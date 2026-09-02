#include "graph_compiler.h"
#include "error_codes.h"
#include "backend_placement_capabilities.h"
#include "execution_placement_plan.h"
#include "data_registry.h"
#include "dense_activation_configuration_policy.h"
#include "arrow_dataset.h"
#include "parquet_backed_dataset.h"
#include "label_column_resolver.h"
#include "graph_topology_utils.h"
#include "worker_defaults.h"
#include "node_metadata_registry.h"
#include "pipeline_runtime_capabilities.h"
#include "cyxwiz/backend_placement_observation.h"
#include "cyxwiz/recurrent_cuda_placement.h"
#include "../gui/loaders/data_loader.h"
#include "../gui/node_import_guardrails.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace cyxwiz {

namespace {
// Local helpers for issue collection. Kept in an anonymous namespace so
// they don't pollute the cyxwiz public surface.

void AddIssue(TrainingConfiguration& config, IssueLevel level,
              const std::string& message, int node_id = -1,
              const std::string& node_name = "",
              const std::string& error_code = "") {
    ValidationIssue issue;
    issue.level = level;
    issue.message = message;
    issue.node_id = node_id;
    issue.node_name = node_name;
    issue.error_code = error_code.empty()
        ? errors::Compiler::GenericIssue
        : error_code;
    config.issues.push_back(std::move(issue));
}

bool IsDatasetSourceType(gui::NodeType type) {
    return type == gui::NodeType::DataInput ||
           type == gui::NodeType::DatasetInput;
}

std::string DatasetNameFromNode(const gui::MLNode& node) {
    auto it = node.parameters.find("dataset_name");
    if (it != node.parameters.end() && !it->second.empty()) {
        return it->second;
    }
    it = node.parameters.find("dataset");
    return it != node.parameters.end() ? it->second : std::string{};
}

std::string DatasetRoleFromNode(const gui::MLNode& node) {
    auto it = node.parameters.find("dataset_role");
    return it == node.parameters.end() ? std::string{} : it->second;
}

const gui::MLNode* FindNodeById(
    const std::vector<gui::MLNode>& nodes,
    int node_id) {
    for (const auto& node : nodes) {
        if (node.id == node_id) return &node;
    }
    return nullptr;
}

const gui::NodePin* FindInputPinByName(
    const gui::MLNode& node,
    const std::string& pin_name) {
    for (const auto& pin : node.inputs) {
        if (pin.name == pin_name) return &pin;
    }
    return nullptr;
}

const gui::NodePin* FindPinById(
    const std::vector<gui::NodePin>& pins,
    int pin_id) {
    for (const auto& pin : pins) {
        if (pin.id == pin_id) return &pin;
    }
    return nullptr;
}

void CollectDatasetSourceOrigins(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    int node_id,
    int output_pin_id,
    std::unordered_set<int>& active_path,
    std::unordered_set<int>& origins,
    bool& invalid_path) {
    const auto* node = FindNodeById(nodes, node_id);
    const auto* output_pin = node
        ? FindPinById(node->outputs, output_pin_id)
        : nullptr;
    if (!node || !output_pin || output_pin->type != gui::PinType::Dataset) {
        invalid_path = true;
        return;
    }

    if (IsDatasetSourceType(node->type)) {
        origins.insert(node->id);
        return;
    }

    const auto runtime_support = ResolvePipelineRuntimeSupport(node->type);
    if (runtime_support.mode != PipelineRuntimeSupportMode::OperatorBacked ||
        runtime_support.implementation_owner !=
            PipelineRuntimeImplementationOwner::PipelineOperatorFactory ||
        !runtime_support.materializer_arrow_table_supported) {
        invalid_path = true;
        return;
    }

    if (!active_path.insert(node->id).second) {
        invalid_path = true;
        return;
    }

    bool found_dataset_input = false;
    for (const auto& link : links) {
        if (link.to_node != node->id) continue;
        const auto* input_pin = FindPinById(node->inputs, link.to_pin);
        if (!input_pin || input_pin->type != gui::PinType::Dataset) continue;
        found_dataset_input = true;
        CollectDatasetSourceOrigins(nodes, links, link.from_node, link.from_pin,
                                    active_path, origins, invalid_path);
    }
    if (!found_dataset_input) invalid_path = true;
    active_path.erase(node->id);
}

const gui::MLNode* FindDatasetSourceConnectedToSplitInput(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    const gui::MLNode& split_node,
    const std::string& pin_name,
    bool* has_connection = nullptr,
    bool* has_multiple_sources = nullptr) {
    if (has_connection) *has_connection = false;
    if (has_multiple_sources) *has_multiple_sources = false;

    const auto* input_pin = FindInputPinByName(split_node, pin_name);
    if (!input_pin) return nullptr;

    std::unordered_set<int> origins;
    std::unordered_set<int> active_path;
    bool invalid_path = false;
    for (const auto& link : links) {
        if (link.to_node != split_node.id || link.to_pin != input_pin->id) {
            continue;
        }
        if (has_connection) *has_connection = true;

        CollectDatasetSourceOrigins(nodes, links, link.from_node, link.from_pin,
                                    active_path, origins, invalid_path);
    }
    if (origins.size() > 1) {
        if (has_multiple_sources) *has_multiple_sources = true;
        return nullptr;
    }
    if (invalid_path || origins.empty()) return nullptr;
    return FindNodeById(nodes, *origins.begin());
}

bool IsLossNodeType(gui::NodeType type);

bool SplitReachesLoss(
    const gui::MLNode& split_node,
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links) {
    const auto reachable = CollectReachableNodeIds(split_node.id, links);
    for (const auto& node : nodes) {
        if (IsLossNodeType(node.type) && reachable.count(node.id) > 0) {
            return true;
        }
    }
    return false;
}

DatasetModality DatasetModalityFromNode(const gui::MLNode& node) {
    const auto category = node.parameters.find("file_category");
    if (category == node.parameters.end()) return DatasetModality::Unknown;
    if (category->second == "image") return DatasetModality::Image;
    if (category->second == "audio") return DatasetModality::Audio;
    if (category->second == "text") return DatasetModality::Text;
    if (category->second == "timeseries") return DatasetModality::TimeSeries;
    if (category->second == "tabular") return DatasetModality::Tabular;
    return DatasetModality::Unknown;
}

DatasetStorageKind DatasetStorageKindFromRegistry(
    const std::string& dataset_name) {
    if (dataset_name.empty()) return DatasetStorageKind::Unknown;
    auto& registry = DataRegistry::Instance();
    if (registry.GetArrowDataset(dataset_name)) {
        return DatasetStorageKind::InMemoryArrow;
    }
    if (registry.GetParquetBackedDataset(dataset_name)) {
        return DatasetStorageKind::DiskBackedParquet;
    }
    if (registry.IsSparseFeatureDataset(dataset_name)) {
        return DatasetStorageKind::SparseFeatureCSR;
    }
    if (registry.IsImageDataset(dataset_name)) {
        return DatasetStorageKind::ImageCached;
    }
    if (registry.IsAudioDataset(dataset_name)) {
        return DatasetStorageKind::AudioCached;
    }
    if (registry.IsTextDataset(dataset_name)) {
        return DatasetStorageKind::TextCached;
    }
    return DatasetStorageKind::Unknown;
}

std::string FingerprintArrowSchema(
    const std::shared_ptr<arrow::Schema>& schema,
    const std::string& label_column,
    bool features_only) {
    if (!schema) return {};
    std::ostringstream identity;
    identity << (features_only ? "feature_schema.v1\n" : "schema.v1\n");
    for (const auto& field : schema->fields()) {
        if (features_only &&
            (field->name() == label_column ||
             field->name().rfind("__", 0) == 0)) {
            continue;
        }
        identity << field->name() << ':' << field->type()->ToString()
                 << ':' << (field->nullable() ? "nullable" : "required")
                 << '\n';
    }
    return StablePartitionFingerprint(identity.str());
}

std::string FingerprintFileIdentity(const std::string& source_path) {
    if (source_path.empty()) return {};
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path path(source_path);
    if (!fs::exists(path, ec) || ec) return {};
    const auto size = fs::is_regular_file(path, ec)
        ? static_cast<uintmax_t>(fs::file_size(path, ec))
        : uintmax_t{0};
    if (ec) return {};
    const auto modified = fs::last_write_time(path, ec);
    if (ec) return {};
    const auto modified_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        modified.time_since_epoch()).count();
    std::ostringstream identity;
    identity << "file_identity.v1\n"
             << fs::absolute(path, ec).lexically_normal().generic_string() << '\n'
             << size << '\n'
             << modified_ns << '\n';
    return ec ? std::string{} : StablePartitionFingerprint(identity.str());
}

void PopulateDatasetSourceProvenance(DatasetSourceRef& source) {
    if (!source.IsSupplied()) return;
    auto& registry = DataRegistry::Instance();
    std::shared_ptr<arrow::Schema> schema;
    if (auto arrow_dataset = registry.GetArrowDataset(source.dataset_name)) {
        source.storage_kind = DatasetStorageKind::InMemoryArrow;
        source.row_count = arrow_dataset->GetNumRows();
        schema = arrow_dataset->GetSchema();
    } else if (auto parquet_dataset =
                   registry.GetParquetBackedDataset(source.dataset_name)) {
        source.storage_kind = DatasetStorageKind::DiskBackedParquet;
        source.row_count = parquet_dataset->GetNumRows();
        schema = parquet_dataset->GetSchema();
    } else if (registry.IsSparseFeatureDataset(source.dataset_name)) {
        source.storage_kind = DatasetStorageKind::SparseFeatureCSR;
    }
    if (schema) {
        source.schema_fingerprint = FingerprintArrowSchema(
            schema, source.label_column, false);
        source.feature_schema_fingerprint = FingerprintArrowSchema(
            schema, source.label_column, true);
    }
    if (auto source_path = registry.GetTabularSourcePath(source.dataset_name)) {
        source.source_fingerprint = FingerprintFileIdentity(*source_path);
    }
}

DatasetSourceRef DatasetSourceFromNode(
    const gui::MLNode& node,
    bool externally_supplied) {
    DatasetSourceRef source;
    source.dataset_name = DatasetNameFromNode(node);
    source.source_node_id = node.id;
    source.externally_supplied = externally_supplied;
    source.modality = DatasetModalityFromNode(node);
    source.storage_kind =
        DatasetStorageKindFromRegistry(source.dataset_name);
    return source;
}

void FinalizeResolvedPartitionContract(TrainingConfiguration& config) {
    auto& partitions = config.dataset_roles;
    PopulateDatasetSourceProvenance(partitions.train);
    PopulateDatasetSourceProvenance(partitions.dev);
    PopulateDatasetSourceProvenance(partitions.test);
    auto& policy = partitions.policy;
    policy.train_ratio = config.train_ratio;
    policy.dev_ratio = config.val_ratio;
    policy.test_ratio = config.test_ratio;
    policy.seed = config.split_seed;
    policy.stratified = config.stratified && !config.is_time_series;

    const bool all_roles_external =
        partitions.dev.IsSupplied() && partitions.test.IsSupplied();
    const bool parquet_stratified_fallback =
        policy.stratified &&
        partitions.train.storage_kind ==
            DatasetStorageKind::DiskBackedParquet;
    if (all_roles_external) {
        policy.method = PartitionSplitMethod::None;
        policy.shuffle = false;
    } else if (parquet_stratified_fallback) {
        // ParquetArrowBatcher currently derives roles through deterministic
        // row-group slicing. Keep the requested config intact for its runtime
        // diagnostic, but record the policy that will actually execute.
        policy.method = PartitionSplitMethod::Random;
        policy.shuffle = false;
        policy.stratified = false;
    } else if (config.is_time_series) {
        policy.method = PartitionSplitMethod::TimeOrdered;
        policy.shuffle = false;
    } else if (policy.stratified) {
        policy.method = PartitionSplitMethod::Stratified;
        // The existing stratified partitioner currently shares Data Loader's
        // shuffle flag. Phase 2 will move this setting fully into SplitPolicy.
        policy.shuffle = config.shuffle;
    } else {
        policy.method = PartitionSplitMethod::Random;
        // Ratio slicing is deterministic and contiguous when no stratified
        // partition column is generated.
        policy.shuffle = false;
    }

    auto& manifest = partitions.manifest;
    manifest.train_origin = PartitionOrigin::External;
    manifest.dev_origin = partitions.dev.IsSupplied()
        ? PartitionOrigin::External
        : PartitionOrigin::Derived;
    manifest.test_origin = partitions.test.IsSupplied()
        ? PartitionOrigin::External
        : PartitionOrigin::Derived;
    manifest.dev_resolution_reason = partitions.dev.IsSupplied()
        ? "preserved in full from the connected Validation Dataset input"
        : "derived only from Training Dataset using Data Split policy";
    manifest.test_resolution_reason = partitions.test.IsSupplied()
        ? "preserved in full from the connected Test Dataset input"
        : "derived only from Training Dataset using Data Split policy";
    manifest.label_column = partitions.train.label_column;
    manifest.split_method = policy.method;
    manifest.train_ratio = config.train_ratio;
    manifest.dev_ratio = config.val_ratio;
    manifest.test_ratio = config.test_ratio;
    manifest.seed = config.split_seed;
    manifest.shuffle = policy.shuffle;
    manifest.stratified = policy.stratified;
    manifest.training_source_fingerprint =
        partitions.train.source_fingerprint;
    manifest.validation_source_fingerprint = partitions.dev.IsSupplied()
        ? partitions.dev.source_fingerprint
        : partitions.train.source_fingerprint;
    manifest.test_source_fingerprint = partitions.test.IsSupplied()
        ? partitions.test.source_fingerprint
        : partitions.train.source_fingerprint;
    manifest.feature_schema_fingerprint =
        partitions.train.feature_schema_fingerprint;

    const auto compatibility_with_train = [&](const DatasetSourceRef& role) {
        if (!role.IsSupplied()) return PartitionCompatibility::Compatible;
        if (partitions.train.schema_fingerprint.empty() ||
            role.schema_fingerprint.empty()) {
            return PartitionCompatibility::Unknown;
        }
        return partitions.train.schema_fingerprint == role.schema_fingerprint
            ? PartitionCompatibility::Compatible
            : PartitionCompatibility::Unknown;
    };
    manifest.dev_compatibility = compatibility_with_train(partitions.dev);
    manifest.test_compatibility = compatibility_with_train(partitions.test);

    int64_t compile_train_rows = -1;
    int64_t compile_dev_rows = partitions.dev.IsSupplied()
        ? partitions.dev.row_count : -1;
    int64_t compile_test_rows = partitions.test.IsSupplied()
        ? partitions.test.row_count : -1;
    if (partitions.train.row_count >= 0 &&
        policy.method != PartitionSplitMethod::Stratified) {
        const int64_t source_rows = partitions.train.row_count;
        compile_train_rows = static_cast<int64_t>(
            static_cast<double>(source_rows) * policy.train_ratio);
        int64_t derived_dev_rows = static_cast<int64_t>(
            static_cast<double>(source_rows) * policy.dev_ratio);
        int64_t derived_test_rows = std::max<int64_t>(
            0, source_rows - compile_train_rows - derived_dev_rows);
        if (!partitions.dev.IsSupplied()) compile_dev_rows = derived_dev_rows;
        if (!partitions.test.IsSupplied()) compile_test_rows = derived_test_rows;
    }
    FinalizePartitionManifest(
        partitions, compile_train_rows, compile_dev_rows, compile_test_rows);
}

bool IsOutputNodeType(gui::NodeType type) {
    return type == gui::NodeType::Output ||
           type == gui::NodeType::SequenceTagOutput;
}

bool IsPreTrainInspectionNode(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::DataProfiler:
        case gui::NodeType::DescribeStats:
        case gui::NodeType::CorrelationMatrix:
        case gui::NodeType::SampleRows:
        case gui::NodeType::ValueCounts:
        case gui::NodeType::DataValidator:
            return true;
        default:
            return false;
    }
}

bool HasReachablePreTrainInspectionNode(
    const std::vector<gui::MLNode>& nodes,
    const std::unordered_set<int>& dataset_reachable) {
    for (const auto& node : nodes) {
        if (dataset_reachable.count(node.id) > 0 &&
            IsPreTrainInspectionNode(node.type)) {
            return true;
        }
    }
    return false;
}

size_t ParsePositiveSizeParam(const gui::MLNode& node,
                              const std::string& key,
                              size_t fallback) {
    auto it = node.parameters.find(key);
    if (it == node.parameters.end() || it->second.empty()) {
        return fallback;
    }
    try {
        long long parsed = std::stoll(it->second);
        return parsed > 0 ? static_cast<size_t>(parsed) : fallback;
    } catch (...) {
        return fallback;
    }
}

std::string FormatBytesForIssue(long double bytes) {
    constexpr long double kGiB =
        1024.0L * 1024.0L * 1024.0L;
    constexpr long double kMiB = 1024.0L * 1024.0L;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    if (bytes >= kGiB) {
        oss << static_cast<double>(bytes / kGiB) << " GiB";
    } else {
        oss << static_cast<double>(bytes / kMiB) << " MiB";
    }
    return oss.str();
}

std::string TextVectorizerName(gui::NodeType type) {
    if (type == gui::NodeType::CountVectorizer) {
        return "CountVectorizer";
    }
    return "TFIDFVectorizer";
}

bool IsSparseOutputValue(const std::map<std::string, std::string>& parameters) {
    auto output_format = parameters.find("output_format");
    if (output_format == parameters.end()) return false;
    std::string value = output_format->second;
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    return value == "sparse";
}

void ValidateDenseTextVectorizerMaterializerMemory(
    TrainingConfiguration& config,
    const std::vector<gui::MLNode>& nodes,
    const std::unordered_set<int>& training_path_ids) {
    if (config.dataset_name.empty()) {
        return;
    }

    const gui::MLNode* vectorizer_node = nullptr;
    for (const auto& node : nodes) {
        if (training_path_ids.count(node.id) == 0) {
            continue;
        }
        if ((node.type == gui::NodeType::TFIDFVectorizer ||
             node.type == gui::NodeType::CountVectorizer) &&
            !IsSparseOutputValue(node.parameters)) {
            vectorizer_node = &node;
            break;
        }
    }
    if (!vectorizer_node) {
        return;
    }

    auto& reg = DataRegistry::Instance();
    size_t sample_count = 0;
    size_t full_vocab_size = 0;
    if (const auto* text_entry =
            reg.GetTextDatasetEntry(config.dataset_name)) {
        sample_count = text_entry->num_samples;
        full_vocab_size = text_entry->vocab_size;
    }
    if (sample_count == 0) {
        if (auto arrow_ds = reg.GetArrowDataset(config.dataset_name)) {
            sample_count = static_cast<size_t>(arrow_ds->GetNumRows());
        } else if (auto pq_ds =
                       reg.GetParquetBackedDataset(config.dataset_name)) {
            sample_count = static_cast<size_t>(pq_ds->GetNumRows());
        }
    }
    if (sample_count == 0) {
        return;
    }

    const size_t max_features =
        ParsePositiveSizeParam(*vectorizer_node, "max_features", 2000);
    const std::string ngram_range =
        vectorizer_node->parameters.count("ngram_range")
            ? vectorizer_node->parameters.at("ngram_range")
            : "1,1";
    const std::string stop_words =
        vectorizer_node->parameters.count("stop_words")
            ? vectorizer_node->parameters.at("stop_words")
            : "english";
    constexpr long double kLargeBoundedOutputBytes =
        2.0L * 1024.0L * 1024.0L * 1024.0L;
    constexpr long double kElementBytes = 4.0L;
    const long double estimated_bytes =
        static_cast<long double>(sample_count) *
        static_cast<long double>(max_features) *
        kElementBytes;

    if (estimated_bytes <= kLargeBoundedOutputBytes) {
        return;
    }

    std::ostringstream msg;
    msg << TextVectorizerName(vectorizer_node->type)
        << " dense output is large: rows=" << sample_count;
    if (full_vocab_size > 0) {
        msg << ", full_vocab=" << full_vocab_size;
    } else {
        msg << ", full_vocab=unknown";
    }
    msg
        << ", max_features=" << max_features
        << ", ngram_range=" << ngram_range
        << ", stop_words=" << stop_words
        << ", estimated Arrow feature allocation="
        << FormatBytesForIssue(estimated_bytes)
        << ". Dense text vectorizers are bounded by max_features, but this "
           "wide table may still pressure host memory before training starts. "
           "Lower max_features, use a sampled dataset, or wait for the sparse "
           "feature path before scaling this graph.";
    AddIssue(config, IssueLevel::Warning, msg.str(),
             vectorizer_node->id, vectorizer_node->name);
}

bool RequestsSparseFeatureOutput(const gui::MLNode& node) {
    if (node.type != gui::NodeType::CountVectorizer &&
        node.type != gui::NodeType::TFIDFVectorizer) {
        return false;
    }
    return IsSparseOutputValue(node.parameters);
}

bool IsTrainingModelLayerType(gui::NodeType type) {
    for (const auto& capability :
         GetPipelineSupportedTrainingRoleCapabilities()) {
        if (capability.node_type == type &&
            capability.role == PipelineTrainingSupportRole::ModelLayer) {
            return true;
        }
    }
    return ResolvePipelineTrainingBackendSupport(type).mode ==
           PipelineTrainingBackendSupportMode::UnsupportedSequentialModelLayer;
}

void ValidateSparseFeatureTrainingContract(
    TrainingConfiguration& config,
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    const std::unordered_set<int>& training_path_ids,
    const std::vector<int>& sorted_ids) {

    std::vector<const gui::MLNode*> sparse_vectorizers;
    for (int node_id : sorted_ids) {
        if (training_path_ids.count(node_id) == 0) continue;
        const auto* node = FindNodeById(nodes, node_id);
        if (node && RequestsSparseFeatureOutput(*node)) {
            sparse_vectorizers.push_back(node);
        }
    }
    if (sparse_vectorizers.empty()) return;

    const auto* vectorizer = sparse_vectorizers.front();
    const size_t sparse_width = ParsePositiveSizeParam(
        *vectorizer, "max_features", 2000);
    config.text_preprocessing.has_vectorizer_node = true;
    config.text_preprocessing.max_length = static_cast<int>(sparse_width);
    config.input_size = sparse_width;
    config.input_shape = {sparse_width};
    if (sparse_vectorizers.size() > 1) {
        AddIssue(config, IssueLevel::Error,
                 "Sparse feature training supports exactly one reachable "
                 "CountVectorizer or TFIDFVectorizer on the selected path",
                 vectorizer->id, vectorizer->name,
                 errors::Compiler::UnsupportedTrainingNode);
    }

    if (config.sequence_batch.enabled || config.is_time_series) {
        AddIssue(config, IssueLevel::Error,
                 "Sparse feature training currently supports only non-sequence "
                 "tabular batches; sequence and time-series contracts require "
                 "a dedicated sparse batch shape owner",
                 vectorizer->id, vectorizer->name,
                 errors::Compiler::UnsupportedTrainingNode);
    }

    const auto downstream = CollectReachableNodeIds(vectorizer->id, links);
    const gui::MLNode* first_model_layer = nullptr;
    bool downstream_materializer_reported = false;
    for (int node_id : sorted_ids) {
        if (node_id == vectorizer->id ||
            training_path_ids.count(node_id) == 0 ||
            downstream.count(node_id) == 0) {
            continue;
        }
        const auto* node = FindNodeById(nodes, node_id);
        if (!node) continue;

        const auto runtime_support =
            ResolvePipelineRuntimeSupport(node->type);
        if (runtime_support.materializer_arrow_table_supported &&
            !downstream_materializer_reported) {
            AddIssue(config, IssueLevel::Error,
                     "Sparse output from node '" + vectorizer->name +
                         "' must be the final preprocessing output; downstream "
                         "operator '" + node->name + "' still requires an Arrow table",
                     node->id, node->name,
                     errors::Compiler::UnsupportedTrainingNode);
            downstream_materializer_reported = true;
        }
        if (!first_model_layer && IsTrainingModelLayerType(node->type)) {
            first_model_layer = node;
        }
    }

    if (!first_model_layer || first_model_layer->type != gui::NodeType::Dense) {
        const std::string actual = first_model_layer
            ? "'" + first_model_layer->name + "'"
            : "no reachable model layer";
        AddIssue(config, IssueLevel::Error,
                 "Sparse feature batches currently require Dense as the first "
                 "model layer; selected path has " + actual,
                 first_model_layer ? first_model_layer->id : vectorizer->id,
                 first_model_layer ? first_model_layer->name : vectorizer->name,
                 errors::Compiler::UnsupportedTrainingNode);
    }

    const DatasetSourceRef* external_roles[] = {
        &config.dataset_roles.dev, &config.dataset_roles.test};
    const char* role_names[] = {"Development", "Test"};
    for (size_t index = 0; index < 2; ++index) {
        const auto& role = *external_roles[index];
        if (!role.IsSupplied() ||
            role.storage_kind == DatasetStorageKind::Unknown ||
            role.storage_kind == DatasetStorageKind::SparseFeatureCSR) {
            continue;
        }
        AddIssue(config, IssueLevel::Error,
                 std::string(role_names[index]) +
                     " dataset must use SparseFeatureCSR when the selected "
                     "training path emits sparse features",
                 role.source_node_id, role.dataset_name,
                 errors::Compiler::UnsupportedTrainingNode);
    }
}

bool IsLossNodeType(gui::NodeType type) {
    return type == gui::NodeType::MSELoss ||
           type == gui::NodeType::CrossEntropyLoss ||
           type == gui::NodeType::FocalLoss ||
           type == gui::NodeType::BCELoss ||
           type == gui::NodeType::BCEWithLogits ||
           type == gui::NodeType::L1Loss ||
           type == gui::NodeType::SmoothL1Loss ||
           type == gui::NodeType::HuberLoss ||
           type == gui::NodeType::NLLLoss ||
           type == gui::NodeType::SoftDiceLoss ||
           type == gui::NodeType::TverskyLoss ||
           type == gui::NodeType::JaccardLoss;
}

bool IsTrueParameter(const gui::MLNode& node, const char* key) {
    const auto it = node.parameters.find(key);
    if (it == node.parameters.end()) return false;
    std::string value = it->second;
    std::transform(
        value.begin(), value.end(), value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value == "true" || value == "1" || value == "yes" ||
           value == "on";
}

TargetContract InferTargetContract(
    const std::vector<gui::MLNode>& nodes,
    const std::unordered_set<int>& training_path_ids,
    const gui::MLNode* loss_node) {
    TargetContract contract;

    // Every currently executable tensor loss consumes an external target
    // stream. Future target-free estimator objectives should leave this false;
    // RL compilers should resolve Environment rather than a dataset label.
    contract.required_by_objective =
        loss_node && IsLossNodeType(loss_node->type);

    for (const auto& node : nodes) {
        if (training_path_ids.count(node.id) == 0) continue;

        if (node.type == gui::NodeType::TimeSeriesWindow) {
            contract.origin = TargetOrigin::GraphGenerated;
            contract.value_kind = TargetValueKind::Continuous;
            contract.producer_node_id = node.id;
            contract.producer_node_name = node.name;
            contract.primary_column = "y";
            contract.width = ParsePositiveSizeParam(
                node, "label_width", 1);
            return contract;
        }

        // Causal language-model targets are shifted/generated from the input
        // token sequence. They are not a raw Data Input label column either.
        if (IsTrueParameter(node, "create_causal_lm_targets") ||
            IsTrueParameter(node, "sequence_create_causal_lm_targets")) {
            contract.origin = TargetOrigin::GraphGenerated;
            contract.value_kind = TargetValueKind::TokenIds;
            contract.producer_node_id = node.id;
            contract.producer_node_name = node.name;
            contract.primary_column = "target_ids";
            contract.width = 1;
            return contract;
        }
    }

    return contract;
}

bool IsSupportedOptimizerNodeType(gui::NodeType type) {
    return type == gui::NodeType::SGD ||
           type == gui::NodeType::Adam ||
           type == gui::NodeType::AdamW ||
           type == gui::NodeType::RMSprop ||
           type == gui::NodeType::Adagrad ||
           type == gui::NodeType::NAdam;
}

const char* IssueLevelLabel(IssueLevel level) {
    switch (level) {
        case IssueLevel::Error:   return "ERROR";
        case IssueLevel::Warning: return "WARN";
        case IssueLevel::Info:    return "INFO";
    }
    return "?";
}

std::string ToLowerAscii(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
    return value;
}

std::string TrimAscii(std::string value) {
    auto is_space = [](unsigned char c) {
        return std::isspace(c) != 0;
    };
    value.erase(value.begin(),
                std::find_if(value.begin(), value.end(),
                             [&](char c) { return !is_space(static_cast<unsigned char>(c)); }));
    value.erase(std::find_if(value.rbegin(), value.rend(),
                             [&](char c) { return !is_space(static_cast<unsigned char>(c)); }).base(),
                value.end());
    return value;
}

bool IsTruthyParameterValue(const std::string& value) {
    const std::string lower = ToLowerAscii(value);
    return lower == "true" || lower == "1" || lower == "yes" ||
           lower == "on";
}

bool IsNeutralUnsupportedParameterValue(const std::string& key,
                                        const std::string& value) {
    const std::string lower = ToLowerAscii(value);
    return lower.empty() || lower == "false" || lower == "0" ||
           lower == "none" || lower == "off" ||
           (key == "balance_mode" && lower == "none") ||
           (key == "class_weight" && lower == "none");
}

std::vector<std::string> PresentUnsupportedParameters(
    const std::map<std::string, std::string>& parameters,
    const std::vector<const char*>& names,
    bool ignore_neutral_values = false) {
    std::vector<std::string> present;
    for (const char* name : names) {
        auto it = parameters.find(name);
        if (it == parameters.end()) {
            continue;
        }
        if (ignore_neutral_values &&
            IsNeutralUnsupportedParameterValue(name, it->second)) {
            continue;
        }
        present.push_back(name);
    }
    return present;
}

std::string JoinNames(const std::vector<std::string>& values) {
    std::ostringstream out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << values[i];
    }
    return out.str();
}

const std::string* FindParam(
    const std::map<std::string, std::string>& params,
    std::initializer_list<const char*> keys) {
    for (const char* key : keys) {
        auto it = params.find(key);
        if (it != params.end()) {
            return &it->second;
        }
    }
    return nullptr;
}

bool ParseFloatVectorLiteral(const std::string& raw,
                             std::vector<float>& values,
                             std::string& error) {
    std::string text = raw;
    for (char& c : text) {
        if (c == '[' || c == ']' || c == '(' || c == ')' ||
            c == ',' || c == ';') {
            c = ' ';
        }
    }

    values.clear();
    std::istringstream in(text);
    std::string token;
    while (in >> token) {
        try {
            size_t parsed = 0;
            const float weight = std::stof(token, &parsed);
            if (parsed != token.size() || !std::isfinite(weight) ||
                weight < 0.0f) {
                throw std::runtime_error("invalid weight");
            }
            values.push_back(weight);
        } catch (...) {
            error = "invalid float weight '" + token + "'";
            return false;
        }
    }
    return true;
}

void ValidateCrossEntropyWeightParams(
    TrainingConfiguration& config,
    const gui::MLNode& loss_node) {
    const auto& params = loss_node.parameters;
    const std::string* class_weight = FindParam(params, {"class_weight"});
    const std::string* explicit_weights =
        FindParam(params, {"class_weights", "weight", "weights"});

    if (class_weight && IsNeutralUnsupportedParameterValue(
            "class_weight", *class_weight)) {
        return;
    }

    const std::string* vector_source = explicit_weights;
    if (class_weight) {
        const std::string mode = ToLowerAscii(TrimAscii(*class_weight));
        if (mode == "balanced") {
            // Runtime resolves this from the supported train split before
            // constructing the loss. Unsupported dataset paths warn later and
            // fall back to unweighted CrossEntropy.
            return;
        }
        if (mode == "manual") {
            if (!explicit_weights ||
                IsNeutralUnsupportedParameterValue("class_weights",
                                                   *explicit_weights)) {
                AddIssue(
                    config,
                    IssueLevel::Error,
                    "CrossEntropy class_weight=manual requires class_weights.",
                    loss_node.id,
                    loss_node.name,
                    errors::Compiler::InvalidParameter);
                return;
            }
        } else if (!explicit_weights) {
            vector_source = class_weight;
        }
    }

    if (!vector_source ||
        IsNeutralUnsupportedParameterValue("class_weights", *vector_source)) {
        return;
    }

    std::vector<float> weights;
    std::string error;
    if (!ParseFloatVectorLiteral(*vector_source, weights, error)) {
        AddIssue(
            config,
            IssueLevel::Error,
            "CrossEntropy class_weights parse error: " + error,
            loss_node.id,
            loss_node.name,
            errors::Compiler::InvalidParameter);
        return;
    }

    const size_t expected_classes =
        config.preprocessing.num_classes > 0
            ? config.preprocessing.num_classes
            : config.output_size;
    if (expected_classes > 0 && weights.size() != expected_classes) {
        AddIssue(
            config,
            IssueLevel::Error,
            "CrossEntropy class_weights size (" +
                std::to_string(weights.size()) +
                ") does not match class/output count (" +
                std::to_string(expected_classes) + ")",
            loss_node.id,
            loss_node.name,
            errors::Compiler::LabelOutputShapeMismatch);
    }
}

void ValidateCrossEntropyLabelSmoothing(
    TrainingConfiguration& config,
    const gui::MLNode& loss_node) {
    const std::string* raw =
        FindParam(loss_node.parameters, {"label_smoothing"});
    if (!raw || IsNeutralUnsupportedParameterValue("label_smoothing", *raw)) {
        return;
    }

    const std::string value = TrimAscii(*raw);
    try {
        size_t parsed = 0;
        const float smoothing = std::stof(value, &parsed);
        if (parsed != value.size() || !std::isfinite(smoothing) ||
            smoothing < 0.0f || smoothing >= 1.0f) {
            throw std::runtime_error("invalid label_smoothing");
        }
    } catch (...) {
        AddIssue(
            config,
            IssueLevel::Error,
            "CrossEntropy label_smoothing must be a finite float in [0, 1).",
            loss_node.id,
            loss_node.name,
            errors::Compiler::InvalidParameter);
    }
}

void ValidateBCEWithLogitsPosWeight(
    TrainingConfiguration& config,
    const gui::MLNode& loss_node) {
    const std::string* raw = FindParam(loss_node.parameters, {"pos_weight"});
    if (!raw || IsNeutralUnsupportedParameterValue("pos_weight", *raw)) {
        return;
    }

    const std::string value = TrimAscii(*raw);
    try {
        size_t parsed = 0;
        const float pos_weight = std::stof(value, &parsed);
        if (parsed != value.size() || !std::isfinite(pos_weight) ||
            pos_weight <= 0.0f) {
            throw std::runtime_error("invalid pos_weight");
        }
    } catch (...) {
        AddIssue(
            config,
            IssueLevel::Error,
            "BCEWithLogits pos_weight must be a positive finite float.",
            loss_node.id,
            loss_node.name,
            errors::Compiler::InvalidParameter);
    }
}

const char* ImplementationStatusLabel(NodeImplementationStatus status) {
    switch (status) {
        case NodeImplementationStatus::Implemented:
            return "implemented";
        case NodeImplementationStatus::Template:
            return "template/deferred";
        case NodeImplementationStatus::Deprecated:
            return "deprecated";
        case NodeImplementationStatus::External:
            return "external";
    }
    return "unknown";
}

const char* PreprocessingDomainLabel(PreprocessingDomain domain) {
    switch (domain) {
        case PreprocessingDomain::Tabular:    return "tabular";
        case PreprocessingDomain::Image:      return "image";
        case PreprocessingDomain::Audio:      return "audio";
        case PreprocessingDomain::Text:       return "text";
        case PreprocessingDomain::TimeSeries: return "time-series";
        case PreprocessingDomain::General:    return "general";
    }
    return "unknown";
}

bool IsGraphRuntimeMergeOp(gui::NodeType type) {
    return type == gui::NodeType::Add ||
           type == gui::NodeType::Multiply ||
           type == gui::NodeType::Average ||
           type == gui::NodeType::Concatenate;
}

bool IsGraphRuntimeBinaryMaskOp(gui::NodeType type) {
    return type == gui::NodeType::TensorCompare ||
           type == gui::NodeType::TensorLogicalMask;
}

bool IsGraphRuntimeLinalgOp(gui::NodeType type) {
    return type == gui::NodeType::TensorDot;
}

bool IsGraphRuntimeFanInOp(gui::NodeType type) {
    return IsGraphRuntimeMergeOp(type) ||
           IsGraphRuntimeBinaryMaskOp(type) ||
           IsGraphRuntimeLinalgOp(type);
}

bool HasParam(const std::map<std::string, std::string>& params,
              const char* key) {
    return params.find(key) != params.end();
}

std::string LowerParamValue(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return value;
}

bool ParamIsEnabled(const std::map<std::string, std::string>& params,
                    const char* key) {
    auto it = params.find(key);
    if (it == params.end()) {
        return false;
    }

    std::string value = LowerParamValue(it->second);
    return value == "true" || value == "1" || value == "yes" ||
           value == "on";
}

bool LooksLikeSequenceBatchContract(const gui::MLNode& node,
                                    std::string& matched_key) {
    const auto& params = node.parameters;

    if (node.type == gui::NodeType::NERSequenceBuilder ||
        node.type == gui::NodeType::TokenVocabulary ||
        node.type == gui::NodeType::POSVocabulary ||
        node.type == gui::NodeType::NERTagVocabulary) {
        matched_key = "first_class_sequence_node";
        return true;
    }

    if (node.type == gui::NodeType::DataInput ||
        node.type == gui::NodeType::DatasetInput) {
        const char* category_keys[] = {
            "file_category",
            "dataset_category",
            "task_type"
        };

        for (const char* key : category_keys) {
            auto it = params.find(key);
            if (it == params.end()) {
                continue;
            }
            const std::string value = LowerParamValue(it->second);
            if (value == "sequence" || value == "sequence_text" ||
                value == "sequence_tagging" || value == "token_tagging" ||
                value == "ner") {
                matched_key = key;
                return true;
            }
        }

        const char* sequence_column_keys[] = {
            "token_column",
            "tokens_column",
            "token_sequence_column",
            "tag_column",
            "tags_column",
            "tag_sequence_column",
            "pos_column",
            "pos_sequence_column",
            "sentence_id_column",
            "sequence_id_column"
        };

        for (const char* key : sequence_column_keys) {
            if (HasParam(params, key)) {
                matched_key = key;
                return true;
            }
        }
    }

    if (node.type == gui::NodeType::DataLoader) {
        const char* sequence_loader_keys[] = {
            "batch_layout",
            "word_ids",
            "pos_ids",
            "tag_ids",
            "target_ids",
            "attention_mask",
            "sequence_lengths",
            "ignore_index",
            "target_ignore_index",
            "causal_lm_targets",
            "create_causal_lm_targets",
            "shifted_targets"
        };

        for (const char* key : sequence_loader_keys) {
            if (HasParam(params, key)) {
                matched_key = key;
                return true;
            }
        }
    }

    return false;
}

bool LooksLikeGenerativeTrainingSketch(const gui::MLNode& node,
                                       std::string& matched_key);

void ExtractSequenceBatchContractFromNode(const gui::MLNode& node,
                                          SequenceBatchConfig& sequence) {
    const auto& params = node.parameters;

    if (node.type == gui::NodeType::DataInput ||
        node.type == gui::NodeType::DatasetInput) {
        auto copy_param = [&params](const char* from, std::string& to) {
            auto it = params.find(from);
            if (it != params.end() && !it->second.empty()) {
                to = it->second;
            }
        };

        std::string matched_key;
        if (LooksLikeSequenceBatchContract(node, matched_key)) {
            sequence.enabled = true;
        }

        copy_param("token_column", sequence.token_column);
        copy_param("tokens_column", sequence.token_column);
        copy_param("token_sequence_column", sequence.token_column);
        copy_param("pos_column", sequence.pos_column);
        copy_param("pos_sequence_column", sequence.pos_column);
        copy_param("tag_column", sequence.tag_column);
        copy_param("tags_column", sequence.tag_column);
        copy_param("tag_sequence_column", sequence.tag_column);
        copy_param("target_column", sequence.target_column);
        copy_param("target_ids_column", sequence.target_column);
        copy_param("decoder_target_column", sequence.target_column);
        copy_param("sentence_id_column", sequence.sentence_id_column);
        copy_param("sequence_id_column", sequence.sentence_id_column);

        std::string generative_key;
        if (LooksLikeGenerativeTrainingSketch(node, generative_key)) {
            sequence.enabled = true;
            sequence.create_causal_lm_targets = true;
        }
    }

    if (node.type == gui::NodeType::NERSequenceBuilder) {
        sequence.enabled = true;
        auto copy_param = [&params](const char* from, std::string& to) {
            auto it = params.find(from);
            if (it != params.end() && !it->second.empty()) {
                to = it->second;
            }
        };

        copy_param("token_column", sequence.token_column);
        copy_param("tokens_column", sequence.token_column);
        copy_param("token_sequence_column", sequence.token_column);
        copy_param("pos_column", sequence.pos_column);
        copy_param("pos_sequence_column", sequence.pos_column);
        copy_param("tag_column", sequence.tag_column);
        copy_param("tags_column", sequence.tag_column);
        copy_param("tag_sequence_column", sequence.tag_column);
        copy_param("target_column", sequence.target_column);
        copy_param("target_ids_column", sequence.target_column);
        copy_param("decoder_target_column", sequence.target_column);
        copy_param("sentence_id_column", sequence.sentence_id_column);
        copy_param("sequence_id_column", sequence.sentence_id_column);

        auto mask_it = params.find("create_attention_mask");
        if (mask_it != params.end()) {
            const std::string value = LowerParamValue(mask_it->second);
            sequence.create_attention_mask =
                value != "false" && value != "0" && value != "off";
        }

        auto ignore_it = params.find("ignore_index");
        if (ignore_it != params.end()) {
            try {
                sequence.ignore_index = std::stoi(ignore_it->second);
            } catch (...) {
                sequence.ignore_index = -100;
            }
        }

        auto max_sequence_it = params.find("max_sequence_length");
        if (max_sequence_it != params.end()) {
            try {
                sequence.max_sequence_length =
                    std::max(0, std::stoi(max_sequence_it->second));
            } catch (...) {
                sequence.max_sequence_length = 0;
            }
        }

        auto target_ignore_it = params.find("target_ignore_index");
        if (target_ignore_it != params.end()) {
            try {
                sequence.target_ignore_index = std::stoi(target_ignore_it->second);
            } catch (...) {
                sequence.target_ignore_index = -100;
            }
        }
    }

    if (node.type == gui::NodeType::DataLoader) {
        std::string matched_key;
        if (LooksLikeSequenceBatchContract(node, matched_key)) {
            sequence.enabled = true;
        }

        auto layout_it = params.find("batch_layout");
        if (layout_it != params.end()) {
            sequence.batch_first = LowerParamValue(layout_it->second) != "time_first";
        }

        auto mask_it = params.find("attention_mask");
        if (mask_it != params.end()) {
            const std::string value = LowerParamValue(mask_it->second);
            sequence.create_attention_mask =
                value != "false" && value != "0" && value != "off";
        }

        auto causal_it = params.find("create_causal_lm_targets");
        if (causal_it == params.end()) {
            causal_it = params.find("causal_lm_targets");
        }
        if (causal_it != params.end()) {
            const std::string value = LowerParamValue(causal_it->second);
            sequence.create_causal_lm_targets =
                value != "false" && value != "0" && value != "off";
        }
        if (HasParam(params, "target_ids") || HasParam(params, "shifted_targets")) {
            sequence.create_causal_lm_targets = true;
        }

        auto ignore_it = params.find("ignore_index");
        if (ignore_it != params.end()) {
            try {
                sequence.ignore_index = std::stoi(ignore_it->second);
            } catch (...) {
                sequence.ignore_index = -100;
            }
        }

        auto max_sequence_it = params.find("max_sequence_length");
        if (max_sequence_it == params.end()) {
            max_sequence_it = params.find("sequence_length");
        }
        if (max_sequence_it != params.end()) {
            try {
                sequence.max_sequence_length =
                    std::max(0, std::stoi(max_sequence_it->second));
            } catch (...) {
                sequence.max_sequence_length = 0;
            }
        }

        auto target_ignore_it = params.find("target_ignore_index");
        if (target_ignore_it != params.end()) {
            try {
                sequence.target_ignore_index = std::stoi(target_ignore_it->second);
            } catch (...) {
                sequence.target_ignore_index = -100;
            }
        }
    }

    if (node.type == gui::NodeType::TextPadding) {
        auto mask_it = params.find("create_attention_mask");
        if (mask_it != params.end()) {
            const std::string value = LowerParamValue(mask_it->second);
            sequence.create_attention_mask =
                value != "false" && value != "0" && value != "off";
        }

        auto max_sequence_it = params.find("max_length");
        if (max_sequence_it == params.end()) {
            max_sequence_it = params.find("max_sequence_length");
        }
        if (max_sequence_it != params.end()) {
            try {
                sequence.max_sequence_length =
                    std::max(0, std::stoi(max_sequence_it->second));
            } catch (...) {
                sequence.max_sequence_length = 0;
            }
        }
    }

    if (node.type == gui::NodeType::CrossEntropyLoss) {
        auto ignore_it = params.find("ignore_index");
        if (ignore_it != params.end()) {
            try {
                sequence.ignore_index = std::stoi(ignore_it->second);
            } catch (...) {
                sequence.ignore_index = -100;
            }
        }

        auto target_ignore_it = params.find("target_ignore_index");
        if (target_ignore_it != params.end()) {
            try {
                sequence.target_ignore_index = std::stoi(target_ignore_it->second);
            } catch (...) {
                sequence.target_ignore_index = -100;
            }
        }
    }
}

void ExtractSequenceBatchContract(
    const std::vector<gui::MLNode>& nodes,
    const std::unordered_set<int>& training_path_ids,
    TrainingConfiguration& config) {
    if (training_path_ids.empty()) {
        return;
    }

    for (const auto& node : nodes) {
        if (training_path_ids.count(node.id) == 0) {
            continue;
        }
        ExtractSequenceBatchContractFromNode(node, config.sequence_batch);
    }
}

bool LooksLikeGenerativeTrainingSketch(const gui::MLNode& node,
                                       std::string& matched_key) {
    const auto& params = node.parameters;
    const char* enabled_keys[] = {
        "causal",
        "causal_mask",
        "autoregressive",
        "decoder_only",
        "generation",
        "generate",
        "language_model",
        "causal_lm",
        "teacher_forcing"
    };

    for (const char* key : enabled_keys) {
        if (ParamIsEnabled(params, key)) {
            matched_key = key;
            return true;
        }
    }

    const char* design_keys[] = {
        "shifted_targets",
        "shift_targets",
        "target_shift",
        "next_token_target",
        "prompt_column",
        "completion_column",
        "decoder_target_column"
    };

    for (const char* key : design_keys) {
        if (HasParam(params, key)) {
            matched_key = key;
            return true;
        }
    }

    return false;
}

bool LooksLikeImportedFineTuningSketch(const gui::MLNode& node,
                                       std::string& matched_key) {
    if (node.type == gui::NodeType::DNNModelLoad ||
        node.type == gui::NodeType::PretrainedYOLO ||
        node.type == gui::NodeType::PretrainedMobileNet ||
        node.type == gui::NodeType::PretrainedOpenPose ||
        node.type == gui::NodeType::PretrainedFaceNet) {
        matched_key = "pretrained model node";
        return true;
    }

    const auto& params = node.parameters;
    const char* enabled_keys[] = {
        "pretrained",
        "fine_tune",
        "fine_tuning",
        "transfer_learning",
        "enable_transfer_learning",
        "load_optimizer_state",
        "allow_shape_mismatch",
        "freeze",
        "freeze_layers",
        "unfreeze_layers",
        "lora"
    };

    for (const char* key : enabled_keys) {
        if (ParamIsEnabled(params, key)) {
            matched_key = key;
            return true;
        }
    }

    const char* design_keys[] = {
        "pretrained_model",
        "pretrained_model_path",
        "pretrained_checkpoint",
        "checkpoint_path",
        "weights_path",
        "model_path",
        "base_model",
        "adapter_path",
        "import_format",
        "onnx_path",
        "safetensors_path",
        "gguf_path",
        "freeze_mode",
        "unfreeze_last_n",
        "frozen_layers",
        "trainable_layers"
    };

    for (const char* key : design_keys) {
        if (HasParam(params, key)) {
            matched_key = key;
            return true;
        }
    }

    return false;
}

bool LooksLikeRLTrainingSketch(const gui::MLNode& node,
                               std::string& matched_key) {
    if (node.type == gui::NodeType::GymEnvironment ||
        node.type == gui::NodeType::ReplayBufferNode ||
        node.type == gui::NodeType::PolicyNetwork ||
        node.type == gui::NodeType::ValueNetwork ||
        node.type == gui::NodeType::RLTraining) {
        matched_key = "RL node";
        return true;
    }

    const auto& params = node.parameters;
    const char* enabled_keys[] = {
        "rl_training",
        "reinforcement_learning",
        "policy_gradient",
        "actor_critic",
        "replay_buffer",
        "target_network",
        "rollout_buffer"
    };

    for (const char* key : enabled_keys) {
        if (ParamIsEnabled(params, key)) {
            matched_key = key;
            return true;
        }
    }

    const char* design_keys[] = {
        "env_name",
        "environment",
        "reward_column",
        "action_column",
        "state_column",
        "next_state_column",
        "done_column",
        "rollout_steps",
        "episode_length",
        "rl_gamma",
        "gae_lambda",
        "policy_loss",
        "value_loss",
        "entropy_bonus"
    };

    for (const char* key : design_keys) {
        if (HasParam(params, key)) {
            matched_key = key;
            return true;
        }
    }

    return false;
}

bool LooksLikeDetectionSegmentationTrainingSketch(const gui::MLNode& node,
                                                  std::string& matched_key) {
    if (node.type == gui::NodeType::DNNDetect ||
        node.type == gui::NodeType::DNNPoseEstimate ||
        node.type == gui::NodeType::DNNFaceDetect ||
        node.type == gui::NodeType::PretrainedYOLO ||
        node.type == gui::NodeType::PretrainedOpenPose ||
        node.type == gui::NodeType::PretrainedFaceNet ||
        node.type == gui::NodeType::NonMaxSuppression ||
        node.type == gui::NodeType::ThresholdFilter) {
        matched_key = "detection node";
        return true;
    }

    const auto& params = node.parameters;
    const char* enabled_keys[] = {
        "object_detection",
        "detection_training",
        "instance_segmentation",
        "semantic_segmentation",
        "segmentation_training",
        "mask_training",
        "yolo_training",
        "pose_estimation"
    };

    for (const char* key : enabled_keys) {
        if (ParamIsEnabled(params, key)) {
            matched_key = key;
            return true;
        }
    }

    const char* design_keys[] = {
        "bbox_column",
        "bboxes_column",
        "boxes_column",
        "box_target_column",
        "mask_column",
        "masks_column",
        "segmentation_mask_column",
        "class_target_column",
        "object_id_column",
        "image_id_column",
        "area_column",
        "iscrowd_column",
        "anchor_boxes",
        "anchors",
        "nms_threshold",
        "iou_threshold",
        "map_metric"
    };

    for (const char* key : design_keys) {
        if (HasParam(params, key)) {
            matched_key = key;
            return true;
        }
    }

    return false;
}

bool LooksLikeTimeDistributedTrainingSketch(const gui::MLNode& node,
                                            std::string& matched_key) {
    const auto& params = node.parameters;
    const char* enabled_keys[] = {
        "time_distributed",
        "per_timestep",
        "per_token_head"
    };

    for (const char* key : enabled_keys) {
        if (ParamIsEnabled(params, key)) {
            matched_key = key;
            return true;
        }
    }

    return false;
}

bool LooksLikeReconstructionGenerativeTrainingSketch(
    const gui::MLNode& node,
    std::string& matched_key) {
    const char* sketch_names[] = {
        "Autoencoder",
        "Autoencoder_Conv",
        "VAE",
        "VAE_Conv",
        "GAN",
        "GAN_Basic",
        "DCGAN",
        "WGAN",
        "CycleGAN",
        "Diffusion",
        "GANLoss"
    };

    for (const char* name : sketch_names) {
        if (node.name == name) {
            matched_key = name;
            return true;
        }
    }

    const auto& params = node.parameters;
    const char* enabled_keys[] = {
        "autoencoder_training",
        "vae_training",
        "gan_training",
        "diffusion_training"
    };

    for (const char* key : enabled_keys) {
        if (ParamIsEnabled(params, key)) {
            matched_key = key;
            return true;
        }
    }

    const char* design_keys[] = {
        "reconstruction_target_column",
        "latent_mean",
        "latent_logvar",
        "logvar",
        "kl_loss_weight",
        "beta_vae",
        "generator_loss",
        "discriminator_loss",
        "gradient_penalty",
        "noise_scheduler",
        "diffusion_scheduler",
        "diffusion_timestep",
        "timestep_embedding",
        "noise_prediction_target"
    };

    for (const char* key : design_keys) {
        if (HasParam(params, key)) {
            matched_key = key;
            return true;
        }
    }

    return false;
}

bool LooksLikeMetricLearningTrainingSketch(const gui::MLNode& node,
                                           std::string& matched_key) {
    switch (node.type) {
        case gui::NodeType::PairDatasetBuilder:
            matched_key = "PairDatasetBuilder";
            return true;
        case gui::NodeType::TripletDatasetBuilder:
            matched_key = "TripletDatasetBuilder";
            return true;
        case gui::NodeType::SharedEncoder:
            matched_key = "SharedEncoder";
            return true;
        case gui::NodeType::SiameseBranch:
            matched_key = "SiameseBranch";
            return true;
        case gui::NodeType::ContrastiveLoss:
            matched_key = "ContrastiveLoss";
            return true;
        case gui::NodeType::CosineEmbeddingLoss:
            matched_key = "CosineEmbeddingLoss";
            return true;
        case gui::NodeType::TripletLoss:
            matched_key = "TripletLoss";
            return true;
        case gui::NodeType::PairMetrics:
            matched_key = "PairMetrics";
            return true;
        case gui::NodeType::RetrievalMetrics:
            matched_key = "RetrievalMetrics";
            return true;
        case gui::NodeType::EmbeddingOutput:
            matched_key = "EmbeddingOutput";
            return true;
        case gui::NodeType::PairScoreOutput:
            matched_key = "PairScoreOutput";
            return true;
        default:
            break;
    }

    const char* sketch_names[] = {
        "PairDatasetBuilder",
        "TripletDatasetBuilder",
        "SharedEncoder",
        "SiameseBranch",
        "ContrastiveLoss",
        "CosineEmbeddingLoss",
        "TripletLoss",
        "PairMetrics",
        "RetrievalMetrics",
        "EmbeddingOutput",
        "PairScoreOutput"
    };

    for (const char* name : sketch_names) {
        if (node.name == name) {
            matched_key = name;
            return true;
        }
    }

    const auto& params = node.parameters;
    const char* enabled_keys[] = {
        "metric_learning",
        "shared_encoder",
        "tied_weights"
    };

    for (const char* key : enabled_keys) {
        if (ParamIsEnabled(params, key)) {
            matched_key = key;
            return true;
        }
    }

    const char* design_keys[] = {
        "anchor_column",
        "positive_column",
        "negative_column",
        "sample_a_column",
        "sample_b_column",
        "pair_label_column",
        "triplet_id_column",
        "pair_id_column"
    };

    for (const char* key : design_keys) {
        if (HasParam(params, key)) {
            matched_key = key;
            return true;
        }
    }

    return false;
}

bool LooksLikeGNNTrainingSketch(const gui::MLNode& node,
                                std::string& matched_key) {
    const char* sketch_names[] = {
        "GraphConv",
        "GraphConvolution",
        "GCNConv",
        "GATConv",
        "GraphSAGEConv",
        "SAGEConv",
        "GINConv",
        "MessagePassing",
        "GNNLayer",
        "GraphReadout",
        "GraphPooling"
    };

    for (const char* name : sketch_names) {
        if (node.name == name) {
            matched_key = name;
            return true;
        }
    }

    const auto& params = node.parameters;
    const char* enabled_keys[] = {
        "gnn_training",
        "graph_neural_network",
        "message_passing",
        "node_classification",
        "link_prediction",
        "graph_classification"
    };

    for (const char* key : enabled_keys) {
        if (ParamIsEnabled(params, key)) {
            matched_key = key;
            return true;
        }
    }

    const char* design_keys[] = {
        "edge_index",
        "edge_index_column",
        "edge_attr",
        "edge_attr_column",
        "node_feature_column",
        "node_features_column",
        "adjacency_matrix_column"
    };

    for (const char* key : design_keys) {
        if (HasParam(params, key)) {
            matched_key = key;
            return true;
        }
    }

    return false;
}

bool IsTensorInputPin(const gui::MLNode& node, int pin_id) {
    for (const auto& pin : node.inputs) {
        if (pin.id == pin_id) {
            return pin.is_input && pin.type == gui::PinType::Tensor;
        }
    }
    return false;
}

int CountConnectedSelectedTensorInputs(
    const gui::MLNode& node,
    const std::vector<gui::NodeLink>& links,
    const std::unordered_set<int>& selected_node_ids) {
    int connected = 0;
    for (const auto& link : links) {
        if (link.to_node != node.id) {
            continue;
        }
        if (selected_node_ids.count(link.from_node) == 0 ||
            selected_node_ids.count(link.to_node) == 0) {
            continue;
        }
        if (IsTensorInputPin(node, link.to_pin)) {
            ++connected;
        }
    }
    return connected;
}

bool HasConnectedSelectedInputPinNamed(
    const gui::MLNode& node,
    const std::vector<gui::NodeLink>& links,
    const std::unordered_set<int>& selected_node_ids,
    const char* pin_name) {
    for (const auto& pin : node.inputs) {
        if (!pin.is_input || pin.name != pin_name) {
            continue;
        }
        for (const auto& link : links) {
            if (link.to_node == node.id &&
                link.to_pin == pin.id &&
                selected_node_ids.count(link.from_node) > 0 &&
                selected_node_ids.count(link.to_node) > 0) {
                return true;
            }
        }
    }
    return false;
}

bool HasConnectedSelectedOutputPinNamed(
    const gui::MLNode& node,
    const std::vector<gui::NodeLink>& links,
    const std::unordered_set<int>& selected_node_ids,
    const char* pin_name) {
    for (const auto& pin : node.outputs) {
        if (pin.is_input || pin.name != pin_name) {
            continue;
        }
        for (const auto& link : links) {
            if (link.from_node == node.id &&
                link.from_pin == pin.id &&
                selected_node_ids.count(link.from_node) > 0 &&
                selected_node_ids.count(link.to_node) > 0) {
                return true;
            }
        }
    }
    return false;
}

// Build the legacy single-string error_message from the issues list so
// existing callers that only look at error_message keep working.
std::string JoinErrorMessages(const std::vector<ValidationIssue>& issues) {
    std::ostringstream out;
    bool first = true;
    for (const auto& i : issues) {
        if (i.level != IssueLevel::Error) continue;
        if (!first) out << "; ";
        first = false;
        if (!i.node_name.empty()) {
            out << "[" << i.node_name << "] ";
        }
        if (!i.error_code.empty()) {
            out << "[" << i.error_code << "] ";
        }
        out << i.message;
    }
    return out.str();
}

void ApplyTextInputShape(TrainingConfiguration& config) {
    if (config.preprocessing_domain != PreprocessingDomain::Text) {
        return;
    }

    int max_length = config.text_preprocessing.max_length;
    const bool graph_overrides_length =
        config.text_preprocessing.has_tokenizer_node ||
        config.text_preprocessing.has_vectorizer_node ||
        config.text_preprocessing.has_padding_node;

    if (!graph_overrides_length && !config.dataset_name.empty()) {
        if (const auto* entry =
                DataRegistry::Instance().GetTextDatasetEntry(config.dataset_name)) {
            max_length = entry->max_length;
        }
    }

    if (max_length <= 0) {
        max_length = 1;
    }

    config.input_shape = {static_cast<size_t>(max_length)};
    config.input_size = static_cast<size_t>(max_length);
}

bool ParseBoolParam(const std::map<std::string, std::string>& params,
                    const std::string& key,
                    bool fallback = false) {
    auto it = params.find(key);
    if (it == params.end()) {
        return fallback;
    }
    return it->second == "true" || it->second == "1";
}

size_t ParseSizeParam(const std::map<std::string, std::string>& params,
                      const std::string& key,
                      size_t fallback) {
    auto it = params.find(key);
    if (it == params.end()) {
        return fallback;
    }
    try {
        const int parsed = std::stoi(it->second);
        return parsed > 0 ? static_cast<size_t>(parsed) : fallback;
    } catch (...) {
        return fallback;
    }
}

size_t EstimateSequenceLength(const CompiledLayer& layer) {
    if (!layer.input_shape.empty()) {
        return layer.input_shape[0];
    }
    return 0;
}

void AddBackendPlacementReports(TrainingConfiguration& config) {
    for (const auto& layer : config.layers) {
        const auto capability = backend_placement::ClassifyLayer(layer.type);
        switch (capability.kind) {
            case backend_placement::LayerCapabilityKind::ArrayFireTensor:
            {
                auto placement =
                    backend_placement::BuildArrayFireTensorPlacement(layer);
                config.backend_placements.push_back(placement);
                if (placement.status == BackendPlacementStatus::Cpu) {
                    AddIssue(config, IssueLevel::Warning,
                             placement.explanation + " Reason code: " +
                                 placement.reason_code + ".",
                             layer.node_id, layer.name);
                }
                continue;
            }
            case backend_placement::LayerCapabilityKind::CpuBackedModelLayer:
                config.backend_placements.push_back(
                    backend_placement::BuildCpuBackedModelLayerPlacement(layer));
                continue;
            case backend_placement::LayerCapabilityKind::UnsupportedSequentialModelLayer:
                config.backend_placements.push_back(
                    backend_placement::BuildUnsupportedSequentialModelPlacement(
                        layer,
                        ResolvePipelineUnsupportedSequentialModelLayerReason(
                            layer.type)));
                continue;
            case backend_placement::LayerCapabilityKind::Unclassified:
            {
                auto placement =
                    backend_placement::BuildUnclassifiedPlacement(layer);
                config.backend_placements.push_back(placement);
                AddIssue(config, IssueLevel::Warning,
                         placement.explanation + " Reason code: " +
                             placement.reason_code + ".",
                         layer.node_id, layer.name);
                continue;
            }
            case backend_placement::LayerCapabilityKind::TimeDistributedSequenceWrapper:
            {
                auto placement =
                    backend_placement::BuildTimeDistributedSequenceWrapperPlacement(
                        layer);
                config.backend_placements.push_back(placement);
                AddIssue(config, IssueLevel::Warning,
                         placement.explanation + " Reason code: " +
                             placement.reason_code + ".",
                         layer.node_id, layer.name);
                continue;
            }
            case backend_placement::LayerCapabilityKind::Recurrent:
                break;
        }

        const size_t hidden_size =
            ParseSizeParam(layer.parameters, "hidden_size", 128);
        const bool bidirectional =
            ParseBoolParam(layer.parameters, "bidirectional", false);
        const bool return_sequences =
            ParseBoolParam(layer.parameters, "return_sequences", false);
        const size_t num_layers =
            ParseSizeParam(layer.parameters, "num_layers", 1);
        const size_t seq_len = EstimateSequenceLength(layer);
        const size_t input_size =
            layer.input_shape.size() >= 2 ? layer.input_shape[1] : 0;

        RecurrentCudaPlacementRequest request;
        request.kind = layer.type == gui::NodeType::GRU
            ? RecurrentLayerKind::GRU
            : RecurrentLayerKind::LSTM;
        request.batch_size = static_cast<size_t>(std::max(1, config.batch_size));
        request.seq_len = seq_len;
        request.input_size = input_size;
        request.hidden_size = hidden_size;
        request.num_layers = num_layers;
        request.bidirectional = bidirectional;
        request.return_sequences = return_sequences;

        const auto decision = EvaluateRecurrentCudaPlacement(request);
        BackendPlacementObservation cached_observation;
        bool has_cached_observation =
            decision.should_attempt_arrayfire_cuda &&
            TryGetRecurrentCudaPlacementObservation(request, cached_observation);
        if (decision.should_attempt_arrayfire_cuda && !has_cached_observation) {
            has_cached_observation =
                TryRunRecurrentCudaPreflightProbe(request, cached_observation);
        }
        const bool cached_cuda_overflow =
            has_cached_observation &&
            cached_observation.reason_code ==
                BackendPlacementObservationReason::CudaJitParamOverflow;
        const bool cached_failure_observation = has_cached_observation;
        BackendPlacementEntry placement;
        placement.node_id = layer.node_id;
        placement.node_name = layer.name;
        placement.node_type = decision.layer_name;
        placement.requested_backend = "auto";
        placement.expected_backend =
            cached_failure_observation ? "CPU" : decision.expected_backend;
        placement.fallback_backend = decision.fallback_backend;
        placement.status = cached_failure_observation
            ? BackendPlacementStatus::Cpu
            : (decision.should_attempt_arrayfire_cuda
                   ? BackendPlacementStatus::Gpu
                   : BackendPlacementStatus::Cpu);
        placement.reason_code = cached_failure_observation
            ? (cached_cuda_overflow
                   ? RecurrentCudaPlacementReason::CudaJitParamOverflowRisk
                   : cached_observation.reason_code)
            : decision.reason_code;
        if (cached_failure_observation) {
            placement.observation_source = cached_observation.source;
            placement.observation_device = cached_observation.device;
            placement.observation_dtype = cached_observation.dtype;
            placement.observation_shape_signature =
                cached_observation.shape_signature;
            placement.observation_detail = cached_observation.detail;
            placement.observation_timestamp = cached_observation.timestamp;
            placement.observation_probe_outcome =
                cached_observation.probe_outcome;
            placement.observation_probe_scope =
                cached_observation.probe_scope;
        }
        const std::string observation_source_label =
            cached_observation.source ==
                    BackendPlacementObservationSource::PreflightProbe
                ? "preflight probe observation"
                : cached_observation.source ==
                          BackendPlacementObservationSource::RuntimeFallback
                      ? "runtime fallback observation"
                      : "runtime/probe observation";
        placement.explanation = cached_failure_observation
            ? (cached_cuda_overflow
                   ? decision.layer_name +
                         " recurrent step is expected to run on CPU because a previous " +
                         observation_source_label + " for this exact "
                         "backend/device/dtype/shape reported CUDA generated-kernel "
                         "formal-parameter overflow (reason=" +
                         cached_observation.reason_code +
                         ", source=" + cached_observation.source +
                         "). This is separate from VRAM capacity. Device: " +
                         cached_observation.device + ". Shape signature: " +
                         cached_observation.shape_signature + "."
                   : decision.layer_name +
                         " recurrent step is expected to run on CPU because a previous " +
                         observation_source_label + " for this exact "
                         "backend/device/dtype/shape reported a backend failure "
                         "(reason=" + cached_observation.reason_code +
                         ", source=" + cached_observation.source +
                         "). Device: " + cached_observation.device +
                         ". Shape signature: " +
                         cached_observation.shape_signature + ".")
            : (decision.should_attempt_arrayfire_cuda
                   ? decision.layer_name + " recurrent step is allowed on ArrayFire CUDA by the current placement policy."
                   : decision.reason);
        placement.suggested_action = cached_failure_observation
            ? "Training can continue. Use CPU for this recurrent shape until "
              "a fused/native CUDA recurrent kernel or exact successful "
              "backend probe proves the shape safe on this device."
            : (decision.should_attempt_arrayfire_cuda
                   ? "No action needed."
                   : "Training can continue. To keep this recurrent step on GPU, use a future fused/native CUDA recurrent kernel or exact backend probe; reducing hidden_size, sequence length, layers, or bidirectionality may help only for LSTM estimator-limited shapes.");
        config.backend_placements.push_back(placement);

        if (decision.should_attempt_arrayfire_cuda &&
            !cached_failure_observation) {
            continue;
        }

        const std::string issue_name =
            decision.layer_name + " hidden_size=" + std::to_string(hidden_size);
        std::ostringstream msg;
        msg << decision.layer_name << " layer is valid, but "
            << placement.explanation
            << " Reason code: " << placement.reason_code << ". "
            << "Runtime will use the same placement policy instead of "
            << "repeatedly attempting CUDA and falling back every batch.";
        AddIssue(config, IssueLevel::Warning, msg.str(),
                 layer.node_id, issue_name);
    }

    for (const int graph_op_node_id : config.graph_op_node_ids) {
        auto it = std::find_if(
            config.graph_plan.nodes.begin(),
            config.graph_plan.nodes.end(),
            [graph_op_node_id](const CompiledGraphNode& node) {
                return node.node_id == graph_op_node_id;
            });
        if (it == config.graph_plan.nodes.end()) {
            continue;
        }
        config.backend_placements.push_back(
            backend_placement::BuildGraphRuntimePlacement(*it));
    }
}

void FinalizePinMemoryTransferStatus(TrainingConfiguration& config) {
    auto& status = config.pin_memory_transfer;
    if (!status.requested) {
        return;
    }

    const auto summary = config.SummarizeBackendPlacements();
    const bool cpu_only =
        summary.total > 0 && summary.cpu == summary.total;
    if (cpu_only) {
        status.backend = "CPU";
        status.effective_mode =
            PinMemoryTransferMode::PinnedRequestedButNotApplicable;
        status.reason_code =
            PinMemoryTransferReason::CpuBackendNotApplicable;
        status.message =
            "DataLoader pin_memory=true is not applicable because the compiled "
            "training placement is CPU-only; regular host memory will be used";
    } else {
        status.backend =
            (summary.gpu > 0 || summary.mixed > 0 || summary.risk > 0)
                ? "gpu"
                : "auto";
        status.effective_mode =
            PinMemoryTransferMode::PinnedRequestedButUnsupported;
        status.reason_code =
            PinMemoryTransferReason::BackendUnavailable;
        status.message =
            "DataLoader pin_memory=true is unsupported by current batchers and "
            "will be ignored; serialized for compatibility only until a pinned "
            "host-memory transfer backend exists";
    }

    spdlog::warn("GraphCompiler: {}", status.message);
    AddIssue(config,
             IssueLevel::Warning,
             status.message,
             status.node_id,
             status.node_name);
}

bool ParseFloatParam(const std::map<std::string, std::string>& params,
                     const std::string& key,
                     float fallback,
                     float& value,
                     std::string& error) {
    auto it = params.find(key);
    if (it == params.end() || it->second.empty()) {
        value = fallback;
        return true;
    }

    try {
        size_t parsed_chars = 0;
        value = std::stof(it->second, &parsed_chars);
        if (parsed_chars != it->second.size()) {
            error = key + " must be a float";
            return false;
        }
        if (!std::isfinite(value)) {
            error = key + " must be finite";
            return false;
        }
        return true;
    } catch (...) {
        error = key + " must be a float";
        return false;
    }
}

void ExtractOptimizerConfiguration(const gui::MLNode& node,
                                   TrainingConfiguration& config) {
    config.optimizer_type = node.type;
    config.learning_rate = 0.001f;
    config.momentum = 0.0f;
    config.beta1 = 0.9f;
    config.beta2 = 0.999f;
    config.epsilon = 1e-8f;
    config.rmsprop_alpha = 0.99f;
    config.weight_decay = 0.0f;

    switch (node.type) {
        case gui::NodeType::SGD:
            config.learning_rate = 0.01f;
            config.momentum = 0.9f;
            break;
        case gui::NodeType::AdamW:
            config.weight_decay = 0.01f;
            break;
        case gui::NodeType::Adagrad:
            config.learning_rate = 0.01f;
            config.epsilon = 1e-10f;
            break;
        case gui::NodeType::NAdam:
            config.learning_rate = 0.002f;
            break;
        default:
            break;
    }

    const auto parse = [&](const std::string& key,
                           float fallback,
                           float minimum,
                           bool minimum_inclusive,
                           float maximum,
                           bool maximum_inclusive,
                           float& destination) {
        std::string error;
        float value = fallback;
        if (!ParseFloatParam(node.parameters, key, fallback, value, error)) {
            AddIssue(config, IssueLevel::Error,
                     "Invalid optimizer parameter: " + error,
                     node.id, node.name, errors::Compiler::InvalidParameter);
            return;
        }
        const bool below = minimum_inclusive ? value < minimum
                                             : value <= minimum;
        const bool above = maximum_inclusive ? value > maximum
                                             : value >= maximum;
        if (below || above) {
            AddIssue(config, IssueLevel::Error,
                     "Invalid optimizer parameter '" + key +
                         "': value is outside the supported range",
                     node.id, node.name, errors::Compiler::InvalidParameter);
            return;
        }
        destination = value;
    };

    const auto canonical_lr = node.parameters.find("learning_rate");
    const auto legacy_lr = node.parameters.find("lr");
    const bool has_canonical_lr = canonical_lr != node.parameters.end() &&
                                  !canonical_lr->second.empty();
    const bool has_legacy_lr = legacy_lr != node.parameters.end() &&
                               !legacy_lr->second.empty();
    const std::string lr_key = has_canonical_lr ? "learning_rate"
                                                : (has_legacy_lr ? "lr"
                                                                 : "learning_rate");
    parse(lr_key, config.learning_rate, 0.0f, false,
          std::numeric_limits<float>::max(), true, config.learning_rate);
    if (has_canonical_lr && has_legacy_lr &&
        canonical_lr->second != legacy_lr->second) {
        AddIssue(config, IssueLevel::Warning,
                 "Both learning_rate and legacy lr are present; "
                 "learning_rate is authoritative",
                 node.id, node.name, errors::Compiler::InvalidParameter);
    }

    switch (node.type) {
        case gui::NodeType::SGD:
            parse("momentum", config.momentum, 0.0f, true, 1.0f, false,
                  config.momentum);
            if (node.parameters.count("weight_decay")) {
                AddIssue(config, IssueLevel::Warning,
                         "SGD weight_decay is not supported by the backend and "
                         "will be ignored",
                         node.id, node.name,
                         errors::Compiler::UnsupportedTrainingNode);
            }
            break;
        case gui::NodeType::Adam:
        case gui::NodeType::AdamW:
        case gui::NodeType::NAdam:
            parse("beta1", config.beta1, 0.0f, true, 1.0f, false,
                  config.beta1);
            parse("beta2", config.beta2, 0.0f, true, 1.0f, false,
                  config.beta2);
            parse("epsilon", config.epsilon, 0.0f, false,
                  std::numeric_limits<float>::max(), true, config.epsilon);
            if (node.type == gui::NodeType::AdamW) {
                parse("weight_decay", config.weight_decay, 0.0f, true,
                      std::numeric_limits<float>::max(), true,
                      config.weight_decay);
            }
            break;
        case gui::NodeType::RMSprop:
            parse("alpha", config.rmsprop_alpha, 0.0f, true, 1.0f, false,
                  config.rmsprop_alpha);
            parse("epsilon", config.epsilon, 0.0f, false,
                  std::numeric_limits<float>::max(), true, config.epsilon);
            parse("momentum", config.momentum, 0.0f, true, 1.0f, false,
                  config.momentum);
            break;
        case gui::NodeType::Adagrad:
            parse("epsilon", config.epsilon, 0.0f, false,
                  std::numeric_limits<float>::max(), true, config.epsilon);
            if (node.parameters.count("lr_decay")) {
                AddIssue(config, IssueLevel::Warning,
                         "Adagrad lr_decay is not supported by the backend and "
                         "will be ignored",
                         node.id, node.name,
                         errors::Compiler::UnsupportedTrainingNode);
            }
            break;
        default:
            break;
    }
}

bool ValidateTensorScalarMathParams(gui::NodeType type,
                                    const std::map<std::string, std::string>& params,
                                    std::string& error) {
    if (type == gui::NodeType::TensorPow) {
        float exponent = 2.0f;
        return ParseFloatParam(params, "exponent", 2.0f, exponent, error);
    }

    if (type == gui::NodeType::TensorClip) {
        float min_val = 0.0f;
        float max_val = 1.0f;
        if (!ParseFloatParam(params, "min", 0.0f, min_val, error) ||
            !ParseFloatParam(params, "max", 1.0f, max_val, error)) {
            return false;
        }
        if (min_val > max_val) {
            error = "TensorClip min must be <= max";
            return false;
        }
    }

    return true;
}

const gui::MLNode* FindFirstLossNodeInSet(
    const std::vector<gui::MLNode>& nodes,
    const std::unordered_set<int>& node_ids) {

    for (const auto& node : nodes) {
        if (IsLossNodeType(node.type) &&
            node_ids.count(node.id) > 0) {
            return &node;
        }
    }
    return nullptr;
}

const gui::MLNode* FindFirstOptimizerNodeInSet(
    const std::vector<gui::MLNode>& nodes,
    const std::unordered_set<int>& node_ids) {

    for (const auto& node : nodes) {
        if (IsSupportedOptimizerNodeType(node.type) &&
            node_ids.count(node.id) > 0) {
            return &node;
        }
    }
    return nullptr;
}

void ValidateSingleDatasetReachableLossNode(
    const std::vector<gui::MLNode>& nodes,
    const std::unordered_set<int>& dataset_reachable,
    TrainingConfiguration& config) {

    if (dataset_reachable.empty()) {
        return;
    }

    std::vector<const gui::MLNode*> reachable_losses;
    for (const auto& node : nodes) {
        if (IsLossNodeType(node.type) &&
            dataset_reachable.count(node.id) > 0) {
            reachable_losses.push_back(&node);
        }
    }

    if (reachable_losses.size() <= 1) {
        return;
    }

    std::ostringstream names;
    for (size_t i = 0; i < reachable_losses.size(); ++i) {
        if (i > 0) {
            names << ", ";
        }
        names << "'" << reachable_losses[i]->name << "'";
    }

    AddIssue(
        config,
        IssueLevel::Error,
        "Current Studio training supports exactly one dataset-reachable "
        "loss node. Found " + std::to_string(reachable_losses.size()) +
        " losses (" + names.str() + "). Multi-head/multi-task or "
        "alternating-loss training needs a first-class loss aggregation "
        "contract before it can compile truthfully.");
}

void ValidateSingleDatasetSourceForSelectedLoss(
    const std::vector<gui::MLNode>& nodes,
    const gui::MLNode* selected_loss,
    const std::vector<gui::NodeLink>& links,
    TrainingConfiguration& config) {

    if (!selected_loss) {
        return;
    }

    const auto loss_ancestors = CollectAncestorNodeIds(selected_loss->id, links);
    std::vector<const gui::MLNode*> selected_sources;
    for (const auto& node : nodes) {
        if (!IsDatasetSourceType(node.type) ||
            loss_ancestors.count(node.id) == 0) {
            continue;
        }

        bool saw_selected_path_route = false;
        bool all_routes_are_evaluation_inputs = true;
        for (const auto& link : links) {
            if (link.from_node != node.id ||
                loss_ancestors.count(link.to_node) == 0) {
                continue;
            }
            saw_selected_path_route = true;

            const gui::MLNode* target = FindNodeById(nodes, link.to_node);
            if (!target || target->type != gui::NodeType::DataSplit) {
                all_routes_are_evaluation_inputs = false;
                break;
            }
            const auto* validation_pin =
                FindInputPinByName(*target, "Validation Dataset");
            const auto* test_pin = FindInputPinByName(*target, "Test Dataset");
            const bool is_evaluation_input =
                (validation_pin && link.to_pin == validation_pin->id) ||
                (test_pin && link.to_pin == test_pin->id);
            if (!is_evaluation_input) {
                all_routes_are_evaluation_inputs = false;
                break;
            }
        }
        const bool topology_assigned_evaluation_role =
            saw_selected_path_route && all_routes_are_evaluation_inputs;

        // Dev/Test sources routed through their named Data Split inputs are
        // evaluation roles, not additional model inputs. Any other route still
        // counts as a source on the selected training path and fails below.
        if (!topology_assigned_evaluation_role) {
            selected_sources.push_back(&node);
        }
    }

    if (selected_sources.size() <= 1) {
        return;
    }

    std::ostringstream names;
    for (size_t i = 0; i < selected_sources.size(); ++i) {
        if (i > 0) {
            names << ", ";
        }
        names << "'" << selected_sources[i]->name << "'";
    }

    AddIssue(
        config,
        IssueLevel::Error,
        "Current Studio training supports exactly one dataset source on "
        "the selected loss path. Found " +
        std::to_string(selected_sources.size()) + " sources (" +
        names.str() + ") feeding loss '" + selected_loss->name +
        "'. Multi-input/shared-weight training needs a typed named-batch "
        "contract before it can compile truthfully.",
        selected_loss->id,
        selected_loss->name);
}

bool ContainsWhenFiltered(
    const std::unordered_set<int>& node_ids,
    int node_id) {

    return node_ids.empty() || node_ids.count(node_id) > 0;
}

std::vector<std::string> ParseOrderedColumnList(const std::string& value) {
    std::vector<std::string> columns;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        token = TrimAscii(token);
        if (!token.empty()) columns.push_back(std::move(token));
    }
    return columns;
}

std::vector<std::string> ExpectedTargetColumns(
    const TargetContract& target) {
    std::vector<std::string> columns;
    if (target.primary_column.empty() || target.width == 0) return columns;
    columns.reserve(target.width);
    columns.push_back(target.primary_column);
    for (size_t i = 1; i < target.width; ++i) {
        columns.push_back(target.primary_column + "_" + std::to_string(i));
    }
    return columns;
}

void ExtractRegressionTargetTransform(
    const std::vector<gui::MLNode>& nodes,
    const std::unordered_set<int>& training_path_ids,
    TrainingConfiguration& config) {
    const gui::MLNode* selected = nullptr;
    for (const auto& node : nodes) {
        if (node.type != gui::NodeType::StandardScaler ||
            !ContainsWhenFiltered(training_path_ids, node.id)) {
            continue;
        }
        const auto role = node.parameters.find("transform_role");
        const std::string role_value = role == node.parameters.end()
            ? "features"
            : ToLowerAscii(TrimAscii(role->second));
        if (role_value != "regression_target") continue;
        if (selected) {
            AddIssue(config, IssueLevel::Error,
                     "Only one Standard Scaler may declare the regression "
                     "target transform on the selected training path.",
                     node.id, node.name,
                     errors::Compiler::GenericIssue);
            return;
        }
        selected = &node;
    }

    if (!selected) return;
    auto& contract = config.regression_target_transform;
    contract.enabled = true;
    contract.node_id = selected->id;
    contract.node_name = selected->name;
    contract.operator_name = "StandardScaler";

    if (config.target.value_kind != TargetValueKind::Continuous ||
        !config.target.IsResolved()) {
        AddIssue(config, IssueLevel::Error,
                 "Regression target scaling requires a resolved continuous "
                 "target contract.",
                 selected->id, selected->name,
                 errors::Data::RequiredLabelColumnMissing);
        return;
    }

    const auto expected = ExpectedTargetColumns(config.target);
    const auto columns_it = selected->parameters.find("columns");
    const auto configured = ParseOrderedColumnList(
        columns_it == selected->parameters.end() ? "" : columns_it->second);
    if (configured != expected) {
        AddIssue(config, IssueLevel::Error,
                 "Regression target scaler Columns must exactly match the "
                 "ordered target contract: " + JoinNames(expected) + ".",
                 selected->id, selected->name,
                 errors::Compiler::LabelOutputShapeMismatch);
        return;
    }

    const auto mode = selected->parameters.find("operation_mode");
    if (mode == selected->parameters.end() ||
        ToLowerAscii(TrimAscii(mode->second)) != "transform_only") {
        AddIssue(config, IssueLevel::Error,
                 "Regression target scaling must use Transform Only with "
                 "statistics fitted on Train data.",
                 selected->id, selected->name,
                 errors::Data::ColumnTypeMismatch);
        return;
    }
    const auto path = selected->parameters.find("state_path");
    contract.state_path = path == selected->parameters.end()
        ? ""
        : TrimAscii(path->second);
    if (contract.state_path.empty()) {
        AddIssue(config, IssueLevel::Error,
                 "Regression target scaling requires the fitted Train state "
                 "artifact path.",
                 selected->id, selected->name,
                 errors::Data::ColumnTypeMismatch);
        return;
    }

    const auto label = selected->parameters.find("label_col");
    const auto excluded = selected->parameters.find("exclude_columns");
    if ((label != selected->parameters.end() &&
         !TrimAscii(label->second).empty()) ||
        (excluded != selected->parameters.end() &&
         !TrimAscii(excluded->second).empty())) {
        AddIssue(config, IssueLevel::Error,
                 "Regression target scaler must not exclude target columns; "
                 "leave Label column and Additional exclusions empty.",
                 selected->id, selected->name,
                 errors::Compiler::LabelOutputShapeMismatch);
        return;
    }

    contract.target_columns = expected;
    AddIssue(config, IssueLevel::Info,
             "Regression target scaling is declared from Train-fitted state; "
             "loss remains in transformed space while MAE/RMSE are restored "
             "to original target units.",
             selected->id, selected->name);
}

bool HasConnectedInputAfterFirst(const gui::MLNode& node,
                                 const std::vector<gui::NodeLink>& links) {
    if (node.inputs.size() <= 1) {
        return false;
    }
    std::unordered_set<int> connected_input_pins;
    connected_input_pins.reserve(links.size());
    for (const auto& link : links) {
        connected_input_pins.insert(link.to_pin);
    }
    for (size_t i = 1; i < node.inputs.size(); ++i) {
        if (connected_input_pins.count(node.inputs[i].id) > 0) {
            return true;
        }
    }
    return false;
}

bool CheckedMultiplySize(size_t lhs, size_t rhs, size_t& out) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    out = lhs * rhs;
    return true;
}

size_t ShapeElementCount(const std::vector<size_t>& shape) {
    size_t count = 1;
    for (size_t dim : shape) {
        size_t next = 0;
        if (!CheckedMultiplySize(count, dim, next)) {
            throw std::overflow_error("shape element count overflow");
        }
        count = next;
    }
    return count;
}

bool ResolveReshapeTargetShape(const std::map<std::string, std::string>& params,
                               const std::vector<size_t>& input_shape,
                               std::vector<size_t>& target_shape,
                               std::string& error) {
    auto it = params.find("shape");
    if (it == params.end() || it->second.empty()) {
        error = "Reshape/View requires a non-empty shape parameter";
        return false;
    }
    if (input_shape.empty()) {
        error = "Reshape/View requires a known input shape";
        return false;
    }

    std::string shape_str = it->second;
    shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), '['), shape_str.end());
    shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ']'), shape_str.end());
    shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ' '), shape_str.end());

    std::vector<int64_t> dims;
    std::stringstream ss(shape_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            error = "Reshape/View shape contains an empty dimension";
            return false;
        }
        try {
            dims.push_back(std::stoll(token));
        } catch (...) {
            error = "Reshape/View shape contains a non-integer dimension";
            return false;
        }
    }

    if (dims.empty()) {
        error = "Reshape/View requires at least one target dimension";
        return false;
    }

    int infer_index = -1;
    size_t known_product = 1;
    for (size_t i = 0; i < dims.size(); ++i) {
        const int64_t dim = dims[i];
        if (dim == -1) {
            if (infer_index != -1) {
                error = "Reshape/View allows at most one -1 inferred dimension";
                return false;
            }
            infer_index = static_cast<int>(i);
            continue;
        }
        if (dim <= 0) {
            error = "Reshape/View dimensions must be positive, except one optional -1";
            return false;
        }
        size_t next = 0;
        if (!CheckedMultiplySize(known_product, static_cast<size_t>(dim), next)) {
            error = "Reshape/View target shape product overflows";
            return false;
        }
        known_product = next;
    }

    size_t input_elements = 0;
    try {
        input_elements = ShapeElementCount(input_shape);
    } catch (...) {
        error = "Reshape/View input shape product overflows";
        return false;
    }

    target_shape.clear();
    target_shape.reserve(dims.size());
    if (infer_index >= 0) {
        if (known_product == 0 || input_elements % known_product != 0) {
            error = "Reshape/View inferred dimension is not divisible by input elements";
            return false;
        }
        dims[static_cast<size_t>(infer_index)] =
            static_cast<int64_t>(input_elements / known_product);
    } else if (known_product != input_elements) {
        error = "Reshape/View target shape must preserve element count";
        return false;
    }

    for (int64_t dim : dims) {
        target_shape.push_back(static_cast<size_t>(dim));
    }
    return true;
}

bool ParseIntParam(const std::map<std::string, std::string>& params,
                   const std::string& key,
                   int fallback,
                   int& out,
                   std::string& error) {
    auto it = params.find(key);
    if (it == params.end() || it->second.empty()) {
        out = fallback;
        return true;
    }
    try {
        out = std::stoi(it->second);
        return true;
    } catch (...) {
        error = key + " must be an integer";
        return false;
    }
}

bool ParsePositiveShapeParam(const std::map<std::string, std::string>& params,
                             const std::string& key,
                             const std::string& op_name,
                             std::vector<size_t>& dims,
                             std::string& error) {
    auto it = params.find(key);
    if (it == params.end() || it->second.empty()) {
        error = op_name + " requires a non-empty " + key + " parameter";
        return false;
    }

    std::string shape_str = it->second;
    shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), '['), shape_str.end());
    shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ']'), shape_str.end());
    shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ' '), shape_str.end());

    dims.clear();
    std::stringstream ss(shape_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            error = op_name + " shape contains an empty dimension";
            return false;
        }
        try {
            const int64_t dim = std::stoll(token);
            if (dim <= 0) {
                error = op_name + " shape dimensions must be positive";
                return false;
            }
            dims.push_back(static_cast<size_t>(dim));
        } catch (...) {
            error = op_name + " shape contains a non-integer dimension";
            return false;
        }
    }

    if (dims.empty()) {
        error = op_name + " requires at least one target dimension";
        return false;
    }
    return true;
}

bool ParseIndexListParam(const std::map<std::string, std::string>& params,
                         const std::string& key,
                         const std::string& op_name,
                         std::vector<int>& indices,
                         std::string& error) {
    auto it = params.find(key);
    if (it == params.end() || it->second.empty()) {
        error = op_name + " requires a non-empty " + key + " parameter";
        return false;
    }

    std::string list_str = it->second;
    list_str.erase(std::remove(list_str.begin(), list_str.end(), '['), list_str.end());
    list_str.erase(std::remove(list_str.begin(), list_str.end(), ']'), list_str.end());
    list_str.erase(std::remove(list_str.begin(), list_str.end(), ' '), list_str.end());

    indices.clear();
    std::stringstream ss(list_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            error = op_name + " indices contains an empty value";
            return false;
        }
        try {
            indices.push_back(std::stoi(token));
        } catch (...) {
            error = op_name + " indices contains a non-integer value";
            return false;
        }
    }

    if (indices.empty()) {
        error = op_name + " requires at least one index";
        return false;
    }
    return true;
}

bool NormalizeCompilerDim(int dim,
                          int rank,
                          bool allow_end,
                          const std::string& op_name,
                          int& normalized,
                          std::string& error) {
    const int min_dim = allow_end ? -rank - 1 : -rank;
    const int max_dim = allow_end ? rank : rank - 1;
    if (dim < min_dim || dim > max_dim) {
        error = op_name + " dimension is out of range";
        return false;
    }
    normalized = dim < 0 ? dim + rank + (allow_end ? 1 : 0) : dim;
    return true;
}

bool ResolveSqueezeTargetShape(const std::map<std::string, std::string>& params,
                               const std::vector<size_t>& input_shape,
                               std::vector<size_t>& target_shape,
                               std::string& error) {
    if (input_shape.empty()) {
        error = "Squeeze requires a known input shape";
        return false;
    }

    int dim = -1;
    if (!ParseIntParam(params, "dim", -1, dim, error)) {
        error = "Squeeze " + error;
        return false;
    }

    target_shape.clear();
    if (dim == -1) {
        for (size_t size : input_shape) {
            if (size != 1) {
                target_shape.push_back(size);
            }
        }
    } else {
        int axis = 0;
        if (!NormalizeCompilerDim(dim,
                                  static_cast<int>(input_shape.size()),
                                  false,
                                  "Squeeze",
                                  axis,
                                  error)) {
            return false;
        }
        if (input_shape[static_cast<size_t>(axis)] != 1) {
            error = "Squeeze selected dimension must have size 1";
            return false;
        }
        for (size_t i = 0; i < input_shape.size(); ++i) {
            if (i != static_cast<size_t>(axis)) {
                target_shape.push_back(input_shape[i]);
            }
        }
    }

    if (target_shape.empty()) {
        target_shape.push_back(1);
    }
    return true;
}

bool ResolveUnsqueezeTargetShape(const std::map<std::string, std::string>& params,
                                 const std::vector<size_t>& input_shape,
                                 std::vector<size_t>& target_shape,
                                 std::string& error) {
    if (input_shape.empty()) {
        error = "Unsqueeze requires a known input shape";
        return false;
    }

    int dim = 0;
    if (!ParseIntParam(params, "dim", 0, dim, error)) {
        error = "Unsqueeze " + error;
        return false;
    }

    int axis = 0;
    if (!NormalizeCompilerDim(dim,
                              static_cast<int>(input_shape.size()),
                              true,
                              "Unsqueeze",
                              axis,
                              error)) {
        return false;
    }

    target_shape.clear();
    target_shape.reserve(input_shape.size() + 1);
    for (int i = 0; i < static_cast<int>(input_shape.size()); ++i) {
        if (i == axis) {
            target_shape.push_back(1);
        }
        target_shape.push_back(input_shape[static_cast<size_t>(i)]);
    }
    if (axis == static_cast<int>(input_shape.size())) {
        target_shape.push_back(1);
    }
    return true;
}

bool ResolvePermuteTargetShape(const std::map<std::string, std::string>& params,
                               const std::vector<size_t>& input_shape,
                               std::vector<size_t>& target_shape,
                               std::vector<int>& normalized_dims,
                               std::string& error) {
    if (input_shape.empty()) {
        error = "Permute requires a known input shape";
        return false;
    }

    auto it = params.find("dims");
    if (it == params.end() || it->second.empty()) {
        error = "Permute requires a non-empty dims parameter";
        return false;
    }

    std::string dims_str = it->second;
    dims_str.erase(std::remove(dims_str.begin(), dims_str.end(), '['), dims_str.end());
    dims_str.erase(std::remove(dims_str.begin(), dims_str.end(), ']'), dims_str.end());
    dims_str.erase(std::remove(dims_str.begin(), dims_str.end(), ' '), dims_str.end());

    const int rank = static_cast<int>(input_shape.size());
    std::stringstream ss(dims_str);
    std::string token;
    normalized_dims.clear();
    std::vector<bool> seen(input_shape.size(), false);

    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            error = "Permute dims contains an empty dimension";
            return false;
        }

        int raw_dim = 0;
        try {
            raw_dim = std::stoi(token);
        } catch (...) {
            error = "Permute dims contains a non-integer dimension";
            return false;
        }

        int axis = 0;
        if (!NormalizeCompilerDim(raw_dim, rank, false, "Permute", axis, error)) {
            return false;
        }
        if (seen[static_cast<size_t>(axis)]) {
            error = "Permute dims must not contain duplicates";
            return false;
        }
        seen[static_cast<size_t>(axis)] = true;
        normalized_dims.push_back(axis);
    }

    if (normalized_dims.size() != input_shape.size()) {
        error = "Permute dims must match input rank";
        return false;
    }

    target_shape.clear();
    target_shape.reserve(normalized_dims.size());
    for (int axis : normalized_dims) {
        target_shape.push_back(input_shape[static_cast<size_t>(axis)]);
    }
    return true;
}

bool ResolveTensorBroadcastTargetShape(gui::NodeType type,
                                       const std::map<std::string, std::string>& params,
                                       const std::vector<size_t>& input_shape,
                                       std::vector<size_t>& target_shape,
                                       std::string& error) {
    const std::string op_name = type == gui::NodeType::TensorBroadcastTo
        ? "TensorBroadcastTo"
        : "TensorExpand";
    if (input_shape.empty()) {
        error = op_name + " requires a known input shape";
        return false;
    }
    if (!ParsePositiveShapeParam(params, "shape", op_name, target_shape, error)) {
        return false;
    }
    if (target_shape.size() < input_shape.size()) {
        error = op_name + " target sample rank must be >= input sample rank";
        return false;
    }

    const size_t sample_pad = target_shape.size() - input_shape.size();
    for (size_t axis = 0; axis < target_shape.size(); ++axis) {
        const size_t input_dim = axis < sample_pad ? 1 : input_shape[axis - sample_pad];
        const size_t target_dim = target_shape[axis];
        if (input_dim != 1 && input_dim != target_dim) {
            error = op_name + " target shape is not broadcast-compatible";
            return false;
        }
    }
    return true;
}

bool ResolveTensorIndexSelectTargetShape(const std::map<std::string, std::string>& params,
                                         const std::vector<size_t>& input_shape,
                                         std::vector<size_t>& target_shape,
                                         std::string& error) {
    if (input_shape.empty()) {
        error = "TensorIndexSelect requires a known input shape";
        return false;
    }

    int dim = 0;
    if (!ParseIntParam(params, "dim", 0, dim, error)) {
        error = "TensorIndexSelect " + error;
        return false;
    }

    int normalized_dim = 0;
    if (!NormalizeCompilerDim(dim,
                              static_cast<int>(input_shape.size()),
                              false,
                              "TensorIndexSelect",
                              normalized_dim,
                              error)) {
        return false;
    }

    std::vector<int> indices;
    if (!ParseIndexListParam(params, "indices", "TensorIndexSelect", indices, error)) {
        return false;
    }

    const int dim_size = static_cast<int>(input_shape[static_cast<size_t>(normalized_dim)]);
    for (int index : indices) {
        int normalized = index;
        if (normalized < 0) {
            normalized += dim_size;
        }
        if (normalized < 0 || normalized >= dim_size) {
            error = "TensorIndexSelect index is out of range";
            return false;
        }
    }

    target_shape = input_shape;
    target_shape[static_cast<size_t>(normalized_dim)] = indices.size();
    return true;
}

bool ResolveShapeOpTargetShape(gui::NodeType type,
                               const std::map<std::string, std::string>& params,
                               const std::vector<size_t>& input_shape,
                               std::vector<size_t>& target_shape,
                               std::string& error) {
    switch (type) {
        case gui::NodeType::Reshape:
        case gui::NodeType::View:
            return ResolveReshapeTargetShape(params, input_shape, target_shape, error);
        case gui::NodeType::Squeeze:
            return ResolveSqueezeTargetShape(params, input_shape, target_shape, error);
        case gui::NodeType::Unsqueeze:
            return ResolveUnsqueezeTargetShape(params, input_shape, target_shape, error);
        case gui::NodeType::Permute: {
            std::vector<int> ignored_dims;
            return ResolvePermuteTargetShape(params,
                                             input_shape,
                                             target_shape,
                                             ignored_dims,
                                             error);
        }
        case gui::NodeType::TensorBroadcastTo:
        case gui::NodeType::TensorExpand:
            return ResolveTensorBroadcastTargetShape(type,
                                                     params,
                                                     input_shape,
                                                     target_shape,
                                                     error);
        case gui::NodeType::TensorIndexSelect:
            return ResolveTensorIndexSelectTargetShape(params,
                                                       input_shape,
                                                       target_shape,
                                                       error);
        default:
            error = "Unsupported shape operation";
            return false;
    }
}

bool ResolveReductionTargetShape(gui::NodeType type,
                                 const std::map<std::string, std::string>& params,
                                 const std::vector<size_t>& input_shape,
                                 std::vector<size_t>& target_shape,
                                 std::string& error) {
    if (type != gui::NodeType::TensorSum &&
        type != gui::NodeType::TensorMean &&
        type != gui::NodeType::TensorMax &&
        type != gui::NodeType::TensorMin &&
        type != gui::NodeType::TensorProd &&
        type != gui::NodeType::TensorVar &&
        type != gui::NodeType::TensorStd) {
        error = "Unsupported tensor reduction";
        return false;
    }
    if (input_shape.empty()) {
        error = "Tensor reduction requires a known input shape";
        return false;
    }

    int dim = -1;
    if (!ParseIntParam(params, "dim", -1, dim, error)) {
        error = "Tensor reduction " + error;
        return false;
    }
    const bool keepdim = ParseBoolParam(params, "keepdim", false);

    target_shape.clear();
    if (dim == -1) {
        if (keepdim) {
            target_shape.assign(input_shape.size(), 1);
        } else {
            target_shape = {1};
        }
        return true;
    }

    if (dim < 0 || dim >= static_cast<int>(input_shape.size())) {
        error = "Tensor reduction dim is out of range";
        return false;
    }

    target_shape = input_shape;
    if (keepdim) {
        target_shape[static_cast<size_t>(dim)] = 1;
    } else {
        target_shape.erase(target_shape.begin() + dim);
        if (target_shape.empty()) {
            target_shape.push_back(1);
        }
    }
    return true;
}

bool ValidateTensorMaskParams(gui::NodeType type,
                              const std::map<std::string, std::string>& params,
                              std::string& error) {
    if (type == gui::NodeType::TensorCompare) {
        auto op_it = params.find("op");
        const std::string op = op_it != params.end() ? op_it->second : ">";
        if (op != ">" && op != ">=" && op != "<" &&
            op != "<=" && op != "==" && op != "!=") {
            error = "TensorCompare op must be one of >, >=, <, <=, ==, !=";
            return false;
        }

        float scalar = 0.0f;
        if (!ParseFloatParam(params, "scalar", 0.0f, scalar, error)) {
            error = "TensorCompare " + error;
            return false;
        }
        return true;
    }

    if (type == gui::NodeType::TensorLogicalMask) {
        auto op_it = params.find("op");
        const std::string op = op_it != params.end() ? op_it->second : "not";
        if (op != "not") {
            error = "TensorLogicalMask currently supports only op=not in the sequential runtime";
            return false;
        }
        return true;
    }

    error = "Unsupported tensor mask operation";
    return false;
}

void ValidateTrainingPathImplementationStatus(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    const std::unordered_set<int>& training_path_ids,
    TrainingConfiguration& config) {

    if (training_path_ids.empty()) {
        return;
    }

    auto& registry = NodeMetadataRegistry::Instance();
    if (!registry.IsInitialized()) {
        registry.Initialize();
    }

    for (const auto& node : nodes) {
        if (training_path_ids.count(node.id) == 0) {
            continue;
        }

        if (node.type == gui::NodeType::TransformerDecoder &&
            HasConnectedSelectedInputPinNamed(
                node,
                links,
                training_path_ids,
                "Memory")) {
            std::ostringstream msg;
            msg << "TransformerDecoder node '" << node.name
                << "' has a connected Memory input on the selected training "
                   "path, but Studio currently supports only decoder-only "
                   "causal self-attention through TransformerDecoderModule. "
                   "Seq2seq/cross-attention generation needs a graph-level "
                   "encoder-memory contract, shifted-token targets, causal "
                   "mask ownership, and inference/generation semantics before "
                   "it can compile truthfully.";
            AddIssue(config, IssueLevel::Error, msg.str(), node.id,
                     node.name, errors::Compiler::UnsupportedTrainingNode);
            continue;
        }

        if (node.type == gui::NodeType::MultiHeadAttention &&
            (HasConnectedSelectedInputPinNamed(
                 node,
                 links,
                 training_path_ids,
                 "Key") ||
             HasConnectedSelectedInputPinNamed(
                 node,
                 links,
                 training_path_ids,
                 "Value") ||
             HasConnectedSelectedInputPinNamed(
                 node,
                 links,
                 training_path_ids,
                 "Context") ||
             HasConnectedSelectedInputPinNamed(
                 node,
                 links,
                 training_path_ids,
                 "Mask"))) {
            std::ostringstream msg;
            msg << "MultiHeadAttention node '" << node.name
                << "' has a connected Key/Value/Context/Mask input on the "
                   "selected "
                   "training path, but Studio currently supports only "
                   "single-input self-attention through "
                   "MultiHeadAttentionModule. Cross-attention and masking "
                   "need a graph-level multi-input tensor contract, mask ownership, "
                   "export/load metadata, and inference semantics before it "
                   "can compile truthfully.";
            AddIssue(config, IssueLevel::Error, msg.str(), node.id,
                     node.name, errors::Compiler::UnsupportedTrainingNode);
            continue;
        }

        std::string target_design_key;
        if (gui::detail::IsDenseEncodedSequencePlaceholder(node,
                                                           target_design_key)) {
            std::ostringstream msg;
            msg << "Node '" << node.name
                << "' is encoded as Dense but contains target-design marker '"
                << target_design_key
                << "'. This graph needs first-class sequence/NER nodes and "
                   "cannot be compiled as a Dense layer.";
            AddIssue(config, IssueLevel::Error, msg.str(), node.id,
                     node.name, errors::Compiler::UnsupportedTrainingNode);
            continue;
        }

        std::string generative_key;
        if (LooksLikeGenerativeTrainingSketch(node, generative_key)) {
            std::ostringstream msg;
            msg << "Node '" << node.name
                << "' sketches decoder/generative training via '"
                << generative_key
                << "', but this is not the tested causal language-model "
                   "contract. Use an explicit TransformerDecoder stack with "
                   "shifted-token targets, token-level CrossEntropy, tokenizer "
                   "packaging, and the greedy generation path before marking "
                   "the graph as a supported generation workflow.";
            AddIssue(config, IssueLevel::Error, msg.str(), node.id,
                     node.name, errors::Compiler::UnsupportedTrainingNode);
            continue;
        }

        std::string fine_tuning_key;
        if (LooksLikeImportedFineTuningSketch(node, fine_tuning_key)) {
            std::ostringstream msg;
            msg << "Node '" << node.name
                << "' sketches imported/pretrained fine-tuning via '"
                << fine_tuning_key
                << "', but Studio training does not have a model-import to "
                   "training-graph contract. This path needs parameter "
                   "mapping, shape validation, freeze/unfreeze ownership, "
                   "optimizer-state compatibility, and tokenizer/preprocessor "
                   "packaging before it can compile truthfully.";
            AddIssue(config, IssueLevel::Error, msg.str(), node.id,
                     node.name, errors::Compiler::UnsupportedTrainingNode);
            continue;
        }

        std::string rl_key;
        if (LooksLikeRLTrainingSketch(node, rl_key)) {
            std::ostringstream msg;
            msg << "Node '" << node.name
                << "' sketches reinforcement-learning training via '"
                << rl_key
                << "', but Studio training is currently a supervised "
                   "single-batch executor. This path needs an environment "
                   "stepping loop, rollout/replay buffer schema, policy/value "
                   "loss contracts, target-network handling, and episodic "
                   "metrics before it can compile truthfully.";
            AddIssue(config, IssueLevel::Error, msg.str(), node.id,
                     node.name, errors::Compiler::UnsupportedTrainingNode);
            continue;
        }

        std::string detection_key;
        if (LooksLikeDetectionSegmentationTrainingSketch(node, detection_key)) {
            std::ostringstream msg;
            msg << "Node '" << node.name
                << "' sketches detection/segmentation training via '"
                << detection_key
                << "', but Studio training does not have a detection target "
                   "schema or multi-head loss contract. This path needs "
                   "box/mask/class target materialization, variable-object "
                   "batching, detection heads, loss aggregation, NMS/evaluation "
                   "metrics, and output packaging before it can compile "
                   "truthfully.";
            AddIssue(config, IssueLevel::Error, msg.str(), node.id, node.name);
            continue;
        }

        std::string time_distributed_key;
        if (LooksLikeTimeDistributedTrainingSketch(node,
                                                   time_distributed_key)) {
            std::ostringstream msg;
            msg << "Node '" << node.name
                << "' sketches per-timestep/per-token training via '"
                << time_distributed_key
                << "', but Studio training does not have a trainable "
                   "TimeDistributed wrapper. This path needs an inner-layer "
                   "binding, sequence-shape preservation, token-level loss "
                   "shape validation, padding ignore support, and per-token "
                   "metrics before it can compile truthfully.";
            AddIssue(config, IssueLevel::Error, msg.str(), node.id, node.name);
            continue;
        }

        std::string reconstruction_key;
        if (LooksLikeReconstructionGenerativeTrainingSketch(
                node,
                reconstruction_key)) {
            std::ostringstream msg;
            msg << "Node '" << node.name
                << "' sketches autoencoder/VAE/GAN/diffusion training via '"
                << reconstruction_key
                << "', but Studio training currently has a single supervised "
                   "optimizer step. This path needs reconstruction-target "
                   "routing, latent KL-loss contracts, alternating optimizer "
                   "or adversarial-step orchestration, diffusion noise "
                   "schedules, and generation output packaging before it can "
                   "compile truthfully.";
            AddIssue(config, IssueLevel::Error, msg.str(), node.id, node.name);
            continue;
        }

        std::string metric_learning_key;
        if (LooksLikeMetricLearningTrainingSketch(node,
                                                  metric_learning_key)) {
            std::ostringstream msg;
            const auto support =
                ResolvePipelineTrainingBackendSupport(node.type);
            if (support.mode == PipelineTrainingBackendSupportMode::
                                    UnsupportedTrainingWorkflow &&
                support.reason != nullptr) {
                msg << "Node '" << node.name << "' " << support.reason;
            } else {
                msg << "Node '" << node.name
                    << "' sketches metric-learning/Siamese training via '"
                    << metric_learning_key
                    << "', but no typed visual graph or TrainingExecutor "
                       "owner is registered for the workflow.";
            }
            AddIssue(config, IssueLevel::Error, msg.str(), node.id, node.name,
                     errors::Compiler::UnsupportedTrainingNode);
            continue;
        }

        std::string gnn_key;
        if (LooksLikeGNNTrainingSketch(node, gnn_key)) {
            std::ostringstream msg;
            msg << "Node '" << node.name
                << "' sketches graph-neural-network/GNN training via '"
                << gnn_key
                << "', but Studio training does not have a graph batch "
                   "contract. This path needs graph batch schemas, "
                   "edge-index/adjacency routing, message-passing kernels, "
                   "node/edge/graph target contracts, neighborhood batching, "
                   "and graph-level output packaging before it can compile "
                   "truthfully.";
            AddIssue(config, IssueLevel::Error, msg.str(), node.id, node.name);
            continue;
        }

        const auto training_support =
            ResolvePipelineTrainingBackendSupport(node.type);
        if (!training_support.compile_supported &&
            training_support.mode ==
                PipelineTrainingBackendSupportMode::UnsupportedSequentialModelLayer) {
            std::ostringstream msg;
            msg << "Node '" << node.name << "' "
                << training_support.reason;
            AddIssue(config, IssueLevel::Error, msg.str(), node.id,
                     node.name, errors::Compiler::UnsupportedTrainingNode);
            continue;
        }

        const NodeMetadata* metadata = registry.GetMetadata(node.type);
        if (!metadata) {
            continue;
        }
        if (metadata->status == NodeImplementationStatus::Implemented) {
            continue;
        }
        if (IsGraphRuntimeFanInOp(node.type)) {
            continue;
        }

        std::ostringstream msg;
        msg << "Node '" << node.name << "' is "
            << ImplementationStatusLabel(metadata->status)
            << " and cannot run in the training path";
        if (!metadata->brief_description.empty()) {
            msg << " (" << metadata->brief_description << ")";
        }
        AddIssue(config, IssueLevel::Error, msg.str(), node.id,
                 node.name, errors::Compiler::UnsupportedTrainingNode);
    }
}

void ValidateUnsupportedTrainingControlNodes(
    const std::vector<gui::MLNode>& nodes,
    TrainingConfiguration& config) {

    for (const auto& node : nodes) {
        const auto training_support =
            ResolvePipelineTrainingBackendSupport(node.type);
        if (training_support.compile_supported ||
            training_support.mode !=
                PipelineTrainingBackendSupportMode::UnsupportedTrainingControl) {
            continue;
        }

        std::ostringstream msg;
        msg << "Node '" << node.name << "' is " << training_support.reason;
        AddIssue(config, IssueLevel::Error, msg.str(), node.id,
                 node.name, errors::Compiler::UnsupportedTrainingNode);
    }
}

void CollectGraphRuntimeOpNodeIds(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    const std::vector<int>& sorted_node_ids,
    const std::unordered_set<int>& training_path_ids,
    TrainingConfiguration& config) {
    if (training_path_ids.empty()) {
        return;
    }

    std::unordered_map<int, const gui::MLNode*> by_id;
    by_id.reserve(nodes.size());
    for (const auto& node : nodes) {
        by_id[node.id] = &node;
    }

    for (int node_id : sorted_node_ids) {
        if (training_path_ids.count(node_id) == 0) {
            continue;
        }

        auto it = by_id.find(node_id);
        if (it == by_id.end()) {
            continue;
        }
        const gui::MLNode& node = *it->second;
        if (!IsGraphRuntimeFanInOp(node.type)) {
            continue;
        }

        const int connected_inputs =
            CountConnectedSelectedTensorInputs(node, links, training_path_ids);
        if (connected_inputs < 2) {
            std::ostringstream msg;
            msg << "Graph runtime node '" << node.name
                << "' requires at least two connected tensor inputs";
            AddIssue(config, IssueLevel::Error, msg.str(), node.id, node.name);
            continue;
        }
        if ((IsGraphRuntimeBinaryMaskOp(node.type) ||
             IsGraphRuntimeLinalgOp(node.type)) &&
            connected_inputs != 2) {
            std::ostringstream msg;
            msg << "Graph runtime node '" << node.name
                << "' requires exactly two connected tensor inputs";
            AddIssue(config, IssueLevel::Error, msg.str(), node.id, node.name);
            continue;
        }
        if (IsGraphRuntimeBinaryMaskOp(node.type)) {
            auto op_it = node.parameters.find("op");
            const std::string op = op_it != node.parameters.end()
                ? op_it->second
                : (node.type == gui::NodeType::TensorCompare ? ">" : "and");
            if (node.type == gui::NodeType::TensorCompare &&
                op != ">" && op != ">=" && op != "<" &&
                op != "<=" && op != "==" && op != "!=") {
                AddIssue(config, IssueLevel::Error,
                         "TensorCompare graph op must use one of >, >=, <, <=, ==, !=",
                         node.id, node.name);
                continue;
            }
            if (node.type == gui::NodeType::TensorLogicalMask &&
                op != "and" && op != "or") {
                AddIssue(config, IssueLevel::Error,
                         "TensorLogicalMask graph op supports only op=and or op=or with two inputs",
                         node.id, node.name);
                continue;
            }
        }

        // Runtime tensor-shape checks remain in GraphExecutableModel, where
        // concrete shapes and broadcast/concat dimensions are known. The
        // compiler only records deliberately-enabled graph ops here.
        config.graph_op_node_ids.push_back(node.id);
    }
}
} // anonymous namespace

TrainingConfiguration GraphCompiler::Compile(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    bool allow_unloaded_data,
    const std::string& placement_observation_cache_path)
{
    TrainingConfiguration config;

    // === Structural validation ===
    // Collect all structural errors at once (the old ValidateGraph stopped
    // at the first one). is_valid is determined at the end of Compile by
    // checking whether config.issues contains any Error-level entries.
    const gui::MLNode* dataset_node = FindDatasetInputNode(nodes, links);
    const std::unordered_set<int> dataset_reachable =
        dataset_node
            ? CollectReachableNodeIds(dataset_node->id, links)
            : std::unordered_set<int>{};
    const gui::MLNode* loss_node = FindFirstLossNodeInSet(nodes, dataset_reachable);
    if (!loss_node) {
        loss_node = FindLossNode(nodes);
    }
    const gui::MLNode* optimizer_node = FindOptimizerNode(nodes);
    std::unordered_set<int> training_path_ids;
    if (dataset_node && loss_node) {
        const auto loss_ancestors = CollectAncestorNodeIds(loss_node->id, links);
        for (int node_id : dataset_reachable) {
            if (loss_ancestors.count(node_id) > 0) {
                training_path_ids.insert(node_id);
            }
        }
    }
    config.target = InferTargetContract(
        nodes, training_path_ids, loss_node);
    if (config.target.IsGeneratedByGraph()) {
        spdlog::info(
            "GraphCompiler: target contract resolved from graph node '{}' "
            "(column='{}', width={}); a raw Data Input label is not required",
            config.target.producer_node_name,
            config.target.primary_column,
            config.target.width);
    }

    if (nodes.empty()) {
        AddIssue(config,
                 IssueLevel::Error,
                 "Graph is empty - add nodes to create a model",
                 -1,
                 "",
                 errors::Compiler::MissingTrainingPathNode);
    } else {
        if (!dataset_node) {
            AddIssue(config, IssueLevel::Error,
                     "Graph must have a DataInput or DatasetInput node",
                     -1,
                     "",
                     errors::Compiler::MissingTrainingPathNode);
        }
        if (!loss_node) {
            AddIssue(config, IssueLevel::Error,
                     "Graph must have a loss function (MSELoss, CrossEntropyLoss, etc.)",
                     -1,
                     "",
                     errors::Compiler::MissingTrainingPathNode);
        }
        if (!optimizer_node) {
            AddIssue(config, IssueLevel::Error,
                     "Graph must have an optimizer (SGD, Adam, AdamW, RMSprop, Adagrad, or NAdam)",
                     -1,
                     "",
                     errors::Compiler::MissingTrainingPathNode);
        }
        bool has_model_layer = false;
        for (const auto& node : nodes) {
            if (IsModelLayer(node.type)) { has_model_layer = true; break; }
        }
        if (!has_model_layer) {
            AddIssue(config, IssueLevel::Error,
                     "Graph must have at least one model layer (Dense, Conv2D, etc.)",
                     -1,
                     "",
                     errors::Compiler::MissingTrainingPathNode);
        }
        if (HasCycle(nodes, links)) {
            AddIssue(config, IssueLevel::Error,
                     "Graph contains a cycle - remove circular connections",
                     -1,
                     "",
                     errors::Compiler::InvalidConnectivity);
        }

        // Pin-connectivity checks. The runtime currently ignores pin
        // topology (reads dataset_name from registry instead), so a
        // graph with a disconnected Loss.Targets pin will train
        // anyway. These checks make the canvas the source of truth at
        // compile time so users can't ship a visually-broken graph.
        ValidateRequiredInputsConnected(nodes, links, config);
        ValidateRequiredOutputsConnected(nodes, links, config);
        ValidateLossTargetsReachLabels(nodes, links, config);
        ValidateLossPredictionsReachModel(nodes, links, config);
        ValidateOptimizerReachesLoss(nodes, links, config);
        ValidateSingleDatasetReachableLossNode(nodes, dataset_reachable, config);
        ValidateSingleDatasetSourceForSelectedLoss(nodes, loss_node, links, config);
        ValidateUnsupportedTrainingControlNodes(nodes, config);

        if (dataset_node && !HasReachablePreTrainInspectionNode(nodes, dataset_reachable)) {
            AddIssue(config, IssueLevel::Warning,
                     "No pre-train data inspection node found - consider adding "
                     "DataProfiler, DescribeStats, ValueCounts, SampleRows, "
                     "CorrelationMatrix, or DataValidator before training to check "
                     "missing values, class balance, column types, and label suitability",
                     dataset_node->id, dataset_node->name);
        }
    }

    // Extract dataset configuration
    if (dataset_node) {
        // dataset_name parameter is what DataInputDialog::Apply writes when
        // a dataset is loaded. The legacy "dataset" key was used by the
        // older DatasetInput dialog and is now obsolete — fall back to it
        // only if dataset_name is empty, so older project files still load.
        config.dataset_name = DatasetNameFromNode(*dataset_node);
        config.dataset_roles.train = DatasetSourceFromNode(
            *dataset_node, /*externally_supplied=*/true);

        const gui::MLNode* role_split = FindFirstReachableNodeOfType(
            nodes, dataset_reachable, gui::NodeType::DataSplit);
        if (role_split) {
            bool train_has_connection = false;
            bool train_has_multiple_sources = false;
            const auto* topology_train = FindDatasetSourceConnectedToSplitInput(
                nodes, links, *role_split, "Training Dataset",
                &train_has_connection, &train_has_multiple_sources);
            if (train_has_multiple_sources) {
                AddIssue(config, IssueLevel::Error,
                         "Data Split 'Training Dataset' accepts exactly one "
                         "Dataset source",
                         role_split->id, role_split->name,
                         errors::Compiler::InvalidConnectivity);
            } else if (train_has_connection && !topology_train) {
                AddIssue(config, IssueLevel::Error,
                         "Training role must originate from one Data Input "
                         "Dataset output",
                         role_split->id, role_split->name,
                         errors::Compiler::InvalidConnectivity);
            } else if (topology_train && topology_train->id != dataset_node->id) {
                AddIssue(config, IssueLevel::Error,
                         "Selected Training Dataset does not match the source "
                         "connected to Data Split",
                         role_split->id, role_split->name,
                         errors::Compiler::InvalidConnectivity);
            }

            const auto resolve_optional_role = [&](
                const char* pin_name,
                const char* role_name,
                DatasetSourceRef& output) {
                bool has_connection = false;
                bool has_multiple_sources = false;
                const auto* source = FindDatasetSourceConnectedToSplitInput(
                    nodes, links, *role_split, pin_name,
                    &has_connection, &has_multiple_sources);
                if (has_multiple_sources) {
                    AddIssue(config, IssueLevel::Error,
                             std::string("Data Split '") + pin_name +
                                 "' accepts exactly one Dataset source",
                             role_split->id, role_split->name,
                             errors::Compiler::InvalidConnectivity);
                    return;
                }
                if (has_connection && !source) {
                    AddIssue(config, IssueLevel::Error,
                             std::string(role_name) +
                                 " role must originate from one Data Input "
                                 "Dataset output",
                             role_split->id, role_split->name,
                             errors::Compiler::InvalidConnectivity);
                    return;
                }
                if (source) {
                    output = DatasetSourceFromNode(
                        *source, /*externally_supplied=*/true);
                }
            };
            resolve_optional_role(
                "Validation Dataset", "Validation", config.dataset_roles.dev);
            resolve_optional_role(
                "Test Dataset", "Test", config.dataset_roles.test);
        }

        for (const auto& node : nodes) {
            if (!IsDatasetSourceType(node.type)) continue;
            const std::string legacy_role = DatasetRoleFromNode(node);
            if (legacy_role.empty()) continue;
            AddIssue(
                config,
                IssueLevel::Info,
                "Legacy Data Input dataset_role='" + legacy_role +
                    "' is ignored. Dataset roles are assigned by the named "
                    "Training/Validation/Test inputs on Data Split.",
                node.id,
                node.name);
        }

        // === New error checks tied to the dataset node ===
        //
        // Registry is the source of truth for "is data loaded", NOT the
        // `data_loaded` param hint on the node. The hint is cached dialog
        // state that can go stale when an async Apply completes after the
        // dialog has already closed — PollAsyncLoadResult only runs while
        // the dialog is visible, so the provisional "false" that Apply
        // sets before launching the worker gets stuck. This mirrors the
        // DataInputDialog constructor's design, which probes the registry
        // directly instead of trusting the param hint (see CLAUDE.md under
        // "Async load + UX").
        //
        // The decision matrix is:
        //   in_registry=yes → data is loaded, proceed (ignore stale hint)
        //   in_registry=no, dataset_name empty → never-applied, block
        //   in_registry=no, dataset_name set, hint=true → registry wiped,
        //     "re-apply" error
        //   in_registry=no, dataset_name set, hint=false → never-applied
        //     (or async still running at compile time), block
        const std::string loaded_param = dataset_node->parameters.count("data_loaded")
            ? dataset_node->parameters.at("data_loaded")
            : std::string("false");

        // Registry probe via the loader factory. Each loader checks its
        // own registry map (tabular: Arrow+Parquet, image/audio/text:
        // their entries). GetByRegisteredDataset returns non-null iff
        // ANY loader claims the name — same semantics as the 5-way OR
        // this replaces.
        const bool in_registry = !config.dataset_name.empty() &&
            loaders::GetByRegisteredDataset(config.dataset_name) != nullptr;

        if (in_registry) {
            // Data IS loaded, regardless of the hint. Log when we had to
            // override a stale hint so it's traceable at training time.
            if (loaded_param != "true") {
                spdlog::info("GraphCompiler: dataset '{}' found in registry "
                             "despite data_loaded='{}' hint (stale from "
                             "async Apply after dialog close) - proceeding",
                             config.dataset_name, loaded_param);
            }
        } else if (allow_unloaded_data) {
            spdlog::info("GraphCompiler: dataset '{}' not in registry, but "
                         "compile is running in deployment mode - skipping "
                         "data_loaded validation",
                         config.dataset_name);
        } else if (config.dataset_name.empty()) {
            if (loaded_param == "true") {
                AddIssue(config, IssueLevel::Error,
                         "Data marked loaded but no dataset_name parameter set",
                         dataset_node->id, dataset_node->name);
            } else {
                AddIssue(config, IssueLevel::Error,
                         "Data is not loaded - open the node and click Apply",
                         dataset_node->id, dataset_node->name);
            }
        } else {
            // dataset_name is set but registry has nothing under it.
            if (loaded_param == "true") {
                AddIssue(config, IssueLevel::Error,
                         "Dataset '" + config.dataset_name +
                         "' is marked loaded but missing from registry - "
                         "re-apply the DataInput node",
                         dataset_node->id, dataset_node->name);
            } else {
                AddIssue(config, IssueLevel::Error,
                         "Data is not loaded - open the node and click Apply",
                         dataset_node->id, dataset_node->name);
            }
        }

        // Check 3: label column. Required for tabular supervised training,
        // but image / audio / text datasets get labels from folder
        // structure or an explicit dialog column, not from a flag on
        // the DataInput node — don't emit a misleading warning for them.
        // Each loader reports LabelsFromStructure() — Tabular returns
        // false; Image/Audio/Text return true.
        auto cat_it_lbl = dataset_node->parameters.find("file_category");
        const std::string cat_str = (cat_it_lbl != dataset_node->parameters.end())
            ? cat_it_lbl->second : std::string();
        bool labels_from_structure = false;
        if (auto* cat_loader = loaders::GetByCategory(
                loaders::FileCategoryFromString(cat_str))) {
            labels_from_structure = cat_loader->LabelsFromStructure();
        }

        const std::string requested_label_col = dataset_node->parameters.count("label_column")
            ? dataset_node->parameters.at("label_column")
            : std::string();

        std::shared_ptr<arrow::Schema> tabular_schema;
        auto& data_registry = DataRegistry::Instance();
        if (auto arrow_dataset =
                data_registry.GetArrowDataset(config.dataset_name)) {
            tabular_schema = arrow_dataset->GetSchema();
        } else if (auto parquet_dataset =
                       data_registry.GetParquetBackedDataset(config.dataset_name)) {
            tabular_schema = parquet_dataset->GetSchema();
        }

        const int resolved_label_index = ResolveLabelColumnIndex(
            tabular_schema, requested_label_col);
        const std::string resolved_label_col = resolved_label_index >= 0
            ? tabular_schema->field(resolved_label_index)->name()
            : std::string{};
        config.dataset_roles.train.label_column = resolved_label_col;

        if (!config.target.IsGeneratedByGraph()) {
            if (!resolved_label_col.empty()) {
                config.target.origin = TargetOrigin::DatasetColumn;
                config.target.producer_node_id = dataset_node->id;
                config.target.producer_node_name = dataset_node->name;
                config.target.primary_column = resolved_label_col;
                config.target.width = 1;
            } else if (labels_from_structure) {
                config.target.origin = TargetOrigin::DatasetStructure;
                config.target.producer_node_id = dataset_node->id;
                config.target.producer_node_name = dataset_node->name;
                config.target.width = 1;
            }
        }

        if (!requested_label_col.empty() && tabular_schema &&
            resolved_label_index < 0 && !labels_from_structure) {
            if (config.target.IsGeneratedByGraph()) {
                AddIssue(config, IssueLevel::Warning,
                         "Selected source label column '" + requested_label_col +
                             "' is not present in registered dataset '" +
                             config.dataset_name + "', but it is not used because "
                             "node '" + config.target.producer_node_name +
                             "' generates the objective targets",
                         dataset_node->id, dataset_node->name);
            } else {
                AddIssue(config,
                         config.target.required_by_objective
                             ? IssueLevel::Error
                             : IssueLevel::Warning,
                         "Selected label column '" + requested_label_col +
                             "' is not present in registered dataset '" +
                             config.dataset_name + "'",
                         dataset_node->id, dataset_node->name,
                         config.target.required_by_objective
                             ? errors::Data::RequiredLabelColumnMissing
                             : "");
            }
        } else if (requested_label_col.empty() &&
                   !resolved_label_col.empty()) {
            AddIssue(config, IssueLevel::Info,
                     "Auto-resolved label column '" + resolved_label_col +
                         "' from the registered dataset schema",
                     dataset_node->id, dataset_node->name);
        } else if (requested_label_col.empty() && !labels_from_structure &&
                   config.target.required_by_objective &&
                   !config.target.IsResolved()) {
            const IssueLevel level = tabular_schema
                ? IssueLevel::Error
                : IssueLevel::Warning;
            AddIssue(config, level,
                     tabular_schema
                         ? "No label column selected and no common label name "
                           "was found in the registered dataset schema"
                         : "No label column selected - apply the Data Input "
                           "node so the compiler can inspect its schema",
                     dataset_node->id, dataset_node->name,
                     tabular_schema
                         ? errors::Data::RequiredLabelColumnMissing
                         : "");
        }

        // Extract input shape from dataset node
        if (dataset_node->parameters.count("shape")) {
            // Parse shape string like "[28, 28, 1]"
            std::string shape_str = dataset_node->parameters.at("shape");
            // Simple parsing - remove brackets and split by comma
            shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), '['), shape_str.end());
            shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ']'), shape_str.end());
            shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ' '), shape_str.end());

            size_t pos = 0;
            while ((pos = shape_str.find(',')) != std::string::npos) {
                config.input_shape.push_back(std::stoul(shape_str.substr(0, pos)));
                shape_str.erase(0, pos + 1);
            }
            if (!shape_str.empty()) {
                config.input_shape.push_back(std::stoul(shape_str));
            }
        }

        // Calculate input size. Plain datasets use flattened feature width.
        // Sequence language-model datasets keep input_size as per-token
        // feature width while input_shape preserves [seq_len, features] for
        // downstream sequence wrappers such as TimeDistributed.
        const bool sequence_lm_input =
            ParseBoolParam(dataset_node->parameters,
                           "create_causal_lm_targets",
                           false) ||
            ParseBoolParam(dataset_node->parameters,
                           "sequence_create_causal_lm_targets",
                           false);
        size_t sequence_length =
            ParseSizeParam(dataset_node->parameters,
                           "max_sequence_length",
                           0);
        if (sequence_length == 0) {
            sequence_length =
                ParseSizeParam(dataset_node->parameters,
                               "sequence_max_sequence_length",
                               0);
        }
        if (sequence_lm_input && !config.input_shape.empty()) {
            if (config.input_shape.size() == 1 && sequence_length > 0) {
                const size_t feature_width = config.input_shape[0];
                config.input_shape = {sequence_length, feature_width};
                config.input_size = feature_width;
            } else if (config.input_shape.size() >= 2) {
                config.input_size = config.input_shape.back();
            } else {
                config.input_size = config.input_shape[0];
            }
        } else {
            config.input_size = 1;
            for (size_t dim : config.input_shape) {
                config.input_size *= dim;
            }
        }

        // Tabular Data Input nodes do not need a manually-entered shape.
        // Derive the exact feature width from the same numeric-column
        // contract used by Arrow batchers. This prevents a silent
        // Linear(1, ...) fallback followed by a native dimension mismatch
        // when the first real batch contains many columns.
        if (tabular_schema && config.input_shape.empty() &&
            resolved_label_index >= 0) {
            const size_t feature_count = CountNumericBatchFeatureColumns(
                tabular_schema, resolved_label_index);
            if (feature_count == 0) {
                AddIssue(config, IssueLevel::Error,
                         "Registered dataset '" + config.dataset_name +
                             "' has no numeric feature columns after excluding "
                             "label column '" + resolved_label_col + "'",
                         dataset_node->id, dataset_node->name,
                         errors::Data::ColumnTypeMismatch);
            } else {
                config.input_shape = {feature_count};
                config.input_size = feature_count;
                spdlog::info(
                    "GraphCompiler: derived tabular input width {} from "
                    "dataset '{}' (label='{}')",
                    feature_count, config.dataset_name, resolved_label_col);
            }
        }
    }

    // Extract split ratios from the DataSplit node on the selected data path.
    if (const gui::MLNode* split_node = FindFirstReachableNodeOfType(
            nodes, dataset_reachable, gui::NodeType::DataSplit)) {
        config.has_data_split = true;
        AddIssue(config,
                 IssueLevel::Info,
                 "Data Split resolves its named Dataset inputs into one "
                 "Train/Validation/Test partition contract for Data Loader.",
                 split_node->id,
                 split_node->name);
        try {
            if (split_node->parameters.count("train_ratio"))
                config.train_ratio = std::stof(split_node->parameters.at("train_ratio"));
            if (split_node->parameters.count("val_ratio"))
                config.val_ratio = std::stof(split_node->parameters.at("val_ratio"));
            if (split_node->parameters.count("test_ratio"))
                config.test_ratio = std::stof(split_node->parameters.at("test_ratio"));
            if (split_node->parameters.count("seed"))
                config.split_seed = std::stoi(split_node->parameters.at("seed"));
            if (split_node->parameters.count("stratified"))
                config.stratified =
                    IsTruthyParameterValue(split_node->parameters.at("stratified"));
        } catch (const std::exception& e) {
            spdlog::warn("GraphCompiler: DataSplit param parse error ({}) - using defaults", e.what());
        }
    }
    if (config.has_data_split &&
        (config.dataset_roles.dev.IsSupplied() ||
         config.dataset_roles.test.IsSupplied())) {
        if (config.dataset_roles.dev.IsSupplied()) {
            config.val_ratio = 0.0f;
        }
        if (config.dataset_roles.test.IsSupplied()) {
            config.test_ratio = 0.0f;
        }
        config.train_ratio =
            std::max(0.0f, 1.0f - config.val_ratio - config.test_ratio);
        spdlog::info(
            "GraphCompiler: resolved supplied dataset roles - "
            "derived train={:.2f}, val={:.2f}, test={:.2f}; "
            "external dev={}, external test={}",
            config.train_ratio, config.val_ratio, config.test_ratio,
            config.dataset_roles.dev.IsSupplied(),
            config.dataset_roles.test.IsSupplied());
    }
    if (config.has_data_split) {
        spdlog::info("GraphCompiler: DataSplit node found - train={:.2f}, val={:.2f}, test={:.2f}, seed={}, stratified={}",
                     config.train_ratio, config.val_ratio, config.test_ratio,
                     config.split_seed, config.stratified);
    } else {
        spdlog::info("GraphCompiler: No DataSplit node - using defaults (train=0.80, val=0.10, test=0.10)");
    }

    // Extract training-loop config from DataLoader node if present.
    // DataLoader owns batch_size / epochs / shuffle / drop_last / num_workers —
    // all the "how do I iterate training" hyperparameters.
    if (const gui::MLNode* loader_node = FindFirstReachableNodeOfType(
            nodes, dataset_reachable, gui::NodeType::DataLoader)) {
        config.has_data_loader = true;
        try {
            if (loader_node->parameters.count("batch_size"))
                config.batch_size = std::stoi(loader_node->parameters.at("batch_size"));
            if (loader_node->parameters.count("epochs"))
                config.epochs = std::stoi(loader_node->parameters.at("epochs"));
            if (loader_node->parameters.count("shuffle"))
                config.shuffle = (loader_node->parameters.at("shuffle") == "true");
            if (loader_node->parameters.count("drop_last"))
                config.drop_last = (loader_node->parameters.at("drop_last") == "true");
            if (loader_node->parameters.count("num_workers"))
                config.num_workers = std::stoi(loader_node->parameters.at("num_workers"));
            const int requested_workers = config.num_workers;
            config.num_workers = ClampNumWorkersToPlatform(config.num_workers);
            if (config.num_workers != requested_workers) {
                spdlog::warn("GraphCompiler: clamping DataLoader num_workers from {} to {} based on platform",
                             requested_workers, config.num_workers);
            }
            if (loader_node->parameters.count("prefetch_factor")) {
                config.prefetch_factor = std::max(0, std::stoi(loader_node->parameters.at("prefetch_factor")));
                spdlog::info("GraphCompiler: DataLoader prefetch_factor={} will enable a bounded "
                             "async batch queue on supported Arrow/Parquet batchers; num_workers={} "
                             "still controls synchronous per-batch conversion inside each fetch",
                             config.prefetch_factor, config.num_workers);
            }
            if (loader_node->parameters.count("log_interval"))
                config.log_interval = std::max(0, std::stoi(loader_node->parameters.at("log_interval")));
            if (loader_node->parameters.count("validation_freq"))
                config.validation_freq = std::max(1, std::stoi(loader_node->parameters.at("validation_freq")));
            if (loader_node->parameters.count("seed"))
                config.dataloader_seed = std::max(0, std::stoi(loader_node->parameters.at("seed")));
            if (loader_node->parameters.count(
                    training_contract::kGradientAccumulationStepsKey)) {
                config.grad_accum_steps =
                    training_contract::ClampGradientAccumulationSteps(
                        std::stoi(loader_node->parameters.at(
                            training_contract::kGradientAccumulationStepsKey)));
            }
            if (loader_node->parameters.count("balance_classes"))
                config.balance_classes =
                    IsTruthyParameterValue(loader_node->parameters.at("balance_classes"));
            if (loader_node->parameters.count("balance_mode")) {
                config.balance_mode =
                    ToLowerAscii(loader_node->parameters.at("balance_mode"));
                if (config.balance_mode == "weighted sampler") {
                    config.balance_mode = "weighted_sampler";
                }
            }
            if (loader_node->parameters.count("weighted_sampler") &&
                IsTruthyParameterValue(loader_node->parameters.at("weighted_sampler"))) {
                config.balance_classes = true;
                config.balance_mode = "weighted_sampler";
            }
            if (loader_node->parameters.count("oversample") &&
                IsTruthyParameterValue(loader_node->parameters.at("oversample"))) {
                config.balance_classes = true;
                config.balance_mode = "oversample";
            }
            if (loader_node->parameters.count("undersample") &&
                IsTruthyParameterValue(loader_node->parameters.at("undersample"))) {
                config.balance_classes = true;
                config.balance_mode = "undersample";
            }
            if (loader_node->parameters.count("balance_target"))
                config.balance_target =
                    ToLowerAscii(loader_node->parameters.at("balance_target"));
            if (loader_node->parameters.count("balance_seed"))
                config.balance_seed = std::max(
                    0, std::stoi(loader_node->parameters.at("balance_seed")));
            if (config.balance_classes &&
                (config.balance_mode.empty() || config.balance_mode == "none")) {
                config.balance_mode = "oversample";
            }
            config.pin_memory_transfer.node_id = loader_node->id;
            config.pin_memory_transfer.node_name = loader_node->name;
            config.pin_memory_transfer.batch_size = config.batch_size;
            if (loader_node->parameters.count("pin_memory") &&
                loader_node->parameters.at("pin_memory") == "true") {
                config.pin_memory_transfer.requested = true;
            }
            if (loader_node->parameters.count("save_best_checkpoint"))
                config.save_best_checkpoint = (loader_node->parameters.at("save_best_checkpoint") == "true");
            if (loader_node->parameters.count("early_stopping_patience"))
                config.early_stopping_patience = std::stoi(loader_node->parameters.at("early_stopping_patience"));
            if (loader_node->parameters.count("checkpoint_dir"))
                config.checkpoint_dir = loader_node->parameters.at("checkpoint_dir");
        } catch (const std::exception& e) {
            spdlog::warn("GraphCompiler: DataLoader param parse error ({}) - using defaults", e.what());
        }
    }
    if (config.has_data_loader) {
        spdlog::info("GraphCompiler: DataLoader node found - batch_size={}, epochs={}, shuffle={}, drop_last={}, num_workers={}, prefetch_factor={}, log_interval={}, validation_freq={}, seed={}, grad_accum_steps={}, balance_classes={}, balance_mode='{}', balance_target='{}', balance_seed={}, save_best_checkpoint={}, early_stopping_patience={}, checkpoint_dir='{}'",
                     config.batch_size, config.epochs, config.shuffle, config.drop_last, config.num_workers, config.prefetch_factor,
                     config.log_interval, config.validation_freq, config.dataloader_seed,
                     config.grad_accum_steps, config.balance_classes,
                     config.balance_mode, config.balance_target,
                     config.balance_seed, config.save_best_checkpoint, config.early_stopping_patience,
                     config.checkpoint_dir);
        if (config.num_workers > 0) {
            spdlog::info("GraphCompiler: num_workers={} will be forwarded to supported batchers",
                         config.num_workers);
        }
    } else {
        spdlog::info("GraphCompiler: No DataLoader node - using defaults (batch_size=32, epochs=10, shuffle=true, drop_last=false, log_interval=10, validation_freq=1, seed=42, grad_accum_steps=1)");
    }

    if (loss_node) {
        const auto loss_reachable = CollectReachableNodeIds(loss_node->id, links);
        if (const gui::MLNode* path_optimizer_node =
                FindFirstOptimizerNodeInSet(nodes, loss_reachable)) {
            optimizer_node = path_optimizer_node;
        }
    }
    ExtractSequenceBatchContract(nodes, training_path_ids, config);
    if (loss_node && config.sequence_batch.enabled) {
        ExtractSequenceBatchContractFromNode(*loss_node, config.sequence_batch);
    }
    if (config.sequence_batch.enabled) {
        for (const auto& node : nodes) {
            if (node.type == gui::NodeType::TextPadding) {
                ExtractSequenceBatchContractFromNode(node, config.sequence_batch);
            }
        }
    }
    ValidateTrainingPathImplementationStatus(nodes, links, training_path_ids, config);

    if (dataset_node) {
        config.data_source_node_id = dataset_node->id;
    }
    if (loss_node) {
        config.loss_node_id = loss_node->id;
    }
    if (optimizer_node) {
        config.optimizer_node_id = optimizer_node->id;
    }

    // Phase 4 Time-Series detection. If the graph contains a
    // TimeSeriesWindow node, mark the config so the training dispatch
    // routes the Arrow batcher in regression mode (float labels, no
    // one-hot) and drives index selection from the __partition__ column
    // emitted by TimeSeriesSplitOperator. The downstream batcher is the
    // same ArrowDatasetBatcher used by tabular classification — only
    // the constructor args and SetRegressionMode switch differ.
    //
    // Also override config.input_size to match the operator's
    // input_width. The raw CSV column count is meaningless once
    // TimeSeriesWindow materializes the table — the first Dense layer
    // receives `input_width` features (x_0..x_{input_width-1}), not
    // the original per-row scalar. Without this override
    // BuildSequentialFromConfig would construct Linear(1 -> hidden_units)
    // and crash at the first forward pass with a dimension mismatch.
    for (const auto& node : nodes) {
        if (node.type == gui::NodeType::TimeSeriesWindow &&
            ContainsWhenFiltered(training_path_ids, node.id)) {
            config.is_time_series = true;
            int input_width = 12;
            auto iw_it = node.parameters.find("input_width");
            if (iw_it != node.parameters.end() && !iw_it->second.empty()) {
                try { input_width = std::stoi(iw_it->second); }
                catch (...) { /* fall back to default */ }
            }
            int label_width = 1;
            auto lw_it = node.parameters.find("label_width");
            if (lw_it != node.parameters.end() && !lw_it->second.empty()) {
                try { label_width = std::stoi(lw_it->second); }
                catch (...) { /* runtime parameter validation reports it */ }
            }
            if (label_width > 0) {
                config.time_series_target_width =
                    static_cast<size_t>(label_width);
            }
            // Multivariate: input_size = input_width * (1 + num_feature_cols).
            // feature_cols is comma-separated; count non-empty tokens.
            // Single-variate (empty feature_cols) keeps input_size = input_width.
            int num_features = 1;
            auto fc_it = node.parameters.find("feature_cols");
            if (fc_it != node.parameters.end() && !fc_it->second.empty()) {
                int extras = 0;
                std::stringstream ss(fc_it->second);
                std::string token;
                while (std::getline(ss, token, ',')) {
                    size_t start = token.find_first_not_of(" \t");
                    if (start != std::string::npos) ++extras;
                }
                num_features = 1 + extras;
            }
            const int total_input = input_width * num_features;
            if (total_input > 0) {
                config.input_size = static_cast<size_t>(total_input);
                config.input_shape = {static_cast<size_t>(total_input)};
            }
            spdlog::info("GraphCompiler: TimeSeriesWindow found - regression mode, "
                         "input_size={} (input_width={} x num_features={}), "
                         "target_width={}",
                         config.input_size, input_width, num_features,
                         config.time_series_target_width);
            break;
        }
    }
    FinalizeResolvedPartitionContract(config);

    // === Early domain detection ===
    // Shape inference below needs the dataset domain before it sees the
    // first model layer. Text graphs in particular use max_length as the
    // synthetic/debug input width; if this is left empty, Local Debug falls
    // back to [1] and Studio Debugger reports cosmetic shape mismatches.
    if (dataset_node) {
        auto cat_it = dataset_node->parameters.find("file_category");
        const std::string cat = (cat_it != dataset_node->parameters.end())
            ? cat_it->second : std::string();

        if (auto* cat_loader = loaders::GetByCategory(
                loaders::FileCategoryFromString(cat))) {
            config.preprocessing_domain = cat_loader->Domain(cat);
        }
    }

    // Get topologically sorted node IDs
    std::vector<int> sorted_ids = TopologicalSort(nodes, links);
    ValidateSparseFeatureTrainingContract(
        config, nodes, links, training_path_ids, sorted_ids);
    CollectGraphRuntimeOpNodeIds(
        nodes,
        links,
        sorted_ids,
        training_path_ids,
        config);
    config.graph_plan = BuildCompiledGraphPlan(
        nodes,
        links,
        sorted_ids,
        training_path_ids,
        config.data_source_node_id,
        config.loss_node_id,
        config.optimizer_node_id);
    config.metric_learning_graph =
        AnalyzeMetricLearningGraphContract(config.graph_plan);
    config.bert_encoder_graph =
        AnalyzeBertEncoderGraphContract(config.graph_plan);
    for (const auto& blocker : config.bert_encoder_graph.blockers) {
        AddIssue(config,
                 IssueLevel::Error,
                 blocker,
                 -1,
                 "",
                 errors::Compiler::UnsupportedTrainingNode);
    }

    // CrossEntropy/Focal consume logits and apply softmax internally.
    bool using_cross_entropy =
        loss_node &&
        (loss_node->type == gui::NodeType::CrossEntropyLoss ||
         loss_node->type == gui::NodeType::FocalLoss);

    // Log all nodes in the graph for debugging
    spdlog::info("GraphCompiler: Processing {} nodes (using_cross_entropy={})", sorted_ids.size(), using_cross_entropy);
    for (int node_id : sorted_ids) {
        const gui::MLNode* n = FindNodeById(node_id, nodes);
        if (n) {
            spdlog::debug("GraphCompiler: Node[{}] '{}' type={}", n->id, n->name, static_cast<int>(n->type));
        }
    }

    // Extract model layers and preprocessing in execution order
    std::vector<size_t> current_shape = config.input_shape;

    for (int node_id : sorted_ids) {
        const gui::MLNode* node = FindNodeById(node_id, nodes);
        if (!node) continue;
        const bool is_execution_node =
            IsPreprocessing(node->type) ||
            IsModelLayer(node->type) ||
            IsActivation(node->type);
        if (is_execution_node &&
            !ContainsWhenFiltered(training_path_ids, node->id)) {
            spdlog::debug("GraphCompiler: Skipping node '{}' outside selected training path",
                          node->name);
            continue;
        }

        // Handle preprocessing nodes
        if (IsPreprocessing(node->type)) {
            spdlog::info("GraphCompiler: Found preprocessing node '{}' (type={})", node->name, static_cast<int>(node->type));
            ExtractPreprocessing(*node, config);
            ApplyTextInputShape(config);
            current_shape = config.input_shape;
            if (node->type == gui::NodeType::Normalize) {
                spdlog::info("GraphCompiler: Normalization enabled - mean={}, std={}",
                             config.preprocessing.norm_mean, config.preprocessing.norm_std);
            }
            continue;
        }

        // Skip Softmax when using a logits-based class loss.
        // This prevents double-softmax which kills gradients
        if (node->type == gui::NodeType::Softmax && using_cross_entropy) {
            spdlog::warn(
                "GraphCompiler: Softmax layer skipped - {} loss applies softmax internally",
                config.GetLossName());
            continue;
        }

        if (IsGraphRuntimeFanInOp(node->type) &&
            HasConnectedInputAfterFirst(*node, links)) {
            continue;
        }

        // Handle model layers and activations
        if (IsModelLayer(node->type) || IsActivation(node->type)) {
            if (const auto reason =
                    ResolveInvalidDenseActivationConfigurationReason(
                        node->type, node->parameters)) {
                AddIssue(config, IssueLevel::Error, *reason, node->id,
                         node->name, errors::Compiler::InvalidParameter);
            }
            if (const auto reason =
                    ResolveInvalidSequenceProjectionConfigurationReason(
                        node->type, node->parameters)) {
                AddIssue(config, IssueLevel::Error, *reason, node->id,
                         node->name, errors::Compiler::InvalidParameter);
            }
            if (const auto reason =
                    ResolveInvalidNormalizationRegularizationConfigurationReason(
                        node->type, node->parameters)) {
                AddIssue(config, IssueLevel::Error, *reason, node->id,
                         node->name, errors::Compiler::InvalidParameter);
            }
            if (const auto reason = ResolveInvalidTransformerConfigurationReason(
                    node->type, node->parameters)) {
                AddIssue(config, IssueLevel::Error, *reason, node->id,
                         node->name, errors::Compiler::InvalidParameter);
            }
            CompiledLayer layer = ExtractLayerConfig(*node);
            layer.input_shape = current_shape;

            if (const auto reason =
                    ResolvePipelineUnsupportedSequentialModelConfigurationReason(
                        node->type, layer.parameters)) {
                AddIssue(config, IssueLevel::Error, *reason, node->id,
                         node->name,
                         errors::Compiler::UnsupportedTrainingNode);
            }
            if ((node->type == gui::NodeType::LSTM ||
                 node->type == gui::NodeType::GRU) &&
                HasConnectedSelectedOutputPinNamed(
                    *node, links, training_path_ids, "Hidden")) {
                AddIssue(
                    config,
                    IssueLevel::Error,
                    "The legacy Hidden output pin is not routed by the Engine "
                    "sequential model. Leave Hidden disconnected and use Output.",
                    node->id,
                    node->name,
                    errors::Compiler::UnsupportedTrainingNode);
            }

            // Infer output shape
            if (node->type == gui::NodeType::Reshape ||
                node->type == gui::NodeType::View ||
                node->type == gui::NodeType::Squeeze ||
                node->type == gui::NodeType::Unsqueeze ||
                node->type == gui::NodeType::Permute ||
                node->type == gui::NodeType::TensorBroadcastTo ||
                node->type == gui::NodeType::TensorExpand ||
                node->type == gui::NodeType::TensorIndexSelect) {
                std::string error;
                bool ok = false;
                if (node->type == gui::NodeType::Permute) {
                    ok = ResolvePermuteTargetShape(layer.parameters,
                                                   current_shape,
                                                   layer.output_shape,
                                                   layer.dims,
                                                   error);
                } else {
                    ok = ResolveShapeOpTargetShape(node->type,
                                                   layer.parameters,
                                                   current_shape,
                                                   layer.output_shape,
                                                   error);
                }
                if (!ok) {
                    AddIssue(config, IssueLevel::Error, error, node->id,
                             node->name, errors::Compiler::TensorShapeMismatch);
                    layer.output_shape = current_shape;
                }
            } else if (node->type == gui::NodeType::TensorPow ||
                       node->type == gui::NodeType::TensorClip) {
                std::string error;
                if (!ValidateTensorScalarMathParams(node->type,
                                                    layer.parameters,
                                                    error)) {
                    AddIssue(config, IssueLevel::Error, error, node->id,
                             node->name, errors::Compiler::InvalidParameter);
                }
                layer.output_shape = current_shape;
            } else if (node->type == gui::NodeType::TensorCompare ||
                       node->type == gui::NodeType::TensorLogicalMask) {
                std::string error;
                if (HasConnectedInputAfterFirst(*node, links)) {
                    error = node->type == gui::NodeType::TensorCompare
                        ? "TensorCompare currently supports only scalar comparison; disconnect input B"
                        : "TensorLogicalMask currently supports only unary op=not; disconnect input B";
                    AddIssue(config, IssueLevel::Error, error, node->id,
                             node->name, errors::Compiler::InvalidParameter);
                } else if (!ValidateTensorMaskParams(node->type,
                                                     layer.parameters,
                                                     error)) {
                    AddIssue(config, IssueLevel::Error, error, node->id,
                             node->name, errors::Compiler::InvalidParameter);
                }
                layer.output_shape = current_shape;
            } else if (node->type == gui::NodeType::TensorSum ||
                       node->type == gui::NodeType::TensorMean ||
                       node->type == gui::NodeType::TensorMax ||
                       node->type == gui::NodeType::TensorMin ||
                       node->type == gui::NodeType::TensorProd ||
                       node->type == gui::NodeType::TensorVar ||
                       node->type == gui::NodeType::TensorStd) {
                std::string error;
                if (!ResolveReductionTargetShape(node->type,
                                                 layer.parameters,
                                                 current_shape,
                                                 layer.output_shape,
                                                 error)) {
                    AddIssue(config, IssueLevel::Error, error, node->id,
                             node->name, errors::Compiler::TensorShapeMismatch);
                    layer.output_shape = current_shape;
                }
            } else if (node->type == gui::NodeType::TimeDistributed) {
                if (current_shape.size() < 2) {
                    AddIssue(config,
                             IssueLevel::Error,
                             "TimeDistributed requires sequence input shape [seq_len, features]",
                             node->id,
                             node->name,
                             errors::Compiler::TensorShapeMismatch);
                    layer.output_shape = current_shape;
                } else {
                    layer.output_shape = InferOutputShape(layer, current_shape);
                }
            } else {
                layer.output_shape = InferOutputShape(layer, current_shape);
            }
            current_shape = layer.output_shape;

            config.layers.push_back(layer);

            // Track output size from final class/logit projection layers.
            if (node->type == gui::NodeType::Dense ||
                node->type == gui::NodeType::TimeDistributed) {
                config.output_size = layer.units;
            }
        }
    }

    if (config.is_time_series && config.output_size > 0 &&
        config.output_size != config.time_series_target_width) {
        AddIssue(
            config,
            IssueLevel::Error,
            "TimeSeriesWindow target width (" +
                std::to_string(config.time_series_target_width) +
                ") does not match the model output width (" +
                std::to_string(config.output_size) +
                "). Set the final projection to the forecast horizon.",
            -1,
            "",
            errors::Compiler::LabelOutputShapeMismatch);
    }

    // Extract loss configuration
    if (loss_node) {
        config.loss_type = loss_node->type;
        config.loss_params = loss_node->parameters;

        // Resolve target semantics once, at compile time. Target-producing
        // graph operators may already be more specific; otherwise the
        // objective supplies the narrow default used by runtime metrics.
        if (config.target.value_kind == TargetValueKind::Unspecified) {
            switch (config.loss_type) {
                case gui::NodeType::MSELoss:
                case gui::NodeType::L1Loss:
                case gui::NodeType::SmoothL1Loss:
                case gui::NodeType::HuberLoss:
                    config.target.value_kind = TargetValueKind::Continuous;
                    break;
                default:
                    config.target.value_kind = TargetValueKind::Categorical;
                    break;
            }
        }

        const auto unsupported_sample_weight_params = PresentUnsupportedParameters(
            loss_node->parameters,
            {"sample_weight", "sample_weights"},
            true);
        if (!unsupported_sample_weight_params.empty()) {
            AddIssue(
                config,
                IssueLevel::Warning,
                "Sample-weight loss parameters are present but not "
                "implemented for graph training and will be ignored: " +
                    JoinNames(unsupported_sample_weight_params) + ".",
                loss_node->id,
                loss_node->name);
        }
        if (loss_node->type != gui::NodeType::CrossEntropyLoss) {
            const auto class_weight_params = PresentUnsupportedParameters(
                loss_node->parameters,
                {"weight", "class_weight", "class_weights", "weights",
                 "label_smoothing"},
                true);
            if (!class_weight_params.empty()) {
                AddIssue(
                    config,
                    IssueLevel::Warning,
                    "CrossEntropy loss parameters are supported only on "
                    "CrossEntropyLoss and will be ignored here: " +
                        JoinNames(class_weight_params) + ".",
                    loss_node->id,
                    loss_node->name);
            }
        }
        if (loss_node->type != gui::NodeType::BCEWithLogits) {
            const auto pos_weight_params = PresentUnsupportedParameters(
                loss_node->parameters,
                {"pos_weight"},
                true);
            if (!pos_weight_params.empty()) {
                AddIssue(
                    config,
                    IssueLevel::Warning,
                    "pos_weight is supported only on BCEWithLogits and "
                    "will be ignored here.",
                    loss_node->id,
                    loss_node->name);
            }
        }
    }

    ExtractRegressionTargetTransform(nodes, training_path_ids, config);

    // Extract optimizer configuration
    if (optimizer_node) {
        ExtractOptimizerConfiguration(*optimizer_node, config);
    }

    // Set one-hot encoding if we have classification (CrossEntropy loss)
    if (config.loss_type == gui::NodeType::CrossEntropyLoss ||
        config.loss_type == gui::NodeType::FocalLoss) {
        if (config.loss_type == gui::NodeType::CrossEntropyLoss) {
            config.preprocessing.has_onehot = true;
        }

        // Try to get num_classes from Output node's class-count parameters.
        const gui::MLNode* output_node = FindFirstReachableNodeOfType(
            nodes, training_path_ids, gui::NodeType::Output);
        if (!output_node) {
            output_node = FindFirstReachableNodeOfType(
                nodes, training_path_ids, gui::NodeType::SequenceTagOutput);
        }
        if (!output_node) {
            output_node = FindOutputNode(nodes);
        }
        if (output_node) {
            auto it = output_node->parameters.find(
                output_node->type == gui::NodeType::SequenceTagOutput
                    ? "num_tags"
                    : "num_classes");
            if (output_node->type == gui::NodeType::Output &&
                it == output_node->parameters.end()) {
                it = output_node->parameters.find("classes");
            }
            if (it != output_node->parameters.end() && !it->second.empty()) {
                try {
                    config.preprocessing.num_classes = std::stoul(it->second);
                    spdlog::info("GraphCompiler: num_classes={} from output node",
                                 config.preprocessing.num_classes);
                } catch (...) {
                    config.preprocessing.num_classes = 0;  // Will fall back below
                }
            }
        }

        // Fallback to output_size if Output node doesn't specify classes
        if (config.preprocessing.num_classes == 0) {
            config.preprocessing.num_classes = config.output_size;
            spdlog::warn("GraphCompiler: num_classes not specified in Output node, using output_size={}",
                         config.output_size);
        }

        if (config.output_size > 0 && config.output_size < 2) {
            AddIssue(config, IssueLevel::Error,
                     config.GetLossName() +
                     " requires at least two prediction logits; "
                     "the selected model path outputs " +
                     std::to_string(config.output_size),
                     -1,
                     "",
                     errors::Compiler::LabelOutputShapeMismatch);
        }

        if (config.preprocessing.num_classes > 0 &&
            config.output_size > 0 &&
            config.preprocessing.num_classes != config.output_size) {
            AddIssue(config, IssueLevel::Error,
                     config.GetLossName() + " class count (" +
                     std::to_string(config.preprocessing.num_classes) +
                     ") does not match the model output size (" +
                     std::to_string(config.output_size) + ")",
                     -1,
                     "",
                     errors::Compiler::LabelOutputShapeMismatch);
        }

        if (loss_node && loss_node->type == gui::NodeType::CrossEntropyLoss) {
            ValidateCrossEntropyWeightParams(config, *loss_node);
            ValidateCrossEntropyLabelSmoothing(config, *loss_node);
        }
    }

    if (config.loss_type == gui::NodeType::BCELoss ||
        config.loss_type == gui::NodeType::BCEWithLogits) {
        if (config.output_size > 0 && config.output_size != 1) {
            const char* loss_name = config.loss_type == gui::NodeType::BCELoss
                ? "BCELoss"
                : "BCEWithLogits";
            AddIssue(config, IssueLevel::Error,
                     std::string(loss_name) +
                     " requires a single prediction output for binary "
                     "classification; the selected model path outputs " +
                     std::to_string(config.output_size),
                     -1,
                     "",
                     errors::Compiler::LabelOutputShapeMismatch);
        }
        if (loss_node && loss_node->type == gui::NodeType::BCEWithLogits) {
            ValidateBCEWithLogitsPosWeight(config, *loss_node);
        }
    }

    // === Post-compile sanity checks ===
    // These need the values populated by the layer-extraction passes
    // above, so they live at the end of Compile().

    if (!placement_observation_cache_path.empty()) {
        std::string cache_error;
        if (!LoadBackendPlacementObservationCache(
                placement_observation_cache_path,
                &cache_error)) {
            AddIssue(config,
                     IssueLevel::Warning,
                     "Backend placement observation cache could not be loaded: " +
                         cache_error);
        }
    }
    AddBackendPlacementReports(config);
    config.compiler_placement_fingerprint =
        FingerprintPlacementEntries(config.backend_placements);
    FinalizePinMemoryTransferStatus(config);

    // DataSplit ratios should sum to ~1.0. Drift > 0.05 is almost
    // certainly a typo or stale state from the user adjusting one
    // ratio without rebalancing the others.
    if (config.has_data_split) {
        float sum = config.train_ratio + config.val_ratio + config.test_ratio;
        if (std::abs(sum - 1.0f) > 0.05f) {
            std::ostringstream msg;
            msg << "DataSplit ratios sum to " << sum
                << " (expected 1.0) - train=" << config.train_ratio
                << ", val=" << config.val_ratio
                << ", test=" << config.test_ratio;
            AddIssue(config, IssueLevel::Warning, msg.str());
        }
        if (config.val_ratio < 1e-6f) {
            AddIssue(config, IssueLevel::Warning,
                     "Validation split is 0 - training will run without validation metrics",
                     -1,
                     "",
                     errors::Data::InvalidSplit);
        }
    }

    // batch_size must fit in the train set. We can only check the upper
    // bound when the dataset is actually loaded — otherwise the row count
    // is unknown and the existing data_loaded check above already fired.
    if (config.batch_size <= 0) {
        AddIssue(config, IssueLevel::Error,
                 "batch_size must be positive (got " +
                 std::to_string(config.batch_size) + ")",
                 -1,
                 "",
                 errors::Compiler::InvalidParameter);
    } else if (dataset_node && !config.dataset_name.empty()) {
        auto& reg = DataRegistry::Instance();
        int64_t total_rows = 0;
        if (auto arrow_ds = reg.GetArrowDataset(config.dataset_name)) {
            total_rows = arrow_ds->GetNumRows();
        } else if (auto pq_ds = reg.GetParquetBackedDataset(config.dataset_name)) {
            total_rows = pq_ds->GetNumRows();
        }
        if (total_rows > 0) {
            int64_t train_rows = static_cast<int64_t>(total_rows * config.train_ratio);
            if (config.batch_size > train_rows) {
                AddIssue(config, IssueLevel::Error,
                         "batch_size (" + std::to_string(config.batch_size) +
                         ") is larger than the train split (" +
                         std::to_string(train_rows) + " rows)",
                         -1,
                         "",
                         errors::Memory::BatchTooLarge);
            } else if (config.batch_size > train_rows / 2) {
                AddIssue(config, IssueLevel::Warning,
                         "batch_size (" + std::to_string(config.batch_size) +
                         ") is more than half the train split (" +
                         std::to_string(train_rows) + " rows) - few iterations per epoch",
                         -1,
                         "",
                         errors::Memory::BatchTooLarge);
            }
        }
    }

    ValidateDenseTextVectorizerMaterializerMemory(
        config, nodes, training_path_ids);

    // === Domain detection ===
    // The DataInput node carries file_category (tabular / image / audio /
    // text / timeseries). The owning loader reports its PreprocessingDomain
    // via Domain(file_category) — TabularLoader splits Tabular vs
    // TimeSeries based on the category string; other loaders return a
    // fixed value.
    //
    // Per-domain notes preserved from the pre-refactor code:
    //   - TimeSeries: TimeSeriesWindow runs as a real Cat-1
    //     IPipelineOperator in the materializer pass, NOT as a config
    //     extractor. The compile gate's checks below (label column,
    //     batch size vs row count) run on the pre-materialized CSV —
    //     validating the user's raw column pick, not the post-window
    //     x_0..x_n schema.
    //   - Audio / Text: either the graph has extractor nodes (Spectrogram
    //     / MelSpec / MFCC for audio, TextTokenizer / TextVocabulary /
    //     TextPadding for text) that populate config.audio_preprocessing
    //     / text_preprocessing later, or we fall back to the dialog-
    //     baked defaults on the registry entry. The end-of-Compile()
    //     log reports which mode won.
    if (dataset_node) {
        auto cat_it = dataset_node->parameters.find("file_category");
        const std::string cat = (cat_it != dataset_node->parameters.end())
            ? cat_it->second : std::string();

        if (auto* cat_loader = loaders::GetByCategory(
                loaders::FileCategoryFromString(cat))) {
            config.preprocessing_domain = cat_loader->Domain(cat);
        }

        for (const auto& node : nodes) {
            if (!ContainsWhenFiltered(training_path_ids, node.id)) {
                continue;
            }

            auto node_domain = GetPreprocessingNodeDomain(node.type);
            if (!node_domain ||
                *node_domain == PreprocessingDomain::General ||
                *node_domain == config.preprocessing_domain) {
                continue;
            }

            std::ostringstream msg;
            msg << "Preprocessing node '" << node.name << "' is for "
                << PreprocessingDomainLabel(*node_domain)
                << " data, but the selected DataInput is "
                << PreprocessingDomainLabel(config.preprocessing_domain)
                << ". Use a matching DataInput category or replace this "
                   "preprocessing node.";
            AddIssue(config, IssueLevel::Error, msg.str(), node.id,
                     node.name, errors::Data::ColumnTypeMismatch);
        }

        // is_image kept as a local for the image-specific checks below
        // (Resize, Augmentation, etc.) — those stay domain-scoped and
        // don't need to move into the loader yet.
        const bool is_image = (cat == "image");

        if (is_image) {

            // Check 1: Resize node required for image datasets.
            // Without it, images load at their native (variable) size,
            // which can't be batched AND can cause massive tensors that
            // OOM the GPU. This is the most critical image check.
            bool has_resize = false;
            for (const auto& node : nodes) {
                if (node.type == gui::NodeType::Resize) {
                    has_resize = true;
                    break;
                }
            }
            if (!has_resize) {
                AddIssue(config, IssueLevel::Error,
                         "Image datasets require a Resize node to set target "
                         "dimensions. Without it, images load at their native "
                         "size which can exceed GPU memory. Add a Resize node "
                         "between DataInput and the first layer (e.g. "
                         "Resize 64x64 for fast training, 224x224 for accuracy).",
                         dataset_node->id, dataset_node->name);
            }

            // Check 2: Flatten required before Dense layers for images.
            // Images are 3D (H×W×C); Dense expects a 1D flat vector.
            if (!config.layers.empty()) {
                auto first_layer = config.layers[0];
                if (first_layer.type == gui::NodeType::Dense) {
                    bool has_flatten_before_dense = false;
                    for (const auto& node : nodes) {
                        if (node.type == gui::NodeType::Flatten) {
                            has_flatten_before_dense = true;
                            break;
                        }
                    }
                    if (!has_flatten_before_dense) {
                        AddIssue(config, IssueLevel::Error,
                                 "First model layer is Dense but no Flatten node "
                                 "found. Image data is 3D (H x W x C) and needs "
                                 "Flatten before Dense layers.",
                                 first_layer.node_id, first_layer.name);
                    }
                }
            }

            // Check 3: Memory estimation for image models.
            // Estimate total model memory (weights + gradients + optimizer)
            // and warn if it's likely to exceed reasonable GPU memory.
            if (has_resize && config.image_preprocessing.target_width > 0) {
                size_t img_features = static_cast<size_t>(
                    config.image_preprocessing.target_width) *
                    config.image_preprocessing.target_height * 3;
                size_t total_params = 0;
                size_t prev_size = img_features;
                for (const auto& layer : config.layers) {
                    if (layer.type == gui::NodeType::Dense) {
                        total_params += prev_size * layer.units + layer.units;
                        prev_size = layer.units;
                    } else if (layer.type == gui::NodeType::Flatten) {
                        // prev_size stays the same (just reshaped)
                    }
                }
                // Memory = weights × 4 (weights + grads + adam_m + adam_v)
                // + batch activations (batch_size × largest_layer)
                size_t model_bytes = total_params * 4 * sizeof(float);
                size_t batch_bytes = static_cast<size_t>(config.batch_size) *
                    img_features * sizeof(float);
                size_t total_est = model_bytes + batch_bytes;

                double est_mb = total_est / (1024.0 * 1024.0);

                std::ostringstream mem_msg;
                mem_msg << "Estimated GPU memory: " << std::fixed
                        << std::setprecision(0) << est_mb << " MB ("
                        << total_params << " parameters, batch_size="
                        << config.batch_size << ", input="
                        << config.image_preprocessing.target_width << "x"
                        << config.image_preprocessing.target_height << "x3)";

                // Real GPU usage is ~3-5x the raw parameter estimate due to
                // CUDA context (~300 MB), forward/backward activations,
                // matmul temporaries, and optimizer state. Use a 4x
                // multiplier for a realistic bound.
                double realistic_mb = est_mb * 4.0;

                if (realistic_mb > 3000) {
                    // Suggest a smaller size that would fit
                    int safe_features = 64 * 64 * 3;  // 12288
                    mem_msg << ". WARNING: real GPU usage is ~"
                            << std::setprecision(0) << realistic_mb
                            << " MB (incl. activations + optimizer + CUDA "
                            << "overhead). This will likely OOM on your GPU. "
                            << "Set the Resize node to a smaller size "
                            << "(e.g. 64x64 for " << safe_features
                            << " features)";
                    AddIssue(config, IssueLevel::Error, mem_msg.str());
                } else if (realistic_mb > 2000) {
                    mem_msg << ". This is close to typical GPU memory limits "
                            << "— consider reducing Resize dimensions or "
                            << "batch_size if training crashes";
                    AddIssue(config, IssueLevel::Warning, mem_msg.str());
                } else {
                    AddIssue(config, IssueLevel::Info, mem_msg.str());
                }
            }

            // Check 4: Pipeline ordering — Normalize before Resize is
            // almost certainly wrong (normalization stats are scale-
            // dependent). Walk the TOPOLOGICAL order (from links), not
            // the raw node vector (creation order), so the check reflects
            // the actual data flow.
            {
                std::vector<int> topo = TopologicalSort(nodes, links);
                bool seen_normalize = false;
                for (int nid : topo) {
                    if (!ContainsWhenFiltered(training_path_ids, nid)) {
                        continue;
                    }
                    const gui::MLNode* n = FindNodeById(nid, nodes);
                    if (!n) continue;
                    if (n->type == gui::NodeType::Normalize) {
                        seen_normalize = true;
                    }
                    if (n->type == gui::NodeType::Resize && seen_normalize) {
                        AddIssue(config, IssueLevel::Warning,
                                 "Normalize appears before Resize in the pipeline. "
                                 "Normalization is scale-dependent — resizing after "
                                 "normalization gives wrong per-pixel values. Move "
                                 "Resize before Normalize.");
                        break;
                    }
                }
            }
        }
    }

    // Audio preprocessing mode summary — surface whether the graph's
    // feature node overrode the dialog defaults, or whether we fell
    // back. Useful for debugging "why isn't my MelSpectrogram n_mels=64
    // taking effect" without having to grep through the logs for
    // individual extractor calls.
    if (config.preprocessing_domain == PreprocessingDomain::Audio) {
        if (config.audio_preprocessing.has_feature_node) {
            spdlog::info("GraphCompiler: audio feature extraction driven by graph "
                         "node (type={})",
                         static_cast<int>(config.audio_preprocessing.feature_type));
        } else {
            spdlog::info("GraphCompiler: audio feature extraction uses dialog defaults "
                         "(no Spectrogram/MelSpectrogram/MFCC node in graph)");
        }
    }

    // Text preprocessing is no longer extracted into TrainingConfiguration.
    // TextTokenizer is a real Arrow operator; TextVocabulary/TextPadding
    // fold into it during PipelineMaterializer. If no materialized Arrow
    // table is selected, TextDatasetBatcher uses the dialog-baked defaults
    // from DataRegistry::TextDatasetEntry.
    if (config.preprocessing_domain == PreprocessingDomain::Text) {
        spdlog::info("GraphCompiler: text preprocessing nodes are materialized "
                     "through Arrow operators; legacy text fallback uses dialog "
                     "defaults from the registered dataset");
    }

    // Final verdict: is_valid is the absence of any Error-level issue.
    // Warnings and Info don't block training.
    config.is_valid = !config.HasErrors();
    config.error_message = JoinErrorMessages(config.issues);

    spdlog::info("GraphCompiler: Compiled {} layers, input_size={}, output_size={}, "
                 "issues: {} errors / {} warnings / {} info, valid={}",
                 config.layers.size(), config.input_size, config.output_size,
                 config.CountIssues(IssueLevel::Error),
                 config.CountIssues(IssueLevel::Warning),
                 config.CountIssues(IssueLevel::Info),
                 config.is_valid);
    for (const auto& issue : config.issues) {
        const std::string prefix =
            std::string("  [") + IssueLevelLabel(issue.level) + "] " +
            (issue.node_name.empty() ? "" : ("[" + issue.node_name + "] ")) +
            (issue.error_code.empty() ? "" : ("[" + issue.error_code + "] ")) +
            issue.message;
        switch (issue.level) {
            case IssueLevel::Error:
                spdlog::error("{}", prefix);
                break;
            case IssueLevel::Warning:
                spdlog::warn("{}", prefix);
                break;
            case IssueLevel::Info:
                spdlog::info("{}", prefix);
                break;
        }
    }
    if (!config.backend_placements.empty()) {
        const auto summary = config.SummarizeBackendPlacements();
        spdlog::info(
            "GraphCompiler: Backend placement plan: total={}, gpu={}, cpu={}, mixed={}, risk={}, unsupported={}, unknown={}",
            summary.total,
            summary.gpu,
            summary.cpu,
            summary.mixed,
            summary.risk,
            summary.unsupported,
            summary.unknown);
        spdlog::info(
            "GraphCompiler: Backend placement fingerprint={}",
            config.compiler_placement_fingerprint);
        for (const auto& placement : config.backend_placements) {
            spdlog::info("  [{}] {} '{}' -> expected={}, fallback={}, reason={}",
                         placement.status,
                         placement.node_type,
                         placement.node_name,
                         placement.expected_backend,
                         placement.fallback_backend.empty() ? "none" : placement.fallback_backend,
                         placement.reason_code);
        }
    }

    return config;
}

bool GraphCompiler::ValidateGraph(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    std::string& error)
{
    // Check for empty graph
    if (nodes.empty()) {
        error = "Graph is empty - add nodes to create a model";
        return false;
    }

    // Check for dataset input
    if (!FindDatasetInputNode(nodes, links)) {
        error = "Graph must have a DatasetInput node";
        return false;
    }

    // Check for at least one model layer
    bool has_model_layer = false;
    for (const auto& node : nodes) {
        if (IsModelLayer(node.type)) {
            has_model_layer = true;
            break;
        }
    }
    if (!has_model_layer) {
        error = "Graph must have at least one model layer (Dense, Conv2D, etc.)";
        return false;
    }

    // Check for loss function
    if (!FindLossNode(nodes)) {
        error = "Graph must have a loss function (MSELoss or CrossEntropyLoss)";
        return false;
    }

    // Check for optimizer
    if (!FindOptimizerNode(nodes)) {
        error = "Graph must have an optimizer (SGD, Adam, AdamW, RMSprop, Adagrad, or NAdam)";
        return false;
    }

    // Check for cycles
    if (HasCycle(nodes, links)) {
        error = "Graph contains a cycle - remove circular connections";
        return false;
    }

    return true;
}

std::vector<int> GraphCompiler::TopologicalSort(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links)
{
    // Build adjacency list and in-degree count
    std::map<int, std::vector<int>> adj;
    std::map<int, int> in_degree;

    for (const auto& node : nodes) {
        adj[node.id] = {};
        in_degree[node.id] = 0;
    }

    for (const auto& link : links) {
        adj[link.from_node].push_back(link.to_node);
        in_degree[link.to_node]++;
    }

    // Kahn's algorithm
    std::queue<int> queue;
    for (const auto& node : nodes) {
        if (in_degree[node.id] == 0) {
            queue.push(node.id);
        }
    }

    std::vector<int> sorted;
    while (!queue.empty()) {
        int node_id = queue.front();
        queue.pop();
        sorted.push_back(node_id);

        for (int neighbor : adj[node_id]) {
            in_degree[neighbor]--;
            if (in_degree[neighbor] == 0) {
                queue.push(neighbor);
            }
        }
    }

    return sorted;
}

const gui::MLNode* GraphCompiler::FindNodeById(int id, const std::vector<gui::MLNode>& nodes) const {
    for (const auto& node : nodes) {
        if (node.id == id) return &node;
    }
    return nullptr;
}

std::vector<int> GraphCompiler::GetConnectedNodes(
    int from_node_id,
    const std::vector<gui::NodeLink>& links) const
{
    std::vector<int> connected;
    for (const auto& link : links) {
        if (link.from_node == from_node_id) {
            connected.push_back(link.to_node);
        }
    }
    return connected;
}

std::vector<int> GraphCompiler::GetInputNodes(
    int to_node_id,
    const std::vector<gui::NodeLink>& links) const
{
    std::vector<int> inputs;
    for (const auto& link : links) {
        if (link.to_node == to_node_id) {
            inputs.push_back(link.from_node);
        }
    }
    return inputs;
}

const gui::MLNode* GraphCompiler::FindDatasetInputNode(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links) const {
    // Modern graphs assign the Training role exclusively through the named
    // Data Split input. Prefer that topology before any legacy fallback so a
    // stale Data Input `dataset_role` parameter cannot select the run source.
    for (const auto& node : nodes) {
        if (node.type != gui::NodeType::DataSplit ||
            !SplitReachesLoss(node, nodes, links)) {
            continue;
        }
        if (const auto* training_source =
                FindDatasetSourceConnectedToSplitInput(
                    nodes, links, node, "Training Dataset")) {
            return training_source;
        }
    }

    const gui::MLNode* first_source = nullptr;
    const gui::MLNode* first_connected_source = nullptr;
    for (const auto& node : nodes) {
        if (!IsDatasetSourceType(node.type)) {
            continue;
        }

        if (!first_source) {
            first_source = &node;
        }

        if (!HasOutgoingLink(node.id, links)) {
            continue;
        }

        if (!first_connected_source) {
            first_connected_source = &node;
        }

        const auto reachable = CollectReachableNodeIds(node.id, links);
        for (const auto& candidate : nodes) {
            if (IsLossNodeType(candidate.type) &&
                reachable.count(candidate.id) > 0) {
                return &node;
            }
        }
    }
    return first_connected_source ? first_connected_source : first_source;
}

const gui::MLNode* GraphCompiler::FindLossNode(const std::vector<gui::MLNode>& nodes) const {
    for (const auto& node : nodes) {
        if (IsLossNodeType(node.type)) {
            return &node;
        }
    }
    return nullptr;
}

const gui::MLNode* GraphCompiler::FindOptimizerNode(const std::vector<gui::MLNode>& nodes) const {
    for (const auto& node : nodes) {
        if (IsSupportedOptimizerNodeType(node.type)) {
            return &node;
        }
    }
    return nullptr;
}

const gui::MLNode* GraphCompiler::FindOutputNode(const std::vector<gui::MLNode>& nodes) const {
    for (const auto& node : nodes) {
        if (IsOutputNodeType(node.type)) {
            return &node;
        }
    }
    return nullptr;
}

bool GraphCompiler::IsModelLayer(gui::NodeType type) const {
    if (IsPipelineUnsupportedSequentialModelLayer(type)) {
        return true;
    }

    switch (type) {
        case gui::NodeType::Dense:
        case gui::NodeType::Conv2D:
        case gui::NodeType::MaxPool2D:
        case gui::NodeType::AvgPool2D:
        case gui::NodeType::GlobalMaxPool:
        case gui::NodeType::GlobalAvgPool:
        case gui::NodeType::Flatten:
        case gui::NodeType::Reshape:
        case gui::NodeType::View:
        case gui::NodeType::Permute:
        case gui::NodeType::Squeeze:
        case gui::NodeType::Unsqueeze:
        case gui::NodeType::TensorBroadcastTo:
        case gui::NodeType::TensorExpand:
        case gui::NodeType::TensorIndexSelect:
        case gui::NodeType::TensorAbs:
        case gui::NodeType::TensorExp:
        case gui::NodeType::TensorLog:
        case gui::NodeType::TensorSqrt:
        case gui::NodeType::TensorSign:
        case gui::NodeType::TensorPow:
        case gui::NodeType::TensorClip:
        case gui::NodeType::TensorCompare:
        case gui::NodeType::TensorLogicalMask:
        case gui::NodeType::TensorSum:
        case gui::NodeType::TensorMean:
        case gui::NodeType::TensorMax:
        case gui::NodeType::TensorMin:
        case gui::NodeType::TensorProd:
        case gui::NodeType::TensorVar:
        case gui::NodeType::TensorStd:
        case gui::NodeType::Dropout:
        case gui::NodeType::BatchNorm:
        case gui::NodeType::LayerNorm:
        case gui::NodeType::MultiHeadAttention:
        case gui::NodeType::ConvTranspose2D:
        case gui::NodeType::Upsample:
        case gui::NodeType::PixelShuffle:
        case gui::NodeType::PolicyNetwork:
        case gui::NodeType::ValueNetwork:
        case gui::NodeType::Embedding:
        case gui::NodeType::TransformerEncoder:
        case gui::NodeType::PositionalEncoding:
        case gui::NodeType::TransformerDecoder:
        // Recurrent layers — required for text/time-series models.
        // Omitting them here caused the LSTM smoke test to silently
        // drop the LSTM node from `config_.layers`, leaving
        // training_executor to feed [batch, seq, embed] straight into
        // a Dense layer sized for [batch, seq*embed] — runtime shape
        // mismatch and crash.
        case gui::NodeType::LSTM:
        case gui::NodeType::GRU:
        case gui::NodeType::RNN:
        case gui::NodeType::Bidirectional:
        case gui::NodeType::TimeDistributed:
            return true;
        default:
            return false;
    }
}

bool GraphCompiler::IsActivation(gui::NodeType type) const {
    return IsExecutableActivationNode(type);
}

// ---------------------------------------------------------------------------
// Table-driven preprocessing registry
//
// Each entry maps a NodeType to a preprocessing domain and an optional
// extraction function. IsPreprocessing and ExtractPreprocessing both
// consult this table instead of maintaining separate switch statements.
//
// Adding a new preprocessing node for any domain is a single line here
// plus a static extraction function. No switch cases to update in
// multiple places.
// ---------------------------------------------------------------------------

namespace {

using ExtractorFn = void(*)(const gui::MLNode&, TrainingConfiguration&);

struct PreprocessingNodeSpec {
    gui::NodeType type;
    PreprocessingDomain domain;
    ExtractorFn extractor;  // nullptr = recognized but not yet wired
};

// --- Tabular extractors (migrated from the old switch) ---

static void ExtractReshape(const gui::MLNode& node, TrainingConfiguration& config) {
    config.preprocessing.has_reshape = true;
    if (node.parameters.count("shape")) {
        std::string shape_str = node.parameters.at("shape");
        shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), '['), shape_str.end());
        shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ']'), shape_str.end());
        shape_str.erase(std::remove(shape_str.begin(), shape_str.end(), ' '), shape_str.end());

        size_t pos = 0;
        while ((pos = shape_str.find(',')) != std::string::npos) {
            config.preprocessing.reshape_dims.push_back(std::stoi(shape_str.substr(0, pos)));
            shape_str.erase(0, pos + 1);
        }
        if (!shape_str.empty()) {
            config.preprocessing.reshape_dims.push_back(std::stoi(shape_str));
        }
    }
}

static void ExtractOneHot(const gui::MLNode& node, TrainingConfiguration& config) {
    config.preprocessing.has_onehot = true;
    if (node.parameters.count("num_classes"))
        config.preprocessing.num_classes = std::stoul(node.parameters.at("num_classes"));
}

// --- Image extractors (Phase 1) ---

static void ExtractImageResize(const gui::MLNode& node, TrainingConfiguration& config) {
    config.image_preprocessing.resize_mode = ResizeMode::Exact;
    if (node.parameters.count("width"))
        config.image_preprocessing.target_width = std::stoi(node.parameters.at("width"));
    if (node.parameters.count("height"))
        config.image_preprocessing.target_height = std::stoi(node.parameters.at("height"));
    if (node.parameters.count("mode")) {
        std::string mode = node.parameters.at("mode");
        if (mode == "exact")       config.image_preprocessing.resize_mode = ResizeMode::Exact;
        else if (mode == "fit")    config.image_preprocessing.resize_mode = ResizeMode::AspectFit;
        else if (mode == "fill")   config.image_preprocessing.resize_mode = ResizeMode::AspectFill;
        else if (mode == "center") config.image_preprocessing.resize_mode = ResizeMode::Center;
    }
    spdlog::info("GraphCompiler: Resize {}x{} mode={}",
                 config.image_preprocessing.target_width,
                 config.image_preprocessing.target_height,
                 static_cast<int>(config.image_preprocessing.resize_mode));
}

static void ExtractImageNormalize(const gui::MLNode& node, TrainingConfiguration& config) {
    // Domain-aware Normalize: when upstream is image data, populate
    // both the legacy GraphPreprocessingConfig (for backward-compat
    // with the tabular training path) and the image-specific config.
    // The image batcher reads image_preprocessing; the tabular batcher
    // reads config.preprocessing. Both work.
    if (config.preprocessing_domain == PreprocessingDomain::Image) {
        config.image_preprocessing.enable_denoise = false;  // not related but clear the field
        // Image normalize is handled by the ImageTransformPipeline;
        // the legacy fields are also set so existing tabular code paths
        // don't break if they run accidentally.
    }
    // Always populate the legacy tabular fields (backward-compat).
    config.preprocessing.has_normalization = true;
    if (node.parameters.count("mean"))
        config.preprocessing.norm_mean = std::stof(node.parameters.at("mean"));
    if (node.parameters.count("std"))
        config.preprocessing.norm_std = std::stof(node.parameters.at("std"));
}

static void ExtractGrayscale(const gui::MLNode& /*node*/, TrainingConfiguration& config) {
    config.image_preprocessing.convert_to_grayscale = true;
}

static void ExtractImageGaussianBlur(const gui::MLNode& node, TrainingConfiguration& config) {
    config.image_preprocessing.blur_config.enabled = true;
    config.image_preprocessing.blur_config.type = BlurType::Gaussian;
    if (node.parameters.count("kernel_size"))
        config.image_preprocessing.blur_config.kernel_size = std::stoi(node.parameters.at("kernel_size"));
    if (node.parameters.count("sigma"))
        config.image_preprocessing.blur_config.sigma = std::stof(node.parameters.at("sigma"));
}

// --- Audio extractors (Phase 2.1) ---
//
// Each extractor sets has_feature_node=true (except AudioAugmentation,
// which sets has_augmentation=true) and copies the GUI node's parameters
// into config.audio_preprocessing. AudioDatasetBatcher later merges this
// into its AudioDatasetConfig, overriding the dialog-baked defaults on
// AudioDatasetEntry. If no audio feature node is found in the graph,
// has_feature_node stays false and the batcher uses the dialog defaults
// — "fallback mode", the standard UX.

static void ExtractSpectrogram(const gui::MLNode& node, TrainingConfiguration& config) {
    config.audio_preprocessing.has_feature_node = true;
    config.audio_preprocessing.feature_type = AudioFeatureType::Spectrogram;
    if (node.parameters.count("n_fft"))
        config.audio_preprocessing.n_fft = std::stoi(node.parameters.at("n_fft"));
    if (node.parameters.count("hop_length"))
        config.audio_preprocessing.hop_length = std::stoi(node.parameters.at("hop_length"));
    if (node.parameters.count("log_scale"))
        config.audio_preprocessing.log_scale = (node.parameters.at("log_scale") == "true");
    spdlog::info("GraphCompiler: Spectrogram n_fft={}, hop={}, log={}",
                 config.audio_preprocessing.n_fft,
                 config.audio_preprocessing.hop_length,
                 config.audio_preprocessing.log_scale);
}

static void ExtractMelSpectrogram(const gui::MLNode& node, TrainingConfiguration& config) {
    config.audio_preprocessing.has_feature_node = true;
    config.audio_preprocessing.feature_type = AudioFeatureType::MelSpectrogram;
    if (node.parameters.count("n_fft"))
        config.audio_preprocessing.n_fft = std::stoi(node.parameters.at("n_fft"));
    if (node.parameters.count("hop_length"))
        config.audio_preprocessing.hop_length = std::stoi(node.parameters.at("hop_length"));
    if (node.parameters.count("n_mels"))
        config.audio_preprocessing.n_mels = std::stoi(node.parameters.at("n_mels"));
    if (node.parameters.count("fmin"))
        config.audio_preprocessing.fmin = std::stof(node.parameters.at("fmin"));
    if (node.parameters.count("fmax"))
        config.audio_preprocessing.fmax = std::stof(node.parameters.at("fmax"));
    if (node.parameters.count("log_scale"))
        config.audio_preprocessing.log_scale = (node.parameters.at("log_scale") == "true");
    spdlog::info("GraphCompiler: MelSpectrogram n_mels={}, n_fft={}, hop={}",
                 config.audio_preprocessing.n_mels,
                 config.audio_preprocessing.n_fft,
                 config.audio_preprocessing.hop_length);
}

static void ExtractMFCC(const gui::MLNode& node, TrainingConfiguration& config) {
    config.audio_preprocessing.has_feature_node = true;
    config.audio_preprocessing.feature_type = AudioFeatureType::MFCC;
    if (node.parameters.count("n_mfcc"))
        config.audio_preprocessing.n_mfcc = std::stoi(node.parameters.at("n_mfcc"));
    if (node.parameters.count("n_fft"))
        config.audio_preprocessing.n_fft = std::stoi(node.parameters.at("n_fft"));
    if (node.parameters.count("hop_length"))
        config.audio_preprocessing.hop_length = std::stoi(node.parameters.at("hop_length"));
    if (node.parameters.count("n_mels"))
        config.audio_preprocessing.n_mels = std::stoi(node.parameters.at("n_mels"));
    spdlog::info("GraphCompiler: MFCC n_mfcc={}, n_mels={}, n_fft={}",
                 config.audio_preprocessing.n_mfcc,
                 config.audio_preprocessing.n_mels,
                 config.audio_preprocessing.n_fft);
}

static void ExtractAudioAugmentation(const gui::MLNode& node, TrainingConfiguration& config) {
    config.audio_preprocessing.has_augmentation = true;
    if (node.parameters.count("noise_level"))
        config.audio_preprocessing.noise_level = std::stof(node.parameters.at("noise_level"));
    if (node.parameters.count("time_stretch"))
        config.audio_preprocessing.time_stretch = (node.parameters.at("time_stretch") == "true");
    if (node.parameters.count("pitch_shift"))
        config.audio_preprocessing.pitch_shift = (node.parameters.at("pitch_shift") == "true");
    spdlog::info("GraphCompiler: AudioAugmentation noise={}, time_stretch={}, pitch_shift={}",
                 config.audio_preprocessing.noise_level,
                 config.audio_preprocessing.time_stretch,
                 config.audio_preprocessing.pitch_shift);
}

// --- Text preprocessing (Arrow materializer path) ---
// TextTokenizer is applied by PipelineMaterializer at runtime. The compiler
// still extracts its shape contract so recurrent preflight uses the graph's
// current max_length instead of the registered dataset's dialog default.
static void ExtractTextTokenizerShape(
    const gui::MLNode& node,
    TrainingConfiguration& config) {
    config.text_preprocessing.has_tokenizer_node = true;
    config.text_preprocessing.has_padding_node = true;
    if (node.parameters.count("max_length")) {
        config.text_preprocessing.max_length =
            static_cast<int>(ParseSizeParam(node.parameters, "max_length", 512));
    }
    if (node.parameters.count("tokenizer_type")) {
        config.text_preprocessing.tokenizer_type =
            static_cast<int>(ParseSizeParam(node.parameters, "tokenizer_type", 1));
    }
    config.text_preprocessing.lowercase =
        ParseBoolParam(node.parameters, "lowercase", true);
    config.text_preprocessing.do_padding =
        ParseBoolParam(node.parameters, "padding", true);
    config.text_preprocessing.do_truncation =
        ParseBoolParam(node.parameters, "truncation", true);
    if (node.parameters.count("pad_value")) {
        config.text_preprocessing.pad_value =
            static_cast<int>(ParseSizeParam(node.parameters, "pad_value", 0));
    }
    spdlog::info("GraphCompiler: TextTokenizer max_length={} drives text input shape",
                 config.text_preprocessing.max_length);
}

static void ExtractTextVectorizerShape(
    const gui::MLNode& node,
    TrainingConfiguration& config) {
    config.text_preprocessing.has_vectorizer_node = true;
    const int max_features = static_cast<int>(
        ParseSizeParam(node.parameters, "max_features", 2000));
    config.text_preprocessing.max_length = std::max(1, max_features);
    spdlog::info(
        "GraphCompiler: text vectorizer max_features={} drives text input shape",
        config.text_preprocessing.max_length);
}

// TextVocabulary and TextPadding fold into the reachable tokenizer in
// PipelineMaterializer; they do not own a separate training shape.

// --- TimeSeries extractors (Phase 4 — deferred) ---

// --- The table ---

static const PreprocessingNodeSpec kPreprocessingSpecs[] = {
    // Tabular (existing, migrated from switch)
    {gui::NodeType::Normalize,          PreprocessingDomain::Tabular,     ExtractImageNormalize},
    {gui::NodeType::TensorReshape,      PreprocessingDomain::Tabular,     ExtractReshape},
    {gui::NodeType::OneHotEncode,       PreprocessingDomain::Tabular,     ExtractOneHot},
    {gui::NodeType::StandardScaler,     PreprocessingDomain::General,     nullptr},
    // General (domain-agnostic data pipeline nodes — no extraction needed)
    {gui::NodeType::DataSplit,          PreprocessingDomain::General,     nullptr},
    {gui::NodeType::DataLoader,         PreprocessingDomain::General,     nullptr},
    // Image (Phase 1)
    {gui::NodeType::Resize,             PreprocessingDomain::Image,       ExtractImageResize},
    {gui::NodeType::CenterCrop,         PreprocessingDomain::Image,       nullptr},
    {gui::NodeType::RandomCrop,         PreprocessingDomain::Image,       nullptr},
    {gui::NodeType::HorizontalFlip,     PreprocessingDomain::Image,       nullptr},
    {gui::NodeType::VerticalFlip,       PreprocessingDomain::Image,       nullptr},
    {gui::NodeType::ImageRotate,        PreprocessingDomain::Image,       nullptr},
    {gui::NodeType::ColorJitter,        PreprocessingDomain::Image,       nullptr},
    {gui::NodeType::ImageGaussianBlur,  PreprocessingDomain::Image,       ExtractImageGaussianBlur},
    {gui::NodeType::Grayscale,          PreprocessingDomain::Image,       ExtractGrayscale},
    {gui::NodeType::Augmentation,       PreprocessingDomain::Image,       nullptr},
    // Audio (Phase 2.1)
    {gui::NodeType::AudioInput,         PreprocessingDomain::Audio,       nullptr},
    {gui::NodeType::Spectrogram,        PreprocessingDomain::Audio,       ExtractSpectrogram},
    {gui::NodeType::MelSpectrogram,     PreprocessingDomain::Audio,       ExtractMelSpectrogram},
    {gui::NodeType::MFCC,               PreprocessingDomain::Audio,       ExtractMFCC},
    {gui::NodeType::AudioAugmentation,  PreprocessingDomain::Audio,       ExtractAudioAugmentation},
    // Text (Arrow materializer path; extract shape only)
    {gui::NodeType::TextTokenizer,      PreprocessingDomain::Text,        ExtractTextTokenizerShape},
    {gui::NodeType::TFIDFVectorizer,    PreprocessingDomain::Text,        ExtractTextVectorizerShape},
    {gui::NodeType::TextVocabulary,     PreprocessingDomain::Text,        nullptr},
    {gui::NodeType::TextPadding,        PreprocessingDomain::Text,        nullptr},
    {gui::NodeType::NERSequenceBuilder, PreprocessingDomain::General,     nullptr},
    {gui::NodeType::TokenVocabulary,    PreprocessingDomain::Text,        nullptr},
    {gui::NodeType::POSVocabulary,      PreprocessingDomain::Text,        nullptr},
    {gui::NodeType::NERTagVocabulary,   PreprocessingDomain::Text,        nullptr},
    // TimeSeries (Phase 4)
    {gui::NodeType::TimeSeriesWindow,   PreprocessingDomain::TimeSeries,  nullptr},
    {gui::NodeType::TimeSeriesFeatures, PreprocessingDomain::TimeSeries,  nullptr},
    {gui::NodeType::TimeSeriesSplit,    PreprocessingDomain::TimeSeries,  nullptr},
    {gui::NodeType::SeasonalNaive,      PreprocessingDomain::TimeSeries,  nullptr},
    {gui::NodeType::LogTransform,       PreprocessingDomain::TimeSeries,  nullptr},
    {gui::NodeType::Differencing,       PreprocessingDomain::TimeSeries,  nullptr},
};

} // anonymous namespace

bool GraphCompiler::IsPreprocessing(gui::NodeType type) const {
    for (const auto& spec : kPreprocessingSpecs) {
        if (spec.type == type) return true;
    }
    return false;
}

std::optional<PreprocessingDomain> GraphCompiler::GetPreprocessingNodeDomain(gui::NodeType type) const {
    for (const auto& spec : kPreprocessingSpecs) {
        if (spec.type == type) {
            return spec.domain;
        }
    }
    return std::nullopt;
}

CompiledLayer GraphCompiler::ExtractLayerConfig(const gui::MLNode& node) const {
    CompiledLayer layer;
    layer.type = node.type;
    layer.node_id = node.id;
    layer.name = gui::EffectiveNodeName(node);
    layer.parameters = node.parameters;

    // Unsupported sequential layers are retained in the compiled inventory so
    // placement and diagnostics can identify them. Do not interpret their
    // parameters: legacy graphs may contain design-era encodings (for example
    // Conv2D padding="same") that have no executable ModelBuilder contract.
    const auto training_support =
        ResolvePipelineTrainingBackendSupport(node.type);
    if (!training_support.compile_supported &&
        training_support.mode ==
            PipelineTrainingBackendSupportMode::UnsupportedSequentialModelLayer) {
        return layer;
    }

    // Extract specific parameters
    switch (node.type) {
        case gui::NodeType::Dense: {
            DenseActivationConfiguration configuration;
            if (!ResolveDenseActivationConfiguration(
                    node.type, node.parameters, configuration)) {
                layer.units = static_cast<int>(configuration.dense_units);
            }
            break;
        }

        case gui::NodeType::TimeDistributed:
            layer.units = static_cast<int>(ParseSizeParam(
                node.parameters,
                "units",
                ParseSizeParam(node.parameters, "out_features", 0)));
            break;

        case gui::NodeType::Embedding:
            // Embedding params (num_embeddings, embedding_dim) stay in
            // the generic `parameters` map and are read by the training
            // executor. No dedicated fields on CompiledLayer to keep
            // the struct slim; the raw parameter passthrough is enough.
            break;

        case gui::NodeType::Conv2D:
            if (node.parameters.count("filters"))
                layer.filters = std::stoi(node.parameters.at("filters"));
            if (node.parameters.count("kernel_size"))
                layer.kernel_size = std::stoi(node.parameters.at("kernel_size"));
            if (node.parameters.count("stride"))
                layer.stride = std::stoi(node.parameters.at("stride"));
            if (node.parameters.count("padding"))
                layer.padding = std::stoi(node.parameters.at("padding"));
            break;

        case gui::NodeType::MaxPool2D:
        case gui::NodeType::AvgPool2D:
            if (node.parameters.count("pool_size"))
                layer.pool_size = std::stoi(node.parameters.at("pool_size"));
            if (node.parameters.count("stride"))
                layer.stride = std::stoi(node.parameters.at("stride"));
            break;

        case gui::NodeType::BatchNorm:
        case gui::NodeType::Dropout:
        case gui::NodeType::LayerNorm:
            // The shared policy validates these raw persisted parameters and
            // ModelBuilder consumes the exact resolved values. Avoid a second
            // parser in CompiledLayer fields, which previously changed zero
            // Dropout/BatchNorm values into defaults.
            break;

        case gui::NodeType::LeakyReLU: {
            DenseActivationConfiguration configuration;
            if (!ResolveDenseActivationConfiguration(
                    node.type, node.parameters, configuration)) {
                layer.negative_slope = configuration.negative_slope;
            }
            break;
        }

        case gui::NodeType::ELU: {
            DenseActivationConfiguration configuration;
            if (!ResolveDenseActivationConfiguration(
                    node.type, node.parameters, configuration)) {
                layer.alpha = configuration.elu_alpha;
            }
            break;
        }

        case gui::NodeType::ConvTranspose2D:
            if (node.parameters.count("in_channels"))
                layer.in_channels = std::stoi(node.parameters.at("in_channels"));
            if (node.parameters.count("out_channels"))
                layer.filters = std::stoi(node.parameters.at("out_channels"));
            if (node.parameters.count("kernel_size"))
                layer.kernel_size = std::stoi(node.parameters.at("kernel_size"));
            if (node.parameters.count("stride"))
                layer.stride = std::stoi(node.parameters.at("stride"));
            if (node.parameters.count("padding"))
                layer.padding = std::stoi(node.parameters.at("padding"));
            if (node.parameters.count("output_padding"))
                layer.output_padding = std::stoi(node.parameters.at("output_padding"));
            break;

        case gui::NodeType::Upsample:
            if (node.parameters.count("scale_factor"))
                layer.scale_factor = std::stoi(node.parameters.at("scale_factor"));
            if (node.parameters.count("mode"))
                layer.upsample_mode = std::stoi(node.parameters.at("mode"));
            break;

        case gui::NodeType::PixelShuffle:
            if (node.parameters.count("upscale_factor"))
                layer.scale_factor = std::stoi(node.parameters.at("upscale_factor"));
            break;

        default:
            break;
    }

    return layer;
}

void GraphCompiler::ExtractPreprocessing(
    const gui::MLNode& node,
    TrainingConfiguration& config) const
{
    for (const auto& spec : kPreprocessingSpecs) {
        if (spec.type == node.type) {
            if (spec.extractor) {
                spec.extractor(node, config);
            }
            return;
        }
    }
}

std::vector<size_t> GraphCompiler::InferOutputShape(
    const CompiledLayer& layer,
    const std::vector<size_t>& input_shape) const
{
    std::vector<size_t> output_shape;

    switch (layer.type) {
        case gui::NodeType::Dense:
            // Dense: [...] -> [units]
            output_shape = {static_cast<size_t>(layer.units)};
            break;

        case gui::NodeType::TimeDistributed: {
            const size_t units = layer.units > 0
                ? static_cast<size_t>(layer.units)
                : ParseSizeParam(layer.parameters, "units", 64);
            if (input_shape.size() >= 2) {
                output_shape = {input_shape[0], units};
            } else {
                output_shape = {units};
            }
            break;
        }

        case gui::NodeType::Embedding: {
            // Embedding: [seq_len] -> [seq_len, embedding_dim]
            // Input shape is the sequence length (from text dataset
            // max_length). Output adds the embedding_dim channel so a
            // downstream Flatten produces seq_len * embedding_dim.
            int embed_dim = 64;
            auto it = layer.parameters.find("embedding_dim");
            if (it != layer.parameters.end()) {
                try { embed_dim = std::stoi(it->second); } catch (...) {}
            }
            if (!input_shape.empty()) {
                output_shape = {input_shape[0],
                                static_cast<size_t>(embed_dim)};
            } else {
                output_shape = {static_cast<size_t>(embed_dim)};
            }
            break;
        }

        case gui::NodeType::LSTM:
        case gui::NodeType::GRU:
        case gui::NodeType::RNN: {
            const size_t hidden_size =
                ParseSizeParam(layer.parameters, "hidden_size", 128);
            const bool bidirectional =
                ParseBoolParam(layer.parameters, "bidirectional", false);
            const bool return_sequences =
                ParseBoolParam(layer.parameters, "return_sequences", false);
            const size_t output_features =
                hidden_size * (bidirectional ? 2 : 1);

            if (return_sequences) {
                const size_t seq_len = input_shape.empty() ? 1 : input_shape[0];
                output_shape = {seq_len, output_features};
            } else {
                output_shape = {output_features};
            }
            break;
        }

        case gui::NodeType::Conv2D:
            // Conv2D: [H, W, C] -> [(H + 2*padding - kernel_size) / stride + 1, W', filters]
            if (input_shape.size() >= 2) {
                size_t out_h = (input_shape[0] + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
                size_t out_w = (input_shape[1] + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
                output_shape = {out_h, out_w, static_cast<size_t>(layer.filters)};
            }
            break;

        case gui::NodeType::MaxPool2D:
        case gui::NodeType::AvgPool2D:
            // Pool2D: [H, W, C] -> [H/pool_size, W/pool_size, C]
            if (input_shape.size() >= 3) {
                int stride = layer.stride > 0 ? layer.stride : layer.pool_size;
                size_t out_h = (input_shape[0] - layer.pool_size) / stride + 1;
                size_t out_w = (input_shape[1] - layer.pool_size) / stride + 1;
                output_shape = {out_h, out_w, input_shape[2]};
            }
            break;

        case gui::NodeType::GlobalMaxPool:
        case gui::NodeType::GlobalAvgPool:
            // Global pooling: [H, W, C] -> [C]
            if (input_shape.size() >= 3) {
                output_shape = {input_shape[2]};  // Just channels remain
            } else if (!input_shape.empty()) {
                output_shape = {input_shape.back()};
            }
            break;

        case gui::NodeType::Flatten:
            // Flatten: [H, W, C] -> [H*W*C]
            {
                size_t flat_size = 1;
                for (size_t dim : input_shape) flat_size *= dim;
                output_shape = {flat_size};
            }
            break;

        case gui::NodeType::Reshape:
        case gui::NodeType::View:
        case gui::NodeType::Permute:
        case gui::NodeType::Squeeze:
        case gui::NodeType::Unsqueeze:
        case gui::NodeType::TensorBroadcastTo:
        case gui::NodeType::TensorExpand:
        case gui::NodeType::TensorIndexSelect: {
            std::string error;
            std::vector<int> ignored_dims;
            const bool ok = layer.type == gui::NodeType::Permute
                ? ResolvePermuteTargetShape(layer.parameters,
                                            input_shape,
                                            output_shape,
                                            ignored_dims,
                                            error)
                : ResolveShapeOpTargetShape(layer.type,
                                            layer.parameters,
                                            input_shape,
                                            output_shape,
                                            error);
            if (!ok) {
                spdlog::warn("GraphCompiler: {}", error);
                output_shape = input_shape;
            }
            break;
        }

        case gui::NodeType::TensorSum:
        case gui::NodeType::TensorMean:
        case gui::NodeType::TensorMax:
        case gui::NodeType::TensorMin:
        case gui::NodeType::TensorProd:
        case gui::NodeType::TensorVar:
        case gui::NodeType::TensorStd: {
            std::string error;
            if (!ResolveReductionTargetShape(layer.type,
                                             layer.parameters,
                                             input_shape,
                                             output_shape,
                                             error)) {
                spdlog::warn("GraphCompiler: {}", error);
                output_shape = input_shape;
            }
            break;
        }

        // Layers/activations that don't change shape
        case gui::NodeType::Dropout:
        case gui::NodeType::BatchNorm:
        case gui::NodeType::ReLU:
        case gui::NodeType::LeakyReLU:
        case gui::NodeType::ELU:
        case gui::NodeType::GELU:
        case gui::NodeType::Swish:
        case gui::NodeType::Mish:
        case gui::NodeType::Sigmoid:
        case gui::NodeType::Tanh:
        case gui::NodeType::Softmax:
        case gui::NodeType::TensorAbs:
        case gui::NodeType::TensorExp:
        case gui::NodeType::TensorLog:
        case gui::NodeType::TensorSqrt:
        case gui::NodeType::TensorSign:
        case gui::NodeType::TensorPow:
        case gui::NodeType::TensorClip:
        case gui::NodeType::TensorCompare:
        case gui::NodeType::TensorLogicalMask:
            output_shape = input_shape;
            break;

        default:
            output_shape = input_shape;
            break;
    }

    return output_shape;
}

bool GraphCompiler::HasCycle(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links) const
{
    // Build adjacency list
    std::map<int, std::vector<int>> adj;
    for (const auto& node : nodes) {
        adj[node.id] = {};
    }
    for (const auto& link : links) {
        adj[link.from_node].push_back(link.to_node);
    }

    // DFS-based cycle detection
    std::set<int> white;  // Not visited
    std::set<int> gray;   // Currently visiting
    std::set<int> black;  // Fully processed

    for (const auto& node : nodes) {
        white.insert(node.id);
    }

    std::function<bool(int)> dfs = [&](int node_id) -> bool {
        white.erase(node_id);
        gray.insert(node_id);

        for (int neighbor : adj[node_id]) {
            if (gray.count(neighbor)) {
                // Back edge found - cycle exists
                return true;
            }
            if (white.count(neighbor) && dfs(neighbor)) {
                return true;
            }
        }

        gray.erase(node_id);
        black.insert(node_id);
        return false;
    };

    while (!white.empty()) {
        int node_id = *white.begin();
        if (dfs(node_id)) {
            return true;
        }
    }

    return false;
}

bool GraphCompiler::IsFullyConnected(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links) const
{
    if (nodes.empty()) return true;

    // BFS from first node
    std::set<int> visited;
    std::queue<int> queue;

    // Build undirected adjacency
    std::map<int, std::set<int>> adj;
    for (const auto& node : nodes) {
        adj[node.id] = {};
    }
    for (const auto& link : links) {
        adj[link.from_node].insert(link.to_node);
        adj[link.to_node].insert(link.from_node);
    }

    queue.push(nodes[0].id);
    visited.insert(nodes[0].id);

    while (!queue.empty()) {
        int node_id = queue.front();
        queue.pop();

        for (int neighbor : adj[node_id]) {
            if (!visited.count(neighbor)) {
                visited.insert(neighbor);
                queue.push(neighbor);
            }
        }
    }

    return visited.size() == nodes.size();
}

// ---------------------------------------------------------------------------
// Pin-connectivity validation
//
// The runtime path (TrainingExecutor) currently bypasses graph topology
// and reads the dataset by name from DataRegistry, so a graph with a
// disconnected Loss.Targets pin will train anyway and silently produce
// nonsense gradients. The architectural fix is the multi-day "walk
// pins, not registry" rewrite tracked in tofix.md. Until that lands,
// these compile-time checks make the canvas the source of truth: the
// user can't ship a graph whose required pins are unwired or whose
// label stream never reaches the loss.
//
// The checks here are deliberately conservative — they only flag
// situations that are unambiguously wrong. Anything that depends on
// runtime semantics (e.g. "this Tensor pin produces a label-shaped
// tensor in practice") is left alone to avoid false positives.
// ---------------------------------------------------------------------------

void GraphCompiler::ValidateRequiredInputsConnected(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    TrainingConfiguration& config) const
{
    // Build the set of input pin IDs that have at least one incoming link.
    std::unordered_set<int> connected_input_pins;
    connected_input_pins.reserve(links.size());
    for (const auto& link : links) {
        connected_input_pins.insert(link.to_pin);
    }

    // Walk every node's input pins and flag required ones with no incoming
    // link. Output pins are skipped intentionally — the existing pin
    // metadata defaults `is_required=true` on every output, which would
    // generate false positives for legitimately-optional outputs (LSTM's
    // Hidden, MultiHeadAttention's Attn Weights, DataSplit's Val/Test
    // streams when the user only trains). When those get explicit
    // `is_required=false` markers, output validation can join.
    for (const auto& node : nodes) {
        for (const auto& pin : node.inputs) {
            if (!pin.is_required) continue;
            if (connected_input_pins.count(pin.id) > 0) continue;

            std::ostringstream msg;
            msg << "Required input pin '" << pin.name
                << "' on node '" << node.name
                << "' has no incoming connection";
            AddIssue(config, IssueLevel::Error, msg.str(),
                     node.id, node.name,
                     errors::Compiler::InvalidConnectivity);
        }
    }
}

void GraphCompiler::ValidateRequiredOutputsConnected(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    TrainingConfiguration& config) const
{
    // Output-side symmetry of ValidateRequiredInputsConnected. Catches
    // "dangling" producers: a DataLoader whose Labels output goes
    // nowhere, a Normalize whose Output isn't consumed, etc. Only
    // outputs that were explicitly marked is_required=false (LSTM's
    // Hidden, attention's Attn Weights, DataSplit's Val/Test,
    // Output.Predictions, Optimizer.State) are skipped — everything
    // else must have at least one downstream consumer.
    std::unordered_set<int> connected_output_pins;
    connected_output_pins.reserve(links.size());
    for (const auto& link : links) {
        connected_output_pins.insert(link.from_pin);
    }

    for (const auto& node : nodes) {
        for (const auto& pin : node.outputs) {
            if (!pin.is_required) continue;
            if (connected_output_pins.count(pin.id) > 0) continue;

            std::ostringstream msg;
            msg << "Required output pin '" << pin.name
                << "' on node '" << node.name
                << "' has no outgoing connection - the produced data is "
                   "never consumed";
            AddIssue(config, IssueLevel::Error, msg.str(),
                     node.id, node.name,
                     errors::Compiler::InvalidConnectivity);
        }
    }
}

void GraphCompiler::ValidateLossTargetsReachLabels(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    TrainingConfiguration& config) const
{
    // Build a quick (node_id, pin_id) → pin lookup so we can recover pin
    // type during the BFS. NodePin IDs are unique across the graph, so a
    // flat pin_id → pointer map is enough.
    std::unordered_map<int, const gui::NodePin*> pin_by_id;
    std::unordered_map<int, int> pin_to_node;  // pin_id → owning node_id
    for (const auto& node : nodes) {
        for (const auto& pin : node.inputs) {
            pin_by_id[pin.id] = &pin;
            pin_to_node[pin.id] = node.id;
        }
        for (const auto& pin : node.outputs) {
            pin_by_id[pin.id] = &pin;
            pin_to_node[pin.id] = node.id;
        }
    }

    // Reverse adjacency: input pin ID → list of upstream output pin IDs
    // feeding it. Lets us walk backwards from any input pin.
    std::unordered_map<int, std::vector<int>> upstream_outputs;
    upstream_outputs.reserve(links.size());
    for (const auto& link : links) {
        upstream_outputs[link.to_pin].push_back(link.from_pin);
    }

    // Node-id → vector<input pin id> so that when BFS arrives at a node
    // (via an output pin), we can step into its inputs and continue the
    // walk upstream through the node's own inputs.
    std::unordered_map<int, std::vector<int>> node_input_pins;
    node_input_pins.reserve(nodes.size());
    for (const auto& node : nodes) {
        std::vector<int> ids;
        ids.reserve(node.inputs.size());
        for (const auto& pin : node.inputs) ids.push_back(pin.id);
        node_input_pins[node.id] = std::move(ids);
    }

    auto is_loss_node = [](gui::NodeType t) {
        return t == gui::NodeType::MSELoss ||
               t == gui::NodeType::CrossEntropyLoss ||
               t == gui::NodeType::FocalLoss ||
               t == gui::NodeType::BCELoss ||
               t == gui::NodeType::BCEWithLogits ||
               t == gui::NodeType::L1Loss ||
               t == gui::NodeType::SmoothL1Loss ||
               t == gui::NodeType::HuberLoss ||
               t == gui::NodeType::NLLLoss ||
               t == gui::NodeType::SoftDiceLoss ||
               t == gui::NodeType::TverskyLoss ||
               t == gui::NodeType::JaccardLoss;
    };

    // For each loss node, find its Targets input pin (the one tagged
    // PinType::Labels) and BFS backwards. If no ancestor output pin is
    // PinType::Labels, the targets stream isn't real labels — the user
    // wired Targets to something else (a model layer's Tensor output,
    // a random preprocessing tensor, etc.). That's the canonical "pin
    // is fooling user" case.
    for (const auto& node : nodes) {
        if (!is_loss_node(node.type)) continue;

        // Identify Targets pin. Every loss node above creates the
        // Targets input as PinType::Labels (see node_editor_nodes.cpp);
        // older graphs may have it as PinType::Tensor — fall back to
        // matching by name.
        const gui::NodePin* targets_pin = nullptr;
        for (const auto& pin : node.inputs) {
            if (pin.type == gui::PinType::Labels) {
                targets_pin = &pin;
                break;
            }
        }
        if (!targets_pin) {
            for (const auto& pin : node.inputs) {
                if (pin.name == "Targets") {
                    targets_pin = &pin;
                    break;
                }
            }
        }
        if (!targets_pin) continue;  // Loss with no Targets input — can't check.

        // If the Targets pin is unconnected, the required-input check
        // will have already flagged it; skip to avoid duplicate errors.
        if (upstream_outputs.find(targets_pin->id) == upstream_outputs.end()) {
            continue;
        }

        // BFS backwards from targets_pin. State is the queue of input
        // pin IDs to walk upstream from. visited tracks visited input
        // pin IDs to bound the walk.
        std::queue<int> queue;
        std::unordered_set<int> visited_inputs;
        queue.push(targets_pin->id);
        visited_inputs.insert(targets_pin->id);

        bool found_labels_source = false;
        while (!queue.empty() && !found_labels_source) {
            int input_pin_id = queue.front();
            queue.pop();

            auto it = upstream_outputs.find(input_pin_id);
            if (it == upstream_outputs.end()) continue;

            for (int upstream_out_pin_id : it->second) {
                auto pin_it = pin_by_id.find(upstream_out_pin_id);
                if (pin_it == pin_by_id.end()) continue;
                if (pin_it->second->type == gui::PinType::Labels) {
                    found_labels_source = true;
                    break;
                }
                // Not a Labels source itself — keep walking upstream
                // through this output pin's owning node's inputs.
                auto owner_it = pin_to_node.find(upstream_out_pin_id);
                if (owner_it == pin_to_node.end()) continue;
                auto inputs_it = node_input_pins.find(owner_it->second);
                if (inputs_it == node_input_pins.end()) continue;
                for (int upstream_in_pin_id : inputs_it->second) {
                    if (visited_inputs.insert(upstream_in_pin_id).second) {
                        queue.push(upstream_in_pin_id);
                    }
                }
            }
        }

        if (!found_labels_source) {
            std::ostringstream msg;
            msg << "Loss node '" << node.name
                << "' has its Targets pin wired, but the upstream chain "
                   "never passes through a Labels-typed pin. Use "
                   "DataLoader.Labels for new graphs; DataInput.Labels and "
                   "DataSplit.*Labels are legacy compatibility sources. "
                   "The model is being trained "
                   "against the wrong stream.";
            AddIssue(config, IssueLevel::Error, msg.str(),
                     node.id, node.name,
                     errors::Compiler::InvalidConnectivity);
        }
    }
}

void GraphCompiler::ValidateLossPredictionsReachModel(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    TrainingConfiguration& config) const
{
    // Symmetric to ValidateLossTargetsReachLabels: BFS upstream from
    // each Loss node's Predictions pin, but the success criterion is
    // "ancestor node is a model layer or Output node" instead of
    // "ancestor pin is PinType::Labels". Catches the case where a user
    // wires Predictions to a dataset/source path, a Normalize output, or
    // (worst) the Labels stream — all of which would silently train
    // garbage before the runtime fix lands.
    std::unordered_map<int, int> pin_to_node;  // pin_id → owning node_id
    for (const auto& node : nodes) {
        for (const auto& pin : node.inputs)  pin_to_node[pin.id] = node.id;
        for (const auto& pin : node.outputs) pin_to_node[pin.id] = node.id;
    }

    std::unordered_map<int, std::vector<int>> upstream_outputs;
    upstream_outputs.reserve(links.size());
    for (const auto& link : links) {
        upstream_outputs[link.to_pin].push_back(link.from_pin);
    }

    std::unordered_map<int, std::vector<int>> node_input_pins;
    node_input_pins.reserve(nodes.size());
    for (const auto& node : nodes) {
        std::vector<int> ids;
        ids.reserve(node.inputs.size());
        for (const auto& pin : node.inputs) ids.push_back(pin.id);
        node_input_pins[node.id] = std::move(ids);
    }

    auto is_loss_node = [](gui::NodeType t) {
        return t == gui::NodeType::MSELoss ||
               t == gui::NodeType::CrossEntropyLoss ||
               t == gui::NodeType::FocalLoss ||
               t == gui::NodeType::BCELoss ||
               t == gui::NodeType::BCEWithLogits ||
               t == gui::NodeType::L1Loss ||
               t == gui::NodeType::SmoothL1Loss ||
               t == gui::NodeType::HuberLoss ||
               t == gui::NodeType::NLLLoss ||
               t == gui::NodeType::SoftDiceLoss ||
               t == gui::NodeType::TverskyLoss ||
               t == gui::NodeType::JaccardLoss;
    };

    // Build node_id → node* lookup so the BFS can check owning node
    // type without a linear search per visited pin.
    std::unordered_map<int, const gui::MLNode*> node_by_id;
    node_by_id.reserve(nodes.size());
    for (const auto& node : nodes) node_by_id[node.id] = &node;

    for (const auto& node : nodes) {
        if (!is_loss_node(node.type)) continue;

        // Predictions is the FIRST input on every loss node above —
        // it's PinType::Tensor. Match by name "Predictions" for older
        // graphs that may not preserve ordering.
        const gui::NodePin* predictions_pin = nullptr;
        for (const auto& pin : node.inputs) {
            if (pin.name == "Predictions") {
                predictions_pin = &pin;
                break;
            }
        }
        if (!predictions_pin && !node.inputs.empty()) {
            // Fall back to the first input pin — older loss nodes may
            // not have set the name explicitly.
            predictions_pin = &node.inputs[0];
        }
        if (!predictions_pin) continue;

        // Already-flagged disconnect — skip to avoid duplicate errors.
        if (upstream_outputs.find(predictions_pin->id) == upstream_outputs.end()) {
            continue;
        }

        std::queue<int> queue;
        std::unordered_set<int> visited_inputs;
        queue.push(predictions_pin->id);
        visited_inputs.insert(predictions_pin->id);

        bool found_model_source = false;
        while (!queue.empty() && !found_model_source) {
            int input_pin_id = queue.front();
            queue.pop();

            auto it = upstream_outputs.find(input_pin_id);
            if (it == upstream_outputs.end()) continue;

            for (int upstream_out_pin_id : it->second) {
                auto owner_it = pin_to_node.find(upstream_out_pin_id);
                if (owner_it == pin_to_node.end()) continue;
                auto node_it = node_by_id.find(owner_it->second);
                if (node_it == node_by_id.end()) continue;

                const gui::MLNode* upstream_node = node_it->second;
                if (IsOutputNodeType(upstream_node->type) ||
                    IsModelLayer(upstream_node->type)) {
                    found_model_source = true;
                    break;
                }

                // Activation nodes (ReLU/Sigmoid/etc.) are NOT in
                // IsModelLayer but a chain ending Activation → Loss
                // is legitimate (the Activation's own input traces
                // back to a model layer). Keep walking upstream
                // through this node's inputs.
                auto inputs_it = node_input_pins.find(owner_it->second);
                if (inputs_it == node_input_pins.end()) continue;
                for (int upstream_in_pin_id : inputs_it->second) {
                    if (visited_inputs.insert(upstream_in_pin_id).second) {
                        queue.push(upstream_in_pin_id);
                    }
                }
            }
        }

        if (!found_model_source) {
            std::ostringstream msg;
            msg << "Loss node '" << node.name
                << "' has its Predictions pin wired, but the upstream "
                   "chain never passes through a model layer or Output "
                   "node. The loss is being computed against a "
                   "non-prediction tensor (often the Labels stream "
                   "wired by mistake, or a raw DataInput tensor with "
                   "no model in between).";
            AddIssue(config, IssueLevel::Error, msg.str(),
                     node.id, node.name,
                     errors::Compiler::InvalidConnectivity);
        }
    }
}

void GraphCompiler::ValidateOptimizerReachesLoss(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    TrainingConfiguration& config) const
{
    // Closes the last visible pin vector in the compile → loss → optimizer
    // chain. The pin-type system's "Tensor is universal" rule means the
    // user CAN wire a Dense.Output (Tensor) directly into Optimizer.Loss
    // (Loss) — the ImNodes link validation won't block it. The required-
    // input check then passes because the pin has SOME incoming wire.
    // Without this BFS, a graph with Optimizer.Loss fed by a random
    // tensor would train with nonsense gradients.
    std::unordered_map<int, int> pin_to_node;
    for (const auto& node : nodes) {
        for (const auto& pin : node.inputs)  pin_to_node[pin.id] = node.id;
        for (const auto& pin : node.outputs) pin_to_node[pin.id] = node.id;
    }

    std::unordered_map<int, std::vector<int>> upstream_outputs;
    upstream_outputs.reserve(links.size());
    for (const auto& link : links) {
        upstream_outputs[link.to_pin].push_back(link.from_pin);
    }

    std::unordered_map<int, std::vector<int>> node_input_pins;
    node_input_pins.reserve(nodes.size());
    for (const auto& node : nodes) {
        std::vector<int> ids;
        ids.reserve(node.inputs.size());
        for (const auto& pin : node.inputs) ids.push_back(pin.id);
        node_input_pins[node.id] = std::move(ids);
    }

    std::unordered_map<int, const gui::MLNode*> node_by_id;
    node_by_id.reserve(nodes.size());
    for (const auto& node : nodes) node_by_id[node.id] = &node;

    auto is_optimizer = [](gui::NodeType t) {
        return t == gui::NodeType::SGD ||
               t == gui::NodeType::Adam ||
               t == gui::NodeType::AdamW ||
               t == gui::NodeType::RMSprop ||
               t == gui::NodeType::Adagrad ||
               t == gui::NodeType::NAdam;
    };

    auto is_loss_node = [](gui::NodeType t) {
        return t == gui::NodeType::MSELoss ||
               t == gui::NodeType::CrossEntropyLoss ||
               t == gui::NodeType::FocalLoss ||
               t == gui::NodeType::BCELoss ||
               t == gui::NodeType::BCEWithLogits ||
               t == gui::NodeType::L1Loss ||
               t == gui::NodeType::SmoothL1Loss ||
               t == gui::NodeType::HuberLoss ||
               t == gui::NodeType::NLLLoss ||
               t == gui::NodeType::SoftDiceLoss ||
               t == gui::NodeType::TverskyLoss ||
               t == gui::NodeType::JaccardLoss;
    };

    for (const auto& node : nodes) {
        if (!is_optimizer(node.type)) continue;

        // Optimizer nodes have a single Loss-typed input pin. Match by
        // PinType first (cleanest) with a name fallback for older graphs.
        const gui::NodePin* loss_pin = nullptr;
        for (const auto& pin : node.inputs) {
            if (pin.type == gui::PinType::Loss) { loss_pin = &pin; break; }
        }
        if (!loss_pin) {
            for (const auto& pin : node.inputs) {
                if (pin.name == "Loss") { loss_pin = &pin; break; }
            }
        }
        if (!loss_pin) continue;

        // Disconnect is handled by the required-input check.
        if (upstream_outputs.find(loss_pin->id) == upstream_outputs.end()) {
            continue;
        }

        std::queue<int> queue;
        std::unordered_set<int> visited_inputs;
        queue.push(loss_pin->id);
        visited_inputs.insert(loss_pin->id);

        bool found_loss_node = false;
        while (!queue.empty() && !found_loss_node) {
            int input_pin_id = queue.front();
            queue.pop();

            auto it = upstream_outputs.find(input_pin_id);
            if (it == upstream_outputs.end()) continue;

            for (int upstream_out_pin_id : it->second) {
                auto owner_it = pin_to_node.find(upstream_out_pin_id);
                if (owner_it == pin_to_node.end()) continue;
                auto node_it = node_by_id.find(owner_it->second);
                if (node_it == node_by_id.end()) continue;

                if (is_loss_node(node_it->second->type)) {
                    found_loss_node = true;
                    break;
                }

                // Not a loss node — keep walking. Unlikely to hit in
                // practice (optimizer is normally one hop from a loss)
                // but the walk stays general for the edge case where
                // the user inserted something between loss and
                // optimizer.
                auto inputs_it = node_input_pins.find(owner_it->second);
                if (inputs_it == node_input_pins.end()) continue;
                for (int upstream_in_pin_id : inputs_it->second) {
                    if (visited_inputs.insert(upstream_in_pin_id).second) {
                        queue.push(upstream_in_pin_id);
                    }
                }
            }
        }

        if (!found_loss_node) {
            std::ostringstream msg;
            msg << "Optimizer node '" << node.name
                << "' has its Loss pin wired, but the upstream chain "
                   "never passes through a loss function (MSELoss, "
                   "CrossEntropyLoss, etc.). The optimizer will "
                   "backprop from a non-loss tensor — training is "
                   "meaningless.";
            AddIssue(config, IssueLevel::Error, msg.str(),
                     node.id, node.name,
                     errors::Compiler::InvalidConnectivity);
        }
    }
}

} // namespace cyxwiz
