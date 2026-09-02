#include "pipeline_materializer.h"

#include "node_executors/pipeline_operator.h"
#include "node_executors/pipeline_operator_factory.h"
#include "materialization_memory_guard.h"
#include "pipeline_runtime_capabilities.h"
#include "process_memory_snapshot.h"

#include <arrow/table.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <map>
#include <new>
#include <optional>
#include <queue>
#include <unordered_set>
#include <vector>

namespace cyxwiz {

namespace {

// Deliberately not derived from std::exception: several operators translate
// backend std::exceptions into Arrow errors. This private control signal must
// reach the materializer boundary without being relabelled by those catches.
class MaterializationCancelled final {};

struct MaterializationCapacityExceeded final {
    std::string message;
};

struct MaterializationPreflightComplete final {
    PipelineOperatorProgress event;
};

void SetMaterializationFailure(
    MaterializeTableResult& result,
    MaterializationFailureKind kind,
    std::string message,
    const gui::MLNode* node = nullptr) {
    result.success = false;
    result.failure_kind = kind;
    result.error_message = std::move(message);
    if (node) {
        result.failed_node_id = node->id;
        result.failed_node_name = node->name;
    }
}

bool StopIfMaterializationCancelled(
    const PipelineOperatorExecutionContext& context,
    MaterializeTableResult& result,
    const gui::MLNode* node = nullptr) {
    if (!context.IsCancellationRequested()) return false;
    SetMaterializationFailure(
        result,
        MaterializationFailureKind::Cancelled,
        node
            ? "PipelineMaterializer: cancelled at node '" + node->name + "'"
            : "PipelineMaterializer: cancelled",
        node);
    return true;
}

bool IsDataInputNode(const gui::MLNode& node) {
    return node.type == gui::NodeType::DataInput ||
           node.type == gui::NodeType::DatasetInput;
}

std::string DatasetNameForNode(const gui::MLNode& node) {
    auto dataset_name = node.parameters.find("dataset_name");
    if (dataset_name != node.parameters.end() && !dataset_name->second.empty()) {
        return dataset_name->second;
    }

    auto legacy_dataset = node.parameters.find("dataset");
    if (legacy_dataset != node.parameters.end() && !legacy_dataset->second.empty()) {
        return legacy_dataset->second;
    }

    return {};
}

const gui::MLNode* FindDataInputNode(
    const std::vector<gui::MLNode>& nodes,
    const std::string& source_dataset_name) {

    if (!source_dataset_name.empty()) {
        for (const auto& n : nodes) {
            if (IsDataInputNode(n) &&
                DatasetNameForNode(n) == source_dataset_name) {
                return &n;
            }
        }
        return nullptr;
    }

    for (const auto& n : nodes) {
        if (IsDataInputNode(n)) {
            return &n;
        }
    }
    return nullptr;
}

const gui::MLNode* FindNodeById(int id, const std::vector<gui::MLNode>& nodes) {
    for (const auto& n : nodes) {
        if (n.id == id) return &n;
    }
    return nullptr;
}

bool IsFoldedTextConfigNode(gui::NodeType type) {
    return type == gui::NodeType::TextVocabulary ||
           type == gui::NodeType::TextPadding;
}

std::optional<gui::NodeType> ResolveArrowTableMaterializerOperatorType(
    gui::NodeType type) {

    const auto support = ResolvePipelineRuntimeSupport(type);
    if (support.mode != PipelineRuntimeSupportMode::OperatorBacked ||
        support.implementation_owner !=
            PipelineRuntimeImplementationOwner::PipelineOperatorFactory ||
        !support.materializer_arrow_table_supported ||
        support.materializer_storage_support !=
            PipelineMaterializerStorageSupport::ArrowTableOnly ||
        !support.operator_type.has_value()) {
        return std::nullopt;
    }

    return support.operator_type;
}

bool IsArrowTableMaterializerOperator(gui::NodeType type) {
    return ResolveArrowTableMaterializerOperatorType(type).has_value();
}

bool RequestsSparseFeatureOutput(
    const std::map<std::string, std::string>& params) {
    auto output_format = params.find("output_format");
    if (output_format == params.end()) return false;
    std::string value = output_format->second;
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    return value == "sparse";
}

bool HasReachableMaterializerOperator(
    int node_id,
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    std::unordered_set<int>& visiting);

bool ValidateSparseOutputIsTerminal(
    const gui::MLNode& node,
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    std::string& error) {
    for (const auto& link : links) {
        if (link.from_node != node.id) continue;
        std::unordered_set<int> visiting;
        if (HasReachableMaterializerOperator(
                link.to_node, nodes, links, visiting)) {
            error = "PipelineMaterializer: sparse output from node '" +
                    node.name + "' must be the final preprocessing output";
            return false;
        }
    }
    return true;
}

bool HasReachableMaterializerOperator(
    int node_id,
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    std::unordered_set<int>& visiting) {

    if (!visiting.insert(node_id).second) {
        return false;
    }

    const gui::MLNode* node = FindNodeById(node_id, nodes);
    if (!node) {
        return false;
    }

    if (IsArrowTableMaterializerOperator(node->type)) {
        return true;
    }

    for (const auto& link : links) {
        if (link.from_node != node_id) {
            continue;
        }
        if (HasReachableMaterializerOperator(
                link.to_node, nodes, links, visiting)) {
            return true;
        }
    }

    return false;
}

bool HasReachableCycle(
    int node_id,
    const std::vector<gui::NodeLink>& links,
    std::unordered_set<int>& visiting,
    std::unordered_set<int>& visited,
    int& cycle_node_id) {

    if (visiting.find(node_id) != visiting.end()) {
        cycle_node_id = node_id;
        return true;
    }
    if (visited.find(node_id) != visited.end()) {
        return false;
    }

    visiting.insert(node_id);
    for (const auto& link : links) {
        if (link.from_node != node_id) {
            continue;
        }
        if (HasReachableCycle(
                link.to_node, links, visiting, visited, cycle_node_id)) {
            return true;
        }
    }

    visiting.erase(node_id);
    visited.insert(node_id);
    return false;
}

bool ValidateLinearMaterializerOperatorPath(
    const gui::MLNode& data_input,
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    std::string& error) {

    std::unordered_set<int> cycle_visiting;
    std::unordered_set<int> cycle_checked;
    int cycle_node_id = -1;
    if (HasReachableCycle(
            data_input.id, links, cycle_visiting, cycle_checked,
            cycle_node_id)) {
        const gui::MLNode* node = FindNodeById(cycle_node_id, nodes);
        const std::string node_name =
            node ? node->name : std::to_string(cycle_node_id);
        error = "PipelineMaterializer: cyclic graph path involving node '" +
                node_name +
                "' is not supported by the Arrow-table materializer";
        return false;
    }

    std::queue<int> queue;
    std::unordered_set<int> visited;
    queue.push(data_input.id);
    visited.insert(data_input.id);

    while (!queue.empty()) {
        const int node_id = queue.front();
        queue.pop();

        const gui::MLNode* current_node = FindNodeById(node_id, nodes);
        if (current_node &&
            RequestsSparseFeatureOutput(current_node->parameters) &&
            !ValidateSparseOutputIsTerminal(
                *current_node, nodes, links, error)) {
            return false;
        }

        std::vector<int> operator_relevant_children;
        for (const auto& link : links) {
            if (link.from_node != node_id) {
                continue;
            }

            std::unordered_set<int> visiting;
            if (HasReachableMaterializerOperator(
                    link.to_node, nodes, links, visiting)) {
                operator_relevant_children.push_back(link.to_node);
            }

            if (visited.insert(link.to_node).second) {
                queue.push(link.to_node);
            }
        }

        if (operator_relevant_children.size() > 1) {
            const gui::MLNode* node = FindNodeById(node_id, nodes);
            const std::string node_name =
                node ? node->name : std::to_string(node_id);
            error = "PipelineMaterializer: branched operator paths from node '" +
                    node_name +
                    "' are not supported by the Arrow-table materializer";
            return false;
        }
    }

    return true;
}

void FoldTextConfigNodeParams(
    const gui::MLNode& config_node,
    std::map<std::string, std::string>& params) {

    if (config_node.type == gui::NodeType::TextVocabulary) {
        auto min_freq = config_node.parameters.find("min_freq");
        if (min_freq != config_node.parameters.end()) {
            params["min_word_freq"] = min_freq->second;
        }
        auto max_vocab = config_node.parameters.find("max_vocab_size");
        if (max_vocab != config_node.parameters.end()) {
            params["max_vocab_size"] = max_vocab->second;
        }
        auto vocab_file = config_node.parameters.find("vocab_file");
        if (vocab_file != config_node.parameters.end()) {
            params["vocab_file"] = vocab_file->second;
            params["vocab_build_if_missing"] = "true";
        }
    } else if (config_node.type == gui::NodeType::TextPadding) {
        auto max_length = config_node.parameters.find("max_length");
        if (max_length != config_node.parameters.end()) {
            params["max_length"] = max_length->second;
        }
        auto pad_value = config_node.parameters.find("pad_value");
        if (pad_value != config_node.parameters.end()) {
            params["pad_value"] = pad_value->second;
        }
    }
}

void FoldReachableTextConfigParams(
    const gui::MLNode& tokenizer_node,
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    std::map<std::string, std::string>& params) {

    std::queue<int> queue;
    std::unordered_set<int> visited;
    queue.push(tokenizer_node.id);
    visited.insert(tokenizer_node.id);

    while (!queue.empty()) {
        const int node_id = queue.front();
        queue.pop();

        for (const auto& link : links) {
            if (link.from_node != node_id ||
                !visited.insert(link.to_node).second) {
                continue;
            }

            const gui::MLNode* child = FindNodeById(link.to_node, nodes);
            if (!child || !IsFoldedTextConfigNode(child->type)) {
                continue;
            }

            FoldTextConfigNodeParams(*child, params);
            queue.push(child->id);
        }
    }
}

std::map<std::string, std::string> BuildOperatorParams(
    const gui::MLNode& node,
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links) {

    auto params = node.parameters;
    if (node.type == gui::NodeType::TextTokenizer) {
        FoldReachableTextConfigParams(node, nodes, links, params);
    }
    return params;
}

} // namespace

MaterializationCacheability PipelineMaterializer::EvaluateCacheability(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    const std::string& source_dataset_name) {
    MaterializationCacheability result;
    const gui::MLNode* data_input =
        FindDataInputNode(nodes, source_dataset_name);
    if (!data_input) {
        result.cacheable = false;
        result.reason = "materialization source node is unavailable";
        return result;
    }

    std::string graph_shape_error;
    if (!ValidateLinearMaterializerOperatorPath(
            *data_input, nodes, links, graph_shape_error)) {
        result.cacheable = false;
        result.reason = graph_shape_error;
        return result;
    }

    auto& factory = PipelineOperatorFactory::Instance();
    std::queue<int> queue;
    std::unordered_set<int> visited;
    queue.push(data_input->id);
    visited.insert(data_input->id);
    while (!queue.empty()) {
        const int node_id = queue.front();
        queue.pop();
        const gui::MLNode* node = FindNodeById(node_id, nodes);
        if (!node) {
            continue;
        }

        const auto operator_type =
            ResolveArrowTableMaterializerOperatorType(node->type);
        if (node->id != data_input->id && operator_type.has_value()) {
            auto op = factory.Create(*operator_type);
            if (!op) {
                result.cacheable = false;
                result.reason = "operator factory could not inspect node '" +
                                node->name + "'";
                return result;
            }
            std::string error;
            const auto parameters = BuildOperatorParams(*node, nodes, links);
            if (!op->Configure(parameters, error)) {
                result.cacheable = false;
                result.reason = "node '" + node->name +
                                "' configuration is invalid: " + error;
                return result;
            }
            if (!op->IsCacheable()) {
                result.cacheable = false;
                result.reason = "node '" + node->name +
                                "' reads or writes fitted state";
                return result;
            }
            std::vector<PipelineOperatorCacheDependency> dependencies;
            if (!op->CollectCacheDependencies(dependencies, error)) {
                result.cacheable = false;
                result.valid = false;
                result.reason = "node '" + node->name +
                                "' cache dependency is invalid: " + error;
                return result;
            }
            for (const auto& dependency : dependencies) {
                MaterializationCacheDependencyIdentity identity;
                if (!ResolveMaterializationCacheDependencyIdentity(
                        node->id, dependency.role, dependency.path,
                        identity, &error)) {
                    result.cacheable = false;
                    result.valid = false;
                    result.reason = "node '" + node->name +
                                    "' cache dependency is invalid: " + error;
                    return result;
                }
                result.dependencies.push_back(std::move(identity));
            }
        }

        for (const auto& link : links) {
            if (link.from_node == node_id &&
                visited.insert(link.to_node).second) {
                queue.push(link.to_node);
            }
        }
    }
    return result;
}

MaterializeTableResult PipelineMaterializer::PreflightTable(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    const std::shared_ptr<arrow::Table>& source_table,
    const std::string& source_dataset_name,
    MaterializationMemoryContext memory_context) {
    PipelineOperatorExecutionContext execution_context;
    execution_context.memory = std::move(memory_context);
    execution_context.stop_after_memory_preflight = true;
    return MaterializeTable(
        nodes, links, source_table, source_dataset_name, {},
        std::move(execution_context));
}

MaterializeTableResult PipelineMaterializer::MaterializeTable(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    const std::shared_ptr<arrow::Table>& source_table,
    const std::string& source_dataset_name,
    PipelineOperatorProgressCallback progress_callback,
    PipelineOperatorExecutionContext execution_context) try {

    MaterializeTableResult result;
    result.table = source_table;

    if (StopIfMaterializationCancelled(execution_context, result)) {
        return result;
    }

    if (!source_table) {
        SetMaterializationFailure(
            result,
            MaterializationFailureKind::Error,
            "PipelineMaterializer: source Arrow table is null");
        return result;
    }

    const gui::MLNode* data_input = FindDataInputNode(nodes, source_dataset_name);
    if (!data_input) {
        if (!source_dataset_name.empty()) {
            SetMaterializationFailure(
                result,
                MaterializationFailureKind::Error,
                "PipelineMaterializer: source dataset '" +
                source_dataset_name +
                "' does not match any DataInput/DatasetInput node");
        }
        return result;
    }

    auto& factory = PipelineOperatorFactory::Instance();

    bool any_materializable = false;
    for (const auto& n : nodes) {
        if (n.id == data_input->id) continue;
        if (IsArrowTableMaterializerOperator(n.type)) {
            any_materializable = true;
            break;
        }
    }
    if (!any_materializable) {
        return result;
    }

    std::string graph_shape_error;
    if (!ValidateLinearMaterializerOperatorPath(
            *data_input, nodes, links, graph_shape_error)) {
        SetMaterializationFailure(
            result, MaterializationFailureKind::Error, graph_shape_error);
        return result;
    }

    auto current_table = source_table;
    const auto process_memory_baseline =
        execution_context.CaptureProcessMemory();
    const auto budget_snapshot = execution_context.memory.snapshot_override
        .value_or(DetectMaterializationMemorySnapshot());
    const uint64_t safe_memory_budget_bytes = SaturatingScaleBytes(
        budget_snapshot.available_bytes,
        execution_context.memory.policy.blocked_fraction);
    std::queue<int> queue;
    std::unordered_set<int> visited;
    queue.push(data_input->id);
    visited.insert(data_input->id);

    while (!queue.empty()) {
        const int node_id = queue.front();
        queue.pop();

        const gui::MLNode* node = FindNodeById(node_id, nodes);
        if (!node) continue;
        if (StopIfMaterializationCancelled(execution_context, result, node)) {
            return result;
        }

        const auto operator_type =
            ResolveArrowTableMaterializerOperatorType(node->type);
        if (node->id != data_input->id && operator_type.has_value()) {
            auto op = factory.Create(*operator_type);
            if (!op) {
                SetMaterializationFailure(
                    result,
                    MaterializationFailureKind::Error,
                    "PipelineMaterializer: factory returned null for node '" +
                    node->name +
                    "' (runtime support allows materialization but Create failed)",
                    node);
                return result;
            }

            op->SetExecutionContext(execution_context);

            op->SetProgressCallback(
                [progress_callback,
                 execution_context,
                 process_memory_baseline,
                 budget_snapshot,
                 safe_memory_budget_bytes,
                 node](
                    const PipelineOperatorProgress& event) {
                    if (execution_context.IsCancellationRequested()) {
                        throw MaterializationCancelled();
                    }
                    auto with_node = event;
                    with_node.node_id = node->id;
                    with_node.node_name = node->name;
                    with_node.available_memory_bytes =
                        budget_snapshot.available_bytes;
                    with_node.safe_memory_budget_bytes =
                        safe_memory_budget_bytes;
                    const auto process_memory =
                        execution_context.CaptureProcessMemory();
                    with_node.process_memory_detected =
                        process_memory.detected;
                    with_node.process_resident_memory_bytes =
                        process_memory.resident_bytes;
                    with_node.process_private_memory_bytes =
                        process_memory.private_bytes;
                    with_node.process_private_memory_name =
                        process_memory.private_metric_name;
                    with_node.process_memory_source = process_memory.source;
                    if (process_memory.detected &&
                        process_memory_baseline.detected &&
                        process_memory.resident_bytes >
                            process_memory_baseline.resident_bytes) {
                        with_node.process_resident_growth_bytes =
                            process_memory.resident_bytes -
                            process_memory_baseline.resident_bytes;
                    }
                    const uint64_t configured_limit =
                        execution_context.memory.policy.hard_limit_bytes;
                    const uint64_t runtime_limit = configured_limit > 0 &&
                            safe_memory_budget_bytes > 0
                        ? std::min(configured_limit,
                                   safe_memory_budget_bytes)
                        : configured_limit > 0
                            ? configured_limit
                            : safe_memory_budget_bytes;
                    // Estimate-only preflight must report the operator's
                    // projected risk. Incidental process growth while setting
                    // up the preflight is not materialization growth and must
                    // not reclassify a warning as a runtime block.
                    if (!execution_context.stop_after_memory_preflight &&
                        runtime_limit > 0 &&
                        with_node.process_resident_growth_bytes >=
                            runtime_limit) {
                        with_node.status = "blocked";
                        with_node.memory_risk_level = "blocked";
                        with_node.message =
                            "Runtime memory guard stopped node '" +
                            node->name + "': process resident growth " +
                            FormatMaterializationBytes(
                                with_node.process_resident_growth_bytes) +
                            " reached the materialization limit " +
                            FormatMaterializationBytes(runtime_limit) +
                            ". Reduce input rows or output dimensions and retry.";
                        if (progress_callback) {
                            progress_callback(with_node);
                        }
                        throw MaterializationCapacityExceeded{
                            with_node.message};
                    }
                    if (progress_callback) {
                        progress_callback(with_node);
                    }
                    if (execution_context.stop_after_memory_preflight &&
                        !with_node.memory_risk_level.empty()) {
                        throw MaterializationPreflightComplete{
                            std::move(with_node)};
                    }
                    if (execution_context.IsCancellationRequested()) {
                        throw MaterializationCancelled();
                    }
                });

            std::string err;
            auto params = BuildOperatorParams(*node, nodes, links);
            if (!op->Configure(params, err)) {
                SetMaterializationFailure(
                    result,
                    MaterializationFailureKind::Error,
                    "PipelineMaterializer: Configure failed for node '" +
                    node->name + "': " + err,
                    node);
                return result;
            }

            const bool sparse_output = RequestsSparseFeatureOutput(params);
            if (sparse_output && !op->SupportsSparseFeatureOutput()) {
                SetMaterializationFailure(
                    result,
                    MaterializationFailureKind::Error,
                    "PipelineMaterializer: node '" + node->name +
                        "' requested sparse output but its operator does not "
                        "support SparseFeatureDataset publication",
                    node);
                return result;
            }
            if (sparse_output && !ValidateSparseOutputIsTerminal(
                    *node, nodes, links, err)) {
                SetMaterializationFailure(
                    result, MaterializationFailureKind::Error, err, node);
                return result;
            }

            arrow::Status apply_status = arrow::Status::OK();
            try {
                if (sparse_output) {
                    std::string sparse_name = source_dataset_name.empty()
                        ? DatasetNameForNode(*data_input)
                        : source_dataset_name;
                    if (sparse_name.empty()) {
                        sparse_name = "pipeline_sparse_features";
                    }
                    sparse_name += PipelineMaterializer::kMaterializedSuffix;
                    auto applied = op->ApplySparse(current_table, sparse_name);
                    if (applied.ok()) {
                        result.sparse_dataset = applied.ValueOrDie();
                    } else {
                        apply_status = applied.status();
                    }
                } else {
                    auto applied = op->Apply(current_table);
                    if (applied.ok()) {
                        current_table = applied.ValueOrDie();
                    } else {
                        apply_status = applied.status();
                    }
                }
            } catch (const MaterializationCancelled&) {
                SetMaterializationFailure(
                    result,
                    MaterializationFailureKind::Cancelled,
                    "PipelineMaterializer: cancelled at node '" +
                        node->name + "'",
                    node);
                return result;
            } catch (const MaterializationCapacityExceeded& capacity) {
                SetMaterializationFailure(
                    result,
                    MaterializationFailureKind::Capacity,
                    capacity.message,
                    node);
                return result;
            } catch (MaterializationPreflightComplete& preflight) {
                result.memory_preflight_observed = true;
                result.memory_preflight = std::move(preflight.event);
                result.table = source_table;
                return result;
            } catch (const std::bad_alloc&) {
                SetMaterializationFailure(
                    result,
                    MaterializationFailureKind::Capacity,
                    "PipelineMaterializer: allocation failed for node '" +
                        node->name +
                        "'. Reduce input rows or output dimensions and retry.",
                    node);
                return result;
            }
            if (!apply_status.ok()) {
                const auto failure_kind = apply_status.IsCancelled()
                    ? MaterializationFailureKind::Cancelled
                    : apply_status.IsCapacityError()
                        ? MaterializationFailureKind::Capacity
                        : MaterializationFailureKind::Error;
                SetMaterializationFailure(
                    result,
                    failure_kind,
                    "PipelineMaterializer: Apply failed for node '" +
                        node->name + "': " + apply_status.message(),
                    node);
                return result;
            }

            if (StopIfMaterializationCancelled(
                    execution_context, result, node)) {
                return result;
            }

            ++result.operators_applied;
            spdlog::info("PipelineMaterializer: applied operator '{}' on node '{}'",
                         op->GetName(), node->name);
        }

        for (const auto& link : links) {
            if (link.from_node == node_id &&
                visited.insert(link.to_node).second) {
                queue.push(link.to_node);
            }
        }
    }

    result.table = current_table;
    return result;
} catch (const std::bad_alloc&) {
    MaterializeTableResult result;
    result.table = source_table;
    SetMaterializationFailure(
        result,
        MaterializationFailureKind::Capacity,
        "PipelineMaterializer: allocation failed while preparing the operator "
        "pipeline. Reduce input rows or output dimensions and retry.");
    return result;
}

} // namespace cyxwiz
