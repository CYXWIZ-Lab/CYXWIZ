#include "../src/core/arrow_dataset.h"
#include "../src/core/async_task_manager.h"
#include "../src/core/data_registry.h"
#include "../src/core/dataset_batcher.h"
#include "../src/core/model_builder.h"
#include "../src/core/node_executors/pipeline_operator_factory.h"
#include "../src/core/node_executors/text_tokenizer_operator.h"
#include "../src/core/parquet_backed_dataset.h"
#include "../src/core/pipeline_materializer.h"
#include "../src/core/pipeline_runtime_capabilities.h"
#include "../src/core/sequence_arrow_batcher.h"
#include "../src/core/sequence_tag_metrics.h"
#include "../src/core/training_run_comparison.h"
#include "../src/gui/graph_training_launcher.h"

#include <arrow/api.h>

#include <atomic>
#include <cmath>
#include <cctype>
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <nlohmann/json.hpp>

namespace {

constexpr const char* kDatasetName = "gui_text_runtime";
constexpr const char* kMaterializedDatasetName = "gui_text_runtime__materialized";
constexpr const char* kUnusedDatasetName = "unused_gui_text_runtime";
constexpr const char* kScopeArrowDatasetName = "gui_text_runtime_scope_arrow";
constexpr const char* kScopeParquetDatasetName = "gui_text_runtime_scope_parquet";
constexpr const char* kScopeImageDatasetName = "gui_text_runtime_scope_image";
constexpr const char* kScopeAudioDatasetName = "gui_text_runtime_scope_audio";
constexpr const char* kScopeTextDatasetName = "gui_text_runtime_scope_legacy_text";
constexpr const char* kSavedNerDatasetName = "gui_text_runtime_scope_ner";

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

using json = nlohmann::json;

bool WaitFor(const std::function<bool()>& predicate,
             std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    do {
        cyxwiz::AsyncTaskManager::Instance().ProcessCompletedCallbacks();
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (std::chrono::steady_clock::now() < deadline);

    cyxwiz::AsyncTaskManager::Instance().ProcessCompletedCallbacks();
    return predicate();
}

std::filesystem::path FindRepoRoot() {
    auto dir = std::filesystem::current_path();
    while (!dir.empty()) {
        if (std::filesystem::exists(dir / "cyxwiz-engine" / "CMakeLists.txt") &&
            std::filesystem::exists(dir / "examples" / "cyxgraph")) {
            return dir;
        }
        const auto parent = dir.parent_path();
        if (parent == dir) {
            break;
        }
        dir = parent;
    }
    return std::filesystem::current_path();
}

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    Check(in.is_open(), "could not open " + path.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void CheckFinite(float value, const std::string& message) {
    Check(std::isfinite(value), message);
}

bool ParseBoolValue(const std::string& value, bool fallback) {
    std::string normalized = value;
    for (auto& ch : normalized) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }

    if (normalized == "true" || normalized == "1" || normalized == "yes" ||
        normalized == "on") {
        return true;
    }
    if (normalized == "false" || normalized == "0" || normalized == "off" ||
        normalized == "no") {
        return false;
    }
    return fallback;
}

std::string JsonToString(const json& value) {
    if (value.is_string()) {
        return value.get<std::string>();
    }
    if (value.is_number_integer()) {
        return std::to_string(value.get<long long>());
    }
    if (value.is_number_float()) {
        std::ostringstream out;
        out << value.get<double>();
        return out.str();
    }
    if (value.is_boolean()) {
        return value.get<bool>() ? "true" : "false";
    }
    return value.dump();
}

std::map<std::string, std::string> ReadParametersFromJson(
    const json& params_json) {
    std::map<std::string, std::string> params;
    if (!params_json.is_object()) {
        return params;
    }
    for (auto it = params_json.begin(); it != params_json.end(); ++it) {
        params[it.key()] = JsonToString(it.value());
    }
    return params;
}

std::pair<std::vector<gui::MLNode>, std::vector<gui::NodeLink>>
LoadCyxGraphForTest(const std::filesystem::path& path) {
    const auto root = json::parse(ReadFile(path));
    Check(root.contains("nodes") && root["nodes"].is_array(),
          "saved graph should include nodes");
    Check(root.contains("links") && root["links"].is_array(),
          "saved graph should include links");

    std::vector<gui::MLNode> nodes;
    nodes.reserve(root["nodes"].size());
    for (const auto& node_json : root["nodes"]) {
        gui::MLNode node;
        node.id = node_json.value("id", 0);
        node.type = static_cast<gui::NodeType>(node_json.value("type", 0));
        node.name = node_json.value("name", std::string("node"));
        node.category = static_cast<gui::NodeCategory>(
            node_json.value("category", static_cast<int>(node.category)));
        if (node_json.contains("parameters")) {
            node.parameters = ReadParametersFromJson(node_json["parameters"]);
        }
        nodes.push_back(std::move(node));
    }

    std::vector<gui::NodeLink> links;
    links.reserve(root["links"].size());
    for (const auto& link_json : root["links"]) {
        gui::NodeLink link;
        link.id = link_json.value("id", 0);
        link.from_node = link_json.value("from_node", 0);
        link.to_node = link_json.value("to_node", 0);
        link.from_pin = link_json.value("from_pin", 0);
        link.to_pin = link_json.value("to_pin", 0);
        links.push_back(std::move(link));
    }

    return {nodes, links};
}

bool ParseBoolParam(const std::map<std::string, std::string>& params,
                   const std::string& key,
                   bool fallback) {
    auto it = params.find(key);
    if (it == params.end()) {
        return fallback;
    }
    return ParseBoolValue(it->second, fallback);
}

int ParseIntParam(const std::map<std::string, std::string>& params,
                 const std::string& key,
                 int fallback) {
    auto it = params.find(key);
    if (it == params.end()) {
        return fallback;
    }
    try {
        return std::stoi(it->second);
    } catch (...) {
        return fallback;
    }
}

float ParseFloatParam(const std::map<std::string, std::string>& params,
                     const std::string& key,
                     float fallback) {
    auto it = params.find(key);
    if (it == params.end()) {
        return fallback;
    }
    try {
        return std::stof(it->second);
    } catch (...) {
        return fallback;
    }
}

void AddLayerIfSequenceNode(const gui::MLNode& node,
                           cyxwiz::TrainingConfiguration& config) {
    if (node.type == gui::NodeType::Embedding ||
        node.type == gui::NodeType::Concatenate ||
        node.type == gui::NodeType::LSTM ||
        node.type == gui::NodeType::Dropout ||
        node.type == gui::NodeType::TimeDistributed) {
        cyxwiz::CompiledLayer layer;
        layer.type = node.type;
        layer.parameters = node.parameters;
        const auto it = node.parameters.find("units");
        if (it != node.parameters.end()) {
            try {
                layer.units = std::stoi(it->second);
            } catch (...) {
                layer.units = 0;
            }
        }
        config.layers.push_back(std::move(layer));
    }
}

cyxwiz::TrainingConfiguration MakeSavedNerGraphConfig(
    const std::vector<gui::MLNode>& nodes,
    const std::string& dataset_name,
    const std::filesystem::path& checkpoint_dir) {
    cyxwiz::TrainingConfiguration config;
    config.is_valid = true;
    config.dataset_name = dataset_name;
    config.preprocessing_domain = cyxwiz::PreprocessingDomain::Text;
    config.loss_type = gui::NodeType::CrossEntropyLoss;
    config.optimizer_type = gui::NodeType::Adam;
    config.learning_rate = 0.001f;
    config.train_ratio = 0.8f;
    config.val_ratio = 0.1f;
    config.test_ratio = 0.1f;
    config.split_seed = 42;
    config.batch_size = 32;
    config.epochs = 8;
    config.shuffle = true;
    config.drop_last = false;
    config.num_workers = 4;
    config.prefetch_factor = 2;
    config.log_interval = 10;
    config.validation_freq = 1;
    config.dataloader_seed = 42;
    config.grad_accum_steps = 1;
    config.save_best_checkpoint = true;
    config.early_stopping_patience = 5;
    config.checkpoint_dir = checkpoint_dir.string();
    config.sequence_batch.enabled = true;
    config.sequence_batch.token_column = "tokens";
    config.sequence_batch.pos_column = "pos_tags";
    config.sequence_batch.tag_column = "ner_tags";
    config.sequence_batch.sentence_id_column = "sentence_id";
    config.sequence_batch.create_attention_mask = true;
    config.sequence_batch.ignore_index = 0;
    config.sequence_batch.target_ignore_index = 0;
    config.sequence_batch.max_sequence_length = 96;

    for (const auto& node : nodes) {
        if (node.type == gui::NodeType::DataInput) {
            config.data_source_node_id = node.id;
        } else if (node.type == gui::NodeType::DataSplit) {
            config.has_data_split = true;
            config.train_ratio =
                ParseFloatParam(node.parameters, "train_ratio", 0.80f);
            config.val_ratio = ParseFloatParam(node.parameters, "val_ratio", 0.10f);
            config.test_ratio = ParseFloatParam(node.parameters, "test_ratio", 0.10f);
            config.split_seed = ParseIntParam(node.parameters, "seed", 42);
        } else if (node.type == gui::NodeType::DataLoader) {
            config.has_data_loader = true;
            config.batch_size = ParseIntParam(node.parameters, "batch_size", 32);
            config.epochs = ParseIntParam(node.parameters, "epochs", 8);
            config.shuffle = ParseBoolParam(node.parameters, "shuffle", true);
            config.drop_last = ParseBoolParam(node.parameters, "drop_last", false);
            config.num_workers = ParseIntParam(node.parameters, "num_workers", 4);
            config.prefetch_factor =
                ParseIntParam(node.parameters, "prefetch_factor", 2);
            config.log_interval = ParseIntParam(node.parameters, "log_interval", 10);
            config.validation_freq =
                ParseIntParam(node.parameters, "validation_freq", 1);
            config.dataloader_seed = ParseIntParam(node.parameters, "seed", 42);
            config.grad_accum_steps =
                ParseIntParam(node.parameters, "grad_accum_steps", 1);
            config.save_best_checkpoint =
                ParseBoolParam(node.parameters, "save_best_checkpoint", true);
            config.early_stopping_patience =
                ParseIntParam(node.parameters, "early_stopping_patience", 5);
        } else if (node.type == gui::NodeType::NERSequenceBuilder) {
            if (node.parameters.count("token_column") > 0) {
                config.sequence_batch.token_column = node.parameters.at("token_column");
            }
            if (node.parameters.count("pos_column") > 0) {
                config.sequence_batch.pos_column = node.parameters.at("pos_column");
            }
            if (node.parameters.count("tag_column") > 0) {
                config.sequence_batch.tag_column = node.parameters.at("tag_column");
            }
            if (node.parameters.count("sentence_id_column") > 0) {
                config.sequence_batch.sentence_id_column =
                    node.parameters.at("sentence_id_column");
            }
            config.sequence_batch.create_attention_mask =
                ParseBoolParam(node.parameters, "create_attention_mask",
                               config.sequence_batch.create_attention_mask);
            config.sequence_batch.ignore_index =
                ParseIntParam(node.parameters, "ignore_index",
                              config.sequence_batch.ignore_index);
            config.sequence_batch.target_ignore_index =
                ParseIntParam(node.parameters, "target_ignore_index",
                              config.sequence_batch.target_ignore_index);
            config.sequence_batch.max_sequence_length =
                ParseIntParam(node.parameters, "max_sequence_length",
                              config.sequence_batch.max_sequence_length);
        } else if (node.type == gui::NodeType::TextPadding) {
            config.sequence_batch.create_attention_mask =
                ParseBoolParam(node.parameters, "create_attention_mask",
                               config.sequence_batch.create_attention_mask);
            config.sequence_batch.max_sequence_length =
                ParseIntParam(node.parameters, "max_length",
                              config.sequence_batch.max_sequence_length);
        } else if (node.type == gui::NodeType::CrossEntropyLoss) {
            config.sequence_batch.ignore_index =
                ParseIntParam(node.parameters, "ignore_index",
                              config.sequence_batch.ignore_index);
            config.sequence_batch.target_ignore_index =
                ParseIntParam(node.parameters, "target_ignore_index",
                              config.sequence_batch.target_ignore_index);
        } else if (node.type == gui::NodeType::Adam) {
            config.optimizer_node_id = node.id;
            config.learning_rate = ParseFloatParam(
                node.parameters, "learning_rate",
                ParseFloatParam(node.parameters, "lr", config.learning_rate));
        } else {
            AddLayerIfSequenceNode(node, config);
        }
    }

    return config;
}

std::shared_ptr<arrow::Array> FinishStringArray(
    const std::vector<std::string>& values) {
    arrow::StringBuilder builder;
    for (const auto& value : values) {
        auto st = builder.Append(value);
        Check(st.ok(), st.ToString());
    }

    std::shared_ptr<arrow::Array> array;
    auto st = builder.Finish(&array);
    Check(st.ok(), st.ToString());
    return array;
}

std::shared_ptr<arrow::Table> MakeTextTable() {
    auto text = FinishStringArray({
        "Small happy text sample",
        "Another happy sample",
        "Sad text pipeline example",
        "Another sad example",
    });
    auto label = FinishStringArray({"positive", "positive", "negative", "negative"});

    auto schema = arrow::schema({
        arrow::field("text", arrow::utf8()),
        arrow::field("label", arrow::utf8()),
    });
    return arrow::Table::Make(schema, {text, label}, 4);
}

std::shared_ptr<arrow::Table> MakeSequenceTable() {
    auto tokens = FinishStringArray({
        "John lives in Berlin",
        "Mary works in Paris",
    });
    auto ner_tags = FinishStringArray({
        "B-PER O O B-LOC",
        "B-PER O O B-LOC",
    });

    auto schema = arrow::schema({
        arrow::field("tokens", arrow::utf8()),
        arrow::field("ner_tags", arrow::utf8()),
    });
    return arrow::Table::Make(schema, {tokens, ner_tags}, 2);
}

std::vector<std::string> SplitSimpleCsvRow(const std::string& line) {
    std::vector<std::string> values;
    std::string value;
    bool in_quotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                value.push_back('"');
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
            continue;
        }
        if (ch == ',' && !in_quotes) {
            values.push_back(value);
            value.clear();
            continue;
        }
        value.push_back(ch);
    }

    values.push_back(value);
    return values;
}

std::shared_ptr<arrow::Table> LoadNerSentencesCsvTable(
    const std::filesystem::path& csv_path) {
    std::ifstream in(csv_path);
    Check(in.is_open(), "NER sentence CSV should open: " + csv_path.string());

    std::string line;
    Check(static_cast<bool>(std::getline(in, line)),
          "NER sentence CSV should include header");
    const auto header = SplitSimpleCsvRow(line);

    auto find_index = [&](const std::string& name) {
        for (int i = 0; i < static_cast<int>(header.size()); ++i) {
            if (header[i] == name) {
                return i;
            }
        }
        return -1;
    };

    const int token_idx = find_index("tokens");
    const int pos_idx = find_index("pos_tags");
    const int tag_idx = find_index("ner_tags");
    const int sentence_idx = find_index("sentence_id");

    Check(token_idx >= 0, "NER sentence CSV must include tokens column");
    Check(pos_idx >= 0, "NER sentence CSV must include pos_tags column");
    Check(tag_idx >= 0, "NER sentence CSV must include ner_tags column");

    std::vector<std::string> sentence_values;
    std::vector<std::string> token_values;
    std::vector<std::string> pos_values;
    std::vector<std::string> tag_values;
    const auto max_index = std::max(
        {token_idx, pos_idx, tag_idx, sentence_idx >= 0 ? sentence_idx : 0});
    const bool has_sentence_id = sentence_idx >= 0;

    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto values = SplitSimpleCsvRow(line);
        Check(static_cast<int>(values.size()) > max_index,
              "NER sentence CSV row should include required fields");
        if (has_sentence_id) {
            sentence_values.push_back(values[sentence_idx]);
        } else {
            sentence_values.push_back(std::to_string(sentence_values.size()));
        }
        token_values.push_back(values[token_idx]);
        pos_values.push_back(values[pos_idx]);
        tag_values.push_back(values[tag_idx]);
    }

    auto schema = arrow::schema({
        arrow::field("sentence_id", arrow::utf8()),
        arrow::field("tokens", arrow::utf8()),
        arrow::field("pos_tags", arrow::utf8()),
        arrow::field("ner_tags", arrow::utf8()),
    });
    return arrow::Table::Make(schema,
                              {FinishStringArray(sentence_values),
                               FinishStringArray(token_values),
                               FinishStringArray(pos_values),
                               FinishStringArray(tag_values)},
                              static_cast<int64_t>(token_values.size()));
}

gui::MLNode MakeDataInputNode(
    const std::string& dataset_name = kDatasetName) {
    gui::MLNode node;
    node.id = 1;
    node.type = gui::NodeType::DataInput;
    node.category = gui::NodeCategory::DataPipeline;
    node.name = "Data Input";
    node.parameters = {
        {"dataset_name", dataset_name},
        {"data_loaded", "true"},
        {"file_category", "text"},
        {"label_column", "label"},
    };
    return node;
}

gui::MLNode MakeUnusedDataInputNode() {
    gui::MLNode node;
    node.id = 99;
    node.type = gui::NodeType::DataInput;
    node.category = gui::NodeCategory::DataPipeline;
    node.name = "Unused Data Input";
    node.parameters = {
        {"dataset_name", kUnusedDatasetName},
        {"data_loaded", "true"},
        {"file_category", "text"},
        {"label_column", "wrong_label"},
    };
    return node;
}

gui::MLNode MakeTokenizerNode() {
    gui::MLNode node;
    node.id = 2;
    node.type = gui::NodeType::TextTokenizer;
    node.category = gui::NodeCategory::TextProcessing;
    node.name = "Text Tokenizer";
    node.parameters = {
        {"text_col", "text"},
        {"label_col", "label"},
        {"tokenizer_type", "1"},
        {"max_length", "4"},
        {"lowercase", "true"},
        {"min_word_freq", "1"},
        {"max_vocab_size", "100"},
    };
    return node;
}

gui::MLNode MakeOptimizerNode(
    int id,
    const std::string& name,
    const std::string& epochs,
    const std::string& batch_size) {

    gui::MLNode node;
    node.id = id;
    node.type = gui::NodeType::Adam;
    node.category = gui::NodeCategory::Training;
    node.name = name;
    node.parameters = {
        {"epochs", epochs},
        {"batch_size", batch_size},
    };
    return node;
}

cyxwiz::TrainingConfiguration MakeTrainingConfig(
    const std::filesystem::path& checkpoint_dir) {
    cyxwiz::TrainingConfiguration config;
    config.is_valid = true;
    config.dataset_name = kDatasetName;
    config.input_size = 4;
    config.input_shape = {4};
    config.output_size = 2;
    config.preprocessing_domain = cyxwiz::PreprocessingDomain::Text;
    config.loss_type = gui::NodeType::CrossEntropyLoss;
    config.optimizer_type = gui::NodeType::Adam;
    config.learning_rate = 0.001f;
    config.train_ratio = 0.75f;
    config.shuffle = false;
    config.epochs = 1;
    config.batch_size = 2;
    config.data_source_node_id = 1;
    config.optimizer_node_id = 4;
    config.num_workers = 0;
    config.save_best_checkpoint = false;
    config.early_stopping_patience = 0;
    config.checkpoint_dir = checkpoint_dir.string();

    cyxwiz::CompiledLayer dense;
    dense.type = gui::NodeType::Dense;
    dense.units = 2;
    config.layers.push_back(dense);
    return config;
}

void CheckUnsupportedMaterializerSource(
    const cyxwiz::MaterializeResult& result,
    const std::string& dataset_name,
    cyxwiz::PipelineMaterializerSourceKind source_kind,
    cyxwiz::PipelineStorageBackend backend,
    const std::string& label) {

    Check(result.success, result.error_message);
    Check(result.effective_dataset_name == dataset_name,
          label + " source should pass through unchanged");
    Check(result.operators_applied == 0,
          label + " source should not apply Arrow operators");
    Check(result.source_kind == source_kind,
          label + " source should report its source kind");
    Check(result.skipped_unsupported_source,
          label + " source should report unsupported-source skip");
    const auto backend_support =
        cyxwiz::ResolvePipelineMaterializerStorageBackendSupport(backend);
    Check(backend_support.reason != nullptr,
          label + " materializer backend reason should be registered");
    Check(result.unsupported_source_reason == backend_support.reason,
          label + " source should expose central materializer skip reason");
    Check(result.diagnostic_message.find(
              cyxwiz::PipelineMaterializerSourceKindName(source_kind)) !=
              std::string::npos,
          label + " source should expose source kind in materializer diagnostic");
    Check(result.diagnostic_message.find(result.unsupported_source_reason) !=
              std::string::npos,
          label + " source should expose central reason in materializer diagnostic");
}

void TestTrainingRunComparisonRecord() {
    cyxwiz::TrainingConfiguration config;
    config.dataset_name = "sentiment_v1";
    config.epochs = 8;
    config.batch_size = 32;
    config.learning_rate = 0.0002f;
    config.train_ratio = 0.70f;
    config.val_ratio = 0.15f;
    config.test_ratio = 0.15f;
    config.preprocessing_domain = cyxwiz::PreprocessingDomain::Text;
    config.sequence_batch.enabled = true;
    config.save_best_checkpoint = true;
    config.early_stopping_patience = 3;
    config.checkpoint_dir = "runs/sentiment";
    config.dataloader_seed = 123;
    config.split_seed = 17;
    config.shuffle = false;
    config.dataset_roles.train.dataset_name = "sentiment_v1";
    config.dataset_roles.train.label_column = "label";
    config.dataset_roles.test.dataset_name = "sentiment_test_v1";
    config.dataset_roles.test.label_column = "label";
    config.dataset_roles.test.externally_supplied = true;
    config.dataset_roles.policy.method =
        cyxwiz::PartitionSplitMethod::Random;
    config.dataset_roles.policy.shuffle = false;
    config.dataset_roles.manifest.training_source_fingerprint = "train-content-v1";
    config.dataset_roles.manifest.validation_source_fingerprint = "train-content-v1";
    config.dataset_roles.manifest.test_source_fingerprint = "test-content-v1";
    config.dataset_roles.manifest.feature_schema_fingerprint = "features-v1";
    config.dataset_roles.manifest.dev_compatibility =
        cyxwiz::PartitionCompatibility::Compatible;
    config.dataset_roles.manifest.test_compatibility =
        cyxwiz::PartitionCompatibility::Compatible;
    config.dataset_roles.manifest.dev_leakage =
        cyxwiz::PartitionLeakageStatus::Passed;
    config.dataset_roles.manifest.test_leakage =
        cyxwiz::PartitionLeakageStatus::Unavailable;
    config.dataset_roles.manifest.test_status_reason =
        "no shared stable identifier and exact-row scan limit exceeded";
    cyxwiz::CompiledLayer gru;
    gru.type = gui::NodeType::GRU;
    gru.parameters["hidden_size"] = "96";
    gru.parameters["num_layers"] = "2";
    gru.parameters["bidirectional"] = "true";
    config.layers.push_back(gru);

    cyxwiz::TrainingMetrics metrics;
    metrics.train_loss = 0.25f;
    metrics.train_accuracy = 0.91f;
    metrics.train_sample_count = 700;
    metrics.val_sample_count = 150;
    metrics.test_sample_count = 150;
    metrics.val_loss_history = {0.9f, 0.7f, 0.8f};
    metrics.val_accuracy_history = {0.65f, 0.72f, 0.70f};
    metrics.has_validation_metrics = true;
    metrics.test_loss = 0.68f;
    metrics.test_accuracy = 0.71f;
    metrics.has_test_metrics = true;
    metrics.checkpoint_used = "runs/sentiment/best_checkpoint.cyxckpt";

    const auto record = cyxwiz::MakeTrainingRunComparisonRecord(
        "run-001", config, metrics, 12.5f,
        metrics.checkpoint_used,
        "complete");

    Check(record.run_id == "run-001", "run comparison should keep run id");
    Check(record.run_status == "complete",
          "run comparison should keep run status");
    Check(record.dataset_name == "sentiment_v1",
          "run comparison should keep dataset");
    Check(record.preprocessing_domain == "text",
          "run comparison should keep preprocessing domain");
    Check(record.sequence_batch_enabled,
          "run comparison should preserve sequence batch mode");
    Check(record.primary_layer_type == "GRU",
          "run comparison should keep primary layer type");
    Check(record.architecture_summary == "GRU",
          "run comparison should summarize architecture");
    Check(record.model_layer_count == 1,
          "run comparison should count model layers");
    Check(record.model_family == "GRU",
          "run comparison should detect GRU family");
    Check(record.bidirectional,
          "run comparison should preserve bidirectional flag");
    Check(record.hidden_size == 96,
          "run comparison should preserve hidden size");
    Check(record.num_layers == 2,
          "run comparison should preserve recurrent layer count");
    Check(record.train_ratio == 0.70f,
          "run comparison should preserve train split ratio");
    Check(record.val_ratio == 0.15f,
          "run comparison should preserve validation split ratio");
    Check(record.test_ratio == 0.15f,
          "run comparison should preserve test split ratio");
    Check(record.train_sample_count == 700,
          "run comparison should preserve train sample count");
    Check(record.val_sample_count == 150,
          "run comparison should preserve validation sample count");
    Check(record.test_sample_count == 150,
          "run comparison should preserve test sample count");
    Check(record.train_source_name == "sentiment_v1",
          "run comparison should record training source name");
    Check(record.dev_source_name == "sentiment_v1",
          "run comparison should record derived dev source name");
    Check(record.test_source_name == "sentiment_test_v1",
          "run comparison should record external test source name");
    Check(record.train_origin == "external",
          "run comparison should mark training source external");
    Check(record.dev_origin == "derived",
          "run comparison should mark absent dev role derived");
    Check(record.test_origin == "external",
          "run comparison should mark supplied test role external");
    Check(record.train_label_column == "label" &&
              record.dev_label_column == "label" &&
              record.test_label_column == "label",
          "run comparison should record role label columns");
    Check(record.partition_manifest_fingerprint.size() == 16,
          "run comparison should include stable partition fingerprint");
    Check(record.dev_schema_compatibility == "compatible" &&
              record.test_schema_compatibility == "compatible" &&
              record.dev_leakage_status == "passed" &&
              record.test_leakage_status == "unavailable" &&
              !record.test_partition_status_reason.empty(),
          "run comparison should preserve structured role validation status");
    const auto repeated_record = cyxwiz::MakeTrainingRunComparisonRecord(
        "run-001", config, metrics, 12.5f,
        metrics.checkpoint_used,
        "complete");
    Check(repeated_record.partition_manifest_fingerprint ==
              record.partition_manifest_fingerprint,
          "same partition manifest inputs should produce same fingerprint");
    auto changed_loader_seed_config = config;
    changed_loader_seed_config.dataloader_seed = 999;
    const auto changed_loader_seed_record =
        cyxwiz::MakeTrainingRunComparisonRecord(
            "run-001", changed_loader_seed_config, metrics, 12.5f,
            metrics.checkpoint_used, "complete");
    Check(changed_loader_seed_record.partition_manifest_fingerprint ==
              record.partition_manifest_fingerprint,
          "Data Loader seed must not change the partition manifest");
    auto changed_split_seed_config = config;
    changed_split_seed_config.split_seed = 18;
    const auto changed_split_seed_record =
        cyxwiz::MakeTrainingRunComparisonRecord(
            "run-001", changed_split_seed_config, metrics, 12.5f,
            metrics.checkpoint_used, "complete");
    Check(changed_split_seed_record.partition_manifest_fingerprint !=
              record.partition_manifest_fingerprint,
          "Data Split seed must change the partition manifest");
    auto changed_source_config = config;
    changed_source_config.dataset_roles.manifest.test_source_fingerprint =
        "test-content-v2";
    const auto changed_source_record =
        cyxwiz::MakeTrainingRunComparisonRecord(
            "run-001", changed_source_config, metrics, 12.5f,
            metrics.checkpoint_used, "complete");
    Check(changed_source_record.partition_manifest_fingerprint !=
              record.partition_manifest_fingerprint,
          "source content identity must change the partition manifest");
    auto changed_schema_config = config;
    changed_schema_config.dataset_roles.manifest.feature_schema_fingerprint =
        "features-v2";
    const auto changed_schema_record =
        cyxwiz::MakeTrainingRunComparisonRecord(
            "run-001", changed_schema_config, metrics, 12.5f,
            metrics.checkpoint_used, "complete");
    Check(changed_schema_record.partition_manifest_fingerprint !=
              record.partition_manifest_fingerprint,
          "feature schema identity must change the partition manifest");
    auto changed_method_config = config;
    changed_method_config.dataset_roles.policy.method =
        cyxwiz::PartitionSplitMethod::TimeOrdered;
    const auto changed_method_record =
        cyxwiz::MakeTrainingRunComparisonRecord(
            "run-001", changed_method_config, metrics, 12.5f,
            metrics.checkpoint_used, "complete");
    Check(changed_method_record.partition_manifest_fingerprint !=
              record.partition_manifest_fingerprint,
          "split method must change the partition manifest");
    auto changed_stratified_config = config;
    changed_stratified_config.stratified = true;
    changed_stratified_config.dataset_roles.policy.stratified = true;
    changed_stratified_config.dataset_roles.policy.method =
        cyxwiz::PartitionSplitMethod::Stratified;
    const auto changed_stratified_record =
        cyxwiz::MakeTrainingRunComparisonRecord(
            "run-001", changed_stratified_config, metrics, 12.5f,
            metrics.checkpoint_used, "complete");
    Check(changed_stratified_record.partition_manifest_fingerprint !=
              record.partition_manifest_fingerprint,
          "stratification must change the partition manifest");
    auto changed_metrics = metrics;
    changed_metrics.test_sample_count = 151;
    const auto changed_record = cyxwiz::MakeTrainingRunComparisonRecord(
        "run-001", config, changed_metrics, 12.5f,
        metrics.checkpoint_used,
        "complete");
    Check(changed_record.partition_manifest_fingerprint !=
              record.partition_manifest_fingerprint,
          "resolved row-count changes should alter partition fingerprint");
    Check(cyxwiz::CompareTrainingRunPartitions(record, repeated_record) ==
              cyxwiz::TrainingRunPartitionCompatibility::SameManifest,
          "identical partition manifests should be directly comparable");
    Check(cyxwiz::CompareTrainingRunPartitions(record, changed_record) ==
              cyxwiz::TrainingRunPartitionCompatibility::DifferentManifest,
          "different partition manifests should not be directly comparable");
    auto unknown_partition_record = record;
    unknown_partition_record.partition_manifest_fingerprint.clear();
    Check(cyxwiz::CompareTrainingRunPartitions(
              record, unknown_partition_record) ==
              cyxwiz::TrainingRunPartitionCompatibility::Unknown,
          "missing partition provenance should produce unknown compatibility");
    Check(std::string(cyxwiz::TrainingRunPartitionCompatibilityLabel(
              cyxwiz::TrainingRunPartitionCompatibility::DifferentManifest)) ==
              "different",
          "comparison UI should label different partition manifests explicitly");
    Check(record.best_val_loss == 0.7f,
          "run comparison should compute best validation loss");
    Check(record.best_val_accuracy == 0.72f,
          "run comparison should compute best validation accuracy");
    Check(record.best_val_loss_epoch == 2,
          "run comparison should report best validation loss epoch");
    Check(record.best_val_accuracy_epoch == 2,
          "run comparison should report best validation accuracy epoch");
    Check(record.has_validation_metrics,
          "run comparison should mark validation metrics present");
    Check(record.has_test_metrics,
          "run comparison should mark test metrics present");
    Check(record.checkpoint_used == "runs/sentiment/best_checkpoint.cyxckpt",
          "run comparison should keep checkpoint used");
    Check(record.final_test_accuracy == 0.71f,
          "run comparison should keep final test accuracy");

    const std::string csv = cyxwiz::TrainingRunComparisonTableSummary({record});
    Check(csv.find("run_id,run_status,dataset_name,preprocessing_domain,"
                   "sequence_batch_enabled") == 0,
          "run comparison CSV should include stable header");
    Check(csv.find("partition_manifest_fingerprint") != std::string::npos,
          "run comparison CSV should include partition manifest column");
    Check(csv.find("test_leakage_status") != std::string::npos &&
              csv.find("exact-row scan limit exceeded") != std::string::npos,
          "run comparison CSV should include structured role-check disclosure");
    Check(csv.find("run-001,complete,sentiment_v1,text,true,GRU") !=
              std::string::npos,
          "run comparison CSV should include record row");
    Check(csv.find("sentiment_v1,sentiment_v1,sentiment_test_v1") !=
              std::string::npos,
          "run comparison CSV should include role source names");
    Check(csv.find("external,derived,external,label,label,label") !=
              std::string::npos,
          "run comparison CSV should include role origins and labels");
    const auto output_path =
        std::filesystem::temp_directory_path() /
        "cyxwiz_training_run_comparison" /
        "runs.csv";
    std::string error;
    Check(cyxwiz::WriteTrainingRunComparisonCsv(output_path, {record}, &error),
          "run comparison CSV export should succeed: " + error);
    Check(std::filesystem::exists(output_path),
          "run comparison CSV export should create output file");

    auto weaker = record;
    weaker.run_id = "run-002";
    weaker.final_test_accuracy = 0.60f;
    auto sorted = cyxwiz::SortTrainingRunComparisonsByBestMetric({weaker, record});
    Check(sorted.size() == 2,
          "run comparison sort should keep all records");
    Check(sorted.front().run_id == "run-001",
          "run comparison sort should prefer higher test accuracy");

    auto tie_b = record;
    tie_b.run_id = "run-tie-b";
    auto tie_a = record;
    tie_a.run_id = "run-tie-a";
    auto tied_sorted =
        cyxwiz::SortTrainingRunComparisonsByBestMetric({tie_b, tie_a});
    Check(tied_sorted.front().run_id == "run-tie-a",
          "run comparison sort should use run id as deterministic final tie-breaker");

    config.checkpoint_dir.clear();
    const auto default_checkpoint_record =
        cyxwiz::MakeTrainingRunComparisonRecord(
            "run-default-checkpoint", config, metrics, 1.0f);
    Check(default_checkpoint_record.checkpoint_used ==
              "default .cyxwiz/checkpoints run folder",
          "run comparison should make default checkpoint root explicit");

    config.save_best_checkpoint = false;
    const auto final_state_record =
        cyxwiz::MakeTrainingRunComparisonRecord(
            "run-final-state", config, metrics, 1.0f);
    Check(final_state_record.checkpoint_used == "final epoch model state",
          "run comparison should not imply checkpoint use when checkpointing is disabled");

    cyxwiz::TrainingMetrics zero_metrics;
    zero_metrics.has_validation_metrics = true;
    zero_metrics.has_test_metrics = true;
    const auto zero_record = cyxwiz::MakeTrainingRunComparisonRecord(
        "run-zero", config, zero_metrics, 1.0f);
    Check(zero_record.has_validation_metrics,
          "run comparison should not infer validation availability from nonzero values");
    Check(zero_record.has_test_metrics,
          "run comparison should not infer test availability from nonzero values");
}

} // namespace

namespace cyxwiz {

PipelineOperatorFactory& PipelineOperatorFactory::Instance() {
    static PipelineOperatorFactory instance;
    return instance;
}

PipelineOperatorFactory::PipelineOperatorFactory() = default;

std::unique_ptr<IPipelineOperator> PipelineOperatorFactory::Create(
    gui::NodeType type) const {
    if (type == gui::NodeType::TextTokenizer) {
        return std::make_unique<TextTokenizerOperator>();
    }
    return nullptr;
}

bool PipelineOperatorFactory::HasOperator(gui::NodeType type) const {
    return type == gui::NodeType::TextTokenizer;
}

void PipelineOperatorFactory::RegisterCreator(gui::NodeType, Creator) {}

std::vector<gui::NodeType> PipelineOperatorFactory::GetSupportedTypes() const {
    return {gui::NodeType::TextTokenizer};
}

} // namespace cyxwiz

int main() {
    TestTrainingRunComparisonRecord();

    const auto repo_root = FindRepoRoot();
    const auto work_dir =
        std::filesystem::temp_directory_path() /
        "cyxwiz_text_gui_training_launch";
    std::filesystem::create_directories(work_dir);
    std::filesystem::current_path(work_dir);

    auto& registry = cyxwiz::DataRegistry::Instance();
    registry.UnregisterTabularDataset(kDatasetName);
    registry.UnregisterTabularDataset(kMaterializedDatasetName);
    registry.UnregisterTabularDataset(kUnusedDatasetName);
    Check(registry.RegisterArrowTable(MakeTextTable(), kDatasetName) != nullptr,
          "raw text Arrow dataset should register");
    Check(registry.RegisterArrowTable(MakeTextTable(), kUnusedDatasetName) != nullptr,
          "unused raw text Arrow dataset should register");

    std::vector<gui::MLNode> nodes = {
        MakeUnusedDataInputNode(),
        MakeOptimizerNode(3, "Stale Adam", "99", "99"),
        MakeDataInputNode(),
        MakeTokenizerNode(),
        MakeOptimizerNode(4, "Selected Adam", "", ""),
    };
    std::vector<gui::NodeLink> links = {
        {1, 1, 0, 2, 0, gui::LinkType::TensorFlow},
    };

    registry.UnregisterTabularDataset(kScopeArrowDatasetName);
    registry.UnregisterTabularDataset(
        std::string(kScopeArrowDatasetName) +
        cyxwiz::PipelineMaterializer::kMaterializedSuffix);
    registry.UnregisterTabularDataset(kScopeParquetDatasetName);
    registry.UnregisterImageDataset(kScopeImageDatasetName);
    registry.UnregisterAudioDataset(kScopeAudioDatasetName);
    Check(registry.RegisterArrowTable(MakeTextTable(), kScopeArrowDatasetName) != nullptr,
          "Arrow source should register for materializer scope test");
    std::vector<gui::MLNode> scope_nodes = {
        MakeDataInputNode(kScopeArrowDatasetName),
        MakeTokenizerNode(),
    };
    auto arrow_scope = cyxwiz::PipelineMaterializer::Materialize(
        scope_nodes, links, registry, kScopeArrowDatasetName);
    Check(arrow_scope.success, arrow_scope.error_message);
    Check(arrow_scope.source_kind ==
              cyxwiz::PipelineMaterializerSourceKind::ArrowTable,
          "Arrow source should report ArrowTable source kind");
    Check(!arrow_scope.skipped_unsupported_source,
          "Arrow source should not report unsupported-source skip");
    Check(arrow_scope.unsupported_source_reason.empty(),
          "Arrow source should not report unsupported-source reason");
    Check(arrow_scope.operators_applied == 1,
          "Arrow source should apply tokenizer through registry materializer");
    registry.UnregisterTabularDataset(kScopeArrowDatasetName);
    registry.UnregisterTabularDataset(
        std::string(kScopeArrowDatasetName) +
        cyxwiz::PipelineMaterializer::kMaterializedSuffix);

    const auto parquet_path = work_dir / "materializer_scope.parquet";
    std::remove(parquet_path.string().c_str());
    cyxwiz::ArrowDataset parquet_fixture(
        MakeTextTable(), kScopeParquetDatasetName);
    Check(parquet_fixture.ExportParquet(parquet_path.string()),
          "materializer scope Parquet fixture should export");
    auto parquet_dataset = cyxwiz::ParquetBackedDataset::Open(
        parquet_path.string(), kScopeParquetDatasetName);
    Check(parquet_dataset != nullptr,
          "materializer scope Parquet fixture should open");
    registry.RegisterParquetBacked(kScopeParquetDatasetName, parquet_dataset);
    auto parquet_scope = cyxwiz::PipelineMaterializer::Materialize(
        scope_nodes, links, registry, kScopeParquetDatasetName);
    CheckUnsupportedMaterializerSource(
        parquet_scope,
        kScopeParquetDatasetName,
        cyxwiz::PipelineMaterializerSourceKind::ParquetBacked,
        cyxwiz::PipelineStorageBackend::ParquetBacked,
        "Parquet-backed");
    registry.UnregisterTabularDataset(kScopeParquetDatasetName);
    parquet_dataset.reset();
    std::remove(parquet_path.string().c_str());

    cyxwiz::DataRegistry::ImageDatasetEntry image_entry;
    image_entry.folder_path = "images";
    image_entry.num_images = 4;
    image_entry.num_classes = 2;
    registry.RegisterImageDataset(kScopeImageDatasetName, image_entry);
    auto image_scope = cyxwiz::PipelineMaterializer::Materialize(
        scope_nodes, links, registry, kScopeImageDatasetName);
    CheckUnsupportedMaterializerSource(
        image_scope,
        kScopeImageDatasetName,
        cyxwiz::PipelineMaterializerSourceKind::ImageDataset,
        cyxwiz::PipelineStorageBackend::ImageDataset,
        "Image");
    registry.UnregisterImageDataset(kScopeImageDatasetName);

    cyxwiz::DataRegistry::AudioDatasetEntry audio_entry;
    audio_entry.folder_path = "audio";
    audio_entry.num_samples = 4;
    audio_entry.num_classes = 2;
    registry.RegisterAudioDataset(kScopeAudioDatasetName, audio_entry);
    auto audio_scope = cyxwiz::PipelineMaterializer::Materialize(
        scope_nodes, links, registry, kScopeAudioDatasetName);
    CheckUnsupportedMaterializerSource(
        audio_scope,
        kScopeAudioDatasetName,
        cyxwiz::PipelineMaterializerSourceKind::AudioDataset,
        cyxwiz::PipelineStorageBackend::AudioDataset,
        "Audio");
    registry.UnregisterAudioDataset(kScopeAudioDatasetName);

    cyxwiz::DataRegistry::TextDatasetEntry text_entry;
    text_entry.source_path = "legacy_text.csv";
    text_entry.text_column = "text";
    text_entry.label_column = "label";
    text_entry.num_samples = 3;
    registry.RegisterTextDataset(kScopeTextDatasetName, text_entry);
    auto text_scope = cyxwiz::PipelineMaterializer::Materialize(
        scope_nodes, links, registry, kScopeTextDatasetName);
    CheckUnsupportedMaterializerSource(
        text_scope,
        kScopeTextDatasetName,
        cyxwiz::PipelineMaterializerSourceKind::TextDataset,
        cyxwiz::PipelineStorageBackend::TextDataset,
        "Legacy text");
    registry.UnregisterTextDataset(kScopeTextDatasetName);

    std::atomic<bool> dispatch_called{false};
    std::atomic<bool> callback_started{false};
    std::atomic<bool> callback_finished{false};

    auto config = MakeTrainingConfig(work_dir / "checkpoints");
    auto dispatch = [&](cyxwiz::TrainingConfiguration dispatch_config,
                        const std::string& dataset_name,
                        const std::string& label_column,
                        int epochs,
                        int batch_size,
                        std::weak_ptr<cyxwiz::TrainingPlotPanel>,
                        std::function<void(bool)> callback) {
        dispatch_called.store(true);
        Check(dataset_name == kMaterializedDatasetName,
              "dispatch should receive materialized dataset");
        Check(dispatch_config.dataset_name == kMaterializedDatasetName,
              "config dataset name should match materialized dataset");
        Check(label_column == "y", "dispatch should receive runtime y label");
        Check(epochs == 1, "epochs should come from compiled config");
        Check(batch_size == 2, "batch size should come from compiled config");
        Check(!dispatch_config.save_best_checkpoint,
              "save_best_checkpoint should come from compiled config");
        Check(dispatch_config.early_stopping_patience == 0,
              "early stopping patience should come from compiled config");
        Check(dispatch_config.checkpoint_dir ==
                  (work_dir / "checkpoints").string(),
              "checkpoint directory should come from compiled config");

        if (callback) {
            callback(true);
            callback_started.store(true);
        }

        auto dataset = registry.GetArrowDataset(dataset_name);
        Check(dataset != nullptr, "materialized Arrow dataset should exist");
        auto table = dataset->GetArrowTable();
        Check(table != nullptr, "materialized table should exist");
        Check(table->GetColumnByName("tok_0") != nullptr,
              "materialized table should expose token columns");
        Check(table->GetColumnByName("y") != nullptr,
              "materialized table should expose y label");

        cyxwiz::ArrowDatasetBatcher batcher(
            dataset,
            label_column,
            batch_size,
            /*shuffle=*/false,
            dispatch_config.train_ratio,
            /*is_training=*/true);
        batcher.SetOneHotEncoding(2);

        auto batch = batcher.GetNextBatch();
        Check(batch.IsValid(), "GUI launch batch should be valid");
        Check(batch.data.Shape().size() == 2, "GUI launch features should be 2D");
        Check(batch.data.Shape()[1] == 4,
              "GUI launch feature width should equal tokenizer max_length");

        auto built = cyxwiz::BuildSequentialFromConfig(dispatch_config);
        Check(built.ok(), "GUI launch should build model/loss/optimizer");
        auto predictions = built.model->Forward(batch.data);
        auto loss = built.loss->Forward(predictions, batch.labels);
        Check(loss.NumElements() == 1, "GUI launch loss should be scalar");
        CheckFinite(loss.Data<float>()[0], "GUI launch loss should be finite");

        auto grad = built.loss->Backward(predictions, batch.labels);
        built.model->Backward(grad);
        built.model->UpdateParameters(built.optimizer.get());

        if (callback) {
            callback(false);
            callback_finished.store(true);
        }
        return true;
    };

    auto result = gui::StartGraphTrainingFromCompiledConfig(
        nodes,
        links,
        std::move(config),
        registry,
        std::weak_ptr<cyxwiz::TrainingPlotPanel>{},
        [](bool) {},
        dispatch);

    Check(result.started, result.error_message);
    Check(WaitFor([&] { return callback_finished.load(); },
                  std::chrono::seconds(20)),
          "dispatch should run through the training finish callback");
    Check(dispatch_called.load(), "dispatch should be called");
    Check(callback_started.load(), "training start callback should fire");
    Check(result.effective_dataset_name == kDatasetName,
          "queued result should report source dataset");
    Check(result.label_column == "label",
          "queued result should report source label column");
    Check(result.operators_applied == 0,
          "queued result should not report async materializer operators");
    Check(result.materializer_source_kind ==
              cyxwiz::PipelineMaterializerSourceKind::Unknown,
          "queued result should not report async materializer source kind");
    Check(!result.materializer_skipped_unsupported_source,
          "queued result should not report async unsupported-source status");
    Check(result.materialization_cache_enabled,
          "queued result should expose enabled materialization cache");
    Check(result.materialization_cache_mode == cyxwiz::MaterializationCacheMode::Auto,
          "queued result should expose automatic materialization cache mode");
    Check(result.epochs == 1, "result epochs should match config");
    Check(result.batch_size == 2, "result batch size should match config");

    auto sequence_config =
        MakeTrainingConfig(work_dir / "sequence_launch_checkpoints");
    const std::string sequence_dataset_name = "gui_sequence_runtime";
    Check(registry.RegisterArrowTable(MakeSequenceTable(), sequence_dataset_name) !=
              nullptr,
          "sequence runtime dataset should register");
    sequence_config.dataset_name = sequence_dataset_name;
    sequence_config.sequence_batch.enabled = true;
    sequence_config.sequence_batch.token_column = "tokens";
    sequence_config.sequence_batch.tag_column = "ner_tags";
    sequence_config.sequence_batch.create_attention_mask = true;
    sequence_config.sequence_batch.max_sequence_length = 5;

    auto sequence_dataset = std::make_shared<cyxwiz::ArrowDataset>(
        MakeSequenceTable(), sequence_dataset_name);
    auto sequence_build = cyxwiz::BuildSequenceBatcherFromArrowDataset(
        sequence_dataset, sequence_config, 2);
    Check(sequence_build.success(),
          "sequence Arrow batcher bridge should build: " +
              sequence_build.error_message);
    Check(sequence_build.sample_count == 2,
          "sequence Arrow batcher bridge should preserve sample count");
    Check(sequence_build.sequence_length == 5,
          "sequence Arrow batcher bridge should expose sequence length");
    Check(sequence_build.token_vocabulary_size > 0,
          "sequence Arrow batcher bridge should expose token vocabulary size");
    Check(sequence_build.tag_vocabulary_size ==
              sequence_build.id_to_label.size(),
          "sequence Arrow batcher bridge should expose tag vocabulary size");
    Check(!sequence_build.id_to_label.empty(),
          "sequence Arrow batcher bridge should expose tag label vocabulary");
    auto sequence_batch = sequence_build.batcher->GetNextSequenceBatch();
    Check(sequence_batch.IsSupervised(),
          "sequence Arrow batcher bridge should produce supervised sequence batches");
    Check(sequence_batch.sequence_length == 5,
          "sequence Arrow batcher bridge should honor max sequence length");
    Check(sequence_batch.HasAttentionMask(),
          "sequence Arrow batcher bridge should honor attention mask config");

    auto normalized_sequence_config = sequence_config;
    normalized_sequence_config.input_size = 0;
    normalized_sequence_config.input_shape.clear();
    normalized_sequence_config.output_size = 99;
    normalized_sequence_config.layers.clear();

    cyxwiz::CompiledLayer sequence_embedding;
    sequence_embedding.type = gui::NodeType::Embedding;
    sequence_embedding.parameters["num_embeddings"] = "1";
    sequence_embedding.parameters["embedding_dim"] = "6";
    normalized_sequence_config.layers.push_back(sequence_embedding);

    cyxwiz::CompiledLayer sequence_head;
    sequence_head.type = gui::NodeType::TimeDistributed;
    sequence_head.units = 1;
    sequence_head.parameters["units"] = "1";
    normalized_sequence_config.layers.push_back(sequence_head);

    cyxwiz::ApplySequenceBatcherBuildResultToTrainingConfig(
        sequence_build, normalized_sequence_config);
    Check(normalized_sequence_config.input_size == 5,
          "sequence config normalization should set input sequence length");
    Check(normalized_sequence_config.output_size ==
              sequence_build.tag_vocabulary_size,
          "sequence config normalization should set tag output size");
    Check(normalized_sequence_config.layers[0]
              .parameters["num_embeddings"] ==
              std::to_string(sequence_build.token_vocabulary_size),
          "sequence config normalization should set embedding vocabulary size");
    Check(normalized_sequence_config.layers[1].units ==
              static_cast<int>(sequence_build.tag_vocabulary_size),
          "sequence config normalization should set token head units");
    Check(normalized_sequence_config.layers[1]
              .parameters["units"] ==
              std::to_string(sequence_build.tag_vocabulary_size),
          "sequence config normalization should set token head units parameter");

    std::atomic<bool> sequence_dispatch_called{false};
    auto sequence_dispatch = [&](
        cyxwiz::TrainingConfiguration dispatch_config,
        const std::string&,
        const std::string&,
        int,
        int,
        std::weak_ptr<cyxwiz::TrainingPlotPanel>,
        std::function<void(bool)>) {
        sequence_dispatch_called.store(true);
        Check(dispatch_config.sequence_batch.enabled,
              "sequence launch should preserve sequence_batch.enabled");
        Check(dispatch_config.sequence_batch.token_column == "tokens",
              "sequence launch should preserve token column");
        Check(dispatch_config.sequence_batch.tag_column == "ner_tags",
              "sequence launch should preserve tag column");
        return true;
    };
    auto sequence_result = gui::StartGraphTrainingFromCompiledConfig(
        {MakeDataInputNode(sequence_dataset_name)},
        {},
        std::move(sequence_config),
        registry,
        std::weak_ptr<cyxwiz::TrainingPlotPanel>{},
        [](bool) {},
        sequence_dispatch);

    Check(sequence_result.started,
          sequence_result.error_message);
    Check(WaitFor([&] { return sequence_dispatch_called.load(); },
                  std::chrono::seconds(20)),
          "sequence batch launch should call dispatch");

    const auto ner_graph_path =
        repo_root / "examples/cyxgraph/NER/ner_bilstm_sequence_tagger.cyxgraph";
    const auto ner_csv_path =
        repo_root / "examples/cyxgraph/NER/generated/ner_sentences.csv";

    registry.UnregisterTabularDataset(kSavedNerDatasetName);
    auto [saved_ners_nodes, saved_ners_links] =
        LoadCyxGraphForTest(ner_graph_path);
    auto ner_dataset = LoadNerSentencesCsvTable(ner_csv_path);
    Check(registry.RegisterArrowTable(ner_dataset, kSavedNerDatasetName) != nullptr,
          "saved NER Arrow dataset should register");
    auto saved_ners_config =
        MakeSavedNerGraphConfig(saved_ners_nodes, kSavedNerDatasetName,
                                work_dir / "saved_ner_checkpoints");
    for (auto& node : saved_ners_nodes) {
        if (node.type == gui::NodeType::DataInput) {
            node.parameters["dataset_name"] = kSavedNerDatasetName;
            break;
        }
    }

    std::atomic<bool> saved_ners_dispatch_called{false};
    std::atomic<bool> saved_ners_start_callback{false};
    std::atomic<bool> saved_ners_finish_callback{false};
    auto saved_ners_dispatch = [&](cyxwiz::TrainingConfiguration dispatch_config,
                                  const std::string& materialized_dataset_name,
                                  const std::string& label_column,
                                  int epochs,
                                  int batch_size,
                                  std::weak_ptr<cyxwiz::TrainingPlotPanel>,
                                  std::function<void(bool)> callback) {
        saved_ners_dispatch_called.store(true);
        Check(materialized_dataset_name == kSavedNerDatasetName,
              "saved NER launch should use registered dataset name");
        Check(label_column.empty(),
              "saved NER launch should keep empty label column without "
              "materializer operator relabeling");
        Check(dispatch_config.sequence_batch.enabled,
              "saved NER launch should keep sequence enabled");
        Check(dispatch_config.sequence_batch.token_column == "tokens",
              "saved NER launch should use tokens as sequence input");
        Check(dispatch_config.sequence_batch.pos_column == "pos_tags",
              "saved NER launch should use pos_tags as POS input");
        Check(dispatch_config.sequence_batch.tag_column == "ner_tags",
              "saved NER launch should use ner_tags as target tags");
        Check(epochs == 8, "saved NER launch should keep DataLoader epochs");
        Check(batch_size == 32, "saved NER launch should keep DataLoader batch size");
        Check(dispatch_config.sequence_batch.max_sequence_length == 96,
              "saved NER launch should keep configured sequence length");

        if (callback) {
            callback(true);
            saved_ners_start_callback.store(true);
        }

        auto dataset = registry.GetArrowDataset(materialized_dataset_name);
        Check(dataset != nullptr, "saved NER dispatch should resolve Arrow dataset");
        auto sequence_build = cyxwiz::BuildSequenceBatcherFromArrowDataset(
            dataset, dispatch_config, batch_size);
        Check(sequence_build.success(),
              "saved NER launch should build sequence batcher: " +
                  sequence_build.error_message);
        Check(sequence_build.sample_count == static_cast<size_t>(ner_dataset->num_rows()),
              "saved NER launch should preserve row count");
        Check(sequence_build.id_to_label.size() == sequence_build.tag_vocabulary_size,
              "saved NER launch should expose full tag vocabulary");
        Check(sequence_build.tag_vocabulary_size > 0,
              "saved NER launch should infer a non-empty tag vocabulary");
        sequence_build.batcher->SetPhase(cyxwiz::BatcherPhase::Train);
        const auto saved_ner_train_samples = sequence_build.batcher->GetNumSamples();
        const size_t total_ner_samples =
            static_cast<size_t>(ner_dataset->num_rows());
        const size_t expected_train_samples = std::min<size_t>(
            total_ner_samples,
            static_cast<size_t>(std::floor(
                total_ner_samples * dispatch_config.train_ratio)));
        size_t expected_val_samples = std::min<size_t>(
            total_ner_samples,
            static_cast<size_t>(std::floor(
                total_ner_samples * dispatch_config.val_ratio)));
        if (expected_train_samples + expected_val_samples > total_ner_samples) {
            expected_val_samples =
                expected_train_samples >= total_ner_samples
                    ? 0
                    : total_ner_samples - expected_train_samples;
        }
        const size_t expected_test_samples =
            total_ner_samples - expected_train_samples - expected_val_samples;

        Check(saved_ner_train_samples == expected_train_samples,
              "saved NER split should allocate expected train samples");
        Check(sequence_build.batcher->GetNumBatches() == 1u,
              "saved NER launch should have one train batch with batch size 32");
        sequence_build.batcher->SetPhase(cyxwiz::BatcherPhase::Val);
        const auto saved_ner_val_samples = sequence_build.batcher->GetNumSamples();
        Check(saved_ner_val_samples == expected_val_samples,
              "saved NER split should allocate expected val samples");
        Check(sequence_build.batcher->GetNumBatches() ==
              (expected_val_samples + batch_size - 1) / batch_size ||
              expected_val_samples == 0,
              "saved NER split should report expected val batch count");
        sequence_build.batcher->SetPhase(cyxwiz::BatcherPhase::Test);
        const auto saved_ner_test_samples = sequence_build.batcher->GetNumSamples();
        Check(saved_ner_test_samples == expected_test_samples,
              "saved NER split should allocate expected test samples");
        Check(sequence_build.batcher->GetNumBatches() ==
                  (expected_test_samples + batch_size - 1) / batch_size ||
              expected_test_samples == 0,
              "saved NER split should report expected test batch count");
        Check(saved_ner_train_samples + saved_ner_val_samples +
                  saved_ner_test_samples ==
                  total_ner_samples,
              "saved NER split should partition all rows across phases");
        auto batch = sequence_build.batcher->GetNextSequenceBatch();
        Check(batch.IsSupervised(),
              "saved NER launch should build supervised sequence batches");
        Check(batch.HasAttentionMask(),
              "saved NER launch should build attention masks");
        cyxwiz::ApplySequenceBatcherBuildResultToTrainingConfig(
            sequence_build, dispatch_config);
        Check(dispatch_config.output_size ==
                  static_cast<int>(sequence_build.tag_vocabulary_size),
              "saved NER launch should align classifier width with sequence tags");

        const auto batch_shape = batch.word_ids.Shape();
        Check(batch_shape.size() == 2,
              "saved NER decode should use batched token ids");
        std::vector<float> logits(batch_shape[0] * batch_shape[1] *
                                      sequence_build.tag_vocabulary_size,
                                  -1.0f);
        const auto* gold_ids = batch.tag_ids.Data<int64_t>();
        for (size_t i = 0; i < batch.tag_ids.NumElements(); ++i) {
            const int64_t gold = gold_ids[i];
            if (gold < 0 ||
                static_cast<size_t>(gold) >= sequence_build.tag_vocabulary_size) {
                continue;
            }
            logits[i * sequence_build.tag_vocabulary_size +
                   static_cast<size_t>(gold)] = 1.0f;
        }
        const auto predictions =
            cyxwiz::Tensor({batch_shape[0], batch_shape[1],
                            sequence_build.tag_vocabulary_size},
                           logits.data(), cyxwiz::DataType::Float32);
        Check(predictions.Shape().size() == 3,
              "saved NER decode should produce sequence logits");
        Check(predictions.Shape()[0] == batch.word_ids.Shape()[0] &&
                  predictions.Shape()[1] == batch.word_ids.Shape()[1],
              "saved NER decode should preserve batch and token dimensions");
        Check(predictions.Shape()[2] == sequence_build.tag_vocabulary_size,
              "saved NER decode should produce logits for every BIO tag");
        const auto predicted_ids = cyxwiz::ArgmaxSequenceTagLogits(predictions);
        Check(predicted_ids.Shape() == batch.tag_ids.Shape(),
              "saved NER decode should return per-token predicted ids");
        const auto metrics =
            cyxwiz::ComputeSequenceTagMetricsFromLogits(
                predictions, batch.tag_ids, sequence_build.id_to_label,
                dispatch_config.sequence_batch.ignore_index);
        Check(std::isfinite(metrics.token_accuracy),
              "saved NER decode metrics should be finite");
        Check(predicted_ids.NumElements() == batch.tag_ids.NumElements(),
              "saved NER decode ids should match target count");
        Check(!sequence_build.id_to_label.empty(),
              "saved NER decode should have label vocabulary available");

        const auto* predicted_data = predicted_ids.Data<int64_t>();
        const auto* gold_data = batch.tag_ids.Data<int64_t>();
        bool observed_gold = false;
        for (size_t i = 0; i < batch.tag_ids.NumElements(); ++i) {
            const int64_t gold = gold_data[i];
            if (gold == dispatch_config.sequence_batch.ignore_index) {
                continue;
            }
            Check(predicted_data[i] >= 0 &&
                      static_cast<size_t>(predicted_data[i]) <
                          sequence_build.id_to_label.size(),
                  "saved NER decode should map all predicted ids to vocab");
            Check(gold >= 0 &&
                      static_cast<size_t>(gold) < sequence_build.id_to_label.size(),
                  "saved NER decode should map gold ids to vocab");
            Check(!sequence_build.id_to_label[static_cast<size_t>(gold)].empty(),
                  "saved NER decode should map gold ids to a defined label");
            Check(!sequence_build.id_to_label[static_cast<size_t>(predicted_data[i])].empty(),
                  "saved NER decode should map predictions to a defined label");
            observed_gold = true;
            break;
        }
        Check(observed_gold,
              "saved NER decode should contain at least one non-padding token");

        if (callback) {
            callback(false);
            saved_ners_finish_callback.store(true);
        }
        return true;
    };
    auto saved_ner_result = gui::StartGraphTrainingFromCompiledConfig(
        saved_ners_nodes,
        saved_ners_links,
        std::move(saved_ners_config),
        registry,
        std::weak_ptr<cyxwiz::TrainingPlotPanel>{},
        [](bool) {},
        saved_ners_dispatch);

    Check(saved_ner_result.started, saved_ner_result.error_message);
    Check(WaitFor([&] { return saved_ners_finish_callback.load(); },
                  std::chrono::seconds(20)),
          "saved NER launch should run through the finish callback");
    Check(saved_ners_dispatch_called.load(),
          "saved NER launch should call dispatch");
    Check(saved_ners_start_callback.load(),
          "saved NER launch should start callback");
    Check(saved_ner_result.effective_dataset_name == kSavedNerDatasetName,
          "saved NER result should use registered dataset name");
    Check(saved_ner_result.operators_applied == 0,
          "saved NER result should not apply Arrow operators in test binary");
    Check(!saved_ner_result.materializer_skipped_unsupported_source,
          "saved NER result should not skip unsupported Arrow materializer source");

    registry.UnregisterTabularDataset(kSavedNerDatasetName);

    registry.UnregisterTabularDataset(kDatasetName);
    registry.UnregisterTabularDataset(kMaterializedDatasetName);
    registry.UnregisterTabularDataset(kUnusedDatasetName);

    cyxwiz::DataRegistry::TextDatasetEntry launch_text_entry = text_entry;
    launch_text_entry.num_samples = 3;
    registry.RegisterTextDataset(kScopeTextDatasetName, launch_text_entry);

    auto legacy_text_config = MakeTrainingConfig(work_dir / "legacy_text_checkpoints");
    legacy_text_config.dataset_name = kScopeTextDatasetName;
    std::atomic<bool> legacy_text_dispatch_called{false};
    auto legacy_text_dispatch = [&](cyxwiz::TrainingConfiguration dispatch_config,
                                    const std::string& dataset_name,
                                    const std::string& label_column,
                                    int epochs,
                                    int batch_size,
                                    std::weak_ptr<cyxwiz::TrainingPlotPanel>,
                                    std::function<void(bool)> callback) {
        legacy_text_dispatch_called.store(true);
        Check(dispatch_config.dataset_name == kScopeTextDatasetName,
              "legacy text config should keep original dataset");
        Check(dataset_name == kScopeTextDatasetName,
              "legacy text dispatch should receive original dataset");
        Check(label_column == "label",
              "legacy text dispatch should keep configured label column");
        Check(epochs == 1, "legacy text epochs should match config");
        Check(batch_size == 2, "legacy text batch size should match config");
        if (callback) {
            callback(true);
        }
        return true;
    };

    std::vector<gui::MLNode> legacy_text_nodes = {
        MakeDataInputNode(kScopeTextDatasetName),
        MakeTokenizerNode(),
        MakeOptimizerNode(4, "Text Adam", "", ""),
    };
    auto legacy_text_result = gui::StartGraphTrainingFromCompiledConfig(
        legacy_text_nodes,
        links,
        std::move(legacy_text_config),
        registry,
        std::weak_ptr<cyxwiz::TrainingPlotPanel>{},
        [](bool) {},
        legacy_text_dispatch);

    Check(legacy_text_result.started, legacy_text_result.error_message);
    Check(WaitFor([&] { return legacy_text_dispatch_called.load(); },
                  std::chrono::seconds(20)),
          "legacy text dispatch should be called");
    Check(legacy_text_result.effective_dataset_name == kScopeTextDatasetName,
          "legacy text result should keep original dataset");
    Check(legacy_text_result.operators_applied == 0,
          "legacy text result should not apply Arrow materializer operators");
    Check(legacy_text_result.materializer_source_kind ==
              cyxwiz::PipelineMaterializerSourceKind::Unknown,
          "legacy text queued result should not report async source kind");
    Check(!legacy_text_result.materializer_skipped_unsupported_source,
          "legacy text queued result should not report async materializer skip");
    Check(legacy_text_result.status_title == "Training launch queued",
          "legacy text queued result should expose queued launch status");
    registry.UnregisterTextDataset(kScopeTextDatasetName);

    cyxwiz::AsyncTaskManager::Instance().Shutdown();
    std::cout << "Text GUI training launch helper passed\n";
    return 0;
}
