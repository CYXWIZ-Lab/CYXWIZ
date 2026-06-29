#include "model_exporter.h"
#include "training_executor.h"
#include "node_executors/tree_model_artifact.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>

#ifdef CYXWIZ_HAS_ONNX_EXPORT
#include <onnx/onnx_pb.h>
#endif

namespace cyxwiz {

namespace {

struct InferredSequenceConfig {
    bool enabled = false;
    bool batch_first = true;
    bool create_attention_mask = true;
    bool create_causal_lm_targets = false;
    size_t max_sequence_length = 0;
    int64_t word_pad_id = 0;
    int64_t pos_pad_id = 0;
    int64_t tag_ignore_index = -100;
    int64_t target_ignore_index = -100;
    std::string token_vocab_path;
    std::string pos_vocab_path;
    std::string tag_vocab_path;
};

bool LooksLikeTokenizerParams(const nlohmann::json& params) {
    return params.contains("tokenizer_type") ||
           params.contains("method") ||
           params.contains("vocab_file") ||
           params.contains("text_col") ||
           params.contains("text_column") ||
           params.contains("max_length") ||
           params.contains("min_word_freq") ||
           params.contains("max_vocab_size");
}

void CopyParamIfPresent(nlohmann::json& target,
                        const nlohmann::json& params,
                        const char* key) {
    if (!params.contains(key)) {
        return;
    }
    target[key] = params[key];
}

bool InferTextTokenizerAssetsFromGraph(
    const std::string& graph_json,
    std::string& out_config_json,
    std::string& out_vocab_path
) {
    out_config_json.clear();
    out_vocab_path.clear();

    if (graph_json.empty()) {
        return false;
    }

    try {
        const auto graph = nlohmann::json::parse(graph_json);
        if (!graph.contains("nodes") || !graph["nodes"].is_array()) {
            return false;
        }

        nlohmann::json package_config;
        package_config["version"] = "1.0";
        package_config["source"] = "graph";
        package_config["effective"] = nlohmann::json::object();
        package_config["nodes"] = nlohmann::json::array();

        bool found = false;
        std::string candidate_vocab_path;

        for (const auto& node : graph["nodes"]) {
            if (!node.contains("parameters") || !node["parameters"].is_object()) {
                continue;
            }

            const auto& params = node["parameters"];
            if (!LooksLikeTokenizerParams(params)) {
                continue;
            }

            found = true;

            nlohmann::json node_record;
            if (node.contains("id")) {
                node_record["id"] = node["id"];
            }
            if (node.contains("name")) {
                node_record["name"] = node["name"];
            }
            node_record["parameters"] = params;
            package_config["nodes"].push_back(std::move(node_record));

            auto& effective = package_config["effective"];
            CopyParamIfPresent(effective, params, "method");
            CopyParamIfPresent(effective, params, "tokenizer_type");
            CopyParamIfPresent(effective, params, "max_length");
            CopyParamIfPresent(effective, params, "lowercase");
            CopyParamIfPresent(effective, params, "min_word_freq");
            CopyParamIfPresent(effective, params, "max_vocab_size");
            CopyParamIfPresent(effective, params, "pad_value");
            CopyParamIfPresent(effective, params, "text_col");
            CopyParamIfPresent(effective, params, "text_column");
            CopyParamIfPresent(effective, params, "label_col");
            CopyParamIfPresent(effective, params, "label_column");
            CopyParamIfPresent(effective, params, "vocab_file");

            if (params.contains("vocab_file") &&
                params["vocab_file"].is_string() &&
                !params["vocab_file"].get<std::string>().empty()) {
                candidate_vocab_path = params["vocab_file"].get<std::string>();
            }
        }

        if (!found) {
            return false;
        }

        if (!candidate_vocab_path.empty()) {
            package_config["effective"]["vocab_file_packaged"] =
                "tokenizer/vocab.txt";
            if (std::filesystem::exists(candidate_vocab_path)) {
                out_vocab_path = candidate_vocab_path;
            } else {
                package_config["effective"]["vocab_file_packaged"] = nullptr;
                package_config["effective"]["vocab_file_package_warning"] =
                    "Original vocab_file path did not exist at export time";
            }
        }

        out_config_json = package_config.dump(4);
        return true;
    } catch (const std::exception& e) {
        spdlog::warn("Could not infer text tokenizer assets from graph: {}",
                     e.what());
        return false;
    }
}

std::string ToLowerInvariant(const std::string& value) {
    std::string lowered = value;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return lowered;
}

bool ReadInt64Param(
    const nlohmann::json& params,
    const char* key,
    int64_t& out_value
) {
    if (!params.contains(key)) {
        return false;
    }

    const auto& value = params[key];
    if (value.is_number_integer()) {
        out_value = value.get<int64_t>();
        return true;
    }
    if (value.is_number_unsigned()) {
        out_value = static_cast<int64_t>(value.get<uint64_t>());
        return true;
    }
    if (value.is_number_float()) {
        out_value = static_cast<int64_t>(value.get<double>());
        return true;
    }
    if (value.is_string()) {
        try {
            out_value = std::stoll(value.get<std::string>());
            return true;
        } catch (...) {
            return false;
        }
    }
    return false;
}

bool ReadBoolParam(
    const nlohmann::json& params,
    const char* key,
    bool& out_value
) {
    if (!params.contains(key)) {
        return false;
    }

    const auto& value = params[key];
    if (value.is_boolean()) {
        out_value = value.get<bool>();
        return true;
    }
    if (value.is_number_integer()) {
        out_value = value.get<int64_t>() != 0;
        return true;
    }
    if (value.is_string()) {
        const auto lowered = ToLowerInvariant(value.get<std::string>());
        if (lowered == "true" || lowered == "1" || lowered == "yes" ||
            lowered == "on") {
            out_value = true;
            return true;
        }
        if (lowered == "false" || lowered == "0" || lowered == "off") {
            out_value = false;
            return true;
        }
    }
    return false;
}

bool InferSequenceAssetsFromGraph(
    const std::string& graph_json,
    InferredSequenceConfig& out_config
) {
    out_config = InferredSequenceConfig{};
    if (graph_json.empty()) {
        return false;
    }

    try {
        const auto graph = nlohmann::json::parse(graph_json);
        if (!graph.contains("nodes") || !graph["nodes"].is_array()) {
            return false;
        }

        const auto is_sequence_like_key = [](const std::string& key) {
            static const char* keys[] = {
                "token_column",
                "tokens_column",
                "token_sequence_column",
                "pos_column",
                "pos_sequence_column",
                "tag_column",
                "tags_column",
                "tag_sequence_column",
                "sentence_id_column",
                "sequence_id_column",
                "max_sequence_length",
                "sequence_length",
                "ignore_index",
                "target_ignore_index",
                "causal_lm_targets",
                "create_causal_lm_targets",
                "shifted_targets",
                "batch_layout",
                "attention_mask",
                "create_attention_mask",
                "return_sequences",
                "sequence_payload",
                "tag_vocab_file",
                "vocab_file"
            };

            for (const char* known_key : keys) {
                if (key == known_key) {
                    return true;
                }
            }
            return false;
        };

        for (const auto& node : graph["nodes"]) {
            if (!node.contains("parameters") || !node["parameters"].is_object()) {
                continue;
            }

            const auto& params = node["parameters"];
            const std::string node_name =
                node.contains("name") && node["name"].is_string()
                    ? node["name"].get<std::string>()
                    : "";
            const std::string node_name_lower = ToLowerInvariant(node_name);

            bool looks_sequence_like = false;
            for (const auto& param : params.items()) {
                if (is_sequence_like_key(param.key())) {
                    looks_sequence_like = true;
                    break;
                }
            }

            if (!looks_sequence_like) {
                // Keep this conservative: only promote to sequence assets if we
                // clearly see sequence semantics in parameters.
                const bool has_vocab_like = params.contains("vocab_file") ||
                                           params.contains("tag_vocab_file") ||
                                           params.contains("outside_tag") ||
                                           params.contains("pad_tag") ||
                                           params.contains("bio_scheme");
                if (!has_vocab_like) {
                    continue;
                }
                looks_sequence_like = has_vocab_like;
            }

            if (looks_sequence_like) {
                out_config.enabled = true;
            }

            if (params.contains("batch_layout")) {
                bool batch_first = true;
                if (ReadBoolParam(params, "batch_layout", batch_first)) {
                    out_config.batch_first = batch_first;
                } else {
                    const auto layout =
                        ToLowerInvariant(params["batch_layout"].get<std::string>());
                    out_config.batch_first = layout != "time_first";
                }
            }

            bool attention_mask = out_config.create_attention_mask;
            if (ReadBoolParam(params, "create_attention_mask", attention_mask) ||
                ReadBoolParam(params, "attention_mask", attention_mask)) {
                out_config.create_attention_mask = attention_mask;
            }

            bool causal_lm_targets = out_config.create_causal_lm_targets;
            if (ReadBoolParam(params, "create_causal_lm_targets", causal_lm_targets) ||
                ReadBoolParam(params, "causal_lm_targets", causal_lm_targets) ||
                (params.contains("target_ids") && params["target_ids"].is_array()) ||
                (params.contains("shifted_targets") && params["shifted_targets"].is_array())) {
                out_config.create_causal_lm_targets = causal_lm_targets;
            }

            int64_t max_sequence_length = out_config.max_sequence_length;
            if (ReadInt64Param(params, "max_sequence_length", max_sequence_length) ||
                ReadInt64Param(params, "sequence_length", max_sequence_length)) {
                out_config.max_sequence_length =
                    max_sequence_length < 0 ? 0 : static_cast<size_t>(max_sequence_length);
            }

            int64_t tag_ignore_index = out_config.tag_ignore_index;
            if (ReadInt64Param(params, "ignore_index", tag_ignore_index)) {
                out_config.tag_ignore_index = tag_ignore_index;
            }

            int64_t target_ignore_index = out_config.target_ignore_index;
            if (ReadInt64Param(params, "target_ignore_index", target_ignore_index)) {
                out_config.target_ignore_index = target_ignore_index;
            }

            int64_t token_pad_id = out_config.word_pad_id;
            if (ReadInt64Param(params, "token_pad_value", token_pad_id)) {
                out_config.word_pad_id = token_pad_id;
            }
            int64_t pos_pad_id = out_config.pos_pad_id;
            if (ReadInt64Param(params, "pos_pad_value", pos_pad_id)) {
                out_config.pos_pad_id = pos_pad_id;
            }

            auto infer_vocab_for_node = [&]() {
                auto resolve_path = [&](const nlohmann::json& path_value) {
                    if (!path_value.is_string()) {
                        return std::string{};
                    }
                    const auto path = path_value.get<std::string>();
                    return path;
                };

                if (params.contains("tag_vocab_file")) {
                    const auto tag_path = resolve_path(params["tag_vocab_file"]);
                    if (!tag_path.empty()) {
                        out_config.tag_vocab_path = tag_path;
                    }
                    return;
                }

                if (!params.contains("vocab_file")) {
                    return;
                }

                const auto vocab_path = resolve_path(params["vocab_file"]);
                if (vocab_path.empty()) {
                    return;
                }

                const bool tag_like = params.contains("outside_tag") ||
                                      params.contains("pad_tag") ||
                                      params.contains("bio_scheme") ||
                                      (node_name_lower.find("tag") != std::string::npos);
                const bool pos_like = params.contains("pad_tag") == false &&
                                     !tag_like &&
                                     (node_name_lower.find("pos") != std::string::npos);

                if (tag_like) {
                    out_config.tag_vocab_path = vocab_path;
                } else if (pos_like) {
                    out_config.pos_vocab_path = vocab_path;
                } else {
                    out_config.token_vocab_path = vocab_path;
                }
            };
            infer_vocab_for_node();
        }

        return out_config.enabled;
    } catch (const std::exception&) {
        spdlog::warn("Could not infer sequence assets from graph");
        out_config = InferredSequenceConfig{};
        return false;
    }
}

} // namespace

ExportResult ModelExporter::Export(
    SequentialModel& model,
    const Optimizer* optimizer,
    const TrainingMetrics* training_metrics,
    const std::string& graph_json,
    const std::string& output_path,
    const ExportOptions& options,
    ProgressCallback progress_cb
) {
    auto start_time = std::chrono::steady_clock::now();

    // Determine format from options or path
    ModelFormat format = options.format;
    if (format == ModelFormat::Unknown) {
        format = DetectFormat(output_path);
    }

    ExportResult result;

    switch (format) {
        case ModelFormat::CyxModel:
            result = ExportCyxModel(model, optimizer, training_metrics,
                                    graph_json, output_path, options, progress_cb);
            break;

        case ModelFormat::ONNX:
            result = ExportONNX(model, output_path, options, progress_cb);
            break;

        case ModelFormat::Safetensors:
            result = ExportSafetensors(model, output_path, options, progress_cb);
            break;

        case ModelFormat::GGUF:
            result = ExportGGUF(model, output_path, options, progress_cb);
            break;

        default:
            result.success = false;
            result.error_message = "Unknown or unsupported export format";
            return result;
    }

    auto end_time = std::chrono::steady_clock::now();
    result.export_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();

    return result;
}

ExportResult ModelExporter::ExportCyxModel(
    SequentialModel& model,
    const Optimizer* optimizer,
    const TrainingMetrics* training_metrics,
    const std::string& graph_json,
    const std::string& output_path,
    const ExportOptions& options,
    ProgressCallback progress_cb
) {
    ExportResult result;
    result.output_path = output_path;

    if (progress_cb) progress_cb(0, 6, "Preparing model data...");

    try {
        ExportOptions resolved_options = options;
        if (resolved_options.include_tokenizer_assets &&
            resolved_options.text_tokenizer_config_json.empty()) {
            std::string inferred_config;
            std::string inferred_vocab_path;
            if (InferTextTokenizerAssetsFromGraph(graph_json,
                                                  inferred_config,
                                                  inferred_vocab_path)) {
                resolved_options.text_tokenizer_config_json =
                    std::move(inferred_config);
                if (resolved_options.text_tokenizer_vocab_path.empty()) {
                    resolved_options.text_tokenizer_vocab_path =
                        std::move(inferred_vocab_path);
                }
            }
        }

        InferredSequenceConfig inferred_sequence_config;
        if (resolved_options.include_sequence_assets &&
            InferSequenceAssetsFromGraph(graph_json, inferred_sequence_config)) {
            if (resolved_options.sequence_token_vocabulary_path.empty() &&
                !inferred_sequence_config.token_vocab_path.empty()) {
                resolved_options.sequence_token_vocabulary_path =
                    inferred_sequence_config.token_vocab_path;
            }
            if (resolved_options.sequence_pos_vocabulary_path.empty() &&
                !inferred_sequence_config.pos_vocab_path.empty()) {
                resolved_options.sequence_pos_vocabulary_path =
                    inferred_sequence_config.pos_vocab_path;
            }
            if (resolved_options.sequence_tag_vocabulary_path.empty() &&
                !inferred_sequence_config.tag_vocab_path.empty()) {
                resolved_options.sequence_tag_vocabulary_path =
                    inferred_sequence_config.tag_vocab_path;
            }

            resolved_options.sequence_max_sequence_length =
                inferred_sequence_config.max_sequence_length != 0
                    ? inferred_sequence_config.max_sequence_length
                    : resolved_options.sequence_max_sequence_length;
            resolved_options.sequence_batch_first =
                inferred_sequence_config.batch_first;
            resolved_options.sequence_create_attention_mask =
                inferred_sequence_config.create_attention_mask;
            resolved_options.sequence_create_causal_lm_targets =
                inferred_sequence_config.create_causal_lm_targets;
            resolved_options.sequence_word_pad_id =
                inferred_sequence_config.word_pad_id;
            resolved_options.sequence_pos_pad_id =
                inferred_sequence_config.pos_pad_id;
            resolved_options.sequence_tag_ignore_index =
                inferred_sequence_config.tag_ignore_index;
            resolved_options.sequence_target_ignore_index =
                inferred_sequence_config.target_ignore_index;
        }

        // 1. Create manifest
        ModelManifest manifest = CreateManifest(model, training_metrics,
                                                resolved_options);
        manifest.has_tokenizer =
            !resolved_options.text_tokenizer_config_json.empty();
        manifest.has_vocabulary =
            !resolved_options.text_tokenizer_vocab_path.empty();
        manifest.has_tree_model_artifact =
            resolved_options.include_tree_model_artifact;
        if (manifest.has_tree_model_artifact) {
            if (!resolved_options.tree_model_artifact_json.empty()) {
                try {
                    const auto artifact_json =
                        nlohmann::json::parse(
                            resolved_options.tree_model_artifact_json);
                    manifest.tree_model_type =
                        artifact_json.value("model_type", "");
                } catch (const std::exception&) {
                    manifest.tree_model_type.clear();
                }
            } else if (!resolved_options.tree_model_artifact_path.empty()) {
                manifest.tree_model_type = ReadTreeModelArtifactType(
                    resolved_options.tree_model_artifact_path);
            }
        }
        if (resolved_options.include_sequence_assets) {
            manifest.has_sequence =
                inferred_sequence_config.enabled ||
                !resolved_options.sequence_token_vocabulary_path.empty() ||
                !resolved_options.sequence_pos_vocabulary_path.empty() ||
                !resolved_options.sequence_tag_vocabulary_path.empty() ||
                resolved_options.sequence_max_sequence_length != 0;
            manifest.has_sequence_token_vocabulary =
                !resolved_options.sequence_token_vocabulary_path.empty();
            manifest.has_sequence_pos_vocabulary =
                !resolved_options.sequence_pos_vocabulary_path.empty();
            manifest.has_sequence_tag_vocabulary =
                !resolved_options.sequence_tag_vocabulary_path.empty();
            manifest.sequence_batch_first =
                resolved_options.sequence_batch_first;
            manifest.sequence_create_attention_mask =
                resolved_options.sequence_create_attention_mask;
            manifest.sequence_create_causal_lm_targets =
                resolved_options.sequence_create_causal_lm_targets;
            manifest.sequence_max_sequence_length =
                resolved_options.sequence_max_sequence_length;
            manifest.sequence_word_pad_id =
                resolved_options.sequence_word_pad_id;
            manifest.sequence_pos_pad_id =
                resolved_options.sequence_pos_pad_id;
            manifest.sequence_tag_ignore_index =
                resolved_options.sequence_tag_ignore_index;
            manifest.sequence_target_ignore_index =
                resolved_options.sequence_target_ignore_index;
            manifest.sequence_token_vocabulary_path =
                manifest.has_sequence_token_vocabulary ? "sequence/token_vocab.txt" : "";
            manifest.sequence_pos_vocabulary_path =
                manifest.has_sequence_pos_vocabulary ? "sequence/pos_vocab.txt" : "";
            manifest.sequence_tag_vocabulary_path =
                manifest.has_sequence_tag_vocabulary ? "sequence/tag_vocab.txt" : "";
        }
        if (manifest.has_tree_model_artifact) {
            manifest.tree_model_artifact_path = "tree/model.json";
        }

        if (progress_cb) progress_cb(1, 6, "Extracting weights...");

        // 2. Extract weights from model
        std::map<std::string, std::vector<uint8_t>> weights;
        std::map<std::string, std::vector<int64_t>> weight_shapes;

        auto params = model.GetParameters();
        result.num_parameters = 0;
        result.total_tensor_bytes = 0;

        for (const auto& [name, tensor] : params) {
            weights[name] = TensorToBytes(tensor);
            weight_shapes[name] = GetTensorShape(tensor);
            result.num_parameters += static_cast<int>(tensor.NumElements());
            result.total_tensor_bytes += tensor.NumBytes();
        }

        result.num_layers = static_cast<int>(model.Size());
        manifest.num_parameters = result.num_parameters;
        manifest.num_layers = result.num_layers;

        if (progress_cb) progress_cb(2, 6, "Creating training config...");

        // 3. Create training config
        TrainingConfig config = CreateTrainingConfig(optimizer, training_metrics);

        if (progress_cb) progress_cb(3, 6, "Creating training history...");

        // 4. Create training history
        TrainingHistory* history_ptr = nullptr;
        TrainingHistory history;
        if (options.include_training_history && training_metrics) {
            history = CreateTrainingHistory(training_metrics);
            history_ptr = &history;
            manifest.has_training_history = true;
        }

        if (progress_cb) progress_cb(4, 6, "Extracting optimizer state...");

        // 5. Extract optimizer state (if requested)
        // Note: Optimizer state export requires adding GetState() to Optimizer interface
        // For now, we store basic optimizer info in the config
        std::map<std::string, std::vector<uint8_t>>* optimizer_state_ptr = nullptr;
        if (options.include_optimizer_state && optimizer) {
            // TODO: Implement optimizer state export when Optimizer::GetState() is added
            // For now, optimizer state is not exported but config contains optimizer type/lr
            manifest.has_optimizer_state = false;
        }

        // Update manifest flags
        manifest.has_graph = options.include_graph && !graph_json.empty();

        if (progress_cb) progress_cb(5, 6, "Writing .cyxmodel file...");

        // 6. Create the .cyxmodel archive
        formats::CyxModelFormat cyxmodel;
        bool success = cyxmodel.Create(
            output_path,
            manifest,
            resolved_options.include_graph ? graph_json : "",
            config,
            history_ptr,
            weights,
            weight_shapes,
            optimizer_state_ptr,
            resolved_options
        );

        if (!success) {
            result.success = false;
            result.error_message = cyxmodel.GetLastError();
            last_error_ = result.error_message;
            return result;
        }

        if (progress_cb) progress_cb(6, 6, "Export complete!");

        // Get file size
        if (std::filesystem::exists(output_path)) {
            if (std::filesystem::is_directory(output_path)) {
                // Directory-based format - sum up all files
                size_t total_size = 0;
                for (const auto& entry : std::filesystem::recursive_directory_iterator(output_path)) {
                    if (entry.is_regular_file()) {
                        total_size += entry.file_size();
                    }
                }
                result.file_size_bytes = total_size;
            } else {
                result.file_size_bytes = std::filesystem::file_size(output_path);
            }
        }

        result.success = true;
        spdlog::info("Exported model to {} ({} parameters, {} layers, {} bytes)",
                     output_path, result.num_parameters, result.num_layers, result.file_size_bytes);

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("Export failed: ") + e.what();
        last_error_ = result.error_message;
        spdlog::error("Model export failed: {}", e.what());
    }

    return result;
}

ExportResult ModelExporter::ExportONNX(
    SequentialModel& model,
    const std::string& output_path,
    const ExportOptions& options,
    ProgressCallback progress_cb
) {
    ExportResult result;
    result.output_path = output_path;

#ifndef CYXWIZ_HAS_ONNX_EXPORT
    result.success = false;
    result.error_message = "ONNX export support not compiled. Build with CYXWIZ_ENABLE_ONNX=ON and ensure onnx package is installed.";
    last_error_ = result.error_message;
    return result;
#else
    try {
        if (progress_cb) progress_cb(0, 5, "Creating ONNX model...");

        // Create ONNX ModelProto
        onnx::ModelProto model_proto;
        model_proto.set_ir_version(8);  // ONNX IR version 8
        model_proto.set_producer_name("CyxWiz Engine");
        model_proto.set_producer_version("0.2.0");
        model_proto.set_domain("ai.cyxwiz");
        model_proto.set_model_version(1);

        // Set opset import
        auto* opset = model_proto.add_opset_import();
        opset->set_domain("");  // Default ONNX domain
        opset->set_version(options.onnx_opset_version > 0 ? options.onnx_opset_version : 17);

        if (progress_cb) progress_cb(1, 5, "Building computation graph...");

        // Create graph
        auto* graph = model_proto.mutable_graph();
        graph->set_name(options.model_name.empty() ? "cyxwiz_model" : options.model_name);

        // Get model parameters
        auto params = model.GetParameters();
        result.num_parameters = 0;
        result.total_tensor_bytes = 0;

        // Determine input shape from first Linear layer
        std::vector<int64_t> input_shape = {-1};  // -1 for dynamic batch size
        int64_t current_size = 0;

        // Find input size from first Linear layer
        for (const auto& [name, tensor] : params) {
            if (name.find("weight") != std::string::npos) {
                auto shape = tensor.Shape();
                if (shape.size() == 2) {
                    input_shape.push_back(static_cast<int64_t>(shape[1]));  // in_features
                    current_size = static_cast<int64_t>(shape[0]);  // out_features
                    break;
                }
            }
        }

        // Add graph input
        auto* graph_input = graph->add_input();
        graph_input->set_name("input");
        auto* input_type = graph_input->mutable_type()->mutable_tensor_type();
        input_type->set_elem_type(onnx::TensorProto::FLOAT);
        auto* input_shape_proto = input_type->mutable_shape();
        for (auto dim : input_shape) {
            auto* dim_proto = input_shape_proto->add_dim();
            if (dim == -1) {
                dim_proto->set_dim_param("batch_size");
            } else {
                dim_proto->set_dim_value(dim);
            }
        }

        if (progress_cb) progress_cb(2, 5, "Adding weights as initializers...");

        // Add weights as initializers
        for (const auto& [name, tensor] : params) {
            auto* initializer = graph->add_initializer();
            initializer->set_name(name);
            initializer->set_data_type(onnx::TensorProto::FLOAT);

            // Set shape
            for (size_t dim : tensor.Shape()) {
                initializer->add_dims(static_cast<int64_t>(dim));
            }

            // Set raw data
            size_t num_bytes = tensor.NumBytes();
            initializer->set_raw_data(tensor.Data(), num_bytes);

            result.num_parameters += static_cast<int>(tensor.NumElements());
            result.total_tensor_bytes += num_bytes;
        }

        if (progress_cb) progress_cb(3, 5, "Creating ONNX operators...");

        // Build ONNX nodes from model layers
        std::string current_input = "input";
        int node_idx = 0;
        int layer_idx = 0;

        // Iterate through model modules and create corresponding ONNX nodes
        // Note: This requires knowledge of the layer types in SequentialModel
        // We'll use the parameter names to infer layer structure

        // Group parameters by layer
        std::map<int, std::map<std::string, const Tensor*>> layer_params;
        for (const auto& [name, tensor] : params) {
            // Parse layer index from name like "layer0.weight", "layer0.bias"
            if (name.find("layer") == 0) {
                size_t dot_pos = name.find('.');
                if (dot_pos != std::string::npos) {
                    int idx = std::stoi(name.substr(5, dot_pos - 5));
                    std::string param_name = name.substr(dot_pos + 1);
                    layer_params[idx][param_name] = &tensor;
                }
            }
        }

        // Create ONNX nodes for each layer
        for (const auto& [idx, params_map] : layer_params) {
            auto weight_it = params_map.find("weight");
            auto bias_it = params_map.find("bias");

            if (weight_it != params_map.end()) {
                const Tensor* weight = weight_it->second;
                auto weight_shape = weight->Shape();

                if (weight_shape.size() == 2) {
                    // Linear layer -> Gemm node
                    std::string weight_name = "layer" + std::to_string(idx) + ".weight";
                    std::string bias_name = "layer" + std::to_string(idx) + ".bias";
                    std::string output_name = "gemm_" + std::to_string(node_idx) + "_out";

                    auto* gemm_node = graph->add_node();
                    gemm_node->set_name("Gemm_" + std::to_string(node_idx));
                    gemm_node->set_op_type("Gemm");
                    gemm_node->add_input(current_input);
                    gemm_node->add_input(weight_name);
                    if (bias_it != params_map.end()) {
                        gemm_node->add_input(bias_name);
                    }
                    gemm_node->add_output(output_name);

                    // Gemm attributes: Y = alpha * A * B^T + beta * C
                    auto* alpha_attr = gemm_node->add_attribute();
                    alpha_attr->set_name("alpha");
                    alpha_attr->set_f(1.0f);
                    alpha_attr->set_type(onnx::AttributeProto::FLOAT);

                    auto* beta_attr = gemm_node->add_attribute();
                    beta_attr->set_name("beta");
                    beta_attr->set_f(1.0f);
                    beta_attr->set_type(onnx::AttributeProto::FLOAT);

                    auto* transB_attr = gemm_node->add_attribute();
                    transB_attr->set_name("transB");
                    transB_attr->set_i(1);  // Transpose weight matrix
                    transB_attr->set_type(onnx::AttributeProto::INT);

                    current_input = output_name;
                    current_size = static_cast<int64_t>(weight_shape[0]);
                    node_idx++;
                    layer_idx++;

                    // Check if next layer is an activation (based on layer count)
                    // For simplicity, add ReLU after each linear except the last
                    // This is a heuristic - ideally we'd have explicit layer type info
                    if (idx < static_cast<int>(layer_params.size()) - 1) {
                        // Add ReLU activation
                        std::string relu_output = "relu_" + std::to_string(node_idx) + "_out";
                        auto* relu_node = graph->add_node();
                        relu_node->set_name("Relu_" + std::to_string(node_idx));
                        relu_node->set_op_type("Relu");
                        relu_node->add_input(current_input);
                        relu_node->add_output(relu_output);
                        current_input = relu_output;
                        node_idx++;
                    }
                }
            }
        }

        // Add Softmax at the end (common for classification)
        if (options.add_softmax_output) {
            std::string softmax_output = "softmax_out";
            auto* softmax_node = graph->add_node();
            softmax_node->set_name("Softmax_output");
            softmax_node->set_op_type("Softmax");
            softmax_node->add_input(current_input);
            softmax_node->add_output(softmax_output);

            auto* axis_attr = softmax_node->add_attribute();
            axis_attr->set_name("axis");
            axis_attr->set_i(-1);  // Last axis
            axis_attr->set_type(onnx::AttributeProto::INT);

            current_input = softmax_output;
        }

        // Add graph output
        auto* graph_output = graph->add_output();
        graph_output->set_name("output");
        auto* output_type = graph_output->mutable_type()->mutable_tensor_type();
        output_type->set_elem_type(onnx::TensorProto::FLOAT);
        auto* output_shape = output_type->mutable_shape();
        auto* batch_dim = output_shape->add_dim();
        batch_dim->set_dim_param("batch_size");
        auto* feature_dim = output_shape->add_dim();
        feature_dim->set_dim_value(current_size);

        // Rename the last node's output to "output"
        if (graph->node_size() > 0) {
            auto* last_node = graph->mutable_node(graph->node_size() - 1);
            last_node->set_output(0, "output");
        }

        if (progress_cb) progress_cb(4, 5, "Writing ONNX file...");

        // Serialize to file
        std::ofstream output_file(output_path, std::ios::binary);
        if (!output_file) {
            result.success = false;
            result.error_message = "Failed to open output file: " + output_path;
            last_error_ = result.error_message;
            return result;
        }

        if (!model_proto.SerializeToOstream(&output_file)) {
            result.success = false;
            result.error_message = "Failed to serialize ONNX model";
            last_error_ = result.error_message;
            return result;
        }

        output_file.close();

        if (progress_cb) progress_cb(5, 5, "Export complete!");

        result.success = true;
        result.num_layers = layer_idx;
        result.file_size_bytes = std::filesystem::file_size(output_path);

        spdlog::info("Exported model to ONNX: {} ({} layers, {} parameters, {} bytes)",
                     output_path, result.num_layers, result.num_parameters, result.file_size_bytes);

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("ONNX export failed: ") + e.what();
        last_error_ = result.error_message;
        spdlog::error("ONNX export failed: {}", e.what());
    }

    return result;
#endif
}

ExportResult ModelExporter::ExportSafetensors(
    SequentialModel& model,
    const std::string& output_path,
    const ExportOptions& options,
    ProgressCallback progress_cb
) {
    ExportResult result;
    result.output_path = output_path;

    if (progress_cb) progress_cb(0, 3, "Preparing tensors...");

    try {
        // Safetensors format:
        // [8 bytes] header_size (little-endian u64)
        // [header_size bytes] JSON header with tensor metadata
        // [remaining] tensor data (contiguous, aligned)

        auto params = model.GetParameters();

        // Build header JSON
        nlohmann::json header;
        size_t data_offset = 0;
        std::vector<std::pair<std::string, const Tensor*>> ordered_tensors;

        for (const auto& [name, tensor] : params) {
            nlohmann::json tensor_info;

            // Determine dtype string
            std::string dtype_str = "F32";
            switch (tensor.GetDataType()) {
                case DataType::Float32: dtype_str = "F32"; break;
                case DataType::Float64: dtype_str = "F64"; break;
                case DataType::Int32: dtype_str = "I32"; break;
                case DataType::Int64: dtype_str = "I64"; break;
                case DataType::UInt8: dtype_str = "U8"; break;
            }

            tensor_info["dtype"] = dtype_str;

            // Shape
            std::vector<int64_t> shape;
            for (size_t dim : tensor.Shape()) {
                shape.push_back(static_cast<int64_t>(dim));
            }
            tensor_info["shape"] = shape;

            // Offsets
            size_t tensor_size = tensor.NumBytes();
            tensor_info["data_offsets"] = std::vector<size_t>{data_offset, data_offset + tensor_size};

            header[name] = tensor_info;
            data_offset += tensor_size;

            ordered_tensors.push_back({name, &tensor});
            result.num_parameters += static_cast<int>(tensor.NumElements());
        }

        // Add metadata
        if (!options.model_name.empty() || !options.author.empty()) {
            nlohmann::json metadata;
            if (!options.model_name.empty()) metadata["model_name"] = options.model_name;
            if (!options.author.empty()) metadata["author"] = options.author;
            if (!options.description.empty()) metadata["description"] = options.description;
            header["__metadata__"] = metadata;
        }

        if (progress_cb) progress_cb(1, 3, "Writing header...");

        // Serialize header
        std::string header_str = header.dump();

        // Pad header to 8-byte alignment
        while ((header_str.size() + 8) % 8 != 0) {
            header_str += ' ';
        }

        uint64_t header_size = header_str.size();

        // Write file
        std::ofstream file(output_path, std::ios::binary);
        if (!file) {
            result.success = false;
            result.error_message = "Failed to open output file: " + output_path;
            last_error_ = result.error_message;
            return result;
        }

        // Write header size (little-endian)
        file.write(reinterpret_cast<const char*>(&header_size), 8);

        // Write header
        file.write(header_str.c_str(), header_str.size());

        if (progress_cb) progress_cb(2, 3, "Writing tensor data...");

        // Write tensor data in order
        for (const auto& [name, tensor] : ordered_tensors) {
            file.write(static_cast<const char*>(tensor->Data()), tensor->NumBytes());
            result.total_tensor_bytes += tensor->NumBytes();
        }

        file.close();

        if (progress_cb) progress_cb(3, 3, "Export complete!");

        result.success = true;
        result.num_layers = static_cast<int>(model.Size());
        result.file_size_bytes = std::filesystem::file_size(output_path);

        spdlog::info("Exported model to Safetensors: {} ({} tensors, {} bytes)",
                     output_path, ordered_tensors.size(), result.file_size_bytes);

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("Safetensors export failed: ") + e.what();
        last_error_ = result.error_message;
        spdlog::error("Safetensors export failed: {}", e.what());
    }

    return result;
}

ExportResult ModelExporter::ExportGGUF(
    SequentialModel& /*model*/,
    const std::string& output_path,
    const ExportOptions& /*options*/,
    ProgressCallback progress_cb
) {
    ExportResult result;
    result.output_path = output_path;

    // GGUF format is complex and primarily used for LLMs
    // Basic implementation here for simple models

    if (progress_cb) progress_cb(0, 1, "GGUF export...");

    result.success = false;
    result.error_message = "GGUF export is planned for future release. Use .cyxmodel or .safetensors for now.";
    last_error_ = result.error_message;

    return result;
}

std::vector<ModelFormat> ModelExporter::GetSupportedFormats() {
    std::vector<ModelFormat> formats = {
        ModelFormat::CyxModel,
        ModelFormat::Safetensors
    };

    // Note: Use CYXWIZ_HAS_ONNX_EXPORT (not CYXWIZ_HAS_ONNX) to distinguish between:
    // - CYXWIZ_HAS_ONNX: ONNX Runtime available for inference (model loading)
    // - CYXWIZ_HAS_ONNX_EXPORT: ONNX protobuf available for export (model saving)
    // On macOS, Runtime is ON (vendored binary) but Export is OFF (protobuf conflicts)
#ifdef CYXWIZ_HAS_ONNX_EXPORT
    formats.push_back(ModelFormat::ONNX);
#endif

    // GGUF support coming soon
    // formats.push_back(ModelFormat::GGUF);

    return formats;
}

bool ModelExporter::IsFormatSupported(ModelFormat format) {
    switch (format) {
        case ModelFormat::CyxModel:
            return true;
        case ModelFormat::Safetensors:
            return true;
        case ModelFormat::ONNX:
            // Use CYXWIZ_HAS_ONNX_EXPORT to check for model export capability
            // (not CYXWIZ_HAS_ONNX which only indicates inference runtime availability)
#ifdef CYXWIZ_HAS_ONNX_EXPORT
            return true;
#else
            return false;
#endif
        case ModelFormat::GGUF:
            return false;  // Not yet implemented
        default:
            return false;
    }
}

std::string ModelExporter::GetExtension(ModelFormat format) {
    return GetFormatExtension(format);
}

ModelFormat ModelExporter::DetectFormat(const std::string& path) {
    std::filesystem::path p(path);
    std::string ext = p.extension().string();
    return GetFormatFromExtension(ext);
}

bool ModelExporter::ValidateForExport(
    SequentialModel& model,
    ModelFormat format,
    std::string& error_message
) {
    // Check if model has any layers
    if (model.Size() == 0) {
        error_message = "Model has no layers";
        return false;
    }

    // Check if model has parameters
    auto params = model.GetParameters();
    if (params.empty()) {
        error_message = "Model has no trainable parameters";
        return false;
    }

    // Format-specific validation
    switch (format) {
        case ModelFormat::ONNX:
            // Use CYXWIZ_HAS_ONNX_EXPORT for model export validation
            // (CYXWIZ_HAS_ONNX only indicates inference capability, not export)
#ifndef CYXWIZ_HAS_ONNX_EXPORT
            error_message = "ONNX export support not compiled";
            return false;
#endif
            // TODO: Validate that all layer types are supported in ONNX
            break;

        case ModelFormat::GGUF:
            error_message = "GGUF export not yet implemented";
            return false;

        default:
            break;
    }

    return true;
}

std::vector<uint8_t> ModelExporter::TensorToBytes(const Tensor& tensor) {
    size_t num_bytes = tensor.NumBytes();
    std::vector<uint8_t> bytes(num_bytes);

    const void* data = tensor.Data();
    if (data && num_bytes > 0) {
        std::memcpy(bytes.data(), data, num_bytes);
    }

    return bytes;
}

std::vector<int64_t> ModelExporter::GetTensorShape(const Tensor& tensor) {
    const auto& shape = tensor.Shape();
    std::vector<int64_t> result;
    result.reserve(shape.size());

    for (size_t dim : shape) {
        result.push_back(static_cast<int64_t>(dim));
    }

    return result;
}

int ModelExporter::CountParameters(SequentialModel& model) {
    int total = 0;
    auto params = model.GetParameters();
    for (const auto& [name, tensor] : params) {
        total += static_cast<int>(tensor.NumElements());
    }
    return total;
}

ModelManifest ModelExporter::CreateManifest(
    SequentialModel& model,
    const TrainingMetrics* metrics,
    const ExportOptions& options
) {
    ModelManifest manifest;

    manifest.version = "1.0";
    manifest.format = "cyxmodel";
    manifest.cyxwiz_version = "0.2.0";  // TODO: Get from version header

    // Timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&time_t));
    manifest.created = buf;

    // Model info
    manifest.model_name = options.model_name.empty() ? "Untitled Model" : options.model_name;
    manifest.model_type = "SequentialModel";
    manifest.num_parameters = CountParameters(model);
    manifest.num_layers = static_cast<int>(model.Size());

    // Training summary (if available)
    if (metrics) {
        manifest.epochs_trained = metrics->current_epoch;
        manifest.final_accuracy = metrics->train_accuracy;
        manifest.final_loss = metrics->train_loss;
    }

    // Metadata
    manifest.author = options.author;
    manifest.description = options.description;
    manifest.custom_metadata = options.custom_metadata;
    if (options.include_tokenizer_assets &&
        !options.text_tokenizer_config_json.empty()) {
        manifest.custom_metadata["text_tokenizer"] = "packaged";
    }
    if (options.include_tokenizer_assets &&
        !options.text_tokenizer_vocab_path.empty()) {
        manifest.custom_metadata["text_vocabulary"] = "packaged";
    }

    return manifest;
}

TrainingConfig ModelExporter::CreateTrainingConfig(
    const Optimizer* optimizer,
    const TrainingMetrics* metrics
) {
    TrainingConfig config;

    if (optimizer) {
        // Get optimizer type name
        // Note: This is a simplified implementation
        // In a full implementation, we'd have a proper type system
        config.optimizer_type = "Adam";  // Default, should be extracted from optimizer
        config.learning_rate = static_cast<float>(optimizer->GetLearningRate());

        // Get other optimizer parameters if available
        // This depends on the optimizer interface
    }

    if (metrics) {
        config.epochs = metrics->total_epochs;
        config.batch_size = metrics->total_batches > 0 ? 32 : 0;  // Approximation
    }

    return config;
}

TrainingHistory ModelExporter::CreateTrainingHistory(const TrainingMetrics* metrics) {
    TrainingHistory history;

    if (metrics) {
        history.loss_history = metrics->loss_history;
        history.accuracy_history = metrics->accuracy_history;
        history.val_loss_history = metrics->val_loss_history;
        history.val_accuracy_history = metrics->val_accuracy_history;

        // Compute best metrics
        if (!history.loss_history.empty()) {
            auto min_it = std::min_element(history.loss_history.begin(), history.loss_history.end());
            history.best_loss = *min_it;
            history.best_epoch = static_cast<int>(std::distance(history.loss_history.begin(), min_it));
        }

        if (!history.accuracy_history.empty()) {
            auto max_it = std::max_element(history.accuracy_history.begin(), history.accuracy_history.end());
            history.best_accuracy = *max_it;
        }
    }

    return history;
}

} // namespace cyxwiz
