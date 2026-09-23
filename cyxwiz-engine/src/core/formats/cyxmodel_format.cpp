#include "cyxmodel_format.h"
#include "cyxmodel_archive.h"
#include <set>
#include <limits>
#include <spdlog/spdlog.h>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <cstring>

namespace cyxwiz {
namespace formats {



// JSON serialization for ModelManifest
nlohmann::json CyxModelFormat::ManifestToJson(const ModelManifest& manifest) {
    nlohmann::json j;
    j["version"] = manifest.version;
    j["format"] = manifest.format;
    j["created"] = manifest.created;
    j["cyxwiz_version"] = manifest.cyxwiz_version;

    j["model"]["name"] = manifest.model_name;
    j["model"]["type"] = manifest.model_type;
    j["model"]["family"] = manifest.model_family;
    j["model"]["supports_generation"] = manifest.supports_generation;
    j["model"]["generation_output_contract"] =
        manifest.generation_output_contract;
    j["model"]["supports_bert_encoder"] = manifest.supports_bert_encoder;
    j["model"]["bert_encoder_task"] = manifest.bert_encoder_task;
    j["model"]["bert_encoder_input_kind"] =
        manifest.bert_encoder_input_kind;
    j["model"]["bert_encoder_output_contract"] =
        manifest.bert_encoder_output_contract;
    j["model"]["bert_encoder_has_attention_mask"] =
        manifest.bert_encoder_has_attention_mask;
    j["model"]["bert_encoder_requires_token_type_ids"] =
        manifest.bert_encoder_requires_token_type_ids;
    j["model"]["num_parameters"] = manifest.num_parameters;
    j["model"]["num_layers"] = manifest.num_layers;

    j["training"]["epochs_trained"] = manifest.epochs_trained;
    j["training"]["final_accuracy"] = manifest.final_accuracy;
    j["training"]["final_loss"] = manifest.final_loss;

    j["metadata"]["author"] = manifest.author;
    j["metadata"]["description"] = manifest.description;
    j["metadata"]["custom"] = manifest.custom_metadata;

    j["content"]["has_optimizer_state"] = manifest.has_optimizer_state;
    j["content"]["has_training_history"] = manifest.has_training_history;
    j["content"]["has_graph"] = manifest.has_graph;
    j["content"]["has_tokenizer"] = manifest.has_tokenizer;
    j["content"]["has_vocabulary"] = manifest.has_vocabulary;
    j["content"]["has_sequence"] = manifest.has_sequence;
    j["content"]["has_sequence_token_vocabulary"] =
        manifest.has_sequence_token_vocabulary;
    j["content"]["has_sequence_pos_vocabulary"] =
        manifest.has_sequence_pos_vocabulary;
    j["content"]["has_sequence_tag_vocabulary"] =
        manifest.has_sequence_tag_vocabulary;
    j["content"]["has_tree_model_artifact"] =
        manifest.has_tree_model_artifact;
    j["content"]["tree_model_type"] = manifest.tree_model_type;
    j["content"]["tree_model_artifact_path"] =
        manifest.tree_model_artifact_path;
    j["content"]["sequence_batch_first"] = manifest.sequence_batch_first;
    j["content"]["sequence_create_attention_mask"] =
        manifest.sequence_create_attention_mask;
    j["content"]["sequence_create_causal_lm_targets"] =
        manifest.sequence_create_causal_lm_targets;
    j["content"]["sequence_max_sequence_length"] =
        manifest.sequence_max_sequence_length;
    j["content"]["sequence_word_pad_id"] = manifest.sequence_word_pad_id;
    j["content"]["sequence_pos_pad_id"] = manifest.sequence_pos_pad_id;
    j["content"]["sequence_tag_ignore_index"] =
        manifest.sequence_tag_ignore_index;
    j["content"]["sequence_target_ignore_index"] =
        manifest.sequence_target_ignore_index;
    j["content"]["sequence_token_vocabulary_path"] =
        manifest.sequence_token_vocabulary_path;
    j["content"]["sequence_pos_vocabulary_path"] =
        manifest.sequence_pos_vocabulary_path;
    j["content"]["sequence_tag_vocabulary_path"] =
        manifest.sequence_tag_vocabulary_path;

    return j;
}

ModelManifest CyxModelFormat::JsonToManifest(const nlohmann::json& j) {
    ModelManifest manifest;

    manifest.version = j.value("version", "1.0");
    manifest.format = j.value("format", "cyxmodel");
    manifest.created = j.value("created", "");
    manifest.cyxwiz_version = j.value("cyxwiz_version", "");

    if (j.contains("model")) {
        manifest.model_name = j["model"].value("name", "");
        manifest.model_type = j["model"].value("type", "");
        manifest.model_family = j["model"].value("family", "");
        manifest.supports_generation =
            j["model"].value("supports_generation", false);
        manifest.generation_output_contract =
            j["model"].value("generation_output_contract", "");
        manifest.supports_bert_encoder =
            j["model"].value("supports_bert_encoder", false);
        manifest.bert_encoder_task =
            j["model"].value("bert_encoder_task", "");
        manifest.bert_encoder_input_kind =
            j["model"].value("bert_encoder_input_kind", "");
        manifest.bert_encoder_output_contract =
            j["model"].value("bert_encoder_output_contract", "");
        manifest.bert_encoder_has_attention_mask =
            j["model"].value("bert_encoder_has_attention_mask", false);
        manifest.bert_encoder_requires_token_type_ids =
            j["model"].value("bert_encoder_requires_token_type_ids", false);
        manifest.num_parameters = j["model"].value("num_parameters", 0);
        manifest.num_layers = j["model"].value("num_layers", 0);
    }

    if (j.contains("training")) {
        manifest.epochs_trained = j["training"].value("epochs_trained", 0);
        manifest.final_accuracy = j["training"].value("final_accuracy", 0.0f);
        manifest.final_loss = j["training"].value("final_loss", 0.0f);
    }

    if (j.contains("metadata")) {
        manifest.author = j["metadata"].value("author", "");
        manifest.description = j["metadata"].value("description", "");
        if (j["metadata"].contains("custom")) {
            manifest.custom_metadata = j["metadata"]["custom"].get<std::map<std::string, std::string>>();
        }
    }

    if (j.contains("content")) {
        manifest.has_optimizer_state = j["content"].value("has_optimizer_state", false);
        manifest.has_training_history = j["content"].value("has_training_history", false);
        manifest.has_graph = j["content"].value("has_graph", false);
        manifest.has_tokenizer = j["content"].value("has_tokenizer", false);
        manifest.has_vocabulary = j["content"].value("has_vocabulary", false);
        manifest.has_sequence = j["content"].value("has_sequence", false);
        manifest.has_sequence_token_vocabulary =
            j["content"].value("has_sequence_token_vocabulary", false);
        manifest.has_sequence_pos_vocabulary =
            j["content"].value("has_sequence_pos_vocabulary", false);
        manifest.has_sequence_tag_vocabulary =
            j["content"].value("has_sequence_tag_vocabulary", false);
        manifest.has_tree_model_artifact =
            j["content"].value("has_tree_model_artifact", false);
        manifest.tree_model_type =
            j["content"].value("tree_model_type", "");
        manifest.tree_model_artifact_path =
            j["content"].value("tree_model_artifact_path", "");
        manifest.sequence_batch_first =
            j["content"].value("sequence_batch_first", true);
        manifest.sequence_create_attention_mask =
            j["content"].value("sequence_create_attention_mask", true);
        manifest.sequence_create_causal_lm_targets =
            j["content"].value("sequence_create_causal_lm_targets", false);
        manifest.sequence_max_sequence_length =
            j["content"].value("sequence_max_sequence_length", size_t{0});
        manifest.sequence_word_pad_id =
            j["content"].value("sequence_word_pad_id", int64_t{0});
        manifest.sequence_pos_pad_id =
            j["content"].value("sequence_pos_pad_id", int64_t{0});
        manifest.sequence_tag_ignore_index =
            j["content"].value("sequence_tag_ignore_index", int64_t{-100});
        manifest.sequence_target_ignore_index =
            j["content"].value("sequence_target_ignore_index", int64_t{-100});
        manifest.sequence_token_vocabulary_path =
            j["content"].value("sequence_token_vocabulary_path", "");
        manifest.sequence_pos_vocabulary_path =
            j["content"].value("sequence_pos_vocabulary_path", "");
        manifest.sequence_tag_vocabulary_path =
            j["content"].value("sequence_tag_vocabulary_path", "");
    }

    return manifest;
}

// JSON serialization for TrainingConfig
nlohmann::json CyxModelFormat::ConfigToJson(const TrainingConfig& config) {
    nlohmann::json j;

    j["optimizer"]["type"] = config.optimizer_type;
    j["optimizer"]["learning_rate"] = config.learning_rate;
    j["optimizer"]["momentum"] = config.momentum;
    j["optimizer"]["weight_decay"] = config.weight_decay;
    j["optimizer"]["beta1"] = config.beta1;
    j["optimizer"]["beta2"] = config.beta2;
    j["optimizer"]["epsilon"] = config.epsilon;

    j["training"]["batch_size"] = config.batch_size;
    j["training"]["epochs"] = config.epochs;
    j["training"]["loss_function"] = config.loss_function;

    j["data"]["dataset_name"] = config.dataset_name;
    j["data"]["num_classes"] = config.num_classes;
    j["data"]["input_shape"] = config.input_shape;

    return j;
}

TrainingConfig CyxModelFormat::JsonToConfig(const nlohmann::json& j) {
    TrainingConfig config;

    if (j.contains("optimizer")) {
        config.optimizer_type = j["optimizer"].value("type", "");
        config.learning_rate = j["optimizer"].value("learning_rate", 0.001f);
        config.momentum = j["optimizer"].value("momentum", 0.9f);
        config.weight_decay = j["optimizer"].value("weight_decay", 0.0f);
        config.beta1 = j["optimizer"].value("beta1", 0.9f);
        config.beta2 = j["optimizer"].value("beta2", 0.999f);
        config.epsilon = j["optimizer"].value("epsilon", 1e-8f);
    }

    if (j.contains("training")) {
        config.batch_size = j["training"].value("batch_size", 32);
        config.epochs = j["training"].value("epochs", 0);
        config.loss_function = j["training"].value("loss_function", "");
    }

    if (j.contains("data")) {
        config.dataset_name = j["data"].value("dataset_name", "");
        config.num_classes = j["data"].value("num_classes", 0);
        if (j["data"].contains("input_shape")) {
            config.input_shape = j["data"]["input_shape"].get<std::vector<int64_t>>();
        }
    }

    return config;
}

// JSON serialization for TrainingHistory
nlohmann::json CyxModelFormat::HistoryToJson(const TrainingHistory& history) {
    nlohmann::json j;

    j["loss"] = history.loss_history;
    j["accuracy"] = history.accuracy_history;
    j["val_loss"] = history.val_loss_history;
    j["val_accuracy"] = history.val_accuracy_history;
    j["learning_rate"] = history.learning_rate_history;
    j["timestamps"] = history.epoch_timestamps;

    j["best"]["accuracy"] = history.best_accuracy;
    j["best"]["loss"] = history.best_loss;
    j["best"]["epoch"] = history.best_epoch;

    return j;
}

TrainingHistory CyxModelFormat::JsonToHistory(const nlohmann::json& j) {
    TrainingHistory history;

    if (j.contains("loss")) history.loss_history = j["loss"].get<std::vector<float>>();
    if (j.contains("accuracy")) history.accuracy_history = j["accuracy"].get<std::vector<float>>();
    if (j.contains("val_loss")) history.val_loss_history = j["val_loss"].get<std::vector<float>>();
    if (j.contains("val_accuracy")) history.val_accuracy_history = j["val_accuracy"].get<std::vector<float>>();
    if (j.contains("learning_rate")) history.learning_rate_history = j["learning_rate"].get<std::vector<float>>();
    if (j.contains("timestamps")) history.epoch_timestamps = j["timestamps"].get<std::vector<int64_t>>();

    if (j.contains("best")) {
        history.best_accuracy = j["best"].value("accuracy", 0.0f);
        history.best_loss = j["best"].value("loss", std::numeric_limits<float>::max());
        history.best_epoch = j["best"].value("epoch", 0);
    }

    return history;
}

// JSON serialization for WeightsManifest
nlohmann::json CyxModelFormat::WeightsManifestToJson(const WeightsManifest& manifest) {
    nlohmann::json j;
    j["version"] = manifest.version;
    j["total_tensors"] = manifest.total_tensors;
    j["total_bytes"] = manifest.total_bytes;

    nlohmann::json tensors = nlohmann::json::array();
    for (const auto& t : manifest.tensors) {
        nlohmann::json tensor;
        tensor["name"] = t.name;
        tensor["shape"] = t.shape;
        tensor["dtype"] = static_cast<int>(t.dtype);
        tensor["offset"] = t.offset;
        tensor["size_bytes"] = t.size_bytes;
        tensors.push_back(tensor);
    }
    j["tensors"] = tensors;

    return j;
}

WeightsManifest CyxModelFormat::JsonToWeightsManifest(const nlohmann::json& j) {
    WeightsManifest manifest;

    manifest.version = j.value("version", "1.0");
    manifest.total_tensors = j.value("total_tensors", 0);
    manifest.total_bytes = j.value("total_bytes", 0);

    if (j.contains("tensors")) {
        for (const auto& t : j["tensors"]) {
            TensorMeta meta;
            meta.name = t.value("name", "");
            meta.shape = t["shape"].get<std::vector<int64_t>>();
            meta.dtype = static_cast<TensorDType>(t.value("dtype", 0));
            meta.offset = t.value("offset", 0);
            meta.size_bytes = t.value("size_bytes", 0);
            manifest.tensors.push_back(meta);
        }
    }

    return manifest;
}

// Binary tensor serialization with header
namespace {
size_t TensorPayloadBytes(const std::vector<int64_t>& shape, TensorDType dtype) {
    size_t bytes=0;
    switch(dtype) {
        case TensorDType::Float32: case TensorDType::Int32: bytes=4; break;
        case TensorDType::Float64: case TensorDType::Int64: bytes=8; break;
        case TensorDType::UInt8: bytes=1; break;
        default: throw std::runtime_error("Unsupported native tensor dtype");
    }
    if (shape.size()>8) throw std::runtime_error("Unsupported tensor rank");
    for (const auto dim : shape) {
        if (dim<0 || (dim>0 && bytes>std::numeric_limits<size_t>::max()/static_cast<size_t>(dim)))
            throw std::runtime_error("Invalid tensor dimensions/byte overflow");
        bytes*=static_cast<size_t>(dim);
    }
    return bytes;
}
}

std::vector<uint8_t> CyxModelFormat::SerializeTensorWithHeader(
    const std::vector<uint8_t>& data,
    const std::vector<int64_t>& shape,
    TensorDType dtype
) {
    if (TensorPayloadBytes(shape,dtype)!=data.size()) throw std::runtime_error("Tensor byte count mismatch");
    std::vector<uint8_t> result;

    // Header format:
    // [4 bytes] uint32_t ndims
    // [8 bytes each] int64_t shape[ndims]
    // [4 bytes] uint32_t dtype
    // [data] raw tensor data

    uint32_t ndims = static_cast<uint32_t>(shape.size());
    uint32_t dtype_val = static_cast<uint32_t>(dtype);

    // Calculate total size
    size_t header_size = 4 + (8 * ndims) + 4;
    result.resize(header_size + data.size());

    // Write ndims
    std::memcpy(result.data(), &ndims, 4);
    size_t offset = 4;

    // Write shape
    for (size_t i = 0; i < ndims; ++i) {
        std::memcpy(result.data() + offset, &shape[i], 8);
        offset += 8;
    }

    // Write dtype
    std::memcpy(result.data() + offset, &dtype_val, 4);
    offset += 4;

    // Write data
    std::memcpy(result.data() + offset, data.data(), data.size());

    return result;
}

bool CyxModelFormat::DeserializeTensorWithHeader(
    const std::vector<uint8_t>& data,
    std::vector<uint8_t>& tensor_data,
    std::vector<int64_t>& shape,
    TensorDType& dtype
) {
    if (data.size() < 8) {
        last_error_ = "Invalid tensor data: too small";
        return false;
    }

    size_t offset = 0;

    // Read ndims
    uint32_t ndims;
    std::memcpy(&ndims, data.data() + offset, 4);
    offset += 4;

    // Validate
    if (ndims > 8 || data.size() < offset + (8 * ndims) + 4) {
        last_error_ = "Invalid tensor header";
        return false;
    }

    // Read shape
    shape.resize(ndims);
    for (size_t i = 0; i < ndims; ++i) {
        std::memcpy(&shape[i], data.data() + offset, 8);
        offset += 8;
    }

    // Read dtype
    uint32_t dtype_val;
    std::memcpy(&dtype_val, data.data() + offset, 4);
    dtype = static_cast<TensorDType>(dtype_val);
    offset += 4;

    // Read data
    size_t tensor_size = data.size() - offset;
    try {
        if (TensorPayloadBytes(shape,dtype)!=tensor_size) { last_error_="Tensor byte count mismatch"; return false; }
    } catch(const std::exception& e) { last_error_=e.what(); return false; }
    tensor_data.resize(tensor_size);
    std::memcpy(tensor_data.data(), data.data() + offset, tensor_size);

    return true;
}

// Create .cyxmodel archive
bool CyxModelFormat::Create(
    const std::string& output_path,
    const ModelManifest& manifest,
    const std::string& graph_json,
    const TrainingConfig& config,
    const TrainingHistory* history,
    const std::map<std::string, std::vector<uint8_t>>& weights,
    const std::map<std::string, std::vector<int64_t>>& weight_shapes,
    const std::map<std::string, std::vector<uint8_t>>* optimizer_state,
    const ExportOptions& options,
    const std::map<std::string, TensorDType>* weight_dtypes
) {
    std::map<std::string, std::vector<uint8_t>> files;

    // Create manifest.json
    nlohmann::json manifest_json = ManifestToJson(manifest);
    std::string manifest_str = manifest_json.dump(4);
    files["manifest.json"] = std::vector<uint8_t>(manifest_str.begin(), manifest_str.end());

    // Create graph.cyxgraph
    if (!graph_json.empty() && options.include_graph) {
        files["graph.cyxgraph"] = std::vector<uint8_t>(graph_json.begin(), graph_json.end());
    }

    // Create config.json
    nlohmann::json config_json = ConfigToJson(config);
    std::string config_str = config_json.dump(4);
    files["config.json"] = std::vector<uint8_t>(config_str.begin(), config_str.end());

    // Create history.json (optional)
    if (history && options.include_training_history) {
        nlohmann::json history_json = HistoryToJson(*history);
        std::string history_str = history_json.dump(4);
        files["history.json"] = std::vector<uint8_t>(history_str.begin(), history_str.end());
    }

    // Create weights directory
    WeightsManifest weights_manifest;
    weights_manifest.total_tensors = static_cast<int>(weights.size());
    size_t total_bytes = 0;

    for (const auto& [name, data] : weights) {
        // Get shape
        auto shape_it = weight_shapes.find(name);
        std::vector<int64_t> shape;
        if (shape_it != weight_shapes.end()) {
            shape = shape_it->second;
        }

        // Serialize with header
        const auto dtype = weight_dtypes ? weight_dtypes->at(name) : TensorDType::Float32;
        auto serialized = SerializeTensorWithHeader(data, shape, dtype);

        // Generate filename (replace . and / with _)
        std::string filename = name;
        for (auto& c : filename) {
            if (c == '.' || c == '/') c = '_';
        }
        filename = "weights/" + filename + ".bin";

        if (files.count(filename)) { last_error_ = "Colliding tensor filename: " + filename; return false; }
        files[filename] = serialized;

        // Add to manifest
        TensorMeta meta;
        meta.name = name;
        meta.shape = shape;
        meta.dtype = dtype;
        meta.size_bytes = serialized.size();
        weights_manifest.tensors.push_back(meta);
        total_bytes += serialized.size();
    }

    weights_manifest.total_bytes = total_bytes;
    nlohmann::json weights_manifest_json = WeightsManifestToJson(weights_manifest);
    std::string weights_manifest_str = weights_manifest_json.dump(4);
    files["weights/manifest.json"] = std::vector<uint8_t>(weights_manifest_str.begin(), weights_manifest_str.end());

    // Create optimizer directory (optional)
    if (optimizer_state && options.include_optimizer_state && !optimizer_state->empty()) {
        for (const auto& [name, data] : *optimizer_state) {
            std::string filename = name;
            for (auto& c : filename) {
                if (c == '.' || c == '/') c = '_';
            }
            filename = "optimizer/" + filename + ".bin";
            files[filename] = data;
        }
    }

    // Create tokenizer deployment assets (optional)
    if (options.include_tokenizer_assets) {
        if (!options.text_tokenizer_config_json.empty()) {
            const std::string& tokenizer_config =
                options.text_tokenizer_config_json;
            files["tokenizer/config.json"] =
                std::vector<uint8_t>(tokenizer_config.begin(),
                                     tokenizer_config.end());
        }

        if (!options.text_tokenizer_vocab_data.empty() && !options.text_tokenizer_vocab_path.empty()) {
            last_error_ = "Tokenizer export cannot specify both vocabulary bytes and a file path";
            return false;
        }
        if (!options.text_tokenizer_vocab_data.empty()) {
            files["tokenizer/vocab.txt"] = std::vector<uint8_t>(
                options.text_tokenizer_vocab_data.begin(), options.text_tokenizer_vocab_data.end());
        } else if (!options.text_tokenizer_vocab_path.empty()) {
            std::ifstream vocab_file(options.text_tokenizer_vocab_path,
                                     std::ios::binary);
            if (!vocab_file.is_open()) {
                last_error_ = "Cannot open tokenizer vocabulary file: " +
                              options.text_tokenizer_vocab_path;
                return false;
            }

            std::vector<uint8_t> vocab_bytes(
                (std::istreambuf_iterator<char>(vocab_file)),
                std::istreambuf_iterator<char>());
            files["tokenizer/vocab.txt"] = std::move(vocab_bytes);
        }

        if (!options.text_tokenizer_model_data.empty() && !options.text_tokenizer_model_path.empty()) {
            last_error_ = "Tokenizer export cannot specify both model bytes and a file path";
            return false;
        }
        if (!options.text_tokenizer_model_data.empty()) {
            files["tokenizer/model.spm"] = std::vector<uint8_t>(
                options.text_tokenizer_model_data.begin(),
                options.text_tokenizer_model_data.end());
        } else if (!options.text_tokenizer_model_path.empty()) {
            std::ifstream model_file(options.text_tokenizer_model_path,
                                     std::ios::binary);
            if (!model_file.is_open()) {
                last_error_ = "Cannot open tokenizer model file: " +
                              options.text_tokenizer_model_path;
                return false;
            }

            std::vector<uint8_t> model_bytes(
                (std::istreambuf_iterator<char>(model_file)),
                std::istreambuf_iterator<char>());
            files["tokenizer/model.spm"] = std::move(model_bytes);
        }
    }

    // Create sequence vocabulary assets (optional)
    if (options.include_sequence_assets) {
        if (!options.sequence_token_vocabulary_path.empty()) {
            std::ifstream token_vocab_file(
                options.sequence_token_vocabulary_path,
                std::ios::binary);
            if (!token_vocab_file.is_open()) {
                last_error_ = "Cannot open sequence token vocabulary file: " +
                              options.sequence_token_vocabulary_path;
                return false;
            }

            std::vector<uint8_t> vocab_bytes(
                (std::istreambuf_iterator<char>(token_vocab_file)),
                std::istreambuf_iterator<char>());
            files["sequence/token_vocab.txt"] = std::move(vocab_bytes);
        }

        if (!options.sequence_pos_vocabulary_path.empty()) {
            std::ifstream pos_vocab_file(
                options.sequence_pos_vocabulary_path,
                std::ios::binary);
            if (!pos_vocab_file.is_open()) {
                last_error_ = "Cannot open sequence POS vocabulary file: " +
                              options.sequence_pos_vocabulary_path;
                return false;
            }

            std::vector<uint8_t> vocab_bytes(
                (std::istreambuf_iterator<char>(pos_vocab_file)),
                std::istreambuf_iterator<char>());
            files["sequence/pos_vocab.txt"] = std::move(vocab_bytes);
        }

        if (!options.sequence_tag_vocabulary_path.empty()) {
            std::ifstream tag_vocab_file(
                options.sequence_tag_vocabulary_path,
                std::ios::binary);
            if (!tag_vocab_file.is_open()) {
                last_error_ = "Cannot open sequence tag vocabulary file: " +
                              options.sequence_tag_vocabulary_path;
                return false;
            }

            std::vector<uint8_t> vocab_bytes(
                (std::istreambuf_iterator<char>(tag_vocab_file)),
                std::istreambuf_iterator<char>());
            files["sequence/tag_vocab.txt"] = std::move(vocab_bytes);
        }
    }

    if (options.include_tree_model_artifact) {
        if (!options.tree_model_artifact_json.empty()) {
            const std::string& artifact_json = options.tree_model_artifact_json;
            files["tree/model.json"] =
                std::vector<uint8_t>(artifact_json.begin(),
                                     artifact_json.end());
        } else if (!options.tree_model_artifact_path.empty()) {
            std::ifstream artifact_file(options.tree_model_artifact_path,
                                        std::ios::binary);
            if (!artifact_file.is_open()) {
                last_error_ = "Cannot open tree model artifact file: " +
                              options.tree_model_artifact_path;
                return false;
            }

            std::vector<uint8_t> artifact_bytes(
                (std::istreambuf_iterator<char>(artifact_file)),
                std::istreambuf_iterator<char>());
            files["tree/model.json"] = std::move(artifact_bytes);
        }
    }

    return CreateArchive(output_path, files, options.compress);
}

// Extract .cyxmodel archive
bool CyxModelFormat::Extract(
    const std::string& input_path,
    ModelManifest& manifest,
    std::string& graph_json,
    TrainingConfig& config,
    TrainingHistory* history,
    std::map<std::string, std::vector<uint8_t>>& weights,
    std::map<std::string, std::vector<int64_t>>& weight_shapes,
    std::map<std::string, std::vector<uint8_t>>* optimizer_state,
    const ImportOptions& options,
    std::map<std::string, TensorDType>* weight_dtypes
) {
    std::map<std::string, std::vector<uint8_t>> files;

    // Read from directory
    if (!ReadPackage(input_path, files)) {
        return false;
    }

    return Extract(files,manifest,graph_json,config,history,weights,weight_shapes,
                   optimizer_state,options,weight_dtypes);
}

bool CyxModelFormat::Extract(
    const std::map<std::string, std::vector<uint8_t>>& files,
    ModelManifest& manifest,
    std::string& graph_json,
    TrainingConfig& config,
    TrainingHistory* history,
    std::map<std::string, std::vector<uint8_t>>& weights,
    std::map<std::string, std::vector<int64_t>>& weight_shapes,
    std::map<std::string, std::vector<uint8_t>>* optimizer_state,
    const ImportOptions& options,
    std::map<std::string, TensorDType>* weight_dtypes
) {
    weights.clear(); weight_shapes.clear(); graph_json.clear();
    if (weight_dtypes) weight_dtypes->clear();
    if (optimizer_state) optimizer_state->clear();
    // Parse manifest.json
    auto manifest_it = files.find("manifest.json");
    if (manifest_it == files.end()) {
        last_error_ = "Missing manifest.json";
        return false;
    }
    std::string manifest_str(manifest_it->second.begin(), manifest_it->second.end());
    manifest = JsonToManifest(nlohmann::json::parse(manifest_str));

    // Parse graph.cyxgraph
    auto graph_it = files.find("graph.cyxgraph");
    if (graph_it != files.end()) {
        graph_json = std::string(graph_it->second.begin(), graph_it->second.end());
    }

    const std::pair<bool,const char*> required_assets[] = {
        {manifest.has_graph,"graph.cyxgraph"}, {manifest.has_tokenizer,"tokenizer/config.json"},
        {manifest.has_vocabulary,"tokenizer/vocab.txt"}, {manifest.has_training_history,"history.json"},
        {manifest.has_tree_model_artifact,"tree/model.json"},
        {manifest.has_sequence_token_vocabulary,manifest.sequence_token_vocabulary_path.c_str()},
        {manifest.has_sequence_pos_vocabulary,manifest.sequence_pos_vocabulary_path.c_str()},
        {manifest.has_sequence_tag_vocabulary,manifest.sequence_tag_vocabulary_path.c_str()}};
    for (const auto& [required,name] : required_assets) {
        if (required && !files.count(name)) { last_error_=std::string("Missing declared asset: ")+name; return false; }
    }

    // Parse config.json
    auto config_it = files.find("config.json");
    if (config_it != files.end()) {
        std::string config_str(config_it->second.begin(), config_it->second.end());
        config = JsonToConfig(nlohmann::json::parse(config_str));
    }

    // Parse history.json
    if (history && options.load_training_history) {
        auto history_it = files.find("history.json");
        if (history_it != files.end()) {
            std::string history_str(history_it->second.begin(), history_it->second.end());
            *history = JsonToHistory(nlohmann::json::parse(history_str));
        }
    }

    // Load weights manifest
    auto weights_manifest_it = files.find("weights/manifest.json");
    if (weights_manifest_it != files.end()) {
        std::string wm_str(weights_manifest_it->second.begin(), weights_manifest_it->second.end());
        WeightsManifest weights_manifest = JsonToWeightsManifest(nlohmann::json::parse(wm_str));

        std::set<std::string> tensor_names;
        std::set<std::string> tensor_files;
        // Load each tensor
        for (const auto& meta : weights_manifest.tensors) {
            if (!tensor_names.insert(meta.name).second) { last_error_="Duplicate tensor: "+meta.name; return false; }
            std::string filename = meta.name;
            for (auto& c : filename) {
                if (c == '.' || c == '/') c = '_';
            }
            filename = "weights/" + filename + ".bin";

            if (!tensor_files.insert(filename).second) { last_error_="Colliding tensor filenames: "+filename; return false; }
            auto tensor_it = files.find(filename);
            if (tensor_it != files.end()) {
                std::vector<uint8_t> tensor_data;
                std::vector<int64_t> shape;
                TensorDType dtype;

                if (DeserializeTensorWithHeader(tensor_it->second, tensor_data, shape, dtype)) {
                    if (shape != meta.shape || dtype != meta.dtype || meta.size_bytes != tensor_it->second.size()) {
                        last_error_="Tensor inventory/header mismatch: "+meta.name; return false;
                    }
                    weights[meta.name] = std::move(tensor_data);
                    weight_shapes[meta.name] = std::move(shape);
                    if (weight_dtypes) (*weight_dtypes)[meta.name]=dtype;
                } else { return false; }
            } else { last_error_="Missing tensor payload: "+meta.name; return false; }
        }
    } else { last_error_="Missing weights/manifest.json"; return false; }

    // Load optimizer state
    if (optimizer_state && options.load_optimizer_state) {
        for (const auto& [filename, data] : files) {
            if (filename.substr(0, 10) == "optimizer/" && filename != "optimizer/state.json") {
                std::string name = filename.substr(10);
                if (name.size() > 4 && name.substr(name.size() - 4) == ".bin") {
                    name = name.substr(0, name.size() - 4);
                }
                (*optimizer_state)[name] = data;
            }
        }
    }

    return true;
}

// Probe file for metadata
ProbeResult CyxModelFormat::Probe(const std::string& input_path) {
    ProbeResult result;

    // Check if path exists
    if (!std::filesystem::exists(input_path)) {
        result.error_message = "File not found: " + input_path;
        return result;
    }

    if (std::filesystem::is_directory(input_path)) {
        size_t total_size = 0;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(input_path)) {
            if (entry.is_regular_file()) {
                total_size += entry.file_size();
            }
        }
        result.file_size = total_size;
    } else {
        result.file_size = std::filesystem::file_size(input_path);
    }
    result.format = ModelFormat::CyxModel;

    // Try to read manifest.json
    std::map<std::string, std::vector<uint8_t>> files;
    if (!ReadPackage(input_path,files)) { result.error_message=last_error_; return result; }
    if (!files.count("manifest.json")) { result.error_message="Missing manifest.json"; return result; }
    const auto& manifest_bytes = files.at("manifest.json");
    try {
        nlohmann::json j = nlohmann::json::parse(manifest_bytes.begin(),manifest_bytes.end());
        ModelManifest manifest = JsonToManifest(j);

        result.valid = true;
        result.format_version = CyxModelArchive::IsV3(input_path) ? "CYXW v3" : manifest.version;
        result.model_name = manifest.model_name;
        result.model_family = manifest.model_family;
        result.supports_generation = manifest.supports_generation;
        result.generation_output_contract = manifest.generation_output_contract;
        result.supports_bert_encoder = manifest.supports_bert_encoder;
        result.bert_encoder_task = manifest.bert_encoder_task;
        result.bert_encoder_input_kind = manifest.bert_encoder_input_kind;
        result.bert_encoder_output_contract =
            manifest.bert_encoder_output_contract;
        result.bert_encoder_has_attention_mask =
            manifest.bert_encoder_has_attention_mask;
        result.bert_encoder_requires_token_type_ids =
            manifest.bert_encoder_requires_token_type_ids;
        result.author = manifest.author;
        result.description = manifest.description;
        result.num_parameters = manifest.num_parameters;
        result.num_layers = manifest.num_layers;
        result.epochs_trained = manifest.epochs_trained;
        result.final_accuracy = manifest.final_accuracy;
        result.final_loss = manifest.final_loss;
        result.has_optimizer_state = manifest.has_optimizer_state;
        result.has_training_history = manifest.has_training_history;
        result.has_graph = manifest.has_graph;
        result.has_tokenizer = manifest.has_tokenizer;
        result.has_vocabulary = manifest.has_vocabulary;
        result.has_sequence = manifest.has_sequence;
        result.has_sequence_token_vocabulary =
            manifest.has_sequence_token_vocabulary;
        result.has_sequence_pos_vocabulary =
            manifest.has_sequence_pos_vocabulary;
        result.has_sequence_tag_vocabulary =
            manifest.has_sequence_tag_vocabulary;
        result.has_tree_model_artifact = manifest.has_tree_model_artifact;
        result.tree_model_type = manifest.tree_model_type;
        result.tree_model_artifact_path = manifest.tree_model_artifact_path;
        if (files.count("tree/model.json")) {
            result.has_tree_model_artifact = true;
            if (result.tree_model_artifact_path.empty()) result.tree_model_artifact_path="tree/model.json";
            if (result.tree_model_type.empty()) {
                const auto& bytes=files.at("tree/model.json");
                result.tree_model_type=nlohmann::json::parse(bytes.begin(),bytes.end()).value("model_type",std::string{});
            }
        }
        result.sequence_batch_first = manifest.sequence_batch_first;
        result.sequence_create_attention_mask =
            manifest.sequence_create_attention_mask;
        result.sequence_create_causal_lm_targets =
            manifest.sequence_create_causal_lm_targets;
        result.sequence_max_sequence_length =
            manifest.sequence_max_sequence_length;
        result.sequence_word_pad_id = manifest.sequence_word_pad_id;
        result.sequence_pos_pad_id = manifest.sequence_pos_pad_id;
        result.sequence_tag_ignore_index = manifest.sequence_tag_ignore_index;
        result.sequence_target_ignore_index =
            manifest.sequence_target_ignore_index;
        result.sequence_token_vocabulary_path =
            manifest.sequence_token_vocabulary_path;
        result.sequence_pos_vocabulary_path =
            manifest.sequence_pos_vocabulary_path;
        result.sequence_tag_vocabulary_path =
            manifest.sequence_tag_vocabulary_path;
    } catch (const std::exception& e) {
        result.valid = false;
        result.error_message = "Error parsing manifest: " + std::string(e.what());
    }

    if (files.count("weights/manifest.json")) {
        try {
            const auto& bytes=files.at("weights/manifest.json");
            const auto wm=JsonToWeightsManifest(nlohmann::json::parse(bytes.begin(),bytes.end()));
            for (const auto& tensor : wm.tensors) {
                result.layer_names.push_back(tensor.name);
                result.layer_shapes[tensor.name]=tensor.shape;
            }
        } catch (const std::exception& e) { result.valid=false; result.error_message=e.what(); }
    }
    return result;
}

// Extract graph only
std::string CyxModelFormat::ExtractGraphOnly(const std::string& input_path) {
    std::map<std::string,std::vector<uint8_t>> files;
    if (!ReadPackage(input_path,files)) return {};
    const auto found=files.find("graph.cyxgraph");
    if (found==files.end()) { last_error_="No graph.cyxgraph found"; return {}; }
    return std::string(found->second.begin(),found->second.end());
}

bool CyxModelFormat::ExtractTextTokenizerAssets(
    const std::string& input_path,
    std::string& config_json,
    std::string& vocab_text
) {
    std::string model_data;
    return ExtractTextTokenizerAssets(input_path, config_json, vocab_text, model_data);
}

bool CyxModelFormat::ExtractTextTokenizerAssets(
    const std::string& input_path,
    std::string& config_json,
    std::string& vocab_text,
    std::string& model_data
) {
    std::map<std::string, std::vector<uint8_t>> files;
    if (!ReadPackage(input_path, files)) {
        return false;
    }

    config_json.clear();
    vocab_text.clear();
    model_data.clear();

    auto config_it = files.find("tokenizer/config.json");
    if (config_it != files.end()) {
        config_json.assign(config_it->second.begin(), config_it->second.end());
    }

    auto vocab_it = files.find("tokenizer/vocab.txt");
    if (vocab_it != files.end()) {
        vocab_text.assign(vocab_it->second.begin(), vocab_it->second.end());
    }

    auto model_it = files.find("tokenizer/model.spm");
    if (model_it == files.end()) {
        model_it = files.find("tokenizer/model.spm.fb");
    }
    if (model_it != files.end()) {
        model_data.assign(model_it->second.begin(), model_it->second.end());
    }

    if (config_json.empty() && vocab_text.empty() && model_data.empty()) {
        last_error_ = "No tokenizer assets found";
        return false;
    }

    return true;
}

bool CyxModelFormat::ExtractSequenceVocabularyAssets(
    const std::string& input_path,
    std::string& token_vocab_text,
    std::string& pos_vocab_text,
    std::string& tag_vocab_text
) {
    std::map<std::string, std::vector<uint8_t>> files;
    if (!ReadPackage(input_path, files)) {
        return false;
    }

    token_vocab_text.clear();
    pos_vocab_text.clear();
    tag_vocab_text.clear();

    auto token_it = files.find("sequence/token_vocab.txt");
    if (token_it != files.end()) {
        token_vocab_text.assign(token_it->second.begin(),
                               token_it->second.end());
    }

    auto pos_it = files.find("sequence/pos_vocab.txt");
    if (pos_it != files.end()) {
        pos_vocab_text.assign(pos_it->second.begin(), pos_it->second.end());
    }

    auto tag_it = files.find("sequence/tag_vocab.txt");
    if (tag_it != files.end()) {
        tag_vocab_text.assign(tag_it->second.begin(), tag_it->second.end());
    }

    if (token_vocab_text.empty() && pos_vocab_text.empty() &&
        tag_vocab_text.empty()) {
        last_error_ = "No sequence vocabulary assets found";
        return false;
    }

    return true;
}

bool CyxModelFormat::ExtractTreeModelArtifact(
    const std::string& input_path,
    std::string& artifact_json
) {
    std::map<std::string, std::vector<uint8_t>> files;
    if (!ReadPackage(input_path, files)) {
        return false;
    }

    artifact_json.clear();

    auto artifact_it = files.find("tree/model.json");
    if (artifact_it != files.end()) {
        artifact_json.assign(artifact_it->second.begin(),
                             artifact_it->second.end());
    }

    if (artifact_json.empty()) {
        last_error_ = "No tree model artifact found";
        return false;
    }

    return true;
}

// Directory-based storage (simple fallback)
bool CyxModelFormat::ReadPackage(
    const std::string& input_path,
    std::map<std::string, std::vector<uint8_t>>& files) {
    try { files = CyxModelArchive::Read(input_path); return true; }
    catch (const std::exception& e) { last_error_ = e.what(); return false; }
}

std::string CyxModelFormat::GetTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time), "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

bool CyxModelFormat::CreateArchive(
    const std::string& output_path,
    const std::map<std::string, std::vector<uint8_t>>& files,
    bool compress) {
    try {
        if (compress) throw std::runtime_error("CYXW v3 compression is not implemented; disable Compress");
        CyxModelArchive::WriteBinary(output_path, files);
        spdlog::info("Created CYXW v3 binary model: {}", output_path);
        return true;
    } catch (const std::exception& e) { last_error_ = e.what(); return false; }
}

} // namespace formats
} // namespace cyxwiz
