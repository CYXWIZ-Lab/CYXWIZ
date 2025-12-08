#include "cyxmodel_format.h"
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
std::vector<uint8_t> CyxModelFormat::SerializeTensorWithHeader(
    const std::vector<uint8_t>& data,
    const std::vector<int64_t>& shape,
    TensorDType dtype
) {
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
    const ExportOptions& options
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
        auto serialized = SerializeTensorWithHeader(data, shape, TensorDType::Float32);

        // Generate filename (replace . and / with _)
        std::string filename = name;
        for (auto& c : filename) {
            if (c == '.' || c == '/') c = '_';
        }
        filename = "weights/" + filename + ".bin";

        files[filename] = serialized;

        // Add to manifest
        TensorMeta meta;
        meta.name = name;
        meta.shape = shape;
        meta.dtype = TensorDType::Float32;
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

    // Write to archive or directory
    bool use_zip = output_path.size() > 9 &&
                   output_path.substr(output_path.size() - 9) == ".cyxmodel";

    if (use_zip) {
        // For now, use directory-based storage (ZIP can be added with minizip)
        // Create a directory with .cyxmodel extension (it's just a convention)
        return CreateDirectory(output_path, files);
    } else {
        return CreateDirectory(output_path, files);
    }
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
    const ImportOptions& options
) {
    std::map<std::string, std::vector<uint8_t>> files;

    // Read from directory
    if (!ReadDirectory(input_path, files)) {
        return false;
    }

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

        // Load each tensor
        for (const auto& meta : weights_manifest.tensors) {
            std::string filename = meta.name;
            for (auto& c : filename) {
                if (c == '.' || c == '/') c = '_';
            }
            filename = "weights/" + filename + ".bin";

            auto tensor_it = files.find(filename);
            if (tensor_it != files.end()) {
                std::vector<uint8_t> tensor_data;
                std::vector<int64_t> shape;
                TensorDType dtype;

                if (DeserializeTensorWithHeader(tensor_it->second, tensor_data, shape, dtype)) {
                    weights[meta.name] = std::move(tensor_data);
                    weight_shapes[meta.name] = std::move(shape);
                }
            }
        }
    }

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

    result.file_size = std::filesystem::file_size(input_path);
    result.format = ModelFormat::CyxModel;

    // Try to read manifest.json
    std::filesystem::path manifest_path = std::filesystem::path(input_path) / "manifest.json";
    if (!std::filesystem::exists(manifest_path)) {
        result.error_message = "Missing manifest.json";
        return result;
    }

    std::ifstream manifest_file(manifest_path);
    if (!manifest_file.is_open()) {
        result.error_message = "Cannot open manifest.json";
        return result;
    }

    try {
        nlohmann::json j = nlohmann::json::parse(manifest_file);
        ModelManifest manifest = JsonToManifest(j);

        result.valid = true;
        result.format_version = manifest.version;
        result.model_name = manifest.model_name;
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
    } catch (const std::exception& e) {
        result.error_message = "Error parsing manifest: " + std::string(e.what());
    }

    // Load weights manifest for layer info
    std::filesystem::path weights_manifest_path = std::filesystem::path(input_path) / "weights" / "manifest.json";
    if (std::filesystem::exists(weights_manifest_path)) {
        std::ifstream wm_file(weights_manifest_path);
        if (wm_file.is_open()) {
            try {
                nlohmann::json wm_json = nlohmann::json::parse(wm_file);
                WeightsManifest wm = JsonToWeightsManifest(wm_json);
                for (const auto& t : wm.tensors) {
                    result.layer_names.push_back(t.name);
                    result.layer_shapes[t.name] = t.shape;
                }
            } catch (...) {
                // Ignore errors in weights manifest probe
            }
        }
    }

    return result;
}

// Extract graph only
std::string CyxModelFormat::ExtractGraphOnly(const std::string& input_path) {
    std::filesystem::path graph_path = std::filesystem::path(input_path) / "graph.cyxgraph";

    if (!std::filesystem::exists(graph_path)) {
        last_error_ = "No graph.cyxgraph found";
        return "";
    }

    std::ifstream file(graph_path);
    if (!file.is_open()) {
        last_error_ = "Cannot open graph.cyxgraph";
        return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Directory-based storage (simple fallback)
bool CyxModelFormat::CreateDirectory(
    const std::string& output_path,
    const std::map<std::string, std::vector<uint8_t>>& files
) {
    namespace fs = std::filesystem;

    try {
        // Create root directory
        fs::create_directories(output_path);

        for (const auto& [filename, data] : files) {
            fs::path full_path = fs::path(output_path) / filename;

            // Create parent directories if needed
            fs::create_directories(full_path.parent_path());

            // Write file
            std::ofstream file(full_path, std::ios::binary);
            if (!file.is_open()) {
                last_error_ = "Cannot create file: " + full_path.string();
                return false;
            }
            file.write(reinterpret_cast<const char*>(data.data()), data.size());
        }

        spdlog::info("Created .cyxmodel at: {}", output_path);
        return true;
    } catch (const std::exception& e) {
        last_error_ = "Error creating directory: " + std::string(e.what());
        return false;
    }
}

bool CyxModelFormat::ReadDirectory(
    const std::string& input_path,
    std::map<std::string, std::vector<uint8_t>>& files
) {
    namespace fs = std::filesystem;

    try {
        if (!fs::exists(input_path) || !fs::is_directory(input_path)) {
            last_error_ = "Not a valid directory: " + input_path;
            return false;
        }

        fs::path root(input_path);

        for (const auto& entry : fs::recursive_directory_iterator(input_path)) {
            if (entry.is_regular_file()) {
                // Get relative path
                fs::path rel_path = fs::relative(entry.path(), root);
                std::string filename = rel_path.generic_string();

                // Read file
                std::ifstream file(entry.path(), std::ios::binary);
                if (file.is_open()) {
                    file.seekg(0, std::ios::end);
                    size_t size = file.tellg();
                    file.seekg(0, std::ios::beg);

                    std::vector<uint8_t> data(size);
                    file.read(reinterpret_cast<char*>(data.data()), size);
                    files[filename] = std::move(data);
                }
            }
        }

        return true;
    } catch (const std::exception& e) {
        last_error_ = "Error reading directory: " + std::string(e.what());
        return false;
    }
}

std::string CyxModelFormat::GetTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time), "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

bool CyxModelFormat::IsZipFile(const std::string& path) {
    // Check for ZIP magic bytes: PK\x03\x04
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    char magic[4];
    file.read(magic, 4);
    return magic[0] == 'P' && magic[1] == 'K' && magic[2] == 0x03 && magic[3] == 0x04;
}

// ZIP archive operations (placeholder - implement with minizip when available)
bool CyxModelFormat::CreateArchive(
    const std::string& output_path,
    const std::map<std::string, std::vector<uint8_t>>& files,
    bool compress
) {
    // TODO: Implement with minizip for proper ZIP archive creation
    // For now, fall back to directory-based storage
    spdlog::warn("ZIP archive creation not available, using directory storage");
    return CreateDirectory(output_path, files);
}

bool CyxModelFormat::ExtractArchive(
    const std::string& input_path,
    std::map<std::string, std::vector<uint8_t>>& files
) {
    // TODO: Implement with minizip for proper ZIP archive extraction
    // For now, try directory-based storage
    return ReadDirectory(input_path, files);
}

} // namespace formats
} // namespace cyxwiz
