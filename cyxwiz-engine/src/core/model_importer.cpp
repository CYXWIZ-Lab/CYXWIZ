#include "model_importer.h"
#include "graph_compiler.h"
#include <cyxwiz/sequential.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <cstring>

namespace cyxwiz {

// Forward declaration
static bool IsCyxwBinaryFormat(const std::string& path);

// Helper: Build model architecture from graph JSON
static bool BuildModelFromGraph(
    const std::string& graph_json,
    SequentialModel& model,
    std::string& error_message
) {
    if (graph_json.empty()) {
        error_message = "No graph data available to rebuild model architecture";
        return false;
    }

    try {
        using json = nlohmann::json;
        json j = json::parse(graph_json);

        // Parse nodes
        std::vector<gui::MLNode> nodes;
        for (const auto& node_json : j["nodes"]) {
            gui::MLNode node;
            node.id = node_json["id"];
            node.type = static_cast<gui::NodeType>(node_json["type"].get<int>());
            node.name = node_json["name"];
            if (node_json.contains("parameters")) {
                node.parameters = node_json["parameters"].get<std::map<std::string, std::string>>();
            }
            nodes.push_back(node);
        }

        // Parse links
        std::vector<gui::NodeLink> links;
        for (const auto& link_json : j["links"]) {
            gui::NodeLink link;
            link.id = link_json["id"];
            link.from_node = link_json["from_node"];
            link.to_node = link_json["to_node"];
            links.push_back(link);
        }

        // Compile graph to get layer configuration
        GraphCompiler compiler;
        TrainingConfiguration config = compiler.Compile(nodes, links);

        if (!config.is_valid) {
            error_message = "Graph compilation failed: " + config.error_message;
            return false;
        }

        spdlog::info("Building model from graph: {} layers, input={}, output={}",
                     config.layers.size(), config.input_size, config.output_size);

        // Build model layers from configuration
        size_t current_input_size = config.input_size;

        for (size_t i = 0; i < config.layers.size(); ++i) {
            const auto& layer_cfg = config.layers[i];

            switch (layer_cfg.type) {
                case gui::NodeType::Dense: {
                    size_t out_features = 64;
                    if (layer_cfg.parameters.count("units")) {
                        out_features = std::stoul(layer_cfg.parameters.at("units"));
                    }
                    model.Add<LinearModule>(current_input_size, out_features, true);
                    spdlog::debug("  [{}] Linear({} -> {})", i, current_input_size, out_features);
                    current_input_size = out_features;
                    break;
                }

                case gui::NodeType::ReLU:
                    model.Add<ReLUModule>();
                    break;

                case gui::NodeType::Sigmoid:
                    model.Add<SigmoidModule>();
                    break;

                case gui::NodeType::Tanh:
                    model.Add<TanhModule>();
                    break;

                case gui::NodeType::Softmax:
                    model.Add<SoftmaxModule>();
                    break;

                case gui::NodeType::Output:
                    // Output node is just a terminal marker, not an actual layer
                    // The last Dense layer already outputs the correct features
                    break;

                case gui::NodeType::Dropout: {
                    float rate = 0.5f;
                    if (layer_cfg.parameters.count("rate")) {
                        rate = std::stof(layer_cfg.parameters.at("rate"));
                    }
                    model.Add<DropoutModule>(rate);
                    break;
                }

                default:
                    // Skip non-layer nodes (preprocessing, loss, optimizer, etc.)
                    break;
            }
        }

        if (model.Size() == 0) {
            error_message = "No layers were created from the graph";
            return false;
        }

        spdlog::info("Built model with {} layers from graph", model.Size());
        return true;

    } catch (const std::exception& e) {
        error_message = std::string("Failed to parse graph: ") + e.what();
        return false;
    }
}

ProbeResult ModelImporter::ProbeFile(const std::string& input_path) {
    ProbeResult result;

    // Check file exists
    if (!std::filesystem::exists(input_path)) {
        result.valid = false;
        result.error_message = "File not found: " + input_path;
        return result;
    }

    // Detect format
    result.format = DetectFormat(input_path);

    if (result.format == ModelFormat::Unknown) {
        result.valid = false;
        result.error_message = "Unknown model format";
        return result;
    }

    // Get file size
    if (std::filesystem::is_directory(input_path)) {
        // Directory-based .cyxmodel
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

    // Format-specific probing
    switch (result.format) {
        case ModelFormat::CyxModel: {
            formats::CyxModelFormat cyxmodel;
            result = cyxmodel.Probe(input_path);
            break;
        }

        case ModelFormat::Safetensors: {
            // Read safetensors header
            std::ifstream file(input_path, std::ios::binary);
            if (!file) {
                result.valid = false;
                result.error_message = "Failed to open file";
                return result;
            }

            // Read header size
            uint64_t header_size = 0;
            file.read(reinterpret_cast<char*>(&header_size), 8);

            if (header_size > 100 * 1024 * 1024) {  // Sanity check: 100MB max header
                result.valid = false;
                result.error_message = "Invalid header size";
                return result;
            }

            // Read header JSON
            std::string header_str(header_size, '\0');
            file.read(&header_str[0], header_size);

            try {
                auto header = nlohmann::json::parse(header_str);

                for (auto& [key, value] : header.items()) {
                    if (key == "__metadata__") {
                        if (value.contains("model_name")) {
                            result.model_name = value["model_name"];
                        }
                        if (value.contains("author")) {
                            result.author = value["author"];
                        }
                        if (value.contains("description")) {
                            result.description = value["description"];
                        }
                    } else {
                        // Tensor entry
                        result.layer_names.push_back(key);

                        if (value.contains("shape")) {
                            std::vector<int64_t> shape;
                            for (const auto& dim : value["shape"]) {
                                shape.push_back(dim.get<int64_t>());
                            }
                            result.layer_shapes[key] = shape;

                            // Count parameters
                            int64_t params = 1;
                            for (int64_t dim : shape) params *= dim;
                            result.num_parameters += static_cast<int>(params);
                        }
                    }
                }

                result.num_layers = static_cast<int>(result.layer_names.size());
                result.format_version = "safetensors";
                result.valid = true;

            } catch (const std::exception& e) {
                result.valid = false;
                result.error_message = std::string("Failed to parse header: ") + e.what();
            }
            break;
        }

        case ModelFormat::ONNX:
        case ModelFormat::GGUF:
            result.valid = false;
            result.error_message = "Probing not yet implemented for this format";
            break;

        default:
            result.valid = false;
            result.error_message = "Unknown format";
            break;
    }

    return result;
}

ImportResult ModelImporter::Import(
    const std::string& input_path,
    SequentialModel& model,
    const ImportOptions& options,
    ProgressCallback progress_cb
) {
    auto start_time = std::chrono::steady_clock::now();

    // Detect format
    ModelFormat format = DetectFormat(input_path);

    ImportResult result;

    switch (format) {
        case ModelFormat::CyxModel:
            result = ImportCyxModel(input_path, model, options, progress_cb);
            break;

        case ModelFormat::Safetensors:
            result = ImportSafetensors(input_path, model, options, progress_cb);
            break;

        case ModelFormat::ONNX:
            result = ImportONNX(input_path, model, options, progress_cb);
            break;

        case ModelFormat::GGUF:
            result = ImportGGUF(input_path, model, options, progress_cb);
            break;

        default:
            result.success = false;
            result.error_message = "Unknown or unsupported model format";
            return result;
    }

    auto end_time = std::chrono::steady_clock::now();
    result.load_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();

    return result;
}

// Import from binary CYXW format
ImportResult ModelImporter::ImportCyxModelBinary(
    const std::string& input_path,
    SequentialModel& model,
    const ImportOptions& options,
    ProgressCallback progress_cb
) {
    ImportResult result;

    if (progress_cb) progress_cb(0, 5, "Reading binary .cyxmodel file...");

    try {
        std::ifstream file(input_path, std::ios::binary);
        if (!file) {
            result.error_message = "Failed to open file: " + input_path;
            return result;
        }

        // Read and verify magic
        uint32_t magic;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != 0x43595857) {
            result.error_message = "Invalid magic number";
            return result;
        }

        // Read version
        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 2) {
            result.error_message = "Unsupported format version: " + std::to_string(version);
            return result;
        }

        if (progress_cb) progress_cb(1, 5, "Reading metadata...");

        // Read JSON metadata
        uint64_t json_len;
        file.read(reinterpret_cast<char*>(&json_len), sizeof(json_len));

        std::string json_str(json_len, '\0');
        file.read(json_str.data(), json_len);

        auto meta = nlohmann::json::parse(json_str);

        // Extract model info
        if (meta.contains("metadata")) {
            result.model_name = meta["metadata"].value("name", "");
        }
        result.format_version = "CYXW v2";

        // Read number of modules
        size_t num_modules;
        file.read(reinterpret_cast<char*>(&num_modules), sizeof(num_modules));

        if (progress_cb) progress_cb(2, 5, "Building model from metadata...");

        // Build model architecture from metadata if model is empty
        if (model.Size() == 0 && meta.contains("modules")) {
            for (const auto& mod_json : meta["modules"]) {
                if (!mod_json.value("has_parameters", false)) {
                    // Activation layer - try to infer from name
                    std::string name = mod_json.value("name", "");
                    if (name.find("ReLU") != std::string::npos) {
                        model.Add<ReLUModule>();
                    } else if (name.find("Sigmoid") != std::string::npos) {
                        model.Add<SigmoidModule>();
                    } else if (name.find("Tanh") != std::string::npos) {
                        model.Add<TanhModule>();
                    } else if (name.find("Softmax") != std::string::npos) {
                        model.Add<SoftmaxModule>();
                    } else if (name.find("Dropout") != std::string::npos) {
                        model.Add<DropoutModule>(0.5f);
                    }
                } else if (mod_json.contains("parameters")) {
                    // Linear layer - get shape from parameters
                    auto& params = mod_json["parameters"];
                    for (const auto& p : params) {
                        if (p["name"] == "weight" && p.contains("shape")) {
                            auto shape = p["shape"];
                            if (shape.size() == 2) {
                                size_t out_features = shape[0].get<size_t>();
                                size_t in_features = shape[1].get<size_t>();
                                model.Add<LinearModule>(in_features, out_features, true);
                                break;
                            }
                        }
                    }
                }
            }
            spdlog::info("Built model with {} layers from binary metadata", model.Size());
        }

        if (progress_cb) progress_cb(3, 5, "Loading weights...");

        // Verify module count matches
        if (num_modules != model.Size()) {
            result.warnings.push_back("Module count mismatch: file has " +
                std::to_string(num_modules) + ", model has " + std::to_string(model.Size()));
        }

        // Load weights into model
        size_t modules_to_load = std::min(num_modules, model.Size());
        for (size_t i = 0; i < num_modules; ++i) {
            size_t num_params;
            file.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));

            std::map<std::string, Tensor> params;
            for (size_t j = 0; j < num_params; ++j) {
                // Read parameter name
                size_t name_len;
                file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
                std::string name(name_len, '\0');
                file.read(name.data(), name_len);

                // Read tensor shape
                size_t ndims;
                file.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));
                std::vector<size_t> shape(ndims);
                file.read(reinterpret_cast<char*>(shape.data()), ndims * sizeof(size_t));

                // Read dtype
                DataType dtype;
                file.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));

                // Read data
                size_t num_bytes;
                file.read(reinterpret_cast<char*>(&num_bytes), sizeof(num_bytes));

                Tensor tensor(shape, dtype);
                file.read(reinterpret_cast<char*>(tensor.Data()), num_bytes);

                params[name] = std::move(tensor);
                result.layer_names.push_back("layer_" + std::to_string(i) + "." + name);
                result.num_parameters += static_cast<int>(num_bytes / 4);  // Assume float32
            }

            // Set parameters on model module
            if (i < model.Size()) {
                auto* module = model.GetModule(i);
                if (module && module->HasParameters()) {
                    module->SetParameters(params);
                }
            }
        }

        if (progress_cb) progress_cb(4, 5, "Finalizing...");

        result.success = true;
        result.num_layers = static_cast<int>(model.Size());

        if (progress_cb) progress_cb(5, 5, "Import complete!");

        spdlog::info("Imported binary model: {} ({} layers)", input_path, result.num_layers);

    } catch (const std::exception& e) {
        result.error_message = std::string("Import failed: ") + e.what();
        spdlog::error("ImportCyxModelBinary: {}", result.error_message);
    }

    return result;
}

ImportResult ModelImporter::ImportCyxModel(
    const std::string& input_path,
    SequentialModel& model,
    const ImportOptions& options,
    ProgressCallback progress_cb
) {
    // Check if this is binary format
    if (IsCyxwBinaryFormat(input_path)) {
        spdlog::info("Detected binary CYXW format: {}", input_path);
        return ImportCyxModelBinary(input_path, model, options, progress_cb);
    }

    ImportResult result;

    if (progress_cb) progress_cb(0, 5, "Reading .cyxmodel file...");

    try {
        formats::CyxModelFormat cyxmodel;

        ModelManifest manifest;
        std::string graph_json;
        TrainingConfig config;
        TrainingHistory history;
        std::map<std::string, std::vector<uint8_t>> weights;
        std::map<std::string, std::vector<int64_t>> weight_shapes;
        std::map<std::string, std::vector<uint8_t>> optimizer_state;

        TrainingHistory* history_ptr = options.load_training_history ? &history : nullptr;
        std::map<std::string, std::vector<uint8_t>>* opt_state_ptr =
            options.load_optimizer_state ? &optimizer_state : nullptr;

        bool success = cyxmodel.Extract(
            input_path,
            manifest,
            graph_json,
            config,
            history_ptr,
            weights,
            weight_shapes,
            opt_state_ptr,
            options
        );

        if (!success) {
            result.success = false;
            result.error_message = cyxmodel.GetLastError();
            last_error_ = result.error_message;
            return result;
        }

        if (progress_cb) progress_cb(1, 5, "Building model architecture...");

        // Build model architecture from graph (if available)
        bool model_built = false;
        if (!graph_json.empty()) {
            std::string build_error;
            if (!BuildModelFromGraph(graph_json, model, build_error)) {
                // If graph build fails, log warning but continue
                // (for backward compatibility with models without graphs)
                spdlog::warn("Could not build model from graph: {}", build_error);
                result.warnings.push_back("Could not build model from graph: " + build_error);
            } else {
                model_built = true;
            }
        } else {
            spdlog::warn("No graph.cyxgraph found - model architecture cannot be rebuilt");
            result.warnings.push_back("No graph data - model architecture not available");
        }

        // Fallback: build model from weight shapes if graph compilation failed
        if (!model_built && !weight_shapes.empty()) {
            spdlog::info("Building model from weight shapes (graph unavailable)");

            // Collect Dense layer shapes from weights
            std::vector<std::pair<int, int>> dense_layers;  // (in_features, out_features)
            std::map<int, std::pair<int, int>> layer_shapes;  // layer_idx -> (in, out)

            for (const auto& [name, shape] : weight_shapes) {
                // Parse layer index from name like "layer0.weight" or "layer3.bias"
                if (name.find(".weight") != std::string::npos && shape.size() == 2) {
                    size_t layer_start = name.find("layer");
                    size_t dot_pos = name.find(".");
                    if (layer_start != std::string::npos && dot_pos != std::string::npos) {
                        std::string idx_str = name.substr(layer_start + 5, dot_pos - layer_start - 5);
                        try {
                            int layer_idx = std::stoi(idx_str);
                            int out_features = static_cast<int>(shape[0]);
                            int in_features = static_cast<int>(shape[1]);
                            layer_shapes[layer_idx] = {in_features, out_features};
                        } catch (...) {}
                    }
                }
            }

            // Build model with Dense + ReLU layers
            std::vector<std::pair<int, std::pair<int, int>>> sorted_layers;
            for (const auto& [idx, dims] : layer_shapes) {
                sorted_layers.push_back({idx, dims});
            }

            for (size_t i = 0; i < sorted_layers.size(); ++i) {
                int in_features = sorted_layers[i].second.first;
                int out_features = sorted_layers[i].second.second;

                // Add Linear layer
                model.Add<LinearModule>(in_features, out_features);

                // Add ReLU after all but the last Dense layer
                if (i < sorted_layers.size() - 1) {
                    model.Add<ReLUModule>();
                }
            }

            if (model.Size() > 0) {
                model_built = true;
                spdlog::info("Built model with {} layers from weight shapes", model.Size());

                // Remap weights to match model parameter names
                // Model uses sequential naming: weight, bias for each linear layer
                auto model_params = model.GetParameters();
                std::map<std::string, std::vector<uint8_t>> remapped_weights;
                std::map<std::string, std::vector<int64_t>> remapped_shapes;

                // Map old names to new names by matching shapes
                for (const auto& [model_name, model_tensor] : model_params) {
                    auto model_shape = model_tensor.Shape();

                    // Find matching weight by shape
                    for (const auto& [weight_name, weight_shape] : weight_shapes) {
                        if (weight_shape.size() == model_shape.size()) {
                            bool match = true;
                            for (size_t d = 0; d < weight_shape.size(); ++d) {
                                if (static_cast<size_t>(weight_shape[d]) != model_shape[d]) {
                                    match = false;
                                    break;
                                }
                            }
                            if (match && remapped_weights.find(model_name) == remapped_weights.end()) {
                                // Check if this weight was already used
                                bool already_used = false;
                                for (const auto& [rn, rv] : remapped_weights) {
                                    if (weights.at(weight_name) == rv) {
                                        already_used = true;
                                        break;
                                    }
                                }
                                if (!already_used) {
                                    remapped_weights[model_name] = weights.at(weight_name);
                                    remapped_shapes[model_name] = weight_shape;
                                    spdlog::debug("Remapped weight {} -> {}", weight_name, model_name);
                                    break;
                                }
                            }
                        }
                    }
                }

                // Replace original weights with remapped ones
                weights = std::move(remapped_weights);
                weight_shapes = std::move(remapped_shapes);
            }
        }

        if (progress_cb) progress_cb(2, 5, "Validating model architecture...");

        // Validate model architecture matches weights (skip if no layers built)
        if (options.strict_mode && model.Size() > 0) {
            std::string validation_error;
            if (!ValidateModelArchitecture(model, weight_shapes, validation_error)) {
                result.success = false;
                result.error_message = validation_error;
                last_error_ = validation_error;
                return result;
            }
        }

        if (progress_cb) progress_cb(3, 5, "Loading weights...");

        // Populate model with weights
        if (!PopulateModelWeights(model, weights, weight_shapes, options, result.warnings)) {
            result.success = false;
            result.error_message = last_error_;
            return result;
        }

        if (progress_cb) progress_cb(4, 5, "Finalizing...");

        // Populate result
        result.success = true;
        result.model_name = manifest.model_name;
        result.format_version = manifest.version;
        result.num_parameters = manifest.num_parameters;
        result.num_layers = static_cast<int>(model.Size());  // Use actual model size

        // Collect layer names
        for (const auto& [name, _] : weights) {
            result.layer_names.push_back(name);
        }

        if (progress_cb) progress_cb(5, 5, "Import complete!");

        spdlog::info("Imported model from {} ({} parameters, {} layers)",
                     input_path, result.num_parameters, result.num_layers);

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("Import failed: ") + e.what();
        last_error_ = result.error_message;
        spdlog::error("Model import failed: {}", e.what());
    }

    return result;
}

ImportResult ModelImporter::ImportONNX(
    const std::string& input_path,
    SequentialModel& model,
    const ImportOptions& options,
    ProgressCallback progress_cb
) {
    ImportResult result;

#ifdef CYXWIZ_HAS_ONNX
    // TODO: Implement ONNX import using onnxruntime
    if (progress_cb) progress_cb(0, 1, "ONNX import not yet implemented");
    result.success = false;
    result.error_message = "ONNX import is planned but not yet implemented";
#else
    result.success = false;
    result.error_message = "ONNX support not compiled. Build with CYXWIZ_ENABLE_ONNX=ON";
#endif

    last_error_ = result.error_message;
    return result;
}

ImportResult ModelImporter::ImportSafetensors(
    const std::string& input_path,
    SequentialModel& model,
    const ImportOptions& options,
    ProgressCallback progress_cb
) {
    ImportResult result;

    if (progress_cb) progress_cb(0, 3, "Reading safetensors file...");

    try {
        std::ifstream file(input_path, std::ios::binary);
        if (!file) {
            result.success = false;
            result.error_message = "Failed to open file: " + input_path;
            last_error_ = result.error_message;
            return result;
        }

        // Read header size
        uint64_t header_size = 0;
        file.read(reinterpret_cast<char*>(&header_size), 8);

        // Read header JSON
        std::string header_str(header_size, '\0');
        file.read(&header_str[0], header_size);

        auto header = nlohmann::json::parse(header_str);

        if (progress_cb) progress_cb(1, 3, "Parsing tensor metadata...");

        // Parse tensor metadata
        std::map<std::string, std::vector<uint8_t>> weights;
        std::map<std::string, std::vector<int64_t>> weight_shapes;

        // Calculate data start position
        size_t data_start = 8 + header_size;

        for (auto& [name, info] : header.items()) {
            if (name == "__metadata__") continue;

            // Get shape
            std::vector<int64_t> shape;
            for (const auto& dim : info["shape"]) {
                shape.push_back(dim.get<int64_t>());
            }
            weight_shapes[name] = shape;

            // Get data offsets
            auto offsets = info["data_offsets"];
            size_t start_offset = offsets[0].get<size_t>();
            size_t end_offset = offsets[1].get<size_t>();
            size_t tensor_size = end_offset - start_offset;

            // Read tensor data
            file.seekg(data_start + start_offset);
            std::vector<uint8_t> tensor_data(tensor_size);
            file.read(reinterpret_cast<char*>(tensor_data.data()), tensor_size);

            weights[name] = std::move(tensor_data);

            result.layer_names.push_back(name);
        }

        if (progress_cb) progress_cb(2, 3, "Loading weights into model...");

        // Populate model weights
        if (!PopulateModelWeights(model, weights, weight_shapes, options, result.warnings)) {
            result.success = false;
            result.error_message = last_error_;
            return result;
        }

        // Get metadata if available
        if (header.contains("__metadata__")) {
            auto& meta = header["__metadata__"];
            if (meta.contains("model_name")) {
                result.model_name = meta["model_name"];
            }
        }

        if (progress_cb) progress_cb(3, 3, "Import complete!");

        result.success = true;
        result.format_version = "safetensors";
        result.num_layers = static_cast<int>(weights.size());

        // Count parameters
        for (const auto& [name, shape] : weight_shapes) {
            int64_t params = 1;
            for (int64_t dim : shape) params *= dim;
            result.num_parameters += static_cast<int>(params);
        }

        spdlog::info("Imported model from safetensors: {} ({} tensors)",
                     input_path, weights.size());

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("Safetensors import failed: ") + e.what();
        last_error_ = result.error_message;
        spdlog::error("Safetensors import failed: {}", e.what());
    }

    return result;
}

ImportResult ModelImporter::ImportGGUF(
    const std::string& input_path,
    SequentialModel& model,
    const ImportOptions& options,
    ProgressCallback progress_cb
) {
    ImportResult result;

    if (progress_cb) progress_cb(0, 1, "GGUF import...");

    result.success = false;
    result.error_message = "GGUF import is planned for future release";
    last_error_ = result.error_message;

    return result;
}

std::optional<std::string> ModelImporter::ExtractGraph(const std::string& input_path) {
    if (DetectFormat(input_path) != ModelFormat::CyxModel) {
        last_error_ = "Graph extraction only supported for .cyxmodel files";
        return std::nullopt;
    }

    formats::CyxModelFormat cyxmodel;
    std::string graph_json = cyxmodel.ExtractGraphOnly(input_path);

    if (graph_json.empty()) {
        last_error_ = cyxmodel.GetLastError();
        return std::nullopt;
    }

    return graph_json;
}

std::optional<TrainingHistory> ModelImporter::ExtractHistory(const std::string& input_path) {
    if (DetectFormat(input_path) != ModelFormat::CyxModel) {
        last_error_ = "History extraction only supported for .cyxmodel files";
        return std::nullopt;
    }

    // Use the full Extract method with only history enabled
    formats::CyxModelFormat cyxmodel;

    ModelManifest manifest;
    std::string graph_json;
    TrainingConfig config;
    TrainingHistory history;
    std::map<std::string, std::vector<uint8_t>> weights;
    std::map<std::string, std::vector<int64_t>> weight_shapes;

    ImportOptions options;
    options.load_training_history = true;
    options.load_optimizer_state = false;

    bool success = cyxmodel.Extract(
        input_path,
        manifest,
        graph_json,
        config,
        &history,
        weights,
        weight_shapes,
        nullptr,
        options
    );

    if (!success) {
        last_error_ = cyxmodel.GetLastError();
        return std::nullopt;
    }

    if (history.loss_history.empty() && history.accuracy_history.empty()) {
        last_error_ = "No training history found in model";
        return std::nullopt;
    }

    return history;
}

std::vector<ModelFormat> ModelImporter::GetSupportedFormats() {
    std::vector<ModelFormat> formats = {
        ModelFormat::CyxModel,
        ModelFormat::Safetensors
    };

#ifdef CYXWIZ_HAS_ONNX
    formats.push_back(ModelFormat::ONNX);
#endif

    return formats;
}

bool ModelImporter::IsFormatSupported(ModelFormat format) {
    switch (format) {
        case ModelFormat::CyxModel:
            return true;
        case ModelFormat::Safetensors:
            return true;
        case ModelFormat::ONNX:
#ifdef CYXWIZ_HAS_ONNX
            return true;
#else
            return false;
#endif
        case ModelFormat::GGUF:
            return false;
        default:
            return false;
    }
}

ModelFormat ModelImporter::DetectFormat(const std::string& path) {
    std::filesystem::path p(path);

    // Check if it's a directory (directory-based .cyxmodel)
    if (std::filesystem::is_directory(p)) {
        // Check for manifest.json inside
        if (std::filesystem::exists(p / "manifest.json")) {
            return ModelFormat::CyxModel;
        }
        return ModelFormat::Unknown;
    }

    // Check extension
    std::string ext = p.extension().string();
    ModelFormat format = GetFormatFromExtension(ext);

    if (format != ModelFormat::Unknown) {
        return format;
    }

    // Try magic bytes detection
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return ModelFormat::Unknown;
    }

    char magic[8];
    file.read(magic, 8);

    // GGUF magic: "GGUF"
    if (std::strncmp(magic, "GGUF", 4) == 0) {
        return ModelFormat::GGUF;
    }

    // ZIP magic (for .cyxmodel): PK\x03\x04
    if (magic[0] == 'P' && magic[1] == 'K' && magic[2] == 0x03 && magic[3] == 0x04) {
        return ModelFormat::CyxModel;
    }

    // CYXW magic (binary .cyxmodel): 0x43595857
    if (magic[0] == 'W' && magic[1] == 'X' && magic[2] == 'Y' && magic[3] == 'C') {
        return ModelFormat::CyxModel;  // Binary variant of CyxModel
    }

    return ModelFormat::Unknown;
}

// Check if a file is binary CYXW format
static bool IsCyxwBinaryFormat(const std::string& path) {
    if (std::filesystem::is_directory(path)) return false;

    std::ifstream file(path, std::ios::binary);
    if (!file) return false;

    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    return magic == 0x43595857;  // "CYXW"
}

Tensor ModelImporter::BytesToTensor(
    const std::vector<uint8_t>& data,
    const std::vector<int64_t>& shape,
    TensorDType dtype
) {
    // Convert shape to size_t vector
    std::vector<size_t> tensor_shape;
    tensor_shape.reserve(shape.size());
    for (int64_t dim : shape) {
        tensor_shape.push_back(static_cast<size_t>(dim));
    }

    // Map TensorDType to DataType
    DataType data_type = DataType::Float32;
    switch (dtype) {
        case TensorDType::Float32: data_type = DataType::Float32; break;
        case TensorDType::Float64: data_type = DataType::Float64; break;
        case TensorDType::Int32: data_type = DataType::Int32; break;
        case TensorDType::Int64: data_type = DataType::Int64; break;
        case TensorDType::UInt8: data_type = DataType::UInt8; break;
        default: data_type = DataType::Float32; break;
    }

    // Create tensor with data
    return Tensor(tensor_shape, data.data(), data_type);
}

bool ModelImporter::PopulateModelWeights(
    SequentialModel& model,
    const std::map<std::string, std::vector<uint8_t>>& weights,
    const std::map<std::string, std::vector<int64_t>>& shapes,
    const ImportOptions& options,
    std::vector<std::string>& warnings
) {
    // Get current model parameters
    auto model_params = model.GetParameters();

    // Build weight tensors
    std::map<std::string, Tensor> new_params;

    for (const auto& [name, data] : weights) {
        auto shape_it = shapes.find(name);
        if (shape_it == shapes.end()) {
            last_error_ = "Missing shape for weight: " + name;
            return false;
        }

        // Check if model has this parameter
        auto model_param_it = model_params.find(name);
        if (model_param_it == model_params.end()) {
            if (options.strict_mode) {
                last_error_ = "Model does not have parameter: " + name;
                return false;
            } else {
                warnings.push_back("Skipping unknown parameter: " + name);
                continue;
            }
        }

        // Check shape compatibility
        const auto& expected_shape = model_param_it->second.Shape();
        const auto& loaded_shape = shape_it->second;

        bool shape_match = (expected_shape.size() == loaded_shape.size());
        if (shape_match) {
            for (size_t i = 0; i < expected_shape.size(); ++i) {
                if (expected_shape[i] != static_cast<size_t>(loaded_shape[i])) {
                    shape_match = false;
                    break;
                }
            }
        }

        if (!shape_match && !options.allow_shape_mismatch) {
            std::string msg = "Shape mismatch for " + name + ": expected [";
            for (size_t i = 0; i < expected_shape.size(); ++i) {
                msg += std::to_string(expected_shape[i]);
                if (i < expected_shape.size() - 1) msg += ", ";
            }
            msg += "], got [";
            for (size_t i = 0; i < loaded_shape.size(); ++i) {
                msg += std::to_string(loaded_shape[i]);
                if (i < loaded_shape.size() - 1) msg += ", ";
            }
            msg += "]";

            if (options.strict_mode) {
                last_error_ = msg;
                return false;
            } else {
                warnings.push_back(msg);
                continue;
            }
        }

        // Create tensor (assume float32 for now)
        Tensor tensor = BytesToTensor(data, loaded_shape, TensorDType::Float32);
        new_params[name] = std::move(tensor);
    }

    // Apply weights to model
    model.SetParameters(new_params);

    return true;
}

bool ModelImporter::ValidateModelArchitecture(
    SequentialModel& model,
    const std::map<std::string, std::vector<int64_t>>& weight_shapes,
    std::string& error_message
) {
    auto model_params = model.GetParameters();

    // Check all weights have corresponding model parameters
    for (const auto& [name, shape] : weight_shapes) {
        if (model_params.find(name) == model_params.end()) {
            error_message = "Model missing parameter: " + name;
            return false;
        }
    }

    // Check all model parameters have corresponding weights
    for (const auto& [name, tensor] : model_params) {
        if (weight_shapes.find(name) == weight_shapes.end()) {
            error_message = "Weights missing parameter: " + name;
            return false;
        }
    }

    return true;
}

} // namespace cyxwiz
