#include "model_converter.h"
#include "model_format.h"
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <fstream>
#include <cstring>

namespace cyxwiz {

namespace fs = std::filesystem;
using json = nlohmann::json;

std::string ModelConverter::last_error_;

// Binary format constants
constexpr uint32_t CYXW_MAGIC = 0x43595857;  // "CYXW" in little-endian
constexpr uint32_t CYXW_VERSION = 2;

bool ModelConverter::IsBinaryFormat(const std::string& path) {
    if (!fs::exists(path)) return false;
    if (fs::is_directory(path)) return false;

    std::ifstream file(path, std::ios::binary);
    if (!file) return false;

    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    return magic == CYXW_MAGIC;
}

bool ModelConverter::IsDirectoryFormat(const std::string& path) {
    if (!fs::exists(path)) return false;
    if (!fs::is_directory(path)) return false;

    // Check for manifest.json
    return fs::exists(fs::path(path) / "manifest.json");
}

bool ModelConverter::BinaryToDirectory(
    const std::string& input_path,
    const std::string& output_path,
    ProgressCallback progress_cb
) {
    try {
        if (progress_cb) progress_cb(0, 6, "Opening binary file...");

        // Open binary file
        std::ifstream file(input_path, std::ios::binary);
        if (!file) {
            last_error_ = "Failed to open file: " + input_path;
            return false;
        }

        // Read and verify magic
        uint32_t magic;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != CYXW_MAGIC) {
            last_error_ = "Invalid magic number - not a CyxWiz binary model";
            return false;
        }

        // Read version
        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != CYXW_VERSION) {
            last_error_ = "Unsupported format version: " + std::to_string(version);
            return false;
        }

        if (progress_cb) progress_cb(1, 6, "Reading metadata...");

        // Read JSON metadata
        uint64_t json_len;
        file.read(reinterpret_cast<char*>(&json_len), sizeof(json_len));

        std::string json_str(json_len, '\0');
        file.read(json_str.data(), json_len);

        json meta = json::parse(json_str);

        // Read number of modules
        size_t num_modules;
        file.read(reinterpret_cast<char*>(&num_modules), sizeof(num_modules));

        // Debug: Print metadata modules
        if (meta.contains("modules")) {
            spdlog::info("Metadata has {} modules:", meta["modules"].size());
            for (const auto& mod : meta["modules"]) {
                spdlog::info("  - index={}, name={}, has_params={}",
                    mod.value("index", -1),
                    mod.value("name", "unknown"),
                    mod.value("has_parameters", false));
            }
        }

        if (progress_cb) progress_cb(2, 6, "Creating output directory...");

        // Create output directory
        fs::path out_dir(output_path);
        fs::create_directories(out_dir);
        fs::create_directories(out_dir / "weights");

        if (progress_cb) progress_cb(3, 6, "Writing manifest.json...");

        // Create manifest.json
        json manifest;
        manifest["version"] = "1.0";
        manifest["format"] = "cyxmodel";
        manifest["cyxwiz_version"] = meta["metadata"].value("framework", "CyxWiz") + " " +
                                      meta["metadata"].value("format_version", "2.0");

        // Get current timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
        manifest["created"] = ss.str();

        // Model info
        manifest["model"]["name"] = meta["metadata"].value("name", "Converted Model");
        manifest["model"]["type"] = "SequentialModel";
        manifest["model"]["num_layers"] = num_modules;

        // Count parameters
        int total_params = 0;
        if (meta.contains("modules")) {
            for (const auto& mod : meta["modules"]) {
                if (mod.contains("parameters")) {
                    for (const auto& param : mod["parameters"]) {
                        if (param.contains("shape")) {
                            int param_count = 1;
                            for (const auto& dim : param["shape"]) {
                                param_count *= dim.get<int>();
                            }
                            total_params += param_count;
                        }
                    }
                }
            }
        }
        manifest["model"]["num_parameters"] = total_params;

        // Metadata
        manifest["metadata"]["author"] = "";
        manifest["metadata"]["description"] = meta["metadata"].value("description", "");
        manifest["metadata"]["custom"] = json::object();

        // Content flags - we'll generate a graph from metadata
        manifest["content"]["has_graph"] = meta.contains("modules") && !meta["modules"].empty();
        manifest["content"]["has_optimizer_state"] = false;
        manifest["content"]["has_training_history"] = false;

        // Write manifest
        std::ofstream manifest_file(out_dir / "manifest.json");
        manifest_file << manifest.dump(4);
        manifest_file.close();

        // Generate graph.cyxgraph from binary metadata
        if (meta.contains("modules") && !meta["modules"].empty()) {
            json graph;
            graph["version"] = "1.0";
            graph["framework"] = 0;
            graph["nodes"] = json::array();
            graph["links"] = json::array();

            // Map module names to node types
            // Module names may include details like "Linear(784 -> 512)" or "Dropout(p=0.2)"
            // so we use starts_with matching instead of exact match
            auto getNodeType = [](const std::string& name) -> int {
                if (name.find("Linear") == 0) return 0;       // Dense
                if (name.find("ReLU") == 0) return 29;
                if (name.find("Sigmoid") == 0) return 36;
                if (name.find("Tanh") == 0) return 40;
                if (name.find("Softmax") == 0) return 39;
                if (name.find("Dropout") == 0) return 14;
                if (name.find("BatchNorm") == 0) return 8;
                if (name.find("LayerNorm") == 0) return 17;
                if (name.find("Flatten") == 0) return 16;
                if (name.find("Conv2d") == 0 || name.find("Conv2D") == 0) return 1;
                if (name.find("MaxPool") == 0) return 2;
                if (name.find("AvgPool") == 0) return 3;
                spdlog::warn("Unknown module type '{}', defaulting to Dense", name);
                return 0;  // Default to Dense
            };

            int node_id = 1;
            int pin_id = 1;
            int link_id = 1;
            float pos_x = 0.0f;
            int prev_node_id = -1;
            int prev_pin_id = -1;

            // Add Input node first
            json input_node;
            input_node["id"] = node_id;
            input_node["name"] = "Input";
            input_node["type"] = 77;  // DataInput type
            input_node["pos_x"] = pos_x;
            input_node["pos_y"] = 200.0f;
            input_node["parameters"] = json::object();

            // Try to infer input shape from first Linear layer
            for (const auto& mod : meta["modules"]) {
                std::string name = mod.value("name", "");
                if (name.find("Linear") == 0 && mod.contains("parameters")) {
                    for (const auto& p : mod["parameters"]) {
                        if (p["name"] == "weight" && p.contains("shape")) {
                            auto shape = p["shape"];
                            if (shape.size() >= 2) {
                                int in_features = shape[1].get<int>();
                                input_node["parameters"]["shape"] = "[" + std::to_string(in_features) + "]";
                                spdlog::info("    Inferred input shape: [{}]", in_features);
                                break;
                            }
                        }
                    }
                    break;
                }
            }

            graph["nodes"].push_back(input_node);
            prev_node_id = node_id;
            prev_pin_id = pin_id + 1;  // Output pin
            node_id++;
            pin_id += 2;
            pos_x += 200.0f;

            // Add nodes for each module
            for (const auto& mod : meta["modules"]) {
                std::string mod_name = mod["name"];
                int node_type = getNodeType(mod_name);

                json node;
                node["id"] = node_id;
                node["name"] = mod_name;
                node["type"] = node_type;
                node["pos_x"] = pos_x;
                node["pos_y"] = 200.0f;
                node["parameters"] = json::object();

                // Set parameters based on layer type (use starts_with matching)
                if (mod_name.find("Linear") == 0 && mod.contains("parameters")) {
                    for (const auto& p : mod["parameters"]) {
                        if (p["name"] == "weight" && p.contains("shape")) {
                            auto shape = p["shape"];
                            if (shape.size() >= 2) {
                                int out_features = shape[0].get<int>();
                                node["parameters"]["units"] = std::to_string(out_features);
                                spdlog::info("    Set units={} for Linear layer", out_features);
                            }
                        }
                    }
                } else if (mod_name.find("Dropout") == 0) {
                    // Try to parse dropout rate from name like "Dropout(p=0.200000)"
                    size_t p_pos = mod_name.find("p=");
                    if (p_pos != std::string::npos) {
                        size_t end_pos = mod_name.find(")", p_pos);
                        if (end_pos != std::string::npos) {
                            std::string rate_str = mod_name.substr(p_pos + 2, end_pos - p_pos - 2);
                            node["parameters"]["rate"] = rate_str;
                        }
                    } else if (mod.contains("dropout_rate")) {
                        node["parameters"]["rate"] = mod["dropout_rate"];
                    }
                }

                graph["nodes"].push_back(node);

                // Create link from previous node
                if (prev_node_id > 0) {
                    json link;
                    link["id"] = link_id++;
                    link["from_node"] = prev_node_id;
                    link["from_pin"] = prev_pin_id;
                    link["from_pin_index"] = 0;
                    link["to_node"] = node_id;
                    link["to_pin"] = pin_id;
                    link["to_pin_index"] = 0;
                    link["link_type"] = 0;
                    graph["links"].push_back(link);
                }

                prev_node_id = node_id;
                prev_pin_id = pin_id + 1;  // Output pin
                node_id++;
                pin_id += 2;
                pos_x += 200.0f;
            }

            // Add Softmax node before output (for classification)
            json softmax_node;
            softmax_node["id"] = node_id;
            softmax_node["name"] = "Softmax";
            softmax_node["type"] = 39;  // Softmax type
            softmax_node["pos_x"] = pos_x;
            softmax_node["pos_y"] = 100.0f;
            softmax_node["parameters"] = json::object();
            graph["nodes"].push_back(softmax_node);

            int softmax_node_id = node_id;
            int softmax_in_pin = pin_id;
            int softmax_out_pin = pin_id + 1;

            // Link last layer to Softmax
            json softmax_link;
            softmax_link["id"] = link_id++;
            softmax_link["from_node"] = prev_node_id;
            softmax_link["from_pin"] = prev_pin_id;
            softmax_link["from_pin_index"] = 0;
            softmax_link["to_node"] = softmax_node_id;
            softmax_link["to_pin"] = softmax_in_pin;
            softmax_link["to_pin_index"] = 0;
            softmax_link["link_type"] = 0;
            graph["links"].push_back(softmax_link);

            node_id++;
            pin_id += 2;
            pos_x += 200.0f;

            // Add CrossEntropyLoss node (required for graph compilation)
            json loss_node;
            loss_node["id"] = node_id;
            loss_node["name"] = "CrossEntropy Loss";
            loss_node["type"] = 52;  // CrossEntropyLoss type
            loss_node["pos_x"] = pos_x;
            loss_node["pos_y"] = 300.0f;
            loss_node["parameters"] = json::object();
            loss_node["parameters"]["reduction"] = "mean";
            graph["nodes"].push_back(loss_node);

            int loss_node_id = node_id;
            int loss_pred_pin = pin_id;
            int loss_target_pin = pin_id + 1;
            int loss_out_pin = pin_id + 2;

            // Link last Dense layer to Loss (prediction input)
            json loss_link;
            loss_link["id"] = link_id++;
            loss_link["from_node"] = prev_node_id;
            loss_link["from_pin"] = prev_pin_id;
            loss_link["from_pin_index"] = 0;
            loss_link["to_node"] = loss_node_id;
            loss_link["to_pin"] = loss_pred_pin;
            loss_link["to_pin_index"] = 0;
            loss_link["link_type"] = 0;
            graph["links"].push_back(loss_link);

            // Link Input node to Loss (target input) - use second output pin from Input
            json target_link;
            target_link["id"] = link_id++;
            target_link["from_node"] = 1;  // Input node
            target_link["from_pin"] = 2;   // Second output pin (labels)
            target_link["from_pin_index"] = 1;
            target_link["to_node"] = loss_node_id;
            target_link["to_pin"] = loss_target_pin;
            target_link["to_pin_index"] = 1;
            target_link["link_type"] = 0;
            graph["links"].push_back(target_link);

            node_id++;
            pin_id += 3;
            pos_x += 200.0f;

            // Add Adam optimizer node
            json optimizer_node;
            optimizer_node["id"] = node_id;
            optimizer_node["name"] = "Adam";
            optimizer_node["type"] = 60;  // Adam type
            optimizer_node["pos_x"] = pos_x;
            optimizer_node["pos_y"] = 300.0f;
            optimizer_node["parameters"] = json::object();
            optimizer_node["parameters"]["learning_rate"] = "0.001";
            optimizer_node["parameters"]["beta1"] = "0.9";
            optimizer_node["parameters"]["beta2"] = "0.999";
            optimizer_node["parameters"]["epsilon"] = "1e-8";
            graph["nodes"].push_back(optimizer_node);

            // Link Loss to Optimizer
            json opt_link;
            opt_link["id"] = link_id++;
            opt_link["from_node"] = loss_node_id;
            opt_link["from_pin"] = loss_out_pin;
            opt_link["from_pin_index"] = 0;
            opt_link["to_node"] = node_id;
            opt_link["to_pin"] = pin_id;
            opt_link["to_pin_index"] = 0;
            opt_link["link_type"] = 0;
            graph["links"].push_back(opt_link);

            node_id++;
            pin_id += 2;
            pos_x += 200.0f;

            // Add Output node
            json output_node;
            output_node["id"] = node_id;
            output_node["name"] = "Output";
            output_node["type"] = 50;  // Output type
            output_node["pos_x"] = pos_x;
            output_node["pos_y"] = 100.0f;
            output_node["parameters"] = json::object();
            graph["nodes"].push_back(output_node);

            // Link Softmax to Output
            json final_link;
            final_link["id"] = link_id;
            final_link["from_node"] = softmax_node_id;
            final_link["from_pin"] = softmax_out_pin;
            final_link["from_pin_index"] = 0;
            final_link["to_node"] = node_id;
            final_link["to_pin"] = pin_id;
            final_link["to_pin_index"] = 0;
            final_link["link_type"] = 0;
            graph["links"].push_back(final_link);

            // Write graph.cyxgraph
            std::ofstream graph_file(out_dir / "graph.cyxgraph");
            graph_file << graph.dump(4);
            graph_file.close();

            spdlog::info("Generated graph.cyxgraph with {} nodes from binary metadata",
                         graph["nodes"].size());
        }

        if (progress_cb) progress_cb(4, 6, "Extracting weights...");

        // Create weights manifest
        json weights_manifest;
        weights_manifest["version"] = "1.0";
        weights_manifest["tensors"] = json::array();
        size_t total_bytes = 0;

        // Read and write each module's parameters
        // Layer indices must match module indices (including non-trainable like ReLU)
        // because SequentialModel::GetParameters() uses module indices
        spdlog::info("BinaryToDirectory: Processing {} modules", num_modules);
        for (size_t i = 0; i < num_modules; ++i) {
            size_t num_params;
            file.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));

            // Use module index directly - this matches SequentialModel naming
            int layer_idx = static_cast<int>(i);
            spdlog::info("  Module {}: {} params", i, num_params);

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
                int dtype;
                file.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));

                // Read data
                size_t num_bytes;
                file.read(reinterpret_cast<char*>(&num_bytes), sizeof(num_bytes));
                std::vector<uint8_t> data(num_bytes);
                file.read(reinterpret_cast<char*>(data.data()), num_bytes);

                // Create tensor name matching GraphCompiler convention: layer0.weight, layer0.bias
                std::string tensor_name = "layer" + std::to_string(layer_idx) + "." + name;
                spdlog::info("    Created tensor: {}", tensor_name);
                std::string filename = tensor_name;
                for (auto& c : filename) {
                    if (c == '.' || c == '/') c = '_';
                }
                filename += ".bin";

                // Write tensor file with header
                std::ofstream tensor_file(out_dir / "weights" / filename, std::ios::binary);

                // Write header: ndims, shape, dtype
                uint32_t header_ndims = static_cast<uint32_t>(ndims);
                tensor_file.write(reinterpret_cast<const char*>(&header_ndims), sizeof(header_ndims));

                for (size_t dim : shape) {
                    int64_t dim64 = static_cast<int64_t>(dim);
                    tensor_file.write(reinterpret_cast<const char*>(&dim64), sizeof(dim64));
                }

                uint32_t header_dtype = static_cast<uint32_t>(dtype);
                tensor_file.write(reinterpret_cast<const char*>(&header_dtype), sizeof(header_dtype));

                // Write data
                tensor_file.write(reinterpret_cast<const char*>(data.data()), data.size());
                tensor_file.close();

                // Add to weights manifest
                json tensor_meta;
                tensor_meta["name"] = tensor_name;
                std::vector<int64_t> shape64;
                for (auto s : shape) shape64.push_back(static_cast<int64_t>(s));
                tensor_meta["shape"] = shape64;
                tensor_meta["dtype"] = dtype;
                tensor_meta["size_bytes"] = num_bytes;
                weights_manifest["tensors"].push_back(tensor_meta);

                total_bytes += num_bytes;
            }
        }

        weights_manifest["total_tensors"] = weights_manifest["tensors"].size();
        weights_manifest["total_bytes"] = total_bytes;

        // Write weights manifest
        std::ofstream weights_manifest_file(out_dir / "weights" / "manifest.json");
        weights_manifest_file << weights_manifest.dump(4);
        weights_manifest_file.close();

        if (progress_cb) progress_cb(5, 6, "Writing config.json...");

        // Create empty config.json (binary format doesn't store training config)
        json config;
        config["optimizer_type"] = "";
        config["learning_rate"] = 0.001;
        config["batch_size"] = 32;
        config["epochs"] = 0;

        std::ofstream config_file(out_dir / "config.json");
        config_file << config.dump(4);
        config_file.close();

        if (progress_cb) progress_cb(6, 6, "Conversion complete!");

        spdlog::info("ModelConverter: Converted binary to directory: {} -> {}",
                     input_path, output_path);
        return true;

    } catch (const std::exception& e) {
        last_error_ = std::string("Conversion failed: ") + e.what();
        spdlog::error("ModelConverter::BinaryToDirectory: {}", last_error_);
        return false;
    }
}

bool ModelConverter::DirectoryToBinary(
    const std::string& input_path,
    const std::string& output_path,
    ProgressCallback progress_cb
) {
    try {
        if (progress_cb) progress_cb(0, 5, "Reading manifest...");

        fs::path in_dir(input_path);

        // Read manifest
        std::ifstream manifest_file(in_dir / "manifest.json");
        if (!manifest_file) {
            last_error_ = "Missing manifest.json in " + input_path;
            return false;
        }
        json manifest = json::parse(manifest_file);
        manifest_file.close();

        // Read weights manifest
        std::ifstream weights_manifest_file(in_dir / "weights" / "manifest.json");
        if (!weights_manifest_file) {
            last_error_ = "Missing weights/manifest.json in " + input_path;
            return false;
        }
        json weights_manifest = json::parse(weights_manifest_file);
        weights_manifest_file.close();

        if (progress_cb) progress_cb(1, 5, "Building metadata...");

        // Build binary format metadata
        json meta;
        meta["metadata"]["name"] = manifest["model"].value("name", "Model");
        meta["metadata"]["description"] = manifest["metadata"].value("description", "");
        meta["metadata"]["created_at"] = manifest.value("created", "");
        meta["metadata"]["framework"] = "CyxWiz";
        meta["metadata"]["format_version"] = "2.0";

        // Build modules array from weights
        meta["modules"] = json::array();

        // Group tensors by layer
        std::map<int, std::vector<json>> layer_params;
        int fallback_layer_idx = 0;
        for (const auto& tensor : weights_manifest["tensors"]) {
            std::string name = tensor["name"];
            int layer_idx = fallback_layer_idx;

            // Try to parse "layer_X.param_name" format
            if (name.substr(0, 6) == "layer_") {
                size_t dot_pos = name.find('.');
                if (dot_pos != std::string::npos && dot_pos > 6) {
                    std::string layer_str = name.substr(6, dot_pos - 6);
                    try {
                        layer_idx = std::stoi(layer_str);
                    } catch (...) {
                        // Use fallback index
                        layer_idx = fallback_layer_idx++;
                    }
                } else {
                    layer_idx = fallback_layer_idx++;
                }
            } else {
                // Try other naming conventions like "linear_0.weight"
                size_t underscore_pos = name.find('_');
                size_t dot_pos = name.find('.');
                if (underscore_pos != std::string::npos && dot_pos != std::string::npos &&
                    underscore_pos < dot_pos) {
                    std::string idx_str = name.substr(underscore_pos + 1, dot_pos - underscore_pos - 1);
                    try {
                        layer_idx = std::stoi(idx_str);
                    } catch (...) {
                        layer_idx = fallback_layer_idx++;
                    }
                } else {
                    layer_idx = fallback_layer_idx++;
                }
            }

            layer_params[layer_idx].push_back(tensor);
        }

        for (const auto& [layer_idx, params] : layer_params) {
            json module_info;
            module_info["index"] = layer_idx;
            module_info["name"] = "Layer " + std::to_string(layer_idx);
            module_info["has_parameters"] = !params.empty();
            module_info["trainable"] = true;

            json param_list = json::array();
            for (const auto& p : params) {
                json param_info;
                std::string full_name = p["name"];
                size_t dot_pos = full_name.find('.');
                param_info["name"] = (dot_pos != std::string::npos) ?
                    full_name.substr(dot_pos + 1) : full_name;
                param_info["shape"] = p["shape"];
                param_info["dtype"] = "float32";
                param_list.push_back(param_info);
            }
            module_info["parameters"] = param_list;
            meta["modules"].push_back(module_info);
        }

        if (progress_cb) progress_cb(2, 5, "Creating output file...");

        // Open output file
        std::ofstream file(output_path, std::ios::binary);
        if (!file) {
            last_error_ = "Failed to create output file: " + output_path;
            return false;
        }

        // Write header
        file.write(reinterpret_cast<const char*>(&CYXW_MAGIC), sizeof(CYXW_MAGIC));
        file.write(reinterpret_cast<const char*>(&CYXW_VERSION), sizeof(CYXW_VERSION));

        // Write JSON
        std::string json_str = meta.dump();
        uint64_t json_len = json_str.size();
        file.write(reinterpret_cast<const char*>(&json_len), sizeof(json_len));
        file.write(json_str.c_str(), json_len);

        if (progress_cb) progress_cb(3, 5, "Writing module count...");

        // Write number of modules
        size_t num_modules = layer_params.size();
        file.write(reinterpret_cast<const char*>(&num_modules), sizeof(num_modules));

        if (progress_cb) progress_cb(4, 5, "Writing weights...");

        // Write each module's parameters
        for (const auto& [layer_idx, params] : layer_params) {
            size_t num_params = params.size();
            file.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));

            for (const auto& param : params) {
                // Get param name (without layer prefix)
                std::string full_name = param["name"];
                size_t dot_pos = full_name.find('.');
                std::string param_name = (dot_pos != std::string::npos) ?
                    full_name.substr(dot_pos + 1) : full_name;

                // Write parameter name
                size_t name_len = param_name.size();
                file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
                file.write(param_name.c_str(), name_len);

                // Read tensor file
                std::string tensor_filename = full_name;
                for (auto& c : tensor_filename) {
                    if (c == '.' || c == '/') c = '_';
                }
                tensor_filename += ".bin";

                std::ifstream tensor_file(in_dir / "weights" / tensor_filename, std::ios::binary);
                if (!tensor_file) {
                    last_error_ = "Missing tensor file: " + tensor_filename;
                    return false;
                }

                // Read tensor header
                uint32_t ndims;
                tensor_file.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));

                std::vector<size_t> shape(ndims);
                for (size_t i = 0; i < ndims; ++i) {
                    int64_t dim;
                    tensor_file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
                    shape[i] = static_cast<size_t>(dim);
                }

                uint32_t dtype;
                tensor_file.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));

                // Calculate data size
                size_t num_elements = 1;
                for (size_t s : shape) num_elements *= s;
                size_t num_bytes = num_elements * 4;  // Assume float32

                // Read tensor data
                std::vector<uint8_t> data(num_bytes);
                tensor_file.read(reinterpret_cast<char*>(data.data()), num_bytes);
                tensor_file.close();

                // Write tensor in binary format
                // Shape
                size_t shape_ndims = shape.size();
                file.write(reinterpret_cast<const char*>(&shape_ndims), sizeof(shape_ndims));
                file.write(reinterpret_cast<const char*>(shape.data()), shape_ndims * sizeof(size_t));

                // Dtype
                int dtype_int = static_cast<int>(dtype);
                file.write(reinterpret_cast<const char*>(&dtype_int), sizeof(dtype_int));

                // Data
                file.write(reinterpret_cast<const char*>(&num_bytes), sizeof(num_bytes));
                file.write(reinterpret_cast<const char*>(data.data()), num_bytes);
            }
        }

        file.close();

        if (progress_cb) progress_cb(5, 5, "Conversion complete!");

        spdlog::info("ModelConverter: Converted directory to binary: {} -> {}",
                     input_path, output_path);
        return true;

    } catch (const std::exception& e) {
        last_error_ = std::string("Conversion failed: ") + e.what();
        spdlog::error("ModelConverter::DirectoryToBinary: {}", last_error_);
        return false;
    }
}

} // namespace cyxwiz
