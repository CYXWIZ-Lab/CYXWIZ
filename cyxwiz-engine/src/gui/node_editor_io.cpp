// File I/O Module for Node Editor
// This module contains all file operations for the visual node editor:
// - Graph save/load functionality (JSON serialization)
// - Cross-platform native file dialogs
// - Code export functionality (PyTorch, TensorFlow, Keras, PyCyxWiz)

#include "node_editor.h"
#include "../core/file_dialogs.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <spdlog/spdlog.h>

namespace gui {

// ========== Pattern to Graph Conversion ==========

// Helper: Convert string type name to NodeType enum
static NodeType StringToNodeType(const std::string& type_str) {
    static const std::unordered_map<std::string, NodeType> type_map = {
        // Data pipeline
        {"Input", NodeType::DatasetInput},      // Pattern uses "Input" for data input
        {"DataInput", NodeType::DatasetInput},
        {"DatasetInput", NodeType::DatasetInput},
        {"Output", NodeType::Output},

        // Core layers
        {"Dense", NodeType::Dense},
        {"Conv1D", NodeType::Conv1D},
        {"Conv2D", NodeType::Conv2D},
        {"Conv3D", NodeType::Conv3D},
        {"DepthwiseConv2D", NodeType::DepthwiseConv2D},

        // Pooling
        {"MaxPool2D", NodeType::MaxPool2D},
        {"AvgPool2D", NodeType::AvgPool2D},
        {"GlobalMaxPool", NodeType::GlobalMaxPool},
        {"GlobalAvgPool", NodeType::GlobalAvgPool},
        {"GlobalAvgPool2D", NodeType::GlobalAvgPool},  // Alias
        {"AdaptiveAvgPool", NodeType::AdaptiveAvgPool},

        // Normalization
        {"BatchNorm", NodeType::BatchNorm},
        {"BatchNorm2D", NodeType::BatchNorm},  // Alias
        {"LayerNorm", NodeType::LayerNorm},
        {"GroupNorm", NodeType::GroupNorm},
        {"InstanceNorm", NodeType::InstanceNorm},

        // Regularization
        {"Dropout", NodeType::Dropout},
        {"Flatten", NodeType::Flatten},

        // Recurrent
        {"RNN", NodeType::RNN},
        {"LSTM", NodeType::LSTM},
        {"GRU", NodeType::GRU},
        {"Bidirectional", NodeType::Bidirectional},
        {"Embedding", NodeType::Embedding},

        // Attention & Transformer
        {"MultiHeadAttention", NodeType::MultiHeadAttention},
        {"SelfAttention", NodeType::SelfAttention},
        {"CrossAttention", NodeType::CrossAttention},
        {"LinearAttention", NodeType::LinearAttention},
        {"TransformerEncoder", NodeType::TransformerEncoder},
        {"TransformerDecoder", NodeType::TransformerDecoder},
        {"PositionalEncoding", NodeType::PositionalEncoding},

        // Activations
        {"ReLU", NodeType::ReLU},
        {"LeakyReLU", NodeType::LeakyReLU},
        {"PReLU", NodeType::PReLU},
        {"ELU", NodeType::ELU},
        {"SELU", NodeType::SELU},
        {"GELU", NodeType::GELU},
        {"Swish", NodeType::Swish},
        {"Mish", NodeType::Mish},
        {"Sigmoid", NodeType::Sigmoid},
        {"Tanh", NodeType::Tanh},
        {"Softmax", NodeType::Softmax},

        // Shape operations
        {"Reshape", NodeType::Reshape},
        {"Permute", NodeType::Permute},
        {"Squeeze", NodeType::Squeeze},
        {"Unsqueeze", NodeType::Unsqueeze},
        {"View", NodeType::View},
        {"Split", NodeType::Split},

        // Merge operations
        {"Concatenate", NodeType::Concatenate},
        {"Concat", NodeType::Concatenate},  // Alias
        {"Add", NodeType::Add},
        {"Multiply", NodeType::Multiply},
        {"Average", NodeType::Average},

        // Loss functions
        {"MSELoss", NodeType::MSELoss},
        {"CrossEntropyLoss", NodeType::CrossEntropyLoss},
        {"CrossEntropy", NodeType::CrossEntropyLoss},  // Alias
        {"BCELoss", NodeType::BCELoss},
        {"BCEWithLogits", NodeType::BCEWithLogits},
        {"L1Loss", NodeType::L1Loss},
        {"SmoothL1Loss", NodeType::SmoothL1Loss},
        {"HuberLoss", NodeType::HuberLoss},
        {"NLLLoss", NodeType::NLLLoss},

        // Optimizers
        {"SGD", NodeType::SGD},
        {"Adam", NodeType::Adam},
        {"AdamW", NodeType::AdamW},
        {"RMSprop", NodeType::RMSprop},
        {"Adagrad", NodeType::Adagrad},
        {"NAdam", NodeType::NAdam},

        // Data Pipeline / Preprocessing
        {"Normalize", NodeType::Normalize},
        {"OneHotEncode", NodeType::OneHotEncode},
        {"DataLoader", NodeType::DataLoader},
        {"DataSplit", NodeType::DataSplit},
        {"Augmentation", NodeType::Augmentation},
        {"TensorReshape", NodeType::TensorReshape}
    };

    auto it = type_map.find(type_str);
    if (it != type_map.end()) {
        return it->second;
    }
    spdlog::warn("Unknown node type '{}', defaulting to Dense", type_str);
    return NodeType::Dense;
}

bool NodeEditor::LoadPatternAsGraph(const nlohmann::json& j) {
    using json = nlohmann::json;

    // Clear existing graph
    ClearGraph();

    const auto& tmpl = j["template"];

    // Map string IDs to integer IDs
    std::unordered_map<std::string, int> id_map;
    int next_id = 1;

    // Load nodes from template
    if (tmpl.contains("nodes") && tmpl["nodes"].is_array()) {
        for (const auto& node_json : tmpl["nodes"]) {
            std::string str_id = node_json.value("id", "");
            std::string type_str = node_json.value("type", "Dense");
            std::string name = node_json.value("name", type_str);

            // Convert string type to NodeType enum
            NodeType node_type = StringToNodeType(type_str);

            // Create node with proper pins
            MLNode node = CreateNode(node_type, name);
            node.id = next_id;
            id_map[str_id] = next_id;
            next_id++;

            // Parse position (two formats supported: pos_x/pos_y or pos: [x, y])
            float pos_x = node_json.value("pos_x", 0.0f);
            float pos_y = node_json.value("pos_y", 0.0f);

            if (node_json.contains("pos") && node_json["pos"].is_array() && node_json["pos"].size() >= 2) {
                pos_x = node_json["pos"][0].get<float>();
                pos_y = node_json["pos"][1].get<float>();
            }

            // Apply any node parameters (substitute pattern parameters with defaults)
            if (node_json.contains("params") && node_json["params"].is_object()) {
                for (auto& [key, value] : node_json["params"].items()) {
                    std::string param_value;
                    if (value.is_string()) {
                        param_value = value.get<std::string>();
                        // Handle pattern parameter references like "$hidden1_size"
                        if (!param_value.empty() && param_value[0] == '$') {
                            // Find the default value for this parameter
                            std::string param_name = param_value.substr(1);
                            if (j.contains("parameters") && j["parameters"].is_array()) {
                                for (const auto& p : j["parameters"]) {
                                    if (p.value("name", "") == param_name) {
                                        param_value = p.value("default_value", param_value);
                                        break;
                                    }
                                }
                            }
                        }
                    } else if (value.is_number_integer()) {
                        param_value = std::to_string(value.get<int>());
                    } else if (value.is_number_float()) {
                        param_value = std::to_string(value.get<float>());
                    } else if (value.is_boolean()) {
                        param_value = value.get<bool>() ? "true" : "false";
                    }
                    node.parameters[key] = param_value;
                }
            }

            nodes_.push_back(node);

            // Queue position
            pending_positions_[node.id] = ImVec2(pos_x, pos_y);
        }
    }

    pending_positions_frames_ = 3;

    // Load links from template
    int link_id = 1;
    if (tmpl.contains("links") && tmpl["links"].is_array()) {
        for (const auto& link_json : tmpl["links"]) {
            std::string from_str = link_json.value("from", "");
            std::string to_str = link_json.value("to", "");

            auto from_it = id_map.find(from_str);
            auto to_it = id_map.find(to_str);

            if (from_it == id_map.end() || to_it == id_map.end()) {
                spdlog::warn("Link references unknown node: {} -> {}", from_str, to_str);
                continue;
            }

            int from_node_id = from_it->second;
            int to_node_id = to_it->second;

            const MLNode* from_node = FindNodeById(from_node_id);
            const MLNode* to_node = FindNodeById(to_node_id);

            if (!from_node || !to_node) {
                spdlog::warn("Could not find nodes for link: {} -> {}", from_str, to_str);
                continue;
            }

            // Get pin indices (default to first pin)
            int from_pin_idx = link_json.value("from_pin", 0);
            int to_pin_idx = link_json.value("to_pin", 0);

            // Create link using actual pin IDs
            NodeLink link;
            link.id = link_id++;

            if (from_pin_idx < static_cast<int>(from_node->outputs.size())) {
                link.from_pin = from_node->outputs[from_pin_idx].id;
            } else if (!from_node->outputs.empty()) {
                link.from_pin = from_node->outputs[0].id;
            } else {
                spdlog::warn("Node {} has no output pins", from_str);
                continue;
            }

            if (to_pin_idx < static_cast<int>(to_node->inputs.size())) {
                link.to_pin = to_node->inputs[to_pin_idx].id;
            } else if (!to_node->inputs.empty()) {
                link.to_pin = to_node->inputs[0].id;
            } else {
                spdlog::warn("Node {} has no input pins", to_str);
                continue;
            }

            link.from_node = from_node_id;
            link.to_node = to_node_id;
            link.type = LinkType::TensorFlow;

            links_.push_back(link);
        }
    }

    // Update next IDs
    next_node_id_ = next_id;
    next_link_id_ = link_id;

    std::string name = j.value("name", "Imported Pattern");
    spdlog::info("Loaded pattern '{}' as graph ({} nodes, {} links)",
                 name, nodes_.size(), links_.size());

    return true;
}

// ========== Graph Save/Load Implementation ==========

bool NodeEditor::SaveGraph(const std::string& filepath) {
    using json = nlohmann::json;

    try {
        json j;
        j["version"] = "1.0";
        j["framework"] = static_cast<int>(selected_framework_);

        // Serialize nodes
        json nodes_array = json::array();
        for (const auto& node : nodes_) {
            json node_json;
            node_json["id"] = node.id;
            node_json["type"] = static_cast<int>(node.type);
            node_json["name"] = node.name;
            node_json["parameters"] = node.parameters;

            // Save node position
            auto it = cached_node_positions_.find(node.id);
            ImVec2 pos = (it != cached_node_positions_.end()) ? it->second : ImVec2(0,0);
            node_json["pos_x"] = pos.x;
            node_json["pos_y"] = pos.y;

            nodes_array.push_back(node_json);
        }
        j["nodes"] = nodes_array;

        // Serialize links with pin indices for multi-pin support
        json links_array = json::array();
        for (const auto& link : links_) {
            json link_json;
            link_json["id"] = link.id;
            link_json["from_node"] = link.from_node;
            link_json["from_pin"] = link.from_pin;
            link_json["to_node"] = link.to_node;
            link_json["to_pin"] = link.to_pin;

            // Save pin indices for proper multi-pin node support
            const MLNode* from_node = FindNodeById(link.from_node);
            const MLNode* to_node = FindNodeById(link.to_node);

            int from_pin_index = 0;
            if (from_node) {
                for (size_t i = 0; i < from_node->outputs.size(); ++i) {
                    if (from_node->outputs[i].id == link.from_pin) {
                        from_pin_index = static_cast<int>(i);
                        break;
                    }
                }
            }

            int to_pin_index = 0;
            if (to_node) {
                for (size_t i = 0; i < to_node->inputs.size(); ++i) {
                    if (to_node->inputs[i].id == link.to_pin) {
                        to_pin_index = static_cast<int>(i);
                        break;
                    }
                }
            }

            link_json["from_pin_index"] = from_pin_index;
            link_json["to_pin_index"] = to_pin_index;

            // Save link type for skip connection visualization
            link_json["link_type"] = static_cast<int>(link.type);

            links_array.push_back(link_json);
        }
        j["links"] = links_array;

        // Write to file
        std::ofstream file(filepath);
        if (!file.is_open()) {
            spdlog::error("Failed to open file for writing: {}", filepath);
            return false;
        }

        file << j.dump(4);  // Pretty print with 4-space indent
        current_file_path_ = filepath;
        spdlog::info("Graph saved to: {}", filepath);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Error saving graph: {}", e.what());
        return false;
    }
}

bool NodeEditor::LoadGraph(const std::string& filepath) {
    using json = nlohmann::json;

    try {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            spdlog::error("Failed to open file for reading: {}", filepath);
            return false;
        }

        json j;
        file >> j;

        // Check if this is a pattern template format (has "template" key with nodes inside)
        if (j.contains("template") && j["template"].is_object() &&
            j["template"].contains("nodes")) {
            spdlog::info("Detected pattern template format, converting to graph format");
            return LoadPatternAsGraph(j);
        }

        // Clear existing graph
        ClearGraph();

        // Update next IDs to avoid conflicts
        int max_node_id = 0;
        int max_link_id = 0;

        // Load framework
        if (j.contains("framework")) {
            selected_framework_ = static_cast<CodeFramework>(j["framework"].get<int>());
        }

        // Load nodes
        for (const auto& node_json : j["nodes"]) {
            MLNode node;
            node.id = node_json["id"];
            node.type = static_cast<NodeType>(node_json["type"].get<int>());
            node.name = node_json["name"];

            if (node_json.contains("parameters")) {
                node.parameters = node_json["parameters"].get<std::map<std::string, std::string>>();
            }

            // Recreate pins based on node type using fresh pin IDs
            // Create node with fresh pin IDs
            MLNode template_node = CreateNode(node.type, node.name);
            node.inputs = template_node.inputs;
            node.outputs = template_node.outputs;

            // Update max IDs
            max_node_id = std::max(max_node_id, node.id);

            nodes_.push_back(node);

            // Queue position restore for next render frame (must be inside ImNodes scope)
            if (node_json.contains("pos_x") && node_json.contains("pos_y")) {
                float pos_x = node_json["pos_x"];
                float pos_y = node_json["pos_y"];
                pending_positions_[node.id] = ImVec2(pos_x, pos_y);
            }
        }

        // Need to apply positions for multiple frames because ImNodes needs the node
        // to exist before SetNodeGridSpacePos takes effect
        pending_positions_frames_ = 3;  // Apply for 3 frames to ensure positions stick

        // Build helper to find actual pin ID from file pin ID
        // File format: pins are assigned sequentially per node in order
        // Node 0: input pin 0, output pin 1
        // Node 1: input pin 2, output pin 3
        // etc.
        auto findActualPinId = [this](int file_pin_id, int node_id, bool is_from_pin) -> int {
            // Find the node
            const MLNode* node = FindNodeById(node_id);
            if (!node) return file_pin_id;  // Fallback

            // For from_pin, it's an output pin; for to_pin, it's an input pin
            const auto& pins = is_from_pin ? node->outputs : node->inputs;

            // Calculate expected pin offset: each node before this one contributes pins
            // But since we can't know the original pin assignment, use a simple heuristic:
            // Just use the first pin of the appropriate type
            if (!pins.empty()) {
                // If file_pin_id matches what we'd expect for this node's first output/input, use it
                return pins[0].id;
            }
            return file_pin_id;
        };

        // Load links with pin index support for multi-pin nodes
        for (const auto& link_json : j["links"]) {
            NodeLink link;
            link.id = link_json["id"];
            link.from_node = link_json["from_node"];
            link.to_node = link_json["to_node"];

            // Find actual pin IDs from loaded nodes using pin indices
            const MLNode* from_node = FindNodeById(link.from_node);
            const MLNode* to_node = FindNodeById(link.to_node);

            // Use pin indices if available (new format), otherwise fall back to first pin
            int from_pin_index = 0;
            int to_pin_index = 0;

            if (link_json.contains("from_pin_index")) {
                from_pin_index = link_json["from_pin_index"].get<int>();
            }
            if (link_json.contains("to_pin_index")) {
                to_pin_index = link_json["to_pin_index"].get<int>();
            }

            // Get the actual pin ID using the index
            if (from_node && from_pin_index < static_cast<int>(from_node->outputs.size())) {
                link.from_pin = from_node->outputs[from_pin_index].id;
            } else if (from_node && !from_node->outputs.empty()) {
                link.from_pin = from_node->outputs[0].id;  // Fallback to first
            } else {
                link.from_pin = link_json["from_pin"];  // Legacy fallback
            }

            if (to_node && to_pin_index < static_cast<int>(to_node->inputs.size())) {
                link.to_pin = to_node->inputs[to_pin_index].id;
            } else if (to_node && !to_node->inputs.empty()) {
                link.to_pin = to_node->inputs[0].id;  // Fallback to first
            } else {
                link.to_pin = link_json["to_pin"];  // Legacy fallback
            }

            // Load link type for skip connection visualization
            if (link_json.contains("link_type")) {
                link.type = static_cast<LinkType>(link_json["link_type"].get<int>());
            } else {
                link.type = LinkType::TensorFlow;  // Default for legacy files
            }

            links_.push_back(link);
            max_link_id = std::max(max_link_id, link.id);
        }

        // Update next IDs
        next_node_id_ = max_node_id + 1;
        next_link_id_ = max_link_id + 1;

        current_file_path_ = filepath;
        spdlog::info("Graph loaded from: {} ({} nodes, {} links)",
                     filepath, nodes_.size(), links_.size());
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Error loading graph: {}", e.what());
        return false;
    }
}

std::string NodeEditor::GetGraphJson() const {
    using json = nlohmann::json;

    try {
        json j;
        j["version"] = "1.0";
        j["framework"] = static_cast<int>(selected_framework_);

        // Serialize nodes
        json nodes_array = json::array();
        for (const auto& node : nodes_) {
            json node_json;
            node_json["id"] = node.id;
            node_json["type"] = static_cast<int>(node.type);
            node_json["name"] = node.name;
            node_json["parameters"] = node.parameters;

            // Save node position
            auto it = cached_node_positions_.find(node.id);
            ImVec2 pos = (it != cached_node_positions_.end()) ? it->second : ImVec2(0,0);
            node_json["pos_x"] = pos.x;
            node_json["pos_y"] = pos.y;

            nodes_array.push_back(node_json);
        }
        j["nodes"] = nodes_array;

        // Serialize links with pin indices for multi-pin support
        json links_array = json::array();
        for (const auto& link : links_) {
            json link_json;
            link_json["id"] = link.id;
            link_json["from_node"] = link.from_node;
            link_json["from_pin"] = link.from_pin;
            link_json["to_node"] = link.to_node;
            link_json["to_pin"] = link.to_pin;

            // Save pin indices for proper multi-pin node support
            const MLNode* from_node = FindNodeById(link.from_node);
            const MLNode* to_node = FindNodeById(link.to_node);

            int from_pin_index = 0;
            if (from_node) {
                for (size_t i = 0; i < from_node->outputs.size(); ++i) {
                    if (from_node->outputs[i].id == link.from_pin) {
                        from_pin_index = static_cast<int>(i);
                        break;
                    }
                }
            }

            int to_pin_index = 0;
            if (to_node) {
                for (size_t i = 0; i < to_node->inputs.size(); ++i) {
                    if (to_node->inputs[i].id == link.to_pin) {
                        to_pin_index = static_cast<int>(i);
                        break;
                    }
                }
            }

            link_json["from_pin_index"] = from_pin_index;
            link_json["to_pin_index"] = to_pin_index;

            // Save link type for skip connection visualization
            link_json["link_type"] = static_cast<int>(link.type);

            links_array.push_back(link_json);
        }
        j["links"] = links_array;

        return j.dump(4);  // Pretty print with 4-space indent

    } catch (const std::exception& e) {
        spdlog::error("Error serializing graph: {}", e.what());
        return "";
    }
}

bool NodeEditor::LoadGraphFromString(const std::string& json_string) {
    using json = nlohmann::json;

    if (json_string.empty()) {
        spdlog::error("Cannot load graph from empty JSON string");
        return false;
    }

    try {
        json j = json::parse(json_string);

        // Clear existing graph
        ClearGraph();

        // Update next IDs to avoid conflicts
        int max_node_id = 0;
        int max_link_id = 0;

        // Load framework
        if (j.contains("framework")) {
            selected_framework_ = static_cast<CodeFramework>(j["framework"].get<int>());
        }

        // Load nodes
        for (const auto& node_json : j["nodes"]) {
            MLNode node;
            node.id = node_json["id"];
            node.type = static_cast<NodeType>(node_json["type"].get<int>());
            node.name = node_json["name"];

            if (node_json.contains("parameters")) {
                node.parameters = node_json["parameters"].get<std::map<std::string, std::string>>();
            }

            // Recreate pins based on node type using fresh pin IDs
            MLNode template_node = CreateNode(node.type, node.name);
            node.inputs = template_node.inputs;
            node.outputs = template_node.outputs;

            // Update max IDs
            max_node_id = std::max(max_node_id, node.id);

            nodes_.push_back(node);

            // Queue position restore for next render frame
            if (node_json.contains("pos_x") && node_json.contains("pos_y")) {
                float pos_x = node_json["pos_x"];
                float pos_y = node_json["pos_y"];
                pending_positions_[node.id] = ImVec2(pos_x, pos_y);
            }
        }

        // Need to apply positions for multiple frames
        pending_positions_frames_ = 3;

        // Load links with pin index support
        for (const auto& link_json : j["links"]) {
            NodeLink link;
            link.id = link_json["id"];
            link.from_node = link_json["from_node"];
            link.to_node = link_json["to_node"];

            // Find actual pin IDs from loaded nodes using pin indices
            const MLNode* from_node = FindNodeById(link.from_node);
            const MLNode* to_node = FindNodeById(link.to_node);

            // Use pin indices if available
            int from_pin_index = 0;
            int to_pin_index = 0;

            if (link_json.contains("from_pin_index")) {
                from_pin_index = link_json["from_pin_index"].get<int>();
            }
            if (link_json.contains("to_pin_index")) {
                to_pin_index = link_json["to_pin_index"].get<int>();
            }

            // Get the actual pin ID using the index
            if (from_node && from_pin_index < static_cast<int>(from_node->outputs.size())) {
                link.from_pin = from_node->outputs[from_pin_index].id;
            } else if (from_node && !from_node->outputs.empty()) {
                link.from_pin = from_node->outputs[0].id;
            } else {
                link.from_pin = link_json["from_pin"];
            }

            if (to_node && to_pin_index < static_cast<int>(to_node->inputs.size())) {
                link.to_pin = to_node->inputs[to_pin_index].id;
            } else if (to_node && !to_node->inputs.empty()) {
                link.to_pin = to_node->inputs[0].id;
            } else {
                link.to_pin = link_json["to_pin"];
            }

            // Load link type
            if (link_json.contains("link_type")) {
                link.type = static_cast<LinkType>(link_json["link_type"].get<int>());
            } else {
                link.type = LinkType::TensorFlow;
            }

            links_.push_back(link);
            max_link_id = std::max(max_link_id, link.id);
        }

        // Update next IDs
        next_node_id_ = max_node_id + 1;
        next_link_id_ = max_link_id + 1;

        spdlog::info("Graph loaded from JSON string ({} nodes, {} links)",
                     nodes_.size(), links_.size());
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Error loading graph from JSON string: {}", e.what());
        return false;
    }
}

// ========== Cross-Platform File Dialogs ==========

void NodeEditor::ShowSaveDialog() {
    auto result = cyxwiz::FileDialogs::SaveGraph();
    if (result) {
        if (SaveGraph(*result)) {
            spdlog::info("Graph successfully saved");
        }
    }
}

void NodeEditor::ShowLoadDialog() {
    auto result = cyxwiz::FileDialogs::OpenGraph();
    if (result) {
        if (LoadGraph(*result)) {
            spdlog::info("Graph successfully loaded");
        }
    }
}

// ========== Code Export Implementation ==========

void NodeEditor::ExportCodeToFile() {
    // Validate graph first
    std::string error_message;
    if (!ValidateGraph(error_message)) {
        spdlog::error("Cannot export code: {}", error_message);
        // TODO: Show error dialog to user
        return;
    }

    // Generate code
    auto sorted_ids = TopologicalSort();
    if (sorted_ids.empty()) {
        spdlog::error("Failed to sort graph for code generation");
        return;
    }

    std::string code;
    std::string extension = ".py";
    std::string framework_name;

    switch (selected_framework_) {
        case CodeFramework::PyTorch:
            code = GeneratePyTorchCode(sorted_ids);
            framework_name = "PyTorch";
            break;
        case CodeFramework::TensorFlow:
            code = GenerateTensorFlowCode(sorted_ids);
            framework_name = "TensorFlow";
            break;
        case CodeFramework::Keras:
            code = GenerateKerasCode(sorted_ids);
            framework_name = "Keras";
            break;
        case CodeFramework::PyCyxWiz:
            code = GeneratePyCyxWizCode(sorted_ids);
            framework_name = "PyCyxWiz";
            break;
    }

    // Build the code with header and footer
    std::string header = "# Neural Network Model Generated by CyxWiz\n";
    header += "# Framework: " + framework_name + "\n";
    header += "# Generated on: " + std::string(__DATE__) + " " + std::string(__TIME__) + "\n\n";

    std::string full_code = header + code;

    // Save to file - will be called from ShowExportDialog
    return;
}

void NodeEditor::ShowExportDialog() {
    // Validate graph first
    std::string error_message;
    if (!ValidateGraph(error_message)) {
        spdlog::error("Cannot export code: {}", error_message);
        return;
    }

    // Generate code
    auto sorted_ids = TopologicalSort();
    if (sorted_ids.empty()) {
        spdlog::error("Failed to sort graph for code generation");
        return;
    }

    std::string code;
    std::string framework_name;

    switch (selected_framework_) {
        case CodeFramework::PyTorch:
            code = GeneratePyTorchCode(sorted_ids);
            framework_name = "PyTorch";
            break;
        case CodeFramework::TensorFlow:
            code = GenerateTensorFlowCode(sorted_ids);
            framework_name = "TensorFlow";
            break;
        case CodeFramework::Keras:
            code = GenerateKerasCode(sorted_ids);
            framework_name = "Keras";
            break;
        case CodeFramework::PyCyxWiz:
            code = GeneratePyCyxWizCode(sorted_ids);
            framework_name = "PyCyxWiz";
            break;
    }

    // Build the code with header
    std::string header = "# Neural Network Model Generated by CyxWiz\n";
    header += "# Framework: " + framework_name + "\n";
    header += "# Generated on: " + std::string(__DATE__) + " " + std::string(__TIME__) + "\n\n";

    std::string full_code = header + code;

    // Default filename based on framework
    std::string default_name = "model_" + framework_name + ".py";

    // Show cross-platform save dialog
    auto result = cyxwiz::FileDialogs::SaveScript();
    if (result) {
        std::ofstream file(*result);
        if (file.is_open()) {
            file << full_code;
            file.close();
            spdlog::info("Code exported successfully to: {}", *result);
        } else {
            spdlog::error("Failed to open file for writing: {}", *result);
        }
    }
}

} // namespace gui
