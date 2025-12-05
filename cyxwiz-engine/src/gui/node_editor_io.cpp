// File I/O Module for Node Editor
// This module contains all file operations for the visual node editor:
// - Graph save/load functionality (JSON serialization)
// - Platform-specific file dialogs (Windows native, cross-platform fallback)
// - Code export functionality (PyTorch, TensorFlow, Keras, PyCyxWiz)

#include "node_editor.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <spdlog/spdlog.h>

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#endif

namespace gui {

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

// ========== Platform-Specific File Dialogs ==========

#ifdef _WIN32
void NodeEditor::ShowSaveDialog() {
    char szFile[260] = {0};

    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = "CyxWiz Graph Files\0*.cyxgraph\0All Files\0*.*\0";
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT;
    ofn.lpstrDefExt = "cyxwiz";
    ofn.lpstrTitle = "Save Neural Network Graph";

    if (GetSaveFileNameA(&ofn)) {
        if (SaveGraph(szFile)) {
            spdlog::info("Graph successfully saved");
        }
    }
}

void NodeEditor::ShowLoadDialog() {
    char szFile[260] = {0};

    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = "CyxWiz Graph Files\0*.cyxgraph\0All Files\0*.*\0";
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
    ofn.lpstrTitle = "Load Neural Network Graph";

    if (GetOpenFileNameA(&ofn)) {
        if (LoadGraph(szFile)) {
            spdlog::info("Graph successfully loaded");
        }
    }
}
#endif // _WIN32

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

#ifdef _WIN32
void NodeEditor::ShowExportDialog() {
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

    // Show Windows file save dialog
    char szFile[260] = {0};

    // Suggest a default filename based on framework
    std::string default_name = "model_" + framework_name + ".py";
    strncpy(szFile, default_name.c_str(), sizeof(szFile) - 1);

    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = "Python Files\0*.py\0All Files\0*.*\0";
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT;
    ofn.lpstrDefExt = "py";
    ofn.lpstrTitle = "Export Neural Network Code";

    if (GetSaveFileNameA(&ofn)) {
        // Save code to file
        std::ofstream file(szFile);
        if (file.is_open()) {
            file << full_code;
            file.close();
            spdlog::info("Code exported successfully to: {}", szFile);
        } else {
            spdlog::error("Failed to open file for writing: {}", szFile);
        }
    }
}
#else
// For non-Windows platforms, implement later or use a cross-platform dialog library
void NodeEditor::ShowSaveDialog() {
    SaveGraph("model.cyxgraph");
}

void NodeEditor::ShowLoadDialog() {
    LoadGraph("model.cyxgraph");
}

void NodeEditor::ShowExportDialog() {
    // Export to default filename for now
    std::string filename = "model_export.py";

    // Generate and save code
    ExportCodeToFile();

    // Validate graph first
    std::string error_message;
    if (!ValidateGraph(error_message)) {
        spdlog::error("Cannot export code: {}", error_message);
        return;
    }

    auto sorted_ids = TopologicalSort();
    if (sorted_ids.empty()) {
        return;
    }

    std::string code;
    std::string framework_name;

    switch (selected_framework_) {
        case CodeFramework::PyTorch:
            code = GeneratePyTorchCode(sorted_ids);
            framework_name = "PyTorch";
            filename = "model_pytorch.py";
            break;
        case CodeFramework::TensorFlow:
            code = GenerateTensorFlowCode(sorted_ids);
            framework_name = "TensorFlow";
            filename = "model_tensorflow.py";
            break;
        case CodeFramework::Keras:
            code = GenerateKerasCode(sorted_ids);
            framework_name = "Keras";
            filename = "model_keras.py";
            break;
        case CodeFramework::PyCyxWiz:
            code = GeneratePyCyxWizCode(sorted_ids);
            framework_name = "PyCyxWiz";
            filename = "model_pycyxwiz.py";
            break;
    }

    std::string header = "# Neural Network Model Generated by CyxWiz\n";
    header += "# Framework: " + framework_name + "\n\n";
    std::string full_code = header + code;

    std::ofstream file(filename);
    if (file.is_open()) {
        file << full_code;
        file.close();
        spdlog::info("Code exported to: {}", filename);
    }
}
#endif

} // namespace gui
