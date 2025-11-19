# Save/Load Implementation - Copy-Paste Ready Code

## Files to Modify

### 1. cyxwiz-engine/src/gui/node_editor.cpp

Add these includes at the top of the file (after existing includes):

```cpp
#include <nlohmann/json.hpp>
#include <fstream>
#ifdef _WIN32
#include <windows.h>  // For file dialogs
#include <commdlg.h>
#endif
```

Add these implementations at the end of the file (before the closing `} // namespace gui`):

```cpp
// ========== Save/Load Implementation ==========

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
            ImVec2 pos = ImNodes::GetNodeGridSpacePos(node.id);
            node_json["pos_x"] = pos.x;
            node_json["pos_y"] = pos.y;

            nodes_array.push_back(node_json);
        }
        j["nodes"] = nodes_array;

        // Serialize links
        json links_array = json::array();
        for (const auto& link : links_) {
            json link_json;
            link_json["id"] = link.id;
            link_json["from_node"] = link.from_node;
            link_json["from_pin"] = link.from_pin;
            link_json["to_node"] = link.to_node;
            link_json["to_pin"] = link.to_pin;
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
        int max_pin_id = 0;
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

            // Recreate pins based on node type
            MLNode template_node = CreateNode(node.type, node.name);
            node.inputs = template_node.inputs;
            node.outputs = template_node.outputs;

            // Update max IDs
            max_node_id = std::max(max_node_id, node.id);
            for (const auto& pin : node.inputs) {
                max_pin_id = std::max(max_pin_id, pin.id);
            }
            for (const auto& pin : node.outputs) {
                max_pin_id = std::max(max_pin_id, pin.id);
            }

            nodes_.push_back(node);

            // Restore node position
            if (node_json.contains("pos_x") && node_json.contains("pos_y")) {
                float pos_x = node_json["pos_x"];
                float pos_y = node_json["pos_y"];
                ImNodes::SetNodeGridSpacePos(node.id, ImVec2(pos_x, pos_y));
            }
        }

        // Load links
        for (const auto& link_json : j["links"]) {
            NodeLink link;
            link.id = link_json["id"];
            link.from_node = link_json["from_node"];
            link.from_pin = link_json["from_pin"];
            link.to_node = link_json["to_node"];
            link.to_pin = link_json["to_pin"];
            links_.push_back(link);

            max_link_id = std::max(max_link_id, link.id);
        }

        // Update next IDs
        next_node_id_ = max_node_id + 1;
        next_pin_id_ = max_pin_id + 1;
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

#ifdef _WIN32
void NodeEditor::ShowSaveDialog() {
    char szFile[260] = {0};

    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = "CyxWiz Graph Files\0*.cyxwiz\0All Files\0*.*\0";
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
    ofn.lpstrFilter = "CyxWiz Graph Files\0*.cyxwiz\0All Files\0*.*\0";
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
#else
// For non-Windows platforms, implement later or use a cross-platform dialog library
void NodeEditor::ShowSaveDialog() {
    SaveGraph("model.cyxwiz");
}

void NodeEditor::ShowLoadDialog() {
    LoadGraph("model.cyxwiz");
}
#endif
```

### 2. Update ShowToolbar() in node_editor.cpp

Find the `ShowToolbar()` function and update it to add Save/Load buttons:

```cpp
void NodeEditor::ShowToolbar() {
    // File operations
    if (ImGui::Button("Save Graph")) {
        ShowSaveDialog();
    }
    ImGui::SameLine();

    if (ImGui::Button("Load Graph")) {
        ShowLoadDialog();
    }
    ImGui::SameLine();

    ImGui::Text("|");
    ImGui::SameLine();

    // Existing buttons...
    if (ImGui::Button("Add Dense Layer")) {
        AddNode(NodeType::Dense, "Dense Layer");
    }
    ImGui::SameLine();

    // ... rest of toolbar ...
}
```

## Testing the Implementation

1. **Build the project**:
   ```bash
   cd build/windows-release
   cmake --build . --config Release --target cyxwiz-engine -j 8
   ```

2. **Test Save**:
   - Run CyxWiz Engine
   - Click "Save Graph" button
   - Choose a location and filename
   - File should be saved with `.cyxwiz` extension

3. **Test Load**:
   - Click "Load Graph" button
   - Select a previously saved `.cyxwiz` file
   - Graph should be restored with all nodes, connections, and parameters

4. **Verify**:
   - Check that node positions are preserved
   - Verify all parameters are retained
   - Ensure connections are maintained
   - Test with different parameter values

## File Format Example

A saved `.cyxwiz` file looks like this:

```json
{
    "version": "1.0",
    "framework": 0,
    "nodes": [
        {
            "id": 1,
            "type": 0,
            "name": "Input Layer",
            "parameters": {
                "shape": "28,28,1"
            },
            "pos_x": 100.0,
            "pos_y": 100.0
        },
        {
            "id": 2,
            "type": 1,
            "name": "Dense (128)",
            "parameters": {
                "units": "128",
                "activation": "relu"
            },
            "pos_x": 300.0,
            "pos_y": 100.0
        }
    ],
    "links": [
        {
            "id": 1,
            "from_node": 1,
            "from_pin": 2,
            "to_node": 2,
            "to_pin": 3
        }
    ]
}
```

## Next Steps

After implementing Save/Load:

1. Add keyboard shortcuts (Ctrl+S for Save, Ctrl+O for Open)
2. Implement "Save As" functionality
3. Add recent files list
4. Implement auto-save functionality
5. Add dirty flag to track unsaved changes
