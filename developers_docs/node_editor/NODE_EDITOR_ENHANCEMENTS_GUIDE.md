# Node Editor Enhancements - Implementation Guide

## Completed Features

### 1. Properties Panel Integration ✅
- **Status**: Complete
- **Description**: Integrated standalone Properties docking window with Node Editor
- **Files Modified**:
  - `cyxwiz-engine/src/gui/properties.h`
  - `cyxwiz-engine/src/gui/properties.cpp`
  - `cyxwiz-engine/src/gui/node_editor.h`
  - `cyxwiz-engine/src/gui/node_editor.cpp`
  - `cyxwiz-engine/src/gui/main_window.cpp`

### 2. Expanded Editable Parameters ✅
- **Status**: Complete
- **Supported Node Types**:
  - **Input**: Shape specification (height,width,channels)
  - **Dense**: Units, Activation function (none/relu/sigmoid/tanh/softmax/leaky_relu)
  - **Conv2D**: Filters, Kernel Size, Stride, Padding (same/valid), Activation
  - **MaxPool2D**: Pool Size, Stride
  - **Dropout**: Drop Rate (0.0-0.9 slider)
  - **BatchNorm**: Momentum (0.0-1.0), Epsilon (0.0001-0.01)
  - **Output**: Number of classes

## Remaining Features to Implement

### 3. JSON Serialization for Save/Load

**Implementation Steps**:

1. **Add methods to `node_editor.h`**:
```cpp
public:
    // File operations
    bool SaveGraph(const std::string& filepath);
    bool LoadGraph(const std::string& filepath);

private:
    std::string current_file_path_;  // Track current graph file
```

2. **Implement in `node_editor.cpp`** using `nlohmann/json` (already available via vcpkg):
```cpp
#include <nlohmann/json.hpp>
#include <fstream>

bool NodeEditor::SaveGraph(const std::string& filepath) {
    using json = nlohmann::json;

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
}

bool NodeEditor::LoadGraph(const std::string& filepath) {
    using json = nlohmann::json;

    std::ifstream file(filepath);
    if (!file.is_open()) {
        spdlog::error("Failed to open file for reading: {}", filepath);
        return false;
    }

    json j;
    file >> j;

    // Clear existing graph
    ClearGraph();

    // Load framework
    selected_framework_ = static_cast<CodeFramework>(j["framework"].get<int>());

    // Load nodes
    for (const auto& node_json : j["nodes"]) {
        MLNode node;
        node.id = node_json["id"];
        node.type = static_cast<NodeType>(node_json["type"].get<int>());
        node.name = node_json["name"];
        node.parameters = node_json["parameters"].get<std::map<std::string, std::string>>();

        // Recreate pins (same logic as CreateNode)
        node = CreateNode(node.type, node.name);
        node.id = node_json["id"];  // Preserve original ID
        node.parameters = node_json["parameters"].get<std::map<std::string, std::string>>();

        nodes_.push_back(node);

        // Restore node position
        float pos_x = node_json["pos_x"];
        float pos_y = node_json["pos_y"];
        ImNodes::SetNodeGridSpacePos(node.id, ImVec2(pos_x, pos_y));
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
    }

    current_file_path_ = filepath;
    spdlog::info("Graph loaded from: {}", filepath);
    return true;
}
```

3. **Add menu items to toolbar** in `ShowToolbar()`:
```cpp
if (ImGui::Button("Save Graph")) {
    // TODO: Show file save dialog
    SaveGraph("my_model.cyxwiz");
}
ImGui::SameLine();

if (ImGui::Button("Load Graph")) {
    // TODO: Show file open dialog
    LoadGraph("my_model.cyxwiz");
}
ImGui::SameLine();
```

### 4. File Dialog Integration

Use `ImGuiFileDialog` (available via vcpkg) or Windows native dialogs:

```cpp
// In ShowToolbar():
if (ImGui::Button("Save Graph")) {
    #ifdef _WIN32
    // Use Windows file dialog
    OPENFILENAMEA ofn;
    char szFile[260] = {0};
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = "CyxWiz Files\0*.cyxwiz\0All Files\0*.*\0";
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT;
    ofn.lpstrDefExt = "cyxwiz";

    if (GetSaveFileNameA(&ofn)) {
        SaveGraph(szFile);
    }
    #endif
}
```

### 5. Enhanced Graph Validation

**Add to `node_editor.cpp`**:

```cpp
bool NodeEditor::ValidateGraph(std::string& error_message) {
    if (nodes_.empty()) {
        error_message = "Graph is empty. Add nodes first.";
        return false;
    }

    // Check for Input node
    bool has_input = false;
    bool has_output = false;
    for (const auto& node : nodes_) {
        if (node.type == NodeType::Input) has_input = true;
        if (node.type == NodeType::Output) has_output = true;
    }

    if (!has_input) {
        error_message = "Graph must have at least one Input node.";
        return false;
    }

    if (!has_output) {
        error_message = "Graph must have at least one Output node.";
        return false;
    }

    // Check for cycles using DFS
    if (HasCycle()) {
        error_message = "Graph contains cycles. Neural networks must be acyclic.";
        return false;
    }

    // Check that all nodes are reachable from input
    if (!AllNodesReachable()) {
        error_message = "Some nodes are not connected to the network.";
        return false;
    }

    return true;
}

bool NodeEditor::HasCycle() {
    // Build adjacency list
    std::map<int, std::vector<int>> adj;
    for (const auto& link : links_) {
        adj[link.from_node].push_back(link.to_node);
    }

    // DFS with visited/recursion stack
    std::set<int> visited, rec_stack;

    std::function<bool(int)> dfs = [&](int node_id) -> bool {
        visited.insert(node_id);
        rec_stack.insert(node_id);

        for (int neighbor : adj[node_id]) {
            if (!visited.count(neighbor)) {
                if (dfs(neighbor)) return true;
            } else if (rec_stack.count(neighbor)) {
                return true;  // Cycle detected
            }
        }

        rec_stack.erase(node_id);
        return false;
    };

    for (const auto& node : nodes_) {
        if (!visited.count(node.id)) {
            if (dfs(node.id)) return true;
        }
    }

    return false;
}

bool NodeEditor::AllNodesReachable() {
    // Find all Input nodes
    std::vector<int> input_nodes;
    for (const auto& node : nodes_) {
        if (node.type == NodeType::Input) {
            input_nodes.push_back(node.id);
        }
    }

    if (input_nodes.empty()) return false;

    // Build adjacency list
    std::map<int, std::vector<int>> adj;
    for (const auto& link : links_) {
        adj[link.from_node].push_back(link.to_node);
    }

    // BFS from all input nodes
    std::set<int> reachable;
    std::queue<int> q;

    for (int input_id : input_nodes) {
        q.push(input_id);
        reachable.insert(input_id);
    }

    while (!q.empty()) {
        int current = q.front();
        q.pop();

        for (int neighbor : adj[current]) {
            if (!reachable.count(neighbor)) {
                reachable.insert(neighbor);
                q.push(neighbor);
            }
        }
    }

    // Check if all nodes are reachable
    return reachable.size() == nodes_.size();
}
```

### 6. Code Export to File

```cpp
void NodeEditor::ExportCodeToFile() {
    // Generate code first
    std::string error_msg;
    if (!ValidateGraph(error_msg)) {
        spdlog::error("Cannot export: {}", error_msg);
        // TODO: Show error dialog
        return;
    }

    auto sorted_ids = TopologicalSort();
    std::string code;

    switch (selected_framework_) {
        case CodeFramework::PyTorch:
            code = GeneratePyTorchCode(sorted_ids);
            break;
        case CodeFramework::TensorFlow:
            code = GenerateTensorFlowCode(sorted_ids);
            break;
        case CodeFramework::Keras:
            code = GenerateKerasCode(sorted_ids);
            break;
        case CodeFramework::PyCyxWiz:
            code = GeneratePyCyxWizCode(sorted_ids);
            break;
    }

    // Show file save dialog
    #ifdef _WIN32
    OPENFILENAMEA ofn;
    char szFile[260] = {0};
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = "Python Files\0*.py\0All Files\0*.*\0";
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT;
    ofn.lpstrDefExt = "py";

    if (GetSaveFileNameA(&ofn)) {
        std::ofstream file(szFile);
        if (file.is_open()) {
            file << code;
            spdlog::info("Code exported to: {}", szFile);
        }
    }
    #endif
}
```

### 7. Additional Node Types

Add to `NodeType` enum in `node_editor.h`:

```cpp
enum class NodeType {
    // ... existing types ...

    // Recurrent layers
    LSTM,
    GRU,

    // Pooling
    AvgPool2D,
    GlobalAvgPool2D,
    GlobalMaxPool2D,

    // Normalization
    LayerNorm,
    GroupNorm,
    InstanceNorm,

    // Advanced activations
    ELU,
    SELU,
    Swish,
    GELU
};
```

Then implement in `CreateNode()` and add corresponding code generation logic.

### 8. Visual Improvements

**Better Color Scheme** - Update `GetNodeColor()`:
```cpp
ImU32 NodeEditor::GetNodeColor(NodeType type) {
    switch (type) {
        case NodeType::Input:
            return IM_COL32(52, 152, 219, 255);  // Blue
        case NodeType::Output:
            return IM_COL32(46, 204, 113, 255);  // Green

        case NodeType::Dense:
            return IM_COL32(155, 89, 182, 255);  // Purple
        case NodeType::LSTM:
        case NodeType::GRU:
            return IM_COL32(142, 68, 173, 255);  // Dark Purple

        case NodeType::Conv2D:
            return IM_COL32(52, 152, 219, 255);  // Blue
        case NodeType::MaxPool2D:
        case NodeType::AvgPool2D:
            return IM_COL32(41, 128, 185, 255);  // Dark Blue

        case NodeType::ReLU:
        case NodeType::Sigmoid:
        case NodeType::Tanh:
            return IM_COL32(230, 126, 34, 255);  // Orange

        case NodeType::Dropout:
        case NodeType::BatchNorm:
        case NodeType::LayerNorm:
            return IM_COL32(231, 76, 60, 255);  // Red

        case NodeType::Flatten:
            return IM_COL32(149, 165, 166, 255);  // Gray

        default:
            return IM_COL32(127, 140, 141, 255);  // Default Gray
    }
}
```

**Display Parameters on Nodes** - Update `RenderNodes()`:
```cpp
for (auto& node : nodes_) {
    // ... existing code ...

    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted(node.name.c_str());
    ImNodes::EndNodeTitleBar();

    // Display key parameters on the node
    if (node.type == NodeType::Dense && !node.parameters["units"].empty()) {
        ImGui::Text("Units: %s", node.parameters["units"].c_str());
    }
    else if (node.type == NodeType::Conv2D) {
        if (!node.parameters["filters"].empty()) {
            ImGui::Text("Filters: %s", node.parameters["filters"].c_str());
        }
    }
    else if (node.type == NodeType::Dropout && !node.parameters["rate"].empty()) {
        ImGui::Text("Rate: %s", node.parameters["rate"].c_str());
    }

    // ... rest of node rendering ...
}
```

## Testing Checklist

- [ ] Test expanded parameters in Properties panel
- [ ] Save graph with various node configurations
- [ ] Load saved graph and verify all parameters preserved
- [ ] Test cycle detection with circular connections
- [ ] Test validation with disconnected nodes
- [ ] Export code to file for all frameworks
- [ ] Verify parameter display on nodes
- [ ] Test new node types (LSTM, GRU, etc.)

## Build Instructions

```bash
cd build/windows-release
cmake --build . --config Release --target cyxwiz-engine -j 8
```

## Future Enhancements

1. **Undo/Redo** - Implement command pattern for node operations
2. **Node Search** - Quick search to find nodes by name
3. **Minimap** - Overview of large graphs
4. **Node Groups** - Collapse multiple nodes into reusable blocks
5. **Auto-Layout** - Automatic graph organization
6. **Template Library** - Pre-built network architectures (ResNet, VGG, etc.)
7. **Training Integration** - Direct connection to training backend
8. **Real-time Visualization** - Show activation maps during training
