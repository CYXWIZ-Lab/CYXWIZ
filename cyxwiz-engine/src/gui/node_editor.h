#pragma once

#include <vector>
#include <string>
#include <map>
#include <imgui.h>

// Forward declarations from ImNodes
struct ImNodesEditorContext;

// Forward declarations
namespace cyxwiz {
class ScriptEditorPanel;
}

namespace gui {
class Properties;


// Node types for ML model building
enum class NodeType {
    Input,
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNorm,
    // Activation functions
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    LeakyReLU,
    // Output
    Output,
    // Loss functions
    MSELoss,
    CrossEntropyLoss,
    // Optimizers
    SGD,
    Adam,
    AdamW
};

// Attribute for node pins (inputs/outputs)
enum class PinType {
    Tensor,      // General tensor data
    Parameters,  // Model parameters
    Loss,        // Loss value
    Optimizer    // Optimizer state
};

// Node pin structure
struct NodePin {
    int id;
    PinType type;
    std::string name;
    bool is_input;  // true = input pin, false = output pin
};

// Visual node structure
struct MLNode {
    int id;
    NodeType type;
    std::string name;
    std::vector<NodePin> inputs;
    std::vector<NodePin> outputs;

    // Node-specific parameters (e.g., units for Dense layer)
    std::map<std::string, std::string> parameters;
};

// Connection between nodes
struct NodeLink {
    int id;
    int from_node;
    int from_pin;
    int to_node;
    int to_pin;
};

// Supported code generation frameworks
enum class CodeFramework {
    PyTorch,
    TensorFlow,
    Keras,
    PyCyxWiz
};

class NodeEditor {
public:
    NodeEditor();
    ~NodeEditor();

    void Render();

    // Set script editor for code output
    void SetScriptEditor(cyxwiz::ScriptEditorPanel* editor) { script_editor_ = editor; }

    // Set properties panel for node selection display
    void SetPropertiesPanel(Properties* properties) { properties_panel_ = properties; }

private:
    void ShowToolbar();
    void RenderNodes();
    void HandleInteractions();
    void ShowContextMenu();

    // Helper functions
    unsigned int GetNodeColor(NodeType type);

    // Node management
    void AddNode(NodeType type, const std::string& name);
    MLNode CreateNode(NodeType type, const std::string& name);
    void DeleteNode(int node_id);
    void ClearGraph();

    // Link management
    void CreateLink(int from_pin, int to_pin, int from_node, int to_node);

    // File operations
    bool SaveGraph(const std::string& filepath);
    bool LoadGraph(const std::string& filepath);
    void ShowSaveDialog();
    void ShowLoadDialog();

    // Code generation
    void GeneratePythonCode();
    void GenerateCodeForFramework(CodeFramework framework);
    bool ValidateGraph(std::string& error_message);

    // Graph validation helpers
    bool HasCycle();
    bool AllNodesReachable();
    bool HasInputNode();
    bool HasOutputNode();

    // Framework-specific generators
    std::string GeneratePyTorchCode(const std::vector<int>& sorted_ids);
    std::string GenerateTensorFlowCode(const std::vector<int>& sorted_ids);
    std::string GenerateKerasCode(const std::vector<int>& sorted_ids);
    std::string GeneratePyCyxWizCode(const std::vector<int>& sorted_ids);

    // Framework-specific layer conversion
    std::string NodeTypeToPythonLayer(const MLNode& node);
    std::string NodeTypeToTensorFlowLayer(const MLNode& node, int layer_idx);
    std::string NodeTypeToKerasLayer(const MLNode& node);
    std::string NodeTypeToPyCyxWizLayer(const MLNode& node);

    std::vector<int> TopologicalSort();
    const MLNode* FindNodeById(int node_id) const;

    bool show_window_;

    // Node graph state
    std::vector<MLNode> nodes_;
    std::vector<NodeLink> links_;
    int next_node_id_;
    int next_pin_id_;
    int next_link_id_;

    // UI state
    bool show_context_menu_;
    int context_menu_node_id_;  // -1 if clicking on canvas
    int selected_node_id_;  // Currently selected node for properties panel (-1 = none)
    CodeFramework selected_framework_;  // Current code generation framework

    // ImNodes editor context
    ImNodesEditorContext* editor_context_;

    // Script editor for code output
    cyxwiz::ScriptEditorPanel* script_editor_;

    // Properties panel for node selection display
    Properties* properties_panel_;

    // Current file path for save/load
    std::string current_file_path_;

    // Deferred node addition (to avoid modifying nodes_ while ImNodes is rendering)
    struct PendingNode {
        NodeType type;
        std::string name;
        ImVec2 position;  // Grid space position where node should be created
    };
    std::vector<PendingNode> pending_nodes_;
    ImVec2 context_menu_pos_;  // Mouse position when context menu was opened (grid space)
};

} // namespace gui
