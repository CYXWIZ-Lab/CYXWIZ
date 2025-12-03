#pragma once

#include <vector>
#include <string>
#include <map>
#include <functional>
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
    AdamW,
    // Data Pipeline Nodes
    DatasetInput,       // Load dataset from DataRegistry
    DataLoader,         // Batch iterator with shuffle/drop_last
    Augmentation,       // Transform pipeline for data augmentation
    DataSplit,          // Train/val/test splitter
    TensorReshape,      // Reshape tensor dimensions
    Normalize,          // Normalize values (mean/std)
    OneHotEncode        // Label encoding
};

// Attribute for node pins (inputs/outputs)
enum class PinType {
    Tensor,      // General tensor data
    Labels,      // Label tensor (for classification)
    Parameters,  // Model parameters
    Loss,        // Loss value
    Optimizer,   // Optimizer state
    Dataset      // Dataset handle reference
    // Note: Shape is metadata (node parameter), not a data flow type
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

    // Visibility control for sidebar integration
    bool* GetVisiblePtr() { return &show_window_; }

    // Minimap visibility control
    void SetShowMinimap(bool show) { show_minimap_ = show; }
    bool GetShowMinimap() const { return show_minimap_; }
    bool* GetShowMinimapPtr() { return &show_minimap_; }

    // Access to graph data for compilation
    const std::vector<MLNode>& GetNodes() const { return nodes_; }
    const std::vector<NodeLink>& GetLinks() const { return links_; }

    // Training callback - set by MainWindow to trigger training from node graph
    using TrainCallback = std::function<void(const std::vector<MLNode>&, const std::vector<NodeLink>&)>;
    void SetTrainCallback(TrainCallback callback) { train_callback_ = callback; }

    // Check if graph is ready for training
    bool IsGraphValid() const;

    // Training state control
    void SetTrainingActive(bool active) { is_training_ = active; }
    bool IsTrainingActive() const { return is_training_; }

    // Update DatasetInput node name based on loaded dataset
    void UpdateDatasetNodeName(const std::string& dataset_name);

private:
    void ShowToolbar();
    void RenderNodes();
    void RenderMinimap();
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
    void ExportCodeToFile();
    void ShowExportDialog();

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

    // Training callback
    TrainCallback train_callback_;

    // Training animation state
    bool is_training_ = false;
    float training_animation_time_ = 0.0f;

    // Minimap state
    bool show_minimap_ = true;
    ImVec2 minimap_size_ = ImVec2(180.0f, 140.0f);  // Size of minimap in pixels
    bool minimap_navigating_ = false;  // Is user dragging to navigate within minimap
    ImVec2 minimap_screen_min_;        // Screen-space bounds of minimap (for input blocking)
    ImVec2 minimap_screen_max_;        // Screen-space bounds of minimap (for input blocking)
    bool mouse_over_minimap_ = false;  // True when mouse is over minimap window

    // Minimap position options
    enum class MinimapPosition { TopLeft, TopRight, BottomLeft, BottomRight };
    MinimapPosition minimap_position_ = MinimapPosition::BottomRight;
};

} // namespace gui
