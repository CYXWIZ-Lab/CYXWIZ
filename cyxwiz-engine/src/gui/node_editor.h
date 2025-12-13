#pragma once

#include <vector>
#include <string>
#include <map>
#include <functional>
#include <imgui.h>
#include <nlohmann/json_fwd.hpp>

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
    // ===== Core Layers =====
    Dense,

    // Convolutional Layers
    Conv1D,
    Conv2D,
    Conv3D,
    DepthwiseConv2D,

    // Pooling Layers
    MaxPool2D,
    AvgPool2D,
    GlobalMaxPool,
    GlobalAvgPool,
    AdaptiveAvgPool,

    // Normalization Layers
    BatchNorm,
    LayerNorm,
    GroupNorm,
    InstanceNorm,

    // Regularization
    Dropout,
    Flatten,

    // ===== Recurrent Layers =====
    RNN,
    LSTM,
    GRU,
    Bidirectional,
    TimeDistributed,
    Embedding,

    // ===== Attention & Transformer =====
    MultiHeadAttention,
    SelfAttention,
    CrossAttention,
    LinearAttention,      // O(n) linear attention (Performer/Linear Transformer)
    TransformerEncoder,
    TransformerDecoder,
    PositionalEncoding,

    // ===== Activation Functions =====
    ReLU,
    LeakyReLU,
    PReLU,
    ELU,
    SELU,
    GELU,
    Swish,
    Mish,
    Sigmoid,
    Tanh,
    Softmax,

    // ===== Shape Operations =====
    Reshape,
    Permute,
    Squeeze,
    Unsqueeze,
    View,
    Split,

    // ===== Merge Operations =====
    Concatenate,
    Add,
    Multiply,
    Average,

    // ===== Output =====
    Output,

    // ===== Loss Functions =====
    MSELoss,
    CrossEntropyLoss,
    BCELoss,
    BCEWithLogits,
    L1Loss,
    SmoothL1Loss,
    HuberLoss,
    NLLLoss,

    // ===== Optimizers =====
    SGD,
    Adam,
    AdamW,
    RMSprop,
    Adagrad,
    NAdam,

    // ===== Learning Rate Schedulers =====
    StepLR,
    CosineAnnealing,
    ReduceOnPlateau,
    ExponentialLR,
    WarmupScheduler,

    // ===== Regularization Nodes =====
    L1Regularization,
    L2Regularization,
    ElasticNet,

    // ===== Utility Nodes =====
    Lambda,
    Identity,
    Constant,
    Parameter,

    // ===== Data Pipeline Nodes =====
    DatasetInput,       // Load dataset from DataRegistry
    DataLoader,         // Batch iterator with shuffle/drop_last
    Augmentation,       // Transform pipeline for data augmentation
    DataSplit,          // Train/val/test splitter
    TensorReshape,      // Reshape tensor dimensions (legacy, use Reshape)
    Normalize,          // Normalize values (mean/std)
    OneHotEncode,       // Label encoding

    // ===== Composite Nodes =====
    Subgraph            // Encapsulated subgraph (collapsible)
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

// Pin capacity constants for variadic connections
constexpr int PIN_UNLIMITED = -1;  // No limit on connections
constexpr int PIN_SINGLE = 1;      // Standard single connection

// Node pin structure
struct NodePin {
    int id;
    PinType type;
    std::string name;
    bool is_input;  // true = input pin, false = output pin

    // Variadic pin support - enables multiple connections to a single pin
    bool is_variadic = false;        // True for pins accepting multiple connections
    int min_connections = 0;         // Minimum required connections (0 = optional)
    int max_connections = PIN_SINGLE; // Maximum allowed (-1 = unlimited)

    // Visual indicators
    bool is_required = true;         // Visual indicator for required pins
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

    // Position for pattern insertion (optional - set by InstantiatePattern)
    float initial_pos_x = 0.0f;
    float initial_pos_y = 0.0f;
    bool has_initial_position = false;  // True if position should be applied when inserting
};

// Connection/Link types for visual differentiation
enum class LinkType {
    TensorFlow,         // Standard data flow (default)
    ResidualSkip,       // Residual/Skip connection (additive)
    DenseSkip,          // DenseNet-style skip (concatenative)
    AttentionFlow,      // Attention Q/K/V connections
    GradientFlow,       // Gradient backprop (for visualization)
    ParameterFlow,      // Parameter updates
    LossFlow            // Loss to optimizer
};

// Connection between nodes
struct NodeLink {
    int id;
    int from_node;
    int from_pin;
    int to_node;
    int to_pin;
    LinkType type = LinkType::TensorFlow;  // Connection type for visual styling
};

// Graph snapshot for undo/redo
struct GraphSnapshot {
    std::vector<MLNode> nodes;
    std::vector<NodeLink> links;
    int next_node_id;
    int next_pin_id;
    int next_link_id;
};

// Clipboard data for copy/paste
struct ClipboardData {
    std::vector<MLNode> nodes;
    std::vector<NodeLink> links;  // Internal links only
    bool valid = false;
};

// Supported code generation frameworks
enum class CodeFramework {
    PyTorch,
    TensorFlow,
    Keras,
    PyCyxWiz
};

// Search state for node search/filter feature (Ctrl+F to find existing nodes)
struct SearchState {
    char search_buffer[256] = "";
    std::vector<int> matching_node_ids;
    int current_match_index = -1;
    bool search_visible = false;
};

// Search state for adding nodes via search box (top-right of canvas)
struct NodeAddSearchState {
    char search_buffer[256] = "";
    bool is_active = false;           // Is the search box focused
    bool show_results = false;        // Show dropdown results
    int selected_index = 0;           // Currently selected result (keyboard navigation)
    bool just_activated = false;      // Set focus on next frame
};

// Entry for searchable node
struct SearchableNode {
    NodeType type;
    std::string name;      // Display name (e.g., "Dense (512 units)")
    std::string category;  // Category (e.g., "Layers > Dense / Linear")
    std::string keywords;  // Additional keywords for search
};

// Alignment types for arranging selected nodes
enum class AlignmentType { Left, Center, Right, Top, Middle, Bottom };
enum class DistributeType { Horizontal, Vertical };

// Node group for visual organization
struct NodeGroup {
    int id;
    std::string name;
    std::vector<int> node_ids;
    ImVec4 color;           // RGBA color for group box
    bool collapsed = false;
    float padding = 20.0f;  // Padding around contained nodes
};

// Subgraph data for encapsulated node groups
struct SubgraphData {
    int subgraph_node_id;                    // ID of the parent subgraph node
    std::vector<MLNode> internal_nodes;      // Nodes inside the subgraph
    std::vector<NodeLink> internal_links;    // Links between internal nodes
    std::vector<int> input_pin_mappings;     // External input pin -> internal node pin
    std::vector<int> output_pin_mappings;    // Internal node pin -> external output pin
    bool expanded = false;                   // Whether subgraph is expanded (visible)
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

    // Pattern insertion - add multiple nodes and links from a pattern template
    void InsertPattern(const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links);

    // ID getters for pattern library to generate unique IDs
    int GetNextNodeId() const { return next_node_id_; }
    int GetNextPinId() const { return next_pin_id_; }
    int GetNextLinkId() const { return next_link_id_; }

    // Load graph from file (for Asset Browser integration)
    // Supports both regular graph format and pattern template format
    bool LoadGraph(const std::string& filepath);

    // Load graph from JSON string (for import from .cyxmodel)
    bool LoadGraphFromString(const std::string& json_string);

    // Load pattern template format as graph (converts string IDs to int, resolves parameters)
    bool LoadPatternAsGraph(const nlohmann::json& j);

    // Get current graph as JSON string (for export to .cyxmodel)
    std::string GetGraphJson() const;

    // Show the node editor window
    void Show() { show_window_ = true; }

    // ===== Skip/Residual Connection Helpers =====

    // Add a residual (additive) skip connection between two nodes
    // If target is not an Add node, automatically inserts one
    // Returns the Add node ID (existing or newly created)
    int AddResidualConnection(int from_node_id, int to_node_id);

    // Add a dense (concatenative) skip connection between two nodes
    // If target is not a Concatenate node, automatically inserts one
    // Returns the Concatenate node ID (existing or newly created)
    int AddDenseConnection(int from_node_id, int to_node_id);

    // Wrap selected nodes with a residual block
    // Creates Add node after selection and connects input to Add's second input
    void WrapSelectionWithResidual();

    // Get all skip connections in the graph
    std::vector<NodeLink> GetSkipConnections() const;

    // Check if a link is a skip connection (bypasses multiple layers)
    bool IsSkipConnection(const NodeLink& link) const;

    // Auto-detect and mark skip connections based on graph topology
    void DetectSkipConnections();

    // Node factory - creates a node with proper pins for the given type
    // Made public so PatternBrowser can use it via callback
    MLNode CreateNode(NodeType type, const std::string& name);

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
    void DeleteNode(int node_id);
    void ClearGraph();

    // Link management
    void CreateLink(int from_pin, int to_pin, int from_node, int to_node,
                    LinkType type = LinkType::TensorFlow);

    // Get visual color for a link based on its type
    ImU32 GetLinkColor(LinkType type) const;
    ImU32 GetLinkHoverColor(LinkType type) const;

    // Connection tracking for variadic pins
    int GetConnectionCount(int pin_id) const;
    std::vector<int> GetConnectedPins(int pin_id) const;
    bool IsPinFull(int pin_id) const;
    bool IsPinConnected(int pin_id) const;
    bool CanAcceptConnection(int pin_id) const;
    std::vector<NodeLink> GetLinksToPin(int pin_id) const;
    std::vector<NodeLink> GetLinksFromPin(int pin_id) const;
    const NodePin* FindPinById(int pin_id) const;
    bool ValidateLink(int from_pin, int to_pin, std::string& error) const;

    // File operations
    bool SaveGraph(const std::string& filepath);
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

    // Undo/Redo system
    void SaveUndoState();
    void Undo();
    void Redo();
    bool CanUndo() const { return !undo_stack_.empty(); }
    bool CanRedo() const { return !redo_stack_.empty(); }

    // Clipboard operations
    void SelectAll();
    void ClearSelection();
    void DeleteSelected();
    void CopySelection();
    void CutSelection();
    void PasteClipboard();
    void DuplicateSelection();

    // Helper for finding empty position
    ImVec2 FindEmptyPosition();

    // Keyboard shortcuts
    void HandleKeyboardShortcuts();
    void FrameSelected();
    void FrameAll();

    // Search functionality
    void ShowSearchBar();
    void UpdateSearchResults();
    void NavigateToMatch(int direction);  // +1 = next, -1 = previous
    void HighlightMatchingNodes();

    // Node add search (top-right search box for quick node creation)
    void ShowNodeAddSearch();
    void InitializeSearchableNodes();
    void UpdateNodeAddSearchResults();
    static int FuzzyMatch(const std::string& pattern, const std::string& str);  // Returns match score (0 = no match)

    // Alignment and distribution tools
    void AlignSelectedNodes(AlignmentType type);
    void DistributeSelectedNodes(DistributeType type);
    void AutoLayoutSelection();

    // Node grouping
    void CreateGroupFromSelection(const std::string& name);
    void DeleteGroup(int group_id);
    void UngroupSelection();
    void RenderGroups();
    NodeGroup* FindGroupContainingNode(int node_id);

    // Subgraph encapsulation
    void CreateSubgraphFromSelection(const std::string& name);
    void ExpandSubgraph(int node_id);
    void CollapseSubgraph(int node_id);
    void ToggleSubgraphExpansion(int node_id);
    bool IsSubgraphNode(int node_id) const;
    SubgraphData* GetSubgraphData(int node_id);

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

    // Helper for code generation with variadic inputs
    std::vector<int> GetInputNodeIds(int node_id) const;
    bool IsMergeNode(NodeType type) const;
    bool HasMergeNodes() const;

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

    // Zoom state
    float zoom_ = 1.0f;
    static constexpr float ZOOM_MIN = 0.5f;
    static constexpr float ZOOM_MAX = 2.0f;

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

    // Undo/Redo state
    std::vector<GraphSnapshot> undo_stack_;
    std::vector<GraphSnapshot> redo_stack_;
    static constexpr size_t MAX_UNDO_LEVELS = 50;

    // Clipboard state
    ClipboardData clipboard_;
    std::vector<int> selected_node_ids_;  // Multi-selection support
    ImVec2 paste_offset_ = ImVec2(50.0f, 50.0f);  // Offset for pasted nodes

    // Deferred position setting (for nodes created outside render context)
    std::map<int, ImVec2> pending_positions_;  // node_id -> position
    int pending_positions_frames_ = 0;  // Number of frames to keep applying positions (needed for ImNodes)

    // Cached node positions (updated each frame inside BeginNodeEditor/EndNodeEditor scope)
    // Used by FindEmptyPosition() which may be called outside the editor scope
    std::map<int, ImVec2> cached_node_positions_;

    // Save as Pattern dialog state
    bool show_save_pattern_dialog_ = false;
    char save_pattern_name_[256] = "";
    char save_pattern_description_[1024] = "";

    // Deferred clear flag (to call ImNodes clear inside BeginNodeEditor scope)
    bool pending_clear_imnodes_ = false;

    // Flag to recreate ImNodes editor context (full reset)
    bool pending_context_reset_ = false;

    // Empty graph warning popup state
    bool show_empty_graph_warning_ = false;

    // Search state (Ctrl+F for existing nodes)
    SearchState search_state_;

    // Node add search state (top-right search box)
    NodeAddSearchState node_add_search_;
    std::vector<SearchableNode> all_searchable_nodes_;       // All available nodes for search
    std::vector<std::pair<int, SearchableNode*>> filtered_nodes_;  // Filtered results with scores
    bool searchable_nodes_initialized_ = false;

    // Node groups
    std::vector<NodeGroup> groups_;
    int next_group_id_ = 1;

    // Create group dialog state
    bool show_create_group_dialog_ = false;
    char create_group_name_[256] = "";
    float create_group_color_[4] = {0.2f, 0.5f, 0.8f, 0.3f};

    // Subgraph data storage
    std::vector<SubgraphData> subgraphs_;
};

} // namespace gui
