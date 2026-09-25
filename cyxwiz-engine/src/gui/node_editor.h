#pragma once

#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <memory>
#include <optional>
#include <functional>
#include <atomic>
#include <cstdint>
#include <thread>
#include <utility>
#include <imgui.h>
#include <nlohmann/json_fwd.hpp>

// Forward declarations from ImNodes
struct ImNodesEditorContext;

// Forward declarations
// Forward declare ScriptingEngine in global scripting namespace
#include "../core/graph_model.h"

namespace scripting { class ScriptingEngine; }

namespace cyxwiz {
class ScriptEditorPanel;
class GraphExecutor;
class RLTrainingExecutor;
class TrainingDashboardPanel;
class PipelineExecutor;
class PipelineExecutionTracker;
}

namespace gui {
class Properties;
class ShapeInferenceEngine;

// Icon pack selection for node icons
enum class IconPack {
    FontAwesome,    // Default - FontAwesome 6 icons (solid style)
    Tabler,         // Tabler Icons - clean 2px outline style
    Remix,          // Remix Icon - filled style icons
    Lucide,         // Lucide Icons - clean minimal stroke style
    Iconoir,        // Iconoir Icons - simple consistent stroke icons
    Phosphor        // Phosphor Icons - flexible icon family with multiple weights
};

struct DataBoundaryMigrationResult {
    bool success = false;
    int nodes_migrated = 0;
    int links_removed = 0;
    int links_rerouted = 0;
    std::string message;
};

// Graph snapshot for undo/redo
struct GraphSnapshot {
    std::vector<SubgraphData> subgraphs;
    std::map<int, ImVec2> positions;
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

// Execution modes for the unified canvas (Unified Canvas Phase 2)
enum class ExecutionMode {
    CodeGeneration,   // Generate PyTorch/TF/Keras code (existing behavior)
    DuckDBPipeline,   // Execute data transforms with DuckDB/Arrow (new)
    LocalTraining     // Train models locally with cyxwiz-backend (existing)
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

    // Debouncing state
    float search_debounce_timer_ = 0.0f;      // Time remaining before search triggers
    std::string last_search_query_;           // Last processed query
    bool search_dirty_ = false;               // True if search needs to be updated
};

// Node implementation status for template/coming soon nodes
enum class NodeImplementationStatus {
    Implemented,    // Fully working node
    Template,       // Coming Soon - defined but not implemented
    Deprecated      // Being phased out
};

// Entry for searchable node
struct SearchableNode {
    NodeType type;
    std::string name;      // Display name (e.g., "Dense (512 units)")
    std::string category;  // Category (e.g., "Layers > Dense / Linear")
    std::string keywords;  // Additional keywords for search
    std::string plugin_qualified_name;  // For PluginCustom: "plugin_id:type_name"
    NodeImplementationStatus status = NodeImplementationStatus::Implemented;  // Default to implemented
    std::string description;  // Brief description for info panel
    std::string tooltip;      // Tooltip text (for templates: why not available)
    bool support_blocked = false;  // Derived from NodeMetadata::support_axes
    std::string support_state;     // Compact central support state value
    std::string support_reason;    // Compact central support reason
};

// Alignment types for arranging selected nodes
enum class AlignmentType { Left, Center, Right, Top, Middle, Bottom };
enum class DistributeType { Horizontal, Vertical };

// Node group for visual organization
struct NodeGroup {
    int id;
    std::string name;
    std::string description;  // KNIME-style: bullet-point description shown in header
    std::vector<int> node_ids;
    ImVec4 color;           // RGBA color for group box
    bool collapsed = false;
    float padding = 20.0f;  // Padding around contained nodes
};

// Canvas annotation (KNIME-style sticky note)
struct CanvasAnnotation {
    int id;
    std::string title;
    std::string content;
    ImVec2 position;
    ImVec2 size;
    ImU32 color;            // Background color
    bool is_minimized;

    CanvasAnnotation()
        : id(-1)
        , position(0, 0)
        , size(200, 100)
        , color(IM_COL32(255, 255, 200, 255))  // Default yellow
        , is_minimized(false) {}
};

// Canvas Frame - visual organization box (drag from Node Browser)
struct CanvasFrame {
    int id;
    std::string title;
    std::string description;
    ImVec2 position;        // Canvas position (not screen)
    ImVec2 size;
    ImVec4 color;           // RGBA color
    bool is_selected = false;

    CanvasFrame()
        : id(-1)
        , title("Frame")
        , position(0, 0)
        , size(300, 200)
        , color(0.3f, 0.5f, 0.7f, 0.8f) {}  // Default blue
};

// Annotation color presets
enum class AnnotationColor {
    Yellow,   // Default - IM_COL32(255, 255, 200, 255)
    Blue,     // Info - IM_COL32(200, 220, 255, 255)
    Green,    // Done - IM_COL32(200, 255, 200, 255)
    Orange,   // TODO - IM_COL32(255, 230, 200, 255)
    Pink      // Important - IM_COL32(255, 200, 220, 255)
};

// Validation warning severity levels
enum class ValidationSeverity {
    Error,    // Blocking errors (prevent training)
    Warning,  // Non-blocking warnings (suggest improvements)
    Info      // Informational messages
};

// Validation warning for shape mismatches and other issues
struct ValidationWarning {
    int node_id;                      // Node with the issue
    ValidationSeverity severity;      // Warning severity
    std::string message;              // User-facing message
    std::string suggested_fix;        // Suggested action
    bool has_auto_fix = false;        // Whether auto-fix is available
    int from_node_id = -1;            // Source node (for connection issues)
    int to_node_id = -1;              // Target node (for connection issues)
};

// Execution context for unified canvas (Unified Canvas Phase 2)
struct ExecutionContext {
    ExecutionMode mode;
    std::map<int, std::string> node_results;  // Node ID -> dataset/tensor name in DuckDB or memory
    std::string input_dataset;                // Initial dataset name
    std::string output_dataset;               // Final result dataset name
};

// Unified Canvas Phase 6: Execution state for visualization
enum class NodeExecutionState {
    Idle,        // Not executing
    Pending,     // Waiting to execute
    Executing,   // Currently executing
    Completed,   // Successfully completed
    Error        // Failed with error
};

class NodeEditor {
public:
    NodeEditor();
    ~NodeEditor();

    // Cancels background work this editor owns (the data pipeline run) and
    // drops its pending UI delivery. Called on project close and destruction.
    void CancelOwnedBackgroundWork();

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

    // Icon pack selection (for node icons)
    void SetIconPack(IconPack pack) { icon_pack_ = pack; }
    IconPack GetIconPack() const { return icon_pack_; }

    // Access to graph data for compilation
    const std::vector<MLNode>& GetNodes() const { return nodes_; }
    const std::vector<NodeLink>& GetLinks() const { return links_; }
    int GetSelectedNodeId() const { return selected_node_id_; }
    const std::string& GetCurrentFilePath() const { return current_file_path_; }
    std::string GetNodeTypeDisplayName(NodeType type) const { return GetNodeTypeName(type); }

    // Thread-safe scalar simulation bridge used by Properties.
    bool IsGraphSimulationRunning() const { return is_simulating_.load(); }
    bool TryGetSimulationScalar(int pin_id, float& value) const;
    float GetSimulationTime() const;
    bool SetSimulationNodeParameter(int node_id,
                                    const std::string& key,
                                    const std::string& value);

    // Training callback - set by MainWindow to trigger training from node graph
    using TrainCallback = std::function<void(const std::vector<MLNode>&, const std::vector<NodeLink>&)>;
    void SetTrainCallback(TrainCallback callback) { train_callback_ = callback; }

    // Compile callback - set by MainWindow to dry-run the graph compilation (no training)
    using CompileCallback = std::function<void()>;
    void SetCompileCallback(CompileCallback callback) { compile_callback_ = callback; }

    // Local Debug callback - runs one forward + one backward pass on a
    // synthetic batch to catch shape / NaN / dead-subgraph bugs before
    // real training. Mirrors SetCompileCallback; triggered by the
    // toolbar "Local Debug" button or the F6 shortcut.
    using DebugCallback = std::function<void()>;
    void SetDebugCallback(DebugCallback callback) { debug_callback_ = callback; }

    // Open Custom Node Editor callback - opens the CustomNodeEditorPanel from the Studio toolbar
    using OpenCustomNodeEditorCallback = std::function<void()>;
    void SetOpenCustomNodeEditorCallback(OpenCustomNodeEditorCallback callback) { open_custom_node_editor_callback_ = callback; }

    // Open Studio Debugger callback - opens the debugger panel from the Studio toolbar
    using OpenStudioDebuggerCallback = std::function<void()>;
    void SetOpenStudioDebuggerCallback(OpenStudioDebuggerCallback callback) { open_studio_debugger_callback_ = callback; }

    // Explain Node callback - opens a node-scoped explanation in Studio Debugger.
    using ExplainNodeCallback = std::function<void(int)>;
    void SetExplainNodeCallback(ExplainNodeCallback callback) { explain_node_callback_ = std::move(callback); }

    // Check if graph is ready for training
    bool IsGraphValid() const;
    // Graph replacement admission is checked by all load/clear entry points.
    std::string GetGraphReplacementBlockReason() const;
    bool CanReplaceGraph(const char* operation);
    void SetGraphPreparationBusyCallback(std::function<bool()> callback) {
        graph_preparation_busy_callback_ = std::move(callback);
    }


    // Training state control
    void SetTrainingActive(bool active) { is_training_ = active; }
    bool IsTrainingActive() const { return is_training_; }

    // Unified Canvas Phase 2: Execution mode control
    void SetExecutionMode(ExecutionMode mode) { execution_mode_ = mode; }
    ExecutionMode GetExecutionMode() const { return execution_mode_; }

    // Execute the current graph based on execution mode
    void ExecuteGraph();

    // Execute data pipeline using DuckDB (Phase 2)
    bool ExecuteDataPipeline();

    // Unified Canvas Phase 6: Execution visualization
    void SetNodeExecutionState(int node_id, NodeExecutionState state);
    void SetNodeExecutionError(int node_id, const std::string& error);
    void ClearExecutionStates();

    // Update DatasetInput node name based on loaded dataset
    void UpdateDatasetNodeName(const std::string& dataset_name);

    // Dynamic pins: rebuild a node's pins from plugin-provided data.
    // Called when a trigger parameter changes on a supports_dynamic_pins node.
    void ResolveDynamicPins(int node_id);

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

    // Legacy Data Input/Split/Loader graphs retain their original pin layout
    // until the user explicitly requests this migration.
    bool HasLegacyDataBoundary() const;
    DataBoundaryMigrationResult MigrateLegacyDataBoundary();

    // Get current graph as JSON string (for export to .cyxmodel)
    std::string GetGraphJson() const;

    // Show the node editor window
    void Show() { show_window_ = true; }

    // ===== Data Studio Integration (Phase 5 Week 7) =====

    /**
     * Set dataset from Data Studio deployment
     * Finds or creates a DatasetInput node and populates it with the dataset
     * Automatically frames the node after creation
     */
    void SetDatasetFromDataStudio(const std::string& dataset_name);

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

    // Deterministic node factory used by CreateNode and registry contract
    // gates. Keeping the ID cursors explicit lets non-GUI callers exercise
    // the exact production dispatch without constructing ImGui/ImNodes state.
    static MLNode CreateNodeWithIds(NodeType type,
                                    const std::string& name,
                                    int& next_node_id,
                                    int& next_pin_id);

    // Unified Canvas Phase 1: Get category for a node type
    static NodeCategory GetCategoryForNodeType(NodeType type);

    // ===== Menu Operations (Public API for Toolbar) =====

    // Add a node at the center of the visible area
    void AddNodeFromMenu(NodeType type, const std::string& name);
    void AddDataInputFromAsset(const std::string& path);

    // Delete currently selected nodes
    void DeleteSelectedNodes();

    // Duplicate currently selected nodes
    void DuplicateSelectedNodes();

    // Group selected nodes with auto-generated name
    void GroupSelectedNodes();

    // Ungroup selected nodes
    void UngroupSelectedNodes();

    // ===== Canvas Annotations (KNIME-style sticky notes) =====

    // Add a new annotation at the center of the visible area
    void AddAnnotation();

    // Add annotation at specific position
    void AddAnnotationAt(const ImVec2& position);

    // Delete annotation by ID
    void DeleteAnnotation(int annotation_id);

    // Delete currently selected annotation
    void DeleteSelectedAnnotation();

    // Get annotations for serialization
    const std::vector<CanvasAnnotation>& GetAnnotations() const { return annotations_; }

    // Set annotations (for loading)
    void SetAnnotations(const std::vector<CanvasAnnotation>& annotations);

    // ===== Workflow Description =====

    // Get workflow description buffer for UI editing
    char* GetWorkflowDescriptionBuffer() { return workflow_description_; }
    size_t GetWorkflowDescriptionBufferSize() const { return sizeof(workflow_description_); }

    // Get/set workflow description as string
    std::string GetWorkflowDescription() const { return workflow_description_; }
    void SetWorkflowDescription(const std::string& desc);

    // ===== Pin / Node State =====
    // Drives pin colors in the node editor. The state machine is:
    //   Default        → red hollow  (graph never compiled, or just edited)
    //   CompileFailed  → red solid   (last Compile flagged this node)
    //   CompilePassed  → green hollow (last Compile validated this node)
    //   Trained        → green solid (training run completed successfully)
    // Cleared on any graph modification so stale state is never shown.

    enum class NodePinState {
        Default,
        CompileFailed,
        CompilePassed,
        Trained
    };

    void SetNodePinState(int node_id, NodePinState state);
    void SetAllNodesPinState(NodePinState state);
    void ClearValidationState();
    NodePinState GetNodePinState(int node_id) const;
    bool HasValidationState() const { return !node_pin_state_.empty(); }

    // Select a node in the canvas and frame it into view.
    void FocusNode(int node_id);

private:
    void ShowToolbar();
    void RenderNodes();
    void RenderHoveredNodeTooltip(int hovered_node_id);
    void SyncPipelineExecutionVisualization();
    void RenderMinimap();
    void HandleInteractions();
    void ShowContextMenu();
    void ConfigureNode(int node_id);
    void ShowSingleNodeContextMenu();  // Node-specific context menu
    void ShowNodeDescriptionEditPopup();  // KNIME-style node description editor

    // Unified Canvas Phase 3: Categorized node palette helpers
    void ShowCategorizedNodeMenu();
    void RenderNodeCategory(NodeCategory category, const char* category_name, const char* icon);
    static const char* GetCategoryIcon(NodeCategory category);
    static const char* GetCategoryName(NodeCategory category);

    // Helper functions
    unsigned int GetNodeColor(NodeType type);
    const char* GetNodeIcon(NodeType type);

    // Node management
    void AddNode(NodeType type, const std::string& name);
    bool CanAddNodeToGraph(NodeType type) const;
    void DeleteNode(int node_id);
    void ClearGraph();

    // Link management
    void CreateLink(int from_pin, int to_pin, int from_node, int to_node,
                    LinkType type = LinkType::TensorFlow);

    // Get visual color for a link based on its type
    ImU32 GetLinkColor(LinkType type) const;
    ImU32 GetLinkHoverColor(LinkType type) const;

    // Get visual color for pins and links based on data type
    ImU32 GetPinTypeColor(PinType type) const;
    ImU32 GetPinTypeHoverColor(PinType type) const;

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

    // Pin lookup optimization
    void RebuildPinLookup();
    void RebuildDataBoundaryPins(MLNode& node, bool legacy_contract);

    // File operations
    bool SaveGraph(const std::string& filepath);
    bool LoadGraphJson(const nlohmann::json& graph_json,
                       const std::string& source_description);
    void ShowSaveDialog();
    void ShowLoadDialog();
    void ExportCodeToFile();
    void ShowExportDialog();

    // Code generation
    void GeneratePythonCode();
    void GenerateCodeForFramework(CodeFramework framework);
    bool ValidateGraph(std::string& error_message);

    // Unified Canvas Phase 4.2: Helper for node type names
    std::string GetNodeTypeName(NodeType type) const;

    // Shape validation (non-blocking warnings)
    std::vector<ValidationWarning> ValidateShapes();
    bool Is4DOutputNode(NodeType type) const;
    bool Expects2DInput(NodeType type) const;

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
    void RenderFrames();
    NodeGroup* FindGroupContainingNode(int node_id);

    // Frame management
    void AddFrameAt(const ImVec2& canvas_position);
    void DeleteFrame(int frame_id);
    const std::vector<CanvasFrame>& GetFrames() const { return frames_; }

    // Annotation rendering and interaction
    void RenderAnnotations();
    void RenderAnnotationEditPopup();
    void HandleAnnotationInteraction();
    CanvasAnnotation* FindAnnotationById(int annotation_id);
    void ShowAnnotationContextMenu(int annotation_id);
    void ShowAnnotationEditPopup();
    ImU32 GetAnnotationColorValue(AnnotationColor color) const;

    // Subgraph encapsulation
    void CreateSubgraphFromSelection(const std::string& name);
    void ExpandSubgraph(int node_id);
    void CollapseSubgraph(int node_id);
    void ToggleSubgraphExpansion(int node_id);
    bool IsSubgraphNode(int node_id) const;
    SubgraphData* GetSubgraphData(int node_id);
    bool IsPreparationRecipeNode(int node_id) const;
    // SQL step input pins follow its alias list (one named pin per input).
    // Returns the ids of removed pins so their links can be dropped.
    std::vector<int> SyncSqlStepInputPins(MLNode& node);
    void SyncParameterDrivenPins();
    // Marks a subgraph as a Preparation Recipe after validating the recipe
    // contract, or turns it back into a visual group. Returns false and shows
    // the reason when the subgraph does not qualify.
    bool SetSubgraphPreparationRecipe(int node_id, bool recipe);
    bool IsSubgraphMember(int node_id) const;

    // Framework-specific generators
    std::string GeneratePyTorchCode(const std::vector<int>& sorted_ids);
    // PyTorch export of the optimizer node's training recipe (tofix112).
    const MLNode* FindExportOptimizerNode() const;
    std::string PyTorchOptimizerSetup() const;
    std::string PyTorchStepLines() const;
    std::string GenerateTensorFlowCode(const std::vector<int>& sorted_ids);
    std::string GenerateKerasCode(const std::vector<int>& sorted_ids);
    std::string GeneratePyCyxWizCode(const std::vector<int>& sorted_ids);
    std::optional<std::string> FindDenseActivationConfigurationError(
        const std::vector<int>& sorted_ids) const;
    std::optional<std::string> FindUnsupportedSequentialLayerError(
        const std::vector<int>& sorted_ids) const;

    // RL-specific code generation
    bool IsRLGraph(const std::vector<int>& sorted_ids) const;
    std::string GenerateRLPyTorchCode(const std::vector<int>& sorted_ids) const;
    std::string GenerateRLPyCyxWizCode(const std::vector<int>& sorted_ids) const;

    // Framework-specific layer conversion
    std::string NodeTypeToPythonLayer(const MLNode& node);
    std::string NodeTypeToTensorFlowLayer(const MLNode& node, int layer_idx);
    std::string NodeTypeToKerasLayer(const MLNode& node);
    std::string NodeTypeToPyCyxWizLayer(const MLNode& node);

    std::vector<int> TopologicalSort();
    const MLNode* FindNodeById(int node_id) const;
    MLNode* FindNodeById(int node_id);  // Non-const version

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

    // Pin lookup optimization - O(1) pin ID -> (node*, pin*) mapping
    std::unordered_map<int, std::pair<MLNode*, NodePin*>> pin_lookup_;

    // Pin state (node_id → NodePinState). Set by Compile/Train, cleared on graph change.
    std::unordered_map<int, NodePinState> node_pin_state_;

    // UI state
    bool show_context_menu_;
    int context_menu_node_id_;  // -1 if clicking on canvas
    int selected_node_id_;  // Currently selected node for properties panel (-1 = none)
    CodeFramework selected_framework_;  // Current code generation framework
    ExecutionMode execution_mode_;      // Unified Canvas Phase 2: Current execution mode

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
        PendingNode(
            NodeType type_value,
            std::string name_value,
            ImVec2 position_value,
            std::unordered_map<std::string, std::string>
                initial_parameters_value = {})
            : type(type_value),
              name(std::move(name_value)),
              position(position_value),
              initial_parameters(std::move(initial_parameters_value)) {}

        NodeType type;
        std::string name;
        ImVec2 position;  // Grid space position where node should be created
        std::unordered_map<std::string, std::string> initial_parameters;
    };
    std::vector<PendingNode> pending_nodes_;
    ImVec2 context_menu_pos_;  // Mouse position when context menu was opened (grid space)

    // Training callback
    TrainCallback train_callback_;

    // Compile callback (dry-run GraphCompiler::Compile and show result popup)
    CompileCallback compile_callback_;

    // Local Debug callback (runs one forward + backward pass on synthetic data)
    DebugCallback debug_callback_;

    // Open Custom Node Editor callback (opens CustomNodeEditorPanel)
    OpenCustomNodeEditorCallback open_custom_node_editor_callback_;

    // Open Studio Debugger callback (opens StudioDebuggerPanel)
    OpenStudioDebuggerCallback open_studio_debugger_callback_;

    // Explain selected node in Studio Debugger.
    ExplainNodeCallback explain_node_callback_;

    // Deferred focus request so sidebar actions never touch ImNodes outside
    // the live editor render scope.
    int pending_focus_node_id_ = -1;

    // Training animation state
    std::atomic<bool> is_training_{false};
    std::function<bool()> graph_preparation_busy_callback_;

    float training_animation_time_ = 0.0f;

    // Zoom state
    float zoom_ = 1.0f;
    static constexpr float ZOOM_MIN = 0.5f;
    static constexpr float ZOOM_MAX = 2.0f;

    // Icon pack selection (for node icons)
    IconPack icon_pack_ = IconPack::FontAwesome;  // Default to FontAwesome

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

    // Auto-insert Flatten dialog state
    bool show_auto_insert_flatten_dialog_ = false;
    int pending_flatten_from_node_ = -1;
    int pending_flatten_to_node_ = -1;
    int pending_flatten_from_pin_ = -1;
    int pending_flatten_to_pin_ = -1;

    // Shape inference engine and validation warnings
    std::unique_ptr<ShapeInferenceEngine> shape_inference_;
    std::vector<ValidationWarning> validation_warnings_;

    // Deferred clear flag (to call ImNodes clear inside BeginNodeEditor scope)
    bool pending_clear_imnodes_ = false;

    // Flag to recreate ImNodes editor context (full reset)
    bool pending_context_reset_ = false;

    // UI-owned rejection feedback, rendered even when the Studio tab is hidden.
    bool graph_busy_dialog_pending_ = false;
    std::string graph_busy_message_;
    // Why a pipeline or recipe action could not proceed (same rendering rule).
    bool pipeline_notice_pending_ = false;
    std::string pipeline_notice_;
    void ShowPipelineNotice(std::string message);
    bool SubmitDataPipelineJson(nlohmann::json pipeline_json);
    // Recipe node -> its step node ids for the pipeline run in flight; the
    // executor reports step states, the recipe node shows their roll-up.
    std::map<int, std::vector<int>> recipe_step_ids_;
    void RollUpRecipeExecutionStates();

    // Empty graph warning popup state
    bool show_empty_graph_warning_ = false;

    // Search state (Ctrl+F for existing nodes)
    SearchState search_state_;

    // Node add search state (top-right search box)
    NodeAddSearchState node_add_search_;
    std::vector<SearchableNode> all_searchable_nodes_;       // All available nodes for search
    std::vector<std::pair<int, SearchableNode*>> filtered_nodes_;  // Filtered results with scores
    bool searchable_nodes_initialized_ = false;

    // Unified Canvas Phase 3: Context menu search filter
    char context_menu_search_[256] = "";
    std::map<NodeCategory, std::vector<std::pair<NodeType, std::string>>> nodes_by_category_;
    bool nodes_by_category_initialized_ = false;

    // Node groups
    std::vector<NodeGroup> groups_;
    int next_group_id_ = 1;

    // Create group dialog state
    bool show_create_group_dialog_ = false;
    char create_group_name_[256] = "";
    float create_group_color_[4] = {0.2f, 0.5f, 0.8f, 0.3f};

    // Canvas annotations (KNIME-style sticky notes)
    std::vector<CanvasAnnotation> annotations_;
    int next_annotation_id_ = 1;
    int selected_annotation_id_ = -1;
    int dragging_annotation_id_ = -1;
    ImVec2 annotation_drag_offset_ = ImVec2(0, 0);
    bool editing_annotation_ = false;
    int editing_annotation_id_ = -1;
    char annotation_edit_title_[256] = "";
    char annotation_edit_content_[2048] = "";

    // Canvas frames (visual organization boxes - drag from Node Browser)
    std::vector<CanvasFrame> frames_;
    int next_frame_id_ = 1;
    int selected_frame_id_ = -1;
    int dragging_frame_id_ = -1;
    int resizing_frame_id_ = -1;
    ImVec2 frame_drag_offset_ = ImVec2(0, 0);
    bool editing_frame_ = false;
    int editing_frame_id_ = -1;
    char frame_edit_title_[256] = "";
    char frame_edit_desc_[1024] = "";
    bool frame_right_clicked_ = false;  // Prevents canvas menu when frame is right-clicked

    // KNIME-style node dragging (icon-only drag to avoid pin-drag moving the node)
    int dragging_knime_node_id_ = -1;
    ImVec2 knime_drag_offset_ = ImVec2(0, 0);

    // KNIME-style: Node description editing
    bool editing_node_description_ = false;
    int editing_node_id_ = -1;
    char node_description_buffer_[1024] = "";
    int right_clicked_node_id_ = -1;  // Node that was right-clicked

    // KNIME-style: Per-node display options (stored in node parameters)
    // show_name: "true" or "false" - whether to show name below node
    // show_description: "true" or "false" - whether to show description below node
    // Use GetNodeDisplayOption() and SetNodeDisplayOption() to access

    // Workflow description (shown in CyxWiz Studio section)
    char workflow_description_[2048] = "";

    // Subgraph data storage
    std::vector<SubgraphData> subgraphs_;

    // ===== Graph Simulation State =====
    std::unique_ptr<cyxwiz::GraphExecutor> graph_executor_;
    std::thread sim_thread_;
    std::atomic<bool> is_simulating_{false};
    std::atomic<bool> sim_stop_requested_{false};
    float sim_rate_hz_ = 60.0f;  // Simulation tick rate

    void OnRunSimulation();
    void OnStopSimulation();
    bool HasSimulationNodes() const;  // Check if graph contains signal/plant nodes

    // ===== RL Training State =====
    std::unique_ptr<cyxwiz::RLTrainingExecutor> rl_executor_;
    std::shared_ptr<cyxwiz::TrainingDashboardPanel> rl_dashboard_;
    bool rl_script_running_ = false;

    // ===== Unified Canvas Phase 2: Data Pipeline Execution =====
    uint64_t pipeline_task_id_ = 0;
    std::shared_ptr<cyxwiz::PipelineExecutionTracker> pipeline_execution_tracker_;
    // Lifetime token for background work this editor owns; destroying the
    // editor cancels that work and discards its pending UI delivery.
    std::shared_ptr<const void> task_owner_token_ = std::make_shared<int>(0);
    bool pipeline_execution_active_ = false;
    ExecutionContext execution_context_;

    // ===== Unified Canvas Phase 6: Execution Visualization =====
    std::map<int, NodeExecutionState> node_execution_states_;  // Node ID -> execution state
    std::map<int, std::string> node_execution_errors_;         // Node ID -> error message
    int currently_executing_node_id_ = -1;                     // Node currently being executed
    float execution_pulse_time_ = 0.0f;                        // Animation time for pulsing effect

    // Scripting engine for Python-based RL training
    scripting::ScriptingEngine* scripting_engine_ = nullptr;
public:
    void SetScriptingEngine(scripting::ScriptingEngine* engine) { scripting_engine_ = engine; }
private:
    bool HasRLNodes() const;
    void OnStartRLTraining();
    void OnStopRLTraining();
    void ExportPolicyONNX(const std::string& output_path);
    bool export_onnx_dialog_open_ = false;

    // Phase 4.7: Performance metrics
    float last_eval_time_ms_ = 0.0f;
};

} // namespace gui

