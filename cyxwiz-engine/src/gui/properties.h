#pragma once

#include <string>
#include <map>
#include <vector>
#include <deque>
#include <functional>
#include <memory>
#include <set>
#include "../core/node_metadata.h"
#include "node_config_dialog.h"
#include "properties_node_editors.h"
#include "properties_shape_info.h"
#include "properties_truth.h"

namespace cyxwiz {
class LiveGraphCompile;
}  // namespace cyxwiz

namespace gui {

// Forward declarations
enum class NodeType;
struct MLNode;
struct NodeLink;
class NodeEditor;

using NodeShapeInfo = properties_shape::NodeShapeInfo;

class Properties {
public:
    Properties();
    ~Properties();

    void Render();

    // Set the currently selected node to display properties for
    void SetSelectedNode(MLNode* node);
    // Select the node and open its dedicated dialog when one is registered.
    // Returns true when a dialog was opened; false means Properties is the
    // configuration surface for this node type.
    bool ConfigureNode(MLNode* node);
    void ClearSelection();
    void ClearNodeReferences();

    // Set the node editor reference for graph access
    void SetNodeEditor(NodeEditor* editor) { node_editor_ = editor; }

    // Visibility control for sidebar integration
    bool* GetVisiblePtr() { return &show_window_; }

    // Force recomputation of shapes (call when graph changes)
    void InvalidateShapes() { shapes_valid_ = false; }

    // The background compile of the canvas (TOFIX123); the AS COMPILED card
    // reads it. Owned by MainWindow.
    void SetLiveCompile(const cyxwiz::LiveGraphCompile* live_compile) { live_compile_ = live_compile; }

    // Opens a registered dataset (a step's output) in the Table Viewer
    // (TOFIX146 "Open in Table Viewer"); `title` names the tab.
    using OpenTableCallback = std::function<void(const std::string& dataset_name, const std::string& title)>;
    void SetOpenTableCallback(OpenTableCallback callback) { open_table_ = std::move(callback); }

    void SetBackendPlacementFacts(
        std::vector<properties_truth::BackendPlacementTruthFact> facts);
    void ClearBackendPlacementFacts();

private:
    void RenderNodeProperties(MLNode& node);
    // General, Compiled, Fusion, Truth, then (unless dialog-only) Parameters,
    // Advanced, Presets and Executor.
    void RenderGenericSections(MLNode& node, const cyxwiz::NodeMetadata* metadata, bool is_dialog_only);

    // Enhanced property sections (Phase 3)
    void RenderGeneralSection(MLNode& node);
    void RenderTruthSummarySection(MLNode& node);
    void RenderParametersSection(MLNode& node, const cyxwiz::NodeMetadata* metadata);
    void RenderPresetsSection(MLNode& node);
    // Word + POS fusion card on a Concatenate (properties_sequence_fusion.cpp).
    void RenderSequenceFusionSection(MLNode& node);
    // AS COMPILED card: the selected node as the compiler built it
    // (properties_compiled_card.cpp, TOFIX123).
    void RenderCompiledSection(MLNode& node);
    bool AnyDataInputLoaded() const;

    // Node executor integration (Phase: Node Executor Architecture)
    void RenderExecutorSection(MLNode& node);

    NodeShapeInfo ComputeNodeShape(int node_id);

    bool show_window_;
    MLNode* selected_node_ = nullptr;
    NodeEditor* node_editor_ = nullptr;

    // Shape caching
    bool shapes_valid_ = false;
    std::map<int, NodeShapeInfo> cached_shapes_;
    std::vector<properties_truth::BackendPlacementTruthFact> backend_placement_facts_;

    // Scope data buffers (node_id -> time/value ring buffer)
    std::map<int, properties_node_editors::ScopeBuffer> scope_buffers_;

    // Phase 3: Enhanced properties state
    std::map<std::string, std::string> validation_errors_;  // param_name -> error message
    bool section_general_open_ = true;
    bool section_parameters_open_ = true;
    bool section_advanced_open_ = false;
    bool section_presets_open_ = false;
    char preset_name_buffer_[64] = {};
    std::set<int> fusion_details_open_;  // Concatenate ids with Details shown
    std::set<int> compiled_details_open_;  // node ids with AS COMPILED Details shown
    const cyxwiz::LiveGraphCompile* live_compile_ = nullptr;

    // KNIME-style configuration dialogs
    std::unique_ptr<NodeConfigDialog> active_dialog_;
    void RenderOpenDialogButton(MLNode& node);

    // SQL step (TOFIX146 board 1, gui/properties_sql_step.cpp): replaces the
    // generic sections for SQLQuery nodes; they stay under its Details.
    struct SqlStepView;
    std::unique_ptr<SqlStepView> sql_view_;
    OpenTableCallback open_table_;
    void RenderSqlStep(MLNode& node, const cyxwiz::NodeMetadata* metadata);
    void RenderSqlStepDetails(MLNode& node, const cyxwiz::NodeMetadata* metadata);
};

} // namespace gui
