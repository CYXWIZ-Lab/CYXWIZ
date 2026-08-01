#pragma once

#include <string>
#include <map>
#include <vector>
#include <deque>
#include <functional>
#include <memory>
#include "../core/node_metadata.h"
#include "node_config_dialog.h"
#include "properties_node_editors.h"
#include "properties_shape_info.h"
#include "properties_truth.h"

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

    void SetBackendPlacementFacts(
        std::vector<properties_truth::BackendPlacementTruthFact> facts);
    void ClearBackendPlacementFacts();

private:
    void RenderNodeProperties(MLNode& node);

    // Enhanced property sections (Phase 3)
    void RenderGeneralSection(MLNode& node);
    void RenderTruthSummarySection(MLNode& node);
    void RenderParametersSection(MLNode& node, const cyxwiz::NodeMetadata* metadata);
    void RenderPresetsSection(MLNode& node);

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
    float scope_demo_time_ = 0.0f;  // Demo animation timer

    // Phase 3: Enhanced properties state
    std::map<std::string, std::string> validation_errors_;  // param_name -> error message
    bool section_general_open_ = true;
    bool section_parameters_open_ = true;
    bool section_advanced_open_ = false;
    bool section_presets_open_ = false;
    char preset_name_buffer_[64] = {};

    // KNIME-style configuration dialogs
    std::unique_ptr<NodeConfigDialog> active_dialog_;
    void RenderOpenDialogButton(MLNode& node);
};

} // namespace gui
