#pragma once

#include <chrono>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>
#include "../core/node_metadata.h"
#include "../core/properties_presentation.h"
#include "node_config_dialog.h"
#include "properties_node_editors.h"
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

// The Properties panel (TOFIX129 A7): draws a PropertiesView
// (core/properties_presentation) that is rebuilt only when the selected
// node, the graph, the compile result or an edit changes.
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

    // An editor changed the node: the view is rebuilt on the next frame.
    void NotifyEdited() { ++edit_serial_; }

    // The background compile of the canvas (TOFIX123); the AS COMPILED card
    // reads it. Owned by MainWindow.
    void SetLiveCompile(const cyxwiz::LiveGraphCompile* live_compile) { live_compile_ = live_compile; }

    void SetBackendPlacementFacts(
        std::vector<properties_truth::BackendPlacementTruthFact> facts);
    void ClearBackendPlacementFacts();

private:
    using View = cyxwiz::properties_view::View;

    // Rebuilds view_ when its key changed (properties.cpp).
    void RefreshView(MLNode& node);
    void RefreshDatasetFacts();

    void RenderEmpty();
    void RenderHeader(MLNode& node);
    void RenderActionsMenu(MLNode& node);
    void RenderSettings(MLNode& node);
    void RenderMetadataRows(MLNode& node);
    void RenderMetadataRow(MLNode& node, const cyxwiz::properties_view::Setting& row);
    void RenderLooseTruths(MLNode& node);
    void RenderViewSettings(MLNode& node);  // Plot / Dashboard nodes: saved settings in words
    void RenderNodeProperties(MLNode& node);  // the custom and fallback editors
    void RenderAdvanced(MLNode& node);
    // The cards (properties_compiled_card.cpp, properties_sequence_fusion.cpp).
    void RenderCompiledCard(MLNode& node);
    void RenderFusionCard(MLNode& node);
    // Node executor integration (K-Means)
    void RenderExecutorSection(MLNode& node);

    bool show_window_;
    MLNode* selected_node_ = nullptr;
    NodeEditor* node_editor_ = nullptr;
    const cyxwiz::LiveGraphCompile* live_compile_ = nullptr;

    // The view and what it was built from.
    View view_;
    std::optional<cyxwiz::NodeMetadata> view_metadata_;  // the view's rows point into it
    cyxwiz::properties_view::Key view_key_;
    bool view_valid_ = false;
    uint64_t edit_serial_ = 0;
    // Dataset facts of the Data Input nodes (registry probes), refreshed
    // at most once a second.
    std::vector<properties_truth::DatasetTruthFact> dataset_facts_;
    uint64_t facts_serial_ = 0;
    std::chrono::steady_clock::time_point facts_time_{};
    std::vector<properties_truth::BackendPlacementTruthFact> backend_placement_facts_;

    // Per-frame: the truths drawn on a row (the rest is listed read-only).
    std::set<std::string> shown_truth_keys_;

    // Scope data buffers (node_id -> time/value ring buffer)
    std::map<int, properties_node_editors::ScopeBuffer> scope_buffers_;

    // Section and disclosure state
    bool section_settings_open_ = true;
    bool section_advanced_open_ = false;
    std::set<int> settings_details_open_;   // node ids with the rows' provenance shown
    std::set<int> fusion_details_open_;     // Concatenate ids with Details shown
    std::set<int> compiled_details_open_;   // node ids with AS COMPILED Details shown
    char name_buffer_[128] = {};
    int name_buffer_node_ = -1;
    char preset_name_buffer_[64] = {};

    // KNIME-style configuration dialogs
    std::unique_ptr<NodeConfigDialog> active_dialog_;
};

} // namespace gui
