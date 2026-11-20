#pragma once

#include "../../core/extension_node_presentation.h"
#include "../../core/node_metadata.h"
#include "../../core/nodes_panel_presentation.h"
#include <functional>
#include <set>
#include <string>
#include <vector>
#include <imgui.h>

namespace gui {

class NodeEditor;  // Forward declaration

/**
 * The Nodes panel (TOFIX129, approved mockup NP-1..NP-8): a search with a
 * clear button, a filter in plain words, a grid or list switch; Studio Tools
 * in one row; Favourites and Recent as chips; the node groups fold, with
 * their counts, and the panel remembers which are open; Plugins, Advanced
 * operations, Can't run yet and Examples fold at the end. A card or a row is
 * added by double-click or drag; a click shows it in the Info panel. What it
 * shows comes from core/nodes_panel_presentation.
 */
class NodeBrowserPanel {
public:
    NodeBrowserPanel();
    ~NodeBrowserPanel() = default;

    void SetNodeEditor(NodeEditor* editor) { node_editor_ = editor; }

    // A card hovered or clicked (the Info panel previews / keeps it).
    void SetNodeHoverCallback(std::function<void(cyxwiz::NodeType)> callback) { on_node_hover_ = std::move(callback); }
    void SetNodeSelectCallback(std::function<void(cyxwiz::NodeType)> callback) { on_node_select_ = std::move(callback); }
    // Extension nodes (TOFIX125) share one NodeType: these carry their type
    // id "<provider_id>:<TypeName>".
    void SetExtensionHoverCallback(std::function<void(const std::string&)> callback) {
        on_extension_hover_ = std::move(callback);
    }
    void SetExtensionSelectCallback(std::function<void(const std::string&)> callback) {
        on_extension_select_ = std::move(callback);
    }
    // An example double-clicked under Examples: opened in the Pattern Browser
    // (its settings, the question before it replaces a graph, its result).
    void SetExampleOpenCallback(std::function<void(const std::string&)> callback) {
        on_example_open_ = std::move(callback);
    }

    void Render();

    // Apps > a loaded plugin or Python bundle (TOFIX125): the Plugins group
    // open and narrowed to that provider, the window shown and focused.
    void ShowPluginNodes(const std::string& provider_id);

    void SetVisible(bool visible) { visible_ = visible; }
    bool* GetVisiblePtr() { return &visible_; }

private:
    // Top rows
    void RenderSearchRow();
    void RenderFilterRow(size_t count);
    void HandleKeys();

    // Browsing
    void RenderStudioTools();
    void RenderFavourites();
    void RenderRecent();
    void RenderNodeChip(const cyxwiz::NodeMetadata* metadata);
    void RenderGroups(const std::vector<cyxwiz::nodes_panel::Group>& groups);
    void RenderPluginsGroup();
    void RenderAdvancedGroup();
    void RenderCantRunGroup();
    void RenderExamplesGroup();
    void RenderSearchResults(const cyxwiz::nodes_panel::SearchResult& found,
                             const std::vector<cyxwiz::ExtensionPaletteEntry>& extension_results);
    // A folding header: open is remembered under `name` (forced open while a
    // filter narrows the list).
    bool FoldHeader(const std::string& name, size_t count, const ImVec4& swatch, const char* note = nullptr);

    // Nodes as cards (grid) or rows (list), with the same actions.
    void RenderNodes(const std::vector<const cyxwiz::NodeMetadata*>& nodes);
    void RenderExtensionNodes(const std::vector<cyxwiz::ExtensionPaletteEntry>& entries);
    void RenderNodeItem(const cyxwiz::NodeMetadata* metadata, float width,
                        const cyxwiz::ExtensionPaletteEntry* extension = nullptr);
    void DrawCard(const cyxwiz::NodeMetadata* metadata, ImVec2 min, float width, bool can_add, bool has_star,
                  bool star_hovered, const cyxwiz::ExtensionPaletteEntry* extension);
    void DrawRow(const cyxwiz::NodeMetadata* metadata, ImVec2 min, float width, bool can_add, bool has_star,
                 bool star_hovered, const cyxwiz::ExtensionPaletteEntry* extension);
    void DrawIcon(const cyxwiz::NodeMetadata* metadata, ImVec2 min, float size,
                  const cyxwiz::ExtensionPaletteEntry* extension);
    // `footer` replaces the model's (a chip adds on a click, not a double-click).
    void RenderNodeTooltip(const cyxwiz::NodeMetadata* metadata, const char* footer = nullptr);

    void SyncUserNodes();
    void LoadPanelSettings();
    void SaveOpenGroups();
    void ClearSearch();
    void AddNode(const cyxwiz::NodeMetadata* metadata, const cyxwiz::ExtensionPaletteEntry* extension = nullptr);
    ImU32 GetNodeColor(cyxwiz::NodeCategory category) const;

    bool visible_ = true;
    char search_[256] = {};
    bool focus_search_next_frame_ = false;
    bool focus_window_next_frame_ = false;
    cyxwiz::nodes_panel::Filter filter_ = cyxwiz::nodes_panel::Filter::All;
    std::string provider_filter_;  // the Plugins group shows only this provider
    bool scroll_to_plugins_ = false;  // Apps > a plugin: bring the Plugins group into view
    int examples_count_ = -1;         // the pattern library's size, counted once

    // Saved with the Engine settings (EngineConfig nodes_panel).
    bool settings_loaded_ = false;
    bool list_view_ = false;  // this session's view; Preferences sets the default
    std::set<std::string> open_groups_;
    bool user_nodes_loaded_ = false;
    std::vector<int> saved_favourites_;
    std::vector<int> saved_recent_;

    NodeEditor* node_editor_ = nullptr;
    std::function<void(cyxwiz::NodeType)> on_node_hover_;
    std::function<void(cyxwiz::NodeType)> on_node_select_;
    std::function<void(const std::string&)> on_extension_hover_;
    std::function<void(const std::string&)> on_extension_select_;
    std::function<void(const std::string&)> on_example_open_;
    cyxwiz::NodeType pinned_node_type_ = cyxwiz::NodeType::Unknown;
    std::string pinned_extension_id_;

    // Layout
    static constexpr float PANEL_MIN_WIDTH = 280.0f;
    static constexpr float NODE_CARD_HEIGHT = 90.0f;  // icon, then a name on up to two lines
    static constexpr float NODE_ICON_SIZE = 40.0f;
    static constexpr float ROW_ICON_SIZE = 20.0f;
    static constexpr float GRID_SPACING = 8.0f;
    // As many columns as cards of at least kMinCardWidth fit (a fixed 3 with a
    // 70 px minimum ran a third column past a narrow panel's edge).
    static constexpr float kMinCardWidth = 84.0f;
    static int ColumnsFor(float width) {
        const int n = static_cast<int>((width + GRID_SPACING) / (kMinCardWidth + GRID_SPACING));
        return n < 1 ? 1 : n;
    }
};

} // namespace gui
