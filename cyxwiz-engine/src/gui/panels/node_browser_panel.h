#pragma once

#include <string>
#include <vector>
#include <functional>
#include <unordered_map>
#include "../../core/node_metadata.h"

namespace gui {

class NodeEditor;  // Forward declaration

/**
 * NodeBrowserPanel - KNIME-style node palette for CyxWiz Studio
 *
 * Features:
 * - Grid-based node display (3 columns) with visual icons
 * - Port indicators showing inputs/outputs
 * - Categorized sections with "Show all" expansion
 * - Search with filtering across name/keywords/description
 * - Filter tags for subcategory filtering
 * - Breadcrumb navigation (Nodes > Category)
 * - Drag-and-drop node creation
 * - Template nodes with "Coming Soon" badges
 */
class NodeBrowserPanel {
public:
    NodeBrowserPanel();
    ~NodeBrowserPanel() = default;

    /**
     * Set reference to node editor for drag-drop node creation
     */
    void SetNodeEditor(NodeEditor* editor) { node_editor_ = editor; }

    /**
     * Set callback for node hover events (for Info Panel integration)
     */
    void SetNodeHoverCallback(std::function<void(cyxwiz::NodeType)> callback) {
        on_node_hover_ = callback;
    }

    /**
     * Render the panel
     */
    void Render();

    /**
     * Focus the search bar (for keyboard shortcuts)
     * Called when user presses / or Ctrl+F
     */
    void FocusSearchBar() { focus_search_next_frame_ = true; }

    /**
     * Clear search and navigate back to root
     */
    void ClearSearch();

    /**
     * Process keyboard shortcuts
     * Call from main window to handle global shortcuts
     * Returns true if shortcut was handled
     */
    bool HandleKeyboardShortcuts();

    /**
     * Check if panel is visible
     */
    bool IsVisible() const { return visible_; }

    /**
     * Toggle panel visibility
     */
    void SetVisible(bool visible) { visible_ = visible; }
    void ToggleVisible() { visible_ = !visible_; }
    bool* GetVisiblePtr() { return &visible_; }

private:
    // Render helpers
    void RenderHeader();
    void RenderStudioSection();    // CyxWiz Studio quick actions, examples, description
    void RenderSearchBar();
    void RenderFilterTags();
    void RenderCategorySection(cyxwiz::NodeCategory category);
    void RenderNodeGrid(const std::vector<const cyxwiz::NodeMetadata*>& nodes, int max_visible = -1);
    void RenderNodeCard(const cyxwiz::NodeMetadata* metadata, float card_width);
    void RenderNodeIcon(const cyxwiz::NodeMetadata* metadata, ImVec2 icon_pos, ImVec2 size);
    void RenderPortIndicators(const cyxwiz::NodeMetadata* metadata, ImVec2 icon_pos, ImVec2 icon_size);
    void RenderNodeTooltip(const cyxwiz::NodeMetadata* metadata);
    std::vector<const cyxwiz::NodeMetadata*> ApplySupportFilter(
        const std::vector<const cyxwiz::NodeMetadata*>& nodes) const;
    bool NodeMatchesSupportFilter(const cyxwiz::NodeMetadata* metadata) const;
    bool IsSupportBlocked(const cyxwiz::NodeMetadata* metadata) const;

    enum class SupportFilterMode {
        All,
        Runnable,
        Blocked,
    };
    const char* GetSupportFilterLabel(SupportFilterMode mode) const;

    // Node creation
    void CreateNodeAtMouse(const cyxwiz::NodeMetadata* metadata);

    // Navigation
    void NavigateToCategory(cyxwiz::NodeCategory category);
    void NavigateBack();

    // Get color for node category
    ImU32 GetNodeColor(cyxwiz::NodeCategory category) const;
    ImU32 GetNodeColorDark(cyxwiz::NodeCategory category) const;

    // State
    bool visible_ = true;
    std::string search_query_;
    bool focus_search_next_frame_ = false;  // Focus search bar on next frame
    bool studio_section_expanded_ = true;   // CyxWiz Studio section state

    // Navigation state (breadcrumb)
    bool showing_all_in_category_ = false;
    cyxwiz::NodeCategory current_category_ = cyxwiz::NodeCategory::Unknown;

    // Category expansion state (for collapsed view)
    std::unordered_map<cyxwiz::NodeCategory, bool> category_show_all_;

    SupportFilterMode support_filter_mode_ = SupportFilterMode::All;

    // Drag-drop state
    const cyxwiz::NodeMetadata* dragging_node_ = nullptr;

    // Reference to node editor
    NodeEditor* node_editor_ = nullptr;

    // Callback for node hover
    std::function<void(cyxwiz::NodeType)> on_node_hover_;

    // Layout constants
    static constexpr float PANEL_MIN_WIDTH = 280.0f;
    static constexpr float NODE_CARD_WIDTH = 85.0f;
    static constexpr float NODE_CARD_HEIGHT = 75.0f;
    static constexpr float NODE_ICON_SIZE = 40.0f;
    static constexpr float GRID_SPACING = 8.0f;
    static constexpr int GRID_COLUMNS = 3;
    static constexpr int VISIBLE_ROWS_COLLAPSED = 3;  // Show 3 rows (9 nodes) when collapsed
};

} // namespace gui
