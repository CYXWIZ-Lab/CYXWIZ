#include "node_browser_panel.h"
#include "../node_editor.h"
#include "../../core/node_metadata_registry.h"
#include "../patterns/pattern_library.h"
#include "../icons.h"
#include <imgui.h>
#include <imgui_internal.h>
#include <algorithm>
#include <cmath>

namespace gui {

NodeBrowserPanel::NodeBrowserPanel() {
    // Initialize category show_all state to false (collapsed)
    auto& registry = cyxwiz::NodeMetadataRegistry::Instance();
    if (!registry.IsInitialized()) {
        registry.Initialize();
    }
    for (auto category : registry.GetCategories()) {
        category_show_all_[category] = false;
    }
}

void NodeBrowserPanel::Render() {
    if (!visible_) return;

    auto& registry = cyxwiz::NodeMetadataRegistry::Instance();
    if (!registry.IsInitialized()) {
        registry.Initialize();
    }

    ImGui::SetNextWindowSizeConstraints(ImVec2(PANEL_MIN_WIDTH, 300), ImVec2(450, FLT_MAX));

    if (ImGui::Begin("Nodes", &visible_)) {
        RenderHeader();
        RenderSearchBar();

        // Show filter tags when in a category view
        if (showing_all_in_category_) {
            RenderFilterTags();
        }

        ImGui::Separator();

        // Main content area with scrolling
        ImGui::BeginChild("NodeGrid", ImVec2(0, 0), false);

        // CyxWiz Studio section (always at top when not searching/in category)
        if (search_query_.empty() && !showing_all_in_category_) {
            RenderStudioSection();
            ImGui::Spacing();
        }

        if (!search_query_.empty()) {
            // Search results mode - show as grid
            auto results = registry.Search(search_query_, true);
            if (results.empty()) {
                ImGui::TextDisabled("No nodes found matching '%s'", search_query_.c_str());
            } else {
                RenderNodeGrid(results);
            }
        } else if (showing_all_in_category_) {
            // Showing all nodes in a specific category
            auto nodes = registry.GetByCategory(current_category_, true);
            RenderNodeGrid(nodes);
        } else {
            // Normal category view
            auto categories = registry.GetCategories();
            for (auto category : categories) {
                RenderCategorySection(category);
            }
        }

        ImGui::EndChild();
    }
    ImGui::End();
}

void NodeBrowserPanel::RenderHeader() {
    // Breadcrumb navigation: "Nodes" or "Nodes > CategoryName"
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.7f, 0.7f, 1.0f));

    if (showing_all_in_category_) {
        // Clickable "Nodes" to go back
        if (ImGui::SmallButton("Nodes")) {
            NavigateBack();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Go back to all categories (Esc)");
        }
        ImGui::SameLine();
        ImGui::TextUnformatted(">");
        ImGui::SameLine();
        ImGui::TextUnformatted(cyxwiz::GetCategoryDisplayName(current_category_).c_str());
    } else {
        ImGui::TextUnformatted("Nodes");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Node Browser - Drag nodes to canvas\nPress / to search");
        }
    }

    ImGui::PopStyleColor();

    // Grid/List toggle button (top right) - placeholder for future
    ImGui::SameLine(ImGui::GetContentRegionAvail().x - 20);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::SmallButton(ICON_FA_GRIP);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Toggle grid/list view (Coming Soon)");
    }
    ImGui::PopStyleColor();
}

void NodeBrowserPanel::RenderSearchBar() {
    ImGui::PushItemWidth(-1);

    char search_buf[256] = {0};
    strncpy(search_buf, search_query_.c_str(), sizeof(search_buf) - 1);

    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8, 6));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);

    // Focus search bar if requested (from keyboard shortcut)
    if (focus_search_next_frame_) {
        ImGui::SetKeyboardFocusHere();
        focus_search_next_frame_ = false;
    }

    if (ImGui::InputTextWithHint("##NodeSearch", ICON_FA_SEARCH " Search all nodes (press /)",
                                  search_buf, sizeof(search_buf))) {
        search_query_ = search_buf;
        // Clear category view when searching
        if (!search_query_.empty()) {
            showing_all_in_category_ = false;
        }
    }

    // Tooltip for search bar
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Search nodes by name, keywords, or description\nShortcut: / or Ctrl+F");
    }

    ImGui::PopStyleVar(2);
    ImGui::PopItemWidth();
}

void NodeBrowserPanel::ClearSearch() {
    search_query_.clear();
    showing_all_in_category_ = false;
    current_category_ = cyxwiz::NodeCategory::Unknown;
}

bool NodeBrowserPanel::HandleKeyboardShortcuts() {
    // Only handle shortcuts when panel is visible and no text input is active
    if (!visible_) return false;

    // Check if any text input is active
    if (ImGui::GetIO().WantTextInput) return false;

    // / key - Focus search bar
    if (ImGui::IsKeyPressed(ImGuiKey_Slash) && !ImGui::GetIO().KeyCtrl) {
        FocusSearchBar();
        return true;
    }

    // Ctrl+F - Focus search bar
    if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_F)) {
        FocusSearchBar();
        return true;
    }

    // Escape - Clear search and go back
    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        if (!search_query_.empty() || showing_all_in_category_) {
            ClearSearch();
            return true;
        }
    }

    return false;
}

void NodeBrowserPanel::RenderStudioSection() {
    // CyxWiz Studio section header (same style as category headers)
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
    ImGui::TextUnformatted("Studio Tools");

    // Horizontal separator line
    ImVec2 line_start = ImGui::GetCursorScreenPos();
    line_start.y -= 2;
    float text_width = ImGui::CalcTextSize("Studio Tools").x + 8;
    ImVec2 line_end = ImVec2(line_start.x + ImGui::GetContentRegionAvail().x, line_start.y);
    line_start.x += text_width;
    ImGui::GetWindowDrawList()->AddLine(line_start, line_end, IM_COL32(100, 100, 100, 128), 1.0f);
    ImGui::PopStyleColor();

    // Calculate card dimensions (same as node cards)
    float available_width = ImGui::GetContentRegionAvail().x;
    float card_width = (available_width - (GRID_COLUMNS - 1) * GRID_SPACING) / GRID_COLUMNS;
    card_width = std::max(card_width, 70.0f);

    // === Note Card ===
    {
        ImGui::PushID("StudioNote");
        ImVec2 card_size(card_width, NODE_CARD_HEIGHT);
        ImVec2 cursor_start = ImGui::GetCursorScreenPos();

        ImGui::InvisibleButton("##notecard", card_size);
        bool hovered = ImGui::IsItemHovered();

        // Double-click to add annotation
        if (hovered && ImGui::IsMouseDoubleClicked(0)) {
            if (node_editor_) {
                node_editor_->AddAnnotation();
            }
        }

        // Drag-drop source for annotation
        if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
            int annotation_type = 1;  // Just a marker value
            ImGui::SetDragDropPayload("ANNOTATION", &annotation_type, sizeof(int));
            ImGui::Text(ICON_FA_COMMENT " Annotation");
            ImGui::EndDragDropSource();
        }

        // Tooltip
        if (hovered) {
            ImGui::BeginTooltip();
            ImGui::TextColored(ImVec4(0.9f, 0.9f, 0.9f, 1.0f), ICON_FA_COMMENT " Annotation");
            ImGui::TextWrapped("Add a sticky note to the canvas for documentation");
            ImGui::Separator();
            ImGui::TextDisabled("Drag to canvas or double-click to add");
            ImGui::TextDisabled("Shortcut: Ctrl+Shift+N");
            ImGui::EndTooltip();
        }

        // Draw card
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        float icon_x = cursor_start.x + (card_width - NODE_ICON_SIZE) * 0.5f;
        float icon_y = cursor_start.y + 4;
        ImVec2 icon_pos(icon_x, icon_y);
        ImVec2 p_max(icon_pos.x + NODE_ICON_SIZE, icon_pos.y + NODE_ICON_SIZE);

        // Yellow background for note
        ImU32 bg_color = IM_COL32(200, 180, 100, 255);
        ImU32 border_color = IM_COL32(150, 130, 70, 255);
        draw_list->AddRectFilled(icon_pos, p_max, bg_color, 6.0f);
        draw_list->AddRect(icon_pos, p_max, border_color, 6.0f, 0, 2.0f);

        // Icon
        ImFont* icon_font = ImGui::GetFont();
        float scaled_icon_size = NODE_ICON_SIZE * 0.55f;
        const char* icon = ICON_FA_COMMENT;
        ImVec2 icon_text_size = icon_font->CalcTextSizeA(scaled_icon_size, FLT_MAX, 0.0f, icon);
        ImVec2 icon_text_pos(icon_pos.x + (NODE_ICON_SIZE - icon_text_size.x) * 0.5f,
                            icon_pos.y + (NODE_ICON_SIZE - icon_text_size.y) * 0.5f);
        draw_list->AddText(icon_font, scaled_icon_size, icon_text_pos, IM_COL32(255, 255, 255, 255), icon);

        // Label
        const char* label = "Note";
        ImVec2 text_size = ImGui::CalcTextSize(label);
        float text_x = cursor_start.x + (card_width - text_size.x) * 0.5f;
        float text_y = icon_y + NODE_ICON_SIZE + 4;
        draw_list->AddText(ImVec2(text_x, text_y), IM_COL32(220, 220, 220, 255), label);

        // Hover highlight
        if (hovered) {
            draw_list->AddRect(cursor_start,
                ImVec2(cursor_start.x + card_width, cursor_start.y + card_size.y),
                IM_COL32(100, 150, 255, 100), 4.0f);
        }
        ImGui::PopID();
    }

    ImGui::SameLine(0, GRID_SPACING);

    // === Group Card ===
    {
        ImGui::PushID("StudioGroup");
        ImVec2 card_size(card_width, NODE_CARD_HEIGHT);
        ImVec2 cursor_start = ImGui::GetCursorScreenPos();

        ImGui::InvisibleButton("##groupcard", card_size);
        bool hovered = ImGui::IsItemHovered();

        // Double-click to group selected nodes
        if (hovered && ImGui::IsMouseDoubleClicked(0)) {
            if (node_editor_) {
                node_editor_->GroupSelectedNodes();
            }
        }

        // Tooltip
        if (hovered) {
            ImGui::BeginTooltip();
            ImGui::TextColored(ImVec4(0.9f, 0.9f, 0.9f, 1.0f), ICON_FA_OBJECT_GROUP " Group");
            ImGui::TextWrapped("Group selected nodes together for organization");
            ImGui::Separator();
            ImGui::TextDisabled("Select nodes first, then double-click");
            ImGui::TextDisabled("Shortcut: Ctrl+G");
            ImGui::EndTooltip();
        }

        // Draw card
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        float icon_x = cursor_start.x + (card_width - NODE_ICON_SIZE) * 0.5f;
        float icon_y = cursor_start.y + 4;
        ImVec2 icon_pos(icon_x, icon_y);
        ImVec2 p_max(icon_pos.x + NODE_ICON_SIZE, icon_pos.y + NODE_ICON_SIZE);

        // Blue background for group
        ImU32 bg_color = IM_COL32(100, 140, 180, 255);
        ImU32 border_color = IM_COL32(70, 100, 140, 255);
        draw_list->AddRectFilled(icon_pos, p_max, bg_color, 6.0f);
        draw_list->AddRect(icon_pos, p_max, border_color, 6.0f, 0, 2.0f);

        // Icon
        ImFont* icon_font = ImGui::GetFont();
        float scaled_icon_size = NODE_ICON_SIZE * 0.55f;
        const char* icon = ICON_FA_OBJECT_GROUP;
        ImVec2 icon_text_size = icon_font->CalcTextSizeA(scaled_icon_size, FLT_MAX, 0.0f, icon);
        ImVec2 icon_text_pos(icon_pos.x + (NODE_ICON_SIZE - icon_text_size.x) * 0.5f,
                            icon_pos.y + (NODE_ICON_SIZE - icon_text_size.y) * 0.5f);
        draw_list->AddText(icon_font, scaled_icon_size, icon_text_pos, IM_COL32(255, 255, 255, 255), icon);

        // Label
        const char* label = "Group";
        ImVec2 text_size = ImGui::CalcTextSize(label);
        float text_x = cursor_start.x + (card_width - text_size.x) * 0.5f;
        float text_y = icon_y + NODE_ICON_SIZE + 4;
        draw_list->AddText(ImVec2(text_x, text_y), IM_COL32(220, 220, 220, 255), label);

        // Hover highlight
        if (hovered) {
            draw_list->AddRect(cursor_start,
                ImVec2(cursor_start.x + card_width, cursor_start.y + card_size.y),
                IM_COL32(100, 150, 255, 100), 4.0f);
        }
        ImGui::PopID();
    }

    ImGui::Spacing();
    ImGui::Spacing();

    // === Examples Section ===
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
    ImGui::TextUnformatted("Examples");

    // Horizontal separator line
    ImVec2 examples_line_start = ImGui::GetCursorScreenPos();
    examples_line_start.y -= 2;
    float examples_text_width = ImGui::CalcTextSize("Examples").x + 8;
    ImVec2 examples_line_end = ImVec2(examples_line_start.x + ImGui::GetContentRegionAvail().x, examples_line_start.y);
    examples_line_start.x += examples_text_width;
    ImGui::GetWindowDrawList()->AddLine(examples_line_start, examples_line_end, IM_COL32(100, 100, 100, 128), 1.0f);
    ImGui::PopStyleColor();

    // Get pattern library
    auto& pattern_lib = patterns::PatternLibrary::Instance();
    if (!pattern_lib.IsInitialized()) {
        pattern_lib.Initialize();
    }

    auto pattern_categories = pattern_lib.GetAvailableCategories();
    for (auto cat : pattern_categories) {
        auto cat_patterns = pattern_lib.GetByCategory(cat);
        if (cat_patterns.empty()) continue;

        // Category tree node
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth;
        const char* cat_icon = patterns::GetPatternCategoryIcon(cat);
        std::string cat_label = std::string(cat_icon) + " " + patterns::PatternCategoryToString(cat);

        if (ImGui::TreeNodeEx(cat_label.c_str(), flags)) {
            for (const auto& pattern : cat_patterns) {
                ImGui::PushID(pattern.id.c_str());

                // Selectable item for each pattern
                bool selected = false;
                if (ImGui::Selectable(pattern.name.c_str(), &selected, ImGuiSelectableFlags_AllowDoubleClick)) {
                    if (ImGui::IsMouseDoubleClicked(0)) {
                        // Load pattern into canvas
                        if (node_editor_) {
                            std::vector<MLNode> nodes;
                            std::vector<NodeLink> links;
                            std::map<std::string, std::string> params;  // Default params
                            static int base_id = 20000;
                            int next_node_id = base_id;
                            int next_link_id = base_id * 100;
                            ImVec2 base_pos(200.0f, 200.0f);

                            patterns::NodeCreatorCallback creator = [this](NodeType type, const std::string& name) {
                                return node_editor_->CreateNode(type, name);
                            };

                            if (pattern_lib.InstantiatePatternWithCreator(
                                pattern.id, params, nodes, links, next_node_id, next_link_id, base_pos, creator)) {
                                node_editor_->InsertPattern(nodes, links);
                                base_id = next_node_id + 100;
                            }
                        }
                    }
                }

                // Tooltip
                if (ImGui::IsItemHovered()) {
                    ImGui::BeginTooltip();
                    ImGui::TextColored(ImVec4(0.9f, 0.9f, 0.9f, 1.0f), "%s", pattern.name.c_str());
                    if (!pattern.description.empty()) {
                        ImGui::TextWrapped("%s", pattern.description.c_str());
                    }
                    ImGui::Separator();
                    ImGui::TextDisabled("Double-click to load");
                    ImGui::EndTooltip();
                }

                ImGui::PopID();
            }
            ImGui::TreePop();
        }
    }

    if (pattern_categories.empty()) {
        ImGui::TextDisabled("No patterns available");
    }

    ImGui::Spacing();
    ImGui::Spacing();
}

void NodeBrowserPanel::RenderFilterTags() {
    auto& registry = cyxwiz::NodeMetadataRegistry::Instance();

    // Category dropdown
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6, 3));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);

    std::string cat_name = cyxwiz::GetCategoryDisplayName(current_category_);
    if (ImGui::BeginCombo("##CatFilter", cat_name.c_str(), ImGuiComboFlags_NoArrowButton)) {
        for (auto category : registry.GetCategories()) {
            bool selected = (category == current_category_);
            if (ImGui::Selectable(cyxwiz::GetCategoryDisplayName(category).c_str(), selected)) {
                NavigateToCategory(category);
            }
        }
        ImGui::EndCombo();
    }

    // Additional filter tags could be added here based on keywords
    // For now, show count of remaining filters
    auto nodes = registry.GetByCategory(current_category_, true);
    if (nodes.size() > 9) {
        ImGui::SameLine();
        ImGui::TextDisabled("+%zu more", nodes.size() - 9);
    }

    ImGui::PopStyleVar(2);
}

void NodeBrowserPanel::RenderCategorySection(cyxwiz::NodeCategory category) {
    auto& registry = cyxwiz::NodeMetadataRegistry::Instance();
    auto nodes = registry.GetByCategory(category, true);

    if (nodes.empty()) return;

    // Category header with horizontal line
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));

    std::string cat_name = cyxwiz::GetCategoryDisplayName(category);
    ImGui::TextUnformatted(cat_name.c_str());

    // Horizontal separator line
    ImVec2 line_start = ImGui::GetCursorScreenPos();
    line_start.y -= 2;
    float text_width = ImGui::CalcTextSize(cat_name.c_str()).x + 8;
    ImVec2 line_end = ImVec2(line_start.x + ImGui::GetContentRegionAvail().x, line_start.y);
    line_start.x += text_width;
    ImGui::GetWindowDrawList()->AddLine(line_start, line_end,
                                        IM_COL32(100, 100, 100, 128), 1.0f);

    ImGui::PopStyleColor();

    // Determine how many nodes to show
    int max_visible = category_show_all_[category] ? -1 : (GRID_COLUMNS * VISIBLE_ROWS_COLLAPSED);

    // Render nodes in grid
    RenderNodeGrid(nodes, max_visible);

    // "Show all" button if there are more nodes
    if (!category_show_all_[category] && (int)nodes.size() > (GRID_COLUMNS * VISIBLE_ROWS_COLLAPSED)) {
        float button_x = ImGui::GetContentRegionAvail().x - 60;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + button_x);

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));

        if (ImGui::SmallButton(("Show all##" + cat_name).c_str())) {
            NavigateToCategory(category);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("View all %zu nodes in %s category", nodes.size(), cat_name.c_str());
        }

        // Circle arrow icon next to "Show all"
        ImGui::SameLine();
        ImGui::TextUnformatted(ICON_FA_CHEVRON_RIGHT);

        ImGui::PopStyleColor(2);
    }

    ImGui::Spacing();
    ImGui::Spacing();
}

void NodeBrowserPanel::RenderNodeGrid(const std::vector<const cyxwiz::NodeMetadata*>& nodes, int max_visible) {
    if (nodes.empty()) return;

    float available_width = ImGui::GetContentRegionAvail().x;
    float card_width = (available_width - (GRID_COLUMNS - 1) * GRID_SPACING) / GRID_COLUMNS;
    card_width = std::max(card_width, 70.0f);  // Minimum width

    int visible_count = (max_visible > 0) ? std::min((int)nodes.size(), max_visible) : (int)nodes.size();

    for (int i = 0; i < visible_count; ++i) {
        if (i % GRID_COLUMNS != 0) {
            ImGui::SameLine(0, GRID_SPACING);
        }

        RenderNodeCard(nodes[i], card_width);
    }
}

void NodeBrowserPanel::RenderNodeCard(const cyxwiz::NodeMetadata* metadata, float card_width) {
    if (!metadata) return;

    auto& registry = cyxwiz::NodeMetadataRegistry::Instance();

    ImGui::PushID(static_cast<int>(metadata->type));

    ImVec2 card_size(card_width, NODE_CARD_HEIGHT);
    ImVec2 cursor_start = ImGui::GetCursorScreenPos();

    // Invisible button for interaction
    ImGui::InvisibleButton("##card", card_size);

    bool hovered = ImGui::IsItemHovered();
    bool clicked = ImGui::IsItemClicked();

    // Handle double-click to create node
    if (hovered && ImGui::IsMouseDoubleClicked(0)) {
        if (!metadata->IsTemplate()) {
            CreateNodeAtMouse(metadata);
            registry.RecordUsage(metadata->type);
        }
    }

    // Handle drag-drop source (only for implemented nodes)
    if (!metadata->IsTemplate() && ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
        ImGui::SetDragDropPayload("NODE_TYPE", &metadata->type, sizeof(cyxwiz::NodeType));
        ImGui::Text("%s %s", metadata->icon.c_str(), metadata->name.c_str());
        dragging_node_ = metadata;
        ImGui::EndDragDropSource();
    }

    // Context menu (right-click)
    if (ImGui::BeginPopupContextItem("##NodeContextMenu")) {
        ImGui::TextDisabled("%s %s", metadata->icon.c_str(), metadata->name.c_str());
        ImGui::Separator();

        if (!metadata->IsTemplate()) {
            if (ImGui::MenuItem(ICON_FA_PLUS " Add to Canvas", "Double-click")) {
                CreateNodeAtMouse(metadata);
                registry.RecordUsage(metadata->type);
            }

            if (ImGui::MenuItem(ICON_FA_STAR " Toggle Favorite")) {
                registry.ToggleFavorite(metadata->type);
            }
        } else {
            ImGui::TextDisabled(ICON_FA_CLOCK " Coming Soon");
            if (ImGui::MenuItem(ICON_FA_THUMBS_UP " Vote for this feature")) {
                // TODO: Increment user_votes and save preferences
            }
        }

        ImGui::Separator();

        if (ImGui::MenuItem(ICON_FA_CIRCLE_INFO " View Documentation")) {
            // Trigger info panel to show this node
            if (on_node_hover_) {
                on_node_hover_(metadata->type);
            }
        }

        if (ImGui::MenuItem(ICON_FA_COPY " Copy Node Name")) {
            ImGui::SetClipboardText(metadata->name.c_str());
        }

        ImGui::EndPopup();
    }

    // Tooltip
    if (hovered && !ImGui::IsMouseDragging(0)) {
        RenderNodeTooltip(metadata);
    }

    // Draw the card
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // Icon position (centered horizontally)
    float icon_x = cursor_start.x + (card_width - NODE_ICON_SIZE) * 0.5f;
    float icon_y = cursor_start.y + 4;
    ImVec2 icon_pos(icon_x, icon_y);
    ImVec2 icon_size(NODE_ICON_SIZE, NODE_ICON_SIZE);

    // Draw node icon with color
    RenderNodeIcon(metadata, icon_pos, icon_size);

    // Draw port indicators
    RenderPortIndicators(metadata, icon_pos, icon_size);

    // Draw node name (centered, below icon)
    std::string display_name = metadata->name;
    if (display_name.length() > 12) {
        display_name = display_name.substr(0, 10) + "...";
    }

    ImVec2 text_size = ImGui::CalcTextSize(display_name.c_str());
    float text_x = cursor_start.x + (card_width - text_size.x) * 0.5f;
    float text_y = icon_y + NODE_ICON_SIZE + 4;

    // Dim text for template nodes
    ImU32 text_color = metadata->IsTemplate() ?
        IM_COL32(150, 150, 150, 255) : IM_COL32(220, 220, 220, 255);

    draw_list->AddText(ImVec2(text_x, text_y), text_color, display_name.c_str());

    // Draw "Coming Soon" badge for template nodes
    if (metadata->IsTemplate()) {
        // Draw small badge in top-right corner
        const char* badge_text = metadata->badge.empty() ? "Soon" : metadata->badge.c_str();
        // Truncate long badges
        std::string badge_display = badge_text;
        if (badge_display.length() > 6) {
            badge_display = badge_display.substr(0, 4) + "..";
        }

        ImVec2 badge_text_size = ImGui::CalcTextSize(badge_display.c_str());
        float badge_padding = 3.0f;
        float badge_width = badge_text_size.x + badge_padding * 2;
        float badge_height = badge_text_size.y + badge_padding;

        float badge_x = cursor_start.x + card_width - badge_width - 2;
        float badge_y = cursor_start.y + 2;

        // Badge background (amber/yellow for "Coming Soon")
        draw_list->AddRectFilled(
            ImVec2(badge_x, badge_y),
            ImVec2(badge_x + badge_width, badge_y + badge_height),
            IM_COL32(180, 140, 40, 230),
            3.0f
        );

        // Badge text
        draw_list->AddText(
            ImVec2(badge_x + badge_padding, badge_y + badge_padding * 0.5f),
            IM_COL32(255, 255, 255, 255),
            badge_display.c_str()
        );
    }

    // Highlight on hover
    if (hovered) {
        draw_list->AddRect(cursor_start,
                          ImVec2(cursor_start.x + card_width, cursor_start.y + card_size.y),
                          IM_COL32(100, 150, 255, 100), 4.0f);
        // Notify info panel of hover
        if (on_node_hover_) {
            on_node_hover_(metadata->type);
        }
    }

    ImGui::PopID();
}

void NodeBrowserPanel::RenderNodeIcon(const cyxwiz::NodeMetadata* metadata, ImVec2 icon_pos, ImVec2 size) {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // Use the passed icon position (calculated in RenderNodeCard)
    ImVec2 p_min = icon_pos;
    ImVec2 p_max(icon_pos.x + size.x, icon_pos.y + size.y);

    // Get category color
    ImU32 bg_color = GetNodeColor(metadata->category);
    ImU32 border_color = GetNodeColorDark(metadata->category);

    // Template nodes are grayed out
    if (metadata->IsTemplate()) {
        bg_color = IM_COL32(120, 120, 120, 255);
        border_color = IM_COL32(80, 80, 80, 255);
    }

    // Draw rounded rectangle background
    draw_list->AddRectFilled(p_min, p_max, bg_color, 6.0f);
    draw_list->AddRect(p_min, p_max, border_color, 6.0f, 0, 2.0f);

    // Draw icon character in center using the icon font
    if (!metadata->icon.empty()) {
        // FontAwesome icons are merged into the main font
        ImFont* icon_font = ImGui::GetFont();
        float scaled_icon_size = size.y * 0.55f;  // Icon takes 55% of box height
        ImVec2 icon_text_size = icon_font->CalcTextSizeA(scaled_icon_size, FLT_MAX, 0.0f, metadata->icon.c_str());
        ImVec2 icon_text_pos(
            p_min.x + (size.x - icon_text_size.x) * 0.5f,
            p_min.y + (size.y - icon_text_size.y) * 0.5f
        );
        draw_list->AddText(icon_font, scaled_icon_size, icon_text_pos, IM_COL32(255, 255, 255, 255), metadata->icon.c_str());
    }
}

void NodeBrowserPanel::RenderPortIndicators(const cyxwiz::NodeMetadata* metadata,
                                            ImVec2 icon_pos, ImVec2 icon_size) {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // Input ports (left side triangles pointing LEFT - into the node)
    if (!metadata->inputs.empty()) {
        float tri_size = 5.0f;
        float y_center = icon_pos.y + icon_size.y * 0.5f;

        // Draw small triangle pointing left (<)
        ImVec2 p1(icon_pos.x - 1, y_center);  // tip points left
        ImVec2 p2(icon_pos.x + tri_size - 1, y_center - tri_size);
        ImVec2 p3(icon_pos.x + tri_size - 1, y_center + tri_size);

        draw_list->AddTriangleFilled(p1, p2, p3, IM_COL32(80, 80, 80, 255));
    }

    // Output ports (right side triangles pointing RIGHT - out of the node)
    if (!metadata->outputs.empty()) {
        float tri_size = 5.0f;
        float y_center = icon_pos.y + icon_size.y * 0.5f;
        float x_right = icon_pos.x + icon_size.x;

        // Draw small triangle pointing right (>)
        ImVec2 p1(x_right - tri_size + 1, y_center - tri_size);
        ImVec2 p2(x_right - tri_size + 1, y_center + tri_size);
        ImVec2 p3(x_right + 1, y_center);  // tip points right

        draw_list->AddTriangleFilled(p1, p2, p3, IM_COL32(80, 80, 80, 255));
    }
}

void NodeBrowserPanel::RenderNodeTooltip(const cyxwiz::NodeMetadata* metadata) {
    ImGui::BeginTooltip();

    // Header with icon and name
    ImGui::TextColored(ImVec4(0.9f, 0.9f, 0.9f, 1.0f), "%s %s",
                       metadata->icon.c_str(), metadata->name.c_str());

    // Brief description
    if (!metadata->brief_description.empty()) {
        ImGui::TextWrapped("%s", metadata->brief_description.c_str());
    }

    ImGui::Separator();

    // Inputs/Outputs
    if (!metadata->inputs.empty()) {
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "Inputs:");
        for (const auto& input : metadata->inputs) {
            ImGui::BulletText("%s", input.name.c_str());
        }
    }

    if (!metadata->outputs.empty()) {
        ImGui::TextColored(ImVec4(0.8f, 0.4f, 0.4f, 1.0f), "Outputs:");
        for (const auto& output : metadata->outputs) {
            ImGui::BulletText("%s", output.name.c_str());
        }
    }

    // Status info for template nodes
    if (metadata->IsTemplate()) {
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.3f, 1.0f),
                          ICON_FA_CLOCK " Coming Soon");
        if (metadata->user_votes > 0) {
            ImGui::SameLine();
            ImGui::Text("(%d votes)", metadata->user_votes);
        }
    }

    // Usage hint
    ImGui::Separator();
    ImGui::TextDisabled("Double-click or drag to add");

    ImGui::EndTooltip();
}

void NodeBrowserPanel::NavigateToCategory(cyxwiz::NodeCategory category) {
    showing_all_in_category_ = true;
    current_category_ = category;
    category_show_all_[category] = true;
}

void NodeBrowserPanel::NavigateBack() {
    showing_all_in_category_ = false;
    current_category_ = cyxwiz::NodeCategory::Unknown;
    // Reset all category show_all states
    for (auto& pair : category_show_all_) {
        pair.second = false;
    }
}

void NodeBrowserPanel::CreateNodeAtMouse(const cyxwiz::NodeMetadata* metadata) {
    if (!node_editor_ || !metadata) return;

    // Only create implemented nodes
    if (metadata->IsTemplate()) {
        return;
    }

    // TODO: Implement direct node creation
    // node_editor_->CreateNode(metadata->type, mouse_pos);
}

ImU32 NodeBrowserPanel::GetNodeColor(cyxwiz::NodeCategory category) const {
    switch (category) {
        case cyxwiz::NodeCategory::DataSources:
            return IM_COL32(100, 181, 246, 255);  // Light Blue
        case cyxwiz::NodeCategory::DataTransform:
            return IM_COL32(38, 166, 154, 255);   // Teal
        case cyxwiz::NodeCategory::Analytics:
            return IM_COL32(156, 39, 176, 255);   // Purple
        case cyxwiz::NodeCategory::Preprocessing:
            return IM_COL32(129, 199, 132, 255);  // Light Green
        case cyxwiz::NodeCategory::Layers:
            return IM_COL32(39, 174, 96, 255);    // Green
        case cyxwiz::NodeCategory::Activation:
            return IM_COL32(243, 156, 18, 255);   // Orange
        case cyxwiz::NodeCategory::Pooling:
            return IM_COL32(155, 89, 182, 255);   // Light Purple
        case cyxwiz::NodeCategory::Normalization:
            return IM_COL32(236, 112, 99, 255);   // Coral
        case cyxwiz::NodeCategory::Attention:
            return IM_COL32(103, 58, 183, 255);   // Deep Purple
        case cyxwiz::NodeCategory::Recurrent:
            return IM_COL32(63, 81, 181, 255);    // Indigo
        case cyxwiz::NodeCategory::ShapeOps:
            return IM_COL32(26, 188, 156, 255);   // Turquoise
        case cyxwiz::NodeCategory::MergeOps:
            return IM_COL32(139, 195, 74, 255);   // Lime Green
        case cyxwiz::NodeCategory::Training:
            return IM_COL32(52, 73, 94, 255);     // Dark Blue Gray
        case cyxwiz::NodeCategory::Regularization:
            return IM_COL32(233, 30, 99, 255);    // Pink
        case cyxwiz::NodeCategory::Utility:
            return IM_COL32(96, 125, 139, 255);   // Blue Gray
        case cyxwiz::NodeCategory::Signal:
            return IM_COL32(0, 150, 136, 255);    // Teal
        case cyxwiz::NodeCategory::DataPipeline:
            return IM_COL32(0, 188, 212, 255);    // Cyan
        case cyxwiz::NodeCategory::DNN:
            return IM_COL32(41, 98, 255, 255);    // Deep Blue
        case cyxwiz::NodeCategory::TextProcessing:
            return IM_COL32(0, 121, 107, 255);    // Dark Teal
        case cyxwiz::NodeCategory::Upsampling:
            return IM_COL32(92, 107, 192, 255);   // Indigo
        case cyxwiz::NodeCategory::TimeSeries:
            return IM_COL32(255, 160, 0, 255);    // Amber
        case cyxwiz::NodeCategory::Audio:
            return IM_COL32(126, 87, 194, 255);   // Purple
        case cyxwiz::NodeCategory::RL:
            return IM_COL32(229, 57, 53, 255);    // Red
        case cyxwiz::NodeCategory::DataExport:
            return IM_COL32(76, 175, 80, 255);    // Green
        case cyxwiz::NodeCategory::Plugin:
            return IM_COL32(68, 136, 170, 255);   // Steel Blue
        case cyxwiz::NodeCategory::Unknown:
        default:
            return IM_COL32(127, 140, 141, 255);  // Gray
    }
}

ImU32 NodeBrowserPanel::GetNodeColorDark(cyxwiz::NodeCategory category) const {
    // Darken the node color for borders
    ImU32 color = GetNodeColor(category);
    int r = (color >> 0) & 0xFF;
    int g = (color >> 8) & 0xFF;
    int b = (color >> 16) & 0xFF;
    return IM_COL32(r * 0.7f, g * 0.7f, b * 0.7f, 255);
}

} // namespace gui
