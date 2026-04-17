// node_editor_add_search.cpp
// Node add search functionality - allows users to quickly find and add nodes via search box

#include "node_editor.h"
#include "icons.h"
#include "../core/node_metadata.h"
#include "../core/node_metadata_registry.h"
#include "../plugin/registries/plugin_node_registry.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <imnodes.h>
#include <algorithm>
#include <cctype>

namespace gui {

// Fuzzy match algorithm - returns a score (higher is better match, 0 = no match)
// Inspired by fzf/sublime text matching
int NodeEditor::FuzzyMatch(const std::string& pattern, const std::string& str) {
    if (pattern.empty()) return 1;  // Empty pattern matches everything
    if (str.empty()) return 0;

    // Convert both to lowercase for case-insensitive matching
    std::string pattern_lower, str_lower;
    pattern_lower.reserve(pattern.size());
    str_lower.reserve(str.size());

    for (char c : pattern) pattern_lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    for (char c : str) str_lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    int score = 0;
    size_t pattern_idx = 0;
    size_t last_match_idx = 0;
    bool consecutive = true;

    for (size_t i = 0; i < str_lower.size() && pattern_idx < pattern_lower.size(); ++i) {
        if (str_lower[i] == pattern_lower[pattern_idx]) {
            // Character matched
            score += 10;  // Base score for match

            // Bonus for consecutive matches
            if (consecutive && i == last_match_idx + 1) {
                score += 15;
            }

            // Bonus for matching at start of word (after space, underscore, or at beginning)
            if (i == 0 || str_lower[i-1] == ' ' || str_lower[i-1] == '_' || str_lower[i-1] == '/') {
                score += 20;
            }

            // Bonus for matching uppercase in original (CamelCase)
            if (i < str.size() && std::isupper(static_cast<unsigned char>(str[i]))) {
                score += 10;
            }

            last_match_idx = i;
            consecutive = true;
            pattern_idx++;
        } else {
            consecutive = false;
        }
    }

    // All pattern characters must be found
    if (pattern_idx < pattern_lower.size()) {
        return 0;  // No match
    }

    // Bonus for shorter strings (more relevant)
    score += static_cast<int>(100 - std::min(str.size(), size_t(100)));

    // Bonus for exact substring match
    if (str_lower.find(pattern_lower) != std::string::npos) {
        score += 50;
    }

    return score;
}

// Initialize the searchable nodes list with all available node types.
//
// As of 2026-04-17 this reads from `NodeMetadataRegistry` instead of a
// hand-coded list of ~300 addNode() calls — the registry is now the
// single source of truth for which nodes exist (see
// docs/plans/node_registration_unification.md). The old hand-coded
// list drifted from the browser (BarChart shipped but was invisible
// in the Nodes panel) and was the direct cause of the "registered in
// two of three places" bug class.
//
// Plugin-provided nodes stay on their existing runtime-registration
// path — PluginNodeRegistry is a separate concern that doesn't flow
// through NodeMetadataRegistry.
void NodeEditor::InitializeSearchableNodes() {
    if (searchable_nodes_initialized_) return;

    all_searchable_nodes_.clear();

    auto join_keywords = [](const std::vector<std::string>& kws) {
        std::string out;
        for (size_t i = 0; i < kws.size(); ++i) {
            if (i) out.push_back(' ');
            out += kws[i];
        }
        return out;
    };

    auto& reg = cyxwiz::NodeMetadataRegistry::Instance();
    if (!reg.IsInitialized()) reg.Initialize();

    // Walk every registered node. include_templates=true so the JSON
    // template entries under resources/node_templates/*.json show up
    // with the "Coming Soon" status — the click handler at line 595
    // short-circuits on status == Template so they display in the
    // dropdown but can't actually be added to the graph.
    for (const auto& cat : reg.GetCategories()) {
        for (const auto* meta : reg.GetByCategory(cat, /*include_templates=*/true)) {
            if (!meta) continue;
            SearchableNode node;
            node.type = meta->type;
            node.name = meta->name;
            node.category = cyxwiz::GetCategoryDisplayName(cat);
            node.keywords = join_keywords(meta->keywords);
            node.status = meta->status;
            node.description = meta->brief_description;
            node.tooltip = meta->help_text;
            all_searchable_nodes_.push_back(std::move(node));
        }
    }


    // Plugin-provided nodes
    try {
        auto plugin_nodes = cyxwiz::plugin::PluginNodeRegistry::Instance().GetAllNodeTypesWithNames();
        for (const auto& [qname, info] : plugin_nodes) {
            SearchableNode sn;
            sn.type = NodeType::PluginCustom;
            sn.name = info.display_name;
            sn.category = "Plugin/" + info.category;
            sn.keywords = info.type_name + " " + info.display_name + " " + info.description + " plugin";
            sn.plugin_qualified_name = qname;
            all_searchable_nodes_.push_back(std::move(sn));
        }
    } catch (const std::exception& e) {
        spdlog::warn("Failed to load plugin nodes into search: {}", e.what());
    } catch (...) {
        spdlog::warn("Unknown error loading plugin nodes into search");
    }

    searchable_nodes_initialized_ = true;
}

// Update filtered results based on current search query
void NodeEditor::UpdateNodeAddSearchResults() {
    filtered_nodes_.clear();

    std::string query(node_add_search_.search_buffer);

    // If query is empty, show all nodes
    if (query.empty()) {
        for (auto& node : all_searchable_nodes_) {
            filtered_nodes_.push_back({100, &node});  // Default score
        }
    } else {
        // Score each node
        for (auto& node : all_searchable_nodes_) {
            // Match against name, category, and keywords
            int name_score = FuzzyMatch(query, node.name);
            int category_score = FuzzyMatch(query, node.category) / 2;  // Lower weight for category
            int keyword_score = FuzzyMatch(query, node.keywords) / 2;   // Lower weight for keywords

            int total_score = std::max({name_score, category_score, keyword_score});

            if (total_score > 0) {
                filtered_nodes_.push_back({total_score, &node});
            }
        }

        // Sort by score (highest first)
        std::sort(filtered_nodes_.begin(), filtered_nodes_.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
    }

    // Limit to top 15 results
    if (filtered_nodes_.size() > 15) {
        filtered_nodes_.resize(15);
    }

    // Reset selection if out of bounds
    if (node_add_search_.selected_index >= static_cast<int>(filtered_nodes_.size())) {
        node_add_search_.selected_index = 0;
    }
}

// Render the node add search UI (top-right of canvas)
void NodeEditor::ShowNodeAddSearch() {
    // Initialize searchable nodes on first call
    if (!searchable_nodes_initialized_) {
        InitializeSearchableNodes();
        UpdateNodeAddSearchResults();
    }

    // Get the content region bounds (the actual canvas area inside the window)
    ImVec2 content_min = ImGui::GetWindowContentRegionMin();
    ImVec2 content_max = ImGui::GetWindowContentRegionMax();
    ImVec2 window_pos = ImGui::GetWindowPos();

    // Calculate canvas bounds in screen coordinates
    ImVec2 canvas_pos(window_pos.x + content_min.x, window_pos.y + content_min.y);
    ImVec2 canvas_size(content_max.x - content_min.x, content_max.y - content_min.y);

    // Position search box in top-right corner of the canvas content area
    // Add offset to position below the toolbar rows (approx 70px for two rows of buttons)
    float search_width = 250.0f;
    float search_height = 28.0f;
    float margin = 10.0f;
    float toolbar_offset = 70.0f;  // Offset to position below the toolbars

    ImVec2 search_pos(canvas_pos.x + canvas_size.x - search_width - margin, canvas_pos.y + toolbar_offset + margin);

    // Create a floating window for the search box that renders on top of ImNodes
    ImGui::SetNextWindowPos(search_pos);
    ImGui::SetNextWindowSize(ImVec2(search_width, search_height));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 5.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.15f, 0.15f, 0.18f, 0.95f));

    ImGuiWindowFlags search_window_flags =
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoScrollWithMouse |
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoDocking;

    // Declare at function scope so it's accessible for the "close dropdown" check
    bool input_focused = false;

    if (ImGui::Begin("##NodeSearchBox", nullptr, search_window_flags)) {
        // Push style for search input
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 5.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8, 6));
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.18f, 0.0f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.2f, 0.2f, 0.24f, 0.5f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0.25f, 0.25f, 0.3f, 0.5f));

        // Search icon
        ImGui::SetCursorPos(ImVec2(8, 6));
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), ICON_FA_MAGNIFYING_GLASS);

        // Input field
        ImGui::SetCursorPos(ImVec2(28, 0));
        ImGui::SetNextItemWidth(search_width - 36);

    // Handle keyboard focus
    if (node_add_search_.just_activated) {
        ImGui::SetKeyboardFocusHere();
        node_add_search_.just_activated = false;
    }

    bool text_changed = ImGui::InputTextWithHint(
        "##node_add_search",
        "Search nodes...",
        node_add_search_.search_buffer,
        sizeof(node_add_search_.search_buffer),
        ImGuiInputTextFlags_EnterReturnsTrue
    );

    // Check if input is active
    bool input_active = ImGui::IsItemActive();
    input_focused = ImGui::IsItemFocused();  // Assign to function-scope variable

    // Update search state
    if (input_active || input_focused) {
        node_add_search_.is_active = true;
        node_add_search_.show_results = true;
    }

    // Debouncing logic: Start timer when text changes
    std::string current_query(node_add_search_.search_buffer);
    if (text_changed || ImGui::IsItemEdited()) {
        // Check if the query actually changed (prevents unnecessary updates)
        if (current_query != node_add_search_.last_search_query_) {
            node_add_search_.search_dirty_ = true;
            node_add_search_.search_debounce_timer_ = 0.15f;  // 150ms debounce delay
        }
    }

    // Decrement timer each frame
    if (node_add_search_.search_debounce_timer_ > 0.0f) {
        node_add_search_.search_debounce_timer_ -= ImGui::GetIO().DeltaTime;
    }

    // Execute search when timer expires and search is dirty
    if (node_add_search_.search_dirty_ && node_add_search_.search_debounce_timer_ <= 0.0f) {
        UpdateNodeAddSearchResults();
        node_add_search_.selected_index = 0;
        node_add_search_.search_dirty_ = false;
        node_add_search_.last_search_query_ = current_query;
    }

    // Handle keyboard navigation
    if (node_add_search_.is_active && node_add_search_.show_results) {
        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
            node_add_search_.selected_index = std::min(
                node_add_search_.selected_index + 1,
                static_cast<int>(filtered_nodes_.size()) - 1
            );
        }
        if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) {
            node_add_search_.selected_index = std::max(node_add_search_.selected_index - 1, 0);
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            node_add_search_.show_results = false;
            node_add_search_.is_active = false;
            node_add_search_.search_buffer[0] = '\0';
        }
        if ((ImGui::IsKeyPressed(ImGuiKey_Enter) || text_changed) && !filtered_nodes_.empty()) {
            // Add the selected node
            SearchableNode* selected = filtered_nodes_[node_add_search_.selected_index].second;

            // Block template nodes - they're not implemented yet
            if (selected->status == NodeImplementationStatus::Template) {
                // Don't add template nodes - just show they're not available
            } else {
                // Set position for new node (center of visible canvas)
                ImVec2 editor_pan = ImNodes::EditorContextGetPanning();
                context_menu_pos_ = ImVec2(
                    canvas_size.x / 2 - editor_pan.x,
                    canvas_size.y / 2 - editor_pan.y
                );

                // For plugin nodes, pass qualified name so CreateNode can look up the registry
                if (selected->type == NodeType::PluginCustom && !selected->plugin_qualified_name.empty())
                    AddNode(selected->type, selected->plugin_qualified_name);
                else
                    AddNode(selected->type, selected->name);

                // Reset search state
                node_add_search_.show_results = false;
                node_add_search_.is_active = false;
                node_add_search_.search_buffer[0] = '\0';
                UpdateNodeAddSearchResults();
            }
        }
    }

        ImGui::PopStyleColor(3);
        ImGui::PopStyleVar(2);
    }
    ImGui::End();
    ImGui::PopStyleColor(1);
    ImGui::PopStyleVar(2);

    // Render dropdown results - only show when user has typed something
    std::string query(node_add_search_.search_buffer);
    bool has_search_text = !query.empty();

    if (node_add_search_.show_results && has_search_text && !filtered_nodes_.empty()) {
        ImVec2 dropdown_pos(search_pos.x, search_pos.y + search_height + 2);
        float dropdown_width = search_width;
        float item_height = 32.0f;
        float dropdown_height = std::min(item_height * filtered_nodes_.size() + 8, 400.0f);

        // Create a separate floating window for dropdown results
        ImGui::SetNextWindowPos(dropdown_pos);
        ImGui::SetNextWindowSize(ImVec2(dropdown_width, dropdown_height));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4, 4));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 5.0f);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.12f, 0.12f, 0.14f, 0.96f));
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.24f, 0.24f, 0.28f, 1.0f));

        ImGuiWindowFlags dropdown_flags =
            ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoDocking |
            ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoNav;

        if (ImGui::Begin("##NodeSearchDropdown", nullptr, dropdown_flags)) {
            for (size_t i = 0; i < filtered_nodes_.size(); ++i) {
                SearchableNode* node = filtered_nodes_[i].second;
                bool is_selected = (static_cast<int>(i) == node_add_search_.selected_index);

                ImVec2 item_pos = ImGui::GetCursorScreenPos();
                ImVec2 item_size(dropdown_width - 8, item_height - 2);

                // Highlight selected item
                if (is_selected) {
                    ImDrawList* draw_list = ImGui::GetWindowDrawList();
                    draw_list->AddRectFilled(
                        item_pos,
                        ImVec2(item_pos.x + item_size.x, item_pos.y + item_size.y),
                        IM_COL32(60, 100, 180, 200),
                        3.0f
                    );
                }

                // Handle click - block template nodes
                bool is_template = (node->status == NodeImplementationStatus::Template);
                ImGui::InvisibleButton(("node_result_" + std::to_string(i)).c_str(), item_size);
                if (ImGui::IsItemClicked() && !is_template) {
                    // Add the clicked node (not template)
                    ImVec2 editor_pan = ImNodes::EditorContextGetPanning();
                    context_menu_pos_ = ImVec2(
                        canvas_size.x / 2 - editor_pan.x,
                        canvas_size.y / 2 - editor_pan.y
                    );

                    AddNode(node->type, node->name);

                    // Reset search state
                    node_add_search_.show_results = false;
                    node_add_search_.is_active = false;
                    node_add_search_.search_buffer[0] = '\0';
                    UpdateNodeAddSearchResults();
                }
                if (ImGui::IsItemHovered()) {
                    node_add_search_.selected_index = static_cast<int>(i);
                    // Show tooltip for template nodes
                    if (is_template && !node->tooltip.empty()) {
                        ImGui::SetTooltip("%s", node->tooltip.c_str());
                    }
                }

                // Draw node name and category (with Coming Soon badge for templates)
                ImGui::SetCursorScreenPos(ImVec2(item_pos.x + 8, item_pos.y + 4));
                if (is_template) {
                    // Grayed out name for template nodes
                    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "%s", node->name.c_str());
                    // Add "Coming Soon" badge
                    ImGui::SameLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.6f, 0.2f, 1.0f));
                    ImGui::TextUnformatted(" [Coming Soon]");
                    ImGui::PopStyleColor();
                } else {
                    ImGui::TextColored(ImVec4(1.0f, 1.0f, 1.0f, 1.0f), "%s", node->name.c_str());
                }

                ImGui::SetCursorScreenPos(ImVec2(item_pos.x + 8, item_pos.y + 18));
                ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.6f, 1.0f), "%s", node->category.c_str());

                ImGui::SetCursorScreenPos(ImVec2(item_pos.x, item_pos.y + item_height));
            }
        }
        ImGui::End();
        ImGui::PopStyleColor(2);
        ImGui::PopStyleVar(2);
    }

    // Close dropdown when clicking outside
    if (node_add_search_.show_results && !input_focused && ImGui::IsMouseClicked(0)) {
        ImVec2 mouse_pos = ImGui::GetMousePos();
        ImVec2 dropdown_check_pos(search_pos.x, search_pos.y);
        float dropdown_height = 32.0f * filtered_nodes_.size() + search_height + 10;

        if (mouse_pos.x < dropdown_check_pos.x || mouse_pos.x > dropdown_check_pos.x + search_width ||
            mouse_pos.y < dropdown_check_pos.y || mouse_pos.y > dropdown_check_pos.y + dropdown_height) {
            node_add_search_.show_results = false;
            node_add_search_.is_active = false;
        }
    }
}

} // namespace gui

