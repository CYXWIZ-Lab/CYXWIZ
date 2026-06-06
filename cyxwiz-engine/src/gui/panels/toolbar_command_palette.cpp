#include "toolbar.h"
#include <imgui.h>
#include <algorithm>
#include <cctype>
#include <cstring>

namespace cyxwiz {

namespace {

const char* ToolSurfaceLabel(ToolSurface surface) {
    switch (surface) {
        case ToolSurface::StandalonePanel: return "Panel";
        case ToolSurface::GraphBackedPanel: return "Graph-backed";
        case ToolSurface::Utility: return "Utility";
        case ToolSurface::Command:
        default: return "Command";
    }
}

} // namespace

void ToolbarPanel::OpenCommandPalette() {
    show_command_palette_ = true;
    focus_search_input_ = true;
    selected_index_ = 0;
    memset(search_buffer_, 0, sizeof(search_buffer_));

    // Initially show all tools
    filtered_tools_.clear();
    for (const auto& tool : all_tools_) {
        filtered_tools_.push_back(&tool);
    }
}

void ToolbarPanel::HandleGlobalShortcuts() {
    // Ctrl+P for command palette
    if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_P)) {
        OpenCommandPalette();
    }
}

std::string ToolbarPanel::ToLowerCase(const std::string& str) const {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

int ToolbarPanel::FuzzyMatch(const std::string& pattern, const std::string& text) const {
    if (pattern.empty()) return 100;  // Empty pattern matches everything

    std::string lowerPattern = ToLowerCase(pattern);
    std::string lowerText = ToLowerCase(text);

    // Exact substring match gets highest score
    if (lowerText.find(lowerPattern) != std::string::npos) {
        return 100;
    }

    // Check if pattern starts text (prefix match)
    if (lowerText.find(lowerPattern) == 0) {
        return 90;
    }

    // Fuzzy character matching
    int score = 0;
    size_t patternIdx = 0;
    size_t lastMatchIdx = 0;
    bool consecutive = true;

    for (size_t i = 0; i < lowerText.size() && patternIdx < lowerPattern.size(); ++i) {
        if (lowerText[i] == lowerPattern[patternIdx]) {
            score += 10;
            // Bonus for consecutive matches
            if (consecutive && i == lastMatchIdx + 1) {
                score += 5;
            } else {
                consecutive = false;
            }
            // Bonus for matching at word boundaries
            if (i == 0 || lowerText[i - 1] == ' ' || lowerText[i - 1] == '_' || lowerText[i - 1] == '-') {
                score += 3;
            }
            lastMatchIdx = i;
            ++patternIdx;
        }
    }

    // Only match if all pattern characters were found
    if (patternIdx != lowerPattern.size()) {
        return 0;
    }

    return score;
}

void ToolbarPanel::UpdateSearchResults(const std::string& query) {
    filtered_tools_.clear();

    if (query.empty()) {
        // Show all tools when query is empty
        for (const auto& tool : all_tools_) {
            tool.match_score = 100;
            filtered_tools_.push_back(&tool);
        }
        return;
    }

    // Score all tools against the query
    for (const auto& tool : all_tools_) {
        int nameScore = FuzzyMatch(query, tool.name);
        int categoryScore = FuzzyMatch(query, tool.category) / 2;  // Lower weight for category
        int keywordScore = FuzzyMatch(query, tool.keywords) / 2;   // Lower weight for keywords

        tool.match_score = std::max({nameScore, categoryScore, keywordScore});

        if (tool.match_score > 0) {
            filtered_tools_.push_back(&tool);
        }
    }

    // Sort by score (descending)
    std::sort(filtered_tools_.begin(), filtered_tools_.end(),
              [](const ToolEntry* a, const ToolEntry* b) {
                  return a->match_score > b->match_score;
              });

    // Reset selection when results change
    selected_index_ = 0;
}

void ToolbarPanel::RenderCommandPalette() {
    if (!show_command_palette_) return;

    // Check ESC key globally (even before window is drawn)
    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        show_command_palette_ = false;
        return;
    }

    // Draw semi-transparent background overlay for click-outside detection
    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
    ImVec2 viewport_pos = ImGui::GetMainViewport()->Pos;
    ImVec2 viewport_size = ImGui::GetMainViewport()->Size;
    draw_list->AddRectFilled(viewport_pos, ImVec2(viewport_pos.x + viewport_size.x, viewport_pos.y + viewport_size.y),
                             IM_COL32(0, 0, 0, 100));

    // Center the modal
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImVec2 window_size(500, 400);
    ImVec2 window_pos(center.x - window_size.x * 0.5f, center.y - window_size.y * 0.3f);
    ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar;

    if (ImGui::Begin("##CommandPalette", &show_command_palette_, flags)) {
        // Check for click outside the command palette window (must be inside Begin/End)
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem | ImGuiHoveredFlags_ChildWindows)) {
            show_command_palette_ = false;
            ImGui::End();
            return;
        }
        // Search input
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10, 8));
        ImGui::PushItemWidth(-1);

        if (focus_search_input_) {
            ImGui::SetKeyboardFocusHere();
            focus_search_input_ = false;
        }

        bool textChanged = ImGui::InputTextWithHint("##SearchInput", "Type to search tools...",
                                                     search_buffer_, sizeof(search_buffer_));
        ImGui::PopItemWidth();
        ImGui::PopStyleVar();

        if (textChanged) {
            UpdateSearchResults(search_buffer_);
        }

        ImGui::Separator();

        // Handle keyboard navigation
        bool selection_changed_by_keyboard = false;
        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
            selected_index_ = std::min(selected_index_ + 1, static_cast<int>(filtered_tools_.size()) - 1);
            selection_changed_by_keyboard = true;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) {
            selected_index_ = std::max(selected_index_ - 1, 0);
            selection_changed_by_keyboard = true;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Enter) && !filtered_tools_.empty()) {
            if (selected_index_ >= 0 && selected_index_ < static_cast<int>(filtered_tools_.size())) {
                const ToolEntry* tool = filtered_tools_[selected_index_];
                if (tool->availability == ToolAvailability::Working && tool->callback) {
                    auto callback = tool->callback;
                    show_command_palette_ = false;
                    callback();
                }
            }
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            show_command_palette_ = false;
        }

        // Results list
        ImGui::BeginChild("##ResultsList", ImVec2(0, 0), false);

        for (int i = 0; i < static_cast<int>(filtered_tools_.size()); ++i) {
            const auto* tool = filtered_tools_[i];
            const bool can_execute = tool->availability == ToolAvailability::Working && tool->callback;

            ImGui::PushID(i);

            bool isSelected = (i == selected_index_);
            if (isSelected) {
                ImGui::PushStyleColor(ImGuiCol_Header, ImGui::GetStyle().Colors[ImGuiCol_HeaderActive]);
            }

            // Selectable row
            if (ImGui::Selectable("##ToolRow", isSelected, ImGuiSelectableFlags_SpanAllColumns, ImVec2(0, 32))) {
                if (can_execute) {
                    show_command_palette_ = false;
                    tool->callback();
                }
            }

            if (isSelected) {
                ImGui::PopStyleColor();
                // Only auto-scroll when selection changed via keyboard, not every frame
                if (selection_changed_by_keyboard) {
                    ImGui::SetScrollHereY();
                }
            }

            // Draw content on top of selectable
            ImGui::SameLine(10);

            // Icon
            ImGui::Text("%s", tool->icon.c_str());
            ImGui::SameLine(40);

            // Tool name
            if (!can_execute) {
                ImGui::TextDisabled("%s", tool->name.c_str());
            } else {
                ImGui::Text("%s", tool->name.c_str());
            }

            if (!tool->status_detail.empty() && ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s", tool->status_detail.c_str());
            }

            // Category badge (right-aligned)
            ImGui::SameLine(ImGui::GetWindowWidth() - 210);
            ImGui::TextDisabled("[%s]", ToolSurfaceLabel(tool->surface));
            ImGui::SameLine(ImGui::GetWindowWidth() - 110);
            ImGui::TextDisabled("[%s]", tool->category.c_str());

            ImGui::PopID();
        }

        if (filtered_tools_.empty()) {
            ImGui::TextDisabled("No matching tools found");
        }

        ImGui::EndChild();
    }
    ImGui::End();
}

} // namespace cyxwiz
