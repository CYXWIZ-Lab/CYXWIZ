// Command Palette (Ctrl+P), listing every menu action from the menu
// presentation model (TOFIX129 step 1.4): same labels, chords, enabled rules
// and hints as the menu bar.
#include "toolbar.h"

#include "../icons.h"

#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <cstring>

namespace cyxwiz {

void ToolbarPanel::OpenCommandPalette() {
    show_command_palette_ = true;
    focus_search_input_ = true;
    selected_index_ = 0;
    std::memset(search_buffer_, 0, sizeof(search_buffer_));
    palette_entries_ = menu::BuildPaletteEntries(menu::BuildMenuModel(BuildMenuInputs()));
    UpdateSearchResults("");
}

std::string ToolbarPanel::ToLowerCase(const std::string& str) const {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return result;
}

int ToolbarPanel::FuzzyMatch(const std::string& pattern, const std::string& text) const {
    if (pattern.empty()) return 100;  // Empty pattern matches everything

    const std::string lower_pattern = ToLowerCase(pattern);
    const std::string lower_text = ToLowerCase(text);

    const size_t at = lower_text.find(lower_pattern);
    if (at == 0) return 100;                 // prefix match
    if (at != std::string::npos) return 90;  // substring match

    // Fuzzy character matching
    int score = 0;
    size_t pattern_idx = 0;
    size_t last_match_idx = 0;
    bool consecutive = true;
    for (size_t i = 0; i < lower_text.size() && pattern_idx < lower_pattern.size(); ++i) {
        if (lower_text[i] != lower_pattern[pattern_idx]) continue;
        score += 10;
        if (consecutive && i == last_match_idx + 1) score += 5;
        else consecutive = false;
        if (i == 0 || lower_text[i - 1] == ' ' || lower_text[i - 1] == '_' || lower_text[i - 1] == '-') score += 3;
        last_match_idx = i;
        ++pattern_idx;
    }
    return pattern_idx == lower_pattern.size() ? score : 0;
}

void ToolbarPanel::UpdateSearchResults(const std::string& query) {
    filtered_entries_.clear();
    std::vector<std::pair<int, int>> scored;  // score, index
    for (int i = 0; i < static_cast<int>(palette_entries_.size()); ++i) {
        const auto& e = palette_entries_[i];
        int score = 100;
        if (!query.empty()) {
            score = std::max({FuzzyMatch(query, e.label), FuzzyMatch(query, e.menu_path) / 2,
                              FuzzyMatch(query, e.shortcut) / 2});
        }
        if (score > 0) scored.push_back({score, i});
    }
    std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (const auto& [score, index] : scored) filtered_entries_.push_back(index);
    selected_index_ = 0;
}

void ToolbarPanel::RenderCommandPalette() {
    if (!show_command_palette_) return;

    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        show_command_palette_ = false;
        return;
    }

    // Dim the workspace so the palette reads as the one active thing.
    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
    const ImVec2 viewport_pos = ImGui::GetMainViewport()->Pos;
    const ImVec2 viewport_size = ImGui::GetMainViewport()->Size;
    draw_list->AddRectFilled(viewport_pos, ImVec2(viewport_pos.x + viewport_size.x, viewport_pos.y + viewport_size.y),
                             IM_COL32(0, 0, 0, 100));

    const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    const ImVec2 window_size(600.0f, 440.0f);
    ImGui::SetNextWindowPos(ImVec2(center.x - window_size.x * 0.5f, center.y - window_size.y * 0.35f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);

    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                   ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar;
    if (!ImGui::Begin("##CommandPalette", &show_command_palette_, flags)) {
        ImGui::End();
        return;
    }
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
        !ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem | ImGuiHoveredFlags_ChildWindows)) {
        show_command_palette_ = false;
        ImGui::End();
        return;
    }

    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 8.0f));
    ImGui::PushItemWidth(-1);
    if (focus_search_input_) {
        ImGui::SetKeyboardFocusHere();
        focus_search_input_ = false;
    }
    const bool text_changed = ImGui::InputTextWithHint("##SearchInput", ICON_FA_MAGNIFYING_GLASS " Search every command",
                                                       search_buffer_, sizeof(search_buffer_));
    ImGui::PopItemWidth();
    ImGui::PopStyleVar();
    if (text_changed) UpdateSearchResults(search_buffer_);

    ImGui::Separator();

    bool moved_by_keyboard = false;
    const int count = static_cast<int>(filtered_entries_.size());
    if (ImGui::IsKeyPressed(ImGuiKey_DownArrow) && count > 0) {
        selected_index_ = std::min(selected_index_ + 1, count - 1);
        moved_by_keyboard = true;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) {
        selected_index_ = std::max(selected_index_ - 1, 0);
        moved_by_keyboard = true;
    }
    auto run = [this](const menu::PaletteEntry& e) {
        if (!e.enabled) return;
        show_command_palette_ = false;
        Dispatch(e.id, e.argument);
    };
    if (ImGui::IsKeyPressed(ImGuiKey_Enter) && selected_index_ >= 0 && selected_index_ < count) {
        run(palette_entries_[filtered_entries_[selected_index_]]);
    }

    ImGui::BeginChild("##ResultsList", ImVec2(0.0f, 0.0f), ImGuiChildFlags_None);
    const float right_edge = ImGui::GetWindowWidth() - 16.0f;
    for (int i = 0; i < count; ++i) {
        const auto& e = palette_entries_[filtered_entries_[i]];
        ImGui::PushID(i);
        const bool selected = i == selected_index_;
        if (selected) ImGui::PushStyleColor(ImGuiCol_Header, ImGui::GetStyle().Colors[ImGuiCol_HeaderActive]);
        if (ImGui::Selectable("##row", selected, ImGuiSelectableFlags_SpanAllColumns, ImVec2(0.0f, 30.0f))) run(e);
        if (selected) {
            ImGui::PopStyleColor();
            if (moved_by_keyboard) ImGui::SetScrollHereY();
        }
        if (ImGui::IsItemHovered()) {
            if (e.planned) ImGui::SetTooltip("Planned: not available yet.");
            else if (!e.enabled) ImGui::SetTooltip("%s", e.disabled_reason.c_str());
            else if (!e.hint.empty()) ImGui::SetTooltip("%s", e.hint.c_str());
        }

        ImGui::SameLine(10.0f);
        ImGui::TextDisabled("%s", IconForAction(e.id));
        ImGui::SameLine(40.0f);
        if (e.enabled) ImGui::TextUnformatted(e.label.c_str());
        else ImGui::TextDisabled("%s%s", e.label.c_str(), e.planned ? "  (planned)" : "");

        // Right side: chord, then the menu path.
        const float path_width = ImGui::CalcTextSize(e.menu_path.c_str()).x;
        const float chord_width = e.shortcut.empty() ? 0.0f : ImGui::CalcTextSize(e.shortcut.c_str()).x + 16.0f;
        ImGui::SameLine(right_edge - path_width - chord_width);
        if (!e.shortcut.empty()) {
            ImGui::TextDisabled("%s", e.shortcut.c_str());
            ImGui::SameLine(right_edge - path_width);
        }
        ImGui::TextDisabled("%s", e.menu_path.c_str());
        ImGui::PopID();
    }
    if (count == 0) ImGui::TextDisabled("No command matches the search.");
    ImGui::EndChild();
    ImGui::End();
}

}  // namespace cyxwiz
