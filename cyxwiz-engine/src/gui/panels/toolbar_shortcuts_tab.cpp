// Preferences > Shortcuts (TOFIX129 step 1.4): every shortcut, grouped by the
// window it belongs to, read from the same table that the menus print and
// the key handler dispatches. Read-only: rebinding is planned.
#include "toolbar.h"

#include "../icons.h"

#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

std::string Lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool Matches(const menu::ShortcutEntry& e, const std::string& needle) {
    if (needle.empty()) return true;
    return Lower(e.action_label).find(needle) != std::string::npos ||
           Lower(e.chord).find(needle) != std::string::npos ||
           Lower(e.menu_path).find(needle) != std::string::npos;
}

}  // namespace

void ToolbarPanel::RenderShortcutsPreferences() {
    const menu::Context contexts[] = {menu::Context::Any, menu::Context::StudioCanvas, menu::Context::ScriptEditor,
                                      menu::Context::ScriptDebugging, menu::Context::ScriptNotebook,
                                      menu::Context::Variables, menu::Context::TableViewer, menu::Context::DataStudio};
    constexpr int kContexts = static_cast<int>(sizeof(contexts) / sizeof(contexts[0]));
    const auto& table = menu::ShortcutTable();
    const std::string needle = Lower(shortcuts_search_);

    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("Shortcuts belong to the focused window. The same key can do the matching action in "
                        "another window: F5 runs a script in the Script Editor and starts training elsewhere.");
    ImGui::PopTextWrapPos();
    ImGui::Spacing();

    const float rail_width = 200.0f;
    if (ImGui::BeginChild("##shortcut_windows", ImVec2(rail_width, 0.0f), ImGuiChildFlags_None)) {
        ImGui::TextDisabled("WINDOW");
        ImGui::Spacing();
        for (int i = 0; i < kContexts; ++i) {
            int count = 0;
            for (const auto& e : table) if (e.context == contexts[i] && Matches(e, needle)) ++count;
            const std::string label = std::string(menu::ContextName(contexts[i])) + "  (" + std::to_string(count) + ")";
            if (ImGui::Selectable(label.c_str(), shortcuts_context_ == i)) shortcuts_context_ = i;
        }
        ImGui::Spacing();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("Console and the other windows list their keys here when their "
                            "screens are reviewed.");
        ImGui::PopTextWrapPos();
    }
    ImGui::EndChild();

    ImGui::SameLine();

    if (ImGui::BeginChild("##shortcut_list", ImVec2(0.0f, 0.0f), ImGuiChildFlags_None)) {
        ImGui::SetNextItemWidth(280.0f);
        ImGui::InputTextWithHint("##shortcut_search", ICON_FA_MAGNIFYING_GLASS " Search actions or keys",
                                 shortcuts_search_, sizeof(shortcuts_search_));
        ImGui::SameLine();
        ImGui::TextDisabled("Rebinding: planned");
        ImGui::Spacing();

        const menu::Context context = contexts[std::clamp(shortcuts_context_, 0, kContexts - 1)];
        const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH |
                                      ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp |
                                      ImGuiTableFlags_NoSavedSettings;
        if (ImGui::BeginTable("##shortcuts", 3, flags, ImVec2(0.0f, 0.0f))) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthStretch, 2.4f);
            ImGui::TableSetupColumn("Shortcut", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            ImGui::TableSetupColumn("Also in menu", ImGuiTableColumnFlags_WidthStretch, 1.3f);
            ImGui::TableHeadersRow();
            int shown = 0;
            for (const auto& e : table) {
                if (e.context != context || !Matches(e, needle)) continue;
                ++shown;
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::PushTextWrapPos(0.0f);
                if (e.planned) ImGui::TextDisabled("%s  (planned)", e.action_label.c_str());
                else ImGui::TextUnformatted(e.action_label.c_str());
                ImGui::PopTextWrapPos();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(e.chord.c_str());
                ImGui::TableNextColumn();
                ImGui::TextDisabled("%s", e.menu_path.c_str());
            }
            if (shown == 0) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextDisabled("No shortcut matches the search.");
            }
            ImGui::EndTable();
        }
    }
    ImGui::EndChild();
}

}  // namespace cyxwiz
