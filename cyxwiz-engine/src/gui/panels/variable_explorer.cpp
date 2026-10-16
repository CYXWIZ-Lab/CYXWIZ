#include "variable_explorer.h"

#include "../../scripting/scripting_engine.h"
#include "../icons.h"
#include "../separate_windows.h"

#include <imgui.h>
#include <imgui_internal.h>

namespace cyxwiz {

VariableExplorerPanel::VariableExplorerPanel() : Panel("Variable Explorer", true) {}

const char* VariableExplorerPanel::GetIcon() const { return ICON_FA_LIST_UL; }

void VariableExplorerPanel::SetScriptingEngine(std::shared_ptr<scripting::ScriptingEngine> engine) {
    scripting_engine_ = std::move(engine);
    view_.SetEngine(scripting_engine_.get());
}

void VariableExplorerPanel::Render() {
    if (!visible_) return;
    ImGui::SetNextWindowSize(ImVec2(900.0f, 320.0f), ImGuiCond_FirstUseEver);
    // Board 9: beside the Console. A saved layout made before this panel had
    // its own entry would open it floating; the first time, join the
    // Console's (or else the Viewport's) dock area.
    for (const char* neighbour : {"Console", "Viewport"}) {
        ImGuiWindow* w = ImGui::FindWindowByName(neighbour);
        if (w && w->DockId != 0) {
            ImGui::SetNextWindowDockID(w->DockId, ImGuiCond_FirstUseEver);
            break;
        }
    }
    // The ### id matches the dock layout and imgui.ini entry whatever the icon.
    static constexpr const char* kName = ICON_FA_LIST_UL " Variable Explorer###VariableExplorer";
    ::gui::NextWindowMayLeave(kName);
    const bool expanded = ImGui::Begin(kName, &visible_);
    ::gui::TabMenu(kName, &visible_);
    if (expanded) {
        focused_ = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
        view_.Render(0.0f);
    }
    ImGui::End();
}

}  // namespace cyxwiz
