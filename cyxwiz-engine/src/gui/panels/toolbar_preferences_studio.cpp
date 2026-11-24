// Preferences > Studio (TOFIX129 B9-10): how the canvas starts. The menus
// change these for the session; here is what every start uses.

#include "toolbar.h"

#include "../../core/engine_config.h"

#include <imgui.h>

namespace cyxwiz {

void ToolbarPanel::RenderStudioPreferences() {
    auto& config = core::EngineConfig::Instance();
    const core::StudioConfig saved = config.GetStudio();
    core::StudioConfig studio = saved;

    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("How the Studio starts. The menus change these for the session; here is what every start "
                        "uses.");
    ImGui::PopTextWrapPos();

    ImGui::Spacing();
    ImGui::Text("Saving");
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::Checkbox("Auto Save on when the Engine starts", &studio.auto_save_at_start);
    ImGui::TextDisabled("File > Auto Save saves open scripts and the graph every minute.");

    ImGui::Spacing();
    ImGui::Text("Canvas");
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::Checkbox("Show the canvas minimap", &studio.minimap);
    ImGui::TextDisabled("View > Minimaps > Studio Minimap (M) shows or hides it for the session.");

    ImGui::Spacing();
    ImGui::Text("Before Train");
    ImGui::Separator();
    ImGui::Spacing();
    bool require = config.RequireDebugBeforeTrain();
    if (ImGui::Checkbox("Require a passing Local Debug run before Train", &require)) {
        config.SetRequireDebugBeforeTrain(require);
        config.Save();
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("Off: Train warns when the graph has no passing Local Debug (F6) run, or changed since. On: "
                        "Train refuses to start until it has one.");
    ImGui::PopTextWrapPos();

    if (!(studio == saved)) {
        config.SetStudio(studio);
        config.Save();
    }
}

}  // namespace cyxwiz
