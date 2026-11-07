// Preferences > Studio Debugger (TOFIX178 6.1c, approved Capture mockup): the
// user's defaults for layer capture and the layer views. Engine-wide, applied
// and saved at once; each run's capture control starts from them.

#include "toolbar.h"

#include "layer_capture_controls.h"

#include "../../core/engine_config.h"
#include "../ui_buttons.h"
#include "../ui_widgets.h"

#include <imgui.h>

namespace cyxwiz {

void ToolbarPanel::RenderStudioDebuggerPreferences() {
    using core::FeatureMapColour;
    using core::LayerCaptureWhen;
    auto& engine_config = core::EngineConfig::Instance();
    const core::StudioDebuggerConfig saved = engine_config.GetStudioDebugger();
    core::StudioDebuggerConfig settings = saved;

    ImGui::Spacing();
    ImGui::Text("Layer capture");
    ImGui::Separator();
    ImGui::Spacing();

    RenderLayerCaptureControls(settings);

    ImGui::PushTextWrapPos(0.0f);
    if (settings.capture_when != LayerCaptureWhen::Off) {
        ImGui::Spacing();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Keep the captures of the last");
        ImGui::SameLine();
        ui::NumberField("##keep_capture_runs", &settings.keep_capture_runs, 1, 1000, 1, 60.0f,
                        "Older runs keep their report; their captures are deleted");
        ImGui::SameLine();
        ImGui::TextUnformatted("runs");
        ImGui::TextDisabled("Kept on the compute device while a run is open; saved with the run in the debug run "
                            "store.");
    }
    ImGui::PopTextWrapPos();

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Text("Layer views");
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::TextUnformatted("Feature-map colour");
    static const char* const kColour[] = {"Heat", "Grayscale"};
    int colour = static_cast<int>(settings.feature_map_colour);
    if (ui::SegmentedControl("##feature_map_colour", kColour, 2, &colour))
        settings.feature_map_colour = static_cast<FeatureMapColour>(colour);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("For every activation layer. Heat runs from black through red to yellow; each view can "
                        "switch it.");
    ImGui::PopTextWrapPos();

    if (!(settings == saved)) {
        engine_config.SetStudioDebugger(settings);
        engine_config.Save();
    }
}

}  // namespace cyxwiz
