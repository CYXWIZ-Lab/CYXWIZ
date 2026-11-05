// Preferences > Studio Debugger (TOFIX178 6.1c, approved Capture mockup): the
// user's defaults for layer capture and the layer views. Engine-wide, applied
// and saved at once; each run's capture control starts from them.

#include "toolbar.h"

#include "../../core/engine_config.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <imgui.h>

#include <string>

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

    ImGui::TextUnformatted("When");
    static const char* const kWhen[] = {"Off", "Local Debug runs", "Training, every N epochs"};
    int when = static_cast<int>(settings.capture_when);
    if (ui::SegmentedControl("##capture_when", kWhen, 3, &when)) settings.capture_when = static_cast<LayerCaptureWhen>(when);

    ImGui::PushTextWrapPos(0.0f);
    if (settings.capture_when == LayerCaptureWhen::Off) {
        ImGui::TextDisabled("Nothing is captured; Local Debug and training run exactly as without it.");
    } else {
        if (settings.capture_when == LayerCaptureWhen::Training) {
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted("Every");
            ImGui::SameLine();
            ui::NumberField("##capture_every", &settings.capture_every_epochs, 1, 1000, 1, 60.0f,
                            "Epochs between snapshots");
            ImGui::SameLine();
            ImGui::TextUnformatted("epochs, and the last one");
            ImGui::TextDisabled("Local Debug runs are captured too. A snapshot runs the sample once, forward and "
                                "backward, with the model in evaluation mode: training comes out the same.");
        }

        ImGui::Spacing();
        ImGui::TextUnformatted("What, for every layer");
        ImGui::Checkbox("Activations: each layer's input and output", &settings.capture_activations);
        ImGui::Checkbox("Parameters: weights and biases; BatchNorm running statistics as kept state",
                        &settings.capture_parameters);
        ImGui::Checkbox("Gradients of the parameters, from the same step", &settings.capture_gradients);
        if (!settings.capture_activations && !settings.capture_parameters && !settings.capture_gradients)
            ImGui::TextColored(ui::CurrentTokens().warning, "Nothing is ticked, so nothing is captured.");

        ImGui::Spacing();
        ImGui::TextUnformatted("On");
        const std::string batch = "A batch of " + std::to_string(core::kLayerCaptureBatchSamples);
        const char* const inputs[] = {"The picked sample", batch.c_str()};
        int input = settings.capture_batch ? 1 : 0;
        if (ui::SegmentedControl("##capture_input", inputs, 2, &input)) settings.capture_batch = input == 1;
        ImGui::TextDisabled("The Studio Debugger's sample number; a batch takes the items after it too.");

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
