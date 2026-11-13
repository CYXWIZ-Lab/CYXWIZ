#include "layer_capture_controls.h"

#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <imgui.h>

#include <string>

namespace cyxwiz {

bool RenderLayerCaptureControls(core::StudioDebuggerConfig& settings) {
    using core::LayerCaptureWhen;
    const core::StudioDebuggerConfig before = settings;

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
    }
    ImGui::PopTextWrapPos();
    return !(settings == before);
}

}  // namespace cyxwiz
