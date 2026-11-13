// Preferences > Training: the user's defaults for training choices that new
// nodes start with. Engine-wide, applied and saved at once.

#include "toolbar.h"

#include "../../core/engine_config.h"
#include "../ui_buttons.h"

#include <imgui.h>

namespace cyxwiz {

void ToolbarPanel::RenderTrainingPreferences() {
    auto& engine_config = core::EngineConfig::Instance();
    const core::TrainingDefaultsConfig saved = engine_config.GetTrainingDefaults();
    core::TrainingDefaultsConfig settings = saved;

    ImGui::Spacing();
    ImGui::Text("Fine-tuning");
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::TextUnformatted("A frozen BatchNorm");
    static const char* const kStatistics[] = {"Keeps its statistics", "Keeps updating statistics"};
    int statistics = settings.frozen_batchnorm_keeps_statistics ? 0 : 1;
    if (ui::SegmentedControl("##frozen_batchnorm_statistics", kStatistics, 2, &statistics))
        settings.frozen_batchnorm_keeps_statistics = statistics == 0;
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("What a new BatchNorm node's \"When frozen\" setting starts as. Keeping its statistics uses "
                        "the loaded running mean and variance while the rest trains, the usual choice for "
                        "fine-tuning on a small dataset. Keeping them updating is what PyTorch does when only "
                        "requires_grad is off. Each node can still be set on its own in Properties.");
    ImGui::PopTextWrapPos();

    if (!(settings == saved)) {
        engine_config.SetTrainingDefaults(settings);
        engine_config.Save();
    }
}

}  // namespace cyxwiz
