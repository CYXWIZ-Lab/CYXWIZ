// Preferences > Training: the user's defaults for training choices that new
// nodes start with. Engine-wide, applied and saved at once.

#include "toolbar.h"

#include "../../core/engine_config.h"
#include "../ui_buttons.h"
#include "../ui_widgets.h"

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

    // How the Training Dashboard's charts start; its View options change
    // them for the session
    using TD = core::TrainingDashboardConfig;
    const TD saved_view = engine_config.GetTrainingDashboard();
    TD view = saved_view;
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Text("Training Dashboard");
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("How the charts start each time the Engine starts. The dashboard's View options change them "
                        "for the session.");
    ImGui::PopTextWrapPos();
    ImGui::Checkbox("Auto scale", &view.auto_scale);
    ImGui::Checkbox("Follow the latest epoch", &view.follow_current_epoch);
    ImGui::SameLine();
    ImGui::BeginDisabled(!view.follow_current_epoch);
    ImGui::TextDisabled("epochs shown");
    ImGui::SameLine();
    ui::NumberField("##dash_epoch_window", &view.epoch_window, TD::kMinEpochWindow, TD::kMaxEpochWindow, 1, 44.0f);
    ImGui::EndDisabled();
    ImGui::Checkbox("Log loss axis", &view.log_loss_axis);
    ImGui::Checkbox("Smooth", &view.smooth);
    ImGui::SameLine();
    ImGui::BeginDisabled(!view.smooth);
    ImGui::TextDisabled("points in the moving average");
    ImGui::SameLine();
    ui::NumberField("##dash_smooth_window", &view.smooth_window, TD::kMinSmoothWindow, TD::kMaxSmoothWindow, 1,
                    44.0f);
    ImGui::EndDisabled();
    if (!(view == saved_view)) {
        engine_config.SetTrainingDashboard(view);
        engine_config.Save();
    }
}

}  // namespace cyxwiz
