// Preferences > Hyperparameter Search (TOFIX178 T4): how a new search starts,
// its strategy, trials, time limit and score. Engine-wide, applied and saved
// at once; the Hyperparameter Search panel follows them.

#include "toolbar.h"

#include "hyperparam_search_panel.h"
#include "../ui_buttons.h"
#include "../../core/engine_config.h"
#include "../../core/hyperparam_search_presentation.h"

#include <imgui.h>

namespace cyxwiz {

void ToolbarPanel::RenderSearchPreferences() {
    auto& engine_config = core::EngineConfig::Instance();
    const core::HyperparamSearchDefaults saved = engine_config.GetSearchDefaults();

    ImGui::Spacing();
    ImGui::Text("A new search starts with");
    ImGui::Separator();
    ImGui::Spacing();

    SearchConfig config;
    search_presentation::ApplyDefaults(config, saved);
    ImGui::PushID("search_defaults");
    const bool changed = SearchSettingsFields(config, false);
    ImGui::PopID();

    ImGui::Spacing();
    if (ui::SecondaryButton("Reset", !(saved == core::HyperparamSearchDefaults{}), "These are the Engine defaults")) {
        engine_config.SetSearchDefaults({});
        engine_config.Save();
    }

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("Train > Hyperparameter Search sets these when it opens, and again when you change them here "
                        "(not while a search runs). What to vary is picked per graph in the panel.");
    ImGui::PopTextWrapPos();

    if (changed) {
        engine_config.SetSearchDefaults(search_presentation::DefaultsOf(config));
        engine_config.Save();
    }
}

}  // namespace cyxwiz
