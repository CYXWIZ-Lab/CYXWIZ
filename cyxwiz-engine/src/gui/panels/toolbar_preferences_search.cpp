// Preferences > Search (TOFIX178 T4): how a new hyperparameter search starts
// (strategy, trials, time limit, score) and how Neural Architecture Search
// searches and how many of the best it trains. Engine-wide, applied and saved
// at once; both panels follow them while they are not running.

#include "toolbar.h"

#include "hyperparam_search_panel.h"
#include "nas_panel.h"
#include "../ui_buttons.h"
#include "../../core/engine_config.h"
#include "../../core/hyperparam_search_presentation.h"
#include "../../core/nas_presentation.h"

#include <imgui.h>

namespace cyxwiz {

void ToolbarPanel::RenderSearchPreferences() {
    auto& engine_config = core::EngineConfig::Instance();
    const core::HyperparamSearchDefaults saved = engine_config.GetSearchDefaults();

    ImGui::Spacing();
    ImGui::Text("Hyperparameter Search");
    ImGui::Separator();
    ImGui::Spacing();

    SearchConfig config;
    search_presentation::ApplyDefaults(config, saved);
    ImGui::PushID("search_defaults");
    const bool changed = SearchSettingsFields(config, false);
    if (ui::SecondaryButton("Reset", !(saved == core::HyperparamSearchDefaults{}), "These are the Engine defaults")) {
        engine_config.SetSearchDefaults({});
        engine_config.Save();
    }
    ImGui::PopID();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("Train > Hyperparameter Search sets these when it opens, and again when you change them here "
                        "(not while a search runs). What to vary is picked per graph in the panel.");
    ImGui::PopTextWrapPos();
    if (changed) {
        engine_config.SetSearchDefaults(search_presentation::DefaultsOf(config));
        engine_config.Save();
    }

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Text("Neural Architecture Search");
    ImGui::Separator();
    ImGui::Spacing();

    const core::ArchitectureSearchDefaults saved_architecture = engine_config.GetArchitectureSearchDefaults();
    NASSearchConfig architecture;
    int train_best = 0;
    nas_presentation::ApplyDefaults(architecture, train_best, saved_architecture);
    ImGui::PushID("architecture_defaults");
    const bool architecture_changed = ArchitectureSearchFields(architecture, &train_best);
    if (ui::SecondaryButton("Reset", !(saved_architecture == core::ArchitectureSearchDefaults{}),
                            "These are the Engine defaults")) {
        engine_config.SetArchitectureSearchDefaults({});
        engine_config.Save();
    }
    ImGui::PopID();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("Tools > Model > Neural Architecture Search sets these when it opens, and again when you "
                        "change them here (not while it searches or trains).");
    ImGui::PopTextWrapPos();
    if (architecture_changed) {
        engine_config.SetArchitectureSearchDefaults(nas_presentation::DefaultsOf(architecture, train_best));
        engine_config.Save();
    }
}

}  // namespace cyxwiz
