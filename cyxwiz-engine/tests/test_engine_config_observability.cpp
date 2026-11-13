#include "../src/core/engine_config.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

} // namespace

int main() {
    const auto unique = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto path = std::filesystem::temp_directory_path() /
        ("cyxwiz_observability_" + unique + ".json");

    auto& config = cyxwiz::core::EngineConfig::Instance();
    const std::vector<cyxwiz::core::RuntimeLogSavedFilterConfig> expected = {
        {"training_errors", "category=training and level>=error"},
        {"legacy_invalid", "unknown=value"}};

    config.SetRuntimeLogSavedFilters(expected);
    Check(config.Save(path), "saved filters should persist to a test config");
    config.SetRuntimeLogSavedFilters({});
    Check(config.Load(path), "saved filters should reload from the test config");
    Check(config.GetRuntimeLogSavedFilters() == expected,
          "valid and invalid saved expressions should survive round trip");

    // Studio Debugger defaults (TOFIX178 6.3c): Where it looks' method and
    // what saliency shows survive a save and load.
    cyxwiz::core::StudioDebuggerConfig debugger = config.GetStudioDebugger();
    debugger.where_it_looks_method = cyxwiz::core::WhereItLooksMethod::Saliency;
    debugger.saliency_show = cyxwiz::core::SaliencyShow::TimesInput;
    debugger.top_images_channels = cyxwiz::core::TopImagesChannels::Strongest;
    debugger.top_images_rank = cyxwiz::core::TopImagesRank::Mean;
    debugger.embedding_method = cyxwiz::core::EmbeddingMethod::Umap;
    debugger.embedding_colour = cyxwiz::core::EmbeddingColour::PredictedClass;
    debugger.unit_weights_order = cyxwiz::core::UnitWeightsOrder::Number;
    config.SetStudioDebugger(debugger);
    Check(config.Save(path), "the Studio Debugger defaults save");
    config.SetStudioDebugger({});
    Check(config.Load(path), "the Studio Debugger defaults reload");
    Check(config.GetStudioDebugger() == debugger, "the layer views' saved choices survive a round trip");

    // Preferences > Simulation: the graph run's defaults, kept within range
    using cyxwiz::core::EngineConfig;
    config.SetSimulationRateHz(5000);
    Check(config.SimulationRateHz() == EngineConfig::kMaxSimulationRateHz, "the tick rate is at most 1000");
    config.SetSimulationRateHz(0);
    Check(config.SimulationRateHz() == EngineConfig::kMinSimulationRateHz, "the tick rate is at least 1");
    config.SetSimulationRateHz(120);
    config.SetOpenSimulationMonitorOnRun(false);
    Check(config.Save(path), "simulation defaults should persist");
    config.SetSimulationRateHz(60);
    config.SetOpenSimulationMonitorOnRun(true);
    Check(config.Load(path), "simulation defaults should reload");
    Check(config.SimulationRateHz() == 120 && !config.OpenSimulationMonitorOnRun(),
          "the tick rate and the open-on-run choice survive a round trip");

    // A plugin's own settings, and the Stress test's defaults (kept in range)
    config.SetPluginSetting("com.example.sim", "viewport.stress", "{\"kick\":1.5}");
    config.SetPolicyStressTestDefaults({3, 50, 0});
    const auto clamped = config.PolicyStressTestDefaults();
    Check(clamped.kind == 3 && clamped.levels == 21 && clamped.episodes == 1, "the stress test defaults stay in range");
    config.SetPolicyStressTestDefaults({4, 7, 12});
    Check(config.Save(path), "plugin settings and stress test defaults should persist");
    config.SetPluginSetting("com.example.sim", "viewport.stress", "changed");
    config.SetPolicyStressTestDefaults({});
    Check(config.Load(path), "plugin settings and stress test defaults should reload");
    std::string stored;
    Check(config.PluginSetting("com.example.sim", "viewport.stress", stored) && stored == "{\"kick\":1.5}",
          "a plugin's setting survives a round trip");
    Check(!config.PluginSetting("com.example.sim", "never.set", stored), "a key the plugin never set is not there");
    const auto reloaded = config.PolicyStressTestDefaults();
    Check(reloaded.kind == 4 && reloaded.levels == 7 && reloaded.episodes == 12,
          "the stress test defaults survive a round trip");
    // Watching a PPO run (Preferences > Simulation), kept in range
    config.SetWatchTraining(7, 0);
    Check(config.WatchTrainingMode() == 2 && config.WatchTrainingEvery() == 1, "the watch default stays in range");
    config.SetWatchTraining(1, 25);
    Check(config.Save(path), "the watch default should persist");
    config.SetWatchTraining(0, 20);
    Check(config.Load(path), "the watch default should reload");
    Check(config.WatchTrainingMode() == 1 && config.WatchTrainingEvery() == 25, "the watch default survives a round trip");

    std::error_code error;
    std::filesystem::remove(path, error);
    Check(!error, "temporary test config should be removable");
    std::cout << "Engine observability config contracts passed\n";
    return 0;
}
