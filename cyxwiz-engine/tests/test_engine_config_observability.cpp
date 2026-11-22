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
    // Preferences > Tools > Background Tasks (TOFIX129 C6): finished tasks listed by default
    cyxwiz::core::MonitorConfig monitor = config.GetMonitor();
    Check(monitor.show_finished_tasks, "Background Tasks lists finished tasks by default");
    monitor.show_finished_tasks = false;
    config.SetMonitor(monitor);
    Check(config.Save(path), "the Background Tasks default should persist");
    monitor.show_finished_tasks = true;
    config.SetMonitor(monitor);
    Check(config.Load(path), "the Background Tasks default should reload");
    Check(!config.GetMonitor().show_finished_tasks, "the Background Tasks default survives a round trip");
    monitor.show_finished_tasks = true;
    config.SetMonitor(monitor);
    // Preferences > General (TOFIX129 B9): how the Engine starts and closes
    Check(!config.StartWithLastProject() && config.RecentProjectsKept() == EngineConfig::kDefaultRecentProjects &&
              !config.AskBeforeClosingAlways(),
          "the general defaults: start page, 10 recent projects, ask when something would be lost");
    config.SetRecentProjectsKept(500);
    Check(config.RecentProjectsKept() == EngineConfig::kMaxRecentProjects, "at most 50 recent projects");
    config.SetRecentProjectsKept(0);
    Check(config.RecentProjectsKept() == EngineConfig::kMinRecentProjects, "at least 1 recent project");
    config.SetStartWithLastProject(true);
    config.SetRecentProjectsKept(25);
    config.SetAskBeforeClosingAlways(true);
    Check(config.Save(path), "the general settings should persist");
    config.SetStartWithLastProject(false);
    config.SetRecentProjectsKept(10);
    config.SetAskBeforeClosingAlways(false);
    Check(config.Load(path), "the general settings should reload");
    Check(config.StartWithLastProject() && config.RecentProjectsKept() == 25 && config.AskBeforeClosingAlways(),
          "the general settings survive a round trip");

    // Preferences > Script Editor, Python, Studio, Training (TOFIX129 B9)
    Check(config.GetEditor() == cyxwiz::core::EditorConfig{} && config.GetPythonStart() == cyxwiz::core::PythonStartConfig{} &&
              config.GetStudio() == cyxwiz::core::StudioConfig{} &&
              config.GetTrainingDashboard() == cyxwiz::core::TrainingDashboardConfig{},
          "the editor, Python start, Studio and dashboard defaults");
    cyxwiz::core::EditorConfig editor;
    editor.tab_size = 3;  // not offered: 4
    editor.new_file_line_endings = 9;
    config.SetEditor(editor);
    Check(config.GetEditor().tab_size == 4 && config.GetEditor().new_file_line_endings == 2, "editor values kept in range");
    editor.tab_size = 2;
    editor.word_wrap = true;
    editor.minimap = false;
    editor.new_file_line_endings = 1;
    editor.trim_trailing_whitespace = true;
    editor.final_newline = false;
    config.SetEditor(editor);
    cyxwiz::core::PythonStartConfig start;
    start.startup_scripts = {"C:/scripts/start.py", "C:/scripts/helpers.py"};
    start.import_numpy = false;
    start.console_commands_kept = 99999;
    config.SetPythonStart(start);
    Check(config.GetPythonStart().console_commands_kept == cyxwiz::core::PythonStartConfig::kMaxCommandsKept,
          "commands kept stays in range");
    start.console_commands_kept = 2000;
    config.SetPythonStart(start);
    cyxwiz::core::StudioConfig studio;
    studio.auto_save_at_start = true;
    studio.minimap = false;
    config.SetStudio(studio);
    cyxwiz::core::TrainingDashboardConfig view;
    view.follow_current_epoch = true;
    view.epoch_window = 1000;
    view.smooth = true;
    view.smooth_window = 1;
    config.SetTrainingDashboard(view);
    Check(config.GetTrainingDashboard().epoch_window == 50 && config.GetTrainingDashboard().smooth_window == 2,
          "dashboard windows stay in range");
    view.epoch_window = 12;
    view.smooth_window = 9;
    config.SetTrainingDashboard(view);
    Check(config.Save(path), "the B9 settings should persist");
    config.SetEditor({});
    config.SetPythonStart({});
    config.SetStudio({});
    config.SetTrainingDashboard({});
    Check(config.Load(path), "the B9 settings should reload");
    Check(config.GetEditor() == editor && config.GetPythonStart() == start && config.GetStudio() == studio &&
              config.GetTrainingDashboard() == view,
          "the editor, Python start, Studio and dashboard settings survive a round trip");

    std::error_code error;
    std::filesystem::remove(path, error);
    Check(!error, "temporary test config should be removable");
    std::cout << "Engine observability config contracts passed\n";
    return 0;
}
