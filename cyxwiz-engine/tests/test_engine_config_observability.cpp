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

    std::error_code error;
    std::filesystem::remove(path, error);
    Check(!error, "temporary test config should be removable");
    std::cout << "Engine observability config contracts passed\n";
    return 0;
}
