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

    std::error_code error;
    std::filesystem::remove(path, error);
    Check(!error, "temporary test config should be removable");
    std::cout << "Engine observability config contracts passed\n";
    return 0;
}
