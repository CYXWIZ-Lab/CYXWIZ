#include "starter_graphs.h"

#include "icons.h"

#include <array>
#include <cstdlib>
#include <filesystem>

namespace gui {

namespace {

// examples/cyxgraph/<filename> next to the folder the Engine was launched
// from, or up to four folders above the working folder (a build tree).
std::string ResolveStarterGraphPath(const char* filename) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (const char* launch_cwd = std::getenv("CYXWIZ_LAUNCH_CWD")) {
        const fs::path candidate = fs::path(launch_cwd) / "examples" / "cyxgraph" / filename;
        if (fs::exists(candidate, ec)) return fs::weakly_canonical(candidate, ec).string();
    }
    const fs::path cwd = fs::current_path(ec);
    const std::array<fs::path, 7> roots = {cwd,
                                           cwd.parent_path(),
                                           cwd.parent_path().parent_path(),
                                           cwd.parent_path().parent_path().parent_path(),
                                           cwd.parent_path().parent_path().parent_path().parent_path(),
                                           cwd / "cyxwiz-engine",
                                           cwd / ".." / "cyxwiz-engine"};
    for (const auto& root : roots) {
        const fs::path candidate = root / "examples" / "cyxgraph" / filename;
        if (fs::exists(candidate, ec)) return fs::weakly_canonical(candidate, ec).string();
    }
    return {};
}

}  // namespace

std::vector<StarterGraph> FindStarterGraphs() {
    struct Def {
        const char* title;
        const char* description;
        const char* domain;
        const char* icon;
        const char* filename;
    };
    static constexpr std::array<Def, 5> defs = {{
        {"Binary image classification", "Cats-vs-dogs training graph for a two-class image dataset.", "Binary classification", ICON_FA_IMAGES, "cats_dogs_classifier.cyxgraph"},
        {"Multiclass image classification", "MNIST MLP graph for a tabular digit dataset with ten classes.", "Multiclass classification", ICON_FA_IMAGE, "mnist_mlp.cyxgraph"},
        {"Text classification", "Mental-health sentiment graph: a Text Tokenizer, an Embedding and a seven-class classifier.", "Text classification", ICON_FA_COMMENTS, "mental_health_sentiment_classifier_v2.cyxgraph"},
        {"Audio classification", "Speech-command graph for labeled command utterances.", "Audio classification", ICON_FA_WAVE_SQUARE, "speech_command_classifier.cyxgraph"},
        {"Time-series forecasting", "Airline-passengers dense forecaster using a real time-series training graph.", "Forecasting / regression", ICON_FA_CHART_LINE, "timeseries/airline_passengers_dense.cyxgraph"}}};
    std::vector<StarterGraph> found;
    for (const auto& d : defs) {
        const std::string path = ResolveStarterGraphPath(d.filename);
        if (!path.empty()) found.push_back({d.title, d.description, d.domain, d.icon, path});
    }
    return found;
}

}  // namespace gui
