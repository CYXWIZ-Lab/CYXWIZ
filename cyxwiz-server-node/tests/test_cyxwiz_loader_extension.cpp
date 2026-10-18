/**
 * CyxWizLoader and extension nodes (TOFIX125 P6 step 6.4)
 *
 * A model with an extension node (a user's C++ plugin or Python node) runs
 * the extension's own code, which the Server Node does not have. The loader
 * used to drop such a layer and load the rest; it now refuses the model.
 */

#include <catch2/catch_test_macros.hpp>

#include "../src/model_loader.h"

#include <nlohmann/json.hpp>
#include <spdlog/sinks/ringbuffer_sink.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

json ExtensionNode() {
    return {{"id", 50},
            {"type", 321},
            {"name", "My Scale"},
            {"parameters", json::object()},
            {"extension",
             {{"contract", 1}, {"type_id", "test.files:Scale"}, {"version", "1.0.0"}, {"content_hash", ""}}}};
}

json DenseNode() {
    return {{"id", 2}, {"type", 0}, {"name", "Dense"}, {"parameters", {{"units", "4"}}}};
}

// A directory package: manifest.json (no "modules") and graph.cyxgraph.
fs::path WritePackage(const std::string& name, const json& manifest, const json& graph) {
    const fs::path dir = fs::temp_directory_path() / "cyxwiz_loader_extension_test" / name;
    fs::remove_all(dir);
    fs::create_directories(dir / "weights");
    std::ofstream(dir / "manifest.json") << manifest.dump(2);
    std::ofstream(dir / "graph.cyxgraph") << graph.dump(2);
    return dir;
}

// Loads `dir` and returns whether the log said why it refused.
bool RefusedForExtension(const fs::path& dir, bool& loaded) {
    auto sink = std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(64);
    auto previous = spdlog::default_logger();
    spdlog::set_default_logger(std::make_shared<spdlog::logger>("loader_test", sink));
    cyxwiz::servernode::CyxWizLoader loader;
    loaded = loader.Load(dir.string()) || loader.IsLoaded();
    spdlog::set_default_logger(previous);
    for (const auto& line : sink->last_formatted()) {
        if (line.find("cannot run extension nodes yet") != std::string::npos &&
            line.find("test.files:Scale") != std::string::npos)
            return true;
    }
    return false;
}

}  // namespace

TEST_CASE("CyxWizLoader refuses a package whose manifest lists extensions", "[cyxwiz_loader][extension]") {
    const json manifest = {{"version", "1.0"},
                           {"format", "cyxmodel"},
                           {"extensions", json::array({{{"type_id", "test.files:Scale"},
                                                        {"version", "1.0.0"},
                                                        {"content_hash", ""}}})}};
    const json graph = {{"nodes", json::array({DenseNode(), ExtensionNode()})}, {"links", json::array()}};
    const auto dir = WritePackage("listed", manifest, graph);

    bool loaded = true;
    REQUIRE(RefusedForExtension(dir, loaded));
    REQUIRE_FALSE(loaded);
}

TEST_CASE("CyxWizLoader refuses a graph with an extension node when the manifest does not list it",
          "[cyxwiz_loader][extension]") {
    const json manifest = {{"version", "1.0"}, {"format", "cyxmodel"}};
    const json graph = {{"nodes", json::array({DenseNode(), ExtensionNode()})}, {"links", json::array()}};
    const auto dir = WritePackage("graph_only", manifest, graph);

    bool loaded = true;
    REQUIRE(RefusedForExtension(dir, loaded));
    REQUIRE_FALSE(loaded);
}
