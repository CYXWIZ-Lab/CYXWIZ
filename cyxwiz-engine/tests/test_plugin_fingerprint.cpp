// Plugin build fingerprint (TOFIX125 P5 step 5.1, decision D4): comparing
// fingerprints, and the loader refusing a plugin built differently or built
// before API 1.2, before any of its code runs.
#include "../src/plugin/plugin_fingerprint.h"
#include "../src/plugin/plugin_loader.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#ifndef OK_PLUGIN
#error "OK_PLUGIN, OTHER_PLUGIN and MISSING_PLUGIN must name the test plugin libraries"
#endif

namespace fs = std::filesystem;
using namespace cyxwiz::plugin;

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok   " : "  FAIL ") << what << "\n";
    if (!condition) ++g_failures;
}

std::string With(std::string fingerprint, const std::string& key, const std::string& value) {
    const auto at = fingerprint.find(key + "=");
    const auto end = fingerprint.find(';', at);
    return fingerprint.replace(at + key.size() + 1,
                               (end == std::string::npos ? fingerprint.size() : end) - at - key.size() - 1, value);
}

std::string Lookup(const std::vector<std::pair<std::string, std::string>>& pairs, const std::string& key) {
    for (const auto& [k, v] : pairs) {
        if (k == key) return v;
    }
    return {};
}

// A plugin folder holding the library and a plugin.json naming it.
fs::path PluginFolder(const fs::path& work, const std::string& name, const fs::path& library) {
    const fs::path folder = work / name;
    fs::create_directories(folder);
    fs::copy_file(library, folder / library.filename(), fs::copy_options::overwrite_existing);
    std::ofstream(folder / "plugin.json") << R"({"id": "com.test.)" << name << R"(", "name": ")" << name
                                          << R"(", "version": "1.0.0", "api_version": "1.3.0",
        "platforms": {"windows": {"library": ")" << library.filename().string()
                                          << R"("}, "linux": {"library": ")" << library.filename().string()
                                          << R"("}, "macos": {"library": ")" << library.filename().string()
                                          << R"("}}})";
    return folder;
}

}  // namespace

int main() {
    std::cout << "the fingerprint\n";
    const std::string engine = CurrentBuildFingerprint();
    const auto pairs = ParseBuildFingerprint(engine);
    std::string keys;
    for (const auto& [k, v] : pairs) keys += k + ",";
    Check(keys == "sdk,api,compiler,stdlib,build,ptr,arrayfire,", "keys in order: " + keys + " (" + engine + ")");
    Check(CompareBuildFingerprints(engine, engine).empty(), "the same build matches");

    std::cout << "what counts as different\n";
    const auto build = CompareBuildFingerprints(engine, With(engine, "build", "debug-other"));
    Check(build.size() == 1 && build[0].key == "build", "build type differs");
    Check(CompareBuildFingerprints(engine, With(engine, "compiler", "gcc-13")).size() == 1, "compiler differs");
    Check(CompareBuildFingerprints(engine, With(engine, "ptr", "32")).size() == 1, "pointer size differs");
    Check(CompareBuildFingerprints(engine, With(engine, "sdk", "0.9")).size() == 1, "another SDK minor differs");
    Check(CompareBuildFingerprints(engine, With(engine, "api", "1.1")).empty(), "an older API minor is accepted");
    Check(CompareBuildFingerprints(engine, With(engine, "api", "1.9")).size() == 1, "a newer API minor is refused");
    Check(CompareBuildFingerprints(engine, With(engine, "api", "2.0")).size() == 1, "another API major is refused");
    Check(CompareBuildFingerprints(engine, With(engine, "arrayfire", "none")).empty(),
          "a plugin that does not include the backend skips the ArrayFire check");
    Check(CompareBuildFingerprints(engine, With(engine, "arrayfire", "3.1")).size() == 1 ||
              Lookup(pairs, "arrayfire") == "none",
          "another ArrayFire version differs");
    const std::string text = DescribeFingerprintMismatch(build);
    Check(text.find("The plugin was built with build=debug-other; this Engine with build=") == 0 &&
              text.find("Rebuild the plugin against the CyxWiz SDK") != std::string::npos,
          "the mismatch message names both values: " + text);
    Check(MissingFingerprintMessage(engine).find("no build fingerprint") != std::string::npos,
          "the missing-fingerprint message");

    std::cout << "the loader\n";
    const fs::path work = fs::temp_directory_path() / "cyxwiz_test_plugin_fingerprint";
    std::error_code ec;
    fs::remove_all(work, ec);
    {
        std::string error;
        auto ok = PluginLoader::LoadFromDirectory(PluginFolder(work, "ok", OK_PLUGIN), error);
        Check(ok != nullptr && ok->instance != nullptr, "a plugin built like the Engine loads" +
                                                            (error.empty() ? std::string() : ": " + error));
        if (ok) PluginLoader::Unload(*ok);
    }
    {
        std::string error;
        auto other = PluginLoader::LoadFromDirectory(PluginFolder(work, "other", OTHER_PLUGIN), error);
        Check(other == nullptr && error.find("The plugin was built with") == 0 &&
                  error.find("build=debug") != std::string::npos,
              "a plugin built differently is refused: " + error);
    }
    {
        std::string error;
        auto missing = PluginLoader::LoadFromDirectory(PluginFolder(work, "missing", MISSING_PLUGIN), error);
        Check(missing == nullptr && error.find("no build fingerprint") != std::string::npos,
              "a plugin built before API 1.2 is refused: " + error);
    }
    {
        std::string error;
        auto other = PluginLoader::LoadFromDLL(fs::path(OTHER_PLUGIN), error);
        Check(other == nullptr && error.find("The plugin was built with") == 0,
              "loading the library directly checks it too");
    }
    fs::remove_all(work, ec);

    std::cout << "every plugin in the repository carries the fingerprint\n";
    for (const auto& item : fs::recursive_directory_iterator(fs::path(CYXWIZ_SOURCE_ROOT) / "plugins", ec)) {
        if (item.path().filename() != "plugin.json") continue;
        const fs::path folder = item.path().parent_path();
        if (!fs::exists(folder / "bin")) continue;  // not built in this configuration
        std::string error;
        auto plugin = PluginLoader::LoadFromDirectory(folder, error);
        const bool fingerprint_ok = error.find("fingerprint") == std::string::npos &&
                                    error.find("The plugin was built with") == std::string::npos;
        Check(fingerprint_ok, folder.filename().string() + (plugin ? ": loads" : ": " + error));
        if (plugin) PluginLoader::Unload(*plugin);
    }

    if (g_failures > 0) {
        std::cout << g_failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "test_plugin_fingerprint: all checks passed\n";
    return 0;
}
