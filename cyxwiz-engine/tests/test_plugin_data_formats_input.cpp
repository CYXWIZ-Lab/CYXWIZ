// A plugin's dataset format through Data Input's path (TOFIX181 step 3): the
// core's external-format hook, format detection and validation, and
// DataRegistry::LoadTabularExternal, with the example plugin
// plugins/examples/data_formats (ARFF) loaded by the real plugin manager.
//   - "auto" resolves a .arff file to the plugin format; .csv stays CSV;
//   - the file streams into the Parquet cache and loads in memory, on disk,
//     or with selected columns; a second load reuses the cache;
//   - a bad file fails with the plugin's reason;
//   - pipeline validation accepts the plugin format while it is installed;
//   - after unloading the plugin the format is refused with a reason.
#include "../src/core/data_input_formats.h"
#include "../src/core/data_registry.h"
#include "../src/core/arrow_dataset.h"
#include "../src/core/parquet_backed_dataset.h"
#include "../src/core/pipeline_runtime_capabilities.h"
#include "../src/plugin/plugin_data_formats.h"
#include "../src/plugin/plugin_manager.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

namespace fs = std::filesystem;
using cyxwiz::DataRegistry;

constexpr const char* kPluginId = "com.cyxwiz.examples.data-formats";
constexpr const char* kFormat = "plugin:com.cyxwiz.examples.data-formats.arff";

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

fs::path PluginDir() {
    auto dir = fs::current_path();
    while (!dir.empty()) {
        const auto candidate = dir / "plugins" / "examples" / "data_formats";
        if (fs::exists(candidate / "plugin.json")) return candidate;
        const auto parent = dir.parent_path();
        if (parent == dir) break;
        dir = parent;
    }
    return {};
}

void Write(const fs::path& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary);
    out << text;
}

bool AllowedInPipelines(const std::string& name) {
    for (const auto& capability : cyxwiz::ResolvePipelineAllowedParameterValues("DataInput")) {
        if (std::string(capability.parameter_name) != "file_type") continue;
        for (const char* value : capability.allowed_values)
            if (value && name == value) return true;
    }
    return false;
}

}  // namespace

int main() {
    const auto dir = PluginDir();
    Check(!dir.empty(), "plugins/examples/data_formats not found; run from the repository root");
    const auto work = fs::temp_directory_path() / "cyxwiz_test_plugin_data_formats_input";
    fs::remove_all(work);
    fs::create_directories(work / "cache");
    const std::string cache = (work / "cache").string();

    const auto arff = work / "iris.arff";
    Write(arff,
          "@relation iris\n"
          "@attribute 'sepal length' numeric\n"
          "@attribute petal numeric\n"
          "@attribute class {a,b}\n"
          "@data\n"
          "5.1,1.4,a\n"
          "4.9,?,b\n"
          "6.2,4.5,a\n");

    // Before the plugin: the built-in ARFF entry explains it cannot read it.
    cyxwiz::plugin::InstallPluginDataFormats();
    Check(cyxwiz::data_input::ResolveFormat("auto", arff.string()) == "arff",
          "without the plugin a .arff file resolves to the built-in, unreadable ARFF entry");

    auto& manager = cyxwiz::plugin::PluginManager::Instance();
    Check(manager.LoadPlugin(dir) && manager.InitializePlugin(kPluginId), "the data formats example loads and starts");

    // --- Detection and validation ---
    Check(cyxwiz::data_input::ResolveFormat("auto", arff.string()) == kFormat,
          "with the plugin \"auto\" resolves a .arff file to the plugin format, got " +
              cyxwiz::data_input::ResolveFormat("auto", arff.string()));
    Check(cyxwiz::data_input::ResolveFormat("auto", (work / "a.csv").string()) == "csv",
          "a .csv file stays CSV");
    Check(cyxwiz::data_input::IsExecutable(kFormat), "the plugin format is executable while installed");
    Check(AllowedInPipelines(kFormat), "pipeline validation accepts the plugin format");
    const auto formats = cyxwiz::data_input::ExternalFormats();
    Check(formats.size() == 1 && formats[0].label == "ARFF (Weka)" && formats[0].version == std::string(kPluginId) + "@1.0.0",
          "the external format list carries the plugin's label and version");
    Check(cyxwiz::data_input::ExternalFormatId(kFormat) >= cyxwiz::data_input::kFirstExternalFormatId &&
              std::strcmp(cyxwiz::data_input::ExternalFormatNameOf(cyxwiz::data_input::ExternalFormatId(kFormat)),
                          kFormat) == 0,
          "the dialog number maps back to the format name");

    // --- Loading ---
    auto& registry = DataRegistry::Instance();
    std::string status;
    auto backend = registry.LoadTabularExternal(kFormat, arff.string(), "iris", false, {}, &status, cache);
    Check(backend == DataRegistry::TabularLoadBackend::InMemory, "a small file loads in memory: " + status);
    auto dataset = registry.GetArrowDataset("iris");
    Check(dataset && dataset->GetNumRows() == 3 && dataset->GetNumColumns() == 3, "3 rows and 3 columns in memory");
    const auto cache_file = cyxwiz::ParquetBackedDataset::GetCacheFilePath(
        arff.string(), std::string("format=") + kFormat + "@" + kPluginId + "@1.0.0", cache);
    Check(fs::exists(cache_file), "the file was streamed into the Parquet cache at " + cache_file);
    const auto cached_at = fs::last_write_time(cache_file);

    backend = registry.LoadTabularExternal(kFormat, arff.string(), "iris_disk", true, {}, &status, cache);
    Check(backend == DataRegistry::TabularLoadBackend::DiskBacked, "force disk-backed keeps it on disk: " + status);
    Check(fs::last_write_time(cache_file) == cached_at, "a second load reuses the cache");

    backend = registry.LoadTabularExternal(kFormat, arff.string(), "iris_one", false, {"sepal length"}, &status, cache);
    dataset = registry.GetArrowDataset("iris_one");
    Check(backend == DataRegistry::TabularLoadBackend::InMemory && dataset && dataset->GetNumColumns() == 1,
          "selected columns load only those columns: " + status);

    const auto bad = work / "bad.arff";
    Write(bad, "@relation bad\n@attribute a numeric\n@attribute b numeric\n@data\n1,2\n3\n");
    backend = registry.LoadTabularExternal(kFormat, bad.string(), "bad", false, {}, &status, cache);
    Check(backend == DataRegistry::TabularLoadBackend::Failed && status.find("line 6") != std::string::npos,
          "a bad file fails with the plugin's reason, got: " + status);

    // --- After unloading ---
    manager.UnloadPlugin(kPluginId);
    Check(!cyxwiz::data_input::IsExecutable(kFormat), "the format is not executable once its plugin is unloaded");
    Check(!AllowedInPipelines(kFormat), "pipeline validation no longer lists it");
    Check(std::string(cyxwiz::data_input::UnsupportedReason(kFormat, false)).find("plugin that is not loaded") !=
              std::string::npos,
          "the reason says the plugin is not loaded");
    backend = registry.LoadTabularExternal(kFormat, arff.string(), "iris_gone", false, {}, &status, cache);
    Check(backend == DataRegistry::TabularLoadBackend::Failed && status.find("not loaded") != std::string::npos,
          "loading fails with the reason, got: " + status);
    dataset = registry.GetArrowDataset("iris");
    Check(dataset && dataset->GetNumRows() == 3, "a dataset loaded through the plugin stays loaded");

    registry.UnregisterTabularDataset("iris");
    registry.UnregisterTabularDataset("iris_one");
    std::error_code ignored;
    fs::remove_all(work, ignored);
    std::cout << "test_plugin_data_formats_input: all checks passed\n";
    return 0;
}
