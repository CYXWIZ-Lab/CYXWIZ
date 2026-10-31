// cyxwiz-plugin-check: load C++ plugin folders the way the Engine does and
// check their trainable nodes (TOFIX125 P5 step 5.3, Part D8).
//
//   cyxwiz-plugin-check [--open <file>]... <plugin folder> [<plugin folder> ...]
//
// For each folder: read plugin.json, load the library (the build fingerprint
// is checked before the plugin is created; the library's static
// initialisers have already run), grant the permissions plugin.json declares
// for this run, initialise the plugin, list the
// nodes it registers, and for each trainable node check its output shape and
// its gradients with the node's default settings, as the Plugin Manager's
// "Check gradient" does; list its dataset formats and analytics (TOFIX181).
// With --open, read each file through the plugin's dataset format as Data
// Input's Apply does (the format its extension and CanLoad pick, every batch,
// the data validated), and run each of its analytics on the columns that
// analytic reads, as Explore > Plugins does.
// Exit code 0 when every folder loads, registers something, every node
// passes, and every file opens and every analytic answers.
#include <cyxwiz/gradient_check.h>
#include "../src/core/extension_node_registry.h"
#include "../src/plugin/plugin_loader.h"
#include "../src/plugin/plugin_manager.h"
#include "../src/plugin/registries/plugin_analytics_registry.h"
#include "../src/plugin/registries/plugin_data_loader_registry.h"
#include "../src/plugin/security/permission_store.h"

#include <arrow/api.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::string ShapeText(const std::vector<size_t>& shape) {
    std::string text;
    for (const size_t d : shape) text += (text.empty() ? "" : ", ") + std::to_string(d);
    return "[" + text + "]";
}

std::string Number(double value) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(4) << value;
    return text.str();
}

bool Reads(cyxwiz::plugin::AnalyticsColumns columns, const arrow::DataType& type) {
    switch (columns) {
        case cyxwiz::plugin::AnalyticsColumns::Numeric:
            return arrow::is_integer(type.id()) || arrow::is_floating(type.id());
        case cyxwiz::plugin::AnalyticsColumns::Text:
            return arrow::is_string(type.id());
        case cyxwiz::plugin::AnalyticsColumns::Any:
            break;
    }
    return true;
}

// Each analytic of the plugin on the columns of `table` it reads (up to its
// maximum), as Explore > Plugins runs it with its default columns.
bool RunAnalytics(const std::string& plugin_id, const std::string& file_name,
                  const std::shared_ptr<arrow::Table>& table) {
    auto& analytics = cyxwiz::plugin::PluginAnalyticsRegistry::Instance();
    bool ok = true;
    for (const auto& analytic : analytics.Analytics()) {
        if (analytic.plugin_id != plugin_id) continue;
        const auto& info = analytic.info;
        std::vector<int> columns;
        std::string names;
        for (int i = 0; i < table->num_columns(); ++i) {
            if (info.max_columns > 0 && static_cast<int>(columns.size()) >= info.max_columns) break;
            if (!Reads(info.columns, *table->field(i)->type())) continue;
            columns.push_back(i);
            names += (names.empty() ? "" : ", ") + table->field(i)->name();
        }
        const int needed = std::max(1, info.min_columns);
        if (static_cast<int>(columns.size()) < needed) {
            std::cout << "  analytic " << info.name << " on " << file_name << ": not run, the file has "
                      << columns.size() << " column(s) it reads and it needs " << needed << "\n";
            continue;
        }
        auto chosen = table->SelectColumns(columns);
        if (!chosen.ok()) {
            std::cout << "  FAIL analytic " << info.name << " on " << file_name << ": " << chosen.status().message()
                      << "\n";
            ok = false;
            continue;
        }
        const auto result =
            analytics.Compute(info.analytics_id, std::make_shared<arrow::TableBatchReader>(*chosen));
        if (!result.success) {
            std::cout << "  FAIL analytic " << info.name << " on " << file_name << " (" << names
                      << "): " << result.error_message << "\n";
            ok = false;
            continue;
        }
        std::string answer;
        if (result.has_scalar) answer = Number(result.scalar_value);
        for (size_t i = 0; i < result.vector_value.size(); ++i) {
            const std::string label = i < result.vector_labels.size() ? result.vector_labels[i] : std::to_string(i + 1);
            answer += (answer.empty() ? "" : ", ") + label + " " + Number(result.vector_value[i]);
        }
        if (answer.empty()) answer = result.text_value.empty() ? "(no values)" : result.text_value;
        std::cout << "  ok   analytic " << info.name << " on " << file_name << " (" << names << "): " << answer
                  << "\n";
    }
    return ok;
}

// Reads `file` through the plugin's dataset format as Data Input's Apply
// does, then runs the plugin's analytics on it.
bool OpenThrough(const std::string& plugin_id, const fs::path& file) {
    auto& loaders = cyxwiz::plugin::PluginDataLoaderRegistry::Instance();
    std::vector<cyxwiz::plugin::PluginDataLoaderRegistry::Loader> own;
    for (const auto& loader : loaders.Loaders())
        if (loader.plugin_id == plugin_id) own.push_back(loader);
    const std::string file_name = file.filename().string();
    if (own.empty()) {
        std::cout << "  open " << file_name << ": the plugin adds no dataset format\n";
        return true;
    }
    if (!fs::is_regular_file(file)) {
        std::cout << "  FAIL open " << file_name << ": no such file (" << file.string() << ")\n";
        return false;
    }
    std::string extension = file.extension().string();
    if (!extension.empty()) extension.erase(0, 1);
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const auto by_extension = std::find_if(own.begin(), own.end(), [&](const auto& loader) {
        return std::find(loader.info.extensions.begin(), loader.info.extensions.end(), extension) !=
               loader.info.extensions.end();
    });
    if (by_extension == own.end()) {
        std::cout << "  FAIL open " << file_name << ": no dataset format of this plugin reads ." << extension
                  << " files\n";
        return false;
    }
    const std::string loader_id = loaders.LoaderForFile(file.string());
    const auto chosen = std::find_if(own.begin(), own.end(),
                                     [&](const auto& loader) { return loader.info.loader_id == loader_id; });
    if (chosen == own.end()) {
        std::cout << "  FAIL open " << file_name << ": " << by_extension->info.format_name
                  << " does not take the file (its CanLoad returned false)\n";
        return false;
    }

    auto opened = loaders.Open(loader_id, file.string());
    if (!opened.ok()) {
        std::cout << "  FAIL open " << file_name << ": " << opened.status().message() << "\n";
        return false;
    }
    std::shared_ptr<arrow::RecordBatchReader> reader = std::move(opened).ValueUnsafe();
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    while (true) {
        std::shared_ptr<arrow::RecordBatch> batch;
        const auto status = reader->ReadNext(&batch);
        if (!status.ok()) {
            std::cout << "  FAIL open " << file_name << ": batch " << batches.size() + 1 << ": " << status.message()
                      << "\n";
            return false;
        }
        if (!batch) break;
        batches.push_back(std::move(batch));
    }
    const auto schema = reader->schema();
    reader.reset();  // releases the plugin's stream, as Apply does before using the data
    auto table = arrow::Table::FromRecordBatches(schema, batches);
    if (!table.ok()) {
        std::cout << "  FAIL open " << file_name << ": " << table.status().message() << "\n";
        return false;
    }
    if (const auto valid = (*table)->ValidateFull(); !valid.ok()) {
        std::cout << "  FAIL open " << file_name << ": the data is not valid Arrow: " << valid.message() << "\n";
        return false;
    }
    std::string columns;
    for (const auto& field : schema->fields())
        columns += (columns.empty() ? "" : ", ") + field->name() + " (" + field->type()->ToString() + ")";
    std::cout << "  ok   open " << file_name << " as " << chosen->info.format_name << ": " << (*table)->num_rows()
              << " rows, " << (*table)->num_columns() << " columns in " << batches.size()
              << (batches.size() == 1 ? " batch" : " batches") << "\n";
    std::cout << "         columns: " << columns << "\n";
    return RunAnalytics(plugin_id, file_name, *table);
}

bool CheckFolder(const fs::path& folder, const std::vector<fs::path>& files) {
    std::cout << folder.string() << "\n";
    cyxwiz::plugin::PluginManifest manifest;
    std::string error;
    if (!cyxwiz::plugin::PluginLoader::ParseManifest(folder, manifest, error)) {
        std::cout << "  FAIL plugin.json: " << error << "\n";
        return false;
    }
    std::cout << "  plugin " << manifest.id << " " << manifest.version.ToString() << " (API "
              << manifest.api_version.ToString() << ")\n";

    auto& manager = cyxwiz::plugin::PluginManager::Instance();
    std::string load_error;
    if (!manager.LoadPlugin(folder, &load_error)) {
        std::cout << "  FAIL the plugin did not load: " << load_error << "\n";
        return false;
    }
    // The Engine asks the user before a plugin with a dangerous permission
    // starts; a command line cannot, so the check grants what plugin.json
    // declares for this run only (the store is never loaded or saved here).
    auto& permissions = cyxwiz::plugin::security::PermissionStore::Instance();
    const std::string version = manifest.version.ToString();
    for (const auto permission : permissions.GetUndecidedDangerous(manifest.id, version, manifest.permissions)) {
        permissions.SetDecision(manifest.id, version, permission,
                                cyxwiz::plugin::security::PermissionDecision::Allowed);
        for (const auto& name : cyxwiz::plugin::PermissionsToStrings(static_cast<uint32_t>(permission)))
            std::cout << "  permission " << name << ": granted for this check only (the Engine asks the user)\n";
    }
    if (!manager.InitializePlugin(manifest.id)) {
        const auto* loaded = manager.GetLoadedPlugin(manifest.id);
        std::cout << "  FAIL the plugin did not initialise"
                  << (loaded && !loaded->error_message.empty() ? ": " + loaded->error_message : std::string())
                  << "\n";
        manager.UnloadPlugin(manifest.id);
        return false;
    }

    auto& registry = cyxwiz::ExtensionNodeRegistry::Instance();
    bool ok = true;
    size_t nodes = 0;
    for (const auto& node : registry.All()) {
        if (node.provider_id != manifest.id) continue;
        ++nodes;
        const auto factory = registry.TensorFactory(node.type_id);
        if (!factory) {
            std::cout << "  node " << node.type_id << ": not trainable, nothing to check\n";
            continue;
        }
        std::map<std::string, std::string> settings;
        for (const auto& parameter : node.metadata.parameters) settings[parameter.name] = parameter.default_value;

        // The gradient check's batches (GradientCheckOptions::input_shapes);
        // a node that refuses one of their sample shapes is checked with the
        // others, and says which it refused and why.
        cyxwiz::GradientCheckOptions options;
        std::vector<std::vector<size_t>> accepted;
        for (const auto& batch : options.input_shapes) {
            const std::vector<size_t> sample(batch.begin() + 1, batch.end());
            const auto shape = factory->InferOutputShape(sample, settings);
            if (shape.ok) {
                accepted.push_back(batch);
            } else {
                std::cout << "  node " << node.type_id << ": refuses a " << ShapeText(sample)
                          << " input with its default settings: " << shape.error << "\n";
            }
        }
        if (accepted.empty()) {
            std::cout << "  FAIL node " << node.type_id << ": refuses every input shape the check uses\n";
            ok = false;
            continue;
        }
        options.input_shapes = accepted;
        const auto result = cyxwiz::CheckModuleGradients(
            [&](const std::vector<size_t>& input_shape) {
                std::string create_error;
                const std::vector<size_t> sample(input_shape.begin() + 1, input_shape.end());
                auto module = factory->CreateModule(sample, settings, create_error);
                if (!module && !create_error.empty()) std::cout << "  CreateModule: " << create_error << "\n";
                return module;
            },
            options);
        std::cout << (result.passed ? "  ok   node " : "  FAIL node ") << node.type_id << ": "
                  << cyxwiz::DescribeGradientCheck(result) << "\n";
        ok = ok && result.passed;
    }
    size_t formats = 0;
    for (const auto& loader : cyxwiz::plugin::PluginDataLoaderRegistry::Instance().Loaders()) {
        if (loader.plugin_id != manifest.id) continue;
        ++formats;
        std::string extensions;
        for (const auto& extension : loader.info.extensions) extensions += (extensions.empty() ? "." : ", .") + extension;
        std::cout << "  ok   dataset format " << loader.info.format_name << " (" << extensions << ")\n";
    }
    size_t analytics = 0;
    for (const auto& analytic : cyxwiz::plugin::PluginAnalyticsRegistry::Instance().Analytics()) {
        if (analytic.plugin_id != manifest.id) continue;
        ++analytics;
        std::cout << "  ok   analytic " << analytic.info.name
                  << (analytic.info.category.empty() ? std::string() : " (" + analytic.info.category + ")") << "\n";
    }
    if (const auto* loaded = manager.GetLoadedPlugin(manifest.id)) {
        for (const auto& reason : loaded->skipped_nodes) {
            std::cout << "  FAIL not added: " << reason << "\n";
            ok = false;
        }
    }
    if (nodes + formats + analytics == 0) {
        std::cout << "  FAIL the plugin registered no nodes, dataset formats or analytics\n";
        ok = false;
    }
    for (const auto& file : files) ok = OpenThrough(manifest.id, file) && ok;
    manager.UnloadPlugin(manifest.id);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<fs::path> folders;
    std::vector<fs::path> files;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--open" && i + 1 < argc) {
            files.push_back(fs::absolute(argv[++i]));
        } else if (argument.rfind("--", 0) == 0) {
            folders.clear();
            break;
        } else {
            folders.push_back(fs::absolute(argument));
        }
    }
    if (folders.empty()) {
        std::cout << "usage: cyxwiz-plugin-check [--open <file>]... <plugin folder> [<plugin folder> ...]\n"
                     "  --open <file>  read the file through the plugin's dataset format and run its analytics\n";
        return 2;
    }
    bool ok = true;
    for (const auto& folder : folders) ok = CheckFolder(folder, files) && ok;
    std::cout << (ok ? "cyxwiz-plugin-check: all plugins pass\n" : "cyxwiz-plugin-check: FAILED\n");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
