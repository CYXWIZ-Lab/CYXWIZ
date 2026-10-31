// Trainable nodes from a real C++ plugin (TOFIX125 P2 step 2.5): the example
// plugin plugins/examples/activation_nodes is loaded through the plugin
// manager, its Softsign and Snake register as trainable extension nodes,
// pass the gradient check through the engine's factory, and train inside the
// tiny causal-LM example graph.
#include <cyxwiz/gradient_check.h>
#include "../src/core/extension_node_registry.h"
#include "../src/core/graph_training_job.h"
#include "../src/plugin/plugin_manager.h"
#include "../src/plugin/plugin_version_constraint.h"
#include "../src/plugin/registries/plugin_panel_registry.h"
#include "../src/plugin/registries/plugin_training_hook_manager.h"
#include "causal_lm_token_window_fixture.h"
#include "route_qualification_test_fixture.h"

#include <cyxwiz/sequential.h>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#ifndef CYXWIZ_SOURCE_ROOT
#error "CYXWIZ_SOURCE_ROOT must name the repository root"
#endif

namespace fs = std::filesystem;

namespace {

constexpr const char* kPluginId = "com.cyxwiz.examples.activation-nodes";
constexpr int kExtensionNodeId = 50;
constexpr int kDecoderNodeId = 8;
constexpr int kDenseNodeId = 9;

int g_failures = 0;

void Check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok   " : "  FAIL ") << what << "\n";
    if (!condition) ++g_failures;
}

nlohmann::json GraphWith(const fs::path& root, const std::string& type_name,
                         const std::map<std::string, std::string>& parameters) {
    nlohmann::json graph = cyxwiz::test::LoadTokenWindowGraph(root);
    for (auto& link : graph["links"]) {
        if (link["from_node"] == kDecoderNodeId && link["to_node"] == kDenseNodeId) {
            link["to_node"] = kExtensionNodeId;
            link["to_pin_index"] = 0;
        }
    }
    nlohmann::json params = nlohmann::json::object();
    for (const auto& [key, value] : parameters) params[key] = value;
    graph["nodes"].push_back(
        {{"id", kExtensionNodeId}, {"type", static_cast<int>(gui::NodeType::PluginCustom)},
         {"category", static_cast<int>(gui::NodeCategory::Plugin)}, {"name", type_name},
         {"description", ""}, {"parameters", params}, {"pos_x", 0.0}, {"pos_y", 0.0},
         {"extension",
          {{"contract", 1}, {"type_id", std::string(kPluginId) + ":" + type_name}, {"version", "1.0.0"},
           {"content_hash", ""},
           {"inputs", nlohmann::json::array({{{"name", "Input"}, {"type", "Tensor"}}})},
           {"outputs", nlohmann::json::array({{{"name", "Output"}, {"type", "Tensor"}}})}}}});
    graph["links"].push_back({{"id", 900}, {"from_node", kExtensionNodeId}, {"from_pin", 0},
                              {"from_pin_index", 0}, {"to_node", kDenseNodeId}, {"to_pin", 0},
                              {"to_pin_index", 0}, {"link_type", 0}});
    return graph;
}

// True while the plugin's library is loaded in this process.
bool LibraryLoaded() {
#ifdef _WIN32
    return ::GetModuleHandleA("activation_nodes.dll") != nullptr;
#else
    return true;
#endif
}

// A module of the trained model's layer stack (the job returns any
// executable model; these graphs train as a layer stack).
cyxwiz::Module* FindModule(cyxwiz::IExecutableModel& executable, const std::string& name) {
    cyxwiz::SequentialModel* model = executable.AsSequentialModel();
    if (!model) return nullptr;
    for (size_t i = 0; i < model->Size(); ++i) {
        if (model->GetModule(i) && model->GetModule(i)->GetName() == name) return model->GetModule(i);
    }
    return nullptr;
}

}  // namespace

int main() {
    const fs::path root = CYXWIZ_SOURCE_ROOT;
    const fs::path plugin_dir = root / "plugins" / "examples" / "activation_nodes";
    const fs::path work = fs::temp_directory_path() / "cyxwiz_plugin_tensor_nodes_test";
    std::error_code ec;
    fs::remove_all(work, ec);
    fs::create_directories(work);
    const fs::path parquet = work / "tokens.parquet";
    if (!cyxwiz::test::WriteTokenWindows(root, parquet, 0, 8)) {
        std::cerr << "could not write the Parquet fixture\n";
        return EXIT_FAILURE;
    }
    cyxwiz::test::InstallQualifiedRouteSnapshot();

    auto& manager = cyxwiz::plugin::PluginManager::Instance();
    auto& registry = cyxwiz::ExtensionNodeRegistry::Instance();

    std::cout << "the plugin registers two trainable nodes\n";
    Check(manager.LoadPlugin(plugin_dir), "the example plugin loads from " + plugin_dir.string());
    Check(manager.InitializePlugin(kPluginId), "it initialises");
    const auto softsign = registry.Find(std::string(kPluginId) + ":Softsign");
    const auto snake = registry.Find(std::string(kPluginId) + ":Snake");
    Check(softsign && softsign->kind == cyxwiz::ExtensionNodeKind::TensorLayer &&
              softsign->role == cyxwiz::ExtensionLayerRole::Activation,
          "Softsign is a trainable activation");
    Check(snake && snake->metadata.parameters.size() == 1 &&
              snake->metadata.parameters[0].name == "initial_a" &&
              snake->metadata.parameters[0].type == "float" &&
              snake->metadata.parameters[0].default_value == "1.0",
          "Snake has its typed parameter");
    Check(snake && snake->menu_category == "Activations" && snake->version == "1.0.0" &&
              snake->color == 0xFF3C8F4A,
          "category, version and colour come from the plugin");

    std::cout << "both pass the gradient check through the engine's factory\n";
    for (const char* name : {"Softsign", "Snake"}) {
        const auto factory = registry.TensorFactory(std::string(kPluginId) + ":" + name);
        Check(factory != nullptr, std::string(name) + " has a factory");
        if (!factory) continue;
        const auto result = cyxwiz::CheckModuleGradients([&](const std::vector<size_t>& shape) {
            std::string error;
            const std::vector<size_t> sample(shape.begin() + 1, shape.end());  // without the batch
            return factory->CreateModule(sample, {{"initial_a", "0.8"}}, error);
        });
        Check(result.passed, std::string(name) + ": " + cyxwiz::DescribeGradientCheck(result));
    }

    std::cout << "a shape refusal from the plugin reaches the engine\n";
    {
        const auto factory = registry.TensorFactory(std::string(kPluginId) + ":Snake");
        const auto refused = factory->InferOutputShape({8, 32}, {{"initial_a", "-1"}});
        Check(!refused.ok && refused.error == "initial_a must be greater than 0",
              "Snake refuses a negative frequency: " + refused.error);
    }

    std::cout << "Snake trains in the example graph and learns its frequency\n";
    {
        cyxwiz::GraphTrainingJobRequest request;
        request.graph_json = GraphWith(root, "Snake", {{"initial_a", "1.0"}}).dump();
        request.dataset_files["tiny_causal_lm_tokens"] = parquet.string();
        request.epochs_override = 3;
        request.checkpoint_dir_override = (work / "ck_snake").string();
        std::vector<float> losses;
        cyxwiz::GraphTrainingJobCallbacks callbacks;
        callbacks.on_epoch = [&losses](int, float train_loss, float, float, float, float) {
            losses.push_back(train_loss);
        };
        auto result = cyxwiz::RunGraphTrainingJob(request, callbacks);
        Check(result.ok, "the job succeeded" + (result.error.empty() ? "" : " (" + result.error + ")"));
        cyxwiz::Module* module = result.model ? FindModule(*result.model, "Snake") : nullptr;
        Check(module != nullptr, "the model holds the plugin's Snake module");
        if (module) {
            const auto params = module->GetParameters();
            const float a = params.count("a") ? params.at("a").ReadData<float>()[0] : 1.0f;
            Check(std::isfinite(a) && std::fabs(a - 1.0f) > 1e-6f,
                  "the optimizer changed a from 1.0 (now " + std::to_string(a) + ")");
        }
        Check(losses.size() == 3 && losses.back() < losses.front(),
              "the loss goes down (" + (losses.empty() ? std::string("none")
                                                       : std::to_string(losses.front()) + " -> " +
                                                             std::to_string(losses.back())) + ")");
    }

    std::cout << "Softsign trains in the example graph\n";
    {
        cyxwiz::GraphTrainingJobRequest request;
        request.graph_json = GraphWith(root, "Softsign", {}).dump();
        request.dataset_files["tiny_causal_lm_tokens"] = parquet.string();
        request.epochs_override = 2;
        request.checkpoint_dir_override = (work / "ck_softsign").string();
        auto result = cyxwiz::RunGraphTrainingJob(request);
        Check(result.ok, "the job succeeded" + (result.error.empty() ? "" : " (" + result.error + ")"));
        Check(result.model && FindModule(*result.model, "Softsign") != nullptr,
              "the model holds the plugin's Softsign module");
    }

    std::cout << "unloading while a model still holds a module keeps the code loaded until it is released\n";
    {
        const std::string type_id = std::string(kPluginId) + ":Snake";
        auto lease = registry.ProviderLease(type_id);  // what GuardedExtensionModule holds
        std::string error;
        auto module = registry.TensorFactory(type_id)->CreateModule({2, 3}, {{"initial_a", "1.0"}}, error);
        Check(module != nullptr && lease != nullptr, "a module and its lease are held");

        manager.UnloadPlugin(kPluginId);
        Check(!registry.Has(type_id), "unloading removes the plugin's nodes at once");
        Check(manager.GetLoadedPlugin(kPluginId) == nullptr, "the plugin is no longer listed");
        Check(LibraryLoaded(), "its library stays loaded while the module lives");
        const float values[6] = {0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f};
        const cyxwiz::Tensor x(std::vector<size_t>{2, 3}, values);
        Check(module && module->Forward(x).NumElements() == 6, "the module still runs after the unload");

        module.reset();
        Check(LibraryLoaded(), "the lease alone still keeps it");
        lease.reset();
        Check(!LibraryLoaded(), "releasing the last holder frees the library");
    }

    std::cout << "unloading with nothing in use frees the library at once\n";
    {
        Check(manager.LoadPlugin(plugin_dir) && manager.InitializePlugin(kPluginId), "loads again");
        Check(registry.Has(std::string(kPluginId) + ":Snake"), "its nodes are back");
        manager.UnloadPlugin(kPluginId);
        Check(!LibraryLoaded(), "nothing held it: the library is freed with the unload");
    }

    std::cout << "a node the Engine refuses is reported with its reason, not only logged\n";
    {
        // Something else already holds the Snake type id, so the plugin's
        // Snake cannot register; Softsign still does.
        struct NoFactory : cyxwiz::ITensorNodeFactory {
            cyxwiz::ExtensionShapeResult InferOutputShape(const std::vector<size_t>&,
                                                          const std::map<std::string, std::string>&) const override {
                return {};
            }
            std::unique_ptr<cyxwiz::Module> CreateModule(const std::vector<size_t>&,
                                                         const std::map<std::string, std::string>&,
                                                         std::string&) const override {
                return nullptr;
            }
        };
        cyxwiz::ExtensionNodeRegistry::Registration taken;
        taken.descriptor.provider_id = kPluginId;
        taken.descriptor.type_name = "Snake";
        taken.descriptor.kind = cyxwiz::ExtensionNodeKind::TensorLayer;
        taken.descriptor.source_path = "a test that took the id first";
        taken.tensor_factory = std::make_shared<NoFactory>();
        std::string error;
        const bool registered = registry.Register(std::move(taken), error);
        Check(registered, "the Snake id is taken first" + (error.empty() ? std::string() : ": " + error));

        Check(manager.LoadPlugin(plugin_dir) && manager.InitializePlugin(kPluginId), "the plugin still loads");
        const auto* loaded = manager.GetLoadedPlugin(kPluginId);
        Check(loaded && loaded->skipped_nodes.size() == 1, "one node is reported as not added");
        Check(loaded && !loaded->skipped_nodes.empty() &&
                  loaded->skipped_nodes[0].find("Snake") != std::string::npos &&
                  loaded->skipped_nodes[0].find("already registered") != std::string::npos,
              "the reason names the node and why: " +
                  (loaded && !loaded->skipped_nodes.empty() ? loaded->skipped_nodes[0] : std::string("none")));
        Check(registry.Has(std::string(kPluginId) + ":Softsign"), "the other node is added");
        manager.UnloadPlugin(kPluginId);
    }

    std::cout << "plugin.json dependency versions\n";
    {
        using cyxwiz::plugin::PluginVersion;
        using cyxwiz::plugin::VersionMeetsConstraint;
        const PluginVersion v = PluginVersion::Parse("1.4.2");
        Check(VersionMeetsConstraint(v, "") && VersionMeetsConstraint(v, "*"), "no constraint accepts any version");
        Check(VersionMeetsConstraint(v, "1.4.2") && !VersionMeetsConstraint(v, "1.4.3"), "a bare version is exact");
        Check(VersionMeetsConstraint(v, ">=1.4") && !VersionMeetsConstraint(v, ">1.4.2") &&
                  VersionMeetsConstraint(v, "<2") && !VersionMeetsConstraint(v, "<=1.4.1"),
              ">=, >, <, <= compare");
        Check(VersionMeetsConstraint(v, "^1.2.0") && !VersionMeetsConstraint(v, "^2.0.0") &&
                  !VersionMeetsConstraint(PluginVersion::Parse("0.3.0"), "^0.2.0"),
              "^ keeps the major (the minor below 1.0)");
        Check(VersionMeetsConstraint(v, "~1.4.0") && !VersionMeetsConstraint(v, "~1.3.0"), "~ keeps major.minor");
        Check(VersionMeetsConstraint(v, ">=1.0 <2.0") && !VersionMeetsConstraint(v, ">=1.0 <1.4"),
              "several constraints must all hold");
        std::string error;
        Check(!VersionMeetsConstraint(v, ">=one", &error) && error.find("cannot read") != std::string::npos,
              "an unreadable constraint is refused with the reason: " + error);
    }

    std::cout << "a plugin starts only after its dependencies\n";
    {
        // A copy of the example plugin under another id, with dependencies.
        const fs::path dependent = work / "dependent";
        fs::create_directories(dependent / "bin");
        fs::copy(plugin_dir / "bin", dependent / "bin", fs::copy_options::recursive | fs::copy_options::overwrite_existing);
        nlohmann::json manifest;
        std::ifstream(plugin_dir / "plugin.json") >> manifest;
        const std::string dependent_id = "com.cyxwiz.test.dependent";
        manifest["id"] = dependent_id;
        const auto start = [&](const nlohmann::json& dependencies) {
            manifest["dependencies"] = dependencies;
            std::ofstream(dependent / "plugin.json") << manifest.dump(2);
            manager.UnloadPlugin(dependent_id);
            std::string error;
            Check(manager.LoadPlugin(dependent, &error), "the dependent copy loads" + (error.empty() ? "" : ": " + error));
            return manager.InitializePlugin(dependent_id);
        };
        const auto reason = [&]() {
            const auto* loaded = manager.GetLoadedPlugin(dependent_id);
            return loaded ? loaded->error_message : std::string("not loaded");
        };
        using cyxwiz::plugin::PluginState;

        // Each result first: the message reads the reason it left.
        const auto outcome = [&](const nlohmann::json& dependencies, PluginState state, const std::string& words) {
            const bool started = start(dependencies);
            return !started && manager.GetPluginState(dependent_id) == state &&
                   reason().find(words) != std::string::npos;
        };
        bool ok = outcome({{{"id", "com.cyxwiz.test.missing"}, {"version", ">=1.0.0"}}}, PluginState::Failed,
                          "not installed");
        Check(ok, "a missing dependency stops it: " + reason());

        manager.UnloadPlugin(kPluginId);
        Check(manager.LoadPlugin(plugin_dir), "the dependency loads (not started)");
        ok = outcome({{{"id", kPluginId}, {"version", ">=2.0.0"}}}, PluginState::Failed, "version 1.0.0 is installed");
        Check(ok, "a dependency of another version stops it: " + reason());

        ok = outcome({{{"id", kPluginId}, {"version", "^1.0.0"}}}, PluginState::Loaded, "waits for");
        Check(ok, "a dependency not started yet makes it wait: " + reason());
        Check(manager.InitializePlugin(kPluginId) &&
                  manager.GetPluginState(dependent_id) == PluginState::Initialized,
              "it starts when its dependency does");
        manager.UnloadPlugin(dependent_id);
        manager.UnloadPlugin(kPluginId);
    }

    std::cout << "a loaded plugin's build output is not locked; a rebuild loads as a new copy\n";
    {
        const fs::path rebuild = work / "rebuild";
        fs::create_directories(rebuild);
        fs::copy(plugin_dir / "bin", rebuild / "bin", fs::copy_options::recursive | fs::copy_options::overwrite_existing);
        nlohmann::json manifest;
        std::ifstream(plugin_dir / "plugin.json") >> manifest;
        const std::string rebuild_id = "com.cyxwiz.test.rebuild";
        manifest["id"] = rebuild_id;
        std::ofstream(rebuild / "plugin.json") << manifest.dump(2);

        std::string error;
        Check(manager.LoadPlugin(rebuild, &error) && manager.InitializePlugin(rebuild_id),
              "the plugin loads" + (error.empty() ? "" : ": " + error));
        const auto* first = manager.GetLoadedPlugin(rebuild_id);
        const fs::path built = first ? first->dll_path : fs::path();
        const fs::path first_copy = first ? first->loaded_from : fs::path();
        Check(!first_copy.empty() && first_copy != built && fs::is_regular_file(first_copy),
              "it runs from a private copy: " + first_copy.string());
        bool writable = false;
        {
            // A rebuild: the linker rewrites the file. Trailing bytes keep
            // the library loadable and change its content.
            std::ofstream out(built, std::ios::binary | std::ios::app);
            out << "rebuilt";
            writable = static_cast<bool>(out);
        }
        Check(writable, "the build output can be rewritten while the plugin is loaded");

        manager.UnloadPlugin(rebuild_id);
        error.clear();
        Check(manager.LoadPlugin(rebuild, &error) && manager.InitializePlugin(rebuild_id),
              "the rebuilt plugin loads again" + (error.empty() ? "" : ": " + error));
        const auto* second = manager.GetLoadedPlugin(rebuild_id);
        const fs::path second_copy = second ? second->loaded_from : fs::path();
        Check(!second_copy.empty() && second_copy.parent_path() != first_copy.parent_path(),
              "the rebuild runs from a new copy: " + second_copy.string());
        manager.UnloadPlugin(rebuild_id);
    }

    std::cout << "a provider offered twice registers once\n";
    {
        // A plugin may offer the same object through QueryInterface and a
        // context Register* call: one registration, no duplicate warnings.
        struct Hook final : cyxwiz::plugin::ITrainingHook {};
        struct Panels final : cyxwiz::plugin::IPanelProvider {
            std::vector<cyxwiz::plugin::PluginPanelInfo> GetPanels() override {
                cyxwiz::plugin::PluginPanelInfo info;
                info.panel_id = "com.cyxwiz.test.twice.panel";
                info.title = "Twice";
                return {info};
            }
            void RenderPanel(const std::string&, bool*) override {}
        };
        Hook hook;
        Panels panels;
        auto& hooks = cyxwiz::plugin::PluginTrainingHookManager::Instance();
        auto& panel_registry = cyxwiz::plugin::PluginPanelRegistry::Instance();
        const size_t hooks_before = hooks.GetHookCount();
        hooks.RegisterHook("com.cyxwiz.test.twice", &hook);
        hooks.RegisterHook("com.cyxwiz.test.twice", &hook);
        Check(hooks.GetHookCount() == hooks_before + 1, "a training hook offered twice is called once");
        panel_registry.Register("com.cyxwiz.test.twice", &panels);
        panel_registry.RegisterDirect("com.cyxwiz.test.twice", "com.cyxwiz.test.twice.panel", "Twice", "", false,
                                      &panels);
        Check(panel_registry.CountForPlugin("com.cyxwiz.test.twice") == 1, "a panel offered twice is listed once");
        hooks.RemoveByPlugin("com.cyxwiz.test.twice");
        panel_registry.RemoveByPlugin("com.cyxwiz.test.twice");
    }

    std::cout << "a folder that cannot load says why\n";
    {
        std::string error;
        const bool loaded = manager.LoadPlugin(work / "no-such-plugin", &error);
        Check(!loaded && !error.empty(), "LoadPlugin gives its reason: " + error);
    }

    fs::remove_all(work, ec);
    if (g_failures > 0) {
        std::cout << g_failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "test_plugin_tensor_nodes: all checks passed\n";
    return 0;
}
