// _cyxwiz_envs: plugin environments (IEnvironmentProvider) in a Python
// process of their own (TOFIX195 S3.4). It loads plugin folders with the
// Engine's own PluginManager, as cyxwiz-plugin-check does, and hands
// batches of copies to Python as numpy arrays. pycyxwiz.rl builds
// gymnasium environments on it, so Stable-Baselines3 or CleanRL can train
// on the same physics the Engine trains on.
//
//   load_plugin(folder) -> plugin id       environments() -> [dict]
//   Batch(key, count, settings) .reset(seed) .step(actions) .close()

#include "plugin_loader.h"
#include "plugin_manager.h"
#include "registries/plugin_environment_registry.h"
#include "security/permission_store.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using cyxwiz::plugin::EnvironmentBatch;
using cyxwiz::plugin::PluginEnvironmentInfo;
using cyxwiz::plugin::PluginEnvironmentRegistry;

namespace {

std::vector<std::string> loaded_plugins;

// Loads and starts the plugin in `folder`. A permission the Engine would ask
// the user about is denied: this process cannot ask.
std::string LoadPlugin(const std::string& folder_text) {
    const std::filesystem::path folder = std::filesystem::absolute(folder_text);
    cyxwiz::plugin::PluginManifest manifest;
    std::string error;
    if (!cyxwiz::plugin::PluginLoader::ParseManifest(folder, manifest, error))
        throw std::runtime_error("plugin.json in " + folder.string() + ": " + error);
    auto& manager = cyxwiz::plugin::PluginManager::Instance();
    if (manager.GetLoadedPlugin(manifest.id)) return manifest.id;
    if (!manager.LoadPlugin(folder, &error)) throw std::runtime_error("The plugin did not load: " + error);
    auto& permissions = cyxwiz::plugin::security::PermissionStore::Instance();
    const std::string version = manifest.version.ToString();
    for (const auto permission : permissions.GetUndecidedDangerous(manifest.id, version, manifest.permissions))
        permissions.SetDecision(manifest.id, version, permission, cyxwiz::plugin::security::PermissionDecision::Denied);
    if (!manager.InitializePlugin(manifest.id)) {
        const auto* loaded = manager.GetLoadedPlugin(manifest.id);
        const std::string reason = loaded ? loaded->error_message : std::string();
        manager.UnloadPlugin(manifest.id);
        throw std::runtime_error("The plugin did not start" + (reason.empty() ? std::string(".") : ": " + reason));
    }
    loaded_plugins.push_back(manifest.id);
    return manifest.id;
}

// Unloads what load_plugin started (at interpreter exit). A batch still
// alive keeps its library loaded until it goes.
void Shutdown() {
    auto& manager = cyxwiz::plugin::PluginManager::Instance();
    for (auto it = loaded_plugins.rbegin(); it != loaded_plugins.rend(); ++it) manager.UnloadPlugin(*it);
    loaded_plugins.clear();
}

py::dict Describe(const PluginEnvironmentRegistry::Environment& environment) {
    const PluginEnvironmentInfo& info = environment.info;
    py::dict d;
    d["key"] = environment.key;
    d["plugin_id"] = environment.plugin_id;
    d["id"] = info.id;
    d["name"] = info.name;
    d["description"] = info.description;
    d["category"] = info.category;
    d["observation_size"] = info.observation_size;
    d["discrete_actions"] = info.discrete_actions;
    d["action_size"] = info.action_size;
    d["action_count"] = info.action_count;
    d["action_low"] = info.action_low;
    d["action_high"] = info.action_high;
    d["max_episode_steps"] = info.max_episode_steps;
    d["step_seconds"] = info.step_seconds;
    py::list settings;
    for (const auto& setting : info.settings) {
        py::dict s;
        s["key"] = setting.key;
        s["label"] = setting.label;
        s["default"] = setting.default_value;
        s["description"] = setting.description;
        settings.append(s);
    }
    d["settings"] = settings;
    return d;
}

py::list Environments() {
    py::list out;
    for (const auto& environment : PluginEnvironmentRegistry::Instance().Environments()) out.append(Describe(environment));
    return out;
}

// Accepts a plugin key ("<plugin id>:<id>") or an id one plugin offers.
std::string ResolveKey(const std::string& name) {
    PluginEnvironmentRegistry::Environment found;
    if (PluginEnvironmentRegistry::Instance().Find(name, found)) return name;
    std::vector<std::string> matches;
    for (const auto& environment : PluginEnvironmentRegistry::Instance().Environments())
        if (environment.info.id == name) matches.push_back(environment.key);
    if (matches.size() == 1) return matches[0];
    if (matches.empty())
        throw std::runtime_error("No loaded plugin offers the environment '" + name + "'. Load its plugin first (load_plugin).");
    throw std::runtime_error("More than one plugin offers '" + name + "': use the full key, such as '" + matches[0] + "'.");
}

class Batch {
public:
    Batch(const std::string& name, int count, const std::map<std::string, std::string>& settings) {
        std::string error;
        batch_ = PluginEnvironmentRegistry::Instance().Create(ResolveKey(name), settings, count, error);
        if (!batch_) throw std::runtime_error(error);
    }

    int Count() const { return Live().Count(); }
    int ObservationSize() const { return Live().ObservationSize(); }
    int ActionSize() const { return Live().ActionSize(); }
    py::dict Info() const {
        return Describe({Live().Key(), Live().PluginId(), Live().Info()});
    }

    py::array_t<float> Reset(uint64_t seed) {
        auto& batch = Live();
        py::array_t<float> obs({batch.Count(), batch.ObservationSize()});
        std::string error;
        bool ok = false;
        {
            py::gil_scoped_release release;
            ok = batch.Reset(seed, obs.mutable_data(), error);
        }
        if (!ok) throw std::runtime_error(error);
        return obs;
    }

    py::tuple Step(const py::array_t<float, py::array::c_style | py::array::forcecast>& actions) {
        auto& batch = Live();
        const py::ssize_t count = batch.Count();
        const py::ssize_t action_size = batch.ActionSize();
        if (actions.size() != count * action_size)
            throw std::invalid_argument("Expected " + std::to_string(count) + " x " + std::to_string(action_size) +
                                        " actions, got " + std::to_string(actions.size()) + " values.");
        const py::ssize_t obs_size = batch.ObservationSize();
        py::array_t<float> obs({count, obs_size});
        py::array_t<float> final_obs({count, obs_size});
        py::array_t<float> rewards(count);
        py::array_t<bool> terminated(count);
        py::array_t<bool> truncated(count);
        std::fill(final_obs.mutable_data(), final_obs.mutable_data() + final_obs.size(), 0.0f);
        std::vector<uint8_t> term(count), trunc(count);
        const cyxwiz::plugin::PluginEnvironmentStepBuffers out{obs.mutable_data(), rewards.mutable_data(), term.data(),
                                                               trunc.data(), final_obs.mutable_data()};
        std::string error;
        bool ok = false;
        {
            py::gil_scoped_release release;
            ok = batch.Step(actions.data(), out, error);
        }
        if (!ok) throw std::runtime_error(error);
        for (py::ssize_t i = 0; i < count; ++i) {
            terminated.mutable_data()[i] = term[i] != 0;
            truncated.mutable_data()[i] = trunc[i] != 0;
        }
        return py::make_tuple(obs, rewards, terminated, truncated, final_obs);
    }

    void Close() { batch_.reset(); }

private:
    EnvironmentBatch& Live() const {
        if (!batch_) throw std::runtime_error("The batch is closed.");
        return *batch_;
    }
    std::unique_ptr<EnvironmentBatch> batch_;
};

}  // namespace

PYBIND11_MODULE(_cyxwiz_envs, m) {
    m.doc() = "Plugin environments for reinforcement learning (use pycyxwiz.rl)";
    m.def("load_plugin", &LoadPlugin, py::arg("folder"),
          "Load and start the plugin in a folder (plugin.json); returns its id");
    m.def("environments", &Environments, "The environments the loaded plugins offer");
    m.def("_shutdown", &Shutdown, "Unload the plugins load_plugin started");
    py::class_<Batch>(m, "Batch")
        .def(py::init<const std::string&, int, const std::map<std::string, std::string>&>(), py::arg("environment"),
             py::arg("count") = 1, py::arg("settings") = std::map<std::string, std::string>{})
        .def_property_readonly("count", &Batch::Count)
        .def_property_readonly("observation_size", &Batch::ObservationSize)
        .def_property_readonly("action_size", &Batch::ActionSize)
        .def_property_readonly("info", &Batch::Info)
        .def("reset", &Batch::Reset, py::arg("seed"))
        .def("step", &Batch::Step, py::arg("actions"),
             "Returns (observations, rewards, terminated, truncated, final_observations)")
        .def("close", &Batch::Close);
    py::module_::import("atexit").attr("register")(m.attr("_shutdown"));
}
