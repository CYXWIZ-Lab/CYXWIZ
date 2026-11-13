#include "plugin_environment_registry.h"

#include "../../core/extension_node_contract.h"
#include "../builtin/classic_control_environments.h"
#include "../security/safe_execute.h"

#include <spdlog/spdlog.h>

namespace cyxwiz::plugin {

namespace {

// The plugin's strings and vectors copied into the Engine's own.
PluginEnvironmentInfo CopyInfo(const PluginEnvironmentInfo& in) {
    PluginEnvironmentInfo out;
    out.id = std::string(in.id.c_str());
    out.name = std::string(in.name.c_str());
    out.description = std::string(in.description.c_str());
    out.category = std::string(in.category.c_str());
    out.observation_size = in.observation_size;
    out.discrete_actions = in.discrete_actions;
    out.action_size = in.action_size;
    out.action_count = in.action_count;
    out.action_low.assign(in.action_low.begin(), in.action_low.end());
    out.action_high.assign(in.action_high.begin(), in.action_high.end());
    out.max_episode_steps = in.max_episode_steps;
    out.step_seconds = in.step_seconds;
    out.view_panel_id = std::string(in.view_panel_id.c_str());
    for (const auto& setting : in.settings) {
        PluginEnvironmentSetting copy;
        copy.key = std::string(setting.key.c_str());
        copy.label = std::string(setting.label.c_str());
        copy.default_value = std::string(setting.default_value.c_str());
        copy.description = std::string(setting.description.c_str());
        copy.kind = setting.kind;
        for (const auto& option : setting.options)
            copy.options.push_back({std::string(option.value.c_str()), std::string(option.label.c_str())});
        copy.section = std::string(setting.section.c_str());
        copy.min = setting.min;
        copy.max = setting.max;
        copy.unit = std::string(setting.unit.c_str());
        out.settings.push_back(std::move(copy));
    }
    return out;
}

// What makes an environment usable at all; empty when it is.
std::string InfoProblem(const PluginEnvironmentInfo& info) {
    if (info.id.empty()) return "it has no id";
    if (info.observation_size <= 0) return "it observes nothing";
    if (info.discrete_actions) {
        if (info.action_size != 1 || info.action_count < 2) return "a discrete action space needs one action of 2 or more choices";
    } else {
        if (info.action_size <= 0) return "it takes no action";
        if (info.action_low.size() != static_cast<size_t>(info.action_size) ||
            info.action_high.size() != static_cast<size_t>(info.action_size))
            return "its action bounds do not match its action size";
    }
    return {};
}

}  // namespace

EnvironmentBatch::~EnvironmentBatch() {
    if (!batch_) return;
    auto* batch = batch_;
    const auto result = security::SafeExecute(plugin_id_, "ReleaseEnvironments", [batch]() { batch->Release(); });
    if (!result.success) spdlog::warn("Plugin {} failed to release its environments: {}", plugin_id_, result.error_message);
}

bool EnvironmentBatch::Reset(uint64_t seed, float* observations, std::string& error) {
    bool ok = false;
    auto* batch = batch_;
    const auto result = security::SafeExecute(plugin_id_, "ResetEnvironments",
                                              [&ok, batch, seed, observations]() { ok = batch->Reset(seed, observations); });
    if (!result.success) {
        error = "The plugin " + plugin_id_ + " failed to reset " + info_.name + ": " + result.error_message;
        return false;
    }
    if (!ok) error = std::string(batch_->LastError().c_str());
    return ok;
}

bool EnvironmentBatch::Step(const float* actions, const PluginEnvironmentStepBuffers& out, std::string& error) {
    bool ok = false;
    auto* batch = batch_;
    const auto result = security::SafeExecute(plugin_id_, "StepEnvironments",
                                              [&ok, batch, actions, &out]() { ok = batch->Step(actions, out); });
    if (!result.success) {
        error = "The plugin " + plugin_id_ + " failed to step " + info_.name + ": " + result.error_message;
        return false;
    }
    if (!ok) error = std::string(batch_->LastError().c_str());
    return ok;
}

bool EnvironmentBatch::ShowCopy(int copy, std::string& error) {
    bool shown = false;
    auto* batch = batch_;
    const auto result =
        security::SafeExecute(plugin_id_, "ShowEnvironmentCopy", [&shown, batch, copy]() { shown = batch->ShowCopy(copy); });
    if (!result.success) {
        error = "The plugin " + plugin_id_ + " failed to show " + info_.name + ": " + result.error_message;
        return false;
    }
    return shown;
}

std::vector<PluginEnvironmentStat> EnvironmentBatch::Stats() const {
    std::vector<PluginEnvironmentStat> stats;
    auto* batch = batch_;
    const auto result = security::SafeExecute(plugin_id_, "EnvironmentStats", [&stats, batch]() {
        for (const auto& s : batch->Stats())
            stats.push_back({std::string(s.label.c_str()), std::string(s.value.c_str()), std::string(s.note.c_str())});
    });
    if (!result.success) return {};
    return stats;
}

void EnvironmentBatch::LabelShownCopy(const PluginShownCopyLabel& label) {
    auto* batch = batch_;
    security::SafeExecute(plugin_id_, "LabelEnvironmentCopy", [&label, batch]() { batch->LabelShownCopy(label); });
}

PluginEnvironmentRegistry::PluginEnvironmentRegistry() {
    auto& builtin = ClassicControlEnvironments();
    for (const auto& info : builtin.GetEnvironments()) Register(kBuiltinEnvironmentProviderId, info, &builtin, nullptr);
}

PluginEnvironmentRegistry& PluginEnvironmentRegistry::Instance() {
    static PluginEnvironmentRegistry instance;
    return instance;
}

void PluginEnvironmentRegistry::Register(const std::string& plugin_id, const PluginEnvironmentInfo& info,
                                         IEnvironmentProvider* provider, std::shared_ptr<void> library_lease) {
    if (!provider) return;
    Registered entry;
    entry.environment.info = CopyInfo(info);
    if (const std::string problem = InfoProblem(entry.environment.info); !problem.empty()) {
        spdlog::warn("PluginEnvironmentRegistry: {} environment '{}' not added: {}", plugin_id,
                     entry.environment.info.id, problem);
        return;
    }
    entry.environment.plugin_id = plugin_id;
    entry.environment.key = MakeExtensionTypeId(plugin_id, entry.environment.info.id);
    entry.provider = provider;
    entry.library_lease = std::move(library_lease);
    std::lock_guard lock(mutex_);
    if (environments_.count(entry.environment.key)) {
        spdlog::warn("PluginEnvironmentRegistry: Duplicate environment {}, skipping", entry.environment.key);
        return;
    }
    const std::string key = entry.environment.key;
    environments_[key] = std::move(entry);
}

void PluginEnvironmentRegistry::RemoveByPlugin(const std::string& plugin_id) {
    std::lock_guard lock(mutex_);
    for (auto it = environments_.begin(); it != environments_.end();) {
        it = it->second.environment.plugin_id == plugin_id ? environments_.erase(it) : std::next(it);
    }
}

std::vector<PluginEnvironmentRegistry::Environment> PluginEnvironmentRegistry::Environments() const {
    std::lock_guard lock(mutex_);
    std::vector<Environment> out;
    out.reserve(environments_.size());
    for (const auto& [key, entry] : environments_) out.push_back(entry.environment);
    return out;
}

bool PluginEnvironmentRegistry::Find(const std::string& key, Environment& out) const {
    std::lock_guard lock(mutex_);
    const auto it = environments_.find(key);
    if (it == environments_.end()) return false;
    out = it->second.environment;
    return true;
}

bool PluginEnvironmentRegistry::Describe(const std::string& key, const std::map<std::string, std::string>& settings,
                                         Environment& out, std::string& error) const {
    IEnvironmentProvider* provider = nullptr;
    std::shared_ptr<void> lease;
    {
        std::lock_guard lock(mutex_);
        const auto it = environments_.find(key);
        if (it == environments_.end()) {
            error = "The environment " + key + " is not available: its plugin is not loaded, or it no longer offers it.";
            return false;
        }
        out = it->second.environment;
        provider = it->second.provider;
        lease = it->second.library_lease;  // the library stays loaded while the plugin is called
    }
    PluginEnvironmentInfo described;
    std::string reason;
    bool ok = false;
    const std::string id = out.info.id;
    const auto result = security::SafeExecute(out.plugin_id, "DescribeEnvironment", [&]() {
        ok = provider->DescribeEnvironment(id, settings, described, reason);
    });
    if (!result.success) {
        error = "The plugin " + out.plugin_id + " failed to describe " + out.info.name + ": " + result.error_message;
        return false;
    }
    if (!ok) {
        error = std::string(reason.c_str());
        return false;
    }
    described = CopyInfo(described);
    if (const std::string problem = InfoProblem(described); !problem.empty() || described.id != id) {
        error = "The plugin " + out.plugin_id + " described " + out.info.name + " wrongly: " +
                (problem.empty() ? std::string("another id") : problem);
        return false;
    }
    out.info = std::move(described);
    return true;
}

size_t PluginEnvironmentRegistry::GetCount() const {
    std::lock_guard lock(mutex_);
    return environments_.size();
}

std::unique_ptr<EnvironmentBatch> PluginEnvironmentRegistry::Create(const std::string& key,
                                                                    const std::map<std::string, std::string>& settings,
                                                                    int count, std::string& error) {
    if (count < 1) {
        error = "Ask for at least one copy of the environment.";
        return nullptr;
    }
    std::unique_ptr<EnvironmentBatch> batch(new EnvironmentBatch());
    // What these settings make of it: the batch is checked against this
    Environment described;
    if (!Describe(key, settings, described, error)) return nullptr;
    IEnvironmentProvider* provider = nullptr;
    {
        std::lock_guard lock(mutex_);
        const auto it = environments_.find(key);
        if (it == environments_.end()) {
            error = "The environment " + key + " is not available: its plugin is not loaded, or it no longer offers it.";
            return nullptr;
        }
        batch->key_ = key;
        batch->plugin_id_ = it->second.environment.plugin_id;
        batch->info_ = described.info;
        batch->library_lease_ = it->second.library_lease;
        provider = it->second.provider;
    }
    batch->count_ = count;

    PluginEnvironmentCreateResult created;
    const std::string id = batch->info_.id;
    const auto result = security::SafeExecute(batch->plugin_id_, "CreateEnvironments",
                                              [&created, provider, &id, &settings, count]() {
                                                  created = provider->CreateEnvironments(id, settings, count);
                                              });
    if (!result.success) {
        error = "The plugin " + batch->plugin_id_ + " failed to create " + batch->info_.name + ": " + result.error_message;
        return nullptr;
    }
    if (!created.batch) {
        error = created.error.empty() ? "The plugin " + batch->plugin_id_ + " could not create " + batch->info_.name + "."
                                      : std::string(created.error.c_str());
        return nullptr;
    }
    batch->batch_ = created.batch;  // released by the handle from here on
    int made[3] = {0, 0, 0};
    auto* made_batch = created.batch;
    const auto sizes = security::SafeExecute(batch->plugin_id_, "DescribeEnvironments", [&made, made_batch]() {
        made[0] = made_batch->Count();
        made[1] = made_batch->ObservationSize();
        made[2] = made_batch->ActionSize();
    });
    if (!sizes.success || made[0] != count || made[1] != batch->info_.observation_size ||
        made[2] != batch->info_.action_size) {
        error = "The plugin " + batch->plugin_id_ + " made " + batch->info_.name +
                " copies that do not match the environment it describes (count, observation or action size).";
        return nullptr;
    }
    return batch;
}

} // namespace cyxwiz::plugin
