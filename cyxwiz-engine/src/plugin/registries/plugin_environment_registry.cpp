#include "plugin_environment_registry.h"

#include "../../core/extension_node_contract.h"
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
    for (const auto& setting : in.settings) {
        out.settings.push_back({std::string(setting.key.c_str()), std::string(setting.label.c_str()),
                                std::string(setting.default_value.c_str()), std::string(setting.description.c_str())});
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
        batch->info_ = it->second.environment.info;
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
