#pragma once

// Plugin environments for reinforcement learning (IEnvironmentProvider,
// plugin API 1.4), and the Engine's own (builtin/classic_control_environments,
// provider id "cyxwiz.builtin", registered when the registry is first used).
// Keyed by "<plugin id>:<environment id>". A batch the
// Engine creates keeps its plugin's library loaded until the batch goes, so a
// training run survives the plugin being disabled under it.

#include "../interfaces/i_environment_provider.h"

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace cyxwiz::plugin {

// The Engine's handle on copies of one plugin environment. Calls into the
// plugin are crash-isolated: a crash comes back as an error.
class EnvironmentBatch {
public:
    ~EnvironmentBatch();
    EnvironmentBatch(const EnvironmentBatch&) = delete;
    EnvironmentBatch& operator=(const EnvironmentBatch&) = delete;

    const std::string& Key() const { return key_; }
    const std::string& PluginId() const { return plugin_id_; }
    const PluginEnvironmentInfo& Info() const { return info_; }
    int Count() const { return count_; }
    int ObservationSize() const { return info_.observation_size; }
    int ActionSize() const { return info_.action_size; }

    bool Reset(uint64_t seed, float* observations, std::string& error);
    bool Step(const float* actions, const PluginEnvironmentStepBuffers& out, std::string& error);

private:
    friend class PluginEnvironmentRegistry;
    EnvironmentBatch() = default;

    std::string key_;
    std::string plugin_id_;
    PluginEnvironmentInfo info_;
    int count_ = 0;
    IEnvironmentBatch* batch_ = nullptr;
    std::shared_ptr<void> library_lease_;  // declared last: released after the batch
};

class PluginEnvironmentRegistry {
public:
    static PluginEnvironmentRegistry& Instance();

    PluginEnvironmentRegistry(const PluginEnvironmentRegistry&) = delete;
    PluginEnvironmentRegistry& operator=(const PluginEnvironmentRegistry&) = delete;

    struct Environment {
        std::string key;  // "<plugin id>:<environment id>"
        std::string plugin_id;
        PluginEnvironmentInfo info;
    };

    // The info is copied into the Engine's own strings. library_lease keeps
    // the plugin's library loaded while a batch lives (may be null in tests).
    void Register(const std::string& plugin_id, const PluginEnvironmentInfo& info, IEnvironmentProvider* provider,
                  std::shared_ptr<void> library_lease);
    void RemoveByPlugin(const std::string& plugin_id);

    std::vector<Environment> Environments() const;
    bool Find(const std::string& key, Environment& out) const;
    size_t GetCount() const;

    // count copies of the environment. Null with the reason when the key is
    // unknown, the plugin refuses or crashes, or the batch it returns does
    // not match what it said the environment is.
    std::unique_ptr<EnvironmentBatch> Create(const std::string& key, const std::map<std::string, std::string>& settings,
                                             int count, std::string& error);

private:
    PluginEnvironmentRegistry();

    struct Registered {
        Environment environment;
        IEnvironmentProvider* provider = nullptr;  // non-owning
        std::shared_ptr<void> library_lease;
    };

    mutable std::mutex mutex_;
    std::map<std::string, Registered> environments_;
};

} // namespace cyxwiz::plugin
