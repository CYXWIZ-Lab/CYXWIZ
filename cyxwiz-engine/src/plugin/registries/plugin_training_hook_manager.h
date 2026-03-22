#pragma once

#include "../interfaces/i_training_hook.h"
#include <string>
#include <vector>
#include <map>
#include <mutex>

namespace cyxwiz::plugin {

class PluginTrainingHookManager {
public:
    static PluginTrainingHookManager& Instance();

    PluginTrainingHookManager(const PluginTrainingHookManager&) = delete;
    PluginTrainingHookManager& operator=(const PluginTrainingHookManager&) = delete;

    void RegisterHook(const std::string& plugin_id, ITrainingHook* hook);
    void RemoveByPlugin(const std::string& plugin_id);

    // Invoke all hooks (called by TrainingExecutor in Phase 3)
    void NotifyTrainingStart(TrainingContext& ctx);
    void NotifyTrainingEnd(TrainingContext& ctx);
    void NotifyEpochStart(TrainingContext& ctx);
    void NotifyEpochEnd(TrainingContext& ctx);
    void NotifyBatchStart(TrainingContext& ctx);
    void NotifyBatchEnd(TrainingContext& ctx);

    // Poll all hooks for early stopping
    bool ShouldStopEarly(const TrainingContext& ctx);

    size_t GetHookCount() const;

private:
    PluginTrainingHookManager() = default;

    struct HookEntry {
        std::string plugin_id;
        ITrainingHook* hook = nullptr;  // non-owning
    };

    mutable std::mutex mutex_;
    std::vector<HookEntry> hooks_;
};

} // namespace cyxwiz::plugin
