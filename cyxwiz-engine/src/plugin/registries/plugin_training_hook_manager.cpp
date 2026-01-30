#include "plugin_training_hook_manager.h"
#include <spdlog/spdlog.h>
#include <algorithm>

namespace cyxwiz::plugin {

PluginTrainingHookManager& PluginTrainingHookManager::Instance() {
    static PluginTrainingHookManager instance;
    return instance;
}

void PluginTrainingHookManager::RegisterHook(const std::string& plugin_id, ITrainingHook* hook) {
    if (!hook) return;
    std::lock_guard lock(mutex_);
    hooks_.push_back(HookEntry{plugin_id, hook});
    spdlog::info("PluginTrainingHookManager: Registered hook from plugin {}", plugin_id);
}

void PluginTrainingHookManager::RemoveByPlugin(const std::string& plugin_id) {
    std::lock_guard lock(mutex_);
    hooks_.erase(
        std::remove_if(hooks_.begin(), hooks_.end(),
            [&](const HookEntry& e) { return e.plugin_id == plugin_id; }),
        hooks_.end()
    );
}

// Helper macro to reduce boilerplate for invoke methods
#define INVOKE_HOOKS(method_name) \
    std::vector<ITrainingHook*> snapshot; \
    { \
        std::lock_guard lock(mutex_); \
        snapshot.reserve(hooks_.size()); \
        for (const auto& e : hooks_) \
            if (e.hook) snapshot.push_back(e.hook); \
    } \
    for (auto* hook : snapshot) { \
        try { \
            hook->method_name(ctx); \
        } catch (const std::exception& e) { \
            spdlog::error("PluginTrainingHookManager: " #method_name " threw: {}", e.what()); \
        } \
    }

void PluginTrainingHookManager::NotifyTrainingStart(TrainingContext& ctx) { INVOKE_HOOKS(OnTrainingStart) }
void PluginTrainingHookManager::NotifyTrainingEnd(TrainingContext& ctx) { INVOKE_HOOKS(OnTrainingEnd) }
void PluginTrainingHookManager::NotifyEpochStart(TrainingContext& ctx) { INVOKE_HOOKS(OnEpochStart) }
void PluginTrainingHookManager::NotifyEpochEnd(TrainingContext& ctx) { INVOKE_HOOKS(OnEpochEnd) }
void PluginTrainingHookManager::NotifyBatchStart(TrainingContext& ctx) { INVOKE_HOOKS(OnBatchStart) }
void PluginTrainingHookManager::NotifyBatchEnd(TrainingContext& ctx) { INVOKE_HOOKS(OnBatchEnd) }

#undef INVOKE_HOOKS

bool PluginTrainingHookManager::ShouldStopEarly(const TrainingContext& ctx) {
    std::vector<ITrainingHook*> snapshot;
    {
        std::lock_guard lock(mutex_);
        snapshot.reserve(hooks_.size());
        for (const auto& e : hooks_)
            if (e.hook) snapshot.push_back(e.hook);
    }
    for (auto* hook : snapshot) {
        try {
            if (hook->ShouldStopEarly(ctx)) return true;
        } catch (const std::exception& e) {
            spdlog::error("PluginTrainingHookManager: ShouldStopEarly threw: {}", e.what());
        }
    }
    return false;
}

size_t PluginTrainingHookManager::GetHookCount() const {
    std::lock_guard lock(mutex_);
    return hooks_.size();
}

} // namespace cyxwiz::plugin
