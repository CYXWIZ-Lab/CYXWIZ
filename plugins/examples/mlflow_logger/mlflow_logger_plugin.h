#pragma once

// MLflow Logger Plugin - Example CyxWiz Plugin
//
// Demonstrates:
//   - IPlugin lifecycle (OnLoad, OnInitialize, OnShutdown, OnUnload)
//   - ITrainingHook for training metric logging
//   - Early stopping via ShouldStopEarly()
//   - CYXWIZ_PLUGIN_ENTRY macro usage

#include "../../cyxwiz-engine/src/plugin/plugin_types.h"
#include "../../cyxwiz-engine/src/plugin/plugin_context.h"
#include "../../cyxwiz-engine/src/plugin/interfaces/i_training_hook.h"

#include <string>
#include <vector>

namespace cyxwiz::plugin::examples {

class MLflowLoggerPlugin : public IPlugin, public ITrainingHook {
public:
    MLflowLoggerPlugin() = default;
    ~MLflowLoggerPlugin() override = default;

    // ===== IPlugin Lifecycle =====
    bool OnLoad(PluginContext& ctx) override;
    bool OnInitialize(PluginContext& ctx) override;
    void OnShutdown(PluginContext& ctx) override;
    void OnUnload(PluginContext& ctx) override;

    const PluginManifest& GetManifest() const override { return manifest_; }
    PluginPermissionFlags GetRequiredPermissions() const override {
        return static_cast<uint32_t>(PluginPermission::Network) |
               static_cast<uint32_t>(PluginPermission::Training);
    }
    PluginState GetState() const override { return state_; }

    // ===== ITrainingHook =====
    void OnTrainingStart(TrainingContext& ctx) override;
    void OnTrainingEnd(TrainingContext& ctx) override;
    void OnEpochEnd(TrainingContext& ctx) override;
    bool ShouldStopEarly(const TrainingContext& ctx) override;

private:
    PluginManifest manifest_;
    PluginState state_ = PluginState::Unloaded;

    // MLflow configuration
    std::string tracking_uri_ = "http://localhost:5000";
    std::string experiment_name_ = "cyxwiz-training";
    std::string current_run_id_;

    // Early stopping state
    int patience_ = 5;                          // Stop after N epochs without improvement
    int epochs_without_improvement_ = 0;
    float best_val_loss_ = std::numeric_limits<float>::max();
};

} // namespace cyxwiz::plugin::examples
