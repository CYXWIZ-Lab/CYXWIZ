#pragma once

// MLflow Logger Plugin - Example CyxWiz Plugin
//
// Demonstrates:
//   - IPlugin lifecycle (OnLoad, OnInitialize, OnShutdown, OnUnload)
//   - ITrainingHook for training metric logging
//   - Early stopping via ShouldStopEarly()
//   - Logging through the plugin context, also from training hooks
//   - CYXWIZ_PLUGIN_ENTRY macro usage

#include <fmt/format.h>
#include <imgui.h>
#include "plugin/plugin_types.h"
#include "plugin/plugin_context.h"
#include "plugin/interfaces/i_training_hook.h"

#include <string>
#include <utility>
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
    void SetImGuiContext(void* ctx) override { ImGui::SetCurrentContext(static_cast<ImGuiContext*>(ctx)); }
    void* QueryInterface(const char* name) override {
        if (std::string(name) == "ITrainingHook") return static_cast<ITrainingHook*>(this);
        return nullptr;
    }

    // ===== ITrainingHook =====
    void OnTrainingStart(TrainingContext& ctx) override;
    void OnTrainingEnd(TrainingContext& ctx) override;
    void OnEpochEnd(TrainingContext& ctx) override;
    bool ShouldStopEarly(const TrainingContext& ctx) override;

private:
    // Training hooks get a TrainingContext, not the plugin context: keep the
    // one OnLoad gives (valid until OnUnload) and log through it, so the lines
    // reach the Engine's log on every platform.
    template <typename... Args>
    void LogInfo(fmt::format_string<Args...> format, Args&&... args) {
        if (ctx_) ctx_->LogInfo(fmt::format(format, std::forward<Args>(args)...));
    }
    template <typename... Args>
    void LogWarn(fmt::format_string<Args...> format, Args&&... args) {
        if (ctx_) ctx_->LogWarn(fmt::format(format, std::forward<Args>(args)...));
    }

    PluginContext* ctx_ = nullptr;
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
