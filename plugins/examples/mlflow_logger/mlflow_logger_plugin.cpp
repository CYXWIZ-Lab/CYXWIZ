#include "mlflow_logger_plugin.h"

#include <spdlog/spdlog.h>
#include <fstream>
#include <nlohmann/json.hpp>

namespace cyxwiz::plugin::examples {

// =============================================================================
// IPlugin Lifecycle
// =============================================================================

bool MLflowLoggerPlugin::OnLoad(PluginContext& ctx) {
    ctx.LogInfo("MLflow Logger: OnLoad");

    // Read optional config file from plugin directory
    // In a real plugin, this would parse mlflow_config.json for tracking URI etc.
    auto config_path = manifest_.plugin_dir / "mlflow_config.json";
    std::ifstream f(config_path);
    if (f.is_open()) {
        try {
            nlohmann::json config = nlohmann::json::parse(f);
            tracking_uri_ = config.value("tracking_uri", tracking_uri_);
            experiment_name_ = config.value("experiment_name", experiment_name_);
            patience_ = config.value("early_stopping_patience", patience_);
            ctx.LogInfo("MLflow Logger: Loaded config from mlflow_config.json");
        } catch (const std::exception& e) {
            ctx.LogWarn("MLflow Logger: Failed to parse config: " + std::string(e.what()));
        }
    }

    state_ = PluginState::Loaded;
    return true;
}

bool MLflowLoggerPlugin::OnInitialize(PluginContext& ctx) {
    ctx.LogInfo("MLflow Logger: OnInitialize");

    // Register as a training hook — requires Training permission
    if (!ctx.RegisterTrainingHook(this)) {
        ctx.LogError("MLflow Logger: Failed to register training hook");
        state_ = PluginState::Failed;
        return false;
    }

    ctx.LogInfo("MLflow Logger: Registered training hook (tracking_uri=" + tracking_uri_ + ")");
    state_ = PluginState::Active;
    return true;
}

void MLflowLoggerPlugin::OnShutdown(PluginContext& ctx) {
    ctx.LogInfo("MLflow Logger: OnShutdown");
    // In a real plugin: end any active MLflow run
    current_run_id_.clear();
    state_ = PluginState::Loaded;
}

void MLflowLoggerPlugin::OnUnload(PluginContext& ctx) {
    ctx.LogInfo("MLflow Logger: OnUnload");
    state_ = PluginState::Unloaded;
}

// =============================================================================
// ITrainingHook - Training Metric Logging
// =============================================================================

void MLflowLoggerPlugin::OnTrainingStart(TrainingContext& ctx) {
    // Reset early stopping state
    epochs_without_improvement_ = 0;
    best_val_loss_ = std::numeric_limits<float>::max();

    // In a real plugin: create MLflow run via HTTP API
    // POST {tracking_uri}/api/2.0/mlflow/runs/create
    current_run_id_ = "simulated-run-001";

    spdlog::info("[MLflow] Training started — experiment='{}', run_id='{}', epochs={}",
                 experiment_name_, current_run_id_, ctx.total_epochs);
}

void MLflowLoggerPlugin::OnTrainingEnd(TrainingContext& ctx) {
    spdlog::info("[MLflow] Training ended — final train_loss={:.4f}, val_loss={:.4f}",
                 ctx.train_loss, ctx.val_loss);

    // In a real plugin: end MLflow run
    // POST {tracking_uri}/api/2.0/mlflow/runs/update
    current_run_id_.clear();
}

void MLflowLoggerPlugin::OnEpochEnd(TrainingContext& ctx) {
    // Log metrics for this epoch
    // In a real plugin: POST to MLflow tracking API
    spdlog::info("[MLflow] Epoch {}/{}: train_loss={:.4f}, train_acc={:.4f}, "
                 "val_loss={:.4f}, val_acc={:.4f}, lr={:.6f}",
                 ctx.current_epoch, ctx.total_epochs,
                 ctx.train_loss, ctx.train_accuracy,
                 ctx.val_loss, ctx.val_accuracy,
                 ctx.learning_rate);

    // Log custom metrics if any
    for (const auto& [name, value] : ctx.custom_metrics) {
        spdlog::info("[MLflow]   {} = {:.4f}", name, value);
    }

    // Track best validation loss for early stopping
    if (ctx.val_loss < best_val_loss_) {
        best_val_loss_ = ctx.val_loss;
        epochs_without_improvement_ = 0;
    } else {
        epochs_without_improvement_++;
    }
}

bool MLflowLoggerPlugin::ShouldStopEarly(const TrainingContext& ctx) {
    if (epochs_without_improvement_ >= patience_) {
        spdlog::warn("[MLflow] Early stopping: val_loss hasn't improved for {} epochs "
                     "(best={:.4f}, current={:.4f})",
                     patience_, best_val_loss_, ctx.val_loss);
        return true;
    }
    return false;
}

} // namespace cyxwiz::plugin::examples

// =============================================================================
// Plugin Entry Point
// =============================================================================

CYXWIZ_PLUGIN_ENTRY(cyxwiz::plugin::examples::MLflowLoggerPlugin)
