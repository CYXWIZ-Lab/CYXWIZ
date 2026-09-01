#pragma once

#include <cyxwiz/scheduler.h>

#include <memory>
#include <optional>
#include <string>
#include <variant>

namespace cyxwiz {

struct StepLRSchedulerSpec {
    int step_size = 10;
    double gamma = 0.1;
};

struct ExponentialLRSchedulerSpec {
    double gamma = 0.95;
};

struct CosineAnnealingLRSchedulerSpec {
    int t_max = 100;
    double eta_min = 0.0;
};

struct ReduceLROnPlateauSchedulerSpec {
    std::string mode = "min";
    double factor = 0.1;
    int patience = 10;
    double threshold = 1.0e-4;
    double min_lr = 1.0e-8;
};

struct LinearWarmupLRSchedulerSpec {
    int warmup_epochs = 1;
    double base_lr = 0.001;
    double start_lr = 0.0;
};

struct OneCycleLRSchedulerSpec {
    double max_lr = 0.1;
    int total_steps = 1;
    double pct_start = 0.3;
    double div_factor = 25.0;
    double final_div_factor = 10000.0;
};

using TrainingSchedulerSpec = std::variant<
    StepLRSchedulerSpec,
    ExponentialLRSchedulerSpec,
    CosineAnnealingLRSchedulerSpec,
    ReduceLROnPlateauSchedulerSpec,
    LinearWarmupLRSchedulerSpec,
    OneCycleLRSchedulerSpec>;

enum class TrainingSchedulerCadence {
    CompletedEpoch,
    ValidatedEpoch,
    OptimizerUpdate,
};

struct TrainingSchedulerResumeState {
    SchedulerState scheduler_state;
    int completed_epochs = 0;
    int completed_optimizer_steps = 0;
};

struct TrainingSchedulerAdvanceResult {
    bool ok = true;
    bool stepped = false;
    double learning_rate = 0.0;
    int completed_epochs = 0;
    int completed_optimizer_steps = 0;
    std::string error;
};

bool ValidateTrainingSchedulerSpec(
    const TrainingSchedulerSpec& specification,
    std::string& error);

TrainingSchedulerCadence GetTrainingSchedulerCadence(
    const TrainingSchedulerSpec& specification);

const char* TrainingSchedulerCadenceName(TrainingSchedulerCadence cadence);

/**
 * Owns the host-side lifecycle cursor for one backend LRScheduler.
 *
 * The controller never performs parameter computation. TrainingExecutor calls
 * it only after a real optimizer update and after a fully completed epoch.
 * Absolute epoch/update cursors are kept beside backend SchedulerState because
 * plateau schedulers may intentionally skip non-validation epochs.
 */
class TrainingSchedulerController {
public:
    explicit TrainingSchedulerController(TrainingSchedulerSpec specification);

    bool Attach(
        Optimizer& optimizer,
        const std::optional<TrainingSchedulerResumeState>& resume_state,
        std::string& error);

    bool Restore(
        const TrainingSchedulerResumeState& resume_state,
        std::string& error);

    TrainingSchedulerAdvanceResult OnOptimizerStep();

    TrainingSchedulerAdvanceResult OnEpochCompleted(
        std::optional<float> validation_loss);

    bool ExportResumeState(
        TrainingSchedulerResumeState& state,
        std::string& error) const;

    TrainingSchedulerCadence GetCadence() const;
    int GetCompletedEpochs() const { return completed_epochs_; }
    int GetCompletedOptimizerSteps() const {
        return completed_optimizer_steps_;
    }
    LRScheduler* GetScheduler() { return scheduler_.get(); }
    const LRScheduler* GetScheduler() const { return scheduler_.get(); }

private:
    bool ValidateResumeState(
        const TrainingSchedulerResumeState& resume_state,
        std::string& error) const;

    TrainingSchedulerSpec specification_;
    std::unique_ptr<LRScheduler> scheduler_;
    Optimizer* optimizer_ = nullptr;
    int completed_epochs_ = 0;
    int completed_optimizer_steps_ = 0;
};

} // namespace cyxwiz
