#include "training_scheduler_controller.h"

#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace cyxwiz {

namespace {

bool IsFiniteNonNegative(double value) {
    return std::isfinite(value) && value >= 0.0;
}

bool IsFinitePositive(double value) {
    return std::isfinite(value) && value > 0.0;
}

std::unique_ptr<LRScheduler> BuildScheduler(
    const TrainingSchedulerSpec& specification,
    Optimizer& optimizer) {
    return std::visit(
        [&](const auto& spec) -> std::unique_ptr<LRScheduler> {
            using Spec = std::decay_t<decltype(spec)>;
            if constexpr (std::is_same_v<Spec, StepLRSchedulerSpec>) {
                return std::make_unique<StepLR>(
                    &optimizer, spec.step_size, spec.gamma);
            } else if constexpr (
                std::is_same_v<Spec, ExponentialLRSchedulerSpec>) {
                return std::make_unique<ExponentialLR>(
                    &optimizer, spec.gamma);
            } else if constexpr (
                std::is_same_v<Spec, CosineAnnealingLRSchedulerSpec>) {
                return std::make_unique<CosineAnnealingLR>(
                    &optimizer, spec.t_max, spec.eta_min);
            } else if constexpr (
                std::is_same_v<Spec, ReduceLROnPlateauSchedulerSpec>) {
                return std::make_unique<ReduceLROnPlateau>(
                    &optimizer,
                    spec.mode,
                    spec.factor,
                    spec.patience,
                    spec.threshold,
                    spec.min_lr);
            } else if constexpr (
                std::is_same_v<Spec, LinearWarmupLRSchedulerSpec>) {
                return std::make_unique<LinearWarmupLR>(
                    &optimizer,
                    spec.warmup_epochs,
                    spec.base_lr,
                    spec.start_lr);
            } else {
                return std::make_unique<OneCycleLR>(
                    &optimizer,
                    spec.max_lr,
                    spec.total_steps,
                    spec.pct_start,
                    spec.div_factor,
                    spec.final_div_factor);
            }
        },
        specification);
}

TrainingSchedulerAdvanceResult MakeResult(
    const TrainingSchedulerController& controller,
    bool stepped,
    std::string error = {}) {
    TrainingSchedulerAdvanceResult result;
    result.ok = error.empty();
    result.stepped = stepped;
    result.completed_epochs = controller.GetCompletedEpochs();
    result.completed_optimizer_steps =
        controller.GetCompletedOptimizerSteps();
    result.error = std::move(error);
    if (const auto* scheduler = controller.GetScheduler()) {
        result.learning_rate = scheduler->GetLR();
    }
    return result;
}

} // namespace

bool ValidateTrainingSchedulerSpec(
    const TrainingSchedulerSpec& specification,
    std::string& error) {
    error.clear();
    std::visit(
        [&](const auto& spec) {
            using Spec = std::decay_t<decltype(spec)>;
            if constexpr (std::is_same_v<Spec, StepLRSchedulerSpec>) {
                if (spec.step_size <= 0 ||
                    !IsFiniteNonNegative(spec.gamma)) {
                    error = "StepLR requires positive step_size and finite "
                            "non-negative gamma";
                }
            } else if constexpr (
                std::is_same_v<Spec, ExponentialLRSchedulerSpec>) {
                if (!IsFiniteNonNegative(spec.gamma)) {
                    error = "ExponentialLR requires finite non-negative gamma";
                }
            } else if constexpr (
                std::is_same_v<Spec, CosineAnnealingLRSchedulerSpec>) {
                if (spec.t_max <= 0 ||
                    !IsFiniteNonNegative(spec.eta_min)) {
                    error = "CosineAnnealingLR requires positive t_max and "
                            "finite non-negative eta_min";
                }
            } else if constexpr (
                std::is_same_v<Spec, ReduceLROnPlateauSchedulerSpec>) {
                if ((spec.mode != "min" && spec.mode != "max") ||
                    !std::isfinite(spec.factor) || spec.factor < 0.0 ||
                    spec.factor >= 1.0 || spec.patience < 0 ||
                    !IsFiniteNonNegative(spec.threshold) ||
                    !IsFiniteNonNegative(spec.min_lr)) {
                    error = "ReduceLROnPlateau requires mode min/max, factor "
                            "in [0,1), non-negative patience, threshold, and "
                            "min_lr";
                }
            } else if constexpr (
                std::is_same_v<Spec, LinearWarmupLRSchedulerSpec>) {
                if (spec.warmup_epochs <= 0 ||
                    !IsFiniteNonNegative(spec.base_lr) ||
                    !IsFiniteNonNegative(spec.start_lr)) {
                    error = "LinearWarmupLR requires positive warmup_epochs "
                            "and finite non-negative base/start learning rates";
                }
            } else {
                if (!IsFinitePositive(spec.max_lr) || spec.total_steps <= 0 ||
                    !std::isfinite(spec.pct_start) || spec.pct_start < 0.0 ||
                    spec.pct_start >= 1.0 ||
                    !IsFinitePositive(spec.div_factor) ||
                    !IsFinitePositive(spec.final_div_factor)) {
                    error = "OneCycleLR requires positive max_lr/total_steps/"
                            "division factors and pct_start in [0,1)";
                }
            }
        },
        specification);
    return error.empty();
}

TrainingSchedulerCadence GetTrainingSchedulerCadence(
    const TrainingSchedulerSpec& specification) {
    if (std::holds_alternative<OneCycleLRSchedulerSpec>(specification)) {
        return TrainingSchedulerCadence::OptimizerUpdate;
    }
    if (std::holds_alternative<ReduceLROnPlateauSchedulerSpec>(
            specification)) {
        return TrainingSchedulerCadence::ValidatedEpoch;
    }
    return TrainingSchedulerCadence::CompletedEpoch;
}

const char* TrainingSchedulerCadenceName(
    TrainingSchedulerCadence cadence) {
    switch (cadence) {
        case TrainingSchedulerCadence::CompletedEpoch:
            return "completed_epoch";
        case TrainingSchedulerCadence::ValidatedEpoch:
            return "validated_epoch";
        case TrainingSchedulerCadence::OptimizerUpdate:
            return "optimizer_update";
    }
    return "unknown";
}

TrainingSchedulerController::TrainingSchedulerController(
    TrainingSchedulerSpec specification)
    : specification_(std::move(specification)) {}

bool TrainingSchedulerController::Attach(
    Optimizer& optimizer,
    const std::optional<TrainingSchedulerResumeState>& resume_state,
    std::string& error) {
    error.clear();
    if (scheduler_ != nullptr) {
        error = "training scheduler is already attached";
        return false;
    }
    if (!ValidateTrainingSchedulerSpec(specification_, error)) {
        return false;
    }
    if (resume_state.has_value() &&
        !ValidateResumeState(*resume_state, error)) {
        return false;
    }

    const double original_lr = optimizer.GetLearningRate();
    try {
        auto scheduler = BuildScheduler(specification_, optimizer);
        if (resume_state.has_value() &&
            !scheduler->ImportState(resume_state->scheduler_state, error)) {
            optimizer.SetLearningRate(original_lr);
            return false;
        }

        optimizer_ = &optimizer;
        scheduler_ = std::move(scheduler);
        completed_epochs_ = resume_state.has_value()
            ? resume_state->completed_epochs
            : 0;
        completed_optimizer_steps_ = resume_state.has_value()
            ? resume_state->completed_optimizer_steps
            : 0;
        return true;
    } catch (const std::exception& exception) {
        optimizer.SetLearningRate(original_lr);
        error = std::string("training scheduler attachment failed: ") +
                exception.what();
        return false;
    }
}

bool TrainingSchedulerController::Restore(
    const TrainingSchedulerResumeState& resume_state,
    std::string& error) {
    error.clear();
    if (!scheduler_ || !optimizer_) {
        error = "training scheduler is not attached";
        return false;
    }
    if (!ValidateResumeState(resume_state, error)) {
        return false;
    }
    if (!scheduler_->ImportState(resume_state.scheduler_state, error)) {
        return false;
    }
    completed_epochs_ = resume_state.completed_epochs;
    completed_optimizer_steps_ = resume_state.completed_optimizer_steps;
    return true;
}

TrainingSchedulerAdvanceResult
TrainingSchedulerController::OnOptimizerStep() {
    if (!scheduler_) {
        return MakeResult(*this, false, "training scheduler is not attached");
    }

    ++completed_optimizer_steps_;
    if (GetCadence() != TrainingSchedulerCadence::OptimizerUpdate) {
        return MakeResult(*this, false);
    }

    try {
        scheduler_->Step(completed_optimizer_steps_);
        return MakeResult(*this, true);
    } catch (const std::exception& exception) {
        return MakeResult(
            *this,
            false,
            std::string("optimizer-update scheduler advance failed: ") +
                exception.what());
    }
}

TrainingSchedulerAdvanceResult
TrainingSchedulerController::OnEpochCompleted(
    std::optional<float> validation_loss) {
    if (!scheduler_) {
        return MakeResult(*this, false, "training scheduler is not attached");
    }

    ++completed_epochs_;
    const auto cadence = GetCadence();
    if (cadence == TrainingSchedulerCadence::OptimizerUpdate) {
        return MakeResult(*this, false);
    }
    if (cadence == TrainingSchedulerCadence::ValidatedEpoch &&
        !validation_loss.has_value()) {
        return MakeResult(*this, false);
    }
    if (validation_loss.has_value() &&
        !std::isfinite(*validation_loss)) {
        return MakeResult(
            *this,
            false,
            "validated-epoch scheduler metric must be finite");
    }

    try {
        scheduler_->Step(
            completed_epochs_,
            validation_loss.value_or(0.0f));
        return MakeResult(*this, true);
    } catch (const std::exception& exception) {
        return MakeResult(
            *this,
            false,
            std::string("epoch scheduler advance failed: ") +
                exception.what());
    }
}

bool TrainingSchedulerController::ExportResumeState(
    TrainingSchedulerResumeState& state,
    std::string& error) const {
    error.clear();
    if (!scheduler_) {
        error = "training scheduler is not attached";
        return false;
    }

    TrainingSchedulerResumeState exported;
    if (!scheduler_->ExportState(exported.scheduler_state, error)) {
        return false;
    }
    exported.completed_epochs = completed_epochs_;
    exported.completed_optimizer_steps = completed_optimizer_steps_;
    if (!ValidateResumeState(exported, error)) {
        return false;
    }
    state = std::move(exported);
    return true;
}

TrainingSchedulerCadence TrainingSchedulerController::GetCadence() const {
    return GetTrainingSchedulerCadence(specification_);
}

bool TrainingSchedulerController::ValidateResumeState(
    const TrainingSchedulerResumeState& resume_state,
    std::string& error) const {
    if (resume_state.completed_epochs < 0 ||
        resume_state.completed_optimizer_steps < 0 ||
        resume_state.scheduler_state.last_step < 0) {
        error = "training scheduler resume cursors cannot be negative";
        return false;
    }

    const int scheduler_cursor = resume_state.scheduler_state.last_step;
    switch (GetCadence()) {
        case TrainingSchedulerCadence::CompletedEpoch:
            if (scheduler_cursor != resume_state.completed_epochs) {
                error = "epoch scheduler state does not match the completed "
                        "epoch cursor";
                return false;
            }
            break;
        case TrainingSchedulerCadence::ValidatedEpoch:
            if (scheduler_cursor > resume_state.completed_epochs) {
                error = "plateau scheduler state exceeds the completed epoch "
                        "cursor";
                return false;
            }
            break;
        case TrainingSchedulerCadence::OptimizerUpdate:
            if (scheduler_cursor !=
                resume_state.completed_optimizer_steps) {
                error = "update scheduler state does not match the completed "
                        "optimizer-step cursor";
                return false;
            }
            break;
    }
    error.clear();
    return true;
}

} // namespace cyxwiz
