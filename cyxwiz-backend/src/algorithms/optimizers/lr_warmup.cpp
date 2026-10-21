#include "cyxwiz/optimizers/lr_warmup.h"
#include "cyxwiz/tensor.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cyxwiz {

namespace {

bool IsValidWarmupType(WarmupType warmup_type) {
    return warmup_type == WarmupType::None ||
           warmup_type == WarmupType::Linear ||
           warmup_type == WarmupType::Cosine;
}

double ComputeWarmupLearningRate(
    int current_step,
    int warmup_steps,
    WarmupType warmup_type,
    double base_lr)
{
    if (warmup_type == WarmupType::None || warmup_steps <= 0 ||
        current_step > warmup_steps) {
        return base_lr;
    }
    if (current_step == 0) return 0.0;

    const double progress =
        static_cast<double>(current_step) / warmup_steps;
    if (warmup_type == WarmupType::Linear) return base_lr * progress;
    return base_lr * 0.5 * (1.0 - std::cos(M_PI * progress));
}

} // namespace

// ============================================================================
// Learning Rate Warmup
// ============================================================================

LRWarmup::LRWarmup(std::unique_ptr<Optimizer> optimizer, int warmup_steps,
                   WarmupType warmup_type, double base_lr)
    : optimizer_(std::move(optimizer)), warmup_steps_(warmup_steps),
      warmup_type_(warmup_type), current_step_(0) {
    if (!optimizer_) {
        throw std::invalid_argument("LRWarmup requires an optimizer");
    }
    if (warmup_steps_ < 0) {
        throw std::invalid_argument("LRWarmup warmup_steps cannot be negative");
    }
    if (!IsValidWarmupType(warmup_type_)) {
        throw std::invalid_argument("LRWarmup type is invalid");
    }

    // If base_lr not specified, use optimizer's initial learning rate
    if (base_lr < 0) {
        base_lr_ = optimizer_->GetLearningRate();
    } else {
        base_lr_ = base_lr;
    }
    if (!std::isfinite(base_lr_) || base_lr_ < 0.0) {
        throw std::invalid_argument(
            "LRWarmup requires a finite non-negative base learning rate");
    }

    const bool warmup_active =
        warmup_type_ != WarmupType::None && warmup_steps_ > 0;
    optimizer_->SetLearningRate(warmup_active ? 0.0 : base_lr_);
}

void LRWarmup::Step(std::map<std::string, Tensor>& parameters,
                    const std::map<std::string, Tensor>& gradients) {
    // Match optimizer.step() followed by scheduler.step(): the current rate
    // belongs to this update, and the newly computed rate belongs to the next.
    optimizer_->Step(parameters, gradients);
    current_step_++;

    optimizer_->SetLearningRate(ComputeWarmupLearningRate(
        current_step_, warmup_steps_, warmup_type_, base_lr_));
}

void LRWarmup::ZeroGrad() {
    optimizer_->ZeroGrad();
}

double LRWarmup::GetCurrentLR() const {
    return optimizer_->GetLearningRate();
}

double LRWarmup::GetWarmupProgress() const {
    if (warmup_type_ == WarmupType::None || warmup_steps_ <= 0) return 1.0;
    double progress = static_cast<double>(current_step_) / warmup_steps_;
    return progress > 1.0 ? 1.0 : progress;
}

bool LRWarmup::IsWarmupComplete() const {
    return warmup_type_ == WarmupType::None ||
           current_step_ >= warmup_steps_;
}

bool LRWarmup::ExportState(
    LRWarmupState& state,
    std::string& error) const
{
    OptimizerState optimizer_state;
    if (!optimizer_->ExportState(optimizer_state, error)) {
        error = "LRWarmup wrapped optimizer state export failed: " + error;
        return false;
    }
    const double expected_lr = ComputeWarmupLearningRate(
        current_step_, warmup_steps_, warmup_type_, base_lr_);
    if (!std::isfinite(optimizer_state.learning_rate) ||
        optimizer_state.learning_rate != expected_lr || current_step_ < 0) {
        error = "LRWarmup contains inconsistent runtime state.";
        return false;
    }

    LRWarmupState exported;
    exported.warmup_steps = warmup_steps_;
    exported.warmup_type = warmup_type_;
    exported.base_learning_rate = base_lr_;
    exported.current_step = current_step_;
    exported.optimizer_state = std::move(optimizer_state);
    state = std::move(exported);
    error.clear();
    return true;
}

bool LRWarmup::ImportState(
    const LRWarmupState& state,
    std::string& error)
{
    if (state.schema_version != 1) {
        error = "LRWarmup state schema version is unsupported.";
        return false;
    }
    if (state.warmup_steps != warmup_steps_ ||
        state.warmup_type != warmup_type_ ||
        !std::isfinite(state.base_learning_rate) ||
        state.base_learning_rate != base_lr_) {
        error =
            "LRWarmup state configuration does not match the active wrapper.";
        return false;
    }
    if (state.current_step < 0) {
        error = "LRWarmup state has a negative step.";
        return false;
    }
    const double expected_lr = ComputeWarmupLearningRate(
        state.current_step, warmup_steps_, warmup_type_, base_lr_);
    if (!std::isfinite(state.optimizer_state.learning_rate) ||
        state.optimizer_state.learning_rate != expected_lr) {
        error = "LRWarmup state learning rate does not match its step.";
        return false;
    }
    if (!optimizer_->ImportState(state.optimizer_state, error)) {
        error = "LRWarmup wrapped optimizer state import failed: " + error;
        return false;
    }

    current_step_ = state.current_step;
    error.clear();
    return true;
}

} // namespace cyxwiz
