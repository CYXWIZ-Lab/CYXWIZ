#include "cyxwiz/optimizers/lr_warmup.h"
#include "cyxwiz/tensor.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cyxwiz {

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

    // If base_lr not specified, use optimizer's initial learning rate
    if (base_lr < 0) {
        base_lr_ = optimizer_->GetLearningRate();
    } else {
        base_lr_ = base_lr;
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

    // Compute warmup multiplier
    double warmup_lr = base_lr_;

    if (current_step_ <= warmup_steps_ && warmup_type_ != WarmupType::None) {
        double progress = static_cast<double>(current_step_) / warmup_steps_;

        switch (warmup_type_) {
            case WarmupType::Linear:
                // Linear warmup: lr increases linearly from 0 to base_lr
                warmup_lr = base_lr_ * progress;
                break;

            case WarmupType::Cosine:
                // Cosine warmup: smoother ramp-up using cosine curve
                // lr = base_lr * 0.5 * (1 - cos(pi * progress))
                warmup_lr = base_lr_ * 0.5 * (1.0 - std::cos(M_PI * progress));
                break;

            default:
                break;
        }
    }

    // Set adjusted learning rate
    optimizer_->SetLearningRate(warmup_lr);
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

} // namespace cyxwiz
