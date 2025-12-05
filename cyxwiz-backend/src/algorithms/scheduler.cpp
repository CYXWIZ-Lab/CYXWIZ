#include <cyxwiz/tensor.h>  // Must be before scheduler.h (optimizer.h uses Tensor in std::map)
#include <cyxwiz/scheduler.h>
#include <spdlog/spdlog.h>
#include <cmath>
#include <algorithm>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cyxwiz {

// ============================================================================
// StepLR Implementation
// ============================================================================

StepLR::StepLR(Optimizer* optimizer, int step_size, double gamma)
    : optimizer_(optimizer)
    , step_size_(step_size)
    , gamma_(gamma)
{
    if (optimizer_) {
        base_lr_ = optimizer_->GetLearningRate();
        current_lr_ = base_lr_;
    }
    spdlog::debug("StepLR: Created with step_size={}, gamma={}", step_size, gamma);
}

void StepLR::Step(int epoch, float /*metric*/) {
    if (!optimizer_) return;

    last_epoch_ = epoch;
    int num_decays = epoch / step_size_;
    current_lr_ = base_lr_ * std::pow(gamma_, num_decays);
    optimizer_->SetLearningRate(current_lr_);

    spdlog::debug("StepLR: Epoch {} - LR = {:.6f}", epoch, current_lr_);
}

void StepLR::Reset() {
    last_epoch_ = 0;
    current_lr_ = base_lr_;
    if (optimizer_) {
        optimizer_->SetLearningRate(current_lr_);
    }
}

// ============================================================================
// ExponentialLR Implementation
// ============================================================================

ExponentialLR::ExponentialLR(Optimizer* optimizer, double gamma)
    : optimizer_(optimizer)
    , gamma_(gamma)
{
    if (optimizer_) {
        base_lr_ = optimizer_->GetLearningRate();
        current_lr_ = base_lr_;
    }
    spdlog::debug("ExponentialLR: Created with gamma={}", gamma);
}

void ExponentialLR::Step(int epoch, float /*metric*/) {
    if (!optimizer_) return;

    last_epoch_ = epoch;
    current_lr_ = base_lr_ * std::pow(gamma_, epoch);
    optimizer_->SetLearningRate(current_lr_);

    spdlog::debug("ExponentialLR: Epoch {} - LR = {:.6f}", epoch, current_lr_);
}

void ExponentialLR::Reset() {
    last_epoch_ = 0;
    current_lr_ = base_lr_;
    if (optimizer_) {
        optimizer_->SetLearningRate(current_lr_);
    }
}

// ============================================================================
// CosineAnnealingLR Implementation
// ============================================================================

CosineAnnealingLR::CosineAnnealingLR(Optimizer* optimizer, int T_max, double eta_min)
    : optimizer_(optimizer)
    , T_max_(T_max)
    , eta_min_(eta_min)
{
    if (optimizer_) {
        base_lr_ = optimizer_->GetLearningRate();
        current_lr_ = base_lr_;
    }
    spdlog::debug("CosineAnnealingLR: Created with T_max={}, eta_min={}", T_max, eta_min);
}

void CosineAnnealingLR::Step(int epoch, float /*metric*/) {
    if (!optimizer_) return;

    last_epoch_ = epoch;

    // Cosine annealing formula
    // lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * T_cur / T_max)) / 2
    double progress = static_cast<double>(epoch % T_max_) / T_max_;
    current_lr_ = eta_min_ + (base_lr_ - eta_min_) * (1.0 + std::cos(M_PI * progress)) / 2.0;

    optimizer_->SetLearningRate(current_lr_);
    spdlog::debug("CosineAnnealingLR: Epoch {} - LR = {:.6f}", epoch, current_lr_);
}

void CosineAnnealingLR::Reset() {
    last_epoch_ = 0;
    current_lr_ = base_lr_;
    if (optimizer_) {
        optimizer_->SetLearningRate(current_lr_);
    }
}

// ============================================================================
// ReduceLROnPlateau Implementation
// ============================================================================

ReduceLROnPlateau::ReduceLROnPlateau(
    Optimizer* optimizer,
    const std::string& mode,
    double factor,
    int patience,
    double threshold,
    double min_lr)
    : optimizer_(optimizer)
    , mode_(mode)
    , factor_(factor)
    , patience_(patience)
    , threshold_(threshold)
    , min_lr_(min_lr)
{
    if (optimizer_) {
        base_lr_ = optimizer_->GetLearningRate();
        current_lr_ = base_lr_;
    }

    // Initialize best metric based on mode
    if (mode_ == "min") {
        best_metric_ = std::numeric_limits<double>::infinity();
    } else {
        best_metric_ = -std::numeric_limits<double>::infinity();
    }

    spdlog::debug("ReduceLROnPlateau: Created with mode={}, factor={}, patience={}",
                  mode, factor, patience);
}

bool ReduceLROnPlateau::IsBetter(float current, double best) const {
    if (mode_ == "min") {
        return current < (best - threshold_);
    } else {
        return current > (best + threshold_);
    }
}

void ReduceLROnPlateau::Step(int epoch, float metric) {
    if (!optimizer_) return;

    last_epoch_ = epoch;

    if (IsBetter(metric, best_metric_)) {
        // Improvement detected
        best_metric_ = metric;
        num_bad_epochs_ = 0;
        spdlog::debug("ReduceLROnPlateau: Improvement detected at epoch {} (metric={:.4f})",
                      epoch, metric);
    } else {
        // No improvement
        num_bad_epochs_++;
        spdlog::debug("ReduceLROnPlateau: No improvement for {} epochs (metric={:.4f}, best={:.4f})",
                      num_bad_epochs_, metric, best_metric_);

        if (num_bad_epochs_ >= patience_) {
            // Reduce learning rate
            double old_lr = current_lr_;
            current_lr_ = std::max(current_lr_ * factor_, min_lr_);

            if (current_lr_ < old_lr) {
                optimizer_->SetLearningRate(current_lr_);
                spdlog::info("ReduceLROnPlateau: Reducing LR from {:.6f} to {:.6f}",
                             old_lr, current_lr_);
            }

            num_bad_epochs_ = 0;
        }
    }
}

void ReduceLROnPlateau::Reset() {
    last_epoch_ = 0;
    num_bad_epochs_ = 0;
    current_lr_ = base_lr_;

    if (mode_ == "min") {
        best_metric_ = std::numeric_limits<double>::infinity();
    } else {
        best_metric_ = -std::numeric_limits<double>::infinity();
    }

    if (optimizer_) {
        optimizer_->SetLearningRate(current_lr_);
    }
}

// ============================================================================
// LinearWarmupLR Implementation
// ============================================================================

LinearWarmupLR::LinearWarmupLR(
    Optimizer* optimizer,
    int warmup_epochs,
    double base_lr,
    double start_lr)
    : optimizer_(optimizer)
    , warmup_epochs_(warmup_epochs)
    , start_lr_(start_lr)
{
    base_lr_ = base_lr;
    current_lr_ = start_lr;

    if (optimizer_) {
        optimizer_->SetLearningRate(current_lr_);
    }

    spdlog::debug("LinearWarmupLR: Created with warmup_epochs={}, base_lr={}, start_lr={}",
                  warmup_epochs, base_lr, start_lr);
}

void LinearWarmupLR::Step(int epoch, float /*metric*/) {
    if (!optimizer_) return;

    last_epoch_ = epoch;

    if (epoch < warmup_epochs_) {
        // Linear warmup: interpolate from start_lr to base_lr
        double progress = static_cast<double>(epoch + 1) / warmup_epochs_;
        current_lr_ = start_lr_ + (base_lr_ - start_lr_) * progress;
    } else {
        // After warmup, maintain base_lr
        current_lr_ = base_lr_;
    }

    optimizer_->SetLearningRate(current_lr_);
    spdlog::debug("LinearWarmupLR: Epoch {} - LR = {:.6f}", epoch, current_lr_);
}

void LinearWarmupLR::Reset() {
    last_epoch_ = 0;
    current_lr_ = start_lr_;
    if (optimizer_) {
        optimizer_->SetLearningRate(current_lr_);
    }
}

// ============================================================================
// OneCycleLR Implementation
// ============================================================================

OneCycleLR::OneCycleLR(
    Optimizer* optimizer,
    double max_lr,
    int total_steps,
    double pct_start,
    double div_factor,
    double final_div_factor)
    : optimizer_(optimizer)
    , max_lr_(max_lr)
    , total_steps_(total_steps)
    , pct_start_(pct_start)
    , div_factor_(div_factor)
    , final_div_factor_(final_div_factor)
{
    initial_lr_ = max_lr / div_factor;
    final_lr_ = max_lr / final_div_factor;
    base_lr_ = max_lr;
    current_lr_ = initial_lr_;

    if (optimizer_) {
        optimizer_->SetLearningRate(current_lr_);
    }

    spdlog::debug("OneCycleLR: Created with max_lr={}, total_steps={}, pct_start={}",
                  max_lr, total_steps, pct_start);
}

void OneCycleLR::Step(int epoch, float /*metric*/) {
    if (!optimizer_) return;

    current_step_ = epoch;

    int warmup_steps = static_cast<int>(total_steps_ * pct_start_);
    int decay_steps = total_steps_ - warmup_steps;

    if (current_step_ < warmup_steps) {
        // Linear warmup from initial_lr to max_lr
        double progress = static_cast<double>(current_step_) / warmup_steps;
        current_lr_ = initial_lr_ + (max_lr_ - initial_lr_) * progress;
    } else {
        // Cosine decay from max_lr to final_lr
        int decay_step = current_step_ - warmup_steps;
        double progress = static_cast<double>(decay_step) / decay_steps;
        progress = std::min(progress, 1.0);

        // Cosine annealing
        current_lr_ = final_lr_ + (max_lr_ - final_lr_) * (1.0 + std::cos(M_PI * progress)) / 2.0;
    }

    optimizer_->SetLearningRate(current_lr_);
    spdlog::debug("OneCycleLR: Step {} - LR = {:.6f}", current_step_, current_lr_);
}

void OneCycleLR::Reset() {
    current_step_ = 0;
    current_lr_ = initial_lr_;
    if (optimizer_) {
        optimizer_->SetLearningRate(current_lr_);
    }
}

// ============================================================================
// Factory Function
// ============================================================================

std::unique_ptr<LRScheduler> CreateScheduler(
    SchedulerType type,
    Optimizer* optimizer,
    double param1,
    int param2,
    double param3)
{
    switch (type) {
        case SchedulerType::StepLR:
            return std::make_unique<StepLR>(optimizer, param2, param1);

        case SchedulerType::ExponentialLR:
            return std::make_unique<ExponentialLR>(optimizer, param1);

        case SchedulerType::CosineAnnealing:
            return std::make_unique<CosineAnnealingLR>(optimizer, param2, param3);

        case SchedulerType::ReduceLROnPlateau:
            return std::make_unique<ReduceLROnPlateau>(optimizer, "min", param1, param2, 1e-4, param3);

        case SchedulerType::LinearWarmup: {
            double base_lr = optimizer ? optimizer->GetLearningRate() : 0.001;
            return std::make_unique<LinearWarmupLR>(optimizer, param2, base_lr, param3);
        }

        case SchedulerType::OneCycleLR:
            return std::make_unique<OneCycleLR>(optimizer, param1, param2);

        default:
            spdlog::error("CreateScheduler: Unknown scheduler type {}", static_cast<int>(type));
            return nullptr;
    }
}

} // namespace cyxwiz
