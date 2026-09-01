#include <cyxwiz/tensor.h>  // Must be before scheduler.h (optimizer.h uses Tensor in std::map)
#include <cyxwiz/scheduler.h>
#include <spdlog/spdlog.h>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cyxwiz {

namespace {

void RequireOptimizer(Optimizer* optimizer, const char* scheduler_name) {
    if (!optimizer) {
        throw std::invalid_argument(
            std::string(scheduler_name) + " requires an optimizer");
    }
}

void RequireFiniteNonNegative(
    double value,
    const char* scheduler_name,
    const char* field_name)
{
    if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(
            std::string(scheduler_name) + " requires finite non-negative " +
            field_name);
    }
}

void RequireFinitePositive(
    double value,
    const char* scheduler_name,
    const char* field_name)
{
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::invalid_argument(
            std::string(scheduler_name) + " requires finite positive " +
            field_name);
    }
}

SchedulerState MakeSchedulerState(
    const char* scheduler_type,
    double base_lr,
    double current_lr,
    int last_step,
    std::map<std::string, double> hyperparameters,
    std::map<std::string, std::string> string_hyperparameters = {},
    std::map<std::string, double> values = {})
{
    SchedulerState state;
    state.scheduler_type = scheduler_type;
    state.base_learning_rate = base_lr;
    state.current_learning_rate = current_lr;
    state.last_step = last_step;
    state.hyperparameters = std::move(hyperparameters);
    state.string_hyperparameters = std::move(string_hyperparameters);
    state.values = std::move(values);
    return state;
}

bool ValidateSchedulerState(
    const SchedulerState& state,
    const char* scheduler_type,
    double expected_base_lr,
    const std::map<std::string, double>& expected_hyperparameters,
    const std::map<std::string, std::string>& expected_string_hyperparameters,
    std::string& error)
{
    error.clear();
    if (state.schema_version != 1) {
        error = std::string(scheduler_type) +
                " scheduler state schema version is unsupported.";
        return false;
    }
    if (state.scheduler_type != scheduler_type) {
        error = "Scheduler state type '" + state.scheduler_type +
                "' is incompatible with " + scheduler_type + ".";
        return false;
    }
    if (!std::isfinite(state.base_learning_rate) ||
        state.base_learning_rate < 0.0 ||
        state.base_learning_rate != expected_base_lr) {
        error = std::string(scheduler_type) +
                " scheduler state has an incompatible base learning rate.";
        return false;
    }
    if (!std::isfinite(state.current_learning_rate) ||
        state.current_learning_rate < 0.0) {
        error = std::string(scheduler_type) +
                " scheduler state has an invalid current learning rate.";
        return false;
    }
    if (state.last_step < 0) {
        error = std::string(scheduler_type) +
                " scheduler state has a negative step.";
        return false;
    }
    if (state.hyperparameters != expected_hyperparameters ||
        state.string_hyperparameters != expected_string_hyperparameters) {
        error = std::string(scheduler_type) +
                " scheduler state configuration does not match the active "
                "scheduler.";
        return false;
    }
    for (const auto& [name, value] : state.values) {
        if (!std::isfinite(value)) {
            error = std::string(scheduler_type) +
                    " scheduler state value '" + name + "' is not finite.";
            return false;
        }
    }
    return true;
}

} // namespace

// ============================================================================
// StepLR Implementation
// ============================================================================

StepLR::StepLR(Optimizer* optimizer, int step_size, double gamma)
    : optimizer_(optimizer)
    , step_size_(step_size)
    , gamma_(gamma)
{
    RequireOptimizer(optimizer_, "StepLR");
    if (step_size_ <= 0) {
        throw std::invalid_argument("StepLR requires positive step_size");
    }
    RequireFiniteNonNegative(gamma_, "StepLR", "gamma");
    base_lr_ = optimizer_->GetLearningRate();
    RequireFiniteNonNegative(base_lr_, "StepLR", "optimizer learning rate");
    current_lr_ = base_lr_;
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

bool StepLR::ExportState(SchedulerState& state, std::string& error) const {
    error.clear();
    state = MakeSchedulerState(
        "StepLR", base_lr_, current_lr_, last_epoch_,
        {{"step_size", static_cast<double>(step_size_)}, {"gamma", gamma_}});
    return true;
}

bool StepLR::ImportState(const SchedulerState& state, std::string& error) {
    const std::map<std::string, double> expected{
        {"step_size", static_cast<double>(step_size_)}, {"gamma", gamma_}};
    if (!ValidateSchedulerState(
            state, "StepLR", base_lr_, expected, {}, error) ||
        !state.values.empty()) {
        if (error.empty()) error = "StepLR scheduler state has unknown values.";
        return false;
    }
    last_epoch_ = state.last_step;
    current_lr_ = state.current_learning_rate;
    optimizer_->SetLearningRate(current_lr_);
    return true;
}

// ============================================================================
// ExponentialLR Implementation
// ============================================================================

ExponentialLR::ExponentialLR(Optimizer* optimizer, double gamma)
    : optimizer_(optimizer)
    , gamma_(gamma)
{
    RequireOptimizer(optimizer_, "ExponentialLR");
    RequireFiniteNonNegative(gamma_, "ExponentialLR", "gamma");
    base_lr_ = optimizer_->GetLearningRate();
    RequireFiniteNonNegative(
        base_lr_, "ExponentialLR", "optimizer learning rate");
    current_lr_ = base_lr_;
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

bool ExponentialLR::ExportState(
    SchedulerState& state,
    std::string& error) const
{
    error.clear();
    state = MakeSchedulerState(
        "ExponentialLR", base_lr_, current_lr_, last_epoch_,
        {{"gamma", gamma_}});
    return true;
}

bool ExponentialLR::ImportState(
    const SchedulerState& state,
    std::string& error)
{
    const std::map<std::string, double> expected{{"gamma", gamma_}};
    if (!ValidateSchedulerState(
            state, "ExponentialLR", base_lr_, expected, {}, error) ||
        !state.values.empty()) {
        if (error.empty()) {
            error = "ExponentialLR scheduler state has unknown values.";
        }
        return false;
    }
    last_epoch_ = state.last_step;
    current_lr_ = state.current_learning_rate;
    optimizer_->SetLearningRate(current_lr_);
    return true;
}

// ============================================================================
// CosineAnnealingLR Implementation
// ============================================================================

CosineAnnealingLR::CosineAnnealingLR(Optimizer* optimizer, int T_max, double eta_min)
    : optimizer_(optimizer)
    , T_max_(T_max)
    , eta_min_(eta_min)
{
    RequireOptimizer(optimizer_, "CosineAnnealingLR");
    if (T_max_ <= 0) {
        throw std::invalid_argument(
            "CosineAnnealingLR requires positive T_max");
    }
    RequireFiniteNonNegative(eta_min_, "CosineAnnealingLR", "eta_min");
    base_lr_ = optimizer_->GetLearningRate();
    RequireFiniteNonNegative(
        base_lr_, "CosineAnnealingLR", "optimizer learning rate");
    current_lr_ = base_lr_;
    spdlog::debug("CosineAnnealingLR: Created with T_max={}, eta_min={}", T_max, eta_min);
}

void CosineAnnealingLR::Step(int epoch, float /*metric*/) {
    if (!optimizer_) return;

    last_epoch_ = epoch;

    // Cosine annealing formula
    // lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * T_cur / T_max)) / 2
    const double progress = static_cast<double>(epoch) / T_max_;
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

bool CosineAnnealingLR::ExportState(
    SchedulerState& state,
    std::string& error) const
{
    error.clear();
    state = MakeSchedulerState(
        "CosineAnnealingLR", base_lr_, current_lr_, last_epoch_,
        {{"T_max", static_cast<double>(T_max_)}, {"eta_min", eta_min_}});
    return true;
}

bool CosineAnnealingLR::ImportState(
    const SchedulerState& state,
    std::string& error)
{
    const std::map<std::string, double> expected{
        {"T_max", static_cast<double>(T_max_)}, {"eta_min", eta_min_}};
    if (!ValidateSchedulerState(
            state, "CosineAnnealingLR", base_lr_, expected, {}, error) ||
        !state.values.empty()) {
        if (error.empty()) {
            error = "CosineAnnealingLR scheduler state has unknown values.";
        }
        return false;
    }
    last_epoch_ = state.last_step;
    current_lr_ = state.current_learning_rate;
    optimizer_->SetLearningRate(current_lr_);
    return true;
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
    RequireOptimizer(optimizer_, "ReduceLROnPlateau");
    if (mode_ != "min" && mode_ != "max") {
        throw std::invalid_argument(
            "ReduceLROnPlateau mode must be 'min' or 'max'");
    }
    if (!std::isfinite(factor_) || factor_ < 0.0 || factor_ >= 1.0) {
        throw std::invalid_argument(
            "ReduceLROnPlateau factor must be finite and in [0, 1)");
    }
    if (patience_ < 0) {
        throw std::invalid_argument(
            "ReduceLROnPlateau patience cannot be negative");
    }
    RequireFiniteNonNegative(
        threshold_, "ReduceLROnPlateau", "threshold");
    RequireFiniteNonNegative(min_lr_, "ReduceLROnPlateau", "min_lr");
    base_lr_ = optimizer_->GetLearningRate();
    RequireFiniteNonNegative(
        base_lr_, "ReduceLROnPlateau", "optimizer learning rate");
    current_lr_ = base_lr_;

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

        if (num_bad_epochs_ > patience_) {
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

bool ReduceLROnPlateau::ExportState(
    SchedulerState& state,
    std::string& error) const
{
    error.clear();
    const bool has_best = std::isfinite(best_metric_);
    state = MakeSchedulerState(
        "ReduceLROnPlateau", base_lr_, current_lr_, last_epoch_,
        {{"factor", factor_},
         {"patience", static_cast<double>(patience_)},
         {"threshold", threshold_},
         {"min_lr", min_lr_}},
        {{"mode", mode_}},
        {{"has_best", has_best ? 1.0 : 0.0},
         {"best_metric", has_best ? best_metric_ : 0.0},
         {"num_bad_epochs", static_cast<double>(num_bad_epochs_)}});
    return true;
}

bool ReduceLROnPlateau::ImportState(
    const SchedulerState& state,
    std::string& error)
{
    const std::map<std::string, double> expected{
        {"factor", factor_},
        {"patience", static_cast<double>(patience_)},
        {"threshold", threshold_},
        {"min_lr", min_lr_}};
    if (!ValidateSchedulerState(
            state, "ReduceLROnPlateau", base_lr_, expected,
            {{"mode", mode_}}, error)) {
        return false;
    }
    const auto has_best = state.values.find("has_best");
    const auto best_metric = state.values.find("best_metric");
    const auto bad_epochs = state.values.find("num_bad_epochs");
    if (state.values.size() != 3 || has_best == state.values.end() ||
        best_metric == state.values.end() || bad_epochs == state.values.end() ||
        (has_best->second != 0.0 && has_best->second != 1.0) ||
        bad_epochs->second < 0.0 ||
        std::floor(bad_epochs->second) != bad_epochs->second) {
        error = "ReduceLROnPlateau scheduler state values are invalid.";
        return false;
    }

    const double imported_best = has_best->second == 1.0
        ? best_metric->second
        : (mode_ == "min"
               ? std::numeric_limits<double>::infinity()
               : -std::numeric_limits<double>::infinity());
    last_epoch_ = state.last_step;
    current_lr_ = state.current_learning_rate;
    best_metric_ = imported_best;
    num_bad_epochs_ = static_cast<int>(bad_epochs->second);
    optimizer_->SetLearningRate(current_lr_);
    return true;
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
    RequireOptimizer(optimizer_, "LinearWarmupLR");
    if (warmup_epochs_ <= 0) {
        throw std::invalid_argument(
            "LinearWarmupLR requires positive warmup_epochs");
    }
    RequireFiniteNonNegative(base_lr, "LinearWarmupLR", "base_lr");
    RequireFiniteNonNegative(start_lr_, "LinearWarmupLR", "start_lr");
    base_lr_ = base_lr;
    current_lr_ = start_lr;
    optimizer_->SetLearningRate(current_lr_);

    spdlog::debug("LinearWarmupLR: Created with warmup_epochs={}, base_lr={}, start_lr={}",
                  warmup_epochs, base_lr, start_lr);
}

void LinearWarmupLR::Step(int epoch, float /*metric*/) {
    if (!optimizer_) return;

    last_epoch_ = epoch;

    if (epoch < warmup_epochs_) {
        // Linear warmup: interpolate from start_lr to base_lr
        double progress = static_cast<double>(epoch) / warmup_epochs_;
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

bool LinearWarmupLR::ExportState(
    SchedulerState& state,
    std::string& error) const
{
    error.clear();
    state = MakeSchedulerState(
        "LinearWarmupLR", base_lr_, current_lr_, last_epoch_,
        {{"warmup_epochs", static_cast<double>(warmup_epochs_)},
         {"start_lr", start_lr_}});
    return true;
}

bool LinearWarmupLR::ImportState(
    const SchedulerState& state,
    std::string& error)
{
    const std::map<std::string, double> expected{
        {"warmup_epochs", static_cast<double>(warmup_epochs_)},
        {"start_lr", start_lr_}};
    if (!ValidateSchedulerState(
            state, "LinearWarmupLR", base_lr_, expected, {}, error) ||
        !state.values.empty()) {
        if (error.empty()) {
            error = "LinearWarmupLR scheduler state has unknown values.";
        }
        return false;
    }
    last_epoch_ = state.last_step;
    current_lr_ = state.current_learning_rate;
    optimizer_->SetLearningRate(current_lr_);
    return true;
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
    RequireOptimizer(optimizer_, "OneCycleLR");
    RequireFinitePositive(max_lr_, "OneCycleLR", "max_lr");
    if (total_steps_ <= 0) {
        throw std::invalid_argument("OneCycleLR requires positive total_steps");
    }
    if (!std::isfinite(pct_start_) || pct_start_ < 0.0 || pct_start_ >= 1.0) {
        throw std::invalid_argument(
            "OneCycleLR pct_start must be finite and in [0, 1)");
    }
    RequireFinitePositive(div_factor_, "OneCycleLR", "div_factor");
    RequireFinitePositive(
        final_div_factor_, "OneCycleLR", "final_div_factor");
    initial_lr_ = max_lr / div_factor;
    final_lr_ = initial_lr_ / final_div_factor;
    base_lr_ = max_lr;
    current_lr_ = initial_lr_;

    optimizer_->SetLearningRate(current_lr_);

    spdlog::debug("OneCycleLR: Created with max_lr={}, total_steps={}, pct_start={}",
                  max_lr, total_steps, pct_start);
}

void OneCycleLR::Step(int epoch, float /*metric*/) {
    if (!optimizer_) return;
    if (epoch > total_steps_) {
        throw std::out_of_range(
            "OneCycleLR step exceeds the configured total_steps");
    }

    current_step_ = epoch;

    const double first_phase_end = pct_start_ * total_steps_ - 1.0;
    const double second_phase_end = static_cast<double>(total_steps_ - 1);
    const auto cosine_anneal = [](double start, double end, double progress) {
        const double cosine = std::cos(M_PI * progress) + 1.0;
        return end + (start - end) * cosine / 2.0;
    };

    if (static_cast<double>(current_step_) <= first_phase_end) {
        const double progress =
            static_cast<double>(current_step_) / first_phase_end;
        current_lr_ = cosine_anneal(initial_lr_, max_lr_, progress);
    } else {
        const double progress =
            (static_cast<double>(current_step_) - first_phase_end) /
            (second_phase_end - first_phase_end);
        current_lr_ = cosine_anneal(max_lr_, final_lr_, progress);
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

bool OneCycleLR::ExportState(
    SchedulerState& state,
    std::string& error) const
{
    error.clear();
    state = MakeSchedulerState(
        "OneCycleLR", base_lr_, current_lr_, current_step_,
        {{"max_lr", max_lr_},
         {"total_steps", static_cast<double>(total_steps_)},
         {"pct_start", pct_start_},
         {"div_factor", div_factor_},
         {"final_div_factor", final_div_factor_}});
    return true;
}

bool OneCycleLR::ImportState(
    const SchedulerState& state,
    std::string& error)
{
    const std::map<std::string, double> expected{
        {"max_lr", max_lr_},
        {"total_steps", static_cast<double>(total_steps_)},
        {"pct_start", pct_start_},
        {"div_factor", div_factor_},
        {"final_div_factor", final_div_factor_}};
    if (!ValidateSchedulerState(
            state, "OneCycleLR", base_lr_, expected, {}, error) ||
        !state.values.empty() || state.last_step > total_steps_) {
        if (error.empty()) {
            error = "OneCycleLR scheduler state values are invalid.";
        }
        return false;
    }
    current_step_ = state.last_step;
    current_lr_ = state.current_learning_rate;
    optimizer_->SetLearningRate(current_lr_);
    return true;
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
