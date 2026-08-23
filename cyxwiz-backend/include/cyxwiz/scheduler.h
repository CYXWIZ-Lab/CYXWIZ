#pragma once

#include "api_export.h"
#include "tensor.h"  // Required before optimizer.h (uses Tensor in std::map)
#include "optimizer.h"
#include <map>
#include <memory>
#include <string>
#include <functional>

namespace cyxwiz {

/**
 * Learning rate scheduler types
 */
enum class SchedulerType {
    StepLR,          // Decay by gamma every step_size epochs
    ExponentialLR,   // Decay by gamma every epoch
    CosineAnnealing, // Cosine annealing without warm restarts
    ReduceLROnPlateau, // Reduce when metric stops improving
    LinearWarmup,    // Linear warmup followed by decay
    OneCycleLR       // 1cycle policy
};

/**
 * Typed, backend-owned scheduler state required to reproduce the next LR.
 * Checkpoint code serializes this envelope without inspecting concrete
 * scheduler internals.
 */
struct SchedulerState {
    int schema_version = 1;
    std::string scheduler_type;
    double base_learning_rate = 0.0;
    double current_learning_rate = 0.0;
    int last_step = 0;
    std::map<std::string, double> hyperparameters;
    std::map<std::string, std::string> string_hyperparameters;
    std::map<std::string, double> values;
};

/**
 * Base class for learning rate schedulers
 */
class CYXWIZ_API LRScheduler {
public:
    virtual ~LRScheduler() = default;

    /**
     * Step the scheduler (call at end of each epoch or batch)
     * @param epoch Current epoch number (1-indexed)
     * @param metric Optional metric value (for plateau-based schedulers)
     */
    virtual void Step(int epoch, float metric = 0.0f) = 0;

    /**
     * Get the current learning rate
     */
    virtual double GetLR() const = 0;

    /**
     * Get scheduler name
     */
    virtual std::string GetName() const = 0;

    /**
     * Reset scheduler state
     */
    virtual void Reset() = 0;

    /** Export complete state required for the next scheduler step. */
    virtual bool ExportState(SchedulerState& state, std::string& error) const {
        state = SchedulerState{};
        error = "This scheduler does not implement exact state export.";
        return false;
    }

    /** Import state transactionally; failure must not mutate the scheduler. */
    virtual bool ImportState(const SchedulerState& state, std::string& error) {
        (void)state;
        error = "This scheduler does not implement exact state import.";
        return false;
    }

protected:
    double base_lr_ = 0.001;
    double current_lr_ = 0.001;
};

/**
 * StepLR - Decays learning rate by gamma every step_size epochs
 *
 * lr = base_lr * gamma^(epoch // step_size)
 */
class CYXWIZ_API StepLR : public LRScheduler {
public:
    /**
     * @param optimizer Optimizer to adjust
     * @param step_size Period of learning rate decay
     * @param gamma Multiplicative factor of learning rate decay (default: 0.1)
     */
    StepLR(Optimizer* optimizer, int step_size, double gamma = 0.1);

    void Step(int epoch, float metric = 0.0f) override;
    double GetLR() const override { return current_lr_; }
    std::string GetName() const override { return "StepLR"; }
    void Reset() override;
    bool ExportState(SchedulerState& state, std::string& error) const override;
    bool ImportState(const SchedulerState& state, std::string& error) override;

private:
    Optimizer* optimizer_;
    int step_size_;
    double gamma_;
    int last_epoch_ = 0;
};

/**
 * ExponentialLR - Decays learning rate by gamma every epoch
 *
 * lr = base_lr * gamma^epoch
 */
class CYXWIZ_API ExponentialLR : public LRScheduler {
public:
    /**
     * @param optimizer Optimizer to adjust
     * @param gamma Multiplicative factor of learning rate decay (default: 0.95)
     */
    ExponentialLR(Optimizer* optimizer, double gamma = 0.95);

    void Step(int epoch, float metric = 0.0f) override;
    double GetLR() const override { return current_lr_; }
    std::string GetName() const override { return "ExponentialLR"; }
    void Reset() override;
    bool ExportState(SchedulerState& state, std::string& error) const override;
    bool ImportState(const SchedulerState& state, std::string& error) override;

private:
    Optimizer* optimizer_;
    double gamma_;
    int last_epoch_ = 0;
};

/**
 * CosineAnnealingLR - Cosine annealing schedule
 *
 * lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * epoch / T_max)) / 2
 */
class CYXWIZ_API CosineAnnealingLR : public LRScheduler {
public:
    /**
     * @param optimizer Optimizer to adjust
     * @param T_max Maximum number of iterations
     * @param eta_min Minimum learning rate (default: 0)
     */
    CosineAnnealingLR(Optimizer* optimizer, int T_max, double eta_min = 0.0);

    void Step(int epoch, float metric = 0.0f) override;
    double GetLR() const override { return current_lr_; }
    std::string GetName() const override { return "CosineAnnealingLR"; }
    void Reset() override;
    bool ExportState(SchedulerState& state, std::string& error) const override;
    bool ImportState(const SchedulerState& state, std::string& error) override;

private:
    Optimizer* optimizer_;
    int T_max_;
    double eta_min_;
    int last_epoch_ = 0;
};

/**
 * ReduceLROnPlateau - Reduce learning rate when a metric has stopped improving
 */
class CYXWIZ_API ReduceLROnPlateau : public LRScheduler {
public:
    /**
     * @param optimizer Optimizer to adjust
     * @param mode 'min' for minimizing metric, 'max' for maximizing
     * @param factor Factor by which the learning rate will be reduced (default: 0.1)
     * @param patience Number of epochs with no improvement after which LR is reduced
     * @param threshold Threshold for measuring improvement (default: 1e-4)
     * @param min_lr Minimum learning rate (default: 1e-8)
     */
    ReduceLROnPlateau(
        Optimizer* optimizer,
        const std::string& mode = "min",
        double factor = 0.1,
        int patience = 10,
        double threshold = 1e-4,
        double min_lr = 1e-8
    );

    void Step(int epoch, float metric = 0.0f) override;
    double GetLR() const override { return current_lr_; }
    std::string GetName() const override { return "ReduceLROnPlateau"; }
    void Reset() override;
    bool ExportState(SchedulerState& state, std::string& error) const override;
    bool ImportState(const SchedulerState& state, std::string& error) override;

    /**
     * Check if improvement was detected
     */
    bool IsImproving() const { return num_bad_epochs_ == 0; }

    /**
     * Get number of epochs without improvement
     */
    int GetBadEpochs() const { return num_bad_epochs_; }

private:
    Optimizer* optimizer_;
    std::string mode_;  // "min" or "max"
    double factor_;
    int patience_;
    double threshold_;
    double min_lr_;

    double best_metric_;
    int num_bad_epochs_ = 0;
    int last_epoch_ = 0;

    bool IsBetter(float current, double best) const;
};

/**
 * LinearWarmupLR - Linear warmup followed by constant or decay
 *
 * During warmup, linearly interpolate from start_lr at construction/step 0 to
 * base_lr at warmup_epochs. After warmup, maintain base_lr. This matches a
 * PyTorch LambdaLR linear-warmup schedule.
 */
class CYXWIZ_API LinearWarmupLR : public LRScheduler {
public:
    /**
     * @param optimizer Optimizer to adjust
     * @param warmup_epochs Number of warmup epochs
     * @param base_lr Target learning rate after warmup
     * @param start_lr Starting learning rate (default: 0)
     */
    LinearWarmupLR(
        Optimizer* optimizer,
        int warmup_epochs,
        double base_lr,
        double start_lr = 0.0
    );

    void Step(int epoch, float metric = 0.0f) override;
    double GetLR() const override { return current_lr_; }
    std::string GetName() const override { return "LinearWarmupLR"; }
    void Reset() override;
    bool ExportState(SchedulerState& state, std::string& error) const override;
    bool ImportState(const SchedulerState& state, std::string& error) override;

    bool IsWarmupComplete() const { return last_epoch_ >= warmup_epochs_; }

private:
    Optimizer* optimizer_;
    int warmup_epochs_;
    double start_lr_;
    int last_epoch_ = 0;
};

/**
 * OneCycleLR - 1cycle learning rate policy
 *
 * The default PyTorch-compatible two-phase policy cosine-anneals the learning
 * rate from initial_lr to max_lr, then from max_lr to final_lr.
 */
class CYXWIZ_API OneCycleLR : public LRScheduler {
public:
    /**
     * @param optimizer Optimizer to adjust
     * @param max_lr Maximum learning rate
     * @param total_steps Total number of steps (epochs * steps_per_epoch)
     * @param pct_start Percentage of cycle spent increasing LR (default: 0.3)
     * @param div_factor Initial LR = max_lr / div_factor (default: 25)
     * @param final_div_factor Final LR = initial_lr / final_div_factor (default: 1e4)
     */
    OneCycleLR(
        Optimizer* optimizer,
        double max_lr,
        int total_steps,
        double pct_start = 0.3,
        double div_factor = 25.0,
        double final_div_factor = 1e4
    );

    void Step(int epoch, float metric = 0.0f) override;
    double GetLR() const override { return current_lr_; }
    std::string GetName() const override { return "OneCycleLR"; }
    void Reset() override;
    bool ExportState(SchedulerState& state, std::string& error) const override;
    bool ImportState(const SchedulerState& state, std::string& error) override;

private:
    Optimizer* optimizer_;
    double max_lr_;
    int total_steps_;
    double pct_start_;
    double div_factor_;
    double final_div_factor_;
    double initial_lr_;
    double final_lr_;

    int current_step_ = 0;
};

/**
 * Factory function to create schedulers
 */
CYXWIZ_API std::unique_ptr<LRScheduler> CreateScheduler(
    SchedulerType type,
    Optimizer* optimizer,
    double param1 = 0.1,   // gamma/factor
    int param2 = 10,       // step_size/patience/warmup_epochs/T_max
    double param3 = 0.0    // min_lr/eta_min
);

} // namespace cyxwiz
