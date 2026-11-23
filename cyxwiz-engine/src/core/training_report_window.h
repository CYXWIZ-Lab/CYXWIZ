#pragma once

#include <cyxwiz/losses/loss_base.h>

#include <cstdint>

namespace cyxwiz {

// The Training Dashboard's batch points (TOFIX178 1.8i): each one is the mean
// over the batches since the previous point, taken from the epoch's running
// sums, so the curve does not restart at every epoch and no point is one
// batch's noise. One per epoch (the sums restart with it).
class ReportWindow {
public:
    // Batch 1 ends a window in the first epoch only (early feedback); a later
    // epoch's first point covers its first log_interval batches. The
    // Dashboard draws the same points (TrainingManager's batch callback).
    static bool Ends(int epoch, int batch_num) { return batch_num > 1 || epoch <= 1; }

    // running = the epoch's loss so far (a mean, or the sum under Reduction::Sum);
    // weight = the loss weight summed over the epoch so far.
    float Loss(Reduction reduction, float running, double weight) {
        if (reduction == Reduction::Sum) {
            const float window = running - static_cast<float>(loss_sum_);
            loss_sum_ = running;
            return window;
        }
        const double sum = static_cast<double>(running) * weight;
        const double added = weight - weight_;
        const float window = added > 0.0 ? static_cast<float>((sum - loss_sum_) / added) : running;
        loss_sum_ = sum;
        weight_ = weight;
        return window;
    }

    // correct / total = the epoch's counts so far; running = their ratio.
    float Accuracy(uint64_t correct, uint64_t total, float running) {
        const uint64_t added = total > total_ ? total - total_ : 0;
        const float window =
            added > 0 ? static_cast<float>(static_cast<double>(correct - correct_) / static_cast<double>(added))
                      : running;
        correct_ = correct;
        total_ = total;
        return window;
    }

private:
    double loss_sum_ = 0.0, weight_ = 0.0;
    uint64_t correct_ = 0, total_ = 0;
};

}  // namespace cyxwiz
