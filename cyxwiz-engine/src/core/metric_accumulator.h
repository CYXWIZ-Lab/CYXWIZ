#pragma once

// One pass's running loss and metrics, summed on the device (TOFIX242 2.2a,
// TRN-08). A batch adds without waiting for the device; Read() brings the sums
// to the host on the report cadence and at the pass's end. Training and
// evaluation use the same accumulator, for every task kind:
//   - the loss, weighted by its mean denominator (a device scalar: the
//     observations a Mean loss divided by) or by a host weight (Sum: 1);
//   - decision counts (classification, pair, triplet, next token): correct,
//     total, and two refusal counts the caller's builder reports (a target
//     outside the classes, a non-finite logit), checked at Read();
//   - regression: |error| and error^2 sums in the target's original units.
// A non-finite loss shows in the summed loss (NaN and infinity propagate), so
// Read() says whether every loss added so far was finite.

#include "regression_target_transform.h"

#include <cyxwiz/loss.h>
#include <cyxwiz/tensor.h>

#include <arrayfire.h>

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace cyxwiz {

class MetricAccumulator {
public:
    struct Totals {
        double loss_sum = 0.0;    // the weighted loss sum (Sum reduction: the losses' sum)
        double weight_sum = 0.0;  // the weights summed (host and device)
        uint64_t correct = 0;
        uint64_t total = 0;
        double absolute_error_sum = 0.0;
        double squared_error_sum = 0.0;
        uint64_t regression_values = 0;
        bool finite = true;  // every loss added was finite
        uint64_t batches = 0;

        // The pass's loss: the weighted mean (Mean), or the sum (Sum).
        double Loss(Reduction reduction) const;
        double Accuracy() const { return total > 0 ? static_cast<double>(correct) / static_cast<double>(total) : 0.0; }
        double Mae() const { return regression_values > 0 ? absolute_error_sum / static_cast<double>(regression_values) : 0.0; }
        double Rmse() const;
    };

    // `loss` is the batch's scalar loss on the device. `device_weight` (may be
    // null) is the loss's mean denominator on the device; otherwise
    // `host_weight` weighs it.
    void AddLoss(const Tensor& loss, const Tensor* device_weight, float host_weight);
    // A device array of 2 or 4 counts: [correct, total] or [correct, total,
    // targets outside the classes, non-finite logits]; any integer or float type.
    void AddCounts(const af::array& counts);
    // Counts a host computation already made (token tagging's entity metrics).
    void AddHostCounts(uint64_t correct, uint64_t total);
    // A regression batch: predictions and targets hold rows x width values.
    void AddRegression(const Tensor& predictions, const Tensor& targets, size_t rows, size_t width,
                       const RegressionTargetTransform& transform);
    // Continues a pass from saved totals (a run resumed inside an epoch).
    void Restore(const Totals& totals);

    // The sums so far: at most two small device reads, named in the trace
    // `sums_operation` (loss, weights, regression errors) and
    // `counts_operation` (decision counts). Throws when a builder reported a
    // target outside the classes or a non-finite logit.
    Totals Read(std::string_view sums_operation, std::string_view counts_operation);

private:
    af::array sums_;    // compute precision: [loss_sum, device_weight_sum, |e| sum, e^2 sum]
    af::array counts_;  // s64: [correct, total, outside the classes, non-finite logits]
    bool has_sums_ = false;
    bool has_counts_ = false;
    double host_weight_sum_ = 0.0;
    uint64_t host_correct_ = 0;
    uint64_t host_total_ = 0;
    uint64_t regression_values_ = 0;
    uint64_t batches_ = 0;
    // Restored totals the device sums continue from
    double restored_loss_sum_ = 0.0;
    double restored_absolute_error_ = 0.0;
    double restored_squared_error_ = 0.0;

    void AddSums(const af::array& values);  // 4 values in sums_'s order
};

}  // namespace cyxwiz
