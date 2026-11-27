#include "metric_accumulator.h"

#include "algorithms/arrayfire_host_materialization.h"
#include "algorithms/arrayfire_host_values.h"  // DeviceHasDouble, HostValuesToDevice, DeviceError

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

af::dtype ComputeType() { return DeviceHasDouble() ? f64 : f32; }

af::array Scalar(const Tensor& value, const char* what) {
    if (value.NumElements() != 1) {
        throw std::invalid_argument(std::string("MetricAccumulator: the ") + what + " is not one value (" +
                                    std::to_string(value.NumElements()) + ")");
    }
    return af::flat(value.GetSemanticArray()).as(ComputeType());
}

}  // namespace

double MetricAccumulator::Totals::Loss(Reduction reduction) const {
    if (reduction == Reduction::Sum) return loss_sum;
    return weight_sum > 0.0 ? loss_sum / weight_sum : 0.0;
}

double MetricAccumulator::Totals::Rmse() const {
    return regression_values > 0 ? std::sqrt(squared_error_sum / static_cast<double>(regression_values)) : 0.0;
}

void MetricAccumulator::AddSums(const af::array& values) {
    af::array next = has_sums_ ? sums_ + values : values;
    next.eval();
    sums_ = next;
    has_sums_ = true;
}

void MetricAccumulator::AddLoss(const Tensor& loss, const Tensor* device_weight, float host_weight) {
    try {
        const af::array value = Scalar(loss, "loss");
        const af::array zero = af::constant(0.0, 1, ComputeType());
        if (device_weight != nullptr) {
            const af::array weight = Scalar(*device_weight, "loss weight");
            AddSums(af::join(0, value * weight, weight, zero, zero));
        } else {
            if (!std::isfinite(host_weight) || host_weight < 0.0f) {
                throw std::invalid_argument("MetricAccumulator: a loss weight must be finite and not negative");
            }
            AddSums(af::join(0, value * static_cast<double>(host_weight), zero, zero, zero));
            host_weight_sum_ += host_weight;
        }
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("MetricAccumulator::AddLoss", e));
    }
    ++batches_;
}

void MetricAccumulator::AddCounts(const af::array& counts) {
    if (counts.elements() != 2 && counts.elements() != 4) {
        throw std::invalid_argument("MetricAccumulator: counts are [correct, total] or [correct, total, outside, "
                                    "non-finite], not " + std::to_string(counts.elements()) + " values");
    }
    try {
        af::array values = af::flat(counts).as(s64);
        if (values.elements() == 2) values = af::join(0, values, af::constant(0, 2, s64));
        af::array next = has_counts_ ? counts_ + values : values;
        next.eval();
        counts_ = next;
        has_counts_ = true;
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("MetricAccumulator::AddCounts", e));
    }
}

void MetricAccumulator::AddHostCounts(uint64_t correct, uint64_t total) {
    host_correct_ += correct;
    host_total_ += total;
}

void MetricAccumulator::AddRegression(const Tensor& predictions, const Tensor& targets, size_t rows, size_t width,
                                      const RegressionTargetTransform& transform) {
    if (rows == 0 || width == 0) return;
    const auto as_rows = [rows, width](const Tensor& values, const char* what) {
        if (values.NumElements() != rows * width) {
            throw std::invalid_argument(std::string("MetricAccumulator: regression ") + what + " hold " +
                                        std::to_string(values.NumElements()) + " values, expected " +
                                        std::to_string(rows) + " x " + std::to_string(width));
        }
        return values.Reshape({rows, width}).GetArrayRowMajor2D().as(ComputeType());
    };
    try {
        af::array error = as_rows(predictions, "predictions") - as_rows(targets, "targets");
        // In the target's original units when a scaler was resolved for it
        if (transform.enabled && transform.resolved && transform.scales.size() == width) {
            const std::vector<double> scales(transform.scales.begin(), transform.scales.end());
            const af::array scale_row = af::moddims(HostValuesToDevice(scales), 1, static_cast<dim_t>(width));
            error = error * af::tile(scale_row, static_cast<unsigned>(rows), 1U);
        }
        const af::array zero = af::constant(0.0, 1, ComputeType());
        AddSums(af::join(0, zero, zero, af::sum(af::flat(af::abs(error))), af::sum(af::flat(error * error))));
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("MetricAccumulator::AddRegression", e));
    }
    regression_values_ += rows * width;
}

void MetricAccumulator::Restore(const Totals& totals) {
    restored_loss_sum_ = totals.loss_sum;
    host_weight_sum_ += totals.weight_sum;
    host_correct_ += totals.correct;
    host_total_ += totals.total;
    restored_absolute_error_ = totals.absolute_error_sum;
    restored_squared_error_ = totals.squared_error_sum;
    regression_values_ += totals.regression_values;
    batches_ += totals.batches;
}

MetricAccumulator::Totals MetricAccumulator::Read(std::string_view sums_operation,
                                                  std::string_view counts_operation) {
    Totals out;
    out.loss_sum = restored_loss_sum_;
    out.weight_sum = host_weight_sum_;
    out.correct = host_correct_;
    out.total = host_total_;
    out.absolute_error_sum = restored_absolute_error_;
    out.squared_error_sum = restored_squared_error_;
    out.regression_values = regression_values_;
    out.batches = batches_;
    try {
        if (has_sums_) {
            double sums[4] = {0.0, 0.0, 0.0, 0.0};
            af::array data = sums_;
            data.eval();
            if (data.type() == f64) {
                MaterializeArrayFireToHost(data, sums, ArrayFireHostSyncCategory::LossScalarReadback,
                                           std::string(sums_operation), "arrayfire_column_major");
            } else {
                float narrow[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                MaterializeArrayFireToHost(data, narrow, ArrayFireHostSyncCategory::LossScalarReadback,
                                           std::string(sums_operation), "arrayfire_column_major");
                for (int i = 0; i < 4; ++i) sums[i] = narrow[i];
            }
            out.loss_sum += sums[0];
            out.weight_sum += sums[1];
            out.absolute_error_sum += sums[2];
            out.squared_error_sum += sums[3];
        }
        if (has_counts_) {
            int64_t counts[4] = {0, 0, 0, 0};
            af::array data = counts_;
            data.eval();
            MaterializeArrayFireToHost(data, counts, ArrayFireHostSyncCategory::MetricScalarReadback,
                                       std::string(counts_operation), "arrayfire_column_major");
            if (counts[2] != 0) throw std::runtime_error("a target is outside the classes the model predicts");
            if (counts[3] != 0) throw std::runtime_error("the model's outputs hold NaN or infinite values");
            out.correct += static_cast<uint64_t>(counts[0]);
            out.total += static_cast<uint64_t>(counts[1]);
        }
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("MetricAccumulator::Read", e));
    }
    out.finite = std::isfinite(out.loss_sum);
    return out;
}

}  // namespace cyxwiz
