#include "debug_numerics.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace cyxwiz {

namespace {

bool PrefixMatches(const std::vector<size_t>& prediction,
                   const std::vector<size_t>& target,
                   size_t target_rank) {
    if (prediction.size() < target_rank || target.size() < target_rank) {
        return false;
    }
    return std::equal(target.begin(), target.begin() + target_rank,
                      prediction.begin());
}

bool IsClassIndexLoss(gui::NodeType loss_type) {
    return loss_type == gui::NodeType::CrossEntropyLoss ||
           loss_type == gui::NodeType::FocalLoss ||
           loss_type == gui::NodeType::NLLLoss;
}

} // namespace

DebugTensorNumericSummary ScanDebugTensorNumerics(
    const Tensor& tensor,
    gui::NodeType producer_type) {
    DebugTensorNumericSummary summary;
    if (tensor.GetDataType() != DataType::Float32) {
        return summary;
    }

    summary.available = true;
    summary.element_count = tensor.NumElements();
    const float* values = tensor.ReadData<float>();
    long double sum = 0.0L;
    long double sum_squares = 0.0L;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();

    const bool element_saturation =
        producer_type == gui::NodeType::Sigmoid ||
        producer_type == gui::NodeType::Tanh ||
        producer_type == gui::NodeType::ReLU;
    if (producer_type == gui::NodeType::Sigmoid) {
        summary.saturation_summary_available = true;
        summary.saturation_definition =
            "output <= 0.01 or output >= 0.99";
    } else if (producer_type == gui::NodeType::Tanh) {
        summary.saturation_summary_available = true;
        summary.saturation_definition = "abs(output) >= 0.99";
    } else if (producer_type == gui::NodeType::ReLU) {
        summary.saturation_summary_available = true;
        summary.saturation_definition = "output <= 0.0";
    } else if (producer_type == gui::NodeType::Softmax) {
        summary.saturation_summary_available = true;
        summary.saturation_definition =
            "row maximum probability >= 0.999";
    }

    for (size_t index = 0; index < summary.element_count; ++index) {
        const double value = static_cast<double>(values[index]);
        if (std::isnan(value)) {
            ++summary.nan_count;
            continue;
        }
        if (std::isinf(value)) {
            ++summary.inf_count;
            continue;
        }
        ++summary.finite_count;
        sum += value;
        sum_squares += static_cast<long double>(value) * value;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        summary.maximum_absolute =
            std::max(summary.maximum_absolute, std::abs(value));
        if (value == 0.0) {
            ++summary.zero_count;
        }
        if (std::abs(value) >= kDebugExplodingAbsoluteThreshold) {
            ++summary.exploding_count;
        }
        if (element_saturation) {
            const bool candidate =
                (producer_type == gui::NodeType::Sigmoid &&
                 (value <= 0.01 || value >= 0.99)) ||
                (producer_type == gui::NodeType::Tanh &&
                 std::abs(value) >= 0.99) ||
                (producer_type == gui::NodeType::ReLU && value <= 0.0);
            if (candidate) {
                ++summary.saturation_candidate_count;
            }
        }
    }

    if (summary.finite_count > 0) {
        summary.minimum = minimum;
        summary.maximum = maximum;
        summary.mean = static_cast<double>(
            sum / static_cast<long double>(summary.finite_count));
        summary.l2_norm = std::sqrt(static_cast<double>(sum_squares));
    }

    if (element_saturation) {
        summary.saturation_observation_count = summary.finite_count;
    } else if (producer_type == gui::NodeType::Softmax &&
               !tensor.Shape().empty()) {
        const size_t class_count = tensor.Shape().back();
        if (class_count > 0 &&
            summary.element_count % class_count == 0) {
            const size_t row_count = summary.element_count / class_count;
            summary.saturation_observation_count = row_count;
            for (size_t row = 0; row < row_count; ++row) {
                double row_maximum = -std::numeric_limits<double>::infinity();
                for (size_t column = 0; column < class_count; ++column) {
                    const double value = static_cast<double>(
                        values[row * class_count + column]);
                    if (std::isfinite(value)) {
                        row_maximum = std::max(row_maximum, value);
                    }
                }
                if (row_maximum >= kDebugSoftmaxProbabilityThreshold) {
                    ++summary.saturation_candidate_count;
                }
            }
        } else {
            summary.saturation_summary_available = false;
            summary.saturation_definition.clear();
        }
    }

    if (summary.saturation_observation_count > 0) {
        summary.saturation_candidate_ratio =
            static_cast<double>(summary.saturation_candidate_count) /
            static_cast<double>(summary.saturation_observation_count);
        summary.saturation_candidate =
            summary.saturation_candidate_ratio >=
            kDebugActivationSaturationRatioThreshold;
    }
    summary.dead_relu_candidate =
        producer_type == gui::NodeType::ReLU && summary.finite_count > 0 &&
        static_cast<double>(summary.zero_count) /
                static_cast<double>(summary.finite_count) >=
            kDebugDeadReluZeroRatioThreshold;
    summary.softmax_saturation_candidate =
        producer_type == gui::NodeType::Softmax &&
        summary.saturation_candidate;
    return summary;
}

void MergeDebugTensorNumerics(DebugTensorNumericSummary& aggregate,
                              const DebugTensorNumericSummary& sample) {
    if (!sample.available) {
        return;
    }
    if (!aggregate.available) {
        aggregate = sample;
        aggregate.saturation_summary_available = false;
        aggregate.saturation_definition.clear();
        aggregate.saturation_candidate_count = 0;
        aggregate.saturation_observation_count = 0;
        aggregate.saturation_candidate_ratio = 0.0;
        aggregate.saturation_candidate = false;
        aggregate.dead_relu_candidate = false;
        aggregate.softmax_saturation_candidate = false;
        return;
    }

    const size_t old_finite_count = aggregate.finite_count;
    const long double combined_sum =
        static_cast<long double>(aggregate.mean) * old_finite_count +
        static_cast<long double>(sample.mean) * sample.finite_count;
    const long double combined_squares =
        static_cast<long double>(aggregate.l2_norm) * aggregate.l2_norm +
        static_cast<long double>(sample.l2_norm) * sample.l2_norm;
    aggregate.element_count += sample.element_count;
    aggregate.finite_count += sample.finite_count;
    aggregate.nan_count += sample.nan_count;
    aggregate.inf_count += sample.inf_count;
    aggregate.zero_count += sample.zero_count;
    aggregate.exploding_count += sample.exploding_count;
    if (sample.finite_count > 0) {
        if (old_finite_count == 0) {
            aggregate.minimum = sample.minimum;
            aggregate.maximum = sample.maximum;
        } else {
            aggregate.minimum = std::min(aggregate.minimum, sample.minimum);
            aggregate.maximum = std::max(aggregate.maximum, sample.maximum);
        }
        aggregate.maximum_absolute = std::max(
            aggregate.maximum_absolute, sample.maximum_absolute);
    }
    if (aggregate.finite_count > 0) {
        aggregate.mean = static_cast<double>(
            combined_sum /
            static_cast<long double>(aggregate.finite_count));
        aggregate.l2_norm = std::sqrt(
            static_cast<double>(combined_squares));
    }
}

void AttachDebugNumericsPayload(DebugTraceRecord& trace,
                                const DebugTensorNumericSummary& summary,
                                const std::string& subject) {
    auto& payload = trace.payload;
    payload["numerics_schema"] = kDebugNumericsSchema;
    payload["numeric_summary_available"] = summary.available;
    payload["numeric_subject"] = subject;
    payload["numeric_host_read_performed"] = summary.available;
    payload["numeric_values_included"] = false;
    payload["numeric_summary_scope"] = "bounded_debug_host_read";
    if (!summary.available) {
        payload["numeric_summary_reason"] =
            "The tensor is not Float32 or was unavailable.";
        return;
    }
    payload["numeric_element_count"] = summary.element_count;
    payload["numeric_finite_count"] = summary.finite_count;
    payload["numeric_nan_count"] = summary.nan_count;
    payload["numeric_inf_count"] = summary.inf_count;
    payload["numeric_zero_count"] = summary.zero_count;
    payload["numeric_zero_ratio"] = summary.finite_count > 0
        ? static_cast<double>(summary.zero_count) /
            static_cast<double>(summary.finite_count)
        : 0.0;
    payload["numeric_exploding_count"] = summary.exploding_count;
    payload["numeric_exploding_abs_threshold"] =
        kDebugExplodingAbsoluteThreshold;
    payload["numeric_exploding_values"] = summary.exploding_count > 0;
    if (summary.finite_count > 0) {
        payload["numeric_min"] = summary.minimum;
        payload["numeric_max"] = summary.maximum;
        payload["numeric_mean"] = summary.mean;
        payload["numeric_max_abs"] = summary.maximum_absolute;
        payload["numeric_l2_norm"] = summary.l2_norm;
    }
    payload["saturation_summary_available"] =
        summary.saturation_summary_available;
    if (summary.saturation_summary_available) {
        payload["saturation_definition"] =
            summary.saturation_definition;
        payload["saturation_candidate_count"] =
            summary.saturation_candidate_count;
        payload["saturation_observation_count"] =
            summary.saturation_observation_count;
        payload["saturation_candidate_ratio"] =
            summary.saturation_candidate_ratio;
        payload["saturation_ratio_threshold"] =
            kDebugActivationSaturationRatioThreshold;
        payload["saturation_candidate"] =
            summary.saturation_candidate;
        payload["dead_relu_candidate"] =
            summary.dead_relu_candidate;
        payload["dead_relu_zero_ratio_threshold"] =
            kDebugDeadReluZeroRatioThreshold;
        payload["softmax_saturation_candidate"] =
            summary.softmax_saturation_candidate;
        payload["softmax_probability_threshold"] =
            kDebugSoftmaxProbabilityThreshold;
    }
}

DebugPredictionTargetCompatibility CheckDebugPredictionTargetShapes(
    const std::vector<size_t>& prediction_shape,
    const std::vector<size_t>& target_shape,
    gui::NodeType loss_type) {
    DebugPredictionTargetCompatibility result;
    if (prediction_shape.empty() || target_shape.empty()) {
        result.reason = "Prediction or target shape is unavailable.";
        return result;
    }
    result.available = true;
    if (prediction_shape == target_shape) {
        result.compatible = true;
        result.reason = "Prediction and target shapes match exactly.";
        return result;
    }

    if (IsClassIndexLoss(loss_type) && prediction_shape.size() >= 2) {
        const size_t expected_target_rank = prediction_shape.size() - 1;
        if (target_shape.size() == expected_target_rank &&
            PrefixMatches(prediction_shape, target_shape,
                          expected_target_rank)) {
            result.compatible = true;
            result.reason =
                "Class-index target shape matches the logits prefix.";
            return result;
        }
        if (target_shape.size() == prediction_shape.size() &&
            target_shape.back() == 1 &&
            PrefixMatches(prediction_shape, target_shape,
                          expected_target_rank)) {
            result.compatible = true;
            result.reason =
                "Singleton class-index target shape matches the logits prefix.";
            return result;
        }
    }

    if ((loss_type == gui::NodeType::BCELoss ||
         loss_type == gui::NodeType::BCEWithLogits) &&
        prediction_shape.size() == target_shape.size() + 1 &&
        prediction_shape.back() == 1 &&
        PrefixMatches(prediction_shape, target_shape, target_shape.size())) {
        result.compatible = true;
        result.reason =
            "Scalar binary targets match a singleton prediction width.";
        return result;
    }

    result.reason =
        "Prediction and target shapes are incompatible for the configured loss.";
    return result;
}

} // namespace cyxwiz
