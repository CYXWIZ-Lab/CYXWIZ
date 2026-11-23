#include "debug_numerics.h"

#include "algorithms/arrayfire_host_values.h"  // DeviceHasDouble, MaterializeArrayFireToHost

#include <arrayfire.h>

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
    size_t softmax_rows = 0;
    size_t class_count = 0;
    if (producer_type == gui::NodeType::Softmax && !tensor.Shape().empty()) {
        class_count = tensor.Shape().back();
        if (class_count > 0 && summary.element_count % class_count == 0) {
            softmax_rows = summary.element_count / class_count;
        } else {
            summary.saturation_summary_available = false;
            summary.saturation_definition.clear();
        }
    }
    if (summary.element_count == 0) {
        return summary;
    }

    // Reduced where the tensor lives; only the counts and five values come
    // back (float64 sums where the device has float64).
    const af::array values = af::flat(tensor.GetArray());
    const af::array finite = !(af::isNaN(values) || af::isInf(values));
    const af::dtype precision = DeviceHasDouble() ? f64 : f32;
    const af::array kept = af::select(finite, values, 0.0f).as(precision);
    const float infinity = std::numeric_limits<float>::infinity();

    std::vector<af::array> counts{
        af::count(af::isNaN(values)),
        af::count(af::isInf(values)),
        af::count(finite),
        af::count(finite && values == 0.0f),
        af::count(finite && af::abs(values) >= static_cast<float>(kDebugExplodingAbsoluteThreshold)),
    };
    if (element_saturation) {
        const af::array candidate =
            producer_type == gui::NodeType::Sigmoid ? (values <= 0.01f || values >= 0.99f)
            : producer_type == gui::NodeType::Tanh  ? af::abs(values) >= 0.99f
                                                    : values <= 0.0f;
        counts.push_back(af::count(finite && candidate));
    } else if (softmax_rows > 0) {
        // Rows are the row-major leading axes, the classes the last.
        const af::array rows = af::moddims(tensor.GetRowMajorFlatArray(), static_cast<dim_t>(class_count),
                                           static_cast<dim_t>(softmax_rows));
        const af::array row_maximum =
            af::max(af::select(af::isNaN(rows) || af::isInf(rows), -infinity, rows), 0);
        counts.push_back(af::count(row_maximum >= static_cast<float>(kDebugSoftmaxProbabilityThreshold)));
    }
    af::array joined_counts = counts.front();
    for (size_t i = 1; i < counts.size(); ++i) joined_counts = af::join(0, joined_counts, counts[i]);
    std::vector<unsigned> count_values(counts.size());
    MaterializeArrayFireToHost(joined_counts.as(u32), count_values.data(), ArrayFireHostSyncCategory::DebugSampleDump,
                               "ScanDebugTensorNumerics", "arrayfire_aggregate_vector", "bounded_debug_readback",
                               "numeric counts");
    summary.nan_count = count_values[0];
    summary.inf_count = count_values[1];
    summary.finite_count = count_values[2];
    summary.zero_count = count_values[3];
    summary.exploding_count = count_values[4];
    if (counts.size() > 5) {
        summary.saturation_candidate_count = count_values[5];
        summary.saturation_observation_count = element_saturation ? summary.finite_count : softmax_rows;
    }

    if (summary.finite_count > 0) {
        const af::array aggregates = af::join(
            0, af::join(0, af::sum(kept), af::sum(kept * kept)),
            af::join(0, af::min(af::select(finite, values, infinity)).as(precision),
                     af::max(af::select(finite, values, -infinity)).as(precision), af::max(af::abs(kept))));
        // f64 on the device only where it has f64 (aggregates are f32 otherwise).
        std::vector<double> host(5);
        if (precision == f64) {
            MaterializeArrayFireToHost(aggregates, host.data(), ArrayFireHostSyncCategory::DebugSampleDump,
                                       "ScanDebugTensorNumerics", "arrayfire_aggregate_vector",
                                       "bounded_debug_readback", "numeric sums and range");
        } else {
            std::vector<float> narrow(host.size());
            MaterializeArrayFireToHost(aggregates, narrow.data(), ArrayFireHostSyncCategory::DebugSampleDump,
                                       "ScanDebugTensorNumerics", "arrayfire_aggregate_vector",
                                       "bounded_debug_readback", "numeric sums and range");
            std::copy(narrow.begin(), narrow.end(), host.begin());
        }
        summary.mean = host[0] / static_cast<double>(summary.finite_count);
        summary.l2_norm = std::sqrt(host[1]);
        summary.minimum = host[2];
        summary.maximum = host[3];
        summary.maximum_absolute = host[4];
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
    payload["numeric_summary_scope"] = "device_reduction_bounded_readback";
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
