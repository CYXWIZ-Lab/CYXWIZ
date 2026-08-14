#pragma once

#include "debug_trace_record.h"
#include "../gui/node_editor.h"

#include <cyxwiz/tensor.h>

#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz {

inline constexpr const char* kDebugNumericsSchema =
    "cyxwiz.debug.numerics.v1";
inline constexpr double kDebugExplodingAbsoluteThreshold = 1.0e6;
inline constexpr double kDebugDeadReluZeroRatioThreshold = 0.99;
inline constexpr double kDebugActivationSaturationRatioThreshold = 0.95;
inline constexpr double kDebugSoftmaxProbabilityThreshold = 0.999;

struct DebugTensorNumericSummary {
    bool available = false;
    size_t element_count = 0;
    size_t finite_count = 0;
    size_t nan_count = 0;
    size_t inf_count = 0;
    size_t zero_count = 0;
    size_t exploding_count = 0;
    double minimum = 0.0;
    double maximum = 0.0;
    double mean = 0.0;
    double maximum_absolute = 0.0;
    double l2_norm = 0.0;

    bool saturation_summary_available = false;
    std::string saturation_definition;
    size_t saturation_candidate_count = 0;
    size_t saturation_observation_count = 0;
    double saturation_candidate_ratio = 0.0;
    bool saturation_candidate = false;
    bool dead_relu_candidate = false;
    bool softmax_saturation_candidate = false;
};

struct DebugPredictionTargetCompatibility {
    bool available = false;
    bool compatible = false;
    std::string reason;
};

DebugTensorNumericSummary ScanDebugTensorNumerics(
    const Tensor& tensor,
    gui::NodeType producer_type);

void MergeDebugTensorNumerics(DebugTensorNumericSummary& aggregate,
                              const DebugTensorNumericSummary& sample);

void AttachDebugNumericsPayload(DebugTraceRecord& trace,
                                const DebugTensorNumericSummary& summary,
                                const std::string& subject);

DebugPredictionTargetCompatibility CheckDebugPredictionTargetShapes(
    const std::vector<size_t>& prediction_shape,
    const std::vector<size_t>& target_shape,
    gui::NodeType loss_type);

} // namespace cyxwiz
