#pragma once

#include "debug_trace_record.h"
#include "model_builder.h"
#include "../gui/node_editor.h"

#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz {

class DebugLossMetricExplainer {
public:
    static constexpr const char* kSchema =
        "cyxwiz.debug.loss_metric_explanation.v1";
    static constexpr size_t kMaxRows = 64;
    static constexpr size_t kMaxParametersPerRow = 16;
    static constexpr size_t kMaxClassWeights = 32;

    DebugTraceRecord BuildTrace(
        const std::string& run_id,
        const TrainingConfiguration& config,
        const std::vector<gui::MLNode>& nodes,
        const std::vector<DebugTraceRecord>& traces) const;
};

} // namespace cyxwiz
