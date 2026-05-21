#pragma once

#include "debug_trace_record.h"
#include "graph_compiler.h"
#include "../gui/node_editor.h"
#include <string>
#include <vector>

namespace cyxwiz {

class TextPreprocessingTracer {
public:
    std::vector<DebugTraceRecord> TraceSample(
        const TrainingConfiguration& config,
        const std::vector<gui::MLNode>& nodes,
        const std::string& run_id,
        size_t sample_index) const;

    std::vector<DebugTraceRecord> TraceFirstSample(
        const TrainingConfiguration& config,
        const std::vector<gui::MLNode>& nodes,
        const std::string& run_id) const;
};

} // namespace cyxwiz
