#pragma once

#include "debug_graph_trace_executor.h"
#include <arrow/table.h>
#include <memory>
#include <string>

namespace cyxwiz {

class DebugOperatorTraceAdapter {
public:
    DebugGraphTraceStep BuildStep(
        int node_id,
        const std::string& node_name,
        const std::string& node_type,
        const std::shared_ptr<arrow::Table>& input,
        const std::shared_ptr<arrow::Table>& output,
        float duration_ms = 0.0f) const;
};

} // namespace cyxwiz
