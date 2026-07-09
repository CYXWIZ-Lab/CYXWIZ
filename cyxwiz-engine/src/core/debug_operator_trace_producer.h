#pragma once

#include "debug_trace_record.h"
#include "../gui/node_editor.h"

#include <arrow/table.h>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

class DebugOperatorTraceProducer {
public:
    std::vector<DebugTraceRecord> TracePreprocessingGraph(
        const std::string& run_id,
        const std::vector<gui::MLNode>& nodes,
        const std::vector<gui::NodeLink>& links,
        const std::shared_ptr<arrow::Table>& source_table,
        const std::string& source_dataset_name = {},
        size_t selected_sample_index = 0,
        size_t max_debug_rows = 32) const;

private:
    std::vector<DebugTraceRecord> TraceTextTokenizer(
        const std::string& run_id,
        const gui::MLNode& node,
        const std::vector<gui::MLNode>& nodes,
        const std::vector<gui::NodeLink>& links,
        const std::shared_ptr<arrow::Table>& input,
        std::shared_ptr<arrow::Table>* output) const;

    DebugTraceRecord BuildWarningTrace(
        const std::string& run_id,
        const gui::MLNode& node,
        const std::shared_ptr<arrow::Table>& input,
        const std::string& message,
        const std::string& error_code = errors::Runtime::ExecutionFailed) const;
};

} // namespace cyxwiz
