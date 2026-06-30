#include "log_transform_operator.h"
#include "../profiler_trace.h"
#include "ts_column_utils.h"

#include <spdlog/spdlog.h>
#include <cmath>
#include <cstdint>
#include <utility>

namespace cyxwiz {

namespace {

void ReportProgress(const PipelineOperatorProgressCallback& callback,
                    std::string stage,
                    std::string message,
                    float progress,
                    uint64_t processed_items = 0,
                    uint64_t total_items = 0,
                    uint64_t estimated_memory_bytes = 0) {
    if (!callback) return;
    PipelineOperatorProgress event;
    event.stage = std::move(stage);
    event.message = std::move(message);
    event.progress = progress;
    event.processed_items = processed_items;
    event.total_items = total_items;
    event.estimated_memory_bytes = estimated_memory_bytes;
    callback(event);
}

} // namespace

bool LogTransformOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {

    value_col_.clear();

    auto it = params.find("value_col");
    if (it == params.end() || it->second.empty()) {
        error = "LogTransform: 'value_col' parameter is required";
        return false;
    }
    value_col_ = it->second;
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
LogTransformOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz LogTransform Materializer");
    if (!input) {
        return arrow::Status::Invalid("LogTransform: input table is null");
    }

    ReportProgress(progress_callback_, "read_column",
                   "Reading column for log transform", 0.10f, 0,
                   static_cast<uint64_t>(input->num_rows()));
    auto column = input->GetColumnByName(value_col_);
    if (!column) {
        return arrow::Status::KeyError(
            "LogTransform: column '" + value_col_ + "' not found");
    }

    int col_idx = input->schema()->GetFieldIndex(value_col_);

    std::vector<float> values;
    std::string bad_type;
    if (!ReadColumnAsFloat(column, values, bad_type)) {
        return arrow::Status::TypeError(
            "LogTransform: column '" + value_col_ +
            "' has unsupported type '" + bad_type + "'");
    }

    ReportProgress(progress_callback_, "transform",
                   "Applying log1p transform", 0.60f,
                   static_cast<uint64_t>(values.size()),
                   static_cast<uint64_t>(values.size()),
                   static_cast<uint64_t>(values.size() * sizeof(float)));
    for (float& v : values) {
        v = std::log1p(v);
    }

    ReportProgress(progress_callback_, "write_output",
                   "Writing transformed column", 0.90f,
                   static_cast<uint64_t>(values.size()),
                   static_cast<uint64_t>(values.size()),
                   static_cast<uint64_t>(values.size() * sizeof(float)));
    ARROW_ASSIGN_OR_RAISE(
        auto out_table,
        ReplaceColumnWithFloat(input, col_idx, values,
                               static_cast<int64_t>(values.size())));

    spdlog::info("LogTransform: applied log1p to '{}' ({} values)",
                 value_col_, values.size());
    ReportProgress(progress_callback_, "complete",
                   "Log transform complete", 1.0f,
                   static_cast<uint64_t>(values.size()),
                   static_cast<uint64_t>(values.size()),
                   static_cast<uint64_t>(values.size() * sizeof(float)));
    return out_table;
}

} // namespace cyxwiz
