#include "log_transform_operator.h"
#include "../profiler_trace.h"
#include "feature_matrix_utils.h"
#include "materialization_memory_preflight.h"
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

    auto column = input->GetColumnByName(value_col_);
    if (!column) {
        return arrow::Status::KeyError(
            "LogTransform: column '" + value_col_ + "' not found");
    }
    if (!IsNumericChunked(column)) {
        return arrow::Status::TypeError(
            "LogTransform: column '" + value_col_ +
            "' has unsupported type '" + column->type()->ToString() + "'");
    }

    const uint64_t planned_rows =
        static_cast<uint64_t>(std::max<int64_t>(0, input->num_rows()));
    const auto transform_estimate = EstimateDenseMaterializationMemory(
        planned_rows, 2, static_cast<uint64_t>(sizeof(float)));
    ARROW_ASSIGN_OR_RAISE(
        auto preflight_estimate,
        EmitMaterializationMemoryPreflight(
            transform_estimate,
            "LogTransform",
            "planned_row_buffers",
            "Reduce input rows or transform a smaller materialized dataset.",
            GetMaterializationMemoryContext(),
            progress_callback_,
            SaturatingMaterializationItemCount(planned_rows, 2),
            0.15f));
    ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));

    ReportProgress(progress_callback_, "read_column",
                   "Reading column for log transform", 0.20f, 0,
                   planned_rows,
                   preflight_estimate.estimated_peak_bytes);

    int col_idx = input->schema()->GetFieldIndex(value_col_);

    std::vector<float> values;
    std::string bad_type;
    if (!ReadColumnAsFloat(
            column, values, bad_type, GetCancellationQuery())) {
        ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
        return arrow::Status::TypeError(
            "LogTransform: column '" + value_col_ +
            "' has unsupported type '" + bad_type + "'");
    }

    ReportProgress(progress_callback_, "transform",
                   "Applying log1p transform", 0.60f,
                   static_cast<uint64_t>(values.size()),
                   static_cast<uint64_t>(values.size()),
                    preflight_estimate.estimated_peak_bytes);
    for (size_t i = 0; i < values.size(); ++i) {
        if ((i & 1023) == 0) {
            ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
        }
        values[i] = std::log1p(values[i]);
    }

    ReportProgress(progress_callback_, "write_output",
                   "Writing transformed column", 0.90f,
                   static_cast<uint64_t>(values.size()),
                   static_cast<uint64_t>(values.size()),
                    preflight_estimate.estimated_peak_bytes);
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
                    preflight_estimate.estimated_peak_bytes);
    return out_table;
}

} // namespace cyxwiz
