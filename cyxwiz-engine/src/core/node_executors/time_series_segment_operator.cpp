#include "time_series_segment_operator.h"

#include "../materialization_memory_guard.h"
#include "../profiler_trace.h"

#include <arrow/api.h>
#include <arrow/builder.h>
#include <arrow/compute/api.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace cyxwiz {
namespace {

bool IsSupportedTimestampType(const std::shared_ptr<arrow::DataType>& type) {
    if (!type) return false;
    switch (type->id()) {
    case arrow::Type::TIMESTAMP:
    case arrow::Type::DATE32:
    case arrow::Type::DATE64:
    case arrow::Type::STRING:
    case arrow::Type::LARGE_STRING:
        return true;
    default:
        return false;
    }
}

void ReportProgress(const PipelineOperatorProgressCallback& callback,
                    std::string stage, std::string message, float progress,
                    uint64_t processed, uint64_t total,
                    uint64_t estimated_memory_bytes = 0) {
    if (!callback) return;
    PipelineOperatorProgress event;
    event.stage = std::move(stage);
    event.message = std::move(message);
    event.status = "running";
    event.progress = progress;
    event.processed_items = processed;
    event.total_items = total;
    event.estimated_memory_bytes = estimated_memory_bytes;
    callback(event);
}

} // namespace

bool TimeSeriesSegmentOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    timestamp_col_.clear();
    gap_threshold_seconds_ = 0.0;
    segment_col_ = "__segment_id";
    delta_col_ = "__time_delta_seconds";

    const auto timestamp = params.find("timestamp_col");
    if (timestamp == params.end() || timestamp->second.empty()) {
        error = "TimeSeriesSegment: timestamp_col is required";
        return false;
    }
    timestamp_col_ = timestamp->second;

    const auto threshold = params.find("gap_threshold_seconds");
    if (threshold == params.end() || threshold->second.empty()) {
        error = "TimeSeriesSegment: gap_threshold_seconds is required";
        return false;
    }
    try {
        size_t consumed = 0;
        gap_threshold_seconds_ = std::stod(threshold->second, &consumed);
        if (consumed != threshold->second.size()) {
            throw std::invalid_argument("trailing characters");
        }
    } catch (...) {
        error = "TimeSeriesSegment: invalid gap_threshold_seconds: " +
                threshold->second;
        return false;
    }
    const double max_seconds =
        static_cast<double>((std::numeric_limits<int64_t>::max)()) / 1000000.0;
    if (!std::isfinite(gap_threshold_seconds_) ||
        gap_threshold_seconds_ <= 0.0 ||
        gap_threshold_seconds_ > max_seconds) {
        error =
            "TimeSeriesSegment: gap_threshold_seconds must be finite and > 0";
        return false;
    }

    if (const auto segment = params.find("segment_col");
        segment != params.end() && !segment->second.empty()) {
        segment_col_ = segment->second;
    }
    if (const auto delta = params.find("delta_col");
        delta != params.end() && !delta->second.empty()) {
        delta_col_ = delta->second;
    }
    if (segment_col_ == delta_col_) {
        error = "TimeSeriesSegment: segment_col and delta_col must differ";
        return false;
    }
    if (segment_col_ == timestamp_col_ || delta_col_ == timestamp_col_) {
        error = "TimeSeriesSegment: output names must not replace timestamp_col";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Schema>>
TimeSeriesSegmentOperator::InferOutputSchema(
    const std::shared_ptr<arrow::Schema>& input_schema) {
    if (!input_schema) {
        return arrow::Status::Invalid(
            "TimeSeriesSegment: input schema is null");
    }
    if (input_schema->GetFieldIndex(segment_col_) >= 0) {
        return arrow::Status::Invalid(
            "TimeSeriesSegment: output segment column '", segment_col_,
            "' already exists");
    }
    if (input_schema->GetFieldIndex(delta_col_) >= 0) {
        return arrow::Status::Invalid(
            "TimeSeriesSegment: output delta column '", delta_col_,
            "' already exists");
    }
    auto fields = input_schema->fields();
    fields.push_back(arrow::field(segment_col_, arrow::int64(), false));
    fields.push_back(arrow::field(delta_col_, arrow::float64(), true));
    return arrow::schema(fields, input_schema->metadata());
}

arrow::Result<std::shared_ptr<arrow::Table>>
TimeSeriesSegmentOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz TimeSeriesSegment Materializer");
    if (!input) {
        return arrow::Status::Invalid(
            "TimeSeriesSegment: input table is null");
    }
    if (timestamp_col_.empty() || gap_threshold_seconds_ <= 0.0) {
        return arrow::Status::Invalid(
            "TimeSeriesSegment: Apply called before Configure succeeded");
    }
    if (input->num_rows() == 0) {
        return arrow::Status::Invalid(
            "TimeSeriesSegment: input has zero rows");
    }
    ARROW_RETURN_NOT_OK(InferOutputSchema(input->schema()).status());

    const auto timestamp_column = input->GetColumnByName(timestamp_col_);
    if (!timestamp_column) {
        return arrow::Status::KeyError(
            "TimeSeriesSegment: timestamp column '", timestamp_col_,
            "' not found");
    }
    if (!IsSupportedTimestampType(timestamp_column->type())) {
        return arrow::Status::TypeError(
            "TimeSeriesSegment: timestamp column '", timestamp_col_,
            "' has unsupported type '", timestamp_column->type()->ToString(),
            "' (expected timestamp, date, string, or large_string)");
    }

    const uint64_t rows = static_cast<uint64_t>(input->num_rows());
    const auto estimate = EstimateDenseMaterializationMemory(
        rows, 2, static_cast<uint64_t>(sizeof(int64_t)));
    const auto decision = EvaluateMaterializationMemory(
        estimate, GetMaterializationMemoryContext());
    std::ostringstream message;
    message << "TimeSeriesSegment memory preflight: risk="
            << MaterializationMemoryRiskName(decision.risk)
            << ", rows=" << rows << ", estimated_peak="
            << FormatMaterializationBytes(estimate.estimated_peak_bytes)
            << ". " << decision.reason;
    if (progress_callback_) {
        PipelineOperatorProgress event;
        event.stage = "TimeSeriesSegment memory preflight";
        event.message = message.str();
        event.status = MaterializationMemoryRiskToProgressStatus(decision.risk);
        event.progress = 0.03f;
        event.total_items = rows;
        event.estimated_memory_bytes = estimate.estimated_peak_bytes;
        event.memory_risk_level =
            MaterializationMemoryRiskName(decision.risk);
        progress_callback_(event);
    }
    if (decision.blocked) {
        return arrow::Status::CapacityError(
            "Materialization blocked: ", message.str());
    }

    ReportProgress(progress_callback_, "Parsing timestamps",
                   "Casting timestamp column to microsecond precision",
                   0.10f, 0, rows, estimate.estimated_peak_bytes);
    arrow::compute::CastOptions cast_options;
    cast_options.allow_time_truncate = false;
    const auto timestamp_type =
        arrow::timestamp(arrow::TimeUnit::MICRO);
    std::vector<std::shared_ptr<arrow::Array>> timestamp_chunks;
    timestamp_chunks.reserve(timestamp_column->num_chunks());
    for (int chunk_index = 0;
         chunk_index < timestamp_column->num_chunks(); ++chunk_index) {
        auto cast_result = arrow::compute::Cast(
            *timestamp_column->chunk(chunk_index),
            timestamp_type, cast_options);
        if (!cast_result.ok()) {
            return arrow::Status::Invalid(
                "TimeSeriesSegment: failed to parse timestamp column '",
                timestamp_col_, "': ",
                cast_result.status().ToString());
        }
        timestamp_chunks.push_back(cast_result.ValueOrDie());
    }

    const int64_t threshold_microseconds = static_cast<int64_t>(
        std::ceil(gap_threshold_seconds_ * 1000000.0));
    arrow::Int64Builder segment_builder(arrow::default_memory_pool());
    arrow::DoubleBuilder delta_builder(arrow::default_memory_pool());
    ARROW_RETURN_NOT_OK(segment_builder.Reserve(input->num_rows()));
    ARROW_RETURN_NOT_OK(delta_builder.Reserve(input->num_rows()));

    int64_t segment = 0;
    int64_t previous = 0;
    bool have_previous = false;
    int64_t row_index = 0;
    int64_t gap_count = 0;
    int64_t max_gap_microseconds = 0;
    for (const auto& chunk : timestamp_chunks) {
        const auto timestamps =
            std::static_pointer_cast<arrow::TimestampArray>(chunk);
        for (int64_t index = 0; index < timestamps->length(); ++index) {
            if (timestamps->IsNull(index)) {
                return arrow::Status::Invalid(
                    "TimeSeriesSegment: null timestamp at row ", row_index);
            }
            const int64_t current = timestamps->Value(index);
            if (!have_previous) {
                ARROW_RETURN_NOT_OK(segment_builder.Append(segment));
                ARROW_RETURN_NOT_OK(delta_builder.AppendNull());
                previous = current;
                have_previous = true;
                ++row_index;
                continue;
            }
            if (current <= previous) {
                return arrow::Status::Invalid(
                    "TimeSeriesSegment: timestamps must be strictly "
                    "increasing at row ", row_index);
            }
            const int64_t delta_microseconds = current - previous;
            max_gap_microseconds =
                (std::max)(max_gap_microseconds, delta_microseconds);
            if (delta_microseconds >= threshold_microseconds) {
                ++segment;
                ++gap_count;
            }
            ARROW_RETURN_NOT_OK(segment_builder.Append(segment));
            ARROW_RETURN_NOT_OK(delta_builder.Append(
                static_cast<double>(delta_microseconds) / 1000000.0));
            previous = current;
            ++row_index;
            if ((row_index % 65536) == 0 ||
                row_index == input->num_rows()) {
                const float progress =
                    0.20f + 0.70f * static_cast<float>(row_index) /
                                static_cast<float>(input->num_rows());
                ReportProgress(
                    progress_callback_, "Detecting timestamp gaps",
                    "Assigning continuous time-series segments",
                    progress, static_cast<uint64_t>(row_index), rows,
                    estimate.estimated_peak_bytes);
            }
        }
    }

    std::shared_ptr<arrow::Array> segment_array;
    std::shared_ptr<arrow::Array> delta_array;
    ARROW_RETURN_NOT_OK(segment_builder.Finish(&segment_array));
    ARROW_RETURN_NOT_OK(delta_builder.Finish(&delta_array));

    ARROW_ASSIGN_OR_RAISE(
        auto with_segments,
        input->AddColumn(
            input->num_columns(),
            arrow::field(segment_col_, arrow::int64(), false),
            std::make_shared<arrow::ChunkedArray>(segment_array)));
    ARROW_ASSIGN_OR_RAISE(
        auto output,
        with_segments->AddColumn(
            with_segments->num_columns(),
            arrow::field(delta_col_, arrow::float64(), true),
            std::make_shared<arrow::ChunkedArray>(delta_array)));

    spdlog::info(
        "TimeSeriesSegment: {} rows -> {} segments, {} gaps "
        "(timestamp_col='{}', threshold={}s, max_gap={}s)",
        input->num_rows(), segment + 1, gap_count, timestamp_col_,
        gap_threshold_seconds_,
        static_cast<double>(max_gap_microseconds) / 1000000.0);
    ReportProgress(progress_callback_, "Complete",
                   "Timestamp integrity and segmentation complete",
                   1.0f, rows, rows,
                   estimate.estimated_peak_bytes);
    return output;
}

} // namespace cyxwiz
