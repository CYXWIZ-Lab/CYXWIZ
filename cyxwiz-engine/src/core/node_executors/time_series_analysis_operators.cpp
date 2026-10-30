#include "time_series_analysis_operators.h"
#include "../materialization_memory_guard.h"
#include "../profiler_trace.h"
#include "../time_steps.h"
#include "feature_matrix_utils.h"
#include "ts_column_utils.h"

#include <cyxwiz/time_series.h>

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

void ReportProgress(const PipelineOperatorProgressCallback& callback,
                    std::string stage,
                    std::string message,
                    double progress,
                    uint64_t processed = 0,
                    uint64_t total = 0,
                    uint64_t memory = 0) {
    if (!callback) {
        return;
    }
    PipelineOperatorProgress event;
    event.stage = std::move(stage);
    event.message = std::move(message);
    event.status = "running";
    event.progress = static_cast<float>(progress);
    event.processed_items = processed;
    event.total_items = total;
    event.estimated_memory_bytes = memory;
    callback(event);
}

// ---- reading the signal

template <typename ArrayT>
void AppendAsDouble(const std::shared_ptr<arrow::Array>& chunk, std::vector<double>& out) {
    const auto& a = static_cast<const ArrayT&>(*chunk);
    for (int64_t i = 0; i < a.length(); ++i) out.push_back(a.IsNull(i) ? NAN : static_cast<double>(a.Value(i)));
}

// One numeric column as doubles, in table order. A missing value is refused:
// the analysis needs every row, and a node cannot drop rows without saying so.
arrow::Status ReadSignalAsDouble(const std::shared_ptr<arrow::Table>& input, const std::string& col_name, const std::string& op_name,
                                 std::vector<double>& out) {
    auto col = input->GetColumnByName(col_name);
    if (!col) return arrow::Status::KeyError(op_name + ": signal column '" + col_name + "' not found");
    out.clear();
    out.reserve(static_cast<size_t>(col->length()));
    for (const auto& chunk : col->chunks()) {
        switch (chunk->type_id()) {
            case arrow::Type::INT8: AppendAsDouble<arrow::Int8Array>(chunk, out); break;
            case arrow::Type::INT16: AppendAsDouble<arrow::Int16Array>(chunk, out); break;
            case arrow::Type::INT32: AppendAsDouble<arrow::Int32Array>(chunk, out); break;
            case arrow::Type::INT64: AppendAsDouble<arrow::Int64Array>(chunk, out); break;
            case arrow::Type::UINT8: AppendAsDouble<arrow::UInt8Array>(chunk, out); break;
            case arrow::Type::UINT16: AppendAsDouble<arrow::UInt16Array>(chunk, out); break;
            case arrow::Type::UINT32: AppendAsDouble<arrow::UInt32Array>(chunk, out); break;
            case arrow::Type::UINT64: AppendAsDouble<arrow::UInt64Array>(chunk, out); break;
            case arrow::Type::FLOAT: AppendAsDouble<arrow::FloatArray>(chunk, out); break;
            case arrow::Type::DOUBLE: AppendAsDouble<arrow::DoubleArray>(chunk, out); break;
            default:
                return arrow::Status::TypeError(op_name + ": signal column '" + col_name + "' must be numeric (got '" + chunk->type()->ToString() + "')");
        }
    }
    size_t missing = 0;
    for (const double v : out)
        if (!std::isfinite(v)) ++missing;
    if (missing > 0)
        return arrow::Status::Invalid(op_name + ": signal column '" + col_name + "' has " + std::to_string(missing) +
                                      " missing value(s); fill or drop those rows first (Fill Missing Values)");
    return arrow::Status::OK();
}

// ---- building columns

// A float64 array; a value that is not finite becomes null (classical
// decomposition leaves its edges empty, a forecast column is empty in-sample).
arrow::Result<std::shared_ptr<arrow::Array>> BuildDoubleArray(const std::vector<double>& values) {
    arrow::DoubleBuilder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(static_cast<int64_t>(values.size())));
    for (double v : values) {
        if (std::isfinite(v)) ARROW_RETURN_NOT_OK(builder.Append(v));
        else ARROW_RETURN_NOT_OK(builder.AppendNull());
    }
    std::shared_ptr<arrow::Array> array;
    ARROW_RETURN_NOT_OK(builder.Finish(&array));
    return array;
}

arrow::Result<std::shared_ptr<arrow::Table>> AppendF64Column(const std::shared_ptr<arrow::Table>& table, const std::string& name,
                                                             const std::vector<double>& values) {
    ARROW_ASSIGN_OR_RAISE(auto array, BuildDoubleArray(values));
    return table->AddColumn(table->num_columns(), arrow::field(name, arrow::float64()), std::make_shared<arrow::ChunkedArray>(array));
}

arrow::Result<std::shared_ptr<arrow::Array>> BuildInt32Array(const std::vector<int32_t>& values) {
    arrow::Int32Builder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(static_cast<int64_t>(values.size())));
    for (int32_t v : values) ARROW_RETURN_NOT_OK(builder.Append(v));
    std::shared_ptr<arrow::Array> array;
    ARROW_RETURN_NOT_OK(builder.Finish(&array));
    return array;
}

// An int32 array with nulls where the value is below zero (a lag count that does not apply).
arrow::Result<std::shared_ptr<arrow::Array>> BuildOptionalInt32Array(const std::vector<int32_t>& values) {
    arrow::Int32Builder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(static_cast<int64_t>(values.size())));
    for (int32_t v : values) {
        if (v >= 0) ARROW_RETURN_NOT_OK(builder.Append(v));
        else ARROW_RETURN_NOT_OK(builder.AppendNull());
    }
    std::shared_ptr<arrow::Array> array;
    ARROW_RETURN_NOT_OK(builder.Finish(&array));
    return array;
}

arrow::Result<std::shared_ptr<arrow::Array>> BuildBoolArray(const std::vector<bool>& values) {
    arrow::BooleanBuilder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(static_cast<int64_t>(values.size())));
    for (bool v : values) ARROW_RETURN_NOT_OK(builder.Append(v));
    std::shared_ptr<arrow::Array> array;
    ARROW_RETURN_NOT_OK(builder.Finish(&array));
    return array;
}

arrow::Result<std::shared_ptr<arrow::Array>> BuildStringArray(const std::vector<std::string>& values) {
    arrow::StringBuilder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(static_cast<int64_t>(values.size())));
    for (const auto& v : values) ARROW_RETURN_NOT_OK(builder.Append(v));
    std::shared_ptr<arrow::Array> array;
    ARROW_RETURN_NOT_OK(builder.Finish(&array));
    return array;
}

arrow::Result<std::vector<int8_t>> ReadPartitionMetadataAsInt8(
    const std::shared_ptr<arrow::ChunkedArray>& column,
    const std::string& op_name) {
    if (!column) {
        return arrow::Status::KeyError(
            op_name + ": __partition__ column is missing");
    }

    std::vector<int8_t> values;
    values.reserve(static_cast<size_t>(column->length()));
    const auto append_value = [&](int64_t value) -> arrow::Status {
        if (value < std::numeric_limits<int8_t>::min() ||
            value > std::numeric_limits<int8_t>::max()) {
            return arrow::Status::Invalid(
                op_name + ": __partition__ value " +
                std::to_string(value) + " is outside int8 range");
        }
        values.push_back(static_cast<int8_t>(value));
        return arrow::Status::OK();
    };

    for (const auto& chunk : column->chunks()) {
        for (int64_t row = 0; row < chunk->length(); ++row) {
            if (chunk->IsNull(row)) {
                return arrow::Status::Invalid(
                    op_name + ": __partition__ contains nulls");
            }
            switch (chunk->type_id()) {
            case arrow::Type::INT8:
                ARROW_RETURN_NOT_OK(append_value(
                    std::static_pointer_cast<arrow::Int8Array>(chunk)->Value(row)));
                break;
            case arrow::Type::INT16:
                ARROW_RETURN_NOT_OK(append_value(
                    std::static_pointer_cast<arrow::Int16Array>(chunk)->Value(row)));
                break;
            case arrow::Type::INT32:
                ARROW_RETURN_NOT_OK(append_value(
                    std::static_pointer_cast<arrow::Int32Array>(chunk)->Value(row)));
                break;
            case arrow::Type::INT64:
                ARROW_RETURN_NOT_OK(append_value(
                    std::static_pointer_cast<arrow::Int64Array>(chunk)->Value(row)));
                break;
            case arrow::Type::UINT8:
                ARROW_RETURN_NOT_OK(append_value(
                    std::static_pointer_cast<arrow::UInt8Array>(chunk)->Value(row)));
                break;
            case arrow::Type::UINT16:
                ARROW_RETURN_NOT_OK(append_value(
                    std::static_pointer_cast<arrow::UInt16Array>(chunk)->Value(row)));
                break;
            case arrow::Type::UINT32: {
                const uint32_t value =
                    std::static_pointer_cast<arrow::UInt32Array>(chunk)->Value(row);
                if (value > static_cast<uint32_t>(
                                std::numeric_limits<int8_t>::max())) {
                    return arrow::Status::Invalid(
                        op_name + ": __partition__ value is outside int8 range");
                }
                ARROW_RETURN_NOT_OK(append_value(static_cast<int64_t>(value)));
                break;
            }
            case arrow::Type::UINT64: {
                const uint64_t value =
                    std::static_pointer_cast<arrow::UInt64Array>(chunk)->Value(row);
                if (value > static_cast<uint64_t>(
                                std::numeric_limits<int8_t>::max())) {
                    return arrow::Status::Invalid(
                        op_name + ": __partition__ value is outside int8 range");
                }
                ARROW_RETURN_NOT_OK(append_value(static_cast<int64_t>(value)));
                break;
            }
            default:
                return arrow::Status::TypeError(
                    op_name + ": __partition__ must be an integer column "
                    "(got '" + chunk->type()->ToString() + "')");
            }
        }
    }
    return values;
}

// ---- parameters

bool ParseSignalColumn(const std::map<std::string, std::string>& params, const std::string& op_name, std::string& signal_col, std::string& error) {
    auto s = params.find("signal_col");
    if (s == params.end() || s->second.empty()) {
        error = op_name + ": 'signal_col' parameter is required";
        return false;
    }
    signal_col = s->second;
    return true;
}

bool ParseIntOptional(const std::map<std::string, std::string>& params, const std::string& key, int& out, const std::string& op_name, std::string& error) {
    auto it = params.find(key);
    if (it == params.end() || it->second.empty()) return true;
    try {
        size_t consumed = 0;
        out = std::stoi(it->second, &consumed);
        if (consumed != it->second.size()) throw std::invalid_argument("trailing characters");
        return true;
    } catch (...) {
        error = op_name + ": '" + key + "' is not a valid integer: " + it->second;
        return false;
    }
}

bool ParseDoubleOptional(const std::map<std::string, std::string>& params, const std::string& key, double& out, const std::string& op_name, std::string& error) {
    auto it = params.find(key);
    if (it == params.end() || it->second.empty()) return true;
    try {
        out = std::stod(it->second);
        return true;
    } catch (...) {
        error = op_name + ": '" + key + "' is not a valid float: " + it->second;
        return false;
    }
}

bool ParseBoolOptional(const std::map<std::string, std::string>& params, const std::string& key, bool& out, const std::string& op_name, std::string& error) {
    auto it = params.find(key);
    if (it == params.end() || it->second.empty()) return true;
    if (it->second == "true") out = true;
    else if (it->second == "false") out = false;
    else {
        error = op_name + ": '" + key + "' must be 'true' or 'false' (got '" + it->second + "')";
        return false;
    }
    return true;
}

bool ParseChoice(const std::map<std::string, std::string>& params, const std::string& key, std::string& out, const std::vector<std::string>& allowed,
                 const std::string& op_name, std::string& error) {
    auto it = params.find(key);
    if (it == params.end() || it->second.empty()) return true;
    out = NormalizeTimeSeriesParameterChoice(it->second);
    if (std::find(allowed.begin(), allowed.end(), out) != allowed.end()) return true;
    std::string list;
    for (const auto& a : allowed) list += (list.empty() ? "'" : " / '") + a + "'";
    error = op_name + ": '" + key + "' must be " + list + " (got '" + out + "')";
    return false;
}

// A string parameter ("" when absent).
std::string ParseText(const std::map<std::string, std::string>& params, const std::string& key) {
    auto it = params.find(key);
    return it == params.end() ? std::string() : it->second;
}

bool ParseHorizonAndLevel(const std::map<std::string, std::string>& params, int& horizon, double& level, const std::string& op_name, std::string& error) {
    if (!ParseIntOptional(params, "horizon", horizon, op_name, error)) return false;
    if (horizon < 0) {
        error = op_name + ": 'horizon' must be >= 0 (got " + std::to_string(horizon) + ")";
        return false;
    }
    if (!ParseDoubleOptional(params, "level", level, op_name, error)) return false;
    if (!(level > 0.5 && level < 1.0)) {
        error = op_name + ": 'level' must be between 0.5 and 1 (got " + std::to_string(level) + ")";
        return false;
    }
    return true;
}

// The period to use: the typed one, else the strongest period of the
// detrended periodogram when a season of that length stands out.
arrow::Result<int> PeriodOf(const std::vector<double>& signal, int typed, const std::string& op_name) {
    if (typed >= 2) return typed;
    const SeasonalityResult season = TimeSeries::DetectSeasonality(signal);
    if (!season.success || !season.has_seasonality || season.detected_period < 2 ||
        static_cast<size_t>(2 * season.detected_period) > signal.size())
        return arrow::Status::Invalid(op_name + ": no seasonal period stands out in the signal; set 'period'");
    return season.detected_period;
}

// ---- future rows

// The time column as seconds since 1970 (dates) or numbers, when it can be
// continued: a date, timestamp or numeric column without gaps in time order.
bool ReadTimes(const std::shared_ptr<arrow::ChunkedArray>& col, std::vector<double>& t, series::TimeKind& kind, double& unit_seconds) {
    if (!col || col->null_count() > 0 || col->length() < 2) return false;
    unit_seconds = 1.0;
    kind = series::TimeKind::Numbers;
    for (const auto& chunk : col->chunks()) {
        switch (chunk->type_id()) {
            case arrow::Type::DATE32:
                kind = series::TimeKind::Dates;
                unit_seconds = series::kDaySeconds;
                AppendAsDouble<arrow::Date32Array>(chunk, t);
                break;
            case arrow::Type::DATE64:
                kind = series::TimeKind::Dates;
                unit_seconds = 1e-3;
                AppendAsDouble<arrow::Date64Array>(chunk, t);
                break;
            case arrow::Type::TIMESTAMP: {
                kind = series::TimeKind::Dates;
                switch (static_cast<const arrow::TimestampType&>(*chunk->type()).unit()) {
                    case arrow::TimeUnit::SECOND: unit_seconds = 1.0; break;
                    case arrow::TimeUnit::MILLI: unit_seconds = 1e-3; break;
                    case arrow::TimeUnit::MICRO: unit_seconds = 1e-6; break;
                    case arrow::TimeUnit::NANO: unit_seconds = 1e-9; break;
                }
                AppendAsDouble<arrow::TimestampArray>(chunk, t);
                break;
            }
            case arrow::Type::INT32: AppendAsDouble<arrow::Int32Array>(chunk, t); break;
            case arrow::Type::INT64: AppendAsDouble<arrow::Int64Array>(chunk, t); break;
            case arrow::Type::FLOAT: AppendAsDouble<arrow::FloatArray>(chunk, t); break;
            case arrow::Type::DOUBLE: AppendAsDouble<arrow::DoubleArray>(chunk, t); break;
            default: return false;
        }
    }
    if (kind == series::TimeKind::Dates)
        for (double& v : t) v *= unit_seconds;
    return std::is_sorted(t.begin(), t.end());
}

// The time column's next `horizon` values in its own type (calendar months
// for a monthly step), or nulls when the column cannot be continued.
arrow::Result<std::shared_ptr<arrow::Array>> FutureTimeArray(const std::shared_ptr<arrow::ChunkedArray>& col, int horizon) {
    std::vector<double> t;
    series::TimeKind kind = series::TimeKind::Numbers;
    double unit = 1.0;
    if (!ReadTimes(col, t, kind, unit)) return arrow::MakeArrayOfNull(col->type(), horizon);
    const series::Step step = series::StepOf(t, kind);
    if (!(step.seconds > 0.0)) return arrow::MakeArrayOfNull(col->type(), horizon);
    const std::vector<double> future = series::FutureTimes(t, step, kind, horizon);
    std::shared_ptr<arrow::Array> array;
    switch (col->type()->id()) {
        case arrow::Type::DATE32: {
            arrow::Date32Builder b;
            for (double v : future) ARROW_RETURN_NOT_OK(b.Append(static_cast<int32_t>(std::llround(v / series::kDaySeconds))));
            ARROW_RETURN_NOT_OK(b.Finish(&array));
            break;
        }
        case arrow::Type::DATE64: {
            arrow::Date64Builder b;
            for (double v : future) ARROW_RETURN_NOT_OK(b.Append(std::llround(v * 1e3)));
            ARROW_RETURN_NOT_OK(b.Finish(&array));
            break;
        }
        case arrow::Type::TIMESTAMP: {
            arrow::TimestampBuilder b(col->type(), arrow::default_memory_pool());
            for (double v : future) ARROW_RETURN_NOT_OK(b.Append(std::llround(v / unit)));
            ARROW_RETURN_NOT_OK(b.Finish(&array));
            break;
        }
        case arrow::Type::INT32: {
            arrow::Int32Builder b;
            for (double v : future) ARROW_RETURN_NOT_OK(b.Append(static_cast<int32_t>(std::llround(v))));
            ARROW_RETURN_NOT_OK(b.Finish(&array));
            break;
        }
        case arrow::Type::INT64: {
            arrow::Int64Builder b;
            for (double v : future) ARROW_RETURN_NOT_OK(b.Append(std::llround(v)));
            ARROW_RETURN_NOT_OK(b.Finish(&array));
            break;
        }
        case arrow::Type::FLOAT: {
            arrow::FloatBuilder b;
            for (double v : future) ARROW_RETURN_NOT_OK(b.Append(static_cast<float>(v)));
            ARROW_RETURN_NOT_OK(b.Finish(&array));
            break;
        }
        case arrow::Type::DOUBLE: {
            arrow::DoubleBuilder b;
            for (double v : future) ARROW_RETURN_NOT_OK(b.Append(v));
            ARROW_RETURN_NOT_OK(b.Finish(&array));
            break;
        }
        default: return arrow::MakeArrayOfNull(col->type(), horizon);
    }
    return array;
}

std::shared_ptr<arrow::Schema> NullableSchema(const std::shared_ptr<arrow::Schema>& schema) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (const auto& f : schema->fields()) fields.push_back(f->WithNullable(true));
    return arrow::schema(fields, schema->metadata());
}

// The input with `horizon` rows added at the end: empty, except the time
// column, which continues its step.
arrow::Result<std::shared_ptr<arrow::Table>> WithFutureRows(const std::shared_ptr<arrow::Table>& input, int horizon, const std::string& time_col) {
    if (horizon <= 0) return input;
    std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;
    for (int c = 0; c < input->num_columns(); ++c) {
        const auto& col = input->column(c);
        std::shared_ptr<arrow::Array> tail;
        if (!time_col.empty() && input->field(c)->name() == time_col) {
            ARROW_ASSIGN_OR_RAISE(tail, FutureTimeArray(col, horizon));
        } else {
            ARROW_ASSIGN_OR_RAISE(tail, arrow::MakeArrayOfNull(col->type(), horizon));
        }
        auto chunks = col->chunks();
        chunks.push_back(tail);
        columns.push_back(std::make_shared<arrow::ChunkedArray>(chunks, col->type()));
    }
    return arrow::Table::Make(NullableSchema(input->schema()), columns, input->num_rows() + horizon);
}

// fitted and residual over the input rows; with a horizon, the future rows
// and forecast / low / high over them.
arrow::Result<std::shared_ptr<arrow::Table>> WithForecast(const std::shared_ptr<arrow::Table>& input, const std::vector<double>& signal,
                                                         const ForecastResult& result, int horizon, const std::string& time_col, const std::string& op_name) {
    const size_t n = signal.size();
    if (result.fitted_values.size() != n)
        return arrow::Status::ExecutionError(op_name + ": backend returned " + std::to_string(result.fitted_values.size()) + " fitted values, expected " + std::to_string(n));
    if (horizon > 0 && static_cast<int>(result.forecast.size()) != horizon)
        return arrow::Status::ExecutionError(op_name + ": backend returned " + std::to_string(result.forecast.size()) + " forecasts, expected " + std::to_string(horizon));
    const size_t rows = n + static_cast<size_t>(horizon);
    std::vector<double> fitted(rows, NAN), residual(rows, NAN), forecast(rows, NAN), low(rows, NAN), high(rows, NAN);
    for (size_t i = 0; i < n; ++i) {
        fitted[i] = result.fitted_values[i];
        residual[i] = std::isfinite(fitted[i]) ? signal[i] - fitted[i] : NAN;
    }
    for (size_t h = 0; h < static_cast<size_t>(horizon); ++h) {
        forecast[n + h] = result.forecast[h];
        if (h < result.lower_bound.size()) low[n + h] = result.lower_bound[h];
        if (h < result.upper_bound.size()) high[n + h] = result.upper_bound[h];
    }
    ARROW_ASSIGN_OR_RAISE(auto out, WithFutureRows(input, horizon, time_col));
    ARROW_ASSIGN_OR_RAISE(out, AppendF64Column(out, "fitted", fitted));
    ARROW_ASSIGN_OR_RAISE(out, AppendF64Column(out, "residual", residual));
    if (horizon > 0) {
        ARROW_ASSIGN_OR_RAISE(out, AppendF64Column(out, "forecast", forecast));
        ARROW_ASSIGN_OR_RAISE(out, AppendF64Column(out, "low", low));
        ARROW_ASSIGN_OR_RAISE(out, AppendF64Column(out, "high", high));
    }
    return out;
}

arrow::Result<std::shared_ptr<arrow::Schema>> ForecastSchema(const std::shared_ptr<arrow::Schema>& input_schema, int horizon, const std::string& op_name) {
    if (!input_schema) return arrow::Status::Invalid(op_name + ": input schema is null");
    auto fields = NullableSchema(input_schema)->fields();
    fields.push_back(arrow::field("fitted", arrow::float64()));
    fields.push_back(arrow::field("residual", arrow::float64()));
    if (horizon > 0) {
        fields.push_back(arrow::field("forecast", arrow::float64()));
        fields.push_back(arrow::field("low", arrow::float64()));
        fields.push_back(arrow::field("high", arrow::float64()));
    }
    return arrow::schema(fields, input_schema->metadata());
}

std::vector<bool> MarkSignificant(int size, const std::vector<int>& lags) {
    std::vector<bool> out(static_cast<size_t>(size), false);
    for (int lag : lags) {
        if (lag >= 0 && lag < size) {
            out[static_cast<size_t>(lag)] = true;
        }
    }
    return out;
}

// ---- memory preflight

std::string BuildTimeSeriesAnalysisMemoryPreflightMessage(
    const std::string& op_name,
    const MaterializationMemoryEstimate& estimate,
    const MaterializationMemoryDecision& decision,
    const std::string& suggestion) {
    std::ostringstream ss;
    ss << op_name << " memory preflight: risk="
       << MaterializationMemoryRiskName(decision.risk)
       << ", samples=" << estimate.rows
       << ", planned_columns=" << estimate.output_features
       << ", raw=" << FormatMaterializationBytes(estimate.raw_output_bytes)
       << ", estimated_peak="
       << FormatMaterializationBytes(estimate.estimated_peak_bytes)
       << ", available="
       << FormatMaterializationBytes(decision.available_bytes)
       << ", safe_budget="
       << FormatMaterializationBytes(decision.safe_budget_bytes)
       << ". " << decision.reason
       << ". Suggestion: " << suggestion;
    return ss.str();
}

arrow::Result<MaterializationMemoryEstimate> EmitSignalAnalysisMemoryPreflight(
    const std::shared_ptr<arrow::Table>& input,
    const std::string& signal_col,
    const std::string& op_name,
    uint64_t planned_columns,
    const std::string& suggestion,
    const MaterializationMemoryContext& memory_context,
    const PipelineOperatorProgressCallback& callback,
    uint64_t& planned_samples) {
    auto signal_column = input->GetColumnByName(signal_col);
    if (!signal_column) {
        return arrow::Status::KeyError(
            op_name + ": signal column '" + signal_col + "' not found");
    }
    if (!IsNumericChunked(signal_column)) {
        std::string got = signal_column->num_chunks() > 0
            ? signal_column->chunk(0)->type()->ToString()
            : "<empty>";
        return arrow::Status::TypeError(
            op_name + ": signal column '" + signal_col +
            "' must be numeric (got '" + got + "')");
    }

    planned_samples =
        static_cast<uint64_t>(std::max<int64_t>(0, signal_column->length()));
    if (planned_samples == 0) {
        return arrow::Status::Invalid(op_name + ": signal column is empty");
    }

    const auto estimate = EstimateDenseMaterializationMemory(
        planned_samples, planned_columns, static_cast<uint64_t>(sizeof(double)));
    const auto decision = EvaluateMaterializationMemory(
        estimate, memory_context);
    const std::string preflight_message =
        BuildTimeSeriesAnalysisMemoryPreflightMessage(
            op_name, estimate, decision, suggestion);

    uint64_t planned_cells = 0;
    if (!CheckedMulU64(planned_samples, planned_columns, planned_cells)) {
        planned_cells = (std::numeric_limits<uint64_t>::max)();
    }

    if (callback) {
        PipelineOperatorProgress event;
        event.stage = op_name + " memory preflight";
        event.message = preflight_message;
        event.status = MaterializationMemoryRiskToProgressStatus(decision.risk);
        event.progress = 0.03f;
        event.processed_items = 0;
        event.total_items = planned_cells;
        event.estimated_memory_bytes = estimate.estimated_peak_bytes;
        event.memory_risk_level = MaterializationMemoryRiskName(decision.risk);
        callback(event);
    }
    if (decision.blocked) {
        return arrow::Status::CapacityError(
            "Materialization blocked: " + preflight_message);
    }
    return estimate;
}

// Preflight, then the signal.
arrow::Result<std::vector<double>> ReadSignal(const std::shared_ptr<arrow::Table>& input, const std::string& signal_col, const std::string& op_name,
                                              uint64_t planned_columns, const std::string& suggestion, const MaterializationMemoryContext& memory,
                                              const PipelineOperatorProgressCallback& callback) {
    if (!input) return arrow::Status::Invalid(op_name + ": input is null");
    uint64_t planned_samples = 0;
    ARROW_ASSIGN_OR_RAISE(auto estimate, EmitSignalAnalysisMemoryPreflight(input, signal_col, op_name, planned_columns, suggestion, memory, callback, planned_samples));
    ReportProgress(callback, "Reading signal", "Reading " + op_name + " signal column '" + signal_col + "'", 0.10, 0, planned_samples, estimate.estimated_peak_bytes);
    std::vector<double> signal;
    ARROW_RETURN_NOT_OK(ReadSignalAsDouble(input, signal_col, op_name, signal));
    return signal;
}

} // namespace

// ============================================================================
// TimeSeriesDecompositionOperator
// ============================================================================

bool TimeSeriesDecompositionOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    signal_col_.clear();
    period_ = -1;
    method_ = "additive";
    algorithm_ = "stl";
    seasonal_window_ = 7;
    robust_ = false;

    if (!ParseSignalColumn(params, GetName(), signal_col_, error)) return false;
    if (!ParseIntOptional(params, "period", period_, GetName(), error)) return false;
    if (period_ != -1 && period_ < 2) {
        error = GetName() + ": 'period' must be >= 2, or -1 to find it (got " + std::to_string(period_) + ")";
        return false;
    }
    if (!ParseChoice(params, "method", method_, {"additive", "multiplicative"}, GetName(), error)) return false;
    if (!ParseChoice(params, "algorithm", algorithm_, {"stl", "classical"}, GetName(), error)) return false;
    if (!ParseIntOptional(params, "seasonal_window", seasonal_window_, GetName(), error)) return false;
    if (seasonal_window_ < 3) {
        error = GetName() + ": 'seasonal_window' must be >= 3 (got " + std::to_string(seasonal_window_) + ")";
        return false;
    }
    if (!ParseBoolOptional(params, "robust", robust_, GetName(), error)) return false;
    return true;
}

arrow::Result<std::shared_ptr<arrow::Schema>> TimeSeriesDecompositionOperator::InferOutputSchema(const std::shared_ptr<arrow::Schema>& input_schema) {
    if (!input_schema) return arrow::Status::Invalid(GetName() + ": input schema is null");
    auto fields = input_schema->fields();
    fields.push_back(arrow::field("trend", arrow::float64()));
    fields.push_back(arrow::field("seasonal", arrow::float64()));
    fields.push_back(arrow::field("residual", arrow::float64()));
    return arrow::schema(fields, input_schema->metadata());
}

arrow::Result<std::shared_ptr<arrow::Table>> TimeSeriesDecompositionOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz TimeSeriesDecomposition Materializer");
    ARROW_ASSIGN_OR_RAISE(const std::vector<double> signal,
                          ReadSignal(input, signal_col_, GetName(), 4, "reduce signal rows, or decompose a sampled / windowed signal first.",
                                     GetMaterializationMemoryContext(), progress_callback_));
    ARROW_ASSIGN_OR_RAISE(const int period, PeriodOf(signal, period_, GetName()));
    if (static_cast<int>(signal.size()) < 2 * period)
        return arrow::Status::Invalid(GetName() + ": need at least 2 * period (" + std::to_string(2 * period) + ") rows, got " + std::to_string(signal.size()));

    ReportProgress(progress_callback_, "Decomposing signal", "Running " + algorithm_ + " decomposition with period " + std::to_string(period), 0.45, 0,
                   static_cast<uint64_t>(signal.size()));
    const DecompositionResult result = algorithm_ == "stl" ? TimeSeries::STLDecompose(signal, period, seasonal_window_, -1, -1, robust_)
                                                            : TimeSeries::Decompose(signal, period, method_);
    if (!result.success) return arrow::Status::ExecutionError(GetName() + ": " + algorithm_ + " decomposition failed: " + result.error_message);

    ReportProgress(progress_callback_, "Appending columns", "Appending trend, seasonal and residual columns", 0.85, static_cast<uint64_t>(signal.size()),
                   static_cast<uint64_t>(signal.size()));
    auto out = input;
    ARROW_ASSIGN_OR_RAISE(out, AppendF64Column(out, "trend", result.trend));
    ARROW_ASSIGN_OR_RAISE(out, AppendF64Column(out, "seasonal", result.seasonal));
    ARROW_ASSIGN_OR_RAISE(out, AppendF64Column(out, "residual", result.residual));

    spdlog::info("{}: n={} period={}{} algorithm={} trend_strength={:.4f} seasonal_strength={:.4f}", GetName(), signal.size(), period,
                 period_ < 2 ? " (found)" : "", algorithm_ == "stl" ? "stl" : method_, result.trend_strength, result.seasonal_strength);
    ReportProgress(progress_callback_, "Complete", "TimeSeriesDecomposition materialization complete", 1.0, static_cast<uint64_t>(signal.size()),
                   static_cast<uint64_t>(signal.size()));
    return out;
}

// ============================================================================
// ARIMAOperator
// ============================================================================

bool ARIMAOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    signal_col_.clear();
    p_ = d_ = q_ = -1;
    horizon_ = 12;
    level_ = 0.95;
    time_col_.clear();

    if (!ParseSignalColumn(params, GetName(), signal_col_, error)) return false;
    if (!ParseIntOptional(params, "p", p_, GetName(), error)) return false;
    if (!ParseIntOptional(params, "d", d_, GetName(), error)) return false;
    if (!ParseIntOptional(params, "q", q_, GetName(), error)) return false;
    for (const auto [name, value] : {std::pair{"p", p_}, std::pair{"d", d_}, std::pair{"q", q_}})
        if (value < -1 || value > 5) {
            error = GetName() + ": '" + name + "' must be between 0 and 5, or -1 to choose it (got " + std::to_string(value) + ")";
            return false;
        }
    if (!ParseHorizonAndLevel(params, horizon_, level_, GetName(), error)) return false;
    time_col_ = ParseText(params, "time_col");
    return true;
}

arrow::Result<std::shared_ptr<arrow::Schema>> ARIMAOperator::InferOutputSchema(const std::shared_ptr<arrow::Schema>& input_schema) {
    return ForecastSchema(input_schema, horizon_, GetName());
}

arrow::Result<std::shared_ptr<arrow::Table>> ARIMAOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz ARIMA Materializer");
    ARROW_ASSIGN_OR_RAISE(const std::vector<double> signal,
                          ReadSignal(input, signal_col_, GetName(), 5, "reduce signal rows, or fit ARIMA on a sampled / windowed signal first.",
                                     GetMaterializationMemoryContext(), progress_callback_));
    ReportProgress(progress_callback_, "Fitting model", p_ < 0 || q_ < 0 ? "Choosing the ARIMA orders by AIC" : "Fitting the ARIMA model", 0.45, 0,
                   static_cast<uint64_t>(signal.size()));
    const ForecastResult result = TimeSeries::ARIMA(signal, horizon_, p_, d_, q_, level_);
    if (!result.success) return arrow::Status::ExecutionError(GetName() + ": fit failed: " + result.error_message);

    ReportProgress(progress_callback_, "Appending columns", "Appending fitted, residual and forecast columns", 0.85, static_cast<uint64_t>(signal.size()),
                   static_cast<uint64_t>(signal.size()));
    ARROW_ASSIGN_OR_RAISE(auto out, WithForecast(input, signal, result, horizon_, time_col_, GetName()));
    spdlog::info("{}: n={} {} horizon={} level={} RMSE={:.4f} AIC={:.4f}", GetName(), signal.size(), result.method, horizon_, level_, result.rmse, result.aic);
    ReportProgress(progress_callback_, "Complete", "ARIMA materialization complete", 1.0, static_cast<uint64_t>(out->num_rows()), static_cast<uint64_t>(out->num_rows()));
    return out;
}

// ============================================================================
// ExponentialSmoothingOperator
// ============================================================================

bool ExponentialSmoothingOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    signal_col_.clear();
    method_ = "holt_winters";
    seasonal_ = "additive";
    alpha_ = beta_ = gamma_ = -1.0;
    period_ = -1;
    damped_ = false;
    horizon_ = 12;
    level_ = 0.95;
    time_col_.clear();

    if (!ParseSignalColumn(params, GetName(), signal_col_, error)) return false;
    if (!ParseChoice(params, "method", method_, {"simple", "holt", "holt_winters"}, GetName(), error)) return false;
    if (!ParseChoice(params, "seasonal", seasonal_, {"additive", "multiplicative"}, GetName(), error)) return false;
    if (!ParseDoubleOptional(params, "alpha", alpha_, GetName(), error)) return false;
    if (!ParseDoubleOptional(params, "beta", beta_, GetName(), error)) return false;
    if (!ParseDoubleOptional(params, "gamma", gamma_, GetName(), error)) return false;
    for (const auto [name, value] : {std::pair{"alpha", alpha_}, std::pair{"beta", beta_}, std::pair{"gamma", gamma_}})
        if (value != -1.0 && !(value >= 0.0 && value <= 1.0)) {
            error = GetName() + ": '" + name + "' must be between 0 and 1, or -1 to estimate it (got " + std::to_string(value) + ")";
            return false;
        }
    if (!ParseIntOptional(params, "period", period_, GetName(), error)) return false;
    if (!ParseBoolOptional(params, "damped", damped_, GetName(), error)) return false;
    if (method_ == "holt_winters" && period_ < 2 && period_ != -1) {
        error = GetName() + ": holt_winters requires 'period' >= 2 (or -1 to find it)";
        return false;
    }
    if (!ParseHorizonAndLevel(params, horizon_, level_, GetName(), error)) return false;
    time_col_ = ParseText(params, "time_col");
    return true;
}

arrow::Result<std::shared_ptr<arrow::Schema>> ExponentialSmoothingOperator::InferOutputSchema(const std::shared_ptr<arrow::Schema>& input_schema) {
    return ForecastSchema(input_schema, horizon_, GetName());
}

arrow::Result<std::shared_ptr<arrow::Table>> ExponentialSmoothingOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz ExponentialSmoothing Materializer");
    ARROW_ASSIGN_OR_RAISE(const std::vector<double> signal,
                          ReadSignal(input, signal_col_, GetName(), 5, "reduce signal rows, or fit the smoothing on a sampled / windowed signal first.",
                                     GetMaterializationMemoryContext(), progress_callback_));
    ReportProgress(progress_callback_, "Fitting model", "Fitting " + method_ + " exponential smoothing", 0.45, 0, static_cast<uint64_t>(signal.size()));
    ForecastResult result;
    if (method_ == "simple") {
        result = TimeSeries::SimpleES(signal, horizon_, alpha_, level_);
    } else if (method_ == "holt") {
        result = TimeSeries::HoltLinear(signal, horizon_, alpha_, beta_, damped_, level_);
    } else {
        ARROW_ASSIGN_OR_RAISE(const int period, PeriodOf(signal, period_, GetName()));
        result = TimeSeries::HoltWinters(signal, horizon_, period, seasonal_, alpha_, beta_, gamma_, level_);
    }
    if (!result.success) return arrow::Status::ExecutionError(GetName() + ": " + method_ + " fit failed: " + result.error_message);

    ReportProgress(progress_callback_, "Appending columns", "Appending fitted, residual and forecast columns", 0.85, static_cast<uint64_t>(signal.size()),
                   static_cast<uint64_t>(signal.size()));
    ARROW_ASSIGN_OR_RAISE(auto out, WithForecast(input, signal, result, horizon_, time_col_, GetName()));
    spdlog::info("{}: n={} {} horizon={} level={} RMSE={:.4f} AIC={:.4f}", GetName(), signal.size(), result.method, horizon_, level_, result.rmse, result.aic);
    ReportProgress(progress_callback_, "Complete", "ExponentialSmoothing materialization complete", 1.0, static_cast<uint64_t>(out->num_rows()),
                   static_cast<uint64_t>(out->num_rows()));
    return out;
}

// ============================================================================
// ACFOperator / PACFOperator
// ============================================================================

namespace {

bool ConfigureCorrelogram(const std::map<std::string, std::string>& params, const std::string& op_name, std::string& signal_col, int& max_lag, double& confidence,
                          std::string& error) {
    signal_col.clear();
    max_lag = -1;
    confidence = 0.95;
    if (!ParseSignalColumn(params, op_name, signal_col, error)) return false;
    if (!ParseIntOptional(params, "max_lag", max_lag, op_name, error)) return false;
    if (!ParseIntOptional(params, "lags", max_lag, op_name, error)) return false;
    if (max_lag == 0 || max_lag < -1) {
        error = op_name + ": max_lag must be -1 or >= 1";
        return false;
    }
    if (!ParseDoubleOptional(params, "confidence", confidence, op_name, error)) return false;
    if (!(confidence > 0.5 && confidence < 1.0)) {
        error = op_name + ": 'confidence' must be between 0.5 and 1 (got " + std::to_string(confidence) + ")";
        return false;
    }
    return true;
}

std::shared_ptr<arrow::Schema> CorrelogramSchema(bool acf) {
    std::vector<std::shared_ptr<arrow::Field>> fields = {
        arrow::field("lag", arrow::int32()),
        arrow::field(acf ? "acf" : "pacf", arrow::float64()),
        arrow::field("confidence_lower", arrow::float64()),
        arrow::field("confidence_upper", arrow::float64()),
        arrow::field("significant", arrow::boolean()),
    };
    if (acf) {
        fields.push_back(arrow::field("ljung_box_q", arrow::float64()));
        fields.push_back(arrow::field("ljung_box_p", arrow::float64()));
    }
    return arrow::schema(fields);
}

arrow::Result<std::shared_ptr<arrow::Table>> CorrelogramTable(const std::shared_ptr<arrow::Table>& input, const std::string& signal_col, int max_lag, double confidence,
                                                              bool acf, const std::string& op_name, const MaterializationMemoryContext& memory,
                                                              const PipelineOperatorProgressCallback& callback) {
    ARROW_ASSIGN_OR_RAISE(const std::vector<double> signal,
                          ReadSignal(input, signal_col, op_name, acf ? 7 : 5, "reduce signal rows, cap max_lag, or run on a sampled / windowed signal first.", memory, callback));
    ReportProgress(callback, acf ? "Computing ACF" : "Computing PACF", acf ? "Computing the autocorrelation by lag" : "Computing the partial autocorrelation by lag", 0.50, 0,
                   static_cast<uint64_t>(signal.size()));
    const AutocorrelationResult result = TimeSeries::ComputeACFPACF(signal, max_lag, confidence);
    if (!result.success) return arrow::Status::ExecutionError(op_name + ": " + result.error_message);

    const auto& values = acf ? result.acf : result.pacf;
    const int n = static_cast<int>(values.size());
    std::vector<int32_t> lags;
    for (int i = 0; i < n; ++i) lags.push_back(i);
    ARROW_ASSIGN_OR_RAISE(auto lag_arr, BuildInt32Array(lags));
    ARROW_ASSIGN_OR_RAISE(auto value_arr, BuildDoubleArray(values));
    ARROW_ASSIGN_OR_RAISE(auto lower_arr, BuildDoubleArray(acf ? result.confidence_lower : result.pacf_confidence_lower));
    ARROW_ASSIGN_OR_RAISE(auto upper_arr, BuildDoubleArray(acf ? result.confidence_upper : result.pacf_confidence_upper));
    ARROW_ASSIGN_OR_RAISE(auto sig_arr, BuildBoolArray(MarkSignificant(n, acf ? result.significant_acf_lags : result.significant_pacf_lags)));
    std::vector<std::shared_ptr<arrow::Array>> arrays = {lag_arr, value_arr, lower_arr, upper_arr, sig_arr};
    if (acf) {
        std::vector<double> q(static_cast<size_t>(n), NAN), p(static_cast<size_t>(n), NAN);
        for (int k = 1; k < n; ++k) {
            if (static_cast<size_t>(k) < result.ljung_box_q.size()) q[static_cast<size_t>(k)] = result.ljung_box_q[static_cast<size_t>(k)];
            if (static_cast<size_t>(k) < result.ljung_box_p.size()) p[static_cast<size_t>(k)] = result.ljung_box_p[static_cast<size_t>(k)];
        }
        ARROW_ASSIGN_OR_RAISE(auto q_arr, BuildDoubleArray(q));
        ARROW_ASSIGN_OR_RAISE(auto p_arr, BuildDoubleArray(p));
        arrays.push_back(q_arr);
        arrays.push_back(p_arr);
    }
    spdlog::info("{}: n={} lags={} confidence={}{}", op_name, signal.size(), n - 1, confidence,
                 acf ? " Ljung-Box p=" + std::to_string(result.ljung_box_pvalue) + " at " + std::to_string(result.ljung_box_lags) + " lags" : "");
    ReportProgress(callback, "Complete", op_name + " materialization complete", 1.0, static_cast<uint64_t>(n), static_cast<uint64_t>(n));
    return arrow::Table::Make(CorrelogramSchema(acf), arrays);
}

} // namespace

bool ACFOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    return ConfigureCorrelogram(params, GetName(), signal_col_, max_lag_, confidence_, error);
}

arrow::Result<std::shared_ptr<arrow::Schema>> ACFOperator::InferOutputSchema(const std::shared_ptr<arrow::Schema>&) { return CorrelogramSchema(true); }

arrow::Result<std::shared_ptr<arrow::Table>> ACFOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz ACF Materializer");
    return CorrelogramTable(input, signal_col_, max_lag_, confidence_, true, GetName(), GetMaterializationMemoryContext(), progress_callback_);
}

bool PACFOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    return ConfigureCorrelogram(params, GetName(), signal_col_, max_lag_, confidence_, error);
}

arrow::Result<std::shared_ptr<arrow::Schema>> PACFOperator::InferOutputSchema(const std::shared_ptr<arrow::Schema>&) { return CorrelogramSchema(false); }

arrow::Result<std::shared_ptr<arrow::Table>> PACFOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz PACF Materializer");
    return CorrelogramTable(input, signal_col_, max_lag_, confidence_, false, GetName(), GetMaterializationMemoryContext(), progress_callback_);
}

// ============================================================================
// StationarityTestOperator
// ============================================================================

namespace {

std::shared_ptr<arrow::Schema> StationaritySchema() {
    return arrow::schema({
        arrow::field("differencing", arrow::int32()),
        arrow::field("adf_statistic", arrow::float64()),
        arrow::field("adf_pvalue", arrow::float64()),
        arrow::field("adf_lags", arrow::int32()),
        arrow::field("kpss_statistic", arrow::float64()),
        arrow::field("kpss_pvalue", arrow::float64()),
        arrow::field("kpss_lags", arrow::int32()),
        arrow::field("stationary", arrow::boolean()),
        arrow::field("suggested_differencing", arrow::int32()),
        arrow::field("analysis", arrow::utf8()),
    });
}

} // namespace

bool StationarityTestOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    signal_col_.clear();
    max_lags_ = -1;
    regression_ = "c";
    if (!ParseSignalColumn(params, GetName(), signal_col_, error)) return false;
    if (!ParseIntOptional(params, "max_lags", max_lags_, GetName(), error)) return false;
    if (max_lags_ < -1) {
        error = GetName() + ": max_lags must be -1 or >= 0";
        return false;
    }
    if (!ParseChoice(params, "regression", regression_, {"c", "ct"}, GetName(), error)) return false;
    return true;
}

arrow::Result<std::shared_ptr<arrow::Schema>> StationarityTestOperator::InferOutputSchema(const std::shared_ptr<arrow::Schema>&) { return StationaritySchema(); }

arrow::Result<std::shared_ptr<arrow::Table>> StationarityTestOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz StationarityTest Materializer");
    ARROW_ASSIGN_OR_RAISE(const std::vector<double> signal,
                          ReadSignal(input, signal_col_, GetName(), 3, "reduce signal rows, cap max_lags, or test a sampled / windowed signal first.",
                                     GetMaterializationMemoryContext(), progress_callback_));
    ReportProgress(progress_callback_, "Testing stationarity", "Running ADF and KPSS on the series and its differences", 0.55, 0, static_cast<uint64_t>(signal.size()));
    const StationarityResult result = TimeSeries::TestStationarity(signal, max_lags_, regression_);
    if (!result.success) return arrow::Status::ExecutionError(GetName() + ": " + result.error_message);

    // One row for the series, one for each difference tried.
    std::vector<int32_t> differencing = {0}, adf_lags = {static_cast<int32_t>(result.adf_lags)}, kpss_lags = {static_cast<int32_t>(result.kpss_lags)};
    std::vector<double> adf_stat = {result.adf_statistic}, adf_p = {result.adf_pvalue}, kpss_stat = {result.kpss_statistic}, kpss_p = {result.kpss_pvalue};
    std::vector<bool> stationary = {result.is_stationary};
    std::vector<std::string> analysis = {result.analysis};
    for (const auto& d : result.differenced) {
        differencing.push_back(static_cast<int32_t>(d.d));
        adf_lags.push_back(-1);
        kpss_lags.push_back(-1);
        adf_stat.push_back(d.adf_statistic);
        adf_p.push_back(d.adf_pvalue);
        kpss_stat.push_back(d.kpss_statistic);
        kpss_p.push_back(d.kpss_pvalue);
        stationary.push_back(d.stationary);
        analysis.push_back(std::string("differenced ") + std::to_string(d.d) + (d.d == 1 ? " time: " : " times: ") + (d.stationary ? "stationary enough" : "not stationary yet"));
    }
    std::vector<int32_t> suggested(differencing.size(), static_cast<int32_t>(result.suggested_differencing));

    ARROW_ASSIGN_OR_RAISE(auto d_arr, BuildInt32Array(differencing));
    ARROW_ASSIGN_OR_RAISE(auto adf_stat_arr, BuildDoubleArray(adf_stat));
    ARROW_ASSIGN_OR_RAISE(auto adf_p_arr, BuildDoubleArray(adf_p));
    ARROW_ASSIGN_OR_RAISE(auto adf_lags_arr, BuildOptionalInt32Array(adf_lags));
    ARROW_ASSIGN_OR_RAISE(auto kpss_stat_arr, BuildDoubleArray(kpss_stat));
    ARROW_ASSIGN_OR_RAISE(auto kpss_p_arr, BuildDoubleArray(kpss_p));
    ARROW_ASSIGN_OR_RAISE(auto kpss_lags_arr, BuildOptionalInt32Array(kpss_lags));
    ARROW_ASSIGN_OR_RAISE(auto stationary_arr, BuildBoolArray(stationary));
    ARROW_ASSIGN_OR_RAISE(auto suggested_arr, BuildInt32Array(suggested));
    ARROW_ASSIGN_OR_RAISE(auto analysis_arr, BuildStringArray(analysis));

    spdlog::info("{}: n={} regression={} ADF p={:.4f} KPSS p={:.4f} stationary={} suggested d={}", GetName(), signal.size(), regression_, result.adf_pvalue,
                 result.kpss_pvalue, result.is_stationary, result.suggested_differencing);
    ReportProgress(progress_callback_, "Complete", "StationarityTest materialization complete", 1.0, static_cast<uint64_t>(differencing.size()),
                   static_cast<uint64_t>(differencing.size()));
    return arrow::Table::Make(StationaritySchema(), {d_arr, adf_stat_arr, adf_p_arr, adf_lags_arr, kpss_stat_arr, kpss_p_arr, kpss_lags_arr, stationary_arr, suggested_arr, analysis_arr});
}

// ============================================================================
// SeasonalityDetectorOperator
// ============================================================================

namespace {

std::shared_ptr<arrow::Schema> SeasonalitySchema() {
    return arrow::schema({
        arrow::field("period", arrow::int32()),
        arrow::field("share", arrow::float64()),
        arrow::field("harmonic_of", arrow::int32()),
        arrow::field("is_primary", arrow::boolean()),
        arrow::field("has_seasonality", arrow::boolean()),
        arrow::field("strength", arrow::float64()),
        arrow::field("analysis", arrow::utf8()),
    });
}

} // namespace

bool SeasonalityDetectorOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    signal_col_.clear();
    min_period_ = 2;
    max_period_ = -1;
    if (!ParseSignalColumn(params, GetName(), signal_col_, error)) return false;
    if (!ParseIntOptional(params, "min_period", min_period_, GetName(), error)) return false;
    if (!ParseIntOptional(params, "max_period", max_period_, GetName(), error)) return false;
    if (min_period_ < 2) {
        error = GetName() + ": min_period must be >= 2";
        return false;
    }
    if (max_period_ != -1 && max_period_ <= min_period_) {
        error = GetName() + ": max_period must be -1 or > min_period";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Schema>> SeasonalityDetectorOperator::InferOutputSchema(const std::shared_ptr<arrow::Schema>&) { return SeasonalitySchema(); }

arrow::Result<std::shared_ptr<arrow::Table>> SeasonalityDetectorOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz SeasonalityDetector Materializer");
    ARROW_ASSIGN_OR_RAISE(const std::vector<double> signal,
                          ReadSignal(input, signal_col_, GetName(), 3, "reduce signal rows, narrow the period search, or detect on a sampled / windowed signal first.",
                                     GetMaterializationMemoryContext(), progress_callback_));
    ReportProgress(progress_callback_, "Detecting seasonality", "Ranking the periods of the detrended periodogram", 0.55, 0, static_cast<uint64_t>(signal.size()));
    const SeasonalityResult result = TimeSeries::DetectSeasonality(signal, min_period_, max_period_);
    if (!result.success) return arrow::Status::ExecutionError(GetName() + ": " + result.error_message);

    std::vector<int32_t> periods, harmonic_of;
    std::vector<double> shares, strengths;
    std::vector<bool> primary, has;
    std::vector<std::string> analysis;
    const auto add = [&](int period, double share, int harmonic) {
        const bool is_primary = period == result.detected_period;
        periods.push_back(static_cast<int32_t>(period));
        shares.push_back(share);
        harmonic_of.push_back(static_cast<int32_t>(harmonic));
        primary.push_back(is_primary);
        has.push_back(result.has_seasonality);
        strengths.push_back(is_primary ? result.strength : NAN);
        analysis.push_back(is_primary ? result.analysis : harmonic > 0 ? "a harmonic of " + std::to_string(harmonic) : "");
    };
    if (result.candidate_periods.empty()) {
        add(result.detected_period, result.detected_share, 0);
    } else {
        for (size_t i = 0; i < result.candidate_periods.size(); ++i)
            add(result.candidate_periods[i], i < result.candidate_strengths.size() ? result.candidate_strengths[i] : 0.0, i < result.harmonic_of.size() ? result.harmonic_of[i] : 0);
    }

    ARROW_ASSIGN_OR_RAISE(auto period_arr, BuildInt32Array(periods));
    ARROW_ASSIGN_OR_RAISE(auto share_arr, BuildDoubleArray(shares));
    ARROW_ASSIGN_OR_RAISE(auto harmonic_arr, BuildInt32Array(harmonic_of));
    ARROW_ASSIGN_OR_RAISE(auto primary_arr, BuildBoolArray(primary));
    ARROW_ASSIGN_OR_RAISE(auto has_arr, BuildBoolArray(has));
    ARROW_ASSIGN_OR_RAISE(auto strength_arr, BuildDoubleArray(strengths));
    ARROW_ASSIGN_OR_RAISE(auto analysis_arr, BuildStringArray(analysis));

    spdlog::info("{}: n={} period={} share={:.3f} strength={:.3f} candidates={}", GetName(), signal.size(), result.detected_period, result.detected_share, result.strength,
                 periods.size());
    ReportProgress(progress_callback_, "Complete", "SeasonalityDetector materialization complete", 1.0, static_cast<uint64_t>(periods.size()), static_cast<uint64_t>(periods.size()));
    return arrow::Table::Make(SeasonalitySchema(), {period_arr, share_arr, harmonic_arr, primary_arr, has_arr, strength_arr, analysis_arr});
}

// ============================================================================
// SeasonalNaiveOperator
// ============================================================================

bool SeasonalNaiveOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    seasonal_period_ = 0;
    const auto it = params.find("seasonal_period");
    if (it == params.end() || it->second.empty()) {
        error = GetName() + ": 'seasonal_period' parameter is required";
        return false;
    }
    try {
        size_t consumed = 0;
        seasonal_period_ = std::stoi(it->second, &consumed);
        if (consumed != it->second.size()) {
            throw std::invalid_argument("trailing characters");
        }
    } catch (...) {
        error = GetName() + ": 'seasonal_period' is not a valid integer: " +
                it->second;
        return false;
    }
    if (seasonal_period_ < 1) {
        error = GetName() + ": seasonal_period must be >= 1";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Schema>>
SeasonalNaiveOperator::InferOutputSchema(
    const std::shared_ptr<arrow::Schema>& input_schema) {
    if (!input_schema) {
        return arrow::Status::Invalid(GetName() + ": input schema is null");
    }

    std::vector<std::shared_ptr<arrow::Field>> fields = {
        arrow::field("window_index", arrow::int64()),
        arrow::field("horizon", arrow::int32()),
        arrow::field("actual", arrow::float64()),
        arrow::field("prediction", arrow::float64()),
        arrow::field("error", arrow::float64()),
    };
    if (input_schema->GetFieldIndex("__target_start_index") >= 0) {
        fields.push_back(arrow::field("__target_index", arrow::int64()));
    }
    if (input_schema->GetFieldIndex("__partition__") >= 0) {
        fields.push_back(arrow::field("__partition__", arrow::int8()));
    }
    return arrow::schema(std::move(fields));
}

arrow::Result<std::shared_ptr<arrow::Table>>
SeasonalNaiveOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz SeasonalNaive Materializer");
    if (!input || !input->schema()) {
        return arrow::Status::Invalid(GetName() + ": input table is null");
    }
    if (seasonal_period_ < 1) {
        return arrow::Status::Invalid(
            GetName() + ": Apply called before Configure succeeded");
    }
    if (input->num_rows() == 0) {
        return arrow::Status::Invalid(GetName() + ": input table has zero rows");
    }

    const auto schema = input->schema();
    int input_width = 0;
    while (schema->GetFieldIndex("x_" + std::to_string(input_width)) >= 0) {
        ++input_width;
    }
    if (input_width == 0) {
        return arrow::Status::Invalid(
            GetName() + ": expected TimeSeriesWindow feature column 'x_0'");
    }
    if (seasonal_period_ > input_width) {
        return arrow::Status::Invalid(
            GetName() + ": seasonal_period (" +
            std::to_string(seasonal_period_) +
            ") exceeds available input width (" +
            std::to_string(input_width) + ")");
    }

    std::vector<std::string> target_names;
    if (schema->GetFieldIndex("y") < 0) {
        return arrow::Status::Invalid(
            GetName() + ": expected TimeSeriesWindow target column 'y'");
    }
    target_names.push_back("y");
    for (int horizon = 1;; ++horizon) {
        const std::string name = "y_" + std::to_string(horizon);
        if (schema->GetFieldIndex(name) < 0) break;
        target_names.push_back(name);
    }

    std::vector<std::vector<float>> seasonal_inputs;
    seasonal_inputs.reserve(static_cast<size_t>(seasonal_period_));
    const int seasonal_start = input_width - seasonal_period_;
    for (int offset = 0; offset < seasonal_period_; ++offset) {
        const std::string name =
            "x_" + std::to_string(seasonal_start + offset);
        std::vector<float> values;
        std::string bad_type;
        if (!ReadColumnAsFloat(input->GetColumnByName(name), values, bad_type)) {
            return arrow::Status::TypeError(
                GetName() + ": feature column '" + name +
                "' must be numeric (got '" + bad_type + "')");
        }
        seasonal_inputs.push_back(std::move(values));
    }

    std::vector<std::vector<float>> targets;
    targets.reserve(target_names.size());
    for (const auto& name : target_names) {
        std::vector<float> values;
        std::string bad_type;
        if (!ReadColumnAsFloat(input->GetColumnByName(name), values, bad_type)) {
            return arrow::Status::TypeError(
                GetName() + ": target column '" + name +
                "' must be numeric (got '" + bad_type + "')");
        }
        targets.push_back(std::move(values));
    }

    const int64_t input_rows = input->num_rows();
    const int64_t horizon_count = static_cast<int64_t>(targets.size());
    if (input_rows > std::numeric_limits<int64_t>::max() / horizon_count) {
        return arrow::Status::CapacityError(
            GetName() + ": expanded output row count overflows int64");
    }
    const int64_t output_rows = input_rows * horizon_count;
    const bool has_target_index =
        schema->GetFieldIndex("__target_start_index") >= 0;
    const bool has_partition = schema->GetFieldIndex("__partition__") >= 0;
    const uint64_t output_columns =
        5ULL + (has_target_index ? 1ULL : 0ULL) +
        (has_partition ? 1ULL : 0ULL);
    const auto estimate = EstimateDenseMaterializationMemory(
        static_cast<uint64_t>(output_rows), output_columns, sizeof(double));
    const auto decision = EvaluateMaterializationMemory(
        estimate, GetMaterializationMemoryContext());
    const std::string preflight_message =
        BuildTimeSeriesAnalysisMemoryPreflightMessage(
            GetName(), estimate, decision,
            "reduce source rows or forecast horizon, filter to one partition, "
            "or use a future chunked baseline materializer");
    if (progress_callback_) {
        PipelineOperatorProgress event;
        event.stage = "SeasonalNaive memory preflight";
        event.message = preflight_message;
        event.status = MaterializationMemoryRiskToProgressStatus(decision.risk);
        event.progress = 0.03f;
        event.estimated_memory_bytes = estimate.estimated_peak_bytes;
        event.memory_risk_level = MaterializationMemoryRiskName(decision.risk);
        event.total_items = static_cast<uint64_t>(output_rows);
        progress_callback_(event);
    }
    if (decision.blocked) {
        return arrow::Status::CapacityError(
            "Materialization blocked: " + preflight_message);
    }

    std::vector<int64_t> target_starts;
    if (has_target_index) {
        auto column = input->GetColumnByName("__target_start_index");
        if (!column || column->type()->id() != arrow::Type::INT64) {
            return arrow::Status::TypeError(
                GetName() + ": __target_start_index must be int64");
        }
        target_starts.reserve(static_cast<size_t>(input_rows));
        for (const auto& chunk : column->chunks()) {
            auto array = std::static_pointer_cast<arrow::Int64Array>(chunk);
            for (int64_t row = 0; row < array->length(); ++row) {
                if (array->IsNull(row)) {
                    return arrow::Status::Invalid(
                        GetName() + ": __target_start_index contains nulls");
                }
                target_starts.push_back(array->Value(row));
            }
        }
    }

    std::vector<int8_t> partitions;
    if (has_partition) {
        auto column = input->GetColumnByName("__partition__");
        ARROW_ASSIGN_OR_RAISE(
            partitions, ReadPartitionMetadataAsInt8(column, GetName()));
    }

    arrow::Int64Builder window_builder;
    arrow::Int32Builder horizon_builder;
    arrow::DoubleBuilder actual_builder;
    arrow::DoubleBuilder prediction_builder;
    arrow::DoubleBuilder error_builder;
    arrow::Int64Builder target_index_builder;
    arrow::Int8Builder partition_builder;
    ARROW_RETURN_NOT_OK(window_builder.Reserve(output_rows));
    ARROW_RETURN_NOT_OK(horizon_builder.Reserve(output_rows));
    ARROW_RETURN_NOT_OK(actual_builder.Reserve(output_rows));
    ARROW_RETURN_NOT_OK(prediction_builder.Reserve(output_rows));
    ARROW_RETURN_NOT_OK(error_builder.Reserve(output_rows));
    if (has_target_index) {
        ARROW_RETURN_NOT_OK(target_index_builder.Reserve(output_rows));
    }
    if (has_partition) {
        ARROW_RETURN_NOT_OK(partition_builder.Reserve(output_rows));
    }

    ReportProgress(progress_callback_, "Generating forecasts",
                   "Repeating the latest seasonal cycle across forecast horizons",
                   0.10, 0, static_cast<uint64_t>(output_rows),
                   estimate.estimated_peak_bytes);
    int64_t written = 0;
    for (int64_t row = 0; row < input_rows; ++row) {
        for (int64_t horizon = 0; horizon < horizon_count; ++horizon) {
            const float actual = targets[static_cast<size_t>(horizon)]
                                      [static_cast<size_t>(row)];
            const float prediction =
                seasonal_inputs[static_cast<size_t>(
                    horizon % static_cast<int64_t>(seasonal_period_))]
                               [static_cast<size_t>(row)];
            ARROW_RETURN_NOT_OK(window_builder.Append(row));
            ARROW_RETURN_NOT_OK(
                horizon_builder.Append(static_cast<int32_t>(horizon + 1)));
            if (std::isfinite(actual) && std::isfinite(prediction)) {
                ARROW_RETURN_NOT_OK(actual_builder.Append(actual));
                ARROW_RETURN_NOT_OK(prediction_builder.Append(prediction));
                ARROW_RETURN_NOT_OK(
                    error_builder.Append(static_cast<double>(actual) -
                                         static_cast<double>(prediction)));
            } else {
                ARROW_RETURN_NOT_OK(actual_builder.AppendNull());
                ARROW_RETURN_NOT_OK(prediction_builder.AppendNull());
                ARROW_RETURN_NOT_OK(error_builder.AppendNull());
            }
            if (has_target_index) {
                ARROW_RETURN_NOT_OK(target_index_builder.Append(
                    target_starts[static_cast<size_t>(row)] + horizon));
            }
            if (has_partition) {
                ARROW_RETURN_NOT_OK(partition_builder.Append(
                    partitions[static_cast<size_t>(row)]));
            }
            ++written;
        }
        if ((row + 1) == input_rows || ((row + 1) % 256) == 0) {
            ReportProgress(
                progress_callback_, "Generating forecasts",
                "Writing long-form seasonal-naive predictions",
                0.10 + 0.85 * static_cast<double>(row + 1) /
                           static_cast<double>(input_rows),
                static_cast<uint64_t>(written),
                static_cast<uint64_t>(output_rows),
                estimate.estimated_peak_bytes);
        }
    }

    std::vector<std::shared_ptr<arrow::Array>> arrays;
    arrays.reserve(static_cast<size_t>(output_columns));
    std::shared_ptr<arrow::Array> array;
    ARROW_RETURN_NOT_OK(window_builder.Finish(&array));
    arrays.push_back(std::move(array));
    ARROW_RETURN_NOT_OK(horizon_builder.Finish(&array));
    arrays.push_back(std::move(array));
    ARROW_RETURN_NOT_OK(actual_builder.Finish(&array));
    arrays.push_back(std::move(array));
    ARROW_RETURN_NOT_OK(prediction_builder.Finish(&array));
    arrays.push_back(std::move(array));
    ARROW_RETURN_NOT_OK(error_builder.Finish(&array));
    arrays.push_back(std::move(array));
    if (has_target_index) {
        ARROW_RETURN_NOT_OK(target_index_builder.Finish(&array));
        arrays.push_back(std::move(array));
    }
    if (has_partition) {
        ARROW_RETURN_NOT_OK(partition_builder.Finish(&array));
        arrays.push_back(std::move(array));
    }

    ARROW_ASSIGN_OR_RAISE(auto output_schema, InferOutputSchema(schema));
    spdlog::info(
        "SeasonalNaive: {} windows x {} horizons -> {} prediction rows "
        "(period={}, input_width={}, partition_metadata={})",
        input_rows, horizon_count, output_rows, seasonal_period_, input_width,
        has_partition);
    ReportProgress(progress_callback_, "Complete",
                   "SeasonalNaive forecast materialization complete", 1.0,
                   static_cast<uint64_t>(output_rows),
                   static_cast<uint64_t>(output_rows),
                   estimate.estimated_peak_bytes);
    return arrow::Table::Make(output_schema, std::move(arrays), output_rows);
}

} // namespace cyxwiz
