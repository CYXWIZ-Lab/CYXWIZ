#include "time_series_analysis_operators.h"
#include "../materialization_memory_guard.h"
#include "../profiler_trace.h"
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

// Append a float64 column to the table.
arrow::Result<std::shared_ptr<arrow::Table>> AppendF64Column(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& name,
    const std::vector<double>& values) {
    arrow::DoubleBuilder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(static_cast<int64_t>(values.size())));
    for (double v : values) ARROW_RETURN_NOT_OK(builder.Append(v));
    std::shared_ptr<arrow::Array> arr;
    ARROW_RETURN_NOT_OK(builder.Finish(&arr));
    auto field = arrow::field(name, arrow::float64());
    auto chunked = std::make_shared<arrow::ChunkedArray>(arr);
    return table->AddColumn(table->num_columns(), field, chunked);
}

// Read one numeric signal column as vector<double>.
arrow::Status ReadSignalAsDouble(
    const std::shared_ptr<arrow::Table>& input,
    const std::string& col_name,
    const std::string& op_name,
    std::vector<double>& out) {

    auto col = input->GetColumnByName(col_name);
    if (!col) {
        return arrow::Status::KeyError(
            op_name + ": signal column '" + col_name + "' not found");
    }
    std::vector<float> floats;
    std::string bad;
    if (!ReadColumnAsFloat(col, floats, bad)) {
        return arrow::Status::TypeError(
            op_name + ": signal column '" + col_name +
            "' must be numeric (got '" + bad + "')");
    }
    out.assign(floats.begin(), floats.end());
    return arrow::Status::OK();
}

// Compute residual = signal - fitted, padding NaNs where fitted is shorter.
std::vector<double> ComputeResidual(
    const std::vector<double>& signal,
    const std::vector<double>& fitted) {
    std::vector<double> residual(signal.size(), 0.0);
    for (size_t i = 0; i < signal.size(); ++i) {
        residual[i] = (i < fitted.size()) ? signal[i] - fitted[i] : 0.0;
    }
    return residual;
}

bool ParseIntOptional(const std::map<std::string, std::string>& params,
                      const std::string& key, int& out,
                      const std::string& op_name, std::string& error) {
    auto it = params.find(key);
    if (it == params.end() || it->second.empty()) return true;
    try { out = std::stoi(it->second); return true; }
    catch (...) {
        error = op_name + ": '" + key + "' is not a valid integer: " + it->second;
        return false;
    }
}

bool ParseDoubleOptional(const std::map<std::string, std::string>& params,
                         const std::string& key, double& out,
                         const std::string& op_name, std::string& error) {
    auto it = params.find(key);
    if (it == params.end() || it->second.empty()) return true;
    try { out = std::stod(it->second); return true; }
    catch (...) {
        error = op_name + ": '" + key + "' is not a valid float: " + it->second;
        return false;
    }
}

arrow::Result<std::shared_ptr<arrow::Array>> BuildInt32Array(
    const std::vector<int32_t>& values) {
    arrow::Int32Builder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(static_cast<int64_t>(values.size())));
    for (int32_t v : values) ARROW_RETURN_NOT_OK(builder.Append(v));
    std::shared_ptr<arrow::Array> array;
    ARROW_RETURN_NOT_OK(builder.Finish(&array));
    return array;
}

arrow::Result<std::shared_ptr<arrow::Array>> BuildDoubleArray(
    const std::vector<double>& values) {
    arrow::DoubleBuilder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(static_cast<int64_t>(values.size())));
    for (double v : values) ARROW_RETURN_NOT_OK(builder.Append(v));
    std::shared_ptr<arrow::Array> array;
    ARROW_RETURN_NOT_OK(builder.Finish(&array));
    return array;
}

arrow::Result<std::shared_ptr<arrow::Array>> BuildBoolArray(
    const std::vector<bool>& values) {
    arrow::BooleanBuilder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(static_cast<int64_t>(values.size())));
    for (bool v : values) ARROW_RETURN_NOT_OK(builder.Append(v));
    std::shared_ptr<arrow::Array> array;
    ARROW_RETURN_NOT_OK(builder.Finish(&array));
    return array;
}

arrow::Result<std::shared_ptr<arrow::Array>> BuildStringArray(
    const std::vector<std::string>& values) {
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

bool ParseSignalColumn(const std::map<std::string, std::string>& params,
                       const std::string& op_name,
                       std::string& signal_col,
                       std::string& error) {
    auto s = params.find("signal_col");
    if (s == params.end() || s->second.empty()) {
        error = op_name + ": 'signal_col' parameter is required";
        return false;
    }
    signal_col = s->second;
    return true;
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

} // namespace

// ============================================================================
// TimeSeriesDecompositionOperator
// ============================================================================

bool TimeSeriesDecompositionOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    signal_col_.clear();
    period_ = 0;
    method_ = "additive";
    algorithm_ = "classical";

    auto s = params.find("signal_col");
    if (s == params.end() || s->second.empty()) {
        error = GetName() + ": 'signal_col' parameter is required";
        return false;
    }
    signal_col_ = s->second;

    if (!ParseIntOptional(params, "period", period_, GetName(), error)) return false;
    if (period_ < 2) {
        error = GetName() + ": 'period' must be >= 2 (got " +
                std::to_string(period_) + ")";
        return false;
    }

    auto m = params.find("method");
    if (m != params.end() && !m->second.empty()) {
        method_ = NormalizeTimeSeriesParameterChoice(m->second);
        if (method_ != "additive" && method_ != "multiplicative") {
            error = GetName() + ": 'method' must be 'additive' / "
                    "'multiplicative' (got '" + method_ + "')";
            return false;
        }
    }

    auto a = params.find("algorithm");
    if (a != params.end() && !a->second.empty()) {
        algorithm_ = NormalizeTimeSeriesParameterChoice(a->second);
        if (algorithm_ != "classical" && algorithm_ != "stl") {
            error = GetName() + ": 'algorithm' must be 'classical' / 'stl' "
                    "(got '" + algorithm_ + "')";
            return false;
        }
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
TimeSeriesDecompositionOperator::Apply(
    const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz TimeSeriesDecomposition Materializer");

    if (!input) return arrow::Status::Invalid(GetName() + ": input is null");

    uint64_t planned_samples = 0;
    ARROW_ASSIGN_OR_RAISE(auto preflight_estimate,
        EmitSignalAnalysisMemoryPreflight(
            input, signal_col_, GetName(), 4,
            "reduce signal rows, decompose a sampled/windowed signal first, or use a future chunked decomposition path.",
            GetMaterializationMemoryContext(),
            progress_callback_, planned_samples));

    ReportProgress(progress_callback_, "Reading signal",
                   "Reading decomposition signal column '" + signal_col_ + "'",
                   0.10,
                   0,
                   planned_samples,
                   preflight_estimate.estimated_peak_bytes);
    std::vector<double> signal;
    ARROW_RETURN_NOT_OK(ReadSignalAsDouble(input, signal_col_, GetName(), signal));

    if (static_cast<int>(signal.size()) < 2 * period_) {
        return arrow::Status::Invalid(
            GetName() + ": need at least 2*period (" +
            std::to_string(2 * period_) + ") samples, got " +
            std::to_string(signal.size()));
    }

    const uint64_t estimated_signal_bytes =
        preflight_estimate.estimated_peak_bytes;
    ReportProgress(progress_callback_, "Decomposing signal",
                   "Running " + algorithm_ + " time-series decomposition",
                   0.45,
                   0,
                   static_cast<uint64_t>(signal.size()),
                   estimated_signal_bytes);
    DecompositionResult result;
    if (algorithm_ == "stl") {
        result = TimeSeries::STLDecompose(signal, period_);
    } else {
        result = TimeSeries::Decompose(signal, period_, method_);
    }
    if (!result.success) {
        return arrow::Status::ExecutionError(
            GetName() + ": " + algorithm_ + " decomposition failed: " +
            result.error_message);
    }

    // Backend returns parallel-length trend/seasonal/residual vectors.
    ReportProgress(progress_callback_, "Appending columns",
                   "Appending trend, seasonal, and residual columns",
                   0.85,
                   static_cast<uint64_t>(signal.size()),
                   static_cast<uint64_t>(signal.size()),
                   static_cast<uint64_t>(signal.size()) * 3 * sizeof(double));
    auto out = input;
    ARROW_ASSIGN_OR_RAISE(out, AppendF64Column(out, "trend", result.trend));
    ARROW_ASSIGN_OR_RAISE(out, AppendF64Column(out, "seasonal", result.seasonal));
    ARROW_ASSIGN_OR_RAISE(out, AppendF64Column(out, "residual", result.residual));

    spdlog::info("{}: n={} period={} method={} algorithm={} "
                 "trend_strength={:.4f} seasonal_strength={:.4f}",
                 GetName(), signal.size(), period_, method_, algorithm_,
                 result.trend_strength, result.seasonal_strength);
    ReportProgress(progress_callback_, "Complete",
                   "TimeSeriesDecomposition materialization complete",
                   1.0,
                   static_cast<uint64_t>(signal.size()),
                   static_cast<uint64_t>(signal.size()),
                   static_cast<uint64_t>(signal.size()) * 3 * sizeof(double));
    return out;
}

// ============================================================================
// ARIMAOperator
// ============================================================================

bool ARIMAOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    signal_col_.clear();
    p_ = -1;
    d_ = -1;
    q_ = -1;

    auto s = params.find("signal_col");
    if (s == params.end() || s->second.empty()) {
        error = GetName() + ": 'signal_col' parameter is required";
        return false;
    }
    signal_col_ = s->second;

    if (!ParseIntOptional(params, "p", p_, GetName(), error)) return false;
    if (!ParseIntOptional(params, "d", d_, GetName(), error)) return false;
    if (!ParseIntOptional(params, "q", q_, GetName(), error)) return false;
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
ARIMAOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz ARIMA Materializer");

    if (!input) return arrow::Status::Invalid(GetName() + ": input is null");

    uint64_t planned_samples = 0;
    ARROW_ASSIGN_OR_RAISE(auto preflight_estimate,
        EmitSignalAnalysisMemoryPreflight(
            input, signal_col_, GetName(), 3,
            "reduce signal rows, fit ARIMA on a sampled/windowed signal first, or use a future chunked forecasting path.",
            GetMaterializationMemoryContext(),
            progress_callback_, planned_samples));

    ReportProgress(progress_callback_, "Reading signal",
                   "Reading ARIMA signal column '" + signal_col_ + "'",
                   0.10,
                   0,
                   planned_samples,
                   preflight_estimate.estimated_peak_bytes);
    std::vector<double> signal;
    ARROW_RETURN_NOT_OK(ReadSignalAsDouble(input, signal_col_, GetName(), signal));

    // horizon=0 so the backend produces in-sample fitted values only.
    const uint64_t estimated_signal_bytes =
        preflight_estimate.estimated_peak_bytes;
    ReportProgress(progress_callback_, "Fitting model",
                   "Fitting ARIMA model",
                   0.45,
                   0,
                   static_cast<uint64_t>(signal.size()),
                   estimated_signal_bytes);
    auto result = TimeSeries::ARIMA(signal, 0, p_, d_, q_);
    if (!result.success) {
        return arrow::Status::ExecutionError(
            GetName() + ": fit failed: " + result.error_message);
    }

    if (result.fitted_values.size() != signal.size()) {
        return arrow::Status::ExecutionError(
            GetName() + ": backend returned " +
            std::to_string(result.fitted_values.size()) +
            " fitted values, expected " + std::to_string(signal.size()));
    }

    auto residual = ComputeResidual(signal, result.fitted_values);

    ReportProgress(progress_callback_, "Appending columns",
                   "Appending ARIMA fitted and residual columns",
                   0.85,
                   static_cast<uint64_t>(signal.size()),
                   static_cast<uint64_t>(signal.size()),
                   static_cast<uint64_t>(signal.size()) * 2 * sizeof(double));
    auto out = input;
    ARROW_ASSIGN_OR_RAISE(out, AppendF64Column(out, "fitted", result.fitted_values));
    ARROW_ASSIGN_OR_RAISE(out, AppendF64Column(out, "residual", residual));

    spdlog::info("{}: n={} p={} d={} q={} RMSE={:.4f} AIC={:.4f} BIC={:.4f}",
                 GetName(), signal.size(), p_, d_, q_,
                 result.rmse, result.aic, result.bic);
    ReportProgress(progress_callback_, "Complete",
                   "ARIMA materialization complete",
                   1.0,
                   static_cast<uint64_t>(signal.size()),
                   static_cast<uint64_t>(signal.size()),
                   static_cast<uint64_t>(signal.size()) * 2 * sizeof(double));
    return out;
}

// ============================================================================
// ExponentialSmoothingOperator
// ============================================================================

bool ExponentialSmoothingOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    signal_col_.clear();
    method_ = "simple";
    alpha_ = -1.0;
    beta_ = -1.0;
    gamma_ = -1.0;
    period_ = -1;
    damped_ = false;

    auto s = params.find("signal_col");
    if (s == params.end() || s->second.empty()) {
        error = GetName() + ": 'signal_col' parameter is required";
        return false;
    }
    signal_col_ = s->second;

    auto m = params.find("method");
    if (m != params.end() && !m->second.empty()) {
        method_ = NormalizeTimeSeriesParameterChoice(m->second);
        if (method_ != "simple" && method_ != "holt" && method_ != "holt_winters") {
            error = GetName() + ": 'method' must be 'simple' / 'holt' / "
                    "'holt_winters' (got '" + method_ + "')";
            return false;
        }
    }

    if (!ParseDoubleOptional(params, "alpha", alpha_, GetName(), error)) return false;
    if (!ParseDoubleOptional(params, "beta", beta_, GetName(), error)) return false;
    if (!ParseDoubleOptional(params, "gamma", gamma_, GetName(), error)) return false;
    if (!ParseIntOptional(params, "period", period_, GetName(), error)) return false;

    auto d = params.find("damped");
    if (d != params.end() && !d->second.empty()) {
        if (d->second == "true") {
            damped_ = true;
        } else if (d->second == "false") {
            damped_ = false;
        } else {
            error = GetName() + ": 'damped' must be 'true' or 'false' (got '" +
                    d->second + "')";
            return false;
        }
    }

    if (method_ == "holt_winters" && period_ < 2 && period_ != -1) {
        error = GetName() + ": holt_winters requires 'period' >= 2 (or -1 for auto-detect)";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
ExponentialSmoothingOperator::Apply(
    const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz ExponentialSmoothing Materializer");

    if (!input) return arrow::Status::Invalid(GetName() + ": input is null");

    uint64_t planned_samples = 0;
    ARROW_ASSIGN_OR_RAISE(auto preflight_estimate,
        EmitSignalAnalysisMemoryPreflight(
            input, signal_col_, GetName(), 3,
            "reduce signal rows, fit smoothing on a sampled/windowed signal first, or use a future chunked forecasting path.",
            GetMaterializationMemoryContext(),
            progress_callback_, planned_samples));

    ReportProgress(progress_callback_, "Reading signal",
                   "Reading ExponentialSmoothing signal column '" +
                   signal_col_ + "'",
                   0.10,
                   0,
                   planned_samples,
                   preflight_estimate.estimated_peak_bytes);
    std::vector<double> signal;
    ARROW_RETURN_NOT_OK(ReadSignalAsDouble(input, signal_col_, GetName(), signal));

    const uint64_t estimated_signal_bytes =
        preflight_estimate.estimated_peak_bytes;
    ReportProgress(progress_callback_, "Fitting model",
                   "Fitting " + method_ + " exponential smoothing model",
                   0.45,
                   0,
                   static_cast<uint64_t>(signal.size()),
                   estimated_signal_bytes);
    ForecastResult result;
    if (method_ == "simple") {
        result = TimeSeries::SimpleES(signal, 0, alpha_);
    } else if (method_ == "holt") {
        result = TimeSeries::HoltLinear(signal, 0, alpha_, beta_, damped_);
    } else {
        // holt_winters
        result = TimeSeries::HoltWinters(signal, 0, period_, "additive",
                                          alpha_, beta_, gamma_);
    }
    if (!result.success) {
        return arrow::Status::ExecutionError(
            GetName() + ": " + method_ + " fit failed: " + result.error_message);
    }

    if (result.fitted_values.size() != signal.size()) {
        return arrow::Status::ExecutionError(
            GetName() + ": backend returned " +
            std::to_string(result.fitted_values.size()) +
            " fitted values, expected " + std::to_string(signal.size()));
    }

    auto residual = ComputeResidual(signal, result.fitted_values);

    ReportProgress(progress_callback_, "Appending columns",
                   "Appending smoothing fitted and residual columns",
                   0.85,
                   static_cast<uint64_t>(signal.size()),
                   static_cast<uint64_t>(signal.size()),
                   static_cast<uint64_t>(signal.size()) * 2 * sizeof(double));
    auto out = input;
    ARROW_ASSIGN_OR_RAISE(out, AppendF64Column(out, "fitted", result.fitted_values));
    ARROW_ASSIGN_OR_RAISE(out, AppendF64Column(out, "residual", residual));

    spdlog::info("{}: n={} method={} RMSE={:.4f} AIC={:.4f}",
                 GetName(), signal.size(), method_, result.rmse, result.aic);
    ReportProgress(progress_callback_, "Complete",
                   "ExponentialSmoothing materialization complete",
                   1.0,
                   static_cast<uint64_t>(signal.size()),
                   static_cast<uint64_t>(signal.size()),
                   static_cast<uint64_t>(signal.size()) * 2 * sizeof(double));
    return out;
}

// ============================================================================
// ACFOperator
// ============================================================================

bool ACFOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    signal_col_.clear();
    max_lag_ = -1;

    if (!ParseSignalColumn(params, GetName(), signal_col_, error)) return false;
    if (!ParseIntOptional(params, "max_lag", max_lag_, GetName(), error)) return false;
    if (!ParseIntOptional(params, "lags", max_lag_, GetName(), error)) return false;
    if (max_lag_ == 0 || max_lag_ < -1) {
        error = GetName() + ": max_lag must be -1 or >= 1";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
ACFOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz ACF Materializer");

    if (!input) return arrow::Status::Invalid(GetName() + ": input is null");

    uint64_t planned_samples = 0;
    ARROW_ASSIGN_OR_RAISE(auto preflight_estimate,
        EmitSignalAnalysisMemoryPreflight(
            input, signal_col_, GetName(), 5,
            "reduce signal rows, cap max_lag, run ACF on a sampled/windowed signal first, or use a future chunked correlation path.",
            GetMaterializationMemoryContext(),
            progress_callback_, planned_samples));

    ReportProgress(progress_callback_, "Reading signal",
                   "Reading ACF signal column '" + signal_col_ + "'",
                   0.10,
                   0,
                   planned_samples,
                   preflight_estimate.estimated_peak_bytes);
    std::vector<double> signal;
    ARROW_RETURN_NOT_OK(ReadSignalAsDouble(input, signal_col_, GetName(), signal));

    ReportProgress(progress_callback_, "Computing ACF",
                   "Computing autocorrelation function",
                   0.50,
                   0,
                   static_cast<uint64_t>(signal.size()),
                   preflight_estimate.estimated_peak_bytes);
    auto result = TimeSeries::ComputeACF(signal, max_lag_);
    if (!result.success) {
        return arrow::Status::ExecutionError(GetName() + ": " + result.error_message);
    }

    const int n = static_cast<int>(result.acf.size());
    std::vector<int32_t> lags;
    lags.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) lags.push_back(i);

    ARROW_ASSIGN_OR_RAISE(auto lag_arr, BuildInt32Array(lags));
    ARROW_ASSIGN_OR_RAISE(auto acf_arr, BuildDoubleArray(result.acf));
    ARROW_ASSIGN_OR_RAISE(auto lower_arr, BuildDoubleArray(result.confidence_lower));
    ARROW_ASSIGN_OR_RAISE(auto upper_arr, BuildDoubleArray(result.confidence_upper));
    ARROW_ASSIGN_OR_RAISE(auto sig_arr, BuildBoolArray(
        MarkSignificant(n, result.significant_acf_lags)));

    auto schema = arrow::schema({
        arrow::field("lag", arrow::int32()),
        arrow::field("acf", arrow::float64()),
        arrow::field("confidence_lower", arrow::float64()),
        arrow::field("confidence_upper", arrow::float64()),
        arrow::field("significant", arrow::boolean()),
    });
    ReportProgress(progress_callback_, "Complete",
                   "ACF materialization complete",
                   1.0,
                   static_cast<uint64_t>(n),
                   static_cast<uint64_t>(n),
                   static_cast<uint64_t>(n) *
                   (sizeof(int32_t) + 3 * sizeof(double) + sizeof(bool)));
    return arrow::Table::Make(schema, {lag_arr, acf_arr, lower_arr, upper_arr, sig_arr});
}

// ============================================================================
// PACFOperator
// ============================================================================

bool PACFOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    signal_col_.clear();
    max_lag_ = -1;

    if (!ParseSignalColumn(params, GetName(), signal_col_, error)) return false;
    if (!ParseIntOptional(params, "max_lag", max_lag_, GetName(), error)) return false;
    if (!ParseIntOptional(params, "lags", max_lag_, GetName(), error)) return false;
    if (max_lag_ == 0 || max_lag_ < -1) {
        error = GetName() + ": max_lag must be -1 or >= 1";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
PACFOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz PACF Materializer");

    if (!input) return arrow::Status::Invalid(GetName() + ": input is null");

    uint64_t planned_samples = 0;
    ARROW_ASSIGN_OR_RAISE(auto preflight_estimate,
        EmitSignalAnalysisMemoryPreflight(
            input, signal_col_, GetName(), 5,
            "reduce signal rows, cap max_lag, run PACF on a sampled/windowed signal first, or use a future chunked correlation path.",
            GetMaterializationMemoryContext(),
            progress_callback_, planned_samples));

    ReportProgress(progress_callback_, "Reading signal",
                   "Reading PACF signal column '" + signal_col_ + "'",
                   0.10,
                   0,
                   planned_samples,
                   preflight_estimate.estimated_peak_bytes);
    std::vector<double> signal;
    ARROW_RETURN_NOT_OK(ReadSignalAsDouble(input, signal_col_, GetName(), signal));

    ReportProgress(progress_callback_, "Computing PACF",
                   "Computing partial autocorrelation function",
                   0.50,
                   0,
                   static_cast<uint64_t>(signal.size()),
                   preflight_estimate.estimated_peak_bytes);
    auto result = TimeSeries::ComputePACF(signal, max_lag_);
    if (!result.success) {
        return arrow::Status::ExecutionError(GetName() + ": " + result.error_message);
    }

    const int n = static_cast<int>(result.pacf.size());
    std::vector<int32_t> lags;
    lags.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) lags.push_back(i);

    ARROW_ASSIGN_OR_RAISE(auto lag_arr, BuildInt32Array(lags));
    ARROW_ASSIGN_OR_RAISE(auto pacf_arr, BuildDoubleArray(result.pacf));
    ARROW_ASSIGN_OR_RAISE(auto lower_arr, BuildDoubleArray(result.confidence_lower));
    ARROW_ASSIGN_OR_RAISE(auto upper_arr, BuildDoubleArray(result.confidence_upper));
    ARROW_ASSIGN_OR_RAISE(auto sig_arr, BuildBoolArray(
        MarkSignificant(n, result.significant_pacf_lags)));

    auto schema = arrow::schema({
        arrow::field("lag", arrow::int32()),
        arrow::field("pacf", arrow::float64()),
        arrow::field("confidence_lower", arrow::float64()),
        arrow::field("confidence_upper", arrow::float64()),
        arrow::field("significant", arrow::boolean()),
    });
    ReportProgress(progress_callback_, "Complete",
                   "PACF materialization complete",
                   1.0,
                   static_cast<uint64_t>(n),
                   static_cast<uint64_t>(n),
                   static_cast<uint64_t>(n) *
                   (sizeof(int32_t) + 3 * sizeof(double) + sizeof(bool)));
    return arrow::Table::Make(schema, {lag_arr, pacf_arr, lower_arr, upper_arr, sig_arr});
}

// ============================================================================
// StationarityTestOperator
// ============================================================================

bool StationarityTestOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    signal_col_.clear();
    max_lags_ = -1;

    if (!ParseSignalColumn(params, GetName(), signal_col_, error)) return false;
    if (!ParseIntOptional(params, "max_lags", max_lags_, GetName(), error)) return false;
    if (max_lags_ < -1) {
        error = GetName() + ": max_lags must be -1 or >= 0";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
StationarityTestOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz StationarityTest Materializer");

    if (!input) return arrow::Status::Invalid(GetName() + ": input is null");

    uint64_t planned_samples = 0;
    ARROW_ASSIGN_OR_RAISE(auto preflight_estimate,
        EmitSignalAnalysisMemoryPreflight(
            input, signal_col_, GetName(), 3,
            "reduce signal rows, cap max_lags, test a sampled/windowed signal first, or use a future chunked stationarity path.",
            GetMaterializationMemoryContext(),
            progress_callback_, planned_samples));

    ReportProgress(progress_callback_, "Reading signal",
                   "Reading StationarityTest signal column '" + signal_col_ + "'",
                   0.10,
                   0,
                   planned_samples,
                   preflight_estimate.estimated_peak_bytes);
    std::vector<double> signal;
    ARROW_RETURN_NOT_OK(ReadSignalAsDouble(input, signal_col_, GetName(), signal));

    ReportProgress(progress_callback_, "Testing stationarity",
                   "Running stationarity tests",
                   0.55,
                   0,
                   static_cast<uint64_t>(signal.size()),
                   preflight_estimate.estimated_peak_bytes);
    auto result = TimeSeries::TestStationarity(signal, max_lags_);
    if (!result.success) {
        return arrow::Status::ExecutionError(GetName() + ": " + result.error_message);
    }

    ARROW_ASSIGN_OR_RAISE(auto adf_stat, BuildDoubleArray({result.adf_statistic}));
    ARROW_ASSIGN_OR_RAISE(auto adf_p, BuildDoubleArray({result.adf_pvalue}));
    ARROW_ASSIGN_OR_RAISE(auto adf_st, BuildBoolArray({result.adf_stationary}));
    ARROW_ASSIGN_OR_RAISE(auto kpss_stat, BuildDoubleArray({result.kpss_statistic}));
    ARROW_ASSIGN_OR_RAISE(auto kpss_p, BuildDoubleArray({result.kpss_pvalue}));
    ARROW_ASSIGN_OR_RAISE(auto kpss_st, BuildBoolArray({result.kpss_stationary}));
    ARROW_ASSIGN_OR_RAISE(auto stationary, BuildBoolArray({result.is_stationary}));
    ARROW_ASSIGN_OR_RAISE(auto diff, BuildInt32Array(
        {static_cast<int32_t>(result.suggested_differencing)}));
    ARROW_ASSIGN_OR_RAISE(auto rolling, BuildInt32Array(
        {static_cast<int32_t>(result.rolling_window)}));
    ARROW_ASSIGN_OR_RAISE(auto analysis, BuildStringArray({result.analysis}));

    auto schema = arrow::schema({
        arrow::field("adf_statistic", arrow::float64()),
        arrow::field("adf_pvalue", arrow::float64()),
        arrow::field("adf_stationary", arrow::boolean()),
        arrow::field("kpss_statistic", arrow::float64()),
        arrow::field("kpss_pvalue", arrow::float64()),
        arrow::field("kpss_stationary", arrow::boolean()),
        arrow::field("is_stationary", arrow::boolean()),
        arrow::field("suggested_differencing", arrow::int32()),
        arrow::field("rolling_window", arrow::int32()),
        arrow::field("analysis", arrow::utf8()),
    });
    ReportProgress(progress_callback_, "Complete",
                   "StationarityTest materialization complete",
                   1.0,
                   1,
                   1);
    return arrow::Table::Make(schema, {
        adf_stat, adf_p, adf_st, kpss_stat, kpss_p, kpss_st,
        stationary, diff, rolling, analysis});
}

// ============================================================================
// SeasonalityDetectorOperator
// ============================================================================

bool SeasonalityDetectorOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
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

arrow::Result<std::shared_ptr<arrow::Table>>
SeasonalityDetectorOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz SeasonalityDetector Materializer");

    if (!input) return arrow::Status::Invalid(GetName() + ": input is null");

    uint64_t planned_samples = 0;
    ARROW_ASSIGN_OR_RAISE(auto preflight_estimate,
        EmitSignalAnalysisMemoryPreflight(
            input, signal_col_, GetName(), 3,
            "reduce signal rows, narrow the period search, detect seasonality on a sampled/windowed signal first, or use a future chunked seasonality path.",
            GetMaterializationMemoryContext(),
            progress_callback_, planned_samples));

    ReportProgress(progress_callback_, "Reading signal",
                   "Reading SeasonalityDetector signal column '" +
                   signal_col_ + "'",
                   0.10,
                   0,
                   planned_samples,
                   preflight_estimate.estimated_peak_bytes);
    std::vector<double> signal;
    ARROW_RETURN_NOT_OK(ReadSignalAsDouble(input, signal_col_, GetName(), signal));

    ReportProgress(progress_callback_, "Detecting seasonality",
                   "Detecting candidate seasonal periods",
                   0.55,
                   0,
                   static_cast<uint64_t>(signal.size()),
                   preflight_estimate.estimated_peak_bytes);
    auto result = TimeSeries::DetectSeasonality(signal, min_period_, max_period_);
    if (!result.success) {
        return arrow::Status::ExecutionError(GetName() + ": " + result.error_message);
    }

    std::vector<int32_t> periods;
    std::vector<double> strengths;
    if (result.candidate_periods.empty()) {
        periods.push_back(static_cast<int32_t>(result.detected_period));
        strengths.push_back(result.strength);
    } else {
        for (int period : result.candidate_periods) {
            periods.push_back(static_cast<int32_t>(period));
        }
        strengths.reserve(periods.size());
        for (size_t i = 0; i < periods.size(); ++i) {
            strengths.push_back(i < result.candidate_strengths.size()
                                    ? result.candidate_strengths[i]
                                    : 0.0);
        }
    }

    std::vector<bool> primary(periods.size(), false);
    for (size_t i = 0; i < periods.size(); ++i) {
        primary[i] = periods[i] == result.detected_period;
    }
    std::vector<bool> has_seasonality(periods.size(), result.has_seasonality);
    std::vector<std::string> analysis(periods.size(), result.analysis);

    ARROW_ASSIGN_OR_RAISE(auto period_arr, BuildInt32Array(periods));
    ARROW_ASSIGN_OR_RAISE(auto strength_arr, BuildDoubleArray(strengths));
    ARROW_ASSIGN_OR_RAISE(auto primary_arr, BuildBoolArray(primary));
    ARROW_ASSIGN_OR_RAISE(auto has_arr, BuildBoolArray(has_seasonality));
    ARROW_ASSIGN_OR_RAISE(auto analysis_arr, BuildStringArray(analysis));

    auto schema = arrow::schema({
        arrow::field("period", arrow::int32()),
        arrow::field("strength", arrow::float64()),
        arrow::field("is_primary", arrow::boolean()),
        arrow::field("has_seasonality", arrow::boolean()),
        arrow::field("analysis", arrow::utf8()),
    });
    ReportProgress(progress_callback_, "Complete",
                   "SeasonalityDetector materialization complete",
                   1.0,
                   static_cast<uint64_t>(periods.size()),
                   static_cast<uint64_t>(periods.size()));
    return arrow::Table::Make(schema, {
        period_arr, strength_arr, primary_arr, has_arr, analysis_arr});
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
