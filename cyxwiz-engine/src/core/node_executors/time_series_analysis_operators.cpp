#include "time_series_analysis_operators.h"
#include "ts_column_utils.h"

#include <cyxwiz/time_series.h>

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

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

} // namespace

// ============================================================================
// TimeSeriesDecompositionOperator
// ============================================================================

bool TimeSeriesDecompositionOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
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
        method_ = m->second;
        if (method_ != "additive" && method_ != "multiplicative") {
            error = GetName() + ": 'method' must be 'additive' / "
                    "'multiplicative' (got '" + method_ + "')";
            return false;
        }
    }

    auto a = params.find("algorithm");
    if (a != params.end() && !a->second.empty()) {
        algorithm_ = a->second;
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
    if (!input) return arrow::Status::Invalid(GetName() + ": input is null");

    std::vector<double> signal;
    ARROW_RETURN_NOT_OK(ReadSignalAsDouble(input, signal_col_, GetName(), signal));

    if (static_cast<int>(signal.size()) < 2 * period_) {
        return arrow::Status::Invalid(
            GetName() + ": need at least 2*period (" +
            std::to_string(2 * period_) + ") samples, got " +
            std::to_string(signal.size()));
    }

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
    auto out = input;
    ARROW_ASSIGN_OR_RAISE(out, AppendF64Column(out, "trend", result.trend));
    ARROW_ASSIGN_OR_RAISE(out, AppendF64Column(out, "seasonal", result.seasonal));
    ARROW_ASSIGN_OR_RAISE(out, AppendF64Column(out, "residual", result.residual));

    spdlog::info("{}: n={} period={} method={} algorithm={} "
                 "trend_strength={:.4f} seasonal_strength={:.4f}",
                 GetName(), signal.size(), period_, method_, algorithm_,
                 result.trend_strength, result.seasonal_strength);
    return out;
}

// ============================================================================
// ARIMAOperator
// ============================================================================

bool ARIMAOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
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
    if (!input) return arrow::Status::Invalid(GetName() + ": input is null");

    std::vector<double> signal;
    ARROW_RETURN_NOT_OK(ReadSignalAsDouble(input, signal_col_, GetName(), signal));

    // horizon=0 so the backend produces in-sample fitted values only.
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

    auto out = input;
    ARROW_ASSIGN_OR_RAISE(out, AppendF64Column(out, "fitted", result.fitted_values));
    ARROW_ASSIGN_OR_RAISE(out, AppendF64Column(out, "residual", residual));

    spdlog::info("{}: n={} p={} d={} q={} RMSE={:.4f} AIC={:.4f} BIC={:.4f}",
                 GetName(), signal.size(), p_, d_, q_,
                 result.rmse, result.aic, result.bic);
    return out;
}

// ============================================================================
// ExponentialSmoothingOperator
// ============================================================================

bool ExponentialSmoothingOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    auto s = params.find("signal_col");
    if (s == params.end() || s->second.empty()) {
        error = GetName() + ": 'signal_col' parameter is required";
        return false;
    }
    signal_col_ = s->second;

    auto m = params.find("method");
    if (m != params.end() && !m->second.empty()) {
        method_ = m->second;
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
    if (d != params.end()) damped_ = (d->second == "true");

    if (method_ == "holt_winters" && period_ < 2 && period_ != -1) {
        error = GetName() + ": holt_winters requires 'period' >= 2 (or -1 for auto-detect)";
        return false;
    }
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
ExponentialSmoothingOperator::Apply(
    const std::shared_ptr<arrow::Table>& input) {
    if (!input) return arrow::Status::Invalid(GetName() + ": input is null");

    std::vector<double> signal;
    ARROW_RETURN_NOT_OK(ReadSignalAsDouble(input, signal_col_, GetName(), signal));

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

    auto out = input;
    ARROW_ASSIGN_OR_RAISE(out, AppendF64Column(out, "fitted", result.fitted_values));
    ARROW_ASSIGN_OR_RAISE(out, AppendF64Column(out, "residual", residual));

    spdlog::info("{}: n={} method={} RMSE={:.4f} AIC={:.4f}",
                 GetName(), signal.size(), method_, result.rmse, result.aic);
    return out;
}

} // namespace cyxwiz
