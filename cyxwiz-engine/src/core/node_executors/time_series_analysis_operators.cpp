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
    if (!input) return arrow::Status::Invalid(GetName() + ": input is null");

    std::vector<double> signal;
    ARROW_RETURN_NOT_OK(ReadSignalAsDouble(input, signal_col_, GetName(), signal));

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
    if (!input) return arrow::Status::Invalid(GetName() + ": input is null");

    std::vector<double> signal;
    ARROW_RETURN_NOT_OK(ReadSignalAsDouble(input, signal_col_, GetName(), signal));

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
    if (!input) return arrow::Status::Invalid(GetName() + ": input is null");

    std::vector<double> signal;
    ARROW_RETURN_NOT_OK(ReadSignalAsDouble(input, signal_col_, GetName(), signal));

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
    if (!input) return arrow::Status::Invalid(GetName() + ": input is null");

    std::vector<double> signal;
    ARROW_RETURN_NOT_OK(ReadSignalAsDouble(input, signal_col_, GetName(), signal));

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
    return arrow::Table::Make(schema, {
        period_arr, strength_arr, primary_arr, has_arr, analysis_arr});
}

} // namespace cyxwiz
