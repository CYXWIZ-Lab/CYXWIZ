#include "series_explore.h"

#include "explore_read.h"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <optional>

namespace cyxwiz::series {

namespace {

using explore_read::NumberExpr;
using explore_read::Numbers;

constexpr double kDay = kDaySeconds;

std::string Ident(const std::string& name) {
    std::string out = "\"";
    for (const char c : name) out += c == '"' ? "\"\"" : std::string(1, c);
    return out + "\"";
}

// DuckDB's type of a column ("" when the table or column is not there).
std::string ColumnType(const std::string& table, const std::string& column, const QueryRunner& run) {
    QueryRequest request;
    request.sql = "SELECT typeof(" + Ident(column) + ") AS t FROM " + Ident(table) + " LIMIT 1";
    request.label = "Series: the time column";
    const QueryResult result = run(request);
    if (!result.ok || !result.table || result.table->num_rows() == 0) return "";
    const auto labels = explore_read::Labels(*result.table, "t");
    return labels.empty() || !labels[0] ? "" : *labels[0];
}

bool IsNumericType(const std::string& type) {
    static const char* const kNumeric[] = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT",
                                           "UINTEGER", "UBIGINT", "FLOAT", "DOUBLE", "DECIMAL"};
    for (const char* n : kNumeric)
        if (type.rfind(n, 0) == 0) return true;
    return false;
}

bool IsDateType(const std::string& type) { return type.rfind("DATE", 0) == 0 || type.rfind("TIMESTAMP", 0) == 0; }

// The time column as seconds since 1970: a date column as it is, a text by the
// common shapes (ISO date-times, YYYY-MM, YYYY/MM/DD, DD/MM/YYYY, YYYY), NULL otherwise.
std::string TimeExpr(const std::string& column, const std::string& type) {
    const std::string c = Ident(column);
    if (IsDateType(type)) return "epoch(CAST(" + c + " AS TIMESTAMP))";
    return "epoch(COALESCE(TRY_CAST(" + c + " AS TIMESTAMP), try_strptime(" + c +
           ", ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M', '%Y-%m-%d %H:%M', '%Y-%m', '%Y/%m/%d', '%d/%m/%Y', '%d.%m.%Y', '%Y'])))";
}

// Period from the series: the strongest non-harmonic peak of the detrended
// periodogram when a seasonal pattern of that length explains enough, else the step's.
void FindPeriod(Series& s, int typed) {
    if (typed > 0) {
        s.period = typed;
        return;
    }
    if (s.values.size() >= 8) {
        const SeasonalityResult season = TimeSeries::DetectSeasonality(s.values);
        if (season.success && season.has_seasonality && static_cast<int>(s.values.size()) >= 2 * season.detected_period) {
            s.period = season.detected_period;
            s.period_found = true;
            s.period_share = season.detected_share;
            return;
        }
    }
    if (s.step.natural_period >= 2 && static_cast<int>(s.values.size()) >= 2 * s.step.natural_period) s.period = s.step.natural_period;
}

std::vector<double> Padded(const std::vector<double>& short_series, size_t n) {
    std::vector<double> out(n, NAN);
    const size_t offset = n - std::min(n, short_series.size());
    for (size_t i = 0; i < short_series.size() && offset + i < n; ++i) out[offset + i] = short_series[i];
    return out;
}

ForecastResult RunMethod(const std::vector<double>& y, const ForecastRequest& r, Method method, int horizon, int period) {
    switch (method) {
        case Method::Simple: return TimeSeries::SimpleES(y, horizon, r.alpha, r.level);
        case Method::Holt: return TimeSeries::HoltLinear(y, horizon, r.alpha, r.beta, r.damped, r.level);
        case Method::HoltWinters: return TimeSeries::HoltWinters(y, horizon, period, r.multiplicative ? "multiplicative" : "additive", r.alpha, r.beta, r.gamma, r.level);
        case Method::Arima: return TimeSeries::ARIMA(y, horizon, r.p, r.d, r.q, r.level);
        case Method::SeasonalNaive: return TimeSeries::SeasonalNaive(y, horizon, std::max(period, 1), r.level);
    }
    return {};
}

}  // namespace

const char* MethodName(Method m) {
    switch (m) {
        case Method::Simple: return "Simple exponential smoothing";
        case Method::Holt: return "Holt";
        case Method::HoltWinters: return "Holt-Winters";
        case Method::Arima: return "ARIMA";
        case Method::SeasonalNaive: return "Seasonal naive";
    }
    return "";
}

double Mape(const std::vector<double>& actual, const std::vector<double>& forecast) {
    double sum = 0.0;
    int count = 0;
    for (size_t i = 0; i < actual.size() && i < forecast.size(); ++i)
        if (std::fabs(actual[i]) > 1e-12) {
            sum += std::fabs((actual[i] - forecast[i]) / actual[i]);
            ++count;
        }
    return count > 0 ? 100.0 * sum / count : NAN;
}

double Rmse(const std::vector<double>& actual, const std::vector<double>& forecast) {
    double sum = 0.0;
    size_t count = std::min(actual.size(), forecast.size());
    for (size_t i = 0; i < count; ++i) sum += (actual[i] - forecast[i]) * (actual[i] - forecast[i]);
    return count > 0 ? std::sqrt(sum / static_cast<double>(count)) : NAN;
}

// ---- reading

SeriesAnswer ReadSeries(const std::string& table, const SeriesRequest& request, const QueryRunner& run, const explore::MissingTexts& texts) {
    SeriesAnswer a;
    Series& s = a.series;
    if (request.value.empty()) return a.error = "Pick the value column.", a;
    if (request.time == request.value) return a.error = "The time and the value are the same column.", a;
    std::vector<QueryParam> params;
    std::vector<std::string> exprs;
    std::string type;
    if (!request.time.empty()) {
        type = ColumnType(table, request.time, run);
        if (type.empty()) return a.error = "The time column \"" + request.time + "\" was not found.", a;
        s.kind = IsNumericType(type) ? TimeKind::Numbers : TimeKind::Dates;
        exprs.push_back((s.kind == TimeKind::Numbers ? "CAST(" + Ident(request.time) + " AS DOUBLE)" : TimeExpr(request.time, type)) + " AS t");
    }
    exprs.push_back(NumberExpr(request.value, texts, params, "v"));
    exprs.push_back("CAST(row_number() OVER () AS DOUBLE) AS r");
    QueryRequest query;
    query.sql = "SELECT " + exprs[0];
    for (size_t i = 1; i < exprs.size(); ++i) query.sql += ", " + exprs[i];
    query.sql += " FROM " + Ident(table);
    if (!request.time.empty()) query.sql += " ORDER BY t NULLS LAST, r";
    query.params = std::move(params);
    query.label = "Series: " + request.value;
    const QueryResult result = run(query);
    if (!result.ok) return a.error = result.error.empty() ? "Cancelled." : result.error, a;
    if (!result.table) return a.error = "The query returned nothing.", a;
    s.rows = result.table->num_rows();
    const auto t = request.time.empty() ? std::vector<std::optional<double>>{} : Numbers(*result.table, "t");
    const auto v = Numbers(*result.table, "v");
    const auto order = Numbers(*result.table, "r");
    int64_t missing_time = 0;
    double last_order = -1.0;
    for (int64_t i = 0; i < s.rows; ++i) {
        const size_t k = static_cast<size_t>(i);
        if (k >= v.size() || !v[k]) continue;
        if (!request.time.empty()) {
            if (k >= t.size() || !t[k]) {
                ++missing_time;
                continue;
            }
            s.t.push_back(*t[k]);
        } else {
            s.t.push_back(static_cast<double>(s.values.size()));
        }
        s.values.push_back(*v[k]);
        if (k < order.size() && order[k]) {
            if (*order[k] < last_order) ++s.unordered;
            last_order = *order[k];
        }
    }
    s.used = static_cast<int64_t>(s.values.size());
    if (!request.time.empty() && s.kind == TimeKind::Dates && s.used > 0) {
        // A text column that mostly does not read as dates is not a time column.
        int64_t with_value = 0;
        for (const auto& value : v)
            if (value) ++with_value;
        if (missing_time > with_value / 10)
            return a.error = "\"" + request.time + "\" does not read as dates (" + std::to_string(missing_time) + " of " + std::to_string(with_value) +
                             " rows have no date); pick a date column, or the row order.", a;
    }
    if (s.used < 8) return a.error = std::to_string(s.used) + " rows have both the time and the value: too few (8 or more).", a;
    s.step = StepOf(s.t, s.kind);
    if (s.kind == TimeKind::Dates)
        for (const double seconds : s.t) s.labels.push_back(DateLabel(seconds, s.step));
    FindPeriod(s, request.period);
    return a;
}

// ---- the answers

PartsAnswer RunParts(const std::string& table, const PartsRequest& request, const QueryRunner& run, const explore::MissingTexts& texts) {
    PartsAnswer a;
    a.request = request;
    const SeriesAnswer read = ReadSeries(table, request.base, run, texts);
    if (!read.ok()) return a.error = read.error, a;
    a.series = read.series;
    if (a.series.period < 2) return a.error = "No seasonal period was found; type one to split the series into its parts.", a;
    if (static_cast<int64_t>(a.series.values.size()) < 2 * a.series.period)
        return a.error = "The series needs two full periods (" + std::to_string(2 * a.series.period) + " rows).", a;
    a.parts = request.stl ? TimeSeries::STLDecompose(a.series.values, a.series.period, request.seasonal_window, -1, -1, request.robust)
                          : TimeSeries::Decompose(a.series.values, a.series.period, request.multiplicative ? "multiplicative" : "additive");
    if (!a.parts.success) a.error = a.parts.error_message;
    return a;
}

SeasonalityAnswer RunSeasonality(const std::string& table, const SeriesRequest& request, const QueryRunner& run, const explore::MissingTexts& texts) {
    SeasonalityAnswer a;
    a.request = request;
    const SeriesAnswer read = ReadSeries(table, request, run, texts);
    if (!read.ok()) return a.error = read.error, a;
    a.series = read.series;
    a.seasonality = TimeSeries::DetectSeasonality(a.series.values);
    if (!a.seasonality.success) return a.error = a.seasonality.error_message, a;
    const int period = a.series.period;
    if (period < 2 || static_cast<int64_t>(a.series.values.size()) < 2 * period) return a;
    // The season at each position: the mean of the detrended series over the
    // rows at that position. A calendar step places each row by its date
    // (the month, the weekday), so a missing row does not shift the rest; any
    // other step counts positions along the rows.
    const DecompositionResult d = TimeSeries::Decompose(a.series.values, period, "additive");
    if (!d.success) return a;
    static const char* const kMonths[12] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
    static const char* const kDays[7] = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"};
    const bool by_month = a.series.kind == TimeKind::Dates && a.series.step.name == "monthly" && period == 12;
    const bool by_weekday = a.series.kind == TimeKind::Dates && a.series.step.name == "daily" && period == 7;
    std::vector<double> sum(static_cast<size_t>(period), 0.0);
    std::vector<int> count(static_cast<size_t>(period), 0);
    for (size_t i = 0; i < a.series.values.size(); ++i) {
        if (!std::isfinite(d.trend[i])) continue;
        size_t position = i % static_cast<size_t>(period);
        if (by_month) {
            int y, m, dd, hh, mm, ss;
            SecondsToCivil(a.series.t[i], y, m, dd, hh, mm, ss);
            position = static_cast<size_t>(m - 1);
        } else if (by_weekday) {
            const long long day = static_cast<long long>(std::floor(a.series.t[i] / kDay));
            position = static_cast<size_t>(((day + 3) % 7 + 7) % 7);  // 1970-01-01 was a Thursday
        }
        sum[position] += a.series.values[i] - d.trend[i];
        ++count[position];
    }
    double mean = 0.0;
    int filled = 0;
    for (int i = 0; i < period; ++i)
        if (count[static_cast<size_t>(i)] > 0) {
            sum[static_cast<size_t>(i)] /= count[static_cast<size_t>(i)];
            mean += sum[static_cast<size_t>(i)];
            ++filled;
        }
    if (filled > 0) mean /= filled;
    for (int i = 0; i < period; ++i) {
        a.season_by_position.push_back(count[static_cast<size_t>(i)] > 0 ? sum[static_cast<size_t>(i)] - mean : NAN);
        a.position_names.push_back(by_month ? kMonths[i] : by_weekday ? kDays[i] : std::to_string(i + 1));
    }
    return a;
}

StationarityAnswer RunStationarity(const std::string& table, const StationarityRequest& request, const QueryRunner& run, const explore::MissingTexts& texts) {
    StationarityAnswer a;
    a.request = request;
    const SeriesAnswer read = ReadSeries(table, request.base, run, texts);
    if (!read.ok()) return a.error = read.error, a;
    a.series = read.series;
    a.tests = TimeSeries::TestStationarity(a.series.values, -1, request.trend ? "ct" : "c");
    if (!a.tests.success) return a.error = a.tests.error_message, a;
    const int n = static_cast<int>(a.series.values.size());
    a.window = request.window > 0 ? request.window : (a.series.period >= 2 ? a.series.period : std::max(10, n / 10));
    a.window = std::min(a.window, n);
    a.rolling_mean = Padded(TimeSeries::RollingMean(a.series.values, a.window), static_cast<size_t>(n));
    a.rolling_sd = Padded(TimeSeries::RollingStd(a.series.values, a.window), static_cast<size_t>(n));
    return a;
}

CorrelogramAnswer RunCorrelogram(const std::string& table, const CorrelogramRequest& request, const QueryRunner& run, const explore::MissingTexts& texts) {
    CorrelogramAnswer a;
    a.request = request;
    const SeriesAnswer read = ReadSeries(table, request.base, run, texts);
    if (!read.ok()) return a.error = read.error, a;
    a.series = read.series;
    const std::vector<double> y = request.differenced ? TimeSeries::Difference(a.series.values, 1) : a.series.values;
    const int lags = std::max(1, std::min(request.lags, static_cast<int>(y.size()) / 2 - 1));
    a.correlogram = TimeSeries::ComputeACFPACF(y, lags, 0.95);
    if (!a.correlogram.success) a.error = a.correlogram.error_message;
    return a;
}

ForecastAnswer RunForecast(const std::string& table, const ForecastRequest& request, const QueryRunner& run, const explore::MissingTexts& texts) {
    ForecastAnswer a;
    a.request = request;
    const SeriesAnswer read = ReadSeries(table, request.base, run, texts);
    if (!read.ok()) return a.error = read.error, a;
    a.series = read.series;
    if (request.horizon < 1 || request.horizon > 10000) return a.error = "The horizon must be 1 to 10,000 steps.", a;
    const int period = a.series.period;
    const bool seasonal = request.method == Method::HoltWinters || request.method == Method::SeasonalNaive;
    if (seasonal && period < 2) return a.error = std::string(MethodName(request.method)) + " needs a seasonal period; none was found, so type one.", a;
    const std::vector<double>& y = a.series.values;
    const int n = static_cast<int>(y.size());
    a.fit = RunMethod(y, request, request.method, request.horizon, period);
    if (!a.fit.success) return a.error = a.fit.error_message, a;
    a.future_t = FutureTimes(a.series.t, a.series.step, a.series.kind, request.horizon);
    for (const double t : a.future_t)
        a.future_labels.push_back(a.series.kind == TimeKind::Dates ? DateLabel(t, a.series.step) : Trim(t + (a.series.kind == TimeKind::Order ? 1.0 : 0.0)));

    // The held-out check: every method on the first n - H rows, scored on the last H.
    const int minimum = std::max(period >= 2 ? 2 * period : 0, 10);
    // The held-out check (D6-6): the last H rows; by default twice the period (else 12), at most a third of the rows.
    int holdout = request.holdout;
    if (holdout < 0) holdout = std::min(period >= 2 ? 2 * period : 12, static_cast<int>(n) / 3);
    if (holdout > 0 && n - holdout < minimum)
        a.holdout_note = "The last " + std::to_string(holdout) + " rows cannot be held out: fewer than " + std::to_string(minimum) + " rows would be left to fit on.";
    if (holdout > 0 && n - holdout >= minimum) {
        a.holdout = holdout;
        const std::vector<double> train(y.begin(), y.end() - a.holdout);
        a.actual.assign(y.end() - a.holdout, y.end());
        std::vector<Method> methods = {request.method, Method::Simple, Method::Holt};
        if (period >= 2) methods.push_back(Method::HoltWinters);
        if (request.method != Method::Arima && request.p >= 0 && request.q >= 0) methods.push_back(Method::Arima);
        if (period >= 2) methods.push_back(Method::SeasonalNaive);
        for (size_t i = 0; i < methods.size(); ++i) {
            bool seen = false;
            for (size_t j = 0; j < i; ++j) seen = seen || methods[j] == methods[i];
            if (seen) continue;
            HoldoutScore score;
            score.method = methods[i];
            score.name = MethodName(methods[i]);
            if (methods[i] == Method::Arima && request.method == Method::Arima) {
                // The same orders the full fit chose.
                ForecastRequest fixed = request;
                const auto& p = a.fit.parameters;
                int pp = 0, dd = 0, qq = 0;
                std::sscanf(a.fit.method.c_str(), "ARIMA(%d,%d,%d)", &pp, &dd, &qq);
                (void)p;
                fixed.p = pp;
                fixed.d = dd;
                fixed.q = qq;
                const ForecastResult f = RunMethod(train, fixed, Method::Arima, a.holdout, period);
                if (f.success) score.forecast = f.forecast;
                else score.error = f.error_message;
            } else {
                const ForecastResult f = RunMethod(train, request, methods[i], a.holdout, period);
                if (f.success) score.forecast = f.forecast;
                else score.error = f.error_message;
            }
            if (score.error.empty()) {
                score.mape = Mape(a.actual, score.forecast);
                score.rmse = Rmse(a.actual, score.forecast);
                double abs_sum = 0.0;
                for (size_t k = 0; k < a.actual.size() && k < score.forecast.size(); ++k) abs_sum += std::fabs(a.actual[k] - score.forecast[k]);
                score.mae = abs_sum / static_cast<double>(a.actual.size());
            }
            a.scores.push_back(std::move(score));
        }
    }
    return a;
}

// ---- into the graph

namespace {
std::string Number(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.6g", v);
    return buf;
}
}  // namespace

std::map<std::string, std::string> NodeParameters(const PartsRequest& r) {
    return {{"signal_col", r.base.value},
            {"period", std::to_string(r.base.period > 0 ? r.base.period : -1)},
            {"algorithm", r.stl ? "stl" : "classical"},
            {"method", r.multiplicative ? "multiplicative" : "additive"},
            {"seasonal_window", std::to_string(r.seasonal_window)},
            {"robust", r.robust ? "true" : "false"}};
}

std::map<std::string, std::string> NodeParameters(const SeriesRequest& r) {
    return {{"signal_col", r.value}, {"min_period", "2"}, {"max_period", "-1"}};
}

std::map<std::string, std::string> NodeParameters(const StationarityRequest& r) {
    return {{"signal_col", r.base.value}, {"regression", r.trend ? "ct" : "c"}, {"max_lags", "-1"}};
}

std::map<std::string, std::string> NodeParameters(const CorrelogramRequest& r) {
    return {{"signal_col", r.base.value}, {"max_lag", std::to_string(r.lags)}, {"confidence", "0.95"}};
}

std::map<std::string, std::string> NodeParameters(const ForecastRequest& r) {
    std::map<std::string, std::string> p = {{"signal_col", r.base.value}, {"time_col", r.base.time}, {"horizon", std::to_string(r.horizon)}, {"level", Number(r.level)}};
    switch (r.method) {
        case Method::Arima:
            p["p"] = std::to_string(r.p);
            p["d"] = std::to_string(r.d);
            p["q"] = std::to_string(r.q);
            break;
        case Method::Simple:
        case Method::Holt:
        case Method::HoltWinters:
            p["method"] = r.method == Method::Simple ? "simple" : r.method == Method::Holt ? "holt" : "holt_winters";
            p["seasonal"] = r.multiplicative ? "multiplicative" : "additive";
            p["period"] = std::to_string(r.base.period > 0 ? r.base.period : -1);
            p["damped"] = r.damped ? "true" : "false";
            p["alpha"] = Number(r.alpha);
            p["beta"] = Number(r.beta);
            p["gamma"] = Number(r.gamma);
            break;
        case Method::SeasonalNaive: return {};
    }
    return p;
}

const char* ForecastNodeType(Method method) {
    switch (method) {
        case Method::Arima: return "ARIMAForecaster";
        case Method::Simple:
        case Method::Holt:
        case Method::HoltWinters: return "ExponentialSmoothing";
        case Method::SeasonalNaive: return "";
    }
    return "";
}

}  // namespace cyxwiz::series
