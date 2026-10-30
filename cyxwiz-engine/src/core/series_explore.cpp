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

constexpr double kDay = 86400.0;
constexpr double kMonth = 30.436875 * kDay;  // the mean Gregorian month

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

double Median(std::vector<double> v) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

std::string Trim(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.6g", v);
    return buf;
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

// ---- dates (proleptic Gregorian; Howard Hinnant's algorithms)

double CivilToSeconds(int y, int m, int d, int hour, int minute, int second) {
    y -= m <= 2;
    const int era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    const long long days = static_cast<long long>(era) * 146097 + static_cast<long long>(doe) - 719468;
    return static_cast<double>(days) * kDay + hour * 3600.0 + minute * 60.0 + second;
}

void SecondsToCivil(double seconds, int& y, int& m, int& d, int& hour, int& minute, int& second) {
    const double days_d = std::floor(seconds / kDay);
    double rest = seconds - days_d * kDay;
    long long z = static_cast<long long>(days_d) + 719468;
    const long long era = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = static_cast<unsigned>(z - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const long long year = static_cast<long long>(yoe) + era * 400;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp = (5 * doy + 2) / 153;
    d = static_cast<int>(doy - (153 * mp + 2) / 5 + 1);
    m = static_cast<int>(mp < 10 ? mp + 3 : mp - 9);
    y = static_cast<int>(year + (m <= 2));
    hour = static_cast<int>(rest / 3600.0);
    rest -= hour * 3600.0;
    minute = static_cast<int>(rest / 60.0);
    second = static_cast<int>(std::lround(rest - minute * 60.0));
}

std::string DateLabel(double seconds, const Step& step) {
    int y, m, d, hh, mm, ss;
    SecondsToCivil(seconds, y, m, d, hh, mm, ss);
    char buf[40];
    if (step.seconds > 0.0 && step.seconds < kDay) std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d", y, m, d, hh, mm);
    else std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d", y, m, d);
    return buf;
}

Step StepOf(const std::vector<double>& t, TimeKind kind) {
    Step s;
    if (kind == TimeKind::Order) {
        s.seconds = 1.0;
        s.even = true;
        return s;
    }
    if (t.size() < 2) return s;
    std::vector<double> gaps(t.size() - 1);
    for (size_t i = 1; i < t.size(); ++i) gaps[i - 1] = t[i] - t[i - 1];
    const double median = Median(gaps);
    if (!(median > 0.0)) {
        s.name = "no step";
        return s;
    }
    double lo = 0.9 * median, hi = 1.1 * median;
    if (kind == TimeKind::Numbers) {
        s.seconds = median;
        s.name = "every " + Trim(median);
    } else {
        const double days = median / kDay;
        if (days >= 27.5 && days <= 31.5) {
            s.name = "monthly";
            s.seconds = kMonth;
            s.natural_period = 12;
            lo = 28.0 * kDay;
            hi = 31.0 * kDay;
        } else if (days >= 89.0 && days <= 93.0) {
            s.name = "quarterly";
            s.seconds = 3.0 * kMonth;
            s.natural_period = 4;
            lo = 89.0 * kDay;
            hi = 93.0 * kDay;
        } else if (days >= 364.0 && days <= 367.0) {
            s.name = "yearly";
            s.seconds = 365.2425 * kDay;
            lo = 364.0 * kDay;
            hi = 367.0 * kDay;
        } else if (std::fabs(days - 7.0) < 0.01) {
            s.name = "weekly";
            s.seconds = 7.0 * kDay;
            s.natural_period = 52;
        } else if (std::fabs(days - 1.0) < 0.01) {
            s.name = "daily";
            s.seconds = kDay;
            s.natural_period = 7;
        } else if (std::fabs(median - 3600.0) < 1.0) {
            s.name = "hourly";
            s.seconds = 3600.0;
            s.natural_period = 24;
        } else if (std::fabs(median - 60.0) < 0.01) {
            s.name = "every minute";
            s.seconds = 60.0;
        } else {
            s.seconds = median;
            if (median >= kDay) s.name = "every " + Trim(days) + " days";
            else if (median >= 3600.0) s.name = "every " + Trim(median / 3600.0) + " hours";
            else if (median >= 60.0) s.name = "every " + Trim(median / 60.0) + " minutes";
            else s.name = "every " + Trim(median) + " seconds";
        }
    }
    size_t within = 0;
    for (const double g : gaps)
        if (g >= lo && g <= hi) ++within;
    s.even = static_cast<double>(within) >= 0.95 * static_cast<double>(gaps.size());
    return s;
}

std::vector<double> FutureTimes(const std::vector<double>& t, const Step& step, TimeKind kind, int count) {
    std::vector<double> out;
    if (t.empty() || count <= 0) return out;
    const double last = t.back();
    const int months = step.name == "monthly" ? 1 : step.name == "quarterly" ? 3 : step.name == "yearly" ? 12 : 0;
    if (kind == TimeKind::Dates && months > 0) {
        int y, m, d, hh, mm, ss;
        SecondsToCivil(last, y, m, d, hh, mm, ss);
        for (int k = 1; k <= count; ++k) {
            int total = (y * 12 + (m - 1)) + months * k;
            const int ny = total / 12, nm = total % 12 + 1;
            static const int kLength[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
            int length = kLength[nm - 1];
            if (nm == 2 && ((ny % 4 == 0 && ny % 100 != 0) || ny % 400 == 0)) length = 29;
            out.push_back(CivilToSeconds(ny, nm, std::min(d, length), hh, mm, ss));
        }
        return out;
    }
    const double size = kind == TimeKind::Order ? 1.0 : step.seconds;
    for (int k = 1; k <= count; ++k) out.push_back(last + size * k);
    return out;
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
    exprs.push_back("row_number() OVER () AS r");
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
    if (request.holdout > 0 && n - request.holdout >= minimum) {
        a.holdout = request.holdout;
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

}  // namespace cyxwiz::series
