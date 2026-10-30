// Data Studio's Explore tabs Series and Forecast (TOFIX129 D6): reading a
// series through the session query service (dates as text or a date column,
// numbers, the row order; rows missing either left out; the table sorted by
// time), the step and the period, the five answers on the airline passengers
// series (their figures are the backend's, checked against statsmodels in
// test_computation_truth_time_series), the held-out check, future dates, and
// the words.

#include "../src/core/series_explore.h"
#include "../src/core/series_presentation.h"
#include "../src/core/session_query_engine.h"

#include <arrow/api.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace cyxwiz;
namespace fs = std::filesystem;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void Near(double a, double b, double tol, const std::string& what) {
    Check(std::fabs(a - b) <= tol * (1.0 + std::fabs(b)), what + ": " + std::to_string(a) + " vs " + std::to_string(b));
}

void Contains(const std::string& text, const std::string& part, const std::string& what) {
    Check(text.find(part) != std::string::npos, what + ": \"" + part + "\" not in \"" + text + "\"");
}

// The airline passengers CSV (Month as YYYY-MM text, Passengers).
void ReadAirline(std::vector<std::string>& months, std::vector<double>& passengers) {
    std::ifstream in(CYXWIZ_AIRLINE_CSV);
    Check(in.good(), std::string("cannot open ") + CYXWIZ_AIRLINE_CSV);
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        const size_t comma = line.find(',');
        if (comma == std::string::npos) continue;
        months.push_back(line.substr(0, comma));
        passengers.push_back(std::stod(line.substr(comma + 1)));
    }
}

// The airline table with a gap (row 50 has no value) and two rows swapped, plus a
// numeric "index" column and a DATE column "day" (daily, 1 to 144).
std::shared_ptr<arrow::Table> AirlineTable(const std::vector<std::string>& months, const std::vector<double>& passengers) {
    arrow::StringBuilder month;
    arrow::DoubleBuilder value, index;
    arrow::Date32Builder day;
    std::vector<size_t> order(months.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::swap(order[20], order[21]);
    for (const size_t i : order) {
        (void)month.Append(months[i]);
        if (i == 50) (void)value.AppendNull();
        else (void)value.Append(passengers[i]);
        (void)index.Append(static_cast<double>(i) * 0.5);
        (void)day.Append(static_cast<int32_t>(i));  // days since 1970-01-01
    }
    std::shared_ptr<arrow::Array> ma, va, ia, da;
    (void)month.Finish(&ma);
    (void)value.Finish(&va);
    (void)index.Finish(&ia);
    (void)day.Finish(&da);
    return arrow::Table::Make(arrow::schema({arrow::field("Month", arrow::utf8()), arrow::field("Passengers", arrow::float64()),
                                             arrow::field("index", arrow::float64()), arrow::field("day", arrow::date32())}),
                              {ma, va, ia, da});
}
}  // namespace

int main() {
    std::vector<std::string> months;
    std::vector<double> passengers;
    ReadAirline(months, passengers);
    Check(months.size() == 144, "144 months");

    // ---- pure helpers
    Near(series::CivilToSeconds(1970, 1, 1), 0.0, 1e-12, "epoch");
    Near(series::CivilToSeconds(2000, 3, 1), 951868800.0, 1e-12, "2000-03-01");
    int y, m, d, hh, mm, ss;
    series::SecondsToCivil(951868800.0 + 3600.0 * 5 + 60.0 * 7, y, m, d, hh, mm, ss);
    Check(y == 2000 && m == 3 && d == 1 && hh == 5 && mm == 7, "civil back");
    std::vector<double> monthly;
    for (int i = 0; i < 24; ++i) monthly.push_back(series::CivilToSeconds(2020 + i / 12, i % 12 + 1, 31 - (i % 12 == 1 ? 2 : 0)));  // month ends, Feb 29
    const series::Step step = series::StepOf(monthly, series::TimeKind::Dates);
    Check(step.name == "monthly" && step.even && step.natural_period == 12, "month ends read as monthly: " + step.name);
    const auto future = series::FutureTimes(monthly, step, series::TimeKind::Dates, 3);
    Check(future.size() == 3 && series::DateLabel(future[0], step) == "2022-01-31" && series::DateLabel(future[1], step) == "2022-02-28" &&
              series::DateLabel(future[2], step) == "2022-03-31",
          "monthly future dates keep the day and clamp February: " + series::DateLabel(future[1], step));
    std::vector<double> daily;
    for (int i = 0; i < 60; ++i) daily.push_back(86400.0 * i);
    const series::Step daily_step = series::StepOf(daily, series::TimeKind::Dates);
    Check(daily_step.name == "daily" && daily_step.natural_period == 7 && daily_step.even, "daily step");
    daily.erase(daily.begin() + 11, daily.begin() + 13);  // two missing days: one gap of 3 in 57
    Check(series::StepOf(daily, series::TimeKind::Dates).name == "daily" && series::StepOf(daily, series::TimeKind::Dates).even, "a few gaps keep it even");
    for (size_t i = 0; i + 1 < daily.size(); i += 2) daily[i + 1] += 86400.0 * 2;  // every other gap is 3 days: uneven
    Check(!series::StepOf(daily, series::TimeKind::Dates).even, "many gaps make it uneven");
    std::vector<double> hourly;
    for (int i = 0; i < 50; ++i) hourly.push_back(3600.0 * i);
    Check(series::StepOf(hourly, series::TimeKind::Dates).name == "hourly", "hourly step");
    Check(series::DateLabel(3600.0 * 13 + 60.0 * 5, series::StepOf(hourly, series::TimeKind::Dates)) == "1970-01-01 13:05", "a label with the time below a day");
    Check(series::StepOf({0.0, 0.5, 1.0, 1.5}, series::TimeKind::Numbers).name == "every 0.5", "numbers step");
    Near(series::Mape({100.0, 200.0}, {110.0, 180.0}), 10.0, 1e-12, "MAPE");
    Near(series::Rmse({1.0, 2.0}, {2.0, 4.0}), std::sqrt(2.5), 1e-12, "RMSE");

    // ---- reading
    const fs::path root = fs::temp_directory_path() / "cyxwiz_test_series_explore";
    std::error_code ec;
    fs::remove_all(root, ec);
    SessionQueryEngine engine(root.string());
    std::string error;
    Check(engine.AttachArrow("airline", 1, AirlineTable(months, passengers), &error), "attach: " + error);
    const QueryRunner run = [&](const QueryRequest& r) { return engine.Run(r); };

    series::SeriesRequest req;
    req.time = "Month";
    req.value = "Passengers";
    const auto read = series::ReadSeries("airline", req, run, {});
    Check(read.ok(), "read: " + read.error);
    const auto& s = read.series;
    Check(s.kind == series::TimeKind::Dates && s.rows == 144 && s.used == 143, "dates, 143 of 144 rows: " + std::to_string(s.used));
    Check(s.step.name == "monthly" && s.step.even && s.labels.front() == "1949-01-01" && s.labels.back() == "1960-12-01", "monthly from YYYY-MM text: " + s.labels.front());
    Check(s.unordered == 1 && std::is_sorted(s.t.begin(), s.t.end()), "sorted by time, the swapped pair counted once: " + std::to_string(s.unordered));
    Check(s.period == 12 && s.period_found && s.period_share > 0.4, "period 12 found: " + std::to_string(s.period) + " share " + std::to_string(s.period_share));
    Check(series_view::SeriesWords(s) == "143 months, monthly, 1949-01 .. 1960-12", "series words: " + series_view::SeriesWords(s));
    Check(series_view::LeftOut(s) == "1 row has no value or time and is left out. The table was not in time order; the rows were sorted.", "left out: " + series_view::LeftOut(s));
    Contains(series_view::PeriodWords(s), "found: 12", "period words");

    req.time = "day";
    const auto by_day = series::ReadSeries("airline", req, run, {});
    Check(by_day.ok() && by_day.series.step.name == "daily" && by_day.series.labels[1] == "1970-01-02" && by_day.series.unordered > 0, "a DATE column: daily, the swapped rows counted: " + by_day.error);
    req.time = "index";
    const auto by_index = series::ReadSeries("airline", req, run, {});
    Check(by_index.ok() && by_index.series.kind == series::TimeKind::Numbers && by_index.series.step.name == "every 0.5", "a number column: " + by_index.error + by_index.series.step.name);
    req.time = "";
    const auto by_order = series::ReadSeries("airline", req, run, {});
    Check(by_order.ok() && by_order.series.kind == series::TimeKind::Order && by_order.series.used == 143 && by_order.series.t[5] == 5.0, "row order");
    Check(series_view::SeriesWords(by_order.series) == "143 rows in table order", series_view::SeriesWords(by_order.series));
    req.time = "Passengers";
    req.value = "Passengers";
    Check(!series::ReadSeries("airline", req, run, {}).ok(), "time = value refused");
    req.time = "Month";
    req.value = "Month";
    Check(!series::ReadSeries("airline", req, run, {}).ok(), "a text value is refused");

    // ---- the answers (figures are the backend's, which the truth test checks)
    series::PartsRequest parts;
    parts.base = {"Month", "Passengers", 0};
    const auto p = series::RunParts("airline", parts, run, {});
    Check(p.ok() && p.parts.method == "stl" && p.parts.trend_strength > 0.99 && p.parts.seasonal_strength > 0.98, "STL parts: " + p.error);
    Contains(series_view::PartsSentence(p), "Passengers is almost all trend and season", "parts words");
    Contains(series_view::PartsSentence(p), "the season grows with the level", "parts words: growing season");
    Contains(series_view::PartsSentence(p), "the season peaks in ", "parts words: the month named");
    parts.stl = false;
    parts.multiplicative = true;
    const auto classical = series::RunParts("airline", parts, run, {});
    Check(classical.ok() && classical.parts.method == "multiplicative" && std::isnan(classical.parts.trend.front()), "classical multiplicative parts: " + classical.error);
    Contains(series_view::PartsCsv(p), "time,Passengers,trend,seasonal,residual\n1949-01-01,112,", "parts CSV");

    const auto season = series::RunSeasonality("airline", parts.base, run, {});
    Check(season.ok() && season.seasonality.detected_period == 12 && season.position_names.size() == 12 && season.position_names[0] == "Jan" && season.position_names[6] == "Jul",
          "season by month: " + season.error);
    Contains(series_view::SeasonalitySentence(season), "A yearly season (period 12) carries", "season words");
    Contains(series_view::SeasonalitySentence(season), "Jul is the high", "season words: July");

    series::StationarityRequest stationarity;
    stationarity.base = parts.base;
    const auto st = series::RunStationarity("airline", stationarity, run, {});
    Check(st.ok() && !st.tests.is_stationary && st.tests.suggested_differencing == 1 && st.window == 12 && std::isnan(st.rolling_mean[5]) && std::isfinite(st.rolling_mean[11]),
          "stationarity: " + st.error);
    Contains(series_view::StationaritySentence(st), "One difference makes it stationary enough", "stationarity words");
    Contains(series_view::StationaritySentence(st), "the rolling spread grows", "stationarity words: spread");

    series::CorrelogramRequest correlogram;
    correlogram.base = parts.base;
    correlogram.lags = 40;
    const auto c = series::RunCorrelogram("airline", correlogram, run, {});
    Check(c.ok() && c.correlogram.max_lag == 40 && c.correlogram.ljung_box_pvalue < 1e-10, "correlogram: " + c.error);
    Contains(series_view::CorrelogramSentence(c), "The ACF decays slowly, which is a trend, and rises again at 12, 24, which is the season", "correlogram words");
    correlogram.differenced = true;
    const auto cd = series::RunCorrelogram("airline", correlogram, run, {});
    Check(cd.ok() && cd.correlogram.acf.size() == 41, "differenced correlogram");

    series::ForecastRequest forecast;
    forecast.base = parts.base;
    forecast.method = series::Method::HoltWinters;
    forecast.horizon = 24;
    forecast.holdout = 24;
    const auto f = series::RunForecast("airline", forecast, run, {});
    Check(f.ok(), "forecast: " + f.error);
    Check(f.future_labels.size() == 24 && f.future_labels[0] == "1961-01-01" && f.future_labels[23] == "1962-12-01", "future months: " + f.future_labels[0]);
    Check(f.holdout == 24 && f.scores.size() == 4 && f.scores.front().method == series::Method::HoltWinters && f.scores.back().method == series::Method::SeasonalNaive,
          "the held-out check scores the chosen method first and the naive baseline last");
    Check(f.scores.front().mape < f.scores.back().mape && f.scores.front().mape < 10.0, "Holt-Winters beats repeating last year on the held-out months: " +
                                                                                               std::to_string(f.scores.front().mape) + " vs " + std::to_string(f.scores.back().mape));
    Check(f.fit.forecast[0] > 400.0 && f.fit.forecast[0] < 500.0 && f.fit.lower_bound[0] < f.fit.forecast[0] && f.fit.upper_bound[0] > f.fit.forecast[0], "a sensible forecast with its range");
    Contains(series_view::ForecastSentence(f), "Holt-Winters (additive trend, multiplicative season, period 12; alpha ", "forecast words");
    Contains(series_view::ForecastSentence(f), "nineteen times in twenty", "level words");
    Contains(series_view::HoldoutSentence(f), "Checked on the last 24 months held out", "holdout words");
    Contains(series_view::HoldoutSentence(f), "times better than repeating last year", "holdout words: baseline");
    Contains(series_view::ForecastCsv(f), "1961-01-01,,,", "forecast CSV future rows");

    forecast.method = series::Method::Arima;
    forecast.p = 2;
    forecast.d = 1;
    forecast.q = 1;
    forecast.holdout = 0;
    const auto arima = series::RunForecast("airline", forecast, run, {});
    Check(arima.ok() && arima.fit.method == "ARIMA(2,1,1)" && arima.scores.empty(), "ARIMA without the check: " + arima.error);
    Contains(series_view::ForecastSentence(arima), "ARIMA(2,1,1) (ar.L1 ", "ARIMA words");

    // Into the graph: the nodes' parameters.
    Check(series::NodeParameters(parts).at("algorithm") == "classical" && series::NodeParameters(parts).at("method") == "multiplicative" &&
              series::NodeParameters(parts).at("period") == "-1",
          "Decomposition parameters");
    Check(series::NodeParameters(stationarity).at("regression") == "c" && series::NodeParameters(correlogram).at("max_lag") == "40", "test and correlogram parameters");
    Check(series::NodeParameters(forecast).at("p") == "2" && series::NodeParameters(forecast).at("time_col") == "Month" && series::NodeParameters(forecast).at("horizon") == "24" &&
              std::string(series::ForecastNodeType(forecast.method)) == "ARIMAForecaster",
          "ARIMA node parameters");
    forecast.method = series::Method::HoltWinters;
    Check(series::NodeParameters(forecast).at("method") == "holt_winters" && series::NodeParameters(forecast).at("seasonal") == "multiplicative" &&
              std::string(series::ForecastNodeType(forecast.method)) == "ExponentialSmoothing",
          "Exp. Smoothing node parameters");
    forecast.method = series::Method::SeasonalNaive;
    Check(series::NodeParameters(forecast).empty() && std::string(series::ForecastNodeType(forecast.method)).empty(), "the baseline has no node");
    forecast.method = series::Method::Simple;
    forecast.horizon = 0;
    Check(!series::RunForecast("airline", forecast, run, {}).ok(), "a horizon of 0 is refused");

    if (std::getenv("CYXWIZ_SHOW_WORDS")) {
        const char* gap = "\n\n";
        std::cout << series_view::SeriesWords(s) << " | " << series_view::LeftOut(s) << " | " << series_view::PeriodWords(s) << gap
                  << series_view::PartsSentence(p) << gap << series_view::PartsSentence(classical) << gap << series_view::SeasonalitySentence(season) << gap
                  << series_view::StationaritySentence(st) << gap << series_view::CorrelogramSentence(c) << gap << series_view::CorrelogramSentence(cd) << gap
                  << series_view::ForecastSentence(f) << '\n' << series_view::HoldoutSentence(f) << gap << series_view::ForecastSentence(arima) << '\n'
                  << series_view::HoldoutSentence(arima) << '\n';
        for (const auto& sc : f.scores) std::cout << sc.name << ": MAPE " << sc.mape << " RMSE " << sc.rmse << " " << sc.error << '\n';
    }
    fs::remove_all(root, ec);
    std::cout << "series explore: reading, steps, periods, answers, the held-out check and words. OK\n";
    return 0;
}
