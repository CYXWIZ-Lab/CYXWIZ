#pragma once

// The answers behind Data Studio's Explore tabs Series and Forecast
// (TOFIX129 D6, approved mockup d6_time_series_mockup.html). One time column
// (dates, numbers, or the row order), one value column, read through a
// QueryRunner and sorted by time; rows missing either are left out and
// counted; the step between rows is read from the times (monthly, daily, ...)
// and the period is found from the detrended periodogram unless typed. The
// analysis runs in cyxwiz::TimeSeries (checked against statsmodels / scipy in
// test_computation_truth_time_series); what is new here (the time column,
// the step, the held-out check, future dates) is checked in test_series_explore.

#include "data_explore.h"

#include <cyxwiz/time_series.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace cyxwiz::series {

enum class TimeKind { Order, Numbers, Dates };

struct SeriesRequest {
    std::string time;   // "" = the row order
    std::string value;
    int period = 0;     // 0: find it
};

// A time step read from the dates (or numbers).
struct Step {
    std::string name;        // "monthly", "daily", "every 3 days", "every 0.5" (numbers), "" (order)
    double seconds = 0.0;    // dates: the step in seconds (a month: 30.436875 days); numbers: the step
    bool even = false;       // at least 95% of the gaps are the step (a month: 28 .. 31 days)
    int natural_period = 0;  // the period the step suggests (monthly 12, quarterly 4, daily 7, hourly 24), else 0
};

struct Series {
    TimeKind kind = TimeKind::Order;
    std::vector<double> t;         // dates: seconds since 1970; numbers: the values; order: 0, 1, 2 ...
    std::vector<double> values;
    std::vector<std::string> labels;  // dates as text (YYYY-MM-DD, with the time when the step is below a day)
    int64_t rows = 0;              // rows in the table
    int64_t used = 0;              // rows with both the time and the value
    int64_t unordered = 0;         // rows the sort moved (the table was not in time order)
    Step step;
    int period = 0;                // the one used
    bool period_found = false;     // from the periodogram (else typed, or the step's)
    double period_share = 0.0;     // the found period's share of the detrended power
};

struct SeriesAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    Series series;
};

// Reads the series: the time column as dates (a date column, or a text that
// reads as one), as numbers, or the row order; the value column as numbers.
SeriesAnswer ReadSeries(const std::string& table, const SeriesRequest& request, const QueryRunner& run, const explore::MissingTexts& texts);

// ---- Series: Parts

struct PartsRequest {
    SeriesRequest base;
    bool stl = true;
    bool multiplicative = false;  // classical only
    int seasonal_window = 7;      // STL
    bool robust = false;          // STL
};

struct PartsAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    PartsRequest request;
    Series series;
    DecompositionResult parts;
};

PartsAnswer RunParts(const std::string& table, const PartsRequest& request, const QueryRunner& run, const explore::MissingTexts& texts);

// ---- Series: Seasonality

struct SeasonalityAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    SeriesRequest request;
    Series series;
    SeasonalityResult seasonality;          // detrended periodogram, ranked candidates, harmonics
    std::vector<double> season_by_position;  // the additive season at each position of the period (means)
    std::vector<std::string> position_names; // "Jan" .. for a monthly step, "Mon" .. weekly days, else "1" ..
};

SeasonalityAnswer RunSeasonality(const std::string& table, const SeriesRequest& request, const QueryRunner& run,
                                 const explore::MissingTexts& texts);

// ---- Series: Stationarity

struct StationarityRequest {
    SeriesRequest base;
    bool trend = false;   // KPSS / ADF around a trend ("ct") instead of a level ("c")
    int window = 0;       // rolling window (0: the period, else max(10, n / 10))
};

struct StationarityAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    StationarityRequest request;
    Series series;
    StationarityResult tests;              // on the series, with the differenced runs and the suggested d
    std::vector<double> rolling_mean, rolling_sd;
    int window = 0;
};

StationarityAnswer RunStationarity(const std::string& table, const StationarityRequest& request, const QueryRunner& run,
                                   const explore::MissingTexts& texts);

// ---- Series: Correlogram

struct CorrelogramRequest {
    SeriesRequest base;
    int lags = 40;
    bool differenced = false;  // of the first difference
};

struct CorrelogramAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    CorrelogramRequest request;
    Series series;
    AutocorrelationResult correlogram;
};

CorrelogramAnswer RunCorrelogram(const std::string& table, const CorrelogramRequest& request, const QueryRunner& run,
                                 const explore::MissingTexts& texts);

// ---- Forecast

enum class Method { Simple, Holt, HoltWinters, Arima, SeasonalNaive };

struct ForecastRequest {
    SeriesRequest base;
    Method method = Method::HoltWinters;
    int horizon = 12;
    double level = 0.95;
    bool damped = false;           // Holt, Holt-Winters
    bool multiplicative = true;    // Holt-Winters season
    double alpha = -1, beta = -1, gamma = -1;  // -1: estimated
    int p = -1, d = -1, q = -1;    // ARIMA; -1: chosen
    int holdout = 0;               // the last rows held out for the check (0: none)
};

struct HoldoutScore {
    Method method = Method::Simple;
    std::string name;
    std::vector<double> forecast;  // over the held-out rows
    double mape = NAN, rmse = NAN, mae = NAN;
    std::string error;             // when that method could not run on the shorter series
};

struct ForecastAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    ForecastRequest request;
    Series series;
    ForecastResult fit;                  // the chosen method on every row
    std::vector<std::string> future_labels;  // the horizon's dates (or numbers / indices)
    std::vector<double> future_t;
    // The held-out check: the last `holdout` rows, every method fitted on the rest.
    int holdout = 0;
    std::vector<double> actual;
    std::vector<HoldoutScore> scores;    // the chosen method first, the seasonal naive baseline last
};

ForecastAnswer RunForecast(const std::string& table, const ForecastRequest& request, const QueryRunner& run,
                           const explore::MissingTexts& texts);

// ---- Pure helpers (test_series_explore)

// The step of sorted times (dates in seconds, or numbers): its name, size,
// whether the gaps are even, and the period it suggests.
Step StepOf(const std::vector<double>& t, TimeKind kind);
// Dates `count` steps after the last one (calendar months for a monthly step).
std::vector<double> FutureTimes(const std::vector<double>& t, const Step& step, TimeKind kind, int count);
// A date in seconds since 1970 as "YYYY-MM-DD" (with " HH:MM" when the step is below a day).
std::string DateLabel(double seconds, const Step& step);
// seconds since 1970 of a civil date, and back.
double CivilToSeconds(int year, int month, int day, int hour = 0, int minute = 0, int second = 0);
void SecondsToCivil(double seconds, int& year, int& month, int& day, int& hour, int& minute, int& second);
// Mean absolute percentage error (rows at 0 skipped) and RMSE of a forecast against actual values.
double Mape(const std::vector<double>& actual, const std::vector<double>& forecast);
double Rmse(const std::vector<double>& actual, const std::vector<double>& forecast);

const char* MethodName(Method m);

}  // namespace cyxwiz::series
