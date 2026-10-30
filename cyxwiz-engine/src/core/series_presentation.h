#pragma once

// Words of Data Studio's Explore tabs Series and Forecast (TOFIX129 D6,
// approved mockup d6_time_series_mockup.html): what the series is, its parts,
// its season, whether it is stationary, its correlogram, the forecast and the
// held-out check, and the CSV exports. Answers in, text out, so the tabs and
// their test agree. No ImGui, no queries.

#include "series_explore.h"

#include <string>
#include <vector>

namespace cyxwiz::series_view {

const char* MethodLabel(series::Method m);
// "month", "day", "week", "quarter", "year", "hour", "step".
std::string StepUnit(const series::Step& step);
// "144 months, monthly, 1949-01 .. 1960-12" / "200 rows in order".
std::string SeriesWords(const series::Series& s);
// "2 rows have no value or date and are left out. The table was not in time order; the rows were sorted." ("" when nothing).
std::string LeftOut(const series::Series& s);
// "found: 12 (53% of the detrended power)" / "typed" / "from the step" / "none found".
std::string PeriodWords(const series::Series& s);
// A number in a few digits (explore_view::Figure).
std::string Num(double v);

std::string PartsSentence(const series::PartsAnswer& a);
std::string SeasonalitySentence(const series::SeasonalityAnswer& a);
std::string StationaritySentence(const series::StationarityAnswer& a);
std::string CorrelogramSentence(const series::CorrelogramAnswer& a);
// "Holt-Winters (additive trend, multiplicative season, period 12) forecasts 443 next month, between 413 and 475 nineteen times in twenty, rising to 617 in 1961-07."
std::string ForecastSentence(const series::ForecastAnswer& a);
// "Checked on the last 24 months held out, it was 6.4% off on average: 2.4 times better than repeating last year (15.5%)." ("" without a check).
std::string HoldoutSentence(const series::ForecastAnswer& a);
// What the method is, with its fitted parameters: "additive trend, multiplicative season, period 12; alpha 0.319, beta 0.000, gamma 0.601"; ARIMA: its terms.
std::string ModelWords(const series::ForecastAnswer& a);

// CSV exports.
std::string PartsCsv(const series::PartsAnswer& a);
std::string SeasonalityCsv(const series::SeasonalityAnswer& a);
std::string StationarityCsv(const series::StationarityAnswer& a);
std::string CorrelogramCsv(const series::CorrelogramAnswer& a);
std::string ForecastCsv(const series::ForecastAnswer& a);

}  // namespace cyxwiz::series_view
