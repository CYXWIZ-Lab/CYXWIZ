// How long the time series fits take on a long series (TOFIX129 D6): a
// MetroPT-3-sized series (1.5 million rows, a daily season of 8,640 rows of
// 10 seconds) through STL, the smoothing fits and ARIMA, with the time of
// each. Not a check: it prints, for the ticket and the docs.

#include <cyxwiz/time_series.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace cyxwiz;

namespace {
double Seconds(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
}  // namespace

int main(int argc, char** argv) {
    const size_t n = argc > 1 ? static_cast<size_t>(std::atol(argv[1])) : 1516948;
    const int period = argc > 2 ? std::atoi(argv[2]) : 8640;
    std::mt19937 gen(7);
    std::normal_distribution<double> noise(0.0, 1.0);
    std::vector<double> y(n);
    for (size_t i = 0; i < n; ++i)
        y[i] = 60.0 + 5.0 * std::sin(2.0 * M_PI * static_cast<double>(i) / period) + 0.000002 * static_cast<double>(i) + noise(gen);
    std::cout << "n=" << n << " period=" << period << "\n";

    auto t0 = std::chrono::steady_clock::now();
    const SeasonalityResult season = TimeSeries::DetectSeasonality(y);
    std::cout << "DetectSeasonality " << Seconds(t0) << " s: period " << season.detected_period << " share " << season.detected_share << " strength " << season.strength << "\n";

    t0 = std::chrono::steady_clock::now();
    const DecompositionResult stl = TimeSeries::STLDecompose(y, period);
    std::cout << "STL " << Seconds(t0) << " s: " << (stl.success ? "ok" : stl.error_message) << " strengths " << stl.trend_strength << " " << stl.seasonal_strength << "\n";

    t0 = std::chrono::steady_clock::now();
    const ForecastResult simple = TimeSeries::SimpleES(y, 12);
    std::cout << "SimpleES " << Seconds(t0) << " s: " << (simple.success ? "ok" : simple.error_message) << "\n";

    t0 = std::chrono::steady_clock::now();
    const ForecastResult holt = TimeSeries::HoltLinear(y, 12);
    std::cout << "Holt " << Seconds(t0) << " s: " << (holt.success ? "ok" : holt.error_message) << "\n";

    t0 = std::chrono::steady_clock::now();
    const ForecastResult hw = TimeSeries::HoltWinters(y, 12, period, "additive");
    std::cout << "HoltWinters " << Seconds(t0) << " s: " << (hw.success ? "ok" : hw.error_message) << " alpha " << (hw.parameters.count("alpha") ? hw.parameters.at("alpha") : -1)
              << "\n";

    t0 = std::chrono::steady_clock::now();
    const StationarityResult st = TimeSeries::TestStationarity(y);
    std::cout << "Stationarity " << Seconds(t0) << " s: " << (st.success ? "ok" : st.error_message) << " d " << st.suggested_differencing << "\n";

    t0 = std::chrono::steady_clock::now();
    const AutocorrelationResult acf = TimeSeries::ComputeACFPACF(y, 40);
    std::cout << "Correlogram " << Seconds(t0) << " s: " << (acf.success ? "ok" : acf.error_message) << "\n";

    t0 = std::chrono::steady_clock::now();
    const ForecastResult arima = TimeSeries::ARIMA(y, 12, 1, 1, 1);
    std::cout << "ARIMA(1,1,1) " << Seconds(t0) << " s: " << (arima.success ? "ok" : arima.error_message) << "\n";
    return 0;
}
