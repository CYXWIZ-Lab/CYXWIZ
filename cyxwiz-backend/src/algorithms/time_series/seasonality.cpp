// Prevent Windows min/max macros from interfering with std::min/std::max
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <cyxwiz/time_series.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace cyxwiz {

// Constants
constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 2.0 * PI;

// Seasonality Detection
// ============================================================================

SeasonalityResult TimeSeries::DetectSeasonality(const std::vector<double>& data, int min_period, int max_period) {
    SeasonalityResult result;

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

    int n = static_cast<int>(data.size());
    if (max_period < 0) {
        max_period = n / 2;
    }
    max_period = (std::min)(max_period, n / 2);

    if (min_period < 2) min_period = 2;
    if (max_period <= min_period) {
        result.error_message = "Invalid period range";
        return result;
    }

    try {
        // Compute periodogram
        auto [periodogram, frequencies] = Periodogram(data);
        result.periodogram = periodogram;
        result.frequencies = frequencies;

        // Convert to periods
        result.periods.resize(frequencies.size());
        for (size_t i = 0; i < frequencies.size(); i++) {
            if (frequencies[i] > 1e-10) {
                result.periods[i] = 1.0 / frequencies[i];
            } else {
                result.periods[i] = 0;
            }
        }

        // Find peaks in periodogram within valid period range
        double max_power = 0;
        double total_power = 0;

        for (size_t i = 1; i < periodogram.size(); i++) {
            total_power += periodogram[i];
        }

        for (size_t i = 1; i < periodogram.size() - 1; i++) {
            double period = result.periods[i];
            if (period >= min_period && period <= max_period) {
                // Local maximum check
                if (periodogram[i] > periodogram[i - 1] && periodogram[i] > periodogram[i + 1]) {
                    int period_int = static_cast<int>(std::round(period));
                    double strength = periodogram[i] / total_power;

                    result.candidate_periods.push_back(period_int);
                    result.candidate_strengths.push_back(strength);

                    if (periodogram[i] > max_power) {
                        max_power = periodogram[i];
                        result.detected_period = period_int;
                        result.strength = strength;
                    }
                }
            }
        }

        // Also check ACF for confirmation
        auto acf = ComputeACF(data, max_period);
        if (acf.success) {
            for (int lag = min_period; lag <= max_period && lag < static_cast<int>(acf.acf.size()) - 1; lag++) {
                if (acf.acf[lag] > acf.acf[lag - 1] && acf.acf[lag] > acf.acf[lag + 1] &&
                    acf.acf[lag] > 0.1) {
                    result.acf_peaks.push_back(lag);
                }
            }
        }

        // Determine if seasonality is significant
        result.has_seasonality = (result.strength > 0.05 && result.detected_period >= min_period);

        // Analysis text
        if (result.has_seasonality) {
            result.analysis = "Seasonality detected with period " + std::to_string(result.detected_period) +
                              " (strength: " + std::to_string(result.strength) + ")";
        } else {
            result.analysis = "No significant seasonality detected";
        }

        result.success = true;

    } catch (const std::exception& e) {
        result.error_message = std::string("Seasonality detection failed: ") + e.what();
    }

    return result;
}

std::pair<std::vector<double>, std::vector<double>> TimeSeries::Periodogram(const std::vector<double>& data) {
    int n = static_cast<int>(data.size());
    if (n == 0) return {{}, {}};

    double mean = Mean(data);

    // Compute raw periodogram using DFT
    std::vector<double> periodogram;
    std::vector<double> frequencies;

    int n_freq = n / 2 + 1;
    periodogram.resize(n_freq);
    frequencies.resize(n_freq);

    for (int k = 0; k < n_freq; k++) {
        double cos_sum = 0, sin_sum = 0;
        for (int t = 0; t < n; t++) {
            double angle = TWO_PI * k * t / n;
            cos_sum += (data[t] - mean) * std::cos(angle);
            sin_sum += (data[t] - mean) * std::sin(angle);
        }
        periodogram[k] = (cos_sum * cos_sum + sin_sum * sin_sum) / n;
        frequencies[k] = static_cast<double>(k) / n;
    }

    return {periodogram, frequencies};
}

// ============================================================================

} // namespace cyxwiz
