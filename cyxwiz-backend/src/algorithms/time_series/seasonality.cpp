// Seasonality (TOFIX129 D6): the periodogram of the detrended series through
// the device FFT (scipy.signal.periodogram of scipy.signal.detrend, scaling
// "spectrum"), its peaks ranked by their share of the power with harmonics
// named, the strength of the best period from a classical decomposition, and
// the ACF's peaks.

#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/time_series.h"
#include "series_device.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace cyxwiz {

std::pair<std::vector<double>, std::vector<double>> TimeSeries::Periodogram(const std::vector<double>& data) {
    std::vector<double> power, frequency;
    if (!series_detail::SeriesRefusal(data, 4, "The periodogram").empty()) return {power, frequency};
    try {
        series_detail::Periodogram(series_detail::LinearDetrend(data), power, frequency);
    } catch (const af::exception&) {
        power.clear();
        frequency.clear();
    }
    return {power, frequency};
}

SeasonalityResult TimeSeries::DetectSeasonality(const std::vector<double>& data, int min_period, int max_period) {
    SeasonalityResult r;
    if (const std::string refusal = series_detail::SeriesRefusal(data, 8, "Seasonality detection"); !refusal.empty()) {
        r.error_message = refusal;
        return r;
    }
    const int n = static_cast<int>(data.size());
    if (min_period < 2) min_period = 2;
    // A season has to repeat: unless a longer one is asked for, the period
    // goes up to a quarter of the rows (four full cycles), so a slow wave over
    // a long series is not read as its season.
    if (max_period < 0) max_period = n / 4;
    if (max_period > n / 2) max_period = n / 2;
    if (max_period < min_period) {
        r.error_message = "Invalid period range: the longest period is below the shortest";
        return r;
    }
    try {
        std::vector<double> power, frequency;
        series_detail::Periodogram(series_detail::LinearDetrend(data), power, frequency);
        r.periodogram = power;
        r.frequencies = frequency;
        r.periods.assign(power.size(), INFINITY);
        for (size_t k = 1; k < power.size(); ++k) r.periods[k] = 1.0 / frequency[k];
        double total = 0.0;
        for (size_t k = 1; k < power.size(); ++k) total += power[k];

        // Local maxima with a period in range, strongest first.
        struct Peak {
            size_t k;
            double period, share;
        };
        std::vector<Peak> peaks;
        for (size_t k = 1; k + 1 < power.size(); ++k) {
            const double period = r.periods[k];
            if (period < min_period - 0.5 || period > max_period + 0.5) continue;
            if (power[k] > power[k - 1] && power[k] >= power[k + 1]) peaks.push_back({k, period, total > 0.0 ? power[k] / total : 0.0});
        }
        std::stable_sort(peaks.begin(), peaks.end(), [](const Peak& a, const Peak& b) { return a.share > b.share; });
        for (const Peak& p : peaks) {
            const int rounded = static_cast<int>(std::lround(p.period));
            if (std::find(r.candidate_periods.begin(), r.candidate_periods.end(), rounded) != r.candidate_periods.end()) continue;
            // A harmonic: a stronger candidate's period is a whole multiple of this one.
            int parent = 0;
            for (size_t i = 0; i < r.candidate_periods.size() && parent == 0; ++i) {
                const double ratio = static_cast<double>(r.candidate_periods[i]) / p.period;
                const double whole = std::round(ratio);
                if (whole >= 2.0 && std::fabs(ratio - whole) <= 0.05 * whole) parent = r.candidate_periods[i];
            }
            r.candidate_periods.push_back(rounded);
            r.candidate_strengths.push_back(p.share);
            r.harmonic_of.push_back(parent);
        }
        for (size_t i = 0; i < r.candidate_periods.size(); ++i) {
            if (r.harmonic_of[i] != 0) continue;
            r.detected_period = r.candidate_periods[i];
            r.detected_share = r.candidate_strengths[i];
            break;
        }
        // Its strength: how much of the detrended variation a seasonal pattern of that period explains.
        if (r.detected_period >= 2 && n >= 2 * r.detected_period) {
            const DecompositionResult d = Decompose(data, r.detected_period, "additive");
            if (d.success) r.strength = d.seasonal_strength;
        } else {
            r.strength = r.detected_share;
        }
        // Found: a pattern of that length explains 30% of what is left after the trend,
        // and its peak carries at least 5% of the detrended power (not one wave among many).
        r.has_seasonality = r.detected_period >= min_period && r.strength >= 0.3 && r.detected_share >= 0.05;

        // Peaks of the autocorrelation outside its band.
        const AutocorrelationResult acf = ComputeACF(data, std::min(max_period, n / 2 - 1));
        if (acf.success)
            for (size_t k = 2; k + 1 < acf.acf.size(); ++k)
                if (acf.acf[k] > acf.acf[k - 1] && acf.acf[k] >= acf.acf[k + 1] && acf.acf[k] > acf.confidence_upper[k] - acf.acf[k])
                    r.acf_peaks.push_back(static_cast<int>(k));

        if (r.detected_period == 0) {
            r.analysis = "No seasonal period stands out after removing the trend.";
        } else {
            r.analysis = "The strongest period is " + std::to_string(r.detected_period) + " (" +
                         std::to_string(static_cast<int>(std::lround(100.0 * r.detected_share))) + "% of the detrended power); " +
                         (r.has_seasonality ? "a seasonal pattern of that period explains " : "a seasonal pattern of that period explains only ") +
                         std::to_string(static_cast<int>(std::lround(100.0 * r.strength))) + "% of what is left after the trend.";
        }
        r.success = true;
    } catch (const af::exception& e) {
        r.error_message = std::string("Seasonality detection failed on the ArrayFire device: ") + e.what();
    }
    return r;
}

}  // namespace cyxwiz
