// The correlogram (TOFIX129 D6): autocorrelation through the device FFT,
// partial autocorrelation by the Yule-Walker recursion on it, Bartlett's
// band, and the Ljung-Box test with a chi-squared p-value. Equal to
// statsmodels acf(fft=True, alpha), pacf(method="ywm", alpha) and
// acorr_ljungbox (test_computation_truth_time_series).

#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/time_series.h"
#include "cyxwiz/statistical_tests.h"
#include "series_device.h"

#include <algorithm>
#include <cmath>

namespace cyxwiz {

namespace {

int DefaultMaxLag(size_t n) { return std::max(1, std::min(40, static_cast<int>(n) / 2 - 1)); }

// The partial autocorrelations from the autocorrelations (Durbin-Levinson:
// the last Yule-Walker coefficient at each order).
std::vector<double> PartialFromAcf(const std::vector<double>& acf) {
    const size_t k = acf.size();
    std::vector<double> pacf(k, 0.0), phi(k, 0.0), previous(k, 0.0);
    if (k == 0) return pacf;
    pacf[0] = 1.0;
    for (size_t m = 1; m < k; ++m) {
        double num = acf[m], den = 1.0;
        for (size_t j = 1; j < m; ++j) {
            num -= previous[j] * acf[m - j];
            den -= previous[j] * acf[j];
        }
        const double phi_mm = den != 0.0 ? num / den : 0.0;
        phi[m] = phi_mm;
        for (size_t j = 1; j < m; ++j) phi[j] = previous[j] - phi_mm * previous[m - j];
        previous = phi;
        pacf[m] = phi_mm;
    }
    return pacf;
}

// How many leading lags are outside the band (the cut-off order; at most 5).
int CutOff(const std::vector<double>& values, const std::vector<double>& half_width) {
    int order = 0;
    for (size_t k = 1; k < values.size() && k < half_width.size() && order < 5; ++k) {
        if (std::fabs(values[k]) > half_width[k]) ++order;
        else break;
    }
    return order;
}

void FillLjungBox(const std::vector<double>& acf, size_t n, AutocorrelationResult& r) {
    const size_t max_lag = acf.size() - 1;
    r.ljung_box_q.assign(max_lag + 1, 0.0);
    r.ljung_box_p.assign(max_lag + 1, 1.0);
    double sum = 0.0;
    for (size_t k = 1; k <= max_lag; ++k) {
        sum += acf[k] * acf[k] / static_cast<double>(n - k);
        r.ljung_box_q[k] = static_cast<double>(n) * static_cast<double>(n + 2) * sum;
        r.ljung_box_p[k] = StatisticalTests::ChiSquareSf(r.ljung_box_q[k], static_cast<double>(k));
    }
    r.ljung_box_lags = static_cast<int>(std::min<size_t>(10, max_lag));
    r.ljung_box_statistic = r.ljung_box_q[static_cast<size_t>(r.ljung_box_lags)];
    r.ljung_box_pvalue = r.ljung_box_p[static_cast<size_t>(r.ljung_box_lags)];
}

AutocorrelationResult Correlogram(const std::vector<double>& data, int max_lag, double confidence, bool want_acf, bool want_pacf) {
    AutocorrelationResult r;
    r.confidence_level = confidence;
    if (const std::string refusal = series_detail::SeriesRefusal(data, 4, "The correlogram"); !refusal.empty()) {
        r.error_message = refusal;
        return r;
    }
    if (!(confidence > 0.5 && confidence < 1.0)) {
        r.error_message = "The confidence level must be between 0.5 and 1";
        return r;
    }
    const size_t n = data.size();
    if (max_lag < 0) max_lag = DefaultMaxLag(n);
    if (max_lag < 1 || static_cast<size_t>(max_lag) >= n) {
        r.error_message = "max_lag must be between 1 and n - 1";
        return r;
    }
    try {
        const std::vector<double> acov = series_detail::Autocovariance(data, max_lag);
        if (!(acov[0] > 0.0)) {
            r.error_message = "The series has the same value in every row";
            return r;
        }
        std::vector<double> acf(acov.size());
        for (size_t k = 0; k < acov.size(); ++k) acf[k] = acov[k] / acov[0];
        r.max_lag = max_lag;
        r.lags.resize(acf.size());
        for (size_t k = 0; k < acf.size(); ++k) r.lags[k] = static_cast<double>(k);
        const double z = series_detail::NormalPpf(1.0 - (1.0 - confidence) / 2.0);
        // Bartlett: the variance of r_k grows with the earlier autocorrelations.
        std::vector<double> half(acf.size(), 0.0);
        double cumulative = 0.0;
        for (size_t k = 1; k < acf.size(); ++k) {
            half[k] = z * std::sqrt((1.0 + 2.0 * cumulative) / static_cast<double>(n));
            cumulative += acf[k] * acf[k];
        }
        if (want_acf) {
            r.acf = acf;
            r.confidence_upper.resize(acf.size());
            r.confidence_lower.resize(acf.size());
            for (size_t k = 0; k < acf.size(); ++k) {
                r.confidence_upper[k] = acf[k] + half[k];
                r.confidence_lower[k] = acf[k] - half[k];
            }
            for (size_t k = 1; k < acf.size(); ++k)
                if (std::fabs(acf[k]) > half[k]) r.significant_acf_lags.push_back(static_cast<int>(k));
            r.suggested_ma_order = CutOff(acf, half);
        }
        if (want_pacf) {
            r.pacf = PartialFromAcf(acf);
            const double band = z / std::sqrt(static_cast<double>(n));
            r.pacf_confidence_upper.resize(acf.size());
            r.pacf_confidence_lower.resize(acf.size());
            const std::vector<double> bands(acf.size(), band);
            for (size_t k = 0; k < acf.size(); ++k) {
                r.pacf_confidence_upper[k] = r.pacf[k] + band;
                r.pacf_confidence_lower[k] = r.pacf[k] - band;
            }
            for (size_t k = 1; k < acf.size(); ++k)
                if (std::fabs(r.pacf[k]) > band) r.significant_pacf_lags.push_back(static_cast<int>(k));
            r.suggested_ar_order = CutOff(r.pacf, bands);
        }
        FillLjungBox(acf, n, r);
        r.success = true;
    } catch (const af::exception& e) {
        r.error_message = std::string("The correlogram failed on the ArrayFire device: ") + e.what();
    }
    return r;
}

}  // namespace

AutocorrelationResult TimeSeries::ComputeACF(const std::vector<double>& data, int max_lag) {
    return Correlogram(data, max_lag, 0.95, true, false);
}

AutocorrelationResult TimeSeries::ComputePACF(const std::vector<double>& data, int max_lag) {
    return Correlogram(data, max_lag, 0.95, false, true);
}

AutocorrelationResult TimeSeries::ComputeACFPACF(const std::vector<double>& data, int max_lag, double confidence) {
    return Correlogram(data, max_lag, confidence, true, true);
}

double TimeSeries::LjungBoxTest(const std::vector<double>& data, int lags) {
    const AutocorrelationResult r = Correlogram(data, lags, 0.95, true, false);
    if (!r.success || lags < 1 || static_cast<size_t>(lags) >= r.ljung_box_p.size()) return NAN;
    return r.ljung_box_p[static_cast<size_t>(lags)];
}

}  // namespace cyxwiz
