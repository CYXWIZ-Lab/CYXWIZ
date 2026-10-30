// Stationarity (TOFIX129 D6): the augmented Dickey-Fuller test with its lags
// chosen by AIC and MacKinnon's p-values and critical values (statsmodels
// adfuller), the KPSS test with Newey-West lags and the 1992 table
// (statsmodels kpss), both on the series and its differences for the
// suggested order of differencing. The regressions run on the device
// (StatisticalTests::FitLinear); the long-run variance uses the device FFT.

#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/time_series.h"
#include "cyxwiz/statistical_tests.h"
#include "mackinnon.h"
#include "series_device.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace cyxwiz {

namespace {

int TrendTerms(const std::string& regression) {
    if (regression == "n") return 0;
    if (regression == "c") return 1;
    if (regression == "ct") return 2;
    return 3;
}

// The ADF regression of dy_t on y_{t-1}, dy_{t-1} .. dy_{t-lags} and the trend
// terms, over the last `nobs` rows of `xdiff` (statsmodels' lagmat with
// trim="both"). The level's t is t[intercept ? 1 : 0].
LinearFit AdfRegression(const std::vector<double>& x, const std::vector<double>& xdiff, int lags, int nobs, const std::string& regression) {
    const size_t n = x.size();
    const size_t first = xdiff.size() - static_cast<size_t>(nobs);  // first response row
    const std::vector<double> response(xdiff.begin() + static_cast<std::ptrdiff_t>(first), xdiff.end());
    std::vector<std::vector<double>> columns;
    std::vector<std::string> names;
    std::vector<double> level(static_cast<size_t>(nobs));
    for (int i = 0; i < nobs; ++i) level[static_cast<size_t>(i)] = x[n - 1 - static_cast<size_t>(nobs) + static_cast<size_t>(i)];
    columns.push_back(level);
    names.push_back("level");
    for (int lag = 1; lag <= lags; ++lag) {
        std::vector<double> column(static_cast<size_t>(nobs));
        for (int i = 0; i < nobs; ++i) column[static_cast<size_t>(i)] = xdiff[first + static_cast<size_t>(i) - static_cast<size_t>(lag)];
        columns.push_back(column);
        names.push_back("dlag" + std::to_string(lag));
    }
    const int trend = TrendTerms(regression);
    if (trend >= 2) {
        std::vector<double> t(static_cast<size_t>(nobs));
        for (int i = 0; i < nobs; ++i) t[static_cast<size_t>(i)] = static_cast<double>(i + 1);
        columns.push_back(t);
        names.push_back("trend");
        if (trend == 3) {
            for (double& v : t) v *= v;
            columns.push_back(t);
            names.push_back("trend2");
        }
    }
    return StatisticalTests::FitLinear(columns, response, names, trend >= 1);
}

std::string Short(double p) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%.3f", p);
    return buf;
}

}  // namespace

StationarityResult TimeSeries::ADFTest(const std::vector<double>& data, int max_lags, const std::string& regression, bool autolag) {
    StationarityResult r;
    r.regression = regression;
    if (regression != "c" && regression != "ct" && regression != "ctt" && regression != "n") {
        r.error_message = "regression must be \"c\", \"ct\", \"ctt\" or \"n\"";
        return r;
    }
    if (const std::string refusal = series_detail::SeriesRefusal(data, 8, "The ADF test"); !refusal.empty()) {
        r.error_message = refusal;
        return r;
    }
    const int n = static_cast<int>(data.size());
    if (*std::max_element(data.begin(), data.end()) == *std::min_element(data.begin(), data.end())) {
        r.error_message = "The series has the same value in every row";
        return r;
    }
    const int ntrend = TrendTerms(regression);
    if (max_lags < 0) {
        max_lags = static_cast<int>(std::ceil(12.0 * std::pow(n / 100.0, 0.25)));
        max_lags = std::min(n / 2 - ntrend - 1, max_lags);
        if (max_lags < 0) {
            r.error_message = "The series is too short for the ADF test";
            return r;
        }
    } else if (max_lags > n / 2 - ntrend - 1) {
        r.error_message = "max_lags must be below n / 2 - 1 - (trend terms)";
        return r;
    }
    std::vector<double> xdiff(static_cast<size_t>(n - 1));
    for (int i = 0; i + 1 < n; ++i) xdiff[static_cast<size_t>(i)] = data[static_cast<size_t>(i + 1)] - data[static_cast<size_t>(i)];
    int used = max_lags;
    double icbest = NAN;
    if (autolag) {
        // Every lag length fitted on the same rows (those left after max_lags); the smallest AIC wins.
        const int nobs = n - 1 - max_lags;
        int best = 0;
        for (int lag = 0; lag <= max_lags; ++lag) {
            const LinearFit fit = AdfRegression(data, xdiff, lag, nobs, regression);
            if (!fit.ok) {
                r.error_message = "The ADF regression failed: " + fit.error;
                return r;
            }
            if (std::isnan(icbest) || fit.aic < icbest) {
                icbest = fit.aic;
                best = lag;
            }
        }
        used = best;
    }
    const int nobs = n - 1 - used;
    const LinearFit fit = AdfRegression(data, xdiff, used, nobs, regression);
    if (!fit.ok) {
        r.error_message = "The ADF regression failed: " + fit.error;
        return r;
    }
    r.adf_statistic = fit.t[ntrend >= 1 ? 1 : 0];
    r.adf_pvalue = mackinnon::PValue(r.adf_statistic, regression);
    double crit[3];
    mackinnon::CriticalValues(regression, nobs, crit);
    r.adf_critical = {{"1%", crit[0]}, {"5%", crit[1]}, {"10%", crit[2]}};
    r.adf_stationary = r.adf_pvalue < 0.05;
    r.adf_lags = used;
    r.adf_nobs = nobs;
    r.adf_icbest = autolag ? icbest : fit.aic;
    r.success = true;
    return r;
}

StationarityResult TimeSeries::KPSSTest(const std::vector<double>& data, const std::string& regression, int nlags) {
    StationarityResult r;
    r.regression = regression;
    if (regression != "c" && regression != "ct") {
        r.error_message = "regression must be \"c\" or \"ct\"";
        return r;
    }
    if (const std::string refusal = series_detail::SeriesRefusal(data, 8, "The KPSS test"); !refusal.empty()) {
        r.error_message = refusal;
        return r;
    }
    const size_t n = data.size();
    std::vector<double> resids;
    double crit[4];
    if (regression == "ct") {
        std::vector<double> t(n);
        for (size_t i = 0; i < n; ++i) t[i] = static_cast<double>(i + 1);
        const LinearFit fit = StatisticalTests::FitLinear({t}, data, {"t"});
        if (!fit.ok) {
            r.error_message = "The KPSS regression failed: " + fit.error;
            return r;
        }
        resids = fit.residuals;
        crit[0] = 0.119, crit[1] = 0.146, crit[2] = 0.176, crit[3] = 0.216;
    } else {
        const double mean = std::accumulate(data.begin(), data.end(), 0.0) / static_cast<double>(n);
        resids.resize(n);
        for (size_t i = 0; i < n; ++i) resids[i] = data[i] - mean;
        crit[0] = 0.347, crit[1] = 0.463, crit[2] = 0.574, crit[3] = 0.739;
    }
    if (nlags >= static_cast<int>(n)) {
        r.error_message = "nlags must be below the number of observations";
        return r;
    }
    try {
        // sum resid[t] resid[t + i] is n times the biased autocovariance (the residuals have mean 0).
        const int covlags = static_cast<int>(std::pow(static_cast<double>(n), 2.0 / 9.0));
        const std::vector<double> acov = series_detail::Autocovariance(resids, std::max(covlags, std::max(0, nlags)));
        const auto dot = [&](int i) { return acov[static_cast<size_t>(i)] * static_cast<double>(n); };
        if (nlags < 0) {
            // Hobijn et al.: the Newey-West truncation from the residuals' autocovariances.
            double s0 = dot(0) / static_cast<double>(n), s1 = 0.0;
            for (int i = 1; i <= covlags; ++i) {
                const double prod = dot(i) / (static_cast<double>(n) / 2.0);
                s0 += prod;
                s1 += i * prod;
            }
            const double s_hat = s1 / s0;
            const double gamma_hat = 1.1447 * std::pow(s_hat * s_hat, 1.0 / 3.0);
            nlags = static_cast<int>(gamma_hat * std::pow(static_cast<double>(n), 1.0 / 3.0));
            nlags = std::min(nlags, static_cast<int>(n) - 1);
        }
        const std::vector<double> acov2 = nlags > covlags ? series_detail::Autocovariance(resids, nlags) : acov;
        double eta = 0.0, cumulative = 0.0;
        for (const double v : resids) {
            cumulative += v;
            eta += cumulative * cumulative;
        }
        eta /= static_cast<double>(n) * static_cast<double>(n);
        double s_hat = acov2[0] * static_cast<double>(n);
        for (int i = 1; i <= nlags; ++i) s_hat += 2.0 * acov2[static_cast<size_t>(i)] * static_cast<double>(n) * (1.0 - i / (nlags + 1.0));
        s_hat /= static_cast<double>(n);
        r.kpss_statistic = eta / s_hat;
        // p interpolated in Table 1 of Kwiatkowski et al. (1992), clipped at its ends.
        const double pvals[4] = {0.10, 0.05, 0.025, 0.01};
        double p = pvals[3];
        if (r.kpss_statistic <= crit[0]) p = pvals[0];
        else if (r.kpss_statistic >= crit[3]) p = pvals[3];
        else
            for (int i = 0; i < 3; ++i)
                if (r.kpss_statistic <= crit[i + 1]) {
                    p = pvals[i] + (pvals[i + 1] - pvals[i]) * (r.kpss_statistic - crit[i]) / (crit[i + 1] - crit[i]);
                    break;
                }
        r.kpss_pvalue = p;
        r.kpss_critical = {{"10%", crit[0]}, {"5%", crit[1]}, {"2.5%", crit[2]}, {"1%", crit[3]}};
        r.kpss_stationary = r.kpss_pvalue >= 0.05;
        r.kpss_lags = nlags;
        r.success = true;
    } catch (const af::exception& e) {
        r.error_message = std::string("The KPSS test failed on the ArrayFire device: ") + e.what();
    }
    return r;
}

StationarityResult TimeSeries::TestStationarity(const std::vector<double>& data, int max_lags, const std::string& regression) {
    StationarityResult r = ADFTest(data, max_lags, regression, true);
    if (!r.success) return r;
    const StationarityResult kpss = KPSSTest(data, regression == "ct" ? "ct" : "c", -1);
    if (!kpss.success) {
        r.success = false;
        r.error_message = kpss.error_message;
        return r;
    }
    r.kpss_statistic = kpss.kpss_statistic;
    r.kpss_pvalue = kpss.kpss_pvalue;
    r.kpss_critical = kpss.kpss_critical;
    r.kpss_stationary = kpss.kpss_stationary;
    r.kpss_lags = kpss.kpss_lags;
    r.is_stationary = r.adf_stationary && r.kpss_stationary;

    // The order of differencing: the smallest d (up to 2) at which the series is
    // stationary enough: KPSS does not reject stationarity at 5% and ADF does not
    // keep a unit root (p below 0.10). KPSS alone (Hyndman's ndiffs) misses some
    // random walks; both at 5% over-differences series like the airline one.
    const auto enough = [](double adf_p, double kpss_p) { return kpss_p >= 0.05 && adf_p < 0.10; };
    r.suggested_differencing = enough(r.adf_pvalue, r.kpss_pvalue) ? 0 : -1;
    std::vector<double> series = data;
    for (int d = 1; d <= 2 && r.suggested_differencing < 0; ++d) {
        series = Difference(series, 1);
        const StationarityResult adf = ADFTest(series, -1, "c", true);
        const StationarityResult k = KPSSTest(series, "c", -1);
        if (!adf.success || !k.success) break;
        StationarityResult::Differenced diff;
        diff.d = d;
        diff.adf_statistic = adf.adf_statistic;
        diff.adf_pvalue = adf.adf_pvalue;
        diff.kpss_statistic = k.kpss_statistic;
        diff.kpss_pvalue = k.kpss_pvalue;
        diff.stationary = enough(adf.adf_pvalue, k.kpss_pvalue);
        r.differenced.push_back(diff);
        if (diff.stationary) r.suggested_differencing = d;
    }

    const int window = std::max(10, static_cast<int>(data.size()) / 10);
    try {
        series_detail::Rolling(data, window, r.rolling_mean, r.rolling_std);
        r.rolling_window = window;
    } catch (const af::exception& e) {
        r.success = false;
        r.error_message = std::string("The rolling statistics failed on the ArrayFire device: ") + e.what();
        return r;
    }

    const std::string around = regression == "ct" ? "a trend" : "its mean";
    if (r.is_stationary) {
        r.analysis = "Stationary around " + around + ": ADF rejects a unit root (p " + Short(r.adf_pvalue) + ") and KPSS does not reject stationarity (p " +
                     Short(r.kpss_pvalue) + ").";
    } else if (r.adf_stationary && !r.kpss_stationary) {
        r.analysis = "The tests disagree: ADF rejects a unit root (p " + Short(r.adf_pvalue) + ") but KPSS rejects stationarity (p " + Short(r.kpss_pvalue) +
                     "); the series may be stationary around a trend, or changing slowly.";
    } else if (!r.adf_stationary && r.kpss_stationary) {
        r.analysis = "The tests disagree: KPSS does not reject stationarity (p " + Short(r.kpss_pvalue) + ") but ADF cannot reject a unit root (p " +
                     Short(r.adf_pvalue) + "); the series is short, or the test has little power here.";
    } else {
        r.analysis = "Not stationary: ADF cannot reject a unit root (p " + Short(r.adf_pvalue) + ") and KPSS rejects stationarity (p " + Short(r.kpss_pvalue) + ").";
    }
    if (r.suggested_differencing > 0) {
        const StationarityResult::Differenced& chosen = r.differenced.back();
        r.analysis += std::string(" ") + (r.suggested_differencing == 1 ? "One difference makes" : "Two differences make") + " it stationary enough (KPSS p " +
                      Short(chosen.kpss_pvalue) + ", ADF p " + Short(chosen.adf_pvalue) + ").";
    } else if (r.suggested_differencing < 0) {
        r.analysis += " Two differences still do not make it stationary.";
    }
    return r;
}

std::vector<double> TimeSeries::Difference(const std::vector<double>& data, int order) {
    std::vector<double> out = data;
    for (int k = 0; k < order && out.size() > 1; ++k) {
        std::vector<double> next(out.size() - 1);
        for (size_t i = 0; i + 1 < out.size(); ++i) next[i] = out[i + 1] - out[i];
        out = std::move(next);
    }
    return out;
}

std::vector<double> TimeSeries::SeasonalDifference(const std::vector<double>& data, int period, int order) {
    std::vector<double> out = data;
    if (period < 1) return out;
    for (int k = 0; k < order && out.size() > static_cast<size_t>(period); ++k) {
        std::vector<double> next(out.size() - static_cast<size_t>(period));
        for (size_t i = 0; i < next.size(); ++i) next[i] = out[i + static_cast<size_t>(period)] - out[i];
        out = std::move(next);
    }
    return out;
}

}  // namespace cyxwiz
