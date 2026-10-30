// Decomposition (TOFIX129 D6). Classical: the trend by a two-sided moving
// average on the device (empty at the edges), the seasonal part as the means
// by position of the detrended series, as statsmodels seasonal_decompose.
// STL: Cleveland's seasonal-trend decomposition by loess, a port of the
// Fortran routine statsmodels wraps (stl.f: ss, fts, ess, est, rwts), which
// is sequential by nature and runs on the host over the series.

#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/time_series.h"
#include "series_device.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace cyxwiz {

namespace {

double FiniteVariance(const std::vector<double>& v) {
    double sum = 0.0;
    size_t count = 0;
    for (const double x : v)
        if (std::isfinite(x)) {
            sum += x;
            ++count;
        }
    if (count < 2) return 0.0;
    const double mean = sum / static_cast<double>(count);
    double ss = 0.0;
    for (const double x : v)
        if (std::isfinite(x)) ss += (x - mean) * (x - mean);
    return ss / static_cast<double>(count);
}

void FinishStrengths(DecompositionResult& r) {
    r.trend_strength = series_detail::Strength(r.residual, r.trend);
    r.seasonal_strength = series_detail::Strength(r.residual, r.seasonal);
    r.residual_variance = FiniteVariance(r.residual);
}

// ---- STL (Cleveland et al. 1990), indices as in the Fortran: xs, nleft, nright are 1-based.

struct Stl {
    int n = 0, np = 0, ns = 0, nt = 0, nl = 0, isdeg = 1, itdeg = 1, ildeg = 1, nsjump = 1, ntjump = 1, nljump = 1;
    std::vector<double> y, season, trend, rw;
    std::vector<std::vector<double>> work;  // 5 x (n + 2 np)
    bool use_rw = false;

    // Loess estimate at xs from y[nleft - 1 .. nright - 1]; NaN when every weight is 0.
    double Est(const double* y_, int n_, int len, int ideg, int xs, int nleft, int nright, double* w, bool userw, const double* rw_) const {
        const double range = static_cast<double>(n_) - 1.0;
        double h = static_cast<double>(std::max(xs - nleft, nright - xs));
        if (len > n_) h += static_cast<double>((len - n_) / 2);
        const double h9 = 0.999 * h, h1 = 0.001 * h;
        double a = 0.0;
        for (int j = nleft - 1; j < nright; ++j) {
            w[j] = 0.0;
            const double r = std::fabs(static_cast<double>(j + 1 - xs));
            if (r <= h9) {
                if (r <= h1) w[j] = 1.0;
                else w[j] = std::pow(1.0 - std::pow(r / h, 3.0), 3.0);
                if (userw) w[j] *= rw_[j];
                a += w[j];
            }
        }
        if (a <= 0.0) return NAN;
        for (int j = nleft - 1; j < nright; ++j) w[j] /= a;
        if (h > 0.0 && ideg > 0) {
            a = 0.0;
            for (int j = nleft - 1; j < nright; ++j) a += w[j] * static_cast<double>(j + 1);
            double b = static_cast<double>(xs) - a, c = 0.0;
            for (int j = nleft - 1; j < nright; ++j) c += w[j] * (static_cast<double>(j + 1) - a) * (static_cast<double>(j + 1) - a);
            if (std::sqrt(c) > 0.001 * range) {
                b /= c;
                for (int j = nleft - 1; j < nright; ++j) w[j] *= (b * (static_cast<double>(j + 1) - a) + 1.0);
            }
        }
        double ys = 0.0;
        for (int j = nleft - 1; j < nright; ++j) ys += w[j] * y_[j];
        return ys;
    }

    // Loess smooth of y_[0 .. n_ - 1] into ys (res is scratch of length n_).
    void Ess(const double* y_, int n_, int len, int ideg, int njump, bool userw, const double* rw_, double* ys, double* res) const {
        if (n_ < 2) {
            ys[0] = y_[0];
            return;
        }
        const int newnj = std::min(njump, n_ - 1);
        int nleft = 1, nright = n_;
        const auto at = [&](int i) {
            ys[i] = Est(y_, n_, len, ideg, i + 1, nleft, nright, res, userw, rw_);
            if (std::isnan(ys[i])) ys[i] = y_[i];
        };
        if (len >= n_) {
            nleft = 1;
            nright = n_;
            for (int i = 0; i < n_; i += newnj) at(i);
        } else if (newnj == 1) {
            const int nsh = (len + 2) / 2;
            nleft = 1;
            nright = len;
            for (int i = 0; i < n_; ++i) {
                if (i + 1 > nsh && nright != n_) {
                    ++nleft;
                    ++nright;
                }
                at(i);
            }
        } else {
            const int nsh = (len + 1) / 2;
            for (int i = 0; i < n_; i += newnj) {
                if (i + 1 < nsh) {
                    nleft = 1;
                    nright = len;
                } else if (i + 1 >= n_ - nsh + 1) {
                    nleft = n_ - len + 1;
                    nright = n_;
                } else {
                    nleft = i + 1 - nsh + 1;
                    nright = len + i + 1 - nsh;
                }
                at(i);
            }
        }
        if (newnj == 1) return;
        for (int i = 0; i < n_ - newnj; i += newnj) {
            const double delta = (ys[i + newnj] - ys[i]) / newnj;
            for (int j = i; j < i + newnj; ++j) ys[j] = ys[i] + delta * (j - i);
        }
        const int k = ((n_ - 1) / newnj) * newnj + 1;
        if (k != n_) {
            ys[n_ - 1] = Est(y_, n_, len, ideg, n_, nleft, nright, res, userw, rw_);
            if (std::isnan(ys[n_ - 1])) ys[n_ - 1] = y_[n_ - 1];
            if (k != n_ - 1) {
                const double delta = (ys[n_ - 1] - ys[k - 1]) / (n_ - k);
                for (int j = k; j < n_; ++j) ys[j] = ys[k - 1] + delta * (j + 1 - k);
            }
        }
    }

    static void Ma(const double* x, int n_, int len, double* ave) {
        const int newn = n_ - len + 1;
        double v = 0.0;
        for (int i = 0; i < len; ++i) v += x[i];
        ave[0] = v / len;
        for (int j = 1, k = len, m = 0; j < newn; ++j, ++k, ++m) {
            v += x[k] - x[m];
            ave[j] = v / len;
        }
    }

    // Low-pass filter of the cycle-subseries smooth: moving averages of np, np and 3.
    void Fts() {
        const int total = n + 2 * np;
        Ma(work[1].data(), total, np, work[2].data());
        Ma(work[2].data(), total - np + 1, np, work[0].data());
        Ma(work[0].data(), total - 2 * np + 2, 3, work[2].data());
    }

    // Smooth each cycle-subseries of work[0] (y - trend) into work[1], extended one period each way.
    void Ss() {
        double* y_ = work[0].data();
        double* season_ = work[1].data();
        double* work1 = work[2].data();
        double* work2 = work[3].data();
        double* work3 = work[4].data();
        double* work4 = season.data();  // scratch, as the Fortran does
        for (int j = 0; j < np; ++j) {
            const int k = (n - (j + 1)) / np + 1;
            for (int i = 0; i < k; ++i) work1[i] = y_[i * np + j];
            if (use_rw)
                for (int i = 0; i < k; ++i) work3[i] = rw[static_cast<size_t>(i * np + j)];
            Ess(work1, k, ns, isdeg, nsjump, use_rw, work3, work2 + 1, work4);
            const int nright = std::min(ns, k);
            work2[0] = Est(work1, k, ns, isdeg, 0, 1, nright, work4, use_rw, work3);
            if (std::isnan(work2[0])) work2[0] = work2[1];
            const int nleft = std::max(1, k - ns + 1);
            work2[k + 1] = Est(work1, k, ns, isdeg, k + 1, nleft, k, work4, use_rw, work3);
            if (std::isnan(work2[k + 1])) work2[k + 1] = work2[k];
            for (int m = 0; m < k + 2; ++m) season_[m * np + j] = work2[m];
        }
    }

    void OneStep(int inner) {
        for (int it = 0; it < inner; ++it) {
            for (int i = 0; i < n; ++i) work[0][static_cast<size_t>(i)] = y[static_cast<size_t>(i)] - trend[static_cast<size_t>(i)];
            Ss();
            Fts();
            Ess(work[2].data(), n, nl, ildeg, nljump, false, work[3].data(), work[0].data(), work[4].data());
            for (int i = 0; i < n; ++i) {
                season[static_cast<size_t>(i)] = work[1][static_cast<size_t>(np + i)] - work[0][static_cast<size_t>(i)];
                work[0][static_cast<size_t>(i)] = y[static_cast<size_t>(i)] - season[static_cast<size_t>(i)];
            }
            Ess(work[0].data(), n, nt, itdeg, ntjump, use_rw, rw.data(), trend.data(), work[2].data());
        }
    }

    // Robustness weights from the residuals (bisquare, 6 x the median absolute residual).
    void Rwts() {
        for (int i = 0; i < n; ++i) rw[static_cast<size_t>(i)] = std::fabs(y[static_cast<size_t>(i)] - work[0][static_cast<size_t>(i)]);
        std::vector<double> sorted = rw;
        const int mid0 = n / 2, mid1 = n - mid0 - 1;
        std::nth_element(sorted.begin(), sorted.begin() + mid0, sorted.end());
        const double a = sorted[static_cast<size_t>(mid0)];
        std::nth_element(sorted.begin(), sorted.begin() + mid1, sorted.end());
        const double cmad = 3.0 * (a + sorted[static_cast<size_t>(mid1)]);
        if (cmad == 0.0) {
            std::fill(rw.begin(), rw.end(), 1.0);
            return;
        }
        const double c9 = 0.999 * cmad, c1 = 0.001 * cmad;
        for (double& w : rw) {
            if (w <= c1) w = 1.0;
            else if (w <= c9) w = std::pow(1.0 - std::pow(w / cmad, 2.0), 2.0);
            else w = 0.0;
        }
    }

    void Fit(int inner, int outer) {
        std::fill(season.begin(), season.end(), 0.0);
        std::fill(trend.begin(), trend.end(), 0.0);
        std::fill(rw.begin(), rw.end(), 1.0);
        use_rw = false;
        for (int k = 0;;) {
            OneStep(inner);
            if (++k > outer) break;
            for (int i = 0; i < n; ++i) work[0][static_cast<size_t>(i)] = trend[static_cast<size_t>(i)] + season[static_cast<size_t>(i)];
            Rwts();
            use_rw = true;
        }
    }
};

}  // namespace

DecompositionResult TimeSeries::Decompose(const std::vector<double>& data, int period, const std::string& method) {
    DecompositionResult r;
    r.period = period;
    r.method = method;
    if (method != "additive" && method != "multiplicative") {
        r.error_message = "method must be \"additive\" or \"multiplicative\"";
        return r;
    }
    if (period < 2) {
        r.error_message = "The period must be at least 2";
        return r;
    }
    if (const std::string refusal = series_detail::SeriesRefusal(data, 2 * static_cast<size_t>(period), "Decomposition"); !refusal.empty()) {
        r.error_message = refusal + " (two full periods)";
        return r;
    }
    const bool multiplicative = method == "multiplicative";
    if (multiplicative)
        for (const double v : data)
            if (v <= 0.0) {
                r.error_message = "A multiplicative decomposition needs values above 0";
                return r;
            }
    try {
        const size_t n = data.size(), m = static_cast<size_t>(period);
        r.original = data;
        // The centred moving average: for an even period, half weights at both ends.
        std::vector<double> filter;
        if (period % 2 == 0) {
            filter.assign(m + 1, 1.0 / period);
            filter.front() = filter.back() = 0.5 / period;
        } else {
            filter.assign(m, 1.0 / period);
        }
        r.trend = series_detail::TwoSidedFilter(data, filter);
        std::vector<double> detrended(n);
        for (size_t i = 0; i < n; ++i) detrended[i] = multiplicative ? data[i] / r.trend[i] : data[i] - r.trend[i];
        std::vector<double> averages = series_detail::SeasonalMeans(detrended, period);
        const double centre = std::accumulate(averages.begin(), averages.end(), 0.0) / static_cast<double>(m);
        for (double& a : averages) a = multiplicative ? a / centre : a - centre;
        r.seasonal.resize(n);
        r.residual.resize(n);
        for (size_t i = 0; i < n; ++i) {
            r.seasonal[i] = averages[i % m];
            r.residual[i] = multiplicative ? data[i] / r.seasonal[i] / r.trend[i] : detrended[i] - r.seasonal[i];
        }
        FinishStrengths(r);
        r.success = true;
    } catch (const af::exception& e) {
        r.error_message = std::string("Decomposition failed on the ArrayFire device: ") + e.what();
    }
    return r;
}

DecompositionResult TimeSeries::STLDecompose(const std::vector<double>& data, int period, int seasonal_window, int trend_window,
                                             int low_pass_window, bool robust) {
    DecompositionResult r;
    r.period = period;
    r.method = "stl";
    r.robust = robust;
    if (period < 2) {
        r.error_message = "The period must be at least 2";
        return r;
    }
    if (const std::string refusal = series_detail::SeriesRefusal(data, 2 * static_cast<size_t>(period), "STL"); !refusal.empty()) {
        r.error_message = refusal + " (two full periods)";
        return r;
    }
    if (seasonal_window < 3 || seasonal_window % 2 == 0) {
        r.error_message = "The seasonal window must be an odd number of 3 or more";
        return r;
    }
    if (trend_window < 0) {
        trend_window = static_cast<int>(std::ceil(1.5 * period / (1.0 - 1.5 / seasonal_window)));
        if (trend_window % 2 == 0) ++trend_window;
    }
    if (trend_window < 3 || trend_window % 2 == 0 || trend_window <= period) {
        r.error_message = "The trend window must be an odd number above the period";
        return r;
    }
    if (low_pass_window < 0) {
        low_pass_window = period + 1;
        if (low_pass_window % 2 == 0) ++low_pass_window;
    }
    if (low_pass_window < 3 || low_pass_window % 2 == 0 || low_pass_window <= period) {
        r.error_message = "The low-pass window must be an odd number above the period";
        return r;
    }
    r.seasonal_window = seasonal_window;
    r.trend_window = trend_window;
    r.low_pass_window = low_pass_window;
    const int n = static_cast<int>(data.size());
    Stl stl;
    stl.n = n;
    stl.np = period;
    stl.ns = seasonal_window;
    stl.nt = trend_window;
    stl.nl = low_pass_window;
    stl.y = data;
    stl.season.assign(static_cast<size_t>(n), 0.0);
    stl.trend.assign(static_cast<size_t>(n), 0.0);
    stl.rw.assign(static_cast<size_t>(n), 1.0);
    stl.work.assign(5, std::vector<double>(static_cast<size_t>(n + 2 * period), 0.0));
    stl.Fit(robust ? 2 : 5, robust ? 15 : 0);
    r.original = data;
    r.trend = stl.trend;
    r.seasonal = stl.season;
    r.residual.resize(static_cast<size_t>(n));
    for (size_t i = 0; i < r.residual.size(); ++i) r.residual[i] = data[i] - r.trend[i] - r.seasonal[i];
    FinishStrengths(r);
    r.success = true;
    return r;
}

}  // namespace cyxwiz
