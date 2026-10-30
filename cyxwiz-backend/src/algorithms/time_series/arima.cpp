// ARIMA (TOFIX129 D6): the series differenced d times, an ARMA(p, q) with a
// mean (when d = 0) fitted by exact Gaussian maximum likelihood through the
// Kalman filter on Harvey's state space (what statsmodels ARIMA does by
// default), stationarity and invertibility kept by the partial-autocorrelation
// transform, orders chosen by AIC when asked, forecasts from the filtered
// state and intervals from the psi weights. The filter is sequential by nature
// and runs on the host over the series.

#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/time_series.h"
#include "series_device.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>

namespace cyxwiz {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Unconstrained x -> AR coefficients of a stationary polynomial (statsmodels
// constrain_stationary_univariate): partial autocorrelations x / sqrt(1 + x^2),
// then the Durbin-Levinson recursion.
std::vector<double> Constrain(const std::vector<double>& x) {
    const size_t k = x.size();
    std::vector<double> r(k), a(k, 0.0), previous(k, 0.0);
    for (size_t i = 0; i < k; ++i) r[i] = x[i] / std::sqrt(1.0 + x[i] * x[i]);
    for (size_t j = 0; j < k; ++j) {
        a[j] = r[j];
        for (size_t i = 0; i < j; ++i) a[i] = previous[i] - r[j] * previous[j - 1 - i];
        previous = a;
    }
    return a;
}

// The inverse: AR coefficients of a stationary polynomial -> unconstrained x (NaN when not stationary).
std::vector<double> Unconstrain(const std::vector<double>& phi) {
    const size_t k = phi.size();
    std::vector<double> a = phi, r(k, 0.0);
    for (size_t j = k; j-- > 0;) {
        const double rj = a[j];
        if (!(std::fabs(rj) < 1.0)) return std::vector<double>(k, NAN);
        r[j] = rj;
        std::vector<double> previous(j, 0.0);
        for (size_t i = 0; i < j; ++i) previous[i] = (a[i] + rj * a[j - 1 - i]) / (1.0 - rj * rj);
        for (size_t i = 0; i < j; ++i) a[i] = previous[i];
    }
    std::vector<double> x(k);
    for (size_t i = 0; i < k; ++i) x[i] = r[i] / std::sqrt(1.0 - r[i] * r[i]);
    return x;
}

struct Arma {
    std::vector<double> ar, ma;
    double mu = 0.0;
};

struct Filtered {
    double loglik = -INFINITY;  // concentrated over sigma^2
    double sigma2 = NAN;
    std::vector<double> innovations;  // v_t
    std::vector<double> state;        // a_{n|n}: the filtered state after the last observation
    std::vector<std::vector<double>> covariance;  // P_{n|n}
    bool ok = false;
};

// Solve the Lyapunov equation P = T P T' + R R' for the stationary state covariance (r x r, r small).
std::vector<std::vector<double>> StationaryCovariance(const std::vector<std::vector<double>>& T, const std::vector<double>& R) {
    const size_t r = R.size(), m = r * r;
    // (I - T kron T) vec(P) = vec(R R')
    std::vector<std::vector<double>> A(m, std::vector<double>(m + 1, 0.0));
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < r; ++j) {
            const size_t row = i * r + j;
            A[row][row] += 1.0;
            for (size_t k = 0; k < r; ++k)
                for (size_t l = 0; l < r; ++l) A[row][k * r + l] -= T[i][k] * T[j][l];
            A[row][m] = R[i] * R[j];
        }
    for (size_t c = 0; c < m; ++c) {
        size_t pivot = c;
        for (size_t rr = c + 1; rr < m; ++rr)
            if (std::fabs(A[rr][c]) > std::fabs(A[pivot][c])) pivot = rr;
        std::swap(A[c], A[pivot]);
        if (std::fabs(A[c][c]) < 1e-300) continue;
        for (size_t rr = 0; rr < m; ++rr) {
            if (rr == c) continue;
            const double f = A[rr][c] / A[c][c];
            if (f == 0.0) continue;
            for (size_t cc = c; cc <= m; ++cc) A[rr][cc] -= f * A[c][cc];
        }
    }
    std::vector<std::vector<double>> P(r, std::vector<double>(r, 0.0));
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < r; ++j) P[i][j] = std::fabs(A[i * r + j][i * r + j]) < 1e-300 ? 0.0 : A[i * r + j][m] / A[i * r + j][i * r + j];
    return P;
}

// Harvey's state space for the ARMA: state dimension r = max(p, q + 1).
struct StateSpace {
    size_t r = 1;
    std::vector<std::vector<double>> T;
    std::vector<double> R;
    explicit StateSpace(const Arma& model) {
        r = std::max(model.ar.size(), model.ma.size() + 1);
        T.assign(r, std::vector<double>(r, 0.0));
        R.assign(r, 0.0);
        for (size_t i = 0; i < r; ++i) {
            if (i < model.ar.size()) T[i][0] = model.ar[i];
            if (i + 1 < r) T[i][i + 1] = 1.0;
        }
        R[0] = 1.0;
        for (size_t j = 0; j < model.ma.size(); ++j) R[j + 1] = model.ma[j];
    }
    std::vector<double> Apply(const std::vector<double>& a) const {
        std::vector<double> out(r, 0.0);
        for (size_t i = 0; i < r; ++i)
            for (size_t k = 0; k < r; ++k) out[i] += T[i][k] * a[k];
        return out;
    }
};

Filtered KalmanFilter(const std::vector<double>& w, const Arma& model) {
    Filtered f;
    const StateSpace ss(model);
    const size_t r = ss.r, n = w.size();
    std::vector<double> a(r, 0.0);
    std::vector<std::vector<double>> P = StationaryCovariance(ss.T, ss.R);
    for (size_t i = 0; i < r; ++i)
        if (!std::isfinite(P[i][i]) || P[i][i] < 0.0) return f;
    double sum_log_f = 0.0, sum_v2_f = 0.0;
    f.innovations.resize(n);
    std::vector<double> K(r), Ta(r);
    std::vector<std::vector<double>> TP(r, std::vector<double>(r));
    for (size_t t = 0; t < n; ++t) {
        const double v = w[t] - model.mu - a[0];
        const double F = P[0][0];
        if (!(F > 0.0) || !std::isfinite(F)) return f;
        f.innovations[t] = v;
        sum_log_f += std::log(F);
        sum_v2_f += v * v / F;
        for (size_t i = 0; i < r; ++i) K[i] = P[i][0] / F;
        for (size_t i = 0; i < r; ++i) a[i] += K[i] * v;
        for (size_t i = 0; i < r; ++i)
            for (size_t j = 0; j < r; ++j) P[i][j] -= K[i] * K[j] * F;
        if (t + 1 == n) break;
        // Predict: a = T a, P = T P T' + R R'.
        a = ss.Apply(a);
        for (size_t i = 0; i < r; ++i)
            for (size_t j = 0; j < r; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < r; ++k) s += ss.T[i][k] * P[k][j];
                TP[i][j] = s;
            }
        for (size_t i = 0; i < r; ++i)
            for (size_t j = 0; j < r; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < r; ++k) s += TP[i][k] * ss.T[j][k];
                P[i][j] = s + ss.R[i] * ss.R[j];
            }
    }
    f.sigma2 = sum_v2_f / static_cast<double>(n);
    f.loglik = -0.5 * static_cast<double>(n) * (std::log(2.0 * kPi) + std::log(f.sigma2) + 1.0) - 0.5 * sum_log_f;
    f.state = a;
    f.covariance = P;
    f.ok = std::isfinite(f.loglik);
    return f;
}

struct Problem {
    const std::vector<double>& w;
    int p, q;
    bool mean;
    Arma Unpack(const std::vector<double>& x) const {
        Arma m;
        m.ar = Constrain(std::vector<double>(x.begin(), x.begin() + p));
        const std::vector<double> ma = Constrain(std::vector<double>(x.begin() + p, x.begin() + p + q));
        m.ma.resize(ma.size());
        for (size_t i = 0; i < ma.size(); ++i) m.ma[i] = -ma[i];  // invertible: 1 + theta B has its roots outside the circle
        m.mu = mean ? x[static_cast<size_t>(p + q)] : 0.0;
        return m;
    }
    double NegativeLogLik(const std::vector<double>& x) const {
        const Filtered f = KalmanFilter(w, Unpack(x));
        return f.ok ? -f.loglik : 1e300;
    }
};

// Nelder-Mead (all coordinates free), restarted once from its result.
std::vector<double> Minimise(const std::function<double(const std::vector<double>&)>& f, std::vector<double> x, double step, int max_evals) {
    const size_t d = x.size();
    if (d == 0) return x;
    for (int round = 0; round < 2; ++round) {
        std::vector<std::vector<double>> simplex(d + 1, x);
        std::vector<double> values(d + 1);
        for (size_t k = 1; k <= d; ++k) simplex[k][k - 1] += step;
        for (size_t k = 0; k <= d; ++k) values[k] = f(simplex[k]);
        int evals = static_cast<int>(d + 1);
        std::vector<size_t> order(d + 1);
        while (evals < max_evals) {
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return values[a] < values[b]; });
            const size_t best = order[0], worst = order[d], second = order[d - 1];
            double spread = 0.0;
            for (size_t k = 0; k <= d; ++k)
                for (size_t i = 0; i < d; ++i) spread = std::max(spread, std::fabs(simplex[k][i] - simplex[best][i]));
            if (std::fabs(values[worst] - values[best]) <= 1e-10 * (1.0 + std::fabs(values[best])) && spread < 1e-6) break;
            std::vector<double> centroid(d, 0.0);
            for (size_t k = 0; k <= d; ++k)
                if (k != worst)
                    for (size_t i = 0; i < d; ++i) centroid[i] += simplex[k][i] / d;
            const auto along = [&](double t) {
                std::vector<double> z(d);
                for (size_t i = 0; i < d; ++i) z[i] = centroid[i] + t * (simplex[worst][i] - centroid[i]);
                return z;
            };
            const std::vector<double> reflected = along(-1.0);
            const double fr = f(reflected);
            ++evals;
            if (fr < values[best]) {
                const std::vector<double> expanded = along(-2.0);
                const double fe = f(expanded);
                ++evals;
                simplex[worst] = fe < fr ? expanded : reflected;
                values[worst] = std::min(fe, fr);
            } else if (fr < values[second]) {
                simplex[worst] = reflected;
                values[worst] = fr;
            } else {
                const std::vector<double> contracted = along(fr < values[worst] ? -0.5 : 0.5);
                const double fc = f(contracted);
                ++evals;
                if (fc < std::min(fr, values[worst])) {
                    simplex[worst] = contracted;
                    values[worst] = fc;
                } else {
                    for (size_t k = 0; k <= d; ++k) {
                        if (k == best) continue;
                        for (size_t i = 0; i < d; ++i) simplex[k][i] = simplex[best][i] + 0.5 * (simplex[k][i] - simplex[best][i]);
                        values[k] = f(simplex[k]);
                        ++evals;
                    }
                }
            }
        }
        x = simplex[static_cast<size_t>(std::min_element(values.begin(), values.end()) - values.begin())];
        step *= 0.1;
    }
    return x;
}

struct Fitted {
    Arma model;
    Filtered filter;
    double aic = NAN, bic = NAN;
    int k = 0;
    bool ok = false;
};

Fitted FitArma(const std::vector<double>& w, int p, int q, bool mean) {
    Fitted out;
    const Problem problem{w, p, q, mean};
    const size_t n = w.size();
    const double average = std::accumulate(w.begin(), w.end(), 0.0) / static_cast<double>(n);
    // Two starts: everything 0, and the Yule-Walker autoregression.
    std::vector<std::vector<double>> starts;
    std::vector<double> zero(static_cast<size_t>(p + q + (mean ? 1 : 0)), 0.0);
    if (mean) zero.back() = average;
    starts.push_back(zero);
    if (p > 0) {
        const AutocorrelationResult acf = TimeSeries::ComputePACF(w, p);
        if (acf.success) {
            // The PACF values are the partial autocorrelations the transform wants.
            std::vector<double> x = zero;
            for (int i = 0; i < p; ++i) {
                const double r = std::clamp(acf.pacf[static_cast<size_t>(i + 1)], -0.98, 0.98);
                x[static_cast<size_t>(i)] = r / std::sqrt(1.0 - r * r);
            }
            starts.push_back(x);
        }
    }
    double best = INFINITY;
    std::vector<double> best_x;
    for (const auto& start : starts) {
        const std::vector<double> x = Minimise([&](const std::vector<double>& z) { return problem.NegativeLogLik(z); }, start, 0.5, 4000 * static_cast<int>(start.size() + 1));
        const double value = problem.NegativeLogLik(x);
        if (value < best) {
            best = value;
            best_x = x;
        }
    }
    if (!std::isfinite(best)) return out;
    out.model = problem.Unpack(best_x);
    out.filter = KalmanFilter(w, out.model);
    if (!out.filter.ok) return out;
    out.k = p + q + 1 + (mean ? 1 : 0);
    out.aic = -2.0 * out.filter.loglik + 2.0 * out.k;
    out.bic = -2.0 * out.filter.loglik + std::log(static_cast<double>(n)) * out.k;
    out.ok = true;
    return out;
}

// psi weights of theta(B) / phi(B) / (1 - B)^d, j = 0 .. count - 1.
std::vector<double> PsiWeights(const Arma& m, int d, int count) {
    std::vector<double> psi(static_cast<size_t>(count), 0.0);
    if (count == 0) return psi;
    psi[0] = 1.0;
    for (int j = 1; j < count; ++j) {
        double v = static_cast<size_t>(j) <= m.ma.size() ? m.ma[static_cast<size_t>(j - 1)] : 0.0;
        for (size_t i = 0; i < m.ar.size() && static_cast<int>(i) < j; ++i) v += m.ar[i] * psi[static_cast<size_t>(j) - 1 - i];
        psi[static_cast<size_t>(j)] = v;
    }
    for (int k = 0; k < d; ++k)
        for (int j = 1; j < count; ++j) psi[static_cast<size_t>(j)] += psi[static_cast<size_t>(j - 1)];
    return psi;
}

}  // namespace

ForecastResult TimeSeries::ARIMA(const std::vector<double>& data, int horizon, int p, int d, int q, double level) {
    ForecastResult r;
    r.method = "ARIMA";
    r.horizon = horizon;
    r.level = level;
    if (const std::string refusal = series_detail::SeriesRefusal(data, 10, "ARIMA"); !refusal.empty()) {
        r.error_message = refusal;
        return r;
    }
    if (horizon < 0 || !(level > 0.0 && level < 1.0)) {
        r.error_message = "The horizon cannot be negative and the level is between 0 and 1";
        return r;
    }
    if (p > 5 || q > 5 || d > 2) {
        r.error_message = "ARIMA orders go up to p 5, d 2, q 5";
        return r;
    }
    if (d < 0) {
        const StationarityResult s = TestStationarity(data, -1, "c");
        d = s.success ? std::max(0, std::min(2, s.suggested_differencing < 0 ? 2 : s.suggested_differencing)) : 1;
    }
    std::vector<double> w = Difference(data, d);
    const int n = static_cast<int>(w.size());
    const bool mean = d == 0;
    const int need = std::max(p, 0) + std::max(q, 0) + 5;
    if (n < need || n < 8) {
        r.error_message = "Too few values after differencing for these orders";
        return r;
    }
    // Orders by AIC when asked (p, q up to 3).
    Fitted fit;
    if (p < 0 || q < 0) {
        double best = INFINITY;
        // The bounds first: p and q take the best orders as the search goes.
        // Orders up to 3 each; a long series (over 200,000 rows) tries up to 1 each, every fit being a full Kalman pass.
        const int top = n > 200000 ? 1 : 3;
        const int p_lo = p < 0 ? 0 : p, p_hi = p < 0 ? top : p, q_lo = q < 0 ? 0 : q, q_hi = q < 0 ? top : q;
        for (int pp = p_lo; pp <= p_hi; ++pp)
            for (int qq = q_lo; qq <= q_hi; ++qq) {
                if (n < pp + qq + 5) continue;
                const Fitted candidate = FitArma(w, pp, qq, mean);
                if (candidate.ok && candidate.aic < best) {
                    best = candidate.aic;
                    fit = candidate;
                    p = pp;
                    q = qq;
                }
            }
    } else {
        fit = FitArma(w, p, q, mean);
    }
    if (!fit.ok) {
        r.error_message = "The ARIMA likelihood could not be maximised for these orders";
        return r;
    }
    r.method = "ARIMA(" + std::to_string(p) + "," + std::to_string(d) + "," + std::to_string(q) + ")";
    for (int i = 0; i < p; ++i) r.parameters["ar.L" + std::to_string(i + 1)] = fit.model.ar[static_cast<size_t>(i)];
    for (int i = 0; i < q; ++i) r.parameters["ma.L" + std::to_string(i + 1)] = fit.model.ma[static_cast<size_t>(i)];
    if (mean) r.parameters["const"] = fit.model.mu;
    r.parameters["sigma2"] = fit.filter.sigma2;
    r.log_likelihood = fit.filter.loglik;
    r.aic = fit.aic;
    r.bic = fit.bic;
    r.n_params = fit.k;

    // One-step fits of the differenced series back on the original scale.
    const size_t total = data.size();
    r.fitted_values.assign(total, NAN);
    r.residuals.assign(total, NAN);
    std::vector<double> w_fit(static_cast<size_t>(n));
    for (int t = 0; t < n; ++t) w_fit[static_cast<size_t>(t)] = w[static_cast<size_t>(t)] - fit.filter.innovations[static_cast<size_t>(t)];
    for (int t = 0; t < n; ++t) {
        // y_t = w_t + (y_{t-1} for d = 1; 2 y_{t-1} - y_{t-2} for d = 2), t counted on the original index.
        const size_t i = static_cast<size_t>(t + d);
        double base = 0.0;
        if (d == 1) base = data[i - 1];
        else if (d == 2) base = 2.0 * data[i - 1] - data[i - 2];
        r.fitted_values[i] = base + w_fit[static_cast<size_t>(t)];
        r.residuals[i] = data[i] - r.fitted_values[i];
    }
    double sse = 0.0, abs_sum = 0.0, ape_sum = 0.0;
    int count = 0, ape_count = 0;
    for (size_t i = static_cast<size_t>(d); i < total; ++i) {
        const double e = r.residuals[i];
        sse += e * e;
        abs_sum += std::fabs(e);
        ++count;
        if (std::fabs(data[i]) > 1e-12) {
            ape_sum += std::fabs(e / data[i]);
            ++ape_count;
        }
    }
    r.sse = sse;
    r.mse = sse / std::max(count, 1);
    r.rmse = std::sqrt(r.mse);
    r.mae = abs_sum / std::max(count, 1);
    r.mape = ape_count > 0 ? 100.0 * ape_sum / ape_count : NAN;

    // Forecasts from the filtered state; the levels by integrating; intervals from the psi weights.
    const StateSpace ss(fit.model);
    std::vector<double> a = fit.filter.state;
    std::vector<double> w_forecast(static_cast<size_t>(horizon));
    for (int h = 0; h < horizon; ++h) {
        a = ss.Apply(a);
        w_forecast[static_cast<size_t>(h)] = fit.model.mu + a[0];
    }
    std::vector<double> last(data.end() - std::min<size_t>(total, 2), data.end());
    r.forecast.resize(static_cast<size_t>(horizon));
    for (int h = 0; h < horizon; ++h) {
        double value = w_forecast[static_cast<size_t>(h)];
        if (d >= 1) value += last.back();
        if (d == 2) value += last.back() - last[last.size() - 2];
        r.forecast[static_cast<size_t>(h)] = value;
        if (d >= 1) {
            last.push_back(value);
            if (last.size() > 2) last.erase(last.begin());
        }
    }
    const std::vector<double> psi = PsiWeights(fit.model, d, horizon);
    const double z = series_detail::NormalPpf(1.0 - (1.0 - level) / 2.0);
    r.lower_bound.resize(static_cast<size_t>(horizon));
    r.upper_bound.resize(static_cast<size_t>(horizon));
    double var = 0.0;
    for (int h = 0; h < horizon; ++h) {
        var += psi[static_cast<size_t>(h)] * psi[static_cast<size_t>(h)];
        const double half = z * std::sqrt(fit.filter.sigma2 * var);
        r.lower_bound[static_cast<size_t>(h)] = r.forecast[static_cast<size_t>(h)] - half;
        r.upper_bound[static_cast<size_t>(h)] = r.forecast[static_cast<size_t>(h)] + half;
    }
    char summary[160];
    std::snprintf(summary, sizeof(summary), "%s: log-likelihood %.2f, AIC %.1f, sigma2 %.4g", r.method.c_str(), r.log_likelihood, r.aic, fit.filter.sigma2);
    r.model_summary = summary;
    r.success = true;
    return r;
}

}  // namespace cyxwiz
