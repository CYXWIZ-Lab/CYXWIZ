// Exponential smoothing (TOFIX129 D6): simple, Holt (damped or not) and
// Holt-Winters (additive or multiplicative season), with the smoothing
// parameters and the initial states estimated by least squares exactly as
// statsmodels ExponentialSmoothing(initialization_method="estimated") does
// (the same recursions, start values, bounds and grid search), prediction
// intervals from the additive-error ETS variances (Hyndman et al. 2008;
// simulated for a multiplicative season), and the seasonal naive baseline.
// The recursions are sequential by nature and run on the host over the series.

#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/time_series.h"
#include "series_device.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <random>

namespace cyxwiz {

namespace {

constexpr double kLowerBound = 1.4901161193847656e-08;  // sqrt(machine epsilon), as statsmodels

enum class Season { None, Additive, Multiplicative };

struct Spec {
    bool trend = false;
    bool damped = false;
    Season season = Season::None;
    int m = 0;  // the period (0 without a season)
};

// Parameters in the order statsmodels keeps them: alpha, beta, gamma, l0, b0, phi, s0[m].
struct Params {
    double alpha = 0.0, beta = 0.0, gamma = 0.0, l0 = 0.0, b0 = 0.0, phi = 1.0;
    std::vector<double> s0;
};

struct Run {
    std::vector<double> fitted;  // n + h values: in-sample one-step fits, then forecasts
    std::vector<double> level, trend, season;
    double sse = 0.0;
};

// The recursions of statsmodels' _predict: y[i - 1] updates the state i, the fit at i is state i's forecast.
Run Smooth(const std::vector<double>& y, const Spec& spec, const Params& p, int h) {
    const int n = static_cast<int>(y.size()), m = std::max(spec.m, 1);
    const double alpha = p.alpha, beta = spec.trend ? p.beta : 0.0, gamma = spec.season != Season::None ? p.gamma : 0.0;
    const double phi = spec.damped ? p.phi : 1.0;
    std::vector<double> l(static_cast<size_t>(n + h + 1), 0.0), b(static_cast<size_t>(n + h + 1), 0.0), s(static_cast<size_t>(n + h + m + 1), 0.0);
    l[0] = p.l0;
    b[0] = spec.trend ? p.b0 : 0.0;
    for (int i = 0; i < spec.m; ++i) s[static_cast<size_t>(i)] = p.s0[static_cast<size_t>(i)];
    const auto trended = [&](double level, double slope) { return spec.trend ? level + slope : level; };
    for (int i = 1; i <= n; ++i) {
        const double yi = y[static_cast<size_t>(i - 1)];
        const double previous = trended(l[static_cast<size_t>(i - 1)], phi * b[static_cast<size_t>(i - 1)]);
        if (spec.season == Season::Multiplicative) {
            l[static_cast<size_t>(i)] = alpha * yi / s[static_cast<size_t>(i - 1)] + (1.0 - alpha) * previous;
            if (spec.trend) b[static_cast<size_t>(i)] = beta * (l[static_cast<size_t>(i)] - l[static_cast<size_t>(i - 1)]) + (1.0 - beta) * phi * b[static_cast<size_t>(i - 1)];
            s[static_cast<size_t>(i + m - 1)] = gamma * yi / previous + (1.0 - gamma) * s[static_cast<size_t>(i - 1)];
        } else if (spec.season == Season::Additive) {
            l[static_cast<size_t>(i)] = alpha * yi - alpha * s[static_cast<size_t>(i - 1)] + (1.0 - alpha) * previous;
            if (spec.trend) b[static_cast<size_t>(i)] = beta * (l[static_cast<size_t>(i)] - l[static_cast<size_t>(i - 1)]) + (1.0 - beta) * phi * b[static_cast<size_t>(i - 1)];
            s[static_cast<size_t>(i + m - 1)] = gamma * yi - gamma * previous + (1.0 - gamma) * s[static_cast<size_t>(i - 1)];
        } else {
            l[static_cast<size_t>(i)] = alpha * yi + (1.0 - alpha) * previous;
            if (spec.trend) b[static_cast<size_t>(i)] = beta * (l[static_cast<size_t>(i)] - l[static_cast<size_t>(i - 1)]) + (1.0 - beta) * phi * b[static_cast<size_t>(i - 1)];
        }
    }
    Run run;
    run.level.assign(l.begin() + 1, l.begin() + 1 + n);
    run.trend.assign(b.begin() + 1, b.begin() + 1 + n);
    if (spec.season != Season::None) run.season.assign(s.begin() + m, s.begin() + m + n);
    // Beyond n the level stays, the slope accumulates (damped: phi + phi^2 + ...), the season repeats.
    for (int t = n + 1; t <= n + h; ++t) l[static_cast<size_t>(t)] = l[static_cast<size_t>(n)];
    std::vector<double> slope(static_cast<size_t>(n + h + 1), 0.0);
    if (spec.trend) {
        for (int t = 0; t < n; ++t) slope[static_cast<size_t>(t)] = phi * b[static_cast<size_t>(t)];
        double phi_sum = 0.0, phi_power = 1.0;
        for (int j = 0; j <= h; ++j) {
            phi_power *= phi;
            phi_sum += phi_power;
            slope[static_cast<size_t>(n + j)] = spec.damped ? b[static_cast<size_t>(n)] * phi_sum : b[static_cast<size_t>(n)] * static_cast<double>(j + 1);
        }
    }
    if (spec.season != Season::None)
        for (int j = 0; j <= h + 1; ++j) s[static_cast<size_t>(n + m - 1 + j)] = s[static_cast<size_t>(n - 1 + j % m)];
    run.fitted.resize(static_cast<size_t>(n + h));
    for (int t = 0; t < n + h; ++t) {
        const double base = trended(l[static_cast<size_t>(t)], slope[static_cast<size_t>(t)]);
        run.fitted[static_cast<size_t>(t)] = spec.season == Season::Multiplicative ? base * s[static_cast<size_t>(t)]
                                             : spec.season == Season::Additive   ? base + s[static_cast<size_t>(t)]
                                                                                 : base;
    }
    for (int t = 0; t < n; ++t) {
        const double e = run.fitted[static_cast<size_t>(t)] - y[static_cast<size_t>(t)];
        run.sse += e * e;
    }
    if (!std::isfinite(run.sse)) run.sse = INFINITY;
    return run;
}

// statsmodels' start values: the heuristic (a centred moving average over the
// first cycles for the season, a line through the first ten values for the
// level and slope), or the simple rule for a short series.
Params StartValues(const std::vector<double>& y, const Spec& spec) {
    Params p;
    const int n = static_cast<int>(y.size()), m = spec.m;
    std::vector<double> base = y;
    if (spec.season != Season::None) {
        if (n < 10 + 2 * (m / 2)) {
            // Simple: the first cycle's mean, the slope from the second cycle, the first cycle's deviations.
            p.l0 = std::accumulate(y.begin(), y.begin() + m, 0.0) / m;
            if (spec.trend) {
                double sum = 0.0;
                for (int i = m; i < 2 * m; ++i) sum += (y[static_cast<size_t>(i)] - y[static_cast<size_t>(i - m)]) / m;
                p.b0 = sum / m;
            }
            p.s0.resize(static_cast<size_t>(m));
            for (int i = 0; i < m; ++i) p.s0[static_cast<size_t>(i)] = spec.season == Season::Additive ? y[static_cast<size_t>(i)] - p.l0 : y[static_cast<size_t>(i)] / p.l0;
            return p;
        }
        int cycles = std::min(5, n / m);
        cycles = std::max(cycles, static_cast<int>(std::ceil((10.0 + 2 * (m / 2)) / m)));
        const int len = cycles * m;
        std::vector<double> part(y.begin(), y.begin() + len);
        // pandas rolling(m, center=True).mean(), then for an even m the mean of neighbours (shift(-1).rolling(2)).
        std::vector<double> trend(static_cast<size_t>(len), NAN);
        const int before = m / 2, after = m - 1 - before;  // pandas centres an even window with one more before
        for (int i = 0; i < len; ++i) {
            if (i - before < 0 || i + after >= len) continue;
            double sum = 0.0;
            for (int j = i - before; j <= i + after; ++j) sum += part[static_cast<size_t>(j)];
            trend[static_cast<size_t>(i)] = sum / m;
        }
        if (m % 2 == 0) {
            std::vector<double> smoothed(static_cast<size_t>(len), NAN);
            for (int i = 0; i < len; ++i) {
                const double a = i + 1 < len ? trend[static_cast<size_t>(i + 1)] : NAN;  // shift(-1)
                const double prev = i < len ? trend[static_cast<size_t>(i)] : NAN;
                if (std::isfinite(a) && std::isfinite(prev)) smoothed[static_cast<size_t>(i)] = 0.5 * (a + prev);
            }
            trend = smoothed;
        }
        std::vector<double> detrended(static_cast<size_t>(len), NAN);
        for (int i = 0; i < len; ++i)
            if (std::isfinite(trend[static_cast<size_t>(i)]))
                detrended[static_cast<size_t>(i)] = spec.season == Season::Additive ? part[static_cast<size_t>(i)] - trend[static_cast<size_t>(i)]
                                                                                    : part[static_cast<size_t>(i)] / trend[static_cast<size_t>(i)];
        p.s0.assign(static_cast<size_t>(m), 0.0);
        for (int j = 0; j < m; ++j) {
            double sum = 0.0;
            int count = 0;
            for (int i = j; i < len; i += m)
                if (std::isfinite(detrended[static_cast<size_t>(i)])) {
                    sum += detrended[static_cast<size_t>(i)];
                    ++count;
                }
            p.s0[static_cast<size_t>(j)] = count > 0 ? sum / count : (spec.season == Season::Additive ? 0.0 : 1.0);
        }
        const double mean = std::accumulate(p.s0.begin(), p.s0.end(), 0.0) / m;
        for (double& v : p.s0) v = spec.season == Season::Additive ? v - mean : v / mean;
        base.clear();
        for (const double v : trend)
            if (std::isfinite(v)) base.push_back(v);
    } else if (n < 10) {
        p.l0 = y[0];
        if (spec.trend) p.b0 = y[1] - y[0];
        return p;
    }
    // A line through the first ten values: level = intercept, slope.
    const int k = std::min(10, static_cast<int>(base.size()));
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    for (int i = 0; i < k; ++i) {
        const double x = i + 1.0;
        sx += x;
        sy += base[static_cast<size_t>(i)];
        sxx += x * x;
        sxy += x * base[static_cast<size_t>(i)];
    }
    const double slope = (k * sxy - sx * sy) / (k * sxx - sx * sx);
    const double intercept = (sy - slope * sx) / k;
    p.l0 = intercept;
    if (spec.trend) p.b0 = slope;
    return p;
}

// The free parameters, in statsmodels' order, with the smoothing ones on the
// unrestricted scale (alpha in (0, 1); beta <= alpha; gamma <= 1 - alpha).
struct Fitter {
    const std::vector<double>& y;
    Spec spec;
    Params fixed;      // the values of the parameters that are not estimated
    std::vector<bool> sel;  // alpha, beta, gamma, l0, b0, phi, s0...
    bool level_bounded = false;  // a multiplicative season or trend keeps the level above 0

    Params Unpack(const std::vector<double>& x) const {
        Params p = fixed;
        size_t at = 0;
        const auto take = [&](double& target) {
            if (sel[at]) target = x[at];
        };
        double a = fixed.alpha, b = fixed.beta, g = fixed.gamma;
        if (sel[0]) a = std::clamp(x[0], 0.0, 1.0);
        if (sel[1]) b = std::clamp(x[1], 0.0, 1.0);
        if (sel[2]) g = std::clamp(x[2], 0.0, 1.0);
        if (sel[0]) a = kLowerBound + a * (1.0 - 2.0 * kLowerBound);
        if (sel[1]) b = b * std::min(a, 1.0);
        if (sel[2]) g = g * std::min(1.0 - a, 1.0);
        p.alpha = a;
        p.beta = b;
        p.gamma = g;
        at = 3;
        take(p.l0);
        at = 4;
        take(p.b0);
        at = 5;
        if (sel[5]) p.phi = std::clamp(x[5], 0.8, 0.995);
        if (level_bounded) p.l0 = std::max(p.l0, 0.0);
        for (size_t i = 0; i < p.s0.size(); ++i)
            if (sel[6 + i]) p.s0[i] = spec.season == Season::Multiplicative ? std::max(x[6 + i], 0.0) : x[6 + i];
        return p;
    }

    double Sse(const std::vector<double>& x) const { return Smooth(y, spec, Unpack(x), 0).sse; }
};

// Nelder-Mead over the free coordinates of x (the others stay), restarted once.
void NelderMead(const std::function<double(const std::vector<double>&)>& f, std::vector<double>& x, const std::vector<bool>& free,
                const std::vector<double>& scale, int max_evals) {
    std::vector<size_t> idx;
    for (size_t i = 0; i < free.size(); ++i)
        if (free[i]) idx.push_back(i);
    const size_t d = idx.size();
    if (d == 0) return;
    const auto full = [&](const std::vector<double>& z) {
        std::vector<double> out = x;
        for (size_t i = 0; i < d; ++i) out[idx[i]] = z[i];
        return out;
    };
    std::vector<std::vector<double>> simplex(d + 1, std::vector<double>(d));
    std::vector<double> values(d + 1);
    for (size_t i = 0; i < d; ++i) simplex[0][i] = x[idx[i]];
    for (size_t k = 1; k <= d; ++k) {
        simplex[k] = simplex[0];
        simplex[k][k - 1] += scale[idx[k - 1]];
    }
    for (size_t k = 0; k <= d; ++k) values[k] = f(full(simplex[k]));
    int evals = static_cast<int>(d + 1);
    std::vector<size_t> order(d + 1);
    while (evals < max_evals) {
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return values[a] < values[b]; });
        const size_t best = order[0], worst = order[d], second = order[d - 1];
        if (std::fabs(values[worst] - values[best]) <= 1e-12 * (1.0 + std::fabs(values[best]))) {
            double spread = 0.0;
            for (size_t k = 0; k <= d; ++k)
                for (size_t i = 0; i < d; ++i) spread = std::max(spread, std::fabs(simplex[k][i] - simplex[best][i]) / (scale[idx[i]] + 1e-300));
            if (spread < 1e-7) break;
        }
        std::vector<double> centroid(d, 0.0);
        for (size_t k = 0; k <= d; ++k)
            if (k != worst)
                for (size_t i = 0; i < d; ++i) centroid[i] += simplex[k][i] / d;
        const auto along = [&](double t) {
            std::vector<double> z(d);
            for (size_t i = 0; i < d; ++i) z[i] = centroid[i] + t * (simplex[worst][i] - centroid[i]);
            return z;
        };
        std::vector<double> reflected = along(-1.0);
        double fr = f(full(reflected));
        ++evals;
        if (fr < values[best]) {
            std::vector<double> expanded = along(-2.0);
            const double fe = f(full(expanded));
            ++evals;
            if (fe < fr) {
                simplex[worst] = expanded;
                values[worst] = fe;
            } else {
                simplex[worst] = reflected;
                values[worst] = fr;
            }
        } else if (fr < values[second]) {
            simplex[worst] = reflected;
            values[worst] = fr;
        } else {
            std::vector<double> contracted = along(fr < values[worst] ? -0.5 : 0.5);
            const double fc = f(full(contracted));
            ++evals;
            if (fc < std::min(fr, values[worst])) {
                simplex[worst] = contracted;
                values[worst] = fc;
            } else {
                for (size_t k = 0; k <= d; ++k) {
                    if (k == best) continue;
                    for (size_t i = 0; i < d; ++i) simplex[k][i] = simplex[best][i] + 0.5 * (simplex[k][i] - simplex[best][i]);
                    values[k] = f(full(simplex[k]));
                    ++evals;
                }
            }
        }
    }
    const size_t best = static_cast<size_t>(std::min_element(values.begin(), values.end()) - values.begin());
    x = full(simplex[best]);
}

std::vector<double> Linspace(double lo, double hi, int count) {
    std::vector<double> out;
    if (count <= 1) return {lo};
    for (int i = 0; i < count; ++i) out.push_back(lo + (hi - lo) * i / (count - 1));
    return out;
}

// Fit by least squares as statsmodels: start values, the grid over the
// smoothing parameters, then the local search over every free parameter.
Params Fit(const std::vector<double>& y, const Spec& spec, double alpha, double beta, double gamma, double phi) {
    Fitter fitter{y, spec, StartValues(y, spec), {}, spec.season == Season::Multiplicative};
    const int m = spec.m;
    fitter.sel.assign(static_cast<size_t>(6 + m), false);
    fitter.sel[0] = alpha < 0.0;
    fitter.sel[1] = spec.trend && beta < 0.0;
    fitter.sel[2] = spec.season != Season::None && gamma < 0.0;
    // The initial states are estimated with the weights, as statsmodels does,
    // up to a period of 50; a longer season (a day of seconds) keeps its
    // heuristic start values, or the search would run over thousands of states.
    const bool estimate_states = m <= 50;
    fitter.sel[3] = estimate_states;
    fitter.sel[4] = spec.trend && estimate_states;
    fitter.sel[5] = spec.damped && phi < 0.0;
    for (int i = 0; i < m; ++i) fitter.sel[static_cast<size_t>(6 + i)] = estimate_states;
    const double init_alpha = alpha >= 0.0 ? alpha : 0.5 / std::max(m, 1);
    fitter.fixed.alpha = init_alpha;
    fitter.fixed.beta = beta >= 0.0 ? beta : (spec.trend ? 0.1 * init_alpha : 0.0);
    fitter.fixed.gamma = gamma >= 0.0 ? gamma : (spec.season != Season::None ? 0.05 * (1.0 - init_alpha) : 0.0);
    fitter.fixed.phi = phi >= 0.0 ? phi : 0.99;

    // The smoothing parameters on their natural scale, chosen on statsmodels' grid with the states fixed.
    double a = fitter.fixed.alpha, b = fitter.fixed.beta, g = fitter.fixed.gamma;
    const int free_smoothing = static_cast<int>(fitter.sel[0]) + static_cast<int>(fitter.sel[1]) + static_cast<int>(fitter.sel[2]);
    if (free_smoothing > 0) {
        // statsmodels' grid (87 points shared by the free parameters); a long
        // series gets a coarser one so the grid spends at most ~2e8 point
        // evaluations (the local search after it finds the optimum either way).
        int ns = 87 / free_smoothing;
        const double budget = 2e8 / static_cast<double>(std::max<size_t>(1, y.size()));
        ns = std::min(ns, std::max(3, static_cast<int>(std::pow(std::max(27.0, budget), 1.0 / free_smoothing))));
        const auto sse_at = [&](double aa, double bb, double gg) {
            Params p = fitter.fixed;
            p.alpha = aa;
            p.beta = bb;
            p.gamma = gg;
            return Smooth(y, spec, p, 0).sse;
        };
        double best = INFINITY;
        const std::vector<double> alphas = fitter.sel[0] ? Linspace(0.005, 0.995, ns) : std::vector<double>{a};
        for (const double aa : alphas) {
            const std::vector<double> betas = fitter.sel[1] ? Linspace(0.0, std::min(1.0, aa), static_cast<int>(std::ceil(ns * std::sqrt(aa)))) : std::vector<double>{b};
            const std::vector<double> gammas = fitter.sel[2] ? Linspace(0.0, std::min(1.0, 1.0 - aa), static_cast<int>(std::ceil(ns * std::sqrt(1.0 - aa)))) : std::vector<double>{g};
            for (const double bb : betas)
                for (const double gg : gammas) {
                    const double v = sse_at(aa, bb, gg);
                    if (v < best) {
                        best = v;
                        a = aa;
                        b = bb;
                        g = gg;
                    }
                }
        }
    }
    // To the unrestricted scale, strictly inside the bounds.
    std::vector<double> x(static_cast<size_t>(6 + m), 0.0), scale(x.size(), 0.1);
    const double eps = 1e-4;
    a = std::clamp(a, eps, 1.0 - eps);
    x[0] = (a - kLowerBound) / (1.0 - 2.0 * kLowerBound);
    x[1] = std::clamp(b / std::max(a, 1e-12), eps, 1.0 - eps);
    x[2] = std::clamp(g / std::max(1.0 - a, 1e-12), eps, 1.0 - eps);
    x[3] = fitter.fixed.l0;
    x[4] = fitter.fixed.b0;
    x[5] = std::clamp(fitter.fixed.phi, 0.8 + eps, 0.995 - eps);
    for (int i = 0; i < m; ++i) x[static_cast<size_t>(6 + i)] = fitter.fixed.s0[static_cast<size_t>(i)];
    const double level_scale = std::max(1e-3, 0.05 * std::fabs(fitter.fixed.l0));
    scale[3] = level_scale;
    scale[4] = std::max(1e-3, 0.1 * std::fabs(fitter.fixed.b0) + 0.01 * level_scale);
    scale[5] = 0.02;
    for (int i = 0; i < m; ++i) scale[static_cast<size_t>(6 + i)] = spec.season == Season::Multiplicative ? 0.02 : 0.1 * level_scale;
    const auto objective = [&](const std::vector<double>& z) { return fitter.Sse(z); };
    // The local search: statsmodels' cap, or on a long series what ~4e8 point evaluations allow (at least 200).
    const int max_evals = std::min(4000 * static_cast<int>(x.size()), std::max(200, static_cast<int>(4e8 / static_cast<double>(std::max<size_t>(1, y.size())))));
    NelderMead(objective, x, fitter.sel, scale, max_evals);
    for (double& v : scale) v *= 0.1;
    NelderMead(objective, x, fitter.sel, scale, max_evals);
    return fitter.Unpack(x);
}

const char* SeasonName(Season s) { return s == Season::Additive ? "additive" : s == Season::Multiplicative ? "multiplicative" : "none"; }

ForecastResult Forecast(const std::vector<double>& y, const Spec& spec, int horizon, double alpha, double beta, double gamma, double phi,
                        double level, const char* name) {
    ForecastResult r;
    r.method = name;
    r.horizon = horizon;
    r.level = level;
    const size_t need = spec.season != Season::None ? std::max<size_t>(2 * static_cast<size_t>(spec.m), 10) : 3;
    if (const std::string refusal = series_detail::SeriesRefusal(y, need, name); !refusal.empty()) {
        r.error_message = refusal;
        return r;
    }
    if (horizon < 0) {
        r.error_message = "The horizon cannot be negative";
        return r;
    }
    if (!(level > 0.0 && level < 1.0)) {
        r.error_message = "The interval level must be between 0 and 1";
        return r;
    }
    if (spec.season == Season::Multiplicative)
        for (const double v : y)
            if (v <= 0.0) {
                r.error_message = "A multiplicative season needs values above 0";
                return r;
            }
    for (const double v : {alpha, beta, gamma})
        if (v > 1.0) {
            r.error_message = "Smoothing parameters are between 0 and 1 (negative: estimated)";
            return r;
        }
    const Params p = Fit(y, spec, alpha, beta, gamma, phi);
    const Run run = Smooth(y, spec, p, horizon);
    const int n = static_cast<int>(y.size()), m = spec.m;
    r.fitted_values.assign(run.fitted.begin(), run.fitted.begin() + n);
    r.forecast.assign(run.fitted.begin() + n, run.fitted.end());
    r.residuals.resize(static_cast<size_t>(n));
    double abs_sum = 0.0, ape_sum = 0.0;
    int ape_count = 0;
    for (int t = 0; t < n; ++t) {
        const double e = y[static_cast<size_t>(t)] - r.fitted_values[static_cast<size_t>(t)];
        r.residuals[static_cast<size_t>(t)] = e;
        abs_sum += std::fabs(e);
        if (std::fabs(y[static_cast<size_t>(t)]) > 1e-12) {
            ape_sum += std::fabs(e / y[static_cast<size_t>(t)]);
            ++ape_count;
        }
    }
    r.sse = run.sse;
    r.mse = run.sse / n;
    r.rmse = std::sqrt(r.mse);
    r.mae = abs_sum / n;
    r.mape = ape_count > 0 ? 100.0 * ape_sum / ape_count : NAN;
    const int k = m * (spec.season != Season::None) + 2 * spec.trend + 2 + (spec.damped ? 1 : 0);
    r.n_params = k;
    r.aic = n * std::log(run.sse / n) + 2.0 * k;
    r.bic = n * std::log(run.sse / n) + k * std::log(static_cast<double>(n));
    r.parameters["alpha"] = p.alpha;
    if (spec.trend) r.parameters["beta"] = p.beta;
    if (spec.season != Season::None) r.parameters["gamma"] = p.gamma;
    if (spec.damped) r.parameters["phi"] = p.phi;
    r.parameters["initial_level"] = p.l0;
    if (spec.trend) r.parameters["initial_trend"] = p.b0;
    for (int i = 0; i < m; ++i) r.parameters["initial_seasons." + std::to_string(i)] = p.s0[static_cast<size_t>(i)];
    r.parameters["sigma2"] = r.mse;

    // The prediction interval. Additive error, additive or no season: the ETS
    // variance sigma^2 (1 + sum_{j<h} c_j^2) with c_j = alpha + beta phi_j + gamma 1[j mod m = 0];
    // a multiplicative season: simulated paths of the same recursion.
    const double z = series_detail::NormalPpf(1.0 - (1.0 - level) / 2.0);
    const double sigma2 = r.mse;
    r.lower_bound.resize(static_cast<size_t>(horizon));
    r.upper_bound.resize(static_cast<size_t>(horizon));
    if (spec.season != Season::Multiplicative) {
        double sum = 0.0, phi_sum = 0.0, phi_power = 1.0;
        for (int h = 1; h <= horizon; ++h) {
            const double var = sigma2 * (1.0 + sum);
            const double half = z * std::sqrt(var);
            r.lower_bound[static_cast<size_t>(h - 1)] = r.forecast[static_cast<size_t>(h - 1)] - half;
            r.upper_bound[static_cast<size_t>(h - 1)] = r.forecast[static_cast<size_t>(h - 1)] + half;
            // c_h for the next step
            phi_power *= spec.damped ? p.phi : 1.0;
            phi_sum += phi_power;
            double c = p.alpha;
            if (spec.trend) c += p.beta * phi_sum;
            if (spec.season == Season::Additive && m > 0 && h % m == 0) c += p.gamma;
            sum += c * c;
        }
    } else if (horizon > 0) {
        const int paths = 5000;
        std::mt19937_64 rng(20261030);
        std::normal_distribution<double> noise(0.0, std::sqrt(sigma2));
        std::vector<std::vector<double>> samples(static_cast<size_t>(horizon), std::vector<double>(paths));
        const double phi_v = spec.damped ? p.phi : 1.0;
        for (int path = 0; path < paths; ++path) {
            double l = run.level.back(), b = spec.trend ? run.trend.back() : 0.0;
            std::vector<double> s(run.season.end() - m, run.season.end());
            for (int h = 0; h < horizon; ++h) {
                const double base = l + phi_v * b;
                const double seasonal = s[static_cast<size_t>(h % m)];
                const double value = base * seasonal + noise(rng);
                samples[static_cast<size_t>(h)][static_cast<size_t>(path)] = value;
                const double e = value - base * seasonal;
                const double new_l = base + p.alpha * e / seasonal;
                if (spec.trend) b = phi_v * b + p.beta * e / seasonal;
                s[static_cast<size_t>(h % m)] = seasonal + p.gamma * e / base;
                l = new_l;
            }
        }
        for (int h = 0; h < horizon; ++h) {
            auto& v = samples[static_cast<size_t>(h)];
            std::sort(v.begin(), v.end());
            const double lo = (1.0 - level) / 2.0, hi = 1.0 - lo;
            r.lower_bound[static_cast<size_t>(h)] = v[static_cast<size_t>(lo * (paths - 1))];
            r.upper_bound[static_cast<size_t>(h)] = v[static_cast<size_t>(hi * (paths - 1))];
        }
    }
    char summary[256];
    std::snprintf(summary, sizeof(summary), "%s: alpha %.3f%s%s%s, season %s, SSE %.4g, AIC %.1f", name, p.alpha,
                  spec.trend ? (", beta " + std::to_string(p.beta).substr(0, 5)).c_str() : "",
                  spec.season != Season::None ? (", gamma " + std::to_string(p.gamma).substr(0, 5)).c_str() : "",
                  spec.damped ? (", phi " + std::to_string(p.phi).substr(0, 5)).c_str() : "", SeasonName(spec.season), run.sse, r.aic);
    r.model_summary = summary;
    r.success = true;
    return r;
}

}  // namespace

ForecastResult TimeSeries::SimpleES(const std::vector<double>& data, int horizon, double alpha, double level) {
    Spec spec;
    return Forecast(data, spec, horizon, alpha, 0.0, 0.0, 1.0, level, "Simple exponential smoothing");
}

ForecastResult TimeSeries::HoltLinear(const std::vector<double>& data, int horizon, double alpha, double beta, bool damped, double level) {
    Spec spec;
    spec.trend = true;
    spec.damped = damped;
    return Forecast(data, spec, horizon, alpha, beta, 0.0, damped ? -1.0 : 1.0, level, damped ? "Damped Holt" : "Holt");
}

ForecastResult TimeSeries::HoltWinters(const std::vector<double>& data, int horizon, int period, const std::string& seasonal_type, double alpha,
                                       double beta, double gamma, double level) {
    ForecastResult refusal;
    if (seasonal_type != "additive" && seasonal_type != "multiplicative") {
        refusal.error_message = "seasonal_type must be \"additive\" or \"multiplicative\"";
        return refusal;
    }
    if (period < 0) {
        const SeasonalityResult season = DetectSeasonality(data);
        period = season.success && season.detected_period >= 2 ? season.detected_period : 12;
    }
    if (period < 2) {
        refusal.error_message = "The period must be at least 2";
        return refusal;
    }
    Spec spec;
    spec.trend = true;
    spec.season = seasonal_type == "additive" ? Season::Additive : Season::Multiplicative;
    spec.m = period;
    ForecastResult r = Forecast(data, spec, horizon, alpha, beta, gamma, 1.0, level, "Holt-Winters");
    r.parameters["period"] = period;
    return r;
}

ForecastResult TimeSeries::MovingAverageForecast(const std::vector<double>& data, int window, int horizon) {
    ForecastResult r;
    r.method = "Moving average";
    r.horizon = horizon;
    if (window < 1 || static_cast<size_t>(window) > data.size() || horizon < 0) {
        r.error_message = "The window must be between 1 and the number of values";
        return r;
    }
    if (const std::string refusal = series_detail::SeriesRefusal(data, static_cast<size_t>(window), "The moving average"); !refusal.empty()) {
        r.error_message = refusal;
        return r;
    }
    std::vector<double> mean, sd;
    series_detail::Rolling(data, window, mean, sd);
    const size_t n = data.size();
    r.fitted_values.assign(n, NAN);
    r.residuals.assign(n, NAN);
    double sse = 0.0, abs_sum = 0.0;
    int count = 0;
    for (size_t t = static_cast<size_t>(window); t < n; ++t) {
        r.fitted_values[t] = mean[t - 1];
        r.residuals[t] = data[t] - mean[t - 1];
        sse += r.residuals[t] * r.residuals[t];
        abs_sum += std::fabs(r.residuals[t]);
        ++count;
    }
    r.sse = sse;
    r.mse = count > 0 ? sse / count : NAN;
    r.rmse = std::sqrt(r.mse);
    r.mae = count > 0 ? abs_sum / count : NAN;
    const double last = mean.back(), sigma = std::sqrt(r.mse);
    const double z = series_detail::NormalPpf(0.975);
    r.forecast.assign(static_cast<size_t>(horizon), last);
    r.lower_bound.assign(static_cast<size_t>(horizon), last - z * sigma);
    r.upper_bound.assign(static_cast<size_t>(horizon), last + z * sigma);
    r.parameters["window"] = window;
    r.success = true;
    return r;
}

ForecastResult TimeSeries::SeasonalNaive(const std::vector<double>& data, int horizon, int period, double level) {
    ForecastResult r;
    r.method = "Seasonal naive";
    r.horizon = horizon;
    r.level = level;
    if (period < 1) {
        r.error_message = "The period must be at least 1";
        return r;
    }
    if (const std::string refusal = series_detail::SeriesRefusal(data, 2 * static_cast<size_t>(period), "The seasonal naive forecast"); !refusal.empty()) {
        r.error_message = refusal;
        return r;
    }
    if (horizon < 0 || !(level > 0.0 && level < 1.0)) {
        r.error_message = "The horizon cannot be negative and the level is between 0 and 1";
        return r;
    }
    const size_t n = data.size(), m = static_cast<size_t>(period);
    r.fitted_values.assign(n, NAN);
    r.residuals.assign(n, NAN);
    double sse = 0.0, abs_sum = 0.0;
    for (size_t t = m; t < n; ++t) {
        r.fitted_values[t] = data[t - m];
        r.residuals[t] = data[t] - data[t - m];
        sse += r.residuals[t] * r.residuals[t];
        abs_sum += std::fabs(r.residuals[t]);
    }
    const double count = static_cast<double>(n - m);
    r.sse = sse;
    r.mse = sse / count;
    r.rmse = std::sqrt(r.mse);
    r.mae = abs_sum / count;
    // The interval grows with the number of whole cycles ahead: sigma sqrt(k + 1), k = floor((h - 1) / m).
    const double z = series_detail::NormalPpf(1.0 - (1.0 - level) / 2.0);
    for (int h = 1; h <= horizon; ++h) {
        const double value = data[n - m + static_cast<size_t>((h - 1) % period)];
        const double k = std::floor((h - 1.0) / period);
        const double half = z * r.rmse * std::sqrt(k + 1.0);
        r.forecast.push_back(value);
        r.lower_bound.push_back(value - half);
        r.upper_bound.push_back(value + half);
    }
    r.parameters["period"] = period;
    r.parameters["sigma2"] = r.mse;
    r.model_summary = "Seasonal naive: each value repeats the one " + std::to_string(period) + " steps before it";
    r.success = true;
    return r;
}

}  // namespace cyxwiz
