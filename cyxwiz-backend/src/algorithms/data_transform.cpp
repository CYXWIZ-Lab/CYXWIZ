// Column transforms, statistics and the normality test on ArrayFire (TOFIX140): the
// CPU option is ArrayFire's CPU backend, there are no hand-written CPU kernels, and a
// build without ArrayFire refuses. Each column crosses to the device once and its
// result comes back once; float64 where the active device has it, else float32.
// Semantics follow scikit-learn / scipy (StandardScaler, RobustScaler,
// QuantileTransformer, scipy.stats.boxcox / yeojohnson / normaltest); PyTorch
// reference fixtures check them (generate_data_transform_fixtures.py).

#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/data_transform.h"
#include "arrayfire_host_values.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace cyxwiz {

namespace {

using Columns = std::vector<std::vector<double>>;

constexpr double kDegenerate = 1e-10;  // zero range / variance / IQR / max-abs

template <typename Result>
Result Fail(Result result, std::string message) {
    result.success = false;
    result.error_message = std::move(message);
    if constexpr (std::is_same_v<Result, TransformResult>) result.transformed_data.clear();
    return result;
}

#ifdef CYXWIZ_HAS_ARRAYFIRE

std::string ColumnMessage(size_t column, const std::string& message) {
    return "Column " + std::to_string(column) + ": " + message;
}

enum class PowerFamily { BoxCox, YeoJohnson };

constexpr dim_t kMaxTileElements = dim_t{1} << 22;  // lambda-profile batch size
constexpr double kZeroLambda = 1e-12;

// A device row of host values in the precision of `like`.
af::array RowLike(const std::vector<double>& values, const af::array& like) {
    af::array row = HostValuesToDevice(values);
    return af::moddims(row, 1, static_cast<dim_t>(values.size())).as(like.type());
}

// numpy / torch "linear" percentile of an ascending column at each fraction in
// [0, 1]: lerp(s[floor], s[ceil], t), using b - (b - a)(1 - t) for t >= 0.5.
af::array Percentiles(const af::array& sorted, const af::array& fractions) {
    const af::array position = fractions * static_cast<double>(sorted.elements() - 1);
    const af::array lower = af::floor(position);
    const af::array t = position - lower;
    const af::array a = af::lookup(sorted, lower.as(u32));
    const af::array b = af::lookup(sorted, af::ceil(position).as(u32));
    const af::array diff = b - a;
    return af::select(t >= 0.5, b - diff * (1.0 - t), a + diff * t);
}

// Percentiles at host fractions, back on the host.
std::vector<double> PercentilesAt(const af::array& sorted, const std::vector<double>& fractions,
                                  const char* operation) {
    return DeviceToHostValues(Percentiles(sorted, HostValuesToDevice(fractions).as(sorted.type())), operation);
}

// Linear quantiles of an ascending column at i / (size - 1), i = 0 .. size - 1. The
// sorted position i (n - 1) / (size - 1) is split in integers, so it never rounds off a
// data point (a rounded position changes which repeated values match a quantile exactly).
af::array ExactQuantiles(const af::array& sorted, int size) {
    const dim_t n = sorted.elements();
    const af::dim4 dims(size);
    const af::array denominator = af::constant(static_cast<long long>(size - 1), dims, s64);
    const af::array numerator = af::range(dims, 0, s64) * af::constant(static_cast<long long>(n - 1), dims, s64);
    const af::array remainder = af::mod(numerator, denominator);
    const af::array lower = (numerator - remainder) / denominator;
    const af::array upper = af::min(lower + af::constant(1LL, dims, s64), af::constant(static_cast<long long>(n - 1), dims, s64));
    const af::array t = remainder.as(sorted.type()) / static_cast<double>(size - 1);
    const af::array a = af::lookup(sorted, lower.as(u32));
    const af::array b = af::lookup(sorted, upper.as(u32));
    const af::array diff = b - a;
    return af::select(t >= 0.5, b - diff * (1.0 - t), a + diff * t);
}

double PopulationStd(const af::array& x) {
    const af::array centered = x - af::mean<double>(x);
    return std::sqrt(af::mean<double>(centered * centered));
}

// Box-Cox / Yeo-Johnson of the column x at each lambda in the row `lambdas`
// (k entries): an [n, k] matrix, one transformed column per lambda.
af::array PowerTransformed(const af::array& x, const af::array& lambdas, PowerFamily family) {
    const dim_t n = x.elements(), k = lambdas.elements();
    const af::array lam = af::tile(lambdas, static_cast<unsigned>(n), 1);
    if (family == PowerFamily::BoxCox) {
        const af::array logx = af::tile(af::log(x), 1, static_cast<unsigned>(k));
        return af::select(af::abs(lam) < kZeroLambda, logx, af::expm1(lam * logx) / lam);
    }
    const af::array values = af::tile(x, 1, static_cast<unsigned>(k));
    const af::array lp = af::log1p(af::abs(values));
    const af::array positive = af::select(af::abs(lam) < kZeroLambda, lp, af::expm1(lam * lp) / lam);
    const af::array other = 2.0 - lam;
    const af::array negative = -af::select(af::abs(other) < kZeroLambda, lp, af::expm1(other * lp) / other);
    return af::select(values >= 0.0, positive, negative);
}

// Profile log-likelihood at each lambda (scipy boxcox_llf / yeojohnson_llf):
// (lambda - 1) * J - n/2 * log(var(y)), J = sum log x (Box-Cox) or
// sum sign(x) log1p(|x|) (Yeo-Johnson), var the population variance.
std::vector<double> LogLikelihoods(const af::array& x, PowerFamily family, const std::vector<double>& lambdas,
                                   const char* operation) {
    const dim_t n = x.elements();
    const double jacobian = family == PowerFamily::BoxCox
        ? af::sum<double>(af::log(x))
        : af::sum<double>(af::select(x >= 0.0, af::log1p(af::abs(x)), -af::log1p(af::abs(x))));
    const size_t chunk = static_cast<size_t>(std::max<dim_t>(1, kMaxTileElements / n));
    std::vector<double> result;
    result.reserve(lambdas.size());
    for (size_t start = 0; start < lambdas.size(); start += chunk) {
        const std::vector<double> part(lambdas.begin() + static_cast<std::ptrdiff_t>(start),
                                       lambdas.begin() + static_cast<std::ptrdiff_t>(std::min(lambdas.size(), start + chunk)));
        const af::array lam = RowLike(part, x);
        const af::array y = PowerTransformed(x, lam, family);
        const af::array centered = y - af::tile(af::mean(y, 0), static_cast<unsigned>(n), 1);
        const af::array variance = af::mean(centered * centered, 0);
        const af::array llf = (lam - 1.0) * jacobian - 0.5 * static_cast<double>(n) * af::log(variance);
        const std::vector<double> values = DeviceToHostValues(llf, operation);
        result.insert(result.end(), values.begin(), values.end());
    }
    return result;
}

std::vector<double> Linspace(double start, double stop, int count) {
    std::vector<double> values(static_cast<size_t>(count));
    const double step = (stop - start) / (count - 1);
    for (int i = 0; i < count; ++i) values[static_cast<size_t>(i)] = start + i * step;
    values.back() = stop;
    return values;
}

// Index of the largest finite value, or -1.
int BestFinite(const std::vector<double>& values) {
    int best = -1;
    for (size_t i = 0; i < values.size(); ++i) {
        if (std::isfinite(values[i]) && (best < 0 || values[i] > values[static_cast<size_t>(best)])) {
            best = static_cast<int>(i);
        }
    }
    return best;
}

// Maximum-likelihood lambda: the coarse profile across [lambda_min, lambda_max],
// then eight rounds of 33-point refinement around the best point (each round
// narrows the bracket 16x, to ~1e-10 of the grid spacing).
BoxCoxLambdaResult SearchLambda(const af::array& x, PowerFamily family, double lambda_min, double lambda_max,
                                int n_steps, const char* operation) {
    BoxCoxLambdaResult result;
    result.lambdas_tested = Linspace(lambda_min, lambda_max, n_steps + 1);
    result.log_likelihoods = LogLikelihoods(x, family, result.lambdas_tested, operation);
    const int coarse = BestFinite(result.log_likelihoods);
    if (coarse < 0) return Fail(result, "No lambda in the range gives a finite likelihood");

    double best = result.lambdas_tested[static_cast<size_t>(coarse)];
    double best_ll = result.log_likelihoods[static_cast<size_t>(coarse)];
    double low = result.lambdas_tested[static_cast<size_t>(std::max(coarse - 1, 0))];
    double high = result.lambdas_tested[static_cast<size_t>(std::min(coarse + 1, n_steps))];
    constexpr int kPoints = 33;
    for (int round = 0; round < 8 && high > low; ++round) {
        const std::vector<double> lambdas = Linspace(low, high, kPoints);
        const std::vector<double> lls = LogLikelihoods(x, family, lambdas, operation);
        const int i = BestFinite(lls);
        if (i >= 0 && lls[static_cast<size_t>(i)] >= best_ll) {
            best = lambdas[static_cast<size_t>(i)];
            best_ll = lls[static_cast<size_t>(i)];
        }
        const double half_width = (high - low) / (kPoints - 1);
        low = std::max(lambda_min, best - half_width);
        high = std::min(lambda_max, best + half_width);
    }
    result.optimal_lambda = best;
    result.best_log_likelihood = best_ll;
    result.success = true;
    return result;
}

// Rejects what the lambda search cannot handle; empty when the column is usable.
std::string LambdaSearchRefusal(const af::array& x, PowerFamily family, double lambda_min, double lambda_max,
                                int n_steps) {
    if (n_steps < 2 || !(lambda_min < lambda_max)) return "Lambda range needs lambda_min < lambda_max and at least 2 steps";
    if (x.elements() < 2) return "Needs at least 2 values";
    if (family == PowerFamily::BoxCox && !af::allTrue<bool>(x > 0.0)) return "Box-Cox requires strictly positive data";
    if (PopulationStd(x) < kDegenerate) return "Data has zero variance (constant values)";
    return {};
}

// Standard normal quantile (Wichura, AS241 PPND16; relative error ~1e-16).
af::array NormalQuantile(const af::array& p) {
    const auto poly = [](const af::array& r, std::initializer_list<double> c) {
        af::array value = af::constant(0.0, r.dims(), r.type());
        for (auto it = std::rbegin(c); it != std::rend(c); ++it) value = value * r + *it;
        return value;
    };
    const af::array q = p - 0.5;
    const af::array rc = 0.180625 - q * q;
    const af::array central = q *
        poly(rc, {3.3871328727963666080e0, 1.3314166789178437745e+2, 1.9715909503065514427e+3,
                  1.3731693765509461125e+4, 4.5921953931549871457e+4, 6.7265770927008700853e+4,
                  3.3430575583588128105e+4, 2.5090809287301226727e+3}) /
        poly(rc, {1.0, 4.2313330701600911252e+1, 6.8718700749205790830e+2, 5.3941960214247511077e+3,
                  2.1213794301586595867e+4, 3.9307895800092710610e+4, 2.8729085735721942674e+4,
                  5.2264952788528545610e+3});
    const af::array r = af::sqrt(-af::log(af::select(q < 0.0, p, 1.0 - p)));
    const af::array rm = r - 1.6;
    const af::array inner_tail = poly(rm, {1.42343711074968357734e0, 4.63033784615654529590e0, 5.76949722146069140550e0,
                                     3.64784832476320460504e0, 1.27045825245236838258e0, 2.41780725177450611770e-1,
                                     2.27238449892691845833e-2, 7.74545014278341407640e-4}) /
        poly(rm, {1.0, 2.05319162663775882187e0, 1.67638483018380384940e0, 6.89767334985100004550e-1,
                  1.48103976427480074590e-1, 1.51986665636164571966e-2, 5.47593808499534494600e-4,
                  1.05075007164441684324e-9});
    const af::array rf = r - 5.0;
    const af::array outer_tail = poly(rf, {6.65790464350110377720e0, 5.46378491116411436990e0, 1.78482653991729133580e0,
                                    2.96560571828504891230e-1, 2.65321895265761230930e-2, 1.24266094738807843860e-3,
                                    2.71155556874348757815e-5, 2.01033439929228813265e-7}) /
        poly(rf, {1.0, 5.99832206555887937690e-1, 1.36929880922735805310e-1, 1.48753612908506148525e-2,
                  7.86869131145613259100e-4, 1.84631831751005468180e-5, 1.42151175831644588870e-7,
                  2.04426310338993978564e-15});
    const af::array tail = af::select(r <= 5.0, inner_tail, outer_tail);
    return af::select(af::abs(q) <= 0.425, central, af::select(q < 0.0, -tail, tail));
}

// For each x, the number of entries of the ascending q that are <= x
// (strict: < x); a vectorised binary search.
af::array CountBelow(const af::array& q, const af::array& x, bool strict) {
    const int size = static_cast<int>(q.elements());
    af::array low = af::constant(0, x.dims(), s32);
    af::array high = af::constant(size, x.dims(), s32);
    const af::array last = af::constant(size - 1, x.dims(), s32);
    for (int span = size + 1; span > 0; span /= 2) {  // floor(log2(size + 1)) + 1 halvings
        const af::array active = low < high;
        const af::array mid = (low + high) >> 1;
        const af::array probe = af::lookup(q, af::min(mid, last).as(u32));
        const af::array go_right = active && (strict ? probe < x : probe <= x);
        low = af::select(go_right, mid + 1, low);
        high = af::select(active && !go_right, mid, high);
    }
    return low;
}

// np.interp(x, q, refs) (forward) averaged with -np.interp(-x, -q[::-1],
// -refs[::-1]) (reverse), as QuantileTransformer does to centre repeated quantiles.
af::array QuantileInterpolate(const af::array& q, const af::array& refs, const af::array& x) {
    const int size = static_cast<int>(q.elements());
    const auto at = [](const af::array& values, const af::array& index) { return af::lookup(values, index.as(u32)); };
    const af::array first = af::constant(0, x.dims(), s32), last = af::constant(size - 1, x.dims(), s32);
    const auto clamp = [&](const af::array& index) { return af::max(af::min(index, last), first); };

    const af::array k = CountBelow(q, x, false) - 1;  // rightmost q[k] <= x
    const af::array k0 = clamp(k), k1 = clamp(k + 1);
    const af::array qk = at(q, k0), rk = at(refs, k0);
    const af::array forward_slope = (at(refs, k1) - rk) / (at(q, k1) - qk);
    af::array forward = af::select(qk == x, rk, forward_slope * (x - qk) + rk);
    forward = af::select(k < 0, at(refs, first), forward);
    forward = af::select(k >= size - 1, at(refs, last), forward);

    const af::array m = CountBelow(q, x, true);  // leftmost q[m] >= x
    const af::array m0 = clamp(m), m1 = clamp(m - 1);
    const af::array qm = at(q, m0), rm = at(refs, m0);
    const af::array reverse_slope = (rm - at(refs, m1)) / (qm - at(q, m1));
    af::array reverse = af::select(qm == x, rm, rm - reverse_slope * (qm - x));
    reverse = af::select(m <= 0, at(refs, first), reverse);
    reverse = af::select(m >= size, at(refs, last), reverse);
    return 0.5 * (forward + reverse);
}

// Runs `transform` on each column on the device. It returns the transformed column,
// or sets `refusal` (and the result is the column's error).
template <typename Transform>
TransformResult TransformEachColumn(const Columns& data, const char* method, const char* operation,
                                    Transform&& transform) {
    TransformResult result;
    result.method = method;
    if (data.empty()) return Fail(result, "Empty data");
    try {
        for (size_t col = 0; col < data.size(); ++col) {
            if (data[col].empty()) return Fail(result, ColumnMessage(col, "Empty data"));
            const af::array x = HostValuesToDevice(data[col]);
            std::string refusal;
            const af::array y = transform(x, "col" + std::to_string(col) + "_", result.params, refusal);
            if (!refusal.empty()) return Fail(result, ColumnMessage(col, refusal));
            result.transformed_data.push_back(DeviceToHostValues(y, operation));
        }
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        return Fail(result, DeviceError(operation, e));
    }
}

#else

std::string NoArrayFire(const char* operation) {
    return std::string(operation) + " runs on ArrayFire, and this build has no ArrayFire";
}

#endif

}  // namespace

// ============================================================================
// Input checks
// ============================================================================

bool DataTransform::CanApplyLogTransform(const std::vector<double>& data, bool use_log1p) {
    return std::all_of(data.begin(), data.end(), [use_log1p](double x) { return use_log1p ? x > -1.0 : x > 0.0; });
}

bool DataTransform::CanApplyBoxCox(const std::vector<double>& data) {
    return std::all_of(data.begin(), data.end(), [](double x) { return x > 0.0; });
}

// ============================================================================
// Statistics
// ============================================================================

ColumnStats DataTransform::ComputeColumnStats(const std::vector<double>& data) {
    ColumnStats stats;
    if (data.empty()) {
        stats.error_message = "Empty data";
        return stats;
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::array x = HostValuesToDevice(data);
        stats.min = af::min<double>(x);
        stats.max = af::max<double>(x);
        stats.mean = af::mean<double>(x);
        stats.std_dev = PopulationStd(x);
        const std::vector<double> quartiles =
            PercentilesAt(af::sort(x), {0.25, 0.5, 0.75}, "DataTransform::ComputeColumnStats");
        stats.q1 = quartiles[0];
        stats.median = quartiles[1];
        stats.q3 = quartiles[2];
        stats.iqr = stats.q3 - stats.q1;
        stats.count = static_cast<int>(data.size());
    } catch (const af::exception& e) {
        stats = ColumnStats{};
        stats.error_message = DeviceError("DataTransform::ComputeColumnStats", e);
    }
#else
    stats.error_message = NoArrayFire("DataTransform::ComputeColumnStats");
#endif
    return stats;
}

NormalityTestResult DataTransform::DAgostinoPearsonTest(const std::vector<double>& data) {
    NormalityTestResult result;
    result.test_name = "D'Agostino-Pearson";
    if (data.size() < 20) {
        result.error_message = "Needs at least 20 values";
        return result;
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    double m2 = 0.0, m3 = 0.0, m4 = 0.0;
    try {
        const af::array x = HostValuesToDevice(data);
        const af::array d = x - af::mean<double>(x);
        const af::array d2 = d * d;
        m2 = af::mean<double>(d2);
        m3 = af::mean<double>(d2 * d);
        m4 = af::mean<double>(d2 * d2);
    } catch (const af::exception& e) {
        result.error_message = DeviceError("DataTransform::DAgostinoPearsonTest", e);
        return result;
    }
    if (m2 <= 0.0) {
        result.error_message = "Data has zero variance (constant values)";
        return result;
    }
    // scipy.stats.skewtest / kurtosistest on the biased moments, combined.
    const double n = static_cast<double>(data.size());
    const double skew = m3 / std::pow(m2, 1.5);
    double y = skew * std::sqrt((n + 1.0) * (n + 3.0) / (6.0 * (n - 2.0)));
    const double beta2 = 3.0 * (n * n + 27.0 * n - 70.0) * (n + 1.0) * (n + 3.0) /
                         ((n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0));
    const double w2 = -1.0 + std::sqrt(2.0 * (beta2 - 1.0));
    const double delta = 1.0 / std::sqrt(0.5 * std::log(w2));
    const double alpha = std::sqrt(2.0 / (w2 - 1.0));
    if (y == 0.0) y = 1.0;
    const double z_skew = delta * std::log(y / alpha + std::sqrt((y / alpha) * (y / alpha) + 1.0));

    const double b2 = m4 / (m2 * m2);
    const double expected = 3.0 * (n - 1.0) / (n + 1.0);
    const double var_b2 = 24.0 * n * (n - 2.0) * (n - 3.0) / ((n + 1.0) * (n + 1.0) * (n + 3.0) * (n + 5.0));
    const double standardized = (b2 - expected) / std::sqrt(var_b2);
    const double sqrt_beta1 = 6.0 * (n * n - 5.0 * n + 2.0) / ((n + 7.0) * (n + 9.0)) *
                              std::sqrt(6.0 * (n + 3.0) * (n + 5.0) / (n * (n - 2.0) * (n - 3.0)));
    const double a = 6.0 + 8.0 / sqrt_beta1 * (2.0 / sqrt_beta1 + std::sqrt(1.0 + 4.0 / (sqrt_beta1 * sqrt_beta1)));
    const double term1 = 1.0 - 2.0 / (9.0 * a);
    const double denom = 1.0 + standardized * std::sqrt(2.0 / (a - 4.0));
    if (denom == 0.0) {
        result.error_message = "Kurtosis test is undefined for this data";
        return result;
    }
    const double term2 = (denom > 0.0 ? 1.0 : -1.0) * std::cbrt((1.0 - 2.0 / a) / std::abs(denom));
    const double z_kurt = (term1 - term2) / std::sqrt(2.0 / (9.0 * a));

    result.statistic = z_skew * z_skew + z_kurt * z_kurt;
    result.p_value = std::exp(-0.5 * result.statistic);  // chi-squared, 2 degrees of freedom
    result.is_normal = result.p_value > 0.05;
    result.computed = true;
#else
    result.error_message = NoArrayFire("DataTransform::DAgostinoPearsonTest");
#endif
    return result;
}

// ============================================================================
// Scalers
// ============================================================================

TransformResult DataTransform::Normalize(const Columns& data, double range_min, double range_max) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    TransformResult result = TransformEachColumn(data, "normalize", "DataTransform::Normalize",
        [&](const af::array& x, const std::string& prefix, auto& params, std::string& refusal) -> af::array {
            const double low = af::min<double>(x), high = af::max<double>(x);
            if (high - low < kDegenerate) {
                refusal = "Data has zero range (constant values)";
                return {};
            }
            params[prefix + "min"] = low;
            params[prefix + "max"] = high;
            return (x - low) / (high - low) * (range_max - range_min) + range_min;
        });
    if (result.success) {
        result.params["range_min"] = range_min;
        result.params["range_max"] = range_max;
    }
    return result;
#else
    TransformResult result;
    result.method = "normalize";
    return Fail(result, NoArrayFire("DataTransform::Normalize"));
#endif
}

TransformResult DataTransform::Standardize(const Columns& data) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    return TransformEachColumn(data, "standardize", "DataTransform::Standardize",
        [](const af::array& x, const std::string& prefix, auto& params, std::string& refusal) -> af::array {
            const double mean = af::mean<double>(x);
            const double std_dev = PopulationStd(x);
            if (std_dev < kDegenerate) {
                refusal = "Data has zero variance (constant values)";
                return {};
            }
            params[prefix + "mean"] = mean;
            params[prefix + "std"] = std_dev;
            return (x - mean) / std_dev;
        });
#else
    TransformResult result;
    result.method = "standardize";
    return Fail(result, NoArrayFire("DataTransform::Standardize"));
#endif
}

TransformResult DataTransform::LogTransform(const Columns& data, const std::string& base, bool use_log1p) {
    if (base != "natural" && base != "log10" && base != "log2") {
        TransformResult result;
        result.method = "log_transform";
        return Fail(result, "Unknown log base '" + base + "' (natural, log10 or log2)");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    TransformResult result = TransformEachColumn(data, "log_transform", "DataTransform::LogTransform",
        [&](const af::array& x, const std::string&, auto&, std::string& refusal) -> af::array {
            if (!af::allTrue<bool>(use_log1p ? x > -1.0 : x > 0.0)) {
                refusal = use_log1p ? "Data contains values <= -1 (log1p requires x > -1)"
                                    : "Data contains non-positive values (log requires x > 0)";
                return {};
            }
            const af::array natural = use_log1p ? af::log1p(x) : af::log(x);
            if (base == "natural") return natural;
            return natural / std::log(base == "log10" ? 10.0 : 2.0);
        });
#else
    TransformResult result;
    result.method = "log_transform";
    result = Fail(result, NoArrayFire("DataTransform::LogTransform"));
#endif
    result.params["base"] = base == "natural" ? 0.0 : base == "log10" ? 10.0 : 2.0;
    result.params["use_log1p"] = use_log1p ? 1.0 : 0.0;
    return result;
}

TransformResult DataTransform::RobustScale(const Columns& data) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    return TransformEachColumn(data, "robust_scale", "DataTransform::RobustScale",
        [](const af::array& x, const std::string& prefix, auto& params, std::string& refusal) -> af::array {
            const std::vector<double> q = PercentilesAt(af::sort(x), {0.25, 0.5, 0.75}, "DataTransform::RobustScale");
            const double iqr = q[2] - q[0];
            if (iqr < kDegenerate) {
                refusal = "Data has zero IQR";
                return {};
            }
            params[prefix + "median"] = q[1];
            params[prefix + "iqr"] = iqr;
            return (x - q[1]) / iqr;
        });
#else
    TransformResult result;
    result.method = "robust_scale";
    return Fail(result, NoArrayFire("DataTransform::RobustScale"));
#endif
}

TransformResult DataTransform::MaxAbsScale(const Columns& data) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    return TransformEachColumn(data, "maxabs_scale", "DataTransform::MaxAbsScale",
        [](const af::array& x, const std::string& prefix, auto& params, std::string& refusal) -> af::array {
            const double max_abs = af::max<double>(af::abs(x));
            if (max_abs < kDegenerate) {
                refusal = "Data is all zeros";
                return {};
            }
            params[prefix + "maxabs"] = max_abs;
            return x / max_abs;
        });
#else
    TransformResult result;
    result.method = "maxabs_scale";
    return Fail(result, NoArrayFire("DataTransform::MaxAbsScale"));
#endif
}

TransformResult DataTransform::QuantileTransform(const Columns& data, const std::string& output_distribution,
                                                 int n_quantiles) {
    const bool normal = output_distribution == "normal";
    if (!normal && output_distribution != "uniform") {
        TransformResult result;
        result.method = "quantile_transform";
        return Fail(result, "Unknown output distribution '" + output_distribution + "' (uniform or normal)");
    }
    if (n_quantiles < 2) {
        TransformResult result;
        result.method = "quantile_transform";
        return Fail(result, "Needs at least 2 quantiles");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    TransformResult result = TransformEachColumn(data, "quantile_transform", "DataTransform::QuantileTransform",
        [&](const af::array& x, const std::string&, auto&, std::string& refusal) -> af::array {
            const dim_t n = x.elements();
            if (n < 2) {
                refusal = "Needs at least 2 values";
                return {};
            }
            const int size = static_cast<int>(std::min<dim_t>(n_quantiles, n));
            const af::array refs = af::range(af::dim4(size), 0, x.type()) / static_cast<double>(size - 1);
            const af::array quantiles = af::scan(ExactQuantiles(af::sort(x), size), 0, AF_BINARY_MAX);
            const double lower_x = af::min<double>(quantiles), upper_x = af::max<double>(quantiles);

            af::array y = QuantileInterpolate(quantiles, refs, x);
            constexpr double kBoundsThreshold = 1e-7;
            const af::array upper = normal ? x + kBoundsThreshold > upper_x : x == upper_x;
            const af::array lower = normal ? x - kBoundsThreshold < lower_x : x == lower_x;
            y = af::select(upper, 1.0, y);
            y = af::select(lower, 0.0, y);
            if (!normal) return y;
            const double clip = kBoundsThreshold - std::numeric_limits<double>::epsilon();
            return NormalQuantile(af::max(af::min(y, af::constant(1.0 - clip, y.dims(), y.type())),
                                          af::constant(clip, y.dims(), y.type())));
        });
#else
    TransformResult result;
    result.method = "quantile_transform";
    result = Fail(result, NoArrayFire("DataTransform::QuantileTransform"));
#endif
    result.params["output_distribution"] = normal ? 1.0 : 0.0;
    result.params["n_quantiles"] = static_cast<double>(n_quantiles);
    return result;
}

// ============================================================================
// Box-Cox and Yeo-Johnson
// ============================================================================

namespace {

#ifdef CYXWIZ_HAS_ARRAYFIRE
BoxCoxLambdaResult FindLambda(const std::vector<double>& data, PowerFamily family, double lambda_min,
                              double lambda_max, int n_steps, const char* operation) {
    BoxCoxLambdaResult result;
    if (data.empty()) return Fail(result, "Empty data");
    try {
        const af::array x = HostValuesToDevice(data);
        const std::string refusal = LambdaSearchRefusal(x, family, lambda_min, lambda_max, n_steps);
        if (!refusal.empty()) return Fail(result, refusal);
        return SearchLambda(x, family, lambda_min, lambda_max, n_steps, operation);
    } catch (const af::exception& e) {
        return Fail(result, DeviceError(operation, e));
    }
}

TransformResult PowerColumn(const std::vector<double>& data, PowerFamily family, double lambda, bool auto_lambda,
                            const char* method, const char* operation) {
    TransformResult result;
    result.method = method;
    if (data.empty()) return Fail(result, "Empty data");
    try {
        const af::array x = HostValuesToDevice(data);
        if (family == PowerFamily::BoxCox && !af::allTrue<bool>(x > 0.0)) {
            return Fail(result, "Box-Cox requires strictly positive data");
        }
        double use_lambda = lambda;
        if (auto_lambda) {
            const std::string refusal = LambdaSearchRefusal(x, family, -5.0, 5.0, 100);
            if (!refusal.empty()) return Fail(result, refusal);
            const BoxCoxLambdaResult search = SearchLambda(x, family, -5.0, 5.0, 100, operation);
            if (!search.success) return Fail(result, search.error_message);
            use_lambda = search.optimal_lambda;
        }
        const af::array y = PowerTransformed(x, RowLike({use_lambda}, x), family);
        result.transformed_data.push_back(DeviceToHostValues(y, operation));
        result.params["lambda"] = use_lambda;
        result.success = true;
        return result;
    } catch (const af::exception& e) {
        return Fail(result, DeviceError(operation, e));
    }
}
#endif

}  // namespace

BoxCoxLambdaResult DataTransform::FindOptimalLambda(const std::vector<double>& data, double lambda_min,
                                                    double lambda_max, int n_steps) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    return FindLambda(data, PowerFamily::BoxCox, lambda_min, lambda_max, n_steps, "DataTransform::FindOptimalLambda");
#else
    return Fail(BoxCoxLambdaResult{}, NoArrayFire("DataTransform::FindOptimalLambda"));
#endif
}

BoxCoxLambdaResult DataTransform::FindOptimalYeoJohnsonLambda(const std::vector<double>& data, double lambda_min,
                                                              double lambda_max, int n_steps) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    return FindLambda(data, PowerFamily::YeoJohnson, lambda_min, lambda_max, n_steps,
                      "DataTransform::FindOptimalYeoJohnsonLambda");
#else
    return Fail(BoxCoxLambdaResult{}, NoArrayFire("DataTransform::FindOptimalYeoJohnsonLambda"));
#endif
}

TransformResult DataTransform::BoxCoxColumn(const std::vector<double>& data, double lambda, bool auto_lambda) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    return PowerColumn(data, PowerFamily::BoxCox, lambda, auto_lambda, "boxcox", "DataTransform::BoxCoxColumn");
#else
    TransformResult result;
    result.method = "boxcox";
    return Fail(result, NoArrayFire("DataTransform::BoxCoxColumn"));
#endif
}

TransformResult DataTransform::YeoJohnsonColumn(const std::vector<double>& data, double lambda, bool auto_lambda) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    return PowerColumn(data, PowerFamily::YeoJohnson, lambda, auto_lambda, "yeojohnson",
                       "DataTransform::YeoJohnsonColumn");
#else
    TransformResult result;
    result.method = "yeojohnson";
    return Fail(result, NoArrayFire("DataTransform::YeoJohnsonColumn"));
#endif
}

} // namespace cyxwiz
