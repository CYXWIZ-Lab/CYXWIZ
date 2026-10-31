#ifdef _WIN32
#define NOMINMAX
#endif

#include "series_device.h"

#include "../arrayfire_host_values.h"
#include "cyxwiz/statistical_tests.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace cyxwiz::series_detail {

std::string SeriesRefusal(const std::vector<double>& data, size_t min_n, const char* what) {
    if (data.size() < min_n) return std::string(what) + " needs at least " + std::to_string(min_n) + " values (" + std::to_string(data.size()) + " given)";
    for (const double v : data)
        if (!std::isfinite(v)) return std::string(what) + ": the series has a missing or infinite value";
    return "";
}

af::dtype ComputeType() { return DeviceHasDouble() ? f64 : f32; }

af::array ToDevice(const std::vector<double>& values) { return HostValuesToDevice(values).as(ComputeType()); }

std::vector<double> ToHost(const af::array& values, const char* operation) { return DeviceToHostValues(values, operation); }

namespace {

dim_t PowerOfTwoAtLeast(dim_t n) {
    dim_t p = 1;
    while (p < n) p *= 2;
    return p;
}

af::dtype ComplexType() { return DeviceHasDouble() ? c64 : c32; }

}  // namespace

std::vector<double> Autocovariance(const std::vector<double>& x, int max_lag) {
    const dim_t n = static_cast<dim_t>(x.size());
    max_lag = std::max(0, std::min<int>(max_lag, static_cast<int>(n) - 1));
    af::array d = ToDevice(x);
    d = d - af::mean<double>(d);
    const dim_t size = PowerOfTwoAtLeast(2 * n);
    af::array padded = af::constant(0.0, size, ComputeType());
    padded(af::seq(0, static_cast<double>(n - 1))) = d;
    const af::array spectrum = af::fft(padded);
    const af::array power = af::real(spectrum * af::conjg(spectrum)).as(ComplexType());
    const af::array sums = af::real(af::ifftNorm(power, 1.0 / static_cast<double>(size)));
    std::vector<double> c = ToHost(sums(af::seq(0, static_cast<double>(max_lag))), "TimeSeries autocovariance");
    for (double& v : c) v /= static_cast<double>(n);
    return c;
}

void Periodogram(const std::vector<double>& x, std::vector<double>& power, std::vector<double>& frequency) {
    const dim_t n = static_cast<dim_t>(x.size());
    af::array d = ToDevice(x);
    d = d - af::mean<double>(d);
    const af::array spectrum = af::fft(d);
    const dim_t half = n / 2;
    af::array p = af::real(spectrum * af::conjg(spectrum))(af::seq(0, static_cast<double>(half)));
    p = p / (static_cast<double>(n) * static_cast<double>(n));
    power = ToHost(p, "TimeSeries periodogram");
    // One-sided: every frequency but 0 (and Nyquist when n is even) carries its mirror too.
    for (size_t k = 1; k < power.size(); ++k)
        if (!(n % 2 == 0 && static_cast<dim_t>(k) == half)) power[k] *= 2.0;
    frequency.resize(power.size());
    for (size_t k = 0; k < frequency.size(); ++k) frequency[k] = static_cast<double>(k) / static_cast<double>(n);
}

std::vector<double> LinearDetrend(const std::vector<double>& x) {
    std::vector<double> t(x.size());
    std::iota(t.begin(), t.end(), 0.0);
    const LinearFit fit = StatisticalTests::FitLinear({t}, x, {"t"});
    if (!fit.ok) return x;
    return fit.residuals;
}

std::vector<double> TwoSidedFilter(const std::vector<double>& x, const std::vector<double>& filter) {
    const size_t n = x.size(), length = filter.size();
    std::vector<double> out(n, NAN);
    if (length == 0 || length % 2 == 0 || n < length) return out;
    const af::array filtered = af::convolve1(ToDevice(x), ToDevice(filter), AF_CONV_DEFAULT);
    std::vector<double> values = ToHost(filtered, "TimeSeries two-sided filter");
    const size_t trim = (length - 1) / 2;
    for (size_t i = trim; i + trim < n; ++i) out[i] = values[i];
    return out;
}

std::vector<double> SeasonalMeans(const std::vector<double>& x, int period) {
    const size_t n = x.size(), m = static_cast<size_t>(period);
    const size_t cycles = (n + m - 1) / m;
    std::vector<double> padded(cycles * m, NAN);
    std::copy(x.begin(), x.end(), padded.begin());
    // [period, cycles] on the device: the mean over the finite values of each row.
    const af::array table = af::moddims(ToDevice(padded), static_cast<dim_t>(m), static_cast<dim_t>(cycles));
    const af::array finite = !af::isNaN(table);
    const af::array sums = af::sum(af::select(finite, table, 0.0), 1);
    const af::array counts = af::sum(finite.as(ComputeType()), 1);
    return ToHost(sums / counts, "TimeSeries seasonal means");
}

void Rolling(const std::vector<double>& x, int window, std::vector<double>& mean, std::vector<double>& sd) {
    const size_t n = x.size(), w = static_cast<size_t>(std::max(1, window));
    mean.assign(n, NAN);
    sd.assign(n, NAN);
    if (n < w) return;
    // Window sums by a device convolution of x and x^2 (the window's mean, then
    // its sample variance), on the series less its mean so the sums do not
    // cancel on a float32 device (the variance is shift-invariant).
    const af::array raw = ToDevice(x);
    const double centre = af::mean<double>(raw);
    const af::array d = raw - centre;
    const af::array ones = af::constant(1.0, static_cast<dim_t>(w), ComputeType());
    const std::vector<double> s1 = ToHost(af::convolve1(d, ones, AF_CONV_EXPAND), "TimeSeries rolling");
    const std::vector<double> s2 = ToHost(af::convolve1(d * d, ones, AF_CONV_EXPAND), "TimeSeries rolling");
    for (size_t i = w - 1; i < n; ++i) {
        const double sum = s1[i], sum2 = s2[i];
        mean[i] = sum / static_cast<double>(w) + centre;
        if (w > 1) sd[i] = std::sqrt(std::max(0.0, (sum2 - sum * sum / static_cast<double>(w)) / static_cast<double>(w - 1)));
    }
}

double NormalPpf(double q) {
    if (!(q > 0.0 && q < 1.0)) return q <= 0.0 ? -INFINITY : INFINITY;
    // Newton on the normal CDF from a rational start (Abramowitz-Stegun 26.2.23).
    const double p = q < 0.5 ? q : 1.0 - q;
    const double t = std::sqrt(-2.0 * std::log(p));
    double z = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t);
    if (q < 0.5) z = -z;
    for (int i = 0; i < 6; ++i) {
        const double err = StatisticalTests::NormalCdf(z) - q;
        const double density = std::exp(-0.5 * z * z) / std::sqrt(2.0 * 3.14159265358979323846);
        if (density <= 0.0) break;
        z -= err / density;
    }
    return z;
}

double Strength(const std::vector<double>& residual, const std::vector<double>& component) {
    std::vector<double> r, s;
    for (size_t i = 0; i < residual.size() && i < component.size(); ++i)
        if (std::isfinite(residual[i]) && std::isfinite(component[i])) {
            r.push_back(residual[i]);
            s.push_back(residual[i] + component[i]);
        }
    if (r.size() < 2) return 0.0;
    const auto var = [](const std::vector<double>& v) {
        const double mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
        double ss = 0.0;
        for (const double x : v) ss += (x - mean) * (x - mean);
        return ss / static_cast<double>(v.size());
    };
    const double vs = var(s);
    if (!(vs > 0.0)) return 0.0;
    return std::max(0.0, 1.0 - var(r) / vs);
}

}  // namespace cyxwiz::series_detail
