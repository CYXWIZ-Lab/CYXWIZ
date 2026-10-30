#pragma once

// Device helpers shared by the time-series analysis (TOFIX129 D6): a series
// on the ArrayFire device in the compute precision, autocovariance and the
// periodogram through the device FFT, two-sided filters by device convolution,
// linear detrending by the device least-squares fit, and the host-side
// distribution helpers the tests need. The series stays small (its length),
// so what is sequential by nature (loess, smoothing recursions, the Kalman
// filter) runs on the host; what has a vector form runs on the device.

#include <string>
#include <vector>

#include <arrayfire.h>

namespace cyxwiz::series_detail {

// Empty when `data` has at least `min_n` finite values; else why not.
std::string SeriesRefusal(const std::vector<double>& data, size_t min_n, const char* what);

af::dtype ComputeType();
af::array ToDevice(const std::vector<double>& values);
std::vector<double> ToHost(const af::array& values, const char* operation);

// Biased autocovariance c_k = (1 / n) sum (x_t - mean)(x_{t+k} - mean), k = 0 .. max_lag,
// by the device FFT of the zero-padded series.
std::vector<double> Autocovariance(const std::vector<double>& x, int max_lag);

// One-sided periodogram of the mean-removed series (scipy.signal.periodogram
// with a boxcar window and scaling "spectrum"): power at frequency k / n for
// k = 0 .. n / 2, the interior frequencies doubled.
void Periodogram(const std::vector<double>& x, std::vector<double>& power, std::vector<double>& frequency);

// x minus its least-squares line (scipy.signal.detrend type "linear").
std::vector<double> LinearDetrend(const std::vector<double>& x);

// Two-sided moving filter of odd length (statsmodels convolution_filter,
// nsides 2): the first and last (length - 1) / 2 values are NaN.
std::vector<double> TwoSidedFilter(const std::vector<double>& x, const std::vector<double>& filter);

// Means by position in the cycle, over the values that are finite
// (statsmodels seasonal_mean with nanmean): `period` values.
std::vector<double> SeasonalMeans(const std::vector<double>& x, int period);

// Rolling mean and sample standard deviation over `window` values (the
// first window - 1 are NaN; pandas rolling(window).mean() / .std()).
void Rolling(const std::vector<double>& x, int window, std::vector<double>& mean, std::vector<double>& sd);

// The normal quantile with P(Z <= z) = q.
double NormalPpf(double q);

// Variance of the finite values (population) and the strength
// 1 - var(residual) / var(residual + component) (Hyndman), 0 when negative.
double Strength(const std::vector<double>& residual, const std::vector<double>& component);

}  // namespace cyxwiz::series_detail
