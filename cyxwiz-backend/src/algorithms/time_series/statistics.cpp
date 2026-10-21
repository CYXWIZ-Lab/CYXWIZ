// Prevent Windows min/max macros from interfering with std::min/std::max
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <cyxwiz/time_series.h>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace cyxwiz {
// ============================================================================
// Utility Functions
// ============================================================================

double TimeSeries::Mean(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

double TimeSeries::Variance(const std::vector<double>& data) {
    if (data.size() < 2) return 0.0;
    double m = Mean(data);
    double sum = 0.0;
    for (double x : data) {
        sum += (x - m) * (x - m);
    }
    return sum / (data.size() - 1);
}

double TimeSeries::StdDev(const std::vector<double>& data) {
    return std::sqrt(Variance(data));
}

std::vector<double> TimeSeries::RollingMean(const std::vector<double>& data, int window) {
    if (window <= 0 || data.size() < static_cast<size_t>(window)) {
        return {};
    }

    std::vector<double> result;
    result.reserve(data.size() - window + 1);

    double sum = 0.0;
    for (int i = 0; i < window; i++) {
        sum += data[i];
    }
    result.push_back(sum / window);

    for (size_t i = window; i < data.size(); i++) {
        sum += data[i] - data[i - window];
        result.push_back(sum / window);
    }

    return result;
}

std::vector<double> TimeSeries::RollingStd(const std::vector<double>& data, int window) {
    if (window <= 1 || data.size() < static_cast<size_t>(window)) {
        return {};
    }

    std::vector<double> result;
    result.reserve(data.size() - window + 1);

    for (size_t i = 0; i <= data.size() - window; i++) {
        double sum = 0.0, sum_sq = 0.0;
        for (int j = 0; j < window; j++) {
            sum += data[i + j];
            sum_sq += data[i + j] * data[i + j];
        }
        double mean = sum / window;
        double var = (sum_sq / window) - (mean * mean);
        result.push_back(std::sqrt((std::max)(0.0, var)));
    }

    return result;
}

std::vector<double> TimeSeries::CenteredMovingAverage(const std::vector<double>& data, int window) {
    if (window <= 0 || data.size() < static_cast<size_t>(window)) {
        return data;
    }

    std::vector<double> result(data.size(), 0.0);
    int half = window / 2;

    // For even window, we need an extra average
    bool even_window = (window % 2 == 0);

    for (size_t i = 0; i < data.size(); i++) {
        int start = static_cast<int>(i) - half;
        int end = static_cast<int>(i) + half;

        if (even_window) {
            // For even window, take average of two moving averages
            if (start < 0 || end >= static_cast<int>(data.size())) {
                result[i] = data[i];  // Edge: keep original
            } else {
                double sum1 = 0.0, sum2 = 0.0;
                for (int j = start; j < end; j++) {
                    sum1 += data[j];
                }
                for (int j = start + 1; j <= end; j++) {
                    sum2 += data[j];
                }
                result[i] = (sum1 / window + sum2 / window) / 2.0;
            }
        } else {
            if (start < 0 || end >= static_cast<int>(data.size())) {
                result[i] = data[i];
            } else {
                double sum = 0.0;
                for (int j = start; j <= end; j++) {
                    sum += data[j];
                }
                result[i] = sum / window;
            }
        }
    }

    return result;
}

} // namespace cyxwiz
