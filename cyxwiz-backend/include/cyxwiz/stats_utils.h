#pragma once

/**
 * stats_utils.h - Lightweight statistical utilities
 *
 * Pure math functions that work on std::vector<double>.
 * No DataTable dependency, header-only for simplicity.
 */

#include "api_export.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace cyxwiz {
namespace stats {

// ===== Basic Statistics =====

inline double Mean(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

inline double Median(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    size_t n = sorted.size();
    return (n % 2 == 0) ? (sorted[n/2-1] + sorted[n/2]) / 2.0 : sorted[n/2];
}

inline double Variance(const std::vector<double>& data, bool sample = true) {
    if (data.size() < 2) return 0.0;
    double mean = Mean(data);
    double sum = 0.0;
    for (double v : data) sum += (v - mean) * (v - mean);
    return sum / (sample ? data.size() - 1 : data.size());
}

inline double StdDev(const std::vector<double>& data, bool sample = true) {
    return std::sqrt(Variance(data, sample));
}

inline double Min(const std::vector<double>& data) {
    return data.empty() ? 0.0 : *std::min_element(data.begin(), data.end());
}

inline double Max(const std::vector<double>& data) {
    return data.empty() ? 0.0 : *std::max_element(data.begin(), data.end());
}

// ===== Percentiles =====

inline double Percentile(const std::vector<double>& data, double p) {
    if (data.empty()) return 0.0;
    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    double idx = p * (sorted.size() - 1);
    size_t lo = static_cast<size_t>(idx);
    size_t hi = std::min(lo + 1, sorted.size() - 1);
    double frac = idx - lo;
    return sorted[lo] * (1 - frac) + sorted[hi] * frac;
}

inline double Q1(const std::vector<double>& data) { return Percentile(data, 0.25); }
inline double Q3(const std::vector<double>& data) { return Percentile(data, 0.75); }
inline double IQR(const std::vector<double>& data) { return Q3(data) - Q1(data); }

// ===== Correlation =====

inline double PearsonCorrelation(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.empty()) return std::nan("");

    size_t n = x.size();
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;

    for (size_t i = 0; i < n; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
        sum_y2 += y[i] * y[i];
    }

    double num = n * sum_xy - sum_x * sum_y;
    double den = std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));

    return (den != 0) ? num / den : std::nan("");
}

// ===== Shape Statistics =====

inline double Skewness(const std::vector<double>& data) {
    if (data.size() < 3) return 0.0;
    double mean = Mean(data);
    double std = StdDev(data, false);
    if (std == 0) return 0.0;

    double sum = 0.0;
    for (double v : data) {
        double z = (v - mean) / std;
        sum += z * z * z;
    }
    return sum / data.size();
}

inline double Kurtosis(const std::vector<double>& data) {
    if (data.size() < 4) return 0.0;
    double mean = Mean(data);
    double std = StdDev(data, false);
    if (std == 0) return 0.0;

    double sum = 0.0;
    for (double v : data) {
        double z = (v - mean) / std;
        sum += z * z * z * z;
    }
    return sum / data.size() - 3.0;  // Excess kurtosis
}

// ===== Descriptive Stats Struct =====

struct DescriptiveStats {
    size_t count = 0;
    double mean = 0.0;
    double median = 0.0;
    double std_dev = 0.0;
    double variance = 0.0;
    double min = 0.0;
    double max = 0.0;
    double q1 = 0.0;
    double q3 = 0.0;
    double iqr = 0.0;
    double skewness = 0.0;
    double kurtosis = 0.0;
};

inline DescriptiveStats ComputeStats(const std::vector<double>& data) {
    DescriptiveStats s;
    if (data.empty()) return s;

    s.count = data.size();
    s.mean = Mean(data);
    s.median = Median(data);
    s.variance = Variance(data);
    s.std_dev = std::sqrt(s.variance);
    s.min = Min(data);
    s.max = Max(data);
    s.q1 = Q1(data);
    s.q3 = Q3(data);
    s.iqr = s.q3 - s.q1;
    s.skewness = Skewness(data);
    s.kurtosis = Kurtosis(data);

    return s;
}

// ===== Outlier Detection =====

inline std::vector<size_t> DetectOutliersIQR(const std::vector<double>& data, double k = 1.5) {
    std::vector<size_t> outliers;
    double q1 = Q1(data);
    double q3 = Q3(data);
    double iqr = q3 - q1;
    double lower = q1 - k * iqr;
    double upper = q3 + k * iqr;

    for (size_t i = 0; i < data.size(); i++) {
        if (data[i] < lower || data[i] > upper) {
            outliers.push_back(i);
        }
    }
    return outliers;
}

inline std::vector<size_t> DetectOutliersZScore(const std::vector<double>& data, double threshold = 3.0) {
    std::vector<size_t> outliers;
    double mean = Mean(data);
    double std = StdDev(data);
    if (std == 0) return outliers;

    for (size_t i = 0; i < data.size(); i++) {
        if (std::abs(data[i] - mean) / std > threshold) {
            outliers.push_back(i);
        }
    }
    return outliers;
}

} // namespace stats
} // namespace cyxwiz
