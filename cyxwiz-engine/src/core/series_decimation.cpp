#include "series_decimation.h"

#include <algorithm>

namespace cyxwiz::series {

Decimated MinMaxDecimate(const std::vector<double>& x, const std::vector<double>& y, size_t max_points) {
    Decimated out;
    const size_t n = std::min(x.size(), y.size());
    if (n <= max_points || max_points < 4) {
        out.x.assign(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(n));
        out.y.assign(y.begin(), y.begin() + static_cast<std::ptrdiff_t>(n));
        return out;
    }
    const size_t buckets = max_points / 2;
    out.x.reserve(buckets * 2 + 2);
    out.y.reserve(buckets * 2 + 2);
    // The first value is always drawn (the line starts where the data does).
    out.x.push_back(x[0]);
    out.y.push_back(y[0]);
    for (size_t b = 0; b < buckets; ++b) {
        const size_t begin = b * n / buckets;
        const size_t end = (b + 1) * n / buckets;
        if (begin >= end) continue;
        size_t lo = begin, hi = begin;
        for (size_t i = begin + 1; i < end; ++i) {
            if (y[i] < y[lo]) lo = i;
            if (y[i] > y[hi]) hi = i;
        }
        const size_t first = std::min(lo, hi), second = std::max(lo, hi);
        if (first != 0) {
            out.x.push_back(x[first]);
            out.y.push_back(y[first]);
        }
        if (second != first) {
            out.x.push_back(x[second]);
            out.y.push_back(y[second]);
        }
    }
    // The newest value is always drawn (the line ends where training is).
    if (out.x.back() != x[n - 1] || out.y.back() != y[n - 1]) {
        out.x.push_back(x[n - 1]);
        out.y.push_back(y[n - 1]);
    }
    return out;
}

std::vector<double> MovingAverage(const std::vector<double>& values, int window) {
    std::vector<double> smoothed;
    smoothed.reserve(values.size());
    const size_t w = static_cast<size_t>(std::max(1, window));
    double running_sum = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        running_sum += values[i];
        if (i >= w) running_sum -= values[i - w];
        smoothed.push_back(running_sum / static_cast<double>(std::min(i + 1, w)));
    }
    return smoothed;
}

}  // namespace cyxwiz::series
