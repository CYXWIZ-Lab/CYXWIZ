#pragma once

#include <cstddef>
#include <vector>

namespace cyxwiz::series {

// A line series reduced for drawing (TOFIX134 P0 item 8). Up to
// max_points values are kept: the series is cut into buckets and each bucket
// keeps its lowest and highest value in their original order, so spikes and
// dips stay visible. A series that already fits is kept as it is.
struct Decimated {
    std::vector<double> x;
    std::vector<double> y;
};

Decimated MinMaxDecimate(const std::vector<double>& x, const std::vector<double>& y, size_t max_points);

// Mean of the last `window` values at each point (fewer at the start).
std::vector<double> MovingAverage(const std::vector<double>& values, int window);

}  // namespace cyxwiz::series
