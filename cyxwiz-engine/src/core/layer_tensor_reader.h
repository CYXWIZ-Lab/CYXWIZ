#pragma once

// Reads for the Studio Debugger's layer views (TOFIX178 6.2): every
// reduction runs where the captured tensor lives, and the host gets only
// what a view draws (the shown channels, a page of heatmap rows, a
// histogram), each read attributed as a debug readback. Shapes are
// row-major as Tensor::Shape() reports them (see layer_view_presentation.h).

#include "layer_view_presentation.h"

#include <cyxwiz/tensor.h>

#include <cstddef>
#include <vector>

namespace cyxwiz::layer_view {

// One sample of an activation, on the device: [H, W, C, N] -> [H, W, C],
// [N, ...] -> [...]; a tensor without a batch axis comes back whole.
Tensor SampleOf(const Tensor& activation, size_t sample);

// Per channel of a spatial sample [H, W, C]: the mean and the maximum.
struct ChannelSummary {
    std::vector<float> means;
    std::vector<float> maxima;
};
ChannelSummary ReadChannelSummary(const Tensor& spatial_sample);

// The pictures of `channels` of a spatial sample [H, W, C]: H x W values per
// channel, row by row, one channel after another.
std::vector<float> ReadChannelImages(const Tensor& spatial_sample, const std::vector<size_t>& channels);

// Every value, row-major (a flat sample's units, a vector parameter).
std::vector<float> ReadValues(const Tensor& tensor);

// A histogram of the finite values over [low, high] (the scan's range).
struct Histogram {
    std::vector<unsigned> counts;
    float low = 0.0f;
    float high = 0.0f;
};
Histogram ReadHistogram(const Tensor& tensor, float low, float high, size_t bins);

// Rows [row_begin, row_begin + row_count) of the tensor as LayoutAsMatrix
// lays it out, row-major (row_count x columns values).
std::vector<float> ReadMatrixRows(const TensorInfo& info, const Tensor& tensor, size_t row_begin, size_t row_count);

}  // namespace cyxwiz::layer_view
