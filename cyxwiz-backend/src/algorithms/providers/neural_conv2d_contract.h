#pragma once

// The device-resident Conv2D contract shared by every provider tenant
// (TOFIX140 A1b): request validation, geometry and batch chunking. The
// layouts are documented on NeuralOpRequest (neural_provider.h).

#include "cyxwiz/neural_provider.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>

namespace cyxwiz::conv2d_contract {

struct Geometry {
    int H = 0, W = 0, C = 0, N = 0, Cout = 0, k = 0, stride = 1, pad = 0, OH = 0, OW = 0;
    size_t K = 0, P = 0;  // k*k*C, OH*OW
    size_t InputElements() const { return static_cast<size_t>(H) * W * C * N; }
    size_t WeightElements() const { return K * Cout; }
    size_t OutputElements() const { return P * Cout * N; }
};

inline Geometry GeometryOf(const NeuralOpRequest& r) {
    Geometry g;
    g.H = static_cast<int>(r.conv_height);
    g.W = static_cast<int>(r.conv_width);
    g.C = static_cast<int>(r.input);
    g.N = static_cast<int>(r.batch);
    g.Cout = static_cast<int>(r.hidden);
    g.k = static_cast<int>(r.conv_kernel);
    g.stride = static_cast<int>(r.conv_stride);
    g.pad = static_cast<int>(r.conv_padding);
    g.OH = (g.H + 2 * g.pad - g.k) / g.stride + 1;
    g.OW = (g.W + 2 * g.pad - g.k) / g.stride + 1;
    g.K = static_cast<size_t>(g.k) * g.k * g.C;
    g.P = static_cast<size_t>(g.OH) * g.OW;
    return g;
}

// "" when the request is a valid device-resident Conv2D, else the reason.
inline std::string ContractError(const NeuralOpRequest& r) {
    if (!r.device_resident) return "conv2d runs device-resident only";
    if (r.dtype != DataType::Float32) return "conv2d needs Float32";
    if (r.batch == 0 || r.input == 0 || r.hidden == 0 || r.conv_height == 0 || r.conv_width == 0 ||
        r.conv_kernel == 0 || r.conv_stride == 0) {
        return "conv2d sizes must be positive";
    }
    const size_t limit = static_cast<size_t>(std::numeric_limits<int>::max() / 4);
    if (r.batch > limit || r.input > limit || r.hidden > limit || r.conv_height > limit ||
        r.conv_width > limit || r.conv_kernel > limit || r.conv_stride > limit || r.conv_padding > limit) {
        return "conv2d sizes exceed the int range";
    }
    if (r.conv_height + 2 * r.conv_padding < r.conv_kernel ||
        r.conv_width + 2 * r.conv_padding < r.conv_kernel) {
        return "conv2d kernel does not fit the padded input";
    }
    const size_t K = r.conv_kernel * r.conv_kernel * r.input;
    const size_t OH = (r.conv_height + 2 * r.conv_padding - r.conv_kernel) / r.conv_stride + 1;
    const size_t OW = (r.conv_width + 2 * r.conv_padding - r.conv_kernel) / r.conv_stride + 1;
    if (K > limit || OH * OW > limit) return "conv2d column matrix exceeds the int range";
    return {};
}

// Samples per pass so a column matrix stays within ~128 MB (at least one).
inline int Chunk(const Geometry& g) {
    constexpr size_t kMaxColumnFloats = size_t{32} << 20;
    const size_t per_sample = std::max<size_t>(1, g.K * g.P);
    return static_cast<int>(std::max<size_t>(1, std::min<size_t>(g.N, kMaxColumnFloats / per_sample)));
}

}  // namespace cyxwiz::conv2d_contract
