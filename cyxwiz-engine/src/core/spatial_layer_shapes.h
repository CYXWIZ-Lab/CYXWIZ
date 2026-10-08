#pragma once

// The one shape rule for the spatial (CNN) layers (TOFIX140 A1): what a
// layer makes of a [H,W,C] sample. The graph compiler (the As compiled
// card, the parameter count), the spatial sequential head (ModelBuilder) and
// the training ingress all read this header, so they never disagree.
// Formulas are PyTorch's (torch.nn.Conv2d / MaxPool2d / AvgPool2d /
// ConvTranspose2d): the fixtures in tests/computation_truth prove them.
// Header-only, no Tensor, no backend dependency.

#include "graph_model.h"
#include "upsampling_configuration_policy.h"

#include <cstddef>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace cyxwiz::spatial {

using Params = std::map<std::string, std::string>;

// The resolved geometry of a convolution or pooling layer.
struct Geometry {
    int channels_out = 0;    // Conv2D filters / ConvTranspose2D out_channels; 0 = keeps C
    int kernel = 1;
    int stride = 1;
    int padding = 0;
    int output_padding = 0;  // ConvTranspose2D only
    int groups = 1;          // GroupNorm only
};

inline int ParseIntParam(const Params& params, const char* key, int fallback) {
    const auto it = params.find(key);
    if (it == params.end() || it->second.empty()) return fallback;
    try {
        size_t consumed = 0;
        const int value = std::stoi(it->second, &consumed);
        if (consumed != it->second.size()) throw std::invalid_argument(key);
        return value;
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(key) + " must be a whole number, not '" + it->second + "'");
    }
}

// "same" keeps the sample size at stride 1 ((k-1)/2 each side, odd kernels);
// "valid" is no padding; a number is the padding itself.
inline int ResolvePadding(const Params& params, int kernel) {
    const auto it = params.find("padding");
    if (it == params.end() || it->second.empty() || it->second == "valid") return 0;
    if (it->second == "same") {
        if (kernel % 2 == 0)
            throw std::invalid_argument("padding 'same' needs an odd kernel size (kernel " + std::to_string(kernel) + ")");
        return (kernel - 1) / 2;
    }
    return ParseIntParam(params, "padding", 0);
}

// Layers that change or need the [H,W,C] layout.
inline bool IsSpatialLayer(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::Conv2D:
        case gui::NodeType::MaxPool2D:
        case gui::NodeType::AvgPool2D:
        case gui::NodeType::ConvTranspose2D:
        case gui::NodeType::GroupNorm:
        case gui::NodeType::InstanceNorm:
        case gui::NodeType::Upsample:
        case gui::NodeType::PixelShuffle:
            return true;
        default:
            return false;
    }
}

// Element-wise layers that pass a [H,W,C,N] tensor through unchanged.
inline bool IsShapePreservingLayer(gui::NodeType type) {
    switch (type) {
        case gui::NodeType::ReLU:
        case gui::NodeType::LeakyReLU:
        case gui::NodeType::ELU:
        case gui::NodeType::SELU:
        case gui::NodeType::GELU:
        case gui::NodeType::Swish:
        case gui::NodeType::Mish:
        case gui::NodeType::Sigmoid:
        case gui::NodeType::Tanh:
        case gui::NodeType::Dropout:
            return true;
        default:
            return false;
    }
}

// Reads the layer's geometry from its persisted parameters. Throws
// std::invalid_argument with the reason when a value cannot be used.
inline Geometry ResolveGeometry(gui::NodeType type, const Params& params, size_t channels_in) {
    Geometry g;
    switch (type) {
        case gui::NodeType::Conv2D:
            g.channels_out = ParseIntParam(params, "filters", 32);
            g.kernel = ParseIntParam(params, "kernel_size", 3);
            g.stride = ParseIntParam(params, "stride", 1);
            g.padding = ResolvePadding(params, g.kernel);
            break;
        case gui::NodeType::MaxPool2D:
        case gui::NodeType::AvgPool2D:
            g.kernel = ParseIntParam(params, "pool_size", 2);
            g.stride = ParseIntParam(params, "stride", g.kernel);
            if (g.stride <= 0) g.stride = g.kernel;
            g.padding = ResolvePadding(params, g.kernel);
            break;
        case gui::NodeType::ConvTranspose2D:
            g.channels_out = ParseIntParam(params, "out_channels", 32);
            g.kernel = ParseIntParam(params, "kernel_size", 3);
            g.stride = ParseIntParam(params, "stride", 2);
            g.padding = ParseIntParam(params, "padding", 1);
            g.output_padding = ParseIntParam(params, "output_padding", 1);
            break;
        case gui::NodeType::GroupNorm:
            g.groups = ParseIntParam(params, "num_groups", 32);
            if (g.groups <= 0 || channels_in % static_cast<size_t>(g.groups) != 0)
                throw std::invalid_argument("num_groups " + std::to_string(g.groups) + " must divide the " +
                                            std::to_string(channels_in) + " input channels");
            break;
        default:
            break;
    }
    if (g.kernel <= 0) throw std::invalid_argument("kernel size must be positive");
    if (g.stride <= 0) throw std::invalid_argument("stride must be positive");
    if (g.padding < 0) throw std::invalid_argument("padding cannot be negative");
    if (type == gui::NodeType::ConvTranspose2D && (g.output_padding < 0 || g.output_padding >= g.stride))
        throw std::invalid_argument("output_padding must be smaller than the stride");
    if (g.channels_out < 0) throw std::invalid_argument("the channel count cannot be negative");
    return g;
}

// The [H,W,C] sample a spatial or shape-preserving layer produces from
// `in` ([H,W,C]). Throws std::invalid_argument when the layer cannot take
// the input (a kernel larger than the sample, a bad parameter).
inline std::vector<size_t> SampleShapeAfter(gui::NodeType type, const Params& params,
                                            const std::vector<size_t>& in) {
    if (in.size() != 3) throw std::invalid_argument("a spatial layer needs a [H,W,C] input sample");
    if (IsShapePreservingLayer(type)) return in;
    const long h = static_cast<long>(in[0]);
    const long w = static_cast<long>(in[1]);
    switch (type) {
        case gui::NodeType::Conv2D:
        case gui::NodeType::MaxPool2D:
        case gui::NodeType::AvgPool2D: {
            const Geometry g = ResolveGeometry(type, params, in[2]);
            const long oh = (h + 2 * g.padding - g.kernel) / g.stride + 1;
            const long ow = (w + 2 * g.padding - g.kernel) / g.stride + 1;
            if (h + 2 * g.padding < g.kernel || w + 2 * g.padding < g.kernel || oh <= 0 || ow <= 0)
                throw std::invalid_argument("kernel " + std::to_string(g.kernel) + " does not fit a " +
                                            std::to_string(h) + "x" + std::to_string(w) + " sample");
            const size_t c = g.channels_out > 0 ? static_cast<size_t>(g.channels_out) : in[2];
            return {static_cast<size_t>(oh), static_cast<size_t>(ow), c};
        }
        case gui::NodeType::ConvTranspose2D: {
            const Geometry g = ResolveGeometry(type, params, in[2]);
            const long oh = (h - 1) * g.stride - 2 * g.padding + g.kernel + g.output_padding;
            const long ow = (w - 1) * g.stride - 2 * g.padding + g.kernel + g.output_padding;
            if (oh <= 0 || ow <= 0) throw std::invalid_argument("the transposed convolution produces an empty sample");
            return {static_cast<size_t>(oh), static_cast<size_t>(ow), static_cast<size_t>(g.channels_out)};
        }
        case gui::NodeType::GroupNorm:
            ResolveGeometry(type, params, in[2]);  // validates the groups
            return in;
        case gui::NodeType::InstanceNorm:
            return in;
        case gui::NodeType::Upsample:
        case gui::NodeType::PixelShuffle: {
            UpsamplingConfiguration resolved;
            if (const auto reason = ResolveUpsamplingConfiguration(type, params, resolved))
                throw std::invalid_argument(*reason);
            return InferUpsamplingSampleShape(type, resolved, in);
        }
        default:
            throw std::invalid_argument("not a spatial layer");
    }
}

}  // namespace cyxwiz::spatial
