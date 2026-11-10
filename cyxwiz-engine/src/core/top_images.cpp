#include "top_images.h"

#include <arrayfire.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace cyxwiz::top_images {

namespace {

size_t Elements(const std::vector<size_t>& shape, size_t from, size_t to) {
    size_t n = 1;
    for (size_t i = from; i < to; ++i) n *= shape[i];
    return n;
}

}  // namespace

Strengths ChannelStrengths(const Tensor& layer_output, Rank rank) {
    const auto& shape = layer_output.Shape();
    const af::array flat = layer_output.GetRowMajorFlatArray().as(f32);
    Strengths s;
    af::array per;  // [C * N] in channel-major order
    if (shape.size() == 4) {
        // [H, W, C, N] row-major: ((h * W + w) * C + c) * N + n, so the map
        // positions are columns of a (C * N) by (H * W) array.
        s.channels = shape[2];
        s.items = shape[3];
        const af::array grid = af::moddims(flat, static_cast<dim_t>(s.channels * s.items),
                                           static_cast<dim_t>(shape[0] * shape[1]));
        per = rank == Rank::Maximum ? af::max(grid, 1) : af::mean(grid, 1);
    } else if (shape.size() == 2) {
        // [N, C] row-major: n * C + c; channel-major is its transpose.
        s.items = shape[0];
        s.channels = shape[1];
        per = af::flat(af::transpose(af::moddims(flat, static_cast<dim_t>(s.channels), static_cast<dim_t>(s.items))));
    } else {
        throw std::runtime_error("a layer output of rank " + std::to_string(shape.size()) +
                                 " has no channels to rank items by");
    }
    s.values.resize(s.channels * s.items);
    if (!s.values.empty()) per.host(s.values.data());
    return s;
}

std::vector<size_t> TopItems(const Strengths& strengths, size_t channel, size_t count) {
    std::vector<size_t> order(strengths.items);
    std::iota(order.begin(), order.end(), size_t{0});
    if (channel >= strengths.channels) return {};
    count = std::min(count, order.size());
    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(count), order.end(),
                      [&](size_t a, size_t b) {
                          const float va = strengths.At(channel, a), vb = strengths.At(channel, b);
                          return va > vb || (va == vb && a < b);
                      });
    order.resize(count);
    return order;
}

Tensor ItemRange(const Tensor& items, size_t begin, size_t end) {
    const auto& shape = items.Shape();
    if (shape.empty() || begin >= end) throw std::runtime_error("an empty item range");
    const bool batch_last = shape.size() == 4;
    const size_t n = batch_last ? shape.back() : shape.front();
    if (end > n) throw std::runtime_error("the item range runs past the batch");
    const size_t per_item = batch_last ? Elements(shape, 0, shape.size() - 1) : Elements(shape, 1, shape.size());
    const af::array flat = items.GetRowMajorFlatArray();
    std::vector<size_t> out_shape = shape;
    if (batch_last) {
        // Row-major index s * N + n: rows are items.
        const af::array by_item = af::moddims(flat, static_cast<dim_t>(n), static_cast<dim_t>(per_item));
        const af::array part = by_item(af::seq(static_cast<double>(begin), static_cast<double>(end - 1)), af::span);
        out_shape.back() = end - begin;
        return Tensor::FromRowMajorFlatArray(af::flat(part), out_shape);
    }
    // Row-major index n * S + s: columns are items.
    const af::array by_item = af::moddims(flat, static_cast<dim_t>(per_item), static_cast<dim_t>(n));
    const af::array part = by_item(af::span, af::seq(static_cast<double>(begin), static_cast<double>(end - 1)));
    out_shape.front() = end - begin;
    return Tensor::FromRowMajorFlatArray(af::flat(part), out_shape);
}

Strengths Run(DebugModelSession& session, const Tensor& items, size_t module_index, Rank rank, size_t chunk,
              const std::function<void(size_t, size_t)>& progress, const std::function<bool()>& cancelled) {
    const auto& shape = items.Shape();
    const size_t n = shape.empty() ? 0 : (shape.size() == 4 ? shape.back() : shape.front());
    Strengths all;
    all.items = n;
    chunk = std::max<size_t>(1, chunk);
    std::vector<Strengths> parts;
    for (size_t begin = 0; begin < n; begin += chunk) {
        if (cancelled && cancelled()) {
            all.cancelled = true;
            return all;
        }
        const size_t end = std::min(n, begin + chunk);
        parts.push_back(ChannelStrengths(session.LayerOutput(ItemRange(items, begin, end), module_index), rank));
        if (progress) progress(end, n);
    }
    if (parts.empty()) return all;
    all.channels = parts.front().channels;
    all.values.assign(all.channels * n, 0.0f);
    size_t offset = 0;
    for (const auto& p : parts) {
        for (size_t c = 0; c < all.channels; ++c)
            for (size_t i = 0; i < p.items; ++i) all.values[c * n + offset + i] = p.At(c, i);
        offset += p.items;
    }
    return all;
}

std::vector<Vote> StrongestVotes(const std::vector<float>& weights, size_t outputs, size_t channels,
                                 size_t per_output) {
    std::vector<Vote> votes;
    if (outputs == 0 || channels == 0 || weights.size() != outputs * channels) return votes;
    const auto strongest = [&](size_t output, int towards, float sign) {
        std::vector<size_t> order(channels);
        std::iota(order.begin(), order.end(), size_t{0});
        const auto w = [&](size_t c) { return sign * weights[output * channels + c]; };
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return w(a) > w(b); });
        for (size_t i = 0; i < std::min(per_output, channels); ++i)
            if (w(order[i]) > 0.0f) votes.push_back({order[i], towards, weights[output * channels + order[i]]});
    };
    if (outputs == 1) {
        strongest(0, 1, 1.0f);   // positive weights raise the logit: class 1
        strongest(0, 0, -1.0f);  // negative weights lower it: class 0
    } else {
        for (size_t o = 0; o < outputs; ++o) strongest(o, static_cast<int>(o), 1.0f);
    }
    return votes;
}

}  // namespace cyxwiz::top_images
