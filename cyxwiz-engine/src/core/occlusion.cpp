#include "occlusion.h"

#include <arrayfire.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cyxwiz::occlusion {

namespace {

size_t Elements(const std::vector<size_t>& shape) {
    size_t n = shape.empty() ? 0 : 1;
    for (size_t d : shape) n *= d;
    return n;
}

std::vector<size_t> Starts(size_t length, size_t patch, size_t step) {
    std::vector<size_t> starts;
    if (patch >= length) return {0};
    for (size_t at = 0; at + patch <= length; at += std::max<size_t>(1, step)) starts.push_back(at);
    if (starts.back() + patch < length) starts.push_back(length - patch);  // flush with the far edge
    return starts;
}

std::vector<float> RowMajor(const Tensor& t) {
    const af::array flat = t.GetRowMajorFlatArray();
    std::vector<float> values(static_cast<size_t>(flat.elements()));
    if (!values.empty()) flat.as(f32).host(values.data());
    return values;
}

}  // namespace

std::optional<InputKind> KindOf(const std::vector<size_t>& sample_shape, bool first_layer_embeds) {
    if (sample_shape.size() == 4 && sample_shape[0] > 1 && sample_shape[1] > 1) return InputKind::Image;
    if (sample_shape.size() == 2 && sample_shape[0] == 1 && sample_shape[1] > 0)
        return first_layer_embeds ? InputKind::Tokens : InputKind::Features;
    return std::nullopt;
}

std::pair<size_t, size_t> PatchGrid(const Plan& plan) {
    if (plan.kind != InputKind::Image || plan.sample_shape.size() != 4) return {0, 0};
    return {Starts(plan.sample_shape[0], plan.patch, plan.step).size(),
            Starts(plan.sample_shape[1], plan.patch, plan.step).size()};
}

std::vector<Piece> PiecesOf(const Plan& plan, const std::vector<float>& token_ids, float pad_id) {
    std::vector<Piece> pieces;
    if (plan.kind == InputKind::Image) {
        if (plan.sample_shape.size() != 4) return pieces;
        const size_t h = plan.sample_shape[0], w = plan.sample_shape[1];
        const size_t ph = std::min(plan.patch, h), pw = std::min(plan.patch, w);
        for (size_t r : Starts(h, plan.patch, plan.step))
            for (size_t c : Starts(w, plan.patch, plan.step)) pieces.push_back({r, c, ph, pw, 0});
        return pieces;
    }
    const size_t n = plan.sample_shape.size() == 2 ? plan.sample_shape[1] : 0;
    for (size_t p = 0; p < n; ++p) {
        if (plan.kind == InputKind::Tokens && p < token_ids.size() && token_ids[p] == pad_id) continue;
        Piece piece;
        piece.position = p;
        pieces.push_back(piece);
    }
    return pieces;
}

double TargetScore(const std::vector<float>& out, layer_view::OutputRule rule, int target) {
    using layer_view::OutputRule;
    if (out.empty()) return 0.0;
    const size_t t = static_cast<size_t>(std::max(0, target));
    switch (rule) {
        case OutputRule::Sigmoid: {
            const double p = 1.0 / (1.0 + std::exp(-static_cast<double>(out[0])));
            return out.size() == 1 ? (t == 0 ? 1.0 - p : p) : 1.0 / (1.0 + std::exp(-static_cast<double>(out[std::min(t, out.size() - 1)])));
        }
        case OutputRule::Softmax: {
            const float top = *std::max_element(out.begin(), out.end());
            double sum = 0.0;
            for (float v : out) sum += std::exp(static_cast<double>(v - top));
            return std::exp(static_cast<double>(out[std::min(t, out.size() - 1)] - top)) / sum;
        }
        case OutputRule::Probability:
            return out.size() == 1 ? (t == 0 ? 1.0 - out[0] : out[0]) : out[std::min(t, out.size() - 1)];
        default:
            return out[std::min(t, out.size() - 1)];
    }
}

Tensor CoveredBatch(const Tensor& sample, const Plan& plan, const std::vector<Piece>& pieces) {
    const size_t s = Elements(plan.sample_shape);
    const size_t k = pieces.size();
    if (s == 0 || k == 0 || sample.NumElements() != s)
        throw std::runtime_error("the sample does not have the planned shape");
    const af::array x = af::tile(af::moddims(sample.GetRowMajorFlatArray().as(f32), static_cast<dim_t>(s), 1), 1,
                                 static_cast<unsigned>(k));
    const af::array index = af::range(af::dim4(static_cast<dim_t>(s), static_cast<dim_t>(k)), 0, f32);
    af::array mask;
    af::array fill;
    if (plan.kind == InputKind::Image) {
        const float w = static_cast<float>(plan.sample_shape[1]);
        const float c = static_cast<float>(plan.sample_shape[2]);
        const af::array row = af::floor(index / (w * c));
        const af::array column = af::floor(index / c) - w * row;
        const af::array channel = index - c * af::floor(index / c);
        std::vector<float> r0(k), c0(k), r1(k), c1(k);
        for (size_t i = 0; i < k; ++i) {
            r0[i] = static_cast<float>(pieces[i].row);
            c0[i] = static_cast<float>(pieces[i].column);
            r1[i] = static_cast<float>(pieces[i].row + pieces[i].rows);
            c1[i] = static_cast<float>(pieces[i].column + pieces[i].columns);
        }
        const auto per_piece = [&](const std::vector<float>& v) {
            return af::tile(af::array(1, static_cast<dim_t>(k), v.data()), static_cast<unsigned>(s), 1);
        };
        mask = (row >= per_piece(r0)) && (row < per_piece(r1)) && (column >= per_piece(c0)) && (column < per_piece(c1));
        std::vector<float> channel_fill(plan.sample_shape[2], 0.0f);
        for (size_t i = 0; i < channel_fill.size() && i < plan.fill.size(); ++i) channel_fill[i] = plan.fill[i];
        fill = af::lookup(af::array(static_cast<dim_t>(channel_fill.size()), channel_fill.data()),
                          af::flat(channel).as(s32));
        fill = af::moddims(fill, static_cast<dim_t>(s), static_cast<dim_t>(k));
    } else {
        std::vector<float> at(k);
        for (size_t i = 0; i < k; ++i) at[i] = static_cast<float>(pieces[i].position);
        mask = index == af::tile(af::array(1, static_cast<dim_t>(k), at.data()), static_cast<unsigned>(s), 1);
        std::vector<float> per_element(s, plan.fill.empty() ? 0.0f : plan.fill[0]);
        if (plan.kind == InputKind::Features)
            for (size_t i = 0; i < s && i < plan.fill.size(); ++i) per_element[i] = plan.fill[i];
        fill = af::tile(af::array(static_cast<dim_t>(s), per_element.data()), 1, static_cast<unsigned>(k));
    }
    const af::array m = mask.as(f32);
    const af::array covered = x * (1.0f - m) + fill * m;  // [S, K], one item per column

    std::vector<size_t> shape = plan.sample_shape;
    if (plan.kind == InputKind::Image) {
        shape.back() = k;  // [H, W, C, K]: row-major index s * K + k
        return Tensor::FromRowMajorFlatArray(af::flat(af::transpose(covered)), shape);
    }
    shape.front() = k;  // [K, T]: row-major index k * S + s
    return Tensor::FromRowMajorFlatArray(af::flat(covered), shape);
}

Result Run(DebugModelSession& session, const Tensor& sample, const Plan& plan, const std::vector<Piece>& pieces,
           layer_view::OutputRule rule, int target, size_t chunk, const std::function<void(size_t, size_t)>& progress,
           const std::function<bool()>& cancelled) {
    Result result;
    result.baseline = TargetScore(RowMajor(session.Forward(sample)), rule, target);
    result.change.reserve(pieces.size());
    chunk = std::max<size_t>(1, chunk);
    for (size_t begin = 0; begin < pieces.size(); begin += chunk) {
        if (cancelled && cancelled()) {
            result.cancelled = true;
            return result;
        }
        const std::vector<Piece> part(pieces.begin() + static_cast<std::ptrdiff_t>(begin),
                                      pieces.begin() + static_cast<std::ptrdiff_t>(std::min(pieces.size(), begin + chunk)));
        const auto out = RowMajor(session.Forward(CoveredBatch(sample, plan, part)));
        const size_t per_item = out.size() / part.size();
        for (size_t i = 0; i < part.size(); ++i) {
            const std::vector<float> item(out.begin() + static_cast<std::ptrdiff_t>(i * per_item),
                                          out.begin() + static_cast<std::ptrdiff_t>((i + 1) * per_item));
            result.change.push_back(TargetScore(item, rule, target) - result.baseline);
        }
        if (progress) progress(result.change.size(), pieces.size());
    }
    return result;
}

}  // namespace cyxwiz::occlusion
