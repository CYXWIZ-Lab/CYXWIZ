#include "group_norm_arrayfire.h"

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>

namespace cyxwiz {
namespace {

struct Extents {
    dim_t h, w, c, b, groups, per_group;  // per_group = H * W * C / groups
};

Extents ExtentsOf(const std::vector<size_t>& shape, size_t groups) {
    Extents e{static_cast<dim_t>(shape[0]), static_cast<dim_t>(shape[1]), static_cast<dim_t>(shape[2]),
              static_cast<dim_t>(shape[3]), static_cast<dim_t>(groups), 0};
    e.per_group = e.h * e.w * (e.c / e.groups);
    return e;
}

// Semantic [H, W, C, B] viewed as [H*W*C/groups, groups, B]: channel c = k + (C/groups) * g.
af::array Grouped(const af::array& x, const Extents& e) { return af::moddims(x, e.per_group, e.groups, e.b); }

af::array PerChannel(const Tensor& parameter, const Extents& e) {
    return af::tile(af::moddims(parameter.GetSemanticArray(), 1, 1, e.c, 1), af::dim4(e.h, e.w, 1, e.b));
}

}  // namespace

GroupNormForward GroupNormForwardArrayFire(const Tensor& input, size_t groups, float eps,
                                           const Tensor* gamma, const Tensor* beta) {
    const Extents e = ExtentsOf(input.Shape(), groups);
    const af::array x = Grouped(input.GetSemanticArray(), e);
    const af::dim4 down(e.per_group);
    const af::array centered = x - af::tile(af::mean(x, 0), down);
    const af::array inverse = 1.0f / af::sqrt(af::mean(centered * centered, 0) + eps);
    const af::array normalized = af::moddims(centered * af::tile(inverse, down), e.h, e.w, e.c, e.b);
    af::array y = normalized;
    if (gamma != nullptr) {
        y = y * PerChannel(*gamma, e) + PerChannel(*beta, e);
    }
    y.eval();
    GroupNormForward out;
    out.output = Tensor::FromSemanticArray(y, input.Shape());
    out.normalized = Tensor::FromSemanticArray(normalized, input.Shape());
    out.std_inv = Tensor::FromSemanticArray(af::moddims(inverse, e.groups, e.b), {groups, input.Shape()[3]});
    return out;
}

GroupNormBackward GroupNormBackwardArrayFire(const Tensor& grad_output, const Tensor& normalized,
                                             const Tensor& std_inv, size_t groups, const Tensor* gamma) {
    const Extents e = ExtentsOf(grad_output.Shape(), groups);
    const af::array dy = grad_output.GetSemanticArray();
    const af::array norm = normalized.GetSemanticArray();
    GroupNormBackward out;
    af::array weighted = dy;
    if (gamma != nullptr) {
        weighted = dy * PerChannel(*gamma, e);
        const size_t channels = grad_output.Shape()[2];
        const auto per_channel_sum = [](const af::array& v) { return af::flat(af::sum(af::sum(af::sum(v, 0), 1), 3)); };
        out.grad_gamma = Tensor::FromSemanticArray(per_channel_sum(dy * norm), {channels});
        out.grad_beta = Tensor::FromSemanticArray(per_channel_sum(dy), {channels});
    }
    const af::dim4 down(e.per_group);
    const af::array w = Grouped(weighted, e);
    const af::array n = Grouped(norm, e);
    const af::array inverse = af::moddims(std_inv.GetSemanticArray(), 1, e.groups, e.b);
    const af::array dx = af::tile(inverse, down) *
                         (w - af::tile(af::mean(w, 0), down) - n * af::tile(af::mean(w * n, 0), down));
    const af::array dx4 = af::moddims(dx, e.h, e.w, e.c, e.b);
    dx4.eval();
    out.grad_input = Tensor::FromSemanticArray(dx4, grad_output.Shape());
    return out;
}

}  // namespace cyxwiz
#endif
