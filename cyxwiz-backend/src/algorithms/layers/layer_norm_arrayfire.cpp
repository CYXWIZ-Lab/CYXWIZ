#include "cyxwiz/layers/normalization.h"
#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>

namespace cyxwiz {
Tensor LayerNormLayer::ForwardArrayFire(const Tensor& input, size_t norm_size) {
    const size_t rows = input.NumElements() / norm_size;
    const af::array x = input.Reshape({rows, norm_size}).GetSemanticArray();
    const af::dim4 across(1, static_cast<dim_t>(norm_size));
    const af::dim4 down(static_cast<dim_t>(rows));
    const af::array centered = x - af::tile(af::mean(x, 1), across);
    const af::array inverse = 1.0f / af::sqrt(af::mean(centered * centered, 1) + eps_);
    const af::array normalized = centered * af::tile(inverse, across);
    af::array y = normalized;
    if (elementwise_affine_) {
        y = y * af::tile(af::moddims(gamma_.GetSemanticArray(), 1, static_cast<dim_t>(norm_size)), down)
              + af::tile(af::moddims(beta_.GetSemanticArray(), 1, static_cast<dim_t>(norm_size)), down);
    }
    // Publish only complete forward caches, with the original row-major shapes.
    y.eval();
    Tensor result = Tensor::FromSemanticArray(y, {rows, norm_size}).Reshape(input.Shape());
    normalized_ = Tensor::FromSemanticArray(normalized, {rows, norm_size}).Reshape(input.Shape());
    std_inv_ = Tensor::FromSemanticArray(af::flat(inverse), {rows});
    cached_input_ = input;
    return result;
}

Tensor LayerNormLayer::BackwardArrayFire(const Tensor& gradient, size_t norm_size) {
    const size_t rows = gradient.NumElements() / norm_size;
    const af::dim4 across(1, static_cast<dim_t>(norm_size));
    const af::array dy = gradient.Reshape({rows, norm_size}).GetSemanticArray();
    const af::array normalized = normalized_.Reshape({rows, norm_size}).GetSemanticArray();
    af::array weighted = dy;
    Tensor gamma_gradient, beta_gradient;
    if (elementwise_affine_) {
        weighted = dy * af::tile(af::moddims(gamma_.GetSemanticArray(), 1,
                                            static_cast<dim_t>(norm_size)), af::dim4(static_cast<dim_t>(rows)));
        gamma_gradient = Tensor::FromSemanticArray(af::flat(af::sum(dy * normalized, 0)), {norm_size});
        beta_gradient = Tensor::FromSemanticArray(af::flat(af::sum(dy, 0)), {norm_size});
    }
    const af::array dx = af::tile(std_inv_.GetSemanticArray(), across) *
        (weighted - af::tile(af::mean(weighted, 1), across)
                  - normalized * af::tile(af::mean(weighted * normalized, 1), across));
    dx.eval(); // Finish lazy input-gradient arithmetic before publishing state.
    Tensor result = Tensor::FromSemanticArray(dx, {rows, norm_size}).Reshape(gradient.Shape());
    if (elementwise_affine_) {
        grad_gamma_ = std::move(gamma_gradient);
        grad_beta_ = std::move(beta_gradient);
    }
    return result;
}
} // namespace cyxwiz
#endif
