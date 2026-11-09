#include "saliency.h"

#include <arrayfire.h>

#include <algorithm>
#include <stdexcept>

namespace cyxwiz::saliency {

namespace {

std::vector<double> Host(const af::array& a) {
    std::vector<float> values(static_cast<size_t>(a.elements()));
    if (!values.empty()) a.as(f32).host(values.data());
    return {values.begin(), values.end()};
}

}  // namespace

std::vector<float> TargetWeights(size_t outputs, layer_view::OutputRule rule, int target) {
    using layer_view::OutputRule;
    std::vector<float> weights(outputs, 0.0f);
    if (outputs == 0) return weights;
    const bool one_score = outputs == 1 && (rule == OutputRule::Sigmoid || rule == OutputRule::Probability);
    if (one_score) {
        weights[0] = target == 0 ? -1.0f : 1.0f;
        return weights;
    }
    weights[static_cast<size_t>(std::clamp(target, 0, static_cast<int>(outputs) - 1))] = 1.0f;
    return weights;
}

Result Reduce(occlusion::InputKind kind, const Tensor& gradient, const Tensor& values) {
    const auto& shape = gradient.Shape();
    if (values.Shape() != shape) throw std::runtime_error("the gradient and the values differ in shape");
    const af::array g = gradient.GetRowMajorFlatArray();
    const af::array x = values.GetRowMajorFlatArray();
    Result result;
    if (kind == occlusion::InputKind::Image || kind == occlusion::InputKind::Tokens) {
        // Image [H, W, C, 1]: a pixel's channels are adjacent. Tokens
        // [1, T, D]: a token's dimensions are adjacent. One column each.
        if (shape.size() != (kind == occlusion::InputKind::Image ? 4u : 3u))
            throw std::runtime_error(kind == occlusion::InputKind::Image ? "an image is [H, W, C, 1]"
                                                                         : "the Embedding's output is [1, T, D]");
        const dim_t inner = static_cast<dim_t>(shape[2]);
        const dim_t columns = static_cast<dim_t>(g.elements()) / std::max<dim_t>(1, inner);
        const af::array gc = af::moddims(g, inner, columns);
        const af::array xc = af::moddims(x, inner, columns);
        if (kind == occlusion::InputKind::Image) {
            result.rows = shape[0];
            result.columns = shape[1];
            result.size = Host(af::max(af::abs(gc), 0));
        } else {
            result.size = Host(af::sqrt(af::sum(gc * gc, 0)));
        }
        result.times_input = Host(af::sum(gc * xc, 0));
        return result;
    }
    result.size = Host(af::abs(g));
    result.times_input = Host(g * x);
    return result;
}

Result Run(DebugModelSession& session, const Tensor& sample, occlusion::InputKind kind, layer_view::OutputRule rule,
           int target, size_t embedding_layer) {
    const Tensor output = session.Forward(sample);
    const auto weights = TargetWeights(output.NumElements(), rule, target);
    const Tensor target_weights = Tensor::FromRowMajorFlatArray(
        af::array(static_cast<dim_t>(weights.size()), weights.data()), output.Shape());
    if (kind == occlusion::InputKind::Tokens) {
        const auto at = session.LayerOutputGradient(sample, target_weights, embedding_layer);
        return Reduce(kind, at.gradient, at.output);
    }
    return Reduce(kind, session.InputGradient(sample, target_weights), sample);
}

}  // namespace cyxwiz::saliency
