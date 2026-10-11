#include <cyxwiz/sequential.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {
// ============================================================================
// BatchNormModule Implementation (BatchNorm1D for MLPs)
// ============================================================================

BatchNormModule::BatchNormModule(size_t num_features, float eps, float momentum)
    : num_features_(num_features)
    , eps_(eps)
    , momentum_(momentum)
{
    if (num_features_ == 0) {
        throw std::invalid_argument(
            "BatchNormModule: num_features must be positive");
    }
    if (!std::isfinite(eps_) || eps_ <= 0.0f) {
        throw std::invalid_argument(
            "BatchNormModule: eps must be a finite number > 0");
    }
    if (!std::isfinite(momentum_) || momentum_ < 0.0f || momentum_ > 1.0f) {
        throw std::invalid_argument(
            "BatchNormModule: momentum must be a finite number in [0, 1]");
    }

    gamma_ = Tensor::Ones({num_features});
    beta_ = Tensor::Zeros({num_features});
    running_mean_ = Tensor::Zeros({num_features});
    running_var_ = Tensor::Ones({num_features});
    grad_gamma_ = Tensor::Zeros({num_features});
    grad_beta_ = Tensor::Zeros({num_features});

    spdlog::debug("BatchNormModule({}) initialized", num_features);
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
namespace {

// A [features] vector as a [1, features] row of the semantic [batch, features] view.
af::array FeatureRow(const Tensor& vector) {
    const af::array values = vector.GetSemanticArray();
    return af::moddims(values, 1, values.elements());
}

}  // namespace
#endif

// Per-feature statistics over the batch, on the semantic (row-major)
// [batch, features] view, matching PyTorch BatchNorm1d.
Tensor BatchNormModule::Forward(const Tensor& input) {
    const auto& shape = input.Shape();
    if (input.GetDataType() != DataType::Float32 || shape.size() != 2 ||
        shape[0] == 0 || shape[1] != num_features_ ||
        shape[0] > (std::numeric_limits<unsigned>::max)() ||
        shape[1] > (std::numeric_limits<unsigned>::max)()) {
        throw std::invalid_argument(
            "BatchNormModule: input must be non-empty Float32 [batch, num_features]");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const size_t batch_size = shape[0];
        const af::array x = input.GetSemanticArray();
        const af::dim4 down(static_cast<dim_t>(batch_size));
        af::array mean = FeatureRow(running_mean_);
        af::array var = FeatureRow(running_var_);
        if (is_training_) {
            mean = af::mean(x, 0);
            const af::array centered = x - af::tile(mean, down);
            var = af::mean(centered * centered, 0);
        }
        const af::array inverse = 1.0f / af::sqrt(var + eps_);
        const af::array normalized = (x - af::tile(mean, down)) * af::tile(inverse, down);
        const af::array y = normalized * af::tile(FeatureRow(gamma_), down) + af::tile(FeatureRow(beta_), down);
        af::eval(y, normalized);
        // Publish state only once every device result exists.
        if (is_training_) {
            // PyTorch keeps the unbiased batch variance in the running estimate
            // (normalisation itself uses the biased one).
            const float unbiased = batch_size > 1
                ? static_cast<float>(batch_size) / static_cast<float>(batch_size - 1) : 1.0f;
            const std::vector<size_t> feature_shape{num_features_};
            running_mean_ = Tensor::FromSemanticArray(
                af::flat((1.0f - momentum_) * FeatureRow(running_mean_) + momentum_ * mean), feature_shape);
            running_var_ = Tensor::FromSemanticArray(
                af::flat((1.0f - momentum_) * FeatureRow(running_var_) + momentum_ * unbiased * var), feature_shape);
            std_inv_ = Tensor::FromSemanticArray(af::flat(inverse), feature_shape);
            normalized_ = Tensor::FromSemanticArray(normalized, shape);
            input_cache_ = input;
        }
        return Tensor::FromSemanticArray(y, shape);
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("BatchNormModule::Forward failed on the ArrayFire device: ") + e.what());
    }
#else
    throw std::runtime_error("BatchNorm runs on ArrayFire, and this build has no ArrayFire");
#endif
}

Tensor BatchNormModule::Backward(const Tensor& grad_output) {
    const auto& shape = grad_output.Shape();
    if (!is_training_ || input_cache_.Shape().size() != 2 ||
        grad_output.GetDataType() != DataType::Float32 ||
        shape != input_cache_.Shape() || shape[1] != num_features_) {
        throw std::invalid_argument(
            "BatchNormModule: backward requires a matching training-mode forward pass");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::array dy = grad_output.GetSemanticArray();
        const af::array normalized = normalized_.GetSemanticArray();
        const af::dim4 down(static_cast<dim_t>(shape[0]));
        const float count = static_cast<float>(shape[0]);
        const af::array weighted = dy * af::tile(FeatureRow(gamma_), down);
        const af::array dx = af::tile(FeatureRow(std_inv_), down) / count *
                             (count * weighted - af::tile(af::sum(weighted, 0), down) -
                              normalized * af::tile(af::sum(weighted * normalized, 0), down));
        const af::array grad_gamma = af::flat(af::sum(dy * normalized, 0));
        const af::array grad_beta = af::flat(af::sum(dy, 0));
        dx.eval();
        af::eval(grad_gamma, grad_beta);
        const std::vector<size_t> feature_shape{num_features_};
        grad_gamma_ = Tensor::FromSemanticArray(grad_gamma, feature_shape);
        grad_beta_ = Tensor::FromSemanticArray(grad_beta, feature_shape);
        return Tensor::FromSemanticArray(dx, shape);
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("BatchNormModule::Backward failed on the ArrayFire device: ") + e.what());
    }
#else
    throw std::runtime_error("BatchNorm runs on ArrayFire, and this build has no ArrayFire");
#endif
}

std::map<std::string, Tensor> BatchNormModule::GetParameters() {
    return {
        {"gamma", gamma_},
        {"beta", beta_},
        {"running_mean", running_mean_},
        {"running_var", running_var_}
    };
}

void BatchNormModule::SetParameters(const std::map<std::string, Tensor>& params) {
    if (params.count("gamma")) gamma_ = params.at("gamma");
    if (params.count("beta")) beta_ = params.at("beta");
    if (params.count("running_mean")) running_mean_ = params.at("running_mean");
    if (params.count("running_var")) running_var_ = params.at("running_var");
}

std::map<std::string, Tensor> BatchNormModule::GetGradients() {
    return {
        {"gamma", grad_gamma_},
        {"beta", grad_beta_}
    };
}

std::string BatchNormModule::GetName() const {
    return "BatchNorm(" + std::to_string(num_features_) + ")";
}

// ============================================================================
// LayerNormModule Implementation
// ============================================================================

LayerNormModule::LayerNormModule(const std::vector<int>& normalized_shape,
                                 float eps,
                                 bool elementwise_affine)
    : layer_(std::make_unique<LayerNormLayer>(normalized_shape,
                                              eps,
                                              elementwise_affine))
    , normalized_shape_(normalized_shape)
    , eps_(eps)
    , elementwise_affine_(elementwise_affine) {
    spdlog::debug("LayerNormModule initialized");
}

Tensor LayerNormModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return layer_->Forward(input);
}

Tensor LayerNormModule::Backward(const Tensor& grad_output) {
    return layer_->Backward(grad_output);
}

std::map<std::string, Tensor> LayerNormModule::GetParameters() {
    if (!elementwise_affine_) {
        return {};
    }
    auto params = layer_->GetParameters();
    params.erase("grad_gamma");
    params.erase("grad_beta");
    return params;
}

void LayerNormModule::SetParameters(
    const std::map<std::string, Tensor>& params) {
    layer_->SetParameters(params);
}

std::map<std::string, Tensor> LayerNormModule::GetGradients() {
    if (!elementwise_affine_) {
        return {};
    }
    auto params = layer_->GetParameters();
    std::map<std::string, Tensor> grads;
    if (params.count("grad_gamma")) {
        grads["gamma"] = params.at("grad_gamma");
    }
    if (params.count("grad_beta")) {
        grads["beta"] = params.at("grad_beta");
    }
    return grads;
}

std::string LayerNormModule::GetName() const {
    std::string shape = "[";
    for (size_t i = 0; i < normalized_shape_.size(); ++i) {
        if (i > 0) {
            shape += ",";
        }
        shape += std::to_string(normalized_shape_[i]);
    }
    shape += "]";
    return "LayerNorm(" + shape + ")";
}

} // namespace cyxwiz
