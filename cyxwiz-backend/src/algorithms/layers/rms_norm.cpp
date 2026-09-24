// RMSNorm (Zhang and Sennrich, arXiv:1910.07467), matching torch.nn.RMSNorm:
//   inv_rms = 1 / sqrt(mean(x^2) + eps),  n = x * inv_rms,  y = n * gamma
// Backward (g = dy * gamma):
//   dx     = inv_rms * (g - n * mean(g * n))
//   dgamma = sum over rows of dy * n
#include "cyxwiz/layers/normalization.h"
#include "layer_arrayfire_utils.h"
#include "layer_utils.h"
#include "../arrayfire_backend_utils.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace cyxwiz {

RMSNormLayer::RMSNormLayer(int normalized_size, float eps, bool elementwise_affine)
    : normalized_size_(0), eps_(eps), elementwise_affine_(elementwise_affine) {
    if (normalized_size <= 0 || !std::isfinite(eps_) || eps_ <= 0.0f) {
        throw std::invalid_argument("RMSNorm requires a positive normalized size and positive eps");
    }
    normalized_size_ = static_cast<size_t>(normalized_size);
    if (elementwise_affine_) {
        gamma_ = Tensor::Ones({normalized_size_});
        grad_gamma_ = Tensor::Zeros({normalized_size_});
    }
}

Tensor RMSNormLayer::Forward(const Tensor& input) {
    if (input.GetDataType() != DataType::Float32) {
        throw std::runtime_error("RMSNorm forward requires Float32 input");
    }
    const auto& shape = input.Shape();
    if (shape.empty() || shape.back() != normalized_size_ || input.NumElements() == 0) {
        throw std::runtime_error("RMSNorm forward expects the last dimension to equal the normalized size");
    }
    if (elementwise_affine_ &&
        (gamma_.GetDataType() != DataType::Float32 || gamma_.Shape() != std::vector<size_t>{normalized_size_})) {
        throw std::runtime_error("RMSNorm forward gamma shape mismatch");
    }
    const size_t rows = input.NumElements() / normalized_size_;

    BackendFallbackReason fallback_reason = BackendFallbackReason::UnsupportedOperation;
    std::string fallback_detail = "ArrayFire is not compiled into this backend";
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        return ForwardArrayFire(input, rows);
    } catch (const af::exception& e) {
        fallback_reason = ClassifyArrayFireBackendFallbackReason(e.what());
        fallback_detail = e.what();
    }
#endif
    ThrowIfArrayFireNativeCpuFallbackForbidden(
        "RMSNormLayer::Forward", fallback_reason, fallback_detail.c_str(),
        BuildArrayFireBackendFallbackContext(BuildTensorShapeContext("input", input.Shape())));
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath, "RMSNormLayer::Forward");

    cached_input_ = input;
    Tensor output(shape, DataType::Float32);
    normalized_ = Tensor(shape, DataType::Float32);
    inv_rms_ = Tensor({rows}, DataType::Float32);
    const float* x = input.ReadData<float>();
    const float* gamma = elementwise_affine_ ? gamma_.ReadData<float>() : nullptr;
    float* y = output.MutableData<float>();
    float* n = normalized_.MutableData<float>();
    float* inv = inv_rms_.MutableData<float>();
    const size_t d = normalized_size_;
    for (size_t row = 0; row < rows; ++row) {
        const size_t offset = row * d;
        double mean_square = 0.0;
        for (size_t i = 0; i < d; ++i) mean_square += static_cast<double>(x[offset + i]) * x[offset + i];
        mean_square /= static_cast<double>(d);
        inv[row] = static_cast<float>(1.0 / std::sqrt(mean_square + eps_));
        for (size_t i = 0; i < d; ++i) {
            n[offset + i] = x[offset + i] * inv[row];
            y[offset + i] = elementwise_affine_ ? n[offset + i] * gamma[i] : n[offset + i];
        }
    }
    return output;
}

Tensor RMSNormLayer::Backward(const Tensor& grad_output) {
    if (grad_output.GetDataType() != DataType::Float32 ||
        grad_output.Shape() != cached_input_.Shape() || normalized_.Shape() != cached_input_.Shape()) {
        throw std::runtime_error("RMSNorm backward requires a Float32 gradient matching the forward input");
    }
    const size_t rows = grad_output.NumElements() / normalized_size_;
    if (inv_rms_.Shape() != std::vector<size_t>{rows}) {
        throw std::runtime_error("RMSNorm backward cache shape mismatch");
    }

    BackendFallbackReason fallback_reason = BackendFallbackReason::UnsupportedOperation;
    std::string fallback_detail = "ArrayFire is not compiled into this backend";
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        return BackwardArrayFire(grad_output, rows);
    } catch (const af::exception& e) {
        fallback_reason = ClassifyArrayFireBackendFallbackReason(e.what());
        fallback_detail = e.what();
    }
#endif
    ThrowIfArrayFireNativeCpuFallbackForbidden(
        "RMSNormLayer::Backward", fallback_reason, fallback_detail.c_str(),
        BuildArrayFireBackendFallbackContext(BuildTensorShapeContext("grad_output", grad_output.Shape())));
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath, "RMSNormLayer::Backward");

    const size_t d = normalized_size_;
    Tensor grad_input(grad_output.Shape(), DataType::Float32);
    const float* dy = grad_output.ReadData<float>();
    const float* n = normalized_.ReadData<float>();
    const float* inv = inv_rms_.ReadData<float>();
    const float* gamma = elementwise_affine_ ? gamma_.ReadData<float>() : nullptr;
    float* dx = grad_input.MutableData<float>();
    float* dgamma = nullptr;
    if (elementwise_affine_) {
        grad_gamma_ = Tensor::Zeros({d});
        dgamma = grad_gamma_.MutableData<float>();
    }
    for (size_t row = 0; row < rows; ++row) {
        const size_t offset = row * d;
        double mean_gn = 0.0;
        for (size_t i = 0; i < d; ++i) {
            const float g = elementwise_affine_ ? dy[offset + i] * gamma[i] : dy[offset + i];
            mean_gn += static_cast<double>(g) * n[offset + i];
            if (dgamma) dgamma[i] += dy[offset + i] * n[offset + i];
        }
        mean_gn /= static_cast<double>(d);
        for (size_t i = 0; i < d; ++i) {
            const float g = elementwise_affine_ ? dy[offset + i] * gamma[i] : dy[offset + i];
            dx[offset + i] = inv[row] * (g - n[offset + i] * static_cast<float>(mean_gn));
        }
    }
    return grad_input;
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
Tensor RMSNormLayer::ForwardArrayFire(const Tensor& input, size_t rows) {
    const size_t d = normalized_size_;
    const af::array x = input.Reshape({rows, d}).GetSemanticArray();
    const af::dim4 across(1, static_cast<dim_t>(d));
    const af::array inverse = 1.0f / af::sqrt(af::mean(x * x, 1) + eps_);
    const af::array normalized = x * af::tile(inverse, across);
    af::array y = normalized;
    if (elementwise_affine_) {
        y = y * af::tile(af::moddims(gamma_.GetSemanticArray(), 1, static_cast<dim_t>(d)),
                         af::dim4(static_cast<dim_t>(rows)));
    }
    y.eval();  // publish only complete forward caches
    Tensor result = Tensor::FromSemanticArray(y, {rows, d}).Reshape(input.Shape());
    normalized_ = Tensor::FromSemanticArray(normalized, {rows, d}).Reshape(input.Shape());
    inv_rms_ = Tensor::FromSemanticArray(af::flat(inverse), {rows});
    cached_input_ = input;
    return result;
}

Tensor RMSNormLayer::BackwardArrayFire(const Tensor& gradient, size_t rows) {
    const size_t d = normalized_size_;
    const af::dim4 across(1, static_cast<dim_t>(d));
    const af::array dy = gradient.Reshape({rows, d}).GetSemanticArray();
    const af::array normalized = normalized_.Reshape({rows, d}).GetSemanticArray();
    af::array g = dy;
    Tensor gamma_gradient;
    if (elementwise_affine_) {
        g = dy * af::tile(af::moddims(gamma_.GetSemanticArray(), 1, static_cast<dim_t>(d)),
                          af::dim4(static_cast<dim_t>(rows)));
        gamma_gradient = Tensor::FromSemanticArray(af::flat(af::sum(dy * normalized, 0)), {d});
    }
    const af::array dx = af::tile(inv_rms_.GetSemanticArray(), across) *
        (g - normalized * af::tile(af::mean(g * normalized, 1), across));
    dx.eval();
    Tensor result = Tensor::FromSemanticArray(dx, {rows, d}).Reshape(gradient.Shape());
    if (elementwise_affine_) grad_gamma_ = std::move(gamma_gradient);
    return result;
}
#endif

std::map<std::string, Tensor> RMSNormLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    if (elementwise_affine_) {
        params["gamma"] = gamma_;
        params["grad_gamma"] = grad_gamma_;
    }
    return params;
}

void RMSNormLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    if (!elementwise_affine_) return;
    const auto it = params.find("gamma");
    if (it == params.end()) return;
    if (it->second.Shape() != std::vector<size_t>{normalized_size_}) {
        throw std::invalid_argument("RMSNorm gamma shape mismatch");
    }
    gamma_ = it->second;
}

}  // namespace cyxwiz
