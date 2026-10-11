#include "cyxwiz/layers/normalization.h"
#include "layer_arrayfire_utils.h"
#include "layer_utils.h"
#include "group_norm_arrayfire.h"
#include "../arrayfire_host_values.h"
#include <limits>

#include <cmath>
#include <stdexcept>
#include <string>

#include <spdlog/spdlog.h>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace cyxwiz {

LayerNormLayer::LayerNormLayer(const std::vector<int>& normalized_shape,
                               float eps, bool elementwise_affine)
    : normalized_shape_(normalized_shape), eps_(eps),
      elementwise_affine_(elementwise_affine) {
    if (normalized_shape_.empty() || !std::isfinite(eps_) || eps_ <= 0.0f) {
        throw std::invalid_argument("LayerNorm requires a non-empty normalized shape and positive eps");
    }

    // Calculate total size of normalized dimensions
    size_t norm_size = 1;
    for (int dim : normalized_shape) {
        if (dim <= 0) {
            throw std::invalid_argument("LayerNorm normalized dimensions must be positive");
        }
        if (norm_size > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
            throw std::overflow_error("LayerNorm normalized shape overflow");
        }
        norm_size *= static_cast<size_t>(dim);
    }

    if (elementwise_affine) {
        gamma_ = Tensor::Ones({norm_size});
        beta_ = Tensor::Zeros({norm_size});
        grad_gamma_ = Tensor::Zeros({norm_size});
        grad_beta_ = Tensor::Zeros({norm_size});
    }
}

Tensor LayerNormLayer::Forward(const Tensor& input) {
    if (input.GetDataType() != DataType::Float32) {
        throw std::runtime_error("LayerNorm forward requires Float32 input");
    }
    const std::vector<size_t>& shape = input.Shape();
    if (shape.size() < normalized_shape_.size()) {
        throw std::runtime_error("LayerNorm forward normalized shape rank exceeds input rank");
    }

    size_t norm_size = 1;
    const size_t suffix_start = shape.size() - normalized_shape_.size();
    for (size_t i = 0; i < normalized_shape_.size(); ++i) {
        const size_t expected = static_cast<size_t>(normalized_shape_[i]);
        if (shape[suffix_start + i] != expected) {
            throw std::runtime_error("LayerNorm forward normalized shape mismatch");
        }
        norm_size *= expected;
    }
    if (input.NumElements() == 0 || norm_size == 0 || input.NumElements() % norm_size != 0) {
        throw std::runtime_error("LayerNorm forward invalid input shape");
    }
    if (elementwise_affine_) {
        const std::vector<size_t> param_shape{norm_size};
        if (gamma_.GetDataType() != DataType::Float32 || beta_.GetDataType() != DataType::Float32 ||
            gamma_.Shape() != param_shape || beta_.Shape() != param_shape) {
            throw std::runtime_error("LayerNorm forward affine parameter mismatch");
        }
    }

    // Any rank: the device path works on [rows, norm_size].
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        return ForwardArrayFire(input, norm_size);
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("LayerNormLayer::Forward failed on the ArrayFire device: ") + e.what());
    }
#else
    (void)norm_size;
    throw std::runtime_error("LayerNorm runs on ArrayFire, and this build has no ArrayFire");
#endif
}

Tensor LayerNormLayer::Backward(const Tensor& grad_output) {
    if (grad_output.GetDataType() != DataType::Float32 ||
        normalized_.GetDataType() != DataType::Float32 ||
        std_inv_.GetDataType() != DataType::Float32) {
        throw std::runtime_error("LayerNorm backward requires Float32 tensors");
    }
    if (grad_output.Shape() != cached_input_.Shape() || normalized_.Shape() != cached_input_.Shape()) {
        throw std::runtime_error("LayerNorm backward gradient/cache shape mismatch");
    }

    size_t norm_size = 1;
    for (int dim : normalized_shape_) {
        norm_size *= static_cast<size_t>(dim);
    }
    if (norm_size == 0 || grad_output.NumElements() % norm_size != 0) {
        throw std::runtime_error("LayerNorm backward invalid normalization size");
    }
    const size_t batch_size = grad_output.NumElements() / norm_size;
    if (std_inv_.Shape() != std::vector<size_t>{batch_size}) {
        throw std::runtime_error("LayerNorm backward std_inv cache shape mismatch");
    }

    if (elementwise_affine_ && (gamma_.GetDataType() != DataType::Float32 ||
        gamma_.Shape() != std::vector<size_t>{norm_size})) {
        throw std::runtime_error("LayerNorm backward gamma shape mismatch");
    }
    // Any rank: the device path works on [rows, norm_size].
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        return BackwardArrayFire(grad_output, norm_size);
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("LayerNormLayer::Backward failed on the ArrayFire device: ") + e.what());
    }
#else
    (void)norm_size;
    throw std::runtime_error("LayerNorm runs on ArrayFire, and this build has no ArrayFire");
#endif
}

std::map<std::string, Tensor> LayerNormLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    if (elementwise_affine_) {
        params["gamma"] = gamma_;
        params["beta"] = beta_;
        params["grad_gamma"] = grad_gamma_;
        params["grad_beta"] = grad_beta_;
    }
    return params;
}

void LayerNormLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    if (params.count("gamma")) gamma_ = params.at("gamma");
    if (params.count("beta")) beta_ = params.at("beta");
}

// ============================================================================
// InstanceNorm2D Layer Implementation
// ============================================================================

InstanceNorm2DLayer::InstanceNorm2DLayer(int num_features, float eps, bool affine)
    : num_features_(num_features), eps_(eps), affine_(affine) {
    if (num_features_ <= 0 || eps_ <= 0.0f) {
        throw std::invalid_argument("InstanceNorm2D requires positive features and eps");
    }

    if (affine) {
        gamma_ = Tensor::Ones({static_cast<size_t>(num_features)});
        beta_ = Tensor::Zeros({static_cast<size_t>(num_features)});
        grad_gamma_ = Tensor::Zeros({static_cast<size_t>(num_features)});
        grad_beta_ = Tensor::Zeros({static_cast<size_t>(num_features)});
    }
}

Tensor InstanceNorm2DLayer::Forward(const Tensor& input) {
    cached_input_ = input;

    ValidateSpatial4DInput(input, "InstanceNorm2D");
    if (input.GetDataType() != DataType::Float32) {
        throw std::runtime_error("InstanceNorm2D forward requires Float32 input");
    }

    const size_t channels = input.Shape()[2];
    if (channels != static_cast<size_t>(num_features_)) {
        throw std::runtime_error("InstanceNorm2D forward channel mismatch");
    }
    const std::vector<size_t> channel_shape{channels};
    if (affine_ &&
        (gamma_.GetDataType() != DataType::Float32 || beta_.GetDataType() != DataType::Float32 ||
         gamma_.Shape() != channel_shape || beta_.Shape() != channel_shape)) {
        throw std::runtime_error("InstanceNorm2D forward affine parameter mismatch");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        auto result = GroupNormForwardArrayFire(input, channels, eps_,
                                                affine_ ? &gamma_ : nullptr, affine_ ? &beta_ : nullptr);
        normalized_ = std::move(result.normalized);
        std_inv_ = std::move(result.std_inv);
        return std::move(result.output);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("InstanceNorm2DLayer::Forward", e));
    }
#else
    throw std::runtime_error("InstanceNorm2D runs on ArrayFire, and this build has no ArrayFire");
#endif
}

Tensor InstanceNorm2DLayer::Backward(const Tensor& grad_output) {
    ValidateSpatial4DInput(cached_input_, "InstanceNorm2D");
    if (grad_output.GetDataType() != DataType::Float32 ||
        normalized_.GetDataType() != DataType::Float32 ||
        std_inv_.GetDataType() != DataType::Float32) {
        throw std::runtime_error("InstanceNorm2D backward requires Float32 tensors");
    }
    if (grad_output.Shape() != cached_input_.Shape() || normalized_.Shape() != cached_input_.Shape()) {
        throw std::runtime_error("InstanceNorm2D backward gradient/cache shape mismatch");
    }

    const size_t channels = cached_input_.Shape()[2];
    const size_t batch_size = cached_input_.Shape()[3];
    const std::vector<size_t> channel_shape{channels};
    const std::vector<size_t> instance_shape{channels, batch_size};
    if (channels != static_cast<size_t>(num_features_) || std_inv_.Shape() != instance_shape) {
        throw std::runtime_error("InstanceNorm2D backward parameter/cache shape mismatch");
    }
    if (affine_ && (gamma_.GetDataType() != DataType::Float32 || gamma_.Shape() != channel_shape)) {
        throw std::runtime_error("InstanceNorm2D backward gamma shape mismatch");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        auto result = GroupNormBackwardArrayFire(grad_output, normalized_, std_inv_, channels,
                                                 affine_ ? &gamma_ : nullptr);
        if (affine_) {
            grad_gamma_ = std::move(result.grad_gamma);
            grad_beta_ = std::move(result.grad_beta);
        }
        return std::move(result.grad_input);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("InstanceNorm2DLayer::Backward", e));
    }
#else
    throw std::runtime_error("InstanceNorm2D runs on ArrayFire, and this build has no ArrayFire");
#endif
}

std::map<std::string, Tensor> InstanceNorm2DLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    if (affine_) {
        params["gamma"] = gamma_;
        params["beta"] = beta_;
        params["grad_gamma"] = grad_gamma_;
        params["grad_beta"] = grad_beta_;
    }
    return params;
}

void InstanceNorm2DLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    if (params.count("gamma")) gamma_ = params.at("gamma");
    if (params.count("beta")) beta_ = params.at("beta");
}

// ============================================================================
// GroupNorm Layer Implementation
// ============================================================================

GroupNormLayer::GroupNormLayer(int num_groups, int num_channels, float eps, bool affine)
    : num_groups_(num_groups), num_channels_(num_channels), eps_(eps), affine_(affine) {

    if (num_groups_ <= 0 || num_channels_ <= 0 || eps_ <= 0.0f ||
        num_channels_ % num_groups_ != 0) {
        throw std::invalid_argument("GroupNorm requires positive groups/channels/eps and divisible channels");
    }

    if (affine) {
        gamma_ = Tensor::Ones({static_cast<size_t>(num_channels)});
        beta_ = Tensor::Zeros({static_cast<size_t>(num_channels)});
        grad_gamma_ = Tensor::Zeros({static_cast<size_t>(num_channels)});
        grad_beta_ = Tensor::Zeros({static_cast<size_t>(num_channels)});
    }
}

Tensor GroupNormLayer::Forward(const Tensor& input) {
    cached_input_ = input;

    ValidateSpatial4DInput(input, "GroupNorm");
    const size_t channels = input.Shape()[2];
    if (channels != static_cast<size_t>(num_channels_)) {
        throw std::runtime_error("GroupNorm forward channel mismatch");
    }

    const std::vector<size_t> channel_shape{channels};
    if (affine_ &&
        (gamma_.GetDataType() != DataType::Float32 || beta_.GetDataType() != DataType::Float32 ||
         gamma_.Shape() != channel_shape || beta_.Shape() != channel_shape)) {
        throw std::runtime_error("GroupNorm forward affine parameter mismatch");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        auto result = GroupNormForwardArrayFire(input, static_cast<size_t>(num_groups_), eps_,
                                                affine_ ? &gamma_ : nullptr, affine_ ? &beta_ : nullptr);
        normalized_ = std::move(result.normalized);
        std_inv_ = std::move(result.std_inv);
        return std::move(result.output);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("GroupNormLayer::Forward", e));
    }
#else
    throw std::runtime_error("GroupNorm runs on ArrayFire, and this build has no ArrayFire");
#endif
}

Tensor GroupNormLayer::Backward(const Tensor& grad_output) {
    ValidateSpatial4DInput(cached_input_, "GroupNorm");
    if (grad_output.GetDataType() != DataType::Float32 ||
        normalized_.GetDataType() != DataType::Float32 ||
        std_inv_.GetDataType() != DataType::Float32) {
        throw std::runtime_error("GroupNorm backward requires Float32 tensors");
    }
    if (grad_output.Shape() != cached_input_.Shape() || normalized_.Shape() != cached_input_.Shape()) {
        throw std::runtime_error("GroupNorm backward gradient/cache shape mismatch");
    }

    const size_t channels = cached_input_.Shape()[2];
    const size_t batch_size = cached_input_.Shape()[3];
    const size_t group_count = static_cast<size_t>(num_groups_);
    const std::vector<size_t> channel_shape{channels};
    const std::vector<size_t> group_shape{group_count, batch_size};
    if (channels != static_cast<size_t>(num_channels_) || channels % group_count != 0 ||
        std_inv_.Shape() != group_shape) {
        throw std::runtime_error("GroupNorm backward parameter/cache shape mismatch");
    }
    if (affine_ && (gamma_.GetDataType() != DataType::Float32 || gamma_.Shape() != channel_shape)) {
        throw std::runtime_error("GroupNorm backward gamma shape mismatch");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        auto result = GroupNormBackwardArrayFire(grad_output, normalized_, std_inv_, group_count,
                                                 affine_ ? &gamma_ : nullptr);
        if (affine_) {
            grad_gamma_ = std::move(result.grad_gamma);
            grad_beta_ = std::move(result.grad_beta);
        }
        return std::move(result.grad_input);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("GroupNormLayer::Backward", e));
    }
#else
    throw std::runtime_error("GroupNorm runs on ArrayFire, and this build has no ArrayFire");
#endif
}

std::map<std::string, Tensor> GroupNormLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    if (affine_) {
        params["gamma"] = gamma_;
        params["beta"] = beta_;
        params["grad_gamma"] = grad_gamma_;
        params["grad_beta"] = grad_beta_;
    }
    return params;
}

void GroupNormLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    if (params.count("gamma")) gamma_ = params.at("gamma");
    if (params.count("beta")) beta_ = params.at("beta");
}

} // namespace cyxwiz
