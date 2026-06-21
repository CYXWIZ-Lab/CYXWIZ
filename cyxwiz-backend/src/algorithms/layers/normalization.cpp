#include "cyxwiz/layer.h"
#include "layer_arrayfire_utils.h"
#include "layer_utils.h"

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
    if (normalized_shape_.empty() || eps_ <= 0.0f) {
        throw std::invalid_argument("LayerNorm requires a non-empty normalized shape and positive eps");
    }

    // Calculate total size of normalized dimensions
    size_t norm_size = 1;
    for (int dim : normalized_shape) {
        if (dim <= 0) {
            throw std::invalid_argument("LayerNorm normalized dimensions must be positive");
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
    cached_input_ = input;

    if (input.GetDataType() != DataType::Float32) {
        throw std::runtime_error("LayerNorm forward CPU fallback requires Float32 input");
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
    const size_t batch_size = input.NumElements() / norm_size;

    if (elementwise_affine_) {
        const std::vector<size_t> param_shape{norm_size};
        if (gamma_.GetDataType() != DataType::Float32 || beta_.GetDataType() != DataType::Float32 ||
            gamma_.Shape() != param_shape || beta_.Shape() != param_shape) {
            throw std::runtime_error("LayerNorm forward affine parameter mismatch");
        }
    }

    Tensor output(shape, DataType::Float32);
    normalized_ = Tensor(shape, DataType::Float32);
    std_inv_ = Tensor({batch_size}, DataType::Float32);

    const float* input_data = input.Data<float>();
    const float* gamma_data = elementwise_affine_ ? gamma_.Data<float>() : nullptr;
    const float* beta_data = elementwise_affine_ ? beta_.Data<float>() : nullptr;
    float* output_data = output.Data<float>();
    float* normalized_data = normalized_.Data<float>();
    float* std_inv_data = std_inv_.Data<float>();

    for (size_t batch = 0; batch < batch_size; ++batch) {
        const size_t offset = batch * norm_size;
        float mean = 0.0f;
        for (size_t i = 0; i < norm_size; ++i) {
            mean += input_data[offset + i];
        }
        mean /= static_cast<float>(norm_size);

        float variance = 0.0f;
        for (size_t i = 0; i < norm_size; ++i) {
            const float centered = input_data[offset + i] - mean;
            variance += centered * centered;
        }
        variance /= static_cast<float>(norm_size);
        std_inv_data[batch] = 1.0f / std::sqrt(variance + eps_);

        for (size_t i = 0; i < norm_size; ++i) {
            normalized_data[offset + i] = (input_data[offset + i] - mean) * std_inv_data[batch];
            output_data[offset + i] = elementwise_affine_
                                          ? gamma_data[i] * normalized_data[offset + i] + beta_data[i]
                                          : normalized_data[offset + i];
        }
    }

    return output;
}

Tensor LayerNormLayer::Backward(const Tensor& grad_output) {
    if (grad_output.GetDataType() != DataType::Float32 ||
        normalized_.GetDataType() != DataType::Float32 ||
        std_inv_.GetDataType() != DataType::Float32) {
        throw std::runtime_error("LayerNorm backward CPU fallback requires Float32 tensors");
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

    Tensor grad_input(grad_output.Shape(), DataType::Float32);
    if (elementwise_affine_) {
        const std::vector<size_t> param_shape{norm_size};
        if (gamma_.GetDataType() != DataType::Float32 || gamma_.Shape() != param_shape) {
            throw std::runtime_error("LayerNorm backward gamma shape mismatch");
        }
        grad_gamma_ = Tensor(param_shape, DataType::Float32);
        grad_beta_ = Tensor(param_shape, DataType::Float32);
    }

    const float* grad_data = grad_output.Data<float>();
    const float* normalized_data = normalized_.Data<float>();
    const float* std_inv_data = std_inv_.Data<float>();
    const float* gamma_data = elementwise_affine_ ? gamma_.Data<float>() : nullptr;
    float* grad_input_data = grad_input.Data<float>();
    float* grad_gamma_data = elementwise_affine_ ? grad_gamma_.Data<float>() : nullptr;
    float* grad_beta_data = elementwise_affine_ ? grad_beta_.Data<float>() : nullptr;
    const float norm_count = static_cast<float>(norm_size);

    for (size_t batch = 0; batch < batch_size; ++batch) {
        const size_t offset = batch * norm_size;
        float sum_dy = 0.0f;
        float sum_dy_norm = 0.0f;
        for (size_t i = 0; i < norm_size; ++i) {
            const float upstream = elementwise_affine_ ? grad_data[offset + i] * gamma_data[i]
                                                       : grad_data[offset + i];
            sum_dy += upstream;
            sum_dy_norm += upstream * normalized_data[offset + i];
            if (elementwise_affine_) {
                grad_gamma_data[i] += grad_data[offset + i] * normalized_data[offset + i];
                grad_beta_data[i] += grad_data[offset + i];
            }
        }

        const float scale = std_inv_data[batch] / norm_count;
        for (size_t i = 0; i < norm_size; ++i) {
            const float upstream = elementwise_affine_ ? grad_data[offset + i] * gamma_data[i]
                                                       : grad_data[offset + i];
            grad_input_data[offset + i] =
                scale * (norm_count * upstream - sum_dy - normalized_data[offset + i] * sum_dy_norm);
        }
    }

    return grad_input;
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
        throw std::runtime_error("InstanceNorm2D forward CPU fallback requires Float32 input");
    }

    const std::vector<size_t>& shape = input.Shape();
    const size_t height = shape[0];
    const size_t width = shape[1];
    const size_t channels = shape[2];
    const size_t batch_size = shape[3];
    if (channels != static_cast<size_t>(num_features_)) {
        throw std::runtime_error("InstanceNorm2D forward channel mismatch");
    }
    const std::vector<size_t> channel_shape{channels};
    if (affine_ &&
        (gamma_.GetDataType() != DataType::Float32 || beta_.GetDataType() != DataType::Float32 ||
         gamma_.Shape() != channel_shape || beta_.Shape() != channel_shape)) {
        throw std::runtime_error("InstanceNorm2D forward affine parameter mismatch");
    }

    Tensor output(shape, DataType::Float32);
    normalized_ = Tensor(shape, DataType::Float32);
    std_inv_ = Tensor({channels, batch_size}, DataType::Float32);

    const float* input_data = input.Data<float>();
    const float* gamma_data = affine_ ? gamma_.Data<float>() : nullptr;
    const float* beta_data = affine_ ? beta_.Data<float>() : nullptr;
    float* output_data = output.Data<float>();
    float* normalized_data = normalized_.Data<float>();
    float* std_inv_data = std_inv_.Data<float>();
    const float sample_count = static_cast<float>(height * width);

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            float mean = 0.0f;
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    mean += input_data[Pool4DIndex(h, w, c, b, width, channels, batch_size)];
                }
            }
            mean /= sample_count;

            float variance = 0.0f;
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    const size_t index = Pool4DIndex(h, w, c, b, width, channels, batch_size);
                    const float centered = input_data[index] - mean;
                    variance += centered * centered;
                }
            }
            variance /= sample_count;
            const float std_inv = 1.0f / std::sqrt(variance + eps_);
            std_inv_data[c * batch_size + b] = std_inv;

            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    const size_t index = Pool4DIndex(h, w, c, b, width, channels, batch_size);
                    normalized_data[index] = (input_data[index] - mean) * std_inv;
                    output_data[index] = affine_
                                             ? gamma_data[c] * normalized_data[index] + beta_data[c]
                                             : normalized_data[index];
                }
            }
        }
    }

    return output;
}

Tensor InstanceNorm2DLayer::Backward(const Tensor& grad_output) {
    ValidateSpatial4DInput(cached_input_, "InstanceNorm2D");
    if (grad_output.GetDataType() != DataType::Float32 ||
        normalized_.GetDataType() != DataType::Float32 ||
        std_inv_.GetDataType() != DataType::Float32) {
        throw std::runtime_error("InstanceNorm2D backward CPU fallback requires Float32 tensors");
    }
    if (grad_output.Shape() != cached_input_.Shape() || normalized_.Shape() != cached_input_.Shape()) {
        throw std::runtime_error("InstanceNorm2D backward gradient/cache shape mismatch");
    }

    const std::vector<size_t>& shape = cached_input_.Shape();
    const size_t height = shape[0];
    const size_t width = shape[1];
    const size_t channels = shape[2];
    const size_t batch_size = shape[3];
    const std::vector<size_t> channel_shape{channels};
    const std::vector<size_t> instance_shape{channels, batch_size};
    if (channels != static_cast<size_t>(num_features_) || std_inv_.Shape() != instance_shape) {
        throw std::runtime_error("InstanceNorm2D backward parameter/cache shape mismatch");
    }
    if (affine_ && (gamma_.GetDataType() != DataType::Float32 || gamma_.Shape() != channel_shape)) {
        throw std::runtime_error("InstanceNorm2D backward gamma shape mismatch");
    }

    Tensor grad_input(shape, DataType::Float32);
    if (affine_) {
        grad_gamma_ = Tensor(channel_shape, DataType::Float32);
        grad_beta_ = Tensor(channel_shape, DataType::Float32);
    }

    const float* grad_data = grad_output.Data<float>();
    const float* normalized_data = normalized_.Data<float>();
    const float* std_inv_data = std_inv_.Data<float>();
    const float* gamma_data = affine_ ? gamma_.Data<float>() : nullptr;
    float* grad_input_data = grad_input.Data<float>();
    float* grad_gamma_data = affine_ ? grad_gamma_.Data<float>() : nullptr;
    float* grad_beta_data = affine_ ? grad_beta_.Data<float>() : nullptr;
    const float sample_count = static_cast<float>(height * width);

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            float sum_dy = 0.0f;
            float sum_dy_norm = 0.0f;
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    const size_t index = Pool4DIndex(h, w, c, b, width, channels, batch_size);
                    const float upstream = affine_ ? grad_data[index] * gamma_data[c] : grad_data[index];
                    sum_dy += upstream;
                    sum_dy_norm += upstream * normalized_data[index];
                    if (affine_) {
                        grad_gamma_data[c] += grad_data[index] * normalized_data[index];
                        grad_beta_data[c] += grad_data[index];
                    }
                }
            }

            const float scale = std_inv_data[c * batch_size + b] / sample_count;
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    const size_t index = Pool4DIndex(h, w, c, b, width, channels, batch_size);
                    const float upstream = affine_ ? grad_data[index] * gamma_data[c] : grad_data[index];
                    grad_input_data[index] =
                        scale * (sample_count * upstream - sum_dy - normalized_data[index] * sum_dy_norm);
                }
            }
        }
    }

    return grad_input;
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
    const std::vector<size_t>& shape = input.Shape();
    const size_t height = shape[0];
    const size_t width = shape[1];
    const size_t channels = shape[2];
    const size_t batch_size = shape[3];
    if (channels != static_cast<size_t>(num_channels_)) {
        throw std::runtime_error("GroupNorm forward channel mismatch");
    }

    const std::vector<size_t> channel_shape{channels};
    if (affine_ &&
        (gamma_.GetDataType() != DataType::Float32 || beta_.GetDataType() != DataType::Float32 ||
         gamma_.Shape() != channel_shape || beta_.Shape() != channel_shape)) {
        throw std::runtime_error("GroupNorm forward affine parameter mismatch");
    }

    const size_t group_count = static_cast<size_t>(num_groups_);
    const size_t channels_per_group = channels / group_count;
    const float sample_count = static_cast<float>(height * width * channels_per_group);
    Tensor output(shape, DataType::Float32);
    normalized_ = Tensor(shape, DataType::Float32);
    std_inv_ = Tensor({group_count, batch_size}, DataType::Float32);

    const float* input_data = input.Data<float>();
    const float* gamma_data = affine_ ? gamma_.Data<float>() : nullptr;
    const float* beta_data = affine_ ? beta_.Data<float>() : nullptr;
    float* output_data = output.Data<float>();
    float* normalized_data = normalized_.Data<float>();
    float* std_inv_data = std_inv_.Data<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t g = 0; g < group_count; ++g) {
            const size_t channel_begin = g * channels_per_group;
            const size_t channel_end = channel_begin + channels_per_group;
            float mean = 0.0f;
            for (size_t c = channel_begin; c < channel_end; ++c) {
                for (size_t h = 0; h < height; ++h) {
                    for (size_t w = 0; w < width; ++w) {
                        mean += input_data[Pool4DIndex(h, w, c, b, width, channels, batch_size)];
                    }
                }
            }
            mean /= sample_count;

            float variance = 0.0f;
            for (size_t c = channel_begin; c < channel_end; ++c) {
                for (size_t h = 0; h < height; ++h) {
                    for (size_t w = 0; w < width; ++w) {
                        const size_t index = Pool4DIndex(h, w, c, b, width, channels, batch_size);
                        const float centered = input_data[index] - mean;
                        variance += centered * centered;
                    }
                }
            }
            variance /= sample_count;
            const float std_inv = 1.0f / std::sqrt(variance + eps_);
            std_inv_data[g * batch_size + b] = std_inv;

            for (size_t c = channel_begin; c < channel_end; ++c) {
                for (size_t h = 0; h < height; ++h) {
                    for (size_t w = 0; w < width; ++w) {
                        const size_t index = Pool4DIndex(h, w, c, b, width, channels, batch_size);
                        normalized_data[index] = (input_data[index] - mean) * std_inv;
                        output_data[index] = affine_
                                                 ? gamma_data[c] * normalized_data[index] + beta_data[c]
                                                 : normalized_data[index];
                    }
                }
            }
        }
    }

    return output;
}

Tensor GroupNormLayer::Backward(const Tensor& grad_output) {
    ValidateSpatial4DInput(cached_input_, "GroupNorm");
    if (grad_output.GetDataType() != DataType::Float32 ||
        normalized_.GetDataType() != DataType::Float32 ||
        std_inv_.GetDataType() != DataType::Float32) {
        throw std::runtime_error("GroupNorm backward CPU fallback requires Float32 tensors");
    }
    if (grad_output.Shape() != cached_input_.Shape() || normalized_.Shape() != cached_input_.Shape()) {
        throw std::runtime_error("GroupNorm backward gradient/cache shape mismatch");
    }

    const std::vector<size_t>& shape = cached_input_.Shape();
    const size_t height = shape[0];
    const size_t width = shape[1];
    const size_t channels = shape[2];
    const size_t batch_size = shape[3];
    const size_t group_count = static_cast<size_t>(num_groups_);
    const size_t channels_per_group = channels / group_count;
    const std::vector<size_t> channel_shape{channels};
    const std::vector<size_t> group_shape{group_count, batch_size};
    if (channels != static_cast<size_t>(num_channels_) || channels % group_count != 0 ||
        std_inv_.Shape() != group_shape) {
        throw std::runtime_error("GroupNorm backward parameter/cache shape mismatch");
    }
    if (affine_ && (gamma_.GetDataType() != DataType::Float32 || gamma_.Shape() != channel_shape)) {
        throw std::runtime_error("GroupNorm backward gamma shape mismatch");
    }

    Tensor grad_input(shape, DataType::Float32);
    if (affine_) {
        grad_gamma_ = Tensor(channel_shape, DataType::Float32);
        grad_beta_ = Tensor(channel_shape, DataType::Float32);
    }

    const float* grad_data = grad_output.Data<float>();
    const float* normalized_data = normalized_.Data<float>();
    const float* std_inv_data = std_inv_.Data<float>();
    const float* gamma_data = affine_ ? gamma_.Data<float>() : nullptr;
    float* grad_input_data = grad_input.Data<float>();
    float* grad_gamma_data = affine_ ? grad_gamma_.Data<float>() : nullptr;
    float* grad_beta_data = affine_ ? grad_beta_.Data<float>() : nullptr;
    const float sample_count = static_cast<float>(height * width * channels_per_group);

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t g = 0; g < group_count; ++g) {
            const size_t channel_begin = g * channels_per_group;
            const size_t channel_end = channel_begin + channels_per_group;
            float sum_dy = 0.0f;
            float sum_dy_norm = 0.0f;
            for (size_t c = channel_begin; c < channel_end; ++c) {
                for (size_t h = 0; h < height; ++h) {
                    for (size_t w = 0; w < width; ++w) {
                        const size_t index = Pool4DIndex(h, w, c, b, width, channels, batch_size);
                        const float upstream = affine_ ? grad_data[index] * gamma_data[c] : grad_data[index];
                        sum_dy += upstream;
                        sum_dy_norm += upstream * normalized_data[index];
                        if (affine_) {
                            grad_gamma_data[c] += grad_data[index] * normalized_data[index];
                            grad_beta_data[c] += grad_data[index];
                        }
                    }
                }
            }

            const float scale = std_inv_data[g * batch_size + b] / sample_count;
            for (size_t c = channel_begin; c < channel_end; ++c) {
                for (size_t h = 0; h < height; ++h) {
                    for (size_t w = 0; w < width; ++w) {
                        const size_t index = Pool4DIndex(h, w, c, b, width, channels, batch_size);
                        const float upstream = affine_ ? grad_data[index] * gamma_data[c] : grad_data[index];
                        grad_input_data[index] =
                            scale * (sample_count * upstream - sum_dy - normalized_data[index] * sum_dy_norm);
                    }
                }
            }
        }
    }

    return grad_input;
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