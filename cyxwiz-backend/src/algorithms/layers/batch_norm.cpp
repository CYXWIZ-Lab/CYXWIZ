#include "cyxwiz/layers/normalization.h"
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

BatchNorm2DLayer::BatchNorm2DLayer(int num_features, float eps, float momentum)
    : num_features_(num_features), eps_(eps), momentum_(momentum) {
    if (num_features_ <= 0 || eps_ <= 0.0f || momentum_ < 0.0f || momentum_ > 1.0f) {
        throw std::invalid_argument("BatchNorm2D requires positive features/eps and momentum in [0, 1]");
    }

    // Initialize gamma (scale) to ones
    gamma_ = Tensor::Ones({static_cast<size_t>(num_features)});

    // Initialize beta (shift) to zeros
    beta_ = Tensor::Zeros({static_cast<size_t>(num_features)});

    // Initialize running statistics
    running_mean_ = Tensor::Zeros({static_cast<size_t>(num_features)});
    running_var_ = Tensor::Ones({static_cast<size_t>(num_features)});

    // Initialize gradient accumulators
    grad_gamma_ = Tensor::Zeros({static_cast<size_t>(num_features)});
    grad_beta_ = Tensor::Zeros({static_cast<size_t>(num_features)});
}

Tensor BatchNorm2DLayer::Forward(const Tensor& input) {
    cached_input_ = input;

    ValidateSpatial4DInput(input, "BatchNorm2D");
    if (gamma_.GetDataType() != DataType::Float32 || beta_.GetDataType() != DataType::Float32 ||
        running_mean_.GetDataType() != DataType::Float32 || running_var_.GetDataType() != DataType::Float32) {
        throw std::runtime_error("BatchNorm2D forward CPU fallback requires Float32 parameters");
    }
    const std::vector<size_t> channel_shape{static_cast<size_t>(num_features_)};
    if (gamma_.Shape() != channel_shape || beta_.Shape() != channel_shape ||
        running_mean_.Shape() != channel_shape || running_var_.Shape() != channel_shape) {
        throw std::runtime_error("BatchNorm2D forward parameter shape mismatch");
    }

    const std::vector<size_t>& shape = input.Shape();
    const size_t height = shape[0];
    const size_t width = shape[1];
    const size_t channels = shape[2];
    const size_t batch_size = shape[3];
    if (channels != static_cast<size_t>(num_features_)) {
        throw std::runtime_error("BatchNorm2D forward channel mismatch");
    }

    Tensor output(shape, DataType::Float32);
    normalized_ = Tensor(shape, DataType::Float32);
    std_inv_ = Tensor(channel_shape, DataType::Float32);

    const float* input_data = input.Data<float>();
    const float* gamma_data = gamma_.Data<float>();
    const float* beta_data = beta_.Data<float>();
    float* running_mean_data = running_mean_.Data<float>();
    float* running_var_data = running_var_.Data<float>();
    float* output_data = output.Data<float>();
    float* normalized_data = normalized_.Data<float>();
    float* std_inv_data = std_inv_.Data<float>();

    std::vector<float> mean(channels, 0.0f);
    std::vector<float> var(channels, 0.0f);
    const float sample_count = static_cast<float>(height * width * batch_size);

    if (training_) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t h = 0; h < height; ++h) {
                    for (size_t w = 0; w < width; ++w) {
                        mean[c] += input_data[Pool4DIndex(h, w, c, b, width, channels, batch_size)];
                    }
                }
            }
            mean[c] /= sample_count;
        }

        for (size_t c = 0; c < channels; ++c) {
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t h = 0; h < height; ++h) {
                    for (size_t w = 0; w < width; ++w) {
                        const float centered = input_data[Pool4DIndex(h, w, c, b, width, channels, batch_size)] - mean[c];
                        var[c] += centered * centered;
                    }
                }
            }
            var[c] /= sample_count;
            running_mean_data[c] = (1.0f - momentum_) * running_mean_data[c] + momentum_ * mean[c];
            running_var_data[c] = (1.0f - momentum_) * running_var_data[c] + momentum_ * var[c];
        }
    } else {
        for (size_t c = 0; c < channels; ++c) {
            mean[c] = running_mean_data[c];
            var[c] = running_var_data[c];
        }
    }

    for (size_t c = 0; c < channels; ++c) {
        std_inv_data[c] = 1.0f / std::sqrt(var[c] + eps_);
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    const size_t index = Pool4DIndex(h, w, c, b, width, channels, batch_size);
                    normalized_data[index] = (input_data[index] - mean[c]) * std_inv_data[c];
                    output_data[index] = gamma_data[c] * normalized_data[index] + beta_data[c];
                }
            }
        }
    }

    return output;
}

Tensor BatchNorm2DLayer::Backward(const Tensor& grad_output) {
    ValidateSpatial4DInput(cached_input_, "BatchNorm2D");
    if (grad_output.GetDataType() != DataType::Float32 ||
        normalized_.GetDataType() != DataType::Float32 ||
        std_inv_.GetDataType() != DataType::Float32 ||
        gamma_.GetDataType() != DataType::Float32) {
        throw std::runtime_error("BatchNorm2D backward CPU fallback requires Float32 tensors");
    }
    if (grad_output.Shape() != cached_input_.Shape() || normalized_.Shape() != cached_input_.Shape()) {
        throw std::runtime_error("BatchNorm2D backward gradient/cache shape mismatch");
    }

    const std::vector<size_t>& shape = cached_input_.Shape();
    const size_t height = shape[0];
    const size_t width = shape[1];
    const size_t channels = shape[2];
    const size_t batch_size = shape[3];
    const std::vector<size_t> channel_shape{channels};
    if (channels != static_cast<size_t>(num_features_) ||
        gamma_.Shape() != channel_shape || std_inv_.Shape() != channel_shape) {
        throw std::runtime_error("BatchNorm2D backward parameter/cache shape mismatch");
    }

    Tensor grad_input(shape, DataType::Float32);
    grad_gamma_ = Tensor(channel_shape, DataType::Float32);
    grad_beta_ = Tensor(channel_shape, DataType::Float32);

    const float* grad_data = grad_output.Data<float>();
    const float* normalized_data = normalized_.Data<float>();
    const float* gamma_data = gamma_.Data<float>();
    const float* std_inv_data = std_inv_.Data<float>();
    float* grad_input_data = grad_input.Data<float>();
    float* grad_gamma_data = grad_gamma_.Data<float>();
    float* grad_beta_data = grad_beta_.Data<float>();
    const float sample_count = static_cast<float>(height * width * batch_size);

    std::vector<float> sum_dy(channels, 0.0f);
    std::vector<float> sum_dy_norm(channels, 0.0f);
    for (size_t c = 0; c < channels; ++c) {
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    const size_t index = Pool4DIndex(h, w, c, b, width, channels, batch_size);
                    sum_dy[c] += grad_data[index];
                    sum_dy_norm[c] += grad_data[index] * normalized_data[index];
                }
            }
        }
        grad_beta_data[c] = sum_dy[c];
        grad_gamma_data[c] = sum_dy_norm[c];
    }

    for (size_t c = 0; c < channels; ++c) {
        const float scale = gamma_data[c] * std_inv_data[c] / sample_count;
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    const size_t index = Pool4DIndex(h, w, c, b, width, channels, batch_size);
                    grad_input_data[index] =
                        scale * (sample_count * grad_data[index] -
                                 sum_dy[c] -
                                 normalized_data[index] * sum_dy_norm[c]);
                }
            }
        }
    }

    return grad_input;
}

std::map<std::string, Tensor> BatchNorm2DLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    params["gamma"] = gamma_;
    params["beta"] = beta_;
    params["running_mean"] = running_mean_;
    params["running_var"] = running_var_;
    params["grad_gamma"] = grad_gamma_;
    params["grad_beta"] = grad_beta_;
    return params;
}

void BatchNorm2DLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    if (params.count("gamma")) {
        gamma_ = params.at("gamma");
    }
    if (params.count("beta")) {
        beta_ = params.at("beta");
    }
    if (params.count("running_mean")) {
        running_mean_ = params.at("running_mean");
    }
    if (params.count("running_var")) {
        running_var_ = params.at("running_var");
    }
}

} // namespace cyxwiz
