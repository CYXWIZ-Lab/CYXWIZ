#include "cyxwiz/layers/normalization.h"
#include "layer_arrayfire_utils.h"
#include "layer_utils.h"
#include "../arrayfire_host_values.h"

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

namespace {

// Semantic [H, W, C, B] as [H*W*B, C]: one column per channel.
af::array ChannelColumns(const af::array& x) {
    const af::dim4 d = x.dims();
    return af::moddims(af::reorder(x, 0, 1, 3, 2), d[0] * d[1] * d[3], d[2]);
}

af::array FromChannelColumns(const af::array& columns, const af::dim4& d) {
    return af::reorder(af::moddims(columns, d[0], d[1], d[3], d[2]), 0, 1, 3, 2);
}

af::array ChannelRow(const Tensor& vector) {
    return af::moddims(vector.GetSemanticArray(), 1, vector.GetSemanticArray().elements());
}

}  // namespace

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
        throw std::runtime_error("BatchNorm2D forward requires Float32 parameters");
    }
    const std::vector<size_t> channel_shape{static_cast<size_t>(num_features_)};
    if (gamma_.Shape() != channel_shape || beta_.Shape() != channel_shape ||
        running_mean_.Shape() != channel_shape || running_var_.Shape() != channel_shape) {
        throw std::runtime_error("BatchNorm2D forward parameter shape mismatch");
    }

    const size_t channels = input.Shape()[2];
    if (channels != static_cast<size_t>(num_features_)) {
        throw std::runtime_error("BatchNorm2D forward channel mismatch");
    }

    try {
        const af::array x4 = input.GetSemanticArray();
        const af::array x = ChannelColumns(x4);
        const af::dim4 down(x.dims(0));
        af::array mean = ChannelRow(running_mean_);
        af::array var = ChannelRow(running_var_);
        if (training_) {
            mean = af::mean(x, 0);
            const af::array centered = x - af::tile(mean, down);
            var = af::mean(centered * centered, 0);
        }
        const af::array inverse = 1.0f / af::sqrt(var + eps_);
        const af::array normalized = (x - af::tile(mean, down)) * af::tile(inverse, down);
        const af::array y = normalized * af::tile(ChannelRow(gamma_), down) + af::tile(ChannelRow(beta_), down);
        const af::array y4 = FromChannelColumns(y, x4.dims());
        const af::array normalized4 = FromChannelColumns(normalized, x4.dims());
        af::eval(y4, normalized4);
        // Publish state only once every device result exists.
        if (training_) {
            const af::array next_mean = (1.0f - momentum_) * ChannelRow(running_mean_) + momentum_ * mean;
            // PyTorch keeps the unbiased variance in the running estimate.
            const dim_t count = x.dims(0);
            const float unbiased = count > 1 ? static_cast<float>(count) / static_cast<float>(count - 1) : 1.0f;
            const af::array next_var = (1.0f - momentum_) * ChannelRow(running_var_) + momentum_ * unbiased * var;
            running_mean_ = Tensor::FromSemanticArray(af::flat(next_mean), channel_shape);
            running_var_ = Tensor::FromSemanticArray(af::flat(next_var), channel_shape);
        }
        std_inv_ = Tensor::FromSemanticArray(af::flat(inverse), channel_shape);
        normalized_ = Tensor::FromSemanticArray(normalized4, input.Shape());
        return Tensor::FromSemanticArray(y4, input.Shape());
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("BatchNorm2DLayer::Forward", e));
    }
}

Tensor BatchNorm2DLayer::Backward(const Tensor& grad_output) {
    ValidateSpatial4DInput(cached_input_, "BatchNorm2D");
    if (grad_output.GetDataType() != DataType::Float32 ||
        normalized_.GetDataType() != DataType::Float32 ||
        std_inv_.GetDataType() != DataType::Float32 ||
        gamma_.GetDataType() != DataType::Float32) {
        throw std::runtime_error("BatchNorm2D backward requires Float32 tensors");
    }
    if (grad_output.Shape() != cached_input_.Shape() || normalized_.Shape() != cached_input_.Shape()) {
        throw std::runtime_error("BatchNorm2D backward gradient/cache shape mismatch");
    }

    const size_t channels = cached_input_.Shape()[2];
    const std::vector<size_t> channel_shape{channels};
    if (channels != static_cast<size_t>(num_features_) ||
        gamma_.Shape() != channel_shape || std_inv_.Shape() != channel_shape) {
        throw std::runtime_error("BatchNorm2D backward parameter/cache shape mismatch");
    }

    try {
        const af::array dy4 = grad_output.GetSemanticArray();
        const af::array dy = ChannelColumns(dy4);
        const af::array normalized = ChannelColumns(normalized_.GetSemanticArray());
        const af::dim4 down(dy.dims(0));
        const float count = static_cast<float>(dy.dims(0));
        const af::array sum_dy = af::sum(dy, 0);
        const af::array sum_dy_norm = af::sum(dy * normalized, 0);
        const af::array scale = ChannelRow(gamma_) * ChannelRow(std_inv_) / count;
        const af::array dx = af::tile(scale, down) *
                             (count * dy - af::tile(sum_dy, down) - normalized * af::tile(sum_dy_norm, down));
        const af::array dx4 = FromChannelColumns(dx, dy4.dims());
        dx4.eval();
        Tensor grad_input = Tensor::FromSemanticArray(dx4, grad_output.Shape());
        grad_gamma_ = Tensor::FromSemanticArray(af::flat(sum_dy_norm), channel_shape);
        grad_beta_ = Tensor::FromSemanticArray(af::flat(sum_dy), channel_shape);
        return grad_input;
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("BatchNorm2DLayer::Backward", e));
    }
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
