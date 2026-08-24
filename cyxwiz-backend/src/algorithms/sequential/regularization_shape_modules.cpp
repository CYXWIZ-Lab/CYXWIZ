#include <cyxwiz/sequential.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {
// ============================================================================
// SoftmaxModule Implementation (ArrayFire)
// ============================================================================

SoftmaxModule::SoftmaxModule(int dim) : dim_(dim) {}

Tensor SoftmaxModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();

#ifdef CYXWIZ_HAS_ARRAYFIRE
    // ArrayFire implementation
    af::array x = input.GetSemanticArray();

    // Softmax: exp(x - max) / sum(exp(x - max))
    // Compute along dim 1 (classes dimension) for [batch, classes] input
    // Note: Use (af::max) to prevent Windows macro conflict
    af::array max_vals = (af::max)(x, 1);  // [batch, 1]
    af::array x_shifted = x - af::tile(max_vals, 1, static_cast<unsigned>(x.dims(1)));  // Subtract max for stability
    af::array exp_x = af::exp(x_shifted);
    af::array sum_exp = af::sum(exp_x, 1);  // [batch, 1]
    af::array softmax = exp_x / af::tile(sum_exp, 1, static_cast<unsigned>(x.dims(1)));

    Tensor output = Tensor::FromSemanticArray(softmax, input.Shape());
    output_cache_ = output.Clone();
    return output;
#else
    // CPU fallback
    const auto& shape = input.Shape();
    size_t batch_size = shape[0];
    size_t num_classes = shape.size() > 1 ? shape[1] : shape[0];

    Tensor output({batch_size, num_classes}, DataType::Float32);
    const float* in_data = input.ReadData<float>();
    float* out_data = output.MutableData<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        float max_val = in_data[b * num_classes];
        for (size_t c = 1; c < num_classes; ++c) {
            max_val = std::max(max_val, in_data[b * num_classes + c]);
        }
        float sum = 0.0f;
        for (size_t c = 0; c < num_classes; ++c) {
            out_data[b * num_classes + c] = std::exp(in_data[b * num_classes + c] - max_val);
            sum += out_data[b * num_classes + c];
        }
        for (size_t c = 0; c < num_classes; ++c) {
            out_data[b * num_classes + c] /= sum;
        }
    }
    output_cache_ = output.Clone();
    return output;
#endif
}

Tensor SoftmaxModule::Backward(const Tensor& grad_output) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    // ArrayFire implementation
    // Softmax backward: grad_input = softmax * (grad_output - sum(grad_output * softmax))
    af::array grad = grad_output.GetSemanticArray();
    af::array soft = output_cache_.GetSemanticArray();

    // Compute dot product per sample: sum(grad * softmax) along classes dimension
    af::array dot = af::sum(grad * soft, 1);  // [batch, 1]

    // grad_input = softmax * (grad - dot)
    af::array grad_input = soft * (grad - af::tile(dot, 1, static_cast<unsigned>(grad.dims(1))));

    return Tensor::FromSemanticArray(grad_input, grad_output.Shape());
#else
    // CPU fallback
    const auto& shape = grad_output.Shape();
    size_t batch_size = shape[0];
    size_t num_classes = shape.size() > 1 ? shape[1] : shape[0];

    Tensor grad_input({batch_size, num_classes}, DataType::Float32);
    const float* grad_data = grad_output.ReadData<float>();
    const float* soft_data = output_cache_.ReadData<float>();
    float* out_data = grad_input.MutableData<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        float dot = 0.0f;
        for (size_t c = 0; c < num_classes; ++c) {
            dot += grad_data[b * num_classes + c] * soft_data[b * num_classes + c];
        }
        for (size_t c = 0; c < num_classes; ++c) {
            out_data[b * num_classes + c] = soft_data[b * num_classes + c] *
                (grad_data[b * num_classes + c] - dot);
        }
    }
    return grad_input;
#endif
}

// ============================================================================
// DropoutModule Implementation (ArrayFire)
// ============================================================================

DropoutModule::DropoutModule(float p) : p_(p), layer_(p) {}

Tensor DropoutModule::Forward(const Tensor& input) {
    return layer_.Forward(input);
}

Tensor DropoutModule::Backward(const Tensor& grad_output) {
    return layer_.Backward(grad_output);
}

void DropoutModule::SetTraining(bool training) {
    Module::SetTraining(training);
    layer_.SetTraining(training);
}

std::string DropoutModule::GetName() const {
    return "Dropout(p=" + std::to_string(p_) + ")";
}

// ============================================================================
// FlattenModule Implementation (ArrayFire)
// ============================================================================

FlattenModule::FlattenModule(int start_dim) : start_dim_(start_dim) {}

Tensor FlattenModule::Forward(const Tensor& input) {
    Tensor output = input.Flatten(start_dim_);
    original_shape_ = input.Shape();
    output_shape_ = output.Shape();
    output_dtype_ = output.GetDataType();
    return output;
}

Tensor FlattenModule::Backward(const Tensor& grad_output) {
    if (original_shape_.empty()) {
        throw std::logic_error(
            "FlattenModule::Backward requires a successful Forward call");
    }
    if (grad_output.Shape() != output_shape_) {
        throw std::runtime_error(
            "FlattenModule::Backward gradient shape does not match Forward output");
    }
    if (grad_output.GetDataType() != output_dtype_) {
        throw std::runtime_error(
            "FlattenModule::Backward gradient dtype does not match Forward output");
    }
    return grad_output.Reshape(original_shape_);
}

} // namespace cyxwiz

