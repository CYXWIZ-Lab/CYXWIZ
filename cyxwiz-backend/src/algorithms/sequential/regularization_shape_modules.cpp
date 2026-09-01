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
namespace {

int NormalizeSoftmaxDimension(int dimension, int rank) {
    if (rank <= 0) {
        throw std::runtime_error("SoftmaxModule requires at least one dimension");
    }
    const int normalized = dimension < 0 ? dimension + rank : dimension;
    if (normalized < 0 || normalized >= rank) {
        throw std::runtime_error("SoftmaxModule dimension is out of range");
    }
    return normalized;
}

std::vector<size_t> SoftmaxRowMajorStrides(
    const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] =
            strides[static_cast<size_t>(i + 1)] *
            shape[static_cast<size_t>(i + 1)];
    }
    return strides;
}

} // namespace

// ============================================================================
// SoftmaxModule Implementation (ArrayFire)
// ============================================================================

SoftmaxModule::SoftmaxModule(int dim) : dim_(dim) {}

Tensor SoftmaxModule::Forward(const Tensor& input) {
    if (input.GetDataType() != DataType::Float32) {
        throw std::runtime_error("SoftmaxModule only supports Float32 tensors");
    }
    const int actual_dim = NormalizeSoftmaxDimension(
        dim_, static_cast<int>(input.Shape().size()));
    input_cache_ = input.Clone();
    if (input.NumElements() == 0) {
        output_cache_ = input.Clone();
        return output_cache_;
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    // ArrayFire implementation
    af::array x = input.GetSemanticArray();

    // Softmax: exp(x - max) / sum(exp(x - max))
    af::array max_vals = (af::max)(x, actual_dim);
    af::dim4 tile_dims(1, 1, 1, 1);
    tile_dims[actual_dim] = x.dims(actual_dim);
    af::array x_shifted = x - af::tile(max_vals, tile_dims);
    af::array exp_x = af::exp(x_shifted);
    af::array sum_exp = af::sum(exp_x, actual_dim);
    af::array softmax = exp_x / af::tile(sum_exp, tile_dims);

    Tensor output = Tensor::FromSemanticArray(softmax, input.Shape());
    output_cache_ = output.Clone();
    return output;
#else
    // CPU fallback
    const auto& shape = input.Shape();
    const std::vector<size_t> strides = SoftmaxRowMajorStrides(shape);
    const size_t axis_size = shape[static_cast<size_t>(actual_dim)];
    const size_t axis_stride = strides[static_cast<size_t>(actual_dim)];
    const size_t outer_count = input.NumElements() / axis_size;

    Tensor output(shape, DataType::Float32);
    const float* in_data = input.ReadData<float>();
    float* out_data = output.MutableData<float>();

    for (size_t outer = 0; outer < outer_count; ++outer) {
        const size_t before_axis = outer / axis_stride;
        const size_t after_axis = outer % axis_stride;
        const size_t base = before_axis * axis_size * axis_stride + after_axis;
        float max_val = in_data[base];
        for (size_t i = 1; i < axis_size; ++i) {
            max_val = std::max(max_val, in_data[base + i * axis_stride]);
        }
        float sum = 0.0f;
        for (size_t i = 0; i < axis_size; ++i) {
            const size_t index = base + i * axis_stride;
            out_data[index] = std::exp(in_data[index] - max_val);
            sum += out_data[index];
        }
        for (size_t i = 0; i < axis_size; ++i) {
            out_data[base + i * axis_stride] /= sum;
        }
    }
    output_cache_ = output.Clone();
    return output;
#endif
}

Tensor SoftmaxModule::Backward(const Tensor& grad_output) {
    if (grad_output.GetDataType() != DataType::Float32 ||
        output_cache_.Shape() != grad_output.Shape()) {
        throw std::runtime_error(
            "SoftmaxModule backward requires a matching successful forward");
    }
    const int actual_dim = NormalizeSoftmaxDimension(
        dim_, static_cast<int>(grad_output.Shape().size()));
    if (grad_output.NumElements() == 0) {
        return grad_output.Clone();
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    // ArrayFire implementation
    // Softmax backward: grad_input = softmax * (grad_output - sum(grad_output * softmax))
    af::array grad = grad_output.GetSemanticArray();
    af::array soft = output_cache_.GetSemanticArray();

    af::array dot = af::sum(grad * soft, actual_dim);
    af::dim4 tile_dims(1, 1, 1, 1);
    tile_dims[actual_dim] = grad.dims(actual_dim);

    // grad_input = softmax * (grad - dot)
    af::array grad_input = soft * (grad - af::tile(dot, tile_dims));

    return Tensor::FromSemanticArray(grad_input, grad_output.Shape());
#else
    // CPU fallback
    const auto& shape = grad_output.Shape();
    const std::vector<size_t> strides = SoftmaxRowMajorStrides(shape);
    const size_t axis_size = shape[static_cast<size_t>(actual_dim)];
    const size_t axis_stride = strides[static_cast<size_t>(actual_dim)];
    const size_t outer_count = grad_output.NumElements() / axis_size;

    Tensor grad_input(shape, DataType::Float32);
    const float* grad_data = grad_output.ReadData<float>();
    const float* soft_data = output_cache_.ReadData<float>();
    float* out_data = grad_input.MutableData<float>();

    for (size_t outer = 0; outer < outer_count; ++outer) {
        const size_t before_axis = outer / axis_stride;
        const size_t after_axis = outer % axis_stride;
        const size_t base = before_axis * axis_size * axis_stride + after_axis;
        float dot = 0.0f;
        for (size_t i = 0; i < axis_size; ++i) {
            const size_t index = base + i * axis_stride;
            dot += grad_data[index] * soft_data[index];
        }
        for (size_t i = 0; i < axis_size; ++i) {
            const size_t index = base + i * axis_stride;
            out_data[index] = soft_data[index] * (grad_data[index] - dot);
        }
    }
    return grad_input;
#endif
}

// ============================================================================
// DropoutModule Implementation (ArrayFire)
// ============================================================================

DropoutModule::DropoutModule(float p) : p_(p), layer_(p) {
    if (!std::isfinite(p_) || p_ < 0.0f || p_ > 1.0f) {
        throw std::invalid_argument(
            "DropoutModule: p must be a finite probability in [0, 1]");
    }
}

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

