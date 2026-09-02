#include "cyxwiz/layers/linear.h"
#include "cyxwiz/backend_placement_observation.h"
#include "../arrayfire_backend_utils.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <spdlog/spdlog.h>
#include <cyxwiz/error_codes.h>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {
namespace {

void ValidateLinearSparseCsrBatchView(
    const LinearSparseCsrBatchView& input,
    size_t expected_columns) {
    if (input.rows == 0 || input.columns == 0) {
        throw std::invalid_argument(
            "LinearLayer sparse CSR input must have positive dimensions");
    }
    if (input.columns != expected_columns) {
        throw std::invalid_argument(
            "LinearLayer sparse CSR input features mismatch. Expected " +
            std::to_string(expected_columns) + ", got " +
            std::to_string(input.columns));
    }
    if (input.rows > static_cast<size_t>(
            (std::numeric_limits<int32_t>::max)()) ||
        input.columns > static_cast<size_t>(
            (std::numeric_limits<int32_t>::max)()) ||
        input.nnz > static_cast<size_t>(
            (std::numeric_limits<int32_t>::max)())) {
        throw std::length_error(
            "LinearLayer sparse CSR dimensions exceed the int32 boundary");
    }
    if (input.row_offsets == nullptr) {
        throw std::invalid_argument(
            "LinearLayer sparse CSR row offsets are null");
    }
    if (input.nnz > 0 &&
        (input.column_indices == nullptr || input.values == nullptr)) {
        throw std::invalid_argument(
            "LinearLayer sparse CSR values or column indices are null");
    }
    if (input.row_offsets[0] != 0 ||
        input.row_offsets[input.rows] != static_cast<int32_t>(input.nnz)) {
        throw std::invalid_argument(
            "LinearLayer sparse CSR row offsets do not bound nnz");
    }
    int32_t previous = 0;
    for (size_t row = 0; row <= input.rows; ++row) {
        const int32_t offset = input.row_offsets[row];
        if (offset < previous || offset < 0 ||
            static_cast<size_t>(offset) > input.nnz) {
            throw std::invalid_argument(
                "LinearLayer sparse CSR row offsets are not canonical");
        }
        previous = offset;
    }
    for (size_t index = 0; index < input.nnz; ++index) {
        const int32_t column = input.column_indices[index];
        if (column < 0 || static_cast<size_t>(column) >= input.columns) {
            throw std::invalid_argument(
                "LinearLayer sparse CSR column index is out of range");
        }
    }
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
void LogLinearInitializationFallbackOnce(
    const char* error_message,
    size_t in_features,
    size_t out_features,
    bool use_bias) {
    const BackendFallbackReason reason =
        ClassifyArrayFireBackendFallbackReason(error_message);
    const std::string context = BuildArrayFireBackendFallbackContext(
        "in_features=" + std::to_string(in_features) +
        "; out_features=" + std::to_string(out_features));
    std::string message =
        "ArrayFire LinearLayer::InitializeWeights failed (reason=" +
        std::string(BackendFallbackReasonName(reason)) +
        "); initializing weights on CPU.";
    message += " Context: ";
    message += context;
    message += ".";
    if (reason != BackendFallbackReason::CudaJitParamOverflow &&
        error_message != nullptr && error_message[0] != '\0') {
        message += " Error: ";
        message += error_message;
    }
    RecordBackendPlacementObservationForActiveDevice(
        "Linear",
        CurrentArrayFireBackendName(),
        "float32",
        BuildLinearPlacementShapeSignature(
            {},
            {out_features, in_features},
            {out_features, in_features},
            "float32",
            use_bias),
        BackendFallbackReasonName(reason),
        BackendPlacementObservationSource::RuntimeFallback,
        message);
    ThrowIfArrayFireNativeCpuFallbackForbidden(
        "LinearLayer::InitializeWeights",
        reason,
        error_message,
        context);
    if (!ShouldLogArrayFireBackendFallbackOnce(
            "LinearLayer::InitializeWeights", reason, context)) {
        return;
    }
    spdlog::warn("{}", message);
}

std::string BuildLinearRuntimeFallbackDetail(
    const char* operation_name,
    BackendFallbackReason reason,
    const char* error_message,
    const std::string& context) {
    return BuildArrayFireBackendFallbackMessage(
        operation_name,
        reason,
        reason != BackendFallbackReason::CudaJitParamOverflow,
        error_message,
        context);
}

std::vector<size_t> BuildLinearOutputShape(size_t batch_size,
                                           size_t out_features,
                                           bool is_batched) {
    return is_batched
        ? std::vector<size_t>{batch_size, out_features}
        : std::vector<size_t>{out_features};
}

std::string BuildLinearRuntimeFallbackContext(size_t in_features,
                                              size_t out_features,
                                              size_t batch_size,
                                              bool use_bias) {
    return BuildArrayFireBackendFallbackContext(
        "in=" + std::to_string(in_features) +
        "; out=" + std::to_string(out_features) +
        "; batch=" + std::to_string(batch_size) +
        "; bias=" + std::string(use_bias ? "true" : "false"));
}

void RecordLinearRuntimeFallback(
    const char* operation_name,
    BackendFallbackReason reason,
    const char* error_message,
    const std::string& context,
    const std::vector<size_t>& lhs_shape,
    const std::vector<size_t>& output_shape,
    size_t in_features,
    size_t out_features,
    bool use_bias) {
    RecordBackendPlacementObservationForActiveDevice(
        "Linear",
        CurrentArrayFireBackendName(),
        "float32",
        BuildLinearPlacementShapeSignature(
            lhs_shape,
            {out_features, in_features},
            output_shape,
            "float32",
            use_bias),
        BackendFallbackReasonName(reason),
        BackendPlacementObservationSource::RuntimeFallback,
        BuildLinearRuntimeFallbackDetail(
            operation_name, reason, error_message, context));
    ThrowIfArrayFireNativeCpuFallbackForbidden(
        operation_name,
        reason,
        error_message,
        context);
}
#endif

} // namespace

LinearLayer::LinearLayer(size_t in_features, size_t out_features, bool use_bias)
    : in_features_(in_features)
    , out_features_(out_features)
    , use_bias_(use_bias)
    , weight_({out_features, in_features}, DataType::Float32)
    , weight_grad_({out_features, in_features}, DataType::Float32)
{
    if (use_bias_) {
        bias_ = Tensor({out_features}, DataType::Float32);
        bias_grad_ = Tensor({out_features}, DataType::Float32);
    }

    // Initialize weights
    InitializeWeights();
}

void LinearLayer::InitializeWeights() {
    // Xavier/Glorot initialization: weights ~ U(-sqrt(6/(in+out)), sqrt(6/(in+out)))
    double limit = std::sqrt(6.0 / (in_features_ + out_features_));

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (IsCurrentArrayFireBackendAvailable()) {
        try {
            af::array w_gpu = af::randu(static_cast<dim_t>(out_features_),
                                         static_cast<dim_t>(in_features_), f32);
            // Scale to [-limit, limit]
            w_gpu = (w_gpu * 2.0f - 1.0f) * static_cast<float>(limit);
            w_gpu.eval();

            weight_ = Tensor::FromArrayRowMajor2D(w_gpu);

            if (use_bias_) {
                bias_ = Tensor::Zeros({out_features_}, DataType::Float32);
            }

            spdlog::info("LinearLayer({}, {}) initialized with Xavier (ArrayFire)", in_features_, out_features_);
            return;
        } catch (const af::exception& e) {
            LogLinearInitializationFallbackOnce(
                e.what(),
                in_features_,
                out_features_,
                use_bias_);
        }
    }
#endif

    // CPU fallback
    weight_ = Tensor::Random({out_features_, in_features_}, DataType::Float32);

    // Scale to [-limit, limit]
    float* weight_data = weight_.MutableData<float>();
    size_t num_weights = out_features_ * in_features_;
    for (size_t i = 0; i < num_weights; i++) {
        weight_data[i] = (weight_data[i] * 2.0f - 1.0f) * static_cast<float>(limit);
    }

    if (use_bias_) {
        bias_ = Tensor::Zeros({out_features_}, DataType::Float32);
    }

    spdlog::debug("LinearLayer({}, {}) initialized with Xavier (CPU)", in_features_, out_features_);
}

Tensor LinearLayer::Forward(const Tensor& input) {
    // Cache input for backward pass
    input_cache_ = input.Clone();

    const auto& input_shape = input.Shape();
    bool is_batched = input_shape.size() == 2;

    if (!is_batched && input_shape.size() != 1) {
        throw std::runtime_error("LinearLayer: Input must be 1D or 2D tensor");
    }

    size_t batch_size = is_batched ? input_shape[0] : 1;
    size_t in_features = is_batched ? input_shape[1] : input_shape[0];

    if (in_features != in_features_) {
        throw std::runtime_error("LinearLayer: Input features mismatch. Expected " +
                               std::to_string(in_features_) + ", got " +
                               std::to_string(in_features));
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (IsCurrentArrayFireBackendAvailable()) {
        const std::string context = BuildLinearRuntimeFallbackContext(
            in_features_, out_features_, batch_size, use_bias_);
        if (ShouldForceArrayFireBackendFallbackForTesting(
                "LinearLayer::Forward")) {
            RecordLinearRuntimeFallback(
                "LinearLayer::Forward",
                BackendFallbackReason::GpuBackendException,
                "forced ArrayFire backend fallback test hook",
                context,
                input_shape,
                BuildLinearOutputShape(batch_size, out_features_, is_batched),
                in_features_,
                out_features_,
                use_bias_);
        } else {
            try {
                af::array input_gpu;
                if (is_batched) {
                    input_gpu = input.GetArrayRowMajor2D().as(af::dtype::f32);
                } else {
                    input_gpu = af::moddims(
                        input.GetArray(),
                        1,
                        static_cast<dim_t>(in_features)).as(af::dtype::f32);
                }

                af::array weight_gpu =
                    weight_.GetArrayRowMajor2D().as(af::dtype::f32);
                af::array output_gpu =
                    af::matmul(input_gpu, weight_gpu, AF_MAT_NONE, AF_MAT_TRANS);
                output_gpu.eval();

                if (use_bias_) {
                    af::array bias_gpu = af::moddims(
                        bias_.GetArray(),
                        1,
                        static_cast<dim_t>(out_features_)).as(af::dtype::f32);
                    output_gpu = output_gpu + af::tile(
                        bias_gpu,
                        static_cast<unsigned int>(batch_size),
                        1);
                    output_gpu.eval();
                }

                if (is_batched) {
                    return Tensor::FromArrayRowMajor2D(output_gpu);
                }

                return Tensor(af::flat(output_gpu));
            } catch (const af::exception& e) {
                const BackendFallbackReason reason =
                    ClassifyArrayFireBackendFallbackReason(e.what());
                RecordLinearRuntimeFallback(
                    "LinearLayer::Forward",
                    reason,
                    e.what(),
                    context,
                    input_shape,
                    BuildLinearOutputShape(batch_size, out_features_, is_batched),
                    in_features_,
                    out_features_,
                    use_bias_);
                const bool log_fallback =
                    ShouldLogArrayFireBackendFallbackOnce(
                        "LinearLayer::Forward", reason, context);
                if (log_fallback) {
                    spdlog::warn("{}",
                                 errors::FormatWarning(
                                     errors::Gpu::KernelExecutionFailed,
                                     BuildLinearRuntimeFallbackDetail(
                                         "LinearLayer::Forward",
                                         reason,
                                         e.what(),
                                         context)));
                }
            }
        }
    }
#endif

    // CPU fallback implementation
    if (is_batched) {
        Tensor output({batch_size, out_features_}, DataType::Float32);
        const float* input_data = input.ReadData<float>();
        const float* weight_data = weight_.ReadData<float>();
        const float* bias_data = use_bias_ ? bias_.ReadData<float>() : nullptr;
        float* output_data = output.MutableData<float>();

        // Matrix multiplication: C = A @ B^T
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t o = 0; o < out_features_; o++) {
                float sum = 0.0f;
                for (size_t i = 0; i < in_features_; i++) {
                    sum += input_data[b * in_features_ + i] * weight_data[o * in_features_ + i];
                }
                if (use_bias_) {
                    sum += bias_data[o];
                }
                output_data[b * out_features_ + o] = sum;
            }
        }

        return output;
    } else {
        // Single sample (no batch dimension)
        Tensor output({out_features_}, DataType::Float32);
        const float* input_data = input.ReadData<float>();
        const float* weight_data = weight_.ReadData<float>();
        const float* bias_data = use_bias_ ? bias_.ReadData<float>() : nullptr;
        float* output_data = output.MutableData<float>();

        for (size_t o = 0; o < out_features_; o++) {
            float sum = 0.0f;
            for (size_t i = 0; i < in_features_; i++) {
                sum += input_data[i] * weight_data[o * in_features_ + i];
            }
            if (use_bias_) {
                sum += bias_data[o];
            }
            output_data[o] = sum;
        }

        return output;
    }
}

Tensor LinearLayer::ForwardSparseCsr(
    const LinearSparseCsrBatchView& input) {
    ValidateLinearSparseCsrBatchView(input, in_features_);

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (IsCurrentArrayFireBackendAvailable()) {
        const std::string context = BuildLinearRuntimeFallbackContext(
            in_features_, out_features_, input.rows, use_bias_);
        if (ShouldForceArrayFireBackendFallbackForTesting(
                "LinearLayer::ForwardSparseCsr")) {
            RecordLinearRuntimeFallback(
                "LinearLayer::ForwardSparseCsr",
                BackendFallbackReason::GpuBackendException,
                "forced ArrayFire backend fallback test hook",
                context,
                {input.rows, input.columns},
                {input.rows, out_features_},
                in_features_,
                out_features_,
                use_bias_);
        } else {
            try {
                af::array output_gpu;
                if (input.nnz == 0) {
                    output_gpu = af::constant(
                        0.0f,
                        static_cast<dim_t>(input.rows),
                        static_cast<dim_t>(out_features_),
                        f32);
                } else {
                    af::array sparse_input = af::sparse(
                        static_cast<dim_t>(input.rows),
                        static_cast<dim_t>(input.columns),
                        static_cast<dim_t>(input.nnz),
                        input.values,
                        input.row_offsets,
                        input.column_indices,
                        f32,
                        AF_STORAGE_CSR,
                        afHost);
                    af::array weight_transposed = af::transpose(
                        weight_.GetArrayRowMajor2D().as(af::dtype::f32));
                    output_gpu = af::matmul(
                        sparse_input,
                        weight_transposed,
                        AF_MAT_NONE,
                        AF_MAT_NONE);
                }
                if (use_bias_) {
                    af::array bias_gpu = af::moddims(
                        bias_.GetArray(),
                        1,
                        static_cast<dim_t>(out_features_)).as(af::dtype::f32);
                    output_gpu = output_gpu + af::tile(
                        bias_gpu,
                        static_cast<unsigned int>(input.rows),
                        1);
                }
                output_gpu.eval();
                return Tensor::FromArrayRowMajor2D(output_gpu);
            } catch (const af::exception& e) {
                const BackendFallbackReason reason =
                    ClassifyArrayFireBackendFallbackReason(e.what());
                RecordLinearRuntimeFallback(
                    "LinearLayer::ForwardSparseCsr",
                    reason,
                    e.what(),
                    context,
                    {input.rows, input.columns},
                    {input.rows, out_features_},
                    in_features_,
                    out_features_,
                    use_bias_);
                if (ShouldLogArrayFireBackendFallbackOnce(
                        "LinearLayer::ForwardSparseCsr", reason, context)) {
                    spdlog::warn("{}", BuildLinearRuntimeFallbackDetail(
                        "LinearLayer::ForwardSparseCsr",
                        reason,
                        e.what(),
                        context));
                }
            }
        }
    }
#endif

    Tensor output({input.rows, out_features_}, DataType::Float32);
    float* output_data = output.MutableData<float>();
    const float* weight_data = weight_.ReadData<float>();
    const float* bias_data = use_bias_ ? bias_.ReadData<float>() : nullptr;
    for (size_t row = 0; row < input.rows; ++row) {
        for (size_t output_feature = 0;
             output_feature < out_features_;
             ++output_feature) {
            output_data[row * out_features_ + output_feature] =
                use_bias_ ? bias_data[output_feature] : 0.0f;
        }
        const size_t begin = static_cast<size_t>(input.row_offsets[row]);
        const size_t end = static_cast<size_t>(input.row_offsets[row + 1]);
        for (size_t index = begin; index < end; ++index) {
            const size_t column =
                static_cast<size_t>(input.column_indices[index]);
            const float value = input.values[index];
            for (size_t output_feature = 0;
                 output_feature < out_features_;
                 ++output_feature) {
                output_data[row * out_features_ + output_feature] +=
                    value * weight_data[
                        output_feature * in_features_ + column];
            }
        }
    }
    return output;
}

Tensor LinearLayer::Backward(const Tensor& grad_output) {
    const auto& grad_shape = grad_output.Shape();
    const auto& input_shape = input_cache_.Shape();
    (void)input_shape;  // Suppress unused variable warning
    bool is_batched = grad_shape.size() == 2;

    size_t batch_size = is_batched ? grad_shape[0] : 1;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (IsCurrentArrayFireBackendAvailable()) {
        const std::string context = BuildLinearRuntimeFallbackContext(
            in_features_, out_features_, batch_size, use_bias_);
        if (ShouldForceArrayFireBackendFallbackForTesting(
                "LinearLayer::Backward")) {
            RecordLinearRuntimeFallback(
                "LinearLayer::Backward",
                BackendFallbackReason::GpuBackendException,
                "forced ArrayFire backend fallback test hook",
                context,
                grad_shape,
                input_cache_.Shape(),
                in_features_,
                out_features_,
                false);
        } else {
            try {
                af::array grad_gpu;
                af::array input_gpu;

                if (is_batched) {
                    grad_gpu =
                        grad_output.GetArrayRowMajor2D().as(af::dtype::f32);
                    input_gpu =
                        input_cache_.GetArrayRowMajor2D().as(af::dtype::f32);
                } else {
                    grad_gpu = af::moddims(
                        grad_output.GetArray(),
                        1,
                        static_cast<dim_t>(out_features_)).as(af::dtype::f32);
                    input_gpu = af::moddims(
                        input_cache_.GetArray(),
                        1,
                        static_cast<dim_t>(in_features_)).as(af::dtype::f32);
                }

                af::array weight_gpu =
                    weight_.GetArrayRowMajor2D().as(af::dtype::f32);

                af::array weight_grad_gpu =
                    af::matmul(grad_gpu, input_gpu, AF_MAT_TRANS, AF_MAT_NONE);
                weight_grad_gpu.eval();
                weight_grad_ = Tensor::FromArrayRowMajor2D(weight_grad_gpu);

                if (use_bias_) {
                    af::array bias_grad_gpu = af::flat(af::sum(grad_gpu, 0));
                    bias_grad_gpu.eval();
                    bias_grad_ = Tensor(bias_grad_gpu);
                }

                af::array grad_input_gpu = af::matmul(grad_gpu, weight_gpu);
                grad_input_gpu.eval();

                if (is_batched) {
                    return Tensor::FromArrayRowMajor2D(grad_input_gpu);
                }

                return Tensor(af::flat(grad_input_gpu));
            } catch (const af::exception& e) {
                const BackendFallbackReason reason =
                    ClassifyArrayFireBackendFallbackReason(e.what());
                RecordLinearRuntimeFallback(
                    "LinearLayer::Backward",
                    reason,
                    e.what(),
                    context,
                    grad_shape,
                    input_cache_.Shape(),
                    in_features_,
                    out_features_,
                    false);
                const bool log_fallback =
                    ShouldLogArrayFireBackendFallbackOnce(
                        "LinearLayer::Backward", reason, context);
                if (log_fallback) {
                    spdlog::warn("{}",
                                 BuildLinearRuntimeFallbackDetail(
                                     "LinearLayer::Backward",
                                     reason,
                                     e.what(),
                                     context));
                }
            }
        }
    }
#endif

    // CPU fallback implementation
    if (is_batched) {
        const float* grad_output_data = grad_output.ReadData<float>();
        const float* input_data = input_cache_.ReadData<float>();
        float* weight_grad_data = weight_grad_.MutableData<float>();

        // Initialize gradients to zero
        std::memset(weight_grad_data, 0, sizeof(float) * out_features_ * in_features_);

        for (size_t o = 0; o < out_features_; o++) {
            for (size_t i = 0; i < in_features_; i++) {
                float grad_sum = 0.0f;
                for (size_t b = 0; b < batch_size; b++) {
                    grad_sum += grad_output_data[b * out_features_ + o] *
                              input_data[b * in_features_ + i];
                }
                weight_grad_data[o * in_features_ + i] = grad_sum;
            }
        }

        if (use_bias_) {
            float* bias_grad_data = bias_grad_.MutableData<float>();
            std::memset(bias_grad_data, 0, sizeof(float) * out_features_);

            for (size_t b = 0; b < batch_size; b++) {
                for (size_t o = 0; o < out_features_; o++) {
                    bias_grad_data[o] += grad_output_data[b * out_features_ + o];
                }
            }

        }

        // Gradient w.r.t. input
        Tensor grad_input({batch_size, in_features_}, DataType::Float32);
        float* grad_input_data = grad_input.MutableData<float>();
        const float* weight_data = weight_.ReadData<float>();

        for (size_t b = 0; b < batch_size; b++) {
            for (size_t i = 0; i < in_features_; i++) {
                float sum = 0.0f;
                for (size_t o = 0; o < out_features_; o++) {
                    sum += grad_output_data[b * out_features_ + o] *
                          weight_data[o * in_features_ + i];
                }
                grad_input_data[b * in_features_ + i] = sum;
            }
        }

        return grad_input;
    } else {
        // Single sample (1D tensors)
        const float* grad_output_data = grad_output.ReadData<float>();
        const float* input_data = input_cache_.ReadData<float>();
        float* weight_grad_data = weight_grad_.MutableData<float>();

        for (size_t o = 0; o < out_features_; o++) {
            for (size_t i = 0; i < in_features_; i++) {
                weight_grad_data[o * in_features_ + i] = grad_output_data[o] * input_data[i];
            }
        }

        if (use_bias_) {
            std::memcpy(bias_grad_.MutableData(), grad_output.ReadData(),
                        sizeof(float) * out_features_);
        }

        Tensor grad_input({in_features_}, DataType::Float32);
        float* grad_input_data = grad_input.MutableData<float>();
        const float* weight_data = weight_.ReadData<float>();

        for (size_t i = 0; i < in_features_; i++) {
            float sum = 0.0f;
            for (size_t o = 0; o < out_features_; o++) {
                sum += weight_data[o * in_features_ + i] * grad_output_data[o];
            }
            grad_input_data[i] = sum;
        }

        return grad_input;
    }
}

void LinearLayer::BackwardSparseCsr(
    const LinearSparseCsrBatchView& input,
    const Tensor& grad_output) {
    ValidateLinearSparseCsrBatchView(input, in_features_);
    const auto& grad_shape = grad_output.Shape();
    if (grad_shape != std::vector<size_t>{input.rows, out_features_}) {
        throw std::invalid_argument(
            "LinearLayer sparse CSR grad_output must have shape [" +
            std::to_string(input.rows) + ", " +
            std::to_string(out_features_) + "]");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (IsCurrentArrayFireBackendAvailable()) {
        const std::string context = BuildLinearRuntimeFallbackContext(
            in_features_, out_features_, input.rows, use_bias_);
        if (ShouldForceArrayFireBackendFallbackForTesting(
                "LinearLayer::BackwardSparseCsr")) {
            RecordLinearRuntimeFallback(
                "LinearLayer::BackwardSparseCsr",
                BackendFallbackReason::GpuBackendException,
                "forced ArrayFire backend fallback test hook",
                context,
                {input.rows, input.columns},
                {out_features_, in_features_},
                in_features_,
                out_features_,
                use_bias_);
        } else {
            try {
                af::array grad_gpu =
                    grad_output.GetArrayRowMajor2D().as(af::dtype::f32);
                af::array weight_grad_gpu;
                if (input.nnz == 0) {
                    weight_grad_gpu = af::constant(
                        0.0f,
                        static_cast<dim_t>(out_features_),
                        static_cast<dim_t>(in_features_),
                        f32);
                } else {
                    af::array sparse_input = af::sparse(
                        static_cast<dim_t>(input.rows),
                        static_cast<dim_t>(input.columns),
                        static_cast<dim_t>(input.nnz),
                        input.values,
                        input.row_offsets,
                        input.column_indices,
                        f32,
                        AF_STORAGE_CSR,
                        afHost);
                    af::array feature_by_output = af::matmul(
                        sparse_input,
                        grad_gpu,
                        AF_MAT_TRANS,
                        AF_MAT_NONE);
                    weight_grad_gpu = af::transpose(feature_by_output);
                }
                weight_grad_gpu.eval();
                weight_grad_ = Tensor::FromArrayRowMajor2D(weight_grad_gpu);

                if (use_bias_) {
                    af::array bias_grad_gpu = af::flat(af::sum(grad_gpu, 0));
                    bias_grad_gpu.eval();
                    bias_grad_ = Tensor(bias_grad_gpu);
                }
                return;
            } catch (const af::exception& e) {
                const BackendFallbackReason reason =
                    ClassifyArrayFireBackendFallbackReason(e.what());
                RecordLinearRuntimeFallback(
                    "LinearLayer::BackwardSparseCsr",
                    reason,
                    e.what(),
                    context,
                    {input.rows, input.columns},
                    {out_features_, in_features_},
                    in_features_,
                    out_features_,
                    use_bias_);
                if (ShouldLogArrayFireBackendFallbackOnce(
                        "LinearLayer::BackwardSparseCsr", reason, context)) {
                    spdlog::warn("{}", BuildLinearRuntimeFallbackDetail(
                        "LinearLayer::BackwardSparseCsr",
                        reason,
                        e.what(),
                        context));
                }
            }
        }
    }
#endif

    const float* grad_data = grad_output.ReadData<float>();
    float* weight_grad_data = weight_grad_.MutableData<float>();
    std::memset(
        weight_grad_data,
        0,
        sizeof(float) * out_features_ * in_features_);
    if (use_bias_) {
        float* bias_grad_data = bias_grad_.MutableData<float>();
        std::memset(bias_grad_data, 0, sizeof(float) * out_features_);
        for (size_t row = 0; row < input.rows; ++row) {
            for (size_t output_feature = 0;
                 output_feature < out_features_;
                 ++output_feature) {
                bias_grad_data[output_feature] +=
                    grad_data[row * out_features_ + output_feature];
            }
        }
    }
    for (size_t row = 0; row < input.rows; ++row) {
        const size_t begin = static_cast<size_t>(input.row_offsets[row]);
        const size_t end = static_cast<size_t>(input.row_offsets[row + 1]);
        for (size_t index = begin; index < end; ++index) {
            const size_t column =
                static_cast<size_t>(input.column_indices[index]);
            const float value = input.values[index];
            for (size_t output_feature = 0;
                 output_feature < out_features_;
                 ++output_feature) {
                weight_grad_data[
                    output_feature * in_features_ + column] +=
                    grad_data[row * out_features_ + output_feature] * value;
            }
        }
    }
}

std::map<std::string, Tensor> LinearLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    params["weight"] = weight_;
    if (use_bias_) {
        params["bias"] = bias_;
    }
    return params;
}

void LinearLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    auto weight_it = params.find("weight");
    if (weight_it != params.end()) {
        weight_ = weight_it->second.Clone();
    }

    if (use_bias_) {
        auto bias_it = params.find("bias");
        if (bias_it != params.end()) {
            bias_ = bias_it->second.Clone();
        }
    }
}

std::map<std::string, Tensor> LinearLayer::GetGradients() {
    std::map<std::string, Tensor> grads;
    grads["weight"] = weight_grad_;
    if (use_bias_) {
        grads["bias"] = bias_grad_;
    }
    return grads;
}

} // namespace cyxwiz
