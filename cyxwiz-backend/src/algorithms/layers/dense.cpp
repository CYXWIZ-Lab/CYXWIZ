#include "cyxwiz/layers/dense.h"
#include "../arrayfire_backend_utils.h"
#include "layer_arrayfire_utils.h"

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

#ifdef CYXWIZ_HAS_ARRAYFIRE
namespace {

void LogDenseArrayFireFallback(
    const char* operation_name,
    BackendFallbackReason reason,
    const char* error_message,
    const std::string& shape_context) {
    const std::string context =
        BuildArrayFireBackendFallbackContext(shape_context);
    const bool log_fallback =
        ShouldLogArrayFireBackendFallbackOnce(operation_name, reason, context);
    if (log_fallback) {
        spdlog::warn("{}",
                     BuildArrayFireBackendFallbackMessage(
                         operation_name, reason,
                         reason != BackendFallbackReason::CudaJitParamOverflow,
                         error_message, context));
    }
}

} // namespace
#endif

DenseLayer::DenseLayer(int in_features, int out_features, bool use_bias)
    : in_features_(in_features), out_features_(out_features), use_bias_(use_bias) {

#ifdef CYXWIZ_HAS_ARRAYFIRE
    // Initialize weights using Xavier initialization
    af::dim4 weight_dims(out_features, in_features);
    af::array w = XavierUniform(in_features, out_features, weight_dims);
    weights_ = AfToTensor(w);

    if (use_bias_) {
        // Initialize bias to zeros
        af::array b = af::constant(0.0f, af::dim4(out_features));
        bias_ = AfToTensor(b);
    }

    // Initialize gradient accumulators
    grad_weights_ = Tensor::Zeros({static_cast<size_t>(out_features),
                                    static_cast<size_t>(in_features)});
    if (use_bias_) {
        grad_bias_ = Tensor::Zeros({static_cast<size_t>(out_features)});
    }
#else
    // CPU fallback: simple random initialization
    weights_ = Tensor::Random({static_cast<size_t>(out_features),
                                static_cast<size_t>(in_features)});
    if (use_bias_) {
        bias_ = Tensor::Zeros({static_cast<size_t>(out_features)});
    }
    grad_weights_ = Tensor::Zeros({static_cast<size_t>(out_features),
                                    static_cast<size_t>(in_features)});
    if (use_bias_) {
        grad_bias_ = Tensor::Zeros({static_cast<size_t>(out_features)});
    }
#endif
}

Tensor DenseLayer::Forward(const Tensor& input) {
    cached_input_ = input;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (ShouldForceArrayFireBackendFallbackForTesting("DenseLayer::Forward")) {
        LogDenseArrayFireFallback(
            "DenseLayer::Forward",
            BackendFallbackReason::GpuBackendException,
            "forced ArrayFire backend fallback test hook",
            BuildTensorShapeContext("input", input.Shape()));
    } else {
        try {
            af::array x = TensorToAf(input);
            af::array w = TensorToAf(weights_);

            // Ensure x is 2D: [batch_size, in_features]
            // Matrix multiply: output = x @ W^T
            // Where W is [out_features, in_features]
            af::array output = af::matmul(x, af::transpose(w));
            output.eval();

            if (use_bias_) {
                af::array b = TensorToAf(bias_);
                // Broadcast row bias across the batch dimension. `output` is
                // semantic row-major [batch, out_features].
                output = output + af::tile(
                    af::transpose(b),
                    static_cast<unsigned int>(output.dims(0)));
                output.eval();
            }

            return AfToTensor(output);
        } catch (const af::exception& e) {
            const BackendFallbackReason reason =
                ClassifyArrayFireBackendFallbackReason(e.what());
            LogDenseArrayFireFallback(
                "DenseLayer::Forward", reason, e.what(),
                BuildTensorShapeContext("input", input.Shape()));
        }
    }
#endif

    const std::vector<size_t>& input_shape = input.Shape();
    const bool is_batched = input_shape.size() == 2;
    if (!is_batched && input_shape.size() != 1) {
        throw std::runtime_error("Dense forward expects a 1D or 2D Float32 input tensor");
    }
    if (input.GetDataType() != DataType::Float32 ||
        weights_.GetDataType() != DataType::Float32 ||
        (use_bias_ && bias_.GetDataType() != DataType::Float32)) {
        throw std::runtime_error("Dense forward CPU fallback requires Float32 tensors");
    }

    const size_t batch_size = is_batched ? input_shape[0] : 1;
    const size_t input_features = is_batched ? input_shape[1] : input_shape[0];
    if (input_features != static_cast<size_t>(in_features_)) {
        throw std::runtime_error("Dense forward input feature mismatch");
    }

    Tensor output(is_batched
                      ? std::vector<size_t>{batch_size, static_cast<size_t>(out_features_)}
                      : std::vector<size_t>{static_cast<size_t>(out_features_)},
                  DataType::Float32);
    const float* input_data = input.Data<float>();
    const float* weight_data = weights_.Data<float>();
    const float* bias_data = use_bias_ ? bias_.Data<float>() : nullptr;
    float* output_data = output.Data<float>();

    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t out = 0; out < static_cast<size_t>(out_features_); ++out) {
            float sum = use_bias_ ? bias_data[out] : 0.0f;
            for (size_t in = 0; in < static_cast<size_t>(in_features_); ++in) {
                const size_t input_index = is_batched ? batch * static_cast<size_t>(in_features_) + in : in;
                sum += input_data[input_index] *
                       weight_data[out * static_cast<size_t>(in_features_) + in];
            }
            const size_t output_index = is_batched ? batch * static_cast<size_t>(out_features_) + out : out;
            output_data[output_index] = sum;
        }
    }

    return output;
}

Tensor DenseLayer::Backward(const Tensor& grad_output) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (ShouldForceArrayFireBackendFallbackForTesting("DenseLayer::Backward")) {
        LogDenseArrayFireFallback(
            "DenseLayer::Backward",
            BackendFallbackReason::GpuBackendException,
            "forced ArrayFire backend fallback test hook",
            BuildTensorShapeContext("grad_output", grad_output.Shape()));
    } else {
        try {
            af::array grad_out = TensorToAf(grad_output);
            af::array x = TensorToAf(cached_input_);
            af::array w = TensorToAf(weights_);

            // Gradient w.r.t weights: dW = grad_out^T @ x
            af::array dW = af::matmul(af::transpose(grad_out), x);
            dW.eval();
            grad_weights_ = AfToTensor(dW);

            // Gradient w.r.t bias: db = sum(grad_out, axis=0)
            if (use_bias_) {
                af::array db = af::sum(grad_out, 0);
                db.eval();
                db = af::moddims(db, af::dim4(db.elements()));
                db.eval();
                grad_bias_ = AfToTensor(db);
            }

            // Gradient w.r.t input: dx = grad_out @ W
            af::array dx = af::matmul(grad_out, w);
            dx.eval();

            return AfToTensor(dx);
        } catch (const af::exception& e) {
            const BackendFallbackReason reason =
                ClassifyArrayFireBackendFallbackReason(e.what());
            LogDenseArrayFireFallback(
                "DenseLayer::Backward", reason, e.what(),
                BuildTensorShapeContext("grad_output", grad_output.Shape()));
        }
    }
#endif

    const std::vector<size_t>& input_shape = cached_input_.Shape();
    const std::vector<size_t>& grad_shape = grad_output.Shape();
    const bool is_batched = input_shape.size() == 2;
    if (!is_batched && input_shape.size() != 1) {
        throw std::runtime_error("Dense backward expects cached 1D or 2D input");
    }
    const std::vector<size_t> expected_grad_shape =
        is_batched ? std::vector<size_t>{input_shape[0], static_cast<size_t>(out_features_)}
                   : std::vector<size_t>{static_cast<size_t>(out_features_)};
    if (grad_shape != expected_grad_shape) {
        throw std::runtime_error("Dense backward gradient shape mismatch");
    }
    if (grad_output.GetDataType() != DataType::Float32 ||
        cached_input_.GetDataType() != DataType::Float32 ||
        weights_.GetDataType() != DataType::Float32) {
        throw std::runtime_error("Dense backward CPU fallback requires Float32 tensors");
    }

    const size_t batch_size = is_batched ? input_shape[0] : 1;
    Tensor grad_input(is_batched
                          ? std::vector<size_t>{batch_size, static_cast<size_t>(in_features_)}
                          : std::vector<size_t>{static_cast<size_t>(in_features_)},
                      DataType::Float32);
    grad_weights_ = Tensor::Zeros({static_cast<size_t>(out_features_),
                                   static_cast<size_t>(in_features_)});
    if (use_bias_) {
        grad_bias_ = Tensor::Zeros({static_cast<size_t>(out_features_)});
    }

    const float* grad_output_data = grad_output.Data<float>();
    const float* input_data = cached_input_.Data<float>();
    const float* weight_data = weights_.Data<float>();
    float* grad_input_data = grad_input.Data<float>();
    float* grad_weight_data = grad_weights_.Data<float>();
    float* grad_bias_data = use_bias_ ? grad_bias_.Data<float>() : nullptr;

    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t in = 0; in < static_cast<size_t>(in_features_); ++in) {
            float grad_sum = 0.0f;
            for (size_t out = 0; out < static_cast<size_t>(out_features_); ++out) {
                const size_t grad_output_index =
                    is_batched ? batch * static_cast<size_t>(out_features_) + out : out;
                grad_sum += grad_output_data[grad_output_index] *
                            weight_data[out * static_cast<size_t>(in_features_) + in];
            }
            const size_t grad_input_index =
                is_batched ? batch * static_cast<size_t>(in_features_) + in : in;
            grad_input_data[grad_input_index] = grad_sum;
        }

        for (size_t out = 0; out < static_cast<size_t>(out_features_); ++out) {
            const size_t grad_output_index =
                is_batched ? batch * static_cast<size_t>(out_features_) + out : out;
            if (use_bias_) {
                grad_bias_data[out] += grad_output_data[grad_output_index];
            }
            for (size_t in = 0; in < static_cast<size_t>(in_features_); ++in) {
                const size_t input_index = is_batched ? batch * static_cast<size_t>(in_features_) + in : in;
                grad_weight_data[out * static_cast<size_t>(in_features_) + in] +=
                    grad_output_data[grad_output_index] * input_data[input_index];
            }
        }
    }

    return grad_input;
}

std::map<std::string, Tensor> DenseLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    params["weights"] = weights_;
    params["grad_weights"] = grad_weights_;
    if (use_bias_) {
        params["bias"] = bias_;
        params["grad_bias"] = grad_bias_;
    }
    return params;
}

void DenseLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    if (params.count("weights")) {
        weights_ = params.at("weights");
    }
    if (params.count("bias") && use_bias_) {
        bias_ = params.at("bias");
    }
}

} // namespace cyxwiz
