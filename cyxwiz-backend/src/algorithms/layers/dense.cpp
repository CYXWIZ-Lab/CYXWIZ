#include "cyxwiz/layers/dense.h"
#include "../arrayfire_backend_utils.h"
#include "layer_arrayfire_utils.h"
#include "cyxwiz/backend_placement_observation.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace cyxwiz {

DenseLayer::DenseLayer(int in_features, int out_features, bool use_bias)
    : in_features_(in_features), out_features_(out_features), use_bias_(use_bias) {

    if (in_features_ <= 0 || out_features_ <= 0) {
        throw std::invalid_argument(
            "DenseLayer: in_features and out_features must be positive");
    }

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
    throw std::runtime_error("Dense runs on ArrayFire, and this build has no ArrayFire");
#endif
}

Tensor DenseLayer::Forward(const Tensor& input) {
    const std::vector<size_t>& input_shape = input.Shape();
    const bool is_batched = input_shape.size() == 2;
    if (!is_batched && input_shape.size() != 1) {
        throw std::invalid_argument(
            "DenseLayer: input must be a rank-1 or rank-2 Float32 tensor");
    }
    if (input.GetDataType() != DataType::Float32) {
        throw std::invalid_argument("DenseLayer: input must be Float32");
    }
    const size_t batch_size = is_batched ? input_shape[0] : 1;
    const size_t input_features = is_batched ? input_shape[1] : input_shape[0];
    if (batch_size == 0 || input_features == 0) {
        throw std::invalid_argument(
            "DenseLayer: input dimensions must be nonzero");
    }
    if (input_features != static_cast<size_t>(in_features_)) {
        throw std::invalid_argument(
            "DenseLayer: input feature mismatch; expected " +
            std::to_string(in_features_) + ", got " +
            std::to_string(input_features));
    }

    cached_input_ = input.Clone();
    has_cached_input_ = false;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = is_batched
            ? input.GetArrayRowMajor2D().as(af::dtype::f32)
            : af::moddims(input.GetSemanticArray(), 1,
                          static_cast<dim_t>(in_features_))
                  .as(af::dtype::f32);
        af::array w = weights_.GetArrayRowMajor2D().as(af::dtype::f32);

        // Ensure x is 2D: [batch_size, in_features]
        // Matrix multiply: output = x @ W^T
        // Where W is [out_features, in_features]
        af::array output =
            af::matmul(x, w, AF_MAT_NONE, AF_MAT_TRANS);
        output.eval();

        if (use_bias_) {
            af::array b = af::moddims(
                bias_.GetSemanticArray(), 1,
                static_cast<dim_t>(out_features_)).as(af::dtype::f32);
            // Broadcast row bias across the batch dimension. `output` is
            // semantic row-major [batch, out_features].
            output = output + af::tile(
                b, static_cast<unsigned int>(batch_size), 1);
            output.eval();
        }

        Tensor result = is_batched
            ? Tensor::FromArrayRowMajor2D(output)
            : Tensor::FromSemanticArray(
                  af::flat(output),
                  {static_cast<size_t>(out_features_)});
        has_cached_input_ = true;
        return result;
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("DenseLayer::Forward failed on the ArrayFire device: ") + e.what());
    }
#else
    throw std::runtime_error("Dense runs on ArrayFire, and this build has no ArrayFire");
#endif
}

Tensor DenseLayer::Backward(const Tensor& grad_output) {
    if (!has_cached_input_) {
        throw std::logic_error(
            "DenseLayer: Backward called before a successful Forward");
    }
    const std::vector<size_t>& input_shape = cached_input_.Shape();
    const std::vector<size_t>& grad_shape = grad_output.Shape();
    const bool is_batched = input_shape.size() == 2;
    const std::vector<size_t> expected_grad_shape =
        is_batched
            ? std::vector<size_t>{input_shape[0],
                                  static_cast<size_t>(out_features_)}
            : std::vector<size_t>{static_cast<size_t>(out_features_)};
    if (grad_shape != expected_grad_shape) {
        throw std::invalid_argument(
            "DenseLayer: grad_output shape mismatch");
    }
    if (grad_output.GetDataType() != DataType::Float32) {
        throw std::invalid_argument(
            "DenseLayer: grad_output must be Float32");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = is_batched
            ? grad_output.GetArrayRowMajor2D().as(af::dtype::f32)
            : af::moddims(grad_output.GetSemanticArray(), 1,
                          static_cast<dim_t>(out_features_))
                  .as(af::dtype::f32);
        af::array x = is_batched
            ? cached_input_.GetArrayRowMajor2D().as(af::dtype::f32)
            : af::moddims(cached_input_.GetSemanticArray(), 1,
                          static_cast<dim_t>(in_features_))
                  .as(af::dtype::f32);
        af::array w = weights_.GetArrayRowMajor2D().as(af::dtype::f32);

        // Gradient w.r.t weights: dW = grad_out^T @ x
        af::array dW =
            af::matmul(grad_out, x, AF_MAT_TRANS, AF_MAT_NONE);
        dW.eval();
        grad_weights_ = Tensor::FromArrayRowMajor2D(dW);

        // Gradient w.r.t bias: db = sum(grad_out, axis=0)
        if (use_bias_) {
            af::array db = af::sum(grad_out, 0);
            db.eval();
            db = af::moddims(db, af::dim4(db.elements()));
            db.eval();
            grad_bias_ = Tensor::FromSemanticArray(
                db, {static_cast<size_t>(out_features_)});
        }

        // Gradient w.r.t input: dx = grad_out @ W
        af::array dx = af::matmul(grad_out, w);
        dx.eval();

        return is_batched
            ? Tensor::FromArrayRowMajor2D(dx)
            : Tensor::FromSemanticArray(
                  af::flat(dx),
                  {static_cast<size_t>(in_features_)});
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("DenseLayer::Backward failed on the ArrayFire device: ") + e.what());
    }
#else
    throw std::runtime_error("Dense runs on ArrayFire, and this build has no ArrayFire");
#endif
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

std::map<std::string, Tensor> DenseLayer::GetGradients() const {
    std::map<std::string, Tensor> gradients;
    gradients["weights"] = grad_weights_;
    if (use_bias_) {
        gradients["bias"] = grad_bias_;
    }
    return gradients;
}

void DenseLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    const auto weights = params.find("weights");
    if (weights != params.end()) {
        if (weights->second.GetDataType() != DataType::Float32 ||
            weights->second.Shape() !=
                std::vector<size_t>{static_cast<size_t>(out_features_),
                                    static_cast<size_t>(in_features_)}) {
            throw std::invalid_argument(
                "DenseLayer: weights must be Float32 [out_features, in_features]");
        }
    }
    const auto bias = params.find("bias");
    if (bias != params.end()) {
        if (!use_bias_) {
            throw std::invalid_argument(
                "DenseLayer: bias supplied to a bias-free layer");
        }
        if (bias->second.GetDataType() != DataType::Float32 ||
            bias->second.Shape() !=
                std::vector<size_t>{static_cast<size_t>(out_features_)}) {
            throw std::invalid_argument(
                "DenseLayer: bias must be Float32 [out_features]");
        }
    }

    // Commit only after every supplied parameter has passed validation so a
    // rejected update cannot leave the layer in a partially modified state.
    if (weights != params.end()) {
        weights_ = weights->second.Clone();
    }
    if (bias != params.end()) {
        bias_ = bias->second.Clone();
    }
}

} // namespace cyxwiz
