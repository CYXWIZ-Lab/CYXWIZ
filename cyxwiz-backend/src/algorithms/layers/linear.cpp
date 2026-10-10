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

        spdlog::debug("LinearLayer({}, {}) initialized with Xavier (ArrayFire)", in_features_, out_features_);
        return;
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("LinearLayer::InitializeWeights failed on the ArrayFire device: ") + e.what());
    }
#else
    (void)limit;
    throw std::runtime_error("Linear runs on ArrayFire, and this build has no ArrayFire");
#endif
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
        throw std::runtime_error(std::string("LinearLayer::Forward failed on the ArrayFire device: ") + e.what());
    }
#else
    throw std::runtime_error("Linear runs on ArrayFire, and this build has no ArrayFire");
#endif
}

Tensor LinearLayer::ForwardSparseCsr(
    const LinearSparseCsrBatchView& input) {
    ValidateLinearSparseCsrBatchView(input, in_features_);

#ifdef CYXWIZ_HAS_ARRAYFIRE
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
        throw std::runtime_error(std::string("LinearLayer::ForwardSparseCsr failed on the ArrayFire device: ") + e.what());
    }
#else
    throw std::runtime_error("Linear runs on ArrayFire, and this build has no ArrayFire");
#endif
}

Tensor LinearLayer::ForwardSequence(const Tensor& input) {
    const auto& shape = input.Shape();
    if (shape.size() != 3 || shape[2] != in_features_) {
        throw std::runtime_error("LinearLayer::ForwardSequence: input must be [batch, seq, in_features]");
    }
    const size_t positions = shape[0] * shape[1];
#ifdef CYXWIZ_HAS_ARRAYFIRE
    {
        try {
            const af::array x = af::moddims(input.GetSemanticArray().as(af::dtype::f32),
                                            static_cast<dim_t>(positions), static_cast<dim_t>(in_features_));
            const af::array w = weight_.GetArrayRowMajor2D().as(af::dtype::f32);
            af::array y = af::matmul(x, w, AF_MAT_NONE, AF_MAT_TRANS);
            if (use_bias_) {
                const af::array b = af::moddims(bias_.GetArray(), 1, static_cast<dim_t>(out_features_)).as(af::dtype::f32);
                y = y + af::tile(b, static_cast<unsigned int>(positions), 1);
            }
            y = af::moddims(y, static_cast<dim_t>(shape[0]), static_cast<dim_t>(shape[1]),
                            static_cast<dim_t>(out_features_));
            y.eval();
            input_cache_ = input;
            return Tensor::FromSemanticArray(y, {shape[0], shape[1], out_features_});
        } catch (const af::exception& e) {
            throw std::runtime_error(std::string("LinearLayer::ForwardSequence failed on the ArrayFire device: ") + e.what());
        }
    }
#endif
    Tensor flat = Forward(input.Reshape({positions, in_features_}));
    return flat.Reshape({shape[0], shape[1], out_features_});
}

Tensor LinearLayer::BackwardSequence(const Tensor& grad_output) {
    const auto& shape = grad_output.Shape();
    if (shape.size() != 3 || shape[2] != out_features_) {
        throw std::runtime_error("LinearLayer::BackwardSequence: grad must be [batch, seq, out_features]");
    }
    const size_t positions = shape[0] * shape[1];
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (input_cache_.Shape().size() == 3) {
        try {
            const af::array dy = af::moddims(grad_output.GetSemanticArray().as(af::dtype::f32),
                                             static_cast<dim_t>(positions), static_cast<dim_t>(out_features_));
            const af::array x = af::moddims(input_cache_.GetSemanticArray().as(af::dtype::f32),
                                            static_cast<dim_t>(positions), static_cast<dim_t>(in_features_));
            const af::array w = weight_.GetArrayRowMajor2D().as(af::dtype::f32);
            {
                ScopedProfileSpan span("Linear.sequence_backward.dw");
                af::array dw = af::matmul(dy, x, AF_MAT_TRANS, AF_MAT_NONE);
                dw.eval();
                weight_grad_ = Tensor::FromArrayRowMajor2D(dw);
            }
            if (use_bias_) {
                ScopedProfileSpan span("Linear.sequence_backward.db");
                af::array db = af::flat(af::sum(dy, 0));
                db.eval();
                bias_grad_ = Tensor(db);
            }
            ScopedProfileSpan span("Linear.sequence_backward.dx");
            af::array dx = af::moddims(af::matmul(dy, w), static_cast<dim_t>(shape[0]),
                                       static_cast<dim_t>(shape[1]), static_cast<dim_t>(in_features_));
            dx.eval();
            return Tensor::FromSemanticArray(dx, {shape[0], shape[1], in_features_});
        } catch (const af::exception& e) {
            throw std::runtime_error(std::string("LinearLayer::BackwardSequence failed on the ArrayFire device: ") + e.what());
        }
    }
#endif
    if (input_cache_.Shape().size() == 3) input_cache_ = input_cache_.Reshape({positions, in_features_});
    Tensor flat = Backward(grad_output.Reshape({positions, out_features_}));
    return flat.Reshape({shape[0], shape[1], in_features_});
}

Tensor LinearLayer::Backward(const Tensor& grad_output) {
    const auto& grad_shape = grad_output.Shape();
    const auto& input_shape = input_cache_.Shape();
    (void)input_shape;  // Suppress unused variable warning
    bool is_batched = grad_shape.size() == 2;

#ifdef CYXWIZ_HAS_ARRAYFIRE
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
        throw std::runtime_error(std::string("LinearLayer::Backward failed on the ArrayFire device: ") + e.what());
    }
#else
    throw std::runtime_error("Linear runs on ArrayFire, and this build has no ArrayFire");
#endif
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
        throw std::runtime_error(std::string("LinearLayer::BackwardSparseCsr failed on the ArrayFire device: ") + e.what());
    }
#else
    throw std::runtime_error("Linear runs on ArrayFire, and this build has no ArrayFire");
#endif
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
