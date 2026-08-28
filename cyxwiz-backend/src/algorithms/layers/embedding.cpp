#include "cyxwiz/layers/embedding.h"
#include "layer_arrayfire_utils.h"
#include "cyxwiz/backend_placement_observation.h"
#include "../arrayfire_backend_utils.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <random>
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

af::array TokenRowsToSemanticOutput(const af::array& token_rows,
                                    bool is_batched,
                                    size_t batch_size,
                                    size_t sequence_length,
                                    size_t embedding_dim) {
    if (!is_batched) {
        return token_rows;
    }
    af::array row_major_linear = af::flat(af::transpose(token_rows));
    af::array reversed = af::moddims(
        row_major_linear,
        af::dim4(CheckedIntDim(embedding_dim, "embedding dimension"),
                 CheckedIntDim(sequence_length, "embedding sequence length"),
                 CheckedIntDim(batch_size, "embedding batch size")));
    return af::reorder(reversed, 2, 1, 0);
}

af::array SemanticGradientToTokenRows(const af::array& gradient,
                                      bool is_batched,
                                      size_t batch_size,
                                      size_t sequence_length,
                                      size_t embedding_dim) {
    if (!is_batched) {
        return gradient;
    }
    af::array row_major_linear = af::flat(af::reorder(gradient, 2, 1, 0));
    af::array reversed = af::moddims(
        row_major_linear,
        af::dim4(CheckedIntDim(embedding_dim, "embedding gradient dimension"),
                 CheckedIntDim(batch_size * sequence_length,
                               "embedding gradient token count")));
    return af::transpose(reversed);
}

} // namespace

static std::string BuildEmbeddingContext(const Tensor& tensor) {
    return BuildArrayFireBackendFallbackContext(
        BuildTensorShapeContext("input", tensor.Shape()));
}

static void LogEmbeddingFallbackOnce(
    const char* operation_name,
    const Tensor& tensor,
    size_t num_embeddings,
    size_t embedding_dim,
    const char* error_message)
{
    const BackendFallbackReason reason = ClassifyArrayFireBackendFallbackReason(error_message);
    const std::string context = BuildEmbeddingContext(tensor);
    const std::string message = BuildArrayFireBackendFallbackMessage(
        operation_name,
        reason,
        reason != BackendFallbackReason::CudaJitParamOverflow,
        error_message,
        context);
    RecordBackendPlacementObservationForActiveDevice(
        "Embedding",
        CurrentArrayFireBackendName(),
        "int32",
        BuildEmbeddingPlacementShapeSignature(
            num_embeddings, embedding_dim, tensor.Shape(), "int32"),
        BackendFallbackReasonName(reason),
        BackendPlacementObservationSource::RuntimeFallback,
        message);
    ThrowIfArrayFireNativeCpuFallbackForbidden(
        operation_name,
        reason,
        error_message,
        context);
    if (ShouldLogArrayFireBackendFallbackOnce(operation_name, reason, context)) {
        spdlog::warn("{}", message);
    }
}

#endif

EmbeddingLayer::EmbeddingLayer(int num_embeddings, int embedding_dim,
                               int padding_idx, float max_norm)
    : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim),
      padding_idx_(padding_idx), max_norm_(max_norm) {

    if (num_embeddings_ < 2) {
        throw std::invalid_argument(
            "EmbeddingLayer: num_embeddings must be >= 2");
    }
    if (embedding_dim_ < 1) {
        throw std::invalid_argument(
            "EmbeddingLayer: embedding_dim must be >= 1");
    }
    if (padding_idx_ < -1 || padding_idx_ >= num_embeddings_) {
        throw std::invalid_argument(
            "EmbeddingLayer: padding_idx must be -1 or a valid token id");
    }
    if (!std::isfinite(max_norm_) || max_norm_ < 0.0f) {
        throw std::invalid_argument(
            "EmbeddingLayer: max_norm must be finite and >= 0");
    }

    InitializeWeights();
}

void EmbeddingLayer::InitializeWeights() {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    // Initialize with normal distribution N(0, 1)
    af::array w = af::randn(af::dim4(num_embeddings_, embedding_dim_), af::dtype::f32);
    w.eval();
    if (padding_idx_ >= 0 && padding_idx_ < num_embeddings_) {
        w(padding_idx_, af::span) = 0.0f;
        w.eval();
    }
    weight_ = Tensor::FromSemanticArray(
        w, {static_cast<size_t>(num_embeddings_),
            static_cast<size_t>(embedding_dim_)});
#else
    weight_ = Tensor::Random({static_cast<size_t>(num_embeddings_),
                               static_cast<size_t>(embedding_dim_)});
#endif

    grad_weight_ = Tensor::Zeros({static_cast<size_t>(num_embeddings_),
                                   static_cast<size_t>(embedding_dim_)});
}

void EmbeddingLayer::NormalizeEmbeddings() {
    if (max_norm_ <= 0.0f) return;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array w = weight_.GetSemanticArray();

        // Compute L2 norm for each embedding
        af::array norms = af::sqrt(af::sum(w * w, 1));
        norms.eval();

        // Create scaling factors (clip to max_norm)
        af::array scale = af::min(max_norm_ / (norms + 1e-8f), 1.0f);
        scale.eval();

        // Apply scaling
        w = w * af::tile(scale, 1, embedding_dim_);
        w.eval();

        weight_ = Tensor::FromSemanticArray(
            w, {static_cast<size_t>(num_embeddings_),
                static_cast<size_t>(embedding_dim_)});
        return;
    } catch (const af::exception& e) {
        LogEmbeddingFallbackOnce(
            "EmbeddingLayer::NormalizeEmbeddings",
            weight_,
            static_cast<size_t>(num_embeddings_),
            static_cast<size_t>(embedding_dim_),
            e.what());
    }
#endif

    float* weights = weight_.MutableData<float>();
    for (int row = 0; row < num_embeddings_; ++row) {
        double squared_norm = 0.0;
        for (int col = 0; col < embedding_dim_; ++col) {
            const double value = weights[row * embedding_dim_ + col];
            squared_norm += value * value;
        }
        const double norm = std::sqrt(squared_norm);
        if (norm > static_cast<double>(max_norm_)) {
            const float scale = static_cast<float>(max_norm_ / norm);
            for (int col = 0; col < embedding_dim_; ++col) {
                weights[row * embedding_dim_ + col] *= scale;
            }
        }
    }
}

Tensor EmbeddingLayer::Forward(const Tensor& input) {
    const auto& shape = input.Shape();
    if (input.GetDataType() != DataType::Int32) {
        throw std::invalid_argument(
            "EmbeddingLayer::Forward: input token ids must be Int32");
    }
    if (shape.size() != 1 && shape.size() != 2) {
        throw std::invalid_argument(
            "EmbeddingLayer::Forward: input must be [sequence] or [batch, sequence]");
    }
    const size_t batch_size = shape.size() == 2 ? shape[0] : 1;
    const size_t sequence_length = shape.size() == 2 ? shape[1] : shape[0];
    if (batch_size == 0 || sequence_length == 0 ||
        batch_size > std::numeric_limits<size_t>::max() / sequence_length) {
        throw std::invalid_argument(
            "EmbeddingLayer::Forward: input dimensions must be nonzero and bounded");
    }
    const size_t total_indices = batch_size * sequence_length;

    // Cache indices for backward pass
    cached_indices_ = input.Clone();

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        if (max_norm_ > 0.0f) {
            NormalizeEmbeddings();
        }

        // Token ids are host ingress for the current batcher contract. The
        // lookup table, gathered vectors, and gradients remain on ArrayFire.
        const int32_t* indices_ptr = input.ReadData<int32_t>();
        af::array w = weight_.GetSemanticArray();
        af::array output_flat = af::constant(
            0.0f,
            af::dim4(CheckedIntDim(total_indices, "embedding token count"),
                     embedding_dim_));
        for (size_t i = 0; i < total_indices; ++i) {
            int32_t idx = indices_ptr[i];
            if (idx >= 0 && idx < num_embeddings_ && idx != padding_idx_) {
                output_flat(CheckedIntDim(i, "embedding index"), af::span) =
                    w(idx, af::span);
            }
        }
        output_flat.eval();

        af::array output = TokenRowsToSemanticOutput(
            output_flat, shape.size() == 2, batch_size, sequence_length,
            static_cast<size_t>(embedding_dim_));
        output.eval();
        const std::vector<size_t> output_shape = shape.size() == 2
            ? std::vector<size_t>{batch_size, sequence_length,
                                  static_cast<size_t>(embedding_dim_)}
            : std::vector<size_t>{sequence_length,
                                  static_cast<size_t>(embedding_dim_)};
        return Tensor::FromSemanticArray(output, output_shape);
    } catch (const af::exception& e) {
        LogEmbeddingFallbackOnce(
            "EmbeddingLayer::Forward",
            input,
            static_cast<size_t>(num_embeddings_),
            static_cast<size_t>(embedding_dim_),
            e.what());
    }
#endif

    // CPU fallback
    std::vector<size_t> out_shape;
    if (shape.size() == 2) {
        out_shape = {batch_size, sequence_length,
                     static_cast<size_t>(embedding_dim_)};
    } else {
        out_shape = {sequence_length, static_cast<size_t>(embedding_dim_)};
    }

    Tensor output(out_shape, DataType::Float32);
    float* out_data = output.MutableData<float>();
    const float* weight_data = weight_.ReadData<float>();
    const int32_t* indices = input.ReadData<int32_t>();

    for (size_t i = 0; i < total_indices; ++i) {
        int32_t idx = indices[i];
        if (idx >= 0 && idx < num_embeddings_ && idx != padding_idx_) {
            std::memcpy(out_data + i * embedding_dim_,
                       weight_data + idx * embedding_dim_,
                       embedding_dim_ * sizeof(float));
        } else {
            std::memset(out_data + i * embedding_dim_, 0, embedding_dim_ * sizeof(float));
        }
    }

    return output;
}

Tensor EmbeddingLayer::Backward(const Tensor& grad_output) {
    if (frozen_) {
        // Return empty tensor - no gradient needed for frozen embeddings
        return Tensor();
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const auto& shape = cached_indices_.Shape();
        if (shape.size() != 1 && shape.size() != 2) {
            throw std::runtime_error(
                "EmbeddingLayer::Backward called before a valid Forward");
        }
        const size_t batch_size = shape.size() == 2 ? shape[0] : 1;
        const size_t sequence_length = shape.size() == 2 ? shape[1] : shape[0];
        const size_t total_indices = batch_size * sequence_length;
        const std::vector<size_t> expected_grad_shape = shape.size() == 2
            ? std::vector<size_t>{batch_size, sequence_length,
                                  static_cast<size_t>(embedding_dim_)}
            : std::vector<size_t>{sequence_length,
                                  static_cast<size_t>(embedding_dim_)};
        if (grad_output.GetDataType() != DataType::Float32 ||
            grad_output.Shape() != expected_grad_shape) {
            throw std::invalid_argument(
                "EmbeddingLayer::Backward: grad_output shape or dtype mismatch");
        }

        const int32_t* indices_ptr = cached_indices_.ReadData<int32_t>();

        // Initialize gradient accumulator
        af::array dw = af::constant(0.0f, af::dim4(num_embeddings_, embedding_dim_));

        af::array grad = SemanticGradientToTokenRows(
            grad_output.GetSemanticArray(), shape.size() == 2, batch_size,
            sequence_length,
            static_cast<size_t>(embedding_dim_));
        grad.eval();

        // Scatter-add gradients to the weight matrix
        for (size_t i = 0; i < total_indices; ++i) {
            int32_t idx = indices_ptr[i];
            if (idx >= 0 && idx < num_embeddings_ && idx != padding_idx_) {
                dw(idx, af::span) += grad(CheckedIntDim(i, "embedding grad index"),
                                          af::span);
            }
        }
        dw.eval();

        grad_weight_ = Tensor::FromSemanticArray(
            dw, {static_cast<size_t>(num_embeddings_),
                 static_cast<size_t>(embedding_dim_)});

        // Return empty tensor (no gradient w.r.t. integer indices)
        return Tensor();
    } catch (const af::exception& e) {
        LogEmbeddingFallbackOnce(
            "EmbeddingLayer::Backward",
            cached_indices_,
            static_cast<size_t>(num_embeddings_),
            static_cast<size_t>(embedding_dim_),
            e.what());
    }
#endif

    // CPU fallback
    const auto& shape = cached_indices_.Shape();
    if (shape.size() != 1 && shape.size() != 2) {
        throw std::runtime_error(
            "EmbeddingLayer::Backward called before a valid Forward");
    }
    const size_t batch_size = shape.size() == 2 ? shape[0] : 1;
    const size_t sequence_length = shape.size() == 2 ? shape[1] : shape[0];
    const size_t total = batch_size * sequence_length;
    const std::vector<size_t> expected_grad_shape = shape.size() == 2
        ? std::vector<size_t>{batch_size, sequence_length,
                              static_cast<size_t>(embedding_dim_)}
        : std::vector<size_t>{sequence_length,
                              static_cast<size_t>(embedding_dim_)};
    if (grad_output.GetDataType() != DataType::Float32 ||
        grad_output.Shape() != expected_grad_shape) {
        throw std::invalid_argument(
            "EmbeddingLayer::Backward: grad_output shape or dtype mismatch");
    }

    // Zero out gradient
    grad_weight_ = Tensor::Zeros({static_cast<size_t>(num_embeddings_),
                                   static_cast<size_t>(embedding_dim_)});
    float* dw = grad_weight_.MutableData<float>();
    const float* grad_data = grad_output.ReadData<float>();
    const int32_t* indices = cached_indices_.ReadData<int32_t>();

    // Scatter-add gradients
    for (size_t i = 0; i < total; i++) {
        int32_t idx = indices[i];
        if (idx >= 0 && idx < num_embeddings_ && idx != padding_idx_) {
            for (int j = 0; j < embedding_dim_; j++) {
                dw[idx * embedding_dim_ + j] += grad_data[i * embedding_dim_ + j];
            }
        }
    }

    return Tensor();
}

Tensor EmbeddingLayer::GetEmbedding(int index) const {
    if (index < 0 || index >= num_embeddings_) {
        throw std::out_of_range("Embedding index out of range");
    }

    Tensor result({static_cast<size_t>(embedding_dim_)}, DataType::Float32);
    const float* src =
        weight_.ReadData<float>() + index * embedding_dim_;
    std::memcpy(result.MutableData<float>(), src,
                embedding_dim_ * sizeof(float));
    return result;
}

void EmbeddingLayer::SetEmbedding(int index, const Tensor& embedding) {
    if (index < 0 || index >= num_embeddings_) {
        throw std::out_of_range("Embedding index out of range");
    }
    if (embedding.GetDataType() != DataType::Float32 ||
        embedding.NumElements() != static_cast<size_t>(embedding_dim_)) {
        throw std::invalid_argument("Embedding dimension mismatch");
    }

    float* dst =
        weight_.MutableData<float>() + index * embedding_dim_;
    std::memcpy(dst, embedding.ReadData<float>(),
                embedding_dim_ * sizeof(float));
}

void EmbeddingLayer::LoadPretrainedWeights(const Tensor& weights, bool freeze) {
    const auto& shape = weights.Shape();
    if (weights.GetDataType() != DataType::Float32 || shape.size() != 2 ||
        shape[0] != static_cast<size_t>(num_embeddings_) ||
        shape[1] != static_cast<size_t>(embedding_dim_)) {
        throw std::invalid_argument(
            "Embedding pretrained weights must be a Float32 matrix with "
            "shape [num_embeddings, embedding_dim]");
    }

    weight_ = weights.Clone();
    frozen_ = freeze;

    // Ensure padding index is zero
    if (padding_idx_ >= 0 && padding_idx_ < num_embeddings_) {
        float* data = weight_.MutableData<float>();
        for (int i = 0; i < embedding_dim_; i++) {
            data[padding_idx_ * embedding_dim_ + i] = 0.0f;
        }
    }
}

std::map<std::string, Tensor> EmbeddingLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    params["weight"] = weight_;
    return params;
}

std::map<std::string, Tensor> EmbeddingLayer::GetGradients() {
    // Match LinearLayer convention: one entry per trainable parameter.
    // Used by EmbeddingModule so the optimizer can update weights through
    // the standard Module::GetGradients() path.
    std::map<std::string, Tensor> grads;
    grads["weight"] = grad_weight_;
    return grads;
}

void EmbeddingLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    if (params.count("weight")) {
        LoadPretrainedWeights(params.at("weight"), frozen_);
    }
}

} // namespace cyxwiz
