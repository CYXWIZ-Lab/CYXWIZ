#include "cyxwiz/layers/embedding.h"
#include "layer_arrayfire_utils.h"
#include "cyxwiz/backend_placement_observation.h"
#include "../arrayfire_backend_utils.h"

#include <cmath>
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
        "cuda",
        "int32",
        BuildEmbeddingPlacementShapeSignature(
            num_embeddings, embedding_dim, tensor.Shape(), "int32"),
        BackendFallbackReasonName(reason),
        BackendPlacementObservationSource::RuntimeFallback,
        message);
    if (ShouldLogArrayFireBackendFallbackOnce(operation_name, reason, context)) {
        spdlog::warn("{}", message);
    }
}

static void LogEmbeddingBackendWarningOnce(
    const char* operation_name,
    const Tensor& tensor,
    const char* error_message)
{
    const BackendFallbackReason reason = ClassifyArrayFireBackendFallbackReason(error_message);
    const std::string context = BuildEmbeddingContext(tensor);
    if (!ShouldLogArrayFireBackendFallbackOnce(operation_name, reason, context)) {
        return;
    }

    std::string message = std::string("ArrayFire ") +
        (operation_name ? operation_name : "embedding operation") +
        " failed (reason=" + BackendFallbackReasonName(reason) +
        "); continuing without the ArrayFire normalization step.";
    if (!context.empty()) {
        message += " Context: ";
        message += context;
        message += ".";
    }
    if (reason != BackendFallbackReason::CudaJitParamOverflow &&
        error_message != nullptr &&
        error_message[0] != '\0') {
        message += " Error: ";
        message += error_message;
    }
    spdlog::warn("{}", message);
}
#endif

EmbeddingLayer::EmbeddingLayer(int num_embeddings, int embedding_dim,
                               int padding_idx, float max_norm)
    : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim),
      padding_idx_(padding_idx), max_norm_(max_norm) {

    InitializeWeights();
}

void EmbeddingLayer::InitializeWeights() {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    // Initialize with normal distribution N(0, 1)
    af::array w = af::randn(af::dim4(num_embeddings_, embedding_dim_), af::dtype::f32);
    w.eval();
    weight_ = AfToTensor(w);

    // Zero out padding index if specified
    if (padding_idx_ >= 0 && padding_idx_ < num_embeddings_) {
        float* data = static_cast<float*>(weight_.Data());
        for (int i = 0; i < embedding_dim_; i++) {
            data[padding_idx_ * embedding_dim_ + i] = 0.0f;
        }
    }
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
        af::array w = TensorToAf(weight_);

        // Compute L2 norm for each embedding
        af::array norms = af::sqrt(af::sum(w * w, 1));
        norms.eval();

        // Create scaling factors (clip to max_norm)
        af::array scale = af::min(max_norm_ / (norms + 1e-8f), 1.0f);
        scale.eval();

        // Apply scaling
        w = w * af::tile(scale, 1, embedding_dim_);
        w.eval();

        weight_ = AfToTensor(w);
    } catch (const af::exception& e) {
        LogEmbeddingBackendWarningOnce(
            "EmbeddingLayer::NormalizeEmbeddings",
            weight_,
            e.what());
    }
#endif
}

Tensor EmbeddingLayer::Forward(const Tensor& input) {
    // Cache indices for backward pass
    cached_indices_ = input.Clone();

#ifdef CYXWIZ_HAS_ARRAYFIRE
    // ArrayFire path is only used for the unbatched case (shape.size()==1).
    // Batched input deliberately falls through to the CPU fallback below
    // because AF's column-major scatter gives the wrong data layout for
    // the next layer. The previous version did `try { throw } catch` per
    // batch which spammed warnings 600+ times per epoch and burned CPU
    // on the exception throw — this gate avoids both.
    if (input.Shape().size() != 2) try {
        // Apply max_norm if specified (single-sample path only)
        if (max_norm_ > 0.0f) {
            NormalizeEmbeddings();
        }

        const auto& shape = input.Shape();
        dim_t seq_len = shape[0];
        dim_t total_indices = seq_len;

        // Get indices as int32
        const int32_t* indices_ptr = input.Data<int32_t>();

        // Get weight matrix
        af::array w = TensorToAf(weight_);  // [num_embeddings, embedding_dim]

        // Vectorized gather: for each index, get the corresponding row
        af::array output_flat = af::constant(0.0f, af::dim4(total_indices, embedding_dim_));
        for (dim_t i = 0; i < total_indices; i++) {
            int32_t idx = indices_ptr[i];
            if (idx >= 0 && idx < num_embeddings_) {
                output_flat(CheckedIntDim(static_cast<size_t>(i), "embedding index"), af::span) =
                    w(idx, af::span);
            }
            // If idx == padding_idx or out of bounds, leave as zero
        }
        output_flat.eval();

        // Reshape to [seq_len, embedding_dim]
        af::array output = af::moddims(output_flat, af::dim4(seq_len, embedding_dim_));
        output.eval();
        return AfToTensor(output);
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
    const auto& shape = input.Shape();
    bool is_batched = shape.size() == 2;

    size_t batch_size = is_batched ? shape[0] : 1;
    size_t seq_len = is_batched ? shape[1] : shape[0];

    std::vector<size_t> out_shape;
    if (is_batched) {
        out_shape = {batch_size, seq_len, static_cast<size_t>(embedding_dim_)};
    } else {
        out_shape = {seq_len, static_cast<size_t>(embedding_dim_)};
    }

    Tensor output(out_shape, DataType::Float32);
    float* out_data = static_cast<float*>(output.Data());
    const float* weight_data = weight_.Data<float>();
    const int32_t* indices = input.Data<int32_t>();

    size_t total = batch_size * seq_len;
    for (size_t i = 0; i < total; i++) {
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
    // Same gating as Forward: batched input goes to the CPU fallback
    // because AF's column-major moddims silently scrambles the row
    // ordering of [batch, seq_len, embed_dim] gradient tensors. Caught
    // when wiring text training — the AF backward returned valid-shaped
    // but content-wrong gradients, leading to slow / unstable learning.
    if (cached_indices_.Shape().size() != 2) try {
        const auto& shape = cached_indices_.Shape();
        dim_t seq_len = shape[0];
        dim_t total_indices = seq_len;

        const int32_t* indices_ptr = cached_indices_.Data<int32_t>();

        // Initialize gradient accumulator
        af::array dw = af::constant(0.0f, af::dim4(num_embeddings_, embedding_dim_));

        // Get flattened gradient output
        af::array grad = TensorToAf(grad_output);
        grad = af::moddims(grad, af::dim4(total_indices, embedding_dim_));
        grad.eval();

        // Scatter-add gradients to the weight matrix
        for (dim_t i = 0; i < total_indices; i++) {
            int32_t idx = indices_ptr[i];
            if (idx >= 0 && idx < num_embeddings_ && idx != padding_idx_) {
                dw(idx, af::span) += grad(CheckedIntDim(static_cast<size_t>(i), "embedding grad index"),
                                          af::span);
            }
        }
        dw.eval();

        grad_weight_ = AfToTensor(dw);

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
    bool is_batched = shape.size() == 2;

    size_t batch_size = is_batched ? shape[0] : 1;
    size_t seq_len = is_batched ? shape[1] : shape[0];
    size_t total = batch_size * seq_len;

    // Zero out gradient
    grad_weight_ = Tensor::Zeros({static_cast<size_t>(num_embeddings_),
                                   static_cast<size_t>(embedding_dim_)});
    float* dw = static_cast<float*>(grad_weight_.Data());
    const float* grad_data = grad_output.Data<float>();
    const int32_t* indices = cached_indices_.Data<int32_t>();

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
    const float* src = weight_.Data<float>() + index * embedding_dim_;
    std::memcpy(result.Data(), src, embedding_dim_ * sizeof(float));
    return result;
}

void EmbeddingLayer::SetEmbedding(int index, const Tensor& embedding) {
    if (index < 0 || index >= num_embeddings_) {
        throw std::out_of_range("Embedding index out of range");
    }
    if (embedding.NumElements() != static_cast<size_t>(embedding_dim_)) {
        throw std::invalid_argument("Embedding dimension mismatch");
    }

    float* dst = static_cast<float*>(weight_.Data()) + index * embedding_dim_;
    std::memcpy(dst, embedding.Data<float>(), embedding_dim_ * sizeof(float));
}

void EmbeddingLayer::LoadPretrainedWeights(const Tensor& weights, bool freeze) {
    const auto& shape = weights.Shape();
    if (shape.size() != 2 ||
        shape[0] != static_cast<size_t>(num_embeddings_) ||
        shape[1] != static_cast<size_t>(embedding_dim_)) {
        throw std::invalid_argument("Weight shape mismatch");
    }

    weight_ = weights.Clone();
    frozen_ = freeze;

    // Ensure padding index is zero
    if (padding_idx_ >= 0 && padding_idx_ < num_embeddings_) {
        float* data = static_cast<float*>(weight_.Data());
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
        weight_ = params.at("weight");
    }
}

} // namespace cyxwiz
