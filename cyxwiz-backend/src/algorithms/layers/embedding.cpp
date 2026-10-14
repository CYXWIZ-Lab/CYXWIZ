#include "cyxwiz/layers/embedding.h"
#include <vector>
#include "layer_arrayfire_utils.h"
#include "../arrayfire_backend_utils.h"
#include "../arrayfire_indexed.h"

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

// One ArrayFire path (the CPU is ArrayFire's CPU backend): a device error is
// reported, not hidden behind host loops.
[[noreturn]] static void ThrowEmbeddingDeviceError(const char* operation, const af::exception& error) {
    throw std::runtime_error(std::string(operation) + " failed on the ArrayFire device: " + error.what());
}

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

    grad_weight_ = Tensor::Zeros({static_cast<size_t>(num_embeddings_),
                                   static_cast<size_t>(embedding_dim_)});
}

void EmbeddingLayer::NormalizeEmbeddings() {
    if (max_norm_ <= 0.0f) return;

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
        ThrowEmbeddingDeviceError("EmbeddingLayer::NormalizeEmbeddings", e);
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

    try {
        if (max_norm_ > 0.0f) {
            NormalizeEmbeddings();
        }

        af::array w = weight_.GetSemanticArray();
        // One gather for all tokens (TOFIX118 P8: the per-token loop issued a
        // device copy per token - 69 of ~280 ms per Berean step). Padding and
        // out-of-range ids give zero rows, as before. Ids are read where they
        // live (a host batch uploads once; device ids stay put).
        (void)CheckedIntDim(total_indices, "embedding token count");
        const af::array ids = input.GetRowMajorFlatArray();
        const af::array valid = (ids >= 0) && (ids < num_embeddings_) && (ids != padding_idx_);
        const af::array safe_ids = af::select(valid, ids, 0.0);
        af::array output_flat = af::lookup(w, safe_ids, 0);
        output_flat = af::select(af::tile(valid, 1, embedding_dim_), output_flat, 0.0);
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
        ThrowEmbeddingDeviceError("EmbeddingLayer::Forward", e);
    }
}

Tensor EmbeddingLayer::Backward(const Tensor& grad_output) {
    if (frozen_) {
        // Return empty tensor - no gradient needed for frozen embeddings
        return Tensor();
    }

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

        (void)total_indices;
        // Initialize gradient accumulator
        af::array dw = af::constant(0.0f, af::dim4(num_embeddings_, embedding_dim_));

        af::array grad = SemanticGradientToTokenRows(
            grad_output.GetSemanticArray(), shape.size() == 2, batch_size,
            sequence_length,
            static_cast<size_t>(embedding_dim_));

        // Scatter-add token rows into the table in one pass, on the device:
        // keep valid tokens, sort by id, sum rows per id (sumByKey), assign once.
        const af::array ids = cached_indices_.GetRowMajorFlatArray();
        const af::array valid = (ids >= 0) && (ids < num_embeddings_) && (ids != padding_idx_);
        const af::array row_array = af::where(valid);
        if (row_array.elements() > 0) {
            const af::array key_array = ids(row_array);
            const af::array token_rows = af::lookup(grad, row_array, 0);
            if (row_array.elements() == 1) {
                // sumByKey needs a key vector; one token is a direct row write.
                dw(key_array, af::span) = token_rows;
            } else {
                af::array sorted_keys, order;
                SortWithIndex(sorted_keys, order, key_array, 0, true);
                af::array unique_keys, sums;
                af::sumByKey(unique_keys, sums, sorted_keys, af::lookup(token_rows, order, 0), 0);
                dw(unique_keys, af::span) = sums;
            }
        }
        if (!pending_weight_gradient_.Shape().empty()) {
            dw = dw + pending_weight_gradient_.GetSemanticArray();
        }
        dw.eval();
        pending_weight_gradient_ = Tensor();

        grad_weight_ = Tensor::FromSemanticArray(
            dw, {static_cast<size_t>(num_embeddings_),
                 static_cast<size_t>(embedding_dim_)});

        // Return empty tensor (no gradient w.r.t. integer indices)
        return Tensor();
    } catch (const af::exception& e) {
        ThrowEmbeddingDeviceError("EmbeddingLayer::Backward", e);
    }
}

void EmbeddingLayer::AddPendingWeightGradient(const Tensor& gradient) {
    const std::vector<size_t> shape{static_cast<size_t>(num_embeddings_), static_cast<size_t>(embedding_dim_)};
    if (gradient.GetDataType() != DataType::Float32 || gradient.Shape() != shape) {
        throw std::invalid_argument("Embedding tied gradient must be Float32 [num_embeddings, embedding_dim]");
    }
    pending_weight_gradient_ = pending_weight_gradient_.Shape().empty() ? gradient : pending_weight_gradient_ + gradient;
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
        try {
            af::array padded = weight_.GetSemanticArray().copy();
            padded(padding_idx_, af::span) = 0.0f;
            padded.eval(); // Complete the update within the observed fallback boundary.
            weight_.SetFromSemanticArray(padded, shape);
            return;
        } catch (const af::exception& e) {
            ThrowEmbeddingDeviceError("EmbeddingLayer::LoadPretrainedWeights", e);
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
