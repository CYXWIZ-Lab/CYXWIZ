#include <cyxwiz/sequential.h>
#include <cmath>
#include <stdexcept>

namespace cyxwiz {
// ============================================================================
// LinearModule Implementation
// ============================================================================

LinearModule::LinearModule(size_t in_features, size_t out_features, bool use_bias)
    : in_features_(in_features)
    , out_features_(out_features)
{
    layer_ = std::make_unique<LinearLayer>(in_features, out_features, use_bias);
}

Tensor LinearModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return layer_->Forward(input);
}

Tensor LinearModule::Backward(const Tensor& grad_output) {
    return layer_->Backward(grad_output);
}

std::map<std::string, Tensor> LinearModule::GetParameters() {
    return layer_->GetParameters();
}

void LinearModule::SetParameters(const std::map<std::string, Tensor>& params) {
    layer_->SetParameters(params);
}

std::map<std::string, Tensor> LinearModule::GetGradients() {
    return layer_->GetGradients();
}

std::string LinearModule::GetName() const {
    return "Linear(" + std::to_string(in_features_) + " -> " + std::to_string(out_features_) + ")";
}

// ============================================================================
// TimeDistributedDenseModule Implementation
// ============================================================================

TimeDistributedDenseModule::TimeDistributedDenseModule(size_t in_features,
                                                       size_t out_features,
                                                       bool use_bias)
    : linear_(in_features, out_features, use_bias)
    , in_features_(in_features)
    , out_features_(out_features) {}

Tensor TimeDistributedDenseModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    input_shape_ = input.Shape();
    if (input_shape_.size() != 3) {
        throw std::runtime_error(
            "TimeDistributedDenseModule: input must be [batch, seq_len, features]");
    }
    if (input_shape_[2] != in_features_) {
        throw std::runtime_error(
            "TimeDistributedDenseModule: Input features mismatch. Expected " +
            std::to_string(in_features_) + ", got " +
            std::to_string(input_shape_[2]));
    }

    const size_t batch = input_shape_[0];
    const size_t seq_len = input_shape_[1];
    Tensor flat = input.Reshape({batch * seq_len, in_features_});
    Tensor projected = linear_.Forward(flat);
    return projected.Reshape({batch, seq_len, out_features_});
}

Tensor TimeDistributedDenseModule::Backward(const Tensor& grad_output) {
    if (input_shape_.size() != 3) {
        throw std::runtime_error(
            "TimeDistributedDenseModule: Backward called before Forward");
    }
    const auto& grad_shape = grad_output.Shape();
    if (grad_shape.size() != 3 ||
        grad_shape[0] != input_shape_[0] ||
        grad_shape[1] != input_shape_[1] ||
        grad_shape[2] != out_features_) {
        throw std::runtime_error(
            "TimeDistributedDenseModule: grad_output must be [batch, seq_len, out_features]");
    }

    const size_t batch = input_shape_[0];
    const size_t seq_len = input_shape_[1];
    Tensor flat_grad = grad_output.Reshape({batch * seq_len, out_features_});
    Tensor flat_input_grad = linear_.Backward(flat_grad);
    return flat_input_grad.Reshape(input_shape_);
}

std::map<std::string, Tensor> TimeDistributedDenseModule::GetParameters() {
    return linear_.GetParameters();
}

void TimeDistributedDenseModule::SetParameters(
    const std::map<std::string, Tensor>& params) {
    linear_.SetParameters(params);
}

std::map<std::string, Tensor> TimeDistributedDenseModule::GetGradients() {
    return linear_.GetGradients();
}

std::string TimeDistributedDenseModule::GetName() const {
    return "TimeDistributedDense(" + std::to_string(in_features_) +
           " -> " + std::to_string(out_features_) + ")";
}

// ============================================================================
// EmbeddingModule Implementation
// ============================================================================
//
// The backend's EmbeddingLayer reads indices as int32, but CyxWiz's
// IBatcher / training pipeline ships every tensor as float32 (for
// uniformity with Dense/Conv/etc.). We bridge the type gap here by
// building an int32 Tensor on every forward pass whose values are the
// rounded floats from the upstream batch. This is the cost of dropping
// Embedding into an otherwise float-only graph — worth it, since the
// alternative (dual-typing the whole batching layer) is much bigger.
//
// Shape contract:
//   Forward input:  [batch, seq_len]                float, integer values
//   Forward output: [batch, seq_len, embedding_dim] float
//
// A Flatten module after this produces [batch, seq_len * embedding_dim]
// which is what a Dense layer head expects.

EmbeddingModule::EmbeddingModule(size_t num_embeddings, size_t embedding_dim,
                                 int padding_idx, float max_norm)
    : num_embeddings_(num_embeddings)
    , embedding_dim_(embedding_dim)
    , padding_idx_(padding_idx)
    , max_norm_(max_norm)
{
    layer_ = std::make_unique<EmbeddingLayer>(
        static_cast<int>(num_embeddings),
        static_cast<int>(embedding_dim),
        padding_idx,
        max_norm);
}

Tensor EmbeddingModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();

    // Convert [batch, seq_len] token IDs to int32. Most production batchers
    // still carry token IDs as float32, while sequence/synthetic paths use
    // integer tensors. Normalize both forms here so the model-facing contract
    // stays narrow.
    const auto& shape = input.Shape();
    size_t total = 1;
    for (auto d : shape) total *= d;

    Tensor int_input(shape, DataType::Int32);
    int32_t* dst = static_cast<int32_t*>(int_input.Data());
    const int32_t vocab_max = static_cast<int32_t>(num_embeddings_) - 1;
    for (size_t i = 0; i < total; ++i) {
        int32_t idx = 0;
        switch (input.GetDataType()) {
            case DataType::Float32:
                idx = static_cast<int32_t>(input.Data<float>()[i]);
                break;
            case DataType::Int32:
                idx = input.Data<int32_t>()[i];
                break;
            case DataType::Int64:
                idx = static_cast<int32_t>(input.Data<int64_t>()[i]);
                break;
            default:
                throw std::runtime_error(
                    "EmbeddingModule: input token ids must be Float32, Int32, or Int64");
        }
        // Clamp to valid range. Out-of-vocab IDs map to 0 (the [PAD]
        // slot by convention — safe fallback).
        if (idx < 0 || idx > vocab_max) idx = 0;
        dst[i] = idx;
    }

    return layer_->Forward(int_input);
}

Tensor EmbeddingModule::Backward(const Tensor& grad_output) {
    // EmbeddingLayer::Backward returns an empty tensor because you
    // can't differentiate w.r.t. integer indices — but it DOES update
    // its own weight gradients internally. Propagating "nothing"
    // upstream is correct: Embedding is almost always the first
    // trainable layer, so there's no upstream tensor gradient anyway.
    return layer_->Backward(grad_output);
}

std::map<std::string, Tensor> EmbeddingModule::GetParameters() {
    return layer_->GetParameters();
}

void EmbeddingModule::SetParameters(const std::map<std::string, Tensor>& params) {
    layer_->SetParameters(params);
}

std::map<std::string, Tensor> EmbeddingModule::GetGradients() {
    return layer_->GetGradients();
}

void EmbeddingModule::LoadPretrainedWeights(const Tensor& weights, bool freeze) {
    layer_->LoadPretrainedWeights(weights, freeze);
    SetTrainable(!freeze);
}

std::string EmbeddingModule::GetName() const {
    return "Embedding(" + std::to_string(num_embeddings_) + " x " +
           std::to_string(embedding_dim_) + ")";
}

// ============================================================================
// PositionalEncodingModule Implementation
// ============================================================================

PositionalEncodingModule::PositionalEncodingModule(
    size_t d_model,
    size_t max_sequence_length)
    : d_model_(d_model)
    , max_sequence_length_(max_sequence_length)
{
    if (d_model_ < 1) d_model_ = 1;
    if (max_sequence_length_ < 1) max_sequence_length_ = 1;
}

Tensor PositionalEncodingModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    const auto& shape = input.Shape();
    if (input.GetDataType() != DataType::Float32 ||
        shape.size() != 3 ||
        shape[2] != d_model_) {
        throw std::runtime_error(
            "PositionalEncodingModule: input must be Float32 [batch, seq_len, d_model]");
    }
    if (shape[1] > max_sequence_length_) {
        throw std::runtime_error(
            "PositionalEncodingModule: sequence length exceeds max_sequence_length");
    }

    Tensor output(shape, DataType::Float32);
    const float* src = input.Data<float>();
    float* dst = output.Data<float>();
    const size_t batch = shape[0];
    const size_t seq_len = shape[1];

    for (size_t b = 0; b < batch; ++b) {
        for (size_t pos = 0; pos < seq_len; ++pos) {
            for (size_t dim = 0; dim < d_model_; ++dim) {
                const size_t offset = (b * seq_len + pos) * d_model_ + dim;
                const double angle = static_cast<double>(pos) /
                    std::pow(10000.0, static_cast<double>(2 * (dim / 2)) /
                                      static_cast<double>(d_model_));
                const double encoded = (dim % 2 == 0)
                    ? std::sin(angle)
                    : std::cos(angle);
                dst[offset] = src[offset] + static_cast<float>(encoded);
            }
        }
    }

    return output;
}

Tensor PositionalEncodingModule::Backward(const Tensor& grad_output) {
    return grad_output.Clone();
}

std::string PositionalEncodingModule::GetName() const {
    return "PositionalEncoding(d_model=" + std::to_string(d_model_) + ")";
}

} // namespace cyxwiz
