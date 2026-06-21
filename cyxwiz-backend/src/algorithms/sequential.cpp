#include <cyxwiz/sequential.h>
#include <cyxwiz/debug_hooks.h>
#include <cyxwiz/layers/linear.h>
#include <cyxwiz/activations/relu.h>
#include <cyxwiz/activations/sigmoid.h>
#include <cyxwiz/activations/tanh.h>
#include <cyxwiz/activation.h>  // For LeakyReLUActivation, ELUActivation, GELUActivation, etc.
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <atomic>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <utility>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

namespace {

std::string ShapeToStringForTrace(const std::vector<size_t>& shape) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) {
            out << ',';
        }
        out << shape[i];
    }
    out << ']';
    return out.str();
}

std::vector<size_t> UnravelIndex(size_t index, const std::vector<size_t>& shape) {
    std::vector<size_t> indices(shape.size(), 0);
    for (size_t i = shape.size(); i-- > 0;) {
        indices[i] = index % shape[i];
        index /= shape[i];
    }
    return indices;
}

size_t RavelIndex(const std::vector<size_t>& indices,
                  const std::vector<size_t>& shape) {
    size_t linear = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
        linear = linear * shape[i] + indices[i];
    }
    return linear;
}

void EmitModelLayerTrace(const char* stage,
                         size_t layer_index,
                         const std::string& layer_name,
                         const std::vector<size_t>& input_shape,
                         const std::vector<size_t>& output_shape,
                         float duration_ms) {
    std::ostringstream message;
    message << "layer=" << layer_index
            << " name=" << layer_name
            << " input=" << ShapeToStringForTrace(input_shape)
            << " output=" << ShapeToStringForTrace(output_shape)
            << " duration_ms=" << duration_ms;
    BackendDebugHooks::EmitDebugEvent(stage, message.str());
}

} // namespace

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

// ============================================================================
// LSTMModule Implementation
// ============================================================================
//
// Wraps cyxwiz::LSTMLayer with two classification-friendly behaviors:
//   1. Keras-style `return_sequences=false` reduction — slices out the
//      last timestep of the full LSTM output so a Dense head can sit
//      directly after the LSTM without an intervening Flatten.
//   2. Symmetric last-step gradient re-expansion in Backward, zeroing
//      all non-terminal steps.
//
// When `return_sequences=true`, the wrapper is a pure passthrough to
// LSTMLayer and output retains the `[batch, seq_len, hidden*dirs]`
// shape — needed for stacked LSTMs and seq-to-seq heads.

LSTMModule::LSTMModule(size_t input_size, size_t hidden_size,
                       size_t num_layers, bool bidirectional,
                       bool return_sequences)
    : input_size_(input_size)
    , hidden_size_(hidden_size)
    , num_layers_(num_layers)
    , bidirectional_(bidirectional)
    , return_sequences_(return_sequences)
{
    layer_ = std::make_unique<LSTMLayer>(
        static_cast<int>(input_size),
        static_cast<int>(hidden_size),
        static_cast<int>(num_layers),
        /*batch_first=*/true,
        bidirectional,
        /*dropout=*/0.0f);
}

Tensor LSTMModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();

    // LSTMLayer returns the full sequence output:
    //   [batch, seq_len, hidden_size * num_directions]
    Tensor full_output = layer_->Forward(input);
    last_full_output_shape_ = full_output.Shape();

    if (return_sequences_) {
        return full_output;
    }

    // Defensive: if the output isn't 3D for some reason, bail out and
    // pass through. Should never happen under normal use, but we'd
    // rather forward a weird tensor than crash on the slice.
    if (last_full_output_shape_.size() != 3) {
        spdlog::warn("LSTMModule: expected 3D output [batch, seq, hidden_dirs] "
                     "but got {}D — passing full output through",
                     last_full_output_shape_.size());
        return full_output;
    }

    const size_t batch = last_full_output_shape_[0];
    const size_t seq_len = last_full_output_shape_[1];
    const size_t hd = last_full_output_shape_[2];

    // Slice out the last timestep: out[:, seq_len-1, :] ? [batch, hd].
    // Row-major layout means sample b's last step is at offset
    //   b * seq_len * hd + (seq_len - 1) * hd
    Tensor last({batch, hd}, DataType::Float32);
    const float* src = full_output.Data<float>();
    float* dst = static_cast<float*>(last.Data());
    for (size_t b = 0; b < batch; ++b) {
        const float* src_step = src + b * seq_len * hd + (seq_len - 1) * hd;
        std::memcpy(dst + b * hd, src_step, hd * sizeof(float));
    }
    return last;
}

Tensor LSTMModule::Backward(const Tensor& grad_output) {
    if (return_sequences_) {
        // Full-sequence mode — grad_output already has shape
        // [batch, seq_len, hidden*dirs]. Pass straight through.
        return layer_->Backward(grad_output);
    }

    // Last-step mode: re-expand [batch, hidden] gradient to the full
    // [batch, seq_len, hidden] shape with zeros everywhere except the
    // terminal step. LSTMLayer::Backward expects the gradient of the
    // whole sequence output; since only the last step fed into the
    // loss, all earlier timesteps have zero contribution.
    if (last_full_output_shape_.size() != 3) {
        spdlog::warn("LSTMModule::Backward called without a 3D shape cache "
                     "— falling back to direct grad passthrough");
        return layer_->Backward(grad_output);
    }

    const size_t batch = last_full_output_shape_[0];
    const size_t seq_len = last_full_output_shape_[1];
    const size_t hd = last_full_output_shape_[2];

    Tensor expanded = Tensor::Zeros({batch, seq_len, hd});
    const float* src = grad_output.Data<float>();
    float* dst = static_cast<float*>(expanded.Data());
    for (size_t b = 0; b < batch; ++b) {
        float* dst_step = dst + b * seq_len * hd + (seq_len - 1) * hd;
        std::memcpy(dst_step, src + b * hd, hd * sizeof(float));
    }
    return layer_->Backward(expanded);
}

std::map<std::string, Tensor> LSTMModule::GetParameters() {
    return layer_->GetParameters();
}

void LSTMModule::SetParameters(const std::map<std::string, Tensor>& params) {
    layer_->SetParameters(params);
}

std::map<std::string, Tensor> LSTMModule::GetGradients() {
    // LSTMLayer doesn't expose GetGradients() yet — it writes gradients
    // directly into the parameters map keyed as "grad_W_ih", "grad_W_hh",
    // etc. (the same "grad_X"-named entries the legacy optimizer path
    // looked up). The SequentialModel's training step uses GetParameters
    // for both weights AND grads through this naming, so we can forward
    // GetParameters() here. When LSTMLayer grows a dedicated
    // GetGradients() (matching LinearLayer / EmbeddingLayer), this
    // passthrough can become layer_->GetGradients().
    return layer_->GetParameters();
}

std::string LSTMModule::GetName() const {
    const int dirs = bidirectional_ ? 2 : 1;
    return "LSTM(" + std::to_string(input_size_) + " -> " +
           std::to_string(hidden_size_ * dirs) +
           (return_sequences_ ? ", seq" : ", last") + ")";
}

// ============================================================================
// GRUModule Implementation — direct mirror of LSTMModule. The slice and
// re-expand logic for return_sequences=false is identical because
// GRULayer matches LSTMLayer's [batch, seq, hidden*dirs] full-output
// contract.
// ============================================================================

static Tensor ReverseSequenceTensor(const Tensor& input, bool batch_first) {
    const auto& shape = input.Shape();
    if (shape.size() != 3) {
        return input.Clone();
    }

    const size_t batch = batch_first ? shape[0] : shape[1];
    const size_t seq_len = batch_first ? shape[1] : shape[0];
    const size_t features = shape[2];

    Tensor output = Tensor::Zeros(shape, input.GetDataType());
    const float* src = input.Data<float>();
    float* dst = output.Data<float>();

    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            const size_t src_t = seq_len - 1 - t;
            const float* src_step = batch_first
                ? src + b * seq_len * features + src_t * features
                : src + src_t * batch * features + b * features;
            float* dst_step = batch_first
                ? dst + b * seq_len * features + t * features
                : dst + t * batch * features + b * features;
            std::memcpy(dst_step, src_step, features * sizeof(float));
        }
    }

    return output;
}

static Tensor ConcatFeatureTensor(const Tensor& left, const Tensor& right) {
    const auto& lshape = left.Shape();
    const auto& rshape = right.Shape();
    if (lshape.size() != 3 || rshape.size() != 3) {
        return left.Clone();
    }

    const size_t batch = lshape[0];
    const size_t seq_len = lshape[1];
    const size_t left_features = lshape[2];
    const size_t right_features = rshape[2];

    Tensor output = Tensor::Zeros({batch, seq_len, left_features + right_features},
                                  left.GetDataType());
    const float* lsrc = left.Data<float>();
    const float* rsrc = right.Data<float>();
    float* dst = output.Data<float>();

    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            float* dst_step = dst + b * seq_len * (left_features + right_features)
                             + t * (left_features + right_features);
            const float* lstep = lsrc + b * seq_len * left_features + t * left_features;
            const float* rstep = rsrc + b * seq_len * right_features + t * right_features;
            std::memcpy(dst_step, lstep, left_features * sizeof(float));
            std::memcpy(dst_step + left_features, rstep, right_features * sizeof(float));
        }
    }

    return output;
}

static Tensor SliceFeatureTensor(const Tensor& input, size_t offset, size_t width) {
    const auto& shape = input.Shape();
    if (shape.size() != 3) {
        return input.Clone();
    }

    const size_t batch = shape[0];
    const size_t seq_len = shape[1];
    const size_t features = shape[2];
    if (offset + width > features) {
        return input.Clone();
    }

    Tensor output = Tensor::Zeros({batch, seq_len, width}, input.GetDataType());
    const float* src = input.Data<float>();
    float* dst = output.Data<float>();

    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            const float* src_step = src + b * seq_len * features + t * features + offset;
            float* dst_step = dst + b * seq_len * width + t * width;
            std::memcpy(dst_step, src_step, width * sizeof(float));
        }
    }

    return output;
}

static std::string NormalizeGRULayerKey(const std::string& key) {
    std::string normalized = key;
    if (normalized.rfind("layer0_", 0) == 0) {
        normalized.erase(0, 7);
    }
    if (normalized.rfind("grad_", 0) == 0) {
        normalized.erase(0, 5);
    }
    return normalized;
}

static std::string MakeGRUBranchKey(size_t layer_idx, const std::string& branch,
                                    const std::string& normalized_key) {
    return "layer" + std::to_string(layer_idx) + "." + branch + "." + normalized_key;
}

GRUModule::GRUModule(size_t input_size, size_t hidden_size,
                     size_t num_layers, bool bidirectional,
                     bool return_sequences)
    : input_size_(input_size)
    , hidden_size_(hidden_size)
    , num_layers_(num_layers)
    , bidirectional_(bidirectional)
    , return_sequences_(return_sequences)
{
    if (bidirectional_) {
        split_bidirectional_path_ = true;
        forward_layers_.reserve(num_layers_);
        reverse_layers_.reserve(num_layers_);

        for (size_t layer = 0; layer < num_layers_; ++layer) {
            const int layer_input_size = (layer == 0)
                ? static_cast<int>(input_size)
                : static_cast<int>(hidden_size * 2);
            forward_layers_.push_back(std::make_unique<GRULayer>(
                layer_input_size,
                static_cast<int>(hidden_size),
                /*num_layers=*/1,
                /*batch_first=*/true,
                /*bidirectional=*/false,
                /*dropout=*/0.0f));
            reverse_layers_.push_back(std::make_unique<GRULayer>(
                layer_input_size,
                static_cast<int>(hidden_size),
                /*num_layers=*/1,
                /*batch_first=*/true,
                /*bidirectional=*/false,
                /*dropout=*/0.0f));
        }

        spdlog::info("[GRUModule] Using split bidirectional GRU path "
                     "({} layer pairs). GPU placement for this split path "
                     "remains disabled until the single-direction ArrayFire "
                     "GRU path has dedicated correctness and timeout coverage.",
                     num_layers_);
    } else {
        layer_ = std::make_unique<GRULayer>(
            static_cast<int>(input_size),
            static_cast<int>(hidden_size),
            static_cast<int>(num_layers),
            /*batch_first=*/true,
            bidirectional,
            /*dropout=*/0.0f);
    }
}

Tensor GRUModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();

    Tensor full_output;
    if (split_bidirectional_path_) {
        Tensor layer_input = input;
        for (size_t layer = 0; layer < num_layers_; ++layer) {
            Tensor forward_output = forward_layers_[layer]->Forward(layer_input);
            Tensor reverse_input = ReverseSequenceTensor(layer_input, /*batch_first=*/true);
            Tensor reverse_output = reverse_layers_[layer]->Forward(reverse_input);
            reverse_output = ReverseSequenceTensor(reverse_output, /*batch_first=*/true);
            layer_input = ConcatFeatureTensor(forward_output, reverse_output);
        }
        full_output = layer_input;
    } else {
        full_output = layer_->Forward(input);
    }
    last_full_output_shape_ = full_output.Shape();

    if (return_sequences_) {
        return full_output;
    }

    if (last_full_output_shape_.size() != 3) {
        spdlog::warn("GRUModule: expected 3D output [batch, seq, hidden_dirs] "
                     "but got {}D — passing full output through",
                     last_full_output_shape_.size());
        return full_output;
    }

    const size_t batch = last_full_output_shape_[0];
    const size_t seq_len = last_full_output_shape_[1];
    const size_t hd = last_full_output_shape_[2];

    Tensor last({batch, hd}, DataType::Float32);
    const float* src = full_output.Data<float>();
    float* dst = static_cast<float*>(last.Data());
    for (size_t b = 0; b < batch; ++b) {
        const float* src_step = src + b * seq_len * hd + (seq_len - 1) * hd;
        std::memcpy(dst + b * hd, src_step, hd * sizeof(float));
    }
    return last;
}

Tensor GRUModule::Backward(const Tensor& grad_output) {
    Tensor upstream = grad_output;
    if (!return_sequences_) {
        if (last_full_output_shape_.size() != 3) {
            spdlog::warn("GRUModule::Backward called without a 3D shape cache "
                         "- falling back to direct grad passthrough");
            return split_bidirectional_path_
                ? Tensor::Zeros(input_cache_.Shape())
                : layer_->Backward(grad_output);
        }

        const size_t batch = last_full_output_shape_[0];
        const size_t seq_len = last_full_output_shape_[1];
        const size_t hd = last_full_output_shape_[2];

        Tensor expanded = Tensor::Zeros({batch, seq_len, hd});
        const float* src = grad_output.Data<float>();
        float* dst = static_cast<float*>(expanded.Data());
        for (size_t b = 0; b < batch; ++b) {
            float* dst_step = dst + b * seq_len * hd + (seq_len - 1) * hd;
            std::memcpy(dst_step, src + b * hd, hd * sizeof(float));
        }
        upstream = expanded;
    }

    if (split_bidirectional_path_) {
        if (upstream.Shape().size() != 3) {
            spdlog::warn("GRUModule::Backward expected 3D upstream gradient "
                         "for split bidirectional path");
            return Tensor::Zeros(input_cache_.Shape());
        }

        Tensor layer_grad = upstream;
        for (int layer = static_cast<int>(num_layers_) - 1; layer >= 0; --layer) {
            const size_t total_features = layer_grad.Shape()[2];
            const size_t half_features = total_features / 2;
            Tensor forward_grad = SliceFeatureTensor(layer_grad, 0, half_features);
            Tensor reverse_grad = SliceFeatureTensor(layer_grad, half_features, half_features);

            Tensor dx_forward = forward_layers_[static_cast<size_t>(layer)]->Backward(forward_grad);
            Tensor dx_reverse = reverse_layers_[static_cast<size_t>(layer)]->Backward(
                ReverseSequenceTensor(reverse_grad, /*batch_first=*/true));
            dx_reverse = ReverseSequenceTensor(dx_reverse, /*batch_first=*/true);

            layer_grad = dx_forward + dx_reverse;
        }

        return layer_grad;
    }

    return layer_->Backward(upstream);
}

std::map<std::string, Tensor> GRUModule::GetParameters() {
    if (split_bidirectional_path_) {
        std::map<std::string, Tensor> params;
        for (size_t layer = 0; layer < num_layers_; ++layer) {
            auto forward_params = forward_layers_[layer]->GetParameters();
            auto reverse_params = reverse_layers_[layer]->GetParameters();
            for (const auto& [key, tensor] : forward_params) {
                if (key.find("grad_") != std::string::npos) continue;
                params[MakeGRUBranchKey(layer, "forward", NormalizeGRULayerKey(key))] = tensor;
            }
            for (const auto& [key, tensor] : reverse_params) {
                if (key.find("grad_") != std::string::npos) continue;
                params[MakeGRUBranchKey(layer, "reverse", NormalizeGRULayerKey(key))] = tensor;
            }
        }
        return params;
    }
    return layer_->GetParameters();
}

void GRUModule::SetParameters(const std::map<std::string, Tensor>& params) {
    if (split_bidirectional_path_) {
        std::vector<std::map<std::string, Tensor>> forward_params(num_layers_);
        std::vector<std::map<std::string, Tensor>> reverse_params(num_layers_);
        for (const auto& [key, tensor] : params) {
            if (key.rfind("layer", 0) != 0) {
                continue;
            }
            const size_t dot1 = key.find('.');
            const size_t dot2 = key.find('.', dot1 == std::string::npos ? 0 : dot1 + 1);
            if (dot1 == std::string::npos || dot2 == std::string::npos) {
                continue;
            }

            const size_t layer_idx = static_cast<size_t>(std::stoul(key.substr(5, dot1 - 5)));
            if (layer_idx >= num_layers_) {
                continue;
            }

            const std::string branch = key.substr(dot1 + 1, dot2 - dot1 - 1);
            const std::string base_key = key.substr(dot2 + 1);
            if (base_key.empty()) {
                continue;
            }
            const std::string child_key = "layer0_" + base_key;
            if (branch == "forward") {
                forward_params[layer_idx][child_key] = tensor;
            } else if (branch == "reverse") {
                reverse_params[layer_idx][child_key] = tensor;
            }
        }
        for (size_t layer = 0; layer < num_layers_; ++layer) {
            forward_layers_[layer]->SetParameters(forward_params[layer]);
            reverse_layers_[layer]->SetParameters(reverse_params[layer]);
        }
        return;
    }
    layer_->SetParameters(params);
}

std::map<std::string, Tensor> GRUModule::GetGradients() {
    auto build_gradient_map = [](const std::map<std::string, Tensor>& params,
                                 const std::string& prefix) {
        std::map<std::string, Tensor> grads;
        for (const auto& [key, value] : params) {
            if (key.find("grad_") == std::string::npos) continue;
            const std::string base = NormalizeGRULayerKey(key);
            grads[prefix + base] = value;
        }
        return grads;
    };

    if (split_bidirectional_path_) {
        std::map<std::string, Tensor> grads;
        for (size_t layer = 0; layer < num_layers_; ++layer) {
            auto forward_grads = build_gradient_map(forward_layers_[layer]->GetParameters(),
                                                     MakeGRUBranchKey(layer, "forward", ""));
            auto reverse_grads = build_gradient_map(reverse_layers_[layer]->GetParameters(),
                                                     MakeGRUBranchKey(layer, "reverse", ""));
            grads.insert(forward_grads.begin(), forward_grads.end());
            grads.insert(reverse_grads.begin(), reverse_grads.end());
        }
        return grads;
    }
    // Same convention as LSTMModule - GRULayer writes "grad_*" keys
    // into its parameters map and the SequentialModel optimizer step
    // reads them via GetParameters().
    return layer_->GetParameters();
}

std::string GRUModule::GetName() const {
    const int dirs = bidirectional_ ? 2 : 1;
    const std::string prefix = split_bidirectional_path_ ? "Bi" : "";
    return prefix + std::string("GRU(") + std::to_string(input_size_) + " -> " +
           std::to_string(hidden_size_ * dirs) +
           (return_sequences_ ? ", seq" : ", last") + ")";
}

void GRUModule::SetTraining(bool training) {
    Module::SetTraining(training);
    if (layer_) layer_->SetTraining(training);
    for (auto& layer : forward_layers_) {
        layer->SetTraining(training);
    }
    for (auto& layer : reverse_layers_) {
        layer->SetTraining(training);
    }
}

// ============================================================================
// TransformerEncoderModule Implementation
// ============================================================================

namespace {

bool IsGradientParameterKey(const std::string& key) {
    if (key.rfind("grad_", 0) == 0) {
        return true;
    }
    return key.find(".grad_") != std::string::npos;
}

std::string NormalizeGradientParameterKey(std::string key) {
    if (key.rfind("grad_", 0) == 0) {
        key.erase(0, 5);
    }

    size_t pos = 0;
    while ((pos = key.find(".grad_", pos)) != std::string::npos) {
        key.replace(pos, 6, ".");
        ++pos;
    }

    return key;
}

} // namespace

TransformerEncoderModule::TransformerEncoderModule(size_t d_model,
                                                   size_t num_heads,
                                                   size_t dim_feedforward,
                                                   float dropout,
                                                   bool norm_first)
    : d_model_(d_model)
    , num_heads_(num_heads)
    , dim_feedforward_(dim_feedforward)
    , dropout_(dropout)
    , norm_first_(norm_first)
{
    if (d_model_ < 1) d_model_ = 1;
    if (num_heads_ < 1) num_heads_ = 1;
    if (d_model_ % num_heads_ != 0) {
        spdlog::warn("TransformerEncoderModule: d_model={} is not divisible "
                     "by num_heads={}; falling back to one head",
                     d_model_, num_heads_);
        num_heads_ = 1;
    }
    if (dim_feedforward_ < 1) dim_feedforward_ = d_model_;
    if (dropout_ < 0.0f) dropout_ = 0.0f;
    if (dropout_ >= 1.0f) dropout_ = 0.999f;

    layer_ = std::make_unique<TransformerEncoderLayer>(
        static_cast<int>(d_model_),
        static_cast<int>(num_heads_),
        static_cast<int>(dim_feedforward_),
        dropout_,
        norm_first_);
}

Tensor TransformerEncoderModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return layer_->Forward(input);
}

Tensor TransformerEncoderModule::Backward(const Tensor& grad_output) {
    return layer_->Backward(grad_output);
}

void TransformerEncoderModule::SetTraining(bool training) {
    Module::SetTraining(training);
    layer_->SetTraining(training);
}

std::map<std::string, Tensor> TransformerEncoderModule::GetParameters() {
    std::map<std::string, Tensor> params;
    for (const auto& [key, value] : layer_->GetParameters()) {
        if (!IsGradientParameterKey(key)) {
            params[key] = value;
        }
    }
    return params;
}

void TransformerEncoderModule::SetParameters(
    const std::map<std::string, Tensor>& params) {
    layer_->SetParameters(params);
}

std::map<std::string, Tensor> TransformerEncoderModule::GetGradients() {
    std::map<std::string, Tensor> grads;
    for (const auto& [key, value] : layer_->GetParameters()) {
        if (IsGradientParameterKey(key)) {
            grads[NormalizeGradientParameterKey(key)] = value;
        }
    }
    return grads;
}

std::string TransformerEncoderModule::GetName() const {
    return "TransformerEncoder(d_model=" + std::to_string(d_model_) +
           ", heads=" + std::to_string(num_heads_) + ")";
}

// ============================================================================
// TransformerDecoderModule Implementation
// ============================================================================

TransformerDecoderModule::TransformerDecoderModule(size_t d_model,
                                                   size_t num_heads,
                                                   size_t dim_feedforward,
                                                   float dropout,
                                                   bool norm_first)
    : d_model_(d_model)
    , num_heads_(num_heads)
    , dim_feedforward_(dim_feedforward)
    , dropout_(dropout)
    , norm_first_(norm_first)
{
    if (d_model_ < 1) d_model_ = 1;
    if (num_heads_ < 1) num_heads_ = 1;
    if (d_model_ % num_heads_ != 0) {
        spdlog::warn("TransformerDecoderModule: d_model={} is not divisible "
                     "by num_heads={}; falling back to one head",
                     d_model_, num_heads_);
        num_heads_ = 1;
    }
    if (dim_feedforward_ < 1) dim_feedforward_ = d_model_;
    if (dropout_ < 0.0f) dropout_ = 0.0f;
    if (dropout_ >= 1.0f) dropout_ = 0.999f;

    layer_ = std::make_unique<TransformerDecoderLayer>(
        static_cast<int>(d_model_),
        static_cast<int>(num_heads_),
        static_cast<int>(dim_feedforward_),
        dropout_,
        norm_first_);
}

Tensor TransformerDecoderModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return layer_->Forward(input);
}

Tensor TransformerDecoderModule::Backward(const Tensor& grad_output) {
    return layer_->Backward(grad_output);
}

void TransformerDecoderModule::SetTraining(bool training) {
    Module::SetTraining(training);
    layer_->SetTraining(training);
}

std::map<std::string, Tensor> TransformerDecoderModule::GetParameters() {
    std::map<std::string, Tensor> params;
    for (const auto& [key, value] : layer_->GetParameters()) {
        if (!IsGradientParameterKey(key)) {
            params[key] = value;
        }
    }
    return params;
}

void TransformerDecoderModule::SetParameters(
    const std::map<std::string, Tensor>& params) {
    layer_->SetParameters(params);
}

std::map<std::string, Tensor> TransformerDecoderModule::GetGradients() {
    std::map<std::string, Tensor> grads;
    for (const auto& [key, value] : layer_->GetParameters()) {
        if (IsGradientParameterKey(key)) {
            grads[NormalizeGradientParameterKey(key)] = value;
        }
    }
    return grads;
}

std::string TransformerDecoderModule::GetName() const {
    return "TransformerDecoder(d_model=" + std::to_string(d_model_) +
           ", heads=" + std::to_string(num_heads_) + ")";
}

// ============================================================================
// BatchNormModule Implementation (BatchNorm1D for MLPs)
// ============================================================================

BatchNormModule::BatchNormModule(size_t num_features, float eps, float momentum)
    : num_features_(num_features)
    , eps_(eps)
    , momentum_(momentum)
{
    // Initialize gamma (scale) to 1, beta (shift) to 0
    gamma_ = Tensor({num_features}, DataType::Float32);
    beta_ = Tensor({num_features}, DataType::Float32);
    running_mean_ = Tensor({num_features}, DataType::Float32);
    running_var_ = Tensor({num_features}, DataType::Float32);
    grad_gamma_ = Tensor({num_features}, DataType::Float32);
    grad_beta_ = Tensor({num_features}, DataType::Float32);

#ifdef CYXWIZ_HAS_ARRAYFIRE
    gamma_ = Tensor(af::constant(1.0f, num_features));
    beta_ = Tensor(af::constant(0.0f, num_features));
    running_mean_ = Tensor(af::constant(0.0f, num_features));
    running_var_ = Tensor(af::constant(1.0f, num_features));
#else
    float* gamma_data = gamma_.Data<float>();
    float* beta_data = beta_.Data<float>();
    float* rm_data = running_mean_.Data<float>();
    float* rv_data = running_var_.Data<float>();
    for (size_t i = 0; i < num_features; ++i) {
        gamma_data[i] = 1.0f;
        beta_data[i] = 0.0f;
        rm_data[i] = 0.0f;
        rv_data[i] = 1.0f;
    }
#endif

    spdlog::debug("BatchNormModule({}) initialized", num_features);
}

Tensor BatchNormModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();

    const auto& shape = input.Shape();
    size_t batch_size = shape[0];
    size_t features = shape.size() > 1 ? shape[1] : shape[0];

#ifdef CYXWIZ_HAS_ARRAYFIRE
    af::array x = input.GetArray();
    // x is [batch, features] in our row-major view
    // ArrayFire sees it as [batch, features] with dims(0)=batch, dims(1)=features

    // gamma/beta are [features], we need them as [1, features] for broadcasting
    af::array gamma = af::moddims(gamma_.GetArray(), 1, features);
    af::array beta = af::moddims(beta_.GetArray(), 1, features);

    if (is_training_) {
        // Compute mean and variance per feature (across batch = dim 0)
        af::array mean = af::mean(x, 0);  // [1, features]
        af::array var = af::var(x, AF_VARIANCE_POPULATION, 0);  // [1, features]

        // Update running statistics
        af::array rm = af::moddims(running_mean_.GetArray(), 1, features);
        af::array rv = af::moddims(running_var_.GetArray(), 1, features);
        rm = (1.0f - momentum_) * rm + momentum_ * mean;
        rv = (1.0f - momentum_) * rv + momentum_ * var;
        running_mean_ = Tensor(af::flat(rm));
        running_var_ = Tensor(af::flat(rv));

        // Normalize: (x - mean) / sqrt(var + eps)
        af::array std_inv = 1.0f / af::sqrt(var + eps_);
        // Tile mean and std_inv to [batch, features]
        af::array mean_tiled = af::tile(mean, batch_size, 1);
        af::array std_inv_tiled = af::tile(std_inv, batch_size, 1);
        af::array x_norm = (x - mean_tiled) * std_inv_tiled;

        // Scale and shift: gamma * x_norm + beta
        af::array gamma_tiled = af::tile(gamma, batch_size, 1);
        af::array beta_tiled = af::tile(beta, batch_size, 1);
        af::array out = gamma_tiled * x_norm + beta_tiled;

        // Cache for backward
        normalized_ = Tensor(x_norm);
        std_inv_ = Tensor(af::flat(std_inv));
        batch_mean_ = Tensor(af::flat(mean));

        return Tensor(out);
    } else {
        // Inference mode: use running statistics
        af::array rm = af::moddims(running_mean_.GetArray(), 1, features);
        af::array rv = af::moddims(running_var_.GetArray(), 1, features);
        af::array std_inv = 1.0f / af::sqrt(rv + eps_);

        af::array rm_tiled = af::tile(rm, batch_size, 1);
        af::array std_inv_tiled = af::tile(std_inv, batch_size, 1);
        af::array x_norm = (x - rm_tiled) * std_inv_tiled;

        af::array gamma_tiled = af::tile(gamma, batch_size, 1);
        af::array beta_tiled = af::tile(beta, batch_size, 1);
        af::array out = gamma_tiled * x_norm + beta_tiled;

        return Tensor(out);
    }
#else
    // CPU fallback
    Tensor output({batch_size, features}, DataType::Float32);
    const float* x_data = input.Data<float>();
    float* out_data = output.Data<float>();
    const float* gamma_data = gamma_.Data<float>();
    const float* beta_data = beta_.Data<float>();

    if (is_training_) {
        // Compute mean per feature
        std::vector<float> mean(features, 0.0f);
        std::vector<float> var(features, 0.0f);

        for (size_t f = 0; f < features; ++f) {
            for (size_t b = 0; b < batch_size; ++b) {
                mean[f] += x_data[b * features + f];
            }
            mean[f] /= batch_size;
        }

        // Compute variance per feature
        for (size_t f = 0; f < features; ++f) {
            for (size_t b = 0; b < batch_size; ++b) {
                float diff = x_data[b * features + f] - mean[f];
                var[f] += diff * diff;
            }
            var[f] /= batch_size;
        }

        // Update running statistics
        float* rm_data = running_mean_.Data<float>();
        float* rv_data = running_var_.Data<float>();
        for (size_t f = 0; f < features; ++f) {
            rm_data[f] = (1.0f - momentum_) * rm_data[f] + momentum_ * mean[f];
            rv_data[f] = (1.0f - momentum_) * rv_data[f] + momentum_ * var[f];
        }

        // Normalize, scale, shift
        normalized_ = Tensor({batch_size, features}, DataType::Float32);
        std_inv_ = Tensor({features}, DataType::Float32);
        float* norm_data = normalized_.Data<float>();
        float* std_inv_data = std_inv_.Data<float>();

        for (size_t f = 0; f < features; ++f) {
            std_inv_data[f] = 1.0f / std::sqrt(var[f] + eps_);
        }

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t f = 0; f < features; ++f) {
                float x_norm = (x_data[b * features + f] - mean[f]) * std_inv_data[f];
                norm_data[b * features + f] = x_norm;
                out_data[b * features + f] = x_norm * gamma_data[f] + beta_data[f];
            }
        }
    } else {
        // Inference mode
        const float* rm_data = running_mean_.Data<float>();
        const float* rv_data = running_var_.Data<float>();

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t f = 0; f < features; ++f) {
                float std_inv = 1.0f / std::sqrt(rv_data[f] + eps_);
                float x_norm = (x_data[b * features + f] - rm_data[f]) * std_inv;
                out_data[b * features + f] = x_norm * gamma_data[f] + beta_data[f];
            }
        }
    }

    return output;
#endif
}

Tensor BatchNormModule::Backward(const Tensor& grad_output) {
    const auto& shape = grad_output.Shape();
    size_t batch_size = shape[0];
    size_t features = shape.size() > 1 ? shape[1] : shape[0];

#ifdef CYXWIZ_HAS_ARRAYFIRE
    af::array grad = grad_output.GetArray();
    // grad is [batch, features]

    // Get cached values - x_norm is [batch, features]
    af::array x_norm = normalized_.GetArray();

    // gamma is [features], reshape to [1, features] for broadcasting
    af::array gamma = af::moddims(gamma_.GetArray(), 1, features);
    af::array std_inv = af::moddims(std_inv_.GetArray(), 1, features);

    // Gradient w.r.t gamma and beta (sum along batch dimension = dim 0)
    af::array d_gamma = af::sum(grad * x_norm, 0);  // [1, features]
    af::array d_beta = af::sum(grad, 0);  // [1, features]

    grad_gamma_ = Tensor(af::flat(d_gamma));
    grad_beta_ = Tensor(af::flat(d_beta));

    // Gradient w.r.t input
    // d_x = (1/N) * std_inv * (N * d_y * gamma - sum(d_y * gamma) - x_norm * sum(d_y * gamma * x_norm))
    float N = static_cast<float>(batch_size);

    af::array gamma_tiled = af::tile(gamma, batch_size, 1);  // [batch, features]
    af::array d_x_norm = grad * gamma_tiled;

    af::array sum_d_x_norm = af::tile(af::sum(d_x_norm, 0), batch_size, 1);  // [batch, features]
    af::array sum_d_x_norm_x_norm = af::tile(af::sum(d_x_norm * x_norm, 0), batch_size, 1);  // [batch, features]

    af::array std_inv_tiled = af::tile(std_inv, batch_size, 1);  // [batch, features]
    af::array d_x = (1.0f / N) * std_inv_tiled *
                    (N * d_x_norm - sum_d_x_norm - x_norm * sum_d_x_norm_x_norm);

    return Tensor(d_x);
#else
    // CPU fallback
    Tensor grad_input({batch_size, features}, DataType::Float32);
    const float* grad_data = grad_output.Data<float>();
    const float* x_norm_data = normalized_.Data<float>();
    const float* std_inv_data = std_inv_.Data<float>();
    const float* gamma_data = gamma_.Data<float>();
    float* grad_input_data = grad_input.Data<float>();
    float* d_gamma_data = grad_gamma_.Data<float>();
    float* d_beta_data = grad_beta_.Data<float>();

    // Initialize gradients
    for (size_t f = 0; f < features; ++f) {
        d_gamma_data[f] = 0.0f;
        d_beta_data[f] = 0.0f;
    }

    // Compute d_gamma, d_beta
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t f = 0; f < features; ++f) {
            d_gamma_data[f] += grad_data[b * features + f] * x_norm_data[b * features + f];
            d_beta_data[f] += grad_data[b * features + f];
        }
    }

    // Compute d_x
    float N = static_cast<float>(batch_size);
    for (size_t f = 0; f < features; ++f) {
        float sum_d_x_norm = 0.0f;
        float sum_d_x_norm_x_norm = 0.0f;

        for (size_t b = 0; b < batch_size; ++b) {
            float d_x_norm = grad_data[b * features + f] * gamma_data[f];
            sum_d_x_norm += d_x_norm;
            sum_d_x_norm_x_norm += d_x_norm * x_norm_data[b * features + f];
        }

        for (size_t b = 0; b < batch_size; ++b) {
            float d_x_norm = grad_data[b * features + f] * gamma_data[f];
            grad_input_data[b * features + f] = (1.0f / N) * std_inv_data[f] *
                (N * d_x_norm - sum_d_x_norm - x_norm_data[b * features + f] * sum_d_x_norm_x_norm);
        }
    }

    return grad_input;
#endif
}

std::map<std::string, Tensor> BatchNormModule::GetParameters() {
    return {
        {"gamma", gamma_},
        {"beta", beta_}
    };
}

void BatchNormModule::SetParameters(const std::map<std::string, Tensor>& params) {
    if (params.count("gamma")) gamma_ = params.at("gamma");
    if (params.count("beta")) beta_ = params.at("beta");
    if (params.count("running_mean")) running_mean_ = params.at("running_mean");
    if (params.count("running_var")) running_var_ = params.at("running_var");
}

std::map<std::string, Tensor> BatchNormModule::GetGradients() {
    return {
        {"gamma", grad_gamma_},
        {"beta", grad_beta_}
    };
}

std::string BatchNormModule::GetName() const {
    return "BatchNorm(" + std::to_string(num_features_) + ")";
}

// ============================================================================
// SoftmaxModule Implementation (ArrayFire)
// ============================================================================

SoftmaxModule::SoftmaxModule(int dim) : dim_(dim) {}

Tensor SoftmaxModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();

#ifdef CYXWIZ_HAS_ARRAYFIRE
    // ArrayFire implementation
    af::array x = input.GetArray();

    // Softmax: exp(x - max) / sum(exp(x - max))
    // Compute along dim 1 (classes dimension) for [batch, classes] input
    // Note: Use (af::max) to prevent Windows macro conflict
    af::array max_vals = (af::max)(x, 1);  // [batch, 1]
    af::array x_shifted = x - af::tile(max_vals, 1, static_cast<unsigned>(x.dims(1)));  // Subtract max for stability
    af::array exp_x = af::exp(x_shifted);
    af::array sum_exp = af::sum(exp_x, 1);  // [batch, 1]
    af::array softmax = exp_x / af::tile(sum_exp, 1, static_cast<unsigned>(x.dims(1)));

    Tensor output(softmax);
    output_cache_ = output.Clone();
    return output;
#else
    // CPU fallback
    const auto& shape = input.Shape();
    size_t batch_size = shape[0];
    size_t num_classes = shape.size() > 1 ? shape[1] : shape[0];

    Tensor output({batch_size, num_classes}, DataType::Float32);
    const float* in_data = input.Data<float>();
    float* out_data = output.Data<float>();

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
    af::array grad = grad_output.GetArray();
    af::array soft = output_cache_.GetArray();

    // Compute dot product per sample: sum(grad * softmax) along classes dimension
    af::array dot = af::sum(grad * soft, 1);  // [batch, 1]

    // grad_input = softmax * (grad - dot)
    af::array grad_input = soft * (grad - af::tile(dot, 1, static_cast<unsigned>(grad.dims(1))));

    return Tensor(grad_input);
#else
    // CPU fallback
    const auto& shape = grad_output.Shape();
    size_t batch_size = shape[0];
    size_t num_classes = shape.size() > 1 ? shape[1] : shape[0];

    Tensor grad_input({batch_size, num_classes}, DataType::Float32);
    const float* grad_data = grad_output.Data<float>();
    const float* soft_data = output_cache_.Data<float>();
    float* out_data = grad_input.Data<float>();

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

DropoutModule::DropoutModule(float p) : p_(p) {
    if (p < 0.0f || p > 1.0f) {
        spdlog::warn("DropoutModule: p={} out of range [0,1], clamping", p);
        p_ = std::clamp(p, 0.0f, 1.0f);
    }
}

Tensor DropoutModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();

    // During eval, just return input
    if (!is_training_) {
        return input.Clone();
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    // ArrayFire implementation
    af::array x = input.GetArray();
    float scale = 1.0f / (1.0f - p_);

    // Generate random mask: values > p are kept (scaled), values <= p are dropped
    af::array rand_vals = af::randu(x.dims());
    af::array keep_mask = (rand_vals > p_).as(af::dtype::f32);  // 1 for keep, 0 for drop
    af::array scaled_mask = keep_mask * scale;

    // Store mask for backward pass
    mask_ = Tensor(scaled_mask);

    // Apply dropout
    af::array output = x * scaled_mask;
    return Tensor(output);
#else
    // CPU fallback
    const auto& shape = input.Shape();
    size_t total = input.NumElements();

    Tensor output(shape, input.GetDataType());
    mask_ = Tensor(shape, DataType::Float32);

    const float* in_data = input.Data<float>();
    float* out_data = output.Data<float>();
    float* mask_data = mask_.Data<float>();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float scale = 1.0f / (1.0f - p_);

    for (size_t i = 0; i < total; ++i) {
        if (dist(gen) > p_) {
            mask_data[i] = scale;
            out_data[i] = in_data[i] * scale;
        } else {
            mask_data[i] = 0.0f;
            out_data[i] = 0.0f;
        }
    }
    return output;
#endif
}

Tensor DropoutModule::Backward(const Tensor& grad_output) {
    if (!is_training_) {
        return grad_output.Clone();
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    // ArrayFire implementation
    af::array grad = grad_output.GetArray();
    af::array mask = mask_.GetArray();

    // grad_input = grad * mask (mask already has scaling applied)
    af::array grad_input = grad * mask;
    return Tensor(grad_input);
#else
    // CPU fallback
    const auto& shape = grad_output.Shape();
    Tensor grad_input(shape, DataType::Float32);

    const float* grad_data = grad_output.Data<float>();
    const float* mask_data = mask_.Data<float>();
    float* out_data = grad_input.Data<float>();

    size_t total = grad_output.NumElements();
    for (size_t i = 0; i < total; ++i) {
        out_data[i] = grad_data[i] * mask_data[i];
    }
    return grad_input;
#endif
}

std::string DropoutModule::GetName() const {
    return "Dropout(p=" + std::to_string(p_) + ")";
}

// ============================================================================
// FlattenModule Implementation (ArrayFire)
// ============================================================================

FlattenModule::FlattenModule(int start_dim) : start_dim_(start_dim) {}

Tensor FlattenModule::Forward(const Tensor& input) {
    original_shape_ = input.Shape();

    // Calculate flattened size from start_dim onwards
    size_t batch_size = 1;
    size_t flat_size = 1;

    for (size_t i = 0; i < original_shape_.size(); ++i) {
        if (static_cast<int>(i) < start_dim_) {
            batch_size *= original_shape_[i];
        } else {
            flat_size *= original_shape_[i];
        }
    }

    // Pure CPU reshape — just copy data with new shape. Flatten has no
    // computation, and going through ArrayFire's moddims scrambles the
    // row-major data layout (column-major AF produces transposed output
    // that LinearLayer can't consume). This approach is both correct
    // and faster than a GPU round-trip for a zero-compute operation.
    const float* in_data = input.Data<float>();
    return Tensor({batch_size, flat_size}, in_data, input.GetDataType());
}

Tensor FlattenModule::Backward(const Tensor& grad_output) {
    // Pure CPU reshape back to original shape (same as Forward).
    const float* grad_data = grad_output.Data<float>();
    return Tensor(original_shape_, grad_data, grad_output.GetDataType());
}

// ============================================================================
// ReshapeModule Implementation
// ============================================================================

ReshapeModule::ReshapeModule(std::vector<size_t> target_sample_shape)
    : target_sample_shape_(std::move(target_sample_shape)) {
    if (target_sample_shape_.empty()) {
        throw std::runtime_error("ReshapeModule: target sample shape must not be empty");
    }
}

Tensor ReshapeModule::Forward(const Tensor& input) {
    original_shape_ = input.Shape();
    if (original_shape_.empty()) {
        throw std::runtime_error("ReshapeModule: input must include a batch dimension");
    }

    std::vector<size_t> target_shape;
    target_shape.reserve(target_sample_shape_.size() + 1);
    target_shape.push_back(original_shape_[0]);
    target_shape.insert(target_shape.end(),
                        target_sample_shape_.begin(),
                        target_sample_shape_.end());

    return input.Reshape(target_shape);
}

Tensor ReshapeModule::Backward(const Tensor& grad_output) {
    return grad_output.Reshape(original_shape_);
}

// ============================================================================
// PermuteModule Implementation
// ============================================================================

PermuteModule::PermuteModule(std::vector<int> sample_dims)
    : sample_dims_(std::move(sample_dims)) {
    if (sample_dims_.empty()) {
        throw std::runtime_error("PermuteModule: sample dims must not be empty");
    }

    inverse_sample_dims_.resize(sample_dims_.size());
    for (size_t i = 0; i < sample_dims_.size(); ++i) {
        const int dim = sample_dims_[i];
        if (dim < 0 || dim >= static_cast<int>(sample_dims_.size())) {
            throw std::runtime_error("PermuteModule: sample dims must be normalized");
        }
        inverse_sample_dims_[static_cast<size_t>(dim)] = static_cast<int>(i);
    }
}

Tensor PermuteModule::Forward(const Tensor& input) {
    if (input.Shape().size() != sample_dims_.size() + 1) {
        throw std::runtime_error("PermuteModule: input rank does not match sample dims");
    }

    std::vector<int> full_dims;
    full_dims.reserve(sample_dims_.size() + 1);
    full_dims.push_back(0);
    for (int dim : sample_dims_) {
        full_dims.push_back(dim + 1);
    }
    return input.Permute(full_dims);
}

Tensor PermuteModule::Backward(const Tensor& grad_output) {
    std::vector<int> full_inverse_dims;
    full_inverse_dims.reserve(inverse_sample_dims_.size() + 1);
    full_inverse_dims.push_back(0);
    for (int dim : inverse_sample_dims_) {
        full_inverse_dims.push_back(dim + 1);
    }
    return grad_output.Permute(full_inverse_dims);
}

// ============================================================================
// TensorUnaryModule Implementation
// ============================================================================

TensorUnaryModule::TensorUnaryModule(TensorUnaryOp op,
                                     float scalar,
                                     float scalar2)
    : op_(op),
      scalar_(scalar),
      scalar2_(scalar2) {}

Tensor TensorUnaryModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();

    Tensor output;
    switch (op_) {
        case TensorUnaryOp::Abs:
            output = input.Abs();
            break;
        case TensorUnaryOp::Exp:
            output = input.Exp();
            break;
        case TensorUnaryOp::Log:
            output = input.Log();
            break;
        case TensorUnaryOp::Sqrt:
            output = input.Sqrt();
            break;
        case TensorUnaryOp::Sign:
            output = input.Sign();
            break;
        case TensorUnaryOp::Pow:
            output = input.Pow(scalar_);
            break;
        case TensorUnaryOp::Clip:
            output = input.Clip(scalar_, scalar2_);
            break;
        default:
            throw std::runtime_error("TensorUnaryModule: unsupported unary op");
    }

    output_cache_ = output.Clone();
    return output;
}

Tensor TensorUnaryModule::Backward(const Tensor& grad_output) {
    switch (op_) {
        case TensorUnaryOp::Abs:
            return grad_output * input_cache_.Sign();
        case TensorUnaryOp::Exp:
            return grad_output * output_cache_;
        case TensorUnaryOp::Log:
            return grad_output / input_cache_;
        case TensorUnaryOp::Sqrt:
            return grad_output / (output_cache_ * 2.0f);
        case TensorUnaryOp::Sign:
            return Tensor::Zeros(grad_output.Shape(), grad_output.GetDataType());
        case TensorUnaryOp::Pow:
            if (scalar_ == 0.0f) {
                return Tensor::Zeros(grad_output.Shape(), grad_output.GetDataType());
            }
            return grad_output * (input_cache_.Pow(scalar_ - 1.0f) * scalar_);
        case TensorUnaryOp::Clip: {
            Tensor mask = Tensor::Zeros(grad_output.Shape(), grad_output.GetDataType());
            for (size_t i = 0; i < input_cache_.NumElements(); ++i) {
                const float value = input_cache_.At(i);
                if (value >= scalar_ && value <= scalar2_) {
                    mask.Set(i, 1.0f);
                }
            }
            return grad_output * mask;
        }
        default:
            throw std::runtime_error("TensorUnaryModule: unsupported unary op");
    }
}

std::string TensorUnaryModule::GetName() const {
    switch (op_) {
        case TensorUnaryOp::Abs:
            return "TensorAbs";
        case TensorUnaryOp::Exp:
            return "TensorExp";
        case TensorUnaryOp::Log:
            return "TensorLog";
        case TensorUnaryOp::Sqrt:
            return "TensorSqrt";
        case TensorUnaryOp::Sign:
            return "TensorSign";
        case TensorUnaryOp::Pow:
            return "TensorPow";
        case TensorUnaryOp::Clip:
            return "TensorClip";
        default:
            return "TensorUnary";
    }
}

// ============================================================================
// TensorReductionModule Implementation
// ============================================================================

TensorReductionModule::TensorReductionModule(TensorReductionOp op,
                                             int dim,
                                             bool keepdim)
    : op_(op),
      dim_(dim),
      keepdim_(keepdim) {}

Tensor TensorReductionModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    original_shape_ = input.Shape();
    if (original_shape_.empty()) {
        throw std::runtime_error("TensorReductionModule: input must include a batch dimension");
    }

    const size_t sample_rank = original_shape_.size() - 1;
    if (sample_rank == 0) {
        throw std::runtime_error("TensorReductionModule: input must include sample dimensions");
    }

    Tensor output = input.Clone();
    reduced_count_ = 1;

    if (dim_ == -1) {
        for (size_t i = 1; i < original_shape_.size(); ++i) {
            reduced_count_ *= original_shape_[i];
        }
        if (reduced_count_ == 0) {
            throw std::runtime_error("TensorReductionModule: cannot reduce empty input");
        }
        if (op_ == TensorReductionOp::Var || op_ == TensorReductionOp::Std) {
            std::vector<size_t> reduced_shape = keepdim_
                ? std::vector<size_t>(original_shape_.size(), 1)
                : std::vector<size_t>{original_shape_[0], 1};
            reduced_shape[0] = original_shape_[0];
            const DataType out_dtype = input.GetDataType() == DataType::Float64
                ? DataType::Float64
                : DataType::Float32;
            output = Tensor(reduced_shape, out_dtype);
            for (size_t batch = 0; batch < original_shape_[0]; ++batch) {
                const size_t offset = batch * reduced_count_;
                double total = 0.0;
                for (size_t i = 0; i < reduced_count_; ++i) {
                    total += static_cast<double>(input.At(offset + i));
                }
                const double mean = total / static_cast<double>(reduced_count_);
                double variance = 0.0;
                for (size_t i = 0; i < reduced_count_; ++i) {
                    const double diff = static_cast<double>(input.At(offset + i)) - mean;
                    variance += diff * diff;
                }
                variance /= static_cast<double>(reduced_count_);
                const double value = op_ == TensorReductionOp::Std
                    ? std::sqrt(variance)
                    : variance;
                output.Set(batch, static_cast<float>(value));
            }
            output_shape_ = output.Shape();
            output_cache_ = output.Clone();
            return output;
        }
        for (int axis = static_cast<int>(sample_rank); axis >= 1; --axis) {
            switch (op_) {
                case TensorReductionOp::Sum:
                case TensorReductionOp::Mean:
                    output = output.Sum(axis, true);
                    break;
                case TensorReductionOp::Max:
                    output = output.Max(axis, true);
                    break;
                case TensorReductionOp::Min:
                    output = output.Min(axis, true);
                    break;
                case TensorReductionOp::Prod:
                    output = output.Prod(axis, true);
                    break;
                default:
                    throw std::runtime_error("TensorReductionModule: unsupported reduction op");
            }
        }
        if (op_ == TensorReductionOp::Mean) {
            output = output / static_cast<float>(reduced_count_);
        }
        if (!keepdim_) {
            output = output.Reshape({original_shape_[0], 1});
        }
        output_shape_ = output.Shape();
        output_cache_ = output.Clone();
        return output;
    }

    if (dim_ < 0 || dim_ >= static_cast<int>(sample_rank)) {
        throw std::runtime_error("TensorReductionModule: dim is out of range");
    }

    const int full_dim = dim_ + 1;
    reduced_count_ = original_shape_[static_cast<size_t>(full_dim)];
    switch (op_) {
        case TensorReductionOp::Sum:
            output = input.Sum(full_dim, keepdim_);
            break;
        case TensorReductionOp::Mean:
            output = input.Mean(full_dim, keepdim_);
            break;
        case TensorReductionOp::Max:
            output = input.Max(full_dim, keepdim_);
            break;
        case TensorReductionOp::Min:
            output = input.Min(full_dim, keepdim_);
            break;
        case TensorReductionOp::Prod:
            output = input.Prod(full_dim, keepdim_);
            break;
        case TensorReductionOp::Var:
            output = input.Var(full_dim, keepdim_);
            break;
        case TensorReductionOp::Std:
            output = input.Std(full_dim, keepdim_);
            break;
        default:
            throw std::runtime_error("TensorReductionModule: unsupported reduction op");
    }
    if (!keepdim_ && output.Shape().size() == 1) {
        output = output.Reshape({original_shape_[0], 1});
    }

    output_shape_ = output.Shape();
    output_cache_ = output.Clone();
    return output;
}

Tensor TensorReductionModule::Backward(const Tensor& grad_output) {
    Tensor grad_input = Tensor::Zeros(original_shape_, grad_output.GetDataType());
    const int full_dim = dim_ >= 0 ? dim_ + 1 : -1;
    const float scale = op_ == TensorReductionOp::Mean
        ? 1.0f / static_cast<float>(reduced_count_)
        : 1.0f;

    auto grad_index_for_input = [&](const std::vector<size_t>& input_indices) {
        std::vector<size_t> grad_indices;

        if (dim_ == -1) {
            if (keepdim_) {
                grad_indices.assign(original_shape_.size(), 0);
                grad_indices[0] = input_indices[0];
            } else {
                grad_indices = {input_indices[0], 0};
            }
        } else {
            grad_indices.reserve(output_shape_.size());
            for (size_t axis = 0; axis < original_shape_.size(); ++axis) {
                if (static_cast<int>(axis) == full_dim) {
                    if (keepdim_) {
                        grad_indices.push_back(0);
                    }
                } else {
                    grad_indices.push_back(input_indices[axis]);
                }
            }
            if (!keepdim_ && grad_indices.size() == 1) {
                grad_indices.push_back(0);
            }
        }

        return RavelIndex(grad_indices, output_shape_);
    };

    std::vector<size_t> tie_counts(output_cache_.NumElements(), 0);
    std::vector<double> group_sums(output_cache_.NumElements(), 0.0);
    std::vector<double> group_nonzero_products(output_cache_.NumElements(), 1.0);
    std::vector<size_t> group_zero_counts(output_cache_.NumElements(), 0);
    if (op_ == TensorReductionOp::Max || op_ == TensorReductionOp::Min) {
        for (size_t i = 0; i < original_shape_.size(); ++i) {
            if (original_shape_[i] == 0) {
                throw std::runtime_error("TensorReductionModule: cannot reduce empty input");
            }
        }
        for (size_t i = 0; i < input_cache_.NumElements(); ++i) {
            const std::vector<size_t> input_indices = UnravelIndex(i, original_shape_);
            const size_t grad_index = grad_index_for_input(input_indices);
            if (input_cache_.At(i) == output_cache_.At(grad_index)) {
                tie_counts[grad_index] += 1;
            }
        }
    } else if (op_ == TensorReductionOp::Prod ||
               op_ == TensorReductionOp::Var ||
               op_ == TensorReductionOp::Std) {
        for (size_t i = 0; i < input_cache_.NumElements(); ++i) {
            const std::vector<size_t> input_indices = UnravelIndex(i, original_shape_);
            const size_t grad_index = grad_index_for_input(input_indices);
            const float input_value = input_cache_.At(i);
            if (op_ == TensorReductionOp::Prod) {
                if (input_value == 0.0f) {
                    group_zero_counts[grad_index] += 1;
                } else {
                    group_nonzero_products[grad_index] *= static_cast<double>(input_value);
                }
            } else {
                group_sums[grad_index] += static_cast<double>(input_value);
            }
        }
    }

    for (size_t i = 0; i < grad_input.NumElements(); ++i) {
        const std::vector<size_t> input_indices = UnravelIndex(i, original_shape_);
        const size_t grad_index = grad_index_for_input(input_indices);
        float value = grad_output.At(grad_index) * scale;
        if (op_ == TensorReductionOp::Max || op_ == TensorReductionOp::Min) {
            if (input_cache_.At(i) != output_cache_.At(grad_index)) {
                value = 0.0f;
            } else {
                value /= static_cast<float>(tie_counts[grad_index]);
            }
        } else if (op_ == TensorReductionOp::Prod) {
            const float input_value = input_cache_.At(i);
            double derivative = 0.0;
            if (group_zero_counts[grad_index] == 0) {
                derivative = group_nonzero_products[grad_index] /
                    static_cast<double>(input_value);
            } else if (group_zero_counts[grad_index] == 1 && input_value == 0.0f) {
                derivative = group_nonzero_products[grad_index];
            }
            value = grad_output.At(grad_index) * static_cast<float>(derivative);
        } else if (op_ == TensorReductionOp::Var ||
                   op_ == TensorReductionOp::Std) {
            const double mean = group_sums[grad_index] /
                static_cast<double>(reduced_count_);
            const double centered = static_cast<double>(input_cache_.At(i)) - mean;
            double derivative = 2.0 * centered / static_cast<double>(reduced_count_);
            if (op_ == TensorReductionOp::Std) {
                const double std_value = static_cast<double>(output_cache_.At(grad_index));
                derivative = std_value == 0.0
                    ? 0.0
                    : centered / (static_cast<double>(reduced_count_) * std_value);
            }
            value = grad_output.At(grad_index) * static_cast<float>(derivative);
        }
        grad_input.Set(i, value);
    }

    return grad_input;
}

std::string TensorReductionModule::GetName() const {
    switch (op_) {
        case TensorReductionOp::Sum:
            return "TensorSum";
        case TensorReductionOp::Mean:
            return "TensorMean";
        case TensorReductionOp::Max:
            return "TensorMax";
        case TensorReductionOp::Min:
            return "TensorMin";
        case TensorReductionOp::Prod:
            return "TensorProd";
        case TensorReductionOp::Var:
            return "TensorVar";
        case TensorReductionOp::Std:
            return "TensorStd";
        default:
            return "TensorReduction";
    }
}

// ============================================================================
// TensorShapeModule Implementation
// ============================================================================

TensorShapeModule::TensorShapeModule(TensorShapeOp op,
                                     std::vector<size_t> target_shape,
                                     int dim,
                                     std::vector<int> indices)
    : op_(op),
      target_shape_(std::move(target_shape)),
      dim_(dim),
      indices_(std::move(indices)) {}

Tensor TensorShapeModule::Forward(const Tensor& input) {
    original_shape_ = input.Shape();
    if (original_shape_.empty()) {
        throw std::runtime_error("TensorShapeModule: input must include a batch dimension");
    }

    const size_t sample_rank = original_shape_.size() - 1;
    if (op_ == TensorShapeOp::BroadcastTo || op_ == TensorShapeOp::Expand) {
        if (target_shape_.size() < sample_rank) {
            throw std::runtime_error("TensorShapeModule: target sample rank is too small");
        }

        sample_pad_ = target_shape_.size() - sample_rank;
        padded_input_shape_.clear();
        padded_input_shape_.reserve(target_shape_.size() + 1);
        padded_input_shape_.push_back(original_shape_[0]);
        for (size_t i = 0; i < sample_pad_; ++i) {
            padded_input_shape_.push_back(1);
        }
        for (size_t i = 1; i < original_shape_.size(); ++i) {
            padded_input_shape_.push_back(original_shape_[i]);
        }

        output_shape_.clear();
        output_shape_.reserve(target_shape_.size() + 1);
        output_shape_.push_back(original_shape_[0]);
        output_shape_.insert(output_shape_.end(), target_shape_.begin(), target_shape_.end());

        for (size_t axis = 0; axis < output_shape_.size(); ++axis) {
            const size_t in_dim = padded_input_shape_[axis];
            const size_t out_dim = output_shape_[axis];
            if (in_dim != 1 && in_dim != out_dim) {
                throw std::runtime_error("TensorShapeModule: incompatible target shape");
            }
        }

        Tensor reshaped = padded_input_shape_ == original_shape_
            ? input.Clone()
            : input.Reshape(padded_input_shape_);
        return op_ == TensorShapeOp::BroadcastTo
            ? reshaped.BroadcastTo(output_shape_)
            : reshaped.Expand(output_shape_);
    }

    if (op_ == TensorShapeOp::IndexSelect) {
        if (sample_rank == 0) {
            throw std::runtime_error("TensorShapeModule: input must include sample dimensions");
        }
        if (indices_.empty()) {
            throw std::runtime_error("TensorShapeModule: indices must not be empty");
        }
        int normalized_dim = dim_;
        if (normalized_dim < 0) {
            normalized_dim += static_cast<int>(sample_rank);
        }
        if (normalized_dim < 0 || normalized_dim >= static_cast<int>(sample_rank)) {
            throw std::runtime_error("TensorShapeModule: dim is out of range");
        }

        normalized_dim_ = normalized_dim;
        const int full_dim = normalized_dim_ + 1;
        const int dim_size = static_cast<int>(original_shape_[static_cast<size_t>(full_dim)]);
        normalized_indices_.clear();
        normalized_indices_.reserve(indices_.size());
        for (int index : indices_) {
            int normalized = index;
            if (normalized < 0) {
                normalized += dim_size;
            }
            if (normalized < 0 || normalized >= dim_size) {
                throw std::out_of_range("TensorShapeModule: selected index out of range");
            }
            normalized_indices_.push_back(normalized);
        }

        output_shape_ = original_shape_;
        output_shape_[static_cast<size_t>(full_dim)] = normalized_indices_.size();
        return input.IndexSelect(full_dim, indices_);
    }

    throw std::runtime_error("TensorShapeModule: unsupported shape op");
}

Tensor TensorShapeModule::Backward(const Tensor& grad_output) {
    Tensor grad_input = Tensor::Zeros(original_shape_, grad_output.GetDataType());

    if (op_ == TensorShapeOp::BroadcastTo || op_ == TensorShapeOp::Expand) {
        for (size_t i = 0; i < grad_output.NumElements(); ++i) {
            const std::vector<size_t> out_indices = UnravelIndex(i, output_shape_);
            std::vector<size_t> padded_indices(padded_input_shape_.size(), 0);
            for (size_t axis = 0; axis < output_shape_.size(); ++axis) {
                padded_indices[axis] = padded_input_shape_[axis] == 1 ? 0 : out_indices[axis];
            }

            std::vector<size_t> input_indices;
            input_indices.reserve(original_shape_.size());
            input_indices.push_back(padded_indices[0]);
            for (size_t axis = 1 + sample_pad_; axis < padded_indices.size(); ++axis) {
                input_indices.push_back(padded_indices[axis]);
            }

            const size_t input_index = RavelIndex(input_indices, original_shape_);
            grad_input.Set(input_index, grad_input.At(input_index) + grad_output.At(i));
        }
        return grad_input;
    }

    if (op_ == TensorShapeOp::IndexSelect) {
        const int full_dim = normalized_dim_ + 1;
        for (size_t i = 0; i < grad_output.NumElements(); ++i) {
            std::vector<size_t> out_indices = UnravelIndex(i, output_shape_);
            std::vector<size_t> input_indices = out_indices;
            input_indices[static_cast<size_t>(full_dim)] =
                static_cast<size_t>(normalized_indices_[out_indices[static_cast<size_t>(full_dim)]]);

            const size_t input_index = RavelIndex(input_indices, original_shape_);
            grad_input.Set(input_index, grad_input.At(input_index) + grad_output.At(i));
        }
        return grad_input;
    }

    throw std::runtime_error("TensorShapeModule: unsupported shape op");
}

std::string TensorShapeModule::GetName() const {
    switch (op_) {
        case TensorShapeOp::BroadcastTo:
            return "TensorBroadcastTo";
        case TensorShapeOp::Expand:
            return "TensorExpand";
        case TensorShapeOp::IndexSelect:
            return "TensorIndexSelect";
        default:
            return "TensorShape";
    }
}

// ============================================================================
// TensorMaskModule Implementation
// ============================================================================

TensorMaskModule::TensorMaskModule(TensorMaskOp op, float scalar)
    : op_(op),
      scalar_(scalar) {}

Tensor TensorMaskModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    Tensor output(input.Shape(), input.GetDataType());

    for (size_t i = 0; i < input.NumElements(); ++i) {
        const float value = input.At(i);
        bool keep = false;
        switch (op_) {
            case TensorMaskOp::CompareGreater:
                keep = value > scalar_;
                break;
            case TensorMaskOp::CompareGreaterEqual:
                keep = value >= scalar_;
                break;
            case TensorMaskOp::CompareLess:
                keep = value < scalar_;
                break;
            case TensorMaskOp::CompareLessEqual:
                keep = value <= scalar_;
                break;
            case TensorMaskOp::CompareEqual:
                keep = value == scalar_;
                break;
            case TensorMaskOp::CompareNotEqual:
                keep = value != scalar_;
                break;
            case TensorMaskOp::LogicalNot:
                keep = value == 0.0f;
                break;
            default:
                throw std::runtime_error("TensorMaskModule: unsupported mask op");
        }
        output.Set(i, keep ? 1.0f : 0.0f);
    }

    return output;
}

Tensor TensorMaskModule::Backward(const Tensor& grad_output) {
    return Tensor::Zeros(input_cache_.Shape(), grad_output.GetDataType());
}

std::string TensorMaskModule::GetName() const {
    switch (op_) {
        case TensorMaskOp::CompareGreater:
            return "TensorCompareGreater";
        case TensorMaskOp::CompareGreaterEqual:
            return "TensorCompareGreaterEqual";
        case TensorMaskOp::CompareLess:
            return "TensorCompareLess";
        case TensorMaskOp::CompareLessEqual:
            return "TensorCompareLessEqual";
        case TensorMaskOp::CompareEqual:
            return "TensorCompareEqual";
        case TensorMaskOp::CompareNotEqual:
            return "TensorCompareNotEqual";
        case TensorMaskOp::LogicalNot:
            return "TensorLogicalNot";
        default:
            return "TensorMask";
    }
}

// ============================================================================
// SequentialModel Implementation
// ============================================================================

Tensor SequentialModel::Forward(const Tensor& input) {
    intermediate_outputs_.clear();
    intermediate_outputs_.reserve(modules_.size() + 1);

    Tensor current = input.Clone();
    intermediate_outputs_.push_back(input.Clone());  // Store input

    const bool trace_layers = BackendDebugHooks::HasDebugEventCallback();
    for (size_t i = 0; i < modules_.size(); ++i) {
        auto& module = modules_[i];
        const auto input_shape = trace_layers ? current.Shape() : std::vector<size_t>{};
        const auto layer_start = std::chrono::steady_clock::now();
        current = module->Forward(current);
        if (trace_layers) {
            const auto duration_ms = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - layer_start).count();
            EmitModelLayerTrace("ModelForward", i, module->GetName(),
                                input_shape, current.Shape(), duration_ms);
        }
        intermediate_outputs_.push_back(current.Clone());
    }

    return current;
}

Tensor SequentialModel::Backward(const Tensor& grad_output) {
    Tensor grad = grad_output.Clone();

    // Backward through modules in reverse order
    const bool trace_layers = BackendDebugHooks::HasDebugEventCallback();
    for (int i = static_cast<int>(modules_.size()) - 1; i >= 0; --i) {
        const auto input_shape = trace_layers ? grad.Shape() : std::vector<size_t>{};
        const auto layer_start = std::chrono::steady_clock::now();
        grad = modules_[i]->Backward(grad);
        if (trace_layers) {
            const auto duration_ms = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - layer_start).count();
            EmitModelLayerTrace("ModelBackward", static_cast<size_t>(i),
                                modules_[i]->GetName(), input_shape, grad.Shape(),
                                duration_ms);
        }
    }

    return grad;
}

std::map<std::string, Tensor> SequentialModel::GetParameters() {
    std::map<std::string, Tensor> all_params;

    for (size_t i = 0; i < modules_.size(); ++i) {
        // Skip frozen layers - their parameters won't be updated
        if (modules_[i]->HasParameters() && modules_[i]->IsTrainable()) {
            auto params = modules_[i]->GetParameters();
            for (auto& [key, tensor] : params) {
                all_params["layer" + std::to_string(i) + "." + key] = tensor;
            }
        }
    }

    return all_params;
}

void SequentialModel::SetParameters(const std::map<std::string, Tensor>& params) {
    // Group parameters by layer index
    std::map<size_t, std::map<std::string, Tensor>> layer_params;

    for (const auto& [key, tensor] : params) {
        // Parse "layerN.param_name"
        if (key.substr(0, 5) == "layer") {
            size_t dot_pos = key.find('.');
            if (dot_pos != std::string::npos) {
                size_t layer_idx = std::stoul(key.substr(5, dot_pos - 5));
                std::string param_name = key.substr(dot_pos + 1);
                layer_params[layer_idx][param_name] = tensor;
            }
        }
    }

    // Set parameters for each layer
    for (auto& [layer_idx, layer_param_map] : layer_params) {
        if (layer_idx < modules_.size()) {
            modules_[layer_idx]->SetParameters(layer_param_map);
        }
    }
}

std::map<std::string, Tensor> SequentialModel::GetGradients() {
    std::map<std::string, Tensor> all_grads;

    for (size_t i = 0; i < modules_.size(); ++i) {
        // Skip frozen layers - don't need their gradients
        if (modules_[i]->HasParameters() && modules_[i]->IsTrainable()) {
            auto grads = modules_[i]->GetGradients();
            for (auto& [key, tensor] : grads) {
                all_grads["layer" + std::to_string(i) + "." + key] = tensor;
            }
        }
    }

    return all_grads;
}

void SequentialModel::UpdateParameters(Optimizer* optimizer) {
    if (!optimizer) {
        spdlog::error("SequentialModel::UpdateParameters: No optimizer provided");
        return;
    }

    auto params = GetParameters();
    auto grads = GetGradients();

    optimizer->Step(params, grads);

    SetParameters(params);
}

void SequentialModel::SetTraining(bool training) {
    for (auto& module : modules_) {
        module->SetTraining(training);
    }
}

void SequentialModel::Summary() const {
    spdlog::info("SequentialModel Summary:");
    spdlog::info("========================");
    for (size_t i = 0; i < modules_.size(); ++i) {
        std::string frozen_marker = modules_[i]->IsTrainable() ? "" : " [FROZEN]";
        spdlog::info("  [{}] {}{}", i, modules_[i]->GetName(), frozen_marker);
    }
    spdlog::info("========================");
}

// ============================================================================
// Transfer Learning Methods
// ============================================================================

void SequentialModel::FreezeLayer(size_t layer_idx) {
    if (layer_idx < modules_.size()) {
        modules_[layer_idx]->Freeze();
        spdlog::debug("SequentialModel: Froze layer {} ({})", layer_idx, modules_[layer_idx]->GetName());
    }
}

void SequentialModel::FreezeUpTo(size_t layer_idx) {
    size_t limit = layer_idx < modules_.size() ? layer_idx : modules_.size();
    for (size_t i = 0; i < limit; ++i) {
        modules_[i]->Freeze();
    }
    if (layer_idx > 0) {
        spdlog::debug("SequentialModel: Froze layers 0 to {}", layer_idx - 1);
    }
}

void SequentialModel::FreezeExceptLast(size_t n) {
    if (modules_.size() > n) {
        FreezeUpTo(modules_.size() - n);
        spdlog::debug("SequentialModel: Froze all except last {} layers", n);
    }
}

void SequentialModel::UnfreezeAll() {
    for (auto& module : modules_) {
        module->Unfreeze();
    }
    spdlog::debug("SequentialModel: Unfroze all layers");
}

bool SequentialModel::IsLayerTrainable(size_t layer_idx) const {
    if (layer_idx < modules_.size()) {
        return modules_[layer_idx]->IsTrainable();
    }
    return false;
}

// ============================================================================
// Factory Function
// ============================================================================

std::unique_ptr<Module> CreateModule(
    ModuleType type,
    const std::map<std::string, std::string>& params)
{
    switch (type) {
        case ModuleType::Linear: {
            size_t in_features = 0;
            size_t out_features = 0;
            bool use_bias = true;

            if (params.count("in_features")) {
                in_features = std::stoul(params.at("in_features"));
            }
            if (params.count("out_features")) {
                out_features = std::stoul(params.at("out_features"));
            }
            if (params.count("units")) {
                out_features = std::stoul(params.at("units"));
            }
            if (params.count("use_bias")) {
                use_bias = params.at("use_bias") == "true";
            }

            if (in_features == 0 || out_features == 0) {
                spdlog::error("CreateModule: Linear requires in_features and out_features");
                return nullptr;
            }

            return std::make_unique<LinearModule>(in_features, out_features, use_bias);
        }

        case ModuleType::ReLU:
            return std::make_unique<ReLUModule>();

        case ModuleType::Sigmoid:
            return std::make_unique<SigmoidModule>();

        case ModuleType::Tanh:
            return std::make_unique<TanhModule>();

        case ModuleType::Softmax: {
            int dim = -1;
            if (params.count("dim")) {
                dim = std::stoi(params.at("dim"));
            }
            return std::make_unique<SoftmaxModule>(dim);
        }

        case ModuleType::Dropout: {
            float p = 0.5f;
            if (params.count("p")) {
                p = std::stof(params.at("p"));
            }
            if (params.count("rate")) {
                p = std::stof(params.at("rate"));
            }
            return std::make_unique<DropoutModule>(p);
        }

        case ModuleType::Flatten: {
            int start_dim = 1;
            if (params.count("start_dim")) {
                start_dim = std::stoi(params.at("start_dim"));
            }
            return std::make_unique<FlattenModule>(start_dim);
        }

        case ModuleType::BatchNorm: {
            size_t num_features = 0;
            float eps = 1e-5f;
            float momentum = 0.1f;

            if (params.count("num_features")) {
                num_features = std::stoul(params.at("num_features"));
            }
            if (params.count("features")) {
                num_features = std::stoul(params.at("features"));
            }
            if (params.count("eps")) {
                eps = std::stof(params.at("eps"));
            }
            if (params.count("momentum")) {
                momentum = std::stof(params.at("momentum"));
            }

            if (num_features == 0) {
                spdlog::error("CreateModule: BatchNorm requires num_features");
                return nullptr;
            }

            return std::make_unique<BatchNormModule>(num_features, eps, momentum);
        }

        case ModuleType::LeakyReLU: {
            float negative_slope = 0.01f;
            if (params.count("negative_slope")) {
                negative_slope = std::stof(params.at("negative_slope"));
            }
            return std::make_unique<LeakyReLUModule>(negative_slope);
        }

        case ModuleType::ELU: {
            float alpha = 1.0f;
            if (params.count("alpha")) {
                alpha = std::stof(params.at("alpha"));
            }
            return std::make_unique<ELUModule>(alpha);
        }

        case ModuleType::GELU:
            return std::make_unique<GELUModule>();

        case ModuleType::Swish:
            return std::make_unique<SwishModule>();

        case ModuleType::Mish:
            return std::make_unique<MishModule>();

        default:
            spdlog::error("CreateModule: Unknown module type {}", static_cast<int>(type));
            return nullptr;
    }
}

} // namespace cyxwiz


