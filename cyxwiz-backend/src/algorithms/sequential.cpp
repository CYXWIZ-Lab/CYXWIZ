#include <cyxwiz/sequential.h>
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

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

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
                                 int padding_idx)
    : num_embeddings_(num_embeddings)
    , embedding_dim_(embedding_dim)
    , padding_idx_(padding_idx)
{
    layer_ = std::make_unique<EmbeddingLayer>(
        static_cast<int>(num_embeddings),
        static_cast<int>(embedding_dim),
        padding_idx);
}

Tensor EmbeddingModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();

    // Convert [batch, seq_len] float → int32. IBatcher stores every
    // feature as float, so even integer token IDs arrive as floats
    // like 1234.0f. We static_cast to int32 here, clamping negatives
    // to 0 as a defensive fallback against NaN rounding.
    const auto& shape = input.Shape();
    size_t total = 1;
    for (auto d : shape) total *= d;

    Tensor int_input(shape, DataType::Int32);
    const float* src = input.Data<float>();
    int32_t* dst = static_cast<int32_t*>(int_input.Data());
    const int32_t vocab_max = static_cast<int32_t>(num_embeddings_) - 1;
    for (size_t i = 0; i < total; ++i) {
        int32_t idx = static_cast<int32_t>(src[i]);
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

std::string EmbeddingModule::GetName() const {
    return "Embedding(" + std::to_string(num_embeddings_) + " x " +
           std::to_string(embedding_dim_) + ")";
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

    // Slice out the last timestep: out[:, seq_len-1, :] → [batch, hd].
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

GRUModule::GRUModule(size_t input_size, size_t hidden_size,
                     size_t num_layers, bool bidirectional,
                     bool return_sequences)
    : input_size_(input_size)
    , hidden_size_(hidden_size)
    , num_layers_(num_layers)
    , bidirectional_(bidirectional)
    , return_sequences_(return_sequences)
{
    // One-shot warning: GRULayer currently has the same three-bug pattern
    // LSTMLayer had pre-2026-04-16 (AF Forward shape bug, CPU Forward
    // doesn't populate AF caches, CPU Backward returns zeros). Smoke tests
    // will run end-to-end but the loss will be flat — gradients are zero.
    // Tracked in docs/Data Studio/tofix.md "GRULayer broken AF Forward +
    // missing CPU Backward".
    static std::atomic<bool> warned{false};
    bool expected = false;
    if (warned.compare_exchange_strong(expected, true)) {
        spdlog::warn("[GRUModule] GRULayer is in a known-broken state — "
                     "AF Forward will fail and the CPU fallback returns "
                     "zero gradients in Backward. Training will run but "
                     "weights will not update. See tofix.md.");
    }

    layer_ = std::make_unique<GRULayer>(
        static_cast<int>(input_size),
        static_cast<int>(hidden_size),
        static_cast<int>(num_layers),
        /*batch_first=*/true,
        bidirectional,
        /*dropout=*/0.0f);
}

Tensor GRUModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();

    Tensor full_output = layer_->Forward(input);
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
    if (return_sequences_) {
        return layer_->Backward(grad_output);
    }

    if (last_full_output_shape_.size() != 3) {
        spdlog::warn("GRUModule::Backward called without a 3D shape cache "
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

std::map<std::string, Tensor> GRUModule::GetParameters() {
    return layer_->GetParameters();
}

void GRUModule::SetParameters(const std::map<std::string, Tensor>& params) {
    layer_->SetParameters(params);
}

std::map<std::string, Tensor> GRUModule::GetGradients() {
    // Same convention as LSTMModule — GRULayer writes "grad_*" keys
    // into its parameters map and the SequentialModel optimizer step
    // reads them via GetParameters().
    return layer_->GetParameters();
}

std::string GRUModule::GetName() const {
    const int dirs = bidirectional_ ? 2 : 1;
    return "GRU(" + std::to_string(input_size_) + " -> " +
           std::to_string(hidden_size_ * dirs) +
           (return_sequences_ ? ", seq" : ", last") + ")";
}

// ============================================================================
// ReLUModule Implementation
// ============================================================================

ReLUModule::ReLUModule() {
    activation_ = std::make_unique<ReLU>();
}

Tensor ReLUModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return activation_->Forward(input);
}

Tensor ReLUModule::Backward(const Tensor& grad_output) {
    return activation_->Backward(grad_output, input_cache_);
}

// ============================================================================
// SigmoidModule Implementation
// ============================================================================

SigmoidModule::SigmoidModule() {
    activation_ = std::make_unique<Sigmoid>();
}

Tensor SigmoidModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return activation_->Forward(input);
}

Tensor SigmoidModule::Backward(const Tensor& grad_output) {
    return activation_->Backward(grad_output, input_cache_);
}

// ============================================================================
// TanhModule Implementation
// ============================================================================

TanhModule::TanhModule() {
    activation_ = std::make_unique<Tanh>();
}

Tensor TanhModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return activation_->Forward(input);
}

Tensor TanhModule::Backward(const Tensor& grad_output) {
    return activation_->Backward(grad_output, input_cache_);
}

// ============================================================================
// LeakyReLUModule Implementation
// ============================================================================

LeakyReLUModule::LeakyReLUModule(float negative_slope)
    : negative_slope_(negative_slope)
{
    activation_ = std::make_unique<LeakyReLUActivation>(negative_slope);
}

Tensor LeakyReLUModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return activation_->Forward(input);
}

Tensor LeakyReLUModule::Backward(const Tensor& grad_output) {
    return activation_->Backward(grad_output, input_cache_);
}

std::string LeakyReLUModule::GetName() const {
    return "LeakyReLU(slope=" + std::to_string(negative_slope_) + ")";
}

// ============================================================================
// ELUModule Implementation
// ============================================================================

ELUModule::ELUModule(float alpha)
    : alpha_(alpha)
{
    activation_ = std::make_unique<ELUActivation>(alpha);
}

Tensor ELUModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return activation_->Forward(input);
}

Tensor ELUModule::Backward(const Tensor& grad_output) {
    return activation_->Backward(grad_output, input_cache_);
}

std::string ELUModule::GetName() const {
    return "ELU(alpha=" + std::to_string(alpha_) + ")";
}

// ============================================================================
// GELUModule Implementation
// ============================================================================

GELUModule::GELUModule() {
    activation_ = std::make_unique<GELUActivation>();
}

Tensor GELUModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return activation_->Forward(input);
}

Tensor GELUModule::Backward(const Tensor& grad_output) {
    return activation_->Backward(grad_output, input_cache_);
}

// ============================================================================
// SwishModule Implementation
// ============================================================================

SwishModule::SwishModule() {
    activation_ = std::make_unique<SwishActivation>();
}

Tensor SwishModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return activation_->Forward(input);
}

Tensor SwishModule::Backward(const Tensor& grad_output) {
    return activation_->Backward(grad_output, input_cache_);
}

// ============================================================================
// MishModule Implementation
// ============================================================================

MishModule::MishModule() {
    activation_ = std::make_unique<MishActivation>();
}

Tensor MishModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    return activation_->Forward(input);
}

Tensor MishModule::Backward(const Tensor& grad_output) {
    return activation_->Backward(grad_output, input_cache_);
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
// SequentialModel Implementation
// ============================================================================

Tensor SequentialModel::Forward(const Tensor& input) {
    intermediate_outputs_.clear();
    intermediate_outputs_.reserve(modules_.size() + 1);

    Tensor current = input.Clone();
    intermediate_outputs_.push_back(input.Clone());  // Store input

    for (auto& module : modules_) {
        current = module->Forward(current);
        intermediate_outputs_.push_back(current.Clone());
    }

    return current;
}

Tensor SequentialModel::Backward(const Tensor& grad_output) {
    Tensor grad = grad_output.Clone();

    // Backward through modules in reverse order
    for (int i = static_cast<int>(modules_.size()) - 1; i >= 0; --i) {
        grad = modules_[i]->Backward(grad);
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
// Serialization Implementation
// ============================================================================

using json = nlohmann::json;
namespace fs = std::filesystem;

// Helper: Get current timestamp as string
static std::string GetCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

// Helper: Convert DataType to string
static std::string DataTypeToString(DataType dtype) {
    switch (dtype) {
        case DataType::Float32: return "float32";
        case DataType::Float64: return "float64";
        case DataType::Int32: return "int32";
        case DataType::Int64: return "int64";
        case DataType::UInt8: return "uint8";
        default: return "float32";
    }
}

// Helper: Write tensor to binary stream
static void WriteTensor(std::ostream& os, const Tensor& tensor) {
    // Write shape
    auto shape = tensor.Shape();
    size_t ndims = shape.size();
    os.write(reinterpret_cast<const char*>(&ndims), sizeof(ndims));
    os.write(reinterpret_cast<const char*>(shape.data()), ndims * sizeof(size_t));

    // Write dtype
    DataType dtype = tensor.GetDataType();
    os.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));

    // Write data
    size_t num_bytes = tensor.NumBytes();
    os.write(reinterpret_cast<const char*>(&num_bytes), sizeof(num_bytes));
    os.write(reinterpret_cast<const char*>(tensor.Data()), num_bytes);
}

// Helper: Read tensor from binary stream
static Tensor ReadTensor(std::istream& is) {
    // Read shape
    size_t ndims;
    is.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));
    std::vector<size_t> shape(ndims);
    is.read(reinterpret_cast<char*>(shape.data()), ndims * sizeof(size_t));

    // Read dtype
    DataType dtype;
    is.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));

    // Read data
    size_t num_bytes;
    is.read(reinterpret_cast<char*>(&num_bytes), sizeof(num_bytes));

    Tensor tensor(shape, dtype);
    is.read(reinterpret_cast<char*>(tensor.Data()), num_bytes);

    return tensor;
}

bool SequentialModel::Save(const std::string& path) const {
    try {
        // Ensure path has .cyxmodel extension
        std::string file_path = path;
        if (file_path.size() < 9 || file_path.substr(file_path.size() - 9) != ".cyxmodel") {
            file_path += ".cyxmodel";
        }

        fs::path model_path(file_path);

        // Create directory if needed
        if (model_path.has_parent_path()) {
            fs::create_directories(model_path.parent_path());
        }

        // Prepare metadata JSON
        json meta;
        meta["metadata"]["name"] = model_name_;
        meta["metadata"]["description"] = model_description_;
        meta["metadata"]["created_at"] = GetCurrentTimestamp();
        meta["metadata"]["framework"] = "CyxWiz";
        meta["metadata"]["format_version"] = "2.0";

        // Save module info
        meta["modules"] = json::array();
        for (size_t i = 0; i < modules_.size(); ++i) {
            json module_info;
            module_info["index"] = i;
            module_info["name"] = modules_[i]->GetName();
            module_info["has_parameters"] = modules_[i]->HasParameters();
            module_info["trainable"] = modules_[i]->IsTrainable();

            if (modules_[i]->HasParameters()) {
                auto params = modules_[i]->GetParameters();
                json param_names = json::array();
                for (const auto& [name, tensor] : params) {
                    json param_info;
                    param_info["name"] = name;
                    param_info["shape"] = tensor.Shape();
                    param_info["dtype"] = DataTypeToString(tensor.GetDataType());
                    param_names.push_back(param_info);
                }
                module_info["parameters"] = param_names;
            }

            meta["modules"].push_back(module_info);
        }

        // Serialize JSON to string
        std::string json_str = meta.dump();

        // Open single .cyxmodel file
        std::ofstream file(file_path, std::ios::binary);
        if (!file) {
            spdlog::error("SequentialModel::Save: Failed to create file: {}", file_path);
            return false;
        }

        // Write header
        // Magic number: "CYXW" (4 bytes)
        const uint32_t magic = 0x43595857;
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));

        // Version: 2 for single-file format (4 bytes)
        const uint32_t version = 2;
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));

        // JSON length (8 bytes)
        uint64_t json_len = json_str.size();
        file.write(reinterpret_cast<const char*>(&json_len), sizeof(json_len));

        // JSON data
        file.write(json_str.c_str(), json_len);

        // Number of modules (8 bytes)
        size_t num_modules = modules_.size();
        file.write(reinterpret_cast<const char*>(&num_modules), sizeof(num_modules));

        // Write each module's parameters
        for (const auto& module : modules_) {
            auto params = module->GetParameters();
            size_t num_params = params.size();
            file.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));

            for (const auto& [name, tensor] : params) {
                // Write parameter name length and name
                size_t name_len = name.size();
                file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
                file.write(name.c_str(), name_len);

                // Write tensor
                WriteTensor(file, tensor);
            }
        }

        file.close();
        spdlog::info("SequentialModel::Save: Saved model to {}", file_path);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("SequentialModel::Save: Exception: {}", e.what());
        return false;
    }
}

bool SequentialModel::Load(const std::string& path) {
    try {
        // Determine file path - add .cyxmodel if not present
        std::string file_path = path;
        if (file_path.size() < 9 || file_path.substr(file_path.size() - 9) != ".cyxmodel") {
            file_path += ".cyxmodel";
        }

        // Open the .cyxmodel file
        std::ifstream file(file_path, std::ios::binary);
        if (!file) {
            spdlog::error("SequentialModel::Load: Failed to open file: {}", file_path);
            return false;
        }

        // Read and verify magic number
        uint32_t magic;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != 0x43595857) {
            spdlog::error("SequentialModel::Load: Invalid magic number (not a CyxWiz model file)");
            return false;
        }

        // Read version
        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (version != 2) {
            spdlog::error("SequentialModel::Load: Unsupported format version: {} (expected 2)", version);
            return false;
        }

        // Read JSON length and data
        uint64_t json_len;
        file.read(reinterpret_cast<char*>(&json_len), sizeof(json_len));

        std::string json_str(json_len, '\0');
        file.read(json_str.data(), json_len);

        // Parse JSON metadata
        json meta = json::parse(json_str);
        if (meta.contains("metadata")) {
            model_name_ = meta["metadata"].value("name", "");
            model_description_ = meta["metadata"].value("description", "");
        }

        // Read number of modules
        size_t num_modules;
        file.read(reinterpret_cast<char*>(&num_modules), sizeof(num_modules));

        if (num_modules != modules_.size()) {
            spdlog::error("SequentialModel::Load: Module count mismatch. Expected {}, got {}",
                         modules_.size(), num_modules);
            return false;
        }

        // Load each module's parameters
        for (size_t i = 0; i < num_modules; ++i) {
            size_t num_params;
            file.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));

            std::map<std::string, Tensor> params;
            for (size_t j = 0; j < num_params; ++j) {
                // Read parameter name
                size_t name_len;
                file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
                std::string name(name_len, '\0');
                file.read(name.data(), name_len);

                // Read tensor
                Tensor tensor = ReadTensor(file);
                params[name] = std::move(tensor);
            }

            modules_[i]->SetParameters(params);
        }

        file.close();
        spdlog::info("SequentialModel::Load: Loaded model from {} ({} modules)", file_path, num_modules);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("SequentialModel::Load: Exception: {}", e.what());
        return false;
    }
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
