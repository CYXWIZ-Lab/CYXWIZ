#include "cyxwiz/layers/attention.h"
#include <algorithm>
#include "../arrayfire_backend_utils.h"
#include "layer_arrayfire_utils.h"

#include <cmath>
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

#ifdef CYXWIZ_HAS_ARRAYFIRE
// One ArrayFire path (the CPU is ArrayFire's CPU backend): a device error is
// reported, not hidden behind host loops; a build without ArrayFire refuses.
#ifdef CYXWIZ_HAS_ARRAYFIRE
[[noreturn]] void ThrowAttentionDeviceError(const char* operation, const af::exception& error) {
    throw std::runtime_error(std::string(operation) + " failed on the ArrayFire device: " + error.what());
}
#else
[[noreturn]] void ThrowAttentionNeedsArrayFire() {
    throw std::runtime_error("Multi-head attention runs on ArrayFire, and this build has no ArrayFire");
}
#endif
#endif

} // namespace

MultiHeadAttentionLayer::MultiHeadAttentionLayer(int embed_dim, int num_heads,
                                                   float dropout, bool use_bias)
    : MultiHeadAttentionLayer(embed_dim, num_heads, dropout, use_bias, num_heads) {}

MultiHeadAttentionLayer::MultiHeadAttentionLayer(int embed_dim, int num_heads, float dropout,
                                                 bool use_bias, int num_kv_heads)
    : embed_dim_(embed_dim), num_heads_(num_heads), dropout_(dropout), use_bias_(use_bias),
      num_kv_heads_(num_kv_heads) {

    if (embed_dim_ <= 0 || num_heads_ <= 0 || !std::isfinite(dropout_) || dropout_ < 0.0f || dropout_ >= 1.0f ||
        embed_dim_ % num_heads_ != 0) {
        throw std::invalid_argument("MultiHeadAttention requires positive divisible dims and dropout in [0, 1)");
    }
    if (num_kv_heads_ <= 0 || num_kv_heads_ > num_heads_ || num_heads_ % num_kv_heads_ != 0) {
        throw std::invalid_argument("MultiHeadAttention num_kv_heads must be positive and divide num_heads");
    }

    head_dim_ = embed_dim / num_heads;
    scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    InitializeWeights();
}

void MultiHeadAttentionLayer::InitializeWeights() {
    // Key/value projections are [num_kv_heads * head_dim, embed_dim] (GQA).
    const int kv_dim = num_kv_heads_ * head_dim_;
    const std::vector<size_t> kv_weight_shape{static_cast<size_t>(kv_dim), static_cast<size_t>(embed_dim_)};
    const std::vector<size_t> kv_bias_shape{static_cast<size_t>(kv_dim)};
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        // Xavier initialization for projection weights
        float limit = std::sqrt(6.0f / (embed_dim_ + embed_dim_));

        af::array w_q = af::randu(af::dim4(embed_dim_, embed_dim_)) * 2.0f * limit - limit;
        const float kv_limit = std::sqrt(6.0f / (embed_dim_ + kv_dim));
        af::array w_k = af::randu(af::dim4(kv_dim, embed_dim_)) * 2.0f * kv_limit - kv_limit;
        af::array w_v = af::randu(af::dim4(kv_dim, embed_dim_)) * 2.0f * kv_limit - kv_limit;
        af::array w_o = af::randu(af::dim4(embed_dim_, embed_dim_)) * 2.0f * limit - limit;
        w_q.eval();
        w_k.eval();
        w_v.eval();
        w_o.eval();

        W_q_ = AfToTensor(w_q);
        W_k_ = AfToTensor(w_k);
        W_v_ = AfToTensor(w_v);
        W_o_ = AfToTensor(w_o);

        if (use_bias_) {
            const std::vector<size_t> bias_shape{
                static_cast<size_t>(embed_dim_)};
            b_q_ = Tensor::Zeros(bias_shape, DataType::Float32);
            b_k_ = Tensor::Zeros(kv_bias_shape, DataType::Float32);
            b_v_ = Tensor::Zeros(kv_bias_shape, DataType::Float32);
            b_o_ = Tensor::Zeros(bias_shape, DataType::Float32);
        }

        // Initialize gradient tensors
        grad_W_q_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)});
        grad_W_k_ = Tensor(kv_weight_shape);
        grad_W_v_ = Tensor(kv_weight_shape);
        grad_W_o_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)});

        if (use_bias_) {
            grad_b_q_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
            grad_b_k_ = Tensor(kv_bias_shape);
            grad_b_v_ = Tensor(kv_bias_shape);
            grad_b_o_ = Tensor(std::vector<size_t>{static_cast<size_t>(embed_dim_)});
        }

        return;
    } catch (const af::exception& e) {
        ThrowAttentionDeviceError("MultiHeadAttentionLayer::InitializeWeights", e);
    }
#else
    (void)kv_weight_shape;
    (void)kv_bias_shape;
    ThrowAttentionNeedsArrayFire();
#endif
}

Tensor MultiHeadAttentionLayer::Forward(const Tensor& input) {
    // Self-attention: Q = K = V = input
    return Forward(input, input, input, nullptr);
}

void MultiHeadAttentionLayer::SetRotaryEmbedding(bool enabled, float base, float fraction) {
    if (enabled && (!std::isfinite(base) || base <= 1.0f)) {
        throw std::invalid_argument("Rotary position embedding base must be finite and greater than 1");
    }
    if (enabled && (!std::isfinite(fraction) || fraction <= 0.0f || fraction > 1.0f)) {
        throw std::invalid_argument("Rotary position embedding fraction must be in (0, 1]");
    }
    const int dims = enabled ? static_cast<int>(std::floor(fraction * head_dim_ / 2.0f + 1e-6f)) * 2 : 0;
    if (enabled && fraction == 1.0f && head_dim_ % 2 != 0) {
        throw std::invalid_argument("Rotary position embedding requires an even head dimension (d_model / num_heads)");
    }
    if (enabled && dims < 2) {
        throw std::invalid_argument("Rotary position embedding fraction leaves fewer than two rotary features per head");
    }
    rope_ = enabled;
    rope_base_ = base;
    rope_dims_ = dims;
}

void MultiHeadAttentionLayer::BeginIncremental(size_t position_offset, int sliding_window) {
    if (position_offset != 0 && position_offset != cache_positions_) {
        throw std::runtime_error("MultiHeadAttention incremental decoding must continue at position " +
                                 std::to_string(cache_positions_) + " (got " + std::to_string(position_offset) + ")");
    }
    incremental_ = true;
    incremental_offset_ = position_offset;
    incremental_window_ = sliding_window;
}

void MultiHeadAttentionLayer::ResetKvCache() {
    cache_k_ = Tensor();
    cache_v_ = Tensor();
    cache_positions_ = 0;
}

void MultiHeadAttentionLayer::DeclareStandardMask(bool causal, int sliding_window) {
    fused_declared_ = true;
    fused_causal_ = causal;
    fused_window_ = causal ? std::max(0, sliding_window) : 0;
}

void MultiHeadAttentionLayer::SetAlibi(bool enabled) {
    alibi_ = enabled;
}

std::vector<float> MultiHeadAttentionLayer::AlibiSlopes() const {
    // Press et al.: for n a power of two, slopes 2^(-8/n * i), i = 1..n; else
    // those of the closest lower power of two plus every other slope of 2x it.
    const auto power_of_two = [](int n) {
        std::vector<float> slopes;
        const double start = std::pow(2.0, -8.0 / n);
        for (int i = 1; i <= n; ++i) slopes.push_back(static_cast<float>(std::pow(start, i)));
        return slopes;
    };
    int closest = 1;
    while (closest * 2 <= num_heads_) closest *= 2;
    std::vector<float> slopes = power_of_two(closest);
    if (closest != num_heads_) {
        const std::vector<float> extra = power_of_two(2 * closest);
        for (int i = 0; static_cast<int>(slopes.size()) < num_heads_; i += 2) slopes.push_back(extra[i]);
    }
    return slopes;
}

void MultiHeadAttentionLayer::SetLogitSoftcap(float cap) {
    if (!std::isfinite(cap) || cap < 0.0f) {
        throw std::invalid_argument("Attention logit soft-cap must be finite and non-negative (0 disables)");
    }
    logit_softcap_ = cap;
}

void MultiHeadAttentionLayer::SetQKNorm(bool enabled, float eps) {
    if (enabled && (!std::isfinite(eps) || eps <= 0.0f)) {
        throw std::invalid_argument("QK normalization epsilon must be finite and positive");
    }
    qk_norm_ = enabled;
    qk_norm_eps_ = eps;
    const std::vector<size_t> shape{static_cast<size_t>(head_dim_)};
    q_norm_gamma_ = enabled ? Tensor::Ones(shape) : Tensor();
    k_norm_gamma_ = enabled ? Tensor::Ones(shape) : Tensor();
    grad_q_norm_gamma_ = enabled ? Tensor::Zeros(shape) : Tensor();
    grad_k_norm_gamma_ = enabled ? Tensor::Zeros(shape) : Tensor();
}

Tensor MultiHeadAttentionLayer::Forward(const Tensor& query, const Tensor& key,
                                         const Tensor& value, const Tensor* attn_mask) {
    if (incremental_) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
        return ForwardIncrementalArrayFire(query);
#else
        throw std::runtime_error("MultiHeadAttention incremental decoding needs the ArrayFire path");
#endif
    }
    const auto& q_shape = query.Shape();
    const auto& k_shape = key.Shape();
    const auto& v_shape = value.Shape();
    if (query.GetDataType() != DataType::Float32 || key.GetDataType() != DataType::Float32 ||
        value.GetDataType() != DataType::Float32) {
        throw std::runtime_error("MultiHeadAttention forward requires Float32 inputs");
    }
    if (q_shape.size() != 3 || k_shape.size() != 3 || v_shape.size() != 3) {
        throw std::invalid_argument("MultiHeadAttention expects [batch, seq_len, embed_dim] tensors");
    }
    if (q_shape[0] == 0 || q_shape[1] == 0 || k_shape[1] == 0 ||
        q_shape[0] != k_shape[0] || k_shape[0] != v_shape[0] ||
        k_shape[1] != v_shape[1] ||
        q_shape[2] != static_cast<size_t>(embed_dim_) ||
        k_shape[2] != static_cast<size_t>(embed_dim_) ||
        v_shape[2] != static_cast<size_t>(embed_dim_)) {
        throw std::runtime_error("MultiHeadAttention forward shape mismatch");
    }
    if (attn_mask != nullptr &&
        (attn_mask->GetDataType() != DataType::Float32 ||
         attn_mask->Shape() != std::vector<size_t>{q_shape[1], k_shape[1]})) {
        throw std::runtime_error("MultiHeadAttention mask must be Float32 [seq_len_q, seq_len_kv]");
    }

    const std::vector<size_t> weight_shape{static_cast<size_t>(embed_dim_), static_cast<size_t>(embed_dim_)};
    const std::vector<size_t> bias_shape{static_cast<size_t>(embed_dim_)};
    const size_t kv_dim = static_cast<size_t>(num_kv_heads_ * head_dim_);
    const std::vector<size_t> kv_weight_shape{kv_dim, static_cast<size_t>(embed_dim_)};
    const std::vector<size_t> kv_bias_shape{kv_dim};
    if (W_q_.GetDataType() != DataType::Float32 || W_k_.GetDataType() != DataType::Float32 ||
        W_v_.GetDataType() != DataType::Float32 || W_o_.GetDataType() != DataType::Float32 ||
        W_q_.Shape() != weight_shape || W_k_.Shape() != kv_weight_shape ||
        W_v_.Shape() != kv_weight_shape || W_o_.Shape() != weight_shape) {
        throw std::runtime_error("MultiHeadAttention forward projection weight mismatch");
    }
    if (use_bias_ &&
        (b_q_.GetDataType() != DataType::Float32 || b_k_.GetDataType() != DataType::Float32 ||
         b_v_.GetDataType() != DataType::Float32 || b_o_.GetDataType() != DataType::Float32 ||
         b_q_.Shape() != bias_shape || b_k_.Shape() != kv_bias_shape ||
         b_v_.Shape() != kv_bias_shape || b_o_.Shape() != bias_shape)) {
        throw std::runtime_error("MultiHeadAttention forward bias mismatch");
    }
    if (qk_norm_) {
        const std::vector<size_t> gamma_shape{static_cast<size_t>(head_dim_)};
        if (q_norm_gamma_.GetDataType() != DataType::Float32 || q_norm_gamma_.Shape() != gamma_shape ||
            k_norm_gamma_.GetDataType() != DataType::Float32 || k_norm_gamma_.Shape() != gamma_shape) {
            throw std::runtime_error("MultiHeadAttention forward QK-norm gamma mismatch");
        }
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        return ForwardArrayFire(query, key, value, attn_mask);
    } catch (const af::exception& e) {
        ThrowAttentionDeviceError("MultiHeadAttentionLayer::Forward", e);
    }
#else
    ThrowAttentionNeedsArrayFire();
#endif
}

Tensor MultiHeadAttentionLayer::Backward(const Tensor& grad_output) {
    const auto& shape = grad_output.Shape();
    if (grad_output.GetDataType() != DataType::Float32 || shape.size() != 3 ||
        cached_query_.Shape().size() != 3 || cached_key_.Shape().size() != 3 ||
        cached_value_.Shape().size() != 3) {
        throw std::runtime_error("MultiHeadAttention backward requires cached 3D Float32 tensors");
    }

    const size_t batch_size = shape[0];
    const size_t seq_len_q = shape[1];
    const size_t seq_len_kv = cached_key_.Shape()[1];
    const size_t embed_dim = static_cast<size_t>(embed_dim_);
    const size_t num_heads = static_cast<size_t>(num_heads_);
    const size_t head_dim = static_cast<size_t>(head_dim_);
    const std::vector<size_t> q_shape{batch_size, seq_len_q, embed_dim};
    const std::vector<size_t> kv_shape{batch_size, seq_len_kv, embed_dim};
    const std::vector<size_t> weight_shape{embed_dim, embed_dim};
    // Projected keys/values and their weights are num_kv_heads * head_dim wide (GQA).
    const size_t kv_dim = static_cast<size_t>(num_kv_heads_) * head_dim;
    const std::vector<size_t> projected_kv_shape{batch_size, seq_len_kv, kv_dim};
    const std::vector<size_t> kv_weight_shape{kv_dim, embed_dim};
    if (shape != q_shape || cached_query_.Shape() != q_shape ||
        cached_key_.Shape() != kv_shape || cached_value_.Shape() != kv_shape ||
        cached_Q_.Shape() != q_shape || cached_K_.Shape() != projected_kv_shape ||
        cached_V_.Shape() != projected_kv_shape || cached_context_.Shape() != q_shape ||
        (!cached_fused_ &&
         cached_attn_weights_.Shape() != std::vector<size_t>{seq_len_q, seq_len_kv, batch_size, num_heads}) ||
        W_q_.Shape() != weight_shape || W_k_.Shape() != kv_weight_shape ||
        W_v_.Shape() != kv_weight_shape || W_o_.Shape() != weight_shape) {
        throw std::runtime_error("MultiHeadAttention backward cache/parameter shape mismatch");
    }
    if (cached_attention_dropout_ && !cached_fused_ &&
        (dropout_mask_.GetDataType() != DataType::Float32 ||
         dropout_mask_.Shape() != std::vector<size_t>{seq_len_q, seq_len_kv, batch_size, num_heads})) {
        throw std::runtime_error("MultiHeadAttention backward dropout mask shape mismatch");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        return BackwardArrayFire(grad_output);
    } catch (const af::exception& e) {
        ThrowAttentionDeviceError("MultiHeadAttentionLayer::Backward", e);
    }
#else
    ThrowAttentionNeedsArrayFire();
#endif
}

std::map<std::string, Tensor> MultiHeadAttentionLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    params["W_q"] = W_q_;
    params["W_k"] = W_k_;
    params["W_v"] = W_v_;
    params["W_o"] = W_o_;
    params["grad_W_q"] = grad_W_q_;
    params["grad_W_k"] = grad_W_k_;
    params["grad_W_v"] = grad_W_v_;
    params["grad_W_o"] = grad_W_o_;

    if (use_bias_) {
        params["b_q"] = b_q_;
        params["b_k"] = b_k_;
        params["b_v"] = b_v_;
        params["b_o"] = b_o_;
        params["grad_b_q"] = grad_b_q_;
        params["grad_b_k"] = grad_b_k_;
        params["grad_b_v"] = grad_b_v_;
        params["grad_b_o"] = grad_b_o_;
    }

    if (qk_norm_) {
        params["q_norm_gamma"] = q_norm_gamma_;
        params["k_norm_gamma"] = k_norm_gamma_;
        params["grad_q_norm_gamma"] = grad_q_norm_gamma_;
        params["grad_k_norm_gamma"] = grad_k_norm_gamma_;
    }

    return params;
}

void MultiHeadAttentionLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    if (params.count("W_q")) W_q_ = params.at("W_q");
    if (params.count("W_k")) W_k_ = params.at("W_k");
    if (params.count("W_v")) W_v_ = params.at("W_v");
    if (params.count("W_o")) W_o_ = params.at("W_o");

    if (use_bias_) {
        if (params.count("b_q")) b_q_ = params.at("b_q");
        if (params.count("b_k")) b_k_ = params.at("b_k");
        if (params.count("b_v")) b_v_ = params.at("b_v");
        if (params.count("b_o")) b_o_ = params.at("b_o");
    }

    if (qk_norm_) {
        if (params.count("q_norm_gamma")) q_norm_gamma_ = params.at("q_norm_gamma");
        if (params.count("k_norm_gamma")) k_norm_gamma_ = params.at("k_norm_gamma");
    }
}

} // namespace cyxwiz
