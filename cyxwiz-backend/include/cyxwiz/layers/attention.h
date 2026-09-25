#pragma once

#include "cyxwiz/api_export.h"
#include "cyxwiz/layers/layer_base.h"
#include "cyxwiz/tensor.h"

#include <map>
#include <string>
#include <vector>

namespace cyxwiz {

class CYXWIZ_API MultiHeadAttentionLayer : public Layer {
public:
    MultiHeadAttentionLayer(int embed_dim, int num_heads,
                            float dropout = 0.0f, bool use_bias = true);
    // Grouped-query attention (GQA; multi-query when num_kv_heads == 1):
    // W_k/W_v are [num_kv_heads * head_dim, embed_dim] and each key/value head
    // serves num_heads / num_kv_heads consecutive query heads. num_kv_heads
    // must divide num_heads. ArrayFire path only when num_kv_heads < num_heads.
    MultiHeadAttentionLayer(int embed_dim, int num_heads, float dropout,
                            bool use_bias, int num_kv_heads);

    Tensor Forward(const Tensor& input) override;
    // Float32 additive [query_length,key_length] mask. A row entirely -infinity
    // contributes zero attention/context and zero Q/K/V gradient. Output bias
    // still applies. Finite mask values remain additive score offsets.
    Tensor Forward(const Tensor& query, const Tensor& key, const Tensor& value,
                   const Tensor* attn_mask = nullptr);

    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "MultiHeadAttention"; }

    Tensor GetAttentionWeights() const { return cached_attn_weights_; }
    Tensor GetLastKeyGradient() const { return cached_grad_key_; }
    Tensor GetLastValueGradient() const { return cached_grad_value_; }

    // Rotary position embedding (RoFormer, arXiv:2104.09864; LLaMA/GPT-NeoX
    // half-split convention): Q and K of every head are rotated by
    // position * base^(-2i/head_dim) before the scores. Positions start at 0
    // for both query and key. Requires an even head_dim and the ArrayFire path.
    // fraction < 1 rotates only the first round_down_even(fraction * head_dim)
    // features of each head (partial RoPE, GPT-NeoX rotary_pct); the
    // frequencies then use that rotary width.
    void SetRotaryEmbedding(bool enabled, float base = 10000.0f, float fraction = 1.0f);
    bool UsesRotaryEmbedding() const { return rope_; }
    float GetRotaryBase() const { return rope_base_; }
    int RotaryDims() const { return rope_dims_; }

    // ALiBi (Press et al. 2022): adds slope_h * (key_pos - query_pos) to the
    // scores; slopes are the paper's geometric sequence (also for head counts
    // that are not a power of two). Intended for causal self-attention.
    void SetAlibi(bool enabled);
    bool UsesAlibi() const { return alibi_; }
    std::vector<float> AlibiSlopes() const;

    // Logit soft-capping (Gemma 2): scores = cap * tanh(scores / cap) before
    // masks and ALiBi. 0 disables.
    void SetLogitSoftcap(float cap);
    float GetLogitSoftcap() const { return logit_softcap_; }
    int GetNumKvHeads() const { return num_kv_heads_; }

    // KV-cached causal decoding (inference). Between Begin/EndIncremental,
    // Forward(q, k, v, mask) treats q as self-attention input for positions
    // [position_offset, +seq_len), ignores `mask`, appends this call's keys and
    // values to the cache and attends causally over the whole cache
    // (sliding_window > 0 limits it). position_offset 0 starts a new cache.
    void BeginIncremental(size_t position_offset, int sliding_window = 0);
    void EndIncremental() { incremental_ = false; }
    void ResetKvCache();
    size_t CachedPositions() const { return cache_positions_; }

    // QK normalization (OLMo 2 / Gemma 3 / Qwen3): each head's query and key
    // vectors are RMS-normalized over head_dim and scaled by learned gammas
    // (q_norm_gamma, k_norm_gamma: [head_dim], shared across heads, init 1)
    // after projection and before RoPE. ArrayFire path only.
    void SetQKNorm(bool enabled, float eps = 1e-5f);
    bool UsesQKNorm() const { return qk_norm_; }
    float GetQKNormEps() const { return qk_norm_eps_; }
    bool UsesBias() const { return use_bias_; }

    int GetEmbedDim() const { return embed_dim_; }
    int GetNumHeads() const { return num_heads_; }
    int GetHeadDim() const { return head_dim_; }

private:
    int embed_dim_;
    int num_heads_;
    int head_dim_;
    float dropout_;
    bool use_bias_;
    float scale_;
    int num_kv_heads_ = 0;
    bool rope_ = false;
    float rope_base_ = 10000.0f;
    int rope_dims_ = 0;
    bool alibi_ = false;
    float logit_softcap_ = 0.0f;
    bool qk_norm_ = false;
    float qk_norm_eps_ = 1e-5f;
    Tensor q_norm_gamma_, k_norm_gamma_;
    Tensor grad_q_norm_gamma_, grad_k_norm_gamma_;

    Tensor W_q_, W_k_, W_v_, W_o_;
    Tensor b_q_, b_k_, b_v_, b_o_;
    Tensor grad_W_q_, grad_W_k_, grad_W_v_, grad_W_o_;
    Tensor grad_b_q_, grad_b_k_, grad_b_v_, grad_b_o_;
    Tensor cached_query_, cached_key_, cached_value_;
    Tensor cached_Q_, cached_K_, cached_V_;
    Tensor cached_attn_weights_;
    Tensor cached_context_;
    Tensor dropout_mask_;
    Tensor cached_grad_key_;
    Tensor cached_grad_value_;
    bool cached_self_attention_ = false;
    bool cached_attention_dropout_ = false;

    void InitializeWeights();
    // Name of the first enabled option the native CPU path cannot run, or null.
    const char* ArrayFireOnlyOption() const {
        if (rope_) return "Rotary position embedding";
        if (qk_norm_) return "QK normalization";
        if (alibi_) return "ALiBi";
        if (logit_softcap_ > 0.0f) return "Attention logit soft-capping";
        if (num_kv_heads_ != num_heads_) return "Grouped-query attention";
        return nullptr;
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    Tensor ForwardArrayFire(const Tensor& query, const Tensor& key,
                            const Tensor& value, const Tensor* attn_mask);
    Tensor BackwardArrayFire(const Tensor& grad_output);
    Tensor ForwardIncrementalArrayFire(const Tensor& input);
#endif
    bool incremental_ = false;
    size_t incremental_offset_ = 0;
    int incremental_window_ = 0;
    size_t cache_positions_ = 0;
    Tensor cache_k_, cache_v_;  // [positions, head_dim, batch, num_kv_heads], after QK-norm and RoPE
};

} // namespace cyxwiz
