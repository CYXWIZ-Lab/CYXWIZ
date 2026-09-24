#pragma once

#include "cyxwiz/api_export.h"
#include "cyxwiz/layers/layer_base.h"
#include "cyxwiz/tensor.h"

#include <map>
#include <string>

namespace cyxwiz {

class CYXWIZ_API MultiHeadAttentionLayer : public Layer {
public:
    MultiHeadAttentionLayer(int embed_dim, int num_heads,
                            float dropout = 0.0f, bool use_bias = true);

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
    void SetRotaryEmbedding(bool enabled, float base = 10000.0f);
    bool UsesRotaryEmbedding() const { return rope_; }
    float GetRotaryBase() const { return rope_base_; }

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
    bool rope_ = false;
    float rope_base_ = 10000.0f;

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
#ifdef CYXWIZ_HAS_ARRAYFIRE
    Tensor ForwardArrayFire(const Tensor& query, const Tensor& key,
                            const Tensor& value, const Tensor* attn_mask);
    Tensor BackwardArrayFire(const Tensor& grad_output);
#endif
};

} // namespace cyxwiz
