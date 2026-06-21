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
    Tensor Forward(const Tensor& query, const Tensor& key, const Tensor& value,
                   const Tensor* attn_mask = nullptr);

    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "MultiHeadAttention"; }

    Tensor GetAttentionWeights() const { return cached_attn_weights_; }
    Tensor GetLastKeyGradient() const { return cached_grad_key_; }
    Tensor GetLastValueGradient() const { return cached_grad_value_; }

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
};

} // namespace cyxwiz
