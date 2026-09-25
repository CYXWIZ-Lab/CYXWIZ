#pragma once

#include "cyxwiz/api_export.h"
#include "cyxwiz/activation.h"
#include "cyxwiz/layers/attention.h"
#include "cyxwiz/layers/dense.h"
#include "cyxwiz/layers/dropout.h"
#include "cyxwiz/layers/layer_base.h"
#include "cyxwiz/layers/normalization.h"
#include "cyxwiz/tensor.h"

#include <map>
#include <memory>
#include <string>

namespace cyxwiz {

// Encoder/decoder constructors reject invalid dimensions or dropout before
// weight initialization. Width must be divisible by a positive head count.
class CYXWIZ_API TransformerEncoderLayer : public Layer {
public:
    TransformerEncoderLayer(int d_model, int nhead, int dim_feedforward = 2048,
                            float dropout = 0.1f, bool norm_first = false);
    // Explicit FFN hidden dropout; legacy construction keeps this at zero.
    TransformerEncoderLayer(int d_model, int nhead, int dim_feedforward,
                            float dropout, bool norm_first, float ffn_dropout);

    Tensor Forward(const Tensor& input) override;
    Tensor Forward(const Tensor& input, const Tensor* src_mask);

    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "TransformerEncoderLayer"; }

    void SetTraining(bool training) override;

private:
    int d_model_;
    int nhead_;
    int dim_feedforward_;
    float dropout_;
    bool norm_first_;

    std::unique_ptr<MultiHeadAttentionLayer> self_attn_;
    std::unique_ptr<LayerNormLayer> norm1_;
    std::unique_ptr<LayerNormLayer> norm2_;
    std::unique_ptr<DenseLayer> linear1_;
    std::unique_ptr<DenseLayer> linear2_;
    std::unique_ptr<DropoutLayer> ffn_dropout_;
    std::unique_ptr<DropoutLayer> dropout1_;
    std::unique_ptr<DropoutLayer> dropout2_;

    Tensor cached_attn_output_;
    Tensor cached_ffn_mid_;
    Tensor cached_residual1_;
    Tensor cached_residual2_;
};

// Block choices beyond the classic layout (tofix112). Defaults reproduce the
// original decoder exactly: LayerNorm, Dense -> ReLU -> Dense, biases on.
enum class TransformerNormType { LayerNorm, RMSNorm };
// External: positions come from outside the block (e.g. a PositionalEncoding
// node). Rope: rotary position embedding inside self-attention. Alibi: linear
// distance bias on the self-attention scores (no position input at all).
enum class TransformerPositionEncoding { External, Rope, Alibi };
// Sequential: x + attn, then + ffn. Parallel (GPT-J, PaLM; pre-norm only):
// y = x + attn(norm1(x)) + ffn(norm1(x)); norm2 is unused.
enum class TransformerBlockLayout { Sequential, Parallel };
enum class TransformerFeedForwardType {
    Mlp,    // down(act(up(x)))
    Gated   // down(act(gate(x)) * up(x)): sigmoid=GLU, ReLU=ReGLU, GELU=GEGLU, SiLU=SwiGLU
};

struct CYXWIZ_API TransformerBlockOptions {
    TransformerNormType norm_type = TransformerNormType::LayerNorm;
    float norm_eps = 1e-5f;
    TransformerFeedForwardType ffn_type = TransformerFeedForwardType::Mlp;
    ActivationType ffn_activation = ActivationType::ReLU;
    bool ffn_bias = true;
    TransformerPositionEncoding position_encoding = TransformerPositionEncoding::External;
    float rope_base = 10000.0f;
    // Bias vectors in the attention Q/K/V/output projections (LLaMA: off).
    bool attention_bias = true;
    // Per-head RMSNorm of queries and keys before the scores (OLMo 2, Gemma 3,
    // Qwen3); uses norm_eps; applies to self-attention.
    bool qk_norm = false;
    // Fraction of each head's features RoPE rotates (partial RoPE, GPT-NeoX).
    float rope_fraction = 1.0f;
    TransformerBlockLayout block_layout = TransformerBlockLayout::Sequential;
    // Sandwich norm (Gemma 2/3; pre-norm only): x + post_norm(sublayer(norm(x))).
    bool sandwich_norm = false;
    // Multiplies the initial weights of the residual output projections
    // (attention W_o, FFN down) - GPT-2 uses 1/sqrt(2 * num_blocks).
    float residual_init_scale = 1.0f;
    // Self-attention logit soft-cap (Gemma 2); 0 disables.
    float attn_logit_softcap = 0.0f;
    // Each position attends to at most this many most recent positions
    // including itself (Mistral); 0 = full causal attention.
    int sliding_window = 0;
    // Key/value heads for grouped-query attention; 0 = num_heads (standard).
    int num_kv_heads = 0;

    bool IsClassic() const {
        return norm_type == TransformerNormType::LayerNorm && norm_eps == 1e-5f &&
               ffn_type == TransformerFeedForwardType::Mlp &&
               ffn_activation == ActivationType::ReLU && ffn_bias &&
               position_encoding == TransformerPositionEncoding::External &&
               attention_bias && !qk_norm && rope_fraction == 1.0f &&
               block_layout == TransformerBlockLayout::Sequential && !sandwich_norm &&
               residual_init_scale == 1.0f && attn_logit_softcap == 0.0f &&
               sliding_window == 0 && num_kv_heads == 0;
    }
};

// Feed-forward activation names shared by the Engine, pycyxwiz and exports:
// relu, gelu (tanh approximation), gelu_exact (erf), silu, mish, elu, selu,
// leaky_relu, sigmoid, tanh, hardswish, squared_relu. Returns false for others.
CYXWIZ_API bool TransformerFfnActivationFromName(const std::string& name, ActivationType& out);

class CYXWIZ_API TransformerDecoderLayer : public Layer {
public:
    TransformerDecoderLayer(int d_model, int nhead, int dim_feedforward = 2048,
                            float dropout = 0.1f, bool norm_first = false);
    // Explicit FFN hidden dropout; legacy construction keeps this at zero.
    TransformerDecoderLayer(int d_model, int nhead, int dim_feedforward,
                            float dropout, bool norm_first, float ffn_dropout);
    // Configurable block: normalization type, FFN type/activation and FFN bias.
    TransformerDecoderLayer(int d_model, int nhead, int dim_feedforward,
                            float dropout, bool norm_first, float ffn_dropout,
                            const TransformerBlockOptions& options);
    const TransformerBlockOptions& GetBlockOptions() const { return options_; }

    Tensor Forward(const Tensor& input) override;
    Tensor Forward(const Tensor& tgt, const Tensor& memory,
                   const Tensor* tgt_mask = nullptr,
                   const Tensor* memory_mask = nullptr);

    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "TransformerDecoderLayer"; }

    void SetTraining(bool training) override;

    // KV-cached decoding of positions [position_offset, +seq_len) through the
    // self-attention cache (decoder-only path, inference).
    Tensor ForwardIncremental(const Tensor& input, size_t position_offset);
    void ResetIncrementalState();

    static Tensor GenerateCausalMask(int size);
    // Causal mask that also hides keys `window` or more positions back.
    static Tensor GenerateCausalMask(int size, int window);
    Tensor GetLastMemoryGradient() const;

private:
    int d_model_;
    int nhead_;
    int dim_feedforward_;
    float dropout_;
    bool norm_first_;

    TransformerBlockOptions options_;
    std::unique_ptr<MultiHeadAttentionLayer> self_attn_;
    std::unique_ptr<MultiHeadAttentionLayer> cross_attn_;
    std::unique_ptr<Layer> norm1_;
    std::unique_ptr<Layer> norm2_;
    std::unique_ptr<Layer> norm3_;
    std::unique_ptr<Layer> post_attn_norm_;  // sandwich_norm only
    std::unique_ptr<Layer> post_ffn_norm_;   // sandwich_norm only
    std::unique_ptr<DenseLayer> linear1_;   // up projection
    std::unique_ptr<DenseLayer> linear2_;   // down projection
    std::unique_ptr<DenseLayer> ffn_gate_;  // gate projection (gated FFN only)
    std::unique_ptr<Activation> ffn_activation_;  // null: classic ReLU path
    std::unique_ptr<DropoutLayer> ffn_dropout_;
    std::unique_ptr<DropoutLayer> dropout1_;
    std::unique_ptr<DropoutLayer> dropout2_;
    std::unique_ptr<DropoutLayer> dropout3_;

    Tensor cached_self_attn_output_;
    Tensor cached_cross_attn_output_;
    Tensor cached_ffn_mid_;       // activation input (MLP) / gate pre-activation (gated)
    Tensor cached_ffn_up_;        // gated: up projection output
    Tensor cached_ffn_gate_act_;  // gated: act(gate) output
    Tensor cached_memory_;
    Tensor cached_residual1_;
    Tensor cached_residual2_;
    Tensor cached_residual3_;
    bool cached_has_cross_attention_ = false;

    std::unique_ptr<Layer> MakeNorm() const;
    Tensor FeedForward(const Tensor& flat_input);          // [rows, d_model] -> [rows, d_model]
    Tensor FeedForwardBackward(const Tensor& grad_flat);   // inverse of FeedForward
    Tensor SequenceFeedForward(const Tensor& input);       // [B,S,d] -> [B,S,d] incl. dropout3
    Tensor SequenceFeedForwardBackward(const Tensor& grad);
    bool UsesModernPreNormPath() const {
        return norm_first_ && (options_.block_layout == TransformerBlockLayout::Parallel || options_.sandwich_norm);
    }
    Tensor ForwardModernPreNorm(const Tensor& input, const Tensor& mask);
    Tensor BackwardModernPreNorm(const Tensor& grad_output);
};

} // namespace cyxwiz
