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
// node). Rope: rotary position embedding inside self-attention.
enum class TransformerPositionEncoding { External, Rope };
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

    bool IsClassic() const {
        return norm_type == TransformerNormType::LayerNorm && norm_eps == 1e-5f &&
               ffn_type == TransformerFeedForwardType::Mlp &&
               ffn_activation == ActivationType::ReLU && ffn_bias &&
               position_encoding == TransformerPositionEncoding::External;
    }
};

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

    static Tensor GenerateCausalMask(int size);
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
};

} // namespace cyxwiz
