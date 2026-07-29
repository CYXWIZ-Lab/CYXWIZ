#pragma once

#include "cyxwiz/api_export.h"
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

class CYXWIZ_API TransformerEncoderLayer : public Layer {
public:
    TransformerEncoderLayer(int d_model, int nhead, int dim_feedforward = 2048,
                            float dropout = 0.1f, bool norm_first = false);

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
    std::unique_ptr<DropoutLayer> dropout1_;
    std::unique_ptr<DropoutLayer> dropout2_;

    Tensor cached_attn_output_;
    Tensor cached_ffn_mid_;
    Tensor cached_residual1_;
    Tensor cached_residual2_;
};

class CYXWIZ_API TransformerDecoderLayer : public Layer {
public:
    TransformerDecoderLayer(int d_model, int nhead, int dim_feedforward = 2048,
                            float dropout = 0.1f, bool norm_first = false);

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

    std::unique_ptr<MultiHeadAttentionLayer> self_attn_;
    std::unique_ptr<MultiHeadAttentionLayer> cross_attn_;
    std::unique_ptr<LayerNormLayer> norm1_;
    std::unique_ptr<LayerNormLayer> norm2_;
    std::unique_ptr<LayerNormLayer> norm3_;
    std::unique_ptr<DenseLayer> linear1_;
    std::unique_ptr<DenseLayer> linear2_;
    std::unique_ptr<DropoutLayer> dropout1_;
    std::unique_ptr<DropoutLayer> dropout2_;
    std::unique_ptr<DropoutLayer> dropout3_;

    Tensor cached_self_attn_output_;
    Tensor cached_cross_attn_output_;
    Tensor cached_ffn_mid_;
    Tensor cached_memory_;
    Tensor cached_residual1_;
    Tensor cached_residual2_;
    Tensor cached_residual3_;
    bool cached_has_cross_attention_ = false;
};

} // namespace cyxwiz
