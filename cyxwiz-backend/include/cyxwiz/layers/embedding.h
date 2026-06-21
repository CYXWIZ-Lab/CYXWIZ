#pragma once

#include "cyxwiz/api_export.h"
#include "cyxwiz/layers/layer_base.h"
#include "cyxwiz/tensor.h"

#include <map>
#include <string>

namespace cyxwiz {

class CYXWIZ_API EmbeddingLayer : public Layer {
public:
    EmbeddingLayer(int num_embeddings, int embedding_dim,
                   int padding_idx = -1, float max_norm = 0.0f);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;

    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "Embedding"; }

    std::map<std::string, Tensor> GetGradients();
    Tensor GetEmbedding(int index) const;
    void SetEmbedding(int index, const Tensor& embedding);
    void LoadPretrainedWeights(const Tensor& weights, bool freeze = false);

    int GetNumEmbeddings() const { return num_embeddings_; }
    int GetEmbeddingDim() const { return embedding_dim_; }
    int GetPaddingIdx() const { return padding_idx_; }
    bool IsFrozen() const { return frozen_; }
    void SetFrozen(bool frozen) { frozen_ = frozen; }

private:
    int num_embeddings_;
    int embedding_dim_;
    int padding_idx_;
    float max_norm_;
    bool frozen_ = false;

    Tensor weight_;
    Tensor grad_weight_;
    Tensor cached_indices_;

    void InitializeWeights();
    void NormalizeEmbeddings();
};

} // namespace cyxwiz
