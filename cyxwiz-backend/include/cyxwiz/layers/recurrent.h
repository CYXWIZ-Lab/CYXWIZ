#pragma once

#include "cyxwiz/api_export.h"
#include "cyxwiz/layers/layer_base.h"
#include "cyxwiz/tensor.h"

#include <map>
#include <string>
#include <vector>

namespace cyxwiz {

class CYXWIZ_API LSTMLayer : public Layer {
public:
    LSTMLayer(int input_size, int hidden_size, int num_layers = 1,
              bool batch_first = true, bool bidirectional = false,
              float dropout = 0.0f);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;

    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "LSTM"; }

    void ResetState();
    void SetHiddenState(const Tensor& h0);
    void SetCellState(const Tensor& c0);
    Tensor GetHiddenState() const { return h_n_; }
    Tensor GetCellState() const { return c_n_; }

    int GetInputSize() const { return input_size_; }
    int GetHiddenSize() const { return hidden_size_; }
    int GetNumLayers() const { return num_layers_; }
    bool IsBatchFirst() const { return batch_first_; }
    bool IsBidirectional() const { return bidirectional_; }
    int GetNumDirections() const { return bidirectional_ ? 2 : 1; }

private:
    int input_size_;
    int hidden_size_;
    int num_layers_;
    bool batch_first_;
    bool bidirectional_;
    float dropout_;

    std::vector<Tensor> W_ih_;
    std::vector<Tensor> W_hh_;
    std::vector<Tensor> b_ih_;
    std::vector<Tensor> b_hh_;
    std::vector<Tensor> W_ih_reverse_;
    std::vector<Tensor> W_hh_reverse_;
    std::vector<Tensor> b_ih_reverse_;
    std::vector<Tensor> b_hh_reverse_;
    std::vector<Tensor> grad_W_ih_;
    std::vector<Tensor> grad_W_hh_;
    std::vector<Tensor> grad_b_ih_;
    std::vector<Tensor> grad_b_hh_;
    std::vector<Tensor> grad_W_ih_reverse_;
    std::vector<Tensor> grad_W_hh_reverse_;
    std::vector<Tensor> grad_b_ih_reverse_;
    std::vector<Tensor> grad_b_hh_reverse_;

    Tensor h_n_;
    Tensor c_n_;

    std::vector<Tensor> cached_inputs_;
    std::vector<Tensor> cached_gates_;
    std::vector<Tensor> cached_cell_states_;
    std::vector<Tensor> cached_hidden_states_;

    void InitializeWeights();
};

class CYXWIZ_API GRULayer : public Layer {
public:
    GRULayer(int input_size, int hidden_size, int num_layers = 1,
             bool batch_first = true, bool bidirectional = false,
             float dropout = 0.0f);

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "GRU"; }

    void ResetState();
    void SetHiddenState(const Tensor& h0);
    Tensor GetHiddenState() const { return h_n_; }

    int GetInputSize() const { return input_size_; }
    int GetHiddenSize() const { return hidden_size_; }
    int GetNumLayers() const { return num_layers_; }
    bool IsBatchFirst() const { return batch_first_; }
    bool IsBidirectional() const { return bidirectional_; }

private:
    int input_size_;
    int hidden_size_;
    int num_layers_;
    bool batch_first_;
    bool bidirectional_;
    float dropout_;

    std::vector<Tensor> W_ih_;
    std::vector<Tensor> W_hh_;
    std::vector<Tensor> b_ih_;
    std::vector<Tensor> b_hh_;
    std::vector<Tensor> W_ih_reverse_;
    std::vector<Tensor> W_hh_reverse_;
    std::vector<Tensor> b_ih_reverse_;
    std::vector<Tensor> b_hh_reverse_;
    std::vector<Tensor> grad_W_ih_;
    std::vector<Tensor> grad_W_hh_;
    std::vector<Tensor> grad_b_ih_;
    std::vector<Tensor> grad_b_hh_;

    Tensor h_n_;

    std::vector<Tensor> cached_inputs_;
    std::vector<Tensor> cached_gates_;
    std::vector<Tensor> cached_hidden_states_;

    void InitializeWeights();
};

} // namespace cyxwiz
