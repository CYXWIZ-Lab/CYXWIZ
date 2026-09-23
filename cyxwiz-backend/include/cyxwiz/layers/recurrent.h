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
    // Owner ruling 2026-09-23 (track68): Forward starts from a ZERO state
    // unless SetHiddenState/SetCellState was called since the last
    // Forward (one-shot). Matches the native provider path and PyTorch's
    // default; h_n_/c_n_ hold the FINAL states after Forward.
    bool initial_state_pending_ = false;

    std::vector<Tensor> cached_inputs_;
    std::vector<Tensor> cached_gates_;
    std::vector<Tensor> cached_cell_states_;
    std::vector<Tensor> cached_hidden_states_;

    // tofix68 P2: when the native neural provider executed Forward, the
    // CPU/AF caches are empty and Backward must use the provider's
    // self-contained recompute+BPTT op instead.
    bool provider_forward_used_ = false;
    bool provider_disabled_after_failure_ = false;

    void InitializeWeights();
};

// Vanilla (Elman) RNN: h_t = act(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh),
// act in {tanh, relu}. Native CPU reference implementation (tofix68
// phase 3); unidirectional, batch-first [batch, seq, features], returns
// the full hidden sequence. Parameter naming follows the LSTM/GRU
// convention: layer<N>_W_ih / W_hh / b_ih / b_hh (+ layer<N>_grad_*).
class CYXWIZ_API RNNLayer : public Layer {
public:
    RNNLayer(int input_size, int hidden_size, int num_layers = 1,
             bool batch_first = true, bool bidirectional = false,
             const std::string& nonlinearity = "tanh");

    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;
    std::map<std::string, Tensor> GetParameters() override;
    void SetParameters(const std::map<std::string, Tensor>& params) override;
    std::string GetName() const override { return "RNN"; }

    void ResetState();
    Tensor GetHiddenState() const { return h_n_; }

    int GetInputSize() const { return input_size_; }
    int GetHiddenSize() const { return hidden_size_; }
    int GetNumLayers() const { return num_layers_; }
    bool UsesTanh() const { return use_tanh_; }

private:
    int input_size_;
    int hidden_size_;
    int num_layers_;
    bool batch_first_;
    bool use_tanh_;

    std::vector<Tensor> W_ih_;
    std::vector<Tensor> W_hh_;
    std::vector<Tensor> b_ih_;
    std::vector<Tensor> b_hh_;
    std::vector<Tensor> grad_W_ih_;
    std::vector<Tensor> grad_W_hh_;
    std::vector<Tensor> grad_b_ih_;
    std::vector<Tensor> grad_b_hh_;

    Tensor h_n_;

    std::vector<Tensor> cached_inputs_;
    std::vector<Tensor> cached_hidden_states_;

    // tofix68 (provider 0.7.0): when the native neural provider executed
    // Forward, the CPU caches are empty and Backward uses the provider's
    // self-contained recompute+BPTT op (mirror of LSTMLayer/GRULayer).
    bool provider_forward_used_ = false;
    bool provider_disabled_after_failure_ = false;
    Tensor provider_input_cache_;

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
    // Owner ruling 2026-09-23 (track68): stateless per Forward unless
    // SetHiddenState was called since the last Forward (one-shot).
    bool initial_state_pending_ = false;

    std::vector<Tensor> cached_inputs_;
    std::vector<Tensor> cached_gates_;
    std::vector<Tensor> cached_hidden_states_;

    // tofix68 P3: when the native neural provider executed Forward, the
    // CPU/AF caches are empty and Backward must use the provider's
    // self-contained recompute+BPTT op instead (mirror of LSTMLayer).
    bool provider_forward_used_ = false;
    bool provider_disabled_after_failure_ = false;

    void InitializeWeights();
};

} // namespace cyxwiz
