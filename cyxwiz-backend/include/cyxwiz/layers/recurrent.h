#pragma once

#include "cyxwiz/api_export.h"
#include "cyxwiz/layers/layer_base.h"
#include "cyxwiz/tensor.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

// LSTM on ArrayFire (the CPU option is ArrayFire's CPU backend). A
// unidirectional layer runs the native neural provider where one serves the
// exact shape, else the ArrayFire recurrence. A bidirectional layer composes,
// per level, a forward and a reverse single-direction single-layer LSTMLayer;
// the reverse child sees the input flipped in time and the outputs join on
// the feature axis, forward first (PyTorch). Parameter keys: layer{L}_W_ih ...
// and, when bidirectional, layer{L}_W_ih_reverse ...; gradients under
// layer{L}_grad_W_ih (+ _reverse). Final states h_n / c_n are
// [layers * directions, batch, hidden]: index L is level L forward, index
// num_layers + L its reverse.
class CYXWIZ_API LSTMLayer : public Layer {
public:
    LSTMLayer(int input_size, int hidden_size, int num_layers = 1,
              bool batch_first = true, bool bidirectional = false,
              float dropout = 0.0f);
    ~LSTMLayer() override;
    LSTMLayer(const LSTMLayer&) = delete;
    LSTMLayer& operator=(const LSTMLayer&) = delete;

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

    // Unidirectional weights and gradients, one entry per layer.
    std::vector<Tensor> W_ih_;
    std::vector<Tensor> W_hh_;
    std::vector<Tensor> b_ih_;
    std::vector<Tensor> b_hh_;
    std::vector<Tensor> grad_W_ih_;
    std::vector<Tensor> grad_W_hh_;
    std::vector<Tensor> grad_b_ih_;
    std::vector<Tensor> grad_b_hh_;

    // Bidirectional: one forward and one reverse child per level.
    std::vector<std::unique_ptr<LSTMLayer>> forward_levels_;
    std::vector<std::unique_ptr<LSTMLayer>> reverse_levels_;

    Tensor h_n_;
    Tensor c_n_;
    // Owner ruling 2026-09-23 (track68): Forward starts from a ZERO state
    // unless SetHiddenState/SetCellState was called since the last
    // Forward (one-shot), like PyTorch; h_n_/c_n_ hold the FINAL states.
    bool initial_state_pending_ = false;

    // Device caches of the ArrayFire forward, per layer, seq-first.
    std::vector<Tensor> cached_inputs_;
    std::vector<Tensor> cached_gates_;
    std::vector<Tensor> cached_cell_states_;
    std::vector<Tensor> cached_hidden_states_;
    // Inter-layer dropout keep masks of the last training Forward, reused
    // by Backward.
    std::vector<Tensor> dropout_masks_;

    // When the native neural provider ran Forward, Backward uses the
    // provider's self-contained recompute+BPTT op.
    bool provider_forward_used_ = false;
    bool provider_disabled_after_failure_ = false;

    void InitializeWeights();
    bool TryProviderForward(const Tensor& input, Tensor& output);
    bool TryProviderBackward(const Tensor& grad_output, Tensor& grad_input);
    Tensor ForwardArrayFire(const Tensor& input, bool use_initial_state);
    Tensor BackwardArrayFire(const Tensor& grad_output);
    Tensor ForwardBidirectional(const Tensor& input);
};

// Vanilla (Elman) RNN on ArrayFire: h_t = act(W_ih x_t + b_ih + W_hh h_{t-1}
// + b_hh), act in {tanh, relu}. Batch-first [batch, seq, features], returns
// the full hidden sequence. Same routing and bidirectional composition as
// LSTMLayer (native provider for a unidirectional layer where one serves the
// exact shape, else the ArrayFire recurrence). Keys layer{L}_W_ih ... (+
// _reverse), gradients layer{L}_grad_W_ih ...; h_n is
// [layers * directions, batch, hidden], index 2L level L forward, 2L + 1 its
// reverse (unidirectional: index L).
class CYXWIZ_API RNNLayer : public Layer {
public:
    RNNLayer(int input_size, int hidden_size, int num_layers = 1,
             bool batch_first = true, bool bidirectional = false,
             const std::string& nonlinearity = "tanh");
    ~RNNLayer() override;
    RNNLayer(const RNNLayer&) = delete;
    RNNLayer& operator=(const RNNLayer&) = delete;

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
    bool IsBidirectional() const { return bidirectional_; }
    bool UsesTanh() const { return use_tanh_; }

private:
    int input_size_;
    int hidden_size_;
    int num_layers_;
    bool bidirectional_;
    bool use_tanh_;

    std::vector<Tensor> W_ih_;
    std::vector<Tensor> W_hh_;
    std::vector<Tensor> b_ih_;
    std::vector<Tensor> b_hh_;
    std::vector<Tensor> grad_W_ih_;
    std::vector<Tensor> grad_W_hh_;
    std::vector<Tensor> grad_b_ih_;
    std::vector<Tensor> grad_b_hh_;

    // Bidirectional: one forward and one reverse child per level.
    std::vector<std::unique_ptr<RNNLayer>> forward_levels_;
    std::vector<std::unique_ptr<RNNLayer>> reverse_levels_;
    std::vector<Tensor> dropout_masks_;  // always empty: RNN has no dropout

    Tensor h_n_;
    Tensor cached_input_;

    // Device caches of the ArrayFire forward, per layer, seq-first.
    std::vector<Tensor> cached_inputs_;
    std::vector<Tensor> cached_hidden_states_;

    // When the native neural provider ran Forward, Backward uses the
    // provider's self-contained recompute+BPTT op.
    bool provider_forward_used_ = false;
    bool provider_disabled_after_failure_ = false;

    void InitializeWeights();
    bool TryProviderForward(const Tensor& input, Tensor& output);
    bool TryProviderBackward(const Tensor& grad_output, Tensor& grad_input);
    Tensor ForwardArrayFire(const Tensor& input);
    Tensor BackwardArrayFire(const Tensor& grad_output);
};

// GRU on ArrayFire; same routing and bidirectional composition as
// LSTMLayer. Final state h_n is [layers * directions, batch, hidden] with
// index 2L the level-L forward and 2L + 1 its reverse.
class CYXWIZ_API GRULayer : public Layer {
public:
    GRULayer(int input_size, int hidden_size, int num_layers = 1,
             bool batch_first = true, bool bidirectional = false,
             float dropout = 0.0f);
    ~GRULayer() override;
    GRULayer(const GRULayer&) = delete;
    GRULayer& operator=(const GRULayer&) = delete;

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
    std::vector<Tensor> grad_W_ih_;
    std::vector<Tensor> grad_W_hh_;
    std::vector<Tensor> grad_b_ih_;
    std::vector<Tensor> grad_b_hh_;

    std::vector<std::unique_ptr<GRULayer>> forward_levels_;
    std::vector<std::unique_ptr<GRULayer>> reverse_levels_;

    Tensor h_n_;
    // Owner ruling 2026-09-23 (track68): stateless per Forward unless
    // SetHiddenState was called since the last Forward (one-shot).
    bool initial_state_pending_ = false;

    // Device caches, per layer, seq-first. Gates hold r, z, n and the
    // h-side n projection: [seq, batch, 4 * hidden].
    std::vector<Tensor> cached_inputs_;
    std::vector<Tensor> cached_gates_;
    std::vector<Tensor> cached_hidden_states_;
    std::vector<Tensor> dropout_masks_;

    bool provider_forward_used_ = false;
    bool provider_disabled_after_failure_ = false;

    void InitializeWeights();
    bool TryProviderForward(const Tensor& input, Tensor& output);
    bool TryProviderBackward(const Tensor& grad_output, Tensor& grad_input);
    Tensor ForwardArrayFire(const Tensor& input, bool use_initial_state);
    Tensor BackwardArrayFire(const Tensor& grad_output);
    Tensor ForwardBidirectional(const Tensor& input);
};

} // namespace cyxwiz
