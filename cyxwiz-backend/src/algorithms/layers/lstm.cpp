#include "cyxwiz/layers/recurrent.h"
#include "cyxwiz/neural_provider.h"
#include "layer_arrayfire_utils.h"
#include "layer_recurrent_utils.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

namespace cyxwiz {

Tensor LSTMLayer::Forward(const Tensor& input) {
    const auto& input_shape = input.Shape();
    if (input.GetDataType() != DataType::Float32) {
        throw std::invalid_argument("LSTMLayer::Forward expects Float32 input");
    }
    if (input_shape.size() != 3) {
        throw std::invalid_argument(
            "LSTMLayer::Forward expects a rank-3 [batch, sequence, features] "
            "or [sequence, batch, features] tensor");
    }
    if (input_shape[2] != static_cast<size_t>(input_size_)) {
        throw std::invalid_argument(
            "LSTMLayer::Forward input feature dimension does not match input_size");
    }

    cached_input_ = input;
    // Stateless per Forward (owner ruling 2026-09-23, track68): zero h_0/c_0
    // unless an initial state was set since the last Forward.
    const bool use_initial_state = initial_state_pending_;
    initial_state_pending_ = false;
    if (!use_initial_state) {
        h_n_ = Tensor();
        c_n_ = Tensor();
    }
    provider_forward_used_ = false;

    if (bidirectional_) {
        return ForwardBidirectional(input);
    }
    Tensor output;
    if (!use_initial_state && TryProviderForward(input, output)) {
        return output;
    }
    return ForwardArrayFire(input, use_initial_state);
}

// The native neural provider serves unidirectional batch-first LSTMs that
// start from a zero state and need no inter-layer dropout.
bool LSTMLayer::TryProviderForward(const Tensor& input, Tensor& output) {
    if (!batch_first_ || provider_disabled_after_failure_ ||
        (training_ && dropout_ > 0.0f && num_layers_ > 1)) {
        return false;
    }
    const auto& input_shape = input.Shape();
    NeuralOpRequest provider_request;
    provider_request.target = CaptureCurrentNeuralDeviceTarget();
    provider_request.op = NeuralOp::LstmForward;
    provider_request.training = false;  // forward math is identical
    provider_request.dtype = DataType::Float32;
    provider_request.batch = input_shape[0];
    provider_request.seq = input_shape[1];
    provider_request.input = input_shape[2];
    provider_request.hidden = static_cast<size_t>(hidden_size_);
    provider_request.layers = static_cast<size_t>(num_layers_);
    auto provider = NeuralProviderRegistry::Instance().FindSupporting(provider_request);
    if (!provider) {
        return false;
    }
    const size_t hidden = static_cast<size_t>(hidden_size_);
    const size_t layers = static_cast<size_t>(num_layers_);
    Tensor result(std::vector<size_t>{input_shape[0], input_shape[1], hidden});
    Tensor final_hidden(std::vector<size_t>{layers, input_shape[0], hidden});
    Tensor final_cell(std::vector<size_t>{layers, input_shape[0], hidden});
    NeuralOpBuffers buffers;
    buffers.inputs = {&input};
    for (size_t l = 0; l < layers; ++l) {
        buffers.weights.push_back(&W_ih_[l]);
        buffers.weights.push_back(&W_hh_[l]);
        buffers.weights.push_back(&b_ih_[l]);
        buffers.weights.push_back(&b_hh_[l]);
    }
    buffers.outputs = {&result, &final_hidden, &final_cell};
    const auto status = provider->Execute(provider_request, buffers);
    if (!status.ok) {
        spdlog::warn("LSTMLayer::Forward: native provider failed (reason={}), "
                     "running the ArrayFire recurrence: {}",
                     BackendFallbackReasonName(status.reason), status.detail);
        return false;
    }
    provider_forward_used_ = true;
    cached_inputs_.clear();
    cached_gates_.clear();
    cached_cell_states_.clear();
    cached_hidden_states_.clear();
    dropout_masks_.clear();
    h_n_ = final_hidden;
    c_n_ = final_cell;
    output = result;
    return true;
}

Tensor LSTMLayer::ForwardBidirectional(const Tensor& input) {
    const size_t levels = forward_levels_.size();
    const auto distribute = [&](const Tensor& states, const char* what, auto setter) {
        if (states.Shape().empty()) return;
        if (states.Shape().size() != 3 || states.Shape()[0] != 2 * levels) {
            throw std::invalid_argument(std::string(what) +
                                        " must be [num_layers * 2, batch, hidden]");
        }
        for (size_t level = 0; level < levels; ++level) {
            setter(*forward_levels_[level], recurrent_detail::StateAt(states, level, what));
            setter(*reverse_levels_[level], recurrent_detail::StateAt(states, levels + level, what));
        }
    };
    distribute(h_n_, "LSTM initial hidden state",
               [](LSTMLayer& child, const Tensor& state) { child.SetHiddenState(state); });
    distribute(c_n_, "LSTM initial cell state",
               [](LSTMLayer& child, const Tensor& state) { child.SetCellState(state); });

    const Tensor output = recurrent_detail::BidirectionalForward(
        forward_levels_, reverse_levels_, input, batch_first_, dropout_, training_, dropout_masks_);

    std::vector<Tensor> hidden_states;
    std::vector<Tensor> cell_states;
    for (const auto* levels_of_direction : {&forward_levels_, &reverse_levels_}) {
        for (const auto& child : *levels_of_direction) {
            hidden_states.push_back(child->GetHiddenState());
            cell_states.push_back(child->GetCellState());
        }
    }
    h_n_ = recurrent_detail::StackStates(hidden_states);
    c_n_ = recurrent_detail::StackStates(cell_states);
    return output;
}

Tensor LSTMLayer::ForwardArrayFire(const Tensor& input, bool use_initial_state) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const auto& shape = input.Shape();
    const size_t batch = batch_first_ ? shape[0] : shape[1];
    const size_t seq = batch_first_ ? shape[1] : shape[0];
    const size_t layers = static_cast<size_t>(num_layers_);
    const size_t hidden = static_cast<size_t>(hidden_size_);
    const std::vector<size_t> state_shape{layers, batch, hidden};
    const auto initial_state = [&](const Tensor& state, const char* what) {
        if (!use_initial_state || state.Shape().empty()) {
            return af::array(af::constant(0.0f, af::dim4(static_cast<dim_t>(layers),
                                                         static_cast<dim_t>(batch),
                                                         static_cast<dim_t>(hidden))));
        }
        if (state.Shape() != state_shape) {
            throw std::invalid_argument(std::string(what) + " must be [num_layers, batch, hidden]");
        }
        return state.GetSemanticArray();
    };

    try {
        const af::array h0_all = initial_state(h_n_, "LSTM initial hidden state");
        const af::array c0_all = initial_state(c_n_, "LSTM initial cell state");
        const int seq_i = CheckedIntDim(seq, "seq_len");
        const int batch_i = CheckedIntDim(batch, "batch_size");
        const int H = hidden_size_;
        const int G = 4 * H;

        af::array layer_input = input.GetSemanticArray();
        if (batch_first_) {
            layer_input = af::reorder(layer_input, 1, 0, 2);
        }
        af::array h_final = af::constant(0.0f, af::dim4(static_cast<dim_t>(layers), batch_i, H));
        af::array c_final = af::constant(0.0f, af::dim4(static_cast<dim_t>(layers), batch_i, H));

        cached_inputs_.clear();
        cached_gates_.clear();
        cached_cell_states_.clear();
        cached_hidden_states_.clear();
        dropout_masks_.clear();

        for (int layer = 0; layer < num_layers_; ++layer) {
            const af::array W_ih = W_ih_[layer].GetSemanticArray();
            const af::array W_hh_t = af::transpose(W_hh_[layer].GetSemanticArray());
            const af::array b_ih = b_ih_[layer].GetSemanticArray();
            const af::array b_hh_rows = af::tile(af::transpose(b_hh_[layer].GetSemanticArray()),
                                                 static_cast<unsigned int>(batch_i));
            const dim_t in = layer_input.dims(2);

            // Input projections for every time step at once: [seq, batch, 4H].
            af::array input_proj = af::matmul(
                af::moddims(layer_input, af::dim4(static_cast<dim_t>(seq_i) * batch_i, in)),
                af::transpose(W_ih));
            input_proj.eval();
            input_proj = input_proj + af::tile(af::transpose(b_ih),
                                               static_cast<unsigned int>(seq_i * batch_i));
            input_proj = af::moddims(input_proj, af::dim4(seq_i, batch_i, G));
            input_proj.eval();

            af::array h = af::moddims(h0_all(layer, af::span, af::span), af::dim4(batch_i, H));
            af::array c = af::moddims(c0_all(layer, af::span, af::span), af::dim4(batch_i, H));
            af::array h_states = af::constant(0.0f, af::dim4(seq_i + 1, batch_i, H));
            af::array c_states = af::constant(0.0f, af::dim4(seq_i + 1, batch_i, H));
            af::array all_gates = af::constant(0.0f, af::dim4(seq_i, batch_i, G));
            h_states(0, af::span, af::span) = af::moddims(h, af::dim4(1, batch_i, H));
            c_states(0, af::span, af::span) = af::moddims(c, af::dim4(1, batch_i, H));

            for (int t = 0; t < seq_i; ++t) {
                // Pre-activation gates [i | f | g | o]: [batch, 4H].
                af::array gates = af::moddims(input_proj(t, af::span, af::span),
                                              af::dim4(batch_i, G)) +
                                  af::matmul(h, W_hh_t) + b_hh_rows;
                gates.eval();
                const af::array i_gate = af::sigmoid(gates(af::span, af::seq(0, H - 1)));
                const af::array f_gate = af::sigmoid(gates(af::span, af::seq(H, 2 * H - 1)));
                const af::array g_gate = af::tanh(gates(af::span, af::seq(2 * H, 3 * H - 1)));
                const af::array o_gate = af::sigmoid(gates(af::span, af::seq(3 * H, 4 * H - 1)));
                c = f_gate * c + i_gate * g_gate;
                c.eval();
                h = o_gate * af::tanh(c);
                h.eval();
                // Per-step barriers keep the lazy JIT graph (and CUDA's
                // generated-kernel parameter block) bounded.
                h_states(t + 1, af::span, af::span) = af::moddims(h, af::dim4(1, batch_i, H));
                c_states(t + 1, af::span, af::span) = af::moddims(c, af::dim4(1, batch_i, H));
                all_gates(t, af::span, af::span) = af::moddims(gates, af::dim4(1, batch_i, G));
                h_states.eval();
                c_states.eval();
                all_gates.eval();
            }

            cached_inputs_.push_back(Tensor::FromSemanticArray(
                layer_input, {seq, batch, static_cast<size_t>(in)}));
            cached_gates_.push_back(Tensor::FromSemanticArray(
                all_gates, {seq, batch, static_cast<size_t>(G)}));
            cached_hidden_states_.push_back(Tensor::FromSemanticArray(h_states, {seq + 1, batch, hidden}));
            cached_cell_states_.push_back(Tensor::FromSemanticArray(c_states, {seq + 1, batch, hidden}));
            h_final(layer, af::span, af::span) = af::moddims(h, af::dim4(1, batch_i, H));
            c_final(layer, af::span, af::span) = af::moddims(c, af::dim4(1, batch_i, H));

            af::array layer_output = h_states(af::seq(1, seq_i), af::span, af::span);
            if (training_ && dropout_ > 0.0f && layer + 1 < num_layers_) {
                dropout_masks_.push_back(recurrent_detail::MakeDropoutMask({seq, batch, hidden}, dropout_));
                layer_output = layer_output * dropout_masks_.back().GetSemanticArray();
            }
            layer_output.eval();
            layer_input = layer_output;
        }

        h_final.eval();
        c_final.eval();
        h_n_ = Tensor::FromSemanticArray(h_final, state_shape);
        c_n_ = Tensor::FromSemanticArray(c_final, state_shape);
        if (batch_first_) {
            layer_input = af::reorder(layer_input, 1, 0, 2);
        }
        layer_input.eval();
        return Tensor::FromSemanticArray(
            layer_input, batch_first_ ? std::vector<size_t>{batch, seq, hidden}
                                      : std::vector<size_t>{seq, batch, hidden});
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("LSTMLayer::Forward failed on the ArrayFire device: ") +
                                 e.what());
    }
#else
    (void)input;
    (void)use_initial_state;
    recurrent_detail::ThrowWithoutArrayFire("LSTMLayer");
#endif
}

} // namespace cyxwiz
