#include "cyxwiz/layers/recurrent.h"
#include "cyxwiz/neural_provider.h"
#include "layer_arrayfire_utils.h"
#include "layer_recurrent_utils.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

namespace cyxwiz {

GRULayer::GRULayer(int input_size, int hidden_size, int num_layers,
                   bool batch_first, bool bidirectional, float dropout)
    : input_size_(input_size), hidden_size_(hidden_size), num_layers_(num_layers),
      batch_first_(batch_first), bidirectional_(bidirectional), dropout_(dropout) {
    if (input_size <= 0 || hidden_size <= 0 || num_layers <= 0) {
        throw std::invalid_argument("GRULayer needs positive input_size, hidden_size and num_layers");
    }
    if (!(dropout >= 0.0f && dropout < 1.0f)) {
        throw std::invalid_argument("GRULayer dropout must be in [0, 1)");
    }
    if (bidirectional_) {
        for (int level = 0; level < num_layers_; ++level) {
            const int level_input = level == 0 ? input_size_ : 2 * hidden_size_;
            forward_levels_.push_back(std::make_unique<GRULayer>(
                level_input, hidden_size_, 1, batch_first_, false, 0.0f));
            reverse_levels_.push_back(std::make_unique<GRULayer>(
                level_input, hidden_size_, 1, batch_first_, false, 0.0f));
        }
        return;
    }
    InitializeWeights();
}

GRULayer::~GRULayer() = default;

void GRULayer::InitializeWeights() {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    W_ih_.resize(num_layers_);
    W_hh_.resize(num_layers_);
    b_ih_.resize(num_layers_);
    b_hh_.resize(num_layers_);
    grad_W_ih_.resize(num_layers_);
    grad_W_hh_.resize(num_layers_);
    grad_b_ih_.resize(num_layers_);
    grad_b_hh_.resize(num_layers_);
    try {
        const size_t gates = static_cast<size_t>(3 * hidden_size_);
        const size_t hidden = static_cast<size_t>(hidden_size_);
        for (int layer = 0; layer < num_layers_; ++layer) {
            const size_t in = static_cast<size_t>(layer == 0 ? input_size_ : hidden_size_);
            const float limit_ih = std::sqrt(6.0f / static_cast<float>(in + hidden));
            const float limit_hh = std::sqrt(6.0f / static_cast<float>(2 * hidden));
            const af::array w_ih =
                af::randu(af::dim4(static_cast<dim_t>(gates), static_cast<dim_t>(in)), f32) *
                    (2.0f * limit_ih) - limit_ih;
            const af::array w_hh =
                af::randu(af::dim4(static_cast<dim_t>(gates), static_cast<dim_t>(hidden)), f32) *
                    (2.0f * limit_hh) - limit_hh;
            W_ih_[layer] = Tensor::FromSemanticArray(w_ih, {gates, in});
            W_hh_[layer] = Tensor::FromSemanticArray(w_hh, {gates, hidden});
            b_ih_[layer] = Tensor::FromSemanticArray(
                af::constant(0.0f, af::dim4(static_cast<dim_t>(gates))), {gates});
            b_hh_[layer] = Tensor::FromSemanticArray(
                af::constant(0.0f, af::dim4(static_cast<dim_t>(gates))), {gates});
            grad_W_ih_[layer] = Tensor::Zeros({gates, in});
            grad_W_hh_[layer] = Tensor::Zeros({gates, hidden});
            grad_b_ih_[layer] = Tensor::Zeros({gates});
            grad_b_hh_[layer] = Tensor::Zeros({gates});
        }
    } catch (const af::exception& e) {
        throw std::runtime_error(
            std::string("GRULayer weight initialization failed on the ArrayFire device: ") + e.what());
    }
#else
    recurrent_detail::ThrowWithoutArrayFire("GRULayer");
#endif
}

void GRULayer::ResetState() {
    h_n_ = Tensor();
    initial_state_pending_ = false;
    for (auto* levels : {&forward_levels_, &reverse_levels_}) {
        for (auto& child : *levels) child->ResetState();
    }
}

void GRULayer::SetHiddenState(const Tensor& h0) {
    h_n_ = h0.Clone();
    initial_state_pending_ = true;
}

Tensor GRULayer::Forward(const Tensor& input) {
    const auto& input_shape = input.Shape();
    if (input.GetDataType() != DataType::Float32) {
        throw std::invalid_argument("GRULayer::Forward expects Float32 input");
    }
    if (input_shape.size() != 3) {
        throw std::invalid_argument(
            "GRULayer::Forward expects a rank-3 [batch, sequence, features] "
            "or [sequence, batch, features] tensor");
    }
    if (input_shape[2] != static_cast<size_t>(input_size_)) {
        throw std::invalid_argument(
            "GRULayer::Forward input feature dimension does not match input_size");
    }

    cached_input_ = input;
    // Stateless per Forward (owner ruling 2026-09-23, track68).
    const bool use_initial_state = initial_state_pending_;
    initial_state_pending_ = false;
    if (!use_initial_state) {
        h_n_ = Tensor();
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

// The native neural provider serves unidirectional batch-first GRUs that
// start from a zero state and need no inter-layer dropout.
bool GRULayer::TryProviderForward(const Tensor& input, Tensor& output) {
    if (!batch_first_ || provider_disabled_after_failure_ ||
        (training_ && dropout_ > 0.0f && num_layers_ > 1)) {
        return false;
    }
    const auto& input_shape = input.Shape();
    NeuralOpRequest provider_request;
    provider_request.target = CaptureCurrentNeuralDeviceTarget();
    provider_request.op = NeuralOp::GruForward;
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
    NeuralOpBuffers buffers;
    buffers.inputs = {&input};
    for (size_t l = 0; l < layers; ++l) {
        buffers.weights.push_back(&W_ih_[l]);
        buffers.weights.push_back(&W_hh_[l]);
        buffers.weights.push_back(&b_ih_[l]);
        buffers.weights.push_back(&b_hh_[l]);
    }
    buffers.outputs = {&result, &final_hidden};
    const auto status = provider->Execute(provider_request, buffers);
    if (!status.ok) {
        spdlog::warn("GRULayer::Forward: native provider failed (reason={}), "
                     "running the ArrayFire recurrence: {}",
                     BackendFallbackReasonName(status.reason), status.detail);
        return false;
    }
    provider_forward_used_ = true;
    cached_inputs_.clear();
    cached_gates_.clear();
    cached_hidden_states_.clear();
    dropout_masks_.clear();
    h_n_ = final_hidden;
    output = result;
    return true;
}

Tensor GRULayer::ForwardBidirectional(const Tensor& input) {
    const size_t levels = forward_levels_.size();
    if (!h_n_.Shape().empty()) {
        if (h_n_.Shape().size() != 3 || h_n_.Shape()[0] != 2 * levels) {
            throw std::invalid_argument("GRU initial hidden state must be [num_layers * 2, batch, hidden]");
        }
        for (size_t level = 0; level < levels; ++level) {
            forward_levels_[level]->SetHiddenState(
                recurrent_detail::StateAt(h_n_, 2 * level, "GRU initial hidden state"));
            reverse_levels_[level]->SetHiddenState(
                recurrent_detail::StateAt(h_n_, 2 * level + 1, "GRU initial hidden state"));
        }
    }

    const Tensor output = recurrent_detail::BidirectionalForward(
        forward_levels_, reverse_levels_, input, batch_first_, dropout_, training_, dropout_masks_);

    std::vector<Tensor> hidden_states;
    for (size_t level = 0; level < levels; ++level) {
        hidden_states.push_back(forward_levels_[level]->GetHiddenState());
        hidden_states.push_back(reverse_levels_[level]->GetHiddenState());
    }
    h_n_ = recurrent_detail::StackStates(hidden_states);
    return output;
}

Tensor GRULayer::ForwardArrayFire(const Tensor& input, bool use_initial_state) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const auto& shape = input.Shape();
    const size_t batch = batch_first_ ? shape[0] : shape[1];
    const size_t seq = batch_first_ ? shape[1] : shape[0];
    const size_t layers = static_cast<size_t>(num_layers_);
    const size_t hidden = static_cast<size_t>(hidden_size_);
    const std::vector<size_t> state_shape{layers, batch, hidden};
    if (use_initial_state && !h_n_.Shape().empty() && h_n_.Shape() != state_shape) {
        throw std::invalid_argument("GRU initial hidden state must be [num_layers, batch, hidden]");
    }

    try {
        const af::array h0_all = use_initial_state && !h_n_.Shape().empty()
            ? h_n_.GetSemanticArray()
            : af::constant(0.0f, af::dim4(static_cast<dim_t>(layers), static_cast<dim_t>(batch),
                                          static_cast<dim_t>(hidden)));
        const int seq_i = CheckedIntDim(seq, "seq_len");
        const int batch_i = CheckedIntDim(batch, "batch_size");
        const int H = hidden_size_;
        const int G = 3 * H;

        af::array layer_input = input.GetSemanticArray();
        if (batch_first_) {
            layer_input = af::reorder(layer_input, 1, 0, 2);
        }
        af::array h_final = af::constant(0.0f, af::dim4(static_cast<dim_t>(layers), batch_i, H));

        cached_inputs_.clear();
        cached_gates_.clear();
        cached_hidden_states_.clear();
        dropout_masks_.clear();

        for (int layer = 0; layer < num_layers_; ++layer) {
            const af::array W_ih = W_ih_[layer].GetSemanticArray();
            const af::array W_hh_t = af::transpose(W_hh_[layer].GetSemanticArray());
            const af::array b_ih = b_ih_[layer].GetSemanticArray();
            const af::array b_hh_rows = af::tile(af::transpose(b_hh_[layer].GetSemanticArray()),
                                                 static_cast<unsigned int>(batch_i));
            const dim_t in = layer_input.dims(2);

            af::array input_proj = af::matmul(
                af::moddims(layer_input, af::dim4(static_cast<dim_t>(seq_i) * batch_i, in)),
                af::transpose(W_ih));
            input_proj.eval();
            input_proj = input_proj + af::tile(af::transpose(b_ih),
                                               static_cast<unsigned int>(seq_i * batch_i));
            input_proj = af::moddims(input_proj, af::dim4(seq_i, batch_i, G));
            input_proj.eval();

            af::array h = af::moddims(h0_all(layer, af::span, af::span), af::dim4(batch_i, H));
            af::array h_states = af::constant(0.0f, af::dim4(seq_i + 1, batch_i, H));
            af::array all_gates = af::constant(0.0f, af::dim4(seq_i, batch_i, 4 * H));
            h_states(0, af::span, af::span) = af::moddims(h, af::dim4(1, batch_i, H));

            for (int t = 0; t < seq_i; ++t) {
                const af::array x_t = af::moddims(input_proj(t, af::span, af::span), af::dim4(batch_i, G));
                af::array h_proj = af::matmul(h, W_hh_t) + b_hh_rows;
                h_proj.eval();
                const af::array r = af::sigmoid(x_t(af::span, af::seq(0, H - 1)) +
                                                h_proj(af::span, af::seq(0, H - 1)));
                const af::array z = af::sigmoid(x_t(af::span, af::seq(H, 2 * H - 1)) +
                                                h_proj(af::span, af::seq(H, 2 * H - 1)));
                const af::array n_hidden = h_proj(af::span, af::seq(2 * H, 3 * H - 1));
                af::array n = af::tanh(x_t(af::span, af::seq(2 * H, 3 * H - 1)) + r * n_hidden);
                n.eval();
                h = (1.0f - z) * n + z * h;
                h.eval();
                // Gate cache per step: [r | z | n | h-side n projection].
                all_gates(t, af::span, af::span) =
                    af::moddims(af::join(1, r, z, n, n_hidden), af::dim4(1, batch_i, 4 * H));
                h_states(t + 1, af::span, af::span) = af::moddims(h, af::dim4(1, batch_i, H));
                // Per-step barriers keep the lazy JIT graph (and CUDA's
                // generated-kernel parameter block) bounded.
                all_gates.eval();
                h_states.eval();
            }

            cached_inputs_.push_back(Tensor::FromSemanticArray(
                layer_input, {seq, batch, static_cast<size_t>(in)}));
            cached_gates_.push_back(Tensor::FromSemanticArray(all_gates, {seq, batch, 4 * hidden}));
            cached_hidden_states_.push_back(Tensor::FromSemanticArray(h_states, {seq + 1, batch, hidden}));
            h_final(layer, af::span, af::span) = af::moddims(h, af::dim4(1, batch_i, H));

            af::array layer_output = h_states(af::seq(1, seq_i), af::span, af::span);
            if (training_ && dropout_ > 0.0f && layer + 1 < num_layers_) {
                dropout_masks_.push_back(recurrent_detail::MakeDropoutMask({seq, batch, hidden}, dropout_));
                layer_output = layer_output * dropout_masks_.back().GetSemanticArray();
            }
            layer_output.eval();
            layer_input = layer_output;
        }

        h_final.eval();
        h_n_ = Tensor::FromSemanticArray(h_final, state_shape);
        if (batch_first_) {
            layer_input = af::reorder(layer_input, 1, 0, 2);
        }
        layer_input.eval();
        return Tensor::FromSemanticArray(
            layer_input, batch_first_ ? std::vector<size_t>{batch, seq, hidden}
                                      : std::vector<size_t>{seq, batch, hidden});
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("GRULayer::Forward failed on the ArrayFire device: ") +
                                 e.what());
    }
#else
    (void)input;
    (void)use_initial_state;
    recurrent_detail::ThrowWithoutArrayFire("GRULayer");
#endif
}

std::map<std::string, Tensor> GRULayer::GetParameters() {
    std::map<std::string, Tensor> params;
    if (bidirectional_) {
        for (size_t level = 0; level < forward_levels_.size(); ++level) {
            recurrent_detail::AppendBidirectionalParameters(params, *forward_levels_[level], level, false);
            recurrent_detail::AppendBidirectionalParameters(params, *reverse_levels_[level], level, true);
        }
        return params;
    }
    for (int layer = 0; layer < num_layers_; layer++) {
        const std::string prefix = "layer" + std::to_string(layer) + "_";
        params[prefix + "W_ih"] = W_ih_[layer];
        params[prefix + "W_hh"] = W_hh_[layer];
        params[prefix + "b_ih"] = b_ih_[layer];
        params[prefix + "b_hh"] = b_hh_[layer];
        params[prefix + "grad_W_ih"] = grad_W_ih_[layer];
        params[prefix + "grad_W_hh"] = grad_W_hh_[layer];
        params[prefix + "grad_b_ih"] = grad_b_ih_[layer];
        params[prefix + "grad_b_hh"] = grad_b_hh_[layer];
    }
    return params;
}

void GRULayer::SetParameters(const std::map<std::string, Tensor>& params) {
    if (bidirectional_) {
        for (size_t level = 0; level < forward_levels_.size(); ++level) {
            recurrent_detail::SetBidirectionalParameters(params, *forward_levels_[level], level, false);
            recurrent_detail::SetBidirectionalParameters(params, *reverse_levels_[level], level, true);
        }
        return;
    }
    for (int layer = 0; layer < num_layers_; layer++) {
        const std::string prefix = "layer" + std::to_string(layer) + "_";
        if (params.count(prefix + "W_ih")) W_ih_[layer] = params.at(prefix + "W_ih");
        if (params.count(prefix + "W_hh")) W_hh_[layer] = params.at(prefix + "W_hh");
        if (params.count(prefix + "b_ih")) b_ih_[layer] = params.at(prefix + "b_ih");
        if (params.count(prefix + "b_hh")) b_hh_[layer] = params.at(prefix + "b_hh");
    }
}

} // namespace cyxwiz
