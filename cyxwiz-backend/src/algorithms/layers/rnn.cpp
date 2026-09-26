// Vanilla (Elman) RNN native CPU reference (tofix68 phase 3). This is the
// correctness oracle for the provider's pilot kernel, and it fills the
// previously missing RNN backend layer. Deliberately loop-based and
// readable — performance for this family belongs to the staged ArrayFire
// path (future) and the native provider, both parity-gated against this.

#include "cyxwiz/layers/recurrent.h"
#include "../arrayfire_backend_utils.h"
#include "cyxwiz/neural_provider.h"
#include <spdlog/spdlog.h>

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

namespace cyxwiz {

namespace {

void ValidateRnnConstruction(int input_size, int hidden_size, int num_layers,
                             bool batch_first, bool bidirectional,
                             const std::string& nonlinearity) {
    if (input_size <= 0 || hidden_size <= 0 || num_layers <= 0) {
        throw std::invalid_argument(
            "RNNLayer requires positive input_size, hidden_size, num_layers");
    }
    if (!batch_first) {
        throw std::invalid_argument(
            "RNNLayer currently supports batch_first=true only");
    }
    if (bidirectional) {
        throw std::invalid_argument(
            "RNNLayer does not support bidirectional yet; fail closed "
            "instead of silently running unidirectional");
    }
    if (nonlinearity != "tanh" && nonlinearity != "relu") {
        throw std::invalid_argument(
            "RNNLayer nonlinearity must be \"tanh\" or \"relu\"");
    }
}

} // namespace

RNNLayer::RNNLayer(int input_size, int hidden_size, int num_layers,
                   bool batch_first, bool bidirectional,
                   const std::string& nonlinearity)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      num_layers_(num_layers),
      batch_first_(batch_first),
      use_tanh_(nonlinearity == "tanh") {
    ValidateRnnConstruction(input_size, hidden_size, num_layers, batch_first,
                            bidirectional, nonlinearity);
    InitializeWeights();
}

void RNNLayer::InitializeWeights() {
    std::mt19937& generator = NativeRandomEngine();  // model seed on the training worker
    const float bound = 1.0f / std::sqrt(static_cast<float>(hidden_size_));
    std::uniform_real_distribution<float> distribution(-bound, bound);

    W_ih_.clear(); W_hh_.clear(); b_ih_.clear(); b_hh_.clear();
    grad_W_ih_.clear(); grad_W_hh_.clear(); grad_b_ih_.clear();
    grad_b_hh_.clear();
    for (int layer = 0; layer < num_layers_; ++layer) {
        const size_t in_features = layer == 0
            ? static_cast<size_t>(input_size_)
            : static_cast<size_t>(hidden_size_);
        const size_t hidden = static_cast<size_t>(hidden_size_);
        Tensor W_ih({hidden, in_features});
        Tensor W_hh({hidden, hidden});
        Tensor b_ih({hidden});
        Tensor b_hh({hidden});
        for (Tensor* tensor : {&W_ih, &W_hh, &b_ih, &b_hh}) {
            float* data = tensor->Data<float>();
            for (size_t i = 0; i < tensor->NumElements(); ++i) {
                data[i] = distribution(generator);
            }
        }
        W_ih_.push_back(std::move(W_ih));
        W_hh_.push_back(std::move(W_hh));
        b_ih_.push_back(std::move(b_ih));
        b_hh_.push_back(std::move(b_hh));
        grad_W_ih_.emplace_back(
            Tensor::Zeros({hidden, in_features}));
        grad_W_hh_.emplace_back(Tensor::Zeros({hidden, hidden}));
        grad_b_ih_.emplace_back(Tensor::Zeros({hidden}));
        grad_b_hh_.emplace_back(Tensor::Zeros({hidden}));
    }
}

void RNNLayer::ResetState() {
    h_n_ = Tensor();
}

Tensor RNNLayer::Forward(const Tensor& input) {
    const auto& shape = input.Shape();
    if (shape.size() != 3 ||
        shape[2] != static_cast<size_t>(input_size_)) {
        throw std::invalid_argument(
            "RNNLayer::Forward expects [batch, seq, input_size] input");
    }
    const size_t batch = shape[0];
    const size_t seq = shape[1];
    const size_t hidden = static_cast<size_t>(hidden_size_);

    // tofix68 (provider 0.7.0): route through the native neural provider
    // when one serves the run's selected device and supports the exact
    // tuple (stacked, unidirectional, batch-first — RNNLayer's whole
    // contract). Mirror of LSTMLayer::Forward.
    provider_forward_used_ = false;
    if (!provider_disabled_after_failure_) {
        NeuralOpRequest provider_request;
        provider_request.target = CaptureCurrentNeuralDeviceTarget();
        provider_request.op = NeuralOp::RnnForward;
        provider_request.training = false;  // forward math is identical
        provider_request.dtype = DataType::Float32;
        provider_request.batch = batch;
        provider_request.seq = seq;
        provider_request.input = shape[2];
        provider_request.hidden = hidden;
        provider_request.layers = static_cast<size_t>(num_layers_);
        provider_request.activation = use_tanh_ ? NeuralActivation::Tanh
                                                : NeuralActivation::Relu;
        if (auto provider = NeuralProviderRegistry::Instance()
                                .FindSupporting(provider_request)) {
            const size_t layers = static_cast<size_t>(num_layers_);
            Tensor output(std::vector<size_t>{batch, seq, hidden});
            Tensor final_hidden(std::vector<size_t>{layers, batch, hidden});
            NeuralOpBuffers buffers;
            buffers.inputs = {&input};
            for (size_t l = 0; l < layers; ++l) {
                buffers.weights.push_back(&W_ih_[l]);
                buffers.weights.push_back(&W_hh_[l]);
                buffers.weights.push_back(&b_ih_[l]);
                buffers.weights.push_back(&b_hh_[l]);
            }
            buffers.outputs = {&output, &final_hidden};
            const auto status = provider->Execute(provider_request, buffers);
            if (status.ok) {
                provider_forward_used_ = true;
                provider_input_cache_ = input.Clone();
                cached_inputs_.clear();
                cached_hidden_states_.clear();
                // RNNLayer reports the TOP layer's final state [batch, hidden].
                h_n_ = Tensor({batch, hidden},
                              final_hidden.ReadData<float>() +
                                  (layers - 1) * batch * hidden,
                              DataType::Float32);
                return output;
            }
            spdlog::warn(
                "RNNLayer::Forward: native provider failed (reason={}), "
                "falling back to the CPU reference: {}",
                BackendFallbackReasonName(status.reason), status.detail);
        }
    }

    cached_inputs_.clear();
    cached_hidden_states_.clear();

    Tensor layer_input = input.Clone();
    for (int layer = 0; layer < num_layers_; ++layer) {
        const size_t in_features = layer == 0
            ? static_cast<size_t>(input_size_)
            : hidden;
        cached_inputs_.push_back(layer_input.Clone());

        const float* x = layer_input.ReadData<float>();
        const float* W_ih = W_ih_[layer].ReadData<float>();
        const float* W_hh = W_hh_[layer].ReadData<float>();
        const float* b_ih = b_ih_[layer].ReadData<float>();
        const float* b_hh = b_hh_[layer].ReadData<float>();

        // Hidden states h_0..h_T stored as [seq+1, batch, hidden]; h_0 = 0.
        Tensor states = Tensor::Zeros({seq + 1, batch, hidden});
        float* h = states.Data<float>();
        Tensor output({batch, seq, hidden});
        float* y = output.Data<float>();

        for (size_t t = 0; t < seq; ++t) {
            const float* h_prev = h + t * batch * hidden;
            float* h_next = h + (t + 1) * batch * hidden;
            for (size_t b = 0; b < batch; ++b) {
                const float* x_t = x + (b * seq + t) * in_features;
                for (size_t j = 0; j < hidden; ++j) {
                    float a = b_ih[j] + b_hh[j];
                    const float* w_row = W_ih + j * in_features;
                    for (size_t i = 0; i < in_features; ++i) {
                        a += w_row[i] * x_t[i];
                    }
                    const float* u_row = W_hh + j * hidden;
                    const float* h_row = h_prev + b * hidden;
                    for (size_t i = 0; i < hidden; ++i) {
                        a += u_row[i] * h_row[i];
                    }
                    const float value = use_tanh_
                        ? std::tanh(a)
                        : (a > 0.0f ? a : 0.0f);
                    h_next[b * hidden + j] = value;
                    y[(b * seq + t) * hidden + j] = value;
                }
            }
        }
        cached_hidden_states_.push_back(std::move(states));
        layer_input = std::move(output);
    }

    // Final hidden state of the top layer: [batch, hidden].
    const float* top =
        cached_hidden_states_.back().ReadData<float>() +
        seq * batch * hidden;
    h_n_ = Tensor({batch, hidden}, top, DataType::Float32);
    return layer_input;
}

Tensor RNNLayer::Backward(const Tensor& grad_output) {
    // tofix68 (provider 0.7.0): provider-executed Forward -> provider
    // backward (self-contained recompute + BPTT). On failure, disable the
    // provider for this instance and recompute Forward on the CPU so the
    // reference BPTT below has its caches.
    if (provider_forward_used_) {
        provider_forward_used_ = false;
        const auto& in_shape = provider_input_cache_.Shape();
        const size_t hidden = static_cast<size_t>(hidden_size_);
        const size_t layers = static_cast<size_t>(num_layers_);
        NeuralOpRequest provider_request;
        provider_request.target = CaptureCurrentNeuralDeviceTarget();
        provider_request.op = NeuralOp::RnnBackward;
        provider_request.training = true;
        provider_request.dtype = DataType::Float32;
        provider_request.batch = in_shape[0];
        provider_request.seq = in_shape[1];
        provider_request.input = in_shape[2];
        provider_request.hidden = hidden;
        provider_request.layers = layers;
        provider_request.activation = use_tanh_ ? NeuralActivation::Tanh
                                                : NeuralActivation::Relu;
        if (auto provider = NeuralProviderRegistry::Instance()
                                .FindSupporting(provider_request)) {
            Tensor grad_input(in_shape);
            NeuralOpBuffers buffers;
            buffers.inputs = {&provider_input_cache_, &grad_output};
            buffers.outputs = {&grad_input};
            for (size_t l = 0; l < layers; ++l) {
                const size_t in = l == 0 ? in_shape[2] : hidden;
                grad_W_ih_[l] = Tensor::Zeros({hidden, in});
                grad_W_hh_[l] = Tensor::Zeros({hidden, hidden});
                grad_b_ih_[l] = Tensor::Zeros({hidden});
                grad_b_hh_[l] = Tensor::Zeros({hidden});
                buffers.weights.push_back(&W_ih_[l]);
                buffers.weights.push_back(&W_hh_[l]);
                buffers.weights.push_back(&b_ih_[l]);
                buffers.weights.push_back(&b_hh_[l]);
                buffers.gradients.push_back(&grad_W_ih_[l]);
                buffers.gradients.push_back(&grad_W_hh_[l]);
                buffers.gradients.push_back(&grad_b_ih_[l]);
                buffers.gradients.push_back(&grad_b_hh_[l]);
            }
            const auto status = provider->Execute(provider_request, buffers);
            if (status.ok) {
                return grad_input;
            }
            spdlog::warn(
                "RNNLayer::Backward: native provider failed (reason={}), "
                "recomputing on the CPU reference: {}",
                BackendFallbackReasonName(status.reason), status.detail);
        }
        provider_disabled_after_failure_ = true;
        Forward(provider_input_cache_);
    }

    if (cached_hidden_states_.empty()) {
        throw std::logic_error(
            "RNNLayer::Backward requires a prior Forward call");
    }
    const auto& shape = grad_output.Shape();
    const size_t batch = shape.size() == 3 ? shape[0] : 0;
    const size_t seq = shape.size() == 3 ? shape[1] : 0;
    const size_t hidden = static_cast<size_t>(hidden_size_);
    if (shape.size() != 3 || shape[2] != hidden ||
        cached_hidden_states_.back().Shape()[1] != batch ||
        cached_hidden_states_.back().Shape()[0] != seq + 1) {
        throw std::invalid_argument(
            "RNNLayer::Backward expects [batch, seq, hidden] gradient "
            "matching the cached forward pass");
    }

    Tensor grad_layer = grad_output.Clone();
    for (int layer = num_layers_ - 1; layer >= 0; --layer) {
        const size_t in_features = layer == 0
            ? static_cast<size_t>(input_size_)
            : hidden;
        const float* dY = grad_layer.ReadData<float>();
        const float* x = cached_inputs_[layer].ReadData<float>();
        const float* h = cached_hidden_states_[layer].ReadData<float>();
        const float* W_ih = W_ih_[layer].ReadData<float>();
        const float* W_hh = W_hh_[layer].ReadData<float>();

        grad_W_ih_[layer] = Tensor::Zeros({hidden, in_features});
        grad_W_hh_[layer] = Tensor::Zeros({hidden, hidden});
        grad_b_ih_[layer] = Tensor::Zeros({hidden});
        grad_b_hh_[layer] = Tensor::Zeros({hidden});
        float* gW_ih = grad_W_ih_[layer].Data<float>();
        float* gW_hh = grad_W_hh_[layer].Data<float>();
        float* gb_ih = grad_b_ih_[layer].Data<float>();
        float* gb_hh = grad_b_hh_[layer].Data<float>();

        Tensor grad_input = Tensor::Zeros({batch, seq, in_features});
        float* dX = grad_input.Data<float>();
        std::vector<float> dh_next(batch * hidden, 0.0f);
        std::vector<float> da(hidden, 0.0f);

        for (size_t t = seq; t-- > 0;) {
            const float* h_t = h + (t + 1) * batch * hidden;
            const float* h_prev = h + t * batch * hidden;
            for (size_t b = 0; b < batch; ++b) {
                for (size_t j = 0; j < hidden; ++j) {
                    const float dh =
                        dY[(b * seq + t) * hidden + j] +
                        dh_next[b * hidden + j];
                    const float activated = h_t[b * hidden + j];
                    const float derivative = use_tanh_
                        ? 1.0f - activated * activated
                        : (activated > 0.0f ? 1.0f : 0.0f);
                    da[j] = dh * derivative;
                    gb_ih[j] += da[j];
                    gb_hh[j] += da[j];
                }
                const float* x_t = x + (b * seq + t) * in_features;
                const float* h_row = h_prev + b * hidden;
                float* dx_t = dX + (b * seq + t) * in_features;
                float* dh_row = dh_next.data() + b * hidden;
                for (size_t i = 0; i < hidden; ++i) {
                    dh_row[i] = 0.0f;
                }
                for (size_t j = 0; j < hidden; ++j) {
                    const float g = da[j];
                    float* gw_row = gW_ih + j * in_features;
                    const float* w_row = W_ih + j * in_features;
                    for (size_t i = 0; i < in_features; ++i) {
                        gw_row[i] += g * x_t[i];
                        dx_t[i] += g * w_row[i];
                    }
                    float* gu_row = gW_hh + j * hidden;
                    const float* u_row = W_hh + j * hidden;
                    for (size_t i = 0; i < hidden; ++i) {
                        gu_row[i] += g * h_row[i];
                        dh_row[i] += g * u_row[i];
                    }
                }
            }
        }
        grad_layer = std::move(grad_input);
    }
    return grad_layer;
}

std::map<std::string, Tensor> RNNLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    for (int layer = 0; layer < num_layers_; ++layer) {
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

void RNNLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    for (int layer = 0; layer < num_layers_; ++layer) {
        const std::string prefix = "layer" + std::to_string(layer) + "_";
        if (params.count(prefix + "W_ih"))
            W_ih_[layer] = params.at(prefix + "W_ih");
        if (params.count(prefix + "W_hh"))
            W_hh_[layer] = params.at(prefix + "W_hh");
        if (params.count(prefix + "b_ih"))
            b_ih_[layer] = params.at(prefix + "b_ih");
        if (params.count(prefix + "b_hh"))
            b_hh_[layer] = params.at(prefix + "b_hh");
    }
}

} // namespace cyxwiz
