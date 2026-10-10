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

RNNLayer::RNNLayer(int input_size, int hidden_size, int num_layers,
                   bool batch_first, bool bidirectional,
                   const std::string& nonlinearity)
    : input_size_(input_size), hidden_size_(hidden_size), num_layers_(num_layers),
      bidirectional_(bidirectional), use_tanh_(nonlinearity == "tanh") {
    if (input_size <= 0 || hidden_size <= 0 || num_layers <= 0) {
        throw std::invalid_argument("RNNLayer needs positive input_size, hidden_size and num_layers");
    }
    if (!batch_first) {
        throw std::invalid_argument("RNNLayer supports batch_first=true only");
    }
    if (nonlinearity != "tanh" && nonlinearity != "relu") {
        throw std::invalid_argument("RNNLayer nonlinearity must be \"tanh\" or \"relu\"");
    }
    if (bidirectional_) {
        for (int level = 0; level < num_layers_; ++level) {
            const int level_input = level == 0 ? input_size_ : 2 * hidden_size_;
            forward_levels_.push_back(std::make_unique<RNNLayer>(
                level_input, hidden_size_, 1, true, false, nonlinearity));
            reverse_levels_.push_back(std::make_unique<RNNLayer>(
                level_input, hidden_size_, 1, true, false, nonlinearity));
        }
        return;
    }
    InitializeWeights();
}

RNNLayer::~RNNLayer() = default;

// PyTorch's nn.RNN initialisation: every weight and bias U(-1/sqrt(H), 1/sqrt(H)).
void RNNLayer::InitializeWeights() {
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
        const size_t hidden = static_cast<size_t>(hidden_size_);
        const float bound = 1.0f / std::sqrt(static_cast<float>(hidden_size_));
        const auto uniform = [bound](const af::dim4& dims) {
            return af::randu(dims, f32) * (2.0f * bound) - bound;
        };
        const dim_t H = hidden_size_;
        for (int layer = 0; layer < num_layers_; ++layer) {
            const size_t in = static_cast<size_t>(layer == 0 ? input_size_ : hidden_size_);
            W_ih_[layer] = Tensor::FromSemanticArray(uniform(af::dim4(H, static_cast<dim_t>(in))), {hidden, in});
            W_hh_[layer] = Tensor::FromSemanticArray(uniform(af::dim4(H, H)), {hidden, hidden});
            b_ih_[layer] = Tensor::FromSemanticArray(uniform(af::dim4(H)), {hidden});
            b_hh_[layer] = Tensor::FromSemanticArray(uniform(af::dim4(H)), {hidden});
            grad_W_ih_[layer] = Tensor::Zeros({hidden, in});
            grad_W_hh_[layer] = Tensor::Zeros({hidden, hidden});
            grad_b_ih_[layer] = Tensor::Zeros({hidden});
            grad_b_hh_[layer] = Tensor::Zeros({hidden});
        }
    } catch (const af::exception& e) {
        throw std::runtime_error(
            std::string("RNNLayer weight initialization failed on the ArrayFire device: ") + e.what());
    }
#else
    recurrent_detail::ThrowWithoutArrayFire("RNNLayer");
#endif
}

void RNNLayer::ResetState() {
    h_n_ = Tensor();
    for (auto* levels : {&forward_levels_, &reverse_levels_}) {
        for (auto& child : *levels) child->ResetState();
    }
}

Tensor RNNLayer::Forward(const Tensor& input) {
    const auto& shape = input.Shape();
    if (input.GetDataType() != DataType::Float32) {
        throw std::invalid_argument("RNNLayer::Forward expects Float32 input");
    }
    if (shape.size() != 3 || shape[2] != static_cast<size_t>(input_size_)) {
        throw std::invalid_argument("RNNLayer::Forward expects [batch, seq, input_size] input");
    }
    cached_input_ = input;
    provider_forward_used_ = false;

    if (bidirectional_) {
        const Tensor output = recurrent_detail::BidirectionalForward(
            forward_levels_, reverse_levels_, input, /*batch_first=*/true,
            /*dropout=*/0.0f, training_, dropout_masks_);
        std::vector<Tensor> hidden_states;
        for (size_t level = 0; level < forward_levels_.size(); ++level) {
            hidden_states.push_back(forward_levels_[level]->GetHiddenState());
            hidden_states.push_back(reverse_levels_[level]->GetHiddenState());
        }
        h_n_ = recurrent_detail::StackStates(hidden_states);
        return output;
    }
    Tensor output;
    if (TryProviderForward(input, output)) {
        return output;
    }
    return ForwardArrayFire(input);
}

bool RNNLayer::TryProviderForward(const Tensor& input, Tensor& output) {
    if (provider_disabled_after_failure_) {
        return false;
    }
    const auto& shape = input.Shape();
    NeuralOpRequest provider_request;
    provider_request.target = CaptureCurrentNeuralDeviceTarget();
    provider_request.op = NeuralOp::RnnForward;
    provider_request.training = false;  // forward math is identical
    provider_request.dtype = DataType::Float32;
    provider_request.batch = shape[0];
    provider_request.seq = shape[1];
    provider_request.input = shape[2];
    provider_request.hidden = static_cast<size_t>(hidden_size_);
    provider_request.layers = static_cast<size_t>(num_layers_);
    provider_request.activation = use_tanh_ ? NeuralActivation::Tanh : NeuralActivation::Relu;
    auto provider = NeuralProviderRegistry::Instance().FindSupporting(provider_request);
    if (!provider) {
        return false;
    }
    const size_t hidden = static_cast<size_t>(hidden_size_);
    const size_t layers = static_cast<size_t>(num_layers_);
    Tensor result(std::vector<size_t>{shape[0], shape[1], hidden});
    Tensor final_hidden(std::vector<size_t>{layers, shape[0], hidden});
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
        spdlog::warn("RNNLayer::Forward: native provider failed (reason={}), "
                     "running the ArrayFire recurrence: {}",
                     BackendFallbackReasonName(status.reason), status.detail);
        return false;
    }
    provider_forward_used_ = true;
    cached_inputs_.clear();
    cached_hidden_states_.clear();
    h_n_ = final_hidden;
    output = result;
    return true;
}

Tensor RNNLayer::ForwardArrayFire(const Tensor& input) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    const auto& shape = input.Shape();
    const size_t batch = shape[0];
    const size_t seq = shape[1];
    const size_t layers = static_cast<size_t>(num_layers_);
    const size_t hidden = static_cast<size_t>(hidden_size_);
    try {
        const int seq_i = CheckedIntDim(seq, "seq_len");
        const int batch_i = CheckedIntDim(batch, "batch_size");
        const int H = hidden_size_;

        af::array layer_input = af::reorder(input.GetSemanticArray(), 1, 0, 2);  // [seq, batch, in]
        af::array h_final = af::constant(0.0f, af::dim4(static_cast<dim_t>(layers), batch_i, H));
        cached_inputs_.clear();
        cached_hidden_states_.clear();

        for (int layer = 0; layer < num_layers_; ++layer) {
            const af::array W_hh_t = af::transpose(W_hh_[layer].GetSemanticArray());
            const af::array b_rows = af::tile(
                af::transpose(b_ih_[layer].GetSemanticArray() + b_hh_[layer].GetSemanticArray()),
                static_cast<unsigned int>(seq_i * batch_i));
            const dim_t in = layer_input.dims(2);

            // Input projections of every step at once: [seq * batch, H].
            af::array input_proj = af::matmul(
                af::moddims(layer_input, af::dim4(static_cast<dim_t>(seq_i) * batch_i, in)),
                af::transpose(W_ih_[layer].GetSemanticArray()));
            input_proj = af::moddims(input_proj + b_rows, af::dim4(seq_i, batch_i, H));
            input_proj.eval();

            af::array h = af::constant(0.0f, af::dim4(batch_i, H));
            af::array h_states = af::constant(0.0f, af::dim4(seq_i + 1, batch_i, H));
            for (int t = 0; t < seq_i; ++t) {
                const af::array pre =
                    af::moddims(input_proj(t, af::span, af::span), af::dim4(batch_i, H)) +
                    af::matmul(h, W_hh_t);
                h = use_tanh_ ? af::tanh(pre) : (af::max)(pre, 0.0f);
                h.eval();
                h_states(t + 1, af::span, af::span) = af::moddims(h, af::dim4(1, batch_i, H));
                // Per-step barrier keeps the lazy JIT graph bounded.
                h_states.eval();
            }

            cached_inputs_.push_back(Tensor::FromSemanticArray(
                layer_input, {seq, batch, static_cast<size_t>(in)}));
            cached_hidden_states_.push_back(Tensor::FromSemanticArray(h_states, {seq + 1, batch, hidden}));
            h_final(layer, af::span, af::span) = af::moddims(h, af::dim4(1, batch_i, H));
            layer_input = h_states(af::seq(1, seq_i), af::span, af::span);
            layer_input.eval();
        }

        h_final.eval();
        h_n_ = Tensor::FromSemanticArray(h_final, {layers, batch, hidden});
        af::array output = af::reorder(layer_input, 1, 0, 2);
        output.eval();
        return Tensor::FromSemanticArray(output, {batch, seq, hidden});
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("RNNLayer::Forward failed on the ArrayFire device: ") +
                                 e.what());
    }
#else
    (void)input;
    recurrent_detail::ThrowWithoutArrayFire("RNNLayer");
#endif
}

Tensor RNNLayer::Backward(const Tensor& grad_output) {
    if (bidirectional_) {
        return recurrent_detail::BidirectionalBackward(
            forward_levels_, reverse_levels_, grad_output, /*batch_first=*/true,
            static_cast<size_t>(hidden_size_), dropout_masks_);
    }
    if (provider_forward_used_) {
        provider_forward_used_ = false;
        Tensor grad_input;
        if (TryProviderBackward(grad_output, grad_input)) {
            return grad_input;
        }
        // The provider forward left no device caches: run the ArrayFire
        // forward for this input and stay off the provider for this layer.
        provider_disabled_after_failure_ = true;
        ForwardArrayFire(cached_input_);
    }
    return BackwardArrayFire(grad_output);
}

bool RNNLayer::TryProviderBackward(const Tensor& grad_output, Tensor& grad_input) {
    const auto& in_shape = cached_input_.Shape();
    const size_t hidden = static_cast<size_t>(hidden_size_);
    NeuralOpRequest provider_request;
    provider_request.target = CaptureCurrentNeuralDeviceTarget();
    provider_request.op = NeuralOp::RnnBackward;
    provider_request.training = true;
    provider_request.dtype = DataType::Float32;
    provider_request.batch = in_shape[0];
    provider_request.seq = in_shape[1];
    provider_request.input = in_shape[2];
    provider_request.hidden = hidden;
    provider_request.layers = static_cast<size_t>(num_layers_);
    provider_request.activation = use_tanh_ ? NeuralActivation::Tanh : NeuralActivation::Relu;
    auto provider = NeuralProviderRegistry::Instance().FindSupporting(provider_request);
    if (!provider) {
        return false;
    }
    Tensor result(in_shape);
    NeuralOpBuffers buffers;
    buffers.inputs = {&cached_input_, &grad_output};
    buffers.outputs = {&result};
    for (size_t l = 0; l < static_cast<size_t>(num_layers_); ++l) {
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
    if (!status.ok) {
        spdlog::warn("RNNLayer::Backward: native provider failed (reason={}), "
                     "recomputing on the ArrayFire recurrence: {}",
                     BackendFallbackReasonName(status.reason), status.detail);
        return false;
    }
    grad_input = result;
    return true;
}

// RNN BPTT. Per step, with dh = upstream + carry from t + 1:
//   da = dh * act'(h_t)   (tanh: 1 - h_t^2; relu: h_t > 0)
//   dW_ih += da^T x_t, dW_hh += da^T h_{t-1}, db_ih = db_hh += sum(da)
//   dx_t = da W_ih, carry = da W_hh
Tensor RNNLayer::BackwardArrayFire(const Tensor& grad_output) {
    if (cached_inputs_.size() != static_cast<size_t>(num_layers_)) {
        throw std::runtime_error("RNNLayer::Backward needs a Forward first");
    }
    const auto& top_h = cached_hidden_states_.back().Shape();  // [seq + 1, batch, H]
    const size_t seq = top_h[0] - 1;
    const size_t batch = top_h[1];
    const size_t hidden = static_cast<size_t>(hidden_size_);
    if (grad_output.Shape() != std::vector<size_t>{batch, seq, hidden}) {
        throw std::invalid_argument("RNNLayer::Backward gradient does not match the Forward output shape");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const int seq_i = CheckedIntDim(seq, "seq_len");
        const int batch_i = CheckedIntDim(batch, "batch_size");
        const int H = hidden_size_;

        af::array layer_grad = af::reorder(grad_output.GetSemanticArray(), 1, 0, 2);
        for (int layer = num_layers_ - 1; layer >= 0; --layer) {
            const af::array W_ih = W_ih_[layer].GetSemanticArray();
            const af::array W_hh = W_hh_[layer].GetSemanticArray();
            const af::array input_cache = cached_inputs_[layer].GetSemanticArray();
            const af::array h_cache = cached_hidden_states_[layer].GetSemanticArray();
            const size_t in = cached_inputs_[layer].Shape()[2];
            const int in_i = CheckedIntDim(in, "layer_input_size");

            af::array dW_ih = af::constant(0.0f, af::dim4(H, in_i));
            af::array dW_hh = af::constant(0.0f, af::dim4(H, H));
            af::array db = af::constant(0.0f, af::dim4(H));
            af::array d_layer_input = af::constant(0.0f, af::dim4(seq_i, batch_i, in_i));
            af::array dh_next = af::constant(0.0f, af::dim4(batch_i, H));

            for (int t = seq_i - 1; t >= 0; --t) {
                const af::array x_t = af::moddims(input_cache(t, af::span, af::span), af::dim4(batch_i, in_i));
                const af::array h_prev = af::moddims(h_cache(t, af::span, af::span), af::dim4(batch_i, H));
                const af::array h_t = af::moddims(h_cache(t + 1, af::span, af::span), af::dim4(batch_i, H));
                const af::array dh =
                    af::moddims(layer_grad(t, af::span, af::span), af::dim4(batch_i, H)) + dh_next;
                const af::array da = use_tanh_ ? dh * (1.0f - h_t * h_t)
                                               : dh * (h_t > 0.0f).as(f32);
                dW_ih = dW_ih + af::matmul(af::transpose(da), x_t);
                dW_hh = dW_hh + af::matmul(af::transpose(da), h_prev);
                db = db + af::moddims(af::sum(da, 0), af::dim4(H));
                d_layer_input(t, af::span, af::span) =
                    af::moddims(af::matmul(da, W_ih), af::dim4(1, batch_i, in_i));
                dh_next = af::matmul(da, W_hh);
                dW_ih.eval();
                dW_hh.eval();
                db.eval();
                d_layer_input.eval();
                dh_next.eval();
            }

            grad_W_ih_[layer] = Tensor::FromSemanticArray(dW_ih, {hidden, in});
            grad_W_hh_[layer] = Tensor::FromSemanticArray(dW_hh, {hidden, hidden});
            grad_b_ih_[layer] = Tensor::FromSemanticArray(db, {hidden});
            grad_b_hh_[layer] = Tensor::FromSemanticArray(db.copy(), {hidden});
            layer_grad = d_layer_input;
        }

        af::array grad_input = af::reorder(layer_grad, 1, 0, 2);
        grad_input.eval();
        return Tensor::FromSemanticArray(grad_input, cached_input_.Shape());
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("RNNLayer::Backward failed on the ArrayFire device: ") +
                                 e.what());
    }
#else
    recurrent_detail::ThrowWithoutArrayFire("RNNLayer");
#endif
}

std::map<std::string, Tensor> RNNLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    if (bidirectional_) {
        for (size_t level = 0; level < forward_levels_.size(); ++level) {
            recurrent_detail::AppendBidirectionalParameters(params, *forward_levels_[level], level, false);
            recurrent_detail::AppendBidirectionalParameters(params, *reverse_levels_[level], level, true);
        }
        return params;
    }
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
    if (bidirectional_) {
        for (size_t level = 0; level < forward_levels_.size(); ++level) {
            recurrent_detail::SetBidirectionalParameters(params, *forward_levels_[level], level, false);
            recurrent_detail::SetBidirectionalParameters(params, *reverse_levels_[level], level, true);
        }
        return;
    }
    for (int layer = 0; layer < num_layers_; ++layer) {
        const std::string prefix = "layer" + std::to_string(layer) + "_";
        if (params.count(prefix + "W_ih")) W_ih_[layer] = params.at(prefix + "W_ih");
        if (params.count(prefix + "W_hh")) W_hh_[layer] = params.at(prefix + "W_hh");
        if (params.count(prefix + "b_ih")) b_ih_[layer] = params.at(prefix + "b_ih");
        if (params.count(prefix + "b_hh")) b_hh_[layer] = params.at(prefix + "b_hh");
    }
}

} // namespace cyxwiz
