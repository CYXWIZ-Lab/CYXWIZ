#include "cyxwiz/layers/recurrent.h"
#include "cyxwiz/neural_provider.h"
#include "layer_arrayfire_utils.h"
#include "layer_recurrent_utils.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

namespace cyxwiz {

Tensor GRULayer::Backward(const Tensor& grad_output) {
    if (bidirectional_) {
        return recurrent_detail::BidirectionalBackward(
            forward_levels_, reverse_levels_, grad_output, batch_first_,
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
        ForwardArrayFire(cached_input_, false);
    }
    return BackwardArrayFire(grad_output);
}

bool GRULayer::TryProviderBackward(const Tensor& grad_output, Tensor& grad_input) {
    const auto& in_shape = cached_input_.Shape();
    NeuralOpRequest provider_request;
    provider_request.target = CaptureCurrentNeuralDeviceTarget();
    provider_request.op = NeuralOp::GruBackward;
    provider_request.training = true;
    provider_request.dtype = DataType::Float32;
    provider_request.batch = in_shape[0];
    provider_request.seq = in_shape[1];
    provider_request.input = in_shape[2];
    provider_request.hidden = static_cast<size_t>(hidden_size_);
    provider_request.layers = static_cast<size_t>(num_layers_);
    auto provider = NeuralProviderRegistry::Instance().FindSupporting(provider_request);
    if (!provider) {
        return false;
    }
    const size_t gate_width = static_cast<size_t>(3 * hidden_size_);
    const size_t hidden = static_cast<size_t>(hidden_size_);
    Tensor result(in_shape);
    NeuralOpBuffers buffers;
    buffers.inputs = {&cached_input_, &grad_output};
    buffers.outputs = {&result};
    for (size_t l = 0; l < static_cast<size_t>(num_layers_); ++l) {
        const size_t in = l == 0 ? in_shape[2] : hidden;
        grad_W_ih_[l] = Tensor::Zeros({gate_width, in});
        grad_W_hh_[l] = Tensor::Zeros({gate_width, hidden});
        grad_b_ih_[l] = Tensor::Zeros({gate_width});
        grad_b_hh_[l] = Tensor::Zeros({gate_width});
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
        spdlog::warn("GRULayer::Backward: native provider failed (reason={}), "
                     "recomputing on the ArrayFire recurrence: {}",
                     BackendFallbackReasonName(status.reason), status.detail);
        return false;
    }
    grad_input = result;
    return true;
}

// GRU BPTT. Per step, with dh = upstream + carry from t + 1:
//   dn = dh (1 - z), dz = dh (h_prev - n), carry dh z
//   dn_pre = dn (1 - n^2), dr = dn_pre hn, d_hn = dn_pre r
//   dgates_x = [dr r(1-r) | dz z(1-z) | dn_pre]   (x-side projections)
//   dgates_h = [dr r(1-r) | dz z(1-z) | d_hn  ]   (h-side; the n slot differs)
Tensor GRULayer::BackwardArrayFire(const Tensor& grad_output) {
    if (cached_inputs_.size() != static_cast<size_t>(num_layers_)) {
        throw std::runtime_error("GRULayer::Backward needs a Forward first");
    }
    const auto& top_h = cached_hidden_states_.back().Shape();  // [seq + 1, batch, H]
    const size_t seq = top_h[0] - 1;
    const size_t batch = top_h[1];
    const size_t hidden = static_cast<size_t>(hidden_size_);
    const std::vector<size_t> expected_grad = batch_first_ ? std::vector<size_t>{batch, seq, hidden}
                                                           : std::vector<size_t>{seq, batch, hidden};
    if (grad_output.Shape() != expected_grad) {
        throw std::invalid_argument("GRULayer::Backward gradient does not match the Forward output shape");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const int seq_i = CheckedIntDim(seq, "seq_len");
        const int batch_i = CheckedIntDim(batch, "batch_size");
        const int H = hidden_size_;
        const int G = 3 * H;

        af::array layer_grad = grad_output.GetSemanticArray();
        if (batch_first_) {
            layer_grad = af::reorder(layer_grad, 1, 0, 2);
        }

        for (int layer = num_layers_ - 1; layer >= 0; --layer) {
            const af::array W_ih = W_ih_[layer].GetSemanticArray();
            const af::array W_hh = W_hh_[layer].GetSemanticArray();
            const af::array input_cache = cached_inputs_[layer].GetSemanticArray();
            const af::array gate_cache = cached_gates_[layer].GetSemanticArray();
            const af::array h_cache = cached_hidden_states_[layer].GetSemanticArray();
            const size_t in = cached_inputs_[layer].Shape()[2];
            const int in_i = CheckedIntDim(in, "layer_input_size");

            af::array dW_ih = af::constant(0.0f, af::dim4(G, in_i));
            af::array dW_hh = af::constant(0.0f, af::dim4(G, H));
            af::array db_ih = af::constant(0.0f, af::dim4(G));
            af::array db_hh = af::constant(0.0f, af::dim4(G));
            af::array d_layer_input = af::constant(0.0f, af::dim4(seq_i, batch_i, in_i));
            af::array dh_next = af::constant(0.0f, af::dim4(batch_i, H));

            for (int t = seq_i - 1; t >= 0; --t) {
                const af::array x_t = af::moddims(input_cache(t, af::span, af::span), af::dim4(batch_i, in_i));
                const af::array gates_t = af::moddims(gate_cache(t, af::span, af::span), af::dim4(batch_i, 4 * H));
                const af::array h_prev = af::moddims(h_cache(t, af::span, af::span), af::dim4(batch_i, H));
                const af::array dh =
                    af::moddims(layer_grad(t, af::span, af::span), af::dim4(batch_i, H)) + dh_next;

                const af::array r = gates_t(af::span, af::seq(0, H - 1));
                const af::array z = gates_t(af::span, af::seq(H, 2 * H - 1));
                const af::array n = gates_t(af::span, af::seq(2 * H, 3 * H - 1));
                const af::array hn_pre = gates_t(af::span, af::seq(3 * H, 4 * H - 1));

                const af::array dn_pre = dh * (1.0f - z) * (1.0f - n * n);
                const af::array d_r_pre = dn_pre * hn_pre * r * (1.0f - r);
                const af::array d_z_pre = dh * (h_prev - n) * z * (1.0f - z);
                const af::array dgates_x = af::join(1, d_r_pre, d_z_pre, dn_pre);
                const af::array dgates_h = af::join(1, d_r_pre, d_z_pre, dn_pre * r);

                dW_ih = dW_ih + af::matmul(af::transpose(dgates_x), x_t);
                dW_hh = dW_hh + af::matmul(af::transpose(dgates_h), h_prev);
                db_ih = db_ih + af::moddims(af::sum(dgates_x, 0), af::dim4(G));
                db_hh = db_hh + af::moddims(af::sum(dgates_h, 0), af::dim4(G));
                d_layer_input(t, af::span, af::span) =
                    af::moddims(af::matmul(dgates_x, W_ih), af::dim4(1, batch_i, in_i));
                dh_next = dh * z + af::matmul(dgates_h, W_hh);
                dW_ih.eval();
                dW_hh.eval();
                db_ih.eval();
                db_hh.eval();
                d_layer_input.eval();
                dh_next.eval();
            }

            const size_t gates_size = static_cast<size_t>(G);
            grad_W_ih_[layer] = Tensor::FromSemanticArray(dW_ih, {gates_size, in});
            grad_W_hh_[layer] = Tensor::FromSemanticArray(dW_hh, {gates_size, hidden});
            grad_b_ih_[layer] = Tensor::FromSemanticArray(db_ih, {gates_size});
            grad_b_hh_[layer] = Tensor::FromSemanticArray(db_hh, {gates_size});

            // The layer below fed this one through its dropout mask.
            if (layer > 0 && static_cast<size_t>(layer - 1) < dropout_masks_.size()) {
                d_layer_input = d_layer_input * dropout_masks_[static_cast<size_t>(layer - 1)].GetSemanticArray();
            }
            d_layer_input.eval();
            layer_grad = d_layer_input;
        }

        if (batch_first_) {
            layer_grad = af::reorder(layer_grad, 1, 0, 2);
        }
        layer_grad.eval();
        return Tensor::FromSemanticArray(layer_grad, cached_input_.Shape());
    } catch (const af::exception& e) {
        throw std::runtime_error(std::string("GRULayer::Backward failed on the ArrayFire device: ") +
                                 e.what());
    }
#else
    recurrent_detail::ThrowWithoutArrayFire("GRULayer");
#endif
}

} // namespace cyxwiz
