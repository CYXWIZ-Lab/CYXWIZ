#include "cyxwiz/layers/recurrent.h"
#include "cyxwiz/debug_hooks.h"
#include "layer_arrayfire_utils.h"
#include "layer_recurrent_utils.h"

#include <atomic>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

namespace cyxwiz {

Tensor GRULayer::Backward(const Tensor& grad_output) {
    // Empty caches mean Forward was never called; return zeros sized by
    // the input we last saw so upstream grad flow is dimension-safe
    // rather than throwing. Mirror of LSTMLayer::Backward's guard.
    if (cached_inputs_.empty() || cached_gates_.empty() ||
        cached_hidden_states_.empty()) {
        static std::atomic<bool> warned_once{false};
        if (!warned_once.exchange(true)) {
            spdlog::warn("GRULayer::Backward: caches empty (Forward not run?) "
                         "— returning zero gradients. This warning fires once.");
        }
        if (cached_input_.NumElements() > 0) {
            return Tensor::Zeros(cached_input_.Shape());
        }
        return Tensor::Zeros(grad_output.Shape());
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (!bidirectional_) try {
        af::array upstream = TensorToAf3DRowMajor(grad_output);
        if (batch_first_) {
            upstream = af::reorder(upstream, 1, 0, 2);
        }

        const auto& input_shape = cached_input_.Shape();
        size_t batch_size, seq_len, input_dim;
        if (batch_first_) {
            batch_size = input_shape[0];
            seq_len = input_shape[1];
            input_dim = input_shape[2];
        } else {
            seq_len = input_shape[0];
            batch_size = input_shape[1];
            input_dim = input_shape[2];
        }

        const int H = hidden_size_;
        const int G = 3 * H;

        if (static_cast<int>(grad_W_ih_.size()) < num_layers_) grad_W_ih_.resize(num_layers_);
        if (static_cast<int>(grad_W_hh_.size()) < num_layers_) grad_W_hh_.resize(num_layers_);
        if (static_cast<int>(grad_b_ih_.size()) < num_layers_) grad_b_ih_.resize(num_layers_);
        if (static_cast<int>(grad_b_hh_.size()) < num_layers_) grad_b_hh_.resize(num_layers_);

        af::array layer_grad = upstream;

        for (int layer = num_layers_ - 1; layer >= 0; --layer) {
            const size_t layer_input_size = (layer == 0)
                ? input_dim : static_cast<size_t>(H);

            af::array W_ih = TensorToAf(W_ih_[layer]);
            af::array W_hh = TensorToAf(W_hh_[layer]);
            af::array input_cache = TensorToAf3DRowMajor(cached_inputs_[layer]);
            af::array gate_cache = TensorToAf3DRowMajor(cached_gates_[layer]);
            af::array h_cache = TensorToAf3DRowMajor(cached_hidden_states_[layer]);

            af::array dW_ih = af::constant(0.0f, af::dim4(G, static_cast<dim_t>(layer_input_size)));
            af::array dW_hh = af::constant(0.0f, af::dim4(G, static_cast<dim_t>(H)));
            af::array db_ih = af::constant(0.0f, af::dim4(G));
            af::array db_hh = af::constant(0.0f, af::dim4(G));
            af::array d_layer_input = af::constant(
                0.0f, af::dim4(seq_len, batch_size, static_cast<dim_t>(layer_input_size)));
            af::array dh_next = af::constant(0.0f, af::dim4(batch_size, H));
            af::array ones = af::constant(1.0f, af::dim4(batch_size, H));
            const int batch_i = CheckedIntDim(static_cast<size_t>(batch_size), "batch_size");
            const int layer_input_i = CheckedIntDim(layer_input_size, "layer_input_size");
            const int hidden_i = CheckedIntDim(static_cast<size_t>(H), "hidden_size");

            for (int64_t t = static_cast<int64_t>(seq_len) - 1; t >= 0; --t) {
                const int t_idx = CheckedIntDim(static_cast<size_t>(t), "t");
                af::array x_t = af::moddims(input_cache(t_idx, af::span, af::span),
                                            af::dim4(batch_i, layer_input_i));
                af::array gates_t = af::moddims(gate_cache(t_idx, af::span, af::span),
                                                af::dim4(batch_i, 4 * hidden_i));
                af::array h_prev = af::moddims(h_cache(t_idx, af::span, af::span),
                                               af::dim4(batch_i, hidden_i));
                af::array dh = af::moddims(layer_grad(t_idx, af::span, af::span),
                                           af::dim4(batch_i, hidden_i));
                dh = dh + dh_next;

                af::array r = gates_t(af::span, af::seq(0, H - 1));
                af::array z = gates_t(af::span, af::seq(H, 2 * H - 1));
                af::array n = gates_t(af::span, af::seq(2 * H, 3 * H - 1));
                af::array hn_pre = gates_t(af::span, af::seq(3 * H, 4 * H - 1));

                af::array dn = dh * (ones - z);
                af::array dz = dh * (h_prev - n);
                af::array dh_prev_direct = dh * z;

                af::array dn_pre = dn * (ones - n * n);
                af::array dr = dn_pre * hn_pre;
                af::array d_hn_pre = dn_pre * r;

                af::array d_r_pre = dr * r * (ones - r);
                af::array d_z_pre = dz * z * (ones - z);

                af::array dgates_x = af::join(1, d_r_pre, d_z_pre, dn_pre);
                af::array dgates_h = af::join(1, d_r_pre, d_z_pre, d_hn_pre);

                dW_ih = dW_ih + af::matmul(af::transpose(dgates_x), x_t);
                dW_hh = dW_hh + af::matmul(af::transpose(dgates_h), h_prev);
                db_ih = db_ih + af::moddims(af::sum(dgates_x, 0), af::dim4(G));
                db_hh = db_hh + af::moddims(af::sum(dgates_h, 0), af::dim4(G));

                af::array dx_t = af::matmul(dgates_x, W_ih);
                d_layer_input(t_idx, af::span, af::span) =
                    af::moddims(dx_t, af::dim4(1, batch_i, layer_input_i));

                dh_next = dh_prev_direct + af::matmul(dgates_h, W_hh);
                dW_ih.eval();
                dW_hh.eval();
                db_ih.eval();
                db_hh.eval();
                d_layer_input.eval();
                dh_next.eval();
            }

            dW_ih.eval();
            dW_hh.eval();
            db_ih.eval();
            db_hh.eval();
            d_layer_input.eval();
            grad_W_ih_[layer] = AfToTensor(dW_ih);
            grad_W_hh_[layer] = AfToTensor(dW_hh);
            grad_b_ih_[layer] = AfToTensor(db_ih);
            grad_b_hh_[layer] = AfToTensor(db_hh);

            layer_grad = d_layer_input;
        }

        if (batch_first_) {
            layer_grad = af::reorder(layer_grad, 1, 0, 2);
        }
        layer_grad.eval();

        return AfToTensor3DRowMajor(layer_grad);
    } catch (const af::exception& e) {
        const BackendFallbackReason reason =
            ClassifyArrayFireBackendFallbackReason(e.what());
        const std::string context = BuildArrayFireBackendFallbackContext(
            BuildTensorShapeContext("grad_output", grad_output.Shape()));
        if (ShouldLogArrayFireBackendFallbackOnce("GRULayer::Backward", reason, context)) {
            const std::string fallback_message =
                BuildArrayFireBackendFallbackMessage(
                    "GRULayer::Backward",
                    reason,
                    reason != BackendFallbackReason::CudaJitParamOverflow,
                    e.what(),
                    context);
            BackendDebugHooks::EmitDebugEvent(
                "GRULayer::Backward",
                fallback_message +
                (bidirectional_ ? " [bidirectional=true]" : " [bidirectional=false]"));
            spdlog::warn("{}", fallback_message);
        }
    }
#endif

    // CPU BPTT for GRU. Reads the row-major caches populated by CPU Forward:
    //   cached_inputs_[L]          [seq_len, batch, layer_input_size]
    //   cached_gates_[L]           [seq_len, batch, 4 * H]
    //                              layout per (t, b):
    //                                [0..H)   r post-sigmoid
    //                                [H..2H)  z post-sigmoid
    //                                [2H..3H) n post-tanh
    //                                [3H..4H) hn_pre   (= b_hh_n + W_hh_n @ h_prev)
    //   cached_hidden_states_[L]   [seq_len + 1, batch, H]   (idx 0 = h_0)
    //
    // GRU forward equations:
    //     r       = sigmoid(x_proj_r + h_proj_r)
    //     z       = sigmoid(x_proj_z + h_proj_z)
    //     n       = tanh(x_proj_n + r * hn_pre)        // hn_pre = h_proj_n
    //     h_new   = (1 - z) * n + z * h_prev
    //
    // BPTT — per timestep, given dh_total = dL/dh_new + dh carry from t+1:
    //     dn       = dh_total * (1 - z)
    //     dz       = dh_total * (h_prev - n)
    //     dh_prev_direct = dh_total * z
    //
    //     dn_pre   = dn * (1 - n*n)                    // tanh'
    //     d_x_proj_n = dn_pre
    //     dr       = dn_pre * hn_pre                   // through r * hn_pre
    //     d_hn_pre = dn_pre * r                        // through r * hn_pre
    //
    //     d_r_pre  = dr * r * (1 - r)                  // sigmoid'
    //     d_z_pre  = dz * z * (1 - z)
    //
    //     dgates_x = [d_r_pre | d_z_pre | d_x_proj_n]  // x-side projections
    //     dgates_h = [d_r_pre | d_z_pre | d_hn_pre  ]  // h-side projections
    //                                                  // (n-slot differs!)
    //
    //     dW_ih += outer(dgates_x, x);   db_ih += dgates_x
    //     dW_hh += outer(dgates_h, h_prev); db_hh += dgates_h
    //     dx     = dgates_x @ W_ih
    //     dh_prev = dh_prev_direct + dgates_h @ W_hh
    //
    // Note: dgates_x and dgates_h DIFFER in the n-slot. The legacy AF
    // backward used the same dgates for both sides which is wrong for GRU
    // (and additionally zeroed the r-slot, so reset gate weights never
    // updated). This implementation handles both correctly.

    const auto& input_shape = cached_input_.Shape();
    size_t batch_size, seq_len, input_dim;
    if (batch_first_) {
        batch_size = input_shape[0];
        seq_len    = input_shape[1];
        input_dim  = input_shape[2];
    } else {
        seq_len    = input_shape[0];
        batch_size = input_shape[1];
        input_dim  = input_shape[2];
    }

    const int H = hidden_size_;
    const int G = 3 * H;

    if (static_cast<int>(grad_W_ih_.size()) < num_layers_) grad_W_ih_.resize(num_layers_);
    if (static_cast<int>(grad_W_hh_.size()) < num_layers_) grad_W_hh_.resize(num_layers_);
    if (static_cast<int>(grad_b_ih_.size()) < num_layers_) grad_b_ih_.resize(num_layers_);
    if (static_cast<int>(grad_b_hh_.size()) < num_layers_) grad_b_hh_.resize(num_layers_);

    // Convert grad_output (in user's batch_first/seq_first layout) into a
    // canonical [seq_len, batch, H] row-major scratch buffer for the
    // top-layer gradient.
    const float* dout = grad_output.Data<float>();
    std::vector<float> layer_grad(seq_len * batch_size * H, 0.0f);
    for (size_t t = 0; t < seq_len; ++t) {
        for (size_t b = 0; b < batch_size; ++b) {
            for (int i = 0; i < H; ++i) {
                float g = batch_first_
                    ? dout[b * seq_len * H + t * H + i]
                    : dout[t * batch_size * H + b * H + i];
                layer_grad[t * batch_size * H + b * H + i] = g;
            }
        }
    }

    std::vector<float> d_layer_input;

    for (int layer = num_layers_ - 1; layer >= 0; --layer) {
        const size_t layer_input_size = (layer == 0)
            ? input_dim : static_cast<size_t>(H);

        const float* W_ih = W_ih_[layer].Data<float>();       // [3H, input_size]
        const float* W_hh = W_hh_[layer].Data<float>();       // [3H, H]
        const float* in_cache = cached_inputs_[layer].Data<float>();
        const float* gate_cache = cached_gates_[layer].Data<float>();  // 4H per (t,b)
        const float* h_cache = cached_hidden_states_[layer].Data<float>();

        std::vector<float> dW_ih(G * layer_input_size, 0.0f);
        std::vector<float> dW_hh(G * H, 0.0f);
        std::vector<float> db_ih(G, 0.0f);
        std::vector<float> db_hh(G, 0.0f);

        d_layer_input.assign(seq_len * batch_size * layer_input_size, 0.0f);

        for (size_t b = 0; b < batch_size; ++b) {
            std::vector<float> dh_next(H, 0.0f);

            for (int64_t t = static_cast<int64_t>(seq_len) - 1; t >= 0; --t) {
                const size_t gate_off = t * batch_size * (4 * H) + b * (4 * H);
                const size_t h_prev_off = t * batch_size * H + b * H;        // cache idx t
                const size_t in_off = t * batch_size * layer_input_size + b * layer_input_size;
                const size_t lg_off = t * batch_size * H + b * H;

                std::vector<float> dgates_x(G, 0.0f);
                std::vector<float> dgates_h(G, 0.0f);

                for (int i = 0; i < H; ++i) {
                    const float r       = gate_cache[gate_off + i];
                    const float z       = gate_cache[gate_off + H + i];
                    const float n       = gate_cache[gate_off + 2 * H + i];
                    const float hn_pre  = gate_cache[gate_off + 3 * H + i];
                    const float h_prev  = h_cache[h_prev_off + i];

                    const float dh_total = layer_grad[lg_off + i] + dh_next[i];

                    const float dn = dh_total * (1.0f - z);
                    const float dz = dh_total * (h_prev - n);
                    const float dh_prev_direct = dh_total * z;

                    const float dn_pre = dn * (1.0f - n * n);
                    const float d_x_proj_n = dn_pre;
                    const float dr = dn_pre * hn_pre;
                    const float d_hn_pre = dn_pre * r;

                    const float d_r_pre = dr * r * (1.0f - r);
                    const float d_z_pre = dz * z * (1.0f - z);

                    dgates_x[i]            = d_r_pre;
                    dgates_x[H + i]        = d_z_pre;
                    dgates_x[2 * H + i]    = d_x_proj_n;

                    dgates_h[i]            = d_r_pre;
                    dgates_h[H + i]        = d_z_pre;
                    dgates_h[2 * H + i]    = d_hn_pre;

                    // Stash the direct (non-gate) carry; gate-side carry
                    // gets added below from dgates_h @ W_hh.
                    dh_next[i] = dh_prev_direct;
                }

                // Weight + bias accumulation.
                //   dW_ih [G, layer_input_size] += outer(dgates_x, x_t)
                //   dW_hh [G, H]                 += outer(dgates_h, h_prev)
                for (int g = 0; g < G; ++g) {
                    const float dgx = dgates_x[g];
                    const float dgh = dgates_h[g];
                    db_ih[g] += dgx;
                    db_hh[g] += dgh;
                    for (size_t k = 0; k < layer_input_size; ++k) {
                        dW_ih[g * layer_input_size + k] += dgx * in_cache[in_off + k];
                    }
                    for (int k = 0; k < H; ++k) {
                        dW_hh[g * H + k] += dgh * h_cache[h_prev_off + k];
                    }
                }

                // dx_t = dgates_x @ W_ih   (shape [layer_input_size])
                for (size_t k = 0; k < layer_input_size; ++k) {
                    float s = 0.0f;
                    for (int g = 0; g < G; ++g) {
                        s += dgates_x[g] * W_ih[g * layer_input_size + k];
                    }
                    d_layer_input[in_off + k] = s;
                }

                // dh_prev (carries to t-1) = dh_prev_direct + dgates_h @ W_hh
                for (int k = 0; k < H; ++k) {
                    float s = 0.0f;
                    for (int g = 0; g < G; ++g) {
                        s += dgates_h[g] * W_hh[g * H + k];
                    }
                    dh_next[k] += s;
                }
            }
        }

        grad_W_ih_[layer] = Tensor({static_cast<size_t>(G), layer_input_size},
                                   dW_ih.data());
        grad_W_hh_[layer] = Tensor({static_cast<size_t>(G), static_cast<size_t>(H)},
                                   dW_hh.data());
        grad_b_ih_[layer] = Tensor({static_cast<size_t>(G)}, db_ih.data());
        grad_b_hh_[layer] = Tensor({static_cast<size_t>(G)}, db_hh.data());

        if (layer > 0) {
            layer_grad = d_layer_input;
        }
    }

    Tensor dx = Tensor::Zeros(cached_input_.Shape());
    float* dx_data = dx.Data<float>();
    for (size_t t = 0; t < seq_len; ++t) {
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t k = 0; k < input_dim; ++k) {
                const float v = d_layer_input[t * batch_size * input_dim + b * input_dim + k];
                if (batch_first_) {
                    dx_data[b * seq_len * input_dim + t * input_dim + k] = v;
                } else {
                    dx_data[t * batch_size * input_dim + b * input_dim + k] = v;
                }
            }
        }
    }
    return dx;
}

} // namespace cyxwiz
