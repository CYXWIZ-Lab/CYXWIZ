#include "cyxwiz/layers/recurrent.h"
#include "lstm_direction_helpers.h"
#include "cyxwiz/debug_hooks.h"
#include "layer_arrayfire_utils.h"
#include "layer_recurrent_utils.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

namespace cyxwiz {

using lstm_detail::RunLSTMCpuDirectionBackward;

Tensor LSTMLayer::Backward(const Tensor& grad_output) {
    // Guard: caches populated by the CPU Forward path above. The AF
    // Forward path is still gated off (column-major reorder bug) so we
    // never reach a state where caches exist but weren't built by CPU.
    // Empty caches mean Forward was never called; return zeros sized
    // by the input we saw (if any) so upstream grad flow is dimension-
    // safe rather than throwing.
    if (cached_inputs_.empty() || cached_gates_.empty() ||
        cached_hidden_states_.empty() || cached_cell_states_.empty()) {
        static std::atomic<bool> warned_once{false};
        if (!warned_once.exchange(true)) {
            spdlog::warn("LSTMLayer::Backward: caches empty (Forward not run?) "
                         "— returning zero gradients. This warning fires once.");
        }
        if (cached_input_.NumElements() > 0) {
            return Tensor::Zeros(cached_input_.Shape());
        }
        return Tensor::Zeros(grad_output.Shape());
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    // AF Backward — uses row-major caches that AF Forward (or CPU
    // Forward) populated via AfToTensor3DRowMajor. Reads them back via
    // TensorToAf3DRowMajor so the AF-side semantic axes match the AF
    // Forward's working layout. Same slice-shape moddims pattern that
    // Forward needed: every (t, span, span) = X assignment to a 3D
    // proxy needs X wrapped in moddims with a leading 1 dim.
    //
    // On AF exception, falls through to the CPU BPTT below — same
    // dual-path pattern as Forward. Loss numerically agrees with the
    // CPU path within fp32 noise on the mini sentiment LSTM smoke test.
    //
    // Bidirectional Backward NOT implemented yet (forward-direction
    // only, like the legacy code). Bidirectional graphs will fall
    // through to CPU.
    constexpr bool kAfBackwardEnabled = true;
    if (kAfBackwardEnabled && !bidirectional_) try {
        af::array dout = TensorToAf3DRowMajor(grad_output);

        // Convert to seq-first if batch_first.
        if (batch_first_) {
            dout = af::reorder(dout, 1, 0, 2);
        }

        const dim_t seq_len = dout.dims(0);
        const dim_t batch_size = dout.dims(1);
        const int gate_size = 4 * hidden_size_;

        af::array layer_grad = dout;

        for (int layer = num_layers_ - 1; layer >= 0; layer--) {
            af::array W_ih = TensorToAf(W_ih_[layer]);
            af::array W_hh = TensorToAf(W_hh_[layer]);

            // Caches are row-major Tensors written by AF/CPU Forward —
            // bring them back to AF column-major with semantic axes
            // matching the Forward pass.
            af::array cached_input = TensorToAf3DRowMajor(cached_inputs_[layer]);
            af::array cached_gates = TensorToAf3DRowMajor(cached_gates_[layer]);
            af::array cached_h = TensorToAf3DRowMajor(cached_hidden_states_[layer]);
            af::array cached_c = TensorToAf3DRowMajor(cached_cell_states_[layer]);

            const dim_t layer_input_size = cached_input.dims(2);

            af::array dW_ih = af::constant(0.0f, W_ih.dims());
            af::array dW_hh = af::constant(0.0f, W_hh.dims());
            af::array db_ih = af::constant(0.0f, af::dim4(gate_size));
            af::array db_hh = af::constant(0.0f, af::dim4(gate_size));

            af::array dh_next = af::constant(0.0f, af::dim4(batch_size, hidden_size_));
            af::array dc_next = af::constant(0.0f, af::dim4(batch_size, hidden_size_));

            af::array d_layer_input = af::constant(0.0f, cached_input.dims());
            const int batch_i = CheckedIntDim(static_cast<size_t>(batch_size), "batch_size");
            const int layer_input_i = CheckedIntDim(
                static_cast<size_t>(layer_input_size), "layer_input_size");

            for (dim_t t = seq_len - 1; t >= 0; t--) {
                const int t_idx = CheckedIntDim(static_cast<size_t>(t), "t");
                // h_prev / c_prev are at cache index t (state BEFORE step t).
                // c_t is at cache index t+1 (state AFTER step t).
                // gates is at cache index t (pre-activation gates from step t).
                af::array h_prev = af::moddims(cached_h(t_idx, af::span, af::span),
                                                af::dim4(batch_i, hidden_size_));
                af::array c_prev = af::moddims(cached_c(t_idx, af::span, af::span),
                                                af::dim4(batch_i, hidden_size_));
                af::array c_t    = af::moddims(cached_c(t_idx + 1, af::span, af::span),
                                                af::dim4(batch_i, hidden_size_));
                af::array gates  = af::moddims(cached_gates(t_idx, af::span, af::span),
                                                af::dim4(batch_i, gate_size));

                // Recompute gate activations from cached pre-activations.
                af::array i_gate = af::sigmoid(gates(af::span, af::seq(0, hidden_size_ - 1)));
                af::array f_gate = af::sigmoid(gates(af::span, af::seq(hidden_size_, 2 * hidden_size_ - 1)));
                af::array g_gate = af::tanh   (gates(af::span, af::seq(2 * hidden_size_, 3 * hidden_size_ - 1)));
                af::array o_gate = af::sigmoid(gates(af::span, af::seq(3 * hidden_size_, 4 * hidden_size_ - 1)));

                // Output gradient for this timestep + carry-over from t+1.
                af::array dh = af::moddims(layer_grad(t_idx, af::span, af::span),
                                            af::dim4(batch_i, hidden_size_));
                dh = dh + dh_next;

                // h = o * tanh(c_t)
                af::array tanh_c = af::tanh(c_t);
                af::array do_pre = dh * tanh_c * o_gate * (1.0f - o_gate);
                af::array dc     = dh * o_gate * (1.0f - tanh_c * tanh_c) + dc_next;

                // c_t = f * c_prev + i * g
                af::array df_pre = dc * c_prev * f_gate * (1.0f - f_gate);
                af::array di_pre = dc * g_gate * i_gate * (1.0f - i_gate);
                af::array dg_pre = dc * i_gate * (1.0f - g_gate * g_gate);
                dc_next = dc * f_gate;

                // Pre-activation gradients in [i | f | g | o] order, matching
                // the cache layout from Forward.
                af::array dgates = af::join(1, di_pre, df_pre, dg_pre, do_pre);

                // Weight + bias grad accumulation.
                af::array x_t = af::moddims(cached_input(t_idx, af::span, af::span),
                                             af::dim4(batch_i, layer_input_i));
                dW_ih = dW_ih + af::matmul(af::transpose(dgates), x_t);
                dW_hh = dW_hh + af::matmul(af::transpose(dgates), h_prev);

                af::array sum_g = af::sum(dgates, 0);  // [1, 4H]
                db_ih = db_ih + af::moddims(sum_g, af::dim4(gate_size));
                db_hh = db_hh + af::moddims(sum_g, af::dim4(gate_size));

                // dx_t = dgates @ W_ih    (shape [batch, layer_input_size])
                // Slice assignment to a rank-3 (1, batch, input) proxy needs
                // explicit moddims to add the leading 1 — same fix Forward
                // uses for h_states/c_states/all_gates writes.
                af::array dx_t = af::matmul(dgates, W_ih);
                d_layer_input(t_idx, af::span, af::span) = af::moddims(
                    dx_t, af::dim4(1, batch_i, layer_input_i));

                // dh_prev = dgates @ W_hh
                dh_next = af::matmul(dgates, W_hh);
            }

            // Stash per-layer weight grads.
            grad_W_ih_[layer] = AfToTensor(dW_ih);
            grad_W_hh_[layer] = AfToTensor(dW_hh);
            grad_b_ih_[layer] = AfToTensor(db_ih);
            grad_b_hh_[layer] = AfToTensor(db_hh);

            // Pass dx down to the next layer (which becomes its layer_grad).
            layer_grad = d_layer_input;
        }

        // Convert back to batch_first if needed.
        if (batch_first_) {
            layer_grad = af::reorder(layer_grad, 1, 0, 2);
        }

        return AfToTensor3DRowMajor(layer_grad);
    } catch (const af::exception& e) {
        BackendDebugHooks::EmitDebugEvent(
            "LSTMLayer::Backward",
            std::string("ArrayFire fallback: ") + e.what() +
            (bidirectional_ ? " [bidirectional=true]" : " [bidirectional=false]"));
        spdlog::warn("ArrayFire LSTMLayer::Backward failed: {}, falling back to CPU", e.what());
    }
#endif

    // CPU BPTT. Reads the row-major caches populated by CPU Forward:
    //   cached_inputs_[L]          [seq_len, batch, layer_input_size]
    //   cached_gates_[L]           [seq_len, batch, 4 * hidden_size]   (pre-activations)
    //   cached_hidden_states_[L]   [seq_len + 1, batch, hidden_size]   (idx 0 = h_0)
    //   cached_cell_states_[L]     [seq_len + 1, batch, hidden_size]   (idx 0 = c_0)
    //
    // Walks layers in reverse (num_layers - 1 → 0), and within each
    // layer walks timesteps in reverse (seq_len - 1 → 0). Accumulates
    // dW_ih, dW_hh, db_ih, db_hh per layer and builds a dL/d(input)
    // tensor that feeds upstream (Embedding in the typical text graph).
    //
    // Gate layout matches Forward: [i | f | g | o] in contiguous slices
    // of 4*hidden_size.

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

    auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    auto tanh_f  = [](float x) { return std::tanh(x); };

    const int H = hidden_size_;
    const int G = 4 * H;

    // Ensure grad buffers exist and are sized correctly.
    if (static_cast<int>(grad_W_ih_.size()) < num_layers_) grad_W_ih_.resize(num_layers_);
    if (static_cast<int>(grad_W_hh_.size()) < num_layers_) grad_W_hh_.resize(num_layers_);
    if (static_cast<int>(grad_b_ih_.size()) < num_layers_) grad_b_ih_.resize(num_layers_);
    if (static_cast<int>(grad_b_hh_.size()) < num_layers_) grad_b_hh_.resize(num_layers_);
    if (static_cast<int>(grad_W_ih_reverse_.size()) < num_layers_) grad_W_ih_reverse_.resize(num_layers_);
    if (static_cast<int>(grad_W_hh_reverse_.size()) < num_layers_) grad_W_hh_reverse_.resize(num_layers_);
    if (static_cast<int>(grad_b_ih_reverse_.size()) < num_layers_) grad_b_ih_reverse_.resize(num_layers_);
    if (static_cast<int>(grad_b_hh_reverse_.size()) < num_layers_) grad_b_hh_reverse_.resize(num_layers_);

    if (bidirectional_) {
        Tensor layer_grad = Tensor::Zeros(cached_input_.Shape());

        for (int layer = num_layers_ - 1; layer >= 0; --layer) {
            const size_t fwd_idx = static_cast<size_t>(layer * 2);
            const size_t rev_idx = fwd_idx + 1;

            auto fwd = RunLSTMCpuDirectionBackward(
                grad_output,
                0,
                cached_inputs_[fwd_idx],
                cached_gates_[fwd_idx],
                cached_hidden_states_[fwd_idx],
                cached_cell_states_[fwd_idx],
                W_ih_[layer],
                W_hh_[layer],
                hidden_size_,
                batch_first_,
                false);
            auto rev = RunLSTMCpuDirectionBackward(
                grad_output,
                static_cast<size_t>(hidden_size_),
                cached_inputs_[rev_idx],
                cached_gates_[rev_idx],
                cached_hidden_states_[rev_idx],
                cached_cell_states_[rev_idx],
                W_ih_reverse_[layer],
                W_hh_reverse_[layer],
                hidden_size_,
                batch_first_,
                true);

            grad_W_ih_[layer] = std::move(fwd.grad_W_ih);
            grad_W_hh_[layer] = std::move(fwd.grad_W_hh);
            grad_b_ih_[layer] = std::move(fwd.grad_b_ih);
            grad_b_hh_[layer] = std::move(fwd.grad_b_hh);
            grad_W_ih_reverse_[layer] = std::move(rev.grad_W_ih);
            grad_W_hh_reverse_[layer] = std::move(rev.grad_W_hh);
            grad_b_ih_reverse_[layer] = std::move(rev.grad_b_ih);
            grad_b_hh_reverse_[layer] = std::move(rev.grad_b_hh);

            const float* fwd_dx = fwd.input_grad.Data<float>();
            const float* rev_dx = rev.input_grad.Data<float>();
            float* layer_dx = layer_grad.Data<float>();
            const size_t elem_count = layer_grad.NumElements();
            for (size_t i = 0; i < elem_count; ++i) {
                layer_dx[i] = fwd_dx[i] + rev_dx[i];
            }
        }

        return layer_grad;
    }

    // Upstream gradient entering the top layer. Shape is the OUTPUT
    // shape (batch_first convention matches the Forward output).
    // Reshape conceptually to [seq_len, batch, H] row-major regardless
    // of batch_first (we index with an explicit if inside the loop).
    const float* dout = grad_output.Data<float>();

    // `layer_grad` holds the gradient flowing into the CURRENT layer's
    // output (i.e. dL/d(layer_output)). For the topmost LSTM layer
    // it's dout; for inner layers it's the dx computed by the later
    // layer. Row-major [seq_len, batch, hidden_size].
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

    // dL/d(input to this layer). Sized per-layer since input_dim varies.
    std::vector<float> d_layer_input;

    for (int layer = num_layers_ - 1; layer >= 0; --layer) {
        const size_t layer_input_size = (layer == 0)
            ? input_dim : static_cast<size_t>(H);

        const float* W_ih = W_ih_[layer].Data<float>();       // [4H, input_size]
        const float* W_hh = W_hh_[layer].Data<float>();       // [4H, H]
        const float* in_cache = cached_inputs_[layer].Data<float>();
        const float* gate_cache = cached_gates_[layer].Data<float>();
        const float* h_cache = cached_hidden_states_[layer].Data<float>();
        const float* c_cache = cached_cell_states_[layer].Data<float>();

        // Weight gradients for this layer, zero-initialized.
        std::vector<float> dW_ih(G * layer_input_size, 0.0f);
        std::vector<float> dW_hh(G * H, 0.0f);
        std::vector<float> db_ih(G, 0.0f);
        std::vector<float> db_hh(G, 0.0f);

        d_layer_input.assign(seq_len * batch_size * layer_input_size, 0.0f);

        for (size_t b = 0; b < batch_size; ++b) {
            std::vector<float> dh_next(H, 0.0f);
            std::vector<float> dc_next(H, 0.0f);

            for (int64_t t = static_cast<int64_t>(seq_len) - 1; t >= 0; --t) {
                const size_t gate_off = t * batch_size * G + b * G;
                const size_t h_prev_off = t * batch_size * H + b * H;      // cache idx t
                const size_t c_prev_off = t * batch_size * H + b * H;
                const size_t c_t_off = (t + 1) * batch_size * H + b * H;   // cache idx t+1
                const size_t in_off = t * batch_size * layer_input_size + b * layer_input_size;
                const size_t lg_off = t * batch_size * H + b * H;

                // dh combines upstream output gradient + carried-over
                // gradient from the next timestep's hidden dependency.
                std::vector<float> dh(H);
                for (int i = 0; i < H; ++i) {
                    dh[i] = layer_grad[lg_off + i] + dh_next[i];
                }

                // Recompute gate activations (cheaper than caching them).
                std::vector<float> i_g(H), f_g(H), g_g(H), o_g(H);
                for (int i = 0; i < H; ++i) {
                    i_g[i] = sigmoid(gate_cache[gate_off + i]);
                    f_g[i] = sigmoid(gate_cache[gate_off + H + i]);
                    g_g[i] = tanh_f (gate_cache[gate_off + 2 * H + i]);
                    o_g[i] = sigmoid(gate_cache[gate_off + 3 * H + i]);
                }

                // h = o * tanh(c_t)
                // c_t = f * c_prev + i * g
                std::vector<float> dgates(G, 0.0f);
                for (int i = 0; i < H; ++i) {
                    const float c_t = c_cache[c_t_off + i];
                    const float c_prev = c_cache[c_prev_off + i];
                    const float tanh_c = tanh_f(c_t);

                    // dh = dh, do_pre = dh * tanh(c) * sigmoid'(o_gate_pre)
                    const float do_pre = dh[i] * tanh_c * o_g[i] * (1.0f - o_g[i]);

                    // dc accumulates from this step's output and carry-over.
                    const float dc = dh[i] * o_g[i] * (1.0f - tanh_c * tanh_c)
                                    + dc_next[i];

                    const float df_pre = dc * c_prev * f_g[i] * (1.0f - f_g[i]);
                    const float di_pre = dc * g_g[i] * i_g[i] * (1.0f - i_g[i]);
                    const float dg_pre = dc * i_g[i] * (1.0f - g_g[i] * g_g[i]);

                    dc_next[i] = dc * f_g[i];

                    // Pre-activation gradients in [i | f | g | o] order.
                    dgates[i]           = di_pre;
                    dgates[H + i]       = df_pre;
                    dgates[2 * H + i]   = dg_pre;
                    dgates[3 * H + i]   = do_pre;
                }

                // Accumulate weight and bias grads.
                // dW_ih [G, layer_input_size] += outer(dgates, x_t)
                // dW_hh [G, H]                 += outer(dgates, h_prev)
                // db_ih [G]                   += dgates  (same for db_hh)
                for (int g = 0; g < G; ++g) {
                    const float dg = dgates[g];
                    db_ih[g] += dg;
                    db_hh[g] += dg;
                    for (size_t k = 0; k < layer_input_size; ++k) {
                        dW_ih[g * layer_input_size + k]
                            += dg * in_cache[in_off + k];
                    }
                    for (int k = 0; k < H; ++k) {
                        dW_hh[g * H + k] += dg * h_cache[h_prev_off + k];
                    }
                }

                // dx_t = dgates @ W_ih        (shape [layer_input_size])
                // dh_prev = dgates @ W_hh     (shape [H])
                for (size_t k = 0; k < layer_input_size; ++k) {
                    float s = 0.0f;
                    for (int g = 0; g < G; ++g) {
                        s += dgates[g] * W_ih[g * layer_input_size + k];
                    }
                    d_layer_input[in_off + k] = s;
                }
                for (int k = 0; k < H; ++k) {
                    float s = 0.0f;
                    for (int g = 0; g < G; ++g) {
                        s += dgates[g] * W_hh[g * H + k];
                    }
                    dh_next[k] = s;
                }
            }
        }

        // Stash per-layer weight grads.
        grad_W_ih_[layer] = Tensor({static_cast<size_t>(G), layer_input_size},
                                   dW_ih.data());
        grad_W_hh_[layer] = Tensor({static_cast<size_t>(G), static_cast<size_t>(H)},
                                   dW_hh.data());
        grad_b_ih_[layer] = Tensor({static_cast<size_t>(G)}, db_ih.data());
        grad_b_hh_[layer] = Tensor({static_cast<size_t>(G)}, db_hh.data());

        // This layer's dL/d(input) becomes the next iteration's
        // dL/d(output) for the layer below. Sizes must match: each
        // inner LSTM layer has input_size = H, so copying into
        // layer_grad is safe only when layer > 0. For layer == 0 we
        // exit the loop with d_layer_input sized [seq_len, batch, input_dim]
        // which we convert back to the user's layout below.
        if (layer > 0) {
            layer_grad = d_layer_input;  // size = seq_len * batch * H
        }
    }

    // Convert d_layer_input from row-major [seq_len, batch, input_dim]
    // to whatever batch_first says.
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