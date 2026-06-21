#include "cyxwiz/layers/recurrent.h"
#include "cyxwiz/debug_hooks.h"
#include "cyxwiz/recurrent_cuda_placement.h"
#include "layer_arrayfire_utils.h"
#include "layer_recurrent_utils.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

#include <spdlog/spdlog.h>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace cyxwiz {

GRULayer::GRULayer(int input_size, int hidden_size, int num_layers,
                   bool batch_first, bool bidirectional, float dropout)
    : input_size_(input_size), hidden_size_(hidden_size), num_layers_(num_layers),
      batch_first_(batch_first), bidirectional_(bidirectional), dropout_(dropout) {

    InitializeWeights();
}

void GRULayer::InitializeWeights() {
    int num_directions = bidirectional_ ? 2 : 1;

    W_ih_.resize(num_layers_);
    W_hh_.resize(num_layers_);
    b_ih_.resize(num_layers_);
    b_hh_.resize(num_layers_);
    grad_W_ih_.resize(num_layers_);
    grad_W_hh_.resize(num_layers_);
    grad_b_ih_.resize(num_layers_);
    grad_b_hh_.resize(num_layers_);

    if (bidirectional_) {
        W_ih_reverse_.resize(num_layers_);
        W_hh_reverse_.resize(num_layers_);
        b_ih_reverse_.resize(num_layers_);
        b_hh_reverse_.resize(num_layers_);
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    for (int layer = 0; layer < num_layers_; layer++) {
        int layer_input_size = (layer == 0) ? input_size_ : hidden_size_ * num_directions;
        int gate_size = 3 * hidden_size_;  // GRU has 3 gates

        float limit_ih = std::sqrt(6.0f / (layer_input_size + hidden_size_));
        float limit_hh = std::sqrt(6.0f / (hidden_size_ + hidden_size_));

        af::array w_ih = af::randu(af::dim4(gate_size, layer_input_size), af::dtype::f32) * 2.0f * limit_ih - limit_ih;
        af::array w_hh = af::randu(af::dim4(gate_size, hidden_size_), af::dtype::f32) * 2.0f * limit_hh - limit_hh;
        af::array b_ih = af::constant(0.0f, af::dim4(gate_size));
        af::array b_hh = af::constant(0.0f, af::dim4(gate_size));

        W_ih_[layer] = AfToTensor(w_ih);
        W_hh_[layer] = AfToTensor(w_hh);
        b_ih_[layer] = AfToTensor(b_ih);
        b_hh_[layer] = AfToTensor(b_hh);

        grad_W_ih_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size), static_cast<size_t>(layer_input_size)});
        grad_W_hh_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size), static_cast<size_t>(hidden_size_)});
        grad_b_ih_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});
        grad_b_hh_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});

        if (bidirectional_) {
            af::array w_ih_r = af::randu(af::dim4(gate_size, layer_input_size), af::dtype::f32) * 2.0f * limit_ih - limit_ih;
            af::array w_hh_r = af::randu(af::dim4(gate_size, hidden_size_), af::dtype::f32) * 2.0f * limit_hh - limit_hh;

            W_ih_reverse_[layer] = AfToTensor(w_ih_r);
            W_hh_reverse_[layer] = AfToTensor(w_hh_r);
            b_ih_reverse_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});
            b_hh_reverse_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});
        }
    }
#else
    for (int layer = 0; layer < num_layers_; layer++) {
        int layer_input_size = (layer == 0) ? input_size_ : hidden_size_ * num_directions;
        int gate_size = 3 * hidden_size_;

        W_ih_[layer] = Tensor::Random({static_cast<size_t>(gate_size), static_cast<size_t>(layer_input_size)});
        W_hh_[layer] = Tensor::Random({static_cast<size_t>(gate_size), static_cast<size_t>(hidden_size_)});
        b_ih_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});
        b_hh_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});
    }
#endif
}

void GRULayer::ResetState() {
    h_n_ = Tensor();
}

void GRULayer::SetHiddenState(const Tensor& h0) {
    h_n_ = h0.Clone();
}

Tensor GRULayer::Forward(const Tensor& input) {
    cached_input_ = input;

    // Hoisted weight init guard — same pattern as LSTMLayer::Forward.
    // GRULayer's constructor calls InitializeWeights() which uses the AF
    // backend; if AF init silently failed the weight tensors carry null
    // data. CPU path needs valid weights before computing anything.
    {
        const auto& shape = input.Shape();
        size_t input_dim = shape.size() == 3
            ? (batch_first_ ? shape[2] : shape[2]) : 0;
        int num_directions = bidirectional_ ? 2 : 1;
        if (W_ih_.empty() || W_ih_[0].Data<float>() == nullptr) {
            for (int layer = 0; layer < num_layers_; layer++) {
                size_t layer_input_size = (layer == 0) ? input_dim
                    : static_cast<size_t>(hidden_size_ * num_directions);
                size_t gate_size = static_cast<size_t>(3 * hidden_size_);
                W_ih_[layer] = Tensor::Random({gate_size, layer_input_size});
                W_hh_[layer] = Tensor::Random({gate_size, static_cast<size_t>(hidden_size_)});
                b_ih_[layer] = Tensor::Zeros({gate_size});
                b_hh_[layer] = Tensor::Zeros({gate_size});
            }
        }
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    const auto& af_guard_shape = input.Shape();
    const size_t af_guard_batch = batch_first_ ? af_guard_shape[0] : af_guard_shape[1];
    const size_t af_guard_seq = batch_first_ ? af_guard_shape[1] : af_guard_shape[0];
    const size_t af_guard_input = af_guard_shape.size() >= 3 ? af_guard_shape[2] : 0;
    const bool af_recurrent_allowed =
        ShouldUseArrayFireRecurrentForward(RecurrentLayerKind::GRU,
                                           af_guard_batch,
                                           af_guard_seq,
                                           af_guard_input,
                                           hidden_size_,
                                           num_layers_,
                                           bidirectional_);
    if (!bidirectional_ && af_recurrent_allowed) try {
        af::array x = TensorToAf3DRowMajor(input);
        if (batch_first_) {
            x = af::reorder(x, 1, 0, 2);
        }

        const dim_t batch_size = x.dims(1);
        const dim_t seq_len = x.dims(0);

        const auto h_shape = h_n_.Shape();
        const bool h_needs_init = h_n_.NumElements() == 0 ||
                                  h_n_.Data<float>() == nullptr ||
                                  h_shape.size() != 3 ||
                                  h_shape[0] != static_cast<size_t>(num_layers_) ||
                                  h_shape[1] != static_cast<size_t>(batch_size) ||
                                  h_shape[2] != static_cast<size_t>(hidden_size_);
        if (h_needs_init) {
            h_n_ = Tensor::Zeros({static_cast<size_t>(num_layers_),
                                  static_cast<size_t>(batch_size),
                                  static_cast<size_t>(hidden_size_)});
        }

        cached_inputs_.clear();
        cached_gates_.clear();
        cached_hidden_states_.clear();
        cached_inputs_.reserve(num_layers_);
        cached_gates_.reserve(num_layers_);
        cached_hidden_states_.reserve(num_layers_);

        af::array layer_input = x;

        for (int layer = 0; layer < num_layers_; ++layer) {
            af::array W_ih = TensorToAf(W_ih_[layer]);
            af::array W_hh = TensorToAf(W_hh_[layer]);
            af::array b_ih = TensorToAf(b_ih_[layer]);
            af::array b_hh = TensorToAf(b_hh_[layer]);

            const dim_t layer_input_size = layer_input.dims(2);
            af::array input_flat = af::moddims(layer_input,
                                               af::dim4(seq_len * batch_size, layer_input_size));
            af::array input_proj = af::matmul(input_flat, af::transpose(W_ih));
            input_proj.eval();
            input_proj = input_proj + af::tile(af::transpose(b_ih),
                                               static_cast<unsigned int>(seq_len * batch_size));
            input_proj.eval();
            input_proj = af::moddims(input_proj, af::dim4(seq_len, batch_size, 3 * hidden_size_));
            input_proj.eval();

            af::array h_full = TensorToAf3DRowMajor(h_n_);
            af::array h = af::moddims(h_full(layer, af::span, af::span),
                                      af::dim4(batch_size, hidden_size_));

            const int seq_i = CheckedIntDim(static_cast<size_t>(seq_len), "seq_len");
            const int batch_i = CheckedIntDim(static_cast<size_t>(batch_size), "batch_size");
            af::array layer_output = af::constant(0.0f, af::dim4(seq_i, batch_i, hidden_size_));
            af::array layer_gates = af::constant(0.0f, af::dim4(seq_i, batch_i, 4 * hidden_size_));
            af::array layer_h_states = af::constant(0.0f, af::dim4(seq_i + 1, batch_i, hidden_size_));
            layer_h_states(0, af::span, af::span) =
                af::moddims(h, af::dim4(1, batch_i, hidden_size_));

            for (dim_t t = 0; t < seq_len; ++t) {
                const int t_idx = CheckedIntDim(static_cast<size_t>(t), "t");
                af::array x_t = af::moddims(input_proj(t_idx, af::span, af::span),
                                            af::dim4(batch_i, 3 * hidden_size_));
                x_t.eval();

                af::array h_proj = af::matmul(h, af::transpose(W_hh));
                h_proj.eval();
                h_proj = h_proj + af::tile(af::transpose(b_hh),
                                           batch_i);
                h_proj.eval();

                af::array gates = x_t + h_proj;
                gates.eval();

                af::array r_gate = gates(af::span, af::seq(0, hidden_size_ - 1));
                af::array z_gate = gates(af::span, af::seq(hidden_size_, 2 * hidden_size_ - 1));
                af::array n_input = x_t(af::span, af::seq(2 * hidden_size_, 3 * hidden_size_ - 1));
                af::array n_hidden = h_proj(af::span, af::seq(2 * hidden_size_, 3 * hidden_size_ - 1));
                r_gate.eval();
                z_gate.eval();
                n_input.eval();
                n_hidden.eval();

                af::array r = af::sigmoid(r_gate);
                af::array z = af::sigmoid(z_gate);
                r.eval();
                z.eval();

                af::array reset_hidden = r * n_hidden;
                reset_hidden.eval();
                af::array n = af::tanh(n_input + reset_hidden);
                n.eval();

                h = (1.0f - z) * n + z * h;
                h.eval();

                af::array gates_t = af::join(1, r, z, n, n_hidden);
                gates_t.eval();
                layer_output(t_idx, af::span, af::span) =
                    af::moddims(h, af::dim4(1, batch_i, hidden_size_));
                layer_gates(t_idx, af::span, af::span) =
                    af::moddims(gates_t, af::dim4(1, batch_i, 4 * hidden_size_));
                layer_h_states(t_idx + 1, af::span, af::span) =
                    af::moddims(h, af::dim4(1, batch_i, hidden_size_));

                // ArrayFire is lazy: without these barriers, recurrent
                // writes accumulate a large fused JIT graph across timesteps
                // and NVRTC can exceed CUDA's 4096-byte formal parameter
                // block even when VRAM is mostly idle.
                layer_output.eval();
                layer_gates.eval();
                layer_h_states.eval();
            }

            af::array h_full_out = TensorToAf3DRowMajor(h_n_);
            h_full_out(layer, af::span, af::span) =
                af::moddims(h, af::dim4(1, batch_size, hidden_size_));
            h_full_out.eval();
            h_n_ = AfToTensor3DRowMajor(h_full_out);

            cached_inputs_.push_back(AfToTensor3DRowMajor(layer_input));
            cached_gates_.push_back(AfToTensor3DRowMajor(layer_gates));
            cached_hidden_states_.push_back(AfToTensor3DRowMajor(layer_h_states));

            layer_input = layer_output;
            layer_input.eval();
        }

        if (batch_first_) {
            layer_input = af::reorder(layer_input, 1, 0, 2);
        }
        layer_input.eval();

        return AfToTensor3DRowMajor(layer_input);
    } catch (const af::exception& e) {
        DisableArrayFireCudaRecurrentAfterFailure(
            RecurrentLayerKind::GRU, "GRULayer::Forward", e.what());
        if (IsCudaJitFormalParameterOverflow(e.what())) {
            spdlog::warn("{}",
                         BuildRecurrentFormalParameterOverflowFallbackMessage(
                             "GRULayer::Forward"));
        } else {
            BackendDebugHooks::EmitDebugEvent(
                "GRULayer::Forward",
                std::string("ArrayFire fallback: ") + e.what() +
                (bidirectional_ ? " [bidirectional=true]" : " [bidirectional=false]"));
            spdlog::warn("ArrayFire GRULayer::Forward failed: {}, falling back to CPU", e.what());
        }
    }
#endif

#if 0
    // ArrayFire GRU Forward — currently disabled. Has the same family of
    // bugs LSTM AF Forward had pre-2026-04-16: 3D column-major scrambling
    // at TensorToAf boundary, slice assignment shape mismatches at
    // h_states(t,...) = h writes, and r_gate/z_gate slot ordering bugs
    // that the buggy AF Backward also depended on. Fixing AF GRU is a
    // perf follow-up — needs the same TensorToAf3DRowMajor +
    // af::moddims(rhs, dim4(1,batch,hidden)) treatment LSTM got. The
    // original AF code is preserved for reference once the perf fix is
    // attempted.
    try {
        af::array x = TensorToAf(input);
        dim_t batch_size, seq_len, input_dim;
        if (batch_first_) {
            batch_size = x.dims(0); seq_len = x.dims(1); input_dim = x.dims(2);
            x = af::reorder(x, 1, 0, 2);
        } else {
            seq_len = x.dims(0); batch_size = x.dims(1); input_dim = x.dims(2);
        }
        // ... (legacy AF body removed for brevity — see git history at
        //      0dc11e1a / pre-fix for the original implementation.)
    } catch (const af::exception& e) {
        BackendDebugHooks::EmitDebugEvent(
            "GRULayer::Forward",
            std::string("ArrayFire fallback: ") + e.what() +
            (bidirectional_ ? " [bidirectional=true]" : " [bidirectional=false]"));
        spdlog::warn("ArrayFire GRULayer::Forward failed: {}, falling back to CPU", e.what());
    }
#endif

    // CPU fallback implementation. Populates cached_inputs_, cached_gates_,
    // cached_hidden_states_ in row-major [seq_len, batch, ...] layout so
    // GRULayer::Backward (CPU BPTT below) can read them. Mirror of the
    // LSTM CPU Forward + cache layout, adapted for GRU's 3-gate structure.
    const auto& shape = input.Shape();
    size_t batch_size, seq_len, input_dim;

    if (batch_first_) {
        batch_size = shape[0];
        seq_len = shape[1];
        input_dim = shape[2];
    } else {
        seq_len = shape[0];
        batch_size = shape[1];
        input_dim = shape[2];
    }

    int num_directions = bidirectional_ ? 2 : 1;

    if (h_n_.NumElements() == 0 || h_n_.Data<float>() == nullptr) {
        h_n_ = Tensor({static_cast<size_t>(num_layers_ * num_directions),
                       batch_size,
                       static_cast<size_t>(hidden_size_)},
                      DataType::Float32);
    }

    size_t out_dim0 = batch_first_ ? batch_size : seq_len;
    size_t out_dim1 = batch_first_ ? seq_len : batch_size;
    size_t out_features = static_cast<size_t>(hidden_size_ * num_directions);
    Tensor output({out_dim0, out_dim1, out_features}, DataType::Float32);

    const float* input_data = input.Data<float>();
    float* output_data = output.Data<float>();
    float* h_data = h_n_.Data<float>();

    auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    auto tanh_f = [](float x) { return std::tanh(x); };

    // Reset caches at the top of every forward pass so BPTT reads the
    // current run's state, not a stale one.
    cached_inputs_.clear();
    cached_gates_.clear();
    cached_hidden_states_.clear();
    cached_inputs_.reserve(num_layers_);
    cached_gates_.reserve(num_layers_);
    cached_hidden_states_.reserve(num_layers_);

    Tensor layer_input = input;
    size_t layer_input_size = input_dim;

    for (int layer = 0; layer < num_layers_; layer++) {
        const float* W_ih = W_ih_[layer].Data<float>();
        const float* W_hh = W_hh_[layer].Data<float>();
        const float* b_ih = b_ih_[layer].Data<float>();
        const float* b_hh = b_hh_[layer].Data<float>();
        const int H = hidden_size_;
        const int G = 3 * H;
        const size_t forward_state_offset =
            static_cast<size_t>(layer * num_directions) * batch_size * static_cast<size_t>(H);

        const size_t layer_output_size = static_cast<size_t>(H * num_directions);
        Tensor layer_output({seq_len, batch_size, layer_output_size}, DataType::Float32);
        float* layer_out = layer_output.Data<float>();
        const float* layer_in = layer_input.Data<float>();

        // Per-layer caches, row-major:
        //   input  [seq_len, batch, layer_input_size]
        //   gates  [seq_len, batch, 4 * H]   layout per (t,b):
        //                [0..H)   r post-sigmoid
        //                [H..2H)  z post-sigmoid
        //                [2H..3H) n post-tanh
        //                [3H..4H) hn_pre   = b_hh_n + W_hh_n @ h_prev
        //                                    (the unmodulated h-side
        //                                     projection feeding n; saved
        //                                     because BPTT needs both r and
        //                                     hn_pre to split d_n into
        //                                     x-side and h-side parts)
        //   h      [seq_len + 1, batch, H]   (idx 0 = h_0)
        Tensor layer_input_cache({seq_len, batch_size, layer_input_size},
                                 DataType::Float32);
        Tensor layer_gates_cache({seq_len, batch_size, static_cast<size_t>(4 * H)},
                                 DataType::Float32);
        Tensor layer_h_cache({seq_len + 1, batch_size, static_cast<size_t>(H)},
                             DataType::Float32);
        float* in_cache_data = layer_input_cache.Data<float>();
        float* gate_cache_data = layer_gates_cache.Data<float>();
        float* h_cache_data = layer_h_cache.Data<float>();

        // Seed h_0 at cache index 0 from h_n_ for all batches.
        for (size_t b = 0; b < batch_size; b++) {
            for (int i = 0; i < H; i++) {
                h_cache_data[0 * batch_size * H + b * H + i] =
                    h_data[forward_state_offset + b * H + i];
            }
        }

        for (size_t b = 0; b < batch_size; b++) {
            std::vector<float> h(H);
            for (int i = 0; i < H; i++) {
                h[i] = h_data[forward_state_offset + b * H + i];
            }

            std::vector<float> x_proj(G), h_proj(G);

            for (size_t t = 0; t < seq_len; t++) {
                const float* x_ptr;
                if (layer == 0) {
                    if (batch_first_) x_ptr = input_data + b * seq_len * input_dim + t * input_dim;
                    else x_ptr = input_data + t * batch_size * input_dim + b * input_dim;
                } else {
                    x_ptr = layer_in + t * batch_size * layer_input_size + b * layer_input_size;
                }

                // x_proj[g] = b_ih[g] + W_ih[g,:] · x
                // h_proj[g] = b_hh[g] + W_hh[g,:] · h
                for (int g = 0; g < G; g++) {
                    float xs = b_ih[g];
                    for (size_t k = 0; k < layer_input_size; k++)
                        xs += W_ih[g * layer_input_size + k] * x_ptr[k];
                    x_proj[g] = xs;

                    float hs = b_hh[g];
                    for (int k = 0; k < H; k++)
                        hs += W_hh[g * H + k] * h[k];
                    h_proj[g] = hs;
                }

                // Snapshot input for BPTT.
                for (size_t k = 0; k < layer_input_size; k++) {
                    in_cache_data[t * batch_size * layer_input_size + b * layer_input_size + k]
                        = x_ptr[k];
                }

                // Apply gate equations and snapshot post-activations.
                //   r = sigmoid(x_proj_r + h_proj_r)
                //   z = sigmoid(x_proj_z + h_proj_z)
                //   n = tanh(x_proj_n + r * h_proj_n)
                //   h_new = (1 - z) * n + z * h_prev
                for (int i = 0; i < H; i++) {
                    float r = sigmoid(x_proj[i] + h_proj[i]);
                    float z = sigmoid(x_proj[H + i] + h_proj[H + i]);
                    float hn_pre = h_proj[2 * H + i];           // unmodulated h-side
                    float n = tanh_f(x_proj[2 * H + i] + r * hn_pre);

                    const size_t off = t * batch_size * (4 * H) + b * (4 * H);
                    gate_cache_data[off + i]           = r;
                    gate_cache_data[off + H + i]       = z;
                    gate_cache_data[off + 2 * H + i]   = n;
                    gate_cache_data[off + 3 * H + i]   = hn_pre;

                    h[i] = (1.0f - z) * n + z * h[i];
                }

                for (int i = 0; i < H; i++)
                    layer_out[t * batch_size * layer_output_size + b * layer_output_size + i] = h[i];

                // Snapshot h_t at cache index t+1.
                for (int i = 0; i < H; i++) {
                    h_cache_data[(t + 1) * batch_size * H + b * H + i] = h[i];
                }
            }

            for (int i = 0; i < H; i++) {
                h_data[forward_state_offset + b * H + i] = h[i];
            }
        }

        if (bidirectional_) {
            const float* W_ih_r = W_ih_reverse_[layer].Data<float>();
            const float* W_hh_r = W_hh_reverse_[layer].Data<float>();
            const float* b_ih_r = b_ih_reverse_[layer].Data<float>();
            const float* b_hh_r = b_hh_reverse_[layer].Data<float>();
            const size_t reverse_state_offset =
                static_cast<size_t>(layer * num_directions + 1) * batch_size * static_cast<size_t>(H);

            for (size_t b = 0; b < batch_size; b++) {
                std::vector<float> h(H);
                for (int i = 0; i < H; i++) {
                    h[i] = h_data[reverse_state_offset + b * H + i];
                }

                std::vector<float> x_proj(G), h_proj(G);

                for (size_t step = 0; step < seq_len; step++) {
                    const size_t t = seq_len - 1 - step;
                    const float* x_ptr;
                    if (layer == 0) {
                        if (batch_first_) x_ptr = input_data + b * seq_len * input_dim + t * input_dim;
                        else x_ptr = input_data + t * batch_size * input_dim + b * input_dim;
                    } else {
                        x_ptr = layer_in + t * batch_size * layer_input_size + b * layer_input_size;
                    }

                    for (int g = 0; g < G; g++) {
                        float xs = b_ih_r[g];
                        for (size_t k = 0; k < layer_input_size; k++)
                            xs += W_ih_r[g * layer_input_size + k] * x_ptr[k];
                        x_proj[g] = xs;

                        float hs = b_hh_r[g];
                        for (int k = 0; k < H; k++)
                            hs += W_hh_r[g * H + k] * h[k];
                        h_proj[g] = hs;
                    }

                    for (int i = 0; i < H; i++) {
                        float r = sigmoid(x_proj[i] + h_proj[i]);
                        float z = sigmoid(x_proj[H + i] + h_proj[H + i]);
                        float hn_pre = h_proj[2 * H + i];
                        float n = tanh_f(x_proj[2 * H + i] + r * hn_pre);
                        h[i] = (1.0f - z) * n + z * h[i];
                    }

                    for (int i = 0; i < H; i++) {
                        layer_out[t * batch_size * layer_output_size +
                                  b * layer_output_size + H + i] = h[i];
                    }
                }

                for (int i = 0; i < H; i++) {
                    h_data[reverse_state_offset + b * H + i] = h[i];
                }
            }
        }

        cached_inputs_.push_back(std::move(layer_input_cache));
        cached_gates_.push_back(std::move(layer_gates_cache));
        cached_hidden_states_.push_back(std::move(layer_h_cache));

        layer_input = layer_output;
        layer_input_size = layer_output_size;
    }

    const float* final_out = layer_input.Data<float>();
    for (size_t t = 0; t < seq_len; t++) {
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t f = 0; f < out_features; f++) {
                float val = final_out[t * batch_size * out_features + b * out_features + f];
                if (batch_first_) output_data[b * seq_len * out_features + t * out_features + f] = val;
                else output_data[t * batch_size * out_features + b * out_features + f] = val;
            }
        }
    }
    return output;
}

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
            }

            grad_W_ih_[layer] = AfToTensor(dW_ih);
            grad_W_hh_[layer] = AfToTensor(dW_hh);
            grad_b_ih_[layer] = AfToTensor(db_ih);
            grad_b_hh_[layer] = AfToTensor(db_hh);

            layer_grad = d_layer_input;
        }

        if (batch_first_) {
            layer_grad = af::reorder(layer_grad, 1, 0, 2);
        }

        return AfToTensor3DRowMajor(layer_grad);
    } catch (const af::exception& e) {
        BackendDebugHooks::EmitDebugEvent(
            "GRULayer::Backward",
            std::string("ArrayFire fallback: ") + e.what() +
            (bidirectional_ ? " [bidirectional=true]" : " [bidirectional=false]"));
        spdlog::warn("ArrayFire GRULayer::Backward failed: {}, falling back to CPU", e.what());
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

std::map<std::string, Tensor> GRULayer::GetParameters() {
    std::map<std::string, Tensor> params;
    for (int layer = 0; layer < num_layers_; layer++) {
        std::string prefix = "layer" + std::to_string(layer) + "_";
        params[prefix + "W_ih"] = W_ih_[layer];
        params[prefix + "W_hh"] = W_hh_[layer];
        params[prefix + "b_ih"] = b_ih_[layer];
        params[prefix + "b_hh"] = b_hh_[layer];
        params[prefix + "grad_W_ih"] = grad_W_ih_[layer];
        params[prefix + "grad_W_hh"] = grad_W_hh_[layer];
        params[prefix + "grad_b_ih"] = grad_b_ih_[layer];
        params[prefix + "grad_b_hh"] = grad_b_hh_[layer];

        if (bidirectional_) {
            params[prefix + "W_ih_reverse"] = W_ih_reverse_[layer];
            params[prefix + "W_hh_reverse"] = W_hh_reverse_[layer];
            params[prefix + "b_ih_reverse"] = b_ih_reverse_[layer];
            params[prefix + "b_hh_reverse"] = b_hh_reverse_[layer];
        }
    }
    return params;
}

void GRULayer::SetParameters(const std::map<std::string, Tensor>& params) {
    for (int layer = 0; layer < num_layers_; layer++) {
        std::string prefix = "layer" + std::to_string(layer) + "_";
        if (params.count(prefix + "W_ih")) W_ih_[layer] = params.at(prefix + "W_ih");
        if (params.count(prefix + "W_hh")) W_hh_[layer] = params.at(prefix + "W_hh");
        if (params.count(prefix + "b_ih")) b_ih_[layer] = params.at(prefix + "b_ih");
        if (params.count(prefix + "b_hh")) b_hh_[layer] = params.at(prefix + "b_hh");

        if (bidirectional_) {
            if (params.count(prefix + "W_ih_reverse")) W_ih_reverse_[layer] = params.at(prefix + "W_ih_reverse");
            if (params.count(prefix + "W_hh_reverse")) W_hh_reverse_[layer] = params.at(prefix + "W_hh_reverse");
            if (params.count(prefix + "b_ih_reverse")) b_ih_reverse_[layer] = params.at(prefix + "b_ih_reverse");
            if (params.count(prefix + "b_hh_reverse")) b_hh_reverse_[layer] = params.at(prefix + "b_hh_reverse");
        }
    }
}

} // namespace cyxwiz
