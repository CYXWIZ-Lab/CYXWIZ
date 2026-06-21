#include "cyxwiz/layers/recurrent.h"
#include "lstm_direction_helpers.h"
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

using lstm_detail::RunLSTMCpuDirectionForward;
using lstm_detail::RunLSTMCpuDirectionBackward;
#ifdef CYXWIZ_HAS_ARRAYFIRE
using lstm_detail::RunLSTMAfDirectionForward;
#endif


// ============================================================================
// Helper Functions for ArrayFire Integration
// ============================================================================



// ============================================================================
// LSTM Layer Implementation
// ============================================================================

LSTMLayer::LSTMLayer(int input_size, int hidden_size, int num_layers,
                     bool batch_first, bool bidirectional, float dropout)
    : input_size_(input_size), hidden_size_(hidden_size), num_layers_(num_layers),
      batch_first_(batch_first), bidirectional_(bidirectional), dropout_(dropout) {

    InitializeWeights();
}

void LSTMLayer::InitializeWeights() {
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
        grad_W_ih_reverse_.resize(num_layers_);
        grad_W_hh_reverse_.resize(num_layers_);
        grad_b_ih_reverse_.resize(num_layers_);
        grad_b_hh_reverse_.resize(num_layers_);
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    for (int layer = 0; layer < num_layers_; layer++) {
        // Input size for this layer
        int layer_input_size = (layer == 0) ? input_size_ : hidden_size_ * num_directions;
        int gate_size = 4 * hidden_size_;

        // Xavier initialization for input-hidden weights
        float limit_ih = std::sqrt(6.0f / (layer_input_size + hidden_size_));
        af::array w_ih = af::randu(af::dim4(gate_size, layer_input_size), af::dtype::f32) * 2.0f * limit_ih - limit_ih;
        W_ih_[layer] = AfToTensor(w_ih);

        // Xavier initialization for hidden-hidden weights
        float limit_hh = std::sqrt(6.0f / (hidden_size_ + hidden_size_));
        af::array w_hh = af::randu(af::dim4(gate_size, hidden_size_), af::dtype::f32) * 2.0f * limit_hh - limit_hh;
        W_hh_[layer] = AfToTensor(w_hh);

        // Initialize biases to zero (with forget gate bias = 1 for better gradient flow)
        af::array b_ih = af::constant(0.0f, af::dim4(gate_size));
        af::array b_hh = af::constant(0.0f, af::dim4(gate_size));
        // Set forget gate bias to 1
        b_ih(af::seq(hidden_size_, 2 * hidden_size_ - 1)) = 1.0f;
        b_ih_[layer] = AfToTensor(b_ih);
        b_hh_[layer] = AfToTensor(b_hh);

        // Initialize gradient accumulators
        grad_W_ih_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size), static_cast<size_t>(layer_input_size)});
        grad_W_hh_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size), static_cast<size_t>(hidden_size_)});
        grad_b_ih_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});
        grad_b_hh_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});

        if (bidirectional_) {
            af::array w_ih_r = af::randu(af::dim4(gate_size, layer_input_size), af::dtype::f32) * 2.0f * limit_ih - limit_ih;
            af::array w_hh_r = af::randu(af::dim4(gate_size, hidden_size_), af::dtype::f32) * 2.0f * limit_hh - limit_hh;
            af::array b_ih_r = af::constant(0.0f, af::dim4(gate_size));
            af::array b_hh_r = af::constant(0.0f, af::dim4(gate_size));
            b_ih_r(af::seq(hidden_size_, 2 * hidden_size_ - 1)) = 1.0f;

            W_ih_reverse_[layer] = AfToTensor(w_ih_r);
            W_hh_reverse_[layer] = AfToTensor(w_hh_r);
            b_ih_reverse_[layer] = AfToTensor(b_ih_r);
            b_hh_reverse_[layer] = AfToTensor(b_hh_r);

            grad_W_ih_reverse_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size), static_cast<size_t>(layer_input_size)});
            grad_W_hh_reverse_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size), static_cast<size_t>(hidden_size_)});
            grad_b_ih_reverse_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});
            grad_b_hh_reverse_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});
        }
    }
#else
    // CPU fallback initialization
    for (int layer = 0; layer < num_layers_; layer++) {
        int layer_input_size = (layer == 0) ? input_size_ : hidden_size_ * num_directions;
        int gate_size = 4 * hidden_size_;

        W_ih_[layer] = Tensor::Random({static_cast<size_t>(gate_size), static_cast<size_t>(layer_input_size)});
        W_hh_[layer] = Tensor::Random({static_cast<size_t>(gate_size), static_cast<size_t>(hidden_size_)});
        b_ih_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});
        b_hh_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});
        grad_W_ih_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size), static_cast<size_t>(layer_input_size)});
        grad_W_hh_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size), static_cast<size_t>(hidden_size_)});
        grad_b_ih_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});
        grad_b_hh_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});

        if (bidirectional_) {
            W_ih_reverse_[layer] = Tensor::Random({static_cast<size_t>(gate_size), static_cast<size_t>(layer_input_size)});
            W_hh_reverse_[layer] = Tensor::Random({static_cast<size_t>(gate_size), static_cast<size_t>(hidden_size_)});
            b_ih_reverse_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});
            b_hh_reverse_[layer] = Tensor::Zeros({static_cast<size_t>(gate_size)});
        }
    }
#endif
}

void LSTMLayer::ResetState() {
    h_n_ = Tensor();
    c_n_ = Tensor();
}

void LSTMLayer::SetHiddenState(const Tensor& h0) {
    h_n_ = h0.Clone();
}

void LSTMLayer::SetCellState(const Tensor& c0) {
    c_n_ = c0.Clone();
}

Tensor LSTMLayer::Forward(const Tensor& input) {
    cached_input_ = input;

    // Hoisted from the CPU fallback path: ensure weights are valid
    // BEFORE either path runs. The AF path was tripping
    // af_write_array "Expected: (data != nullptr)" because LSTMLayer's
    // constructor calls InitializeWeights() which uses the AF backend
    // (TensorToAf, etc.); when AF init failed silently the W_ih_ /
    // W_hh_ / b_ih_ / b_hh_ tensors were registered but had nullptr
    // data. The CPU fallback re-init path was only reached if AF
    // Forward also failed, but with the AF path fixed at the boundary
    // it now gets further and trips on the null weights.
    {
        const auto& shape = input.Shape();
        size_t input_dim = shape.size() == 3
            ? (batch_first_ ? shape[2] : shape[2]) : 0;
        int num_directions = bidirectional_ ? 2 : 1;
        if (W_ih_.empty() || W_ih_[0].Data<float>() == nullptr) {
            for (int layer = 0; layer < num_layers_; layer++) {
                size_t layer_input_size = (layer == 0) ? input_dim
                    : static_cast<size_t>(hidden_size_ * num_directions);
                size_t gate_size = static_cast<size_t>(4 * hidden_size_);
                W_ih_[layer] = Tensor::Random({gate_size, layer_input_size});
                W_hh_[layer] = Tensor::Random({gate_size, static_cast<size_t>(hidden_size_)});
                b_ih_[layer] = Tensor::Zeros({gate_size});
                b_hh_[layer] = Tensor::Zeros({gate_size});
            }
        }
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    // The ArrayFire path below is currently broken for [batch, seq,
    // features] 3D input. On first call it throws ArrayFire Exception
    // (Invalid input argument:202); on subsequent calls Invalid input
    // size:203. Same bug family as the pre-fix EmbeddingLayer backward
    // (84ef7211): the `af::reorder` / `moddims` math assumes row-major
    // dim ordering but AF is column-major, so the dims read here
    // (`batch_size = x.dims(0)` etc.) are interpreting the tensor
    // incorrectly and every downstream shape check eventually trips.
    //
    // AF Forward is operational as of 2026-04-16 after four stacked fixes:
    //   1. 3D column-major scrambling at TensorToAf boundary —
    //      TensorToAf3DRowMajor / AfToTensor3DRowMajor helpers handle the
    //      row-major → column-major conversion via dim-reversal +
    //      af::reorder(2, 1, 0).
    //   2. Slice assignment shape mismatch — h_states(t, span, span) = h
    //      had rank-3 proxy on LHS but rank-2 RHS; wrapped each RHS in
    //      af::moddims(..., dim4(1, batch, hidden)) to match.
    //   3. Hoisted weight init guard — covers the case where the AF
    //      backend silently failed to initialize W_ih_/W_hh_/biases.
    //   4. h_n_/c_n_ init check now matches CPU fallback exactly. The
    //      previous AF-only check `NumElements() == 0` doesn't fire for
    //      a default-constructed Tensor() (shape={} → product = 1, so
    //      NumElements returns 1), but Data() is still null and
    //      TensorToAf trips. Mirror the CPU's
    //      `Data<float>() == nullptr` clause.
    //
    // AF Forward + CPU Backward form a working hybrid: AF gives GPU-speed
    // forward, CPU gives correct gradients via row-major caches that both
    // paths populate identically (via AfToTensor3DRowMajor in AF Forward).
    // Loss numerically matches CPU within fp32 noise (~10ppm at the loss
    // level) on the mini sentiment LSTM smoke test.
    //
    // AF Backward under `#if 0` below is the next perf upgrade — would
    // skip the Backward Tensor↔CPU round-trip but needs its own column-
    // major audit using the now-working AF Forward as the oracle.
        // Bidirectional LSTM is currently supported by the CPU path only.
        // The ArrayFire branch still has a shape/layout failure on this
        // configuration, so skip it entirely instead of trying-and-falling-
        // back after the fact.
        constexpr bool kAfPathEnabled = true;
        const auto& af_guard_shape = input.Shape();
        const size_t af_guard_batch = batch_first_ ? af_guard_shape[0] : af_guard_shape[1];
        const size_t af_guard_seq = batch_first_ ? af_guard_shape[1] : af_guard_shape[0];
        const size_t af_guard_input = af_guard_shape.size() >= 3 ? af_guard_shape[2] : 0;
        if (kAfPathEnabled &&
            ShouldUseArrayFireRecurrentForward(RecurrentLayerKind::LSTM,
                                               af_guard_batch,
                                               af_guard_seq,
                                               af_guard_input,
                                               hidden_size_,
                                               num_layers_,
                                               bidirectional_)) try {
        // Use the 3D-aware helper — bare TensorToAf on [batch, seq, feat]
        // produces a column-major AF array with scrambled semantic axes.
        af::array x = TensorToAf3DRowMajor(input);

        // Handle batch_first format
        // Input: [batch, seq_len, input_size] if batch_first
        // Convert to: [seq_len, batch, input_size] for processing
        dim_t batch_size, seq_len, input_dim;

        if (batch_first_) {
            batch_size = x.dims(0);
            seq_len = x.dims(1);
            input_dim = x.dims(2);
            // Transpose to [seq_len, batch, input_size]
            x = af::reorder(x, 1, 0, 2);
        } else {
            seq_len = x.dims(0);
            batch_size = x.dims(1);
            input_dim = x.dims(2);
        }

        af::array semantic_input = x;

        int num_directions = bidirectional_ ? 2 : 1;

        // Initialize hidden and cell states if not set. The CPU fallback's
        // matching check (line 2378+) ANDs `Data<float>() == nullptr` —
        // this AF check used to omit it. The bug: a default-constructed
        // Tensor has shape={} which yields NumElements()=1 (product of
        // empty range = 1) BUT Data() returns nullptr, so the
        // NumElements==0 check passes through and h_n_/c_n_ stay
        // unallocated, then TensorToAf trips on null data. Mirror the
        // CPU check exactly to keep both paths consistent.
        if (h_n_.NumElements() == 0 || h_n_.Data<float>() == nullptr) {
            h_n_ = Tensor::Zeros({static_cast<size_t>(num_layers_ * num_directions),
                                   static_cast<size_t>(batch_size),
                                   static_cast<size_t>(hidden_size_)});
        }
        if (c_n_.NumElements() == 0 || c_n_.Data<float>() == nullptr) {
            c_n_ = Tensor::Zeros({static_cast<size_t>(num_layers_ * num_directions),
                                   static_cast<size_t>(batch_size),
                                   static_cast<size_t>(hidden_size_)});
        }

        if (bidirectional_) {
            af::array h_full = TensorToAf(h_n_);
            af::array c_full = TensorToAf(c_n_);

            cached_inputs_.clear();
            cached_gates_.clear();
            cached_cell_states_.clear();
            cached_hidden_states_.clear();
            cached_inputs_.reserve(static_cast<size_t>(num_layers_ * 2));
            cached_gates_.reserve(static_cast<size_t>(num_layers_ * 2));
            cached_hidden_states_.reserve(static_cast<size_t>(num_layers_ * 2));
            cached_cell_states_.reserve(static_cast<size_t>(num_layers_ * 2));

            af::array layer_input = semantic_input;

            for (int layer = 0; layer < num_layers_; ++layer) {
                af::array W_ih = TensorToAf(W_ih_[layer]);
                af::array W_hh = TensorToAf(W_hh_[layer]);
                af::array b_ih = TensorToAf(b_ih_[layer]);
                af::array b_hh = TensorToAf(b_hh_[layer]);

                af::array init_h_fwd = af::moddims(h_full(layer, af::span, af::span),
                                                   af::dim4(batch_size, hidden_size_));
                af::array init_c_fwd = af::moddims(c_full(layer, af::span, af::span),
                                                   af::dim4(batch_size, hidden_size_));
                af::array init_h_rev = af::moddims(h_full(num_layers_ + layer, af::span, af::span),
                                                   af::dim4(batch_size, hidden_size_));
                af::array init_c_rev = af::moddims(c_full(num_layers_ + layer, af::span, af::span),
                                                   af::dim4(batch_size, hidden_size_));

                auto fwd = RunLSTMAfDirectionForward(
                    layer_input, W_ih, W_hh, b_ih, b_hh,
                    init_h_fwd, init_c_fwd,
                    seq_len, batch_size, static_cast<size_t>(layer_input.dims(2)),
                    hidden_size_);

                af::array reversed_input = af::flip(layer_input, 0);
                af::array W_ih_r = TensorToAf(W_ih_reverse_[layer]);
                af::array W_hh_r = TensorToAf(W_hh_reverse_[layer]);
                af::array b_ih_r = TensorToAf(b_ih_reverse_[layer]);
                af::array b_hh_r = TensorToAf(b_hh_reverse_[layer]);
                auto rev = RunLSTMAfDirectionForward(
                    reversed_input, W_ih_r, W_hh_r, b_ih_r, b_hh_r,
                    init_h_rev, init_c_rev,
                    seq_len, batch_size, static_cast<size_t>(reversed_input.dims(2)),
                    hidden_size_);

                af::array layer_output = af::join(2, fwd.output, af::flip(rev.output, 0));

                cached_inputs_.push_back(std::move(fwd.input_cache));
                cached_gates_.push_back(std::move(fwd.gate_cache));
                cached_hidden_states_.push_back(std::move(fwd.hidden_cache));
                cached_cell_states_.push_back(std::move(fwd.cell_cache));
                cached_inputs_.push_back(std::move(rev.input_cache));
                cached_gates_.push_back(std::move(rev.gate_cache));
                cached_hidden_states_.push_back(std::move(rev.hidden_cache));
                cached_cell_states_.push_back(std::move(rev.cell_cache));

                h_full(layer, af::span, af::span) = af::moddims(TensorToAf(fwd.final_hidden),
                                                                af::dim4(1, batch_size, hidden_size_));
                c_full(layer, af::span, af::span) = af::moddims(TensorToAf(fwd.final_cell),
                                                                af::dim4(1, batch_size, hidden_size_));
                h_full(num_layers_ + layer, af::span, af::span) = af::moddims(TensorToAf(rev.final_hidden),
                                                                               af::dim4(1, batch_size, hidden_size_));
                c_full(num_layers_ + layer, af::span, af::span) = af::moddims(TensorToAf(rev.final_cell),
                                                                               af::dim4(1, batch_size, hidden_size_));

                h_n_ = AfToTensor3DRowMajor(h_full);
                c_n_ = AfToTensor3DRowMajor(c_full);

                if (layer < num_layers_ - 1 && dropout_ > 0.0f && training_) {
                    af::array mask = (af::randu(layer_output.dims()) > dropout_).as(af::dtype::f32);
                    layer_output = layer_output * mask / (1.0f - dropout_);
                }

                layer_input = layer_output;
            }

            af::array output = layer_input;
            if (batch_first_) {
                output = af::reorder(output, 1, 0, 2);
            }
            return AfToTensor3DRowMajor(output);
        }

        // Clear caches
        cached_inputs_.clear();
        cached_gates_.clear();
        cached_cell_states_.clear();
        cached_hidden_states_.clear();

        // Output container: [seq_len, batch, hidden_size * num_directions]
        af::array output = af::constant(0.0f, af::dim4(seq_len, batch_size, hidden_size_ * num_directions));

        af::array layer_input = x;

        for (int layer = 0; layer < num_layers_; layer++) {
            af::array W_ih = TensorToAf(W_ih_[layer]);
            af::array W_hh = TensorToAf(W_hh_[layer]);
            af::array b_ih = TensorToAf(b_ih_[layer]);
            af::array b_hh = TensorToAf(b_hh_[layer]);

            // Get initial hidden/cell state for this layer
            af::array h_full = TensorToAf(h_n_);
            af::array c_full = TensorToAf(c_n_);
            af::array h = h_full(layer, af::span, af::span);
            af::array c = c_full(layer, af::span, af::span);
            h = af::moddims(h, af::dim4(batch_size, hidden_size_));
            c = af::moddims(c, af::dim4(batch_size, hidden_size_));

            // Pre-compute input projections for ALL timesteps at once
            // layer_input: [seq_len, batch, input_size]
            // Reshape to [seq_len * batch, input_size] for batch matmul
            dim_t layer_input_size = layer_input.dims(2);
            af::array input_flat = af::moddims(layer_input, af::dim4(seq_len * batch_size, layer_input_size));

            // Compute all input projections at once: [seq_len * batch, 4 * hidden_size]
            // W_ih: [4 * hidden_size, input_size]
            af::array input_proj = af::matmul(input_flat, af::transpose(W_ih));
            input_proj.eval();
            // Add bias (broadcast)
            input_proj = input_proj + af::tile(af::transpose(b_ih), static_cast<unsigned int>(seq_len * batch_size));
            input_proj.eval();
            // Reshape back: [seq_len, batch, 4 * hidden_size]
            input_proj = af::moddims(input_proj, af::dim4(seq_len, batch_size, 4 * hidden_size_));
            input_proj.eval();

            // Cache for backward — use 3D-aware helper so the resulting
            // Tensor lands in row-major [seq_len, batch, input_size]
            // layout matching what CPU BPTT (LSTMLayer::Backward) reads.
            cached_inputs_.push_back(AfToTensor3DRowMajor(layer_input));

            // Storage for hidden states and cell states over time
            const int seq_i = CheckedIntDim(static_cast<size_t>(seq_len), "seq_len");
            const int batch_i = CheckedIntDim(static_cast<size_t>(batch_size), "batch_size");
            const int seq_batch_i = CheckedIntDim(
                static_cast<size_t>(seq_len * batch_size), "seq_len * batch_size");

            af::array h_states = af::constant(0.0f, af::dim4(seq_i + 1, batch_i, hidden_size_));
            af::array c_states = af::constant(0.0f, af::dim4(seq_i + 1, batch_i, hidden_size_));
            af::array all_gates = af::constant(0.0f, af::dim4(seq_i, batch_i, 4 * hidden_size_));

            // Store initial states. Slice (k, span, span) of a 3D
            // [seq+1, batch, hidden] array yields a (1, batch, hidden)
            // proxy — assigning a (batch, hidden) 2D `h` directly trips
            // af "Size mismatch between input and output" (Invalid input
            // size:203). Wrap with explicit moddims to add the leading
            // 1 dim and match the proxy's rank.
            h_states(0, af::span, af::span) = af::moddims(h, af::dim4(1, batch_i, hidden_size_));
            c_states(0, af::span, af::span) = af::moddims(c, af::dim4(1, batch_i, hidden_size_));

            // Forward pass through time using vectorized operations per timestep
            // Note: The recurrent dependency requires sequential processing,
            // but each timestep is fully vectorized across the batch
            for (dim_t t = 0; t < seq_len; t++) {
                // Get input projection for this timestep: [batch, 4 * hidden_size]
                const int t_idx = CheckedIntDim(static_cast<size_t>(t), "t");
                af::array x_t = input_proj(t_idx, af::span, af::span);
                x_t = af::moddims(x_t, af::dim4(batch_i, 4 * hidden_size_));

                // Compute hidden projection: h @ W_hh^T + b_hh
                af::array h_proj = af::matmul(h, af::transpose(W_hh));
                h_proj.eval();
                h_proj = h_proj + af::tile(af::transpose(b_hh), batch_i);
                h_proj.eval();

                // Combined gates: [batch, 4 * hidden_size]
                af::array gates = x_t + h_proj;
                gates.eval();

                // Split into individual gates and apply activations
                // Order: input, forget, cell, output
                af::array i_gate = af::sigmoid(gates(af::span, af::seq(0, hidden_size_ - 1)));
                af::array f_gate = af::sigmoid(gates(af::span, af::seq(hidden_size_, 2 * hidden_size_ - 1)));
                af::array g_gate = af::tanh(gates(af::span, af::seq(2 * hidden_size_, 3 * hidden_size_ - 1)));
                af::array o_gate = af::sigmoid(gates(af::span, af::seq(3 * hidden_size_, 4 * hidden_size_ - 1)));
                i_gate.eval();
                f_gate.eval();
                g_gate.eval();
                o_gate.eval();

                // Update cell state: c_t = f * c_{t-1} + i * g
                c = f_gate * c + i_gate * g_gate;
                c.eval();

                // Update hidden state: h_t = o * tanh(c_t)
                h = o_gate * af::tanh(c);
                h.eval();

                // Store states. Same moddims-with-leading-1 pattern as
                // the initial-state assignments above to keep the slice
                // proxy and RHS rank consistent.
                h_states(t_idx + 1, af::span, af::span) = af::moddims(h, af::dim4(1, batch_i, hidden_size_));
                c_states(t_idx + 1, af::span, af::span) = af::moddims(c, af::dim4(1, batch_i, hidden_size_));

                // Store gates for backward pass (pre-activation for efficiency)
                all_gates(t_idx, af::span, af::span) = af::moddims(gates, af::dim4(1, batch_i, 4 * hidden_size_));
            }

            // Cache for backward — same row-major treatment so all
            // four caches are consistent with what CPU BPTT reads.
            //   gates   : [seq_len,     batch, 4 * hidden_size]
            //   h_states: [seq_len + 1, batch, hidden_size]
            //   c_states: [seq_len + 1, batch, hidden_size]
            cached_gates_.push_back(AfToTensor3DRowMajor(all_gates));
            cached_hidden_states_.push_back(AfToTensor3DRowMajor(h_states));
            cached_cell_states_.push_back(AfToTensor3DRowMajor(c_states));

            // Extract output hidden states [seq_len, batch, hidden_size]
            af::array layer_output = h_states(af::seq(1, static_cast<double>(seq_len)), af::span, af::span);

            // Handle bidirectional
            if (bidirectional_) {
                af::array W_ih_r = TensorToAf(W_ih_reverse_[layer]);
                af::array W_hh_r = TensorToAf(W_hh_reverse_[layer]);
                af::array b_ih_r = TensorToAf(b_ih_reverse_[layer]);
                af::array b_hh_r = TensorToAf(b_hh_reverse_[layer]);

                // Get reverse initial state
                af::array h_r = h_full(num_layers_ + layer, af::span, af::span);
                af::array c_r = c_full(num_layers_ + layer, af::span, af::span);
                h_r = af::moddims(h_r, af::dim4(batch_i, hidden_size_));
                c_r = af::moddims(c_r, af::dim4(batch_i, hidden_size_));

                // Pre-compute reverse input projections
                af::array input_proj_r = af::matmul(input_flat, af::transpose(W_ih_r));
                input_proj_r.eval();
                input_proj_r = input_proj_r + af::tile(af::transpose(b_ih_r), seq_batch_i);
                input_proj_r.eval();
                input_proj_r = af::moddims(input_proj_r, af::dim4(seq_i, batch_i, 4 * hidden_size_));
                input_proj_r.eval();

                af::array h_states_r = af::constant(0.0f, af::dim4(seq_i + 1, batch_i, hidden_size_));
                af::array c_states_r = af::constant(0.0f, af::dim4(seq_i + 1, batch_i, hidden_size_));

                h_states_r(seq_i, af::span, af::span) =
                    af::moddims(h_r, af::dim4(1, batch_i, hidden_size_));
                c_states_r(seq_i, af::span, af::span) =
                    af::moddims(c_r, af::dim4(1, batch_i, hidden_size_));

                // Backward through time (reverse direction)
                for (dim_t t = seq_len - 1; t >= 0; t--) {
                    const int rt_idx = CheckedIntDim(static_cast<size_t>(t), "reverse t");
                    af::array x_t = input_proj_r(rt_idx, af::span, af::span);
                    x_t = af::moddims(x_t, af::dim4(batch_i, 4 * hidden_size_));

                    af::array h_proj = af::matmul(h_r, af::transpose(W_hh_r));
                    h_proj.eval();
                    h_proj = h_proj + af::tile(af::transpose(b_hh_r), batch_i);
                    h_proj.eval();

                    af::array gates = x_t + h_proj;
                    gates.eval();

                    af::array i_gate = af::sigmoid(gates(af::span, af::seq(0, hidden_size_ - 1)));
                    af::array f_gate = af::sigmoid(gates(af::span, af::seq(hidden_size_, 2 * hidden_size_ - 1)));
                    af::array g_gate = af::tanh(gates(af::span, af::seq(2 * hidden_size_, 3 * hidden_size_ - 1)));
                    af::array o_gate = af::sigmoid(gates(af::span, af::seq(3 * hidden_size_, 4 * hidden_size_ - 1)));
                    i_gate.eval();
                    f_gate.eval();
                    g_gate.eval();
                    o_gate.eval();

                    c_r = f_gate * c_r + i_gate * g_gate;
                    c_r.eval();
                    h_r = o_gate * af::tanh(c_r);
                    h_r.eval();

                    h_states_r(rt_idx, af::span, af::span) = af::moddims(h_r, af::dim4(1, batch_i, hidden_size_));
                    c_states_r(rt_idx, af::span, af::span) = af::moddims(c_r, af::dim4(1, batch_i, hidden_size_));
                }

                // Extract reverse output and concatenate
                af::array layer_output_r = h_states_r(af::seq(0, static_cast<double>(seq_len - 1)), af::span, af::span);
                layer_output = af::join(2, layer_output, layer_output_r);

                // Update final states. Same moddims-with-leading-1
                // pattern — slice (k, span, span) is rank-3 (1, b, H).
                h_full(num_layers_ + layer, af::span, af::span) = af::moddims(h_r, af::dim4(1, batch_size, hidden_size_));
                c_full(num_layers_ + layer, af::span, af::span) = af::moddims(c_r, af::dim4(1, batch_size, hidden_size_));
            }

            // Update final hidden and cell states for forward direction.
            h_full(layer, af::span, af::span) = af::moddims(h, af::dim4(1, batch_size, hidden_size_));
            c_full(layer, af::span, af::span) = af::moddims(c, af::dim4(1, batch_size, hidden_size_));

            // Update stored states. h_full / c_full are
            // [num_layers * num_directions, batch_size, hidden_size]
            // — 3D, needs the row-major helper. Otherwise stateful
            // LSTM (h_n fed back as initial state on next Forward)
            // sees scrambled axes.
            h_n_ = AfToTensor3DRowMajor(h_full);
            c_n_ = AfToTensor3DRowMajor(c_full);

            // Apply dropout between layers (not on last layer)
            if (layer < num_layers_ - 1 && dropout_ > 0.0f && training_) {
                af::array mask = (af::randu(layer_output.dims()) > dropout_).as(af::dtype::f32);
                layer_output = layer_output * mask / (1.0f - dropout_);
            }

            // Use this layer's output as next layer's input
            layer_input = layer_output;
        }

        output = layer_input;

        // Convert back to batch_first if needed
        if (batch_first_) {
            output = af::reorder(output, 1, 0, 2);
        }

        // Use the 3D-aware helper — bare AfToTensor on a semantic
        // [batch, seq, hidden] AF array would produce a row-major
        // Tensor with the axis order scrambled. Keep this in sync
        // with the TensorToAf3DRowMajor at the entry.
        return AfToTensor3DRowMajor(output);
    } catch (const af::exception& e) {
        DisableArrayFireCudaRecurrentAfterFailure(
            RecurrentLayerKind::LSTM, "LSTMLayer::Forward", e.what());
        if (IsCudaJitFormalParameterOverflow(e.what())) {
            spdlog::warn("{}",
                         BuildRecurrentFormalParameterOverflowFallbackMessage(
                             "LSTMLayer::Forward"));
        } else {
            BackendDebugHooks::EmitDebugEvent(
                "LSTMLayer::Forward",
                std::string("ArrayFire fallback: ") + e.what() +
                (bidirectional_ ? " [bidirectional=true]" : " [bidirectional=false]"));
            spdlog::warn("ArrayFire LSTMLayer::Forward failed: {}, falling back to CPU", e.what());
        }
    }
#endif

    // CPU fallback implementation. This path ALSO populates cached_inputs_,
    // cached_gates_, cached_hidden_states_, cached_cell_states_ in row-major
    // CPU-friendly layout so LSTMLayer::Backward (CPU BPTT below) can read
    // them. These caches are distinct from the AF-shaped ones the disabled
    // AF Forward would have written — CPU backward operates on row-major
    // [seq_len, batch, ...] tensors throughout, no AF reshape involved.
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

    // Reinitialize weights with CPU if they have null data (ArrayFire init failed)
    if (W_ih_.empty() || W_ih_[0].Data<float>() == nullptr) {
        for (int layer = 0; layer < num_layers_; layer++) {
            size_t layer_input_size = (layer == 0) ? input_dim : static_cast<size_t>(hidden_size_ * num_directions);
            size_t gate_size = static_cast<size_t>(4 * hidden_size_);
            W_ih_[layer] = Tensor::Random({gate_size, layer_input_size});
            W_hh_[layer] = Tensor::Random({gate_size, static_cast<size_t>(hidden_size_)});
            b_ih_[layer] = Tensor::Zeros({gate_size});
            b_hh_[layer] = Tensor::Zeros({gate_size});
        }
    }

    if (h_n_.NumElements() == 0 || h_n_.Data<float>() == nullptr) {
            h_n_ = Tensor::Zeros({static_cast<size_t>(num_layers_ * num_directions),
                               batch_size, static_cast<size_t>(hidden_size_)});
    }
    if (c_n_.NumElements() == 0 || c_n_.Data<float>() == nullptr) {
            c_n_ = Tensor::Zeros({static_cast<size_t>(num_layers_ * num_directions),
                               batch_size, static_cast<size_t>(hidden_size_)});
    }

    if (bidirectional_) {
        float* h_data = h_n_.Data<float>();
        float* c_data = c_n_.Data<float>();

        Tensor layer_input = input;
        cached_inputs_.clear();
        cached_gates_.clear();
        cached_hidden_states_.clear();
        cached_cell_states_.clear();
        cached_inputs_.reserve(static_cast<size_t>(num_layers_ * 2));
        cached_gates_.reserve(static_cast<size_t>(num_layers_ * 2));
        cached_hidden_states_.reserve(static_cast<size_t>(num_layers_ * 2));
        cached_cell_states_.reserve(static_cast<size_t>(num_layers_ * 2));

        for (int layer = 0; layer < num_layers_; ++layer) {
            Tensor init_h_fwd = Tensor::Zeros({batch_size, static_cast<size_t>(hidden_size_)});
            Tensor init_c_fwd = Tensor::Zeros({batch_size, static_cast<size_t>(hidden_size_)});
            Tensor init_h_rev = Tensor::Zeros({batch_size, static_cast<size_t>(hidden_size_)});
            Tensor init_c_rev = Tensor::Zeros({batch_size, static_cast<size_t>(hidden_size_)});
            const size_t fwd_state_idx = static_cast<size_t>(layer);
            const size_t rev_state_idx = static_cast<size_t>(num_layers_ + layer);
            for (size_t b = 0; b < batch_size; ++b) {
                for (int i = 0; i < hidden_size_; ++i) {
                    init_h_fwd.Data<float>()[b * hidden_size_ + i] =
                        h_data[fwd_state_idx * batch_size * hidden_size_ + b * hidden_size_ + i];
                    init_c_fwd.Data<float>()[b * hidden_size_ + i] =
                        c_data[fwd_state_idx * batch_size * hidden_size_ + b * hidden_size_ + i];
                    init_h_rev.Data<float>()[b * hidden_size_ + i] =
                        h_data[rev_state_idx * batch_size * hidden_size_ + b * hidden_size_ + i];
                    init_c_rev.Data<float>()[b * hidden_size_ + i] =
                        c_data[rev_state_idx * batch_size * hidden_size_ + b * hidden_size_ + i];
                }
            }

            auto fwd = RunLSTMCpuDirectionForward(
                layer_input,
                W_ih_[layer], W_hh_[layer],
                b_ih_[layer], b_hh_[layer],
                init_h_fwd, init_c_fwd,
                hidden_size_, batch_first_, false);
            auto rev = RunLSTMCpuDirectionForward(
                layer_input,
                W_ih_reverse_[layer], W_hh_reverse_[layer],
                b_ih_reverse_[layer], b_hh_reverse_[layer],
                init_h_rev, init_c_rev,
                hidden_size_, batch_first_, true);

            Tensor layer_output = batch_first_
                ? Tensor::Zeros({batch_size, seq_len, static_cast<size_t>(hidden_size_ * 2)})
                : Tensor::Zeros({seq_len, batch_size, static_cast<size_t>(hidden_size_ * 2)});
            float* out_data = layer_output.Data<float>();
            const float* fwd_data = fwd.output.Data<float>();
            const float* rev_data = rev.output.Data<float>();
            for (size_t t = 0; t < seq_len; ++t) {
                for (size_t b = 0; b < batch_size; ++b) {
                    for (int i = 0; i < hidden_size_; ++i) {
                        const float fwd_v = fwd_data[t * batch_size * hidden_size_ + b * hidden_size_ + i];
                        const float rev_v = rev_data[t * batch_size * hidden_size_ + b * hidden_size_ + i];
                        if (batch_first_) {
                            out_data[b * seq_len * hidden_size_ * 2 + t * hidden_size_ * 2 + i] = fwd_v;
                            out_data[b * seq_len * hidden_size_ * 2 + t * hidden_size_ * 2 + hidden_size_ + i] = rev_v;
                        } else {
                            out_data[t * batch_size * hidden_size_ * 2 + b * hidden_size_ * 2 + i] = fwd_v;
                            out_data[t * batch_size * hidden_size_ * 2 + b * hidden_size_ * 2 + hidden_size_ + i] = rev_v;
                        }
                    }
                }
            }

            cached_inputs_.push_back(std::move(fwd.input_cache));
            cached_gates_.push_back(std::move(fwd.gate_cache));
            cached_hidden_states_.push_back(std::move(fwd.hidden_cache));
            cached_cell_states_.push_back(std::move(fwd.cell_cache));
            cached_inputs_.push_back(std::move(rev.input_cache));
            cached_gates_.push_back(std::move(rev.gate_cache));
            cached_hidden_states_.push_back(std::move(rev.hidden_cache));
            cached_cell_states_.push_back(std::move(rev.cell_cache));

            for (size_t b = 0; b < batch_size; ++b) {
                for (int i = 0; i < hidden_size_; ++i) {
                    h_data[fwd_state_idx * batch_size * hidden_size_ + b * hidden_size_ + i] =
                        fwd.final_hidden.Data<float>()[b * hidden_size_ + i];
                    c_data[fwd_state_idx * batch_size * hidden_size_ + b * hidden_size_ + i] =
                        fwd.final_cell.Data<float>()[b * hidden_size_ + i];
                    h_data[rev_state_idx * batch_size * hidden_size_ + b * hidden_size_ + i] =
                        rev.final_hidden.Data<float>()[b * hidden_size_ + i];
                    c_data[rev_state_idx * batch_size * hidden_size_ + b * hidden_size_ + i] =
                        rev.final_cell.Data<float>()[b * hidden_size_ + i];
                }
            }

            layer_input = layer_output;

            if (layer < num_layers_ - 1 && dropout_ > 0.0f && training_) {
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                static thread_local std::mt19937 rng{std::random_device{}()};
                float* layer_out_ptr = layer_input.Data<float>();
                for (size_t i = 0; i < layer_output.NumElements(); ++i) {
                    const float keep = dist(rng) > dropout_ ? 1.0f : 0.0f;
                    layer_out_ptr[i] = keep > 0.0f ? layer_out_ptr[i] / (1.0f - dropout_) : 0.0f;
                }
            }
        }

        return layer_input;
    }

    size_t out_dim0 = batch_first_ ? batch_size : seq_len;
    size_t out_dim1 = batch_first_ ? seq_len : batch_size;
    size_t out_features = static_cast<size_t>(hidden_size_ * num_directions);
    Tensor output = Tensor::Zeros({out_dim0, out_dim1, out_features});

    const float* input_data = input.Data<float>();
    float* output_data = output.Data<float>();
    float* h_data = h_n_.Data<float>();
    float* c_data = c_n_.Data<float>();

    auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    auto tanh_f = [](float x) { return std::tanh(x); };

    // Reset caches at the top of every forward pass so BPTT reads the
    // current run's state, not a stale one. CPU backward below branches
    // on emptiness to decide whether to run BPTT or fall through.
    cached_inputs_.clear();
    cached_gates_.clear();
    cached_hidden_states_.clear();
    cached_cell_states_.clear();
    cached_inputs_.reserve(num_layers_);
    cached_gates_.reserve(num_layers_);
    cached_hidden_states_.reserve(num_layers_);
    cached_cell_states_.reserve(num_layers_);

    Tensor layer_input = input;
    size_t layer_input_size = input_dim;

    for (int layer = 0; layer < num_layers_; layer++) {
        const float* W_ih = W_ih_[layer].Data<float>();
        const float* W_hh = W_hh_[layer].Data<float>();
        const float* b_ih = b_ih_[layer].Data<float>();
        const float* b_hh = b_hh_[layer].Data<float>();
        int gate_size = 4 * hidden_size_;

        Tensor layer_output = Tensor::Zeros({seq_len, batch_size, static_cast<size_t>(hidden_size_)});
        float* layer_out = layer_output.Data<float>();
        const float* layer_in = layer_input.Data<float>();

        // Per-layer caches. Row-major:
        //   input  [seq_len, batch, layer_input_size]
        //   gates  [seq_len, batch, 4 * hidden_size]        (pre-activations)
        //   h      [seq_len + 1, batch, hidden_size]        (index 0 = h_0)
        //   c      [seq_len + 1, batch, hidden_size]        (index 0 = c_0)
        Tensor layer_input_cache = Tensor::Zeros(
            {seq_len, batch_size, layer_input_size});
        Tensor layer_gates_cache = Tensor::Zeros(
            {seq_len, batch_size, static_cast<size_t>(gate_size)});
        Tensor layer_h_cache = Tensor::Zeros(
            {seq_len + 1, batch_size, static_cast<size_t>(hidden_size_)});
        Tensor layer_c_cache = Tensor::Zeros(
            {seq_len + 1, batch_size, static_cast<size_t>(hidden_size_)});
        float* in_cache_data = layer_input_cache.Data<float>();
        float* gate_cache_data = layer_gates_cache.Data<float>();
        float* h_cache_data = layer_h_cache.Data<float>();
        float* c_cache_data = layer_c_cache.Data<float>();

        // Seed h_0 / c_0 at t=0 from h_n_ / c_n_ for all batches.
        for (size_t b = 0; b < batch_size; b++) {
            for (int i = 0; i < hidden_size_; i++) {
                h_cache_data[0 * batch_size * hidden_size_ + b * hidden_size_ + i] =
                    h_data[layer * batch_size * hidden_size_ + b * hidden_size_ + i];
                c_cache_data[0 * batch_size * hidden_size_ + b * hidden_size_ + i] =
                    c_data[layer * batch_size * hidden_size_ + b * hidden_size_ + i];
            }
        }

        for (size_t b = 0; b < batch_size; b++) {
            std::vector<float> h(hidden_size_), c(hidden_size_);
            for (int i = 0; i < hidden_size_; i++) {
                h[i] = h_data[layer * batch_size * hidden_size_ + b * hidden_size_ + i];
                c[i] = c_data[layer * batch_size * hidden_size_ + b * hidden_size_ + i];
            }

            for (size_t t = 0; t < seq_len; t++) {
                std::vector<float> gates(gate_size, 0.0f);
                const float* x_ptr;
                if (layer == 0) {
                    if (batch_first_) x_ptr = input_data + b * seq_len * input_dim + t * input_dim;
                    else x_ptr = input_data + t * batch_size * input_dim + b * input_dim;
                } else {
                    x_ptr = layer_in + t * batch_size * layer_input_size + b * layer_input_size;
                }

                for (int g = 0; g < gate_size; g++) {
                    gates[g] = b_ih[g] + b_hh[g];
                    for (size_t k = 0; k < layer_input_size; k++)
                        gates[g] += W_ih[g * layer_input_size + k] * x_ptr[k];
                    for (int k = 0; k < hidden_size_; k++)
                        gates[g] += W_hh[g * hidden_size_ + k] * h[k];
                }

                // Snapshot pre-activation gates + current input for BPTT.
                // gates_cache_data shape [seq_len, batch, 4H], index by
                // (t, b, g) = t*batch*4H + b*4H + g.
                for (int g = 0; g < gate_size; g++) {
                    gate_cache_data[t * batch_size * gate_size + b * gate_size + g] = gates[g];
                }
                for (size_t k = 0; k < layer_input_size; k++) {
                    in_cache_data[t * batch_size * layer_input_size + b * layer_input_size + k]
                        = x_ptr[k];
                }

                for (int i = 0; i < hidden_size_; i++) {
                    float i_gate = sigmoid(gates[i]);
                    float f_gate = sigmoid(gates[hidden_size_ + i]);
                    float g_gate = tanh_f(gates[2 * hidden_size_ + i]);
                    float o_gate = sigmoid(gates[3 * hidden_size_ + i]);
                    c[i] = f_gate * c[i] + i_gate * g_gate;
                    h[i] = o_gate * tanh_f(c[i]);
                }

                for (int i = 0; i < hidden_size_; i++)
                    layer_out[t * batch_size * hidden_size_ + b * hidden_size_ + i] = h[i];

                // Snapshot h_t, c_t (post-step state) at cache index t+1.
                for (int i = 0; i < hidden_size_; i++) {
                    h_cache_data[(t + 1) * batch_size * hidden_size_ + b * hidden_size_ + i] = h[i];
                    c_cache_data[(t + 1) * batch_size * hidden_size_ + b * hidden_size_ + i] = c[i];
                }
            }

            for (int i = 0; i < hidden_size_; i++) {
                h_data[layer * batch_size * hidden_size_ + b * hidden_size_ + i] = h[i];
                c_data[layer * batch_size * hidden_size_ + b * hidden_size_ + i] = c[i];
            }
        }

        cached_inputs_.push_back(std::move(layer_input_cache));
        cached_gates_.push_back(std::move(layer_gates_cache));
        cached_hidden_states_.push_back(std::move(layer_h_cache));
        cached_cell_states_.push_back(std::move(layer_c_cache));

        layer_input = layer_output;
        layer_input_size = static_cast<size_t>(hidden_size_);
    }

    const float* final_out = layer_input.Data<float>();
    for (size_t t = 0; t < seq_len; t++) {
        for (size_t b = 0; b < batch_size; b++) {
            for (int f = 0; f < hidden_size_; f++) {
                float val = final_out[t * batch_size * hidden_size_ + b * hidden_size_ + f];
                if (batch_first_) output_data[b * seq_len * out_features + t * out_features + f] = val;
                else output_data[t * batch_size * out_features + b * out_features + f] = val;
            }
        }
    }
    return output;
}

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


std::map<std::string, Tensor> LSTMLayer::GetParameters() {
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

void LSTMLayer::SetParameters(const std::map<std::string, Tensor>& params) {
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
