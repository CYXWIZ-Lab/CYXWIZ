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
#include <cyxwiz/error_codes.h>

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
        const size_t input_dim = input_shape[2];
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
        const size_t af_guard_batch = batch_first_ ? input_shape[0] : input_shape[1];
        const size_t af_guard_seq = batch_first_ ? input_shape[1] : input_shape[0];
        const size_t af_guard_input = input_shape[2];
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
        if (batch_first_) {
            // Transpose to [seq_len, batch, input_size]
            x = af::reorder(x, 1, 0, 2);
        }
        const dim_t seq_len = x.dims(0);
        const dim_t batch_size = x.dims(1);

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
        const auto fallback_reason =
            ClassifyArrayFireBackendFallbackReason(e.what());
        const std::string fallback_context =
            BuildArrayFireBackendFallbackContext(
                BuildTensorShapeContext("input", input.Shape()) +
                "; hidden_size=" + std::to_string(hidden_size_) +
                "; layers=" + std::to_string(num_layers_) +
                "; bidirectional=" +
                std::string(bidirectional_ ? "true" : "false") +
                "; batch_first=" +
                std::string(batch_first_ ? "true" : "false"));
        DisableArrayFireCudaRecurrentAfterFailure(
            RecurrentLayerKind::LSTM,
            "LSTMLayer::Forward",
            af_guard_batch,
            af_guard_seq,
            af_guard_input,
            hidden_size_,
            num_layers_,
            bidirectional_,
            e.what());
        if (fallback_reason == BackendFallbackReason::CudaJitParamOverflow) {
            if (ShouldLogArrayFireBackendFallbackOnce(
                    "LSTMLayer::Forward", fallback_reason,
                    fallback_context)) {
                std::string fallback_message =
                    BuildRecurrentFormalParameterOverflowFallbackMessage(
                        "LSTMLayer::Forward");
                fallback_message += " Context: ";
                fallback_message += fallback_context;
                fallback_message += ".";
                BackendDebugHooks::EmitDebugEvent(
                    "LSTMLayer::Forward",
                    errors::FormatWarning(
                        errors::Gpu::PathDisabledByPolicy,
                        fallback_message) +
                    (bidirectional_ ? " [bidirectional=true]" : " [bidirectional=false]"));
                spdlog::warn("{}",
                             errors::FormatWarning(
                                 errors::Gpu::PathDisabledByPolicy,
                                 fallback_message));
            }
        } else {
            const bool log_fallback =
                ShouldLogArrayFireBackendFallbackOnce(
                    "LSTMLayer::Forward", fallback_reason,
                    fallback_context);
            const std::string fallback_message =
                BuildArrayFireBackendFallbackMessage(
                    "LSTMLayer::Forward", fallback_reason,
                    log_fallback, e.what(), fallback_context);
            if (log_fallback) {
                BackendDebugHooks::EmitDebugEvent(
                    "LSTMLayer::Forward",
                    errors::FormatWarning(
                        errors::Gpu::KernelExecutionFailed,
                        fallback_message) +
                    (bidirectional_ ? " [bidirectional=true]" : " [bidirectional=false]"));
                spdlog::warn("{}",
                             errors::FormatWarning(
                                 errors::Gpu::KernelExecutionFailed,
                                 fallback_message));
            }
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

    // Reinitialize weights with CPU if ArrayFire initialization produced null data.
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

} // namespace cyxwiz
