#pragma once

#include "cyxwiz/tensor.h"

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

// Device-side sequence plumbing shared by LSTMLayer, GRULayer and the
// recurrent modules. Every helper works on rank-3 semantic ArrayFire arrays
// and keeps the data on the device; a build without ArrayFire refuses.
namespace cyxwiz::recurrent_detail {

// Throws "<layer> runs on ArrayFire, and this build has no ArrayFire".
[[noreturn]] void ThrowWithoutArrayFire(const char* layer);

// Flips a [batch, seq, features] (batch_first) or [seq, batch, features]
// sequence along its time axis.
Tensor ReverseTime(const Tensor& sequence, bool batch_first);
// Joins two rank-3 sequences on the feature axis (first, then second).
Tensor JoinFeatures(const Tensor& first, const Tensor& second);
// Features [offset, offset + width) of a rank-3 sequence.
Tensor SliceFeatures(const Tensor& sequence, size_t offset, size_t width);
// Last time step of a batch-first [batch, seq, features] sequence.
Tensor LastTimeStep(const Tensor& sequence);
// Inverse of LastTimeStep for gradients: [batch, features] placed at the last
// of seq_len steps, zeros elsewhere.
Tensor ExpandLastTimeStep(const Tensor& last_step_gradient, size_t seq_len);
// Inverted-dropout keep mask (0 or 1 / (1 - p)) of the given shape.
Tensor MakeDropoutMask(const std::vector<size_t>& shape, float dropout);
// State `index` of a stacked [count, batch, hidden] state as [1, batch, hidden].
Tensor StateAt(const Tensor& states, size_t index, const char* what);
// Stacks [1, batch, hidden] states into [count, batch, hidden].
Tensor StackStates(const std::vector<Tensor>& states);

// Bidirectional composition: per level a forward child and a reverse child
// (single-direction, single-layer layers of the same class, same layout).
// The reverse child sees the level input flipped in time and its output is
// flipped back; the two join on the feature axis, forward first (PyTorch).
// Inter-level dropout masks are stored for Backward.
template <typename LayerT>
Tensor BidirectionalForward(const std::vector<std::unique_ptr<LayerT>>& forward_levels,
                            const std::vector<std::unique_ptr<LayerT>>& reverse_levels,
                            const Tensor& input, bool batch_first, float dropout,
                            bool training, std::vector<Tensor>& dropout_masks) {
    dropout_masks.clear();
    Tensor level_input = input;
    for (size_t level = 0; level < forward_levels.size(); ++level) {
        const Tensor forward_output = forward_levels[level]->Forward(level_input);
        const Tensor reverse_output = ReverseTime(
            reverse_levels[level]->Forward(ReverseTime(level_input, batch_first)), batch_first);
        level_input = JoinFeatures(forward_output, reverse_output);
        if (training && dropout > 0.0f && level + 1 < forward_levels.size()) {
            dropout_masks.push_back(MakeDropoutMask(level_input.Shape(), dropout));
            level_input = level_input * dropout_masks.back();
        }
    }
    return level_input;
}

template <typename LayerT>
Tensor BidirectionalBackward(const std::vector<std::unique_ptr<LayerT>>& forward_levels,
                             const std::vector<std::unique_ptr<LayerT>>& reverse_levels,
                             const Tensor& grad_output, bool batch_first, size_t hidden,
                             const std::vector<Tensor>& dropout_masks) {
    Tensor level_grad = grad_output;
    for (size_t level = forward_levels.size(); level-- > 0;) {
        if (level < dropout_masks.size()) {
            level_grad = level_grad * dropout_masks[level];
        }
        const Tensor dx_forward =
            forward_levels[level]->Backward(SliceFeatures(level_grad, 0, hidden));
        const Tensor dx_reverse = ReverseTime(
            reverse_levels[level]->Backward(
                ReverseTime(SliceFeatures(level_grad, hidden, hidden), batch_first)),
            batch_first);
        level_grad = dx_forward + dx_reverse;
    }
    return level_grad;
}

// Parameter-map plumbing for a bidirectional layer: child key "layer0_X"
// appears as "layer{level}_X" (forward) or "layer{level}_X_reverse".
template <typename LayerT>
void AppendBidirectionalParameters(std::map<std::string, Tensor>& out, LayerT& child,
                                   size_t level, bool reverse) {
    const std::string prefix = "layer" + std::to_string(level) + "_";
    for (const auto& [key, tensor] : child.GetParameters()) {
        out[prefix + key.substr(7) + (reverse ? "_reverse" : "")] = tensor;
    }
}

template <typename LayerT>
void SetBidirectionalParameters(const std::map<std::string, Tensor>& params, LayerT& child,
                                size_t level, bool reverse) {
    const std::string prefix = "layer" + std::to_string(level) + "_";
    std::map<std::string, Tensor> child_params;
    for (const char* name : {"W_ih", "W_hh", "b_ih", "b_hh"}) {
        const auto it = params.find(prefix + name + (reverse ? "_reverse" : ""));
        if (it != params.end()) child_params[std::string("layer0_") + name] = it->second;
    }
    child.SetParameters(child_params);
}

}  // namespace cyxwiz::recurrent_detail
