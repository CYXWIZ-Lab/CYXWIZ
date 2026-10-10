#include <cyxwiz/sequential.h>

#include "../layers/layer_recurrent_utils.h"

#include <spdlog/spdlog.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace cyxwiz {

namespace {

using recurrent_detail::ExpandLastTimeStep;
using recurrent_detail::JoinFeatures;
using recurrent_detail::LastTimeStep;
using recurrent_detail::ReverseTime;
using recurrent_detail::SliceFeatures;

// A bidirectional layer's "layer{L}_X" / "layer{L}_X_reverse" key as the
// module's "layer{L}.forward.X" / "layer{L}.reverse.X".
std::string ToModuleKey(const std::string& layer_key) {
    static const std::string kReverse = "_reverse";
    const size_t underscore = layer_key.find('_');
    std::string name = layer_key.substr(underscore + 1);
    const bool reverse = name.size() > kReverse.size() &&
                         name.compare(name.size() - kReverse.size(), kReverse.size(), kReverse) == 0;
    if (reverse) name.resize(name.size() - kReverse.size());
    return layer_key.substr(0, underscore) + (reverse ? ".reverse." : ".forward.") + name;
}

// Inverse of ToModuleKey; empty for keys that are not "layer{L}.{branch}.X".
std::string ToLayerKey(const std::string& module_key) {
    const size_t dot1 = module_key.find('.');
    const size_t dot2 = dot1 == std::string::npos ? std::string::npos : module_key.find('.', dot1 + 1);
    if (module_key.rfind("layer", 0) != 0 || dot2 == std::string::npos || dot2 + 1 >= module_key.size()) {
        return {};
    }
    const std::string branch = module_key.substr(dot1 + 1, dot2 - dot1 - 1);
    if (branch != "forward" && branch != "reverse") return {};
    return module_key.substr(0, dot1) + "_" + module_key.substr(dot2 + 1) +
           (branch == "reverse" ? "_reverse" : "");
}

// Trainable parameters of a recurrent layer under the module's keys.
template <typename LayerT>
std::map<std::string, Tensor> ModuleParameters(LayerT& layer, bool bidirectional) {
    std::map<std::string, Tensor> params;
    for (const auto& [key, tensor] : layer.GetParameters()) {
        if (key.find("_grad_") != std::string::npos) continue;
        params[bidirectional ? ToModuleKey(key) : key] = tensor;
    }
    return params;
}

// Gradients keyed like the parameters they belong to (the optimizer pairs
// them by name): the layer's "layer{L}_grad_X" becomes "layer{L}_X".
template <typename LayerT>
std::map<std::string, Tensor> ModuleGradients(LayerT& layer, bool bidirectional) {
    std::map<std::string, Tensor> grads;
    for (const auto& [key, tensor] : layer.GetParameters()) {
        const size_t at = key.find("_grad_");
        if (at == std::string::npos) continue;
        const std::string parameter_key = key.substr(0, at + 1) + key.substr(at + 6);
        grads[bidirectional ? ToModuleKey(parameter_key) : parameter_key] = tensor;
    }
    return grads;
}

template <typename LayerT>
void SetModuleParameters(LayerT& layer, const std::map<std::string, Tensor>& params, bool bidirectional) {
    if (!bidirectional) {
        layer.SetParameters(params);
        return;
    }
    std::map<std::string, Tensor> layer_params;
    for (const auto& [key, tensor] : params) {
        const std::string layer_key = ToLayerKey(key);
        if (!layer_key.empty()) layer_params[layer_key] = tensor;
    }
    layer.SetParameters(layer_params);
}

// Keras-style return_sequences=false: the last time step of the full
// [batch, seq, features] output, sliced on the device.
Tensor ReduceSequence(const Tensor& full_output, bool return_sequences, std::vector<size_t>& full_shape) {
    full_shape = full_output.Shape();
    return return_sequences ? full_output : LastTimeStep(full_output);
}

// Backward of ReduceSequence: the last-step gradient re-expanded to the
// full sequence (zeros at every earlier step), on the device.
Tensor ExpandSequenceGradient(const Tensor& grad_output, bool return_sequences,
                              const std::vector<size_t>& full_shape, const char* module) {
    if (return_sequences) return grad_output;
    if (full_shape.size() != 3) {
        throw std::runtime_error(std::string(module) + "::Backward needs a Forward first");
    }
    return ExpandLastTimeStep(grad_output, full_shape[1]);
}

std::string NormalizeChildKey(const std::string& key) {
    std::string normalized = key;
    if (normalized.rfind("layer0_", 0) == 0) normalized.erase(0, 7);
    if (normalized.rfind("grad_", 0) == 0) normalized.erase(0, 5);
    return normalized;
}

std::string MakeBranchKey(size_t layer_idx, const std::string& branch, const std::string& normalized_key) {
    return "layer" + std::to_string(layer_idx) + "." + branch + "." + normalized_key;
}

}  // namespace

// ============================================================================
// LSTMModule / GRUModule: a batch-first LSTMLayer / GRULayer (bidirectional
// on the device inside the layer) plus the return_sequences=false reduction.
// ============================================================================

LSTMModule::LSTMModule(size_t input_size, size_t hidden_size,
                       size_t num_layers, bool bidirectional,
                       bool return_sequences)
    : layer_(std::make_unique<LSTMLayer>(static_cast<int>(input_size), static_cast<int>(hidden_size),
                                         static_cast<int>(num_layers), /*batch_first=*/true,
                                         bidirectional, /*dropout=*/0.0f))
    , input_size_(input_size)
    , hidden_size_(hidden_size)
    , bidirectional_(bidirectional)
    , return_sequences_(return_sequences) {}

Tensor LSTMModule::Forward(const Tensor& input) {
    return ReduceSequence(layer_->Forward(input), return_sequences_, last_full_output_shape_);
}

Tensor LSTMModule::Backward(const Tensor& grad_output) {
    return layer_->Backward(
        ExpandSequenceGradient(grad_output, return_sequences_, last_full_output_shape_, "LSTMModule"));
}

std::map<std::string, Tensor> LSTMModule::GetParameters() {
    return ModuleParameters(*layer_, bidirectional_);
}

void LSTMModule::SetParameters(const std::map<std::string, Tensor>& params) {
    SetModuleParameters(*layer_, params, bidirectional_);
}

std::map<std::string, Tensor> LSTMModule::GetGradients() {
    return ModuleGradients(*layer_, bidirectional_);
}

std::string LSTMModule::GetName() const {
    const int dirs = bidirectional_ ? 2 : 1;
    return std::string(bidirectional_ ? "Bi" : "") + "LSTM(" + std::to_string(input_size_) + " -> " +
           std::to_string(hidden_size_ * dirs) + (return_sequences_ ? ", seq" : ", last") + ")";
}

GRUModule::GRUModule(size_t input_size, size_t hidden_size,
                     size_t num_layers, bool bidirectional,
                     bool return_sequences)
    : layer_(std::make_unique<GRULayer>(static_cast<int>(input_size), static_cast<int>(hidden_size),
                                        static_cast<int>(num_layers), /*batch_first=*/true,
                                        bidirectional, /*dropout=*/0.0f))
    , input_size_(input_size)
    , hidden_size_(hidden_size)
    , bidirectional_(bidirectional)
    , return_sequences_(return_sequences) {}

Tensor GRUModule::Forward(const Tensor& input) {
    return ReduceSequence(layer_->Forward(input), return_sequences_, last_full_output_shape_);
}

Tensor GRUModule::Backward(const Tensor& grad_output) {
    return layer_->Backward(
        ExpandSequenceGradient(grad_output, return_sequences_, last_full_output_shape_, "GRUModule"));
}

std::map<std::string, Tensor> GRUModule::GetParameters() {
    return ModuleParameters(*layer_, bidirectional_);
}

void GRUModule::SetParameters(const std::map<std::string, Tensor>& params) {
    SetModuleParameters(*layer_, params, bidirectional_);
}

std::map<std::string, Tensor> GRUModule::GetGradients() {
    return ModuleGradients(*layer_, bidirectional_);
}

std::string GRUModule::GetName() const {
    const int dirs = bidirectional_ ? 2 : 1;
    return std::string(bidirectional_ ? "Bi" : "") + "GRU(" + std::to_string(input_size_) + " -> " +
           std::to_string(hidden_size_ * dirs) + (return_sequences_ ? ", seq" : ", last") + ")";
}

void GRUModule::SetTraining(bool training) {
    Module::SetTraining(training);
    layer_->SetTraining(training);
}


// ============================================================================
// RNNModule Implementation — mirror of LSTMModule over the vanilla RNN
// reference layer (tofix68 phase 3). Bidirectional splits each level into a
// forward and a time-reversed RNNLayer; the flip / join / slice and the
// last-step reduction run on the device.
// ============================================================================

RNNModule::RNNModule(size_t input_size, size_t hidden_size,
                     size_t num_layers, bool return_sequences,
                     const std::string& nonlinearity, bool bidirectional)
    : input_size_(input_size)
    , hidden_size_(hidden_size)
    , num_layers_(num_layers)
    , return_sequences_(return_sequences)
    , nonlinearity_(nonlinearity)
    , bidirectional_(bidirectional)
{
    if (bidirectional_) {
        split_bidirectional_path_ = true;
        forward_layers_.reserve(num_layers_);
        reverse_layers_.reserve(num_layers_);
        for (size_t layer = 0; layer < num_layers_; ++layer) {
            const int layer_input_size = (layer == 0)
                ? static_cast<int>(input_size)
                : static_cast<int>(hidden_size * 2);
            forward_layers_.push_back(std::make_unique<RNNLayer>(
                layer_input_size, static_cast<int>(hidden_size), 1,
                /*batch_first=*/true, /*bidirectional=*/false, nonlinearity));
            reverse_layers_.push_back(std::make_unique<RNNLayer>(
                layer_input_size, static_cast<int>(hidden_size), 1,
                /*batch_first=*/true, /*bidirectional=*/false, nonlinearity));
        }
        spdlog::info("[RNNModule] Using split bidirectional RNN path "
                     "({} layer pairs); each branch is placed independently.",
                     num_layers_);
    } else {
        layer_ = std::make_unique<RNNLayer>(
            static_cast<int>(input_size),
            static_cast<int>(hidden_size),
            static_cast<int>(num_layers),
            /*batch_first=*/true,
            /*bidirectional=*/false,
            nonlinearity);
    }
}

Tensor RNNModule::Forward(const Tensor& input) {
    Tensor full_output;
    if (split_bidirectional_path_) {
        Tensor layer_input = input;
        for (size_t layer = 0; layer < num_layers_; ++layer) {
            const Tensor forward_output = forward_layers_[layer]->Forward(layer_input);
            const Tensor reverse_output = ReverseTime(
                reverse_layers_[layer]->Forward(ReverseTime(layer_input, /*batch_first=*/true)),
                /*batch_first=*/true);
            layer_input = JoinFeatures(forward_output, reverse_output);
        }
        full_output = layer_input;
    } else {
        full_output = layer_->Forward(input);
    }
    return ReduceSequence(full_output, return_sequences_, last_full_output_shape_);
}

Tensor RNNModule::Backward(const Tensor& grad_output) {
    const Tensor upstream =
        ExpandSequenceGradient(grad_output, return_sequences_, last_full_output_shape_, "RNNModule");
    if (!split_bidirectional_path_) {
        return layer_->Backward(upstream);
    }
    Tensor layer_grad = upstream;
    for (int layer = static_cast<int>(num_layers_) - 1; layer >= 0; --layer) {
        const Tensor dx_forward = forward_layers_[static_cast<size_t>(layer)]->Backward(
            SliceFeatures(layer_grad, 0, hidden_size_));
        const Tensor dx_reverse = ReverseTime(
            reverse_layers_[static_cast<size_t>(layer)]->Backward(
                ReverseTime(SliceFeatures(layer_grad, hidden_size_, hidden_size_), /*batch_first=*/true)),
            /*batch_first=*/true);
        layer_grad = dx_forward + dx_reverse;
    }
    return layer_grad;
}

std::map<std::string, Tensor> RNNModule::GetParameters() {
    if (split_bidirectional_path_) {
        std::map<std::string, Tensor> params;
        for (size_t layer = 0; layer < num_layers_; ++layer) {
            for (const auto& [key, tensor] : forward_layers_[layer]->GetParameters()) {
                if (key.find("grad_") != std::string::npos) continue;
                params[MakeBranchKey(layer, "forward", NormalizeChildKey(key))] = tensor;
            }
            for (const auto& [key, tensor] : reverse_layers_[layer]->GetParameters()) {
                if (key.find("grad_") != std::string::npos) continue;
                params[MakeBranchKey(layer, "reverse", NormalizeChildKey(key))] = tensor;
            }
        }
        return params;
    }
    return layer_->GetParameters();
}

void RNNModule::SetParameters(const std::map<std::string, Tensor>& params) {
    if (split_bidirectional_path_) {
        std::vector<std::map<std::string, Tensor>> forward_params(num_layers_);
        std::vector<std::map<std::string, Tensor>> reverse_params(num_layers_);
        for (const auto& [key, tensor] : params) {
            if (key.rfind("layer", 0) != 0) continue;
            const size_t dot1 = key.find('.');
            const size_t dot2 = key.find('.', dot1 == std::string::npos ? 0 : dot1 + 1);
            if (dot1 == std::string::npos || dot2 == std::string::npos) continue;
            const size_t layer_idx = static_cast<size_t>(std::stoul(key.substr(5, dot1 - 5)));
            if (layer_idx >= num_layers_) continue;
            const std::string branch = key.substr(dot1 + 1, dot2 - dot1 - 1);
            const std::string base_key = key.substr(dot2 + 1);
            if (base_key.empty()) continue;
            const std::string child_key = "layer0_" + base_key;
            if (branch == "forward") {
                forward_params[layer_idx][child_key] = tensor;
            } else if (branch == "reverse") {
                reverse_params[layer_idx][child_key] = tensor;
            }
        }
        for (size_t layer = 0; layer < num_layers_; ++layer) {
            forward_layers_[layer]->SetParameters(forward_params[layer]);
            reverse_layers_[layer]->SetParameters(reverse_params[layer]);
        }
        return;
    }
    layer_->SetParameters(params);
}

std::map<std::string, Tensor> RNNModule::GetGradients() {
    auto build_gradient_map = [](const std::map<std::string, Tensor>& params,
                                 const std::string& prefix) {
        std::map<std::string, Tensor> grads;
        for (const auto& [key, value] : params) {
            if (key.find("grad_") == std::string::npos) continue;
            grads[prefix + NormalizeChildKey(key)] = value;
        }
        return grads;
    };
    if (split_bidirectional_path_) {
        std::map<std::string, Tensor> grads;
        for (size_t layer = 0; layer < num_layers_; ++layer) {
            auto forward_grads = build_gradient_map(forward_layers_[layer]->GetParameters(),
                                                     MakeBranchKey(layer, "forward", ""));
            auto reverse_grads = build_gradient_map(reverse_layers_[layer]->GetParameters(),
                                                     MakeBranchKey(layer, "reverse", ""));
            grads.insert(forward_grads.begin(), forward_grads.end());
            grads.insert(reverse_grads.begin(), reverse_grads.end());
        }
        return grads;
    }
    // RNNLayer writes "grad_*" keys into its parameter map and the
    // SequentialModel optimizer step reads them through GetParameters().
    return layer_->GetParameters();
}

std::string RNNModule::GetName() const {
    const int dirs = bidirectional_ ? 2 : 1;
    const std::string prefix = split_bidirectional_path_ ? "Bi" : "";
    return prefix + "RNN(" + std::to_string(input_size_) + " -> " +
           std::to_string(hidden_size_ * dirs) + ", " + nonlinearity_ +
           (return_sequences_ ? ", seq" : ", last") + ")";
}

} // namespace cyxwiz
