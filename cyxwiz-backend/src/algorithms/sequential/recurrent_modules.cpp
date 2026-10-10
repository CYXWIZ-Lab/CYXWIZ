#include <cyxwiz/sequential.h>

#include "../layers/layer_recurrent_utils.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace cyxwiz {

namespace {

using recurrent_detail::ExpandLastTimeStep;
using recurrent_detail::LastTimeStep;

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
// RNNModule: a batch-first RNNLayer (bidirectional on the device inside the
// layer) plus the return_sequences=false reduction, like LSTMModule.
// ============================================================================

RNNModule::RNNModule(size_t input_size, size_t hidden_size,
                     size_t num_layers, bool return_sequences,
                     const std::string& nonlinearity, bool bidirectional)
    : layer_(std::make_unique<RNNLayer>(static_cast<int>(input_size), static_cast<int>(hidden_size),
                                        static_cast<int>(num_layers), /*batch_first=*/true,
                                        bidirectional, nonlinearity))
    , input_size_(input_size)
    , hidden_size_(hidden_size)
    , return_sequences_(return_sequences)
    , nonlinearity_(nonlinearity)
    , bidirectional_(bidirectional) {}

Tensor RNNModule::Forward(const Tensor& input) {
    return ReduceSequence(layer_->Forward(input), return_sequences_, last_full_output_shape_);
}

Tensor RNNModule::Backward(const Tensor& grad_output) {
    return layer_->Backward(
        ExpandSequenceGradient(grad_output, return_sequences_, last_full_output_shape_, "RNNModule"));
}

std::map<std::string, Tensor> RNNModule::GetParameters() {
    return ModuleParameters(*layer_, bidirectional_);
}

void RNNModule::SetParameters(const std::map<std::string, Tensor>& params) {
    SetModuleParameters(*layer_, params, bidirectional_);
}

std::map<std::string, Tensor> RNNModule::GetGradients() {
    return ModuleGradients(*layer_, bidirectional_);
}

std::string RNNModule::GetName() const {
    const int dirs = bidirectional_ ? 2 : 1;
    return std::string(bidirectional_ ? "Bi" : "") + "RNN(" + std::to_string(input_size_) + " -> " +
           std::to_string(hidden_size_ * dirs) + ", " + nonlinearity_ +
           (return_sequences_ ? ", seq" : ", last") + ")";
}

} // namespace cyxwiz
