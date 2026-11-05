#include "layer_capture.h"

#include "executable_model.h"

#include <algorithm>
#include <utility>

namespace cyxwiz {

const char* CapturedTensorRoleName(CapturedTensorRole role) {
    switch (role) {
        case CapturedTensorRole::Input:     return "input";
        case CapturedTensorRole::Output:    return "output";
        case CapturedTensorRole::Parameter: return "parameter";
        case CapturedTensorRole::State:     return "state";
        case CapturedTensorRole::Gradient:  return "gradient";
    }
    return "output";
}

const CapturedTensor* CapturedLayer::Find(CapturedTensorRole role, std::string_view name) const {
    for (const auto& t : tensors)
        if (t.role == role && (name.empty() || t.name == name)) return &t;
    return nullptr;
}

LayerCapture::LayerCapture(LayerCaptureOptions options, std::vector<BuiltModuleProvenance> provenance)
    : options_(options), provenance_(std::move(provenance)) {}

const CapturedLayer* LayerCapture::FindLayer(int node_id) const {
    for (const auto& layer : layers_)
        if (layer.node_id == node_id) return &layer;
    return nullptr;
}

CapturedLayer& LayerCapture::LayerFor(size_t module_index) {
    for (auto& layer : layers_)
        if (layer.module_index == module_index) return layer;
    CapturedLayer layer;
    layer.module_index = module_index;
    for (const auto& p : provenance_) {
        if (p.module_index && *p.module_index == module_index) {
            layer.node_id = p.node_id;
            layer.node_name = p.node_name.empty() ? p.module_name : p.node_name;
            break;
        }
    }
    layers_.push_back(std::move(layer));
    return layers_.back();
}

void LayerCapture::Hold(CapturedLayer& layer, CapturedTensorRole role, std::string name, const Tensor& tensor) {
    const size_t bytes = tensor.NumBytes();
    if (bytes > options_.byte_budget - std::min(bytes_held_, options_.byte_budget)) {
        ++skipped_tensors_;
        skipped_bytes_ += bytes;
        return;
    }
    bytes_held_ += bytes;
    layer.tensors.push_back({role, std::move(name), tensor});
}

void LayerCapture::RecordForward(size_t module_index, const Tensor& input, const Tensor& output) {
    if (!options_.activations) return;
    CapturedLayer& layer = LayerFor(module_index);
    Hold(layer, CapturedTensorRole::Input, "input", input);
    Hold(layer, CapturedTensorRole::Output, "output", output);
}

void LayerCapture::RecordModuleTensors(IExecutableModel& model) {
    if (!options_.parameters && !options_.gradients) return;
    for (size_t i = 0; i < model.LayerCount(); ++i) {
        Module* module = model.LayerModule(i);
        if (!module) continue;
        const auto parameters = module->GetParameters();
        if (parameters.empty()) continue;
        const auto state = module->GetStateNames();
        const auto is_state = [&](const std::string& name) {
            return std::find(state.begin(), state.end(), name) != state.end();
        };
        CapturedLayer& layer = LayerFor(i);
        if (options_.parameters) {
            for (const auto& [name, tensor] : parameters)
                Hold(layer, is_state(name) ? CapturedTensorRole::State : CapturedTensorRole::Parameter, name, tensor);
        }
        if (options_.gradients) {
            for (const auto& [name, tensor] : module->GetGradients())
                if (!is_state(name)) Hold(layer, CapturedTensorRole::Gradient, name, tensor);
        }
    }
}

} // namespace cyxwiz
