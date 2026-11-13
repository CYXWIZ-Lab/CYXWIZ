#include "weight_start.h"

#include "weight_file.h"

#include <cyxwiz/sequential.h>

#include <map>

namespace cyxwiz {

namespace {

size_t Elements(const std::vector<size_t>& shape) {
    size_t n = 1;
    for (const size_t d : shape) n *= d;
    return n;
}

// The channel count of the feature map flattened into this layer's input: the
// nearest layer before it is a Flatten of a channels-last sample ([H, W, C],
// [L, C], [D, H, W, C]) with as many values as this layer reads, with only
// layers that keep the row as it is (activations, dropout) between. 0 otherwise.
size_t FlattenedChannels(const std::vector<BuiltModuleProvenance>& provenance, size_t at) {
    const size_t width = Elements(provenance[at].input_shape);
    for (size_t i = at; i-- > 0;) {
        const auto& before = provenance[i];
        if (before.compiled_layer_index == provenance[at].compiled_layer_index) continue;
        if (before.node_type == gui::NodeType::Flatten) {
            const bool feature_map = before.input_shape.size() >= 2 && Elements(before.input_shape) == width;
            return feature_map ? before.input_shape.back() : 0;
        }
        if (before.input_shape != before.output_shape || before.output_shape != provenance[at].input_shape) return 0;
    }
    return 0;
}

}  // namespace

std::vector<weight_match::LayerSlot> WeightSlots(IExecutableModel& model,
                                                 const std::vector<BuiltModuleProvenance>& provenance) {
    std::vector<weight_match::LayerSlot> slots;
    for (size_t at = 0; at < provenance.size(); ++at) {
        const auto& entry = provenance[at];
        if (!entry.created()) continue;
        Module* module = model.LayerModule(*entry.module_index);
        if (!module || !module->HasParameters()) continue;
        weight_match::LayerSlot slot;
        slot.node_name = entry.node_name;
        slot.type = entry.node_type;
        slot.type_label = entry.module_name.substr(0, entry.module_name.find('('));
        slot.key_prefix = "layer" + std::to_string(*entry.module_index) + ".";
        for (const auto& [key, tensor] : module->GetParameters()) slot.parameters.emplace_back(key, tensor.Shape());
        if (entry.node_type == gui::NodeType::Dense) slot.flattened_channels = FlattenedChannels(provenance, at);
        if (!slot.parameters.empty()) slots.push_back(std::move(slot));
    }
    return slots;
}

bool StartFromWeightsFile(const std::filesystem::path& path, IExecutableModel& model,
                          const std::vector<BuiltModuleProvenance>& provenance, weight_match::Report& report,
                          std::string& error) {
    report = weight_match::Report{};
    weight_file::WeightFile file;
    if (!weight_file::Read(path, file, error)) return false;
    report = weight_match::Match(file, WeightSlots(model, provenance));
    if (report.loaded == 0) {
        error = "no layer of this graph matches the file. A layer takes the entries named after its node "
                "(node conv1 takes conv1.weight); the file's first entry is " +
                (file.entries.empty() ? std::string("missing") : file.entries.front().name);
        return false;
    }
    std::map<std::string, Tensor> parameters;
    for (const auto& tensor : report.tensors) {
        parameters.emplace(tensor.key, Tensor(tensor.shape, tensor.values.data(), DataType::Float32));
    }
    model.SetParameters(parameters);
    return true;
}

}  // namespace cyxwiz
