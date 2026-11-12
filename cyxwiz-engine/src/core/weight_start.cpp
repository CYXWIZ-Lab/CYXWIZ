#include "weight_start.h"

#include "weight_file.h"

#include <cyxwiz/sequential.h>

#include <map>

namespace cyxwiz {

std::vector<weight_match::LayerSlot> WeightSlots(IExecutableModel& model,
                                                 const std::vector<BuiltModuleProvenance>& provenance) {
    std::vector<weight_match::LayerSlot> slots;
    for (const auto& entry : provenance) {
        if (!entry.created()) continue;
        Module* module = model.LayerModule(*entry.module_index);
        if (!module || !module->HasParameters()) continue;
        weight_match::LayerSlot slot;
        slot.node_name = entry.node_name;
        slot.type = entry.node_type;
        slot.type_label = entry.module_name.substr(0, entry.module_name.find('('));
        slot.key_prefix = "layer" + std::to_string(*entry.module_index) + ".";
        for (const auto& [key, tensor] : module->GetParameters()) slot.parameters.emplace_back(key, tensor.Shape());
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
