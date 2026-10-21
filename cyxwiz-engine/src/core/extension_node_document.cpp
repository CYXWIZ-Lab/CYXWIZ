#include "extension_node_document.h"

#include "extension_node_registry.h"
#include "graph_node_factory.h"

#include <nlohmann/json.hpp>

#include <utility>
#include <vector>

namespace cyxwiz {

namespace {

using json = nlohmann::json;

struct SavedPin {
    std::string name;
    gui::PinType type = gui::PinType::Tensor;
};

json WritePins(const std::vector<gui::NodePin>& pins) {
    json result = json::array();
    for (const auto& pin : pins) {
        result.push_back({{"name", pin.name}, {"type", PinTypeToText(pin.type)}});
    }
    return result;
}

bool ReadPins(const json& block, const char* key, std::vector<SavedPin>& pins, std::string& error) {
    if (!block.contains(key) || !block[key].is_array()) {
        error = std::string("the extension block has no '") + key + "' list";
        return false;
    }
    for (const auto& entry : block[key]) {
        if (!entry.is_object() || !entry.contains("name") || !entry["name"].is_string() ||
            !entry.contains("type") || !entry["type"].is_string()) {
            error = std::string("a pin in '") + key + "' needs a name and a type";
            return false;
        }
        SavedPin pin;
        pin.name = entry["name"].get<std::string>();
        const std::string type_text = entry["type"].get<std::string>();
        if (!PinTypeFromText(type_text, pin.type)) {
            error = "pin '" + pin.name + "' has the unknown type '" + type_text + "'";
            return false;
        }
        pins.push_back(std::move(pin));
    }
    return true;
}

bool SamePins(const std::vector<SavedPin>& saved, const std::vector<gui::NodePin>& current) {
    if (saved.size() != current.size()) return false;
    for (size_t i = 0; i < saved.size(); ++i) {
        if (saved[i].name != current[i].name || saved[i].type != current[i].type) return false;
    }
    return true;
}

void RebuildPins(const std::vector<SavedPin>& saved, bool is_input, int& next_pin_id,
                 std::vector<gui::NodePin>& pins) {
    pins.clear();
    for (const auto& entry : saved) {
        gui::NodePin pin{};
        pin.id = next_pin_id++;
        pin.type = entry.type;
        pin.name = entry.name;
        pin.is_input = is_input;
        pins.push_back(std::move(pin));
    }
}

void AppendNote(std::string& note, const std::string& text) {
    if (!note.empty()) note += " ";
    note += text;
}

}  // namespace

const char* PinTypeToText(gui::PinType type) {
    switch (type) {
        case gui::PinType::Tensor:     return "Tensor";
        case gui::PinType::Labels:     return "Labels";
        case gui::PinType::Parameters: return "Parameters";
        case gui::PinType::Loss:       return "Loss";
        case gui::PinType::Optimizer:  return "Optimizer";
        case gui::PinType::Dataset:    return "Dataset";
    }
    return "Tensor";
}

bool PinTypeFromText(const std::string& text, gui::PinType& type) {
    const gui::PinType all[] = {gui::PinType::Tensor, gui::PinType::Labels, gui::PinType::Parameters,
                                gui::PinType::Loss, gui::PinType::Optimizer, gui::PinType::Dataset};
    for (const auto candidate : all) {
        if (text == PinTypeToText(candidate)) {
            type = candidate;
            return true;
        }
    }
    return false;
}

json WriteExtensionBlock(const gui::MLNode& node) {
    if (node.type != gui::NodeType::PluginCustom) return json();
    return {{"contract", kExtensionBlockContract},
            {"type_id", node.extension_type_id},
            {"version", node.extension_version},
            {"content_hash", node.extension_content_hash},
            {"inputs", WritePins(node.inputs)},
            {"outputs", WritePins(node.outputs)}};
}

bool ReadExtensionNode(const json& node_json, int& next_node_id, int& next_pin_id,
                       gui::MLNode& node, std::string& error) {
    if (!node_json.contains("extension") || !node_json["extension"].is_object()) {
        error = "it is an extension node but has no extension block (saved by an older "
                "version; add the node again from the palette)";
        return false;
    }
    const json& block = node_json["extension"];

    const int contract = block.value("contract", 0);
    if (contract > kExtensionBlockContract) {
        error = "its extension block was written by a newer version of CyxWiz (contract " +
                std::to_string(contract) + ")";
        return false;
    }
    if (contract < 1) {
        error = "its extension block has no contract number";
        return false;
    }

    if (!block.contains("type_id") || !block["type_id"].is_string()) {
        error = "its extension block has no type_id";
        return false;
    }
    const std::string type_id = block["type_id"].get<std::string>();
    std::string provider_id;
    std::string type_name;
    if (!SplitExtensionTypeId(type_id, provider_id, type_name) ||
        !ValidateExtensionTypeId(provider_id, type_name, error)) {
        if (error.empty()) error = "'" + type_id + "' is not a type id (provider:TypeName)";
        error = "its extension block has an invalid type_id: " + error;
        return false;
    }

    std::vector<SavedPin> saved_inputs;
    std::vector<SavedPin> saved_outputs;
    if (!ReadPins(block, "inputs", saved_inputs, error) ||
        !ReadPins(block, "outputs", saved_outputs, error)) {
        return false;
    }
    const std::string saved_version = block.value("version", std::string{});
    const std::string saved_hash = block.value("content_hash", std::string{});

    node = gui::CreateExtensionGraphNode(type_id, next_node_id, next_pin_id);

    if (node.extension_missing) {
        RebuildPins(saved_inputs, true, next_pin_id, node.inputs);
        RebuildPins(saved_outputs, false, next_pin_id, node.outputs);
        node.extension_version = saved_version;
        node.extension_content_hash = saved_hash;
        return true;
    }

    // Pins that follow a parameter were resolved when the graph was saved;
    // the saved list is that resolved state.
    if (node.has_dynamic_pins) {
        RebuildPins(saved_inputs, true, next_pin_id, node.inputs);
        RebuildPins(saved_outputs, false, next_pin_id, node.outputs);
    } else if (!SamePins(saved_inputs, node.inputs) || !SamePins(saved_outputs, node.outputs)) {
        AppendNote(node.extension_load_note,
                   "Its pins changed since the graph was saved (" +
                       std::to_string(saved_inputs.size()) + " in, " +
                       std::to_string(saved_outputs.size()) + " out then; " +
                       std::to_string(node.inputs.size()) + " in, " +
                       std::to_string(node.outputs.size()) + " out now); check its links.");
    }
    if (saved_version != node.extension_version) {
        AppendNote(node.extension_load_note,
                   "Saved with version " + (saved_version.empty() ? "(none)" : saved_version) +
                       ", installed version is " +
                       (node.extension_version.empty() ? "(none)" : node.extension_version) + ".");
    } else if (saved_hash != node.extension_content_hash) {
        AppendNote(node.extension_load_note,
                   "Its code changed since the graph was saved, with the same version number.");
    }
    return true;
}

}  // namespace cyxwiz
