#pragma once

// Trainable extension nodes from a C++ plugin (TOFIX125 P2, plugin API 1.1).
// A plugin returns this from IPlugin::QueryInterface("ITensorNodeProvider").
// Each node is one tensor in, one tensor out (a layer or an activation); its
// module is a real cyxwiz::Module, so it trains like a built-in layer.
//
// Rules for the modules (enforced by the engine's guard where it can):
// - Forward receives the whole batch; dimension 0 is the batch.
// - Backward returns a gradient with the shape of the last Forward's input.
// - GetParameters, GetGradients and SetParameters use the same keys, and
//   HasParameters returns true when there are any.
// - Use cyxwiz::Tensor operations; do not change the input tensor.
// - InferOutputShape and CreateModule may be called from any thread, after
//   every graph edit: keep them cheap.

#include "../../core/extension_node_contract.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {
class Module;
}

namespace cyxwiz::plugin {

struct TensorNodeInfo {
    std::string type_name;     // unique within the plugin, letters, digits, '_'
    std::string display_name;
    std::string category;      // the plugin's own grouping ("Activations")
    std::string description;
    std::string version;       // version of this node's pins and parameters
    uint32_t color = 0;        // header colour, IM_COL32 layout (ABGR); 0 = none
    std::string icon;          // icon font codepoint; empty = the engine's plug
    bool is_activation = false;

    struct Pin {
        std::string name;
        bool is_input = true;
    };
    std::vector<Pin> pins;     // v1: exactly one input and one output

    struct Parameter {
        std::string name;
        std::string type;      // "int", "float", "bool", "string", "enum"
        std::string default_value;
        std::string description;
        std::vector<std::string> choices;  // for "enum"
    };
    std::vector<Parameter> parameters;
};

// Strings cross the library boundary as C strings and are copied by the
// engine, as INodeProvider does.
struct TensorNodeCallback {
    void* user_data = nullptr;
    void (*on_node)(void* user_data, const char* type_name, const char* display_name,
                    const char* category, const char* description, const char* version,
                    uint32_t color, const char* icon, bool is_activation) = nullptr;
    void (*on_pin)(void* user_data, const char* name, bool is_input) = nullptr;
    void (*on_parameter)(void* user_data, const char* name, const char* type,
                         const char* default_value, const char* description,
                         const char* choices_comma_separated) = nullptr;
    void (*on_node_done)(void* user_data) = nullptr;
};

class ITensorNodeProvider {
public:
    virtual ~ITensorNodeProvider() = default;

    virtual std::vector<TensorNodeInfo> GetTensorNodes() = 0;

    // Output shape per sample (without the batch dimension). Pure.
    virtual ExtensionShapeResult InferOutputShape(
        const std::string& type_name,
        const std::vector<size_t>& input_shape,
        const std::map<std::string, std::string>& parameters) = 0;

    // nullptr with `error` set when the module cannot be created.
    virtual std::unique_ptr<cyxwiz::Module> CreateModule(
        const std::string& type_name,
        const std::vector<size_t>& input_shape,
        const std::map<std::string, std::string>& parameters,
        std::string& error) = 0;

    // Default: walks GetTensorNodes() inside the plugin and hands the engine
    // C strings. Plugins do not need to override it.
    virtual void EnumerateTensorNodes(const TensorNodeCallback& cb) {
        for (const auto& node : GetTensorNodes()) {
            cb.on_node(cb.user_data, node.type_name.c_str(), node.display_name.c_str(),
                       node.category.c_str(), node.description.c_str(), node.version.c_str(),
                       node.color, node.icon.c_str(), node.is_activation);
            for (const auto& pin : node.pins) cb.on_pin(cb.user_data, pin.name.c_str(), pin.is_input);
            for (const auto& parameter : node.parameters) {
                std::string choices;
                for (const auto& choice : parameter.choices) {
                    if (!choices.empty()) choices += ",";
                    choices += choice;
                }
                cb.on_parameter(cb.user_data, parameter.name.c_str(), parameter.type.c_str(),
                                parameter.default_value.c_str(), parameter.description.c_str(),
                                choices.c_str());
            }
            cb.on_node_done(cb.user_data);
        }
    }
};

}  // namespace cyxwiz::plugin
