#pragma once

#include <string>
#include <vector>
#include <map>
#include <variant>
#include <cstdint>

namespace cyxwiz::plugin {

// Runtime value for graph execution (matches cyxwiz::NodeValue)
using PluginNodeValue = std::variant<float, std::vector<float>, std::string>;

// Context for runtime node evaluation
struct PluginNodeEvalContext {
    std::string node_type_name;
    std::map<std::string, std::string> parameters;
    std::map<std::string, PluginNodeValue> input_values;  // pin_name -> value
    float sim_time = 0.0f;
    float dt = 0.0f;
};

// Result from runtime node evaluation
struct PluginNodeEvalResult {
    std::map<std::string, PluginNodeValue> output_values;  // pin_name -> value
    bool success = true;
    std::string error_message;
};

struct PluginNodeTypeInfo {
    std::string type_name;              // Unique within plugin (e.g. "GaussianBlur")
    std::string display_name;           // UI display (e.g. "Gaussian Blur")
    std::string category;               // Menu category (e.g. "Image Processing")
    std::string description;

    uint32_t color = 0xFF4488AAUL;      // Node header color (ABGR)
    std::string icon;                   // FontAwesome icon codepoint

    struct PinInfo {
        std::string name;
        std::string type;               // "Tensor", "Image", "Scalar", "Signal", etc.
        bool is_input = true;
    };
    std::vector<PinInfo> pins;

    std::map<std::string, std::string> default_parameters;

    // Dynamic pins: if true, pins can change based on parameters (e.g. MJCF model).
    // When a parameter changes, the engine calls INodeProvider::ResolveDynamicPins().
    bool supports_dynamic_pins = false;
    std::string dynamic_pin_trigger;    // Parameter name that triggers pin resolution (e.g. "mjcf_path")
};

// Result from resolving dynamic pins for a node instance
struct DynamicPinResult {
    std::vector<PluginNodeTypeInfo::PinInfo> pins;
    std::map<std::string, std::string> metadata;  // e.g. "model_name" -> "UR5e"
};

// Callback for enumerating node types across DLL boundary without returning std::vector.
// The DLL calls this callback for each node type, passing C strings that the engine copies.
struct NodeTypeCallback {
    void* user_data;
    void (*on_node)(void* user_data,
                    const char* type_name, const char* display_name,
                    const char* category, const char* description,
                    uint32_t color, const char* icon,
                    bool supports_dynamic_pins, const char* dynamic_pin_trigger);
    void (*on_pin)(void* user_data,
                   const char* name, const char* type, bool is_input);
    void (*on_param)(void* user_data,
                     const char* key, const char* value);
    void (*on_node_done)(void* user_data);  // Signals end of current node's pins/params
};

class INodeProvider {
public:
    virtual ~INodeProvider() = default;

    virtual std::vector<PluginNodeTypeInfo> GetNodeTypes() = 0;

    // Safe cross-DLL enumeration: DLL iterates its own vector and calls back with C strings.
    // Default implementation delegates to GetNodeTypes() — plugins don't need to override.
    virtual void EnumerateNodeTypes(const NodeTypeCallback& cb) {
        auto types = GetNodeTypes();
        for (const auto& info : types) {
            cb.on_node(cb.user_data,
                       info.type_name.c_str(), info.display_name.c_str(),
                       info.category.c_str(), info.description.c_str(),
                       info.color, info.icon.c_str(),
                       info.supports_dynamic_pins, info.dynamic_pin_trigger.c_str());
            for (const auto& pin : info.pins) {
                cb.on_pin(cb.user_data, pin.name.c_str(), pin.type.c_str(), pin.is_input);
            }
            for (const auto& [k, v] : info.default_parameters) {
                cb.on_param(cb.user_data, k.c_str(), v.c_str());
            }
            cb.on_node_done(cb.user_data);
        }
    }

    // Generate framework code for a node instance
    virtual std::string GenerateCode(
        const std::string& node_type_name,
        const std::map<std::string, std::string>& parameters,
        const std::string& framework          // "pytorch", "tensorflow", "keras"
    ) = 0;

    // Resolve dynamic pins based on current parameters.
    // Called when a parameter listed in dynamic_pin_trigger changes.
    // Default returns empty (no dynamic pins).
    virtual DynamicPinResult ResolveDynamicPins(
        const std::string& node_type_name,
        const std::map<std::string, std::string>& parameters) {
        (void)node_type_name;
        (void)parameters;
        return {};
    }

    // Evaluate a node at runtime (for live simulation / graph execution).
    // Called by GraphExecutor each tick for plugin nodes.
    // Default returns failure — plugins must override to support live execution.
    virtual PluginNodeEvalResult EvaluateNode(const PluginNodeEvalContext& ctx) {
        (void)ctx;
        PluginNodeEvalResult r;
        r.success = false;
        r.error_message = "EvaluateNode not implemented";
        return r;
    }
};

} // namespace cyxwiz::plugin
