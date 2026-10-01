#pragma once

// Extension nodes (TOFIX125): node types CyxWiz does not ship, supplied by a
// C++ plugin or a Python node bundle. Every extension node is a
// NodeType::PluginCustom on the canvas; its identity is the string type id
// "<provider_id>:<TypeName>". The description reuses NodeMetadata so typed
// parameters and ports work like a built-in node's.

#include "node_metadata.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

class Module;

namespace plugin {
class INodeProvider;
}

enum class ExtensionNodeKind {
    Signal,       // evaluated per tick by GraphExecutor (simulation); cannot train
    TensorLayer,  // one tensor in, one tensor out; becomes a Module in the model
};

enum class ExtensionLayerRole {
    Layer,
    Activation,
};

enum class ExtensionSource {
    NativePlugin,
    PythonBundle,
};

// Result of the numerical gradient check of a TensorLayer node.
struct ExtensionVerification {
    enum class Status { NotRun, Passed, Failed, Stale };
    Status status = Status::NotRun;
    std::string checked_on;  // ISO date
    double tolerance = 0.0;
    double max_relative_error = 0.0;
    std::string note;
};

struct ExtensionNodeDescriptor {
    std::string type_id;      // "<provider_id>:<type_name>"
    std::string provider_id;  // plugin id or bundle id
    std::string type_name;    // unique within the provider
    std::string version;      // version of the node contract (pins and parameters)
    std::string content_hash; // of the code that implements it
    ExtensionSource source = ExtensionSource::NativePlugin;
    std::string source_path;
    ExtensionNodeKind kind = ExtensionNodeKind::Signal;
    ExtensionLayerRole role = ExtensionLayerRole::Layer;

    // name, category, ports, typed parameters, help. type is PluginCustom.
    NodeMetadata metadata;
    std::string menu_category;         // provider's own category text ("Image Processing")
    uint32_t color = 0xFF4488AAUL;     // node header colour (ABGR)

    // Pins that follow a parameter (a MuJoCo model file, for example).
    bool supports_dynamic_pins = false;
    std::string dynamic_pin_trigger;

    ExtensionVerification verification;
};

struct ExtensionShapeResult {
    bool ok = false;
    std::vector<size_t> output_shape;  // per sample, without the batch dimension
    std::string error;
};

// What a TensorLayer node supplies. Both functions may be called from any
// thread, and the background compile calls them after every graph edit.
class ITensorNodeFactory {
public:
    virtual ~ITensorNodeFactory() = default;

    // Pure: the same inputs give the same answer.
    virtual ExtensionShapeResult InferOutputShape(
        const std::vector<size_t>& input_shape,
        const std::map<std::string, std::string>& parameters) const = 0;

    // nullptr with `error` set when the module cannot be created.
    virtual std::unique_ptr<Module> CreateModule(
        const std::vector<size_t>& input_shape,
        const std::map<std::string, std::string>& parameters,
        std::string& error) const = 0;
};

// "<provider_id>:<type_name>".
std::string MakeExtensionTypeId(const std::string& provider_id, const std::string& type_name);

// False with a reason. Provider ids use a-z, 0-9, '.', '-' and '_'; type
// names use letters, digits and '_'. Neither may be empty.
bool ValidateExtensionTypeId(const std::string& provider_id, const std::string& type_name,
                             std::string& error);

// Splits at the first ':'. False when there is none or a side is empty.
bool SplitExtensionTypeId(const std::string& type_id, std::string& provider_id,
                          std::string& type_name);

}  // namespace cyxwiz
