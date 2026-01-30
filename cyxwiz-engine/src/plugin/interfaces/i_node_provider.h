#pragma once

#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace cyxwiz::plugin {

struct PluginNodeTypeInfo {
    std::string type_name;              // Unique within plugin (e.g. "GaussianBlur")
    std::string display_name;           // UI display (e.g. "Gaussian Blur")
    std::string category;               // Menu category (e.g. "Image Processing")
    std::string description;

    uint32_t color = 0xFF4488AAUL;      // Node header color (ABGR)
    std::string icon;                   // FontAwesome icon codepoint

    struct PinInfo {
        std::string name;
        std::string type;               // "Tensor", "Image", "Scalar", etc.
        bool is_input = true;
    };
    std::vector<PinInfo> pins;

    std::map<std::string, std::string> default_parameters;
};

class INodeProvider {
public:
    virtual ~INodeProvider() = default;

    virtual std::vector<PluginNodeTypeInfo> GetNodeTypes() = 0;

    // Generate framework code for a node instance
    virtual std::string GenerateCode(
        const std::string& node_type_name,
        const std::map<std::string, std::string>& parameters,
        const std::string& framework          // "pytorch", "tensorflow", "keras"
    ) = 0;
};

} // namespace cyxwiz::plugin
