#pragma once

#include <string>
#include <vector>
#include <map>
#include <imgui.h>

#include "../icons.h"

namespace gui::patterns {

// Pattern parameter type
enum class ParameterType {
    Int,
    Float,
    String,
    Bool,
    IntArray,    // e.g., kernel_size [3, 3]
    FloatArray,  // e.g., scales [1.0, 2.0]
    NodeType     // Dropdown of node types (e.g., activation selection)
};

// Parameter definition for a pattern
struct PatternParameter {
    std::string name;           // e.g., "hidden_size"
    ParameterType type;
    std::string default_value;  // Default value as string
    std::string description;    // Help text for tooltip
    std::string min_value;      // Optional min (for numeric types)
    std::string max_value;      // Optional max (for numeric types)
    std::vector<std::string> options;  // For dropdown (NodeType or enum)
};

// Pattern category for organization
enum class PatternCategory {
    Basic,          // MLP, simple feedforward networks
    CNN,            // Convolutional networks
    RNN,            // Recurrent networks (LSTM, GRU)
    Transformer,    // Attention-based architectures
    Generative,     // VAE, autoencoder, GAN
    BuildingBlocks, // Residual block, attention block, etc.
    Custom          // User-defined patterns
};

// Node definition within a pattern template
struct PatternNode {
    std::string id;             // Internal reference ID
    std::string type;           // Node type name (e.g., "Dense", "Conv2D", "$activation")
    std::string name;           // Display name
    float pos_x;                // X position (relative)
    float pos_y;                // Y position (relative)
    std::map<std::string, std::string> params;  // Node parameters (can use $param refs)
};

// Link definition within a pattern template
struct PatternLink {
    std::string from_node;      // Source node ID
    std::string to_node;        // Target node ID
    int from_pin;               // Source pin index (default 0)
    int to_pin;                 // Target pin index (default 0)
};

// Pattern template structure
struct PatternTemplate {
    std::vector<PatternNode> nodes;
    std::vector<PatternLink> links;
};

// Complete pattern definition
struct Pattern {
    std::string id;             // Unique identifier (filename without extension)
    std::string name;           // Display name
    std::string description;    // Description shown in browser
    PatternCategory category;
    std::vector<PatternParameter> parameters;
    PatternTemplate template_data;  // Parsed template
    std::string template_json;      // Raw JSON template (for saving)
    std::vector<std::string> tags;  // Search tags
};

// Helper functions
inline const char* PatternCategoryToString(PatternCategory category) {
    switch (category) {
        case PatternCategory::Basic:         return "Basic";
        case PatternCategory::CNN:           return "CNN";
        case PatternCategory::RNN:           return "RNN";
        case PatternCategory::Transformer:   return "Transformer";
        case PatternCategory::Generative:    return "Generative";
        case PatternCategory::BuildingBlocks: return "Building Blocks";
        case PatternCategory::Custom:        return "Custom";
        default:                             return "Unknown";
    }
}

inline PatternCategory StringToPatternCategory(const std::string& str) {
    if (str == "Basic")           return PatternCategory::Basic;
    if (str == "CNN")             return PatternCategory::CNN;
    if (str == "RNN")             return PatternCategory::RNN;
    if (str == "Transformer")     return PatternCategory::Transformer;
    if (str == "Generative")      return PatternCategory::Generative;
    if (str == "BuildingBlocks" || str == "Building Blocks")
                                  return PatternCategory::BuildingBlocks;
    return PatternCategory::Custom;
}

inline const char* ParameterTypeToString(ParameterType type) {
    switch (type) {
        case ParameterType::Int:        return "int";
        case ParameterType::Float:      return "float";
        case ParameterType::String:     return "string";
        case ParameterType::Bool:       return "bool";
        case ParameterType::IntArray:   return "int[]";
        case ParameterType::FloatArray: return "float[]";
        case ParameterType::NodeType:   return "nodetype";
        default:                        return "unknown";
    }
}

inline ParameterType StringToParameterType(const std::string& str) {
    if (str == "int")       return ParameterType::Int;
    if (str == "float")     return ParameterType::Float;
    if (str == "string")    return ParameterType::String;
    if (str == "bool")      return ParameterType::Bool;
    if (str == "int[]")     return ParameterType::IntArray;
    if (str == "float[]")   return ParameterType::FloatArray;
    if (str == "nodetype")  return ParameterType::NodeType;
    return ParameterType::String;  // Default to string
}

// The icon of a pattern category (Nodes panel examples, Pattern Browser).
inline const char* GetPatternCategoryIcon(PatternCategory category) {
    switch (category) {
        case PatternCategory::Basic:          return ICON_FA_LAYER_GROUP;
        case PatternCategory::CNN:            return ICON_FA_IMAGE;
        case PatternCategory::RNN:            return ICON_FA_CHART_LINE;
        case PatternCategory::Transformer:    return ICON_FA_BRAIN;
        case PatternCategory::Generative:     return ICON_FA_WAND_MAGIC_SPARKLES;
        case PatternCategory::BuildingBlocks: return ICON_FA_CUBES;
        case PatternCategory::Custom:         return ICON_FA_USER;
        default:                              return ICON_FA_CUBE;
    }
}

} // namespace gui::patterns
