#pragma once

#include <string>
#include <vector>
#include "../gui/node_editor.h"  // For NodeType, NodeCategory, PinType

namespace cyxwiz {

// Forward use gui namespace enums
using gui::NodeType;
using gui::NodeCategory;
using gui::PinType;

/**
 * Node implementation status for Node Browser display
 */
enum class NodeImplementationStatus {
    Implemented,    // Fully working - can be used
    Template,       // Defined but not implemented (future/coming soon)
    Deprecated,     // Being phased out
    External        // Requires external integration
};

/**
 * Port definition for node inputs/outputs
 */
struct PortDefinition {
    std::string name;
    PinType type = PinType::Tensor;
    bool required = true;
    std::string description;
};

/**
 * Parameter definition for node configuration
 */
struct ParameterDefinition {
    std::string name;
    std::string type;           // "string", "multiline", "password", "int", "float", "enum", "bool", "file", "directory"
    std::string default_value;
    std::string description;
    std::vector<std::string> enum_values;  // For enum type
    std::string validation;     // Regex or range "0-100"
    std::string display_name;   // Optional UI label; falls back to humanized name
    std::string group;          // Optional UI grouping label
    bool required = false;
    bool advanced = false;
};

/**
 * Support axis for frontend/runtime capability display
 */
struct SupportAxisDefinition {
    std::string name;
    std::string value;
    bool supported = true;
    std::string reason;
};

/**
 * NodeMetadata - Complete metadata for a node type
 * Used by NodeBrowserPanel for display and by Info Panel for documentation
 */
struct NodeMetadata {
    // Identity
    NodeType type = NodeType::Unknown;
    NodeCategory category = NodeCategory::Unknown;
    std::string name;
    std::string icon;           // FontAwesome icon string (UTF-8)

    // Discovery
    std::vector<std::string> keywords;
    int usage_count = 0;        // For "most used" sorting
    bool is_favorite = false;   // User-starred

    // Documentation
    std::string brief_description;  // One-liner for tooltip
    std::string help_text;          // Detailed help
    std::string example_usage;      // Code/workflow example

    // Ports
    std::vector<PortDefinition> inputs;
    std::vector<PortDefinition> outputs;

    // Parameters
    std::vector<ParameterDefinition> parameters;

    // Status
    NodeImplementationStatus status = NodeImplementationStatus::Implemented;
    int user_votes = 0;         // For template nodes (feature requests)
    std::string badge;          // Optional badge text (e.g., "Coming Soon", "Beta")
    std::vector<SupportAxisDefinition> support_axes;

    // Helper methods
    bool IsTemplate() const { return status == NodeImplementationStatus::Template; }
    bool IsDeprecated() const { return status == NodeImplementationStatus::Deprecated; }
    bool IsImplemented() const { return status == NodeImplementationStatus::Implemented; }
};

/**
 * Get display name for a category
 */
inline std::string GetCategoryDisplayName(NodeCategory category) {
    switch (category) {
        // Data I/O
        case NodeCategory::DataSources:     return "I/O";
        case NodeCategory::Database:        return "Database";
        case NodeCategory::CloudStorage:    return "Cloud Storage";
        case NodeCategory::DataTransform:   return "Manipulation";

        // Analytics & Visualization
        case NodeCategory::Analytics:       return "Analytics";
        case NodeCategory::Visualization:   return "Visualization";

        // ML Layers
        case NodeCategory::Layers:          return "ML Layers";
        case NodeCategory::Activation:      return "Activation";
        case NodeCategory::Pooling:         return "Pooling";
        case NodeCategory::Normalization:   return "Normalization";
        case NodeCategory::Attention:       return "Attention";
        case NodeCategory::Recurrent:       return "Recurrent";
        case NodeCategory::ShapeOps:        return "Shape Ops";
        case NodeCategory::MergeOps:        return "Merge Ops";
        case NodeCategory::Upsampling:      return "Upsampling";

        // Training & Models
        case NodeCategory::Training:        return "Training";
        case NodeCategory::Regularization:  return "Regularization";
        case NodeCategory::ModelIO:         return "Model I/O";
        case NodeCategory::MLServices:      return "ML Services";
        case NodeCategory::Explainability:  return "Explainability";

        // Data Processing
        case NodeCategory::Preprocessing:   return "Preprocessing";
        case NodeCategory::DataPipeline:    return "Data Pipeline";
        case NodeCategory::TextProcessing:  return "Text";
        case NodeCategory::TimeSeries:      return "Time Series";
        case NodeCategory::Audio:           return "Audio";
        case NodeCategory::JsonXml:         return "JSON/XML";

        // Specialized
        case NodeCategory::DNN:             return "DNN Models";
        case NodeCategory::RL:              return "RL";
        case NodeCategory::BigData:         return "Big Data";

        // Workflow & UI
        case NodeCategory::Workflow:        return "Workflow";
        case NodeCategory::Widgets:         return "Widgets";
        case NodeCategory::Reporting:       return "Reporting";
        case NodeCategory::Utility:         return "Utility";
        case NodeCategory::Signal:          return "Signal";

        case NodeCategory::Plugin:          return "Plugins";
        default:                            return "Other";
    }
}

/**
 * Get icon for a category (defined in node_metadata_registry.cpp)
 */
const char* GetCategoryIcon(NodeCategory category);

} // namespace cyxwiz
