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
    bool variadic = false;
    int min_connections = 0;
    int max_connections = 1;
};

/**
 * Declares who consumes a persisted node parameter.
 *
 * Runtime is the default contract: compiler, loader, materializer, executor,
 * training, or export code consumes the value. UiOnly must be selected for a
 * field that intentionally controls only an editor, dialog, or visualization.
 */
enum class ParameterConsumption {
    Runtime,
    UiOnly,
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
    ParameterConsumption consumption = ParameterConsumption::Runtime;
};

/**
 * Selects the single Properties editing surface for a node.
 * Automatic uses metadata when parameters exist and the narrow fallback
 * surface otherwise. Dialog and Custom are explicit ownership exceptions.
 */
enum class NodePropertiesEditor {
    Automatic,
    Dialog,
    Custom,
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
    NodePropertiesEditor properties_editor = NodePropertiesEditor::Automatic;

    // Helper methods
    bool IsTemplate() const { return status == NodeImplementationStatus::Template; }
    bool IsDeprecated() const { return status == NodeImplementationStatus::Deprecated; }
    bool IsImplemented() const { return status == NodeImplementationStatus::Implemented; }
};

/**
 * Return whether runtime/training capability truth blocks this node from
 * being added to a graph. Kept beside NodeMetadata so every frontend entry
 * point applies the same support policy.
 */
inline bool IsNodeSupportBlocked(const NodeMetadata& metadata) {
    for (const auto& axis : metadata.support_axes) {
        if (axis.name == "Support State" && axis.value == "blocked") {
            return true;
        }
    }
    // Runtime and compiler axes are lane-specific. A node may be unsupported
    // by PipelineExecutor while remaining valid in the training graph runtime.
    // Capability resolution promotes a true product-wide blocker to the
    // explicit Support State above and/or Template status below.
    return metadata.badge == "Blocked";
}

inline bool CanAddNodeToGraph(const NodeMetadata& metadata) {
    return !metadata.IsTemplate() && !IsNodeSupportBlocked(metadata);
}

/**
 * Apply the declarative static contract to a newly constructed node.
 * Dynamic dialogs may add or replace fields after this bootstrap step.
 */
inline void ApplyStaticNodeMetadataContract(const NodeMetadata& metadata,
                                            gui::MLNode& node,
                                            int& next_pin_id) {
    node.category = metadata.category;

    const auto append_pin = [&next_pin_id](const PortDefinition& port,
                                           bool is_input,
                                           std::vector<gui::NodePin>& pins) {
        gui::NodePin pin{};
        pin.id = next_pin_id++;
        pin.type = port.type;
        pin.name = port.name;
        pin.is_input = is_input;
        pin.description = port.description;
        pin.is_required = port.required;
        pin.is_variadic = port.variadic;
        pin.min_connections = port.min_connections;
        pin.max_connections = port.max_connections;
        pins.push_back(pin);
    };

    for (const auto& input : metadata.inputs) {
        append_pin(input, true, node.inputs);
    }
    for (const auto& output : metadata.outputs) {
        append_pin(output, false, node.outputs);
    }
    for (const auto& parameter : metadata.parameters) {
        node.parameters.emplace(parameter.name, parameter.default_value);
    }
}

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
