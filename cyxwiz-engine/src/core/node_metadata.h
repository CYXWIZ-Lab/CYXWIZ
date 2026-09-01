#pragma once

#include <string>
#include <utility>
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
    PortDefinition() = default;
    PortDefinition(std::string name_value,
                   PinType type_value = PinType::Tensor,
                   bool required_value = true,
                   std::string description_value = {},
                   bool variadic_value = false,
                   int min_connections_value = 0,
                   int max_connections_value = 1)
        : name(std::move(name_value)),
          type(type_value),
          required(required_value),
          description(std::move(description_value)),
          variadic(variadic_value),
          min_connections(min_connections_value),
          max_connections(max_connections_value) {}

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
    ParameterDefinition() = default;
    ParameterDefinition(
        std::string name_value,
        std::string type_value,
        std::string default_value_value = {},
        std::string description_value = {},
        std::vector<std::string> enum_values_value = {},
        std::string validation_value = {},
        std::string display_name_value = {},
        std::string group_value = {},
        bool required_value = false,
        bool advanced_value = false,
        ParameterConsumption consumption_value = ParameterConsumption::Runtime)
        : name(std::move(name_value)),
          type(std::move(type_value)),
          default_value(std::move(default_value_value)),
          description(std::move(description_value)),
          enum_values(std::move(enum_values_value)),
          validation(std::move(validation_value)),
          display_name(std::move(display_name_value)),
          group(std::move(group_value)),
          required(required_value),
          advanced(advanced_value),
          consumption(consumption_value) {}

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
    SupportAxisDefinition() = default;
    SupportAxisDefinition(std::string name_value,
                          std::string value_value,
                          bool supported_value = true,
                          std::string reason_value = {})
        : name(std::move(name_value)),
          value(std::move(value_value)),
          supported(supported_value),
          reason(std::move(reason_value)) {}

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
    NodeMetadata() = default;
    NodeMetadata(
        NodeType type_value,
        NodeCategory category_value,
        std::string name_value,
        std::string icon_value,
        std::vector<std::string> keywords_value = {},
        int usage_count_value = 0,
        bool is_favorite_value = false,
        std::string brief_description_value = {},
        std::string help_text_value = {},
        std::string example_usage_value = {},
        std::vector<PortDefinition> inputs_value = {},
        std::vector<PortDefinition> outputs_value = {},
        std::vector<ParameterDefinition> parameters_value = {},
        NodeImplementationStatus status_value =
            NodeImplementationStatus::Implemented,
        int user_votes_value = 0,
        std::string badge_value = {},
        std::vector<SupportAxisDefinition> support_axes_value = {},
        NodePropertiesEditor properties_editor_value =
            NodePropertiesEditor::Automatic)
        : type(type_value),
          category(category_value),
          name(std::move(name_value)),
          icon(std::move(icon_value)),
          keywords(std::move(keywords_value)),
          usage_count(usage_count_value),
          is_favorite(is_favorite_value),
          brief_description(std::move(brief_description_value)),
          help_text(std::move(help_text_value)),
          example_usage(std::move(example_usage_value)),
          inputs(std::move(inputs_value)),
          outputs(std::move(outputs_value)),
          parameters(std::move(parameters_value)),
          status(status_value),
          user_votes(user_votes_value),
          badge(std::move(badge_value)),
          support_axes(std::move(support_axes_value)),
          properties_editor(properties_editor_value) {}

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
