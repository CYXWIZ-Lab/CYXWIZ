#include "../src/core/node_metadata_registry.h"
#include "../src/core/pipeline_runtime_capabilities.h"
#include "../src/gui/node_type_import_registry.h"
#include "../src/gui/properties_contract.h"
#include "node_contract_inventory.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::string TypeId(gui::NodeType type) {
    return std::to_string(static_cast<int>(type));
}

void CheckGraphImportNameContract(
    const cyxwiz::NodeMetadataRegistry& registry) {
    const auto import_names = gui::GetNodeTypeImportNames();
    Check(import_names.size() == 195,
          "accepted graph-import name count drifted");

    std::set<std::string> names;
    std::size_t compatibility_only_count = 0;
    for (const auto& entry : import_names) {
        const std::string name(entry.name);
        Check(!name.empty(), "graph-import name is empty");
        Check(names.insert(name).second,
              "duplicate graph-import name: " + name);
        Check(entry.node_type != gui::NodeType::Unknown,
              "graph-import name maps to Unknown: " + name);
        const auto resolved = gui::ResolveNodeTypeImportName(entry.name);
        Check(resolved.has_value() && *resolved == entry.node_type,
              "graph-import resolver disagrees for: " + name);
        const auto* metadata = registry.GetMetadata(entry.node_type);
        if (entry.legacy_import_compatibility_only) {
            ++compatibility_only_count;
            Check(metadata == nullptr,
                  "compatibility-only graph-import target gained metadata: " +
                      name);
        } else {
            Check(metadata != nullptr,
                  "graph-import target has no registered metadata: " + name);
        }
    }
    Check(compatibility_only_count == 3,
          "saved-graph compatibility import count drifted");
}

const cyxwiz::SupportAxisDefinition* FindSupportAxis(
    const cyxwiz::NodeMetadata& metadata,
    const std::string& name) {
    for (const auto& axis : metadata.support_axes) {
        if (axis.name == name) {
            return &axis;
        }
    }
    return nullptr;
}

bool IsSupportedParameterType(const std::string& type) {
    static const std::set<std::string> supported = {
        "bool",
        "directory",
        "dropdown",
        "enum",
        "file",
        "float",
        "folder",
        "int",
        "multiline",
        "password",
        "string",
        "text",
    };
    return supported.find(type) != supported.end();
}

bool IsChoiceParameter(const cyxwiz::ParameterDefinition& parameter) {
    return parameter.type == "enum" || parameter.type == "dropdown";
}

void CheckParameterSchema(const cyxwiz::NodeMetadata& metadata) {
    std::set<std::string> names;
    for (const auto& parameter : metadata.parameters) {
        const std::string context =
            TypeId(metadata.type) + "." + parameter.name;
        Check(!parameter.name.empty(),
              "metadata parameter name is empty for node " +
                  TypeId(metadata.type));
        Check(names.insert(parameter.name).second,
              "duplicate metadata parameter " + context);
        Check(IsSupportedParameterType(parameter.type),
              "unsupported metadata parameter type " + context + ": " +
                  parameter.type);

        std::set<std::string> choices;
        for (const auto& choice : parameter.enum_values) {
            Check(!choice.empty(), "empty metadata choice " + context);
            Check(choices.insert(choice).second,
                  "duplicate metadata choice " + context + "=" + choice);
        }

        if (IsChoiceParameter(parameter)) {
            Check(!parameter.enum_values.empty(),
                  "choice parameter has no options: " + context);
            Check(parameter.default_value.empty() ||
                      choices.count(parameter.default_value) == 1,
                  "choice default is not an option: " + context + "=" +
                      parameter.default_value);
        } else {
            Check(parameter.enum_values.empty(),
                  "non-choice parameter declares options: " + context);
        }
    }
}

void CheckPortSchema(const cyxwiz::NodeMetadata& metadata) {
    const auto check_ports = [&](const std::vector<cyxwiz::PortDefinition>& ports,
                                 const char* direction) {
        std::set<std::string> names;
        for (const auto& port : ports) {
            Check(!port.name.empty(),
                  std::string(direction) + " port name is empty for node " +
                      TypeId(metadata.type));
            Check(names.insert(port.name).second,
                  std::string("duplicate ") + direction + " port " +
                      TypeId(metadata.type) + "." + port.name);
        }
    };
    check_ports(metadata.inputs, "input");
    check_ports(metadata.outputs, "output");
}

void CheckStaticCreationContract(const cyxwiz::NodeMetadata& metadata) {
    // Blocked/template entries may retain legacy saved-graph compatibility
    // shapes without advertising a constructible production contract.
    // PluginCustom is populated from a runtime plugin-provided schema.
    if (!metadata.IsImplemented() ||
        !cyxwiz::CanAddNodeToGraph(metadata) ||
        metadata.type == gui::NodeType::PluginCustom) {
        return;
    }

    int next_node_id = 700;
    int next_pin_id = 100;
    const gui::MLNode created = gui::NodeEditor::CreateNodeWithIds(
        metadata.type, metadata.name, next_node_id, next_pin_id);

    const std::string context = TypeId(metadata.type);
    Check(created.id == 700 && next_node_id == 701,
          "production creation node-id allocation drift: " + context);
    Check(created.type == metadata.type && created.name == metadata.name,
          "production creation identity drift: " + context);
    Check(created.category == metadata.category,
          "production creation category drift: " + context);
    Check(created.inputs.size() == metadata.inputs.size() &&
              created.outputs.size() == metadata.outputs.size(),
          "production creation pin-count drift: " + context);
    Check(created.parameters.size() == metadata.parameters.size(),
          "production creation seeded an undeclared or duplicate parameter: " +
              context);
    Check(next_pin_id ==
              100 + static_cast<int>(metadata.inputs.size() +
                                     metadata.outputs.size()),
          "production creation pin-id allocation drift: " + context);

    for (std::size_t index = 0; index < metadata.inputs.size(); ++index) {
        const auto& declared = metadata.inputs[index];
        const auto& actual = created.inputs[index];
        Check(actual.id == 100 + static_cast<int>(index) &&
                  actual.name == declared.name &&
                  actual.type == declared.type && actual.is_input &&
                  actual.is_required == declared.required &&
                  actual.is_variadic == declared.variadic &&
                  actual.min_connections == declared.min_connections &&
                  actual.max_connections == declared.max_connections,
              "production creation input-pin drift: " + context + "." +
                  declared.name);
    }

    for (std::size_t index = 0; index < metadata.outputs.size(); ++index) {
        const auto& declared = metadata.outputs[index];
        const auto& actual = created.outputs[index];
        Check(actual.id ==
                  100 + static_cast<int>(metadata.inputs.size() + index) &&
                  actual.name == declared.name &&
                  actual.type == declared.type && !actual.is_input &&
                  actual.is_required == declared.required &&
                  actual.is_variadic == declared.variadic &&
                  actual.min_connections == declared.min_connections &&
                  actual.max_connections == declared.max_connections,
              "production creation output-pin drift: " + context + "." +
                  declared.name);
    }

    for (const auto& parameter : metadata.parameters) {
        const auto created_parameter =
            created.parameters.find(parameter.name);
        Check(created_parameter != created.parameters.end() &&
                  created_parameter->second == parameter.default_value,
              "production creation parameter-default drift: " + context + "." +
                  parameter.name);
    }
}

void CheckPropertyAndSupportContract(const cyxwiz::NodeMetadata& metadata) {
    std::set<std::string> support_axis_names;
    for (const auto& axis : metadata.support_axes) {
        Check(!axis.name.empty() && !axis.value.empty(),
              "support axis name/value is empty for node " +
                  TypeId(metadata.type));
        Check(support_axis_names.insert(axis.name).second,
              "duplicate support axis " + TypeId(metadata.type) + "." +
                  axis.name);
    }

    const bool declared_block = metadata.IsTemplate() ||
                                metadata.badge == "Blocked" ||
                                cyxwiz::IsNodeSupportBlocked(metadata);
    if (declared_block) {
        Check(!cyxwiz::CanAddNodeToGraph(metadata),
              "blocked node remains addable: " + TypeId(metadata.type));
    }

    if (!metadata.IsImplemented()) {
        return;
    }

    const auto property_path =
        gui::properties_contract::ClassifyPanelContractPath(
            metadata.type, &metadata);
    Check(std::string(gui::properties_contract::PanelContractPathName(
              property_path)) != "unknown",
          "implemented node has no property path: " +
              TypeId(metadata.type));
    if (property_path ==
        gui::properties_contract::PanelContractPath::MetadataRenderer) {
        Check(!metadata.parameters.empty(),
              "metadata property path has no schema: " +
                  TypeId(metadata.type));
    }
    for (const auto& parameter : metadata.parameters) {
        if (parameter.consumption !=
            cyxwiz::ParameterConsumption::UiOnly) {
            continue;
        }
        Check(!parameter.required,
              "UI-only parameter cannot be runtime-required: " +
                  TypeId(metadata.type) + "." + parameter.name);
        Check(property_path ==
                  gui::properties_contract::PanelContractPath::DialogOnly ||
                  property_path ==
                      gui::properties_contract::PanelContractPath::CustomEditor,
              "UI-only parameter has no explicit UI owner: " +
                  TypeId(metadata.type) + "." + parameter.name);
    }

    const auto* owner = FindSupportAxis(metadata, "Implementation Owner");
    const auto* state = FindSupportAxis(metadata, "Support State");
    Check(owner != nullptr,
          "implemented node has no declared implementation owner: " +
              TypeId(metadata.type));
    Check(state != nullptr,
          "implemented node has no declared support state: " +
              TypeId(metadata.type));
    Check(owner->value != "none" && owner->value != "unknown" &&
              owner->value != "unowned_training_workflow",
          "implemented node declares no executable/UI owner: " +
              TypeId(metadata.type) + "=" + owner->value);
    Check(state->value != "blocked" && cyxwiz::CanAddNodeToGraph(metadata),
          "implemented production node is blocked or not addable: " +
              TypeId(metadata.type));
}

void CheckTrainingCapabilityContract(
    const cyxwiz::NodeMetadataRegistry& registry) {
    std::set<int> unsupported_types;
    const auto check_unsupported = [&](const auto& capabilities) {
        for (const auto& capability : capabilities) {
            const int identity = static_cast<int>(capability.node_type);
            Check(unsupported_types.insert(identity).second,
                  "duplicate unsupported training capability: " +
                      TypeId(capability.node_type));
            Check(registry.GetMetadata(capability.node_type) != nullptr,
                  "unsupported training capability has no metadata: " +
                      TypeId(capability.node_type));
            Check(capability.reason != nullptr &&
                      std::string(capability.reason).size() > 16,
                  "unsupported training capability has no useful reason: " +
                      TypeId(capability.node_type));
            Check(capability.primitive_evidence !=
                      cyxwiz::PipelineBackendPrimitiveEvidence::NotApplicable,
                  "unsupported training capability has no primitive evidence: " +
                      TypeId(capability.node_type));
            const auto resolved =
                cyxwiz::ResolvePipelineTrainingBackendSupport(
                    capability.node_type);
            Check(resolved.primitive_evidence ==
                      capability.primitive_evidence,
                  "resolved primitive evidence drifted: " +
                      TypeId(capability.node_type));
        }
    };
    check_unsupported(
        cyxwiz::GetPipelineUnsupportedSequentialModelLayerCapabilities());
    check_unsupported(
        cyxwiz::GetPipelineUnsupportedTrainingControlCapabilities());
    check_unsupported(
        cyxwiz::GetPipelineUnsupportedTrainingWorkflowCapabilities());

    std::map<int, cyxwiz::PipelineTrainingSupportRole> roles_by_type;
    for (const auto& capability :
         cyxwiz::GetPipelineSupportedTrainingRoleCapabilities()) {
        const int identity = static_cast<int>(capability.node_type);
        Check(roles_by_type.emplace(identity, capability.role).second,
              "duplicate supported training role: " +
                  TypeId(capability.node_type));
        Check(registry.GetMetadata(capability.node_type) != nullptr,
              "training role has no registered metadata: " +
                  TypeId(capability.node_type));
        Check(capability.reason != nullptr &&
                  std::string(capability.reason).size() > 16,
              "training role has no useful ownership reason: " +
                  TypeId(capability.node_type));
    }

    std::set<int> backend_types;
    for (const auto& capability :
         cyxwiz::GetPipelineSupportedTrainingBackendCapabilities()) {
        const int identity = static_cast<int>(capability.node_type);
        Check(backend_types.insert(identity).second,
              "duplicate supported training backend node: " +
                  TypeId(capability.node_type));
        const auto role = roles_by_type.find(identity);
        Check(role != roles_by_type.end(),
              "training backend node has no declared role: " +
                  TypeId(capability.node_type));
        Check(role->second == cyxwiz::PipelineTrainingSupportRole::ModelLayer ||
                  role->second ==
                      cyxwiz::PipelineTrainingSupportRole::Activation,
              "non-backend training role leaked into backend support: " +
                  TypeId(capability.node_type));
    }

    for (const auto& [identity, role] : roles_by_type) {
        if (role == cyxwiz::PipelineTrainingSupportRole::ModelLayer ||
            role == cyxwiz::PipelineTrainingSupportRole::Activation) {
            Check(backend_types.count(identity) == 1,
                  "backend execution role is absent from backend support: " +
                      std::to_string(identity));
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    std::filesystem::path inventory_output;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        Check(argument == "--inventory-json",
              "unknown command-line argument: " + argument);
        Check(index + 1 < argc,
              "--inventory-json requires an output path");
        inventory_output = argv[++index];
    }

    auto& registry = cyxwiz::NodeMetadataRegistry::Instance();
    registry.Initialize();

    const auto all_metadata = registry.GetAllMetadata();
    Check(!all_metadata.empty(), "node metadata registry is empty");
    Check(all_metadata.size() == registry.GetNodeCount(),
          "all-metadata traversal does not cover the registry exactly once");

    std::set<int> identities;
    int previous_identity = -1;
    std::size_t implemented_count = 0;
    std::size_t blocked_count = 0;
    std::size_t resource_template_count = 0;
    int next_resource_template_identity = 10000;
    for (const auto* metadata : all_metadata) {
        Check(metadata != nullptr, "all-metadata traversal returned null");
        const int identity = static_cast<int>(metadata->type);
        Check(identity > previous_identity,
              "all-metadata traversal is not strictly identity ordered");
        previous_identity = identity;
        Check(identities.insert(identity).second,
              "node identity appears more than once: " +
                  std::to_string(identity));
        Check(metadata->type != gui::NodeType::Unknown,
              "registered node uses the Unknown identity");
        Check(metadata->category != gui::NodeCategory::Unknown,
              "registered node has no category: " + TypeId(metadata->type));
        Check(!metadata->name.empty(),
              "registered node has no name: " + TypeId(metadata->type));

        if (identity >= static_cast<int>(gui::NodeType::Unknown)) {
            Check(identity == next_resource_template_identity++,
                  "resource template runtime identities are not contiguous: " +
                      std::to_string(identity));
            Check(metadata->IsTemplate(),
                  "resource template runtime identity is not fail-closed: " +
                      std::to_string(identity));
            ++resource_template_count;
        }

        CheckParameterSchema(*metadata);
        CheckPortSchema(*metadata);
        CheckStaticCreationContract(*metadata);
        CheckPropertyAndSupportContract(*metadata);

        if (metadata->IsImplemented()) {
            ++implemented_count;
        }
        if (!cyxwiz::CanAddNodeToGraph(*metadata)) {
            ++blocked_count;
        }
    }

    Check(implemented_count > 0, "registry has no implemented nodes");
    Check(blocked_count > 0, "registry has no fail-closed nodes");
    CheckTrainingCapabilityContract(registry);
    CheckGraphImportNameContract(registry);

    cyxwiz::test::NodeContractInventorySummary inventory_summary;
    const std::string inventory =
        cyxwiz::test::BuildNodeContractInventoryJson(
            all_metadata, inventory_summary);
    Check(!inventory.empty(), "generated node inventory is empty");
    Check(inventory_summary.node_count == all_metadata.size(),
          "generated node inventory does not cover the registry exactly once");
    Check(inventory_summary.implemented_count == implemented_count,
          "generated node inventory implemented count drifted");
    Check(inventory_summary.blocked_count == blocked_count,
          "generated node inventory blocked count drifted");
    Check(inventory_summary.resource_template_count == resource_template_count,
          "generated node inventory resource-template count drifted");
    Check(inventory_summary.unassigned_owning_part_count == 0,
          "generated node inventory contains an unassigned owning Part");
    Check(inventory_summary.unclassified_workflow_lane_count == 0,
          "generated node inventory contains an unclassified workflow lane");
    Check(inventory_summary.graph_import_name_count ==
              gui::GetNodeTypeImportNames().size(),
          "generated node inventory graph-import name count drifted");
    Check(inventory_summary.legacy_compatibility_import_name_count == 3,
          "generated inventory compatibility import count drifted");
    Check(inventory_summary.unclassified_frontend_primitive_gap_count == 0,
          "generated inventory has an unclassified frontend primitive gap");

    if (!inventory_output.empty()) {
        std::string error;
        Check(cyxwiz::test::WriteNodeContractInventoryJson(
                  inventory_output, inventory, error),
              error);
        std::cout << "Generated node inventory: "
                  << inventory_output.string() << '\n';
    }

    std::cout << "Node contract gates passed: nodes=" << all_metadata.size()
              << " implemented=" << implemented_count
              << " blocked=" << blocked_count << '\n';
    return 0;
}
