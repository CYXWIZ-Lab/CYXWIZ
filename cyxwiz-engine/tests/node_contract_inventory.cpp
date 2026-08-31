#include "node_contract_inventory.h"

#include "../src/core/node_metadata.h"
#include "../src/core/pipeline_runtime_capabilities.h"
#include "../src/gui/node_type_import_registry.h"
#include "../src/gui/properties_contract.h"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string_view>
#include <system_error>

#include <nlohmann/json.hpp>

namespace cyxwiz::test {
namespace {

using Json = nlohmann::ordered_json;

const SupportAxisDefinition* FindAxis(const NodeMetadata& metadata,
                                      std::string_view name) {
    for (const auto& axis : metadata.support_axes) {
        if (axis.name == name) return &axis;
    }
    return nullptr;
}

std::string AxisValue(const NodeMetadata& metadata,
                      std::string_view name,
                      std::string fallback = "unknown") {
    const auto* axis = FindAxis(metadata, name);
    return axis == nullptr ? std::move(fallback) : axis->value;
}

const char* ImplementationStatusName(NodeImplementationStatus status) {
    switch (status) {
        case NodeImplementationStatus::Implemented: return "implemented";
        case NodeImplementationStatus::Template: return "template";
        case NodeImplementationStatus::Deprecated: return "deprecated";
        case NodeImplementationStatus::External: return "external";
    }
    return "unknown";
}

const char* ParameterConsumptionName(ParameterConsumption consumption) {
    switch (consumption) {
        case ParameterConsumption::Runtime: return "runtime";
        case ParameterConsumption::UiOnly: return "ui_only";
    }
    return "unknown";
}

const char* PinTypeName(gui::PinType type) {
    switch (type) {
        case gui::PinType::Tensor: return "tensor";
        case gui::PinType::Labels: return "labels";
        case gui::PinType::Parameters: return "parameters";
        case gui::PinType::Loss: return "loss";
        case gui::PinType::Optimizer: return "optimizer";
        case gui::PinType::Dataset: return "dataset";
    }
    return "unknown";
}

std::string OwningPart(const NodeMetadata& metadata) {
    const std::string lane = AxisValue(metadata, "Workflow Lane", "");
    if (lane == "classic_ml") return "E";
    if (lane == "deep_learning") return "F";
    if (lane == "simulation") return "H";

    switch (metadata.category) {
        case gui::NodeCategory::DataSources:
        case gui::NodeCategory::Database:
        case gui::NodeCategory::CloudStorage:
        case gui::NodeCategory::DataTransform:
        case gui::NodeCategory::Preprocessing:
        case gui::NodeCategory::DataPipeline:
        case gui::NodeCategory::TextProcessing:
        case gui::NodeCategory::TimeSeries:
        case gui::NodeCategory::Audio:
        case gui::NodeCategory::JsonXml:
        case gui::NodeCategory::BigData:
            return "D";
        case gui::NodeCategory::Analytics:
        case gui::NodeCategory::Explainability:
            return "E";
        case gui::NodeCategory::Layers:
        case gui::NodeCategory::Activation:
        case gui::NodeCategory::Pooling:
        case gui::NodeCategory::Normalization:
        case gui::NodeCategory::Attention:
        case gui::NodeCategory::Recurrent:
        case gui::NodeCategory::ShapeOps:
        case gui::NodeCategory::MergeOps:
        case gui::NodeCategory::Upsampling:
        case gui::NodeCategory::DNN:
            return "F";
        case gui::NodeCategory::Training:
        case gui::NodeCategory::Regularization:
            return "G";
        case gui::NodeCategory::Visualization:
        case gui::NodeCategory::ModelIO:
        case gui::NodeCategory::MLServices:
        case gui::NodeCategory::RL:
        case gui::NodeCategory::Workflow:
        case gui::NodeCategory::Widgets:
        case gui::NodeCategory::Reporting:
        case gui::NodeCategory::Utility:
        case gui::NodeCategory::Signal:
        case gui::NodeCategory::Plugin:
            return "H";
        case gui::NodeCategory::Unknown:
            return "unassigned";
    }
    return "unassigned";
}

std::string TriageClass(
    const NodeMetadata& metadata,
    PipelineBackendPrimitiveEvidence primitive_evidence) {
    const std::string owner = AxisValue(metadata, "Implementation Owner");
    const std::string state = AxisValue(metadata, "Support State");
    if (metadata.IsDeprecated()) return "duplicate_or_legacy_alias";
    if (metadata.status == NodeImplementationStatus::External ||
        metadata.category == gui::NodeCategory::Plugin) {
        return "external_or_plugin_owned";
    }
    if (owner == "ui_only") return "ui_only_workflow_surface";
    switch (primitive_evidence) {
        case PipelineBackendPrimitiveEvidence::ProvenNodePrimitive:
            return "blocked_backend_primitive_engine_wiring_missing";
        case PipelineBackendPrimitiveEvidence::RelatedHelperOnly:
            return "blocked_related_helper_not_node_primitive";
        case PipelineBackendPrimitiveEvidence::Missing:
            return "blocked_backend_primitive_missing";
        case PipelineBackendPrimitiveEvidence::NotApplicable:
            break;
    }
    if (metadata.IsImplemented() && CanAddNodeToGraph(metadata)) {
        return state == "partial" ? "partially_implemented_bounded_subset"
                                  : "production_and_truthful";
    }
    if (owner == "training_backend") {
        return "blocked_training_contract_incomplete";
    }
    if (owner != "unknown" && owner != "none" &&
        owner != "unowned_training_workflow") {
        return "partially_implemented_bounded_subset";
    }
    if (FindAxis(metadata, "Implementation Owner") != nullptr ||
        FindAxis(metadata, "Support State") != nullptr ||
        FindAxis(metadata, "Runtime") != nullptr ||
        FindAxis(metadata, "Training Backend") != nullptr) {
        return "frontend_exists_backend_or_owner_missing";
    }
    return "speculative_catalog_no_proven_owner";
}

std::string TargetDisposition(std::string_view triage_class) {
    if (triage_class == "production_and_truthful" ||
        triage_class == "ui_only_workflow_surface") {
        return "retain_and_regression_guard";
    }
    if (triage_class == "duplicate_or_legacy_alias") {
        return "map_to_canonical_or_remove_with_migration";
    }
    if (triage_class == "external_or_plugin_owned") {
        return "keep_provider_owned_and_fail_closed_without_provider";
    }
    if (triage_class ==
        "blocked_backend_primitive_engine_wiring_missing") {
        return "integrate_only_after_node_level_correctness_and_runtime_evidence";
    }
    if (triage_class == "blocked_related_helper_not_node_primitive") {
        return "design_and_prove_a_complete_node_owner_before_integration";
    }
    if (triage_class == "blocked_backend_primitive_missing") {
        return "validate_workflow_value_before_implementing_a_new_primitive";
    }
    if (triage_class == "speculative_catalog_no_proven_owner") {
        return "review_workflow_value_before_implementation";
    }
    return "keep_blocked_until_owner_and_required_evidence_exist";
}

Json PortJson(const PortDefinition& port) {
    return {{"name", port.name}, {"type", PinTypeName(port.type)},
            {"required", port.required}, {"variadic", port.variadic},
            {"min_connections", port.min_connections},
            {"max_connections", port.max_connections},
            {"description", port.description}};
}

Json ParameterJson(const ParameterDefinition& parameter) {
    return {{"name", parameter.name}, {"type", parameter.type},
            {"default", parameter.default_value},
            {"display_name", parameter.display_name},
            {"group", parameter.group}, {"required", parameter.required},
            {"advanced", parameter.advanced},
            {"consumption", ParameterConsumptionName(parameter.consumption)},
            {"validation", parameter.validation},
            {"choices", parameter.enum_values},
            {"description", parameter.description}};
}

Json SupportAxisJson(const SupportAxisDefinition& axis) {
    return {{"name", axis.name}, {"value", axis.value},
            {"supported", axis.supported}, {"reason", axis.reason}};
}

std::map<int, std::string> TrainingRoles() {
    std::map<int, std::string> result;
    for (const auto& capability :
         GetPipelineSupportedTrainingRoleCapabilities()) {
        result.emplace(static_cast<int>(capability.node_type),
                       PipelineTrainingSupportRoleName(capability.role));
    }
    return result;
}

std::map<int, Json> LegacyAliases() {
    std::map<int, Json> result;
    for (const auto& capability :
         GetPipelineLegacyAliasDecisionCapabilities()) {
        result[static_cast<int>(capability.canonical_node_type)].push_back({
            {"alias", capability.alias_type_name},
            {"canonical", capability.canonical_type_name},
            {"decision", PipelineLegacyAliasDecisionName(capability.decision)},
            {"reason", capability.reason == nullptr ? "" : capability.reason}});
    }
    return result;
}

std::map<int, Json> GraphImportNames() {
    std::map<int, Json> result;
    for (const auto& entry : gui::GetNodeTypeImportNames()) {
        result[static_cast<int>(entry.node_type)].push_back(
            std::string(entry.name));
    }
    return result;
}

Json StringOrNull(const std::string& value) {
    return value.empty() ? Json(nullptr) : Json(value);
}

} // namespace

std::string BuildNodeContractInventoryJson(
    const std::vector<const NodeMetadata*>& nodes,
    NodeContractInventorySummary& summary) {
    summary = {};
    const int serialized_type_limit = static_cast<int>(gui::NodeType::Unknown);
    const auto training_roles = TrainingRoles();
    const auto legacy_aliases = LegacyAliases();
    const auto graph_import_names = GraphImportNames();
    summary.graph_import_name_count = gui::GetNodeTypeImportNames().size();
    Json graph_import_registry = Json::array();
    std::set<int> registered_identities;
    for (const auto* metadata : nodes) {
        if (metadata != nullptr) {
            registered_identities.insert(static_cast<int>(metadata->type));
        }
    }
    for (const auto& entry : gui::GetNodeTypeImportNames()) {
        if (entry.legacy_import_compatibility_only) {
            ++summary.legacy_compatibility_import_name_count;
        }
        const int target_identity = static_cast<int>(entry.node_type);
        graph_import_registry.push_back({
            {"name", std::string(entry.name)},
            {"target_numeric_identity", target_identity},
            {"target_metadata_registered",
             registered_identities.count(target_identity) == 1},
            {"scope", entry.legacy_import_compatibility_only
                 ? "legacy_import_compatibility_only"
                 : "registered_pattern_import"}});
    }
    Json rows = Json::array();
    std::map<std::string, std::size_t> status_counts;
    std::map<std::string, std::size_t> triage_counts;
    std::map<std::string, std::size_t> owning_part_counts;

    for (const auto* metadata : nodes) {
        if (metadata == nullptr) continue;
        const int identity = static_cast<int>(metadata->type);
        const bool resource_template = identity >= serialized_type_limit;
        const bool blocked = !CanAddNodeToGraph(*metadata);
        const std::string status = ImplementationStatusName(metadata->status);
        const std::string owning_part = OwningPart(*metadata);
        const std::string workflow_lane =
            AxisValue(*metadata, "Workflow Lane", "unclassified");
        const auto pipeline = ResolvePipelineRuntimeSupport(metadata->type);
        const auto training = ResolvePipelineTrainingBackendSupport(metadata->type);
        const auto primitive_evidence =
            training.primitive_evidence !=
                    PipelineBackendPrimitiveEvidence::NotApplicable
                ? training.primitive_evidence
                : pipeline.primitive_evidence;
        const std::string triage_class =
            TriageClass(*metadata, primitive_evidence);
        const auto property_path =
            gui::properties_contract::ClassifyPanelContractPath(
                metadata->type, metadata);

        ++summary.node_count;
        if (metadata->IsImplemented()) ++summary.implemented_count;
        if (blocked) ++summary.blocked_count;
        if (resource_template) ++summary.resource_template_count;
        if (owning_part == "unassigned") {
            ++summary.unassigned_owning_part_count;
        }
        if (workflow_lane == "unclassified") {
            ++summary.unclassified_workflow_lane_count;
        }
        if (metadata->example_usage.empty()) {
            ++summary.missing_representative_workflow_count;
        }
        if (triage_class == "frontend_exists_backend_or_owner_missing") {
            ++summary.unclassified_frontend_primitive_gap_count;
        }
        ++status_counts[status];
        ++triage_counts[triage_class];
        ++owning_part_counts[owning_part];

        Json inputs = Json::array();
        for (const auto& input : metadata->inputs) inputs.push_back(PortJson(input));
        Json outputs = Json::array();
        for (const auto& output : metadata->outputs) outputs.push_back(PortJson(output));
        Json parameters = Json::array();
        Json declared_runtime_parameters = Json::array();
        Json actually_consumed_parameters = Json::array();
        Json unproven_runtime_parameters = Json::array();
        Json ui_parameters = Json::array();
        for (const auto& parameter : metadata->parameters) {
            parameters.push_back(ParameterJson(parameter));
            if (parameter.consumption == ParameterConsumption::UiOnly) {
                ui_parameters.push_back(parameter.name);
                continue;
            }
            declared_runtime_parameters.push_back(parameter.name);
            (metadata->IsImplemented()
                 ? actually_consumed_parameters
                 : unproven_runtime_parameters).push_back(parameter.name);
        }
        Json axes = Json::array();
        for (const auto& axis : metadata->support_axes) {
            axes.push_back(SupportAxisJson(axis));
        }

        Json focused_tests = Json::array({"test_node_contract_gates"});
        if (pipeline.mode != PipelineRuntimeSupportMode::Unknown) {
            focused_tests.push_back("test_pipeline_operator_metadata");
        }
        const auto role = training_roles.find(identity);
        const bool training_declared =
            role != training_roles.end() ||
            IsPipelineSupportedTrainingBackendNode(metadata->type) ||
            IsPipelineUnsupportedSequentialModelLayer(metadata->type) ||
            IsPipelineUnsupportedTrainingControlNode(metadata->type) ||
            IsPipelineUnsupportedTrainingWorkflowNode(metadata->type);
        if (role != training_roles.end()) {
            focused_tests.push_back("test_graph_compiler_deferred_nodes");
        }
        Json aliases = Json::array();
        const auto alias = legacy_aliases.find(identity);
        if (alias != legacy_aliases.end()) aliases = alias->second;
        Json accepted_import_names = Json::array();
        const auto import_names = graph_import_names.find(identity);
        if (import_names != graph_import_names.end()) {
            accepted_import_names = import_names->second;
        }

        const std::string factory_contract =
            metadata->type == gui::NodeType::PluginCustom
                ? "dynamic_provider_schema"
                : metadata->IsImplemented() && CanAddNodeToGraph(*metadata)
                    ? "verified_by_test_node_contract_gates"
                    : "saved_graph_or_catalog_compatibility_only";

        rows.push_back({
            {"identity", {{"numeric", identity},
                          {"kind", resource_template
                              ? "resource_template_runtime_id"
                              : "serialized_node_type"},
                          {"symbolic", metadata->name},
                          {"symbolic_source", resource_template
                              ? "resource_template_name"
                              : "canonical_metadata_name"}}},
            {"name", metadata->name},
            {"category", GetCategoryDisplayName(metadata->category)},
            {"workflow_lane", workflow_lane},
            {"owning_part", owning_part},
            {"metadata_status", status}, {"badge", metadata->badge},
            {"can_add_to_graph", CanAddNodeToGraph(*metadata)},
            {"triage_class", triage_class},
            {"target_disposition", TargetDisposition(triage_class)},
            {"properties", {{"path",
                             gui::properties_contract::PanelContractPathName(
                                  property_path)},
                             {"declared", parameters},
                             {"declared_runtime", declared_runtime_parameters},
                             {"actually_consumed", actually_consumed_parameters},
                             {"unproven_runtime", unproven_runtime_parameters},
                             {"ui_only", ui_parameters}}},
            {"pins", {{"factory_contract", factory_contract},
                      {"inputs", inputs}, {"outputs", outputs}}},
            {"execution", {
                {"declared_owner", AxisValue(
                    *metadata, "Implementation Owner")},
                {"support_state", AxisValue(*metadata, "Support State")},
                {"pipeline", {{"mode", PipelineRuntimeSupportModeName(
                                           pipeline.mode)},
                              {"fail_mode", PipelineRuntimeFailModeName(
                                                pipeline.fail_mode)},
                              {"implementation_owner",
                               PipelineRuntimeImplementationOwnerName(
                                   pipeline.implementation_owner)},
                              {"pipeline_executor",
                               pipeline.pipeline_executor_supported},
                              {"source_node", pipeline.source_node},
                              {"materializer",
                               PipelineMaterializerStorageSupportName(
                                   pipeline.materializer_storage_support)},
                              {"failure_reason",
                               pipeline.fail_closed_reason == nullptr
                                   ? "" : pipeline.fail_closed_reason},
                              {"backend_primitive_evidence",
                               PipelineBackendPrimitiveEvidenceName(
                                   pipeline.primitive_evidence)}}},
                {"training", {{"declared", training_declared},
                              {"role", role == training_roles.end()
                                           ? "none" : role->second},
                              {"backend_mode", training_declared
                                   ? PipelineTrainingBackendSupportModeName(
                                         training.mode)
                                   : "not_declared"},
                              {"backend_primitive_evidence",
                               training_declared
                                   ? PipelineBackendPrimitiveEvidenceName(
                                         training.primitive_evidence)
                                   : "not_applicable"},
                              {"compile_supported", training_declared
                                   ? Json(training.compile_supported)
                                   : Json(nullptr)},
                              {"training_supported", training_declared
                                   ? Json(training.training_supported)
                                   : Json(nullptr)},
                              {"reason", training_declared &&
                                             training.reason != nullptr
                                  ? training.reason : ""}}},
                {"device_selection", "runtime_bound_not_static_node_metadata"},
                {"support_axes", axes}}},
            {"serialization", {
                {"numeric_identity_persisted", !resource_template},
                {"graph_import_names", accepted_import_names},
                {"pipeline_legacy_aliases", aliases}}},
            {"evidence", {{"focused_tests", focused_tests},
                          {"representative_workflow",
                           StringOrNull(metadata->example_usage)}}},
            {"documentation", {{"brief", metadata->brief_description},
                               {"help", metadata->help_text}}}
        });
    }

    Json document = {
        {"schema_version", 1},
        {"source_authorities", Json::array({
            "NodeMetadataRegistry::GetAllMetadata",
            "NodeTypeImportRegistry",
            "PipelineRuntimeCapabilities", "PipelineTrainingCapabilities",
            "PropertiesContract"})},
        {"identity_note",
         "Built-in numeric IDs are serialized NodeType identities. Resource "
         "template IDs are deterministic runtime catalog identities beginning "
         "at 10000. Symbolic identity is the canonical metadata name because "
         "the C++ enum token is not exposed as a separate runtime authority."},
        {"graph_import_registry", std::move(graph_import_registry)},
        {"summary", {{"nodes", summary.node_count},
                     {"implemented", summary.implemented_count},
                     {"blocked", summary.blocked_count},
                     {"resource_templates", summary.resource_template_count},
                     {"unassigned_owning_part",
                      summary.unassigned_owning_part_count},
                     {"unclassified_workflow_lane",
                      summary.unclassified_workflow_lane_count},
                     {"missing_representative_workflow",
                      summary.missing_representative_workflow_count},
                     {"graph_import_names",
                      summary.graph_import_name_count},
                     {"legacy_compatibility_import_names",
                      summary.legacy_compatibility_import_name_count},
                     {"unclassified_frontend_primitive_gaps",
                      summary.unclassified_frontend_primitive_gap_count},
                     {"status_counts", status_counts},
                     {"triage_counts", triage_counts},
                     {"owning_part_counts", owning_part_counts}}},
        {"nodes", std::move(rows)}};
    return document.dump(2) + '\n';
}

bool WriteNodeContractInventoryJson(
    const std::filesystem::path& output_path,
    const std::string& json,
    std::string& error) {
    error.clear();
    std::error_code filesystem_error;
    const auto parent = output_path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, filesystem_error);
        if (filesystem_error) {
            error = "cannot create inventory directory: " +
                    filesystem_error.message();
            return false;
        }
    }
    errno = 0;
    std::ofstream output(
        output_path.string(),
        std::ios::out | std::ios::binary | std::ios::trunc);
    if (!output) {
        error = "cannot open inventory output: " + output_path.string() +
                " (" + std::strerror(errno) + ")";
        return false;
    }
    output.write(json.data(), static_cast<std::streamsize>(json.size()));
    if (!output) {
        error = "cannot write inventory output: " + output_path.string();
        return false;
    }
    return true;
}

} // namespace cyxwiz::test
