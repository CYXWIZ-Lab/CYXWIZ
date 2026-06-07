#include "../src/core/node_executors/pipeline_operator_factory.h"
#include "../src/core/node_metadata_registry.h"
#include "../src/core/pipeline_runtime_capabilities.h"

#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

std::string TypeId(gui::NodeType type) {
    return std::to_string(static_cast<int>(type));
}

bool HasInput(const cyxwiz::NodeMetadata* meta,
              const std::string& name,
              bool required) {
    if (!meta) return false;
    for (const auto& input : meta->inputs) {
        if (input.name == name && input.required == required) {
            return true;
        }
    }
    return false;
}

bool HasInputType(const cyxwiz::NodeMetadata* meta,
                  const std::string& name,
                  gui::PinType type) {
    if (!meta) return false;
    for (const auto& input : meta->inputs) {
        if (input.name == name && input.type == type) {
            return true;
        }
    }
    return false;
}

bool HasOutputType(const cyxwiz::NodeMetadata* meta,
                   const std::string& name,
                   gui::PinType type) {
    if (!meta) return false;
    for (const auto& output : meta->outputs) {
        if (output.name == name && output.type == type) {
            return true;
        }
    }
    return false;
}

bool HasParameter(const cyxwiz::NodeMetadata* meta,
                  const std::string& name) {
    if (!meta) return false;
    for (const auto& param : meta->parameters) {
        if (param.name == name) {
            return true;
        }
    }
    return false;
}

bool HasEnumValue(const cyxwiz::NodeMetadata* meta,
                  const std::string& param_name,
                  const std::string& enum_value) {
    if (!meta) return false;
    for (const auto& param : meta->parameters) {
        if (param.name != param_name) continue;
        for (const auto& value : param.enum_values) {
            if (value == enum_value) {
                return true;
            }
        }
    }
    return false;
}

} // namespace

int main() {
    auto& metadata = cyxwiz::NodeMetadataRegistry::Instance();
    metadata.Initialize();

    auto& factory = cyxwiz::PipelineOperatorFactory::Instance();
    const auto supported = factory.GetSupportedTypes();
    Check(!supported.empty(), "PipelineOperatorFactory should register operators");

    {
        std::set<std::string> operator_names;
        for (const auto& capability : cyxwiz::GetPipelineOperatorRuntimeCapabilities()) {
            Check(operator_names.insert(capability.legacy_type_name).second,
                  std::string("duplicate operator runtime capability: ") +
                      capability.legacy_type_name);
        }

        std::set<std::string> fail_closed_names;
        for (const auto& capability : cyxwiz::GetPipelineFailClosedRuntimeCapabilities()) {
            Check(fail_closed_names.insert(capability.legacy_type_name).second,
                  std::string("duplicate fail-closed runtime capability: ") +
                      capability.legacy_type_name);
        }

        std::set<std::string> legacy_names;
        for (const auto& capability : cyxwiz::GetPipelineLegacyRuntimeCapabilities()) {
            Check(legacy_names.insert(capability.legacy_type_name).second,
                  std::string("duplicate legacy runtime capability: ") +
                      capability.legacy_type_name);
        }

        std::set<std::string> source_names;
        for (const auto& capability : cyxwiz::GetPipelineSourceRuntimeCapabilities()) {
            Check(source_names.insert(capability.legacy_type_name).second,
                  std::string("duplicate source runtime capability: ") +
                      capability.legacy_type_name);
        }

        std::set<std::string> input_arity_names;
        for (const auto& capability : cyxwiz::GetPipelineInputArityRuntimeCapabilities()) {
            Check(input_arity_names.insert(capability.legacy_type_name).second,
                  std::string("duplicate input-arity runtime capability: ") +
                      capability.legacy_type_name);
        }

        std::set<std::string> required_parameter_names;
        for (const auto& capability :
             cyxwiz::GetPipelineRequiredParameterRuntimeCapabilities()) {
            Check(required_parameter_names.insert(capability.legacy_type_name).second,
                  std::string("duplicate required-parameter runtime capability: ") +
                      capability.legacy_type_name);
        }

        std::set<std::string> allowed_parameter_names;
        for (const auto& capability :
             cyxwiz::GetPipelineAllowedParameterValuesRuntimeCapabilities()) {
            const std::string key =
                std::string(capability.legacy_type_name) + "." +
                capability.parameter_name;
            Check(allowed_parameter_names.insert(key).second,
                  "duplicate allowed-parameter runtime capability: " + key);
        }

        std::set<std::string> integer_parameter_names;
        for (const auto& capability :
             cyxwiz::GetPipelineIntegerParameterRuntimeCapabilities()) {
            const std::string key =
                std::string(capability.legacy_type_name) + "." +
                capability.parameter_name;
            Check(integer_parameter_names.insert(key).second,
                  "duplicate integer-parameter runtime capability: " + key);
        }

        std::set<int> unsupported_training_layer_types;
        for (const auto& capability :
             cyxwiz::GetPipelineUnsupportedSequentialModelLayerCapabilities()) {
            const int key = static_cast<int>(capability.node_type);
            Check(unsupported_training_layer_types.insert(key).second,
                  "duplicate unsupported training layer capability: " +
                      TypeId(capability.node_type));
            Check(capability.reason != nullptr &&
                      std::string(capability.reason).size() > 16,
                  "unsupported training layer reason is too weak: " +
                      TypeId(capability.node_type));
            Check(cyxwiz::IsPipelineUnsupportedSequentialModelLayer(
                      capability.node_type),
                  "unsupported training layer capability does not resolve: " +
                      TypeId(capability.node_type));
            const auto support = cyxwiz::ResolvePipelineTrainingBackendSupport(
                capability.node_type);
            Check(support.mode ==
                      cyxwiz::PipelineTrainingBackendSupportMode::
                          UnsupportedSequentialModelLayer,
                  "unsupported training layer should resolve through unified support: " +
                      TypeId(capability.node_type));
            Check(!support.compile_supported && !support.training_supported,
                  "unsupported training layer should block compile/training: " +
                      TypeId(capability.node_type));
            Check(support.reason == capability.reason,
                  "unsupported training layer reason should be shared: " +
                      TypeId(capability.node_type));
        }

        std::set<int> unsupported_training_control_types;
        for (const auto& capability :
             cyxwiz::GetPipelineUnsupportedTrainingControlCapabilities()) {
            const int key = static_cast<int>(capability.node_type);
            Check(unsupported_training_control_types.insert(key).second,
                  "duplicate unsupported training control capability: " +
                      TypeId(capability.node_type));
            Check(capability.reason != nullptr &&
                      std::string(capability.reason).size() > 16,
                  "unsupported training control reason is too weak: " +
                      TypeId(capability.node_type));
            Check(cyxwiz::IsPipelineUnsupportedTrainingControlNode(
                      capability.node_type),
                  "unsupported training control capability does not resolve: " +
                      TypeId(capability.node_type));
            const auto support = cyxwiz::ResolvePipelineTrainingBackendSupport(
                capability.node_type);
            Check(support.mode ==
                      cyxwiz::PipelineTrainingBackendSupportMode::
                          UnsupportedTrainingControl,
                  "unsupported training control should resolve through unified support: " +
                      TypeId(capability.node_type));
            Check(!support.compile_supported && !support.training_supported,
                  "unsupported training control should block compile/training: " +
                      TypeId(capability.node_type));
            Check(support.reason == capability.reason,
                  "unsupported training control reason should be shared: " +
                      TypeId(capability.node_type));
        }

        std::set<int> materializer_storage_backends;
        int materializer_supported_backends = 0;
        for (const auto& capability :
             cyxwiz::GetPipelineMaterializerStorageBackendCapabilities()) {
            const int key = static_cast<int>(capability.backend);
            Check(materializer_storage_backends.insert(key).second,
                  "duplicate materializer storage backend capability: " +
                      std::string(cyxwiz::PipelineStorageBackendName(
                          capability.backend)));
            const auto resolved =
                cyxwiz::ResolvePipelineMaterializerStorageBackendSupport(
                    capability.backend);
            Check(resolved.backend == capability.backend,
                  "materializer storage backend capability does not resolve: " +
                      std::string(cyxwiz::PipelineStorageBackendName(
                          capability.backend)));
            Check(resolved.materializer_supported ==
                      capability.materializer_supported,
                  "materializer storage backend support mismatch: " +
                      std::string(cyxwiz::PipelineStorageBackendName(
                          capability.backend)));
            Check(resolved.storage_support == capability.storage_support,
                  "materializer storage support scope mismatch: " +
                      std::string(cyxwiz::PipelineStorageBackendName(
                          capability.backend)));
            Check(capability.reason != nullptr &&
                      std::string(capability.reason).size() > 16,
                  "materializer storage backend reason is too weak: " +
                      std::string(cyxwiz::PipelineStorageBackendName(
                          capability.backend)));
            if (capability.materializer_supported) {
                ++materializer_supported_backends;
                Check(capability.backend == cyxwiz::PipelineStorageBackend::ArrowTable,
                      "only ArrowTable should be materializer-supported today");
                Check(capability.storage_support ==
                          cyxwiz::PipelineMaterializerStorageSupport::ArrowTableOnly,
                      "ArrowTable materializer support should be ArrowTableOnly");
            } else {
                Check(capability.storage_support ==
                          cyxwiz::PipelineMaterializerStorageSupport::None,
                      "unsupported materializer backend should advertise no storage scope");
            }
        }
        Check(materializer_supported_backends == 1,
              "exactly one materializer storage backend should be supported today");
    }

    for (auto type : supported) {
        auto op = factory.Create(type);
        Check(op != nullptr, "factory returned null for type " + TypeId(type));

        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr, "missing metadata for factory type " + TypeId(type));
        Check(meta->status == cyxwiz::NodeImplementationStatus::Implemented,
              "factory type " + TypeId(type) + " is not marked implemented");
        Check(meta->category != gui::NodeCategory::Unknown,
              "factory type " + TypeId(type) + " has unknown category");
    }

    for (const auto& capability : cyxwiz::GetPipelineOperatorRuntimeCapabilities()) {
        Check(factory.HasOperator(capability.node_type),
              std::string("runtime capability has no factory operator: ") +
                  capability.legacy_type_name);
        auto resolved = cyxwiz::ResolvePipelineOperatorRuntimeType(
            capability.legacy_type_name);
        Check(resolved.has_value(),
              std::string("runtime capability does not resolve: ") +
                  capability.legacy_type_name);
        Check(*resolved == capability.node_type,
              std::string("runtime capability resolves to wrong type: ") +
                  capability.legacy_type_name);
        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(
            capability.legacy_type_name);
        Check(support.mode == cyxwiz::PipelineRuntimeSupportMode::OperatorBacked,
              std::string("runtime support mode is not operator-backed: ") +
                  capability.legacy_type_name);
        Check(support.fail_mode == cyxwiz::PipelineRuntimeFailMode::Real,
              std::string("operator-backed runtime should advertise real fail mode: ") +
                  capability.legacy_type_name);
        Check(support.operator_type.has_value() &&
                  *support.operator_type == capability.node_type,
              std::string("runtime support operator type mismatch: ") +
                  capability.legacy_type_name);
        Check(support.node_type.has_value() &&
                  *support.node_type == capability.node_type,
              std::string("runtime support node type mismatch: ") +
                  capability.legacy_type_name);
        const auto runtime_node_type =
            cyxwiz::ResolvePipelineRuntimeNodeType(capability.legacy_type_name);
        Check(runtime_node_type.has_value() &&
                  *runtime_node_type == capability.node_type,
              std::string("runtime name should resolve to operator node type: ") +
                  capability.legacy_type_name);
        const auto enum_support =
            cyxwiz::ResolvePipelineRuntimeSupport(capability.node_type);
        Check(enum_support.mode ==
                  cyxwiz::PipelineRuntimeSupportMode::OperatorBacked,
              std::string("operator node type should resolve support by enum: ") +
                  capability.legacy_type_name);
        Check(support.materializer_arrow_table_supported,
              std::string("operator-backed runtime should be Arrow-materializer capable: ") +
                  capability.legacy_type_name);
        Check(support.pipeline_executor_supported,
              std::string("operator-backed runtime should be pipeline-executor supported: ") +
                  capability.legacy_type_name);
        Check(support.implementation_owner ==
                  cyxwiz::PipelineRuntimeImplementationOwner::
                      PipelineOperatorFactory,
              std::string("operator-backed runtime should be owned by PipelineOperatorFactory: ") +
                  capability.legacy_type_name);
        Check(support.materializer_storage_support ==
                  cyxwiz::PipelineMaterializerStorageSupport::ArrowTableOnly,
              std::string("operator-backed runtime should advertise Arrow-only materializer scope: ") +
                  capability.legacy_type_name);
        const auto* meta = metadata.GetMetadata(capability.node_type);
        Check(meta != nullptr,
              std::string("operator-backed runtime metadata missing: ") +
                  capability.legacy_type_name);
        Check(meta->help_text.find(cyxwiz::PipelineRuntimeSupportModeName(
                  support.mode)) != std::string::npos,
              std::string("operator-backed metadata should expose runtime support mode: ") +
                  capability.legacy_type_name);
        Check(meta->help_text.find(cyxwiz::PipelineRuntimeFailModeName(
                  support.fail_mode)) != std::string::npos,
              std::string("operator-backed metadata should expose fail mode: ") +
                  capability.legacy_type_name);
        Check(meta->help_text.find("pipeline_executor=supported") !=
                  std::string::npos,
              std::string("operator-backed metadata should expose pipeline executor support: ") +
                  capability.legacy_type_name);
        Check(meta->help_text.find(cyxwiz::PipelineMaterializerStorageSupportName(
                  support.materializer_storage_support)) != std::string::npos,
              std::string("operator-backed metadata should expose materializer support scope: ") +
                  capability.legacy_type_name);
        Check(meta->help_text.find(cyxwiz::PipelineRuntimeImplementationOwnerName(
                  support.implementation_owner)) != std::string::npos,
              std::string("operator-backed metadata should expose implementation owner: ") +
                  capability.legacy_type_name);
    }

    for (const auto& capability : cyxwiz::GetPipelineFailClosedRuntimeCapabilities()) {
        Check(!cyxwiz::IsPipelineOperatorRuntimeNode(capability.legacy_type_name),
              std::string("fail-closed runtime name is also operator-backed: ") +
                  capability.legacy_type_name);
        Check(!cyxwiz::IsPipelineLegacyRuntimeNode(capability.legacy_type_name),
              std::string("fail-closed runtime name is also legacy-dispatched: ") +
                  capability.legacy_type_name);
        Check(cyxwiz::ResolvePipelineFailClosedReason(capability.legacy_type_name) != nullptr,
              std::string("fail-closed runtime name does not resolve: ") +
                  capability.legacy_type_name);
        Check(capability.reason != nullptr && std::string(capability.reason).size() > 8,
              std::string("fail-closed runtime reason is too weak: ") +
                  capability.legacy_type_name);
        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(
            capability.legacy_type_name);
        Check(support.mode == cyxwiz::PipelineRuntimeSupportMode::FailClosed,
              std::string("runtime support mode is not fail-closed: ") +
                  capability.legacy_type_name);
        Check(support.fail_mode == cyxwiz::PipelineRuntimeFailMode::HardFail,
              std::string("fail-closed runtime should advertise hard-fail mode: ") +
                  capability.legacy_type_name);
        Check(support.fail_closed_reason != nullptr,
              std::string("runtime support fail-closed reason missing: ") +
                  capability.legacy_type_name);
        auto expected_metadata_type = capability.metadata_node_type;
        if (!expected_metadata_type.has_value()) {
            expected_metadata_type = capability.node_type;
        }
        Check(support.metadata_node_type == expected_metadata_type,
              std::string("runtime support fail-closed metadata node mismatch: ") +
                  capability.legacy_type_name);
        Check(support.node_type == capability.node_type,
              std::string("runtime support fail-closed node type mismatch: ") +
                  capability.legacy_type_name);
        const auto runtime_node_type =
            cyxwiz::ResolvePipelineRuntimeNodeType(capability.legacy_type_name);
        Check(runtime_node_type == expected_metadata_type,
              std::string("fail-closed runtime name should resolve typed metadata: ") +
                  capability.legacy_type_name);
        Check(!support.materializer_arrow_table_supported,
              std::string("fail-closed runtime should not be Arrow-materializer capable: ") +
                  capability.legacy_type_name);
        Check(!support.pipeline_executor_supported,
              std::string("fail-closed runtime should not be pipeline-executor supported: ") +
                  capability.legacy_type_name);
        Check(support.implementation_owner ==
                  cyxwiz::PipelineRuntimeImplementationOwner::None,
              std::string("fail-closed runtime should advertise no implementation owner: ") +
                  capability.legacy_type_name);
        Check(support.materializer_storage_support ==
                  cyxwiz::PipelineMaterializerStorageSupport::None,
              std::string("fail-closed runtime should not advertise materializer scope: ") +
                  capability.legacy_type_name);
    }

    for (const auto& capability : cyxwiz::GetPipelineLegacyRuntimeCapabilities()) {
        Check(!cyxwiz::IsPipelineOperatorRuntimeNode(capability.legacy_type_name),
              std::string("legacy runtime name is also operator-backed: ") +
                  capability.legacy_type_name);
        Check(!cyxwiz::IsPipelineFailClosedRuntimeNode(capability.legacy_type_name),
              std::string("legacy runtime name is also fail-closed: ") +
                  capability.legacy_type_name);
        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(
            capability.legacy_type_name);
        Check(support.mode == cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
              std::string("runtime support mode is not legacy executor: ") +
                  capability.legacy_type_name);
        Check(support.fail_mode == cyxwiz::PipelineRuntimeFailMode::Real,
              std::string("legacy-dispatched runtime should advertise real fail mode: ") +
                  capability.legacy_type_name);
        Check(support.node_type == capability.node_type,
              std::string("legacy-dispatched runtime node type mismatch: ") +
                  capability.legacy_type_name);
        const auto runtime_node_type =
            cyxwiz::ResolvePipelineRuntimeNodeType(capability.legacy_type_name);
        Check(runtime_node_type == capability.node_type,
              std::string("legacy runtime name should resolve typed metadata: ") +
                  capability.legacy_type_name);
        Check(!support.materializer_arrow_table_supported,
              std::string("legacy-dispatched runtime should not claim Arrow materializer support: ") +
                  capability.legacy_type_name);
        Check(support.pipeline_executor_supported,
              std::string("legacy-dispatched runtime should be pipeline-executor supported: ") +
                  capability.legacy_type_name);
        Check(support.implementation_owner ==
                  cyxwiz::PipelineRuntimeImplementationOwner::PipelineExecutor,
              std::string("legacy-dispatched runtime should be owned by PipelineExecutor: ") +
                  capability.legacy_type_name);
        Check(support.materializer_storage_support ==
                  cyxwiz::PipelineMaterializerStorageSupport::None,
              std::string("legacy-dispatched runtime should not advertise materializer scope: ") +
                  capability.legacy_type_name);
    }

    Check(std::string(cyxwiz::PipelineRuntimeSupportModeName(
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor)) ==
              "legacy_executor",
          "runtime support mode name for legacy executor is stable");
    Check(std::string(cyxwiz::PipelineRuntimeSupportModeName(
              cyxwiz::PipelineRuntimeSupportMode::OperatorBacked)) ==
              "operator_backed",
          "runtime support mode name for operator-backed is stable");
    Check(std::string(cyxwiz::PipelineRuntimeSupportModeName(
              cyxwiz::PipelineRuntimeSupportMode::FailClosed)) ==
              "fail_closed",
          "runtime support mode name for fail-closed is stable");

    Check(std::string(cyxwiz::PipelineRuntimeFailModeName(
              cyxwiz::PipelineRuntimeFailMode::Real)) == "real",
          "runtime fail-mode name for real is stable");
    Check(std::string(cyxwiz::PipelineRuntimeFailModeName(
              cyxwiz::PipelineRuntimeFailMode::HardFail)) == "hard_fail",
          "runtime fail-mode name for hard_fail is stable");
    Check(std::string(cyxwiz::PipelineRuntimeFailModeName(
              cyxwiz::PipelineRuntimeFailMode::Simulated)) == "simulated",
          "runtime fail-mode name for simulated is stable");
    Check(std::string(cyxwiz::PipelineRuntimeFailModeName(
              cyxwiz::PipelineRuntimeFailMode::Passthrough)) == "passthrough",
          "runtime fail-mode name for passthrough is stable");

    Check(std::string(cyxwiz::PipelineRuntimeImplementationOwnerName(
              cyxwiz::PipelineRuntimeImplementationOwner::None)) == "none",
          "runtime implementation owner name for none is stable");
    Check(std::string(cyxwiz::PipelineRuntimeImplementationOwnerName(
              cyxwiz::PipelineRuntimeImplementationOwner::PipelineExecutor)) ==
              "pipeline_executor",
          "runtime implementation owner name for pipeline executor is stable");
    Check(std::string(cyxwiz::PipelineRuntimeImplementationOwnerName(
              cyxwiz::PipelineRuntimeImplementationOwner::
                  PipelineOperatorFactory)) == "pipeline_operator_factory",
          "runtime implementation owner name for operator factory is stable");

    Check(std::string(cyxwiz::PipelineMaterializerStorageSupportName(
              cyxwiz::PipelineMaterializerStorageSupport::None)) == "none",
          "materializer storage support name for none is stable");
    Check(std::string(cyxwiz::PipelineMaterializerStorageSupportName(
              cyxwiz::PipelineMaterializerStorageSupport::ArrowTableOnly)) ==
              "arrow_table_only",
          "materializer storage support name for ArrowTableOnly is stable");

    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::FilterRows)) == "FilterRows",
          "legacy runtime enum lookup for FilterRows is stable");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::FilterRows).mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "FilterRows enum support should resolve to legacy executor");
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::StandardScaler)) == "StandardScaler",
          "operator runtime enum lookup for StandardScaler is stable");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::StandardScaler).mode ==
              cyxwiz::PipelineRuntimeSupportMode::OperatorBacked,
          "StandardScaler enum support should resolve to operator-backed");
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::SVMClassifier)) == "SVMClassifier",
          "fail-closed runtime enum lookup for SVMClassifier is stable");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::SVMClassifier).mode ==
              cyxwiz::PipelineRuntimeSupportMode::FailClosed,
          "SVMClassifier enum support should resolve to fail-closed");

    Check(std::string(cyxwiz::PipelineTrainingBackendSupportModeName(
              cyxwiz::PipelineTrainingBackendSupportMode::Allowed)) ==
              "allowed",
          "training backend support mode name for allowed is stable");
    Check(std::string(cyxwiz::PipelineTrainingBackendSupportModeName(
              cyxwiz::PipelineTrainingBackendSupportMode::
                  UnsupportedSequentialModelLayer)) ==
              "unsupported_sequential_model_layer",
          "training backend support mode name for unsupported layer is stable");
    Check(std::string(cyxwiz::PipelineTrainingBackendSupportModeName(
              cyxwiz::PipelineTrainingBackendSupportMode::
                  UnsupportedTrainingControl)) ==
              "unsupported_training_control",
          "training backend support mode name for unsupported control is stable");

    for (const auto& capability : cyxwiz::GetPipelineSourceRuntimeCapabilities()) {
        Check(cyxwiz::IsPipelineSourceRuntimeNode(capability.legacy_type_name),
              std::string("source runtime name does not resolve: ") +
                  capability.legacy_type_name);
        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(
            capability.legacy_type_name);
        Check(support.mode != cyxwiz::PipelineRuntimeSupportMode::Unknown,
              std::string("source runtime name has unknown support: ") +
                  capability.legacy_type_name);
        Check(support.source_node,
              std::string("source runtime support should carry source-node axis: ") +
                  capability.legacy_type_name);
    }

    for (const auto& capability : cyxwiz::GetPipelineInputArityRuntimeCapabilities()) {
        const auto required_count = cyxwiz::ResolvePipelineRequiredInputCount(
            capability.legacy_type_name);
        Check(required_count.has_value(),
              std::string("input arity runtime name does not resolve: ") +
                  capability.legacy_type_name);
        Check(*required_count == capability.required_input_count,
              std::string("input arity runtime count mismatch: ") +
                  capability.legacy_type_name);
        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(
            capability.legacy_type_name);
        Check(support.mode != cyxwiz::PipelineRuntimeSupportMode::Unknown,
              std::string("input arity runtime name has unknown support: ") +
                  capability.legacy_type_name);
        Check(support.required_input_count.has_value() &&
                  *support.required_input_count ==
                      capability.required_input_count,
              std::string("runtime support should carry input-arity axis: ") +
                  capability.legacy_type_name);
        Check(capability.required_input_count > 1,
              std::string("input arity override should only list multi-input nodes: ") +
                  capability.legacy_type_name);
    }

    for (const auto& capability :
         cyxwiz::GetPipelineRequiredParameterRuntimeCapabilities()) {
        const auto required_parameters = cyxwiz::ResolvePipelineRequiredParameters(
            capability.legacy_type_name);
        Check(!required_parameters.empty(),
              std::string("required-parameter runtime name does not resolve: ") +
                  capability.legacy_type_name);
        Check(required_parameters.size() == capability.required_parameters.size(),
              std::string("required-parameter count mismatch: ") +
                  capability.legacy_type_name);
        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(
            capability.legacy_type_name);
        Check(support.mode != cyxwiz::PipelineRuntimeSupportMode::Unknown,
              std::string("required-parameter runtime name has unknown support: ") +
                  capability.legacy_type_name);
        Check(support.required_parameters.size() ==
                  capability.required_parameters.size(),
              std::string("runtime support should carry required-parameter axis: ") +
                  capability.legacy_type_name);
        for (const char* parameter : capability.required_parameters) {
            Check(parameter != nullptr && std::string(parameter).size() > 1,
                  std::string("required parameter name is too weak: ") +
                      capability.legacy_type_name);
        }
    }

    for (const auto& capability :
         cyxwiz::GetPipelineAllowedParameterValuesRuntimeCapabilities()) {
        const auto allowed_parameters =
            cyxwiz::ResolvePipelineAllowedParameterValues(
                capability.legacy_type_name);
        Check(!allowed_parameters.empty(),
              std::string("allowed-parameter runtime name does not resolve: ") +
                  capability.legacy_type_name);
        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(
            capability.legacy_type_name);
        Check(support.mode != cyxwiz::PipelineRuntimeSupportMode::Unknown,
              std::string("allowed-parameter runtime name has unknown support: ") +
                  capability.legacy_type_name);
        auto supported_axis = std::find_if(
            support.allowed_parameter_values.begin(),
            support.allowed_parameter_values.end(),
            [&capability](
                const cyxwiz::PipelineAllowedParameterValuesRuntimeCapability&
                    runtime_capability) {
                return std::string(runtime_capability.parameter_name) ==
                       capability.parameter_name;
            });
        Check(supported_axis != support.allowed_parameter_values.end(),
              std::string("runtime support should carry allowed-parameter axis: ") +
                  capability.legacy_type_name + "." + capability.parameter_name);
        Check(capability.parameter_name != nullptr &&
                  std::string(capability.parameter_name).size() > 1,
              std::string("allowed parameter name is too weak: ") +
                  capability.legacy_type_name);
        Check(capability.default_value != nullptr,
              std::string("allowed parameter default is missing: ") +
                  capability.legacy_type_name);
        Check(!capability.allowed_values.empty(),
              std::string("allowed parameter value list is empty: ") +
                  capability.legacy_type_name);
    }

    for (const auto& capability :
         cyxwiz::GetPipelineIntegerParameterRuntimeCapabilities()) {
        const auto integer_parameters =
            cyxwiz::ResolvePipelineIntegerParameters(capability.legacy_type_name);
        Check(!integer_parameters.empty(),
              std::string("integer-parameter runtime name does not resolve: ") +
                  capability.legacy_type_name);
        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(
            capability.legacy_type_name);
        Check(support.mode != cyxwiz::PipelineRuntimeSupportMode::Unknown,
              std::string("integer-parameter runtime name has unknown support: ") +
                  capability.legacy_type_name);
        auto supported_axis = std::find_if(
            support.integer_parameters.begin(),
            support.integer_parameters.end(),
            [&capability](
                const cyxwiz::PipelineIntegerParameterRuntimeCapability&
                    runtime_capability) {
                return std::string(runtime_capability.parameter_name) ==
                           capability.parameter_name &&
                       runtime_capability.minimum == capability.minimum &&
                       runtime_capability.comma_separated ==
                           capability.comma_separated;
            });
        Check(supported_axis != support.integer_parameters.end(),
              std::string("runtime support should carry integer-parameter axis: ") +
                  capability.legacy_type_name + "." + capability.parameter_name);
        Check(capability.parameter_name != nullptr &&
                  std::string(capability.parameter_name).size() > 1,
              std::string("integer parameter name is too weak: ") +
                  capability.legacy_type_name);
    }

    const std::vector<gui::NodeType> supported_model_nodes = {
        gui::NodeType::LSTM,
        gui::NodeType::GRU,
    };
    for (auto type : supported_model_nodes) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr, "missing supported model metadata for type " + TypeId(type));
        Check(meta->status == cyxwiz::NodeImplementationStatus::Implemented,
              "supported model type " + TypeId(type) + " should be marked implemented");
        const auto support = cyxwiz::ResolvePipelineTrainingBackendSupport(type);
        Check(support.mode == cyxwiz::PipelineTrainingBackendSupportMode::Allowed,
              "supported model type should be allowed by training backend support: " +
                  TypeId(type));
        Check(support.compile_supported && support.training_supported,
              "supported model type should not be blocked by training backend support: " +
                  TypeId(type));
    }

    for (const auto& capability :
         cyxwiz::GetPipelineFailClosedRuntimeCapabilities()) {
        if (!capability.metadata_node_type.has_value()) {
            continue;
        }
        const auto* meta = metadata.GetMetadata(*capability.metadata_node_type);
        Check(meta != nullptr,
              std::string("missing fail-closed metadata for runtime name ") +
                  capability.legacy_type_name);
        Check(meta->status == cyxwiz::NodeImplementationStatus::Template,
              std::string("fail-closed runtime metadata should not be marked implemented: ") +
                  capability.legacy_type_name);
        Check(meta->badge == "Blocked",
              std::string("fail-closed runtime metadata should carry blocked badge: ") +
                  capability.legacy_type_name);
        Check(capability.reason != nullptr &&
                  meta->help_text.find(capability.reason) != std::string::npos,
              std::string("fail-closed runtime metadata should expose central reason: ") +
                  capability.legacy_type_name);
    }

    const std::vector<gui::NodeType> unsupported_training_nodes = {
        gui::NodeType::Conv2D,
        gui::NodeType::MaxPool2D,
        gui::NodeType::AvgPool2D,
        gui::NodeType::GlobalMaxPool,
        gui::NodeType::GlobalAvgPool,
        gui::NodeType::ConvTranspose2D,
        gui::NodeType::Upsample,
        gui::NodeType::PixelShuffle,
    };
    for (auto type : unsupported_training_nodes) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr,
              "missing unsupported training metadata for type " + TypeId(type));
        Check(meta->status == cyxwiz::NodeImplementationStatus::Template,
              "unsupported training type " + TypeId(type) +
                  " should not be marked implemented");
    }

    const auto* compare = metadata.GetMetadata(gui::NodeType::TensorCompare);
    Check(compare != nullptr, "missing TensorCompare metadata");
    Check(HasInput(compare, "A", true),
          "TensorCompare should expose required A input");
    Check(HasInput(compare, "B", false),
          "TensorCompare should expose optional B input");
    Check(HasEnumValue(compare, "op", "=="),
          "TensorCompare should expose tensor compare operators");

    const auto* logical = metadata.GetMetadata(gui::NodeType::TensorLogicalMask);
    Check(logical != nullptr, "missing TensorLogicalMask metadata");
    Check(HasInput(logical, "A", true),
          "TensorLogicalMask should expose required A input");
    Check(HasInput(logical, "B", false),
          "TensorLogicalMask should expose optional B input");
    Check(HasEnumValue(logical, "op", "not"),
          "TensorLogicalMask should keep unary not");
    Check(HasEnumValue(logical, "op", "and") &&
          HasEnumValue(logical, "op", "or"),
          "TensorLogicalMask should expose binary and/or");

    const auto* ts_window = metadata.GetMetadata(gui::NodeType::TimeSeriesWindow);
    Check(ts_window != nullptr, "missing TimeSeriesWindow metadata");
    Check(HasInputType(ts_window, "Data", gui::PinType::Dataset),
          "TimeSeriesWindow should expose Dataset input");
    Check(HasOutputType(ts_window, "Windowed", gui::PinType::Dataset),
          "TimeSeriesWindow should expose one windowed Dataset output");
    Check(HasParameter(ts_window, "value_col") &&
          HasParameter(ts_window, "feature_cols") &&
          HasParameter(ts_window, "time_col") &&
          HasParameter(ts_window, "input_width") &&
          HasParameter(ts_window, "label_width") &&
          HasParameter(ts_window, "shift"),
          "TimeSeriesWindow should expose canonical operator parameters");

    const auto* ts_features = metadata.GetMetadata(gui::NodeType::TimeSeriesFeatures);
    Check(ts_features != nullptr, "missing TimeSeriesFeatures metadata");
    Check(HasInputType(ts_features, "Data", gui::PinType::Dataset),
          "TimeSeriesFeatures should expose Dataset input");
    Check(HasOutputType(ts_features, "Enriched", gui::PinType::Dataset),
          "TimeSeriesFeatures should expose one enriched Dataset output");
    Check(HasParameter(ts_features, "value_col") &&
          HasParameter(ts_features, "lag_values") &&
          HasParameter(ts_features, "rolling_windows") &&
          HasParameter(ts_features, "rolling_aggregations"),
          "TimeSeriesFeatures should expose canonical operator parameters");

    const auto* ts_split = metadata.GetMetadata(gui::NodeType::TimeSeriesSplit);
    Check(ts_split != nullptr, "missing TimeSeriesSplit metadata");
    Check(HasInputType(ts_split, "Data", gui::PinType::Dataset),
          "TimeSeriesSplit should expose Dataset input");
    Check(HasOutputType(ts_split, "Partitioned", gui::PinType::Dataset),
          "TimeSeriesSplit should expose one partitioned Dataset output");
    Check(HasParameter(ts_split, "train_ratio") &&
          HasParameter(ts_split, "val_ratio") &&
          HasParameter(ts_split, "test_ratio"),
          "TimeSeriesSplit should expose train/val/test ratio parameters");

    std::cout << "Pipeline operator metadata drift guard passed\n";
    return 0;
}
