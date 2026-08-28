#include "../src/core/node_executors/pipeline_operator_factory.h"
#include "../src/core/node_metadata_registry.h"
#include "../src/core/pipeline_runtime_capabilities.h"
#include "../src/core/simulation_runtime_capabilities.h"
#include "../src/gui/data_studio/pipeline_canvas.h"
#include "../src/gui/data_studio/node_registry.h"
#include "../src/gui/node_import_guardrails.h"
#include "../src/gui/properties_contract.h"
#include "../src/gui/properties_truth.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
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

bool ParameterMatches(const cyxwiz::NodeMetadata* meta,
                      const std::string& name,
                      const std::string& type,
                      const std::string& default_value) {
    if (!meta) return false;
    for (const auto& param : meta->parameters) {
        if (param.name == name) {
            return param.type == type && param.default_value == default_value;
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

bool ContainsString(const std::vector<std::string>& values,
                    const std::string& expected) {
    return std::find(values.begin(), values.end(), expected) != values.end();
}

bool ContainsString(const std::set<std::string>& values,
                    const std::string& expected) {
    return values.find(expected) != values.end();
}

bool ContainsNodeType(const std::vector<gui::NodeType>& values,
                      gui::NodeType expected) {
    return std::find(values.begin(), values.end(), expected) != values.end();
}

bool SearchContainsType(cyxwiz::NodeMetadataRegistry& metadata,
                        const std::string& query,
                        gui::NodeType expected) {
    const auto results = metadata.Search(query, true);
    return std::any_of(
        results.begin(),
        results.end(),
        [expected](const cyxwiz::NodeMetadata* meta) {
            return meta != nullptr && meta->type == expected;
        });
}

const cyxwiz::ParameterDefinition* FindParameter(
    const cyxwiz::NodeMetadata* meta,
    const std::string& name) {
    if (!meta) return nullptr;
    for (const auto& param : meta->parameters) {
        if (param.name == name) {
            return &param;
        }
    }
    return nullptr;
}

const cyxwiz::SupportAxisDefinition* FindSupportAxis(
    const cyxwiz::NodeMetadata* meta,
    const std::string& name) {
    if (!meta) return nullptr;
    for (const auto& axis : meta->support_axes) {
        if (axis.name == name) {
            return &axis;
        }
    }
    return nullptr;
}

void CheckSupportAxis(const cyxwiz::NodeMetadata* meta,
                      const std::string& name,
                      const std::string& value,
                      bool supported,
                      const std::string& context) {
    const auto* axis = FindSupportAxis(meta, name);
    Check(axis != nullptr, "missing support axis " + name + ": " + context);
    Check(axis->value == value,
          "support axis " + name + " has wrong value for " + context +
              ": " + axis->value);
    Check(axis->supported == supported,
          "support axis " + name + " has wrong supported flag for " +
              context);
}

void CheckSupportAxisReasonContains(const cyxwiz::NodeMetadata* meta,
                                    const std::string& name,
                                    const std::string& expected,
                                    const std::string& context) {
    const auto* axis = FindSupportAxis(meta, name);
    Check(axis != nullptr, "missing support axis " + name + ": " + context);
    Check(!axis->reason.empty(),
          "support axis " + name + " is missing its reason for " + context);
    if (!expected.empty()) {
        Check(axis->reason.find(expected) != std::string::npos,
              "support axis " + name + " reason is incomplete for " + context);
    }
}

std::string FrontendSupportBlockReasonFromAxes(
    const cyxwiz::NodeMetadata* meta) {
    if (!meta) return {};
    const auto* support_state = FindSupportAxis(meta, "Support State");
    if (support_state && support_state->value == "blocked" &&
        !support_state->reason.empty()) {
        return support_state->reason;
    }
    for (const auto& axis : meta->support_axes) {
        if (!axis.supported && !axis.reason.empty()) {
            return axis.reason;
        }
    }
    return {};
}

bool IsExecutableRuntimeSupportMode(
    cyxwiz::PipelineRuntimeSupportMode mode) {
    return mode == cyxwiz::PipelineRuntimeSupportMode::OperatorBacked ||
           mode == cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor;
}

int RuntimeOwnerCount(const cyxwiz::PipelineRuntimeSupport& support) {
    int owners = 0;
    if (support.implementation_owner ==
        cyxwiz::PipelineRuntimeImplementationOwner::PipelineOperatorFactory) {
        ++owners;
    }
    if (support.implementation_owner ==
        cyxwiz::PipelineRuntimeImplementationOwner::PipelineExecutor) {
        ++owners;
    }
    if (support.implementation_owner ==
        cyxwiz::PipelineRuntimeImplementationOwner::None) {
        ++owners;
    }
    return owners;
}

void CheckRuntimeModeOwnerCompatibility(
    const cyxwiz::PipelineRuntimeSupport& support,
    const std::string& context) {
    switch (support.mode) {
    case cyxwiz::PipelineRuntimeSupportMode::OperatorBacked:
        Check(support.implementation_owner ==
                  cyxwiz::PipelineRuntimeImplementationOwner::
                      PipelineOperatorFactory,
              "operator-backed runtime should be owned by "
              "PipelineOperatorFactory: " +
                  context);
        return;
    case cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor:
        Check(support.implementation_owner ==
                  cyxwiz::PipelineRuntimeImplementationOwner::PipelineExecutor,
              "legacy runtime should be owned by PipelineExecutor: " +
                  context);
        return;
    case cyxwiz::PipelineRuntimeSupportMode::FailClosed:
        Check(support.implementation_owner ==
                  cyxwiz::PipelineRuntimeImplementationOwner::None,
              "fail-closed runtime should not have an execution owner: " +
                  context);
        return;
    case cyxwiz::PipelineRuntimeSupportMode::Unknown:
        Check(support.implementation_owner ==
                  cyxwiz::PipelineRuntimeImplementationOwner::Unknown,
              "unknown runtime should keep owner unknown: " + context);
        return;
    }
}

void CheckRuntimeOwnerContract(
    const cyxwiz::PipelineRuntimeSupport& support,
    cyxwiz::PipelineRuntimeSupportMode expected_mode,
    cyxwiz::PipelineRuntimeImplementationOwner expected_owner,
    const std::string& context) {
    Check(support.mode == expected_mode,
          "runtime support mode drift: " + context);
    Check(support.implementation_owner == expected_owner,
          "runtime implementation owner drift: " + context);
    Check(RuntimeOwnerCount(support) == 1,
          "runtime support should resolve exactly one implementation owner: " +
              context);
    Check(support.implementation_owner !=
              cyxwiz::PipelineRuntimeImplementationOwner::Unknown,
          "runtime support should not leave owner unknown: " + context);
    CheckRuntimeModeOwnerCompatibility(support, context);
}

std::vector<std::string> ParseCatalogEnumValues(
    const std::string& parameter_type) {
    std::vector<std::string> values;
    if (parameter_type.rfind("enum:", 0) != 0) {
        return values;
    }

    const std::string raw_values = parameter_type.substr(5);
    std::size_t start = 0;
    while (start <= raw_values.size()) {
        const std::size_t end = raw_values.find(',', start);
        const std::string value = raw_values.substr(
            start,
            end == std::string::npos ? std::string::npos : end - start);
        if (!value.empty()) {
            values.push_back(value);
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return values;
}

bool IsSupportedPropertiesParameterType(const std::string& type) {
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

std::optional<gui::NodeType> ResolveMetadataNodeTypeForRuntimeName(
    const char* legacy_type_name) {
    if (!legacy_type_name) {
        return std::nullopt;
    }
    const auto support =
        cyxwiz::ResolvePipelineRuntimeSupport(legacy_type_name);
    if (support.metadata_node_type.has_value()) {
        return support.metadata_node_type;
    }
    return support.node_type;
}

bool ShouldAuditRuntimeParametersAgainstMetadata(
    cyxwiz::NodeMetadataRegistry& metadata,
    const char* legacy_type_name,
    const cyxwiz::NodeMetadata*& out_meta) {
    if (const auto* alias =
            cyxwiz::ResolvePipelineLegacyAliasDecision(
                legacy_type_name ? legacy_type_name : "")) {
        if (alias->decision ==
            cyxwiz::PipelineLegacyAliasDecision::HiddenCompatibilityAlias) {
            out_meta = nullptr;
            return false;
        }
    }

    const auto node_type = ResolveMetadataNodeTypeForRuntimeName(legacy_type_name);
    if (!node_type.has_value()) {
        out_meta = nullptr;
        return false;
    }

    out_meta = metadata.GetMetadata(*node_type);
    if (!out_meta || out_meta->status != cyxwiz::NodeImplementationStatus::Implemented) {
        return false;
    }

    return gui::properties_contract::ClassifyPanelContractPath(
               out_meta->type, out_meta) ==
           gui::properties_contract::PanelContractPath::MetadataRenderer;
}

void CheckMetadataHasRuntimeParameter(
    cyxwiz::NodeMetadataRegistry& metadata,
    const char* legacy_type_name,
    const char* parameter_name,
    const char* capability_kind) {
    Check(parameter_name != nullptr && std::string(parameter_name).size() > 1,
          std::string("runtime parameter name is too weak: ") +
              (legacy_type_name ? legacy_type_name : "(null)"));

    const cyxwiz::NodeMetadata* meta = nullptr;
    if (!ShouldAuditRuntimeParametersAgainstMetadata(
            metadata, legacy_type_name, meta)) {
        return;
    }

    Check(HasParameter(meta, parameter_name),
          std::string("metadata-rendered node is missing ") +
              capability_kind + " runtime parameter " +
              legacy_type_name + "." + parameter_name +
              " in " + TypeId(meta->type));
}

bool IsBoolAllowedValues(const std::vector<const char*>& values) {
    return values.size() == 2 &&
           std::find(values.begin(), values.end(), std::string("true")) !=
               values.end() &&
           std::find(values.begin(), values.end(), std::string("false")) !=
               values.end();
}

bool IsMetadataEnumType(const std::string& type) {
    return type == "enum" || type == "dropdown";
}

bool IsCommaSeparatedIntegerMetadataType(const std::string& type) {
    return type == "string" || type == "text" || type == "multiline";
}

void CheckMetadataAllowedParameterShape(
    cyxwiz::NodeMetadataRegistry& metadata,
    const cyxwiz::PipelineAllowedParameterValuesRuntimeCapability& capability) {
    CheckMetadataHasRuntimeParameter(
        metadata,
        capability.legacy_type_name,
        capability.parameter_name,
        "allowed-value");

    const cyxwiz::NodeMetadata* meta = nullptr;
    if (!ShouldAuditRuntimeParametersAgainstMetadata(
            metadata, capability.legacy_type_name, meta)) {
        return;
    }

    const auto* param = FindParameter(meta, capability.parameter_name);
    Check(param != nullptr,
          std::string("metadata-rendered node is missing allowed-value "
                      "runtime parameter ") +
              capability.legacy_type_name + "." + capability.parameter_name);

    const bool bool_values = IsBoolAllowedValues(capability.allowed_values);
    if (bool_values && param->type == "bool") {
        Check(param->default_value.empty() ||
                  param->default_value == capability.default_value,
              std::string("metadata bool default drifts from runtime: ") +
                  capability.legacy_type_name + "." +
                  capability.parameter_name);
        return;
    }

    Check(IsMetadataEnumType(param->type),
          std::string("runtime allowed-value parameter should render as enum "
                      "or bool: ") +
              capability.legacy_type_name + "." + capability.parameter_name +
              " metadata type " + param->type);
    for (const char* value : capability.allowed_values) {
        Check(value != nullptr && ContainsString(param->enum_values, value),
              std::string("metadata enum is missing runtime allowed value: ") +
                  capability.legacy_type_name + "." +
                  capability.parameter_name + "=" + (value ? value : "(null)"));
    }
    Check(param->default_value.empty() ||
              param->default_value == capability.default_value,
          std::string("metadata enum default drifts from runtime: ") +
              capability.legacy_type_name + "." + capability.parameter_name +
              " metadata=" + param->default_value +
              " runtime=" +
              (capability.default_value ? capability.default_value : "(null)"));
}

void CheckMetadataIntegerParameterShape(
    cyxwiz::NodeMetadataRegistry& metadata,
    const cyxwiz::PipelineIntegerParameterRuntimeCapability& capability) {
    CheckMetadataHasRuntimeParameter(
        metadata,
        capability.legacy_type_name,
        capability.parameter_name,
        "integer");

    const cyxwiz::NodeMetadata* meta = nullptr;
    if (!ShouldAuditRuntimeParametersAgainstMetadata(
            metadata, capability.legacy_type_name, meta)) {
        return;
    }

    const auto* param = FindParameter(meta, capability.parameter_name);
    Check(param != nullptr,
          std::string("metadata-rendered node is missing integer runtime "
                      "parameter ") +
              capability.legacy_type_name + "." + capability.parameter_name);
    const bool type_matches = capability.comma_separated
                                  ? IsCommaSeparatedIntegerMetadataType(
                                        param->type)
                                  : param->type == "int";
    Check(type_matches,
          std::string("metadata integer parameter has wrong renderer type: ") +
              capability.legacy_type_name + "." + capability.parameter_name +
              " metadata type " + param->type);
}

void CheckMetadataFloatParameterShape(
    cyxwiz::NodeMetadataRegistry& metadata,
    const cyxwiz::PipelineFloatParameterRuntimeCapability& capability) {
    CheckMetadataHasRuntimeParameter(
        metadata,
        capability.legacy_type_name,
        capability.parameter_name,
        "float");

    const cyxwiz::NodeMetadata* meta = nullptr;
    if (!ShouldAuditRuntimeParametersAgainstMetadata(
            metadata, capability.legacy_type_name, meta)) {
        return;
    }

    const auto* param = FindParameter(meta, capability.parameter_name);
    Check(param != nullptr,
          std::string("metadata-rendered node is missing float runtime "
                      "parameter ") +
              capability.legacy_type_name + "." + capability.parameter_name);
    Check(param->type == "float",
          std::string("metadata float parameter has wrong renderer type: ") +
              capability.legacy_type_name + "." + capability.parameter_name +
              " metadata type " + param->type);
}

void CheckPropertyTruthInventory(cyxwiz::NodeMetadataRegistry& metadata) {
    const auto& specialized =
        gui::properties_truth::SpecializedTruthCoverageNodeTypes();
    Check(!specialized.empty(),
          "tofix48 specialized truth coverage inventory should not be empty");

    for (const auto type : specialized) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr,
              "specialized truth coverage should map to registered metadata: " +
                  TypeId(type));
        Check(meta->status == cyxwiz::NodeImplementationStatus::Implemented,
              "specialized truth coverage should only claim implemented nodes: " +
                  TypeId(type));
    }

    std::set<gui::NodeType> seen;
    std::vector<std::string> implemented_nodes_missing_contract;
    std::vector<std::string> implemented_params_with_bad_schema;
    std::size_t dialog_only_count = 0;
    std::size_t custom_sequence_count = 0;
    std::size_t metadata_rendered_count = 0;
    std::size_t custom_fallback_count = 0;
    const std::vector<std::string> generated_help_prefixes = {
        "Product support:",
        "Runtime support:",
        "Training backend support:",
        "Training role:",
        "Task guidance:",
        "Workflow lane:",
    };

    for (const auto category : metadata.GetCategories()) {
        for (const auto* meta : metadata.GetByCategory(category, true)) {
            if (!meta || !seen.insert(meta->type).second) {
                continue;
            }

            for (const auto& prefix : generated_help_prefixes) {
                Check(meta->help_text.find(prefix) == std::string::npos,
                      "authored help must not contain generated support text: " +
                          meta->name + " -> " + prefix);
            }

            std::set<std::string> param_names;
            for (const auto& param : meta->parameters) {
                if (param.name.empty() ||
                    !param_names.insert(param.name).second ||
                    !IsSupportedPropertiesParameterType(param.type)) {
                    implemented_params_with_bad_schema.push_back(
                        meta->name + "(" + TypeId(meta->type) + "):" +
                        param.name + ":" + param.type);
                }
            }

            if (meta->status != cyxwiz::NodeImplementationStatus::Implemented) {
                continue;
            }

            const auto panel_path =
                gui::properties_contract::ClassifyPanelContractPath(
                    meta->type, meta);
            switch (panel_path) {
                case gui::properties_contract::PanelContractPath::DialogOnly:
                    ++dialog_only_count;
                    Check(gui::properties_contract::IsDialogOnlyPropertiesNode(meta->type),
                          "dialog-only property path should be explicit: " +
                              TypeId(meta->type));
                    break;
                case gui::properties_contract::PanelContractPath::CustomSequenceEditor:
                    ++custom_sequence_count;
                    Check(gui::properties_contract::IsCustomSequencePropertiesNode(meta->type),
                          "custom sequence property path should be explicit: " +
                              TypeId(meta->type));
                    break;
                case gui::properties_contract::PanelContractPath::MetadataRenderer:
                    ++metadata_rendered_count;
                    Check(!meta->parameters.empty(),
                          "metadata-rendered property path requires parameters: " +
                              TypeId(meta->type));
                    break;
                case gui::properties_contract::PanelContractPath::CustomFallbackEditor:
                    ++custom_fallback_count;
                    break;
            }

            const bool has_contract_anchor =
                !meta->parameters.empty() ||
                !meta->inputs.empty() ||
                !meta->outputs.empty() ||
                !meta->support_axes.empty() ||
                gui::properties_truth::HasSpecializedTruthCoverage(meta->type);
            if (!has_contract_anchor) {
                implemented_nodes_missing_contract.push_back(
                    meta->name + "(" + TypeId(meta->type) + ")");
            }
        }
    }

    Check(seen.size() == metadata.GetNodeCount(),
          "property truth inventory should visit every metadata node");
    Check(implemented_params_with_bad_schema.empty(),
          "implemented metadata parameters should use unique names and "
          "properties-supported types; first bad entry: " +
              (implemented_params_with_bad_schema.empty()
                   ? std::string()
                   : implemented_params_with_bad_schema.front()));
    Check(implemented_nodes_missing_contract.empty(),
          "implemented metadata nodes should expose at least one property "
          "contract anchor; first missing node: " +
              (implemented_nodes_missing_contract.empty()
                   ? std::string()
                   : implemented_nodes_missing_contract.front()));
    Check(dialog_only_count > 0,
          "property inventory should classify dialog-only nodes");
    Check(custom_sequence_count > 0,
          "property inventory should classify custom sequence nodes");
    Check(metadata_rendered_count > 0,
          "property inventory should classify metadata-rendered nodes");
    Check(custom_fallback_count > 0,
          "property inventory should classify custom fallback nodes");

    const std::vector<gui::NodeType> expected_specialized = {
        gui::NodeType::DataInput,
        gui::NodeType::DataOutput,
        gui::NodeType::DataConvert,
        gui::NodeType::DeployToNodeEditorNode,
        gui::NodeType::DataLoader,
        gui::NodeType::DataProfiler,
        gui::NodeType::StandardScaler,
        gui::NodeType::MinMaxScaler,
        gui::NodeType::RobustScaler,
        gui::NodeType::LabelEncoder,
        gui::NodeType::OrdinalEncoder,
        gui::NodeType::TargetEncoder,
        gui::NodeType::OutlierDetector,
        gui::NodeType::TFIDFVectorizer,
        gui::NodeType::CountVectorizer,
        gui::NodeType::TextTokenizer,
        gui::NodeType::RegressionMetricsNode,
        gui::NodeType::ClassificationMetricsNode,
        gui::NodeType::ConfusionMatrixNode,
        gui::NodeType::ROCCurveNode,
        gui::NodeType::PRCurveNode,
        gui::NodeType::Dense,
        gui::NodeType::TimeDistributed,
        gui::NodeType::Dropout,
        gui::NodeType::BatchNorm,
        gui::NodeType::LayerNorm,
        gui::NodeType::ReLU,
        gui::NodeType::Sigmoid,
        gui::NodeType::Softmax,
        gui::NodeType::GELU,
        gui::NodeType::Tanh,
        gui::NodeType::LeakyReLU,
        gui::NodeType::Flatten,
        gui::NodeType::Reshape,
        gui::NodeType::View,
        gui::NodeType::Permute,
        gui::NodeType::Squeeze,
        gui::NodeType::Unsqueeze,
        gui::NodeType::LSTM,
        gui::NodeType::GRU,
        gui::NodeType::NERSequenceBuilder,
        gui::NodeType::TokenVocabulary,
        gui::NodeType::POSVocabulary,
        gui::NodeType::NERTagVocabulary,
        gui::NodeType::SequenceTagOutput,
        gui::NodeType::MSELoss,
        gui::NodeType::FocalLoss,
        gui::NodeType::BCELoss,
        gui::NodeType::BCEWithLogits,
        gui::NodeType::L1Loss,
        gui::NodeType::SmoothL1Loss,
        gui::NodeType::HuberLoss,
        gui::NodeType::NLLLoss,
        gui::NodeType::SoftDiceLoss,
        gui::NodeType::TverskyLoss,
        gui::NodeType::JaccardLoss,
        gui::NodeType::Adam,
        gui::NodeType::SGD,
        gui::NodeType::AdamW,
        gui::NodeType::RMSprop,
        gui::NodeType::Adagrad,
        gui::NodeType::NAdam,
        gui::NodeType::Output,
        gui::NodeType::CrossEntropyLoss,
        gui::NodeType::ExportCSV,
        gui::NodeType::ExportParquet,
        gui::NodeType::ExportJSON,
        gui::NodeType::TreeModelPredictor,
    };
    for (const auto type : expected_specialized) {
        Check(ContainsNodeType(specialized, type),
              "tofix48 baseline specialized coverage missing node: " +
                  TypeId(type));
    }

    const std::vector<gui::NodeType> representative_metadata_contracts = {
        gui::NodeType::StandardScaler,
        gui::NodeType::MinMaxScaler,
        gui::NodeType::Dense,
        gui::NodeType::Adam,
        gui::NodeType::MSELoss,
        gui::NodeType::ExportCSV,
        gui::NodeType::DataValidator,
    };
    for (const auto type : representative_metadata_contracts) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr,
              "representative property contract metadata missing: " +
                  TypeId(type));
        Check(!meta->parameters.empty() || !meta->support_axes.empty(),
              "representative property contract should expose metadata "
              "parameters or support truth: " +
                  TypeId(type));
    }

    Check(gui::properties_contract::ClassifyPanelContractPath(
              gui::NodeType::DataInput,
              metadata.GetMetadata(gui::NodeType::DataInput)) ==
              gui::properties_contract::PanelContractPath::DialogOnly,
          "DataInput should stay dialog-only in the side panel");
    Check(gui::properties_contract::ClassifyPanelContractPath(
              gui::NodeType::NERSequenceBuilder,
              metadata.GetMetadata(gui::NodeType::NERSequenceBuilder)) ==
              gui::properties_contract::PanelContractPath::CustomSequenceEditor,
          "NERSequenceBuilder should use the custom sequence editor");
    Check(gui::properties_contract::ClassifyPanelContractPath(
              gui::NodeType::DataLoader,
              metadata.GetMetadata(gui::NodeType::DataLoader)) ==
              gui::properties_contract::PanelContractPath::MetadataRenderer,
          "DataLoader should expose metadata-rendered quick parameters");

    const auto* data_loader = metadata.GetMetadata(gui::NodeType::DataLoader);
    Check(data_loader != nullptr,
          "DataLoader must be registered in the modern node catalog");
    Check(HasInputType(data_loader, "Partitions", gui::PinType::Dataset),
          "DataLoader must consume the resolved Dataset partition contract");
    Check(!HasInputType(data_loader, "Data", gui::PinType::Tensor) &&
              !HasInputType(data_loader, "Labels", gui::PinType::Labels),
          "DataLoader metadata must not expose legacy raw Tensor/Labels inputs");
    Check(HasOutputType(data_loader, "Data", gui::PinType::Tensor) &&
              HasOutputType(data_loader, "Labels", gui::PinType::Labels),
          "DataLoader must expose model-facing batched Data/Labels outputs");
    Check(gui::properties_contract::ClassifyPanelContractPath(
              gui::NodeType::ReLU,
              metadata.GetMetadata(gui::NodeType::ReLU)) ==
              gui::properties_contract::PanelContractPath::CustomFallbackEditor,
          "parameterless implemented nodes should route to fallback properties");

    const auto* data_split = metadata.GetMetadata(gui::NodeType::DataSplit);
    Check(data_split != nullptr,
          "DataSplit must be registered in the modern node catalog");
    Check(data_split->status == cyxwiz::NodeImplementationStatus::Implemented,
          "DataSplit must remain addable rather than a template entry");
    Check(HasInputType(data_split, "Training Dataset", gui::PinType::Dataset) &&
              HasInputType(data_split, "Validation Dataset", gui::PinType::Dataset) &&
              HasInputType(data_split, "Test Dataset", gui::PinType::Dataset),
          "DataSplit must expose Dataset-oriented role inputs");
    Check(HasOutputType(data_split, "Partitions", gui::PinType::Dataset),
          "DataSplit must expose one resolved partition-set output");
    Check(HasParameter(data_split, "train_ratio") &&
              HasParameter(data_split, "val_ratio") &&
              HasParameter(data_split, "test_ratio") &&
              HasParameter(data_split, "stratified") &&
              HasParameter(data_split, "seed"),
          "DataSplit metadata must match its configuration dialog");
    Check(SearchContainsType(metadata, "data split", gui::NodeType::DataSplit),
          "DataSplit must be discoverable by the modern node search");
    const std::vector<gui::NodeType> expected_supported_catalog_nodes = {
        gui::NodeType::TransformerEncoder,
        gui::NodeType::TransformerDecoder,
        gui::NodeType::PositionalEncoding,
        gui::NodeType::ELU,
        gui::NodeType::Swish,
        gui::NodeType::Mish,
    };
    for (const auto type : expected_supported_catalog_nodes) {
        const auto* supported = metadata.GetMetadata(type);
        Check(supported != nullptr,
              "supported catalog node must be discoverable: " + TypeId(type));
        Check(supported->status == cyxwiz::NodeImplementationStatus::Implemented,
              "supported catalog node must not be downgraded to a preview: " + TypeId(type));
        Check(SearchContainsType(metadata, supported->name, type),
              "supported catalog node must be searchable: " + TypeId(type));
    }

    const std::vector<gui::NodeType> expected_catalog_previews = {
        gui::NodeType::PReLU,
        gui::NodeType::Resize,
        gui::NodeType::HuggingFaceDataset,
        gui::NodeType::LinePlot,
        gui::NodeType::PluginCustom,
    };
    for (const auto type : expected_catalog_previews) {
        const auto* preview = metadata.GetMetadata(type);
        Check(preview != nullptr,
              "known catalog preview must be discoverable: " + TypeId(type));
        Check(preview->status == cyxwiz::NodeImplementationStatus::Template,
              "known catalog preview must be visibly unavailable: " + TypeId(type));
        Check(SearchContainsType(metadata, preview->name, type),
              "known catalog preview must be searchable: " + TypeId(type));
    }
    const auto* adam = metadata.GetMetadata(gui::NodeType::Adam);
    Check(adam != nullptr, "Adam metadata should exist");
    Check(HasParameter(adam, "learning_rate"),
          "Adam metadata should expose canonical learning_rate");
    Check(!HasParameter(adam, "lr"),
          "Adam metadata should not expose legacy lr as editable");
    const auto* sgd = metadata.GetMetadata(gui::NodeType::SGD);
    Check(sgd != nullptr, "SGD metadata should exist");
    Check(HasParameter(sgd, "learning_rate"),
          "SGD metadata should expose canonical learning_rate");
    Check(!HasParameter(sgd, "lr"),
          "SGD metadata should not expose legacy lr as editable");
    Check(HasParameter(sgd, "momentum"),
          "SGD metadata should expose runtime-consumed momentum");
    const auto* adamw = metadata.GetMetadata(gui::NodeType::AdamW);
    Check(adamw != nullptr, "AdamW metadata should exist");
    Check(HasParameter(adamw, "weight_decay"),
          "AdamW metadata should expose runtime-consumed weight_decay");
    const auto* batch_norm = metadata.GetMetadata(gui::NodeType::BatchNorm);
    Check(batch_norm != nullptr, "BatchNorm metadata should exist");
    Check(HasParameter(batch_norm, "eps"),
          "BatchNorm metadata should expose compiler-consumed eps");
    Check(!HasParameter(batch_norm, "epsilon"),
          "BatchNorm metadata should not expose legacy epsilon as editable");
    const auto* dense = metadata.GetMetadata(gui::NodeType::Dense);
    Check(dense != nullptr, "Dense metadata should exist");
    Check(!HasParameter(dense, "activation"),
          "Dense metadata should not expose inline activation because "
          "ModelBuilder requires explicit activation nodes");

    for (const auto& capability :
         cyxwiz::GetPipelineRequiredParameterRuntimeCapabilities()) {
        for (const char* parameter : capability.required_parameters) {
            CheckMetadataHasRuntimeParameter(
                metadata,
                capability.legacy_type_name,
                parameter,
                "required");
        }
    }

    for (const auto& capability :
         cyxwiz::GetPipelineAllowedParameterValuesRuntimeCapabilities()) {
        CheckMetadataAllowedParameterShape(metadata, capability);
    }

    for (const auto& capability :
         cyxwiz::GetPipelineIntegerParameterRuntimeCapabilities()) {
        CheckMetadataIntegerParameterShape(metadata, capability);
    }

    for (const auto& capability :
         cyxwiz::GetPipelineFloatParameterRuntimeCapabilities()) {
        CheckMetadataFloatParameterShape(metadata, capability);
    }
}

void CheckMultiHeadAttentionReferenceContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    const auto* meta = metadata.GetMetadata(gui::NodeType::MultiHeadAttention);
    Check(meta != nullptr, "MultiHeadAttention metadata should exist");
    Check(meta->status == cyxwiz::NodeImplementationStatus::Implemented,
          "MultiHeadAttention should remain implemented self-attention");

    Check(HasInput(meta, "Query", true),
          "MultiHeadAttention should require one Query sequence input");
    Check(HasInput(meta, "Key", false),
          "MultiHeadAttention should expose reserved optional Key input");
    Check(HasInput(meta, "Value", false),
          "MultiHeadAttention should expose reserved optional Value input");
    Check(HasInput(meta, "Mask", false),
          "MultiHeadAttention should expose reserved optional Mask input");
    Check(HasOutputType(meta, "Output", gui::PinType::Tensor),
          "MultiHeadAttention should expose its runtime tensor output");
    Check(!HasOutputType(meta, "Attn Weights", gui::PinType::Tensor),
          "MultiHeadAttention must not advertise unmaterialized attention "
          "weights");

    Check(ParameterMatches(meta, "embed_dim", "int", "512"),
          "MultiHeadAttention embed_dim contract should match new-node defaults");
    Check(ParameterMatches(meta, "num_heads", "int", "8"),
          "MultiHeadAttention num_heads contract should match new-node defaults");
    Check(ParameterMatches(meta, "dropout", "float", "0.0"),
          "MultiHeadAttention dropout contract should match ModelBuilder input");
    Check(ParameterMatches(meta, "use_bias", "bool", "true"),
          "MultiHeadAttention use_bias contract should match ModelBuilder input");
    Check(!HasParameter(meta, "batch_first"),
          "MultiHeadAttention must not expose the unused batch_first parameter");
}

void CheckDenseReferenceContract(cyxwiz::NodeMetadataRegistry& metadata) {
    const auto* meta = metadata.GetMetadata(gui::NodeType::Dense);
    Check(meta != nullptr, "Dense metadata should exist");
    Check(meta->category == gui::NodeCategory::Layers,
          "Dense metadata should own its layer category");
    Check(HasInput(meta, "Input", true),
          "Dense metadata should require one input tensor");
    Check(HasOutputType(meta, "Output", gui::PinType::Tensor),
          "Dense metadata should expose one output tensor");
    Check(ParameterMatches(meta, "units", "int", "64"),
          "Dense metadata should own the ordinary units default");
    Check(!HasParameter(meta, "activation"),
          "Dense must require an explicit activation node");
    Check(!HasParameter(meta, "use_bias"),
          "Dense must not expose use_bias while ModelBuilder fixes bias on");
}

void CheckStandardScalerReferenceContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    const auto* meta = metadata.GetMetadata(gui::NodeType::StandardScaler);
    Check(meta != nullptr, "StandardScaler metadata should exist");
    Check(meta->category == gui::NodeCategory::Preprocessing,
          "StandardScaler metadata should own its preprocessing category");
    Check(HasInputType(meta, "Data", gui::PinType::Dataset),
          "StandardScaler should require one table input");
    Check(HasOutputType(meta, "Scaled", gui::PinType::Dataset),
          "StandardScaler should expose one scaled table output");
    Check(meta->parameters.size() == 10,
          "StandardScaler metadata should declare every operator parameter");
    Check(ParameterMatches(meta, "columns", "string", "") &&
              ParameterMatches(meta, "label_col", "string", "") &&
              ParameterMatches(meta, "exclude_columns", "string", "") &&
              ParameterMatches(meta, "with_mean", "bool", "true") &&
              ParameterMatches(meta, "with_std", "bool", "true") &&
              ParameterMatches(meta, "transform_role", "enum", "features") &&
              ParameterMatches(meta, "operation_mode", "enum", "fit_transform") &&
              ParameterMatches(meta, "save_state", "bool", "false") &&
              ParameterMatches(meta, "state_path", "file", "") &&
              ParameterMatches(meta, "state_overwrite", "bool", "false"),
          "StandardScaler defaults should match its operator contract");
    CheckSupportAxis(meta, "Implementation Owner", "pipeline_operator_factory",
                     true, "StandardScaler");
}

void CheckTransformerFamilyReferenceContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    for (const auto type : {gui::NodeType::TransformerEncoder,
                            gui::NodeType::TransformerDecoder}) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr && meta->IsImplemented(),
              "Transformer block metadata should be executable: " + TypeId(type));
        Check(ParameterMatches(meta, "d_model", "int", "512") &&
                  ParameterMatches(meta, "num_heads", "int", "8") &&
                  ParameterMatches(meta, "dim_feedforward", "int", "2048") &&
                  ParameterMatches(meta, "dropout", "float", "0.1") &&
                  ParameterMatches(meta, "norm_first", "bool", "false"),
              "Transformer block metadata should expose the five consumed fields: " +
                  TypeId(type));
        Check(!HasParameter(meta, "num_layers") &&
                  !HasParameter(meta, "nhead") &&
                  !HasParameter(meta, "ff_dim") &&
                  !HasParameter(meta, "d_ff"),
              "Transformer block metadata must not expose compatibility aliases: " +
                  TypeId(type));
    }

    const auto* positional =
        metadata.GetMetadata(gui::NodeType::PositionalEncoding);
    Check(positional != nullptr && positional->IsImplemented(),
          "PositionalEncoding metadata should be executable");
    Check(ParameterMatches(positional, "d_model", "int", "512") &&
              ParameterMatches(positional, "max_sequence_length", "int", "5000") &&
              positional->parameters.size() == 2,
          "PositionalEncoding should expose only its two consumed fields");
    Check(!HasParameter(positional, "max_len") &&
              !HasParameter(positional, "dropout"),
          "PositionalEncoding must not expose legacy or ignored fields");

    using Parameters = std::map<std::string, std::string>;
    cyxwiz::TransformerConfiguration resolved;
    Check(!cyxwiz::ResolveTransformerConfiguration(
              gui::NodeType::TransformerEncoder,
              Parameters{{"d_model", "8"}, {"embed_dim", "8"},
                         {"num_heads", "2"}, {"nhead", "2"},
                         {"dim_feedforward", "32"}, {"ff_dim", "32"},
                         {"d_ff", "32"}, {"dropout_rate", "0.25"},
                         {"norm_first", "1"}, {"num_layers", "1"}},
              resolved),
          "equal-valued legacy transformer aliases should remain readable");
    Check(resolved.model_width == 8 && resolved.num_heads == 2 &&
              resolved.feedforward_width == 32 &&
              std::abs(resolved.dropout - 0.25f) < 1e-6f &&
              resolved.norm_first,
          "the shared transformer policy must resolve every accepted alias into construction values");
    Check(cyxwiz::ResolveInvalidTransformerConfigurationReason(
              gui::NodeType::TransformerEncoder,
              Parameters{{"d_model", "8"}, {"nhead", "3"}}).has_value(),
          "non-divisible head configurations must fail closed");
    Check(cyxwiz::ResolveInvalidTransformerConfigurationReason(
              gui::NodeType::TransformerEncoder,
              Parameters{{"d_model", "8"}, {"num_layers", "6"}}).has_value(),
          "a transformer node must not pretend to expand num_layers");
    Check(cyxwiz::ResolveInvalidTransformerConfigurationReason(
              gui::NodeType::MultiHeadAttention,
              Parameters{{"embed_dim", "8"}, {"d_model", "12"}}).has_value(),
          "conflicting attention width aliases must fail closed");
    Check(!cyxwiz::ResolveInvalidTransformerConfigurationReason(
              gui::NodeType::PositionalEncoding,
              Parameters{{"d_model", "8"}, {"max_len", "32"},
                         {"max_length", "32"}}),
          "equal-valued positional-length aliases should remain readable");
    Check(cyxwiz::ResolveInvalidTransformerConfigurationReason(
              gui::NodeType::PositionalEncoding,
              Parameters{{"max_len", "32"}, {"max_length", "64"}}).has_value(),
          "conflicting positional-length aliases must fail closed");
}

void CheckUtilityNodeFamilyContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    const auto* identity = metadata.GetMetadata(gui::NodeType::Identity);
    Check(identity != nullptr &&
              identity->status == cyxwiz::NodeImplementationStatus::Implemented,
          "Identity should remain implemented by its table operator");
    Check(identity->inputs.size() == 1 && identity->outputs.size() == 1 &&
              HasInputType(identity, "Table", gui::PinType::Dataset) &&
              HasOutputType(identity, "Table", gui::PinType::Dataset),
          "Identity should expose its real Arrow-table passthrough pins");
    Check(identity->parameters.empty(),
          "Identity should not expose parameters its operator does not consume");
    Check(!identity->help_text.empty() && cyxwiz::CanAddNodeToGraph(*identity),
          "Identity should describe and allow its operator-backed table path");
    const auto identity_runtime =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::Identity);
    Check(identity_runtime.mode ==
              cyxwiz::PipelineRuntimeSupportMode::OperatorBacked &&
              identity_runtime.implementation_owner ==
                  cyxwiz::PipelineRuntimeImplementationOwner::PipelineOperatorFactory,
          "Identity runtime truth should resolve to PipelineOperatorFactory");
    CheckSupportAxis(identity, "Implementation Owner",
                     "pipeline_operator_factory", true, "Identity");
    CheckSupportAxis(identity, "Workflow Lane",
                     "data_studio_analytics", true, "Identity");
    Check(cyxwiz::PipelineOperatorFactory::Instance().Create(
              gui::NodeType::Identity) != nullptr,
          "Identity should have a registered pipeline operator");

    const auto check_blocked =
        [&](gui::NodeType type,
            const char* legacy_name,
            std::size_t input_count,
            std::size_t output_count) {
            const auto* meta = metadata.GetMetadata(type);
            Check(meta != nullptr &&
                      meta->status == cyxwiz::NodeImplementationStatus::Template &&
                      meta->badge == "Blocked" &&
                      !cyxwiz::CanAddNodeToGraph(*meta),
                  std::string(legacy_name) +
                      " should remain a blocked compatibility contract");
            Check(meta->inputs.size() == input_count &&
                      meta->outputs.size() == output_count &&
                      !meta->help_text.empty(),
                  std::string(legacy_name) +
                      " should preserve pins and explain its missing owner");
            const auto runtime = cyxwiz::ResolvePipelineRuntimeSupport(type);
            Check(runtime.mode == cyxwiz::PipelineRuntimeSupportMode::FailClosed &&
                      runtime.implementation_owner ==
                          cyxwiz::PipelineRuntimeImplementationOwner::None,
                  std::string(legacy_name) +
                      " should resolve to an unowned fail-closed runtime");
            CheckSupportAxis(meta, "Runtime", "fail_closed", false, legacy_name);
            CheckSupportAxis(meta, "Implementation Owner", "none", false,
                             legacy_name);
            CheckSupportAxis(meta, "Support State", "blocked", false,
                             legacy_name);
            Check(cyxwiz::PipelineOperatorFactory::Instance().Create(type) == nullptr,
                  std::string(legacy_name) +
                      " must not gain a fictional pipeline operator");
        };

    check_blocked(gui::NodeType::Lambda, "Lambda", 1, 1);
    const auto* lambda = metadata.GetMetadata(gui::NodeType::Lambda);
    Check(lambda != nullptr && lambda->parameters.size() == 1 &&
              ParameterMatches(lambda, "function", "string", "lambda x: x"),
          "Lambda should preserve its historical function field only");

    check_blocked(gui::NodeType::Parameter, "Parameter", 0, 1);
    const auto* parameter = metadata.GetMetadata(gui::NodeType::Parameter);
    Check(parameter != nullptr && parameter->parameters.size() == 3 &&
              HasOutputType(parameter, "Parameter", gui::PinType::Tensor) &&
              ParameterMatches(parameter, "shape", "string", "256") &&
              ParameterMatches(parameter, "init", "enum", "xavier") &&
              ParameterMatches(parameter, "requires_grad", "bool", "true"),
          "Parameter should preserve its historical blocked saved-graph contract");
}

void CheckSimulationNodeFamilyContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    for (const auto& capability :
         cyxwiz::kBuiltInSimulationRuntimeCapabilities) {
        const auto* meta = metadata.GetMetadata(capability.node_type);
        Check(meta != nullptr &&
                  meta->status ==
                      cyxwiz::NodeImplementationStatus::Implemented &&
                  cyxwiz::CanAddNodeToGraph(*meta),
              std::string(capability.runtime_name) +
                  " should be addable in the simulation lane");
        Check(!meta->help_text.empty(),
              std::string(capability.runtime_name) +
                  " should explain its GraphExecutor contract");
        CheckSupportAxis(meta, "Workflow Lane", "simulation", true,
                         capability.runtime_name);
        CheckSupportAxis(meta, "Simulation Runtime", "supported", true,
                         capability.runtime_name);
        CheckSupportAxis(meta, "Implementation Owner", "graph_executor", true,
                         capability.runtime_name);
        CheckSupportAxis(meta, "Support State", "real", true,
                         capability.runtime_name);
        const auto pipeline =
            cyxwiz::ResolvePipelineRuntimeSupport(capability.node_type);
        Check(pipeline.mode == cyxwiz::PipelineRuntimeSupportMode::FailClosed &&
                  !pipeline.pipeline_executor_supported,
              std::string(capability.runtime_name) +
                  " should remain rejected by PipelineExecutor");
    }

    const auto* constant = metadata.GetMetadata(gui::NodeType::Constant);
    Check(constant && constant->inputs.empty() &&
              HasOutputType(constant, "Value", gui::PinType::Tensor) &&
              constant->parameters.size() == 1 &&
              ParameterMatches(constant, "value", "float", "1.0"),
          "Constant should expose only its scalar simulation value");

    const auto* slider = metadata.GetMetadata(gui::NodeType::SignalSlider);
    Check(slider && slider->inputs.empty() &&
              HasOutputType(slider, "Value", gui::PinType::Tensor) &&
              slider->parameters.size() == 3 &&
              ParameterMatches(slider, "value", "float", "0.0") &&
              ParameterMatches(slider, "min", "float", "-1.0") &&
              ParameterMatches(slider, "max", "float", "1.0"),
          "SignalSlider metadata should match live executor keys and defaults");

    const auto* sine = metadata.GetMetadata(gui::NodeType::SineWave);
    Check(sine && HasOutputType(sine, "Signal", gui::PinType::Tensor) &&
              sine->parameters.size() == 4 &&
              ParameterMatches(sine, "amplitude", "float", "1.0") &&
              ParameterMatches(sine, "frequency", "float", "1.0") &&
              ParameterMatches(sine, "phase", "float", "0.0") &&
              ParameterMatches(sine, "offset", "float", "0.0"),
          "SineWave metadata should match its complete equation");

    const auto* step = metadata.GetMetadata(gui::NodeType::StepSignal);
    Check(step && step->parameters.size() == 3 &&
              ParameterMatches(step, "step_time", "float", "1.0") &&
              ParameterMatches(step, "initial_value", "float", "0.0") &&
              ParameterMatches(step, "final_value", "float", "1.0"),
          "StepSignal metadata should match GraphExecutor parameter keys");

    const auto* ramp = metadata.GetMetadata(gui::NodeType::RampSignal);
    Check(ramp && ramp->parameters.size() == 3 &&
              ParameterMatches(ramp, "start_value", "float", "0.0") &&
              ParameterMatches(ramp, "end_value", "float", "1.0") &&
              ParameterMatches(ramp, "duration", "float", "5.0"),
          "RampSignal metadata should use duration semantics");

    const auto* scope = metadata.GetMetadata(gui::NodeType::SignalScope);
    Check(scope && scope->inputs.size() == 1 && scope->outputs.empty() &&
              HasInputType(scope, "Signal", gui::PinType::Tensor) &&
              scope->parameters.size() == 2 &&
              ParameterMatches(scope, "window_size", "int", "500") &&
              ParameterMatches(scope, "auto_scale", "bool", "true"),
          "SignalScope metadata should describe one live scalar input and its display controls");
}

void CheckDataInputDialogReferenceContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    const auto* meta = metadata.GetMetadata(gui::NodeType::DataInput);
    Check(meta != nullptr, "DataInput metadata should exist");
    Check(gui::properties_contract::ClassifyPanelContractPath(
              gui::NodeType::DataInput, meta) ==
              gui::properties_contract::PanelContractPath::DialogOnly,
          "DataInput dynamic fields should remain dialog-owned");
    Check(meta->category == gui::NodeCategory::DataSources,
          "DataInput metadata should own its data-source category");
    Check(meta->inputs.empty(), "DataInput should not expose static inputs");
    Check(HasOutputType(meta, "Dataset", gui::PinType::Dataset),
          "DataInput should expose one Dataset artifact");
    Check(meta->parameters.size() == 3,
          "DataInput metadata should contain only its static dialog bootstrap fields");
    Check(ParameterMatches(meta, "file_path", "file", "") &&
              ParameterMatches(meta, "file_type", "enum", "auto") &&
              ParameterMatches(meta, "configured", "bool", "false"),
          "DataInput metadata should own its dialog bootstrap defaults");
    Check(!HasParameter(meta, "chunk_size") &&
              !HasParameter(meta, "enable_streaming"),
          "DataInput must not seed removed streaming fields");
}

void CheckConv2DBlockedReferenceContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    const auto* meta = metadata.GetMetadata(gui::NodeType::Conv2D);
    Check(meta != nullptr, "Conv2D metadata should exist");
    Check(meta->status == cyxwiz::NodeImplementationStatus::Template,
          "Conv2D should remain blocked until ModelBuilder supports it");
    Check(!cyxwiz::CanAddNodeToGraph(*meta),
          "central graph-add policy should reject blocked Conv2D");
    Check(HasInputType(meta, "Input", gui::PinType::Tensor) &&
              HasOutputType(meta, "Output", gui::PinType::Tensor),
          "Conv2D metadata should preserve its saved-graph pin contract");
    Check(meta->parameters.size() == 4,
          "Conv2D metadata should expose only its preserved layer fields");
    Check(ParameterMatches(meta, "filters", "int", "32") &&
              ParameterMatches(meta, "kernel_size", "int", "3") &&
              ParameterMatches(meta, "stride", "int", "1") &&
              ParameterMatches(meta, "padding", "enum", "same"),
          "Conv2D metadata defaults should match the preserved graph contract");
    Check(!HasParameter(meta, "activation"),
          "Conv2D must not advertise an unexecuted inline activation");
    CheckSupportAxis(meta, "Training Backend",
                     "unsupported_sequential_model_layer", false, "Conv2D");
    CheckSupportAxis(meta, "Compile", "unsupported", false, "Conv2D");
    CheckSupportAxisReasonContains(
        meta, "Support State", "not supported", "Conv2D");
}

void CheckConvolutionPoolingBlockedFamilyContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    const std::initializer_list<gui::NodeType> family = {
        gui::NodeType::Conv1D,
        gui::NodeType::Conv2D,
        gui::NodeType::Conv3D,
        gui::NodeType::DepthwiseConv2D,
        gui::NodeType::MaxPool2D,
        gui::NodeType::AvgPool2D,
        gui::NodeType::GlobalMaxPool,
        gui::NodeType::GlobalAvgPool,
        gui::NodeType::AdaptiveAvgPool,
    };
    for (const auto type : family) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr,
              "convolution/pooling metadata should exist: " + TypeId(type));
        Check(meta->status == cyxwiz::NodeImplementationStatus::Template &&
                  meta->badge == "Blocked" &&
                  !cyxwiz::CanAddNodeToGraph(*meta),
              "unowned convolution/pooling node must remain blocked: " +
                  TypeId(type));
        Check(meta->inputs.size() == 1 && meta->outputs.size() == 1 &&
                  HasInputType(meta, "Input", gui::PinType::Tensor) &&
                  HasOutputType(meta, "Output", gui::PinType::Tensor),
              "blocked convolution/pooling pins should remain inspectable: " +
                  TypeId(type));
        CheckSupportAxis(meta, "Training Backend",
                         "unsupported_sequential_model_layer", false,
                         TypeId(type));
        CheckSupportAxis(meta, "Compile", "unsupported", false, TypeId(type));
        CheckSupportAxis(meta, "Training", "unsupported", false, TypeId(type));
    }

    for (const auto type : {gui::NodeType::Conv1D,
                            gui::NodeType::Conv3D}) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta->parameters.size() == 4 &&
                  ParameterMatches(meta, "filters", "int", "32") &&
                  ParameterMatches(meta, "kernel_size", "int", "3") &&
                  ParameterMatches(meta, "stride", "int", "1") &&
                  ParameterMatches(meta, "padding", "enum", "same") &&
                  !HasParameter(meta, "activation"),
              "blocked convolution contract should preserve only real saved fields: " +
                  TypeId(type));
    }

    const auto* depthwise =
        metadata.GetMetadata(gui::NodeType::DepthwiseConv2D);
    Check(depthwise->parameters.size() == 5 &&
              ParameterMatches(depthwise, "filters", "int", "32") &&
              ParameterMatches(depthwise, "kernel_size", "int", "3") &&
              ParameterMatches(depthwise, "stride", "int", "1") &&
              ParameterMatches(depthwise, "padding", "enum", "same") &&
              ParameterMatches(depthwise, "depth_multiplier", "int", "1") &&
              !HasParameter(depthwise, "activation"),
          "DepthwiseConv2D should preserve its compatibility fields without a fictional activation");

    for (const auto type : {gui::NodeType::MaxPool2D,
                            gui::NodeType::AvgPool2D}) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta->parameters.size() == 2 &&
                  ParameterMatches(meta, "pool_size", "int", "2") &&
                  ParameterMatches(meta, "stride", "int", "2") &&
                  !HasParameter(meta, "kernel_size"),
              "pooling metadata must use the saved pool_size/stride contract: " +
                  TypeId(type));
    }

    Check(metadata.GetMetadata(gui::NodeType::GlobalMaxPool)->parameters.empty() &&
              metadata.GetMetadata(gui::NodeType::GlobalAvgPool)->parameters.empty(),
          "global pooling compatibility contracts should have no parameters");
    const auto* adaptive = metadata.GetMetadata(gui::NodeType::AdaptiveAvgPool);
    Check(adaptive->parameters.size() == 1 &&
              ParameterMatches(adaptive, "output_size", "int", "1"),
          "AdaptiveAvgPool should preserve its output_size compatibility field");
}

void CheckBlockedUpsamplingFamilyContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    for (const auto type : {gui::NodeType::ConvTranspose2D,
                            gui::NodeType::Upsample,
                            gui::NodeType::PixelShuffle}) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr,
              "upsampling metadata should exist: " + TypeId(type));
        Check(meta->category == gui::NodeCategory::Upsampling &&
                  meta->status == cyxwiz::NodeImplementationStatus::Template &&
                  meta->badge == "Blocked" &&
                  !cyxwiz::CanAddNodeToGraph(*meta),
              "unowned upsampling node must remain blocked: " + TypeId(type));
        Check(meta->inputs.size() == 1 && meta->outputs.size() == 1 &&
                  HasInputType(meta, "Input", gui::PinType::Tensor) &&
                  HasOutputType(meta, "Output", gui::PinType::Tensor),
              "blocked upsampling pins should remain inspectable: " +
                  TypeId(type));
        CheckSupportAxis(meta, "Training Backend",
                         "unsupported_sequential_model_layer", false,
                         TypeId(type));
        CheckSupportAxis(meta, "Compile", "unsupported", false, TypeId(type));
        CheckSupportAxis(meta, "Training", "unsupported", false, TypeId(type));
    }

    const auto* transpose =
        metadata.GetMetadata(gui::NodeType::ConvTranspose2D);
    Check(transpose->parameters.size() == 6 &&
              ParameterMatches(transpose, "in_channels", "int", "64") &&
              ParameterMatches(transpose, "out_channels", "int", "32") &&
              ParameterMatches(transpose, "kernel_size", "int", "3") &&
              ParameterMatches(transpose, "stride", "int", "2") &&
              ParameterMatches(transpose, "padding", "int", "1") &&
              ParameterMatches(transpose, "output_padding", "int", "1") &&
              !HasParameter(transpose, "filters"),
          "ConvTranspose2D metadata must preserve its six saved fields");

    const auto* upsample = metadata.GetMetadata(gui::NodeType::Upsample);
    Check(upsample->parameters.size() == 2 &&
              ParameterMatches(upsample, "scale_factor", "int", "2") &&
              ParameterMatches(upsample, "mode", "enum", "0") &&
              FindParameter(upsample, "mode")->enum_values ==
                  std::vector<std::string>({"0", "1"}),
          "Upsample metadata must preserve numeric mode compatibility");

    const auto* pixel_shuffle =
        metadata.GetMetadata(gui::NodeType::PixelShuffle);
    Check(pixel_shuffle->parameters.size() == 1 &&
              ParameterMatches(pixel_shuffle, "upscale_factor", "int", "2") &&
              !HasParameter(pixel_shuffle, "scale_factor"),
          "PixelShuffle metadata must use its saved upscale_factor field");
}

void CheckBlockedNormalizationFamilyContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    for (const auto type : {gui::NodeType::GroupNorm,
                            gui::NodeType::InstanceNorm}) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr,
              "normalization metadata should exist: " + TypeId(type));
        Check(meta->category == gui::NodeCategory::Normalization &&
                  meta->status == cyxwiz::NodeImplementationStatus::Template &&
                  meta->badge == "Blocked" &&
                  !cyxwiz::CanAddNodeToGraph(*meta),
              "unowned normalization node must remain blocked: " +
                  TypeId(type));
        Check(meta->inputs.size() == 1 && meta->outputs.size() == 1 &&
                  HasInputType(meta, "Input", gui::PinType::Tensor) &&
                  HasOutputType(meta, "Output", gui::PinType::Tensor),
              "blocked normalization pins should remain inspectable: " +
                  TypeId(type));
        CheckSupportAxis(meta, "Training Backend",
                         "unsupported_sequential_model_layer", false,
                         TypeId(type));
        CheckSupportAxis(meta, "Compile", "unsupported", false, TypeId(type));
        CheckSupportAxis(meta, "Training", "unsupported", false, TypeId(type));
    }

    const auto* group = metadata.GetMetadata(gui::NodeType::GroupNorm);
    Check(group->parameters.size() == 4 &&
              ParameterMatches(group, "num_groups", "int", "32") &&
              ParameterMatches(group, "num_channels", "int", "256") &&
              ParameterMatches(group, "eps", "float", "1e-5") &&
              ParameterMatches(group, "affine", "bool", "true") &&
              !HasParameter(group, "epsilon"),
          "GroupNorm metadata must match its backend constructor and canonical fields");

    const auto* instance = metadata.GetMetadata(gui::NodeType::InstanceNorm);
    Check(instance->parameters.size() == 3 &&
              ParameterMatches(instance, "num_features", "int", "64") &&
              ParameterMatches(instance, "eps", "float", "1e-5") &&
              ParameterMatches(instance, "affine", "bool", "false") &&
              !HasParameter(instance, "epsilon"),
          "InstanceNorm metadata must match its backend constructor and canonical fields");

    std::map<std::string, std::string> saved_group_parameters = {
        {"num_groups", "8"},
        {"num_channels", "64"},
        {"epsilon", "0.0001"},
    };
    cyxwiz::CanonicalizePipelineParameterAliases(
        gui::NodeType::GroupNorm, saved_group_parameters);
    Check(saved_group_parameters.find("epsilon") ==
                  saved_group_parameters.end() &&
              saved_group_parameters["eps"] == "0.0001",
          "saved GroupNorm epsilon should migrate to canonical eps");

    std::map<std::string, std::string> saved_instance_parameters = {
        {"num_features", "32"},
        {"eps", "0.0002"},
        {"epsilon", "0.5"},
    };
    cyxwiz::CanonicalizePipelineParameterAliases(
        gui::NodeType::InstanceNorm, saved_instance_parameters);
    Check(saved_instance_parameters.find("epsilon") ==
                  saved_instance_parameters.end() &&
              saved_instance_parameters["eps"] == "0.0002",
          "canonical InstanceNorm eps should win when both saved keys exist");
}

void CheckBlockedAttentionFamilyContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    const std::vector<gui::NodeType> types = {
        gui::NodeType::SelfAttention,
        gui::NodeType::CrossAttention,
        gui::NodeType::LinearAttention,
    };

    for (const auto type : types) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr,
              "attention metadata should exist: " + TypeId(type));
        Check(meta->category == gui::NodeCategory::Attention &&
                  meta->status == cyxwiz::NodeImplementationStatus::Template &&
                  meta->badge == "Blocked" &&
                  !cyxwiz::CanAddNodeToGraph(*meta),
              "unowned attention node must remain blocked: " + TypeId(type));
        CheckSupportAxis(meta, "Training Backend",
                         "unsupported_sequential_model_layer", false,
                         TypeId(type));
        CheckSupportAxis(meta, "Compile", "unsupported", false, TypeId(type));
        CheckSupportAxis(meta, "Training", "unsupported", false, TypeId(type));
    }

    for (const auto type : {gui::NodeType::SelfAttention,
                            gui::NodeType::CrossAttention}) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta->inputs.size() == 4 &&
                  meta->inputs[0].name == "Query" && meta->inputs[0].required &&
                  meta->inputs[1].name == "Key" && meta->inputs[1].required &&
                  meta->inputs[2].name == "Value" && meta->inputs[2].required &&
                  meta->inputs[3].name == "Mask" && !meta->inputs[3].required &&
                  !HasInputType(meta, "Context", gui::PinType::Tensor),
              "saved self/cross-attention input pin order must remain compatible: " +
                  TypeId(type));
        Check(meta->outputs.size() == 2 &&
                  meta->outputs[0].name == "Output" &&
                  meta->outputs[0].required &&
                  meta->outputs[1].name == "Attn Weights" &&
                  !meta->outputs[1].required,
              "saved self/cross-attention output pin order must remain compatible: " +
                  TypeId(type));
        Check(meta->parameters.size() == 4 &&
                  ParameterMatches(meta, "embed_dim", "int", "512") &&
                  ParameterMatches(meta, "num_heads", "int", "8") &&
                  ParameterMatches(meta, "dropout", "float", "0.0") &&
                  ParameterMatches(meta, "batch_first", "bool", "true"),
              "saved self/cross-attention parameters must remain compatible: " +
                  TypeId(type));
    }

    const auto* linear = metadata.GetMetadata(gui::NodeType::LinearAttention);
    Check(linear->inputs.size() == 4 &&
              linear->inputs[0].name == "Query" && linear->inputs[0].required &&
              linear->inputs[1].name == "Key" && linear->inputs[1].required &&
              linear->inputs[2].name == "Value" && linear->inputs[2].required &&
              linear->inputs[3].name == "Mask" && !linear->inputs[3].required &&
              linear->outputs.size() == 1 &&
              linear->outputs[0].name == "Output" &&
              linear->outputs[0].required,
          "saved linear-attention pin order must remain compatible");
    Check(linear->parameters.size() == 5 &&
              ParameterMatches(linear, "embed_dim", "int", "512") &&
              ParameterMatches(linear, "num_heads", "int", "8") &&
              ParameterMatches(linear, "feature_map", "enum", "elu") &&
              HasEnumValue(linear, "feature_map", "elu") &&
              HasEnumValue(linear, "feature_map", "relu") &&
              HasEnumValue(linear, "feature_map", "favor+") &&
              ParameterMatches(linear, "eps", "float", "1e-6") &&
              ParameterMatches(linear, "causal", "bool", "false"),
          "saved linear-attention parameters must remain compatible");
}

void CheckBlockedRecurrentCompatibilityContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    for (const auto type : {gui::NodeType::RNN,
                            gui::NodeType::Bidirectional}) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr,
              "recurrent compatibility metadata should exist: " + TypeId(type));
        Check(meta->category == gui::NodeCategory::Recurrent &&
                  meta->status == cyxwiz::NodeImplementationStatus::Template &&
                  meta->badge == "Blocked" &&
                  !cyxwiz::CanAddNodeToGraph(*meta),
              "unowned recurrent compatibility node must remain blocked: " +
                  TypeId(type));
        Check(meta->inputs.size() == 1 && meta->outputs.size() >= 1 &&
                  meta->inputs[0].name == "Input" &&
                  meta->inputs[0].required &&
                  meta->outputs[0].name == "Output" &&
                  meta->outputs[0].required,
              "recurrent compatibility pins must remain inspectable: " +
                  TypeId(type));
        CheckSupportAxis(meta, "Training Backend",
                         "unsupported_sequential_model_layer", false,
                         TypeId(type));
        CheckSupportAxis(meta, "Compile", "unsupported", false, TypeId(type));
        CheckSupportAxis(meta, "Training", "unsupported", false, TypeId(type));
    }

    const auto* rnn = metadata.GetMetadata(gui::NodeType::RNN);
    Check(rnn->outputs.size() == 2 &&
              rnn->outputs[1].name == "Hidden" &&
              !rnn->outputs[1].required,
          "RNN must preserve its optional Hidden output pin");
    Check(rnn->parameters.size() == 7 &&
              ParameterMatches(rnn, "input_size", "int", "0") &&
              ParameterMatches(rnn, "hidden_size", "int", "256") &&
              ParameterMatches(rnn, "num_layers", "int", "1") &&
              ParameterMatches(rnn, "bidirectional", "bool", "false") &&
              ParameterMatches(rnn, "return_sequences", "bool", "false") &&
              ParameterMatches(rnn, "dropout", "float", "0.0") &&
              ParameterMatches(rnn, "nonlinearity", "string", "tanh") &&
              !HasParameter(rnn, "activation"),
          "RNN metadata must preserve known saved fields without inventing execution");

    const auto* bidirectional =
        metadata.GetMetadata(gui::NodeType::Bidirectional);
    Check(bidirectional->outputs.size() == 1 &&
              bidirectional->parameters.size() == 1 &&
              ParameterMatches(bidirectional, "merge_mode", "string", "concat"),
          "Bidirectional metadata must preserve its standalone wrapper sketch");
}

void CheckImplementedRecurrentConfigurationContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    for (const auto type : {gui::NodeType::LSTM, gui::NodeType::GRU}) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr &&
                  meta->status == cyxwiz::NodeImplementationStatus::Implemented &&
                  cyxwiz::CanAddNodeToGraph(*meta),
              "implemented recurrent node should remain addable: " + TypeId(type));
        Check(meta->inputs.size() == 1 && meta->inputs[0].name == "Input" &&
                  meta->inputs[0].required && meta->outputs.size() == 2 &&
                  meta->outputs[0].name == "Output" && meta->outputs[0].required &&
                  meta->outputs[1].name == "Hidden" && !meta->outputs[1].required,
              "recurrent pins must preserve saved order and optional Hidden compatibility: " +
                  TypeId(type));
        Check(meta->parameters.size() == 6 &&
                  ParameterMatches(meta, "input_size", "int", "0") &&
                  ParameterMatches(meta, "hidden_size", "int", "256") &&
                  ParameterMatches(meta, "num_layers", "int", "1") &&
                  ParameterMatches(meta, "bidirectional", "bool", "false") &&
                  ParameterMatches(meta, "return_sequences", "bool", "false") &&
                  ParameterMatches(meta, "dropout", "float", "0.0"),
              "recurrent metadata must preserve constructor defaults: " + TypeId(type));
        Check(meta->outputs[1].description.find("does not route") !=
                  std::string::npos,
              "legacy Hidden output must disclose missing Engine routing: " +
                  TypeId(type));
    }

    const std::map<std::string, std::string> supported = {
        {"bidirectional", "false"},
        {"dropout", "0.0"},
    };
    Check(!cyxwiz::ResolvePipelineUnsupportedSequentialModelConfigurationReason(
               gui::NodeType::LSTM, supported),
          "unidirectional LSTM with zero dropout should remain supported");
    Check(!cyxwiz::ResolvePipelineUnsupportedSequentialModelConfigurationReason(
               gui::NodeType::GRU, supported),
          "unidirectional GRU with zero dropout should remain supported");

    auto bidirectional = supported;
    bidirectional["bidirectional"] = "true";
    const auto lstm_bidirectional =
        cyxwiz::ResolvePipelineUnsupportedSequentialModelConfigurationReason(
            gui::NodeType::LSTM, bidirectional);
    Check(lstm_bidirectional &&
              lstm_bidirectional->find("reverse-direction backward") !=
                  std::string::npos,
          "bidirectional LSTM should expose its exact backward gap");
    Check(!cyxwiz::ResolvePipelineUnsupportedSequentialModelConfigurationReason(
               gui::NodeType::GRU, bidirectional),
          "split-path bidirectional GRU should remain supported");

    auto recurrent_dropout = supported;
    recurrent_dropout["dropout"] = "0.2";
    for (const auto type : {gui::NodeType::LSTM, gui::NodeType::GRU}) {
        const auto reason =
            cyxwiz::ResolvePipelineUnsupportedSequentialModelConfigurationReason(
                type, recurrent_dropout);
        Check(reason && reason->find("not wired") != std::string::npos,
              "nonzero recurrent dropout must fail closed: " + TypeId(type));
    }
}

void CheckDataValidatorReferenceContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    const auto* meta = metadata.GetMetadata(gui::NodeType::DataValidator);
    Check(meta != nullptr, "DataValidator metadata should exist");
    Check(meta->category == gui::NodeCategory::Analytics,
          "DataValidator should be categorized as analytics");
    Check(meta->inputs.size() == 1 &&
              HasInputType(meta, "Data", gui::PinType::Dataset),
          "DataValidator should accept one Data dataset");
    Check(meta->outputs.size() == 1 &&
              HasOutputType(meta, "Issues", gui::PinType::Dataset),
          "DataValidator should expose only its runtime issue report");
    Check(meta->parameters.size() == 3,
          "DataValidator should expose only implemented rule families");
    Check(ParameterMatches(meta, "required_columns", "string", "") &&
              ParameterMatches(meta, "not_null_columns", "string", "") &&
              ParameterMatches(meta, "unique_columns", "string", ""),
          "DataValidator rule fields should match executor inputs");
    Check(!HasParameter(meta, "column_types") &&
              !HasParameter(meta, "value_ranges") &&
              !HasParameter(meta, "regex_patterns"),
          "DataValidator must not advertise unsupported rule families");
}

void CheckDataValidatorOutputMigrationGuard() {
    nlohmann::json legacy_graph = nlohmann::json::object();
    Check(gui::detail::PreserveLegacyDataValidatorOutputs(legacy_graph),
          "unversioned graphs should use legacy DataValidator link migration");

    nlohmann::json modern_graph = {
        {"data_validator_contract_version",
         gui::detail::kCurrentDataValidatorContractVersion}};
    Check(!gui::detail::PreserveLegacyDataValidatorOutputs(modern_graph),
          "current DataValidator graphs should use the truthful output index");

    nlohmann::json malformed_graph = {
        {"data_validator_contract_version", "2"}};
    Check(gui::detail::PreserveLegacyDataValidatorOutputs(malformed_graph),
          "malformed DataValidator versions should fail safe to legacy migration");

    int index = -1;
    nlohmann::json issues_link = {{"from_pin_index", 2}};
    Check(gui::detail::ResolveLegacyDataValidatorOutputPinIndex(
              issues_link, index) && index == 0,
          "legacy DataValidator Issues links should migrate from index 2 to 0");

    for (const int unsupported_index : {0, 1}) {
        nlohmann::json unsupported_link = {
            {"from_pin_index", unsupported_index}};
        Check(!gui::detail::ResolveLegacyDataValidatorOutputPinIndex(
                  unsupported_link, index),
              "legacy DataValidator fake outputs must not become Issues links");
    }
}

void CheckTabularAnalyticsFamilyContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    const auto* sample = metadata.GetMetadata(gui::NodeType::SampleRows);
    const auto* counts = metadata.GetMetadata(gui::NodeType::ValueCounts);
    const auto* stats = metadata.GetMetadata(gui::NodeType::DescribeStats);
    const auto* correlation =
        metadata.GetMetadata(gui::NodeType::CorrelationMatrix);

    for (const auto* meta : {sample, counts, stats, correlation}) {
        Check(meta != nullptr, "tabular analytics metadata should exist");
        Check(meta->category == gui::NodeCategory::Analytics,
              "tabular analytics node should use the Analytics category: " +
                  TypeId(meta->type));
        Check(meta->status == cyxwiz::NodeImplementationStatus::Implemented,
              "tabular analytics node should be implemented: " +
                  TypeId(meta->type));
        Check(meta->inputs.size() == 1 &&
                  HasInputType(meta, "Table", gui::PinType::Dataset),
              "tabular analytics node should accept one table dataset: " +
                  TypeId(meta->type));
        Check(cyxwiz::ResolvePipelineRuntimeSupport(meta->type).mode ==
                  cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
              "tabular analytics node should retain its executor owner: " +
                  TypeId(meta->type));
    }

    Check(sample->outputs.size() == 1 &&
              HasOutputType(sample, "Sample", gui::PinType::Dataset) &&
              sample->parameters.size() == 1 &&
              ParameterMatches(sample, "count", "int", "100") &&
              !HasParameter(sample, "n") &&
              !HasParameter(sample, "random_state") &&
              !ContainsString(sample->keywords, "random"),
          "SampleRows should declare deterministic leading-row behavior");

    const auto* count_column = FindParameter(counts, "column");
    Check(counts->outputs.size() == 1 &&
              HasOutputType(counts, "Counts", gui::PinType::Dataset) &&
              counts->parameters.size() == 1 && count_column != nullptr &&
              count_column->required,
          "ValueCounts should require the executor-consumed column");

    Check(stats->outputs.size() == 1 &&
              HasOutputType(stats, "Stats", gui::PinType::Dataset) &&
              stats->parameters.empty() &&
              !HasParameter(stats, "show_percentiles"),
          "DescribeStats should expose its real output without fake options");

    Check(correlation->outputs.size() == 1 &&
              HasOutputType(correlation, "Matrix", gui::PinType::Dataset) &&
              correlation->parameters.empty() &&
              !HasParameter(correlation, "method"),
          "CorrelationMatrix should advertise only implemented Pearson behavior");
}

void CheckEvaluationTableFamilyContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    const std::vector<gui::NodeType> types = {
        gui::NodeType::RegressionMetricsNode,
        gui::NodeType::ClassificationMetricsNode,
        gui::NodeType::ConfusionMatrixNode,
        gui::NodeType::ROCCurveNode,
        gui::NodeType::PRCurveNode,
    };
    for (const auto type : types) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr &&
                  meta->category == gui::NodeCategory::Analytics &&
                  meta->status == cyxwiz::NodeImplementationStatus::Implemented,
              "evaluation table node should be implemented Analytics metadata: " +
                  TypeId(type));
        Check(meta->inputs.size() == 1 &&
                  HasInputType(meta, "Data", gui::PinType::Dataset) &&
                  meta->outputs.size() == 1 &&
                  meta->outputs.front().type == gui::PinType::Dataset,
              "evaluation node should use one Dataset table input/output: " +
                  TypeId(type));
        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(type);
        Check(support.mode ==
                  cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
              "evaluation node should retain PipelineExecutor ownership: " +
                  TypeId(type));
    }

    const auto* regression =
        metadata.GetMetadata(gui::NodeType::RegressionMetricsNode);
    const auto* classification =
        metadata.GetMetadata(gui::NodeType::ClassificationMetricsNode);
    const auto* confusion =
        metadata.GetMetadata(gui::NodeType::ConfusionMatrixNode);
    const auto* roc = metadata.GetMetadata(gui::NodeType::ROCCurveNode);
    const auto* pr = metadata.GetMetadata(gui::NodeType::PRCurveNode);

    Check(HasOutputType(regression, "Metrics", gui::PinType::Dataset) &&
              regression->parameters.size() == 3 &&
              FindParameter(regression, "actual_col")->required &&
              FindParameter(regression, "predicted_col")->required &&
              ParameterMatches(regression, "metrics", "string",
                               "mse,rmse,mae,r2"),
          "RegressionMetrics should expose its consumed columns and metrics");
    Check(HasOutputType(classification, "Metrics", gui::PinType::Dataset) &&
              classification->parameters.size() == 3 &&
              FindParameter(classification, "actual_col")->required &&
              FindParameter(classification, "predicted_col")->required &&
              ParameterMatches(classification, "metrics", "string",
                               "accuracy,precision,recall,f1,weighted_f1,count"),
          "ClassificationMetrics should expose its consumed columns and metrics");
    Check(HasOutputType(confusion, "Matrix", gui::PinType::Dataset) &&
              confusion->parameters.size() == 3 &&
              FindParameter(confusion, "actual_col")->required &&
              FindParameter(confusion, "predicted_col")->required &&
              ParameterMatches(confusion, "normalize", "enum", "none") &&
              HasEnumValue(confusion, "normalize", "true") &&
              HasEnumValue(confusion, "normalize", "pred") &&
              HasEnumValue(confusion, "normalize", "all"),
          "ConfusionMatrix should expose only its table-backed normalization contract");
    for (const auto* curve : {roc, pr}) {
        Check(HasOutputType(curve, "Curve", gui::PinType::Dataset) &&
                  curve->parameters.size() == 3 &&
                  FindParameter(curve, "actual_col")->required &&
                  FindParameter(curve, "score_col")->required &&
                  ParameterMatches(curve, "positive_label", "string", "1"),
              "binary evaluation curve should expose its consumed table fields");
    }
}

void CheckEvaluationTableMigrationGuard() {
    nlohmann::json legacy_graph = nlohmann::json::object();
    Check(gui::detail::PreserveLegacyEvaluationTableInputs(legacy_graph),
          "unversioned graphs should use evaluation-table link migration");

    nlohmann::json current_graph = {
        {"evaluation_table_contract_version",
         gui::detail::kCurrentEvaluationTableContractVersion}};
    Check(!gui::detail::PreserveLegacyEvaluationTableInputs(current_graph),
          "current evaluation-table graphs should preserve current links");

    nlohmann::json malformed_graph = {
        {"evaluation_table_contract_version", "invalid"}};
    Check(gui::detail::PreserveLegacyEvaluationTableInputs(malformed_graph),
          "malformed evaluation-table versions should fail closed as legacy");
    Check(gui::detail::IsLegacySplitInputEvaluationNode(
              gui::NodeType::ConfusionMatrixNode) &&
              gui::detail::IsLegacySplitInputEvaluationNode(
                  gui::NodeType::ROCCurveNode) &&
              gui::detail::IsLegacySplitInputEvaluationNode(
                  gui::NodeType::PRCurveNode) &&
              !gui::detail::IsLegacySplitInputEvaluationNode(
                  gui::NodeType::RegressionMetricsNode),
          "only obsolete split-input evaluation nodes should need migration");
}

void CheckSignalProcessingFamilyContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    const std::vector<gui::NodeType> types = {
        gui::NodeType::FFTNode,
        gui::NodeType::FilterDesigner,
        gui::NodeType::Convolution1D,
    };
    for (const auto type : types) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr &&
                  meta->category == gui::NodeCategory::Signal &&
                  meta->status == cyxwiz::NodeImplementationStatus::Implemented,
              "signal operator should be implemented Signal metadata: " +
                  TypeId(type));
        Check(meta->inputs.size() == 1 &&
                  meta->inputs.front().type == gui::PinType::Dataset &&
                  meta->outputs.size() == 1 &&
                  meta->outputs.front().type == gui::PinType::Dataset,
              "signal operator should use one Dataset input/output: " +
                  TypeId(type));
        const auto* signal_col = FindParameter(meta, "signal_col");
        Check(signal_col != nullptr && signal_col->required,
              "signal operator should require its consumed signal column: " +
                  TypeId(type));
        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(type);
        Check(support.mode ==
                  cyxwiz::PipelineRuntimeSupportMode::OperatorBacked,
              "signal operator should retain PipelineOperatorFactory ownership: " +
                  TypeId(type));

        auto op = cyxwiz::PipelineOperatorFactory::Instance().Create(type);
        Check(op != nullptr,
              "signal metadata should have a pipeline operator: " +
                  TypeId(type));
        std::map<std::string, std::string> params;
        for (const auto& parameter : meta->parameters) {
            params.emplace(parameter.name, parameter.default_value);
        }
        std::string error;
        Check(!op->Configure(params, error),
              "empty required signal column should fail operator configuration: " +
                  TypeId(type));
        params["signal_col"] = "signal";
        error.clear();
        Check(op->Configure(params, error),
              "signal metadata defaults should configure after selecting a column: " +
                  TypeId(type) + ": " + error);
    }

    const auto* fft = metadata.GetMetadata(gui::NodeType::FFTNode);
    const auto* filter = metadata.GetMetadata(gui::NodeType::FilterDesigner);
    const auto* convolution =
        metadata.GetMetadata(gui::NodeType::Convolution1D);
    Check(fft->parameters.size() == 2 &&
              ParameterMatches(fft, "sample_rate", "float", "1.0"),
          "FFT metadata should expose only its operator parameters");
    Check(filter->parameters.size() == 6 &&
              ParameterMatches(filter, "filter_type", "dropdown", "lowpass") &&
              HasEnumValue(filter, "filter_type", "highpass") &&
              HasEnumValue(filter, "filter_type", "bandpass") &&
              HasEnumValue(filter, "filter_type", "bandstop") &&
              ParameterMatches(filter, "cutoff", "float", "0.5") &&
              ParameterMatches(filter, "cutoff_high", "float", "0") &&
              ParameterMatches(filter, "sample_rate", "float", "1.0") &&
              ParameterMatches(filter, "order", "int", "4"),
          "FilterDesigner metadata should match its operator defaults/options");
    Check(convolution->parameters.size() == 2 &&
              FindParameter(convolution, "kernel") != nullptr &&
              FindParameter(convolution, "kernel")->required &&
              ParameterMatches(convolution, "kernel", "string",
                               "0.25,0.5,0.25") &&
              !HasParameter(convolution, "mode"),
          "Convolution1D should not advertise its ignored legacy mode");

    const auto* ifft = metadata.GetMetadata(gui::NodeType::IFFTNode);
    const auto* wavelet = metadata.GetMetadata(gui::NodeType::WaveletTransform);
    Check(ifft != nullptr && ifft->IsTemplate() &&
              wavelet != nullptr && wavelet->IsTemplate(),
          "IFFT and Wavelet should remain explicitly blocked outside this family");
}

void CheckTextVectorizerSentimentFamilyContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    const std::vector<gui::NodeType> types = {
        gui::NodeType::TFIDFVectorizer,
        gui::NodeType::CountVectorizer,
        gui::NodeType::SentimentAnalyzer,
    };
    for (const auto type : types) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr &&
                  meta->category == gui::NodeCategory::TextProcessing &&
                  meta->status == cyxwiz::NodeImplementationStatus::Implemented,
              "text operator should be implemented TextProcessing metadata: " +
                  TypeId(type));
        Check(meta->inputs.size() == 1 &&
                  meta->inputs.front().type == gui::PinType::Dataset &&
                  meta->outputs.size() == 1 &&
                  meta->outputs.front().type == gui::PinType::Dataset,
              "text operator should expose its one-table runtime contract: " +
                  TypeId(type));
        const auto* text_col = FindParameter(meta, "text_col");
        Check(text_col != nullptr && text_col->required,
              "text operator should require its consumed text column: " +
                  TypeId(type));
        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(type);
        Check(support.mode ==
                  cyxwiz::PipelineRuntimeSupportMode::OperatorBacked,
              "text operator should retain PipelineOperatorFactory ownership: " +
                  TypeId(type));

        auto op = cyxwiz::PipelineOperatorFactory::Instance().Create(type);
        Check(op != nullptr,
              "text metadata should have a pipeline operator: " + TypeId(type));
        std::map<std::string, std::string> params;
        for (const auto& parameter : meta->parameters) {
            params.emplace(parameter.name, parameter.default_value);
        }
        std::string error;
        Check(!op->Configure(params, error),
              "empty required text column should fail operator configuration: " +
                  TypeId(type));
        params["text_col"] = "text";
        error.clear();
        Check(op->Configure(params, error),
              "text metadata defaults should configure after selecting a column: " +
                  TypeId(type) + ": " + error);
    }

    const auto* tfidf =
        metadata.GetMetadata(gui::NodeType::TFIDFVectorizer);
    const auto* count =
        metadata.GetMetadata(gui::NodeType::CountVectorizer);
    const auto* sentiment =
        metadata.GetMetadata(gui::NodeType::SentimentAnalyzer);
    for (const auto* vectorizer : {tfidf, count}) {
        Check(HasOutputType(vectorizer, "Vectors", gui::PinType::Dataset) &&
                  !HasOutputType(vectorizer, "Vocabulary", gui::PinType::Dataset) &&
                  ParameterMatches(vectorizer, "ngram_range", "dropdown", "1,1") &&
                  HasEnumValue(vectorizer, "ngram_range", "1,3") &&
                  HasEnumValue(vectorizer, "ngram_range", "2,3") &&
                  HasEnumValue(vectorizer, "ngram_range", "3,3") &&
                  !HasParameter(vectorizer, "ngram_min") &&
                  !HasParameter(vectorizer, "ngram_max") &&
                  ParameterMatches(vectorizer, "operation_mode", "enum",
                                   "fit_transform") &&
                  HasEnumValue(vectorizer, "operation_mode",
                               "transform_only") &&
                  ParameterMatches(vectorizer, "save_state", "bool",
                                   "false") &&
                  ParameterMatches(vectorizer, "state_path", "file", "") &&
                  ParameterMatches(vectorizer, "state_overwrite", "bool",
                                   "false"),
              "vectorizer should expose one output, one canonical n-gram control, and fitted-state workflow");

        auto legacy_op =
            cyxwiz::PipelineOperatorFactory::Instance().Create(vectorizer->type);
        Check(legacy_op != nullptr,
              "legacy vectorizer compatibility should retain its operator");
        std::map<std::string, std::string> legacy_params;
        for (const auto& parameter : vectorizer->parameters) {
            if (parameter.name != "ngram_range") {
                legacy_params.emplace(parameter.name, parameter.default_value);
            }
        }
        legacy_params["text_col"] = "text";
        legacy_params["ngram_min"] = "1";
        legacy_params["ngram_max"] = "3";
        std::string legacy_error;
        Check(legacy_op->Configure(legacy_params, legacy_error),
              "saved vectorizer ngram_min/ngram_max aliases should still configure: " +
                  TypeId(vectorizer->type) + ": " + legacy_error);
    }
    Check(tfidf->parameters.size() == 14 &&
              ParameterMatches(tfidf, "min_df", "int", "1") &&
              ParameterMatches(tfidf, "use_idf", "bool", "true") &&
              ParameterMatches(tfidf, "smooth_idf", "bool", "true"),
          "TFIDF metadata should match its materializer fields");
    Check(count->parameters.size() == 12 &&
              ParameterMatches(count, "binary", "bool", "false"),
          "CountVectorizer metadata should match its materializer fields");
    Check(HasOutputType(sentiment, "Sentiment", gui::PinType::Dataset) &&
              sentiment->parameters.size() == 3 &&
              HasParameter(sentiment, "label_col") &&
              ParameterMatches(sentiment, "method", "enum", "vader") &&
              !HasParameter(sentiment, "model"),
          "SentimentAnalyzer should expose its lexicon table contract without a fake model");
}

void CheckTabularTransformFamilyContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    const std::vector<gui::NodeType> types = {
        gui::NodeType::FilterRows,
        gui::NodeType::SelectColumns,
        gui::NodeType::SortRows,
        gui::NodeType::GroupByAggregate,
        gui::NodeType::FillMissingValues,
        gui::NodeType::RemoveDuplicateRows,
        gui::NodeType::RenameColumns,
    };

    for (const auto type : types) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr &&
                  meta->category == gui::NodeCategory::DataTransform &&
                  meta->status == cyxwiz::NodeImplementationStatus::Implemented,
              "tabular transform should have implemented metadata: " +
                  TypeId(type));
        Check(meta->inputs.size() == 1 &&
                  meta->inputs.front().type == gui::PinType::Dataset &&
                  meta->outputs.size() == 1 &&
                  meta->outputs.front().type == gui::PinType::Dataset,
              "tabular transform should expose one Dataset input/output: " +
                  TypeId(type));

        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(type);
        CheckRuntimeOwnerContract(
            support,
            cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
            cyxwiz::PipelineRuntimeImplementationOwner::PipelineExecutor,
            "tabular transform " + TypeId(type));
    }

    const auto* filter = metadata.GetMetadata(gui::NodeType::FilterRows);
    const auto* select = metadata.GetMetadata(gui::NodeType::SelectColumns);
    const auto* sort = metadata.GetMetadata(gui::NodeType::SortRows);
    const auto* group =
        metadata.GetMetadata(gui::NodeType::GroupByAggregate);
    const auto* fill =
        metadata.GetMetadata(gui::NodeType::FillMissingValues);
    const auto* dedupe =
        metadata.GetMetadata(gui::NodeType::RemoveDuplicateRows);
    const auto* rename =
        metadata.GetMetadata(gui::NodeType::RenameColumns);
    const auto* join = metadata.GetMetadata(gui::NodeType::JoinTables);

    Check(filter->parameters.size() == 1 &&
              ParameterMatches(filter, "condition", "string", "") &&
              FindParameter(filter, "condition")->required,
          "FilterRows should require the condition consumed by its executor");
    Check(select->parameters.size() == 1 &&
              ParameterMatches(select, "columns", "string", "") &&
              FindParameter(select, "columns")->required,
          "SelectColumns should require its canonical column list");
    Check(sort->parameters.size() == 3 &&
              FindParameter(sort, "columns")->required &&
              ParameterMatches(sort, "order", "enum", "asc") &&
              HasEnumValue(sort, "order", "desc") &&
              FindParameter(sort, "ascending")->advanced,
          "SortRows should expose order and retain ascending only as an advanced compatibility alias");
    Check(group->parameters.size() == 2 &&
              FindParameter(group, "group_columns")->required &&
              FindParameter(group, "aggregations")->required &&
              !HasParameter(group, "group_by"),
          "GroupBy should expose only the parameter names consumed by its executor");
    Check(fill->parameters.size() == 8 &&
              ParameterMatches(fill, "value", "string", "0") &&
              !HasParameter(fill, "fill_value"),
          "FillMissing should keep one canonical constant-value control");
    Check(dedupe->parameters.size() == 1 &&
              ParameterMatches(dedupe, "columns", "string", "") &&
              !HasParameter(dedupe, "subset") &&
              !HasParameter(dedupe, "keep"),
          "RemoveDuplicateRows should expose its implemented column scope without ignored controls");
    Check(rename->parameters.size() == 1 &&
              ParameterMatches(rename, "mapping", "string", "") &&
              FindParameter(rename, "mapping")->required &&
              !HasParameter(rename, "rename_map"),
          "RenameColumns should be implemented metadata with one canonical mapping control");
    Check(join != nullptr &&
              join->category == gui::NodeCategory::DataTransform &&
              join->status == cyxwiz::NodeImplementationStatus::Implemented &&
              join->inputs.size() == 2 &&
              HasInputType(join, "Left", gui::PinType::Dataset) &&
              HasInputType(join, "Right", gui::PinType::Dataset) &&
              HasOutputType(join, "Joined", gui::PinType::Dataset),
          "Join should expose two named Dataset inputs and one Dataset output");
    Check(join->parameters.size() == 3 &&
              ParameterMatches(join, "left_on", "string", "") &&
              FindParameter(join, "left_on")->required &&
              ParameterMatches(join, "right_on", "string", "") &&
              FindParameter(join, "right_on")->required &&
              ParameterMatches(join, "join_type", "enum", "inner") &&
              HasEnumValue(join, "join_type", "left") &&
              HasEnumValue(join, "join_type", "right") &&
              HasEnumValue(join, "join_type", "outer") &&
              !HasParameter(join, "on_column"),
          "Join should expose separate canonical keys plus its implemented join modes");
    const auto join_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::JoinTables);
    CheckRuntimeOwnerContract(
        join_support,
        cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
        cyxwiz::PipelineRuntimeImplementationOwner::PipelineExecutor,
        "tabular transform " + TypeId(gui::NodeType::JoinTables));
    Check(join_support.required_input_count.has_value() &&
              *join_support.required_input_count == 2,
          "Join runtime should require exactly two inputs");

    std::map<std::string, std::string> saved_group_parameters = {
        {"group_by", "region"},
        {"aggregations", "COUNT(*) AS rows"},
    };
    cyxwiz::CanonicalizePipelineParameterAliases(
        gui::NodeType::GroupByAggregate, saved_group_parameters);
    Check(saved_group_parameters.find("group_by") ==
                  saved_group_parameters.end() &&
              saved_group_parameters["group_columns"] == "region",
          "saved GroupBy group_by aliases should migrate to the canonical property");

    std::map<std::string, std::string> saved_join_parameters = {
        {"on_column", "record_id"},
        {"join_type", "inner"},
    };
    cyxwiz::CanonicalizePipelineParameterAliases(
        gui::NodeType::JoinTables, saved_join_parameters);
    Check(saved_join_parameters.find("on_column") ==
                  saved_join_parameters.end() &&
              saved_join_parameters["left_on"] == "record_id" &&
              saved_join_parameters["right_on"] == "record_id",
          "saved Join on_column aliases should migrate to both canonical keys");
}

void CheckClusteringFamilyContract(cyxwiz::NodeMetadataRegistry& metadata) {
    const std::vector<gui::NodeType> types = {
        gui::NodeType::KMeansCluster,
        gui::NodeType::DBSCANCluster,
        gui::NodeType::HierarchicalCluster,
        gui::NodeType::GMMCluster,
    };

    for (const auto type : types) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr &&
                  meta->category == gui::NodeCategory::Analytics &&
                  meta->status == cyxwiz::NodeImplementationStatus::Implemented,
              "clustering node should have implemented Analytics metadata: " +
                  TypeId(type));
        Check(meta->inputs.size() == 1 &&
                  HasInputType(meta, "Data", gui::PinType::Dataset) &&
                  meta->outputs.size() == 1 &&
                  HasOutputType(meta, "Clustered", gui::PinType::Dataset),
              "clustering node should expose its input-plus-cluster_id table contract: " +
                  TypeId(type));
        Check(ParameterMatches(meta, "feature_cols", "string", "") &&
                  ParameterMatches(meta, "label_col", "string", ""),
              "clustering node should expose shared feature and label selection: " +
                  TypeId(type));

        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(type);
        CheckRuntimeOwnerContract(
            support,
            cyxwiz::PipelineRuntimeSupportMode::OperatorBacked,
            cyxwiz::PipelineRuntimeImplementationOwner::PipelineOperatorFactory,
            "clustering node " + TypeId(type));

        auto op = cyxwiz::PipelineOperatorFactory::Instance().Create(type);
        Check(op != nullptr,
              "clustering metadata should have a pipeline operator: " +
                  TypeId(type));
        std::map<std::string, std::string> params;
        for (const auto& parameter : meta->parameters) {
            params.emplace(parameter.name, parameter.default_value);
        }
        std::string error;
        Check(op->Configure(params, error),
              "clustering metadata defaults should configure its operator: " +
                  TypeId(type) + ": " + error);
    }

    const auto* kmeans = metadata.GetMetadata(gui::NodeType::KMeansCluster);
    const auto* dbscan = metadata.GetMetadata(gui::NodeType::DBSCANCluster);
    const auto* hierarchical =
        metadata.GetMetadata(gui::NodeType::HierarchicalCluster);
    const auto* gmm = metadata.GetMetadata(gui::NodeType::GMMCluster);

    Check(kmeans->parameters.size() == 8 &&
              ParameterMatches(kmeans, "n_clusters", "int", "8") &&
              ParameterMatches(kmeans, "max_iter", "int", "300") &&
              ParameterMatches(kmeans, "init", "enum", "kmeans++") &&
              HasEnumValue(kmeans, "init", "random") &&
              ParameterMatches(kmeans, "n_init", "int", "10") &&
              ParameterMatches(kmeans, "tol", "float", "0.0001") &&
              ParameterMatches(kmeans, "seed", "int", "0") &&
              !HasParameter(kmeans, "random_state"),
          "KMeans metadata should match every consumed operator field");
    Check(dbscan->parameters.size() == 5 &&
              ParameterMatches(dbscan, "eps", "float", "0.5") &&
              ParameterMatches(dbscan, "min_samples", "int", "5") &&
              ParameterMatches(dbscan, "metric", "enum", "euclidean") &&
              HasEnumValue(dbscan, "metric", "cosine"),
          "DBSCAN metadata should match every consumed operator field");
    Check(hierarchical->parameters.size() == 5 &&
              ParameterMatches(hierarchical, "n_clusters", "int", "3") &&
              ParameterMatches(hierarchical, "linkage", "enum", "ward") &&
              HasEnumValue(hierarchical, "linkage", "single") &&
              ParameterMatches(hierarchical, "metric", "enum", "euclidean"),
          "Hierarchical metadata should match every consumed operator field");
    Check(gmm->parameters.size() == 8 &&
              ParameterMatches(gmm, "n_components", "int", "3") &&
              ParameterMatches(gmm, "covariance_type", "enum", "full") &&
              HasEnumValue(gmm, "covariance_type", "spherical") &&
              ParameterMatches(gmm, "max_iter", "int", "100") &&
              ParameterMatches(gmm, "tol", "float", "0.001") &&
              ParameterMatches(gmm, "n_init", "int", "1") &&
              ParameterMatches(gmm, "seed", "int", "0"),
          "GMM metadata should match every consumed operator field");
}

void CheckPcaContract(cyxwiz::NodeMetadataRegistry& metadata) {
    const auto* pca = metadata.GetMetadata(gui::NodeType::PCANode);
    Check(pca != nullptr &&
              pca->category == gui::NodeCategory::Analytics &&
              pca->status == cyxwiz::NodeImplementationStatus::Implemented,
          "PCA should have implemented Analytics metadata");
    Check(pca->inputs.size() == 1 &&
              HasInputType(pca, "Data", gui::PinType::Dataset) &&
              pca->outputs.size() == 1 &&
              HasOutputType(pca, "Transformed", gui::PinType::Dataset) &&
              !HasOutputType(pca, "Components", gui::PinType::Dataset),
          "PCA should expose only the table its operator materializes");
    Check(pca->parameters.size() == 5 &&
              ParameterMatches(pca, "feature_cols", "string", "") &&
              ParameterMatches(pca, "label_col", "string", "") &&
              ParameterMatches(pca, "n_components", "int", "2") &&
              ParameterMatches(pca, "center", "bool", "true") &&
              ParameterMatches(pca, "scale", "bool", "false") &&
              !HasParameter(pca, "whiten"),
          "PCA metadata should match every consumed operator field without whitening");

    const auto support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::PCANode);
    CheckRuntimeOwnerContract(
        support,
        cyxwiz::PipelineRuntimeSupportMode::OperatorBacked,
        cyxwiz::PipelineRuntimeImplementationOwner::PipelineOperatorFactory,
        "PCA");

    auto op = cyxwiz::PipelineOperatorFactory::Instance().Create(
        gui::NodeType::PCANode);
    Check(op != nullptr, "PCA metadata should have a pipeline operator");
    std::map<std::string, std::string> params;
    for (const auto& parameter : pca->parameters) {
        params.emplace(parameter.name, parameter.default_value);
    }
    std::string error;
    Check(op->Configure(params, error),
          "PCA metadata defaults should configure its operator: " + error);
}

void CheckClassicalRegressionFamilyContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    const std::vector<gui::NodeType> types = {
        gui::NodeType::LinearRegressionNode,
        gui::NodeType::PolynomialRegressionNode,
    };

    for (const auto type : types) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr &&
                  meta->category == gui::NodeCategory::Analytics &&
                  meta->status == cyxwiz::NodeImplementationStatus::Implemented,
              "regression node should have implemented Analytics metadata: " +
                  TypeId(type));
        Check(meta->inputs.size() == 1 &&
                  HasInputType(meta, "Data", gui::PinType::Dataset) &&
                  meta->outputs.size() == 2 &&
                  HasOutputType(meta, "Fitted", gui::PinType::Dataset) &&
                  HasOutputType(meta, "Model", gui::PinType::Parameters),
              "regression fit should expose table and fitted Model outputs: " +
                  TypeId(type));
        Check(FindParameter(meta, "target_col") != nullptr &&
                  FindParameter(meta, "target_col")->required,
              "regression node should require its target column: " +
                  TypeId(type));

        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(type);
        CheckRuntimeOwnerContract(
            support,
            cyxwiz::PipelineRuntimeSupportMode::OperatorBacked,
            cyxwiz::PipelineRuntimeImplementationOwner::PipelineOperatorFactory,
            "regression node " + TypeId(type));

        auto op = cyxwiz::PipelineOperatorFactory::Instance().Create(type);
        Check(op != nullptr,
              "regression metadata should have a pipeline operator: " +
                  TypeId(type));
        std::map<std::string, std::string> params;
        for (const auto& parameter : meta->parameters) {
            params.emplace(parameter.name, parameter.default_value);
        }
        params[type == gui::NodeType::LinearRegressionNode
                   ? "feature_cols"
                   : "feature_col"] = "x";
        params["target_col"] = "y";
        std::string error;
        Check(op->Configure(params, error),
              "regression metadata contract should configure its operator: " +
                  TypeId(type) + ": " + error);
    }

    const auto* linear =
        metadata.GetMetadata(gui::NodeType::LinearRegressionNode);
    const auto* polynomial =
        metadata.GetMetadata(gui::NodeType::PolynomialRegressionNode);
    Check(linear->parameters.size() == 3 &&
              FindParameter(linear, "feature_cols") != nullptr &&
              FindParameter(linear, "feature_cols")->required &&
              ParameterMatches(linear, "fit_intercept", "bool", "true") &&
              FindParameter(linear, "fit_intercept")->group == "Fit Options",
          "LinearRegression metadata should match every consumed operator field");
    Check(polynomial->parameters.size() == 3 &&
              FindParameter(polynomial, "feature_col") != nullptr &&
              FindParameter(polynomial, "feature_col")->required &&
              ParameterMatches(polynomial, "degree", "int", "2") &&
              FindParameter(polynomial, "degree")->validation ==
                  "1-2147483647" &&
              FindParameter(polynomial, "degree")->group == "Fit Options",
          "PolynomialRegression metadata should match its unbounded positive degree contract");

    const auto* predictor =
        metadata.GetMetadata(gui::NodeType::RegressionModelPredictor);
    Check(predictor != nullptr && predictor->inputs.size() == 2 &&
              HasInputType(predictor, "Data", gui::PinType::Dataset) &&
              HasInputType(predictor, "Model", gui::PinType::Parameters) &&
              HasOutputType(predictor, "Predictions", gui::PinType::Dataset) &&
              ParameterMatches(predictor, "prediction_col", "string", "prediction"),
          "RegressionModelPredictor should consume Data plus the fitted Model artifact");
    const auto predictor_support = cyxwiz::ResolvePipelineRuntimeSupport(
        gui::NodeType::RegressionModelPredictor);
    CheckRuntimeOwnerContract(
        predictor_support,
        cyxwiz::PipelineRuntimeSupportMode::OperatorBacked,
        cyxwiz::PipelineRuntimeImplementationOwner::PipelineOperatorFactory,
        "RegressionModelPredictor");
    Check(predictor_support.required_input_count == 2,
          "RegressionModelPredictor should require Data and Model links");

    const auto* svm = metadata.GetMetadata(gui::NodeType::SVMRegressor);
    Check(svm != nullptr && svm->IsTemplate() &&
              cyxwiz::ResolvePipelineRuntimeSupport(
                  gui::NodeType::SVMRegressor).mode ==
                  cyxwiz::PipelineRuntimeSupportMode::FailClosed &&
              cyxwiz::PipelineOperatorFactory::Instance().Create(
                  gui::NodeType::SVMRegressor) == nullptr,
          "SVMRegressor should remain blocked without a runtime owner");
}

void CheckBlockedClassifierFamilyContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    const std::vector<gui::NodeType> types = {
        gui::NodeType::SVMClassifier,
        gui::NodeType::KNNClassifier,
        gui::NodeType::NaiveBayesClassifier,
        gui::NodeType::LogisticRegressionNode,
    };

    for (const auto type : types) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr && meta->IsTemplate() &&
                  meta->badge == "Blocked" &&
                  !cyxwiz::CanAddNodeToGraph(*meta),
              "unimplemented classifier should remain blocked: " +
                  TypeId(type));
        Check(meta->inputs.size() == 2 &&
                  HasInputType(meta, "Train Data", gui::PinType::Dataset) &&
                  HasInputType(meta, "Labels", gui::PinType::Labels) &&
                  meta->outputs.size() == 2 &&
                  HasOutputType(meta, "Model", gui::PinType::Parameters) &&
                  HasOutputType(meta, "Predictions", gui::PinType::Labels),
              "blocked classifier should preserve its saved-graph pin contract: " +
                  TypeId(type));
        Check(meta->brief_description.find("Blocked") != std::string::npos &&
                  meta->help_text.find("No classifier executor") !=
                      std::string::npos,
              "blocked classifier help should state missing runtime ownership: " +
                  TypeId(type));
        Check(cyxwiz::ResolvePipelineRuntimeSupport(type).mode ==
                  cyxwiz::PipelineRuntimeSupportMode::FailClosed &&
                  cyxwiz::PipelineOperatorFactory::Instance().Create(type) ==
                      nullptr,
              "blocked classifier should fail closed without an operator: " +
                  TypeId(type));
    }

    const auto* svm = metadata.GetMetadata(gui::NodeType::SVMClassifier);
    const auto* knn = metadata.GetMetadata(gui::NodeType::KNNClassifier);
    const auto* naive_bayes =
        metadata.GetMetadata(gui::NodeType::NaiveBayesClassifier);
    const auto* logistic =
        metadata.GetMetadata(gui::NodeType::LogisticRegressionNode);
    Check(svm->parameters.size() == 3 &&
              ParameterMatches(svm, "kernel", "enum", "rbf") &&
              ParameterMatches(svm, "C", "float", "1.0") &&
              ParameterMatches(svm, "gamma", "enum", "scale"),
          "SVM preview should preserve its legacy saved parameters");
    Check(knn->parameters.size() == 3 &&
              ParameterMatches(knn, "n_neighbors", "int", "5") &&
              ParameterMatches(knn, "weights", "enum", "uniform") &&
              ParameterMatches(knn, "metric", "enum", "euclidean"),
          "KNN preview should preserve its legacy saved parameters");
    Check(naive_bayes->parameters.size() == 1 &&
              ParameterMatches(naive_bayes, "var_smoothing", "float", "1e-9") &&
              !HasParameter(naive_bayes, "variant"),
          "Naive Bayes preview should not advertise an unseeded variant");
    Check(logistic->parameters.size() == 3 &&
              ParameterMatches(logistic, "C", "float", "1.0") &&
              ParameterMatches(logistic, "solver", "enum", "lbfgs") &&
              ParameterMatches(logistic, "max_iter", "int", "100") &&
              !HasParameter(logistic, "penalty") &&
              !HasOutputType(logistic, "Probabilities", gui::PinType::Dataset),
          "Logistic preview should not advertise uncreated parameters or outputs");
}

void CheckBlockedSchedulerFamilyContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    const std::vector<gui::NodeType> types = {
        gui::NodeType::StepLR,
        gui::NodeType::CosineAnnealing,
        gui::NodeType::ReduceOnPlateau,
        gui::NodeType::ExponentialLR,
        gui::NodeType::WarmupScheduler,
    };

    for (const auto type : types) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr && meta->IsTemplate() &&
                  meta->badge == "Blocked" &&
                  !cyxwiz::CanAddNodeToGraph(*meta),
              "unintegrated scheduler should remain blocked: " +
                  TypeId(type));
        Check(meta->inputs.size() == 1 &&
                  HasInputType(meta, "Optimizer", gui::PinType::Optimizer) &&
                  meta->outputs.size() == 1 &&
                  HasOutputType(meta, "Scheduled", gui::PinType::Optimizer),
              "blocked scheduler should preserve its saved-graph pin contract: " +
                  TypeId(type));
        Check(meta->brief_description.find("Blocked") != std::string::npos &&
                  meta->help_text.find("do not construct, step, restore, or checkpoint") !=
                      std::string::npos,
              "blocked scheduler help should state its missing lifecycle owner: " +
                  TypeId(type));

        const auto support =
            cyxwiz::ResolvePipelineTrainingBackendSupport(type);
        Check(support.mode ==
                  cyxwiz::PipelineTrainingBackendSupportMode::
                      UnsupportedTrainingControl &&
                  !support.compile_supported && !support.training_supported,
              "blocked scheduler should fail closed at compile and training: " +
                  TypeId(type));
        Check(cyxwiz::PipelineOperatorFactory::Instance().Create(type) == nullptr,
              "blocked scheduler should not claim a PipelineExecutor owner: " +
                  TypeId(type));
    }

    const auto* step = metadata.GetMetadata(gui::NodeType::StepLR);
    const auto* cosine =
        metadata.GetMetadata(gui::NodeType::CosineAnnealing);
    const auto* plateau =
        metadata.GetMetadata(gui::NodeType::ReduceOnPlateau);
    const auto* exponential =
        metadata.GetMetadata(gui::NodeType::ExponentialLR);
    const auto* warmup =
        metadata.GetMetadata(gui::NodeType::WarmupScheduler);
    Check(step->parameters.size() == 2 &&
              ParameterMatches(step, "step_size", "int", "10") &&
              ParameterMatches(step, "gamma", "float", "0.1"),
          "StepLR preview should preserve its legacy saved parameters");
    Check(cosine->parameters.size() == 2 &&
              ParameterMatches(cosine, "T_max", "int", "100") &&
              ParameterMatches(cosine, "eta_min", "float", "0.0"),
          "CosineAnnealing preview should preserve its legacy saved parameters");
    Check(plateau->parameters.size() == 3 &&
              ParameterMatches(plateau, "mode", "enum", "min") &&
              HasEnumValue(plateau, "mode", "max") &&
              ParameterMatches(plateau, "factor", "float", "0.1") &&
              ParameterMatches(plateau, "patience", "int", "10"),
          "ReduceOnPlateau preview should preserve its legacy saved parameters");
    Check(exponential->parameters.size() == 1 &&
              ParameterMatches(exponential, "gamma", "float", "0.95"),
          "ExponentialLR preview should preserve its legacy saved parameters");
    Check(warmup->parameters.size() == 2 &&
              ParameterMatches(warmup, "warmup_steps", "int", "1000") &&
              ParameterMatches(warmup, "warmup_ratio", "float", "0.1"),
          "Warmup preview should preserve its legacy saved parameters");
}

void CheckBlockedRegularizationFamilyContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    const std::vector<gui::NodeType> types = {
        gui::NodeType::L1Regularization,
        gui::NodeType::L2Regularization,
        gui::NodeType::ElasticNet,
    };

    for (const auto type : types) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr && meta->IsTemplate() &&
                  meta->badge == "Blocked" &&
                  !cyxwiz::CanAddNodeToGraph(*meta),
              "unintegrated regularization node should remain blocked: " +
                  TypeId(type));
        Check(meta->inputs.size() == 1 &&
                  HasInputType(meta, "Parameters", gui::PinType::Parameters) &&
                  meta->outputs.size() == 1 &&
                  HasOutputType(meta, "Penalty", gui::PinType::Loss),
              "blocked regularization node should preserve its saved-graph pin contract: " +
                  TypeId(type));
        Check(meta->brief_description.find("Blocked") != std::string::npos &&
                  meta->help_text.find("No Engine owner") != std::string::npos &&
                  meta->help_text.find("selected training loss") !=
                      std::string::npos,
              "blocked regularization help should state its missing loss owner: " +
                  TypeId(type));

        const auto support =
            cyxwiz::ResolvePipelineTrainingBackendSupport(type);
        Check(support.mode ==
                  cyxwiz::PipelineTrainingBackendSupportMode::
                      UnsupportedTrainingControl &&
                  !support.compile_supported && !support.training_supported,
              "blocked regularization node should fail closed at compile and training: " +
                  TypeId(type));
        Check(cyxwiz::PipelineOperatorFactory::Instance().Create(type) == nullptr,
              "blocked regularization node should not claim a PipelineExecutor owner: " +
                  TypeId(type));
    }

    const auto* l1 = metadata.GetMetadata(gui::NodeType::L1Regularization);
    const auto* l2 = metadata.GetMetadata(gui::NodeType::L2Regularization);
    const auto* elastic = metadata.GetMetadata(gui::NodeType::ElasticNet);
    Check(l1->parameters.size() == 1 &&
              ParameterMatches(l1, "lambda", "float", "0.01"),
          "L1 preview should preserve its legacy saved parameter");
    Check(l2->parameters.size() == 1 &&
              ParameterMatches(l2, "lambda", "float", "0.01") &&
              l2->help_text.find("AdamW weight decay is a separate") !=
                  std::string::npos,
          "L2 preview should preserve its legacy parameter and distinguish AdamW");
    Check(elastic->parameters.size() == 2 &&
              ParameterMatches(elastic, "lambda", "float", "0.01") &&
              ParameterMatches(elastic, "l1_ratio", "float", "0.5") &&
              !HasParameter(elastic, "l1_lambda") &&
              !HasParameter(elastic, "l2_lambda") &&
              !HasParameter(elastic, "alpha"),
          "Elastic Net preview should preserve only its created legacy parameters");
}

void CheckClassicalTreeFamilyContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    const std::vector<gui::NodeType> training_types = {
        gui::NodeType::DecisionTreeClassifier,
        gui::NodeType::RandomForestClassifier,
        gui::NodeType::GradientBoostingClassifier,
    };

    for (const auto type : training_types) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr &&
                  meta->category == gui::NodeCategory::Analytics &&
                  meta->status == cyxwiz::NodeImplementationStatus::Implemented,
              "tree classifier should have implemented Analytics metadata: " +
                  TypeId(type));
        Check(meta->inputs.size() == 1 &&
                  HasInputType(meta, "Data", gui::PinType::Dataset) &&
                  meta->outputs.size() == 1 &&
                  HasOutputType(meta, "Predictions", gui::PinType::Dataset) &&
                  !HasInputType(meta, "Labels", gui::PinType::Labels) &&
                  !HasOutputType(meta, "Model", gui::PinType::Parameters),
              "tree classifier should expose one table-in/table-out contract: " +
                  TypeId(type));
        Check(FindParameter(meta, "target_col") != nullptr &&
                  FindParameter(meta, "target_col")->required &&
                  ParameterMatches(meta, "feature_cols", "string", "") &&
                  ParameterMatches(meta, "prediction_col", "string", "prediction") &&
                  ParameterMatches(meta, "model_path", "string", ""),
              "tree classifier should expose its shared table/artifact fields: " +
                  TypeId(type));

        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(type);
        CheckRuntimeOwnerContract(
            support,
            cyxwiz::PipelineRuntimeSupportMode::OperatorBacked,
            cyxwiz::PipelineRuntimeImplementationOwner::PipelineOperatorFactory,
            "tree classifier " + TypeId(type));

        auto op = cyxwiz::PipelineOperatorFactory::Instance().Create(type);
        Check(op != nullptr,
              "tree classifier metadata should have a pipeline operator: " +
                  TypeId(type));
        std::map<std::string, std::string> params;
        for (const auto& parameter : meta->parameters) {
            params.emplace(parameter.name, parameter.default_value);
        }
        params["target_col"] = "target";
        std::string error;
        Check(op->Configure(params, error),
              "tree classifier metadata defaults should configure its operator: " +
                  TypeId(type) + ": " + error);
    }

    const auto* decision =
        metadata.GetMetadata(gui::NodeType::DecisionTreeClassifier);
    const auto* forest =
        metadata.GetMetadata(gui::NodeType::RandomForestClassifier);
    const auto* boosting =
        metadata.GetMetadata(gui::NodeType::GradientBoostingClassifier);
    const auto* predictor =
        metadata.GetMetadata(gui::NodeType::TreeModelPredictor);
    Check(decision->parameters.size() == 8 &&
              ParameterMatches(decision, "max_depth", "int", "10") &&
              ParameterMatches(decision, "min_samples_split", "int", "2") &&
              ParameterMatches(decision, "min_samples_leaf", "int", "1") &&
              ParameterMatches(decision, "criterion", "enum", "gini"),
          "DecisionTree metadata should match every consumed operator field");
    Check(forest->parameters.size() == 11 &&
              ParameterMatches(forest, "n_estimators", "int", "100") &&
              ParameterMatches(forest, "max_features", "enum", "sqrt") &&
              ParameterMatches(forest, "seed", "int", "42"),
          "RandomForest metadata should match every consumed operator field");
    Check(boosting->parameters.size() == 9 &&
              ParameterMatches(boosting, "n_estimators", "int", "100") &&
              ParameterMatches(boosting, "learning_rate", "float", "0.1") &&
              ParameterMatches(boosting, "max_depth", "int", "3"),
          "GradientBoosting metadata should match every consumed operator field");
    Check(predictor != nullptr && predictor->inputs.size() == 1 &&
              predictor->outputs.size() == 1 &&
              predictor->parameters.size() == 3 &&
              FindParameter(predictor, "model_path") != nullptr &&
              FindParameter(predictor, "model_path")->required &&
              ParameterMatches(predictor, "prediction_col", "string", "prediction"),
          "TreeModelPredictor should expose its required artifact table contract");

    auto predictor_op = cyxwiz::PipelineOperatorFactory::Instance().Create(
        gui::NodeType::TreeModelPredictor);
    std::string predictor_error;
    Check(predictor_op != nullptr &&
              predictor_op->Configure({{"model_path", "model.json"}},
                                      predictor_error),
          "TreeModelPredictor metadata contract should configure its operator: " +
              predictor_error);
}

void CheckClassicalTreeMigrationGuard() {
    const nlohmann::json legacy_graph = nlohmann::json::object();
    Check(gui::detail::PreserveLegacyClassicalTreeTablePins(legacy_graph),
          "unversioned graphs should use classical-tree pin migration");
    const nlohmann::json current_graph = {
        {"classical_tree_table_contract_version",
         gui::detail::kCurrentClassicalTreeTableContractVersion}};
    Check(!gui::detail::PreserveLegacyClassicalTreeTablePins(current_graph),
          "current classical-tree graphs should use truthful table pins");

    int resolved_index = -1;
    Check(gui::detail::ResolveLegacyClassicalTreeOutputPinIndex(
              {{"from_pin_index", 1}}, resolved_index) &&
              resolved_index == 0,
          "legacy tree Predictions output should migrate from index 1 to 0");
    Check(!gui::detail::ResolveLegacyClassicalTreeOutputPinIndex(
              {{"from_pin_index", 0}}, resolved_index),
          "legacy fictional tree Model output must not become Predictions");
}

void CheckStaticCreationAdapter(cyxwiz::NodeMetadataRegistry& metadata) {
    for (const auto type : {
             gui::NodeType::Dense,
             gui::NodeType::MultiHeadAttention,
             gui::NodeType::TransformerEncoder,
             gui::NodeType::TransformerDecoder,
             gui::NodeType::PositionalEncoding,
             gui::NodeType::StandardScaler,
             gui::NodeType::MinMaxScaler,
             gui::NodeType::RobustScaler,
             gui::NodeType::LabelEncoder,
             gui::NodeType::OrdinalEncoder,
             gui::NodeType::TargetEncoder,
             gui::NodeType::OutlierDetector,
             gui::NodeType::DataValidator,
             gui::NodeType::SampleRows,
             gui::NodeType::ValueCounts,
             gui::NodeType::DescribeStats,
             gui::NodeType::CorrelationMatrix,
             gui::NodeType::RegressionMetricsNode,
             gui::NodeType::ClassificationMetricsNode,
             gui::NodeType::ConfusionMatrixNode,
             gui::NodeType::ROCCurveNode,
             gui::NodeType::PRCurveNode,
             gui::NodeType::FFTNode,
             gui::NodeType::FilterDesigner,
             gui::NodeType::Convolution1D,
             gui::NodeType::TFIDFVectorizer,
             gui::NodeType::CountVectorizer,
             gui::NodeType::SentimentAnalyzer,
             gui::NodeType::FilterRows,
             gui::NodeType::SelectColumns,
             gui::NodeType::SortRows,
             gui::NodeType::GroupByAggregate,
             gui::NodeType::FillMissingValues,
             gui::NodeType::RemoveDuplicateRows,
             gui::NodeType::RenameColumns,
             gui::NodeType::JoinTables,
             gui::NodeType::KMeansCluster,
             gui::NodeType::DBSCANCluster,
             gui::NodeType::HierarchicalCluster,
             gui::NodeType::GMMCluster,
             gui::NodeType::PCANode,
             gui::NodeType::LinearRegressionNode,
             gui::NodeType::PolynomialRegressionNode,
             gui::NodeType::DecisionTreeClassifier,
             gui::NodeType::RandomForestClassifier,
             gui::NodeType::GradientBoostingClassifier,
             gui::NodeType::TreeModelPredictor,
             gui::NodeType::RegressionModelPredictor,
             gui::NodeType::SVMClassifier,
             gui::NodeType::KNNClassifier,
             gui::NodeType::NaiveBayesClassifier,
             gui::NodeType::LogisticRegressionNode,
             gui::NodeType::StepLR,
             gui::NodeType::CosineAnnealing,
             gui::NodeType::ReduceOnPlateau,
             gui::NodeType::ExponentialLR,
             gui::NodeType::WarmupScheduler,
             gui::NodeType::L1Regularization,
             gui::NodeType::L2Regularization,
             gui::NodeType::ElasticNet,
             gui::NodeType::PairDatasetBuilder,
             gui::NodeType::TripletDatasetBuilder,
             gui::NodeType::SharedEncoder,
             gui::NodeType::SiameseBranch,
             gui::NodeType::ContrastiveLoss,
             gui::NodeType::CosineEmbeddingLoss,
             gui::NodeType::TripletLoss,
             gui::NodeType::PairMetrics,
             gui::NodeType::RetrievalMetrics,
             gui::NodeType::EmbeddingOutput,
             gui::NodeType::PairScoreOutput,
              gui::NodeType::Lambda,
              gui::NodeType::Identity,
              gui::NodeType::Parameter,
             gui::NodeType::DataInput,
             gui::NodeType::Conv1D,
             gui::NodeType::Conv2D,
             gui::NodeType::Conv3D,
             gui::NodeType::DepthwiseConv2D,
             gui::NodeType::MaxPool2D,
             gui::NodeType::AvgPool2D,
             gui::NodeType::GlobalMaxPool,
             gui::NodeType::GlobalAvgPool,
             gui::NodeType::AdaptiveAvgPool,
             gui::NodeType::ConvTranspose2D,
             gui::NodeType::Upsample,
             gui::NodeType::PixelShuffle,
             gui::NodeType::ReLU,
             gui::NodeType::Sigmoid,
             gui::NodeType::Tanh,
             gui::NodeType::Softmax,
             gui::NodeType::LeakyReLU,
             gui::NodeType::ELU,
             gui::NodeType::GELU,
             gui::NodeType::Swish,
             gui::NodeType::Mish,
             gui::NodeType::Dropout,
             gui::NodeType::BatchNorm,
             gui::NodeType::LayerNorm,
             gui::NodeType::GroupNorm,
             gui::NodeType::InstanceNorm,
             gui::NodeType::SelfAttention,
             gui::NodeType::CrossAttention,
             gui::NodeType::LinearAttention,
             gui::NodeType::RNN,
             gui::NodeType::Bidirectional,
             gui::NodeType::LSTM,
             gui::NodeType::GRU,
             gui::NodeType::Embedding,
             gui::NodeType::TimeDistributed,
             gui::NodeType::Flatten,
             gui::NodeType::Reshape,
             gui::NodeType::View,
             gui::NodeType::Permute,
             gui::NodeType::Squeeze,
             gui::NodeType::Unsqueeze,
             gui::NodeType::TensorBroadcastTo,
             gui::NodeType::TensorExpand,
             gui::NodeType::TensorIndexSelect,
             gui::NodeType::TensorSum,
             gui::NodeType::TensorMean,
             gui::NodeType::TensorMax,
             gui::NodeType::TensorMin,
             gui::NodeType::TensorProd,
             gui::NodeType::TensorVar,
             gui::NodeType::TensorStd,
             gui::NodeType::TensorPow,
             gui::NodeType::TensorSqrt,
             gui::NodeType::TensorExp,
             gui::NodeType::TensorLog,
             gui::NodeType::TensorAbs,
             gui::NodeType::TensorSign,
             gui::NodeType::TensorClip,
             gui::NodeType::TensorDot,
             gui::NodeType::TensorBatchMatMul,
             gui::NodeType::TensorCompare,
             gui::NodeType::TensorLogicalMask,
             gui::NodeType::MSELoss,
             gui::NodeType::CrossEntropyLoss,
             gui::NodeType::FocalLoss,
             gui::NodeType::SoftDiceLoss,
             gui::NodeType::TverskyLoss,
             gui::NodeType::JaccardLoss,
             gui::NodeType::BCELoss,
             gui::NodeType::BCEWithLogits,
             gui::NodeType::L1Loss,
             gui::NodeType::SmoothL1Loss,
             gui::NodeType::HuberLoss,
             gui::NodeType::NLLLoss,
             gui::NodeType::SGD,
             gui::NodeType::Adam,
             gui::NodeType::AdamW,
             gui::NodeType::RMSprop,
             gui::NodeType::Adagrad,
             gui::NodeType::NAdam,
             gui::NodeType::Output,
         }) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr,
              "creation-adapter reference metadata should exist: " +
                  TypeId(type));

        gui::MLNode created;
        created.type = type;
        int next_pin_id = 100;
        cyxwiz::ApplyStaticNodeMetadataContract(*meta, created, next_pin_id);

        Check(created.category == meta->category,
              "creation adapter category drift: " + TypeId(type));
        Check(created.inputs.size() == meta->inputs.size() &&
                  created.outputs.size() == meta->outputs.size(),
              "creation adapter pin-count drift: " + TypeId(type));
        Check(created.parameters.size() == meta->parameters.size(),
              "creation adapter parameter-count drift: " + TypeId(type));

        for (std::size_t i = 0; i < meta->inputs.size(); ++i) {
            const auto& port = meta->inputs[i];
            const auto& pin = created.inputs[i];
            Check(pin.name == port.name && pin.type == port.type &&
                      pin.is_input && pin.is_required == port.required &&
                      pin.description == port.description,
                  "creation adapter input-pin drift: " + TypeId(type));
        }
        for (std::size_t i = 0; i < meta->outputs.size(); ++i) {
            const auto& port = meta->outputs[i];
            const auto& pin = created.outputs[i];
            Check(pin.name == port.name && pin.type == port.type &&
                      !pin.is_input && pin.is_required == port.required &&
                      pin.description == port.description,
                  "creation adapter output-pin drift: " + TypeId(type));
        }
        for (const auto& parameter : meta->parameters) {
            const auto it = created.parameters.find(parameter.name);
            Check(it != created.parameters.end() &&
                      it->second == parameter.default_value,
                  "creation adapter parameter-default drift: " +
                      TypeId(type) + "." + parameter.name);
        }
    }
}

void CheckPreprocessingScalerEncoderFamilyContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    const auto* minmax = metadata.GetMetadata(gui::NodeType::MinMaxScaler);
    Check(minmax != nullptr && minmax->parameters.size() == 4,
          "MinMaxScaler should declare every operator parameter");
    Check(ParameterMatches(minmax, "columns", "string", "") &&
              ParameterMatches(minmax, "label_col", "string", "") &&
              ParameterMatches(minmax, "min", "float", "0.0") &&
              ParameterMatches(minmax, "max", "float", "1.0"),
          "MinMaxScaler metadata defaults should match its operator");

    const auto* robust = metadata.GetMetadata(gui::NodeType::RobustScaler);
    Check(robust != nullptr && robust->parameters.size() == 6,
          "RobustScaler should declare every operator parameter");
    Check(ParameterMatches(robust, "columns", "string", "") &&
              ParameterMatches(robust, "label_col", "string", "") &&
              ParameterMatches(robust, "with_centering", "bool", "true") &&
              ParameterMatches(robust, "with_scaling", "bool", "true") &&
              ParameterMatches(robust, "quantile_min", "float", "25") &&
              ParameterMatches(robust, "quantile_max", "float", "75"),
          "RobustScaler metadata defaults should match its operator");

    const auto* label = metadata.GetMetadata(gui::NodeType::LabelEncoder);
    const auto* ordinal = metadata.GetMetadata(gui::NodeType::OrdinalEncoder);
    const auto* target = metadata.GetMetadata(gui::NodeType::TargetEncoder);
    const auto* outlier = metadata.GetMetadata(gui::NodeType::OutlierDetector);
    Check(FindParameter(label, "column") != nullptr &&
              FindParameter(label, "column")->required,
          "LabelEncoder should require the column consumed by its operator");
    Check(FindParameter(ordinal, "columns") != nullptr &&
              FindParameter(ordinal, "columns")->required &&
              ParameterMatches(ordinal, "categories", "enum", "auto"),
          "OrdinalEncoder should require columns and expose its supported ordering");
    Check(FindParameter(target, "columns") != nullptr &&
              FindParameter(target, "columns")->required &&
              FindParameter(target, "target_col") != nullptr &&
              FindParameter(target, "target_col")->required &&
              ParameterMatches(target, "smoothing", "float", "1.0"),
          "TargetEncoder should require its categorical and numeric target columns");
    Check(outlier != nullptr && outlier->parameters.size() == 4 &&
              ParameterMatches(outlier, "columns", "string", "all") &&
              ParameterMatches(outlier, "label_col", "string", "") &&
              ParameterMatches(outlier, "method", "dropdown", "iqr") &&
              HasEnumValue(outlier, "method", "zscore") &&
              ParameterMatches(outlier, "threshold", "float", "1.5") &&
              !HasParameter(outlier, "action"),
          "OutlierDetector metadata should match its flag-only operator contract");

    const auto configure_defaults = [](const cyxwiz::NodeMetadata* meta,
                                       bool expected) {
        auto op = cyxwiz::PipelineOperatorFactory::Instance().Create(meta->type);
        Check(op != nullptr,
              "preprocessing metadata should have a pipeline operator: " +
                  TypeId(meta->type));
        std::map<std::string, std::string> params;
        for (const auto& parameter : meta->parameters) {
            params.emplace(parameter.name, parameter.default_value);
        }
        std::string error;
        Check(op->Configure(params, error) == expected,
              "preprocessing metadata defaults disagree with operator validation: " +
                  TypeId(meta->type));
    };

    configure_defaults(minmax, true);
    configure_defaults(robust, true);
    configure_defaults(label, false);
    configure_defaults(ordinal, false);
    configure_defaults(target, false);
    configure_defaults(outlier, true);
}

void CheckCoreLayerFamilyContract(cyxwiz::NodeMetadataRegistry& metadata) {
    const auto* output = metadata.GetMetadata(gui::NodeType::Output);
    Check(output != nullptr && output->inputs.size() == 1 &&
              output->outputs.size() == 1 &&
              HasInputType(output, "Input", gui::PinType::Tensor) &&
              HasOutputType(output, "Predictions", gui::PinType::Tensor) &&
              !output->outputs.front().required,
          "Output should preserve its terminal input and optional identity relay");
    Check(ParameterMatches(output, "num_classes", "int", "10") &&
              !HasParameter(output, "classes"),
          "Output metadata should own canonical num_classes without seeding its legacy alias");

    const auto* softmax = metadata.GetMetadata(gui::NodeType::Softmax);
    Check(softmax != nullptr && !HasParameter(softmax, "dim"),
          "Softmax must not expose a dimension ignored by SoftmaxModule");

    const auto* batch_norm = metadata.GetMetadata(gui::NodeType::BatchNorm);
    Check(ParameterMatches(batch_norm, "eps", "float", "1e-5") &&
              ParameterMatches(batch_norm, "momentum", "float", "0.1"),
          "BatchNorm metadata should match GraphCompiler parameter keys");
    Check(!HasParameter(batch_norm, "epsilon"),
          "BatchNorm must not seed the legacy epsilon spelling on new nodes");

    const auto* layer_norm = metadata.GetMetadata(gui::NodeType::LayerNorm);
    Check(ParameterMatches(layer_norm, "normalized_shape", "string", "") &&
              ParameterMatches(layer_norm, "eps", "float", "1e-5") &&
              ParameterMatches(layer_norm, "elementwise_affine", "bool", "true"),
          "LayerNorm metadata should match ModelBuilder's automatic shape contract");

    cyxwiz::NormalizationRegularizationConfiguration resolved;
    Check(!cyxwiz::ResolveNormalizationRegularizationConfiguration(
               gui::NodeType::Dropout, {{"rate", "0"}}, resolved) &&
              resolved.dropout_rate == 0.0f,
          "Dropout policy should preserve an explicit zero rate");
    Check(!cyxwiz::ResolveNormalizationRegularizationConfiguration(
               gui::NodeType::BatchNorm,
               {{"eps", "0.001"}, {"epsilon", "1e-3"},
                {"momentum", "0"}},
               resolved) &&
              std::fabs(resolved.epsilon - 0.001f) < 1e-8f &&
              resolved.momentum == 0.0f,
          "BatchNorm policy should accept equal epsilon aliases and preserve zero momentum");
    Check(cyxwiz::ResolveInvalidNormalizationRegularizationConfigurationReason(
              gui::NodeType::BatchNorm,
              {{"eps", "1e-5"}, {"epsilon", "1e-4"}}).has_value(),
          "BatchNorm policy should reject conflicting epsilon aliases");
    Check(!cyxwiz::ResolveNormalizationRegularizationConfiguration(
               gui::NodeType::LayerNorm,
               {{"normalized_shape", "4, 8"},
                {"elementwise_affine", "false"}},
               resolved) &&
              !resolved.automatic_normalized_shape &&
              resolved.normalized_shape == std::vector<int>({4, 8}) &&
              !resolved.elementwise_affine,
          "LayerNorm policy should resolve exact trailing dimensions and affine state");
    Check(cyxwiz::ResolveInvalidNormalizationRegularizationConfigurationReason(
              gui::NodeType::LayerNorm,
              {{"normalized_shape", "4,-1"}}).has_value(),
          "LayerNorm policy should reject non-positive explicit dimensions");
}

void CheckShapeOperationFamilyContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    for (const auto type : {
             gui::NodeType::Reshape,
             gui::NodeType::View,
         }) {
        const auto* meta = metadata.GetMetadata(type);
        const auto* shape = FindParameter(meta, "shape");
        Check(shape != nullptr && shape->type == "string" &&
                  shape->default_value.empty() && shape->required,
              "Reshape/View should require a shape instead of seeding a "
              "rank-specific default: " + TypeId(type));
    }

    const auto* permute = metadata.GetMetadata(gui::NodeType::Permute);
    const auto* dims = FindParameter(permute, "dims");
    Check(dims != nullptr && dims->type == "string" &&
              dims->default_value.empty() && dims->required,
          "Permute should require an input-rank-specific dimension order");

    Check(ParameterMatches(metadata.GetMetadata(gui::NodeType::Squeeze),
                           "dim", "int", "-1"),
          "Squeeze should default to the compiler's all-singleton policy");
    Check(ParameterMatches(metadata.GetMetadata(gui::NodeType::Unsqueeze),
                           "dim", "int", "0"),
          "Unsqueeze should default to inserting the leading sample axis");

    for (const auto type : {
             gui::NodeType::TensorBroadcastTo,
             gui::NodeType::TensorExpand,
         }) {
        const auto* shape =
            FindParameter(metadata.GetMetadata(type), "shape");
        Check(shape != nullptr && shape->type == "string" &&
                  shape->default_value.empty() && shape->required,
              "broadcast shape operation should require a target shape: " +
                  TypeId(type));
    }

    const auto* index_select =
        metadata.GetMetadata(gui::NodeType::TensorIndexSelect);
    const auto* indices = FindParameter(index_select, "indices");
    Check(ParameterMatches(index_select, "dim", "int", "0") &&
              indices != nullptr && indices->type == "string" &&
              indices->default_value.empty() && indices->required,
          "TensorIndexSelect should expose a safe dimension and required indices");
}

void CheckTensorMathFamilyContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    for (const auto type : {
             gui::NodeType::TensorSum,
             gui::NodeType::TensorMean,
             gui::NodeType::TensorMax,
             gui::NodeType::TensorMin,
             gui::NodeType::TensorProd,
             gui::NodeType::TensorVar,
             gui::NodeType::TensorStd,
         }) {
        const auto* meta = metadata.GetMetadata(type);
        Check(ParameterMatches(meta, "dim", "int", "-1") &&
                  ParameterMatches(meta, "keepdim", "bool", "false"),
              "tensor reduction metadata should match compiler and "
              "ModelBuilder defaults: " + TypeId(type));
    }

    Check(ParameterMatches(metadata.GetMetadata(gui::NodeType::TensorPow),
                           "exponent", "float", "2.0"),
          "TensorPow metadata should match ModelBuilder's exponent default");

    const auto* clip = metadata.GetMetadata(gui::NodeType::TensorClip);
    Check(ParameterMatches(clip, "min", "float", "0.0") &&
              ParameterMatches(clip, "max", "float", "1.0"),
          "TensorClip metadata should match compiler and ModelBuilder bounds");

    for (const auto type : {
             gui::NodeType::TensorSqrt,
             gui::NodeType::TensorExp,
             gui::NodeType::TensorLog,
             gui::NodeType::TensorAbs,
             gui::NodeType::TensorSign,
         }) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr && meta->parameters.empty(),
              "parameter-free unary tensor node should not seed unused "
              "properties: " + TypeId(type));
    }
}

void CheckTensorFanInFamilyContract(
    cyxwiz::NodeMetadataRegistry& metadata) {
    const auto* dot = metadata.GetMetadata(gui::NodeType::TensorDot);
    Check(dot != nullptr && dot->IsImplemented() &&
              cyxwiz::CanAddNodeToGraph(*dot) &&
              HasInput(dot, "A", true) && HasInput(dot, "B", true) &&
              HasOutputType(dot, "Output", gui::PinType::Tensor) &&
              dot->parameters.empty(),
          "TensorDot should expose its implemented two-input graph contract");

    const auto* batch_matmul =
        metadata.GetMetadata(gui::NodeType::TensorBatchMatMul);
    Check(batch_matmul != nullptr && batch_matmul->IsTemplate() &&
              batch_matmul->badge == "Blocked" &&
              !cyxwiz::CanAddNodeToGraph(*batch_matmul) &&
              HasInput(batch_matmul, "A", true) &&
              HasInput(batch_matmul, "B", true),
          "TensorBatchMatMul should preserve its static contract while "
          "remaining blocked from new graphs");

    const auto* compare = metadata.GetMetadata(gui::NodeType::TensorCompare);
    Check(compare != nullptr && HasInput(compare, "A", true) &&
              HasInput(compare, "B", false) &&
              HasOutputType(compare, "Mask", gui::PinType::Tensor) &&
              ParameterMatches(compare, "op", "enum", ">") &&
              ParameterMatches(compare, "scalar", "float", "0.0") &&
              HasEnumValue(compare, "op", ">") &&
              HasEnumValue(compare, "op", "==") &&
              HasEnumValue(compare, "op", "!="),
          "TensorCompare should expose scalar and optional tensor modes");

    const auto* logical =
        metadata.GetMetadata(gui::NodeType::TensorLogicalMask);
    Check(logical != nullptr && HasInput(logical, "A", true) &&
              HasInput(logical, "B", false) &&
              HasOutputType(logical, "Mask", gui::PinType::Tensor) &&
              ParameterMatches(logical, "op", "enum", "not") &&
              HasEnumValue(logical, "op", "not") &&
              HasEnumValue(logical, "op", "and") &&
              HasEnumValue(logical, "op", "or"),
          "TensorLogicalMask should expose unary and optional binary modes");
}

void CheckLossFamilyContract(cyxwiz::NodeMetadataRegistry& metadata) {
    for (const auto type : {
             gui::NodeType::MSELoss,
             gui::NodeType::CrossEntropyLoss,
             gui::NodeType::FocalLoss,
             gui::NodeType::SoftDiceLoss,
             gui::NodeType::TverskyLoss,
             gui::NodeType::JaccardLoss,
             gui::NodeType::BCELoss,
             gui::NodeType::BCEWithLogits,
             gui::NodeType::L1Loss,
             gui::NodeType::SmoothL1Loss,
             gui::NodeType::HuberLoss,
             gui::NodeType::NLLLoss,
         }) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr && meta->IsImplemented() &&
                  cyxwiz::CanAddNodeToGraph(*meta),
              "executable loss should be addable: " + TypeId(type));
        Check(meta->inputs.size() == 2 && meta->outputs.size() == 1 &&
                  meta->inputs[0].type == gui::PinType::Tensor &&
                  meta->inputs[0].required &&
                  meta->inputs[1].type == gui::PinType::Labels &&
                  meta->inputs[1].required &&
                  meta->outputs[0].type == gui::PinType::Loss &&
                  meta->outputs[0].required,
              "loss pins should expose prediction, semantic target, and loss roles: " +
                  TypeId(type));
        Check(ParameterMatches(meta, "reduction", "enum", "mean") &&
                  HasEnumValue(meta, "reduction", "mean") &&
                  HasEnumValue(meta, "reduction", "sum") &&
                  HasEnumValue(meta, "reduction", "none"),
              "loss reduction should match the backend contract: " +
                  TypeId(type));
    }

    const auto* cross_entropy =
        metadata.GetMetadata(gui::NodeType::CrossEntropyLoss);
    Check(HasInputType(cross_entropy, "Logits", gui::PinType::Tensor) &&
              HasInputType(cross_entropy, "Labels", gui::PinType::Labels) &&
              ParameterMatches(cross_entropy, "ignore_index", "int", "-100") &&
              ParameterMatches(cross_entropy, "label_smoothing", "float", "0.0") &&
              ParameterMatches(cross_entropy, "class_weight", "enum", "none") &&
              ParameterMatches(cross_entropy, "class_weights", "string", ""),
          "CrossEntropy should expose the compiler and backend-owned options");

    const auto* focal = metadata.GetMetadata(gui::NodeType::FocalLoss);
    Check(HasInputType(focal, "Logits", gui::PinType::Tensor) &&
              HasInputType(focal, "Labels", gui::PinType::Labels) &&
              ParameterMatches(focal, "alpha", "float", "0.25") &&
              ParameterMatches(focal, "gamma", "float", "2.0"),
          "FocalLoss should expose logit/class-label semantics and defaults");

    const auto* bce_logits =
        metadata.GetMetadata(gui::NodeType::BCEWithLogits);
    Check(HasInputType(bce_logits, "Logits", gui::PinType::Tensor) &&
              ParameterMatches(bce_logits, "pos_weight", "float", "1.0"),
          "BCEWithLogits should expose its stable-logit and weighting contract");

    Check(ParameterMatches(metadata.GetMetadata(gui::NodeType::SmoothL1Loss),
                           "beta", "float", "1.0") &&
              ParameterMatches(metadata.GetMetadata(gui::NodeType::HuberLoss),
                               "beta", "float", "1.0"),
          "SmoothL1 and its Huber alias should expose the consumed transition width");
    Check(ParameterMatches(metadata.GetMetadata(gui::NodeType::NLLLoss),
                           "ignore_index", "int", "-100") &&
              HasInputType(metadata.GetMetadata(gui::NodeType::NLLLoss),
                           "Log Probabilities", gui::PinType::Tensor),
          "NLLLoss should expose log-probability input and ignore-index semantics");
}

void CheckOptimizerFamilyContract(cyxwiz::NodeMetadataRegistry& metadata) {
    for (const auto type : {
             gui::NodeType::SGD,
             gui::NodeType::Adam,
             gui::NodeType::AdamW,
             gui::NodeType::RMSprop,
             gui::NodeType::Adagrad,
             gui::NodeType::NAdam,
         }) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr && meta->IsImplemented() &&
                  cyxwiz::CanAddNodeToGraph(*meta),
              "executable optimizer should be addable: " + TypeId(type));
        Check(meta->inputs.size() == 1 && meta->outputs.size() == 1 &&
                  meta->inputs[0].name == "Loss" &&
                  meta->inputs[0].type == gui::PinType::Loss &&
                  meta->inputs[0].required &&
                  meta->outputs[0].name == "State" &&
                  meta->outputs[0].type == gui::PinType::Optimizer &&
                  !meta->outputs[0].required,
              "optimizer pins should expose one loss input and optional state output: " +
                  TypeId(type));
    }

    Check(ParameterMatches(metadata.GetMetadata(gui::NodeType::SGD),
                           "learning_rate", "float", "0.01") &&
              ParameterMatches(metadata.GetMetadata(gui::NodeType::SGD),
                               "momentum", "float", "0.9") &&
              !HasParameter(metadata.GetMetadata(gui::NodeType::SGD),
                            "weight_decay"),
          "SGD metadata should match the backend constructor contract");

    for (const auto type : {gui::NodeType::Adam, gui::NodeType::NAdam}) {
        const auto* meta = metadata.GetMetadata(type);
        Check(ParameterMatches(meta, "beta1", "float", "0.9") &&
                  ParameterMatches(meta, "beta2", "float", "0.999") &&
                  ParameterMatches(meta, "epsilon", "float", "1e-8"),
              "Adam-family moments should match backend defaults: " +
                  TypeId(type));
    }
    Check(ParameterMatches(metadata.GetMetadata(gui::NodeType::NAdam),
                           "learning_rate", "float", "0.002"),
          "NAdam should expose its backend learning-rate default");

    const auto* adamw = metadata.GetMetadata(gui::NodeType::AdamW);
    Check(ParameterMatches(adamw, "epsilon", "float", "1e-8") &&
              ParameterMatches(adamw, "weight_decay", "float", "0.01"),
          "AdamW should expose all backend-consumed options");

    const auto* rmsprop = metadata.GetMetadata(gui::NodeType::RMSprop);
    Check(ParameterMatches(rmsprop, "alpha", "float", "0.99") &&
              ParameterMatches(rmsprop, "epsilon", "float", "1e-8") &&
              ParameterMatches(rmsprop, "momentum", "float", "0.0"),
          "RMSprop should expose all backend-consumed options");

    const auto* adagrad = metadata.GetMetadata(gui::NodeType::Adagrad);
    Check(ParameterMatches(adagrad, "learning_rate", "float", "0.01") &&
              ParameterMatches(adagrad, "epsilon", "float", "1e-10") &&
              !HasParameter(adagrad, "lr_decay"),
          "Adagrad should not advertise unsupported learning-rate decay");
}

} // namespace

int main() {
    auto& metadata = cyxwiz::NodeMetadataRegistry::Instance();
    metadata.Initialize();

    CheckPropertyTruthInventory(metadata);
    CheckMultiHeadAttentionReferenceContract(metadata);
    CheckTransformerFamilyReferenceContract(metadata);
    CheckDenseReferenceContract(metadata);
    CheckStandardScalerReferenceContract(metadata);
    CheckUtilityNodeFamilyContract(metadata);
    CheckSimulationNodeFamilyContract(metadata);
    CheckDataInputDialogReferenceContract(metadata);
    CheckConv2DBlockedReferenceContract(metadata);
    CheckConvolutionPoolingBlockedFamilyContract(metadata);
    CheckBlockedUpsamplingFamilyContract(metadata);
    CheckBlockedNormalizationFamilyContract(metadata);
    CheckBlockedAttentionFamilyContract(metadata);
    CheckBlockedRecurrentCompatibilityContract(metadata);
    CheckImplementedRecurrentConfigurationContract(metadata);
    CheckDataValidatorReferenceContract(metadata);
    CheckDataValidatorOutputMigrationGuard();
    CheckTabularAnalyticsFamilyContract(metadata);
    CheckEvaluationTableFamilyContract(metadata);
    CheckEvaluationTableMigrationGuard();
    CheckSignalProcessingFamilyContract(metadata);
    CheckTextVectorizerSentimentFamilyContract(metadata);
    CheckTabularTransformFamilyContract(metadata);
    CheckClusteringFamilyContract(metadata);
    CheckPcaContract(metadata);
    CheckClassicalRegressionFamilyContract(metadata);
    CheckBlockedClassifierFamilyContract(metadata);
    CheckBlockedSchedulerFamilyContract(metadata);
    CheckBlockedRegularizationFamilyContract(metadata);
    CheckClassicalTreeFamilyContract(metadata);
    CheckClassicalTreeMigrationGuard();
    CheckStaticCreationAdapter(metadata);
    CheckPreprocessingScalerEncoderFamilyContract(metadata);
    CheckCoreLayerFamilyContract(metadata);
    CheckShapeOperationFamilyContract(metadata);
    CheckTensorMathFamilyContract(metadata);
    CheckTensorFanInFamilyContract(metadata);
    CheckLossFamilyContract(metadata);
    CheckOptimizerFamilyContract(metadata);

    {
        const auto* data_input = metadata.GetMetadata(gui::NodeType::DataInput);
        Check(data_input != nullptr, "DataInput metadata should exist");
        Check(ContainsString(data_input->keywords, "csv"),
              "DataInput metadata should keep CSV keyword");
        Check(ContainsString(data_input->keywords, "tsv"),
              "DataInput metadata should keep TSV keyword");
        Check(ContainsString(data_input->keywords, "parquet"),
              "DataInput metadata should keep Parquet keyword");
        Check(ContainsString(data_input->keywords, "feather"),
              "DataInput metadata should keep Feather keyword");
        Check(ContainsString(data_input->keywords, "arrow"),
              "DataInput metadata should keep Arrow keyword");
        Check(!ContainsString(data_input->keywords, "json"),
              "DataInput metadata should not advertise JSON runtime input");
        Check(!ContainsString(data_input->keywords, "excel"),
              "DataInput metadata should not advertise Excel runtime input");
        Check(!ContainsString(data_input->keywords, "hdf5"),
              "DataInput metadata should not advertise HDF5 runtime input");
        Check(HasOutputType(data_input, "Dataset", gui::PinType::Dataset),
              "DataInput metadata must expose one Dataset artifact output");
        Check(!HasOutputType(data_input, "Data", gui::PinType::Tensor) &&
                  !HasOutputType(data_input, "Labels", gui::PinType::Labels),
              "DataInput metadata must not expose legacy Tensor/Labels outputs");

        const auto* file_path = FindParameter(data_input, "file_path");
        Check(file_path != nullptr,
              "DataInput metadata should keep file_path parameter");
        Check(file_path->validation.find("*.csv") != std::string::npos &&
                  file_path->validation.find("*.tsv") != std::string::npos &&
                  file_path->validation.find("*.parquet") != std::string::npos &&
                  file_path->validation.find("*.feather") != std::string::npos &&
                  file_path->validation.find("*.arrow") != std::string::npos &&
                  file_path->validation.find("*.ipc") != std::string::npos,
              "DataInput file filter should list runtime-supported tabular formats");
        Check(file_path->validation.find("*.json") == std::string::npos &&
                  file_path->validation.find("*.xlsx") == std::string::npos &&
                  file_path->validation.find("*.hdf5") == std::string::npos,
              "DataInput file filter should not advertise unsupported runtime formats");

        const auto* file_type = FindParameter(data_input, "file_type");
        Check(file_type != nullptr,
              "DataInput metadata should expose canonical file_type selector");
        Check(FindParameter(data_input, "type") == nullptr,
              "DataInput metadata should not recreate the legacy type alias");
        Check(file_type->default_value == "auto",
              "DataInput file_type should default to runtime auto detection");
        Check(ContainsString(file_type->enum_values, "csv") &&
                  ContainsString(file_type->enum_values, "tsv") &&
                  ContainsString(file_type->enum_values, "parquet") &&
                  ContainsString(file_type->enum_values, "feather") &&
                  ContainsString(file_type->enum_values, "arrow") &&
                  ContainsString(file_type->enum_values, "ipc"),
              "DataInput file_type enum should list runtime-supported formats");
        Check(!ContainsString(file_type->enum_values, "json") &&
                  !ContainsString(file_type->enum_values, "excel") &&
                  !ContainsString(file_type->enum_values, "hdf5"),
              "DataInput file_type enum should not advertise unsupported runtime formats");

        const auto source_type_axes =
            cyxwiz::ResolvePipelineAllowedParameterValues("DataInput");
        auto source_type_axis = std::find_if(
            source_type_axes.begin(),
            source_type_axes.end(),
            [](const cyxwiz::PipelineAllowedParameterValuesRuntimeCapability&
                   capability) {
                return std::string(capability.parameter_name) == "source_type";
            });
        Check(source_type_axis != source_type_axes.end(),
              "DataInput runtime support should expose source_type limits");
        const std::vector<std::string> source_type_values(
            source_type_axis->allowed_values.begin(),
            source_type_axis->allowed_values.end());
        Check(ContainsString(source_type_values, "file") &&
                  ContainsString(source_type_values, "folder"),
              "DataInput source_type should allow file and folder");
        Check(!ContainsString(source_type_values, "ml_dataset"),
              "DataInput source_type should not advertise ml_dataset without a runtime owner");
    }

    {
        struct SequenceMetadataCase {
            gui::NodeType type;
            std::string name;
            std::string required_parameter;
        };

        const SequenceMetadataCase cases[] = {
            {gui::NodeType::NERSequenceBuilder, "NER Sequence Builder",
             "token_column"},
            {gui::NodeType::TokenVocabulary, "Token Vocabulary", "min_freq"},
            {gui::NodeType::POSVocabulary, "POS Vocabulary", "min_freq"},
            {gui::NodeType::NERTagVocabulary, "NER Tag Vocabulary",
             "outside_tag"},
            {gui::NodeType::SequenceTagOutput, "Sequence Tag Output",
             "num_tags"},
        };

        for (const auto& node_case : cases) {
            const auto* meta = metadata.GetMetadata(node_case.type);
            Check(meta != nullptr,
                  "sequence metadata missing: " + node_case.name);
            Check(meta->status == cyxwiz::NodeImplementationStatus::Implemented,
                  "sequence metadata should be implemented: " +
                      node_case.name);
            Check(meta->badge != "Blocked",
                  "sequence metadata should not be blocked: " +
                      node_case.name);
            Check(HasParameter(meta, node_case.required_parameter),
                  "sequence metadata missing required parameter: " +
                      node_case.name);
            Check(meta->brief_description.find("sequence") !=
                      std::string::npos ||
                      meta->brief_description.find("vocabulary") !=
                          std::string::npos,
                  "sequence metadata should describe its task: " +
                      node_case.name);
            if (node_case.type != gui::NodeType::SequenceTagOutput) {
                CheckSupportAxis(meta, "Pipeline Executor", "supported", true,
                                 node_case.name);
            }
        }

        Check(HasOutputType(metadata.GetMetadata(gui::NodeType::TokenVocabulary),
                            "Token Vocabulary", gui::PinType::Parameters),
              "TokenVocabulary should expose a Parameters output");
        Check(HasOutputType(metadata.GetMetadata(gui::NodeType::POSVocabulary),
                            "POS Vocabulary", gui::PinType::Parameters),
              "POSVocabulary should expose a Parameters output");
        Check(HasOutputType(metadata.GetMetadata(gui::NodeType::NERTagVocabulary),
                            "NER Tag Vocabulary", gui::PinType::Parameters),
              "NERTagVocabulary should expose a Parameters output");
        Check(HasOutputType(metadata.GetMetadata(gui::NodeType::SequenceTagOutput),
                            "Predictions", gui::PinType::Tensor),
              "SequenceTagOutput should expose a Tensor output");
    }

    {
        struct MetricLearningMetadataCase {
            gui::NodeType type;
            std::string name;
            std::string required_parameter;
        };

        const MetricLearningMetadataCase cases[] = {
            {gui::NodeType::PairDatasetBuilder, "Pair Dataset Builder",
             "sample_a_column"},
            {gui::NodeType::TripletDatasetBuilder, "Triplet Dataset Builder",
             "anchor_column"},
            {gui::NodeType::SharedEncoder, "Shared Encoder", "encoder_id"},
            {gui::NodeType::SiameseBranch, "Siamese Branch", "branch"},
            {gui::NodeType::ContrastiveLoss, "Contrastive Loss", "margin"},
            {gui::NodeType::CosineEmbeddingLoss, "Cosine Embedding Loss",
             "margin"},
            {gui::NodeType::TripletLoss, "Triplet Loss", "margin"},
            {gui::NodeType::PairMetrics, "Pair Metrics", "threshold"},
            {gui::NodeType::RetrievalMetrics, "Retrieval Metrics", "k"},
            {gui::NodeType::EmbeddingOutput, "Embedding Output",
             "include_metadata"},
            {gui::NodeType::PairScoreOutput, "Pair Score Output",
             "score_mode"},
        };

        for (const auto& node_case : cases) {
            const auto* meta = metadata.GetMetadata(node_case.type);
            Check(meta != nullptr,
                  "metric-learning metadata missing: " + node_case.name);
            Check(meta->status == cyxwiz::NodeImplementationStatus::Template,
                  "metric-learning metadata should be template-blocked: " +
                      node_case.name);
            Check(meta->badge == "Blocked",
                  "metric-learning metadata should show blocked badge: " +
                      node_case.name);
            Check(!cyxwiz::CanAddNodeToGraph(*meta),
                  "metric-learning node should remain unavailable: " +
                      node_case.name);
            Check(meta->category == gui::NodeCategory::Training,
                  "metric-learning metadata should live under Training: " +
                      node_case.name);
            Check(HasParameter(meta, node_case.required_parameter),
                  "metric-learning metadata missing required parameter: " +
                      node_case.name);
            Check(meta->brief_description.find("Metric") != std::string::npos ||
                      meta->brief_description.find("metric") !=
                          std::string::npos ||
                      ContainsString(meta->keywords, "metric"),
                  "metric-learning metadata should describe metric-learning: " +
                      node_case.name);
            Check(!meta->help_text.empty(),
                  "metric-learning metadata should explain its bounded contract: " +
                      node_case.name);
            const auto support =
                cyxwiz::ResolvePipelineTrainingBackendSupport(node_case.type);
            Check(support.mode ==
                      cyxwiz::PipelineTrainingBackendSupportMode::
                          UnsupportedTrainingWorkflow &&
                      !support.compile_supported && !support.training_supported,
                  "metric-learning node should fail closed as a training workflow: " +
                      node_case.name);
            Check(cyxwiz::IsPipelineUnsupportedTrainingWorkflowNode(
                      node_case.type),
                  "metric-learning workflow capability should resolve: " +
                      node_case.name);
            Check(cyxwiz::PipelineOperatorFactory::Instance().Create(
                      node_case.type) == nullptr,
                  "metric-learning node should not claim a PipelineExecutor owner: " +
                      node_case.name);
        }

        Check(ParameterMatches(
                  metadata.GetMetadata(gui::NodeType::PairDatasetBuilder),
                  "label_convention", "enum", "contrastive_zero_similar"),
              "PairDatasetBuilder should preserve its label convention");
        Check(ParameterMatches(
                  metadata.GetMetadata(gui::NodeType::SharedEncoder),
                  "encoder_id", "string", "shared_encoder"),
              "SharedEncoder should preserve its legacy identity");
        Check(ParameterMatches(
                  metadata.GetMetadata(gui::NodeType::ContrastiveLoss),
                  "margin", "float", "1.0") &&
                  ParameterMatches(
                      metadata.GetMetadata(gui::NodeType::CosineEmbeddingLoss),
                      "margin", "float", "0.0") &&
                  ParameterMatches(
                      metadata.GetMetadata(gui::NodeType::TripletLoss),
                      "margin", "float", "1.0"),
              "metric-learning losses should preserve their distinct margins");
        Check(ParameterMatches(
                  metadata.GetMetadata(gui::NodeType::PairScoreOutput),
                  "score_mode", "enum", "distance"),
              "PairScoreOutput should preserve its scoring mode");

        Check(HasOutputType(metadata.GetMetadata(gui::NodeType::EmbeddingOutput),
                            "Embedding Records", gui::PinType::Dataset),
              "EmbeddingOutput should expose dataset records");
        Check(HasOutputType(metadata.GetMetadata(gui::NodeType::PairScoreOutput),
                            "Pair Scores", gui::PinType::Dataset),
              "PairScoreOutput should expose pair score records");
    }

    {
        const auto* data_output = metadata.GetMetadata(gui::NodeType::DataOutput);
        Check(data_output != nullptr, "DataOutput metadata should exist");
        const auto* file_type = FindParameter(data_output, "file_type");
        Check(file_type != nullptr,
              "DataOutput metadata should expose runtime output format selector");
        Check(file_type->default_value == "csv",
              "DataOutput file_type should default to CSV");
        Check(ContainsString(file_type->enum_values, "csv") &&
                  ContainsString(file_type->enum_values, "parquet"),
              "DataOutput file_type enum should list runtime-supported output formats");
        Check(!ContainsString(file_type->enum_values, "json") &&
                  !ContainsString(file_type->enum_values, "excel"),
              "DataOutput file_type enum should not advertise unsupported exporters");

        const auto* export_csv = metadata.GetMetadata(gui::NodeType::ExportCSV);
        Check(export_csv != nullptr, "ExportCSV metadata should exist");
        Check(export_csv->brief_description.find("Arrow table") != std::string::npos,
              "ExportCSV metadata should describe the real Arrow-table export path");

        const auto* export_json = metadata.GetMetadata(gui::NodeType::ExportJSON);
        Check(export_json != nullptr, "ExportJSON metadata should exist");
        Check(export_json->brief_description.find("Arrow table") != std::string::npos,
              "ExportJSON metadata should describe the real Arrow-table export path");

        const auto* export_parquet = metadata.GetMetadata(gui::NodeType::ExportParquet);
        Check(export_parquet != nullptr, "ExportParquet metadata should exist");
        Check(export_parquet->brief_description.find("Arrow table") != std::string::npos,
              "ExportParquet metadata should describe the real Arrow-table export path");

        const auto* export_sql = metadata.GetMetadata(gui::NodeType::ExportSQL);
        Check(export_sql != nullptr, "ExportSQL metadata should exist");
        Check(export_sql->status == cyxwiz::NodeImplementationStatus::Template,
              "ExportSQL metadata should remain blocked until SQL export is real");
        Check(export_sql->name.find("planned") != std::string::npos &&
                  export_sql->brief_description.find("not implemented") != std::string::npos,
              "ExportSQL metadata should visibly say the export is planned/not implemented");

        const auto* export_excel = metadata.GetMetadata(gui::NodeType::ExportExcel);
        Check(export_excel != nullptr, "ExportExcel metadata should exist");
        Check(export_excel->status == cyxwiz::NodeImplementationStatus::Template,
              "ExportExcel metadata should remain blocked until Excel export is real");
        Check(export_excel->name.find("planned") != std::string::npos &&
                  export_excel->brief_description.find("not implemented") != std::string::npos,
              "ExportExcel metadata should visibly say the export is planned/not implemented");

        const auto* data_profiler = metadata.GetMetadata(gui::NodeType::DataProfiler);
        Check(data_profiler != nullptr, "DataProfiler metadata should exist");
        Check(!HasParameter(data_profiler, "minimal"),
              "DataProfiler metadata should not expose minimal until the executor consumes it");
    }

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

        std::set<std::string> float_parameter_names;
        for (const auto& capability :
             cyxwiz::GetPipelineFloatParameterRuntimeCapabilities()) {
            const std::string key =
                std::string(capability.legacy_type_name) + "." +
                capability.parameter_name;
            Check(float_parameter_names.insert(key).second,
                  "duplicate float-parameter runtime capability: " + key);
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

        std::set<int> unsupported_training_workflow_types;
        for (const auto& capability :
             cyxwiz::GetPipelineUnsupportedTrainingWorkflowCapabilities()) {
            const int key = static_cast<int>(capability.node_type);
            Check(unsupported_training_workflow_types.insert(key).second,
                  "duplicate unsupported training workflow capability: " +
                      TypeId(capability.node_type));
            Check(capability.reason != nullptr &&
                      std::string(capability.reason).size() > 16,
                  "unsupported training workflow reason is too weak: " +
                      TypeId(capability.node_type));
            Check(cyxwiz::IsPipelineUnsupportedTrainingWorkflowNode(
                      capability.node_type),
                  "unsupported training workflow capability does not resolve: " +
                      TypeId(capability.node_type));
            const auto support = cyxwiz::ResolvePipelineTrainingBackendSupport(
                capability.node_type);
            Check(support.mode ==
                      cyxwiz::PipelineTrainingBackendSupportMode::
                          UnsupportedTrainingWorkflow,
                  "unsupported training workflow should resolve through unified support: " +
                      TypeId(capability.node_type));
            Check(!support.compile_supported && !support.training_supported,
                  "unsupported training workflow should block compile/training: " +
                      TypeId(capability.node_type));
            Check(support.reason == capability.reason,
                  "unsupported training workflow reason should be shared: " +
                      TypeId(capability.node_type));
        }

        std::set<int> supported_training_types;
        for (const auto& capability :
             cyxwiz::GetPipelineSupportedTrainingBackendCapabilities()) {
            const int key = static_cast<int>(capability.node_type);
            Check(supported_training_types.insert(key).second,
                  "duplicate supported training backend capability: " +
                      TypeId(capability.node_type));
            Check(capability.reason != nullptr &&
                      std::string(capability.reason).size() > 16,
                  "supported training backend reason is too weak: " +
                      TypeId(capability.node_type));
            Check(cyxwiz::IsPipelineSupportedTrainingBackendNode(
                      capability.node_type),
                  "supported training backend capability does not resolve: " +
                      TypeId(capability.node_type));
            Check(!cyxwiz::IsPipelineUnsupportedSequentialModelLayer(
                      capability.node_type) &&
                      !cyxwiz::IsPipelineUnsupportedTrainingControlNode(
                          capability.node_type) &&
                      !cyxwiz::IsPipelineUnsupportedTrainingWorkflowNode(
                          capability.node_type),
                  "supported training backend should not overlap unsupported lists: " +
                      TypeId(capability.node_type));
            const auto support = cyxwiz::ResolvePipelineTrainingBackendSupport(
                capability.node_type);
            Check(support.mode ==
                      cyxwiz::PipelineTrainingBackendSupportMode::Allowed,
                  "supported training backend should resolve as allowed: " +
                      TypeId(capability.node_type));
            Check(support.compile_supported && support.training_supported,
                  "supported training backend should allow compile/training: " +
                      TypeId(capability.node_type));
            Check(support.reason == capability.reason,
                  "supported training backend reason should be shared: " +
                      TypeId(capability.node_type));
        }

        std::set<int> materializer_storage_backends;
        int materializer_supported_backends = 0;
        const std::set<std::string> expected_materializer_storage_backends = {
            "ArrowTable",
            "ParquetBacked",
            "ImageDataset",
            "AudioDataset",
            "TextDataset",
        };
        for (const auto& capability :
             cyxwiz::GetPipelineMaterializerStorageBackendCapabilities()) {
            const std::string backend_name =
                cyxwiz::PipelineStorageBackendName(capability.backend);
            const int key = static_cast<int>(capability.backend);
            Check(materializer_storage_backends.insert(key).second,
                  "duplicate materializer storage backend capability: " +
                      backend_name);
            Check(ContainsString(expected_materializer_storage_backends,
                                 backend_name),
                  "unexpected materializer storage backend capability: " +
                      backend_name);
            const auto resolved =
                cyxwiz::ResolvePipelineMaterializerStorageBackendSupport(
                    capability.backend);
            Check(resolved.backend == capability.backend,
                  "materializer storage backend capability does not resolve: " +
                      backend_name);
            Check(resolved.materializer_supported ==
                      capability.materializer_supported,
                  "materializer storage backend support mismatch: " +
                      backend_name);
            Check(resolved.storage_support == capability.storage_support,
                  "materializer storage support scope mismatch: " +
                      backend_name);
            Check(capability.reason != nullptr &&
                      std::string(capability.reason).size() > 16,
                  "materializer storage backend reason is too weak: " +
                      backend_name);
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
        Check(materializer_storage_backends.size() ==
                  expected_materializer_storage_backends.size(),
              "materializer storage scope should pin Arrow plus four pass-through domains");
    }

    for (const auto& capability : cyxwiz::GetPipelineOperatorRuntimeCapabilities()) {
        const std::string name = capability.legacy_type_name;
        CheckRuntimeOwnerContract(
            cyxwiz::ResolvePipelineRuntimeSupport(name),
            cyxwiz::PipelineRuntimeSupportMode::OperatorBacked,
            cyxwiz::PipelineRuntimeImplementationOwner::PipelineOperatorFactory,
            "operator capability " + name);
    }

    for (const auto& capability : cyxwiz::GetPipelineLegacyRuntimeCapabilities()) {
        const std::string name = capability.legacy_type_name;
        CheckRuntimeOwnerContract(
            cyxwiz::ResolvePipelineRuntimeSupport(name),
            cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
            cyxwiz::PipelineRuntimeImplementationOwner::PipelineExecutor,
            "legacy capability " + name);
    }

    for (const auto& capability : cyxwiz::GetPipelineFailClosedRuntimeCapabilities()) {
        const std::string name = capability.legacy_type_name;
        CheckRuntimeOwnerContract(
            cyxwiz::ResolvePipelineRuntimeSupport(name),
            cyxwiz::PipelineRuntimeSupportMode::FailClosed,
            cyxwiz::PipelineRuntimeImplementationOwner::None,
            "fail-closed capability " + name);
    }

    struct ExpectedAliasDecision {
        const char* alias;
        const char* canonical;
        gui::NodeType canonical_type;
        cyxwiz::PipelineLegacyAliasDecision decision;
    };
    const std::vector<ExpectedAliasDecision> expected_alias_decisions = {
        {"SaveDataset", "DataOutput", gui::NodeType::DataOutput,
         cyxwiz::PipelineLegacyAliasDecision::HiddenCompatibilityAlias},
        {"DeployToNodeEditor", "DeployToNodeEditorNode",
         gui::NodeType::DeployToNodeEditorNode,
         cyxwiz::PipelineLegacyAliasDecision::HiddenCompatibilityAlias},
        {"TextClean", "TextCleanNode", gui::NodeType::TextCleanNode,
         cyxwiz::PipelineLegacyAliasDecision::NormalizeToCanonical},
        {"TextTokenize", "TextTokenizer", gui::NodeType::TextTokenizer,
         cyxwiz::PipelineLegacyAliasDecision::HiddenCompatibilityAlias},
        {"TextVectorize", "CountVectorizer", gui::NodeType::CountVectorizer,
         cyxwiz::PipelineLegacyAliasDecision::HiddenCompatibilityAlias},
        {"TSWindow", "TimeSeriesWindow", gui::NodeType::TimeSeriesWindow,
         cyxwiz::PipelineLegacyAliasDecision::HiddenCompatibilityAlias},
        {"TSFeatures", "TimeSeriesFeatures", gui::NodeType::TimeSeriesFeatures,
         cyxwiz::PipelineLegacyAliasDecision::HiddenCompatibilityAlias},
        {"TSLag", "TimeSeriesLag", gui::NodeType::TimeSeriesLag,
         cyxwiz::PipelineLegacyAliasDecision::NormalizeToCanonical},
        {"TSDiff", "Differencing", gui::NodeType::Differencing,
         cyxwiz::PipelineLegacyAliasDecision::HiddenCompatibilityAlias},
        {"PolynomialFeatures", "PolynomialFeaturesNode",
         gui::NodeType::PolynomialFeaturesNode,
         cyxwiz::PipelineLegacyAliasDecision::NormalizeToCanonical},
        {"Binning", "BinningNode", gui::NodeType::BinningNode,
         cyxwiz::PipelineLegacyAliasDecision::NormalizeToCanonical},
    };
    std::set<std::string> observed_alias_decisions;
    for (const auto& expected : expected_alias_decisions) {
        const auto* decision =
            cyxwiz::ResolvePipelineLegacyAliasDecision(expected.alias);
        Check(decision != nullptr,
              std::string("missing alias retirement decision: ") +
                  expected.alias);
        Check(observed_alias_decisions.insert(expected.alias).second,
              std::string("duplicate expected alias decision fixture: ") +
                  expected.alias);
        Check(std::string(decision->canonical_type_name) == expected.canonical,
              std::string("alias canonical target drift: ") + expected.alias);
        Check(decision->canonical_node_type == expected.canonical_type,
              std::string("alias canonical node type drift: ") +
                  expected.alias);
        Check(decision->decision == expected.decision,
              std::string("alias retirement decision drift: ") +
                  expected.alias);
        Check(decision->reason != nullptr &&
                  std::string(decision->reason).size() > 16,
              std::string("alias decision reason is too weak: ") +
                  expected.alias);
        Check(cyxwiz::IsPipelineLegacyRuntimeNode(expected.alias),
              std::string("alias decision should point to a legacy runtime alias: ") +
                  expected.alias);
        Check(cyxwiz::ResolvePipelineRuntimeSupport(
                  decision->canonical_node_type).mode !=
                  cyxwiz::PipelineRuntimeSupportMode::Unknown,
              std::string("alias canonical target must resolve runtime support: ") +
                  expected.alias);
    }
    Check(observed_alias_decisions.size() ==
              cyxwiz::GetPipelineLegacyAliasDecisionCapabilities().size(),
          "alias decision table should contain only expected Track 22 aliases");

    const std::set<std::string> retirement_priority_aliases = {
        "SaveDataset",
        "DeployToNodeEditor",
        "TextClean",
        "TextTokenize",
        "TextVectorize",
        "TSWindow",
        "TSFeatures",
        "TSLag",
        "TSDiff",
        "PolynomialFeatures",
        "Binning",
    };
    std::set<std::string> observed_retirement_priority_aliases;
    for (const auto& capability : cyxwiz::GetPipelineLegacyRuntimeCapabilities()) {
        const std::string name = capability.legacy_type_name;
        if (!ContainsString(retirement_priority_aliases, name)) {
            continue;
        }
        observed_retirement_priority_aliases.insert(name);
        Check(!cyxwiz::IsPipelineOperatorRuntimeNode(name),
              "retirement-priority alias should not be operator-backed yet: " +
                  name);
        Check(!cyxwiz::IsPipelineFailClosedRuntimeNode(name),
              "retirement-priority alias should not be fail-closed: " + name);
        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(name);
        CheckRuntimeOwnerContract(
            support,
            cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
            cyxwiz::PipelineRuntimeImplementationOwner::PipelineExecutor,
            name);
        Check(support.node_type.has_value(),
              "retirement-priority alias should resolve typed metadata before plan construction: " +
                  name);
    }
    for (const auto& alias : retirement_priority_aliases) {
        Check(ContainsString(observed_retirement_priority_aliases, alias),
              "missing retirement-priority alias baseline: " + alias);
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

        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(type);
        Check(support.mode == cyxwiz::PipelineRuntimeSupportMode::OperatorBacked,
              "factory type missing central operator-backed runtime support: " +
                  TypeId(type));
        Check(support.operator_type.has_value() && *support.operator_type == type,
              "factory type resolves to wrong operator runtime type: " +
                  TypeId(type));
        Check(support.implementation_owner ==
                  cyxwiz::PipelineRuntimeImplementationOwner::
                      PipelineOperatorFactory,
              "factory type should be owned by PipelineOperatorFactory: " +
                  TypeId(type));
        Check(support.materializer_arrow_table_supported,
              "factory type should advertise Arrow materializer support: " +
                  TypeId(type));
    }

    for (const auto& data_studio_node :
         cyxwiz::NodeRegistry::Instance().GetAllNodeTypes()) {
        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(
            data_studio_node.type_id);
        Check(support.mode != cyxwiz::PipelineRuntimeSupportMode::Unknown,
              "Data Studio node registry advertises unknown runtime type: " +
                  data_studio_node.type_id);
        Check(support.pipeline_executor_supported,
              "Data Studio node registry advertises unsupported runtime type: " +
                  data_studio_node.type_id);
        Check(data_studio_node.type_id != "ArrowDataset" &&
                  data_studio_node.type_id != "Aggregate" &&
                  data_studio_node.type_id != "DetectOutliers",
              "Data Studio node registry should not advertise stale unsupported type: " +
                  data_studio_node.type_id);
        for (const auto& parameter : data_studio_node.parameters) {
            const auto catalog_values = ParseCatalogEnumValues(parameter.second);
            if (catalog_values.empty()) {
                continue;
            }

            const auto runtime_axis = std::find_if(
                support.allowed_parameter_values.begin(),
                support.allowed_parameter_values.end(),
                [&parameter](
                    const cyxwiz::PipelineAllowedParameterValuesRuntimeCapability&
                        axis) {
                    return std::string(axis.parameter_name) == parameter.first;
                });
            if (runtime_axis == support.allowed_parameter_values.end()) {
                continue;
            }

            for (const auto& catalog_value : catalog_values) {
                const bool runtime_accepts_value = std::find_if(
                    runtime_axis->allowed_values.begin(),
                    runtime_axis->allowed_values.end(),
                    [&catalog_value](const char* runtime_value) {
                        return runtime_value != nullptr &&
                               catalog_value == runtime_value;
                    }) != runtime_axis->allowed_values.end();
                Check(runtime_accepts_value,
                      "Data Studio node registry advertises unsupported enum value: " +
                          data_studio_node.type_id + "." + parameter.first +
                          "=" + catalog_value);
            }
        }
    }

    const std::set<std::string> intentional_quick_add_compatibility_nodes;
    const std::set<std::string> disallowed_quick_add_legacy_aliases = {
        "ArrowDataset",
        "FileInput",
        "SaveDataset",
        "RemoveDuplicates",
        "TextClean",
        "TextTokenize",
        "TextVectorize",
        "TSWindow",
        "TSFeatures",
        "TSLag",
        "TSDiff",
        "PCA",
        "PolynomialFeatures",
        "Binning",
    };
    std::vector<std::string> data_studio_quick_add_type_ids;
    for (const auto& item : cyxwiz::PipelineCanvas::GetQuickAddNodes()) {
        Check(item.label != nullptr && std::string(item.label).size() > 0,
              "Data Studio quick-add item should have a label");
        Check(item.type_id != nullptr && std::string(item.type_id).size() > 0,
              "Data Studio quick-add item should have a type id");
        const std::string type_id = item.type_id;
        data_studio_quick_add_type_ids.push_back(type_id);
        Check(disallowed_quick_add_legacy_aliases.count(type_id) == 0,
              "Data Studio quick-add should not promote legacy alias: " +
                  type_id);
        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(type_id);
        Check(support.mode != cyxwiz::PipelineRuntimeSupportMode::Unknown,
              "Data Studio quick-add node has unknown runtime support: " +
                  type_id);
        Check(support.pipeline_executor_supported,
              "Data Studio quick-add node is not PipelineExecutor-supported: " +
                  type_id);
        Check(support.fail_mode == cyxwiz::PipelineRuntimeFailMode::Real,
              "Data Studio quick-add node should have real fail mode: " +
                  type_id);
        if (intentional_quick_add_compatibility_nodes.count(type_id) == 0) {
            Check(support.node_type.has_value(),
                  "Data Studio quick-add canonical node should resolve typed metadata: " +
                      type_id);
            const auto* meta = metadata.GetMetadata(*support.node_type);
            Check(meta != nullptr,
                  "Data Studio quick-add canonical node metadata missing: " +
                      type_id);
            Check(meta->status == cyxwiz::NodeImplementationStatus::Implemented,
                  "Data Studio quick-add canonical node should be implemented metadata: " +
                      type_id);
        }
    }
    for (const auto& legacy_alias : disallowed_quick_add_legacy_aliases) {
        Check(std::find(data_studio_quick_add_type_ids.begin(),
                        data_studio_quick_add_type_ids.end(),
                        legacy_alias) == data_studio_quick_add_type_ids.end(),
              "Data Studio quick-add contract contains disallowed alias: " +
                  legacy_alias);
    }

    for (const auto& capability : cyxwiz::GetPipelineOperatorRuntimeCapabilities()) {
        Check(!cyxwiz::IsPipelineLegacyRuntimeNode(capability.legacy_type_name),
              std::string("operator-backed runtime name is also legacy-dispatched: ") +
                  capability.legacy_type_name);
        Check(!cyxwiz::IsPipelineFailClosedRuntimeNode(capability.legacy_type_name),
              std::string("operator-backed runtime name is also fail-closed: ") +
                  capability.legacy_type_name);
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
        Check(meta->status == cyxwiz::NodeImplementationStatus::Implemented,
              std::string("operator-backed real runtime metadata should be implemented: ") +
                  capability.legacy_type_name);
        Check(meta->badge != "Blocked",
              std::string("operator-backed real runtime metadata should not be blocked: ") +
                  capability.legacy_type_name);
        CheckSupportAxis(
            meta,
            "Runtime",
            cyxwiz::PipelineRuntimeSupportModeName(support.mode),
            true,
            capability.legacy_type_name);
        CheckSupportAxis(
            meta,
            "Fail Mode",
            cyxwiz::PipelineRuntimeFailModeName(support.fail_mode),
            true,
            capability.legacy_type_name);
        CheckSupportAxis(
            meta,
            "Pipeline Executor",
            "supported",
            true,
            capability.legacy_type_name);
        CheckSupportAxis(
            meta,
            "Materializer",
            cyxwiz::PipelineMaterializerStorageSupportName(
                support.materializer_storage_support),
            true,
            capability.legacy_type_name);
        CheckSupportAxis(
            meta,
            "Implementation Owner",
            cyxwiz::PipelineRuntimeImplementationOwnerName(
                support.implementation_owner),
            true,
            capability.legacy_type_name);
        CheckSupportAxis(
            meta,
            "Support State",
            "real",
            true,
            capability.legacy_type_name);
    }

    const std::set<std::string> allowed_untyped_fail_closed_names = {
        // Legacy canvas aliases with no one-to-one metadata node. The current
        // typed metadata entries are DataSplit and DataInput.
        "TrainTestSplit",
    };

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
        const std::string fail_closed_reason =
            capability.reason != nullptr ? capability.reason : "";
        Check(fail_closed_reason.find("is still a") == std::string::npos &&
                  fail_closed_reason.find("is still an") == std::string::npos,
              std::string("fail-closed runtime reason should describe current hard-fail state: ") +
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
        if (!expected_metadata_type.has_value()) {
            Check(allowed_untyped_fail_closed_names.count(
                      capability.legacy_type_name) > 0,
                  std::string("fail-closed runtime should be typed unless it is a documented legacy alias: ") +
                      capability.legacy_type_name);
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
        if (expected_metadata_type.has_value()) {
            const auto* meta = metadata.GetMetadata(*expected_metadata_type);
            if (meta != nullptr) {
                const std::string reason =
                    capability.reason != nullptr ? capability.reason : "";
                if (capability.blocks_metadata_status) {
                    Check(meta->status == cyxwiz::NodeImplementationStatus::Template,
                          std::string("blocked fail-closed metadata should not be implemented: ") +
                              capability.legacy_type_name);
                    Check(meta->badge == "Blocked",
                          std::string("blocked fail-closed metadata should carry blocked badge: ") +
                              capability.legacy_type_name);
                    CheckSupportAxis(
                        meta,
                        "Support State",
                        "blocked",
                        false,
                        capability.legacy_type_name);
                } else {
                    const auto* support_state = FindSupportAxis(meta, "Support State");
                    Check(support_state != nullptr,
                          std::string("missing support axis Support State: ") +
                              capability.legacy_type_name);
                    Check(support_state->value == "partial" ||
                              support_state->value == "real",
                          std::string("non-blocking fail-closed metadata should keep partial or real support state: ") +
                              capability.legacy_type_name);
                    Check(support_state->supported,
                          std::string("non-blocking fail-closed metadata should keep supported state: ") +
                              capability.legacy_type_name);
                }
                CheckSupportAxis(
                    meta,
                    "Runtime",
                    cyxwiz::PipelineRuntimeSupportModeName(support.mode),
                    false,
                    capability.legacy_type_name);
                CheckSupportAxis(
                    meta,
                    "Fail Mode",
                    cyxwiz::PipelineRuntimeFailModeName(support.fail_mode),
                    false,
                    capability.legacy_type_name);
                CheckSupportAxis(
                    meta,
                    "Pipeline Executor",
                    "unsupported",
                    false,
                    capability.legacy_type_name);
                CheckSupportAxis(
                    meta,
                    "Materializer",
                    cyxwiz::PipelineMaterializerStorageSupportName(
                        support.materializer_storage_support),
                    false,
                    capability.legacy_type_name);
                if (capability.blocks_metadata_status) {
                    const auto* implementation_owner =
                        FindSupportAxis(meta, "Implementation Owner");
                    Check(implementation_owner != nullptr,
                          std::string("missing support axis Implementation Owner: ") +
                              capability.legacy_type_name);
                    Check(implementation_owner->value ==
                              cyxwiz::PipelineRuntimeImplementationOwnerName(
                                  support.implementation_owner) ||
                              implementation_owner->value == "training_backend",
                          std::string("fail-closed metadata should keep runtime owner or stronger training owner: ") +
                              capability.legacy_type_name);
                }
                CheckSupportAxisReasonContains(
                    meta,
                    "Runtime",
                    reason,
                    capability.legacy_type_name);
            }
        }
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
        Check(capability.node_type.has_value(),
              std::string("legacy runtime must resolve typed metadata, not "
                          "string-only dispatch: ") +
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
        if (capability.node_type.has_value()) {
            const auto* meta = metadata.GetMetadata(*capability.node_type);
            if (meta != nullptr) {
                const auto* runtime_axis = FindSupportAxis(meta, "Runtime");
                const bool canonical_operator_metadata =
                    runtime_axis != nullptr &&
                    runtime_axis->value == "operator_backed";
                if (!canonical_operator_metadata) {
                    CheckSupportAxis(
                        meta,
                        "Runtime",
                        cyxwiz::PipelineRuntimeSupportModeName(support.mode),
                        true,
                        capability.legacy_type_name);
                    CheckSupportAxis(
                        meta,
                        "Materializer",
                        cyxwiz::PipelineMaterializerStorageSupportName(
                            support.materializer_storage_support),
                        false,
                        capability.legacy_type_name);
                    CheckSupportAxis(
                        meta,
                        "Implementation Owner",
                        cyxwiz::PipelineRuntimeImplementationOwnerName(
                            support.implementation_owner),
                        true,
                        capability.legacy_type_name);
                }
                CheckSupportAxis(
                    meta,
                    "Fail Mode",
                    cyxwiz::PipelineRuntimeFailModeName(support.fail_mode),
                    true,
                    capability.legacy_type_name);
                CheckSupportAxis(
                    meta,
                    "Pipeline Executor",
                    "supported",
                    true,
                    capability.legacy_type_name);
                CheckSupportAxis(
                    meta,
                    "Support State",
                    "real",
                    true,
                    capability.legacy_type_name);
            }
        }
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
              gui::NodeType::RemoveDuplicateRows)) == "RemoveDuplicateRows",
          "runtime enum lookup for RemoveDuplicateRows should prefer canonical spelling");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(
              gui::NodeType::RemoveDuplicateRows).mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "RemoveDuplicateRows enum support should resolve to legacy executor");
    const auto canonical_dedup_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("RemoveDuplicateRows");
    Check(canonical_dedup_type.has_value() &&
              *canonical_dedup_type == gui::NodeType::RemoveDuplicateRows,
          "canonical RemoveDuplicateRows runtime name should resolve to typed metadata");
    const auto legacy_dedup_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("RemoveDuplicates");
    Check(legacy_dedup_type.has_value() &&
              *legacy_dedup_type == gui::NodeType::RemoveDuplicateRows,
          "legacy RemoveDuplicates runtime name should remain an executable alias");
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::BinningNode)) == "BinningNode",
          "runtime enum lookup for BinningNode should prefer canonical spelling");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::BinningNode).mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "BinningNode enum support should resolve to legacy executor");
    const auto canonical_binning_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("BinningNode");
    Check(canonical_binning_type.has_value() &&
              *canonical_binning_type == gui::NodeType::BinningNode,
          "canonical BinningNode runtime name should resolve to typed metadata");
    const auto legacy_binning_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("Binning");
    Check(legacy_binning_type.has_value() &&
              *legacy_binning_type == gui::NodeType::BinningNode,
          "legacy Binning runtime name should remain an executable alias");
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::PolynomialFeaturesNode)) == "PolynomialFeaturesNode",
          "runtime enum lookup for PolynomialFeaturesNode should prefer canonical spelling");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(
              gui::NodeType::PolynomialFeaturesNode).mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "PolynomialFeaturesNode enum support should resolve to legacy executor");
    const auto canonical_polynomial_features_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("PolynomialFeaturesNode");
    Check(canonical_polynomial_features_type.has_value() &&
              *canonical_polynomial_features_type ==
                  gui::NodeType::PolynomialFeaturesNode,
          "canonical PolynomialFeaturesNode runtime name should resolve to typed metadata");
    const auto legacy_polynomial_features_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("PolynomialFeatures");
    Check(legacy_polynomial_features_type.has_value() &&
              *legacy_polynomial_features_type ==
                  gui::NodeType::PolynomialFeaturesNode,
          "legacy PolynomialFeatures runtime name should remain an executable alias");
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::TimeSeriesLag)) == "TimeSeriesLag",
          "runtime enum lookup for TimeSeriesLag should prefer canonical spelling");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::TimeSeriesLag).mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "TimeSeriesLag enum support should resolve to legacy executor");
    const auto canonical_ts_lag_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("TimeSeriesLag");
    Check(canonical_ts_lag_type.has_value() &&
              *canonical_ts_lag_type == gui::NodeType::TimeSeriesLag,
          "canonical TimeSeriesLag runtime name should resolve to typed metadata");
    const auto legacy_ts_lag_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("TSLag");
    Check(legacy_ts_lag_type.has_value() &&
              *legacy_ts_lag_type == gui::NodeType::TimeSeriesLag,
          "legacy TSLag runtime name should remain an executable alias");
    const auto legacy_ts_window_support =
        cyxwiz::ResolvePipelineRuntimeSupport("TSWindow");
    Check(legacy_ts_window_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "legacy TSWindow runtime name should remain legacy-executor routed");
    Check(legacy_ts_window_support.node_type.has_value() &&
              *legacy_ts_window_support.node_type ==
                  gui::NodeType::TimeSeriesWindow,
          "legacy TSWindow runtime name should resolve to TimeSeriesWindow metadata");
    const auto legacy_ts_window_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("TSWindow");
    Check(legacy_ts_window_type.has_value() &&
              *legacy_ts_window_type == gui::NodeType::TimeSeriesWindow,
          "legacy TSWindow runtime name should remain an executable alias");
    const auto legacy_ts_features_support =
        cyxwiz::ResolvePipelineRuntimeSupport("TSFeatures");
    Check(legacy_ts_features_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "legacy TSFeatures runtime name should remain legacy-executor routed");
    Check(legacy_ts_features_support.node_type.has_value() &&
              *legacy_ts_features_support.node_type ==
                  gui::NodeType::TimeSeriesFeatures,
          "legacy TSFeatures runtime name should resolve to TimeSeriesFeatures metadata");
    const auto legacy_ts_features_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("TSFeatures");
    Check(legacy_ts_features_type.has_value() &&
              *legacy_ts_features_type == gui::NodeType::TimeSeriesFeatures,
          "legacy TSFeatures runtime name should remain an executable alias");
    const auto legacy_ts_diff_support =
        cyxwiz::ResolvePipelineRuntimeSupport("TSDiff");
    Check(legacy_ts_diff_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "legacy TSDiff runtime name should remain legacy-executor routed");
    Check(legacy_ts_diff_support.node_type.has_value() &&
              *legacy_ts_diff_support.node_type == gui::NodeType::Differencing,
          "legacy TSDiff runtime name should resolve to Differencing metadata");
    const auto legacy_ts_diff_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("TSDiff");
    Check(legacy_ts_diff_type.has_value() &&
              *legacy_ts_diff_type == gui::NodeType::Differencing,
          "legacy TSDiff runtime name should remain an executable alias");
    const auto legacy_text_vectorize_support =
        cyxwiz::ResolvePipelineRuntimeSupport("TextVectorize");
    Check(legacy_text_vectorize_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "legacy TextVectorize runtime name should remain legacy-executor routed");
    Check(legacy_text_vectorize_support.node_type.has_value() &&
              *legacy_text_vectorize_support.node_type ==
                  gui::NodeType::CountVectorizer,
          "legacy TextVectorize runtime name should resolve to CountVectorizer metadata");
    const auto legacy_text_vectorize_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("TextVectorize");
    Check(legacy_text_vectorize_type.has_value() &&
              *legacy_text_vectorize_type == gui::NodeType::CountVectorizer,
          "legacy TextVectorize runtime name should remain an executable alias");
    const auto legacy_text_tokenize_support =
        cyxwiz::ResolvePipelineRuntimeSupport("TextTokenize");
    Check(legacy_text_tokenize_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "legacy TextTokenize runtime name should remain legacy-executor routed");
    Check(legacy_text_tokenize_support.node_type.has_value() &&
              *legacy_text_tokenize_support.node_type ==
                  gui::NodeType::TextTokenizer,
          "legacy TextTokenize runtime name should resolve to TextTokenizer metadata");
    const auto legacy_text_tokenize_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("TextTokenize");
    Check(legacy_text_tokenize_type.has_value() &&
              *legacy_text_tokenize_type == gui::NodeType::TextTokenizer,
          "legacy TextTokenize runtime name should remain an executable alias");
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::TextCleanNode)) == "TextCleanNode",
          "runtime enum lookup for TextCleanNode should prefer canonical spelling");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::TextCleanNode).mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "TextCleanNode enum support should resolve to legacy executor");
    const auto canonical_text_clean_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("TextCleanNode");
    Check(canonical_text_clean_type.has_value() &&
              *canonical_text_clean_type == gui::NodeType::TextCleanNode,
          "canonical TextCleanNode runtime name should resolve to typed metadata");
    const auto legacy_text_clean_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("TextClean");
    Check(legacy_text_clean_type.has_value() &&
              *legacy_text_clean_type == gui::NodeType::TextCleanNode,
          "legacy TextClean runtime name should remain an executable alias");
    const auto legacy_save_dataset_support =
        cyxwiz::ResolvePipelineRuntimeSupport("SaveDataset");
    Check(legacy_save_dataset_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "legacy SaveDataset runtime name should remain legacy-executor routed");
    Check(legacy_save_dataset_support.node_type.has_value() &&
              *legacy_save_dataset_support.node_type == gui::NodeType::DataOutput,
          "legacy SaveDataset runtime name should resolve to DataOutput metadata");
    const auto legacy_save_dataset_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("SaveDataset");
    Check(legacy_save_dataset_type.has_value() &&
              *legacy_save_dataset_type == gui::NodeType::DataOutput,
          "legacy SaveDataset runtime name should remain an executable alias");
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::DeployToNodeEditorNode)) == "DeployToNodeEditorNode",
          "runtime enum lookup for DeployToNodeEditorNode should prefer canonical spelling");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(
              gui::NodeType::DeployToNodeEditorNode).mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "DeployToNodeEditorNode enum support should resolve to legacy executor");
    const auto canonical_deploy_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("DeployToNodeEditorNode");
    Check(canonical_deploy_type.has_value() &&
              *canonical_deploy_type == gui::NodeType::DeployToNodeEditorNode,
          "canonical DeployToNodeEditorNode runtime name should resolve to typed metadata");
    const auto legacy_deploy_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("DeployToNodeEditor");
    Check(legacy_deploy_type.has_value() &&
              *legacy_deploy_type == gui::NodeType::DeployToNodeEditorNode,
          "legacy DeployToNodeEditor runtime name should remain an executable alias");
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
               gui::NodeType::PCANode)) == "PCANode",
          "operator runtime enum lookup for PCANode should prefer canonical spelling");
    Check(cyxwiz::ResolvePipelineRuntimeSupport("PCA").mode ==
              cyxwiz::PipelineRuntimeSupportMode::OperatorBacked,
          "legacy PCA runtime name should resolve to operator-backed support");
    const auto legacy_pca_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("PCA");
    Check(legacy_pca_type.has_value() &&
              *legacy_pca_type == gui::NodeType::PCANode,
          "legacy PCA runtime name should remain an executable alias");
    const auto legacy_parquet_input_support =
        cyxwiz::ResolvePipelineRuntimeSupport("ParquetInput");
    Check(legacy_parquet_input_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "legacy ParquetInput runtime name should resolve to legacy executor");
    Check(legacy_parquet_input_support.node_type.has_value() &&
              *legacy_parquet_input_support.node_type == gui::NodeType::DataInput,
          "legacy ParquetInput runtime name should resolve to DataInput metadata");
    const auto legacy_parquet_input_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("ParquetInput");
    Check(legacy_parquet_input_type.has_value() &&
              *legacy_parquet_input_type == gui::NodeType::DataInput,
          "legacy ParquetInput runtime name should remain an executable alias");
    const auto cell_extractor_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::CellExtractor);
    Check(cell_extractor_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "CellExtractor enum support should resolve to legacy executor");
    const auto cell_extractor_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("CellExtractor");
    Check(cell_extractor_type.has_value() &&
              *cell_extractor_type == gui::NodeType::CellExtractor,
          "CellExtractor runtime name should resolve typed metadata");
    const auto cell_updater_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::CellUpdater);
    Check(cell_updater_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "CellUpdater enum support should resolve to legacy executor");
    const auto cell_updater_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("CellUpdater");
    Check(cell_updater_type.has_value() &&
              *cell_updater_type == gui::NodeType::CellUpdater,
          "CellUpdater runtime name should resolve typed metadata");
    const auto row_appender_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::RowAppender);
    Check(row_appender_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "RowAppender enum support should resolve to legacy executor");
    const auto row_appender_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("RowAppender");
    Check(row_appender_type.has_value() &&
              *row_appender_type == gui::NodeType::RowAppender,
          "RowAppender runtime name should resolve typed metadata");
    const auto column_appender_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::ColumnAppender);
    Check(column_appender_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "ColumnAppender enum support should resolve to legacy executor");
    const auto column_appender_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("ColumnAppender");
    Check(column_appender_type.has_value() &&
              *column_appender_type == gui::NodeType::ColumnAppender,
          "ColumnAppender runtime name should resolve typed metadata");
    const auto unpivot_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::Unpivot);
    Check(unpivot_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "Unpivot enum support should resolve to legacy executor");
    const auto unpivot_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("Unpivot");
    Check(unpivot_type.has_value() && *unpivot_type == gui::NodeType::Unpivot,
          "Unpivot runtime name should resolve typed metadata");
    const auto export_json_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::ExportJSON);
    Check(export_json_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "ExportJSON enum support should resolve to legacy executor");
    const auto export_json_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("ExportJSON");
    Check(export_json_type.has_value() &&
              *export_json_type == gui::NodeType::ExportJSON,
          "ExportJSON runtime name should resolve typed metadata");
    const auto export_parquet_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::ExportParquet);
    Check(export_parquet_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "ExportParquet enum support should resolve to legacy executor");
    const auto export_parquet_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("ExportParquet");
    Check(export_parquet_type.has_value() &&
              *export_parquet_type == gui::NodeType::ExportParquet,
          "ExportParquet runtime name should resolve typed metadata");
    const auto rule_engine_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::RuleEngine);
    Check(rule_engine_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "RuleEngine enum support should resolve to legacy executor");
    const auto rule_engine_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("RuleEngine");
    Check(rule_engine_type.has_value() &&
              *rule_engine_type == gui::NodeType::RuleEngine,
          "RuleEngine runtime name should resolve typed metadata");
    const auto unit_converter_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::UnitConverter);
    Check(unit_converter_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "UnitConverter enum support should resolve to legacy executor");
    const auto unit_converter_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("UnitConverter");
    Check(unit_converter_type.has_value() &&
              *unit_converter_type == gui::NodeType::UnitConverter,
          "UnitConverter runtime name should resolve typed metadata");
    const auto calculator_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::CalculatorNode);
    Check(calculator_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "CalculatorNode enum support should resolve to legacy executor");
    const auto calculator_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("CalculatorNode");
    Check(calculator_type.has_value() &&
              *calculator_type == gui::NodeType::CalculatorNode,
          "CalculatorNode runtime name should resolve typed metadata");
    const auto json_path_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::JSONPathExtractor);
    Check(json_path_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "JSONPathExtractor enum support should resolve to legacy executor");
    const auto json_path_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("JSONPathExtractor");
    Check(json_path_type.has_value() &&
              *json_path_type == gui::NodeType::JSONPathExtractor,
          "JSONPathExtractor runtime name should resolve typed metadata");
    const auto regex_tester_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::RegexTester);
    Check(regex_tester_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "RegexTester enum support should resolve to legacy executor");
    const auto regex_tester_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("RegexTester");
    Check(regex_tester_type.has_value() &&
              *regex_tester_type == gui::NodeType::RegexTester,
          "RegexTester runtime name should resolve typed metadata");
    const auto data_profiler_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::DataProfiler);
    Check(data_profiler_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "DataProfiler enum support should resolve to legacy executor");
    const auto data_profiler_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("DataProfiler");
    Check(data_profiler_type.has_value() &&
              *data_profiler_type == gui::NodeType::DataProfiler,
          "DataProfiler runtime name should resolve typed metadata");
    const auto regression_metrics_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::RegressionMetricsNode);
    Check(regression_metrics_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "RegressionMetricsNode enum support should resolve to legacy executor");
    const auto regression_metrics_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("RegressionMetricsNode");
    Check(regression_metrics_type.has_value() &&
              *regression_metrics_type == gui::NodeType::RegressionMetricsNode,
          "RegressionMetricsNode runtime name should resolve typed metadata");
    const auto* regression_metrics_meta =
        metadata.GetMetadata(gui::NodeType::RegressionMetricsNode);
    Check(regression_metrics_meta != nullptr,
          "RegressionMetricsNode metadata should exist");
    Check(regression_metrics_meta->status ==
              cyxwiz::NodeImplementationStatus::Implemented,
          "RegressionMetricsNode metadata should be implemented");
    const auto classification_metrics_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::ClassificationMetricsNode);
    Check(classification_metrics_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "ClassificationMetricsNode enum support should resolve to legacy executor");
    const auto classification_metrics_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("ClassificationMetricsNode");
    Check(classification_metrics_type.has_value() &&
              *classification_metrics_type == gui::NodeType::ClassificationMetricsNode,
          "ClassificationMetricsNode runtime name should resolve typed metadata");
    const auto* classification_metrics_meta =
        metadata.GetMetadata(gui::NodeType::ClassificationMetricsNode);
    Check(classification_metrics_meta != nullptr,
          "ClassificationMetricsNode metadata should exist");
    Check(classification_metrics_meta->status ==
              cyxwiz::NodeImplementationStatus::Implemented,
          "ClassificationMetricsNode metadata should be implemented");
    const auto confusion_matrix_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::ConfusionMatrixNode);
    Check(confusion_matrix_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "ConfusionMatrixNode enum support should resolve to legacy executor");
    const auto confusion_matrix_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("ConfusionMatrixNode");
    Check(confusion_matrix_type.has_value() &&
              *confusion_matrix_type == gui::NodeType::ConfusionMatrixNode,
          "ConfusionMatrixNode runtime name should resolve typed metadata");
    const auto* confusion_matrix_meta =
        metadata.GetMetadata(gui::NodeType::ConfusionMatrixNode);
    Check(confusion_matrix_meta != nullptr,
          "ConfusionMatrixNode metadata should exist");
    Check(confusion_matrix_meta->status ==
              cyxwiz::NodeImplementationStatus::Implemented,
          "ConfusionMatrixNode metadata should be implemented");
    const auto roc_curve_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::ROCCurveNode);
    Check(roc_curve_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "ROCCurveNode enum support should resolve to legacy executor");
    const auto roc_curve_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("ROCCurveNode");
    Check(roc_curve_type.has_value() &&
              *roc_curve_type == gui::NodeType::ROCCurveNode,
          "ROCCurveNode runtime name should resolve typed metadata");
    const auto* roc_curve_meta =
        metadata.GetMetadata(gui::NodeType::ROCCurveNode);
    Check(roc_curve_meta != nullptr,
          "ROCCurveNode metadata should exist");
    Check(roc_curve_meta->status ==
              cyxwiz::NodeImplementationStatus::Implemented,
          "ROCCurveNode metadata should be implemented");
    const auto pr_curve_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::PRCurveNode);
    Check(pr_curve_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "PRCurveNode enum support should resolve to legacy executor");
    const auto pr_curve_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("PRCurveNode");
    Check(pr_curve_type.has_value() &&
              *pr_curve_type == gui::NodeType::PRCurveNode,
          "PRCurveNode runtime name should resolve typed metadata");
    const auto* pr_curve_meta =
        metadata.GetMetadata(gui::NodeType::PRCurveNode);
    Check(pr_curve_meta != nullptr,
          "PRCurveNode metadata should exist");
    Check(pr_curve_meta->status ==
              cyxwiz::NodeImplementationStatus::Implemented,
          "PRCurveNode metadata should be implemented");
    const auto data_validator_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::DataValidator);
    Check(data_validator_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "DataValidator enum support should resolve to legacy executor");
    const auto data_validator_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("DataValidator");
    Check(data_validator_type.has_value() &&
              *data_validator_type == gui::NodeType::DataValidator,
          "DataValidator runtime name should resolve typed metadata");
    const auto* data_validator_meta =
        metadata.GetMetadata(gui::NodeType::DataValidator);
    Check(data_validator_meta != nullptr,
          "DataValidator metadata should exist");
    Check(data_validator_meta->status ==
              cyxwiz::NodeImplementationStatus::Implemented,
          "DataValidator metadata should be implemented");
    const auto sample_rows_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::SampleRows);
    Check(sample_rows_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "SampleRows enum support should resolve to legacy executor");
    const auto sample_rows_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("SampleRows");
    Check(sample_rows_type.has_value() &&
              *sample_rows_type == gui::NodeType::SampleRows,
          "SampleRows runtime name should resolve typed metadata");
    const auto value_counts_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::ValueCounts);
    Check(value_counts_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "ValueCounts enum support should resolve to legacy executor");
    const auto value_counts_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("ValueCounts");
    Check(value_counts_type.has_value() &&
              *value_counts_type == gui::NodeType::ValueCounts,
          "ValueCounts runtime name should resolve typed metadata");
    const auto describe_stats_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::DescribeStats);
    Check(describe_stats_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "DescribeStats enum support should resolve to legacy executor");
    const auto describe_stats_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("DescribeStats");
    Check(describe_stats_type.has_value() &&
              *describe_stats_type == gui::NodeType::DescribeStats,
          "DescribeStats runtime name should resolve typed metadata");
    const auto correlation_matrix_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::CorrelationMatrix);
    Check(correlation_matrix_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "CorrelationMatrix enum support should resolve to legacy executor");
    const auto correlation_matrix_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("CorrelationMatrix");
    Check(correlation_matrix_type.has_value() &&
              *correlation_matrix_type == gui::NodeType::CorrelationMatrix,
          "CorrelationMatrix runtime name should resolve typed metadata");
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::CSVFile)) == "FileInput",
          "legacy runtime enum lookup for CSVFile is stable");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::CSVFile).mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "CSVFile enum support should resolve to legacy executor");
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::ExcelFile)) == "ExcelInput",
          "fail-closed runtime enum lookup for ExcelFile is stable");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::ExcelFile).mode ==
              cyxwiz::PipelineRuntimeSupportMode::FailClosed,
          "ExcelFile enum support should resolve to fail-closed");
    const auto* excel_file_meta = metadata.GetMetadata(gui::NodeType::ExcelFile);
    if (excel_file_meta != nullptr) {
        Check(excel_file_meta->status == cyxwiz::NodeImplementationStatus::Template,
              "ExcelFile metadata should be blocked until Excel loading is real");
    }
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::JSONFile)) == "JSONFile",
          "fail-closed runtime enum lookup for JSONFile is stable");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::JSONFile).mode ==
              cyxwiz::PipelineRuntimeSupportMode::FailClosed,
          "JSONFile enum support should resolve to fail-closed");
    const auto* json_file_meta = metadata.GetMetadata(gui::NodeType::JSONFile);
    if (json_file_meta != nullptr) {
        Check(json_file_meta->status == cyxwiz::NodeImplementationStatus::Template,
              "JSONFile metadata should be blocked until JSON loading is real");
    }
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::SQLQuery)) == "SQLQuery",
          "fail-closed runtime enum lookup for SQLQuery is stable");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::SQLQuery).mode ==
              cyxwiz::PipelineRuntimeSupportMode::FailClosed,
          "SQLQuery enum support should resolve to fail-closed");
    const auto* sql_query_meta = metadata.GetMetadata(gui::NodeType::SQLQuery);
    if (sql_query_meta != nullptr) {
        Check(sql_query_meta->status == cyxwiz::NodeImplementationStatus::Template,
              "SQLQuery metadata should be blocked until SQL loading is real");
    }
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::HDF5Dataset)) == "HDF5Dataset",
          "fail-closed runtime enum lookup for HDF5Dataset is stable");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::HDF5Dataset).mode ==
              cyxwiz::PipelineRuntimeSupportMode::FailClosed,
          "HDF5Dataset enum support should resolve to fail-closed");
    const auto* hdf5_meta = metadata.GetMetadata(gui::NodeType::HDF5Dataset);
    if (hdf5_meta != nullptr) {
        Check(hdf5_meta->status == cyxwiz::NodeImplementationStatus::Template,
              "HDF5Dataset metadata should be blocked until HDF5 loading is real");
    }
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::RESTAPISource)) == "RESTAPISource",
          "fail-closed runtime enum lookup for RESTAPISource is stable");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::RESTAPISource).mode ==
              cyxwiz::PipelineRuntimeSupportMode::FailClosed,
          "RESTAPISource enum support should resolve to fail-closed");
    const auto* rest_api_meta = metadata.GetMetadata(gui::NodeType::RESTAPISource);
    if (rest_api_meta != nullptr) {
        Check(rest_api_meta->status == cyxwiz::NodeImplementationStatus::Template,
              "RESTAPISource metadata should be blocked until REST loading is real");
    }
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::ExportSQL)) == "ExportSQL",
          "fail-closed runtime enum lookup for ExportSQL is stable");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::ExportSQL).mode ==
              cyxwiz::PipelineRuntimeSupportMode::FailClosed,
          "ExportSQL enum support should resolve to fail-closed");
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
    const gui::NodeType additional_fail_closed_enum_cases[] = {
        gui::NodeType::UMAPNode,
        gui::NodeType::SVMRegressor,
        gui::NodeType::LearningCurvesNode,
        gui::NodeType::FeatureImportanceNode,
        gui::NodeType::CrossValidationNode,
        gui::NodeType::VisualizeData,
        gui::NodeType::Normalize,
        gui::NodeType::OneHotEncode,
        gui::NodeType::AudioInput,
        gui::NodeType::Spectrogram,
        gui::NodeType::MelSpectrogram,
        gui::NodeType::MFCC,
        gui::NodeType::ReLU,
        gui::NodeType::Sigmoid,
        gui::NodeType::Tanh,
        gui::NodeType::Softmax,
        gui::NodeType::GELU,
        gui::NodeType::LeakyReLU,
        gui::NodeType::MSELoss,
        gui::NodeType::CrossEntropyLoss,
        gui::NodeType::BCELoss,
        gui::NodeType::BCEWithLogits,
        gui::NodeType::L1Loss,
        gui::NodeType::SmoothL1Loss,
        gui::NodeType::HuberLoss,
        gui::NodeType::NLLLoss,
        gui::NodeType::SoftDiceLoss,
        gui::NodeType::TverskyLoss,
        gui::NodeType::JaccardLoss,
        gui::NodeType::SGD,
        gui::NodeType::Adam,
        gui::NodeType::AdamW,
        gui::NodeType::Add,
        gui::NodeType::Multiply,
        gui::NodeType::Average,
        gui::NodeType::Constant,
        gui::NodeType::Lambda,
        gui::NodeType::Reshape,
        gui::NodeType::View,
        gui::NodeType::Permute,
        gui::NodeType::Split,
        gui::NodeType::Squeeze,
        gui::NodeType::Unsqueeze,
        gui::NodeType::TensorAbs,
        gui::NodeType::TensorClip,
        gui::NodeType::TensorExp,
        gui::NodeType::TensorLog,
        gui::NodeType::TensorPow,
        gui::NodeType::TensorSign,
        gui::NodeType::TensorSqrt,
        gui::NodeType::TensorMean,
        gui::NodeType::TensorSum,
        gui::NodeType::TensorMax,
        gui::NodeType::TensorMin,
        gui::NodeType::TensorProd,
        gui::NodeType::TensorStd,
        gui::NodeType::TensorVar,
        gui::NodeType::TensorDot,
        gui::NodeType::TensorBatchMatMul,
        gui::NodeType::TensorBroadcastTo,
        gui::NodeType::TensorExpand,
        gui::NodeType::TensorIndexSelect,
        gui::NodeType::TensorLogicalMask,
        gui::NodeType::SignalSlider,
        gui::NodeType::SineWave,
        gui::NodeType::StepSignal,
        gui::NodeType::RampSignal,
        gui::NodeType::SignalScope,
        gui::NodeType::QualityAnalyzer,
        gui::NodeType::TableSplitter,
        gui::NodeType::ExportExcel,
        gui::NodeType::IFFTNode,
        gui::NodeType::WaveletTransform,
        gui::NodeType::WordEmbeddings,
        gui::NodeType::NamedEntityRecognizer,
        gui::NodeType::ImagePreprocessor,
        gui::NodeType::ImageFolderDataset,
        gui::NodeType::AugmentationPreset,
        gui::NodeType::TSVFile,
        gui::NodeType::TXTFile,
        gui::NodeType::ARFFFile,
        gui::NodeType::FeatherFile,
        gui::NodeType::ArrowIPCFile,
        gui::NodeType::NumPyFile,
        gui::NodeType::ImageCSVDataset,
        gui::NodeType::StreamingDataset,
        gui::NodeType::FashionMNISTDataset,
        gui::NodeType::CIFAR100Dataset,
        gui::NodeType::AudioFolderDataset,
        gui::NodeType::TimeSeriesCSV,
        gui::NodeType::TextCorpusDataset,
    };
    for (const auto type : additional_fail_closed_enum_cases) {
        Check(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(type) != nullptr,
              "typed fail-closed runtime enum should resolve legacy name: " +
                  TypeId(type));
        Check(cyxwiz::ResolvePipelineRuntimeSupport(type).mode ==
                  cyxwiz::PipelineRuntimeSupportMode::FailClosed,
              "typed fail-closed runtime enum should resolve fail-closed support: " +
                  TypeId(type));
    }

    const gui::NodeType blocked_metadata_cases[] = {
        gui::NodeType::LearningCurvesNode,
        gui::NodeType::FeatureImportanceNode,
        gui::NodeType::CrossValidationNode,
        gui::NodeType::VisualizeData,
        gui::NodeType::ExportExcel,
        gui::NodeType::TableSplitter,
        gui::NodeType::IFFTNode,
        gui::NodeType::WaveletTransform,
        gui::NodeType::WordEmbeddings,
        gui::NodeType::NamedEntityRecognizer,
        gui::NodeType::TSVFile,
        gui::NodeType::TXTFile,
        gui::NodeType::ARFFFile,
        gui::NodeType::FeatherFile,
        gui::NodeType::ArrowIPCFile,
        gui::NodeType::NumPyFile,
        gui::NodeType::ImageCSVDataset,
        gui::NodeType::StreamingDataset,
        gui::NodeType::FashionMNISTDataset,
        gui::NodeType::CIFAR100Dataset,
        gui::NodeType::AudioFolderDataset,
        gui::NodeType::TimeSeriesCSV,
        gui::NodeType::TextCorpusDataset,
    };
    for (const auto type : blocked_metadata_cases) {
        const auto* meta = metadata.GetMetadata(type);
        if (meta == nullptr) {
            continue;
        }
        Check(meta->status == cyxwiz::NodeImplementationStatus::Template,
              "blocked fail-closed metadata should remain template: " +
                  TypeId(type));
    }

    const gui::NodeType training_contract_fail_closed_cases[] = {
        gui::NodeType::Normalize,
        gui::NodeType::OneHotEncode,
        gui::NodeType::AudioInput,
        gui::NodeType::Spectrogram,
        gui::NodeType::MelSpectrogram,
        gui::NodeType::MFCC,
        gui::NodeType::ReLU,
        gui::NodeType::Sigmoid,
        gui::NodeType::Tanh,
        gui::NodeType::Softmax,
        gui::NodeType::GELU,
        gui::NodeType::LeakyReLU,
        gui::NodeType::MSELoss,
        gui::NodeType::CrossEntropyLoss,
        gui::NodeType::BCELoss,
        gui::NodeType::BCEWithLogits,
        gui::NodeType::L1Loss,
        gui::NodeType::SmoothL1Loss,
        gui::NodeType::HuberLoss,
        gui::NodeType::NLLLoss,
        gui::NodeType::SoftDiceLoss,
        gui::NodeType::TverskyLoss,
        gui::NodeType::JaccardLoss,
        gui::NodeType::SGD,
        gui::NodeType::Adam,
        gui::NodeType::AdamW,
        gui::NodeType::Add,
        gui::NodeType::Multiply,
        gui::NodeType::Average,
        gui::NodeType::Constant,
        gui::NodeType::Lambda,
        gui::NodeType::Reshape,
        gui::NodeType::View,
        gui::NodeType::Permute,
        gui::NodeType::Split,
        gui::NodeType::Squeeze,
        gui::NodeType::Unsqueeze,
        gui::NodeType::TensorAbs,
        gui::NodeType::TensorClip,
        gui::NodeType::TensorExp,
        gui::NodeType::TensorLog,
        gui::NodeType::TensorPow,
        gui::NodeType::TensorSign,
        gui::NodeType::TensorSqrt,
        gui::NodeType::TensorMean,
        gui::NodeType::TensorSum,
        gui::NodeType::TensorMax,
        gui::NodeType::TensorMin,
        gui::NodeType::TensorProd,
        gui::NodeType::TensorStd,
        gui::NodeType::TensorVar,
        gui::NodeType::TensorDot,
        gui::NodeType::TensorBatchMatMul,
        gui::NodeType::TensorBroadcastTo,
        gui::NodeType::TensorExpand,
        gui::NodeType::TensorIndexSelect,
        gui::NodeType::TensorLogicalMask,
        gui::NodeType::SignalSlider,
        gui::NodeType::SineWave,
        gui::NodeType::StepSignal,
        gui::NodeType::RampSignal,
        gui::NodeType::SignalScope,
    };
    for (const auto type : training_contract_fail_closed_cases) {
        const auto* meta = metadata.GetMetadata(type);
        if (meta == nullptr) {
            continue;
        }
        Check(meta->status == cyxwiz::NodeImplementationStatus::Implemented ||
                  meta->status == cyxwiz::NodeImplementationStatus::Template,
              "training contract metadata should remain registered while PipelineExecutor fail-closes: " +
                  TypeId(type));
        Check(cyxwiz::ResolvePipelineRuntimeSupport(type).mode ==
                  cyxwiz::PipelineRuntimeSupportMode::FailClosed,
              "training contract node should still fail closed in PipelineExecutor: " +
                  TypeId(type));
    }

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
    Check(std::string(cyxwiz::PipelineTrainingBackendSupportModeName(
              cyxwiz::PipelineTrainingBackendSupportMode::
                  UnsupportedTrainingWorkflow)) ==
              "unsupported_training_workflow",
          "training backend support mode name for unsupported workflow is stable");

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

    // Track 22 Phase 5 validation checklist:
    // - every new executable Data Studio node must put static required
    //   parameters, enum values, integer bounds, and float bounds in central
    //   runtime capability tables;
    // - those static validation facts must resolve to exactly one executable
    //   runtime owner before SQL/operator execution can run;
    // - routing tests must include at least one representative bad-schema case
    //   for each newly supported executable node.
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
        Check(IsExecutableRuntimeSupportMode(support.mode),
              std::string("required-parameter runtime name is not executable: ") +
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
        Check(IsExecutableRuntimeSupportMode(support.mode),
              std::string("allowed-parameter runtime name is not executable: ") +
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
        Check(IsExecutableRuntimeSupportMode(support.mode),
              std::string("integer-parameter runtime name is not executable: ") +
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

    for (const auto& capability :
         cyxwiz::GetPipelineFloatParameterRuntimeCapabilities()) {
        const auto float_parameters =
            cyxwiz::ResolvePipelineFloatParameters(capability.legacy_type_name);
        Check(!float_parameters.empty(),
              std::string("float-parameter runtime name does not resolve: ") +
                  capability.legacy_type_name);
        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(
            capability.legacy_type_name);
        Check(support.mode != cyxwiz::PipelineRuntimeSupportMode::Unknown,
              std::string("float-parameter runtime name has unknown support: ") +
                  capability.legacy_type_name);
        Check(IsExecutableRuntimeSupportMode(support.mode),
              std::string("float-parameter runtime name is not executable: ") +
                  capability.legacy_type_name);
        auto supported_axis = std::find_if(
            support.float_parameters.begin(),
            support.float_parameters.end(),
            [&capability](
                const cyxwiz::PipelineFloatParameterRuntimeCapability&
                    runtime_capability) {
                return std::string(runtime_capability.parameter_name) ==
                           capability.parameter_name &&
                       runtime_capability.minimum == capability.minimum &&
                       runtime_capability.maximum == capability.maximum &&
                       runtime_capability.minimum_inclusive ==
                           capability.minimum_inclusive &&
                       runtime_capability.maximum_inclusive ==
                           capability.maximum_inclusive;
            });
        Check(supported_axis != support.float_parameters.end(),
              std::string("runtime support should carry float-parameter axis: ") +
                  capability.legacy_type_name + "." + capability.parameter_name);
        Check(capability.parameter_name != nullptr &&
                  std::string(capability.parameter_name).size() > 1,
              std::string("float parameter name is too weak: ") +
                  capability.legacy_type_name);
        const bool range_has_bound =
            capability.minimum.has_value() || capability.maximum.has_value();
        const bool range_order_is_valid =
            !capability.minimum.has_value() ||
            !capability.maximum.has_value() ||
            *capability.maximum >= *capability.minimum;
        Check(range_has_bound && range_order_is_valid,
              std::string("float parameter range is invalid: ") +
                  capability.legacy_type_name + "." + capability.parameter_name);
    }

    const std::vector<gui::NodeType> supported_model_nodes = {
        gui::NodeType::Dense,
        gui::NodeType::Dropout,
        gui::NodeType::BatchNorm,
        gui::NodeType::LSTM,
        gui::NodeType::GRU,
        gui::NodeType::Embedding,
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
        Check(cyxwiz::IsPipelineSupportedTrainingBackendNode(type),
              "supported model type should be named in central training support: " +
                  TypeId(type));
        CheckSupportAxis(
            meta,
            "Training Backend",
            cyxwiz::PipelineTrainingBackendSupportModeName(
                cyxwiz::PipelineTrainingBackendSupportMode::Allowed),
            true,
            TypeId(type));
        CheckSupportAxis(meta, "Compile", "supported", true, TypeId(type));
        CheckSupportAxis(meta, "Training", "supported", true, TypeId(type));
        CheckSupportAxis(meta, "Implementation Owner", "training_backend", true, TypeId(type));
        CheckSupportAxis(meta, "Support State", "real", true, TypeId(type));
        if (FindSupportAxis(meta, "Workflow Lane") != nullptr) {
            CheckSupportAxis(meta, "Workflow Lane", "deep_learning", true, TypeId(type));
        }
    }

    Check(std::string(cyxwiz::PipelineTrainingSupportRoleName(
              cyxwiz::PipelineTrainingSupportRole::ModelLayer)) == "model_layer",
          "training role name for model layer is stable");
    Check(std::string(cyxwiz::PipelineTrainingSupportRoleName(
              cyxwiz::PipelineTrainingSupportRole::Activation)) == "activation",
          "training role name for activation is stable");
    Check(std::string(cyxwiz::PipelineTrainingSupportRoleName(
              cyxwiz::PipelineTrainingSupportRole::Loss)) == "loss",
          "training role name for loss is stable");
    Check(std::string(cyxwiz::PipelineTrainingSupportRoleName(
              cyxwiz::PipelineTrainingSupportRole::Optimizer)) == "optimizer",
          "training role name for optimizer is stable");
    Check(std::string(cyxwiz::PipelineTrainingSupportRoleName(
              cyxwiz::PipelineTrainingSupportRole::TrainingControl)) ==
              "training_control",
          "training role name for training control is stable");

    struct TrainingRoleCase {
        gui::NodeType node_type;
        const char* role;
    };
    const std::vector<TrainingRoleCase> training_role_cases = {
        {gui::NodeType::Dense, "model_layer"},
        {gui::NodeType::Dropout, "model_layer"},
        {gui::NodeType::BatchNorm, "model_layer"},
        {gui::NodeType::LSTM, "model_layer"},
        {gui::NodeType::GRU, "model_layer"},
        {gui::NodeType::Embedding, "model_layer"},
        {gui::NodeType::Flatten, "model_layer"},
        {gui::NodeType::TimeDistributed, "model_layer"},
        {gui::NodeType::ReLU, "activation"},
        {gui::NodeType::Sigmoid, "activation"},
        {gui::NodeType::Tanh, "activation"},
        {gui::NodeType::Softmax, "activation"},
        {gui::NodeType::MSELoss, "loss"},
        {gui::NodeType::CrossEntropyLoss, "loss"},
        {gui::NodeType::BCELoss, "loss"},
        {gui::NodeType::BCEWithLogits, "loss"},
        {gui::NodeType::L1Loss, "loss"},
        {gui::NodeType::SmoothL1Loss, "loss"},
        {gui::NodeType::HuberLoss, "loss"},
        {gui::NodeType::NLLLoss, "loss"},
        {gui::NodeType::SoftDiceLoss, "loss"},
        {gui::NodeType::TverskyLoss, "loss"},
        {gui::NodeType::JaccardLoss, "loss"},
        {gui::NodeType::SGD, "optimizer"},
        {gui::NodeType::Adam, "optimizer"},
        {gui::NodeType::AdamW, "optimizer"},
        {gui::NodeType::RMSprop, "optimizer"},
        {gui::NodeType::Adagrad, "optimizer"},
        {gui::NodeType::NAdam, "optimizer"},
    };
    for (const auto& role_case : training_role_cases) {
        const auto* meta = metadata.GetMetadata(role_case.node_type);
        Check(cyxwiz::IsPipelineSupportedTrainingRoleNode(role_case.node_type),
              "training role should be centralized for type " +
                  TypeId(role_case.node_type));
        Check(meta != nullptr,
              "training role metadata should be registered for type " +
                  TypeId(role_case.node_type));
        CheckSupportAxis(
            meta,
            "Training Role",
            role_case.role,
            true,
            TypeId(role_case.node_type));
        if (std::string(role_case.role) == "model_layer") {
            CheckSupportAxis(
                meta,
                "Model Builder",
                "supported",
                true,
                TypeId(role_case.node_type));
        } else if (std::string(role_case.role) == "activation") {
            CheckSupportAxis(
                meta,
                "Activation",
                "supported",
                true,
                TypeId(role_case.node_type));
        } else if (std::string(role_case.role) == "loss") {
            CheckSupportAxis(
                meta,
                "Loss",
                "supported",
                true,
                TypeId(role_case.node_type));
        } else if (std::string(role_case.role) == "optimizer") {
            CheckSupportAxis(
                meta,
                "Optimizer",
                "supported",
                true,
                TypeId(role_case.node_type));
        }
        CheckSupportAxis(
            meta,
            "Implementation Owner",
            "training_backend",
            true,
            TypeId(role_case.node_type));
        CheckSupportAxis(
            meta,
            "Compile",
            "supported",
            true,
            TypeId(role_case.node_type));
        CheckSupportAxis(
            meta,
            "Training",
            "supported",
            true,
            TypeId(role_case.node_type));
        CheckSupportAxisReasonContains(
            meta, "Training Role", "", TypeId(role_case.node_type));
    }

    const auto* linear_regression_meta =
        metadata.GetMetadata(gui::NodeType::LinearRegressionNode);
    Check(linear_regression_meta != nullptr, "LinearRegression metadata should exist");
    CheckSupportAxis(
        linear_regression_meta,
        "Workflow Lane",
        "classic_ml",
        true,
        "LinearRegression");
    CheckSupportAxisReasonContains(
        linear_regression_meta, "Workflow Lane", "classical ML",
        "LinearRegression");

    const auto* polynomial_regression_meta =
        metadata.GetMetadata(gui::NodeType::PolynomialRegressionNode);
    Check(polynomial_regression_meta != nullptr, "PolynomialRegression metadata should exist");
    CheckSupportAxis(
        polynomial_regression_meta,
        "Workflow Lane",
        "classic_ml",
        true,
        "PolynomialRegression");
    CheckSupportAxisReasonContains(
        polynomial_regression_meta, "Workflow Lane", "classical ML",
        "PolynomialRegression");

    const auto* decision_tree_meta =
        metadata.GetMetadata(gui::NodeType::DecisionTreeClassifier);
    Check(decision_tree_meta != nullptr, "DecisionTree metadata should exist");
    Check(decision_tree_meta->status == cyxwiz::NodeImplementationStatus::Implemented,
          "DecisionTree metadata should be implemented");
    Check(decision_tree_meta->badge != "Blocked",
          "DecisionTree metadata should not be blocked");
    Check(HasInputType(decision_tree_meta, "Data", gui::PinType::Dataset),
          "DecisionTree should expose a table input");
    Check(HasOutputType(decision_tree_meta, "Predictions", gui::PinType::Dataset),
          "DecisionTree should expose a table prediction output");
    Check(HasParameter(decision_tree_meta, "target_col") &&
              HasParameter(decision_tree_meta, "feature_cols") &&
              HasParameter(decision_tree_meta, "prediction_col") &&
              HasParameter(decision_tree_meta, "model_path") &&
              HasParameter(decision_tree_meta, "max_depth") &&
              HasParameter(decision_tree_meta, "min_samples_split") &&
              HasParameter(decision_tree_meta, "min_samples_leaf") &&
              HasParameter(decision_tree_meta, "criterion"),
          "DecisionTree should expose classifier properties in metadata");
    Check(HasEnumValue(decision_tree_meta, "criterion", "gini") &&
              HasEnumValue(decision_tree_meta, "criterion", "entropy"),
          "DecisionTree criterion should expose supported split criteria");
    CheckSupportAxis(
        decision_tree_meta,
        "Workflow Lane",
        "classic_ml",
        true,
        "DecisionTreeClassifier");
    CheckSupportAxisReasonContains(
        decision_tree_meta, "Workflow Lane", "classical ML",
        "DecisionTreeClassifier");

    const auto* random_forest_meta =
        metadata.GetMetadata(gui::NodeType::RandomForestClassifier);
    Check(random_forest_meta != nullptr, "RandomForest metadata should exist");
    Check(random_forest_meta->status == cyxwiz::NodeImplementationStatus::Implemented,
          "RandomForest metadata should be implemented");
    Check(random_forest_meta->badge != "Blocked",
          "RandomForest metadata should not be blocked");
    Check(HasInputType(random_forest_meta, "Data", gui::PinType::Dataset),
          "RandomForest should expose a table input");
    Check(HasOutputType(random_forest_meta, "Predictions", gui::PinType::Dataset),
          "RandomForest should expose a table prediction output");
    Check(HasParameter(random_forest_meta, "target_col") &&
              HasParameter(random_forest_meta, "feature_cols") &&
              HasParameter(random_forest_meta, "prediction_col") &&
              HasParameter(random_forest_meta, "model_path") &&
              HasParameter(random_forest_meta, "n_estimators") &&
              HasParameter(random_forest_meta, "max_depth") &&
              HasParameter(random_forest_meta, "min_samples_split") &&
              HasParameter(random_forest_meta, "min_samples_leaf") &&
              HasParameter(random_forest_meta, "criterion") &&
              HasParameter(random_forest_meta, "max_features") &&
              HasParameter(random_forest_meta, "seed"),
          "RandomForest should expose classifier properties in metadata");
    Check(HasEnumValue(random_forest_meta, "criterion", "gini") &&
              HasEnumValue(random_forest_meta, "criterion", "entropy"),
          "RandomForest criterion should expose supported split criteria");
    Check(HasEnumValue(random_forest_meta, "max_features", "sqrt") &&
              HasEnumValue(random_forest_meta, "max_features", "log2") &&
              HasEnumValue(random_forest_meta, "max_features", "all"),
          "RandomForest max_features should expose supported feature modes");
    CheckSupportAxis(
        random_forest_meta,
        "Workflow Lane",
        "classic_ml",
        true,
        "RandomForestClassifier");
    CheckSupportAxisReasonContains(
        random_forest_meta, "Workflow Lane", "classical ML",
        "RandomForestClassifier");

    const auto* gradient_boosting_meta =
        metadata.GetMetadata(gui::NodeType::GradientBoostingClassifier);
    Check(gradient_boosting_meta != nullptr,
          "GradientBoosting metadata should exist");
    Check(gradient_boosting_meta->status ==
              cyxwiz::NodeImplementationStatus::Implemented,
          "GradientBoosting metadata should be implemented");
    Check(gradient_boosting_meta->badge != "Blocked",
          "GradientBoosting metadata should not be blocked");
    Check(HasInputType(gradient_boosting_meta, "Data", gui::PinType::Dataset),
          "GradientBoosting should expose a table input");
    Check(HasOutputType(gradient_boosting_meta, "Predictions",
                        gui::PinType::Dataset),
          "GradientBoosting should expose a table prediction output");
    Check(HasParameter(gradient_boosting_meta, "target_col") &&
              HasParameter(gradient_boosting_meta, "feature_cols") &&
              HasParameter(gradient_boosting_meta, "prediction_col") &&
              HasParameter(gradient_boosting_meta, "model_path") &&
              HasParameter(gradient_boosting_meta, "n_estimators") &&
              HasParameter(gradient_boosting_meta, "learning_rate") &&
              HasParameter(gradient_boosting_meta, "max_depth") &&
              HasParameter(gradient_boosting_meta, "min_samples_split") &&
              HasParameter(gradient_boosting_meta, "min_samples_leaf"),
          "GradientBoosting should expose classifier properties in metadata");
    CheckSupportAxis(
        gradient_boosting_meta,
        "Workflow Lane",
        "classic_ml",
        true,
        "GradientBoostingClassifier");
    CheckSupportAxisReasonContains(
        gradient_boosting_meta, "Workflow Lane", "classical ML",
        "GradientBoostingClassifier");

    const auto* tree_predictor_meta =
        metadata.GetMetadata(gui::NodeType::TreeModelPredictor);
    Check(tree_predictor_meta != nullptr,
          "TreeModelPredictor metadata should exist");
    Check(tree_predictor_meta->status ==
              cyxwiz::NodeImplementationStatus::Implemented,
          "TreeModelPredictor metadata should be implemented");
    Check(tree_predictor_meta->badge != "Blocked",
          "TreeModelPredictor metadata should not be blocked");
    Check(HasInputType(tree_predictor_meta, "Data", gui::PinType::Dataset),
          "TreeModelPredictor should expose a table input");
    Check(HasOutputType(tree_predictor_meta, "Predictions",
                        gui::PinType::Dataset),
          "TreeModelPredictor should expose a table prediction output");
    Check(HasParameter(tree_predictor_meta, "model_path") &&
              HasParameter(tree_predictor_meta, "feature_cols") &&
              HasParameter(tree_predictor_meta, "prediction_col"),
          "TreeModelPredictor should expose inference properties in metadata");
    CheckSupportAxis(
        tree_predictor_meta,
        "Workflow Lane",
        "classic_ml",
        true,
        "TreeModelPredictor");
    CheckSupportAxisReasonContains(
        tree_predictor_meta, "Workflow Lane", "classical ML",
        "TreeModelPredictor");

    for (auto type : {
             gui::NodeType::StandardScaler,
             gui::NodeType::TimeSeriesWindow,
             gui::NodeType::TextTokenizer,
             gui::NodeType::KMeansCluster,
         }) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr,
              "Data Studio analytics lane metadata should exist for type " +
                  TypeId(type));
        CheckSupportAxis(
            meta,
            "Workflow Lane",
            "data_studio_analytics",
            true,
            TypeId(type));
        CheckSupportAxisReasonContains(
            meta, "Workflow Lane", "Data Studio analytics", TypeId(type));
    }

    const auto* mse_loss_meta = metadata.GetMetadata(gui::NodeType::MSELoss);
    Check(mse_loss_meta != nullptr, "MSELoss metadata should exist");
    CheckSupportAxis(mse_loss_meta, "Task Type", "regression", true, "MSELoss");
    CheckSupportAxisReasonContains(
        mse_loss_meta, "Task Type", "numeric targets", "MSELoss");

    const auto* cross_entropy_meta =
        metadata.GetMetadata(gui::NodeType::CrossEntropyLoss);
    Check(cross_entropy_meta != nullptr, "CrossEntropyLoss metadata should exist");
    Check(cross_entropy_meta->name.find("Token CE") != std::string::npos,
          "CrossEntropyLoss metadata should advertise token CE use");
    Check(HasParameter(cross_entropy_meta, "ignore_index"),
          "CrossEntropyLoss metadata should expose ignore_index for padded token labels");
    CheckSupportAxis(
        cross_entropy_meta,
        "Task Type",
        "multiclass_classification",
        true,
        "CrossEntropyLoss");
    CheckSupportAxisReasonContains(
        cross_entropy_meta, "Task Type", "class labels", "CrossEntropyLoss");

    Check(SearchContainsType(metadata, "bce", gui::NodeType::BCELoss),
          "searching bce should find BCE Loss metadata");
    Check(SearchContainsType(metadata, "bce", gui::NodeType::BCEWithLogits),
          "searching bce should find BCE with Logits metadata");
    Check(SearchContainsType(metadata, "binary", gui::NodeType::BCELoss),
          "searching binary should find BCE Loss metadata");
    Check(SearchContainsType(metadata, "binary", gui::NodeType::BCEWithLogits),
          "searching binary should find BCE with Logits metadata");
    Check(SearchContainsType(metadata, "negative log likelihood", gui::NodeType::NLLLoss),
          "searching negative log likelihood should find NLL Loss metadata");
    Check(SearchContainsType(metadata, "soft dice", gui::NodeType::SoftDiceLoss),
          "searching soft dice should find Soft Dice Loss metadata");
    Check(SearchContainsType(metadata, "segmentation", gui::NodeType::SoftDiceLoss),
          "searching segmentation should find Soft Dice Loss metadata");
    Check(SearchContainsType(metadata, "tversky", gui::NodeType::TverskyLoss),
          "searching tversky should find Tversky Loss metadata");
    Check(SearchContainsType(metadata, "segmentation", gui::NodeType::TverskyLoss),
          "searching segmentation should find Tversky Loss metadata");
    Check(SearchContainsType(metadata, "jaccard", gui::NodeType::JaccardLoss),
          "searching jaccard should find Jaccard Loss metadata");
    Check(SearchContainsType(metadata, "iou", gui::NodeType::JaccardLoss),
          "searching iou should find Jaccard Loss metadata");
    Check(SearchContainsType(metadata, "segmentation", gui::NodeType::JaccardLoss),
          "searching segmentation should find Jaccard Loss metadata");
    Check(SearchContainsType(metadata, "accuracy", gui::NodeType::ClassificationMetricsNode),
          "searching accuracy should find Classification Metrics metadata");
    Check(SearchContainsType(metadata, "f1", gui::NodeType::ClassificationMetricsNode),
          "searching f1 should find Classification Metrics metadata");
    Check(SearchContainsType(metadata, "optimization", gui::NodeType::Adam),
          "searching optimization should find training optimizer metadata");
    Check(SearchContainsType(metadata, "optimization", gui::NodeType::CrossEntropyLoss),
          "searching optimization should find training loss metadata");

    const auto* time_distributed_meta =
        metadata.GetMetadata(gui::NodeType::TimeDistributed);
    Check(time_distributed_meta != nullptr,
          "TimeDistributed metadata should exist");
    Check(time_distributed_meta->name.find("Dense") != std::string::npos,
          "TimeDistributed metadata should expose the token dense head label");
    Check(time_distributed_meta->brief_description.find("token-classifier") !=
              std::string::npos,
          "TimeDistributed metadata should explain the token classifier role");
    Check(HasOutputType(time_distributed_meta, "Output", gui::PinType::Tensor),
          "TimeDistributed metadata should expose tensor logits output");
    Check(time_distributed_meta->inputs.size() == 1 &&
              time_distributed_meta->inputs[0].required &&
              time_distributed_meta->parameters.size() == 1 &&
              ParameterMatches(time_distributed_meta, "units", "int", "128") &&
              time_distributed_meta->help_text.find("not a generic wrapper") !=
                  std::string::npos,
          "TimeDistributed should expose only its concrete Dense sequence-head contract");

    const auto* embedding_meta =
        metadata.GetMetadata(gui::NodeType::Embedding);
    Check(embedding_meta != nullptr &&
              embedding_meta->status ==
                  cyxwiz::NodeImplementationStatus::Implemented &&
              embedding_meta->inputs.size() == 1 &&
              embedding_meta->inputs[0].name == "Indices" &&
              embedding_meta->inputs[0].type == gui::PinType::Tensor &&
              embedding_meta->outputs.size() == 1 &&
              embedding_meta->outputs[0].name == "Embeddings",
          "Embedding metadata should expose the executable rank-aware Tensor pins");
    Check(embedding_meta->parameters.size() == 8 &&
              ParameterMatches(embedding_meta, "num_embeddings", "int", "10000") &&
              ParameterMatches(embedding_meta, "embedding_dim", "int", "256") &&
              ParameterMatches(embedding_meta, "padding_idx", "int", "-1") &&
              ParameterMatches(embedding_meta, "max_norm", "float", "0") &&
              ParameterMatches(embedding_meta, "freeze", "bool", "false") &&
              ParameterMatches(embedding_meta, "weights_file", "file", "") &&
              ParameterMatches(embedding_meta, "init_mode", "enum", "normal") &&
              ParameterMatches(embedding_meta, "output_weights_file", "file", "") &&
              !HasParameter(embedding_meta, "embedding_weights_file"),
          "Embedding metadata should centralize runtime and dialog fields without persisting its legacy alias");

    const std::map<std::string, std::string> valid_embedding = {
        {"num_embeddings", "100"},
        {"embedding_dim", "16"},
        {"padding_idx", "0"},
        {"max_norm", "0"},
    };
    Check(!cyxwiz::ResolveInvalidSequenceProjectionConfigurationReason(
               gui::NodeType::Embedding, valid_embedding),
          "valid Embedding configuration should pass the shared policy");
    auto invalid_padding = valid_embedding;
    invalid_padding["padding_idx"] = "100";
    Check(cyxwiz::ResolveInvalidSequenceProjectionConfigurationReason(
              gui::NodeType::Embedding, invalid_padding).has_value(),
          "Embedding padding outside the vocabulary should fail closed");
    const std::map<std::string, std::string> conflicting_head = {
        {"units", "4"}, {"out_features", "5"}};
    Check(cyxwiz::ResolveInvalidSequenceProjectionConfigurationReason(
              gui::NodeType::TimeDistributed, conflicting_head).has_value(),
          "conflicting TimeDistributed width aliases should fail closed");

    const auto* bar_chart_meta = metadata.GetMetadata(gui::NodeType::BarChart);
    Check(bar_chart_meta != nullptr, "BarChart metadata should exist");
    Check(bar_chart_meta->status == cyxwiz::NodeImplementationStatus::Implemented,
          "BarChart should remain an implemented UI workflow node");
    CheckSupportAxis(bar_chart_meta, "Implementation Owner", "ui_only", true, "BarChart");
    CheckSupportAxis(bar_chart_meta, "Support State", "partial", true, "BarChart");
    Check(cyxwiz::CanAddNodeToGraph(*bar_chart_meta),
          "frontend blocked state for UI-only partial nodes should come from support_axes");
    CheckSupportAxisReasonContains(
        bar_chart_meta, "Implementation Owner", "UI/panel workflow surface",
        "BarChart");

    const auto* standard_scaler_meta =
        metadata.GetMetadata(gui::NodeType::StandardScaler);
    Check(standard_scaler_meta != nullptr,
          "StandardScaler metadata should exist for frontend support-axis guard");
    CheckSupportAxis(standard_scaler_meta, "Support State", "real", true,
                     "StandardScaler");
    Check(cyxwiz::CanAddNodeToGraph(*standard_scaler_meta),
          "frontend should not block real operator-backed nodes when support_axes are supported");

    for (const auto& capability :
         cyxwiz::GetPipelineFailClosedRuntimeCapabilities()) {
        auto expected_metadata_type = capability.metadata_node_type;
        if (!expected_metadata_type.has_value()) {
            expected_metadata_type = capability.node_type;
        }
        if (!expected_metadata_type.has_value()) {
            continue;
        }
        const auto* meta = metadata.GetMetadata(*expected_metadata_type);
        if (!meta) {
            Check(!capability.metadata_node_type.has_value(),
                  std::string("missing explicitly mapped fail-closed metadata for runtime name ") +
                      capability.legacy_type_name);
            continue;
        }
        if (capability.blocks_metadata_status) {
            Check(meta->status == cyxwiz::NodeImplementationStatus::Template,
                  std::string("fail-closed runtime metadata should not be marked implemented: ") +
                      capability.legacy_type_name);
            Check(meta->badge == "Blocked",
                  std::string("fail-closed runtime metadata should carry blocked badge: ") +
                      capability.legacy_type_name);
        } else {
            Check(meta->status == cyxwiz::NodeImplementationStatus::Implemented ||
                      meta->status == cyxwiz::NodeImplementationStatus::Template,
                  std::string("non-blocking fail-closed runtime metadata should stay registered: ") +
                      capability.legacy_type_name);
        }
        CheckSupportAxis(
            meta,
            "Runtime",
            "fail_closed",
            false,
            capability.legacy_type_name);
        const auto* runtime_axis = FindSupportAxis(meta, "Runtime");
        Check(runtime_axis != nullptr &&
                  capability.reason != nullptr &&
                  runtime_axis->reason.find(capability.reason) !=
                      std::string::npos,
              std::string("fail-closed runtime support axis should expose reason: ") +
                  capability.legacy_type_name);
        CheckSupportAxis(
            meta,
            "Pipeline Executor",
            "unsupported",
            false,
            capability.legacy_type_name);
        if (capability.blocks_metadata_status) {
            CheckSupportAxis(
                meta,
                "Support State",
                "blocked",
                false,
                capability.legacy_type_name);
        } else {
            const auto* support_state = FindSupportAxis(meta, "Support State");
            Check(support_state != nullptr,
                  std::string("missing support axis Support State: ") +
                      capability.legacy_type_name);
            Check(support_state->value == "partial" ||
                      support_state->value == "real",
                  std::string("non-blocking fail-closed runtime metadata should keep partial or real support state: ") +
                      capability.legacy_type_name);
            Check(support_state->supported,
                  std::string("non-blocking fail-closed runtime metadata should keep supported state: ") +
                      capability.legacy_type_name);
        }
        if (capability.blocks_metadata_status || meta->IsTemplate() ||
            meta->badge == "Blocked") {
            Check(!cyxwiz::CanAddNodeToGraph(*meta),
                  std::string("globally blocked fail-closed metadata should not be addable: ") +
                      capability.legacy_type_name);
        } else {
            Check(cyxwiz::CanAddNodeToGraph(*meta),
                  std::string("lane-specific PipelineExecutor limitations should not block another executable lane: ") +
                      capability.legacy_type_name);
        }
        Check(!FrontendSupportBlockReasonFromAxes(meta).empty(),
              std::string("frontend blocked reason should be available from fail-closed support_axes: ") +
                  capability.legacy_type_name);
    }

    for (const auto& capability :
         cyxwiz::GetPipelineUnsupportedSequentialModelLayerCapabilities()) {
        const auto type = capability.node_type;
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr,
              "missing unsupported training metadata for type " + TypeId(type));
        Check(meta->status == cyxwiz::NodeImplementationStatus::Template,
              "unsupported training type " + TypeId(type) +
                  " should not be marked implemented");
        Check(meta->badge == "Blocked",
              "unsupported training type " + TypeId(type) +
                  " should carry blocked badge");
        CheckSupportAxis(
            meta,
            "Training Backend",
            cyxwiz::PipelineTrainingBackendSupportModeName(
                cyxwiz::PipelineTrainingBackendSupportMode::
                    UnsupportedSequentialModelLayer),
            false,
            TypeId(type));
        CheckSupportAxis(meta, "Model Builder", "unsupported", false, TypeId(type));
        CheckSupportAxis(meta, "Compile", "unsupported", false, TypeId(type));
        CheckSupportAxis(meta, "Training", "unsupported", false, TypeId(type));
        CheckSupportAxis(meta, "Implementation Owner", "training_backend", true, TypeId(type));
        CheckSupportAxis(meta, "Support State", "blocked", false, TypeId(type));
        Check(!cyxwiz::CanAddNodeToGraph(*meta),
              "frontend should block unsupported training nodes from support_axes: " +
                  TypeId(type));
        Check(!FrontendSupportBlockReasonFromAxes(meta).empty(),
              "frontend should find unsupported training reason from support_axes: " +
                  TypeId(type));
        const auto* training_axis = FindSupportAxis(meta, "Training Backend");
        Check(training_axis != nullptr &&
                  capability.reason != nullptr &&
                  training_axis->reason.find(capability.reason) !=
                      std::string::npos,
              "unsupported training type " + TypeId(type) +
                  " should expose reason on structured support axis");
    }

    for (const auto& capability :
         cyxwiz::GetPipelineUnsupportedTrainingControlCapabilities()) {
        const auto* meta = metadata.GetMetadata(capability.node_type);
        Check(meta != nullptr,
              "missing unsupported training control metadata for type " +
                  TypeId(capability.node_type));
        Check(meta->status == cyxwiz::NodeImplementationStatus::Template,
              "unsupported training control " + TypeId(capability.node_type) +
                  " should not be marked implemented");
        Check(meta->badge == "Blocked",
              "unsupported training control " + TypeId(capability.node_type) +
                  " should carry blocked badge");
        CheckSupportAxis(
            meta,
            "Training Backend",
            cyxwiz::PipelineTrainingBackendSupportModeName(
                cyxwiz::PipelineTrainingBackendSupportMode::
                    UnsupportedTrainingControl),
            false,
            TypeId(capability.node_type));
        CheckSupportAxis(
            meta,
            "Training Role",
            cyxwiz::PipelineTrainingSupportRoleName(
                cyxwiz::PipelineTrainingSupportRole::TrainingControl),
            false,
            TypeId(capability.node_type));
        CheckSupportAxis(
            meta,
            "Compile",
            "unsupported",
            false,
            TypeId(capability.node_type));
        CheckSupportAxis(
            meta,
            "Training",
            "unsupported",
            false,
            TypeId(capability.node_type));
        CheckSupportAxis(
            meta,
            "Implementation Owner",
            "training_backend",
            true,
            TypeId(capability.node_type));
        CheckSupportAxis(
            meta,
            "Support State",
            "blocked",
            false,
            TypeId(capability.node_type));
        Check(!cyxwiz::CanAddNodeToGraph(*meta),
              "frontend should block unsupported training controls from support_axes: " +
                  TypeId(capability.node_type));
        Check(!FrontendSupportBlockReasonFromAxes(meta).empty(),
              "frontend should find unsupported training control reason from support_axes: " +
                  TypeId(capability.node_type));
        const auto* training_axis = FindSupportAxis(meta, "Training Backend");
        Check(training_axis != nullptr &&
                  capability.reason != nullptr &&
                  training_axis->reason.find(capability.reason) !=
                      std::string::npos,
              "unsupported training control " + TypeId(capability.node_type) +
                  " should expose reason on structured support axis");
    }

    for (const auto& capability :
         cyxwiz::GetPipelineUnsupportedTrainingWorkflowCapabilities()) {
        const auto* meta = metadata.GetMetadata(capability.node_type);
        Check(meta != nullptr,
              "missing unsupported training workflow metadata for type " +
                  TypeId(capability.node_type));
        Check(meta->status == cyxwiz::NodeImplementationStatus::Template &&
                  meta->badge == "Blocked",
              "unsupported training workflow should remain blocked: " +
                  TypeId(capability.node_type));
        CheckSupportAxis(
            meta,
            "Training Backend",
            cyxwiz::PipelineTrainingBackendSupportModeName(
                cyxwiz::PipelineTrainingBackendSupportMode::
                    UnsupportedTrainingWorkflow),
            false,
            TypeId(capability.node_type));
        CheckSupportAxis(
            meta,
            "Training Role",
            cyxwiz::PipelineTrainingSupportRoleName(
                cyxwiz::PipelineTrainingSupportRole::TrainingWorkflow),
            false,
            TypeId(capability.node_type));
        CheckSupportAxis(meta, "Compile", "unsupported", false,
                         TypeId(capability.node_type));
        CheckSupportAxis(meta, "Training", "unsupported", false,
                         TypeId(capability.node_type));
        CheckSupportAxis(meta, "Implementation Owner",
                         "unowned_training_workflow", false,
                         TypeId(capability.node_type));
        CheckSupportAxis(meta, "Support State", "blocked", false,
                         TypeId(capability.node_type));
        Check(!cyxwiz::CanAddNodeToGraph(*meta) &&
                  !FrontendSupportBlockReasonFromAxes(meta).empty(),
              "frontend should fail closed with a structured workflow reason: " +
                  TypeId(capability.node_type));
        const auto* training_axis = FindSupportAxis(meta, "Training Backend");
        Check(training_axis != nullptr && capability.reason != nullptr &&
                  training_axis->reason.find(capability.reason) !=
                      std::string::npos,
              "unsupported training workflow should expose the canonical reason: " +
                  TypeId(capability.node_type));
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
