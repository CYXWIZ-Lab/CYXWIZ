#pragma once

#include "../gui/node_editor.h"
#include "normalization_regularization_configuration_policy.h"
#include "recurrent_configuration_policy.h"
#include "sequence_projection_configuration_policy.h"
#include "transformer_configuration_policy.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

struct PipelineOperatorRuntimeCapability {
    const char* legacy_type_name;
    gui::NodeType node_type;
};

enum class PipelineBackendPrimitiveEvidence {
    NotApplicable,
    ProvenNodePrimitive,
    RelatedHelperOnly,
    Missing,
};

struct PipelineFailClosedRuntimeCapability {
    const char* legacy_type_name;
    const char* reason;
    std::optional<gui::NodeType> node_type = std::nullopt;
    std::optional<gui::NodeType> metadata_node_type = std::nullopt;
    bool blocks_metadata_status = true;
    PipelineBackendPrimitiveEvidence primitive_evidence =
        PipelineBackendPrimitiveEvidence::NotApplicable;
};

enum class PipelineLegacyAliasDecision {
    Unknown,
    NormalizeToCanonical,
    HiddenCompatibilityAlias,
};

struct PipelineLegacyAliasDecisionCapability {
    const char* alias_type_name;
    const char* canonical_type_name;
    gui::NodeType canonical_node_type;
    PipelineLegacyAliasDecision decision =
        PipelineLegacyAliasDecision::Unknown;
    const char* reason = nullptr;
};

struct PipelineLegacyRuntimeCapability {
    const char* legacy_type_name;
    std::optional<gui::NodeType> node_type = std::nullopt;
};

struct PipelineSourceRuntimeCapability {
    const char* legacy_type_name;
};

struct PipelineInputArityRuntimeCapability {
    const char* legacy_type_name;
    int required_input_count;
};

struct PipelineRequiredParameterRuntimeCapability {
    const char* legacy_type_name;
    std::vector<const char*> required_parameters;
};

struct PipelineAllowedParameterValuesRuntimeCapability {
    const char* legacy_type_name;
    const char* parameter_name;
    const char* default_value;
    std::vector<const char*> allowed_values;
};

struct PipelineIntegerParameterRuntimeCapability {
    const char* legacy_type_name;
    const char* parameter_name;
    int64_t minimum;
    bool comma_separated = false;
    std::vector<int64_t> forbidden_values = {};
};

struct PipelineFloatParameterRuntimeCapability {
    const char* legacy_type_name;
    const char* parameter_name;
    std::optional<double> minimum;
    std::optional<double> maximum;
    bool minimum_inclusive = true;
    bool maximum_inclusive = true;
};

struct PipelineUnsupportedTrainingNodeCapability {
    gui::NodeType node_type;
    const char* reason;
    PipelineBackendPrimitiveEvidence primitive_evidence =
        PipelineBackendPrimitiveEvidence::NotApplicable;
};

struct PipelineSupportedTrainingNodeCapability {
    gui::NodeType node_type;
    const char* reason;
};

enum class PipelineTrainingSupportRole {
    ModelLayer,
    Activation,
    Loss,
    Optimizer,
    TrainingControl,
    TrainingWorkflow,
    DataSource,
    Preprocessing,
};

struct PipelineSupportedTrainingRoleCapability {
    gui::NodeType node_type;
    PipelineTrainingSupportRole role = PipelineTrainingSupportRole::ModelLayer;
    const char* reason;
};

enum class PipelineTrainingBackendSupportMode {
    Allowed,
    UnsupportedSequentialModelLayer,
    UnsupportedTrainingControl,
    UnsupportedTrainingWorkflow,
};

struct PipelineTrainingBackendSupport {
    PipelineTrainingBackendSupportMode mode =
        PipelineTrainingBackendSupportMode::Allowed;
    bool compile_supported = true;
    bool training_supported = true;
    const char* reason = nullptr;
    PipelineBackendPrimitiveEvidence primitive_evidence =
        PipelineBackendPrimitiveEvidence::NotApplicable;
};

enum class PipelineStorageBackend {
    Unknown,
    ArrowTable,
    ParquetBacked,
    ImageDataset,
    AudioDataset,
    TextDataset,
    SparseFeatureDataset,
};

enum class PipelineRuntimeSupportMode {
    Unknown,
    LegacyExecutor,
    OperatorBacked,
    FailClosed,
};

enum class PipelineRuntimeFailMode {
    Unknown,
    Real,
    HardFail,
    Simulated,
    Passthrough,
};

enum class PipelineRuntimeImplementationOwner {
    Unknown,
    None,
    PipelineExecutor,
    PipelineOperatorFactory,
};

enum class PipelineMaterializerStorageSupport {
    None,
    ArrowTableOnly,
};

struct PipelineMaterializerStorageBackendCapability {
    PipelineStorageBackend backend = PipelineStorageBackend::Unknown;
    PipelineMaterializerStorageSupport storage_support =
        PipelineMaterializerStorageSupport::None;
    bool materializer_supported = false;
    const char* reason = nullptr;
};

struct PipelineRuntimeSupport {
    PipelineRuntimeSupport() = default;

    PipelineRuntimeSupport(
        PipelineRuntimeSupportMode mode_value,
        PipelineRuntimeFailMode fail_mode_value,
        std::optional<gui::NodeType> node_type_value,
        std::optional<gui::NodeType> operator_type_value,
        const char* fail_closed_reason_value,
        PipelineMaterializerStorageSupport materializer_storage_support_value,
        bool materializer_arrow_table_supported_value,
        bool pipeline_executor_supported_value)
        : mode(mode_value),
          fail_mode(fail_mode_value),
          node_type(node_type_value),
          operator_type(operator_type_value),
          fail_closed_reason(fail_closed_reason_value),
          materializer_storage_support(materializer_storage_support_value),
          materializer_arrow_table_supported(
              materializer_arrow_table_supported_value),
          pipeline_executor_supported(pipeline_executor_supported_value) {}

    PipelineRuntimeSupportMode mode = PipelineRuntimeSupportMode::Unknown;
    PipelineRuntimeFailMode fail_mode = PipelineRuntimeFailMode::Unknown;
    std::optional<gui::NodeType> node_type = std::nullopt;
    std::optional<gui::NodeType> operator_type = std::nullopt;
    const char* fail_closed_reason = nullptr;
    PipelineMaterializerStorageSupport materializer_storage_support =
        PipelineMaterializerStorageSupport::None;
    bool materializer_arrow_table_supported = false;
    bool pipeline_executor_supported = false;
    bool source_node = false;
    std::optional<int> required_input_count = std::nullopt;
    std::optional<gui::NodeType> metadata_node_type = std::nullopt;
    std::vector<const char*> required_parameters;
    std::vector<PipelineAllowedParameterValuesRuntimeCapability>
        allowed_parameter_values;
    std::vector<PipelineIntegerParameterRuntimeCapability> integer_parameters;
    std::vector<PipelineFloatParameterRuntimeCapability> float_parameters;
    PipelineRuntimeImplementationOwner implementation_owner =
        PipelineRuntimeImplementationOwner::Unknown;
    PipelineBackendPrimitiveEvidence primitive_evidence =
        PipelineBackendPrimitiveEvidence::NotApplicable;
};

// Canonicalize saved/runtime parameter aliases without advertising them in
// new-node metadata. Legacy values may replace constructor defaults while
// importing pattern templates.
void CanonicalizePipelineParameterAliases(
    gui::NodeType node_type,
    std::map<std::string, std::string>& parameters,
    bool prefer_legacy = false);

const std::vector<PipelineOperatorRuntimeCapability>&
GetPipelineOperatorRuntimeCapabilities();

const std::vector<PipelineFailClosedRuntimeCapability>&
GetPipelineFailClosedRuntimeCapabilities();

const std::vector<PipelineLegacyRuntimeCapability>&
GetPipelineLegacyRuntimeCapabilities();

const std::vector<PipelineLegacyAliasDecisionCapability>&
GetPipelineLegacyAliasDecisionCapabilities();

const std::vector<PipelineSourceRuntimeCapability>&
GetPipelineSourceRuntimeCapabilities();

const std::vector<PipelineInputArityRuntimeCapability>&
GetPipelineInputArityRuntimeCapabilities();

const std::vector<PipelineRequiredParameterRuntimeCapability>&
GetPipelineRequiredParameterRuntimeCapabilities();

const std::vector<PipelineAllowedParameterValuesRuntimeCapability>&
GetPipelineAllowedParameterValuesRuntimeCapabilities();

const std::vector<PipelineIntegerParameterRuntimeCapability>&
GetPipelineIntegerParameterRuntimeCapabilities();

const std::vector<PipelineFloatParameterRuntimeCapability>&
GetPipelineFloatParameterRuntimeCapabilities();

const std::vector<PipelineUnsupportedTrainingNodeCapability>&
GetPipelineUnsupportedSequentialModelLayerCapabilities();

const std::vector<PipelineUnsupportedTrainingNodeCapability>&
GetPipelineUnsupportedTrainingControlCapabilities();

const std::vector<PipelineUnsupportedTrainingNodeCapability>&
GetPipelineUnsupportedTrainingWorkflowCapabilities();

const std::vector<PipelineSupportedTrainingNodeCapability>&
GetPipelineSupportedTrainingBackendCapabilities();

const std::vector<PipelineSupportedTrainingRoleCapability>&
GetPipelineSupportedTrainingRoleCapabilities();

const std::vector<PipelineMaterializerStorageBackendCapability>&
GetPipelineMaterializerStorageBackendCapabilities();

PipelineRuntimeSupport ResolvePipelineRuntimeSupport(const std::string& legacy_type_name);

PipelineRuntimeSupport ResolvePipelineRuntimeSupport(gui::NodeType node_type);

const char* PipelineStorageBackendName(PipelineStorageBackend backend);

const char* PipelineRuntimeSupportModeName(PipelineRuntimeSupportMode mode);

const char* PipelineRuntimeFailModeName(PipelineRuntimeFailMode fail_mode);

const char* PipelineRuntimeImplementationOwnerName(
    PipelineRuntimeImplementationOwner owner);

const char* PipelineLegacyAliasDecisionName(
    PipelineLegacyAliasDecision decision);

const char* PipelineMaterializerStorageSupportName(
    PipelineMaterializerStorageSupport support);

const char* PipelineTrainingBackendSupportModeName(
    PipelineTrainingBackendSupportMode mode);

const char* PipelineBackendPrimitiveEvidenceName(
    PipelineBackendPrimitiveEvidence evidence);

const char* PipelineTrainingSupportRoleName(PipelineTrainingSupportRole role);

PipelineMaterializerStorageBackendCapability
ResolvePipelineMaterializerStorageBackendSupport(PipelineStorageBackend backend);

std::optional<gui::NodeType>
ResolvePipelineOperatorRuntimeType(const std::string& legacy_type_name);

bool IsPipelineOperatorRuntimeNode(const std::string& legacy_type_name);

std::optional<gui::NodeType>
ResolvePipelineRuntimeNodeType(const std::string& legacy_type_name);

const char* ResolvePipelineRuntimeLegacyTypeName(gui::NodeType node_type);

const char* ResolvePipelineFailClosedReason(const std::string& legacy_type_name);

bool IsPipelineFailClosedRuntimeNode(const std::string& legacy_type_name);

bool IsPipelineLegacyRuntimeNode(const std::string& legacy_type_name);

const PipelineLegacyAliasDecisionCapability*
ResolvePipelineLegacyAliasDecision(const std::string& alias_type_name);

bool IsPipelineSourceRuntimeNode(const std::string& legacy_type_name);

std::optional<int> ResolvePipelineRequiredInputCount(const std::string& legacy_type_name);

std::vector<const char*>
ResolvePipelineRequiredParameters(const std::string& legacy_type_name);

std::vector<PipelineAllowedParameterValuesRuntimeCapability>
ResolvePipelineAllowedParameterValues(const std::string& legacy_type_name);

std::vector<PipelineIntegerParameterRuntimeCapability>
ResolvePipelineIntegerParameters(const std::string& legacy_type_name);

std::vector<PipelineFloatParameterRuntimeCapability>
ResolvePipelineFloatParameters(const std::string& legacy_type_name);

bool ValidatePipelineRuntimeParameterCapabilities(
    const std::string& subject,
    const std::map<std::string, std::string>& parameters,
    const std::vector<PipelineAllowedParameterValuesRuntimeCapability>&
        allowed_parameter_values,
    const std::vector<PipelineIntegerParameterRuntimeCapability>&
        integer_parameters,
    const std::vector<PipelineFloatParameterRuntimeCapability>&
        float_parameters,
    const char* unsupported_context,
    std::string& error);

const char* ResolvePipelineUnsupportedSequentialModelLayerReason(gui::NodeType node_type);

const char* ResolvePipelineUnsupportedTrainingControlReason(gui::NodeType node_type);

const char* ResolvePipelineUnsupportedTrainingWorkflowReason(gui::NodeType node_type);

bool IsPipelineUnsupportedSequentialModelLayer(gui::NodeType node_type);

bool IsPipelineUnsupportedTrainingControlNode(gui::NodeType node_type);

bool IsPipelineUnsupportedTrainingWorkflowNode(gui::NodeType node_type);

bool IsPipelineSupportedTrainingBackendNode(gui::NodeType node_type);

bool IsPipelineSupportedTrainingRoleNode(gui::NodeType node_type);

PipelineTrainingBackendSupport
ResolvePipelineTrainingBackendSupport(gui::NodeType node_type);

} // namespace cyxwiz
