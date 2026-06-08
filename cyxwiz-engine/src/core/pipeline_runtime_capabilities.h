#pragma once

#include "../gui/node_editor.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

struct PipelineOperatorRuntimeCapability {
    const char* legacy_type_name;
    gui::NodeType node_type;
};

struct PipelineFailClosedRuntimeCapability {
    const char* legacy_type_name;
    const char* reason;
    std::optional<gui::NodeType> node_type = std::nullopt;
    std::optional<gui::NodeType> metadata_node_type = std::nullopt;
};

enum class PipelineLegacyDispatchKind {
    Unknown,
    SaveDataset,
    DeployToNodeEditor,
    TextClean,
    TextTokenize,
    TextVectorize,
    TSWindow,
    TSFeatures,
    TSLag,
    TSDiff,
    PolynomialFeatures,
    Binning,
};

struct PipelineLegacyRuntimeCapability {
    const char* legacy_type_name;
    std::optional<gui::NodeType> node_type = std::nullopt;
    PipelineLegacyDispatchKind dispatch_kind =
        PipelineLegacyDispatchKind::Unknown;
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
};

struct PipelineFloatParameterRuntimeCapability {
    const char* legacy_type_name;
    const char* parameter_name;
    double minimum;
    double maximum;
};

struct PipelineUnsupportedTrainingNodeCapability {
    gui::NodeType node_type;
    const char* reason;
};

enum class PipelineTrainingBackendSupportMode {
    Allowed,
    UnsupportedSequentialModelLayer,
    UnsupportedTrainingControl,
};

struct PipelineTrainingBackendSupport {
    PipelineTrainingBackendSupportMode mode =
        PipelineTrainingBackendSupportMode::Allowed;
    bool compile_supported = true;
    bool training_supported = true;
    const char* reason = nullptr;
};

enum class PipelineStorageBackend {
    Unknown,
    ArrowTable,
    ParquetBacked,
    ImageDataset,
    AudioDataset,
    TextDataset,
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
    PipelineLegacyDispatchKind legacy_dispatch_kind =
        PipelineLegacyDispatchKind::Unknown;
    PipelineRuntimeImplementationOwner implementation_owner =
        PipelineRuntimeImplementationOwner::Unknown;
};

const std::vector<PipelineOperatorRuntimeCapability>&
GetPipelineOperatorRuntimeCapabilities();

const std::vector<PipelineFailClosedRuntimeCapability>&
GetPipelineFailClosedRuntimeCapabilities();

const std::vector<PipelineLegacyRuntimeCapability>&
GetPipelineLegacyRuntimeCapabilities();

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

const std::vector<PipelineMaterializerStorageBackendCapability>&
GetPipelineMaterializerStorageBackendCapabilities();

PipelineRuntimeSupport ResolvePipelineRuntimeSupport(const std::string& legacy_type_name);

PipelineRuntimeSupport ResolvePipelineRuntimeSupport(gui::NodeType node_type);

const char* PipelineStorageBackendName(PipelineStorageBackend backend);

const char* PipelineRuntimeSupportModeName(PipelineRuntimeSupportMode mode);

const char* PipelineRuntimeFailModeName(PipelineRuntimeFailMode fail_mode);

const char* PipelineRuntimeImplementationOwnerName(
    PipelineRuntimeImplementationOwner owner);

const char* PipelineMaterializerStorageSupportName(
    PipelineMaterializerStorageSupport support);

const char* PipelineTrainingBackendSupportModeName(
    PipelineTrainingBackendSupportMode mode);

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

const char* ResolvePipelineUnsupportedSequentialModelLayerReason(gui::NodeType node_type);

const char* ResolvePipelineUnsupportedTrainingControlReason(gui::NodeType node_type);

bool IsPipelineUnsupportedSequentialModelLayer(gui::NodeType node_type);

bool IsPipelineUnsupportedTrainingControlNode(gui::NodeType node_type);

PipelineTrainingBackendSupport
ResolvePipelineTrainingBackendSupport(gui::NodeType node_type);

} // namespace cyxwiz
