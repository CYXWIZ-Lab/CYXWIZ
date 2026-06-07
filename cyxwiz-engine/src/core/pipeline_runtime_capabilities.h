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
};

struct PipelineLegacyRuntimeCapability {
    const char* legacy_type_name;
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

struct PipelineUnsupportedTrainingNodeCapability {
    gui::NodeType node_type;
    const char* reason;
};

enum class PipelineRuntimeSupportMode {
    Unknown,
    LegacyExecutor,
    OperatorBacked,
    FailClosed,
};

enum class PipelineMaterializerStorageSupport {
    None,
    ArrowTableOnly,
};

struct PipelineRuntimeSupport {
    PipelineRuntimeSupportMode mode = PipelineRuntimeSupportMode::Unknown;
    std::optional<gui::NodeType> operator_type = std::nullopt;
    const char* fail_closed_reason = nullptr;
    PipelineMaterializerStorageSupport materializer_storage_support =
        PipelineMaterializerStorageSupport::None;
    bool materializer_arrow_table_supported = false;
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

const std::vector<PipelineUnsupportedTrainingNodeCapability>&
GetPipelineUnsupportedSequentialModelLayerCapabilities();

const std::vector<PipelineUnsupportedTrainingNodeCapability>&
GetPipelineUnsupportedTrainingControlCapabilities();

PipelineRuntimeSupport ResolvePipelineRuntimeSupport(const std::string& legacy_type_name);

std::optional<gui::NodeType>
ResolvePipelineOperatorRuntimeType(const std::string& legacy_type_name);

bool IsPipelineOperatorRuntimeNode(const std::string& legacy_type_name);

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

const char* ResolvePipelineUnsupportedSequentialModelLayerReason(gui::NodeType node_type);

const char* ResolvePipelineUnsupportedTrainingControlReason(gui::NodeType node_type);

bool IsPipelineUnsupportedSequentialModelLayer(gui::NodeType node_type);

bool IsPipelineUnsupportedTrainingControlNode(gui::NodeType node_type);

} // namespace cyxwiz
