#pragma once

#include "../gui/node_editor.h"

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

enum class PipelineRuntimeSupportMode {
    Unknown,
    LegacyExecutor,
    OperatorBacked,
    FailClosed,
};

struct PipelineRuntimeSupport {
    PipelineRuntimeSupportMode mode = PipelineRuntimeSupportMode::Unknown;
    std::optional<gui::NodeType> operator_type = std::nullopt;
    const char* fail_closed_reason = nullptr;
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

PipelineRuntimeSupport ResolvePipelineRuntimeSupport(const std::string& legacy_type_name);

std::optional<gui::NodeType>
ResolvePipelineOperatorRuntimeType(const std::string& legacy_type_name);

bool IsPipelineOperatorRuntimeNode(const std::string& legacy_type_name);

const char* ResolvePipelineFailClosedReason(const std::string& legacy_type_name);

bool IsPipelineFailClosedRuntimeNode(const std::string& legacy_type_name);

bool IsPipelineLegacyRuntimeNode(const std::string& legacy_type_name);

bool IsPipelineSourceRuntimeNode(const std::string& legacy_type_name);

std::optional<int> ResolvePipelineRequiredInputCount(const std::string& legacy_type_name);

} // namespace cyxwiz
