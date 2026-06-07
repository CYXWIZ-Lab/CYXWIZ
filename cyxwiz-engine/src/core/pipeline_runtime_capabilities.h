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

enum class PipelineRuntimeSupportMode {
    Unknown,
    OperatorBacked,
    FailClosed,
};

struct PipelineRuntimeSupport {
    PipelineRuntimeSupportMode mode = PipelineRuntimeSupportMode::Unknown;
    std::optional<gui::NodeType> operator_type = std::nullopt;
    const char* fail_closed_reason = nullptr;
};

const std::vector<PipelineOperatorRuntimeCapability>&
GetPipelineOperatorRuntimeCapabilities();

const std::vector<PipelineFailClosedRuntimeCapability>&
GetPipelineFailClosedRuntimeCapabilities();

PipelineRuntimeSupport ResolvePipelineRuntimeSupport(const std::string& legacy_type_name);

std::optional<gui::NodeType>
ResolvePipelineOperatorRuntimeType(const std::string& legacy_type_name);

bool IsPipelineOperatorRuntimeNode(const std::string& legacy_type_name);

const char* ResolvePipelineFailClosedReason(const std::string& legacy_type_name);

bool IsPipelineFailClosedRuntimeNode(const std::string& legacy_type_name);

} // namespace cyxwiz
