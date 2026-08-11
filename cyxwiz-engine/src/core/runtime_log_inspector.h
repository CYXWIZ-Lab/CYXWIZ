#pragma once

#include "runtime_log_filter.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace cyxwiz {

struct RuntimeLogInspectorCriteria {
    std::array<bool, 6> levels = {true, true, true, true, true, true};
    std::string category;
    std::string source;
    std::string code;
    std::string run_id;
    std::optional<uint64_t> task_id;
    std::optional<int> device_id;
    std::string backend;
    std::string text;
    std::string structured_filter;

    bool operator==(const RuntimeLogInspectorCriteria&) const = default;
};

struct RuntimeLogInspectorRequest {
    RuntimeLogInspectorCriteria criteria;
    uint64_t after_sequence = 0;
    uint64_t through_sequence = std::numeric_limits<uint64_t>::max();
    size_t display_limit = 1000;
};

struct RuntimeLogInspectorResult {
    RuntimeLogQueryResult query;
    std::string effective_filter;
    std::optional<RuntimeLogFilterParseError> filter_error;
};

std::string BuildRuntimeLogInspectorFilter(
    const RuntimeLogInspectorCriteria& criteria);

RuntimeLogInspectorResult QueryRuntimeLogInspector(
    const RuntimeLogStore& store,
    const RuntimeLogInspectorRequest& request);

} // namespace cyxwiz
