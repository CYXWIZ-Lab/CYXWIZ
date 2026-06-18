#pragma once

#include "debug_trace_record.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cyxwiz {

struct DebugExportCorrelationInput {
    std::string artifact_kind;
    std::string artifact_path;
    std::string exporter_name;
    uint64_t graph_hash = 0;
    bool compile_success = false;
    std::string compile_status;
    std::vector<int> source_node_ids;
    std::string generated_content;
    std::string message;
};

class DebugExportCorrelationTracer {
public:
    static constexpr const char* kSchema =
        "cyxwiz.debug.export_correlation.v1";

    DebugTraceRecord BuildTrace(
        const std::string& run_id,
        const DebugExportCorrelationInput& input) const;

    static uint64_t Fingerprint(const std::string& content);
};

} // namespace cyxwiz
