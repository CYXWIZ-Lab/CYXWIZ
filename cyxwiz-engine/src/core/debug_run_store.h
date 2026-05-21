#pragma once

#include "debug_recommendation_engine.h"
#include "debug_trace_record.h"
#include "graph_compiler.h"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

struct DebugRunStoreSummary {
    std::string run_id;
    std::string timestamp;
    uint64_t graph_hash = 0;
    bool success = false;
    size_t issue_count = 0;
    size_t trace_count = 0;
    size_t event_count = 0;
    size_t recommendation_count = 0;
    std::string summary;
    std::string file_path;
};

struct DebugRunStoreRecord {
    DebugRunStoreSummary summary;
    std::vector<ValidationIssue> issues;
    std::vector<DebugTraceRecord> traces;
    std::vector<StudioEventRecord> studio_events;
    std::vector<DebugRecommendation> recommendations;
};

class DebugRunStore {
public:
    static bool Save(const DebugRunStoreRecord& record);
    static std::optional<DebugRunStoreRecord> Load(const std::string& run_id);
    static std::vector<DebugRunStoreSummary> ListRecent(size_t max_runs = 10);
};

} // namespace cyxwiz
