#pragma once

#include "debug_trace_record.h"
#include "graph_compiler.h"
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace cyxwiz {

enum class DebugRunMode {
    Preflight,
    LocalDebug,
    SmokeRun,
    FullTrainTrace
};

struct DebugPreflightResult {
    bool ready = false;
    std::string summary;
    std::vector<ValidationIssue> issues;
};

struct SmokeRunResult;

struct DebugGraphNodeSnapshot {
    int id = -1;
    int type = 0;
    std::string name;
    std::vector<std::pair<std::string, std::string>> parameters;
    size_t input_count = 0;
    size_t output_count = 0;
};

struct DebugGraphLinkSnapshot {
    int id = -1;
    int from_node = -1;
    int from_pin = -1;
    int to_node = -1;
    int to_pin = -1;
    int type = 0;
};

struct DebugSession {
    std::string run_id;
    DebugRunMode mode = DebugRunMode::LocalDebug;
    std::string mode_name;
    uint64_t graph_hash = 0;
    size_t node_count = 0;
    size_t link_count = 0;
    size_t selected_sample_index = 0;
    std::vector<DebugGraphNodeSnapshot> graph_nodes;
    std::vector<DebugGraphLinkSnapshot> graph_links;
    std::string dataset_name;
    std::string sample_summary;
    TrainingConfiguration config;
    DebugPreflightResult preflight;
    std::vector<DebugTraceRecord> traces;
    std::vector<StudioEventRecord> studio_events;
    std::chrono::steady_clock::time_point started_at{};
};

} // namespace cyxwiz
