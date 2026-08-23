#pragma once

#include "debug_session.h"
#include "../gui/node_editor.h"
#include <cstdint>
#include <string>
#include <vector>

namespace cyxwiz {

class DebugSessionManager {
public:
    static DebugSession StartSession(
        const std::string& run_id,
        const std::string& mode_name,
        uint64_t graph_hash,
        const std::vector<gui::MLNode>& nodes,
        const std::vector<gui::NodeLink>& links,
        size_t selected_sample_index);

    static DebugTraceRecord BuildGraphSnapshotTrace(const DebugSession& session);

    static bool FullWorkflowSucceeded(
        bool compile_success,
        bool preflight_ready,
        bool smoke_supported,
        bool smoke_success,
        bool has_debug_result,
        bool debug_success);
};

} // namespace cyxwiz
