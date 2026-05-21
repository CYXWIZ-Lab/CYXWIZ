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
};

} // namespace cyxwiz
