#pragma once

#include "debug_run_store.h"

namespace cyxwiz {

class DebugTrainingGraphDiff {
public:
    static constexpr const char* kSchema =
        "cyxwiz.debug.training_graph_diff.v1";

    DebugTraceRecord BuildTrace(
        const DebugRunStoreRecord& baseline,
        const DebugRunStoreRecord& current) const;
};

} // namespace cyxwiz
