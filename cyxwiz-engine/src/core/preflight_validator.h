#pragma once

#include "debug_session.h"
#include "../gui/node_editor.h"
#include <cstdint>
#include <vector>

namespace cyxwiz {

class PreflightValidator {
public:
    DebugPreflightResult Validate(
        const TrainingConfiguration& config,
        const std::vector<gui::MLNode>& nodes,
        const std::vector<gui::NodeLink>& links,
        uint64_t graph_hash = 0) const;
};

} // namespace cyxwiz
