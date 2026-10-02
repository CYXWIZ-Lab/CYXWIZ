#pragma once

#include "graph_model.h"

#include <string>

namespace cyxwiz {

// The PipelineExecutor's type name for a node type ("DataInput",
// "FilterRows", ...; "Unknown" when the executor has none). One table for
// Execute Pipeline and the Plot node's result lane (TOFIX134 P2).
std::string PipelineTypeName(gui::NodeType type);

}  // namespace cyxwiz
