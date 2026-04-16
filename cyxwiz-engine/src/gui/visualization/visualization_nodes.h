#pragma once

// Visualization node factory helpers. Each Populate* fills in pins and
// parameters on a pre-constructed MLNode for a given chart NodeType —
// the NodeEditor::CreateNode switch just delegates here so the chart
// framework owns its own translation unit instead of bloating
// node_editor_nodes.cpp.
//
// Category contract: these are Cat 2 inspection nodes per CLAUDE.md.
// They have input pins, NO output pins, and do NOT transform data.
// Rendering happens in the corresponding *Dialog class when the user
// double-clicks the node on the canvas. Training skips them.

#include "../node_editor.h"

namespace gui::visualization {

// Bar chart / histogram node. Inputs: Data (Tensor, required), Labels
// (Labels, optional — enables class-distribution mode). Parameters:
// chart_type, column, title, max_bars. See BarChartDialog for the
// rendering contract.
void PopulateBarChartNode(MLNode& node, int& next_pin_id);

} // namespace gui::visualization
