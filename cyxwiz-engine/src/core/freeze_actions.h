#pragma once

// The canvas actions that freeze and unfreeze layers, and the Trainable /
// Frozen count, as functions of the graph's nodes. Pure, so the menu, the
// footer and the tests share them.

#include "frozen_layers.h"
#include "graph_compiler.h"

#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace cyxwiz::frozen {

// The node holds weights and its Frozen setting is on.
bool IsFrozen(const gui::MLNode& node);

// Turns Frozen on or off on the listed nodes that hold weights. Returns how
// many nodes changed.
size_t SetFrozen(std::vector<gui::MLNode>& nodes, const std::set<int>& ids, bool frozen);

// Turns Frozen off everywhere. Returns how many nodes changed.
size_t UnfreezeAll(std::vector<gui::MLNode>& nodes);

// The nodes that hold weights, first layer to last, from a compiled graph.
std::vector<int> ModelOrder(const std::vector<CompiledLayer>& layers);

// Freezes every layer with weights except the last `keep` of the model, which
// are unfrozen (keep 1 = all but the head). Returns how many nodes changed.
size_t FreezeAllButLast(std::vector<gui::MLNode>& nodes, const std::vector<int>& model_order, size_t keep);

// Freezes the nodes with these names (the layers a weights file filled).
// Returns how many nodes changed.
size_t FreezeNamed(std::vector<gui::MLNode>& nodes, const std::set<std::string>& names);

struct Totals {
    long long trainable = 0;
    long long frozen = 0;
};

// Learnable weights that a run will change and that it will keep, from the
// compiled layers and the built model's count per compiled layer index.
Totals CountTotals(const std::vector<CompiledLayer>& layers, const std::map<size_t, long long>& layer_parameters);

// "Trainable 1,026 · Frozen 11,176,512"
std::string TotalsText(const Totals& totals);

}  // namespace cyxwiz::frozen
