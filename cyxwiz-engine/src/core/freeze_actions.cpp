#include "freeze_actions.h"

#include <algorithm>

namespace cyxwiz::frozen {
namespace {

bool SettingOn(const std::map<std::string, std::string>& parameters) {
    const auto it = parameters.find(kFrozen);
    return it != parameters.end() && it->second == "true";
}

// Returns true when the node changed.
bool Apply(gui::MLNode& node, bool frozen) {
    if (!LayerNodeHoldsWeights(node.type) || SettingOn(node.parameters) == frozen) return false;
    node.parameters[kFrozen] = frozen ? "true" : "false";
    return true;
}

std::string Grouped(long long value) {
    std::string digits = std::to_string(value);
    for (int at = static_cast<int>(digits.size()) - 3; at > 0; at -= 3) digits.insert(static_cast<size_t>(at), ",");
    return digits;
}

}  // namespace

bool IsFrozen(const gui::MLNode& node) { return LayerNodeHoldsWeights(node.type) && SettingOn(node.parameters); }

size_t SetFrozen(std::vector<gui::MLNode>& nodes, const std::set<int>& ids, bool frozen) {
    size_t changed = 0;
    for (auto& node : nodes) {
        if (ids.count(node.id) && Apply(node, frozen)) ++changed;
    }
    return changed;
}

size_t UnfreezeAll(std::vector<gui::MLNode>& nodes) {
    size_t changed = 0;
    for (auto& node : nodes) {
        if (Apply(node, false)) ++changed;
    }
    return changed;
}

std::vector<int> ModelOrder(const std::vector<CompiledLayer>& layers) {
    std::vector<int> order;
    for (const auto& layer : layers) {
        if (LayerNodeHoldsWeights(layer.type) && std::find(order.begin(), order.end(), layer.node_id) == order.end())
            order.push_back(layer.node_id);
    }
    return order;
}

size_t FreezeAllButLast(std::vector<gui::MLNode>& nodes, const std::vector<int>& model_order, size_t keep) {
    const size_t first_kept = model_order.size() > keep ? model_order.size() - keep : 0;
    const std::set<int> frozen(model_order.begin(), model_order.begin() + static_cast<std::ptrdiff_t>(first_kept));
    const std::set<int> kept(model_order.begin() + static_cast<std::ptrdiff_t>(first_kept), model_order.end());
    return SetFrozen(nodes, frozen, true) + SetFrozen(nodes, kept, false);
}

size_t FreezeNamed(std::vector<gui::MLNode>& nodes, const std::set<std::string>& names) {
    size_t changed = 0;
    for (auto& node : nodes) {
        if (names.count(node.name) && Apply(node, true)) ++changed;
    }
    return changed;
}

Totals CountTotals(const std::vector<CompiledLayer>& layers, const std::map<size_t, long long>& layer_parameters) {
    Totals totals;
    for (const auto& [index, count] : layer_parameters) {
        if (index >= layers.size()) continue;
        const CompiledLayer& layer = layers[index];
        const bool frozen = LayerNodeHoldsWeights(layer.type) && SettingOn(layer.parameters);
        (frozen ? totals.frozen : totals.trainable) += count;
    }
    return totals;
}

std::string TotalsText(const Totals& totals) {
    return "Trainable " + Grouped(totals.trainable) + " \xC2\xB7 Frozen " + Grouped(totals.frozen);
}

}  // namespace cyxwiz::frozen
