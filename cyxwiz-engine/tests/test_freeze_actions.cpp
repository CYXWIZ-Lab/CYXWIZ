// The canvas Freeze actions and the Trainable / Frozen count (TOFIX178 1.9e)
// on a small graph: stem convolution, BatchNorm, ReLU, a second convolution,
// a head.
#include "core/freeze_actions.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

int checks = 0;
int failures = 0;

void Check(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << "\n";
    }
}

gui::MLNode Node(int id, gui::NodeType type, const std::string& name) {
    gui::MLNode node;
    node.id = id;
    node.type = type;
    node.name = name;
    return node;
}

cyxwiz::CompiledLayer Layer(int node_id, gui::NodeType type, const std::string& name) {
    cyxwiz::CompiledLayer layer;
    layer.node_id = node_id;
    layer.type = type;
    layer.name = name;
    return layer;
}

std::string Frozen(const std::vector<gui::MLNode>& nodes) {
    std::string names;
    for (const auto& node : nodes) {
        if (cyxwiz::frozen::IsFrozen(node)) names += (names.empty() ? "" : ",") + node.name;
    }
    return names;
}

}  // namespace

int main() {
    namespace frozen = cyxwiz::frozen;
    std::vector<gui::MLNode> nodes = {Node(1, gui::NodeType::Conv2D, "conv1"), Node(2, gui::NodeType::BatchNorm, "bn1"),
                                      Node(3, gui::NodeType::ReLU, "relu"), Node(4, gui::NodeType::Conv2D, "conv2"),
                                      Node(5, gui::NodeType::Dense, "fc")};
    Check(Frozen(nodes).empty(), "nothing is frozen on a new graph");

    // Selected nodes: only those with weights change; an activation never does.
    Check(frozen::SetFrozen(nodes, {1, 2, 3}, true) == 2 && Frozen(nodes) == "conv1,bn1",
          "freezing a selection freezes its layers with weights: " + Frozen(nodes));
    Check(nodes[2].parameters.count("frozen") == 0, "a node without weights gets no setting");
    Check(frozen::SetFrozen(nodes, {1, 2, 3}, true) == 0, "freezing again changes nothing");
    Check(frozen::SetFrozen(nodes, {2}, false) == 1 && Frozen(nodes) == "conv1", "unfreezing one node");
    Check(frozen::UnfreezeAll(nodes) == 1 && Frozen(nodes).empty(), "unfreeze all");

    // The model's order comes from the compiled layers, not from the node list.
    const std::vector<cyxwiz::CompiledLayer> layers = {
        Layer(1, gui::NodeType::Conv2D, "conv1"), Layer(2, gui::NodeType::BatchNorm, "bn1"),
        Layer(3, gui::NodeType::ReLU, "relu"), Layer(4, gui::NodeType::Conv2D, "conv2"),
        Layer(5, gui::NodeType::Dense, "fc")};
    const auto order = frozen::ModelOrder(layers);
    Check(order == std::vector<int>({1, 2, 4, 5}), "the model order lists the layers with weights");
    Check(frozen::FreezeAllButLast(nodes, order, 1) == 3 && Frozen(nodes) == "conv1,bn1,conv2",
          "all but the head: " + Frozen(nodes));
    Check(frozen::FreezeAllButLast(nodes, order, 2) == 1 && Frozen(nodes) == "conv1,bn1",
          "all but the last two unfreezes the layer that is now kept: " + Frozen(nodes));
    Check(frozen::FreezeAllButLast(nodes, order, 10) == 2 && Frozen(nodes).empty(),
          "keeping more layers than there are freezes none");
    Check(frozen::FreezeAllButLast(nodes, order, 0) == 4 && Frozen(nodes) == "conv1,bn1,conv2,fc", "keeping none freezes all");
    frozen::UnfreezeAll(nodes);

    // The layers a weights file filled, by name.
    Check(frozen::FreezeNamed(nodes, {"conv1", "bn1", "relu", "missing"}) == 2 && Frozen(nodes) == "conv1,bn1",
          "freezing by name: " + Frozen(nodes));

    // The count: learnable weights per compiled layer, split by the layer's setting.
    auto counted = layers;
    counted[0].parameters["frozen"] = "true";
    counted[1].parameters["frozen"] = "true";
    counted[2].parameters["frozen"] = "true";  // an activation's stray setting does not count as frozen
    const std::map<size_t, long long> per_layer = {{0, 9408}, {1, 128}, {2, 0}, {3, 36864}, {4, 1026}};
    const auto totals = frozen::CountTotals(counted, per_layer);
    Check(totals.frozen == 9536 && totals.trainable == 37890, "trainable and frozen weights are counted apart");
    Check(frozen::TotalsText(totals) == "Trainable 37,890 \xC2\xB7 Frozen 9,536", "the count's words: " + frozen::TotalsText(totals));
    Check(frozen::TotalsText({1026, 11176512}) == "Trainable 1,026 \xC2\xB7 Frozen 11,176,512", "digits are grouped");

    if (failures) {
        std::cerr << failures << " of " << checks << " checks failed\n";
        return 1;
    }
    std::cout << "Freeze actions passed: " << checks << " checks\n";
    return 0;
}
