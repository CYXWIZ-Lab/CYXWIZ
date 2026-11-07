// NAS mutations keep the graph connected: adding a layer splices it into the
// links that fed its target, removing one joins what fed it to what it fed.
// Before, both left the graph cut (a layer with no links / a broken chain).

#include "core/nas_evaluator.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace cyxwiz;
using gui::MLNode;
using gui::NodeLink;
using gui::NodeType;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

MLNode Node(int id, NodeType type, const char* name) {
    MLNode node;
    node.id = id;
    node.type = type;
    node.name = name;
    gui::NodePin in;
    in.id = id * 10;
    in.is_input = true;
    gui::NodePin out;
    out.id = id * 10 + 1;
    node.inputs.push_back(in);
    node.outputs.push_back(out);
    return node;
}

// Input -> Dense -> ReLU -> Dense -> Output, linked output pin to input pin.
std::pair<std::vector<MLNode>, std::vector<NodeLink>> Chain() {
    std::vector<MLNode> nodes = {Node(1, NodeType::DatasetInput, "Input"), Node(2, NodeType::Dense, "Dense 1"),
                                 Node(3, NodeType::ReLU, "ReLU"), Node(4, NodeType::Dense, "Dense 2"),
                                 Node(5, NodeType::Output, "Output")};
    nodes[1].parameters["units"] = "64";
    nodes[3].parameters["units"] = "2";
    std::vector<NodeLink> links;
    for (int i = 1; i < 5; ++i) links.push_back({i, i, i * 10 + 1, i + 1, (i + 1) * 10});
    return {nodes, links};
}

// Every link joins existing pins of the right direction, ids are unique, and
// the output is reachable from the input through every node.
std::string Problem(const std::vector<MLNode>& nodes, const std::vector<NodeLink>& links) {
    std::map<int, const MLNode*> by_id;
    for (const auto& n : nodes) by_id[n.id] = &n;
    std::set<int> link_ids;
    for (const auto& l : links) {
        if (!link_ids.insert(l.id).second) return "duplicate link id " + std::to_string(l.id);
        const auto from = by_id.find(l.from_node);
        const auto to = by_id.find(l.to_node);
        if (from == by_id.end() || to == by_id.end()) return "link to a missing node";
        bool out_ok = false, in_ok = false;
        for (const auto& p : from->second->outputs) out_ok = out_ok || p.id == l.from_pin;
        for (const auto& p : to->second->inputs) in_ok = in_ok || p.id == l.to_pin;
        if (!out_ok || !in_ok) return "link " + std::to_string(l.id) + " names a pin its node lacks";
    }
    std::set<int> reached = {1};
    for (bool grew = true; grew;) {
        grew = false;
        for (const auto& l : links)
            if (reached.count(l.from_node) && reached.insert(l.to_node).second) grew = true;
    }
    for (const auto& n : nodes)
        if (!reached.count(n.id)) return "node " + n.name + " is cut off from the input";
    return {};
}

}  // namespace

int main() {
    const auto [nodes, links] = Chain();
    Check(Problem(nodes, links).empty(), "the starting chain is whole");

    for (unsigned seed = 1; seed <= 40; ++seed) {
        const auto [added_nodes, added_links] =
            NASEvaluator::MutateArchitecture(nodes, links, MutationType::AddLayer, seed);
        const std::string add_problem = Problem(added_nodes, added_links);
        Check(add_problem.empty(), "add layer, seed " + std::to_string(seed) + ": " + add_problem);
        Check(added_nodes.size() == nodes.size() + 1, "add layer adds one node, seed " + std::to_string(seed));
        Check(added_links.size() == links.size() + 1, "add layer adds one link, seed " + std::to_string(seed));

        const auto [removed_nodes, removed_links] =
            NASEvaluator::MutateArchitecture(nodes, links, MutationType::RemoveLayer, seed);
        const std::string remove_problem = Problem(removed_nodes, removed_links);
        Check(remove_problem.empty(), "remove layer, seed " + std::to_string(seed) + ": " + remove_problem);
        Check(removed_nodes.size() == nodes.size() - 1, "remove layer removes one node, seed " + std::to_string(seed));
        Check(removed_links.size() == links.size() - 1, "remove layer leaves one link fewer, seed " + std::to_string(seed));
    }

    if (g_failures) {
        std::printf("%d NAS mutation checks failed\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("NAS mutation passed\n");
    return EXIT_SUCCESS;
}
