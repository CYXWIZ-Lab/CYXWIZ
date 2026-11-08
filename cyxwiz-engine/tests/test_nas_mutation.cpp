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

    // The head (Dense 2, id 4: its units are the class count the loss
    // expects) is never removed, resized or swapped; a swap changes the type of
    // the other layer and keeps the graph whole; added layers fit beside Dense.
    for (unsigned seed = 1; seed <= 40; ++seed) {
        const std::string at = ", seed " + std::to_string(seed);
        const auto [removed, removed_links] = NASEvaluator::MutateArchitecture(nodes, links, MutationType::RemoveLayer, seed);
        bool head_kept = false;
        for (const auto& n : removed) head_kept = head_kept || n.id == 4;
        Check(head_kept, "remove layer keeps the head" + at);

        const auto [resized, resized_links] = NASEvaluator::MutateArchitecture(nodes, links, MutationType::ChangeUnits, seed);
        for (const auto& n : resized)
            if (n.id == 4) Check(n.parameters.at("units") == "2", "change units leaves the head's units" + at);

        const auto [swapped, swapped_links] = NASEvaluator::MutateArchitecture(nodes, links, MutationType::SwapLayer, seed);
        const std::string swap_problem = Problem(swapped, swapped_links);
        Check(swap_problem.empty(), "swap layer" + at + ": " + swap_problem);
        Check(swapped[1].type != NodeType::Dense, "swap layer changes Dense 1's type" + at);
        Check(swapped[3].type == NodeType::Dense, "swap layer leaves the head" + at);

        const auto [added, added_links] = NASEvaluator::MutateArchitecture(nodes, links, MutationType::AddLayer, seed);
        for (const auto& n : added)
            Check(n.type == NodeType::Dense || n.type == NodeType::BatchNorm || n.type == NodeType::Dropout ||
                      n.type == NodeType::ReLU || n.type == NodeType::DatasetInput || n.type == NodeType::Output,
                  "add layer beside Dense adds a feature layer, got " + n.name + at);
    }

    // The score counts the model's layers, not the data nodes or the output,
    // and names each type.
    NASSearchConfig config;
    const auto score = NASEvaluator::ScoreArchitecture(nodes, links, {1, 784}, config);
    Check(score.layer_count == 3, "layer count is the 3 model layers, got " + std::to_string(score.layer_count));
    Check(score.architecture_summary.find("Layers: 2x Dense, 1x ReLU") != std::string::npos,
          "summary names the layers: " + score.architecture_summary);
    Check(score.connectivity_score == 0.0, "a plain chain has no skip connections");

    // A residual block built with an Add node that joins two paths counts as a
    // skip connection.
    auto [res_nodes, res_links] = Chain();
    res_nodes.push_back(Node(6, NodeType::Add, "Add"));
    res_links.push_back({5, 2, 21, 6, 60});  // Dense 1 -> Add
    res_links.push_back({6, 4, 41, 6, 60});  // Dense 2 -> Add
    const auto residual = NASEvaluator::ScoreArchitecture(res_nodes, res_links, {1, 784}, config);
    Check(residual.connectivity_score > 0.0, "an Add joining two paths counts as a skip connection");

    // Parameters: each layer sized by what reaches it, not by the input.
    // 784 -> 128 -> 64 -> 10: 100,480 + 8,256 + 650 (what the compiled model
    // holds). Before, every Dense counted 784 inputs: 158,570.
    {
        std::vector<MLNode> mlp = {Node(1, NodeType::DatasetInput, "Input"), Node(2, NodeType::Dense, "Dense 1"),
                                   Node(3, NodeType::ReLU, "ReLU 1"), Node(4, NodeType::Dense, "Dense 2"),
                                   Node(5, NodeType::ReLU, "ReLU 2"), Node(6, NodeType::Dense, "Dense 3"),
                                   Node(7, NodeType::Output, "Output")};
        mlp[1].parameters["units"] = "128";
        mlp[3].parameters["units"] = "64";
        mlp[5].parameters["units"] = "10";
        std::vector<NodeLink> mlp_links;
        for (int i = 1; i < 7; ++i) mlp_links.push_back({i, i, i * 10 + 1, i + 1, (i + 1) * 10});
        const auto counted = NASEvaluator::ScoreArchitecture(mlp, mlp_links, {1, 784}, config);
        Check(counted.total_params == 109386, "MLP parameters 109,386, got " + std::to_string(counted.total_params));

        // A 'same' convolution keeps 28 x 28: Conv2D 1 -> 32 (320), BatchNorm
        // (64 learnable), Flatten 25,088 -> Dense 10 (250,890).
        std::vector<MLNode> cnn = {Node(1, NodeType::DatasetInput, "Input"), Node(2, NodeType::Conv2D, "Conv"),
                                   Node(3, NodeType::BatchNorm, "BN"), Node(4, NodeType::Flatten, "Flatten"),
                                   Node(5, NodeType::Dense, "Dense"), Node(6, NodeType::Output, "Output")};
        cnn[1].parameters = {{"filters", "32"}, {"kernel_size", "3"}, {"padding", "same"}};
        cnn[4].parameters["units"] = "10";
        std::vector<NodeLink> cnn_links;
        for (int i = 1; i < 6; ++i) cnn_links.push_back({i, i, i * 10 + 1, i + 1, (i + 1) * 10});
        const auto conv = NASEvaluator::ScoreArchitecture(cnn, cnn_links, {1, 1, 28, 28}, config);
        Check(conv.total_params == 320 + 64 + 250890, "CNN parameters 251,274, got " + std::to_string(conv.total_params));
    }

    if (g_failures) {
        std::printf("%d NAS mutation checks failed\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("NAS mutation passed\n");
    return EXIT_SUCCESS;
}
