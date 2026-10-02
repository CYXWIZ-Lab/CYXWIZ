// Plot node result lane plan (TOFIX134 P2 step 2.1): run only the nodes above
// the plot, read a loaded Data Input directly, and say plainly when a node
// above runs only inside training.
#include "../src/core/plot/node_result_plan.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz;
using namespace cyxwiz::plot;
using gui::MLNode;
using gui::NodeLink;
using gui::NodeType;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

int next_pin = 1000;

MLNode Node(int id, NodeType type, const std::string& name, int inputs, int outputs) {
    MLNode n{};
    n.id = id;
    n.type = type;
    n.name = name;
    for (int i = 0; i < inputs; ++i) {
        gui::NodePin p{};
        p.id = next_pin++;
        p.is_input = true;
        n.inputs.push_back(p);
    }
    for (int i = 0; i < outputs; ++i) {
        gui::NodePin p{};
        p.id = next_pin++;
        n.outputs.push_back(p);
    }
    return n;
}

NodeLink Link(int id, const MLNode& from, int out, const MLNode& to, int in) {
    NodeLink l{};
    l.id = id;
    l.from_node = from.id;
    l.from_pin = from.outputs[static_cast<size_t>(out)].id;
    l.to_node = to.id;
    l.to_pin = to.inputs[static_cast<size_t>(in)].id;
    return l;
}
}  // namespace

int main() {
    // Data Input -> Filter Rows -> Plot, and Filter -> Dense (training) beside.
    MLNode input = Node(1, NodeType::DataInput, "Data Input", 0, 1);
    input.parameters["file_path"] = "mnist_784.csv";
    input.parameters["dataset_name"] = "mnist_784";
    MLNode filter = Node(2, NodeType::FilterRows, "Filter Rows", 1, 1);
    filter.parameters["condition"] = "class = 7";
    MLNode dense = Node(3, NodeType::Dense, "Dense", 1, 1);
    MLNode plot = Node(9, NodeType::DescribeStats, "Plot", 1, 0);  // stands in for the Plot node
    std::vector<MLNode> nodes = {input, filter, dense, plot};
    std::vector<NodeLink> links = {Link(1, input, 0, filter, 0), Link(2, filter, 0, dense, 0), Link(3, filter, 0, plot, 0)};
    const auto none_loaded = [](const std::string&) { return false; };

    NodeResultPlan p = PlanNodeResult(9, nodes, links, none_loaded);
    Check(p.state == NodeResultPlan::State::Run, "a runnable chain is run");
    Check(p.feeder_id == 2 && p.feeder_name == "Filter Rows", "the feeder is the node wired in");
    Check(p.run_node_count == 2, "only the two nodes above (not Dense, not the plot)");
    const auto j = nlohmann::json::parse(p.pipeline_json);
    Check(j["nodes"].size() == 2 && j["nodes"][0]["type"] == "DataInput" && j["nodes"][1]["type"] == "FilterRows",
          "pipeline types");
    Check(j["nodes"][1]["parameters"]["condition"] == "class = 7", "parameters kept");
    Check(j["links"].size() == 1 && j["links"][0]["start_node"] == 1 && j["links"][0]["end_node"] == 2 &&
              j["links"][0]["start_pin_index"] == 0 && j["links"][0]["end_pin_index"] == 0,
          "only the link inside the closure, with pin indices");

    // Staleness: an edit above changes the fingerprint; one beside does not.
    const std::string before = p.fingerprint;
    nodes[2].parameters["units"] = "256";  // Dense, not above the plot
    Check(PlanNodeResult(9, nodes, links, none_loaded).fingerprint == before, "an edit beside keeps the result");
    nodes[1].parameters["condition"] = "class = 3";
    Check(PlanNodeResult(9, nodes, links, none_loaded).fingerprint != before, "an edit above makes it out of date");

    // A loaded Data Input wired straight in is read directly.
    std::vector<NodeLink> direct = {Link(4, input, 0, plot, 0)};
    p = PlanNodeResult(9, nodes, direct, [](const std::string& n) { return n == "mnist_784"; });
    Check(p.state == NodeResultPlan::State::Loaded && p.dataset_name == "mnist_784", "loaded dataset read directly");
    p = PlanNodeResult(9, nodes, direct, none_loaded);
    Check(p.state == NodeResultPlan::State::Run && p.run_node_count == 1, "not loaded: the Data Input runs");

    // A node that runs only inside training blocks, and its input is offered.
    MLNode split = Node(4, NodeType::DataSplit, "Train/Val/Test Split", 1, 3);
    nodes.push_back(split);
    std::vector<NodeLink> via_split = {Link(5, input, 0, split, 0), Link(6, split, 0, plot, 0)};
    p = PlanNodeResult(9, nodes, via_split, none_loaded);
    Check(p.state == NodeResultPlan::State::Unavailable, "a split above cannot run");
    Check(p.reason == "Train/Val/Test Split runs only inside training, so its output cannot be plotted yet.",
          "the reason in words: " + p.reason);
    Check(p.alternative_id == 1 && p.alternative_name == "Data Input", "offer to plot its input");

    // Nothing wired in.
    p = PlanNodeResult(9, nodes, {}, none_loaded);
    Check(p.state == NodeResultPlan::State::NotConnected && p.feeder_id == -1, "not connected");
    std::cout << "node result plan: closure only, links and pins, staleness, loaded, unavailable, not connected. OK\n";
    return 0;
}
