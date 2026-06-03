#include "../src/core/graph_topology_utils.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

gui::MLNode Node(int id, gui::NodeType type, const std::string& name) {
    gui::MLNode node;
    node.id = id;
    node.type = type;
    node.name = name;
    return node;
}

gui::NodeLink Link(int id, int from_node, int to_node) {
    gui::NodeLink link;
    link.id = id;
    link.from_node = from_node;
    link.to_node = to_node;
    link.type = gui::LinkType::TensorFlow;
    return link;
}

} // namespace

int main() {
    std::vector<gui::MLNode> nodes = {
        Node(10, gui::NodeType::DataLoader, "stale loader"),
        Node(1, gui::NodeType::DataInput, "stale input"),
        Node(2, gui::NodeType::DataInput, "active input"),
        Node(3, gui::NodeType::DataSplit, "active split"),
        Node(4, gui::NodeType::DataLoader, "active loader"),
        Node(5, gui::NodeType::Dense, "model"),
    };

    std::vector<gui::NodeLink> links = {
        Link(100, 2, 3),
        Link(101, 3, 4),
        Link(102, 4, 5),
    };

    Check(!cyxwiz::HasOutgoingLink(1, links),
          "stale DataInput should not be connected");
    Check(cyxwiz::HasOutgoingLink(2, links),
          "active DataInput should be connected");

    auto reachable = cyxwiz::CollectReachableNodeIds(2, links);
    Check(reachable.count(2) == 1, "source should be reachable from itself");
    Check(reachable.count(3) == 1, "active DataSplit should be reachable");
    Check(reachable.count(4) == 1, "active DataLoader should be reachable");
    Check(reachable.count(10) == 0, "stale DataLoader should not be reachable");

    auto* split = cyxwiz::FindFirstReachableNodeOfType(
        nodes, reachable, gui::NodeType::DataSplit);
    auto* loader = cyxwiz::FindFirstReachableNodeOfType(
        nodes, reachable, gui::NodeType::DataLoader);

    Check(split != nullptr && split->id == 3,
          "should find active reachable DataSplit");
    Check(loader != nullptr && loader->id == 4,
          "should ignore stale first DataLoader and find active one");

    std::cout << "Graph topology utils passed\n";
    return 0;
}
