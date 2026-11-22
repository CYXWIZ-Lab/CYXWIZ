// The graph code export (core/code_export, TOFIX228 1.5a): GUI-free, the same
// text for the same graph, the PyTorch module each node becomes, and the
// refusals (empty graph, cycle, a layer the export cannot write).
#include "../src/core/code_export.h"
#include "../src/core/graph_node_factory.h"

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok   " : "  FAIL ") << what << "\n";
    if (!condition) ++g_failures;
}

bool Contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

struct Graph {
    std::vector<gui::MLNode> nodes;
    std::vector<gui::NodeLink> links;
    int next_node = 1;
    int next_pin = 1;
    int next_link = 1;

    int Add(gui::NodeType type, const std::string& name, std::map<std::string, std::string> params = {}) {
        gui::MLNode node = gui::CreateGraphNode(type, name, next_node, next_pin);
        for (auto& [key, value] : params) node.parameters[key] = value;
        nodes.push_back(node);
        return node.id;
    }
    const gui::MLNode& Node(int id) const {
        for (const auto& node : nodes)
            if (node.id == id) return node;
        throw std::runtime_error("no node " + std::to_string(id));
    }
    // From output `out` of `from` to input `in` of `to`.
    void Connect(int from, int to, size_t out = 0, size_t in = 0) {
        gui::NodeLink link;
        link.id = next_link++;
        link.from_node = from;
        link.from_pin = Node(from).outputs.at(out).id;
        link.to_node = to;
        link.to_pin = Node(to).inputs.at(in).id;
        links.push_back(link);
    }
};

// Data -> Dense 16 -> ReLU -> Dropout -> Dense 3 -> CrossEntropy -> AdamW.
Graph Mlp() {
    Graph g;
    const int data = g.Add(gui::NodeType::DatasetInput, "Data", {{"shape", "[8]"}});
    const int hidden = g.Add(gui::NodeType::Dense, "Hidden", {{"units", "16"}});
    const int relu = g.Add(gui::NodeType::ReLU, "ReLU");
    const int drop = g.Add(gui::NodeType::Dropout, "Dropout", {{"rate", "0.25"}});
    const int head = g.Add(gui::NodeType::Dense, "Head", {{"units", "3"}});
    const int loss = g.Add(gui::NodeType::CrossEntropyLoss, "Loss");
    const int opt = g.Add(gui::NodeType::AdamW, "AdamW",
                          {{"learning_rate", "0.002"}, {"weight_decay", "0.05"}, {"grad_clip_norm", "1.5"}});
    g.Connect(data, hidden);
    g.Connect(hidden, relu);
    g.Connect(relu, drop);
    g.Connect(drop, head);
    g.Connect(head, loss);
    g.Connect(loss, opt);
    return g;
}

void CheckPyTorch() {
    const Graph g = Mlp();
    cyxwiz::CodeExportOptions plain;
    plain.timestamp = false;
    const auto first = cyxwiz::ExportGraphCode(g.nodes, g.links, cyxwiz::CodeFramework::PyTorch, plain);
    const auto second = cyxwiz::ExportGraphCode(g.nodes, g.links, cyxwiz::CodeFramework::PyTorch, plain);
    Check(!first.code.empty(), "PyTorch: the MLP exports");
    Check(first.code == second.code, "PyTorch: unstamped, the same graph gives the same text");
    Check(Contains(first.code, "# Generated at: (not stamped)"), "PyTorch: the unstamped header says so");
    Check(Contains(first.code, "class GeneratedModel(nn.Module):"), "PyTorch: GeneratedModel");
    Check(Contains(first.code, "F.dropout(x, p=0.25, training=self.training)"), "PyTorch: Dropout's rate");
    Check(Contains(first.code, "optim.AdamW(") && Contains(first.code, "lr=0.002") &&
              Contains(first.code, "weight_decay=0.05"),
          "PyTorch: the optimizer node's AdamW, lr and decay");
    Check(Contains(first.code, "clip_grad_norm_(model.parameters(), 1.5)"), "PyTorch: the clip norm");

    // Each parameter layer is one module, named in model order.
    Check(first.modules.size() == 2, "PyTorch: two modules (the two Dense), got " +
                                         std::to_string(first.modules.size()));
    if (first.modules.size() == 2) {
        Check(g.Node(first.modules[0].node_id).name == "Hidden" && first.modules[0].name == "layer0",
              "PyTorch: Hidden is self.layer0");
        Check(g.Node(first.modules[1].node_id).name == "Head" && first.modules[1].name == "layer1",
              "PyTorch: Head is self.layer1");
        Check(Contains(first.code, "self.layer0 = ") && Contains(first.code, "self.layer1 = "),
              "PyTorch: both modules are in __init__");
    }

    const auto stamped = cyxwiz::ExportGraphCode(g.nodes, g.links, cyxwiz::CodeFramework::PyTorch);
    Check(!Contains(stamped.code, "(not stamped)") && Contains(stamped.code, "# Generated at: 20"),
          "PyTorch: by default the header has the local time");
}

void CheckOtherFrameworks() {
    const Graph g = Mlp();
    for (const auto& [framework, name, marker] :
         std::vector<std::tuple<cyxwiz::CodeFramework, std::string, std::string>>{
             {cyxwiz::CodeFramework::TensorFlow, "TensorFlow", "tensorflow"},
             {cyxwiz::CodeFramework::Keras, "Keras", "keras"},
             {cyxwiz::CodeFramework::PyCyxWiz, "PyCyxWiz", "pycyxwiz"}}) {
        const auto out = cyxwiz::ExportGraphCode(g.nodes, g.links, framework);
        Check(Contains(out.code, marker), name + ": exports (names " + marker + ")");
        Check(out.modules.empty(), name + ": no PyTorch module list");
    }
}

void CheckRefusals() {
    Check(cyxwiz::ExportGraphCode({}, {}, cyxwiz::CodeFramework::PyTorch).code.empty(), "an empty graph exports nothing");

    Graph cycle;
    const int a = cycle.Add(gui::NodeType::Dense, "A", {{"units", "4"}});
    const int b = cycle.Add(gui::NodeType::Dense, "B", {{"units", "4"}});
    cycle.Connect(a, b);
    cycle.Connect(b, a);
    Check(cyxwiz::ExportGraphCode(cycle.nodes, cycle.links, cyxwiz::CodeFramework::PyTorch).code.empty(),
          "a graph with a cycle exports nothing");

    Graph blocked = Mlp();
    const int conv3d = blocked.Add(gui::NodeType::Conv3D, "Volume");
    blocked.Connect(blocked.nodes.front().id, conv3d);
    const auto out = cyxwiz::ExportGraphCode(blocked.nodes, blocked.links, cyxwiz::CodeFramework::PyTorch);
    Check(Contains(out.code, "raise ValueError(") && Contains(out.code, "Volume") && Contains(out.code, "Conv3D"),
          "a layer the export cannot write: a script that raises with the node and the reason");
}

}  // namespace

int main() {
    std::cout << "test_code_export\n";
    try {
        CheckPyTorch();
        CheckOtherFrameworks();
        CheckRefusals();
    } catch (const std::exception& e) {
        std::cout << "  FAIL exception: " << e.what() << "\n";
        ++g_failures;
    }
    std::cout << (g_failures == 0 ? "PASS" : "FAIL") << " (" << g_failures << " failures)\n";
    return g_failures == 0 ? 0 : 1;
}
