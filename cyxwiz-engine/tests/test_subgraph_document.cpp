#include "gui/subgraph_document.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <functional>
using namespace gui;
using namespace gui::detail;
using json = nlohmann::json;
namespace {
int checks = 0;
void Check(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void Reject(const std::function<void()>& operation, const char* message) {
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    Check(rejected, message);
}
NodePin Pin(int id, bool input) {
    NodePin pin{};
    pin.id = id; pin.type = PinType::Dataset; pin.name = input ? "Data" : "Result"; pin.is_input = input;
    return pin;
}
MLNode Node(int id, NodeType type, int input, int output) {
    MLNode node{};
    node.id = id; node.type = type; node.category = NodeCategory::DataTransform;
    node.name = "Node " + std::to_string(id);
    node.description = "document fixture";
    node.parameters["text"] = "preserved";
    if (input) node.inputs.push_back(Pin(input, true));
    if (output) node.outputs.push_back(Pin(output, false));
    return node;
}
struct Fixture {
    MLNode source = Node(1, NodeType::DataInput, 0, 11);
    MLNode first = Node(2, NodeType::TextCleanNode, 21, 22);
    MLNode second = Node(3, NodeType::TextCleanNode, 31, 32);
    MLNode sink = Node(4, NodeType::ExportParquet, 41, 0);
    MLNode wrapper = Node(5, NodeType::Subgraph, 51, 52);
    NodeLink internal{2, 2, 22, 3, 31};
    SubgraphData group{5, {first, second}, {internal}, {21}, {32}, false};
    std::vector<MLNode> nodes{source, sink, wrapper};
    std::vector<NodeLink> links{{1, 1, 11, 5, 51}, {3, 5, 52, 4, 41}};
    json Save() { json doc; WriteEditorGraphContent(doc, nodes, links, {group}, {{2,{123,456}}}); return doc; }
};

// Exercise the persistence boundary with the real Engine node factory and fresh IDs.
void ReadFresh(const json& doc) {
    const auto flat = FlattenSubgraphDocument(doc);
    Check(flat.at("nodes").size() == 5, "flatten contains every node exactly once");
    Check(flat.at("links").size() == 3, "flatten includes internal link");
    std::vector<MLNode> nodes;
    int next_pin = 1000;
    int next_node = 1;
    for (const auto& entry : flat.at("nodes")) {
        auto node = NodeEditor::CreateNodeWithIds(static_cast<NodeType>(entry.at("type").get<int>()),
            entry.at("name"), next_node, next_pin);
        node.id = entry.at("id");
        node.parameters = entry.at("parameters").get<std::map<std::string,std::string>>();
        nodes.push_back(std::move(node));
    }
    RestoreSubgraphPins(doc, nodes, next_pin);
    Check(nodes[2].inputs[0].type == PinType::Dataset, "wrapper retains Dataset input type");
    Check(nodes[2].outputs[0].type == PinType::Dataset, "wrapper retains Dataset output type");
    std::vector<NodeLink> links;
    auto find = [&](int id) -> const MLNode& { for (const auto& n : nodes) if (n.id == id) return n; throw std::runtime_error("missing fixture node"); };
    for (const auto& entry : flat.at("links")) {
        const auto& from = find(entry.at("from_node")); const auto& to = find(entry.at("to_node"));
        links.push_back({entry.at("id"), from.id, from.outputs.at(entry.at("from_pin_index")).id,
            to.id, to.inputs.at(entry.at("to_pin_index")).id});
    }
    const auto restored = RestoreSubgraphContents(doc, nodes, links);
    Check(restored.size() == 1 && restored[0].internal_nodes.size() == 2, "restored contents");
    Check(restored[0].internal_links.size() == 1, "restored internal link");
    Check(restored[0].input_pin_mappings[0] == restored[0].internal_nodes[0].inputs[0].id, "input mapping uses regenerated ID");
    Check(restored[0].output_pin_mappings[0] == restored[0].internal_nodes[1].outputs[0].id, "output mapping uses regenerated ID");
    Check(restored[0].internal_links[0].from_pin == restored[0].internal_nodes[0].outputs[0].id, "link uses regenerated ID");
    Check(nodes.size() == (restored[0].expanded ? 5u : 3u), "correct visible nodes");
    Check(links.size() == (restored[0].expanded ? 3u : 2u), "correct visible links");
}
}
int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--read") {
            std::ifstream file(argv[2]); json doc; file >> doc; ReadFresh(doc);
        } else {
            Fixture f;
            const json original = f.Save();
            Check(original["nodes"].size() == 3, "collapsed root contains wrapper and external nodes");
            Check(original["subgraphs"][0]["nodes"][0]["pos_x"] == 123, "internal position preserved");
            Check(original["subgraphs"][0]["nodes"][0]["parameters"]["text"] == "preserved", "parameters preserved");
            ReadFresh(original);
            f.group.expanded = true;
            f.nodes.push_back(f.first); f.nodes.push_back(f.second); f.links.push_back(f.internal);
            f.nodes[3].parameters["text"] = "edited while expanded";
            const json expanded = f.Save();
            Check(expanded["nodes"].size() == 3, "expanded save has no duplicate root internals");
            Check(expanded["subgraphs"][0]["nodes"][0]["parameters"]["text"] == "edited while expanded", "save captures live edit");
            ReadFresh(expanded);
            CaptureExpandedSubgraph(f.group, f.nodes, f.links);
            Check(f.group.internal_nodes[0].parameters.at("text") == "edited while expanded", "collapse captures live edit");
            auto bad = original; bad.erase("subgraphs"); bad.erase("subgraph_contract_version");
            Reject([&]{ FlattenSubgraphDocument(bad); }, "legacy wrapper-only graph rejected");
            bad = original; bad["subgraph_contract_version"] = 9;
            Reject([&]{ FlattenSubgraphDocument(bad); }, "future version rejected");
            bad = original; bad["subgraph_contract_version"] = 1.5;
            Reject([&]{ FlattenSubgraphDocument(bad); }, "fractional version rejected");
            bad = original; bad["subgraphs"][0]["inputs"][0]["pin_index"] = 0.5;
            Reject([&]{ FlattenSubgraphDocument(bad); }, "fractional pin index rejected");
            bad = original; bad["subgraphs"][0]["node_id"] = 900;
            Reject([&]{ FlattenSubgraphDocument(bad); }, "missing wrapper rejected");
            bad = original; bad["subgraphs"].push_back(bad["subgraphs"][0]);
            Reject([&]{ FlattenSubgraphDocument(bad); }, "duplicate owner rejected");
            bad = original; bad["subgraphs"][0]["nodes"][0]["id"] = 1;
            Reject([&]{ FlattenSubgraphDocument(bad); }, "duplicate node rejected");
            bad = original; bad["subgraphs"][0]["links"][0]["id"] = 1;
            Reject([&]{ FlattenSubgraphDocument(bad); }, "duplicate link rejected");
            bad = original; bad["subgraphs"][0]["inputs"][0]["node_id"] = 1;
            Reject([&]{ FlattenSubgraphDocument(bad); }, "foreign boundary rejected");
            bad = original; bad["subgraphs"][0]["nodes"][0]["type"] = static_cast<int>(NodeType::Subgraph);
            Reject([&]{ FlattenSubgraphDocument(bad); }, "nested subgraph rejected");
            bad = original; bad["links"][0]["to_node"] = 2;
            Reject([&]{ FlattenSubgraphDocument(bad); }, "outer-to-hidden edge rejected");
            bad = original; bad["subgraphs"][0]["outputs"][0]["pin_index"] = 17;
            auto nodes = std::vector<MLNode>{f.source, f.wrapper, f.first, f.second, f.sink}; int next = 200;
            Reject([&]{ RestoreSubgraphPins(bad, nodes, next); }, "out of range boundary rejected");
            auto absent_link = f.links; absent_link.pop_back();
            Reject([&]{ auto all=f.nodes; RestoreSubgraphContents(original, all, absent_link); }, "lost internal link rejected");
            f.links.push_back({99, 2, 22, 4, 41});
            Reject([&]{ f.Save(); }, "new cross-boundary edge rejected");
            auto ordinary = json{{"nodes", json::array()}, {"links", json::array()}};
            Check(FlattenSubgraphDocument(ordinary) == ordinary, "ordinary graph unchanged");
            if (argc == 3 && std::string(argv[1]) == "--write") {
                std::ofstream file(argv[2]); file << original.dump(2); if (!file) throw std::runtime_error("cannot write fixture");
            }
        }
        std::cout << "PASS: " << checks << " subgraph document checks\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
