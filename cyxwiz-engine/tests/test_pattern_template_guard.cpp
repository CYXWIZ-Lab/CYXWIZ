#include "../src/gui/patterns/pattern_library.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

std::filesystem::path WritePattern(const std::string& id, const std::string& type) {
    const auto dir = std::filesystem::temp_directory_path() / "cyxwiz_pattern_guard";
    std::filesystem::create_directories(dir);

    const auto path = dir / (id + ".cyxgraph");
    std::ofstream out(path, std::ios::binary);
    Check(out.is_open(), "could not write pattern file");
    out << "{\n"
        << "  \"id\": \"" << id << "\",\n"
        << "  \"name\": \"" << id << "\",\n"
        << "  \"category\": \"Basic\",\n"
        << "  \"template\": {\n"
        << "    \"nodes\": [\n"
        << "      {\"id\": \"n1\", \"type\": \"" << type
        << "\", \"name\": \"" << type << "\", \"pos_x\": 0, \"pos_y\": 0}\n"
        << "    ],\n"
        << "    \"links\": []\n"
        << "  }\n"
        << "}\n";
    return path;
}

gui::MLNode MinimalNode(gui::NodeType type, const std::string& name) {
    gui::MLNode node;
    node.type = type;
    node.name = name;

    gui::NodePin input;
    input.id = 1;
    input.type = gui::PinType::Tensor;
    input.name = "Input";
    input.is_input = true;
    node.inputs.push_back(input);

    gui::NodePin output;
    output.id = 2;
    output.type = gui::PinType::Tensor;
    output.name = "Output";
    output.is_input = false;
    node.outputs.push_back(output);

    return node;
}

} // namespace

int main() {
    auto& library = gui::patterns::PatternLibrary::Instance();
    Check(library.LoadPatternFromFile(WritePattern("guard_dense", "Dense").string()),
          "failed to load implemented-node pattern");
    Check(library.LoadPatternFromFile(WritePattern("guard_add", "Add").string()),
          "failed to load graph-runtime merge pattern");
    Check(library.LoadPatternFromFile(WritePattern("guard_concat", "Concatenate").string()),
          "failed to load deferred merge pattern");
    Check(library.LoadPatternFromFile(WritePattern("guard_tensor_dot", "TensorDot").string()),
          "failed to load graph-runtime dot pattern");
    Check(library.LoadPatternFromFile(WritePattern("guard_batch_matmul", "TensorBatchMatMul").string()),
          "failed to load template-node pattern");
    Check(library.LoadPatternFromFile(WritePattern("guard_typo", "DefinitelyNotANode").string()),
          "failed to load unknown-node pattern");

    std::vector<gui::MLNode> nodes;
    std::vector<gui::NodeLink> links;
    int next_node_id = 100;
    int next_link_id = 1000;
    int creator_calls = 0;
    auto creator = [&creator_calls](gui::NodeType type, const std::string& name) {
        ++creator_calls;
        return MinimalNode(type, name);
    };

    Check(library.InstantiatePatternWithCreator(
              "guard_dense", {}, nodes, links, next_node_id, next_link_id, ImVec2(0, 0), creator),
          "implemented node pattern should instantiate");
    Check(creator_calls == 1, "implemented pattern should call node creator once");
    Check(nodes.size() == 1 && nodes.front().type == gui::NodeType::Dense,
          "implemented pattern created wrong node type");

    nodes.clear();
    links.clear();
    Check(library.InstantiatePatternWithCreator(
              "guard_add", {}, nodes, links, next_node_id, next_link_id, ImVec2(0, 0), creator),
          "implemented graph-runtime Add pattern should instantiate");
    Check(creator_calls == 2, "Add pattern should call node creator once");
    Check(nodes.size() == 1 && nodes.front().type == gui::NodeType::Add,
          "Add pattern created wrong node type");

    nodes.clear();
    links.clear();
    Check(library.InstantiatePatternWithCreator(
              "guard_concat", {}, nodes, links, next_node_id, next_link_id, ImVec2(0, 0), creator),
          "implemented Concatenate pattern should instantiate");
    Check(creator_calls == 3, "Concatenate pattern should call node creator once");
    Check(nodes.size() == 1 && nodes.front().type == gui::NodeType::Concatenate,
          "Concatenate pattern created wrong node type");

    nodes.clear();
    links.clear();
    Check(library.InstantiatePatternWithCreator(
              "guard_tensor_dot", {}, nodes, links, next_node_id, next_link_id, ImVec2(0, 0), creator),
          "implemented TensorDot pattern should instantiate");
    Check(creator_calls == 4, "TensorDot pattern should call node creator once");
    Check(nodes.size() == 1 && nodes.front().type == gui::NodeType::TensorDot,
          "TensorDot pattern created wrong node type");

    nodes.clear();
    links.clear();
    Check(!library.InstantiatePatternWithCreator(
              "guard_batch_matmul", {}, nodes, links, next_node_id, next_link_id, ImVec2(0, 0), creator),
          "template TensorBatchMatMul pattern should be rejected");
    Check(nodes.empty() && links.empty(), "template rejection should leave no partial graph");
    Check(creator_calls == 4, "template rejection should not call node creator");

    nodes.clear();
    links.clear();
    Check(!library.InstantiatePatternWithCreator(
              "guard_typo", {}, nodes, links, next_node_id, next_link_id, ImVec2(0, 0), creator),
          "unknown node pattern should be rejected");
    Check(nodes.empty() && links.empty(), "unknown rejection should leave no partial graph");
    Check(creator_calls == 4, "unknown rejection should not call node creator");

    int next_pin_id = 2000;
    Check(!library.InstantiatePattern(
              "guard_batch_matmul", {}, nodes, links, next_node_id, next_pin_id, next_link_id, ImVec2(0, 0)),
          "legacy instantiation should also reject template nodes");

    std::cout << "Pattern template guard passed\n";
    return 0;
}
