#include <catch2/catch_test_macros.hpp>

#include "../../cyxwiz-engine/src/core/nas_evaluator.h"

#include <string>
#include <vector>

namespace {

gui::MLNode MakeNode(int id, gui::NodeType type) {
    gui::MLNode node{};
    node.id = id;
    node.type = type;
    return node;
}

gui::NodeLink MakeLink(int id, int from_node, int to_node) {
    gui::NodeLink link{};
    link.id = id;
    link.from_node = from_node;
    link.to_node = to_node;
    return link;
}

} // namespace

TEST_CASE("NAS validation requires both graph boundaries", "[nas][validation]") {
    const auto output_only = cyxwiz::NASEvaluator::ValidateArchitecture(
        {MakeNode(1, gui::NodeType::Output)}, {});
    CHECK_FALSE(output_only.first);
    CHECK(output_only.second == "Missing DatasetInput node");

    const auto input_only = cyxwiz::NASEvaluator::ValidateArchitecture(
        {MakeNode(1, gui::NodeType::DatasetInput)}, {});
    CHECK_FALSE(input_only.first);
    CHECK(input_only.second == "Missing Output node");
}

TEST_CASE("NAS validation accepts a connected input-to-output graph",
          "[nas][validation]") {
    const std::vector<gui::MLNode> nodes = {
        MakeNode(1, gui::NodeType::DatasetInput),
        MakeNode(2, gui::NodeType::Output),
    };
    const std::vector<gui::NodeLink> links = {MakeLink(1, 1, 2)};

    const auto result = cyxwiz::NASEvaluator::ValidateArchitecture(nodes, links);
    CHECK(result.first);
    CHECK(result.second.empty());
}
