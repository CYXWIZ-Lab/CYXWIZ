// Word + POS fusion presentation (TOFIX112): the verdict the compiler and the
// Properties panel share, on the real NER example graph.
#include "../src/core/graph_document.h"
#include "../src/core/sequence_fusion_presentation.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::filesystem::path FindRepoRoot() {
    auto dir = std::filesystem::current_path();
    while (!dir.empty()) {
        if (std::filesystem::exists(dir / "examples" / "cyxgraph" / "NER")) return dir;
        const auto parent = dir.parent_path();
        if (parent == dir) break;
        dir = parent;
    }
    return std::filesystem::current_path();
}

cyxwiz::GraphDocument LoadNerGraph() {
    const auto path = FindRepoRoot() / "examples" / "cyxgraph" / "NER" / "ner_bilstm_sequence_tagger.cyxgraph";
    std::ifstream in(path, std::ios::binary);
    Check(in.is_open(), "could not open " + path.string());
    std::stringstream text;
    text << in.rdbuf();
    const auto root = nlohmann::json::parse(text.str());
    cyxwiz::GraphDocumentLoadOptions options;
    options.strict_links = true;
    cyxwiz::GraphDocument document;
    std::string error;
    Check(cyxwiz::BuildGraphDocument(root, root, options, document, error), "NER graph should load: " + error);
    return document;
}

int IdNamed(const std::vector<gui::MLNode>& nodes, const std::string& name) {
    for (const auto& node : nodes) {
        if (node.name == name) return node.id;
    }
    Check(false, "missing node " + name);
    return -1;
}

gui::MLNode& NodeWithId(std::vector<gui::MLNode>& nodes, int id) {
    for (auto& node : nodes) {
        if (node.id == id) return node;
    }
    Check(false, "missing node id " + std::to_string(id));
    return nodes.front();
}

std::string ValueOf(const std::vector<std::pair<std::string, std::string>>& rows, const std::string& key) {
    for (const auto& [k, v] : rows) {
        if (k == key) return v;
    }
    return {};
}

void CheckContract(const cyxwiz::GraphDocument& graph) {
    const auto& nodes = graph.nodes;
    const auto* data = cyxwiz::FindUpstreamDataInput(nodes, graph.links, IdNamed(nodes, "Concat Word + POS"));
    Check(data && data->name == "NER Sentence CSV", "the Concatenate's upstream Data Input is the NER CSV");
    const auto contract = cyxwiz::ReadSequenceTaggingContract(*data);
    Check(contract.is_sequence, "sequence_text is a token-tagging category");
    Check(contract.token_column == "tokens" && contract.pos_column == "pos_tags" &&
              contract.tag_column == "ner_tags" && contract.sentence_id_column == "sentence_id",
          "the contract reads the canonical column keys");
    Check(contract.max_sequence_length == 96 && contract.create_attention_mask,
          "the contract reads max length and attention mask");

    gui::MLNode legacy;
    legacy.parameters = {{"file_category", "ner"}, {"token_sequence_column", "w"}, {"pos_sequence_column", "p"}};
    const auto aliased = cyxwiz::ReadSequenceTaggingContract(legacy);
    Check(aliased.is_sequence && aliased.token_column == "w" && aliased.pos_column == "p",
          "older alias keys are still read");
    Check(!cyxwiz::IsSequenceTaggingCategory("tabular"), "tabular is not token tagging");
}

void CheckCompilingCard(const cyxwiz::GraphDocument& graph) {
    const auto& nodes = graph.nodes;
    const auto card = cyxwiz::BuildSequenceFusionCard(nodes, graph.links, IdNamed(nodes, "Concat Word + POS"));
    Check(card.applies && card.compiles && card.status == "Compiles", "the NER Concatenate compiles as fusion");
    Check(card.reason.empty() && card.action == cyxwiz::SequenceFusionAction::None,
          "a compiling card has no reason and no action");
    Check(ValueOf(card.rows, "Input 1 (word ids)") == "Word Embedding, 30,000 x 100", "word row with grouped vocab");
    Check(ValueOf(card.rows, "Input 2 (POS ids)") == "POS Embedding, 128 x 16", "POS row");
    Check(ValueOf(card.rows, "Output") == "116 features per token", "output width is word + POS");
    Check(ValueOf(card.details, "Compiled layer") == "SequenceFeatureFusion", "details name the layer");
    Check(ValueOf(card.details, "Batch source") == "Sequence DataLoader, Data", "details name the batch output");
    Check(ValueOf(card.details, "POS column") == "pos_tags (NER Sentence CSV)", "details name the POS column");

    const auto word_role = cyxwiz::SequenceFusionEmbeddingRole(nodes, graph.links, IdNamed(nodes, "Word Embedding"));
    const auto pos_role = cyxwiz::SequenceFusionEmbeddingRole(nodes, graph.links, IdNamed(nodes, "POS Embedding"));
    Check(word_role && *word_role == "word ids, fused with POS (Concatenate Input 1)", "word Embedding role");
    Check(pos_role && *pos_role == "POS ids, fused with words (Concatenate Input 2)", "POS Embedding role");
    Check(!cyxwiz::SequenceFusionEmbeddingRole(nodes, graph.links, IdNamed(nodes, "Dropout")),
          "a node outside the fusion has no role");
}

void CheckRejectedCards(const cyxwiz::GraphDocument& graph) {
    const int concat = IdNamed(graph.nodes, "Concat Word + POS");

    auto nodes = graph.nodes;
    NodeWithId(nodes, concat).parameters["dim"] = "1";
    auto card = cyxwiz::BuildSequenceFusionCard(nodes, graph.links, concat);
    Check(card.applies && !card.compiles && card.status == "Will not compile", "dim=1 does not compile");
    Check(card.reason == "Concatenate dim is 1, the sequence axis." &&
              card.fix == "Word and POS features join on the feature axis: dim = -1.",
          "dim=1 names the axis and the fix");
    Check(card.action == cyxwiz::SequenceFusionAction::SetFeatureAxis && card.action_label == "Set dim to -1",
          "dim=1 offers the one safe fix");
    Check(ValueOf(card.rows, "Output").empty() && card.details.empty(),
          "a rejected card shows its inputs but no output or details");

    nodes = graph.nodes;
    NodeWithId(nodes, IdNamed(nodes, "NER Sentence CSV")).parameters["pos_column"] = "";
    card = cyxwiz::BuildSequenceFusionCard(nodes, graph.links, concat);
    Check(!card.compiles && card.reason == "The sequence data declares no POS column.", "missing POS column");
    Check(card.action == cyxwiz::SequenceFusionAction::OpenDataInput && card.action_label == "Open Data Input" &&
              card.data_input_node_id == IdNamed(nodes, "NER Sentence CSV"),
          "a missing POS column opens the Data Input");

    auto links = graph.links;
    const auto& input2 = NodeWithId(nodes, concat).inputs.at(1);
    links.erase(std::remove_if(links.begin(), links.end(), [&](const gui::NodeLink& link) {
        return link.to_node == concat && link.to_pin == input2.id;
    }), links.end());
    card = cyxwiz::BuildSequenceFusionCard(graph.nodes, links, concat);
    Check(!card.compiles && card.reason == "Connect exactly Input 1 and Input 2." &&
              card.action == cyxwiz::SequenceFusionAction::None,
          "rewiring reasons offer no automatic fix");

    const auto verdicts = cyxwiz::AnalyzeSequenceFusion(nodes, graph.links, {}, false);
    Check(verdicts.size() == 1 &&
              cyxwiz::SequenceFusionIssueMessage(verdicts.front())
                      .rfind("Word + POS fusion: The sequence data declares no POS column.", 0) == 0,
          "the compiler's issue text starts with the same reason");

    // A Concatenate that no Embedding feeds is not word + POS fusion.
    nodes = graph.nodes;
    NodeWithId(nodes, IdNamed(nodes, "Word Embedding")).type = gui::NodeType::Dense;
    NodeWithId(nodes, IdNamed(nodes, "POS Embedding")).type = gui::NodeType::Dense;
    Check(!cyxwiz::BuildSequenceFusionCard(nodes, graph.links, concat).applies,
          "a Concatenate without Embedding inputs gets no fusion card");
}

}  // namespace

int main() {
    const auto graph = LoadNerGraph();
    CheckContract(graph);
    CheckCompilingCard(graph);
    CheckRejectedCards(graph);
    std::cout << "Sequence fusion presentation passed\n";
    return 0;
}
