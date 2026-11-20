// The Pattern Browser's model (TOFIX178 4e, approved mockup): chips and their
// counts, the list order, not-ready patterns, the Insert words, the question
// before a full graph lands on a canvas with nodes, the result and failure
// lines, number steps, file names and the Save selection summary. The main
// scenario reads the shipped pattern files; the canvas is catdog's mlp_v2.
#include "core/pattern_browser_presentation.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace pb = cyxwiz::pattern_browser;
namespace fs = std::filesystem;

int g_failures = 0;

void Check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

bool Contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

fs::path FindPatternsDir() {
    for (auto dir = fs::current_path(); !dir.empty(); dir = dir.parent_path()) {
        const auto candidate = dir / "cyxwiz-engine" / "resources" / "patterns";
        if (fs::is_directory(candidate)) return candidate;
        if (dir == dir.parent_path()) break;
    }
    return {};
}

// What the browser builds from a pattern file.
std::vector<pb::Entry> ShippedEntries(const fs::path& dir) {
    std::vector<pb::Entry> entries;
    for (const auto& file : fs::directory_iterator(dir)) {
        if (file.path().extension() != ".cyxgraph") continue;
        std::ifstream in(file.path());
        const auto j = nlohmann::json::parse(in);
        pb::Entry entry;
        entry.id = j.value("id", file.path().stem().string());
        entry.name = j.value("name", entry.id);
        entry.description = j.value("description", "");
        entry.category = j.value("category", "");
        for (const auto& tag : j.value("tags", nlohmann::json::array())) entry.tags.push_back(tag.get<std::string>());
        entry.kind = j.value("kind", "block") == "graph" ? pb::Kind::Graph : pb::Kind::Block;
        entry.needs = j.value("needs", "");
        entry.node_count = j["template"]["nodes"].size();
        for (const auto& node : j["template"]["nodes"])
            if (node.value("name", "") == "Block input") entry.block_input = true;
        entries.push_back(std::move(entry));
    }
    return entries;
}

const pb::Entry* Find(const std::vector<pb::Entry>& entries, const std::string& id) {
    for (const auto& entry : entries)
        if (entry.id == id) return &entry;
    return nullptr;
}

}  // namespace

int main() {
    const auto dir = FindPatternsDir();
    Check(!dir.empty(), "the shipped patterns folder is found");
    if (dir.empty()) return 1;
    auto entries = ShippedEntries(dir);
    Check(entries.size() == 31, "31 shipped patterns (" + std::to_string(entries.size()) + ")");

    // A pattern saved from catdog's mlp_v2 (Save selection as pattern).
    pb::Entry mine;
    mine.id = "user:dense-bn-relu-dropout";
    mine.name = "Dense-BN-ReLU-Dropout";
    mine.description = "Dense 512, BatchNorm, ReLU and Dropout 0.4, saved from catdog mlp_v2.";
    mine.category = "BuildingBlocks";
    mine.kind = pb::Kind::Block;
    mine.node_count = 5;
    mine.mine = true;
    mine.block_input = true;
    entries.push_back(mine);

    // Chips: in the mockup's order, with counts; the user's pattern counts under My patterns only.
    const auto chips = pb::Chips(entries);
    const std::vector<std::pair<std::string, size_t>> expected = {
        {"Basic", 4}, {"CNN", 10}, {"RNN", 5}, {"Transformer", 3}, {"Generative", 5},
        {"Reinforcement learning", 1}, {"Blocks", 3}, {"My patterns", 1}};
    Check(chips.size() == expected.size(), "eight chips");
    for (size_t i = 0; i < chips.size() && i < expected.size(); ++i)
        Check(chips[i].label == expected[i].first && chips[i].count == expected[i].second,
              "chip " + std::to_string(i) + ": " + chips[i].label + " " + std::to_string(chips[i].count));

    // CNN: ready patterns by name, then U-Net under "Not ready yet (1)".
    auto view = pb::List(entries, "CNN", "");
    Check(!view.searching && !view.on_mine && view.note.empty(), "a chip's list is not a search");
    Check(view.ready.size() == 9 && view.not_ready.size() == 1, "CNN: nine ready, one not ready");
    Check(!view.ready.empty() && view.ready.front()->name == "AlexNet", "CNN starts at AlexNet");
    Check(!view.not_ready.empty() && view.not_ready.front()->id == "unet", "U-Net is not ready");
    Check(view.not_ready_heading == "Not ready yet (1)", "the not-ready heading counts them");
    for (size_t i = 1; i < view.ready.size(); ++i)
        Check(view.ready[i - 1]->name <= view.ready[i]->name, "CNN is in name order");

    // Generative: all five wait for training the Engine does not have.
    view = pb::List(entries, "Generative", "");
    Check(view.ready.empty() && view.not_ready.size() == 5, "Generative: five not ready");
    Check(view.not_ready_heading == "Not ready yet (5)", "five in the heading");

    // PPO is under Reinforcement learning, not Custom.
    view = pb::List(entries, "Reinforcement Learning", "");
    Check(view.ready.size() == 1 && view.ready.front()->id == "ppo_mlp", "PPO under Reinforcement learning");

    // Blocks: the shipped three; the user's block-category pattern is under My patterns.
    view = pb::List(entries, "BuildingBlocks", "");
    Check(view.ready.size() == 3, "Blocks lists the three shipped blocks");
    view = pb::List(entries, pb::kMine, "");
    Check(view.on_mine && view.ready.size() == 1 && view.ready.front()->mine, "My patterns lists the saved one");

    // Search covers every category, any case, tags and ids too.
    view = pb::List(entries, "Basic", "  ResNet ");
    Check(view.searching && !view.on_mine, "a query searches");
    Check(view.ready.size() >= 2, "ResNet finds ResNet-18 and Residual Block");
    Check(view.note == std::to_string(view.ready.size() + view.not_ready.size()) + " found in every category",
          "the search note: " + view.note);
    view = pb::List(entries, "Basic", "seq2seq");
    Check(view.ready.empty() && view.not_ready.size() == 1 && view.not_ready.front()->id == "lstm_seq2seq",
          "a search finds a not-ready pattern by id and keeps it apart");
    view = pb::List(entries, "Basic", "zzzz");
    Check(view.ready.empty() && view.not_ready.empty() && view.empty_text == "No pattern matches. Try another word.",
          "no match says so");
    const std::vector<pb::Entry> none;
    view = pb::List(none, pb::kMine, "");
    Check(Contains(view.empty_text, "Save selection as pattern"), "empty My patterns says how to save one");

    // Row marks and the Insert words.
    const auto* resnet = Find(entries, "resnet18");
    const auto* block = Find(entries, "conv_bn_relu");
    Check(resnet && resnet->kind == pb::Kind::Graph && resnet->node_count == 77, "ResNet-18: a full graph of 77 nodes");
    Check(block && block->kind == pb::Kind::Block && block->block_input, "Conv-BN-ReLU: a block with Block input");
    Check(pb::KindLabel(pb::Kind::Graph) == "Full graph" && pb::KindLabel(pb::Kind::Block) == "Block", "kind labels");
    Check(pb::NodesLabel(1) == "1 node" && pb::NodesLabel(77) == "77 nodes", "node counts");
    Check(pb::InsertLabel(pb::Kind::Graph) == "Insert full graph" && pb::InsertLabel(pb::Kind::Block) == "Insert block",
          "insert labels");
    Check(pb::InsertHint(pb::Kind::Graph, 20) == "Asks before it touches the 20 nodes on the canvas", "graph hint");
    Check(pb::InsertHint(pb::Kind::Graph, 0) == "Fills the empty canvas", "graph hint on an empty canvas");
    Check(pb::InsertHint(pb::Kind::Block, 20) == "Goes below the graph", "block hint");
    Check(pb::NotReadyText("X.") == "Not ready yet: X.", "not-ready line");

    // A full graph on mlp_v2's 20 nodes asks first; a block does not; an empty canvas does not.
    Check(pb::NeedsPlacementChoice(pb::Kind::Graph, 20), "a full graph on a canvas with nodes asks");
    Check(!pb::NeedsPlacementChoice(pb::Kind::Graph, 0), "a full graph on an empty canvas does not ask");
    Check(!pb::NeedsPlacementChoice(pb::Kind::Block, 20), "a block does not ask");
    Check(pb::PlacementQuestion(20, "mlp_v2") ==
              "The canvas already holds 20 nodes (mlp_v2). A full graph brings its own Data Input, loss and optimizer.",
          "the question");
    Check(Contains(pb::ReplaceNote("mlp_v2"), "Edit > Undo brings it back"), "replace says Undo brings it back");

    // Result lines.
    Check(pb::DoneText(*resnet, pb::Placement::Replace, "mlp_v2") ==
              "Inserted ResNet-18 (77 nodes), replacing the canvas. Next: double-click Data Input, point it at your "
              "data, Apply, then Train.",
          "replace result");
    Check(Contains(pb::DoneText(*resnet, pb::Placement::Beside, "mlp_v2"), "to the right of mlp_v2"), "beside result");
    Check(pb::DoneText(*block, pb::Placement::Below, "mlp_v2") ==
              "Inserted Conv-BN-ReLU Block (4 nodes) below the graph. Link the layer before it to Block input.",
          "block result");
    pb::Entry no_input = *block;
    no_input.block_input = false;
    Check(!Contains(pb::DoneText(no_input, pb::Placement::Empty, ""), "Block input"),
          "a block without Block input does not mention it");
    Check(pb::FailText("Residual Block", "a link goes to input 3 of \"Add\", which has 2 inputs") ==
              "Could not insert Residual Block: a link goes to input 3 of \"Add\", which has 2 inputs. Nothing was "
              "added to the canvas.",
          "failure line");

    // Number steps and formatting.
    Check(pb::NumberStep(false, "64") == 1.0, "a whole number steps by 1");
    Check(pb::NumberStep(true, "0.3") == 0.05, "a rate steps by 0.05");
    Check(pb::NumberStep(true, "0.001") == 0.001, "a learning rate steps by 0.001");
    Check(pb::FormatNumber(64.0, false) == "64" && pb::FormatNumber(0.35, true) == "0.35" &&
              pb::FormatNumber(0.001, true) == "0.001" && pb::FormatNumber(2.0, true) == "2",
          "numbers as the pattern stores them");

    Check(pb::DeleteQuestion("Dense-BN-ReLU-Dropout", "Recycle Bin") ==
              "Delete Dense-BN-ReLU-Dropout? Its file goes to the Recycle Bin.",
          "delete question");

    // Save selection: mlp_v2's first Dense-BN-ReLU-Dropout row (ids as on the canvas).
    Check(pb::FileStem("Dense-BN-ReLU-Dropout") == "dense-bn-relu-dropout", "file name from the name");
    Check(pb::FileStem("  My  block (v2)! ") == "my-block-v2" && pb::FileStem("!!!") == "pattern", "odd names");
    const std::vector<pb::SelectionNode> canvas = {
        {1, "Data Input", true, false}, {2, "Flatten"},  {3, "Dense 512"},   {4, "BatchNorm 1"},
        {5, "ReLU 1"},                  {6, "Dropout 0.4"}, {7, "Dense 256"}, {8, "Output", false, true}};
    const std::vector<pb::SelectionLink> wires = {{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8}};
    auto summary = pb::SummarizeSelection(canvas, wires, {3, 4, 5, 6}, "mlp_v2");
    Check(summary.nodes == 4 && summary.links == 3 && summary.kind == pb::Kind::Block, "four nodes, three links, a block");
    Check(summary.entries == std::vector<int>{3}, "Dense 512 is where Block input goes");
    Check(summary.text == "4 nodes and 3 links from mlp_v2. The links from Flatten and to Dense 256 leave the "
                          "selection and are not kept.",
          "summary: " + summary.text);
    Check(Contains(summary.kind_note, "Block input feeds Dense 512."), "kind note: " + summary.kind_note);

    summary = pb::SummarizeSelection(canvas, wires, {1, 2, 3, 4, 5, 6, 7, 8}, "mlp_v2");
    Check(summary.kind == pb::Kind::Graph && summary.entries.empty() && summary.links == 7, "the whole graph is a full graph");
    Check(summary.text == "8 nodes and 7 links from mlp_v2.", "nothing dropped: " + summary.text);

    summary = pb::SummarizeSelection(canvas, {{2, 3}, {2, 4}, {4, 5}}, {3, 4, 5}, "");
    Check(summary.dropped.size() == 1 && summary.text == "3 nodes and 1 link. The link from Flatten leaves the "
                                                         "selection and is not kept.",
          "two links from one outside node are named once: " + summary.text);
    Check(summary.entries == std::vector<int>{3, 4}, "each node an outside link fed gets a Block input");

    Check(pb::NameTakenText("Dense-BN-ReLU-Dropout") ==
              "You already have a pattern named Dense-BN-ReLU-Dropout. Save replaces it.",
          "name taken");
    Check(pb::SavedText("X") == "Saved X to My patterns.", "saved toast");

    if (g_failures == 0) std::printf("Pattern Browser presentation: all checks passed\n");
    return g_failures == 0 ? 0 : 1;
}
