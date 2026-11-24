#pragma once

// What the Pattern Browser and the canvas's Save selection as pattern dialog
// show, without ImGui (TOFIX178 4e, approved mockup): the category chips and
// their counts, the list (ready patterns, then the ones not ready yet), a
// row's marks, the Insert button's words, the question before a full graph
// lands on a canvas that has nodes, and the result or failure line. The
// renderers (gui/panels/pattern_browser.cpp, gui/node_editor_save_pattern.cpp)
// draw these; test_pattern_browser_presentation checks them.

#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz::pattern_browser {

enum class Kind { Graph, Block };

// One pattern, as the browser needs it.
struct Entry {
    std::string id;
    std::string name;
    std::string description;
    std::string category;  // the file's category: "CNN", "Reinforcement Learning", "BuildingBlocks", ...
    std::vector<std::string> tags;
    Kind kind = Kind::Block;
    std::string needs;  // why it cannot be inserted yet; empty when it can
    size_t node_count = 0;
    bool mine = false;         // from the user's patterns folder
    bool block_input = false;  // has a Block input node (where the layer before it links)
};

// The chip of the user's own patterns.
inline constexpr const char* kMine = "mine";

// The chips in their order, each with how many patterns it holds. A user's
// pattern counts under My patterns only.
struct Chip {
    std::string key;    // a category, or kMine
    std::string label;  // "Reinforcement learning", "Blocks", "My patterns"
    size_t count = 0;
};
std::vector<Chip> Chips(const std::vector<Entry>& entries);

// The list under the chips: with a query, every pattern whose name,
// description, tags or id hold it (any case), from every category; else
// the chip's patterns. Ready ones first, then the ones not ready yet; each
// part by name.
struct ListView {
    std::vector<const Entry*> ready;
    std::vector<const Entry*> not_ready;
    bool searching = false;
    bool on_mine = false;     // My patterns is shown (its folder line goes under the list)
    std::string note;         // "3 found in every category" while searching
    std::string empty_text;   // when nothing is listed
    std::string not_ready_heading;  // "Not ready yet (2)"
};
ListView List(const std::vector<Entry>& entries, const std::string& chip, const std::string& query);

std::string KindLabel(Kind kind);          // "Full graph" / "Block"
std::string NodesLabel(size_t count);      // "1 node" / "12 nodes"
std::string NotReadyText(const std::string& needs);  // "Not ready yet: ..."
std::string InsertLabel(Kind kind);        // "Insert full graph" / "Insert block"
// The muted words beside Insert.
std::string InsertHint(Kind kind, size_t canvas_nodes);

// A full graph brings its own Data Input, loss and optimizer: on a canvas
// that has nodes the user picks Replace the canvas, Add beside it or Cancel.
bool NeedsPlacementChoice(Kind kind, size_t canvas_nodes);
// `graph` is the canvas graph's name ("mlp_v2"); empty when it has none.
std::string PlacementQuestion(size_t canvas_nodes, const std::string& graph);
std::string ReplaceNote(const std::string& graph);

// What was done, for the green line under the button.
enum class Placement { Empty, Below, Beside, Replace };
std::string DoneText(const Entry& entry, Placement placement, const std::string& graph);
// The orange line when nothing was added.
std::string FailText(const std::string& name, const std::string& reason);

// The - / + step of a number setting: 1 for whole numbers; for decimals,
// 0.001 when the default is below 0.05 (a learning rate), else 0.05.
double NumberStep(bool is_float, const std::string& default_value);
// A typed number setting's value as the pattern stores it ("64", "0.3").
std::string FormatNumber(double value, bool is_float);

// My patterns' delete question; `bin` is "Recycle Bin" or "Trash".
std::string DeleteQuestion(const std::string& name, const std::string& bin);

// --- Save selection as pattern ---

// A pattern's file name from its name: lower case, runs of other
// characters as one "-" ("Dense-BN-ReLU-Dropout" -> "dense-bn-relu-dropout").
std::string FileStem(const std::string& name);

struct SelectionNode {
    int id = 0;
    std::string name;
    bool data_input = false;  // a Data Input
    bool output = false;      // an Output
};
struct SelectionLink {
    int from_node = 0;
    int to_node = 0;
};
// What a selection saves as: its nodes, the links inside it, the links that
// leave it (dropped), the selected nodes an outside link fed (each gets a
// Block input in front of it), and Block or Full graph (a Data Input and an
// Output make a full graph).
struct SelectionSummary {
    size_t nodes = 0;
    size_t links = 0;
    std::vector<std::string> dropped;  // "from Flatten", "to Dense 256"
    std::vector<int> entries;          // in the canvas's node order
    Kind kind = Kind::Block;
    std::string text;       // "4 nodes and 3 links from mlp_v2. The links from ... are not kept."
    std::string kind_note;  // "Block input feeds Dense 512. ..."
};
// `nodes` is the whole canvas, `selected` the chosen ids, `graph` its name.
SelectionSummary SummarizeSelection(const std::vector<SelectionNode>& nodes, const std::vector<SelectionLink>& links,
                                    const std::vector<int>& selected, const std::string& graph);
std::string NameTakenText(const std::string& name);  // "You already have a pattern named X. Save replaces it."
std::string SavedText(const std::string& name);      // "Saved X to My patterns."

}  // namespace cyxwiz::pattern_browser
