#pragma once

// What the Nodes panel shows, without ImGui (TOFIX129 Nodes panel piece,
// approved mockup NP-1..NP-8): the filters in plain words, the groups and
// their counts, the search order, a card's mark and its tooltip. The panel
// (gui/panels/node_browser_panel.cpp) draws these; test_nodes_panel_presentation
// checks them.

#include "node_metadata.h"

#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::nodes_panel {

// The filter button's choices.
enum class Filter { All, TrainsModel, DataSteps, ViewsAndPanels, CantRunYet, Favourites };

struct FilterInfo {
    Filter filter;
    const char* name;  // "Trains a model"
    const char* note;  // "layers, losses, optimizers"
};
const std::vector<FilterInfo>& Filters();
const char* FilterName(Filter filter);

// Whether a node passes a filter. All passes every node; where it is listed
// (its group, Advanced operations, Can't run yet) is the palette's choice.
bool Matches(const NodeMetadata& metadata, Filter filter);

// One folding group: a category's nodes that pass the filter. With All the
// group holds the nodes that run and are not table steps (palette Category);
// with another filter, every node of the category that passes it (never a
// deprecated one).
struct Group {
    NodeCategory category = NodeCategory::Unknown;
    std::string name;
    std::vector<const NodeMetadata*> nodes;
};
using CategoryNodes = std::pair<NodeCategory, std::vector<const NodeMetadata*>>;
std::vector<Group> BuildGroups(const std::vector<CategoryNodes>& by_category, Filter filter);

// A search's results: the nodes the query names (whole name, then the
// name's start, then anywhere in it), then the ones that only mention it in
// a keyword or the description. `found` is the registry's match list, best
// first. The generic plugin entry and deprecated nodes are dropped (plugin
// nodes are listed on their own; a deprecated node is not offered); with
// All, what runs comes before table steps and what can't run yet.
struct SearchResult {
    std::vector<const NodeMetadata*> named;
    std::vector<const NodeMetadata*> related;
};
SearchResult OrderSearch(std::vector<const NodeMetadata*> found, const std::string& query, Filter filter);

// "12 nodes" beside the filter; "13 found" while searching.
std::string CountLine(size_t count, bool searching);

// The small dot at a card's top left: only the exceptions are marked.
enum class Mark { None, Partial, Blocked, Pending };
Mark CardMark(const NodeMetadata& metadata);

// A card's hover text (NP-1): what it does, its pins with their kinds, and
// one line on where it runs. The per-axis support facts stay in the Info
// panel.
enum class Tone { Good, Partial, Blocked, Neutral };
struct PortLine {
    std::string name;
    std::string kind;  // gui::PinTypeName: "Tensor", "Labels", "Dataset", ...
};
struct Tooltip {
    std::string title;
    std::string family;       // "Training · loss"
    std::string description;
    std::vector<PortLine> inputs;
    std::vector<PortLine> outputs;
    std::string runs;         // "Runs in training", "Can't run yet: ..."
    Tone tone = Tone::Neutral;
    std::string footer;       // "Double-click or drag to add ..."
};
Tooltip BuildTooltip(const NodeMetadata& metadata);

// Whether the panel may add the node (double-click, drag, Add to Canvas).
bool CanAdd(const NodeMetadata& metadata);

// Why a node can't run (the first unsupported fact's reason), or empty.
std::string CantRunReason(const NodeMetadata& metadata);

}  // namespace cyxwiz::nodes_panel
