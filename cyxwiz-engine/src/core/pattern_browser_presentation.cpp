#include "pattern_browser_presentation.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>

namespace cyxwiz::pattern_browser {

namespace {

struct ChipDef {
    const char* key;
    const char* label;
};
// The file's category names ("Building Blocks" is the older spelling of BuildingBlocks).
constexpr ChipDef kChips[] = {
    {"Basic", "Basic"},
    {"CNN", "CNN"},
    {"RNN", "RNN"},
    {"Transformer", "Transformer"},
    {"Generative", "Generative"},
    {"Reinforcement Learning", "Reinforcement learning"},
    {"BuildingBlocks", "Blocks"},
    {kMine, "My patterns"},
};

std::string Lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

std::string Trimmed(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

bool InChip(const Entry& entry, const std::string& chip) {
    if (chip == kMine) return entry.mine;
    if (entry.mine) return false;
    if (chip == "BuildingBlocks") return entry.category == "BuildingBlocks" || entry.category == "Building Blocks";
    return entry.category == chip;
}

bool Holds(const Entry& entry, const std::string& lower_query) {
    if (Lower(entry.name).find(lower_query) != std::string::npos) return true;
    if (Lower(entry.description).find(lower_query) != std::string::npos) return true;
    if (Lower(entry.id).find(lower_query) != std::string::npos) return true;
    for (const auto& tag : entry.tags)
        if (Lower(tag).find(lower_query) != std::string::npos) return true;
    return false;
}

// "a", "a and b", "a, b and c"
std::string JoinAnd(const std::vector<std::string>& items) {
    std::string out;
    for (size_t i = 0; i < items.size(); ++i) {
        if (i > 0) out += i + 1 == items.size() ? " and " : ", ";
        out += items[i];
    }
    return out;
}

void AddOnce(std::vector<std::string>& items, std::string item) {
    if (std::find(items.begin(), items.end(), item) == items.end()) items.push_back(std::move(item));
}

std::string Plural(size_t count, const char* one, const char* many) {
    return std::to_string(count) + " " + (count == 1 ? one : many);
}

}  // namespace

std::vector<Chip> Chips(const std::vector<Entry>& entries) {
    std::vector<Chip> chips;
    for (const auto& def : kChips) {
        Chip chip{def.key, def.label, 0};
        for (const auto& entry : entries)
            if (InChip(entry, chip.key)) ++chip.count;
        chips.push_back(std::move(chip));
    }
    return chips;
}

ListView List(const std::vector<Entry>& entries, const std::string& chip, const std::string& query) {
    ListView view;
    const std::string wanted = Lower(Trimmed(query));
    view.searching = !wanted.empty();
    view.on_mine = !view.searching && chip == kMine;
    for (const auto& entry : entries) {
        const bool listed = view.searching ? Holds(entry, wanted) : InChip(entry, chip);
        if (!listed) continue;
        (entry.needs.empty() ? view.ready : view.not_ready).push_back(&entry);
    }
    const auto by_name = [](const Entry* a, const Entry* b) {
        const auto la = Lower(a->name), lb = Lower(b->name);
        return la != lb ? la < lb : a->id < b->id;
    };
    std::sort(view.ready.begin(), view.ready.end(), by_name);
    std::sort(view.not_ready.begin(), view.not_ready.end(), by_name);
    const size_t total = view.ready.size() + view.not_ready.size();
    if (view.searching) view.note = std::to_string(total) + " found in every category";
    if (!view.not_ready.empty()) view.not_ready_heading = "Not ready yet (" + std::to_string(view.not_ready.size()) + ")";
    if (total == 0) {
        view.empty_text = view.searching ? "No pattern matches. Try another word."
                          : view.on_mine ? "Nothing saved yet. Select nodes on the canvas, right-click and choose "
                                           "Actions > Save selection as pattern."
                                         : "No patterns in this category.";
    }
    return view;
}

std::string KindLabel(Kind kind) { return kind == Kind::Graph ? "Full graph" : "Block"; }

std::string NodesLabel(size_t count) { return Plural(count, "node", "nodes"); }

std::string NotReadyText(const std::string& needs) { return "Not ready yet: " + needs; }

std::string InsertLabel(Kind kind) { return kind == Kind::Graph ? "Insert full graph" : "Insert block"; }

std::string InsertHint(Kind kind, size_t canvas_nodes) {
    if (kind == Kind::Graph)
        return canvas_nodes > 0 ? "Asks before it touches the " + NodesLabel(canvas_nodes) + " on the canvas"
                                : "Fills the empty canvas";
    return canvas_nodes > 0 ? "Goes below the graph" : "Goes on the empty canvas";
}

bool NeedsPlacementChoice(Kind kind, size_t canvas_nodes) { return kind == Kind::Graph && canvas_nodes > 0; }

std::string PlacementQuestion(size_t canvas_nodes, const std::string& graph) {
    return "The canvas already holds " + NodesLabel(canvas_nodes) + (graph.empty() ? "" : " (" + graph + ")") +
           ". A full graph brings its own Data Input, loss and optimizer.";
}

std::string ReplaceNote(const std::string& graph) {
    if (graph.empty()) return "Replace keeps nothing of the canvas; Edit > Undo brings it back.";
    return "Replace keeps nothing of " + graph + " on the canvas; Edit > Undo brings it back. Its saved file is "
           "not touched until you save.";
}

std::string DoneText(const Entry& entry, Placement placement, const std::string& graph) {
    const std::string what = "Inserted " + entry.name + " (" + NodesLabel(entry.node_count) + ")";
    if (entry.kind == Kind::Graph) {
        const std::string next = "Next: double-click Data Input, point it at your data, Apply, then Train.";
        switch (placement) {
            case Placement::Replace: return what + ", replacing the canvas. " + next;
            case Placement::Beside:
                return what + " to the right of " + (graph.empty() ? std::string("the graph") : graph) +
                       ". The canvas now holds two graphs: delete the one you do not want before you Train.";
            case Placement::Empty:
            case Placement::Below: return what + ". " + next;
        }
    }
    std::string text = what + (placement == Placement::Empty ? " on the empty canvas." : " below the graph.");
    if (entry.block_input) text += " Link the layer before it to Block input.";
    return text;
}

std::string FailText(const std::string& name, const std::string& reason) {
    return "Could not insert " + name + ": " + reason + ". Nothing was added to the canvas.";
}

double NumberStep(bool is_float, const std::string& default_value) {
    if (!is_float) return 1.0;
    char* end = nullptr;
    const double value = std::strtod(default_value.c_str(), &end);
    if (end == default_value.c_str()) return 0.05;
    return std::fabs(value) < 0.05 && value != 0.0 ? 0.001 : 0.05;
}

std::string FormatNumber(double value, bool is_float) {
    if (!is_float) return std::to_string(static_cast<long long>(std::llround(value)));
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.6f", value);
    std::string text = buffer;
    while (!text.empty() && text.back() == '0') text.pop_back();
    if (!text.empty() && text.back() == '.') text.pop_back();
    return text == "-0" ? "0" : text;
}

std::string DeleteQuestion(const std::string& name, const std::string& bin) {
    return "Delete " + name + "? Its file goes to the " + bin + ".";
}

std::string FileStem(const std::string& name) {
    std::string stem;
    bool dash = false;
    for (unsigned char c : name) {
        if (std::isalnum(c)) {
            if (dash && !stem.empty()) stem += '-';
            stem += static_cast<char>(std::tolower(c));
            dash = false;
        } else {
            dash = true;
        }
    }
    return stem.empty() ? "pattern" : stem;
}

SelectionSummary SummarizeSelection(const std::vector<SelectionNode>& nodes, const std::vector<SelectionLink>& links,
                                    const std::vector<int>& selected, const std::string& graph) {
    SelectionSummary summary;
    const std::set<int> chosen(selected.begin(), selected.end());
    std::set<int> entries;
    bool has_data_input = false, has_output = false;
    const auto name_of = [&](int id) {
        for (const auto& node : nodes)
            if (node.id == id) return node.name;
        return std::string("node ") + std::to_string(id);
    };
    for (const auto& node : nodes) {
        if (!chosen.count(node.id)) continue;
        ++summary.nodes;
        has_data_input = has_data_input || node.data_input;
        has_output = has_output || node.output;
    }
    for (const auto& link : links) {
        const bool from_in = chosen.count(link.from_node) > 0, to_in = chosen.count(link.to_node) > 0;
        if (from_in && to_in) {
            ++summary.links;
        } else if (to_in) {
            AddOnce(summary.dropped, "from " + name_of(link.from_node));
            entries.insert(link.to_node);
        } else if (from_in) {
            AddOnce(summary.dropped, "to " + name_of(link.to_node));
        }
    }
    summary.kind = has_data_input && has_output ? Kind::Graph : Kind::Block;
    if (summary.kind == Kind::Graph) entries.clear();  // a full graph starts at its Data Input
    for (const auto& node : nodes)
        if (entries.count(node.id)) summary.entries.push_back(node.id);

    summary.text = NodesLabel(summary.nodes) + " and " + Plural(summary.links, "link", "links") +
                   (graph.empty() ? std::string() : " from " + graph) + ".";
    if (summary.dropped.size() == 1)
        summary.text += " The link " + summary.dropped.front() + " leaves the selection and is not kept.";
    else if (summary.dropped.size() > 1)
        summary.text += " The links " + JoinAnd(summary.dropped) + " leave the selection and are not kept.";

    if (summary.kind == Kind::Graph) {
        summary.kind_note = "It holds a Data Input and an Output, so it saves as a full graph.";
    } else {
        std::vector<std::string> fed;
        for (int id : summary.entries) fed.push_back(name_of(id));
        summary.kind_note = fed.empty() ? std::string() : "Block input feeds " + JoinAnd(fed) + ". ";
        summary.kind_note += "A selection from a Data Input to an Output saves as a full graph.";
    }
    return summary;
}

std::string NameTakenText(const std::string& name) {
    return "You already have a pattern named " + name + ". Save replaces it.";
}

std::string SavedText(const std::string& name) { return "Saved " + name + " to My patterns."; }

}  // namespace cyxwiz::pattern_browser
