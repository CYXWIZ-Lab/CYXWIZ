// The Nodes panel's model (TOFIX129 Nodes panel piece, approved NP-1..NP-8):
// the filters in plain words, the groups, the search order, a card's mark
// and its tooltip, read from the real node registry.
#include "core/nodes_panel_presentation.h"
#include "core/node_metadata_registry.h"
#include "core/node_palette.h"
#include "core/engine_config.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

bool Contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

}  // namespace

int main() {
    namespace np = cyxwiz::nodes_panel;
    using cyxwiz::NodeType;
    auto& registry = cyxwiz::NodeMetadataRegistry::Instance();
    registry.Initialize();
    registry.SetUserNodes({}, {});

    // Filters: six, in plain words, no code words.
    Check(np::Filters().size() == 6, "six filters");
    for (const auto& f : np::Filters()) {
        const std::string name = f.name;
        Check(!Contains(name, "Pipeline") && !Contains(name, "UI-only") && !Contains(name, "Blocked"),
              "a filter in plain words: " + name);
    }
    Check(std::string(np::FilterName(np::Filter::TrainsModel)) == "Trains a model", "Trains a model");

    // A loss that trains: the tooltip says so; none of the support rows
    // (it showed red "Fail closed" / "Unsupported" on hover).
    const auto* focal = registry.GetMetadata(NodeType::FocalLoss);
    Check(focal != nullptr, "Focal Loss is registered");
    if (focal) {
        const np::Tooltip tip = np::BuildTooltip(*focal);
        Check(tip.title == "Focal Loss", "title");
        Check(tip.family == "Training \xC2\xB7 loss", "family names the role: " + tip.family);
        Check(tip.runs == "Runs in training" && tip.tone == np::Tone::Good, "a loss runs in training: " + tip.runs);
        Check(!Contains(tip.runs, "Fail") && !Contains(tip.runs, "nsupported"), "no support-axis words");
        Check(!tip.inputs.empty() && !tip.outputs.empty(), "pins listed");
        Check(tip.outputs[0].kind == "loss", "the output's kind: " + tip.outputs[0].kind);
        Check(np::Matches(*focal, np::Filter::TrainsModel), "a loss is under Trains a model");
        for (auto type : {NodeType::Adam, NodeType::MSELoss, NodeType::Dense})
            Check(np::Matches(*registry.GetMetadata(type), np::Filter::TrainsModel),
                  "optimizers, losses and layers train a model: " + registry.GetMetadata(type)->name);
        Check(!np::Matches(*focal, np::Filter::ViewsAndPanels), "a loss is not a view");
        Check(np::CanAdd(*focal), "a loss can be added");
    }

    // Data Input: a data source with a data output.
    if (const auto* input = registry.GetMetadata(NodeType::DataInput)) {
        const np::Tooltip tip = np::BuildTooltip(*input);
        Check(tip.tone == np::Tone::Good && Contains(tip.runs, "Runs"), "Data Input runs: " + tip.runs);
        Check(!tip.outputs.empty() && tip.outputs[0].kind == "data", "Data Input outputs data");
    } else {
        Check(false, "Data Input is registered");
    }

    // A node that can't run yet: the reason, not addable, under the filter.
    const cyxwiz::NodeMetadata* cant = nullptr;
    for (const auto* m : registry.GetAllMetadata())
        if (m && cyxwiz::palette::PlaceNode(*m).place == cyxwiz::palette::Place::CantRun) { cant = m; break; }
    Check(cant != nullptr, "the registry has a node that can't run yet");
    if (cant) {
        const np::Tooltip tip = np::BuildTooltip(*cant);
        Check(tip.runs.rfind("Can't run yet: ", 0) == 0 && tip.runs.size() > 15, "why it can't run: " + tip.runs);
        Check(!np::CanAdd(*cant), "not addable");
        Check(np::Matches(*cant, np::Filter::CantRunYet), "listed under Can't run yet");
        Check(np::CardMark(*cant) != np::Mark::None || cant->IsTemplate(), "a card mark or a template badge");
    }

    // Groups: All keeps the palette's Category nodes; the counts are the nodes.
    std::vector<np::CategoryNodes> by_category;
    for (auto category : registry.GetCategories()) by_category.push_back({category, registry.GetByCategory(category, true)});
    const auto all = np::BuildGroups(by_category, np::Filter::All);
    Check(!all.empty(), "groups listed");
    bool io = false, training = false;
    for (const auto& g : all) {
        io |= g.name == "I/O";
        training |= g.name == "Training";
        Check(g.category != cyxwiz::NodeCategory::Plugin, "plugins are their own section");
        for (const auto* m : g.nodes)
            Check(cyxwiz::palette::PlaceNode(*m).place == cyxwiz::palette::Place::Category,
                  "All groups only what runs: " + m->name);
    }
    Check(io && training, "the default open groups exist");
    for (const auto& name : cyxwiz::core::NodesPanelConfig{}.open_groups) {
        bool found = false;
        for (const auto& g : all) found |= g.name == name;
        Check(found, "default open group " + name + " is a group");
    }
    const auto cant_groups = np::BuildGroups(by_category, np::Filter::CantRunYet);
    for (const auto& g : cant_groups)
        for (const auto* m : g.nodes) Check(!np::CanAdd(*m), "Can't run yet lists only what can't be added: " + m->name);

    // Favourites filter: in their groups.
    registry.ToggleFavorite(NodeType::Dense);
    const auto favs = np::BuildGroups(by_category, np::Filter::Favourites);
    Check(favs.size() == 1 && favs[0].nodes.size() == 1 && favs[0].nodes[0]->type == NodeType::Dense,
          "Favourites: the starred node in its group");
    registry.ToggleFavorite(NodeType::Dense);

    // Search: named first (whole name, then start), then Related.
    const auto found = registry.Search("loss", true);
    const auto order = np::OrderSearch(found, "loss", np::Filter::All);
    Check(!order.named.empty(), "losses found");
    for (const auto* m : order.named) {
        std::string lower = m->name;
        for (auto& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        Check(lower.find("loss") != std::string::npos, "named holds the word: " + m->name);
        Check(m->type != NodeType::PluginCustom, "no generic plugin entry");
    }
    for (const auto* m : order.related) {
        std::string lower = m->name;
        for (auto& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        Check(lower.find("loss") == std::string::npos, "related does not name it: " + m->name);
    }
    const auto dense = np::OrderSearch(registry.Search("dense", true), "dense", np::Filter::All);
    Check(!dense.named.empty() && dense.named[0]->name == "Dense", "the exact name first");
    const auto trains = np::OrderSearch(found, "loss", np::Filter::TrainsModel);
    for (const auto* m : trains.named) Check(np::Matches(*m, np::Filter::TrainsModel), "a filter narrows search");

    Check(np::CountLine(1, false) == "1 node" && np::CountLine(12, false) == "12 nodes" &&
              np::CountLine(13, true) == "13 found",
          "count line");

    if (g_failures) {
        std::printf("%d Nodes panel presentation checks failed\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("Nodes panel presentation passed\n");
    return EXIT_SUCCESS;
}
