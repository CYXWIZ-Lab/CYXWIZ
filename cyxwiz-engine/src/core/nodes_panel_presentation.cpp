#include "nodes_panel_presentation.h"

#include "node_palette.h"

#include <algorithm>
#include <cctype>

namespace cyxwiz::nodes_panel {

namespace {

const SupportAxisDefinition* FindAxis(const NodeMetadata& metadata, const char* name) {
    const auto it = std::find_if(metadata.support_axes.begin(), metadata.support_axes.end(),
                                 [name](const SupportAxisDefinition& axis) { return axis.name == name; });
    return it != metadata.support_axes.end() ? &*it : nullptr;
}

bool Supported(const NodeMetadata& metadata, const char* name) {
    const auto* axis = FindAxis(metadata, name);
    return axis && axis->supported;
}

bool IsTrainingCategory(NodeCategory category) {
    switch (category) {
    case NodeCategory::Layers:
    case NodeCategory::Activation:
    case NodeCategory::Pooling:
    case NodeCategory::Normalization:
    case NodeCategory::Attention:
    case NodeCategory::Recurrent:
    case NodeCategory::ShapeOps:
    case NodeCategory::MergeOps:
    case NodeCategory::Training:
    case NodeCategory::Regularization:
        return true;
    default:
        return false;
    }
}

bool Runnable(const NodeMetadata& metadata) {
    return metadata.IsImplemented() && !IsNodeSupportBlocked(metadata);
}

// The rules the filter had as Training / Pipeline / UI-only. Training also
// takes the nodes the training backend owns: losses and optimizers have no
// "Training Backend" fact, so the old rule left every one of them out.
bool TrainsModel(const NodeMetadata& metadata) {
    const auto* owner = FindAxis(metadata, "Implementation Owner");
    const bool backend = Supported(metadata, "Training Backend") ||
                         (owner && owner->supported && owner->value == "training_backend");
    return Runnable(metadata) && backend && Supported(metadata, "Compile") && Supported(metadata, "Training");
}

bool DataStep(const NodeMetadata& metadata) {
    return Runnable(metadata) && Supported(metadata, "Runtime") && Supported(metadata, "Pipeline Executor");
}

bool ViewOrPanel(const NodeMetadata& metadata) {
    if (!Runnable(metadata) || IsTrainingCategory(metadata.category)) return false;
    for (const char* axis : {"Runtime", "Pipeline Executor", "Training Backend", "Compile", "Training"})
        if (FindAxis(metadata, axis)) return false;
    return true;
}

std::string Lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

std::vector<PortLine> Ports(const std::vector<PortDefinition>& ports) {
    std::vector<PortLine> out;
    out.reserve(ports.size());
    for (const auto& port : ports) out.push_back({port.name, gui::PinTypeName(port.type)});
    return out;
}

// "model_layer" -> "model layer".
std::string Words(std::string value) {
    std::replace(value.begin(), value.end(), '_', ' ');
    return value;
}

}  // namespace

const std::vector<FilterInfo>& Filters() {
    static const std::vector<FilterInfo> filters = {
        {Filter::All, "All", "every node that runs, grouped"},
        {Filter::TrainsModel, "Trains a model", "layers, losses, optimizers"},
        {Filter::DataSteps, "Data steps", "loading, preparing, exporting"},
        {Filter::ViewsAndPanels, "Views and panels", "dashboards, plots, notes"},
        {Filter::CantRunYet, "Can't run yet", "previews and blocked, with why"},
        {Filter::Favourites, "Favourites", "the ones you starred"},
    };
    return filters;
}

const char* FilterName(Filter filter) {
    for (const auto& info : Filters())
        if (info.filter == filter) return info.name;
    return "All";
}

bool Matches(const NodeMetadata& metadata, Filter filter) {
    switch (filter) {
    case Filter::All: return true;
    case Filter::TrainsModel: return TrainsModel(metadata);
    case Filter::DataSteps: return DataStep(metadata);
    case Filter::ViewsAndPanels: return ViewOrPanel(metadata);
    case Filter::CantRunYet: return palette::PlaceNode(metadata).place == palette::Place::CantRun;
    case Filter::Favourites: return metadata.is_favorite;
    }
    return true;
}

std::vector<Group> BuildGroups(const std::vector<CategoryNodes>& by_category, Filter filter) {
    std::vector<Group> groups;
    for (const auto& [category, nodes] : by_category) {
        if (category == NodeCategory::Plugin) continue;  // plugin nodes are their own section
        Group group{category, GetCategoryDisplayName(category), {}};
        for (const NodeMetadata* metadata : nodes) {
            if (!metadata) continue;
            const palette::Place place = palette::PlaceNode(*metadata).place;
            const bool listed = filter == Filter::All ? place == palette::Place::Category
                                                      : place != palette::Place::Hidden && Matches(*metadata, filter);
            if (listed) group.nodes.push_back(metadata);
        }
        if (!group.nodes.empty()) groups.push_back(std::move(group));
    }
    return groups;
}

SearchResult OrderSearch(std::vector<const NodeMetadata*> found, const std::string& query, Filter filter) {
    found.erase(std::remove_if(found.begin(), found.end(),
                               [&](const NodeMetadata* m) {
                                   return !m || m->type == NodeType::PluginCustom ||
                                          palette::PlaceNode(*m).place == palette::Place::Hidden ||
                                          !Matches(*m, filter);
                               }),
                found.end());
    if (filter == Filter::All) {
        const auto rank = [](const NodeMetadata* m) {
            switch (palette::PlaceNode(*m).place) {
            case palette::Place::Category: return 0;
            case palette::Place::Advanced: return 1;
            default: return 2;
            }
        };
        std::stable_sort(found.begin(), found.end(),
                         [&](const NodeMetadata* a, const NodeMetadata* b) { return rank(a) < rank(b); });
    }
    const std::string q = Lower(query);
    SearchResult out;
    for (const NodeMetadata* m : found) (Lower(m->name).find(q) != std::string::npos ? out.named : out.related).push_back(m);
    const auto strength = [&](const NodeMetadata* m) {
        const std::string name = Lower(m->name);
        return name == q ? 2 : name.rfind(q, 0) == 0 ? 1 : 0;
    };
    std::stable_sort(out.named.begin(), out.named.end(),
                     [&](const NodeMetadata* a, const NodeMetadata* b) { return strength(a) > strength(b); });
    return out;
}

std::string CountLine(size_t count, bool searching) {
    if (searching) return std::to_string(count) + " found";
    return std::to_string(count) + (count == 1 ? " node" : " nodes");
}

Mark CardMark(const NodeMetadata& metadata) {
    const auto* state = FindAxis(metadata, "Support State");
    if (!state || state->value == "real") return Mark::None;
    if (state->value == "partial") return Mark::Partial;
    if (state->value == "blocked") return Mark::Blocked;
    return Mark::Pending;
}

bool CanAdd(const NodeMetadata& metadata) {
    return CanAddNodeToGraph(metadata) && metadata.made_by.empty();
}

std::string CantRunReason(const NodeMetadata& metadata) {
    if (const auto* state = FindAxis(metadata, "Support State");
        state && !state->reason.empty() && (!state->supported || state->value == "blocked"))
        return state->reason;
    for (const auto& axis : metadata.support_axes)
        if (!axis.supported && !axis.reason.empty()) return axis.reason;
    return {};
}

Tooltip BuildTooltip(const NodeMetadata& metadata) {
    Tooltip out;
    out.title = metadata.name;
    out.family = GetCategoryDisplayName(metadata.category);
    if (const auto* role = FindAxis(metadata, "Training Role"); role && !role->value.empty())
        out.family += " \xC2\xB7 " + Words(role->value);
    out.description = metadata.brief_description;
    out.inputs = Ports(metadata.inputs);
    out.outputs = Ports(metadata.outputs);

    const palette::Placement placed = palette::PlaceNode(metadata);
    const std::string reason = CantRunReason(metadata);
    if (!metadata.made_by.empty()) {
        out.runs = metadata.made_by;
        out.tone = Tone::Neutral;
        out.footer = "Not added from the panel";
        return out;
    }
    if (placed.place == palette::Place::CantRun) {
        out.runs = "Can't run yet: " + (reason.empty() ? Lower(placed.why) : reason);
        out.tone = IsNodeSupportBlocked(metadata) ? Tone::Blocked : Tone::Partial;
        out.footer = "Can't be added yet; click for every detail in Info";
        return out;
    }
    const bool training = TrainsModel(metadata) || FindAxis(metadata, "Training Role") != nullptr;
    const bool data_step = DataStep(metadata);
    if (CardMark(metadata) == Mark::Partial && ViewOrPanel(metadata)) {
        // A view (Dashboard, Plot): its support reason is written for
        // developers ("no graph runtime ... owner is registered").
        out.runs = "Runs in part: a view you open from the node, not in training or as a data step";
        out.tone = Tone::Partial;
    } else if (CardMark(metadata) == Mark::Partial) {
        const auto* state = FindAxis(metadata, "Support State");
        out.runs = "Runs in part" + (state && !state->reason.empty() ? ": " + state->reason : std::string());
        out.tone = Tone::Partial;
    } else if (training && data_step) {
        out.runs = "Runs in training and as a data step";
        out.tone = Tone::Good;
    } else if (training) {
        out.runs = "Runs in training";
        out.tone = Tone::Good;
    } else if (data_step) {
        out.runs = "Runs as a data step";
        out.tone = Tone::Good;
    } else if (ViewOrPanel(metadata)) {
        out.runs = "Shows what reaches it";
        out.tone = Tone::Good;
    } else {
        out.runs = "Runs in the graph";
        out.tone = Tone::Good;
    }
    if (placed.place == palette::Place::Advanced) out.runs += " (a table step: " + placed.advanced_group + ")";
    out.footer = "Double-click or drag to add; click for every detail in Info";
    return out;
}

}  // namespace cyxwiz::nodes_panel
