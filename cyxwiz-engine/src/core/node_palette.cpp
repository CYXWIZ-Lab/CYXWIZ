#include "node_palette.h"

#include "preparation_recipe.h"

namespace cyxwiz::palette {

const std::vector<std::string>& AdvancedGroups() {
    static const std::vector<std::string> groups = {"Rows", "Columns", "Text", "Combine", "SQL and checks"};
    return groups;
}

Placement PlaceNode(const NodeMetadata& metadata) {
    Placement out;
    if (metadata.IsDeprecated()) {
        out.place = Place::Hidden;
        return out;
    }
    if (!CanAddNodeToGraph(metadata)) {
        out.place = Place::CantRun;
        if (IsNodeSupportBlocked(metadata)) out.why = "Blocked";
        else if (metadata.badge == "Coming Soon" || metadata.badge.empty()) out.why = "Coming soon";
        else out.why = metadata.badge;  // "UI-only", ...
        return out;
    }
    if (const char* group = RecipeStepGroup(metadata.type)) {
        out.place = Place::Advanced;
        out.advanced_group = group;
    }
    return out;
}

}  // namespace cyxwiz::palette
