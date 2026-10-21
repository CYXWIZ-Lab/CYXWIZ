#pragma once

// Where the palettes put a node (TOFIX157, the owner's design decision of
// 2026-09-24, data_studio_design.md 10.1; approved mockup): the Nodes panel,
// the canvas search and the right-click menu all ask PlaceNode, so they agree.
//
// - A node that runs shows in its category, as before.
// - A table step (a Preparation Recipe step: filter, sort, columns, text,
//   joins, SQL, checks) shows under "Advanced operations", by its kind (Rows,
//   Columns, Text, Combine, SQL and checks). Data Studio > Prepare adds them.
// - A node that cannot run yet (a template, a blocked node, a preview) shows
//   under "Can't run yet" with why, so what is missing stays in sight; it
//   leaves that list when it runs.
// - A deprecated node is not offered (saved graphs still load it).

#include "node_metadata.h"

#include <string>
#include <vector>

namespace cyxwiz::palette {

enum class Place { Category, Advanced, CantRun, Hidden };

struct Placement {
    Place place = Place::Category;
    std::string advanced_group;  // Advanced: "Rows", "Columns", ...
    std::string why;             // CantRun: "Coming soon", "Blocked", "UI-only", ...
};

Placement PlaceNode(const NodeMetadata& metadata);

// The Advanced operations groups, in the order they are shown.
const std::vector<std::string>& AdvancedGroups();

inline constexpr const char* kAdvancedOperations = "Advanced operations";
inline constexpr const char* kCantRunYet = "Can't run yet";

}  // namespace cyxwiz::palette
