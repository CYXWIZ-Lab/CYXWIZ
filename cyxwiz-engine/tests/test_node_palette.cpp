// Where the palettes put a node (TOFIX157, palette curation): a node that
// runs stays in its category; a recipe step type goes under Advanced
// operations by its kind; a node that cannot run yet goes under "Can't run
// yet" with why; a deprecated one is not offered.
#include "../src/core/node_palette.h"
#include "../src/core/preparation_recipe.h"

#include <cstdio>
#include <string>

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        ++failures;
    }
}

cyxwiz::NodeMetadata Node(gui::NodeType type) {
    cyxwiz::NodeMetadata m;
    m.type = type;
    m.status = cyxwiz::NodeImplementationStatus::Implemented;
    return m;
}

}  // namespace

int main() {
    using cyxwiz::palette::Place;
    using cyxwiz::palette::PlaceNode;

    Check(PlaceNode(Node(gui::NodeType::DataInput)).place == Place::Category, "a source stays in its category");
    Check(PlaceNode(Node(gui::NodeType::TestEvaluationNode)).place == Place::Category, "an evaluation node stays");

    const auto filter = PlaceNode(Node(gui::NodeType::FilterRows));
    Check(filter.place == Place::Advanced && filter.advanced_group == "Rows", "Row Filter: Advanced, Rows");
    Check(PlaceNode(Node(gui::NodeType::SelectColumns)).advanced_group == "Columns", "Column Filter: Columns");
    Check(PlaceNode(Node(gui::NodeType::TextCleanNode)).advanced_group == "Text", "Text Clean: Text");
    Check(PlaceNode(Node(gui::NodeType::JoinTables)).advanced_group == "Combine", "Joiner: Combine");
    Check(PlaceNode(Node(gui::NodeType::SQLQuery)).advanced_group == "SQL and checks", "SQL Query: SQL and checks");
    Check(PlaceNode(Node(gui::NodeType::RowCountCheck)).advanced_group == "SQL and checks", "Check: SQL and checks");
    Check(cyxwiz::IsPreparationRecipeStepType(gui::NodeType::DataValidator) &&
              !cyxwiz::IsPreparationRecipeStepType(gui::NodeType::DataInput),
          "a recipe step is exactly a type with a group");

    auto preview = Node(gui::NodeType::Dense);
    preview.status = cyxwiz::NodeImplementationStatus::Template;
    preview.badge = "Coming Soon";
    const auto soon = PlaceNode(preview);
    Check(soon.place == Place::CantRun && soon.why == "Coming soon", "a preview: can't run yet, coming soon");
    preview.badge = "UI-only";
    Check(PlaceNode(preview).why == "UI-only", "a UI-only template says so");

    auto blocked = Node(gui::NodeType::Dense);
    blocked.badge = "Blocked";
    Check(PlaceNode(blocked).place == Place::CantRun && PlaceNode(blocked).why == "Blocked", "blocked: can't run yet");

    // A step kind that cannot run is under Can't run yet, not Advanced operations.
    auto blocked_step = Node(gui::NodeType::Unpivot);
    blocked_step.status = cyxwiz::NodeImplementationStatus::Template;
    Check(PlaceNode(blocked_step).place == Place::CantRun, "a step that can't run: can't run yet");

    auto old = Node(gui::NodeType::Dense);
    old.status = cyxwiz::NodeImplementationStatus::Deprecated;
    Check(PlaceNode(old).place == Place::Hidden, "deprecated: not offered");

    Check(cyxwiz::palette::AdvancedGroups().size() == 5 && cyxwiz::palette::AdvancedGroups().front() == "Rows",
          "five groups, Rows first");
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("node palette: categories, advanced operations, can't run yet. OK\n");
    return 0;
}
