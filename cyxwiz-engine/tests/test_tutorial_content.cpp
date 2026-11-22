// The built-in tutorials (TOFIX129 C6): every step points at a window the
// Engine really has, the text names today's menus and keys, the list's
// words, and the step card's place beside its window.
#include "../src/core/tutorial_content.h"

#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
}

using namespace cyxwiz::tutorial;

// The windows a step may highlight: their ImGui names as the Engine draws them
// (asset_browser.cpp, properties.cpp, console.cpp, script_editor.cpp,
// viewport.cpp, training_plot_panel.cpp; the Studio by NodeEditor::kWindowId).
const std::set<std::string> kWindows = {"", "Asset Browser", "Properties", "Console", "Script Editor",
                                        "Viewport", "Training Dashboard", "###CyxWiz Studio"};

bool Mentions(const Step& s, const char* text) {
    return s.description.find(text) != std::string::npos || s.hint.find(text) != std::string::npos ||
           s.title.find(text) != std::string::npos;
}

void CheckContent() {
    const auto all = BuiltInTutorials();
    Check(all.size() == 3, "three tutorials");
    Check(all[0].id == "getting_started" && all[1].id == "first_model" && all[2].id == "training_basics",
          "the ids the Help menu and saved progress use");
    std::set<std::string> ids;
    for (const Tutorial& t : all) {
        Check(ids.insert(t.id).second, "tutorial ids are unique: " + t.id);
        Check(!t.description.empty() && !t.category.empty(), t.id + " has a description and a category");
        Check(t.steps.size() >= 2, t.id + " has steps");
        std::set<std::string> step_ids;
        for (const Step& s : t.steps) {
            Check(step_ids.insert(s.id).second, t.id + ": step ids are unique: " + s.id);
            Check(kWindows.count(s.target) == 1, t.id + "/" + s.id + " targets a real window: " + s.target);
            Check(!Mentions(s, "Node Editor"), t.id + "/" + s.id + ": the window is CyxWiz Studio, not Node Editor");
            Check(!Mentions(s, "Ctrl+T"), t.id + "/" + s.id + ": training starts with F5, not Ctrl+T");
            Check(!Mentions(s, "Dataset >"), t.id + "/" + s.id + ": the menu is Data, not Dataset");
            Check(!Mentions(s, "Training >"), t.id + "/" + s.id + ": there is no Training menu");
            Check(!s.title.empty() && !s.description.empty() && !s.hint.empty(), t.id + "/" + s.id + " is complete");
            for (const char c : s.description + s.title + s.hint)
                Check(static_cast<unsigned char>(c) < 128, t.id + "/" + s.id + ": ASCII text (MSVC reads sources in the ANSI page)");
        }
        Check(t.steps.back().target.empty() && t.steps.back().hint.find("Complete") != std::string::npos,
              t.id + " ends on a Complete step");
    }
    const Step& menus = all[0].steps[1];
    for (const char* menu : {"File", "Edit", "View", "Nodes", "Train", "Data", "Tools", "Script", "Deploy", "Simulation",
                             "Apps", "Help"})
        Check(menus.description.find(menu) != std::string::npos, std::string("the menu-bar step names ") + menu);
    Check(Mentions(all[2].steps[4], "F5"), "Training Basics starts training with F5");
    Check(all[2].steps[2].target == "###CyxWiz Studio", "the optimizer step highlights CyxWiz Studio");
}

void CheckListRows() {
    const Tutorial t = BuiltInTutorials()[2];
    const ListRow fresh = DescribeListRow(t, false, -1);
    Check(fresh.status == "6 steps" && fresh.action == "Start" && !fresh.done, "not started: its steps, Start");
    const ListRow running = DescribeListRow(t, false, 2);
    Check(running.status == "In progress, step 3 of 6" && running.action == "Resume", "in progress: Resume");
    const ListRow done = DescribeListRow(t, true, -1);
    Check(done.status == "Done" && done.action == "Start again" && done.done, "done: Start again");
    Check(DescribeListRow(t, true, 0).action == "Resume", "a done tutorial running again: Resume");
}

void CheckPlacement() {
    const float sw = 1920, sh = 1080, w = 380, h = 240;
    const Point centre = PlaceCard(nullptr, w, h, sw, sh);
    Check(centre.x == (sw - w) / 2 && centre.y == (sh - h) / 2, "no window: the middle of the screen");

    const Box asset{0, 48, 190, 1010};  // the Asset Browser at the left edge
    const Point right = PlaceCard(&asset, w, h, sw, sh);
    Check(right.x == 190 + 16 && right.y == 48, "a window at the left: the card beside it on the right");

    const Box properties{1610, 48, 1880, 580};  // Properties at the right
    const Point left = PlaceCard(&properties, w, h, sw, sh);
    Check(left.x == 1610 - 16 - w && left.y == 48, "a window at the right: the card on its left");

    const Box bottom{190, 600, 1610, 1010};  // a wide bottom dock with room above
    const Point above = PlaceCard(&bottom, 1500, h, sw, sh);
    Check(above.y == 600 - 16 - h, "a wide card over a bottom window: above it");

    const Box full{0, 0, 1920, 1080};
    const Point inside = PlaceCard(&full, w, h, sw, sh);
    Check(inside.x >= 8 && inside.y >= 8 && inside.x + w <= sw - 8 && inside.y + h <= sh - 8,
          "a window filling the screen: the card stays inside it");

    const Box low{100, 1000, 300, 1070};
    const Point kept = PlaceCard(&low, w, h, sw, sh);
    Check(kept.y + h <= sh - 8, "the card never runs off the bottom (today it did)");
}

}  // namespace

int main() {
    CheckContent();
    CheckListRows();
    CheckPlacement();
    if (failures == 0) std::printf("test_tutorial_content: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
