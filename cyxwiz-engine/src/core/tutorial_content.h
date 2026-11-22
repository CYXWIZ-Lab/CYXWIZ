#pragma once

// The built-in tutorials (Help > Interactive Tutorials) and the words and
// places the tutorial windows use. Pure data in and out (no ImGui), so the
// content is tested against the Engine's real windows and menus.

#include <string>
#include <vector>

namespace cyxwiz::tutorial {

struct Step {
    std::string id;
    std::string title;
    std::string description;
    std::string target;  // the window to highlight (its ImGui name or ###id); empty: none
    std::string hint;    // what to do now
};

struct Tutorial {
    std::string id;
    std::string name;
    std::string description;
    std::string category;
    std::vector<Step> steps;
};

// Getting Started, Your First Model, Training Basics: the order the list shows them.
std::vector<Tutorial> BuiltInTutorials();

// A tutorial in the list: its status and its button.
struct ListRow {
    std::string status;  // "10 steps", "In progress, step 3 of 6", "Done"
    std::string action;  // "Start", "Resume", "Start again"
    bool done = false;
};
// `current_step` is the 0-based step of the tutorial that runs, or -1.
ListRow DescribeListRow(const Tutorial& tutorial, bool done, int current_step);

struct Box {
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    float Width() const { return x1 - x0; }
    float Height() const { return y1 - y0; }
};

// Where the step card goes: beside its window (right, left, below, above:
// the first side it fits on), else over the window's top-left corner; with no
// window, in the middle of the screen. Always inside the screen.
struct Point {
    float x = 0, y = 0;
};
Point PlaceCard(const Box* target, float card_w, float card_h, float screen_w, float screen_h, float gap = 16.0f);

}  // namespace cyxwiz::tutorial
