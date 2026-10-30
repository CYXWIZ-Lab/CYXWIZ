#pragma once

// The CyxWiz Studio canvas bar's words and folding (TOFIX178 UI-1, approved
// mockup https://claude.ai/artifact/3NUdq1C7mfW2QhiG6NTBD8). Pure: no ImGui.
// Compile and Local Debug each show their last result on the bar; Train
// says when Local Debug is stale and is disabled when the project requires
// a passing one. The bar folds by width only, in a fixed order, and never
// into a menu inside a menu.

#include <cstddef>
#include <string>

namespace cyxwiz::studio_toolbar {

enum class Tone { Neutral, Busy, Ok, Warning, Error };

// The live compile's state (LiveGraphCompile::State), mirrored so this
// header needs no compiler headers.
enum class CompileState { NotCompiled, Compiling, Compiled, Failed };

// A step's last result, as a chip on its button and the head of its menu.
struct StepChip {
    std::string label;        // on the bar: "Ready", "2 errors", "Passed 19:12"
    std::string short_label;  // narrower bars: "Ready", "Changed"
    std::string headline;     // its menu: "Ready to train", "Passed at 18:40; the graph changed since"
    Tone tone = Tone::Neutral;
};

struct CompileInputs {
    CompileState state = CompileState::NotCompiled;
    // The canvas changed after the latest finished compile (one is waiting
    // for the edit pause or running).
    bool stale = false;
    int errors = 0;
    int warnings = 0;
    int nodes = 0;
    // Data Pipeline mode: the graph is not checked as you edit (a data graph
    // has no model); Compile (F7) checks it on request.
    bool live_check_off = false;
};
StepChip CompileChip(const CompileInputs& in);

enum class DebugOutcome { NotRun, Passed, Failed };

struct DebugInputs {
    DebugOutcome outcome = DebugOutcome::NotRun;
    // The canvas differs from the graph the last run used.
    bool graph_changed = false;
    bool running = false;
    std::string at;       // the last run's time, "19:12"
    std::string failure;  // why it failed: "Failed at loss: shape [32,4] vs [32]"
};
StepChip LocalDebugChip(const DebugInputs& in);
// A passed run on the graph as it is now.
bool LocalDebugCurrent(const DebugInputs& in);

struct TrainInputs {
    bool train_available = false;  // a training launcher is wired
    bool graph_valid = false;      // the canvas has a dataset, layers and a loss
    bool require_debug = false;    // EngineConfig require_debug_before_train
    DebugInputs debug;
};
struct TrainButton {
    bool enabled = false;
    std::string disabled_reason;  // when not enabled
    std::string tooltip;          // when enabled
};
TrainButton TrainState(const TrainInputs& in);

// The width of each group at each of its forms, measured by the renderer.
struct BarWidths {
    float fixed = 0.0f;  // never folds: graph, Save, the run group's Train / epoch + Stop, pipeline progress
    float edit_icons = 0.0f, edit_menu = 0.0f;
    float view_full = 0.0f, view_menu = 0.0f;
    float mode_full = 0.0f, mode_short = 0.0f, mode_menu = 0.0f;
    float actions_full = 0.0f, actions_icons = 0.0f;
    float steps_full = 0.0f, steps_short = 0.0f, steps_dots = 0.0f;
    float gaps = 0.0f;  // between the groups
};
// 0 full; 1 labels go (short mode names, action icons, short step chips);
// 2 Edit and View become menus; 3 the mode becomes a menu; 4 the step
// chips become dots and the graph its file stem. The lowest level that
// fits; 4 when none does.
int FoldLevel(const BarWidths& w, float available);
float BarWidth(const BarWidths& w, int level);

// The canvas footer: the counts and what the canvas is doing.
std::string CountsText(size_t nodes, size_t links);
std::string SelectionText(int selected);  // "" when nothing is selected

}  // namespace cyxwiz::studio_toolbar
