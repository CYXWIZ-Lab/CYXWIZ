#pragma once

// The CyxWiz Studio window's words, rules and folding: its bar, its title
// and its footer. Pure: no ImGui.
//
// The bar (TOFIX129 A3, approved mockup
// https://claude.ai/artifact/9nAjNG1P6NWCFm9Sf92vT5) is one row. Compile and
// Local Debug each show their last result; Train says when Local Debug is
// stale and is disabled when the project requires a passing one or while the
// graph's data is still loading. When the row is too narrow its items fold
// into one "More" menu in a fixed order; the graph and the run group (Train,
// or a run's line with Pause / Resume and Stop) never fold.

#include <cstddef>
#include <string>

namespace cyxwiz::studio_toolbar {

enum class Tone { Neutral, Busy, Ok, Warning, Error };

// The live compile's state (LiveGraphCompile::State), mirrored so this
// header needs no compiler headers.
enum class CompileState { NotCompiled, Compiling, Compiled, Failed };

// A step's last result, as a chip on its button and the head of its menu.
struct StepChip {
    std::string label;     // on the bar: "Ready", "2 errors", "Passed 19:12"
    std::string headline;  // its menu: "Ready to train", "Passed at 18:40; the graph changed since"
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

// Whether an action can start now, and why not.
struct Availability {
    bool enabled = true;
    std::string reason;  // when not enabled
};
// Local Debug (the chip's label, its menu, F6): never a second run while one goes.
Availability LocalDebugStart(const DebugInputs& in, size_t nodes);

struct TrainInputs {
    bool train_available = false;  // a training launcher is wired
    bool graph_valid = false;      // the canvas has a dataset, layers and a loss
    bool require_debug = false;    // EngineConfig require_debug_before_train
    bool rl_graph = false;         // an RL graph: PPO trains it, Local Debug does not apply
    // What keeps the graph busy before a run can start ("a Data Input is
    // loading its data"); empty when nothing does.
    std::string busy;
    DebugInputs debug;
};
struct TrainButton {
    bool enabled = false;
    std::string disabled_reason;  // when not enabled
    std::string tooltip;          // when enabled
};
TrainButton TrainState(const TrainInputs& in);

// A supervised run's line on the bar: "Epoch 3 / 10", "Paused at epoch 3 / 10",
// "Waiting for the GPU" (compact: "Waiting").
std::string TrainingRunText(int epoch, int total_epochs, bool paused, bool waiting_for_gpu, bool compact);
// A PPO run's line shortened to its state: its first part without "PPO ",
// a wait as "Waiting", first letter capitalised ("PPO update 196 - 50,176
// steps - CUDA" -> "Update 196"; "PPO waiting for the GPU" -> "Waiting").
std::string CompactRunText(const std::string& run_line);

// ---- The execution mode, the framework and the Code menu.

enum class Mode { CodeGen, DataPipeline, LocalTraining };
const char* ModeName(Mode mode);  // "Code Gen", "Data Pipeline", "Local Training"

struct CodeInputs {
    Mode mode = Mode::CodeGen;
    std::string framework;  // "PyTorch"
    std::string graph;      // the graph's file stem, "mnist_mlp"; "" before the first save
    // Why the graph cannot be turned into code ("The canvas is empty...");
    // empty when it can.
    std::string problem;
};
struct CodeMenu {
    // The framework box is on the bar in Code Gen mode only; elsewhere the
    // Code menu names the framework, so code is never written for one the
    // user cannot see.
    bool framework_on_bar = true;
    std::string generate_label;  // "Generate into Script Editor", "Generate PyTorch Code into Script Editor"
    std::string generate_hint;   // "Opens a new Script Editor tab mnist_mlp_pytorch.py"
    std::string save_label;      // "Save as .py...", "Save PyTorch Code as .py..."
    std::string save_hint;       // "Asks where, then writes mnist_mlp_pytorch.py"
    std::string file_name;       // "mnist_mlp_pytorch.py"
    Availability available;
};
CodeMenu CodeMenuFor(const CodeInputs& in);

// The line under the bar after Generate or Save as .py.
struct CodeResult {
    bool ok = false;
    bool to_file = false;
    std::string framework;
    int lines = 0;
    std::string where;  // the Script Editor tab or the file's path
    std::string at;     // "14:32"
    std::string error;  // when not ok
};
std::string CodeResultText(const CodeResult& result);

// ---- The graph menu, Save and Clear.

struct GraphInputs {
    size_t nodes = 0;
    // What keeps the graph from being replaced (GraphReplacementBlockReason:
    // "training is active ..."); empty when nothing does.
    std::string locked;
};
struct GraphMenu {
    Availability load;          // a graph replaced now would lose what is running
    Availability export_graph;  // a copy of the graph file
    Availability clear;
    Availability duplicate;
    Availability remove;  // Delete selected
};
GraphMenu GraphMenuFor(const GraphInputs& in, int selected);
// Save on an empty canvas says why nothing was written; empty when Save can go.
std::string SaveRefusal(size_t nodes);

// ---- Zoom (the View menu, + / -, the wheel).
constexpr float kZoomMin = 0.25f;
constexpr float kZoomMax = 2.0f;
constexpr float kZoomStep = 0.1f;
// One step in or out, held to the range; a value off the step grid lands on it.
float ZoomStep(float zoom, int direction);
Availability ZoomIn(float zoom);
Availability ZoomOut(float zoom);
std::string ZoomText(float zoom);  // "100%"

// ---- The bar's fold (A3-1).

// The width of each item, measured by the renderer; 0 for an item the bar
// does not show now (the framework outside Code Gen mode).
struct BarWidths {
    float graph = 0.0f;       // the graph's name and menu
    float graph_icon = 0.0f;  // the same as its icon and unsaved mark (the tab names the graph)
    float run = 0.0f;         // the run group: Train, or a run's line with Pause / Resume and Stop; never folds
    float run_compact = 0.0f;  // the run's line as its state ("Epoch 3 / 10", "Update 196")
    float progress = 0.0f;    // a running pipeline's progress and Cancel; never folds
    float more = 0.0f;        // the More button
    float save = 0.0f;
    float edit = 0.0f;
    float view_full = 0.0f, view_short = 0.0f;  // "View - 100%", "View"
    float mode_full = 0.0f, mode_menu = 0.0f;   // the segmented control, "Code Gen" with its menu
    float framework = 0.0f;
    float code = 0.0f;
    float run_pipeline = 0.0f;  // Execute Pipeline, in Data Pipeline mode
    float sim_full = 0.0f, sim_icons = 0.0f;  // a simulation graph's group in place of the mode
    float compile = 0.0f, debug = 0.0f;
    float gap = 0.0f;  // between two groups
};

// What one fold level shows: lower levels show more.
// 0 everything; 1 the mode becomes its menu, View loses the zoom, the
// simulation group its words; then into More, one per level: 2 Local Debug,
// 3 Compile, 4 View, 5 Edit, 6 Code (and Execute Pipeline), 7 the framework,
// 8 Save; 9 the run's line shortens to its state; 10 the mode goes into
// More; 11 the graph's button keeps its icon and unsaved mark (the window's
// tab names the graph). Train (or the run group) never folds.
struct BarFold {
    int level = 0;
    bool mode_menu = false, view_short = false, sim_icons = false;
    bool debug_in_more = false, compile_in_more = false, view_in_more = false, edit_in_more = false;
    bool code_in_more = false, framework_in_more = false, save_in_more = false;
    bool run_compact = false;
    bool mode_in_more = false;
    bool graph_icon = false;
};
constexpr int kMaxFold = 11;
BarFold FoldAt(int level);
// Items in More at this fold (only items the bar shows count); More is drawn when > 0.
int ItemsInMore(const BarWidths& w, const BarFold& fold);
float BarWidth(const BarWidths& w, int level);
// The lowest level that fits; kMaxFold when none does.
int FoldLevel(const BarWidths& w, float available);

// ---- The window and the footer.

// The window's tab (and its own window's title): "CyxWiz Studio · mnist_mlp",
// a dot after it while there are unsaved changes; "Untitled" before the first save.
std::string WindowTitle(const std::string& graph, bool unsaved);
// "11 nodes - 11 links", "2 selected"
std::string CountsText(size_t nodes, size_t links);
std::string SelectionText(int selected);  // "" when nothing is selected
struct Footer {
    std::string text;
    bool locked = false;
};
// The right side: the lock, or what the mode does.
Footer FooterFor(Mode mode, const std::string& framework, const std::string& locked);

// ---- The empty canvas (A3-3).
struct EmptyCanvas {
    const char* title = "Start a graph";
    const char* hint = "Drag a node from the Nodes panel, type a name in Search nodes, or start from an example graph.";
};

}  // namespace cyxwiz::studio_toolbar
