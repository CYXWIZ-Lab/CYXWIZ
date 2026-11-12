// The Studio window's words, rules and folding (core/studio_toolbar_presentation).
#include "../src/core/studio_toolbar_presentation.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

int g_checks = 0;

void Check(bool condition, const std::string& message) {
    ++g_checks;
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

bool Has(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

}  // namespace

int main() {
    using namespace cyxwiz::studio_toolbar;
    const std::string dot = " \xC2\xB7 ";

    // Compile: the live compile's latest result.
    CompileInputs c;
    c.nodes = 11;
    c.state = CompileState::Compiled;
    auto chip = CompileChip(c);
    Check(chip.label == "Ready" && chip.headline == "Ready to train" && chip.tone == Tone::Ok, "a clean compile is Ready");
    c.warnings = 2;
    Check(CompileChip(c).headline == "Ready to train, 2 warnings" && CompileChip(c).tone == Tone::Ok,
          "warnings stay Ready and are named");
    c.errors = 2;
    chip = CompileChip(c);
    Check(chip.label == "2 errors" && chip.headline == "Not ready to train: 2 errors" && chip.tone == Tone::Error,
          "errors are counted");
    c.errors = 1;
    Check(CompileChip(c).label == "1 error", "one error is singular");
    c.stale = true;
    Check(CompileChip(c).label == "Checking..." && CompileChip(c).tone == Tone::Busy,
          "an edit since the last compile is Checking");
    c = {};
    Check(CompileChip(c).label == "Empty", "an empty canvas is Empty, not Checking");
    c.nodes = 3;
    c.state = CompileState::Failed;
    Check(CompileChip(c).tone == Tone::Error && CompileChip(c).label == "Failed", "a compile that threw is Failed");
    c.errors = 3;
    Check(CompileChip(c).label == "3 errors", "an invalid graph counts its errors, not Failed");
    c.live_check_off = true;
    Check(CompileChip(c).label == "Not checked" && CompileChip(c).tone == Tone::Neutral,
          "Data Pipeline mode is not checked as you edit");

    // Local Debug: the last run against the graph now, and never two at once.
    DebugInputs d;
    Check(LocalDebugChip(d).label == "Not run" && LocalDebugChip(d).tone == Tone::Neutral, "no run yet");
    d.outcome = DebugOutcome::Passed;
    d.at = "19:12";
    chip = LocalDebugChip(d);
    Check(chip.label == "Passed 19:12" && chip.tone == Tone::Ok && LocalDebugCurrent(d), "a pass on this graph");
    d.graph_changed = true;
    chip = LocalDebugChip(d);
    Check(chip.label == "Graph changed" && chip.headline == "Passed at 19:12; the graph changed since" &&
              chip.tone == Tone::Warning && !LocalDebugCurrent(d),
          "a pass on an older graph is stale");
    d = {};
    d.outcome = DebugOutcome::Failed;
    d.at = "18:40";
    d.failure = "Failed at loss: shape [32,4] vs [32]";
    Check(LocalDebugChip(d).headline == d.failure && LocalDebugChip(d).tone == Tone::Error, "a failure says where");
    Check(LocalDebugStart(d, 12).enabled, "Local Debug can start on a graph with nodes");
    Check(!LocalDebugStart(d, 0).enabled && Has(LocalDebugStart(d, 0).reason, "canvas is empty"),
          "not on an empty canvas");
    d.running = true;
    Check(LocalDebugChip(d).tone == Tone::Busy, "a run in progress");
    Check(!LocalDebugStart(d, 12).enabled && Has(LocalDebugStart(d, 12).reason, "already going"),
          "a second Local Debug cannot start while one runs");

    // Train: enabled with the debug state in its tooltip, or disabled with the reason.
    TrainInputs t;
    Check(!TrainState(t).enabled && Has(TrainState(t).disabled_reason, "not available"), "no launcher");
    t.train_available = true;
    Check(!TrainState(t).enabled && Has(TrainState(t).disabled_reason, "not ready to train"), "an invalid graph");
    t.graph_valid = true;
    auto train = TrainState(t);
    Check(train.enabled && Has(train.tooltip, "No Local Debug (F6) on this graph yet"),
          "no debug yet: Train runs, the tooltip says so");
    t.busy = "a Data Input is loading its data";
    train = TrainState(t);
    Check(!train.enabled && train.disabled_reason == "Train waits while a Data Input is loading its data. It can start when that ends.",
          "Train waits while the graph's data loads: " + train.disabled_reason);
    t.busy.clear();
    t.debug.outcome = DebugOutcome::Passed;
    t.debug.at = "18:40";
    t.debug.graph_changed = true;
    train = TrainState(t);
    Check(train.enabled &&
              train.tooltip ==
                  "Train (F5)\nLocal Debug passed at 18:40; the graph changed since. Train runs anyway; F6 checks the "
                  "new graph first.",
          "a stale pass: " + train.tooltip);
    t.require_debug = true;
    train = TrainState(t);
    Check(!train.enabled && train.disabled_reason == "This project requires a passing Local Debug (F6) on this graph",
          "a project that requires Local Debug disables Train on a stale pass");
    t.debug.graph_changed = false;
    Check(TrainState(t).enabled && TrainState(t).tooltip == "Train (F5)", "a current pass: plain tooltip");
    TrainInputs rl;
    rl.train_available = rl.graph_valid = rl.rl_graph = true;
    train = TrainState(rl);
    Check(train.enabled && !Has(train.tooltip, "Local Debug") && Has(train.tooltip, "PPO trains the policy"),
          "an RL graph's Train tooltip says PPO trains it: " + train.tooltip);

    // The run's line, full and shortened.
    Check(TrainingRunText(3, 10, false, false, false) == "Epoch 3 / 10" &&
              TrainingRunText(3, 10, false, false, true) == "Epoch 3 / 10",
          "a running epoch");
    Check(TrainingRunText(3, 10, true, false, false) == "Paused at epoch 3 / 10" &&
              TrainingRunText(3, 10, true, false, true) == "Paused 3 / 10",
          "a paused run says so");
    Check(TrainingRunText(0, 10, false, true, false) == "Waiting for the GPU" &&
              TrainingRunText(0, 10, false, true, true) == "Waiting",
          "a run waiting for the GPU");
    Check(CompactRunText("PPO update 196" + dot + "50,176 steps" + dot + "CUDA") == "Update 196" &&
              CompactRunText("PPO waiting for the GPU") == "Waiting" && CompactRunText("Epoch 3 / 5") == "Epoch 3 / 5",
          "a PPO line shortened to its state");

    // The Code menu: the framework on the bar only in Code Gen; elsewhere the
    // menu names it.
    CodeInputs code;
    code.framework = "PyTorch";
    code.graph = "mnist_mlp";
    CodeMenu menu = CodeMenuFor(code);
    Check(menu.framework_on_bar && menu.generate_label == "Generate into Script Editor" &&
              menu.save_label == "Save as .py..." && menu.available.enabled,
          "Code Gen mode: the framework is on the bar");
    Check(menu.file_name == "mnist_mlp_pytorch.py" && menu.generate_hint == "Opens a new Script Editor tab mnist_mlp_pytorch.py" &&
              menu.save_hint == "Asks where, then writes mnist_mlp_pytorch.py",
          "the file is named after the graph and the framework");
    code.mode = Mode::DataPipeline;
    menu = CodeMenuFor(code);
    Check(!menu.framework_on_bar && menu.generate_label == "Generate PyTorch Code into Script Editor" &&
              menu.save_label == "Save PyTorch Code as .py...",
          "another mode: the menu names the framework it uses");
    code.graph.clear();
    code.problem = "The canvas is empty. Add a node first.";
    menu = CodeMenuFor(code);
    Check(!menu.available.enabled && menu.available.reason == code.problem && menu.file_name == "graph_pytorch.py",
          "a graph that cannot be turned into code: both items say why");
    CodeResult result{true, false, "PyTorch", 118, "mnist_mlp_pytorch.py", "14:32", ""};
    Check(CodeResultText(result) == "Generated 118 lines of PyTorch code into Script Editor tab mnist_mlp_pytorch.py" + dot + "14:32",
          "the result line says what went where and when: " + CodeResultText(result));
    result.to_file = true;
    result.where = "D:/Projects/mnist/scripts/mnist_mlp_pytorch.py";
    Check(CodeResultText(result) == "Saved 118 lines of PyTorch code to D:/Projects/mnist/scripts/mnist_mlp_pytorch.py" + dot + "14:32",
          "a saved file names its path");
    result = {false, false, "Keras", 0, "", "", "the graph has a cycle"};
    Check(CodeResultText(result) == "Could not generate Keras code: the graph has a cycle", "a failure says why");
    Check(std::string(ModeName(Mode::DataPipeline)) == "Data Pipeline" && std::string(ModeName(Mode::CodeGen)) == "Code Gen",
          "the mode menu shows the mode's whole name");

    // The graph menu, Save and Clear.
    GraphInputs g;
    g.nodes = 12;
    GraphMenu gm = GraphMenuFor(g, 0);
    Check(gm.load.enabled && gm.export_graph.enabled && gm.clear.enabled, "a free graph: everything");
    Check(!gm.duplicate.enabled && gm.duplicate.reason == "Select nodes first" && !gm.remove.enabled,
          "Duplicate and Delete need a selection");
    g.locked = "training is active (including pause or cleanup)";
    gm = GraphMenuFor(g, 2);
    Check(!gm.load.enabled && gm.load.reason == "Not while training is active (including pause or cleanup)." &&
              !gm.clear.enabled && gm.clear.reason == gm.load.reason && gm.remove.enabled && gm.export_graph.enabled,
          "a locked graph: no Load or Clear (they replace the graph); Delete and Export as with the keys");
    g = {};
    gm = GraphMenuFor(g, 0);
    Check(!gm.clear.enabled && gm.clear.reason == "The canvas is already empty." && !gm.export_graph.enabled,
          "an empty canvas has nothing to clear or export");
    Check(SaveRefusal(0) == "The canvas is empty. Add a node, then save." && SaveRefusal(3).empty(),
          "Save on an empty canvas says why");

    // Zoom: on the 10 % grid, held to 25 % .. 200 %.
    Check(std::fabs(ZoomStep(1.0f, 1) - 1.1f) < 1e-4f && std::fabs(ZoomStep(1.0f, -1) - 0.9f) < 1e-4f, "one step");
    Check(std::fabs(ZoomStep(1.03f, 1) - 1.1f) < 1e-4f && std::fabs(ZoomStep(1.03f, -1) - 1.0f) < 1e-4f,
          "off the grid lands on it");
    Check(std::fabs(ZoomStep(2.0f, 1) - 2.0f) < 1e-4f && std::fabs(ZoomStep(0.25f, -1) - 0.25f) < 1e-4f,
          "held to the range");
    Check(!ZoomIn(2.0f).enabled && ZoomIn(2.0f).reason == "The largest zoom (200%)" && ZoomIn(1.9f).enabled,
          "+ is off at 200 %");
    Check(!ZoomOut(0.25f).enabled && ZoomOut(0.25f).reason == "The smallest zoom (25%)" && ZoomOut(0.3f).enabled,
          "- is off at 25 %");
    Check(ZoomText(1.0f) == "100%" && ZoomText(0.25f) == "25%", "zoom text");

    // The fold: one row, items into More in a fixed order; the graph and the
    // run group never fold.
    BarWidths w;
    w.graph = 150, w.graph_icon = 40, w.run = 80, w.run_compact = 60, w.more = 70, w.save = 70, w.edit = 60;
    w.view_full = 110, w.view_short = 70, w.mode_full = 330, w.mode_menu = 120;
    w.framework = 110, w.code = 80, w.compile = 140, w.debug = 150, w.gap = 10;
    // level 0: 150+80 + 70+60+110+330+110+80+140+150 = 1280, 10 groups -> 9 gaps = 1370
    Check(BarWidth(w, 0) == 1370 && FoldLevel(w, 1400) == 0, "everything fits a wide bar");
    // level 1: the mode menu (-210) and View without the zoom (-40) = 1120
    Check(BarWidth(w, 1) == 1120 && FoldLevel(w, 1150) == 1, "first the mode becomes its menu and View loses the zoom");
    // level 2: Local Debug into More: -150 - gap + More 70 + gap = 1040
    Check(BarWidth(w, 2) == 1040 && FoldLevel(w, 1100) == 2 && ItemsInMore(w, FoldAt(2)) == 1,
          "then Local Debug goes into More");
    Check(BarWidth(w, 3) == 890 && ItemsInMore(w, FoldAt(3)) == 2, "then Compile");
    Check(BarWidth(w, 4) == 810, "then View");
    Check(BarWidth(w, 5) == 740, "then Edit");
    Check(BarWidth(w, 6) == 650, "then Code");
    Check(BarWidth(w, 7) == 530, "then the framework");
    Check(BarWidth(w, 8) == 450 && ItemsInMore(w, FoldAt(8)) == 7, "then Save: graph, mode, More and Train remain");
    Check(BarWidth(w, 9) == 430 && FoldLevel(w, 430) == 9, "then the run's line shortens");
    // level 10: the mode menu (120) and its gap go; More holds 8 items
    Check(BarWidth(w, 10) == 300 && ItemsInMore(w, FoldAt(10)) == 8, "then the mode goes into More");
    // level 11: the graph's button keeps its icon (150 -> 40): graph, More, Train
    Check(BarWidth(w, 11) == 190 && FoldLevel(w, 200) == 11 && FoldLevel(w, 100) == kMaxFold,
          "last, the graph's button is its icon; Train and More stay");
    const BarFold most = FoldAt(kMaxFold);
    Check(most.run_compact && most.save_in_more && most.debug_in_more && most.mode_in_more && most.graph_icon,
          "the narrowest fold");
    // The default dock seen on screen (2026-11-11): about 340 px for the bar
    Check(FoldLevel(w, 340) == 10, "the default dock keeps the graph's name, More and Train");
    BarWidths none = w;  // a bar without the items that would fold first
    none.debug = 0, none.compile = 0;
    Check(ItemsInMore(none, FoldAt(3)) == 0 && BarWidth(none, 3) == BarWidth(none, 1),
          "no More button while nothing it would hold is shown");
    BarWidths pipe = w;  // Data Pipeline mode: no framework, Run pipeline beside Code
    pipe.framework = 0, pipe.run_pipeline = 90;
    Check(ItemsInMore(pipe, FoldAt(6)) == 6 && ItemsInMore(pipe, FoldAt(7)) == 6,
          "Execute Pipeline folds with Code; no framework outside Code Gen");
    BarWidths sim = w;  // a simulation graph: its group in place of the mode, which never folds
    sim.mode_full = sim.mode_menu = sim.framework = sim.code = 0;
    sim.sim_full = 260, sim.sim_icons = 140;
    Check(BarWidth(sim, 0) - BarWidth(sim, 1) == 260 - 140 + 40, "the simulation group loses its words at level 1");
    BarWidths progress = w;
    progress.progress = 300;
    Check(BarWidth(progress, 9) == BarWidth(w, 9) + 310, "a running pipeline's progress never folds");

    // The window and the footer.
    Check(WindowTitle("mnist_mlp", false) == "CyxWiz Studio" + dot + "mnist_mlp" &&
              WindowTitle("mnist_mlp", true) == "CyxWiz Studio" + dot + "mnist_mlp \xE2\x80\xA2" &&
              WindowTitle("", false) == "CyxWiz Studio" + dot + "Untitled",
          "the tab names the graph and its unsaved mark");
    Check(CountsText(11, 11) == "11 nodes" + dot + "11 links" && CountsText(1, 1) == "1 node" + dot + "1 link", "counts");
    Check(SelectionText(0).empty() && SelectionText(2) == "2 selected", "selection");
    Check(FooterFor(Mode::CodeGen, "PyTorch", "").text == "Code Gen mode" + dot + "Generate writes PyTorch code" &&
              !FooterFor(Mode::CodeGen, "PyTorch", "").locked,
          "the footer says what the mode does");
    const Footer locked = FooterFor(Mode::LocalTraining, "PyTorch", "a Data Input is loading its data");
    Check(locked.locked && locked.text == "Graph replacement locked: a Data Input is loading its data", "or the lock");
    const EmptyCanvas empty;
    Check(std::string(empty.title) == "Start a graph" && Has(empty.hint, "Search nodes"), "the empty canvas says how to start");

    std::cout << "studio toolbar presentation: " << g_checks << " checks passed\n";
    return 0;
}
