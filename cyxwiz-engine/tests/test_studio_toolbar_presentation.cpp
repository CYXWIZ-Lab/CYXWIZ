// The Studio canvas bar's words and folding (TOFIX178 UI-1).
#include "../src/core/studio_toolbar_presentation.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

}  // namespace

int main() {
    using namespace cyxwiz::studio_toolbar;

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

    // Local Debug: the last run against the graph now.
    DebugInputs d;
    Check(LocalDebugChip(d).label == "Not run" && LocalDebugChip(d).tone == Tone::Neutral, "no run yet");
    d.outcome = DebugOutcome::Passed;
    d.at = "19:12";
    chip = LocalDebugChip(d);
    Check(chip.label == "Passed 19:12" && chip.short_label == "Passed" && chip.tone == Tone::Ok && LocalDebugCurrent(d),
          "a pass on this graph");
    d.graph_changed = true;
    chip = LocalDebugChip(d);
    Check(chip.label == "Graph changed" && chip.short_label == "Changed" &&
              chip.headline == "Passed at 19:12; the graph changed since" && chip.tone == Tone::Warning &&
              !LocalDebugCurrent(d),
          "a pass on an older graph is stale");
    d = {};
    d.outcome = DebugOutcome::Failed;
    d.at = "18:40";
    d.failure = "Failed at loss: shape [32,4] vs [32]";
    Check(LocalDebugChip(d).headline == d.failure && LocalDebugChip(d).tone == Tone::Error, "a failure says where");
    d.running = true;
    Check(LocalDebugChip(d).tone == Tone::Busy, "a run in progress");

    // Train: enabled with the debug state in its tooltip, or disabled with the reason.
    TrainInputs t;
    Check(!TrainState(t).enabled && TrainState(t).disabled_reason.find("not available") != std::string::npos,
          "no launcher");
    t.train_available = true;
    Check(!TrainState(t).enabled && TrainState(t).disabled_reason.find("not ready to train") != std::string::npos,
          "an invalid graph");
    t.graph_valid = true;
    auto train = TrainState(t);
    Check(train.enabled && train.tooltip.find("No Local Debug (F6) on this graph yet") != std::string::npos,
          "no debug yet: Train runs, the tooltip says so");
    t.debug.outcome = DebugOutcome::Passed;
    t.debug.at = "18:40";
    t.debug.graph_changed = true;
    train = TrainState(t);
    Check(train.enabled &&
              train.tooltip ==
                  "Train (F5)\nLocal Debug passed at 18:40; the graph changed since. Train runs anyway; F6 checks the "
                  "new graph first.",
          "a stale pass, as the mockup words it: " + train.tooltip);
    t.require_debug = true;
    train = TrainState(t);
    Check(!train.enabled && train.disabled_reason == "This project requires a passing Local Debug (F6) on this graph",
          "a project that requires Local Debug disables Train on a stale pass");
    t.debug.graph_changed = false;
    Check(TrainState(t).enabled && TrainState(t).tooltip == "Train (F5)", "a current pass: plain tooltip");

    // Folding: labels first, then Edit / View menus, then the mode menu.
    BarWidths w;
    w.fixed = 300;
    w.edit_icons = 90, w.edit_menu = 60;
    w.view_full = 160, w.view_menu = 110;
    w.mode_full = 330, w.mode_short = 220, w.mode_menu = 100;
    w.actions_full = 200, w.actions_icons = 70;
    w.steps_full = 330, w.steps_short = 190, w.steps_dots = 70;
    w.gaps = 50;
    Check(BarWidth(w, 0) == 1460 && FoldLevel(w, 1500) == 0, "everything fits a wide bar");
    Check(BarWidth(w, 1) == 1080 && FoldLevel(w, 1100) == 1, "labels go first");
    Check(BarWidth(w, 2) == 1000 && FoldLevel(w, 1000) == 2, "then Edit and View become menus");
    Check(BarWidth(w, 3) == 880 && FoldLevel(w, 900) == 3, "then the mode is a menu");
    Check(FoldLevel(w, 760) == 4 && BarWidth(w, 4) == 760, "then the steps are dots");
    Check(FoldLevel(w, 300) == 4, "the narrowest form when nothing fits");

    // The footer.
    Check(CountsText(11, 11) == "11 nodes \xC2\xB7 11 links" && CountsText(1, 1) == "1 node \xC2\xB7 1 link",
          "counts");
    Check(SelectionText(0).empty() && SelectionText(2) == "2 selected", "selection");

    std::cout << "Studio toolbar presentation tests passed\n";
    return 0;
}
