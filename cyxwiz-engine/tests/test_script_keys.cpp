// Script Editor run/debug keys (TOFIX133 P0 item 7): each key does one
// thing; the debugger's keys apply only during a debug session.
#include "../src/core/script_keys.h"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz::scriptkeys;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}
}  // namespace

int main() {
    State idle;
    State running;
    running.script_running = true;
    State debug_running;
    debug_running.debugging = true;
    State paused;
    paused.debugging = true;
    paused.paused = true;
    State notebook;
    notebook.notebook = true;

    // Without a debug session F5/F9/F10 are the run keys, once each.
    Check(Resolve(Key::F5, false, false, false, idle) == Action::RunScript, "F5 runs");
    Check(Resolve(Key::F9, false, false, false, idle) == Action::RunSelection, "F9 runs selection");
    Check(Resolve(Key::F10, false, false, false, idle) == Action::StartDebug, "F10 starts debugging");
    Check(Resolve(Key::F5, false, true, false, running) == Action::StopScript, "Shift+F5 stops a run");
    Check(Resolve(Key::F5, false, false, false, running) == Action::None, "F5 does not start a second run");

    // During a session the debugger's keys win; nothing runs twice.
    Check(Resolve(Key::F5, false, false, false, paused) == Action::Continue, "F5 continues when paused");
    Check(Resolve(Key::F5, false, false, false, debug_running) == Action::None, "F5 while running under debug: nothing");
    Check(Resolve(Key::F9, false, false, false, paused) == Action::ToggleBreakpoint, "F9 toggles a breakpoint");
    Check(Resolve(Key::F10, false, false, false, paused) == Action::StepOver, "F10 steps over");
    Check(Resolve(Key::F11, false, false, false, paused) == Action::StepInto, "F11 steps into");
    Check(Resolve(Key::F11, false, true, false, paused) == Action::StepOut, "Shift+F11 steps out");
    Check(Resolve(Key::F5, false, true, false, debug_running) == Action::StopDebug, "Shift+F5 stops debugging");
    Check(Resolve(Key::F11, false, false, false, idle) == Action::None, "F11 outside a session: nothing");

    // Notebook mode: text-mode run keys are off; M needs Ctrl+Shift.
    Check(Resolve(Key::F9, false, false, false, notebook) == Action::None, "no Run Selection in notebook mode");
    Check(Resolve(Key::Enter, true, false, false, notebook) == Action::None, "no Run Section in notebook mode");
    Check(Resolve(Key::M, true, true, false, notebook) == Action::ToggleNotebook, "Ctrl+Shift+M toggles notebook");
    Check(Resolve(Key::M, false, false, false, notebook) == Action::None, "plain M is the notebook's own key");
    Check(Resolve(Key::Enter, true, false, false, idle) == Action::RunSection, "Ctrl+Enter runs the section");
    Check(Resolve(Key::Space, true, false, false, idle) == Action::Completion, "Ctrl+Space completes");
    Check(Resolve(Key::F5, false, false, true, idle) == Action::None, "Alt combinations are not ours");

    std::cout << "script keys: run keys, debug session keys, notebook mode. OK\n";
    return 0;
}
