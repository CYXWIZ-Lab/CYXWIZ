// Run and debug keys of the Script Editor (TOFIX133 P0 item 7). One table
// decides what a key does, so F5/F9/F10 never run twice: while a debug
// session is active the debugger's keys win (Preferences > Shortcuts,
// "Script Editor, debugging"), otherwise the run keys ("Script Editor").
#pragma once

namespace cyxwiz::scriptkeys {

enum class Key { F5, F9, F10, F11, Enter, Space, M };

enum class Action {
    None,
    RunScript,          // F5
    StopScript,         // Shift+F5 while a script runs
    RunSelection,       // F9
    RunSection,         // Ctrl+Enter (text mode)
    StartDebug,         // F10
    Continue,           // F5 while paused
    StopDebug,          // Shift+F5 while debugging
    StepOver,           // F10 while paused
    StepInto,           // F11 while paused
    StepOut,            // Shift+F11 while paused
    ToggleBreakpoint,   // F9 while debugging
    ToggleNotebook,     // Ctrl+Shift+M
    Completion          // Ctrl+Space
};

struct State {
    bool debugging = false;       // a debug session exists (running or paused)
    bool paused = false;          // stopped at a breakpoint or step
    bool script_running = false;  // a normal run is in progress
    bool notebook = false;        // notebook (cell) mode: run keys are the notebook's
};

Action Resolve(Key key, bool ctrl, bool shift, bool alt, const State& state);

// Completion popup (TOFIX133 P0 item 6, P3 board 6). Tab or Enter inserts
// the selected item; Up/Down move the selection (the editor cursor stays);
// Escape closes.
enum class PopupKey { Tab, Enter, Escape, Up, Down };
enum class PopupAction { Accept, CloseAndType, Close, Previous, Next };
PopupAction ResolvePopupKey(PopupKey key);
// Moves the selection by `delta` within `count` items, wrapping around.
int MoveSelection(int current, int delta, int count);

}  // namespace cyxwiz::scriptkeys
