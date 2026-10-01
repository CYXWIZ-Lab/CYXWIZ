#include "script_keys.h"

namespace cyxwiz::scriptkeys {

Action Resolve(Key key, bool ctrl, bool shift, bool alt, const State& s) {
    if (alt) return Action::None;
    const bool plain = !ctrl && !shift;

    if (key == Key::M) return ctrl && shift ? Action::ToggleNotebook : Action::None;
    if (key == Key::Space) return ctrl && !shift ? Action::Completion : Action::None;

    if (s.debugging) {
        // The debugger owns F5/F9/F10/F11 for the whole session.
        switch (key) {
            case Key::F5:
                if (!ctrl && shift) return Action::StopDebug;
                return plain && s.paused ? Action::Continue : Action::None;
            case Key::F9: return plain ? Action::ToggleBreakpoint : Action::None;
            case Key::F10: return plain && s.paused ? Action::StepOver : Action::None;
            case Key::F11:
                if (!s.paused || ctrl) return Action::None;
                return shift ? Action::StepOut : Action::StepInto;
            default: return Action::None;
        }
    }

    switch (key) {
        case Key::F5:
            if (!ctrl && shift) return s.script_running ? Action::StopScript : Action::None;
            return plain && !s.script_running ? Action::RunScript : Action::None;
        case Key::F9: return plain && !s.script_running && !s.notebook ? Action::RunSelection : Action::None;
        case Key::F10: return plain && !s.script_running ? Action::StartDebug : Action::None;
        case Key::Enter:
            return ctrl && !shift && !s.script_running && !s.notebook ? Action::RunSection : Action::None;
        default: return Action::None;
    }
}

}  // namespace cyxwiz::scriptkeys
