#include "breakpoint_lines.h"

#include <algorithm>

namespace cyxwiz::editor {

void MoveBreakpoints(std::vector<scripting::DebugBreakpoint>& breakpoints, const std::vector<LineEdit>& edits,
                     int line_count) {
    for (const LineEdit& e : edits) {
        for (auto& bp : breakpoints) {
            const int l = bp.line - 1;  // 0-based
            if (e.reset) continue;
            if (l <= e.line) continue;                                   // above, or the edited line itself
            if (l > e.line + e.removed) bp.line += e.added - e.removed;  // below the replaced lines
            else bp.line = e.line + 1;                                   // on joined lines: where they went
        }
    }
    // Keep each line once (the first one there wins) and inside the text.
    std::vector<scripting::DebugBreakpoint> kept;
    for (const auto& bp : breakpoints) {
        if (bp.line < 1 || bp.line > line_count) continue;
        if (BreakpointAt(kept, bp.line)) continue;
        kept.push_back(bp);
    }
    breakpoints = std::move(kept);
}

scripting::DebugBreakpoint* BreakpointAt(std::vector<scripting::DebugBreakpoint>& breakpoints, int line) {
    for (auto& bp : breakpoints)
        if (bp.line == line) return &bp;
    return nullptr;
}

const scripting::DebugBreakpoint* BreakpointAt(const std::vector<scripting::DebugBreakpoint>& breakpoints, int line) {
    for (const auto& bp : breakpoints)
        if (bp.line == line) return &bp;
    return nullptr;
}

bool ToggleBreakpoint(std::vector<scripting::DebugBreakpoint>& breakpoints, int line) {
    auto it = std::find_if(breakpoints.begin(), breakpoints.end(), [&](const auto& bp) { return bp.line == line; });
    if (it != breakpoints.end()) {
        breakpoints.erase(it);
        return false;
    }
    scripting::DebugBreakpoint bp;
    bp.line = line;
    breakpoints.push_back(bp);
    return true;
}

}  // namespace cyxwiz::editor
