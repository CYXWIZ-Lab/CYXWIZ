#pragma once

// Breakpoints follow their lines while the text is edited (TOFIX133 P6,
// board 12): lines inserted or removed above a breakpoint move it; a
// breakpoint on lines that were joined goes to the line they became; two
// that land on one line become one. Pure, no ImGui.

#include "text_document.h"
#include "../../scripting/debug_types.h"

#include <vector>

namespace cyxwiz::editor {

// `breakpoints` hold 1-based lines; `line_count` is the document's line
// count after the edits (a reset keeps those still inside it).
void MoveBreakpoints(std::vector<scripting::DebugBreakpoint>& breakpoints, const std::vector<LineEdit>& edits,
                     int line_count);

// The breakpoint on a 1-based line, or null.
scripting::DebugBreakpoint* BreakpointAt(std::vector<scripting::DebugBreakpoint>& breakpoints, int line);
const scripting::DebugBreakpoint* BreakpointAt(const std::vector<scripting::DebugBreakpoint>& breakpoints, int line);

// Adds a plain breakpoint on the line, or removes the one there. True when added.
bool ToggleBreakpoint(std::vector<scripting::DebugBreakpoint>& breakpoints, int line);

}  // namespace cyxwiz::editor
