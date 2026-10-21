// Breakpoints follow their lines while the text is edited (TOFIX133 P6).
// Edits are made on a real Document, so the line edits are the editor's own.
#include "../src/core/editor/breakpoint_lines.h"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz::editor;
using BP = scripting::DebugBreakpoint;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::vector<BP> Lines(std::initializer_list<int> lines) {
    std::vector<BP> out;
    for (int l : lines) {
        BP bp;
        bp.line = l;
        out.push_back(bp);
    }
    return out;
}

std::string Joined(const std::vector<BP>& bps) {
    std::string s;
    for (const auto& b : bps) s += std::to_string(b.line) + " ";
    return s;
}

const char* kText = "a = 1\nb = 2\nc = 3\nd = 4\ne = 5\nf = 6\n";
}  // namespace

int main() {
    {  // Enter at the end of line 2: breakpoints below move down one.
        Document d;
        d.SetText(kText);
        d.TakeLineEdits();
        auto bps = Lines({2, 4, 6});
        bps[1].condition = "d > 3";
        d.SetCursor(Pos{1, 5});
        d.Newline();
        MoveBreakpoints(bps, d.TakeLineEdits(), d.LineCount());
        Check(Joined(bps) == "2 5 7 ", "newline below line 2: " + Joined(bps));
        Check(bps[1].condition == "d > 3", "settings move with the breakpoint");
    }
    {  // Deleting lines 3-4 (selected whole): 4 goes to 3, 6 moves up two.
        Document d;
        d.SetText(kText);
        d.TakeLineEdits();
        auto bps = Lines({1, 4, 6});
        d.SetSelections({Selection{Pos{2, 0}, Pos{4, 0}, -1}});
        d.Paste("");
        MoveBreakpoints(bps, d.TakeLineEdits(), d.LineCount());
        Check(Joined(bps) == "1 3 4 ", "lines removed above: " + Joined(bps));
    }
    {  // Joining line 3 into 2 (Backspace at its start) with breakpoints on both: one stays.
        Document d;
        d.SetText(kText);
        d.TakeLineEdits();
        auto bps = Lines({2, 3, 5});
        d.SetCursor(Pos{2, 0});
        d.Backspace();
        MoveBreakpoints(bps, d.TakeLineEdits(), d.LineCount());
        Check(Joined(bps) == "2 4 ", "joined lines keep one breakpoint: " + Joined(bps));
    }
    {  // Typing inside a line moves nothing; a reload keeps lines still in the text.
        Document d;
        d.SetText(kText);
        d.TakeLineEdits();
        auto bps = Lines({3, 7});
        d.SetCursor(Pos{2, 1});
        d.Type("x");
        MoveBreakpoints(bps, d.TakeLineEdits(), d.LineCount());
        Check(Joined(bps) == "3 7 ", "typing: " + Joined(bps));
        d.SetText("only\ntwo\n");
        MoveBreakpoints(bps, d.TakeLineEdits(), d.LineCount());
        Check(Joined(bps) == "3 ", "reload drops lines past the end (3 lines: the last empty): " + Joined(bps));
    }
    {
        std::vector<BP> bps;
        Check(ToggleBreakpoint(bps, 4) && bps.size() == 1, "toggle adds");
        Check(BreakpointAt(bps, 4) != nullptr && BreakpointAt(bps, 5) == nullptr, "breakpoint at");
        Check(!ToggleBreakpoint(bps, 4) && bps.empty(), "toggle removes");
    }
    std::cout << "breakpoint lines: newline, removed lines, joined lines, typing, reload, toggle. OK\n";
    return 0;
}
