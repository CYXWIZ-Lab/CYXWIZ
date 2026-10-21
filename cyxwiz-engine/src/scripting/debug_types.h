#pragma once

// Script Editor debugger data (TOFIX133 P6): what a debug run is given and
// what the UI reads back. Plain data, shared by the engine, the notebook
// cell manager and the Script Editor.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace scripting {

struct DebugBreakpoint {
    int line = 0;           // 1-based
    std::string condition;  // Python expression; empty: always
    int hit = 0;            // stop from this hit on; 0: every hit
    bool enabled = true;
};

struct DebugFrame {
    std::string name;
    std::string file;
    int line = 0;
};

// The debugger as the UI sees it, updated by the run's worker thread.
struct DebugSnapshot {
    std::string state = "idle";       // idle, running, paused
    std::string reason;               // breakpoint, step, error, pause
    std::string error;                // the error, or why a condition failed
    std::string file;                 // the debug run's file name
    std::vector<DebugFrame> stack;    // innermost first
    std::map<int, int> hits;          // line -> hits in `file`
    std::uint64_t version = 0;
};

}  // namespace scripting
