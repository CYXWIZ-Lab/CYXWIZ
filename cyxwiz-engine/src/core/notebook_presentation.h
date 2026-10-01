// Notebook wording and states for the Script Editor (TOFIX133 P4 step 4.3,
// approved boards 4-5). Pure data: the renderer only draws what this says.
#pragma once

#include <string>
#include <vector>

namespace cyxwiz::nbview {

enum class Tone { Muted, Success, Error, Running, Warning };

enum class CellRun { Idle, Queued, Running, Success, Error, NotRun };

// "0.02 s", "2.04 s", "12.3 s", "1 min 4 s".
std::string FormatDuration(double seconds);

// The column left of a code cell: [n] and, below it, how the last run went.
struct Gutter {
    enum class Mark { None, Check, Cross };  // drawn as icons before `detail`
    std::string label;   // "[3]", "[*]", "[ ]"
    std::string detail;  // "0.02 s" (with Check), "2.04 s" (with Cross), "3.2 s", "Queued", ""
    Mark mark = Mark::None;
    Tone tone = Tone::Muted;
};
// `seconds` is the last run's time (< 0: not timed); `running_seconds` the
// time so far while it runs.
Gutter GutterFor(CellRun state, int execution_count, double seconds, double running_seconds);

// Kernel chip at the right of the toolbar: "Python 3.12.8 · sentiment · Busy".
struct KernelChip {
    std::string text;
    std::string tooltip;
    Tone tone = Tone::Success;
};
struct KernelFacts {
    std::string python_version;   // "3.12.8"; empty before Python starts
    std::string environment;      // "sentiment" (project) or "system Python"
    bool started = false;         // Python is running in the Engine
    bool busy = false;            // this notebook runs a cell
    bool other_busy = false;      // another script holds Python
    bool restarting = false;
};
KernelChip KernelChipFor(const KernelFacts& facts);

// Left of the status bar: what the notebook is doing.
struct RunFacts {
    bool running = false;
    int run_position = 0;        // 1-based cell being run in this batch
    int run_total = 0;           // cells in this batch
    int stopped_at_count = 0;    // [n] of the cell that stopped the last batch (0 = none)
    bool stopped_by_interrupt = false;
    bool restarting = false;
};
struct RunStatus {
    std::string text;
    Tone tone = Tone::Success;
};
RunStatus RunStatusFor(const RunFacts& facts);

// Under a cell that a batch skipped: "Not run: Run All stopped after the
// error in [5]." (or "... was interrupted in [5].").
std::string NotRunMessage(int stopped_at_count, bool interrupted);

// A traceback frame as a link: "Cell [5], line 1" for a notebook cell,
// "sentiment_analysis_inference.py:236" for a file (full path in `path`).
struct FrameLink {
    std::string label;
    std::string path;   // empty for a cell
    int cell_count = 0; // the cell's [n] when the frame is in a cell
    int line = 0;
};
FrameLink FrameLinkFor(const std::string& file, int line);

// Frames worth showing first: the notebook's own cells and files outside
// site-packages / the standard library; the rest sit behind "Show full
// traceback". Indexes into `files`, innermost last. Never empty when
// `files` is not: with no user frame, the innermost frame is kept.
std::vector<int> UserFrames(const std::vector<std::string>& files);

// Shortcut hints for the notebook (moved from the old hint line into the
// mode's tooltip in the status bar).
const char* NotebookKeysHint();

}  // namespace cyxwiz::nbview
