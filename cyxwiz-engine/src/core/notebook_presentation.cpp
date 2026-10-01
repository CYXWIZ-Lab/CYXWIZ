#include "notebook_presentation.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

namespace cyxwiz::nbview {

namespace {
const char* kDot = " \xC2\xB7 ";  // " · "

std::string Lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    std::replace(s.begin(), s.end(), '\\', '/');
    return s;
}

bool IsCellName(const std::string& file, int* count) {
    // "Cell In[5]"
    const std::string prefix = "Cell In[";
    if (file.rfind(prefix, 0) != 0 || file.empty() || file.back() != ']') return false;
    const std::string digits = file.substr(prefix.size(), file.size() - prefix.size() - 1);
    if (digits.empty() || !std::all_of(digits.begin(), digits.end(), [](char c) { return c >= '0' && c <= '9'; }))
        return false;
    if (count) *count = std::atoi(digits.c_str());
    return true;
}
}  // namespace

std::string FormatDuration(double seconds) {
    char buf[48];
    if (seconds < 0.0) seconds = 0.0;
    if (seconds < 10.0) std::snprintf(buf, sizeof(buf), "%.2f s", seconds);
    else if (seconds < 60.0) std::snprintf(buf, sizeof(buf), "%.1f s", seconds);
    else {
        const long total = static_cast<long>(std::floor(seconds));
        std::snprintf(buf, sizeof(buf), "%ld min %ld s", total / 60, total % 60);
    }
    return buf;
}

Gutter GutterFor(CellRun state, int execution_count, double seconds, double running_seconds) {
    Gutter g;
    g.label = execution_count > 0 ? "[" + std::to_string(execution_count) + "]" : "[ ]";
    switch (state) {
        case CellRun::Running: {
            g.label = "[*]";
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.1f s", std::max(0.0, running_seconds));
            g.detail = buf;
            g.tone = Tone::Running;
            break;
        }
        case CellRun::Queued:
            g.label = "[ ]";
            g.detail = "Queued";
            break;
        case CellRun::Success:
            if (seconds >= 0.0) {
                g.detail = FormatDuration(seconds);
                g.mark = Gutter::Mark::Check;
            }
            g.tone = Tone::Success;
            break;
        case CellRun::Error:
            if (seconds >= 0.0) {
                g.detail = FormatDuration(seconds);
                g.mark = Gutter::Mark::Cross;
            }
            g.tone = Tone::Error;
            break;
        case CellRun::NotRun:
            g.label = "[ ]";
            break;
        case CellRun::Idle:
            break;
    }
    return g;
}

KernelChip KernelChipFor(const KernelFacts& f) {
    KernelChip chip;
    std::string state = "Idle";
    chip.tone = Tone::Success;
    if (f.restarting) {
        state = "Restarting";
        chip.tone = Tone::Warning;
    } else if (f.busy) {
        state = "Busy";
        chip.tone = Tone::Running;
    } else if (f.other_busy) {
        state = "Waiting";
        chip.tone = Tone::Warning;
    } else if (!f.started) {
        state = "Starts on first run";
        chip.tone = Tone::Muted;
    }
    chip.text = "Python";
    if (!f.python_version.empty()) chip.text += " " + f.python_version;
    if (!f.environment.empty()) chip.text += kDot + f.environment;
    chip.text += kDot + state;
    if (f.restarting) chip.tooltip = "Clearing this notebook's variables";
    else if (f.busy) chip.tooltip = "A cell is running. Interrupt stops it at the next Python line.";
    else if (f.other_busy) chip.tooltip = "Another script is running in the Engine's Python; cells wait for it.";
    else if (!f.started) chip.tooltip = "Python starts when the first cell runs.";
    else chip.tooltip = "This notebook's variables stay until Restart.";
    return chip;
}

RunStatus RunStatusFor(const RunFacts& f) {
    RunStatus s;
    if (f.restarting) {
        s.text = "Restarting";
        s.tone = Tone::Warning;
    } else if (f.running) {
        s.tone = Tone::Running;
        s.text = f.run_total > 1 ? "Running cell " + std::to_string(f.run_position) + " of " + std::to_string(f.run_total)
                                 : "Running";
    } else if (f.stopped_at_count > 0) {
        s.tone = f.stopped_by_interrupt ? Tone::Warning : Tone::Error;
        s.text = std::string(f.stopped_by_interrupt ? "Interrupted in [" : "Stopped after an error in [") +
                 std::to_string(f.stopped_at_count) + "]";
    } else {
        s.text = "Ready";
    }
    return s;
}

std::string NotRunMessage(int stopped_at_count, bool interrupted) {
    if (stopped_at_count <= 0) return "Not run.";
    return std::string(interrupted ? "Not run: Run All was interrupted in [" : "Not run: Run All stopped after the error in [") +
           std::to_string(stopped_at_count) + "].";
}

FrameLink FrameLinkFor(const std::string& file, int line) {
    FrameLink link;
    link.line = line;
    int count = 0;
    if (IsCellName(file, &count)) {
        link.cell_count = count;
        link.label = "Cell [" + std::to_string(count) + "], line " + std::to_string(line);
        return link;
    }
    if (!file.empty() && file.front() == '<') {  // <string>, <frozen ...>
        link.label = file + (line > 0 ? ", line " + std::to_string(line) : "");
        return link;
    }
    link.path = file;
    link.label = std::filesystem::path(file).filename().string() + (line > 0 ? ":" + std::to_string(line) : "");
    return link;
}

std::vector<int> UserFrames(const std::vector<std::string>& files) {
    std::vector<int> keep;
    for (int i = 0; i < static_cast<int>(files.size()); ++i) {
        const std::string f = Lower(files[i]);
        if (IsCellName(files[i], nullptr)) {
            keep.push_back(i);
            continue;
        }
        // site-packages, or the standard library: <...>/python3.12/... or
        // <...>/Python312/Lib/... (a project's own lib/ folder is the user's).
        bool stdlib = f.find("/lib/python") != std::string::npos;
        for (size_t at = f.find("/lib/"); !stdlib && at != std::string::npos; at = f.find("/lib/", at + 1)) {
            const size_t seg = f.rfind('/', at - 1);
            const std::string parent = f.substr(seg == std::string::npos ? 0 : seg + 1, at - (seg == std::string::npos ? 0 : seg + 1));
            stdlib = parent.rfind("python", 0) == 0;
        }
        const bool library = f.find("site-packages/") != std::string::npos || stdlib || (!f.empty() && f.front() == '<');
        if (!library) keep.push_back(i);
    }
    if (keep.empty() && !files.empty()) keep.push_back(static_cast<int>(files.size()) - 1);
    return keep;
}

const char* NotebookKeysHint() {
    return "Enter edit  \xC2\xB7  Esc command  \xC2\xB7  Shift+Enter run and select next  \xC2\xB7  Ctrl+Enter run\n"
           "A / B add above / below  \xC2\xB7  M / Y markdown / code  \xC2\xB7  D, D delete  \xC2\xB7  Up / Down select\n"
           "C / O fold the cell / its output  \xC2\xB7  All notebook keys: Preferences > Shortcuts";
}

}  // namespace cyxwiz::nbview
