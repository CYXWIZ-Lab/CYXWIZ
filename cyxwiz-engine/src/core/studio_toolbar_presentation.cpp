#include "studio_toolbar_presentation.h"

namespace cyxwiz::studio_toolbar {

namespace {

std::string Count(int n, const char* one, const char* many) {
    return std::to_string(n) + " " + (n == 1 ? one : many);
}

}  // namespace

StepChip CompileChip(const CompileInputs& in) {
    if (in.nodes == 0) return {"Empty", "Empty", "The canvas is empty", Tone::Neutral};
    if (in.live_check_off)
        return {"Not checked", "Off", "Data Pipeline mode: Compile (F7) checks the graph when you press it",
                Tone::Neutral};
    if (in.stale || in.state == CompileState::NotCompiled || in.state == CompileState::Compiling)
        return {"Checking...", "...", "Checking the graph...", Tone::Busy};
    if (in.errors > 0) {
        const std::string errors = Count(in.errors, "error", "errors");
        return {errors, errors, "Not ready to train: " + errors, Tone::Error};
    }
    if (in.state == CompileState::Failed)
        return {"Failed", "Failed", "The graph could not be checked", Tone::Error};
    if (in.warnings > 0)
        return {"Ready", "Ready", "Ready to train, " + Count(in.warnings, "warning", "warnings"), Tone::Ok};
    return {"Ready", "Ready", "Ready to train", Tone::Ok};
}

StepChip LocalDebugChip(const DebugInputs& in) {
    if (in.running) return {"Running...", "...", "Local Debug is running", Tone::Busy};
    switch (in.outcome) {
        case DebugOutcome::NotRun:
            return {"Not run", "Not run", "No Local Debug run on this graph yet", Tone::Neutral};
        case DebugOutcome::Passed:
            if (in.graph_changed)
                return {"Graph changed", "Changed", "Passed at " + in.at + "; the graph changed since", Tone::Warning};
            return {"Passed " + in.at, "Passed", "Passed at " + in.at + ", graph unchanged", Tone::Ok};
        case DebugOutcome::Failed:
            if (in.graph_changed)
                return {"Graph changed", "Changed", "Failed at " + in.at + "; the graph changed since", Tone::Warning};
            return {"Failed", "Failed", in.failure.empty() ? "Failed at " + in.at : in.failure, Tone::Error};
    }
    return {};
}

bool LocalDebugCurrent(const DebugInputs& in) {
    return in.outcome == DebugOutcome::Passed && !in.graph_changed && !in.running;
}

TrainButton TrainState(const TrainInputs& in) {
    TrainButton out;
    if (!in.train_available) {
        out.disabled_reason = "Training is not available yet (no dataset loaded?)";
        return out;
    }
    if (!in.graph_valid) {
        out.disabled_reason = "The graph is not ready to train. It needs a dataset input, model layers and a loss.";
        return out;
    }
    if (in.require_debug && !LocalDebugCurrent(in.debug)) {
        out.disabled_reason = "This project requires a passing Local Debug (F6) on this graph";
        return out;
    }
    out.enabled = true;
    out.tooltip = "Train (F5)";
    if (in.rl_graph) {
        out.tooltip += "\nPPO trains the policy on the environment; the RL Training Dashboard shows each update.";
        return out;
    }
    const auto& d = in.debug;
    if (d.running) {
        out.tooltip += "\nLocal Debug is still running.";
    } else if (d.outcome == DebugOutcome::NotRun) {
        out.tooltip += "\nNo Local Debug (F6) on this graph yet; Train runs anyway.";
    } else if (d.graph_changed) {
        out.tooltip += "\nLocal Debug " + std::string(d.outcome == DebugOutcome::Passed ? "passed" : "failed") +
                       " at " + d.at + "; the graph changed since. Train runs anyway; F6 checks the new graph first.";
    } else if (d.outcome == DebugOutcome::Failed) {
        out.tooltip += "\nThe last Local Debug failed" + (d.failure.empty() ? std::string() : " (" + d.failure + ")") +
                       "; Train runs anyway.";
    }
    return out;
}

float BarWidth(const BarWidths& w, int level) {
    float total = w.fixed + w.pipeline + w.run + w.gaps;
    total += level >= 2 ? w.edit_menu : w.edit_icons;
    total += level >= 2 ? w.view_menu : w.view_full;
    total += level >= 3 ? w.mode_menu : level >= 1 ? w.mode_short : w.mode_full;
    total += level >= 1 ? w.actions_icons : w.actions_full;
    total += level >= 4 ? w.steps_dots : level >= 1 ? w.steps_short : w.steps_full;
    return total;
}

int FoldLevel(const BarWidths& w, float available) {
    for (int level = 0; level < 4; ++level)
        if (BarWidth(w, level) <= available) return level;
    return 4;
}

namespace {

float StepsWidth(const BarWidths& w, int level) {
    return level >= 4 ? w.steps_dots : level >= 1 ? w.steps_short : w.steps_full;
}

}  // namespace

float RowOneWidth(const BarWidths& w, int level, float group_gap) {
    float total = w.fixed + w.pipeline + 3.0f * group_gap;  // graph | edit | view | the mode (and a pipeline)
    total += level >= 2 ? w.edit_menu : w.edit_icons;
    total += level >= 2 ? w.view_menu : w.view_full;
    total += level >= 3 ? w.mode_menu : level >= 1 ? w.mode_short : w.mode_full;
    total += level >= 1 ? w.actions_icons : w.actions_full;
    return total;
}

float RowTwoWidth(const BarWidths& w, int steps_level, bool compact_run, bool with_mode, float group_gap) {
    return StepsWidth(w, steps_level) + (compact_run ? w.run_compact : w.run) +
           (with_mode ? w.mode_menu + w.actions_icons + w.pipeline + group_gap : 0.0f);
}

BarLayout LayoutBar(const BarWidths& w, float available, float group_gap) {
    BarLayout layout;
    layout.level = FoldLevel(w, available);
    if (BarWidth(w, layout.level) <= available) {  // one row fits
        layout.steps_level = layout.level;
        return layout;
    }
    layout.two_rows = true;
    layout.level = 4;
    for (int level = 0; level < 4; ++level)
        if (RowOneWidth(w, level, group_gap) <= available) {
            layout.level = level;
            break;
        }
    layout.mode_on_row_two = RowOneWidth(w, 4, group_gap) > available;  // the mode, and a pipeline after it
    const bool m = layout.mode_on_row_two;
    layout.steps_level = 4;
    for (const int level : {0, 1})
        if (RowTwoWidth(w, level, false, m, group_gap) <= available) {
            layout.steps_level = level;
            break;
        }
    layout.compact_run = RowTwoWidth(w, layout.steps_level, false, m, group_gap) > available;
    return layout;
}

std::string CompactRunText(const std::string& run_line) {
    std::string text = run_line.substr(0, run_line.find(" \xC2\xB7 "));
    if (text.rfind("PPO ", 0) == 0) text.erase(0, 4);
    if (text.rfind("waiting", 0) == 0) return "Waiting";
    if (!text.empty() && text[0] >= 'a' && text[0] <= 'z') text[0] = static_cast<char>(text[0] - 'a' + 'A');
    return text;
}

std::string CountsText(size_t nodes, size_t links) {
    return std::to_string(nodes) + (nodes == 1 ? " node" : " nodes") + " \xC2\xB7 " + std::to_string(links) +
           (links == 1 ? " link" : " links");
}

std::string SelectionText(int selected) {
    return selected > 0 ? std::to_string(selected) + " selected" : std::string();
}

}  // namespace cyxwiz::studio_toolbar
