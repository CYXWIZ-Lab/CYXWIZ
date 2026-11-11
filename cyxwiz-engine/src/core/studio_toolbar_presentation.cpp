#include "studio_toolbar_presentation.h"

#include <algorithm>
#include <cctype>
#include <cmath>

namespace cyxwiz::studio_toolbar {

namespace {

constexpr const char* kDot = " \xC2\xB7 ";

std::string Count(int n, const char* one, const char* many) {
    return std::to_string(n) + " " + (n == 1 ? one : many);
}

Availability Off(std::string reason) { return {false, std::move(reason)}; }

std::string Lower(std::string text) {
    for (char& c : text) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return text;
}

}  // namespace

StepChip CompileChip(const CompileInputs& in) {
    if (in.nodes == 0) return {"Empty", "The canvas is empty", Tone::Neutral};
    if (in.live_check_off)
        return {"Not checked", "Data Pipeline mode: Compile (F7) checks the graph when you press it", Tone::Neutral};
    if (in.stale || in.state == CompileState::NotCompiled || in.state == CompileState::Compiling)
        return {"Checking...", "Checking the graph...", Tone::Busy};
    if (in.errors > 0) {
        const std::string errors = Count(in.errors, "error", "errors");
        return {errors, "Not ready to train: " + errors, Tone::Error};
    }
    if (in.state == CompileState::Failed) return {"Failed", "The graph could not be checked", Tone::Error};
    if (in.warnings > 0)
        return {"Ready", "Ready to train, " + Count(in.warnings, "warning", "warnings"), Tone::Ok};
    return {"Ready", "Ready to train", Tone::Ok};
}

StepChip LocalDebugChip(const DebugInputs& in) {
    if (in.running) return {"Running...", "Local Debug is running", Tone::Busy};
    switch (in.outcome) {
        case DebugOutcome::NotRun:
            return {"Not run", "No Local Debug run on this graph yet", Tone::Neutral};
        case DebugOutcome::Passed:
            if (in.graph_changed)
                return {"Graph changed", "Passed at " + in.at + "; the graph changed since", Tone::Warning};
            return {"Passed " + in.at, "Passed at " + in.at + ", graph unchanged", Tone::Ok};
        case DebugOutcome::Failed:
            if (in.graph_changed)
                return {"Graph changed", "Failed at " + in.at + "; the graph changed since", Tone::Warning};
            return {"Failed", in.failure.empty() ? "Failed at " + in.at : in.failure, Tone::Error};
    }
    return {};
}

bool LocalDebugCurrent(const DebugInputs& in) {
    return in.outcome == DebugOutcome::Passed && !in.graph_changed && !in.running;
}

Availability LocalDebugStart(const DebugInputs& in, size_t nodes) {
    if (in.running) return Off("A Local Debug run is already going. It ends by itself; the Studio Debugger shows it.");
    if (nodes == 0) return Off("The canvas is empty. Add a Data Input and a model first.");
    return {};
}

TrainButton TrainState(const TrainInputs& in) {
    TrainButton out;
    if (!in.train_available) {
        out.disabled_reason = "Training is not available yet (no dataset loaded?)";
        return out;
    }
    if (!in.busy.empty()) {
        out.disabled_reason = "Train waits while " + in.busy + ". It can start when that ends.";
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

std::string TrainingRunText(int epoch, int total_epochs, bool paused, bool waiting_for_gpu, bool compact) {
    if (waiting_for_gpu) return compact ? "Waiting" : "Waiting for the GPU";
    const std::string of = std::to_string(epoch) + " / " + std::to_string(total_epochs);
    if (paused) return compact ? "Paused " + of : "Paused at epoch " + of;
    return "Epoch " + of;
}

std::string CompactRunText(const std::string& run_line) {
    std::string text = run_line.substr(0, run_line.find(kDot));
    if (text.rfind("PPO ", 0) == 0) text.erase(0, 4);
    if (text.rfind("waiting", 0) == 0) return "Waiting";
    if (!text.empty() && text[0] >= 'a' && text[0] <= 'z') text[0] = static_cast<char>(text[0] - 'a' + 'A');
    return text;
}

const char* ModeName(Mode mode) {
    switch (mode) {
        case Mode::CodeGen: return "Code Gen";
        case Mode::DataPipeline: return "Data Pipeline";
        case Mode::LocalTraining: return "Local Training";
    }
    return "Code Gen";
}

CodeMenu CodeMenuFor(const CodeInputs& in) {
    CodeMenu out;
    out.framework_on_bar = in.mode == Mode::CodeGen;
    const std::string named = out.framework_on_bar ? std::string() : " " + in.framework + " Code";
    out.file_name = (in.graph.empty() ? std::string("graph") : in.graph) + "_" + Lower(in.framework) + ".py";
    out.generate_label = "Generate" + named + " into Script Editor";
    out.save_label = "Save" + named + " as .py...";
    out.generate_hint = "Opens a new Script Editor tab " + out.file_name;
    out.save_hint = "Asks where, then writes " + out.file_name;
    if (!in.problem.empty()) out.available = Off(in.problem);
    return out;
}

std::string CodeResultText(const CodeResult& r) {
    if (!r.ok) return "Could not generate " + r.framework + " code: " + r.error;
    const std::string what = Count(r.lines, "line", "lines") + " of " + r.framework + " code";
    const std::string text = r.to_file ? "Saved " + what + " to " + r.where
                                       : "Generated " + what + " into Script Editor tab " + r.where;
    return r.at.empty() ? text : text + kDot + r.at;
}

GraphMenu GraphMenuFor(const GraphInputs& in, int selected) {
    GraphMenu out;
    const std::string busy = in.locked.empty() ? std::string() : "Not while " + in.locked + ".";
    if (!busy.empty()) out.load = Off(busy);
    if (in.nodes == 0) out.export_graph = Off("The canvas is empty. Add a node first.");
    if (in.nodes == 0) out.clear = Off("The canvas is already empty.");
    else if (!busy.empty()) out.clear = Off(busy);
    if (selected == 0) {
        out.duplicate = Off("Select nodes first");
        out.remove = Off("Select nodes first");
    } else if (!busy.empty()) {
        out.duplicate = Off(busy);
        out.remove = Off(busy);
    }
    return out;
}

std::string SaveRefusal(size_t nodes) {
    return nodes == 0 ? "The canvas is empty. Add a node, then save." : std::string();
}

float ZoomStep(float zoom, int direction) {
    // On the 10 % grid: from 1.03, + goes to 1.1 and - to 1.0.
    const float steps = zoom / kZoomStep;
    const float grid = direction > 0 ? std::floor(steps + 1e-3f) + 1.0f : std::ceil(steps - 1e-3f) - 1.0f;
    return std::clamp(grid * kZoomStep, kZoomMin, kZoomMax);
}

Availability ZoomIn(float zoom) {
    return zoom >= kZoomMax - 1e-4f ? Off("The largest zoom (200%)") : Availability{};
}

Availability ZoomOut(float zoom) {
    return zoom <= kZoomMin + 1e-4f ? Off("The smallest zoom (25%)") : Availability{};
}

std::string ZoomText(float zoom) { return std::to_string(static_cast<int>(zoom * 100.0f + 0.5f)) + "%"; }

BarFold FoldAt(int level) {
    BarFold f;
    f.level = std::clamp(level, 0, kMaxFold);
    f.mode_menu = f.view_short = f.sim_icons = f.level >= 1;
    f.debug_in_more = f.level >= 2;
    f.compile_in_more = f.level >= 3;
    f.view_in_more = f.level >= 4;
    f.edit_in_more = f.level >= 5;
    f.code_in_more = f.level >= 6;
    f.framework_in_more = f.level >= 7;
    f.save_in_more = f.level >= 8;
    f.run_compact = f.level >= 9;
    f.mode_in_more = f.level >= 10;
    f.graph_icon = f.level >= 11;
    return f;
}

int ItemsInMore(const BarWidths& w, const BarFold& f) {
    int n = 0;
    const auto in = [&n](bool folded, float width) {
        if (folded && width > 0.0f) ++n;
    };
    in(f.debug_in_more, w.debug);
    in(f.compile_in_more, w.compile);
    in(f.view_in_more, w.view_full);
    in(f.edit_in_more, w.edit);
    in(f.code_in_more, w.code);
    in(f.code_in_more, w.run_pipeline);
    in(f.framework_in_more, w.framework);
    in(f.save_in_more, w.save);
    in(f.mode_in_more, w.mode_menu);
    return n;
}

float BarWidth(const BarWidths& w, int level) {
    const BarFold f = FoldAt(level);
    float total = (f.graph_icon ? w.graph_icon : w.graph) + (f.run_compact ? w.run_compact : w.run) + w.progress;
    int groups = 2;  // the graph, the run group
    const auto add = [&](bool shown, float width) {
        if (!shown || width <= 0.0f) return;
        total += width;
        ++groups;
    };
    add(!f.save_in_more, w.save);
    add(!f.edit_in_more, w.edit);
    add(!f.view_in_more, f.view_short ? w.view_short : w.view_full);
    add(!f.mode_in_more, f.mode_menu ? w.mode_menu : w.mode_full);
    add(true, f.sim_icons ? w.sim_icons : w.sim_full);
    add(!f.framework_in_more, w.framework);
    add(!f.code_in_more, w.code);
    add(!f.code_in_more, w.run_pipeline);
    add(!f.compile_in_more, w.compile);
    add(!f.debug_in_more, w.debug);
    add(ItemsInMore(w, f) > 0, w.more);
    if (w.progress > 0.0f) ++groups;
    return total + static_cast<float>(groups - 1) * w.gap;
}

int FoldLevel(const BarWidths& w, float available) {
    for (int level = 0; level < kMaxFold; ++level)
        if (BarWidth(w, level) <= available) return level;
    return kMaxFold;
}

std::string WindowTitle(const std::string& graph, bool unsaved) {
    return std::string("CyxWiz Studio") + kDot + (graph.empty() ? std::string("Untitled") : graph) +
           (unsaved ? " \xE2\x80\xA2" : "");
}

std::string CountsText(size_t nodes, size_t links) {
    return std::to_string(nodes) + (nodes == 1 ? " node" : " nodes") + kDot + std::to_string(links) +
           (links == 1 ? " link" : " links");
}

std::string SelectionText(int selected) {
    return selected > 0 ? std::to_string(selected) + " selected" : std::string();
}

Footer FooterFor(Mode mode, const std::string& framework, const std::string& locked) {
    if (!locked.empty()) return {"Graph replacement locked: " + locked, true};
    switch (mode) {
        case Mode::CodeGen: return {std::string("Code Gen mode") + kDot + "Generate writes " + framework + " code", false};
        case Mode::DataPipeline: return {std::string("Data Pipeline mode") + kDot + "training nodes run in Train", false};
        case Mode::LocalTraining: return {std::string("Local Training mode") + kDot + "Train runs the whole graph", false};
    }
    return {};
}

}  // namespace cyxwiz::studio_toolbar
