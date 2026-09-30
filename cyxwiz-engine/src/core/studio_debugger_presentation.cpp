#include "studio_debugger_presentation.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <set>

namespace cyxwiz {
namespace {

std::string Lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return text;
}

std::string FormatSeconds(double seconds) {
    char buffer[32];
    if (seconds < 60.0) {
        std::snprintf(buffer, sizeof(buffer), "%.1f s", seconds);
    } else {
        std::snprintf(buffer, sizeof(buffer), "%d min %d s",
                      static_cast<int>(seconds) / 60, static_cast<int>(seconds) % 60);
    }
    return buffer;
}

StudioDebuggerStep Step(const char* id, const char* name, bool required = true) {
    StudioDebuggerStep step;
    step.id = id;
    step.name = name;
    step.required = required;
    return step;
}

std::string UpperFirstBackend(const std::string& backend) {
    const std::string lower = Lower(backend);
    if (lower.find("cuda") != std::string::npos) return "CUDA";
    if (lower.find("opencl") != std::string::npos) return "OpenCL";
    if (lower.find("oneapi") != std::string::npos) return "oneAPI";
    if (lower.find("cpu") != std::string::npos) return "CPU";
    return backend;
}

} // namespace

const char* StudioDebuggerRunModeLabel(StudioDebuggerRunMode mode) {
    switch (mode) {
        case StudioDebuggerRunMode::FullWorkflow: return "Full Workflow";
        case StudioDebuggerRunMode::Preflight: return "Preflight";
        case StudioDebuggerRunMode::LocalDebug: return "Local Debug";
        case StudioDebuggerRunMode::SmokeRun: return "Smoke Run";
        case StudioDebuggerRunMode::RuntimeTrace: return "Runtime Trace";
    }
    return "Full Workflow";
}

const char* StudioDebuggerRunModeKey(StudioDebuggerRunMode mode) {
    switch (mode) {
        case StudioDebuggerRunMode::FullWorkflow: return "full_workflow";
        case StudioDebuggerRunMode::Preflight: return "preflight";
        case StudioDebuggerRunMode::LocalDebug: return "local_debug";
        case StudioDebuggerRunMode::SmokeRun: return "smoke_run";
        case StudioDebuggerRunMode::RuntimeTrace: return "runtime_trace";
    }
    return "full_workflow";
}

bool ParseStudioDebuggerRunModeKey(const std::string& key, StudioDebuggerRunMode& mode) {
    for (auto candidate : {StudioDebuggerRunMode::FullWorkflow, StudioDebuggerRunMode::Preflight,
                           StudioDebuggerRunMode::LocalDebug, StudioDebuggerRunMode::SmokeRun,
                           StudioDebuggerRunMode::RuntimeTrace}) {
        if (key == StudioDebuggerRunModeKey(candidate)) {
            mode = candidate;
            return true;
        }
    }
    return false;
}

const char* StudioDebuggerOutcomeLabel(StudioDebuggerOutcome outcome) {
    switch (outcome) {
        case StudioDebuggerOutcome::NotStarted: return "Not started";
        case StudioDebuggerOutcome::Running: return "Running";
        case StudioDebuggerOutcome::Passed: return "Passed";
        case StudioDebuggerOutcome::NeedsAttention: return "Needs attention";
        case StudioDebuggerOutcome::Failed: return "Failed";
        case StudioDebuggerOutcome::Stopped: return "Stopped";
        case StudioDebuggerOutcome::Unsupported: return "Unsupported";
    }
    return "Not started";
}

const char* StudioDebuggerOutcomeKey(StudioDebuggerOutcome outcome) {
    switch (outcome) {
        case StudioDebuggerOutcome::NotStarted: return "not_started";
        case StudioDebuggerOutcome::Running: return "running";
        case StudioDebuggerOutcome::Passed: return "passed";
        case StudioDebuggerOutcome::NeedsAttention: return "needs_attention";
        case StudioDebuggerOutcome::Failed: return "failed";
        case StudioDebuggerOutcome::Stopped: return "stopped";
        case StudioDebuggerOutcome::Unsupported: return "unsupported";
    }
    return "not_started";
}

bool ParseStudioDebuggerOutcomeKey(const std::string& key, StudioDebuggerOutcome& outcome) {
    for (auto candidate : {StudioDebuggerOutcome::NotStarted, StudioDebuggerOutcome::Running,
                           StudioDebuggerOutcome::Passed, StudioDebuggerOutcome::NeedsAttention,
                           StudioDebuggerOutcome::Failed, StudioDebuggerOutcome::Stopped,
                           StudioDebuggerOutcome::Unsupported}) {
        if (key == StudioDebuggerOutcomeKey(candidate)) {
            outcome = candidate;
            return true;
        }
    }
    return false;
}

DebuggerTone StudioDebuggerOutcomeTone(StudioDebuggerOutcome outcome) {
    switch (outcome) {
        case StudioDebuggerOutcome::Passed: return DebuggerTone::Success;
        case StudioDebuggerOutcome::NeedsAttention: return DebuggerTone::Warning;
        case StudioDebuggerOutcome::Failed: return DebuggerTone::Danger;
        case StudioDebuggerOutcome::Running: return DebuggerTone::Info;
        case StudioDebuggerOutcome::Stopped: return DebuggerTone::Warning;
        case StudioDebuggerOutcome::Unsupported: return DebuggerTone::Muted;
        case StudioDebuggerOutcome::NotStarted: return DebuggerTone::Muted;
    }
    return DebuggerTone::Muted;
}

const char* StudioDebuggerStepStateLabel(StudioDebuggerStepState state) {
    switch (state) {
        case StudioDebuggerStepState::Pending: return "Waiting";
        case StudioDebuggerStepState::Running: return "Running";
        case StudioDebuggerStepState::Passed: return "Passed";
        case StudioDebuggerStepState::Warning: return "Needs attention";
        case StudioDebuggerStepState::Failed: return "Failed";
        case StudioDebuggerStepState::Skipped: return "Skipped";
        case StudioDebuggerStepState::Unsupported: return "Unsupported";
        case StudioDebuggerStepState::Stopped: return "Stopped";
    }
    return "Waiting";
}

const char* StudioDebuggerStepStateKey(StudioDebuggerStepState state) {
    switch (state) {
        case StudioDebuggerStepState::Pending: return "pending";
        case StudioDebuggerStepState::Running: return "running";
        case StudioDebuggerStepState::Passed: return "passed";
        case StudioDebuggerStepState::Warning: return "warning";
        case StudioDebuggerStepState::Failed: return "failed";
        case StudioDebuggerStepState::Skipped: return "skipped";
        case StudioDebuggerStepState::Unsupported: return "unsupported";
        case StudioDebuggerStepState::Stopped: return "stopped";
    }
    return "pending";
}

bool ParseStudioDebuggerStepStateKey(const std::string& key, StudioDebuggerStepState& state) {
    for (auto candidate : {StudioDebuggerStepState::Pending, StudioDebuggerStepState::Running,
                           StudioDebuggerStepState::Passed, StudioDebuggerStepState::Warning,
                           StudioDebuggerStepState::Failed, StudioDebuggerStepState::Skipped,
                           StudioDebuggerStepState::Unsupported, StudioDebuggerStepState::Stopped}) {
        if (key == StudioDebuggerStepStateKey(candidate)) {
            state = candidate;
            return true;
        }
    }
    return false;
}

DebuggerTone StudioDebuggerStepStateTone(StudioDebuggerStepState state) {
    switch (state) {
        case StudioDebuggerStepState::Passed: return DebuggerTone::Success;
        case StudioDebuggerStepState::Warning: return DebuggerTone::Warning;
        case StudioDebuggerStepState::Failed: return DebuggerTone::Danger;
        case StudioDebuggerStepState::Running: return DebuggerTone::Info;
        case StudioDebuggerStepState::Stopped: return DebuggerTone::Warning;
        case StudioDebuggerStepState::Pending:
        case StudioDebuggerStepState::Skipped:
        case StudioDebuggerStepState::Unsupported:
            return DebuggerTone::Muted;
    }
    return DebuggerTone::Muted;
}

const char* StudioDebuggerDomainLabel(PreprocessingDomain domain) {
    switch (domain) {
        case PreprocessingDomain::Tabular: return "Tabular";
        case PreprocessingDomain::Image: return "Image";
        case PreprocessingDomain::Audio: return "Audio";
        case PreprocessingDomain::Text: return "Text";
        case PreprocessingDomain::TimeSeries: return "Time series";
        case PreprocessingDomain::General: return "General";
    }
    return "Unknown";
}

SmokeCapability EvaluateSmokeCapability(const std::string& graph_domain) {
    SmokeCapability capability;
    const std::string domain = Lower(graph_domain);
    if (domain == "text") {
        capability.supported = true;
        return capability;
    }
    capability.supported = false;
    capability.reason = domain.empty()
        ? "Smoke Run needs a compiled graph with a loaded dataset."
        : "Smoke Run supports text graphs only; this graph's data is " + graph_domain + ".";
    capability.alternative =
        "Use Local Debug to check the model on a synthetic batch, then Train.";
    return capability;
}

std::vector<StudioDebuggerStep> PlanStudioDebuggerSteps(
    StudioDebuggerRunMode mode, const SmokeCapability& smoke) {
    std::vector<StudioDebuggerStep> steps;
    const auto add_smoke = [&](bool required_when_supported) {
        StudioDebuggerStep step = Step("smoke", "Smoke Run", required_when_supported);
        if (!smoke.supported) {
            step.state = StudioDebuggerStepState::Unsupported;
            step.detail = smoke.reason;
        }
        steps.push_back(step);
    };
    switch (mode) {
        case StudioDebuggerRunMode::FullWorkflow:
            steps.push_back(Step("compile", "Compile"));
            steps.push_back(Step("preflight", "Preflight"));
            // In Full Workflow an unsupported Smoke Run is skipped, not failed.
            add_smoke(smoke.supported);
            if (!smoke.supported) steps.back().required = false;
            steps.push_back(Step("local_debug", "Local Debug"));
            steps.push_back(Step("runtime", "Runtime evidence", false));
            break;
        case StudioDebuggerRunMode::Preflight:
            steps.push_back(Step("compile", "Compile"));
            steps.push_back(Step("preflight", "Preflight"));
            break;
        case StudioDebuggerRunMode::LocalDebug:
            steps.push_back(Step("compile", "Compile"));
            steps.push_back(Step("preflight", "Preflight"));
            steps.push_back(Step("local_debug", "Local Debug"));
            break;
        case StudioDebuggerRunMode::SmokeRun:
            steps.push_back(Step("compile", "Compile"));
            steps.push_back(Step("preflight", "Preflight"));
            add_smoke(true);
            break;
        case StudioDebuggerRunMode::RuntimeTrace:
            steps.push_back(Step("runtime", "Runtime evidence"));
            break;
    }
    return steps;
}

StudioDebuggerOutcome AggregateStudioDebuggerOutcome(
    const std::vector<StudioDebuggerStep>& steps, bool stopped, bool has_warnings) {
    bool any_run = false;
    bool any_required_failed = false;
    bool any_warning = has_warnings;
    bool any_required_passed = false;
    bool required_unsupported = false;
    bool any_running = false;
    for (const auto& step : steps) {
        switch (step.state) {
            case StudioDebuggerStepState::Running:
                any_running = true;
                break;
            case StudioDebuggerStepState::Passed:
                any_run = true;
                if (step.required) any_required_passed = true;
                break;
            case StudioDebuggerStepState::Warning:
                any_run = true;
                any_warning = true;
                if (step.required) any_required_passed = true;
                break;
            case StudioDebuggerStepState::Failed:
                any_run = true;
                if (step.required) any_required_failed = true;
                else any_warning = true;
                break;
            case StudioDebuggerStepState::Unsupported:
                if (step.required) required_unsupported = true;
                break;
            case StudioDebuggerStepState::Stopped:
                stopped = true;
                break;
            case StudioDebuggerStepState::Pending:
            case StudioDebuggerStepState::Skipped:
                break;
        }
    }
    if (any_required_failed) return StudioDebuggerOutcome::Failed;
    if (stopped) return StudioDebuggerOutcome::Stopped;
    if (any_running) return StudioDebuggerOutcome::Running;
    if (required_unsupported) return StudioDebuggerOutcome::Unsupported;
    if (!any_run) return StudioDebuggerOutcome::NotStarted;
    if (any_warning) return StudioDebuggerOutcome::NeedsAttention;
    return any_required_passed ? StudioDebuggerOutcome::Passed : StudioDebuggerOutcome::NeedsAttention;
}

bool IsTrainingTraceLive(const TrainingTraceSummary& trace, bool training_active) {
    if (!training_active || !trace.available || trace.run_id.empty()) return false;
    const std::string status = Lower(trace.status);
    return status != "completed" && status != "complete" && status != "cancelled" &&
           status != "canceled" && status != "failed" && status != "stopped" &&
           status != "crashed" && status != "error";
}

DebuggerTone TraceStatusTone(const std::string& raw) {
    const std::string status = Lower(raw);
    if (status == "ok" || status == "passed" || status == "ready" ||
        status == "captured" || status == "completed" || status == "success") {
        return DebuggerTone::Success;
    }
    if (status == "failed" || status == "nan" || status == "inf" || status == "error" ||
        status == "crashed") {
        return DebuggerTone::Danger;
    }
    if (status == "warning" || status == "zero" || status == "shape_mismatch" ||
        status == "blocked" || status == "missing_gradient" || status == "stopped") {
        return DebuggerTone::Warning;
    }
    if (status == "started" || status == "running") return DebuggerTone::Info;
    return DebuggerTone::Muted;
}

std::string TraceStatusLabel(const std::string& raw) {
    const std::string status = Lower(raw);
    if (status.empty()) return "unknown";
    if (status == "nan") return "NaN";
    if (status == "inf") return "Inf";
    std::string label = status;
    std::replace(label.begin(), label.end(), '_', ' ');
    return label;
}

int TraceStatusSeverity(const std::string& raw) {
    switch (TraceStatusTone(raw)) {
        case DebuggerTone::Danger: return 4;
        case DebuggerTone::Warning: return 3;
        case DebuggerTone::Info: return 2;
        case DebuggerTone::Success: return 1;
        case DebuggerTone::Muted:
        case DebuggerTone::Neutral:
            return 0;
    }
    return 0;
}

std::string WorstTraceStatus(const std::vector<std::string>& statuses) {
    std::string worst;
    int worst_rank = -1;
    for (const auto& status : statuses) {
        const int rank = TraceStatusSeverity(status);
        if (rank > worst_rank) {
            worst_rank = rank;
            worst = status;
        }
    }
    return worst;
}

StudioDebuggerCounts CountStudioDebuggerFindings(const StudioDebuggerSnapshot& snapshot) {
    StudioDebuggerCounts counts;
    counts.traces = snapshot.traces.size();
    counts.fixes = snapshot.recommendations.size();
    for (const auto& issue : snapshot.issues) {
        if (issue.level == IssueLevel::Error) ++counts.errors;
        else if (issue.level == IssueLevel::Warning) ++counts.warnings;
    }
    return counts;
}

StudioDebuggerStatusLine BuildStudioDebuggerStatusLine(
    const StudioDebuggerSnapshot& snapshot, bool has_session, bool running,
    const std::string& running_step, float running_progress) {
    StudioDebuggerStatusLine line;
    if (running) {
        line.outcome = StudioDebuggerOutcome::Running;
        char progress[32];
        std::snprintf(progress, sizeof(progress), " · %d%%",
                      static_cast<int>(std::lround(std::clamp(running_progress, 0.0f, 1.0f) * 100.0f)));
        line.summary = (running_step.empty() ? std::string("Preparing") : running_step) + progress;
    } else if (!has_session) {
        line.outcome = StudioDebuggerOutcome::NotStarted;
        line.summary = "Choose a mode and press Run to check this graph before training.";
    } else {
        line.outcome = snapshot.outcome;
        std::string mode = StudioDebuggerRunModeLabel(snapshot.mode);
        switch (snapshot.outcome) {
            case StudioDebuggerOutcome::Stopped:
                line.summary = mode + " stopped";
                break;
            case StudioDebuggerOutcome::Unsupported:
                line.summary = mode + " is not available for this graph";
                break;
            case StudioDebuggerOutcome::NotStarted:
                line.summary = mode + " did not start";
                break;
            default:
                line.summary = mode + " finished";
                break;
        }
        if (snapshot.duration_seconds > 0.0) {
            line.summary += " in " + FormatSeconds(snapshot.duration_seconds);
        }
        if (!snapshot.failure_summary.empty() &&
            (snapshot.outcome == StudioDebuggerOutcome::Failed ||
             snapshot.outcome == StudioDebuggerOutcome::Unsupported ||
             snapshot.outcome == StudioDebuggerOutcome::NotStarted)) {
            line.summary += ": " + snapshot.failure_summary;
        }
    }
    line.label = StudioDebuggerOutcomeLabel(line.outcome);
    line.tone = StudioDebuggerOutcomeTone(line.outcome);
    if (has_session && !running) {
        line.counts = CountStudioDebuggerFindings(snapshot);
        const auto& execution = snapshot.execution;
        if (execution.available && !snapshot.training_trace_historical &&
            !execution.effective_backend.empty()) {
            line.backend = UpperFirstBackend(execution.effective_backend);
            if (!execution.effective_device_name.empty()) {
                line.backend += " · " + execution.effective_device_name;
            }
            std::string residency = execution.residency_verdict.empty()
                ? std::string("residency unobserved") : execution.residency_verdict;
            std::replace(residency.begin(), residency.end(), '_', ' ');
            line.provenance = residency + " · " +
                std::to_string(execution.native_cpu_fallback_count) + " CPU fallbacks";
        } else if (snapshot.training_trace_historical && snapshot.training_trace.available) {
            line.provenance = "Runtime shows historical training evidence (run " +
                              snapshot.training_trace.run_id + ")";
        }
    }
    return line;
}

DebuggerGraphLayout LayoutDebuggerGraph(
    const std::vector<DebuggerGraphNodeInput>& nodes,
    const std::vector<DebuggerGraphLinkInput>& links,
    float node_width, float node_height, float gap_x, float gap_y) {
    DebuggerGraphLayout layout;
    if (nodes.empty()) return layout;

    const bool all_positioned = std::all_of(
        nodes.begin(), nodes.end(), [](const auto& node) { return node.has_position; });
    if (all_positioned) {
        // Keep the canvas arrangement: shift to the origin and scale so the
        // closest pair of neighbours does not overlap at the box size used.
        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float max_y = std::numeric_limits<float>::lowest();
        for (const auto& node : nodes) {
            min_x = std::min(min_x, node.x);
            min_y = std::min(min_y, node.y);
            max_x = std::max(max_x, node.x);
            max_y = std::max(max_y, node.y);
        }
        // Canvas nodes are roughly 160 x 90 including labels; map that
        // spacing onto the debugger box size so boxes keep their relative
        // positions without overlapping.
        const float scale_x = (node_width + gap_x) / 200.0f;
        const float scale_y = (node_height + gap_y) / 130.0f;
        for (const auto& node : nodes) {
            layout.nodes.push_back({node.id, node.name, (node.x - min_x) * scale_x,
                                    (node.y - min_y) * scale_y});
        }
        layout.width = (max_x - min_x) * scale_x + node_width;
        layout.height = (max_y - min_y) * scale_y + node_height;
        layout.from_canvas = true;
        return layout;
    }

    // Layered layout: column = longest path from a source (cycle-safe).
    std::map<int, std::vector<int>> incoming;
    std::set<int> ids;
    for (const auto& node : nodes) ids.insert(node.id);
    for (const auto& link : links) {
        if (ids.count(link.from_node) && ids.count(link.to_node) && link.from_node != link.to_node) {
            incoming[link.to_node].push_back(link.from_node);
        }
    }
    std::map<int, int> depth;
    std::set<int> visiting;
    std::function<int(int)> depth_of = [&](int id) -> int {
        auto found = depth.find(id);
        if (found != depth.end()) return found->second;
        if (visiting.count(id)) return 0;  // cycle: break it here
        visiting.insert(id);
        int best = 0;
        for (int parent : incoming[id]) best = std::max(best, depth_of(parent) + 1);
        visiting.erase(id);
        depth[id] = best;
        return best;
    };
    std::map<int, int> rows_used;
    int max_depth = 0;
    int max_rows = 0;
    for (const auto& node : nodes) {
        const int column = depth_of(node.id);
        const int row = rows_used[column]++;
        max_depth = std::max(max_depth, column);
        max_rows = std::max(max_rows, row + 1);
        layout.nodes.push_back({node.id, node.name, column * (node_width + gap_x),
                                row * (node_height + gap_y)});
    }
    layout.width = (max_depth + 1) * node_width + max_depth * gap_x;
    layout.height = max_rows * node_height + std::max(0, max_rows - 1) * gap_y;
    return layout;
}

std::map<int, DebuggerNodeStatus> AggregateDebuggerNodeStatus(const StudioDebuggerSnapshot& snapshot) {
    std::map<int, DebuggerNodeStatus> result;
    std::map<int, std::vector<std::string>> statuses;
    for (const auto& trace : snapshot.traces) {
        if (trace.node_id < 0) continue;
        auto& entry = result[trace.node_id];
        ++entry.trace_count;
        entry.issue_count += trace.issues.size();
        statuses[trace.node_id].push_back(trace.status);
    }
    for (const auto& issue : snapshot.issues) {
        if (issue.node_id < 0) continue;
        auto& entry = result[issue.node_id];
        ++entry.issue_count;
        statuses[issue.node_id].push_back(issue.level == IssueLevel::Error ? "failed" : "warning");
    }
    for (const auto& recommendation : snapshot.recommendations) {
        if (recommendation.node_id >= 0) ++result[recommendation.node_id].recommendation_count;
    }
    for (auto& [id, entry] : result) {
        entry.worst_status = WorstTraceStatus(statuses[id]);
    }
    return result;
}

DebugTraceRecord BuildStudioDebuggerRunSummaryTrace(const StudioDebuggerSnapshot& snapshot) {
    DebugTraceRecord trace;
    trace.run_id = snapshot.run_id;
    trace.node_id = -1;
    trace.node_name = "Run summary";
    trace.node_type = "StudioDebugger";
    trace.phase = "RunSummary";
    trace.role = DebugTraceRole::StudioEvent;
    trace.status = StudioDebuggerOutcomeKey(snapshot.outcome);
    trace.payload["mode"] = StudioDebuggerRunModeKey(snapshot.mode);
    trace.payload["outcome"] = StudioDebuggerOutcomeKey(snapshot.outcome);
    trace.payload["duration_seconds"] = snapshot.duration_seconds;
    trace.payload["graph_domain"] = snapshot.graph_domain;
    trace.payload["node_count"] = snapshot.node_count;
    trace.payload["link_count"] = snapshot.link_count;
    trace.payload["training_trace_historical"] = snapshot.training_trace_historical;
    trace.payload["failure_summary"] = snapshot.failure_summary;
    nlohmann::json steps = nlohmann::json::array();
    for (const auto& step : snapshot.steps) {
        steps.push_back({{"id", step.id},
                         {"name", step.name},
                         {"detail", step.detail},
                         {"state", StudioDebuggerStepStateKey(step.state)},
                         {"required", step.required},
                         {"seconds", step.seconds}});
    }
    trace.payload["steps"] = steps;
    return trace;
}

bool ApplyStudioDebuggerRunSummaryTrace(StudioDebuggerSnapshot& snapshot) {
    for (auto it = snapshot.traces.rbegin(); it != snapshot.traces.rend(); ++it) {
        if (it->phase != "RunSummary" || !it->payload.is_object()) continue;
        const auto& payload = it->payload;
        StudioDebuggerRunMode mode;
        if (ParseStudioDebuggerRunModeKey(payload.value("mode", std::string{}), mode)) {
            snapshot.mode = mode;
        }
        StudioDebuggerOutcome outcome;
        if (ParseStudioDebuggerOutcomeKey(payload.value("outcome", std::string{}), outcome)) {
            snapshot.outcome = outcome;
        }
        snapshot.duration_seconds = payload.value("duration_seconds", 0.0);
        snapshot.graph_domain = payload.value("graph_domain", std::string{});
        snapshot.node_count = payload.value("node_count", snapshot.node_count);
        snapshot.link_count = payload.value("link_count", snapshot.link_count);
        snapshot.training_trace_historical = payload.value("training_trace_historical", false);
        if (snapshot.failure_summary.empty()) {
            snapshot.failure_summary = payload.value("failure_summary", std::string{});
        }
        snapshot.steps.clear();
        if (payload.contains("steps") && payload["steps"].is_array()) {
            for (const auto& item : payload["steps"]) {
                StudioDebuggerStep step;
                step.id = item.value("id", std::string{});
                step.name = item.value("name", std::string{});
                step.detail = item.value("detail", std::string{});
                StudioDebuggerStepState state;
                if (ParseStudioDebuggerStepStateKey(item.value("state", std::string{}), state)) {
                    step.state = state;
                }
                step.required = item.value("required", true);
                step.seconds = item.value("seconds", 0.0);
                snapshot.steps.push_back(step);
            }
        }
        return true;
    }
    return false;
}

} // namespace cyxwiz
