#pragma once

// The data one Studio Debugger run produces (or a saved run restores). It is
// plain data so the presentation model and tests can use it without ImGui.

#include "crash_run_recorder.h"
#include "debug_executor.h"
#include "debug_recommendation_engine.h"
#include "debug_run_store.h"
#include "debug_session.h"
#include "smoke_run_executor.h"
#include "training_trace_collector.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <utility>
#include <functional>
#include <string>
#include <vector>

namespace cyxwiz {

enum class StudioDebuggerRunMode {
    FullWorkflow = 0,
    Preflight,
    LocalDebug,
    SmokeRun,
    RuntimeTrace
};

// Overall result of a run. One vocabulary for the whole debugger (tofix96:
// distinct passed / failed / unsupported / not_started; plus stopped).
enum class StudioDebuggerOutcome {
    NotStarted = 0,
    Running,
    Passed,
    NeedsAttention,
    Failed,
    Stopped,
    Unsupported
};

enum class StudioDebuggerStepState {
    Pending = 0,
    Running,
    Passed,
    Warning,
    Failed,
    Skipped,
    Unsupported,
    Stopped
};

// One step of a run (Compile, Preflight, Smoke Run, Local Debug, ...).
// `required` steps decide the outcome; an optional step never overrides a
// required failure and a later step never erases an earlier one.
struct StudioDebuggerStep {
    std::string id;
    std::string name;
    std::string detail;
    StudioDebuggerStepState state = StudioDebuggerStepState::Pending;
    bool required = true;
    double seconds = 0.0;
};

// Hooks a run reports through: the current step with overall progress
// (0..1) and the step list, and a cooperative stop checked between steps.
struct StudioDebuggerRunControl {
    std::function<void(const std::vector<StudioDebuggerStep>& steps,
                       const std::string& running_step,
                       float progress)> on_progress;
    std::function<bool()> should_stop;
};

// Inputs captured on the UI thread when a run starts.
struct StudioDebuggerRunInputs {
    std::map<int, std::pair<float, float>> node_positions;
    // Project root for the prepared-data cache shared with Train.
    std::filesystem::path project_root;
};

struct StudioDebuggerSnapshot {
    bool success = false;
    bool has_debug_result = false;
    std::string run_id;

    StudioDebuggerRunMode mode = StudioDebuggerRunMode::FullWorkflow;
    StudioDebuggerOutcome outcome = StudioDebuggerOutcome::NotStarted;
    std::vector<StudioDebuggerStep> steps;
    double duration_seconds = 0.0;
    // Graph data domain from the compiled config (Tabular, Text, Image, ...).
    std::string graph_domain;
    // Training evidence belongs to an earlier, finished training run rather
    // than to this debugger run (tofix96 provenance).
    bool training_trace_historical = false;

    uint64_t graph_hash = 0;
    size_t node_count = 0;
    size_t link_count = 0;

    std::string graph_summary;
    std::string preflight_summary;
    std::string sample_summary = "Synthetic sample 0 (POC)";
    std::string failure_summary;

    DebugPreflightResult preflight;
    std::vector<ValidationIssue> issues;
    std::vector<DebugTraceRecord> traces;
    std::vector<StudioEventRecord> studio_events;
    DebugResult debug_result;
    SmokeRunResult smoke_result;
    CrashRunSummary last_run;
    TrainingTraceSummary training_trace;
    DebugRunExecutionSummary execution;
    std::vector<DebugRecommendation> recommendations;
    std::vector<DebugRunStoreSummary> run_history;
};

} // namespace cyxwiz
