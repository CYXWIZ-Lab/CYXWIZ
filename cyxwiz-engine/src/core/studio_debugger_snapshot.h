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
