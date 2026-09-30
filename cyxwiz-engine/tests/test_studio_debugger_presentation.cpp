// Studio Debugger presentation model contract (TOFIX128): outcome rules,
// step plans, the status vocabulary, the status line, graph layout and the
// run summary round trip. Scenario data mirrors the owner's machine
// (GTX 1050 Ti, sentiment Tabular graph, 7 nodes).

#include "../src/core/studio_debugger_presentation.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace cyxwiz;

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        ++g_failures;
    }
}

StudioDebuggerStep MakeStep(const char* id, StudioDebuggerStepState state, bool required = true) {
    StudioDebuggerStep step;
    step.id = id;
    step.name = id;
    step.state = state;
    step.required = required;
    return step;
}

void TestModeAndStateKeysRoundTrip() {
    for (auto mode : {StudioDebuggerRunMode::FullWorkflow, StudioDebuggerRunMode::Preflight,
                      StudioDebuggerRunMode::LocalDebug, StudioDebuggerRunMode::SmokeRun,
                      StudioDebuggerRunMode::RuntimeTrace}) {
        StudioDebuggerRunMode parsed = StudioDebuggerRunMode::FullWorkflow;
        Check(ParseStudioDebuggerRunModeKey(StudioDebuggerRunModeKey(mode), parsed) && parsed == mode,
              std::string("mode key round trip: ") + StudioDebuggerRunModeLabel(mode));
    }
    for (auto outcome : {StudioDebuggerOutcome::NotStarted, StudioDebuggerOutcome::Running,
                         StudioDebuggerOutcome::Passed, StudioDebuggerOutcome::NeedsAttention,
                         StudioDebuggerOutcome::Failed, StudioDebuggerOutcome::Stopped,
                         StudioDebuggerOutcome::Unsupported}) {
        StudioDebuggerOutcome parsed = StudioDebuggerOutcome::NotStarted;
        Check(ParseStudioDebuggerOutcomeKey(StudioDebuggerOutcomeKey(outcome), parsed) && parsed == outcome,
              std::string("outcome key round trip: ") + StudioDebuggerOutcomeLabel(outcome));
    }
    for (auto state : {StudioDebuggerStepState::Pending, StudioDebuggerStepState::Running,
                       StudioDebuggerStepState::Passed, StudioDebuggerStepState::Warning,
                       StudioDebuggerStepState::Failed, StudioDebuggerStepState::Skipped,
                       StudioDebuggerStepState::Unsupported, StudioDebuggerStepState::Stopped}) {
        StudioDebuggerStepState parsed = StudioDebuggerStepState::Pending;
        Check(ParseStudioDebuggerStepStateKey(StudioDebuggerStepStateKey(state), parsed) && parsed == state,
              std::string("step key round trip: ") + StudioDebuggerStepStateLabel(state));
    }
    StudioDebuggerRunMode mode;
    Check(!ParseStudioDebuggerRunModeKey("bogus", mode), "unknown mode key is rejected");
}

void TestSmokeCapabilityBeforeDispatch() {
    const auto text = EvaluateSmokeCapability("Text");
    Check(text.supported, "text graphs support Smoke Run");
    const auto tabular = EvaluateSmokeCapability("Tabular");
    Check(!tabular.supported, "tabular graphs do not support Smoke Run");
    Check(tabular.reason.find("Tabular") != std::string::npos, "reason names the graph's domain");
    Check(!tabular.alternative.empty(), "an alternative is offered");
    Check(!EvaluateSmokeCapability("").supported, "unknown domain is not supported");
}

void TestStepPlans() {
    const auto tabular = EvaluateSmokeCapability("Tabular");
    const auto full = PlanStudioDebuggerSteps(StudioDebuggerRunMode::FullWorkflow, tabular);
    Check(full.size() == 5, "Full Workflow has five steps");
    bool smoke_found = false;
    for (const auto& step : full) {
        if (step.id == "smoke") {
            smoke_found = true;
            Check(step.state == StudioDebuggerStepState::Unsupported, "tabular smoke step pre-marked unsupported");
            Check(!step.required, "unsupported smoke step is optional in Full Workflow");
        }
        if (step.id == "runtime") Check(!step.required, "runtime evidence is optional in Full Workflow");
    }
    Check(smoke_found, "Full Workflow plans a smoke step");

    const auto smoke_only = PlanStudioDebuggerSteps(StudioDebuggerRunMode::SmokeRun, tabular);
    Check(smoke_only.back().id == "smoke" && smoke_only.back().required,
          "Smoke Run mode keeps smoke required");
    Check(AggregateStudioDebuggerOutcome(smoke_only, false, false) == StudioDebuggerOutcome::Unsupported,
          "Smoke Run on tabular graph is Unsupported before running");

    const auto runtime = PlanStudioDebuggerSteps(StudioDebuggerRunMode::RuntimeTrace, tabular);
    Check(runtime.size() == 1 && runtime[0].required, "Runtime Trace requires runtime evidence");
}

void TestOutcomeAggregation() {
    using S = StudioDebuggerStepState;
    Check(AggregateStudioDebuggerOutcome({}, false, false) == StudioDebuggerOutcome::NotStarted,
          "no steps -> Not started");
    Check(AggregateStudioDebuggerOutcome({MakeStep("a", S::Pending)}, false, false) ==
              StudioDebuggerOutcome::NotStarted,
          "pending only -> Not started");
    Check(AggregateStudioDebuggerOutcome({MakeStep("a", S::Passed), MakeStep("b", S::Passed)}, false, false) ==
              StudioDebuggerOutcome::Passed,
          "all passed -> Passed");
    // tofix96: a later passing step never erases an earlier required failure.
    Check(AggregateStudioDebuggerOutcome({MakeStep("a", S::Failed), MakeStep("b", S::Passed)}, false, false) ==
              StudioDebuggerOutcome::Failed,
          "required failure wins over later pass");
    Check(AggregateStudioDebuggerOutcome({MakeStep("a", S::Passed), MakeStep("b", S::Failed, false)}, false, false) ==
              StudioDebuggerOutcome::NeedsAttention,
          "optional failure -> Needs attention");
    Check(AggregateStudioDebuggerOutcome({MakeStep("a", S::Passed), MakeStep("b", S::Unsupported, false)}, false, false) ==
              StudioDebuggerOutcome::Passed,
          "optional unsupported step does not block Passed");
    Check(AggregateStudioDebuggerOutcome({MakeStep("a", S::Passed), MakeStep("b", S::Warning)}, false, false) ==
              StudioDebuggerOutcome::NeedsAttention,
          "warning step -> Needs attention");
    Check(AggregateStudioDebuggerOutcome({MakeStep("a", S::Passed)}, false, true) ==
              StudioDebuggerOutcome::NeedsAttention,
          "external warnings -> Needs attention");
    Check(AggregateStudioDebuggerOutcome({MakeStep("a", S::Passed), MakeStep("b", S::Stopped)}, false, false) ==
              StudioDebuggerOutcome::Stopped,
          "stopped step -> Stopped");
    Check(AggregateStudioDebuggerOutcome({MakeStep("a", S::Passed), MakeStep("b", S::Pending)}, true, false) ==
              StudioDebuggerOutcome::Stopped,
          "stop flag -> Stopped");
    Check(AggregateStudioDebuggerOutcome({MakeStep("a", S::Failed), MakeStep("b", S::Stopped)}, true, false) ==
              StudioDebuggerOutcome::Failed,
          "failure before stop stays Failed");
    Check(AggregateStudioDebuggerOutcome({MakeStep("a", S::Passed), MakeStep("b", S::Running)}, false, false) ==
              StudioDebuggerOutcome::Running,
          "running step -> Running");
}

void TestTraceStatusVocabulary() {
    Check(TraceStatusTone("ok") == DebuggerTone::Success, "ok is success");
    Check(TraceStatusTone("NaN") == DebuggerTone::Danger, "NaN is danger (case-insensitive)");
    Check(TraceStatusTone("shape_mismatch") == DebuggerTone::Warning, "shape mismatch is warning");
    Check(TraceStatusLabel("shape_mismatch") == "shape mismatch", "label replaces underscores");
    Check(TraceStatusLabel("nan") == "NaN", "nan label");
    Check(WorstTraceStatus({"ok", "shape_mismatch", "ok"}) == "shape_mismatch", "worst picks warning over ok");
    Check(WorstTraceStatus({"warning", "failed"}) == "failed", "worst picks failure");
    Check(WorstTraceStatus({}).empty(), "worst of nothing is empty");
}

StudioDebuggerSnapshot OwnerMachineSnapshot() {
    StudioDebuggerSnapshot snapshot;
    snapshot.run_id = "debug-20261001-101500";
    snapshot.mode = StudioDebuggerRunMode::FullWorkflow;
    snapshot.graph_domain = "Tabular";
    snapshot.node_count = 7;
    snapshot.link_count = 6;
    snapshot.duration_seconds = 18.4;
    snapshot.steps = PlanStudioDebuggerSteps(snapshot.mode, EvaluateSmokeCapability(snapshot.graph_domain));
    for (auto& step : snapshot.steps) {
        if (step.state == StudioDebuggerStepState::Pending) step.state = StudioDebuggerStepState::Passed;
    }
    ValidationIssue warning;
    warning.level = IssueLevel::Warning;
    warning.node_id = 3;
    warning.message = "Dense 64 receives 8000 sparse features";
    snapshot.issues.push_back(warning);
    snapshot.outcome = AggregateStudioDebuggerOutcome(snapshot.steps, false, !snapshot.issues.empty());

    for (int node = 1; node <= 7; ++node) {
        DebugTraceRecord trace;
        trace.run_id = snapshot.run_id;
        trace.node_id = node;
        trace.status = node == 3 ? "shape_mismatch" : "ok";
        snapshot.traces.push_back(trace);
    }
    DebugRecommendation fix;
    fix.node_id = 3;
    fix.title = "Keep TF-IDF output sparse";
    snapshot.recommendations.push_back(fix);

    snapshot.execution.available = true;
    snapshot.execution.effective_backend = "cuda";
    snapshot.execution.effective_device_name = "NVIDIA GeForce GTX 1050 Ti";
    snapshot.execution.residency_verdict = "device_resident";
    snapshot.execution.native_cpu_fallback_count = 0;
    return snapshot;
}

void TestStatusLine() {
    const auto snapshot = OwnerMachineSnapshot();
    Check(snapshot.outcome == StudioDebuggerOutcome::NeedsAttention, "owner scenario needs attention");

    const auto line = BuildStudioDebuggerStatusLine(snapshot, true, false, "", 0.0f);
    Check(line.label == "Needs attention", "status label");
    Check(line.tone == DebuggerTone::Warning, "status tone");
    Check(line.summary == "Full Workflow finished in 18.4 s", "status summary: " + line.summary);
    Check(line.counts.traces == 7 && line.counts.warnings == 1 && line.counts.errors == 0 &&
              line.counts.fixes == 1,
          "status counts");
    Check(line.backend == "CUDA · NVIDIA GeForce GTX 1050 Ti", "backend: " + line.backend);
    Check(line.provenance == "device resident · 0 CPU fallbacks", "provenance: " + line.provenance);

    const auto running = BuildStudioDebuggerStatusLine(snapshot, true, true, "Local Debug", 0.62f);
    Check(running.outcome == StudioDebuggerOutcome::Running, "running outcome");
    Check(running.summary == "Local Debug · 62%", "running summary: " + running.summary);
    Check(running.backend.empty(), "no backend while running");

    const auto empty = BuildStudioDebuggerStatusLine(StudioDebuggerSnapshot{}, false, false, "", 0.0f);
    Check(empty.outcome == StudioDebuggerOutcome::NotStarted, "no session -> Not started");

    // tofix96: historical training evidence never claims this run's backend.
    auto historical = snapshot;
    historical.training_trace_historical = true;
    historical.training_trace.available = true;
    historical.training_trace.run_id = "train-1790797";
    const auto hist_line = BuildStudioDebuggerStatusLine(historical, true, false, "", 0.0f);
    Check(hist_line.backend.empty(), "historical evidence has no live backend");
    Check(hist_line.provenance.find("historical") != std::string::npos &&
              hist_line.provenance.find("train-1790797") != std::string::npos,
          "historical provenance names the training run");

    auto failed = snapshot;
    failed.outcome = StudioDebuggerOutcome::Failed;
    failed.failure_summary = "Preflight blocked: no loss node";
    const auto fail_line = BuildStudioDebuggerStatusLine(failed, true, false, "", 0.0f);
    Check(fail_line.summary.find("no loss node") != std::string::npos, "failure summary is shown");
}

void TestGraphLayout() {
    // Layered: Dataset -> TF-IDF -> Dense -> Dense -> Output, plus a side
    // branch Dataset -> Labels -> Loss <- Output.
    std::vector<DebuggerGraphNodeInput> nodes = {
        {1, "Dataset"}, {2, "TF-IDF"}, {3, "Dense 64"}, {4, "Dense 32"},
        {5, "Output"}, {6, "Labels"}, {7, "Loss"}};
    std::vector<DebuggerGraphLinkInput> links = {
        {1, 2}, {2, 3}, {3, 4}, {4, 5}, {1, 6}, {5, 7}, {6, 7}};
    const auto layered = LayoutDebuggerGraph(nodes, links, 120.0f, 40.0f, 30.0f, 16.0f);
    Check(!layered.from_canvas, "no positions -> layered layout");
    Check(layered.nodes.size() == 7, "all nodes placed");
    auto find = [&](const DebuggerGraphLayout& layout, int id) {
        for (const auto& box : layout.nodes) if (box.id == id) return box;
        return DebuggerGraphNodeBox{};
    };
    Check(find(layered, 1).x == 0.0f, "source in first column");
    Check(find(layered, 7).x == 5 * 150.0f, "loss after output (longest path)");
    Check(find(layered, 6).x == 150.0f && find(layered, 6).y == 56.0f, "labels share column 1, second row");
    Check(layered.width == 6 * 120.0f + 5 * 30.0f, "layered width");
    Check(layered.height == 2 * 40.0f + 16.0f, "layered height");

    // A cycle must not recurse forever.
    const auto cyclic = LayoutDebuggerGraph({{1, "A"}, {2, "B"}}, {{1, 2}, {2, 1}}, 100, 30, 20, 10);
    Check(cyclic.nodes.size() == 2, "cyclic graph laid out");

    // Canvas positions keep the relative arrangement.
    std::vector<DebuggerGraphNodeInput> placed = {
        {1, "Dataset", true, 100.0f, 300.0f}, {2, "Dense", true, 500.0f, 300.0f},
        {3, "Loss", true, 500.0f, 560.0f}};
    const auto canvas = LayoutDebuggerGraph(placed, {{1, 2}, {2, 3}}, 120.0f, 40.0f, 30.0f, 16.0f);
    Check(canvas.from_canvas, "all positions -> canvas layout");
    Check(find(canvas, 1).x == 0.0f && find(canvas, 1).y == 0.0f, "canvas layout starts at origin");
    Check(find(canvas, 2).x > find(canvas, 1).x && find(canvas, 2).y == find(canvas, 1).y,
          "canvas keeps left-to-right order");
    Check(find(canvas, 3).y > find(canvas, 2).y && find(canvas, 3).x == find(canvas, 2).x,
          "canvas keeps top-to-bottom order");

    // One unplaced node falls back to layered.
    placed.push_back({4, "New"});
    Check(!LayoutDebuggerGraph(placed, {}, 120, 40, 30, 16).from_canvas, "partial positions -> layered");
    Check(LayoutDebuggerGraph({}, {}, 120, 40, 30, 16).nodes.empty(), "empty graph");
}

void TestNodeAggregation() {
    const auto status = AggregateDebuggerNodeStatus(OwnerMachineSnapshot());
    Check(status.size() == 7, "every traced node aggregated");
    const auto& dense = status.at(3);
    Check(dense.worst_status == "shape_mismatch", "dense worst status: " + dense.worst_status);
    Check(dense.trace_count == 1 && dense.issue_count == 1 && dense.recommendation_count == 1,
          "dense counts");
    Check(status.at(1).worst_status == "ok", "dataset ok");
}

void TestRunSummaryRoundTrip() {
    auto original = OwnerMachineSnapshot();
    original.training_trace_historical = true;
    original.steps[2].detail = "Smoke Run supports text graphs only";
    const auto trace = BuildStudioDebuggerRunSummaryTrace(original);
    Check(trace.phase == "RunSummary" && trace.role == DebugTraceRole::StudioEvent, "summary trace identity");
    Check(trace.status == "needs_attention", "summary trace status");

    StudioDebuggerSnapshot restored;
    restored.traces.push_back(trace);
    Check(ApplyStudioDebuggerRunSummaryTrace(restored), "summary trace applied");
    Check(restored.mode == original.mode, "mode restored");
    Check(restored.outcome == original.outcome, "outcome restored");
    Check(restored.graph_domain == "Tabular", "domain restored");
    Check(restored.duration_seconds == 18.4, "duration restored");
    Check(restored.training_trace_historical, "provenance restored");
    Check(restored.node_count == 7 && restored.link_count == 6, "counts restored");
    Check(restored.steps.size() == original.steps.size(), "steps restored");
    for (size_t i = 0; i < restored.steps.size() && i < original.steps.size(); ++i) {
        Check(restored.steps[i].id == original.steps[i].id &&
                  restored.steps[i].state == original.steps[i].state &&
                  restored.steps[i].required == original.steps[i].required &&
                  restored.steps[i].detail == original.steps[i].detail,
              "step restored: " + original.steps[i].id);
    }

    StudioDebuggerSnapshot legacy;
    Check(!ApplyStudioDebuggerRunSummaryTrace(legacy), "run without summary trace reports false");
}

} // namespace

int main() {
    TestModeAndStateKeysRoundTrip();
    TestSmokeCapabilityBeforeDispatch();
    TestStepPlans();
    TestOutcomeAggregation();
    TestTraceStatusVocabulary();
    TestStatusLine();
    TestGraphLayout();
    TestNodeAggregation();
    TestRunSummaryRoundTrip();
    if (g_failures != 0) {
        std::cerr << g_failures << " studio debugger presentation check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "studio debugger presentation contract passed\n";
    return EXIT_SUCCESS;
}
