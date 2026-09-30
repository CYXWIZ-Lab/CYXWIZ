#pragma once

// Pure presentation model for the Studio Debugger (TOFIX128): outcome rules,
// step plans, one status vocabulary, the status line and the graph layout.
// No ImGui, no globals; the panel only draws what this returns.

#include "studio_debugger_snapshot.h"

#include <map>
#include <string>
#include <vector>

namespace cyxwiz {

enum class DebuggerTone { Neutral = 0, Success, Warning, Danger, Info, Muted };

const char* StudioDebuggerRunModeLabel(StudioDebuggerRunMode mode);
const char* StudioDebuggerRunModeKey(StudioDebuggerRunMode mode);
bool ParseStudioDebuggerRunModeKey(const std::string& key, StudioDebuggerRunMode& mode);

const char* StudioDebuggerOutcomeLabel(StudioDebuggerOutcome outcome);
const char* StudioDebuggerOutcomeKey(StudioDebuggerOutcome outcome);
bool ParseStudioDebuggerOutcomeKey(const std::string& key, StudioDebuggerOutcome& outcome);
DebuggerTone StudioDebuggerOutcomeTone(StudioDebuggerOutcome outcome);

const char* StudioDebuggerStepStateLabel(StudioDebuggerStepState state);
const char* StudioDebuggerStepStateKey(StudioDebuggerStepState state);
bool ParseStudioDebuggerStepStateKey(const std::string& key, StudioDebuggerStepState& state);
DebuggerTone StudioDebuggerStepStateTone(StudioDebuggerStepState state);

// Smoke Run capability from the graph's data domain, known before dispatch.
struct SmokeCapability {
    bool supported = false;
    std::string reason;       // why not, when unsupported
    std::string alternative;  // what to use instead
};
SmokeCapability EvaluateSmokeCapability(const std::string& graph_domain);

// Steps a mode runs, in order, with which of them decide the outcome.
std::vector<StudioDebuggerStep> PlanStudioDebuggerSteps(
    StudioDebuggerRunMode mode, const SmokeCapability& smoke);

// Outcome from explicit per-step requirements (tofix96): a required failure
// wins, stop is kept, unsupported-only runs are Unsupported, warnings make it
// Needs attention; nothing run is Not started.
StudioDebuggerOutcome AggregateStudioDebuggerOutcome(
    const std::vector<StudioDebuggerStep>& steps, bool stopped, bool has_warnings);

// One vocabulary for raw trace statuses (ok, shape_mismatch, nan, ...).
DebuggerTone TraceStatusTone(const std::string& status);
std::string TraceStatusLabel(const std::string& status);
int TraceStatusSeverity(const std::string& status);  // higher = worse
std::string WorstTraceStatus(const std::vector<std::string>& statuses);

struct StudioDebuggerCounts {
    size_t traces = 0;
    size_t errors = 0;
    size_t warnings = 0;
    size_t fixes = 0;
};
StudioDebuggerCounts CountStudioDebuggerFindings(const StudioDebuggerSnapshot& snapshot);

// The single status line under the command bar.
struct StudioDebuggerStatusLine {
    StudioDebuggerOutcome outcome = StudioDebuggerOutcome::NotStarted;
    std::string label;
    DebuggerTone tone = DebuggerTone::Muted;
    std::string summary;      // "Full Workflow finished in 18.4 s"
    StudioDebuggerCounts counts;
    std::string backend;      // "CUDA · GTX 1050 Ti" or empty
    std::string provenance;   // "device-resident · 0 CPU fallbacks" / "historical training evidence"
};
StudioDebuggerStatusLine BuildStudioDebuggerStatusLine(
    const StudioDebuggerSnapshot& snapshot, bool has_session, bool running,
    const std::string& running_step, float running_progress);

// Graph trace layout: canvas positions when every node has one (scaled to
// fit), otherwise a layered left-to-right layout by dependency depth.
struct DebuggerGraphNodeInput {
    int id = -1;
    std::string name;
    bool has_position = false;
    float x = 0.0f;
    float y = 0.0f;
};
struct DebuggerGraphLinkInput {
    int from_node = -1;
    int to_node = -1;
};
struct DebuggerGraphNodeBox {
    int id = -1;
    std::string name;
    float x = 0.0f;  // top-left, layout space (0,0 = top-left of the graph)
    float y = 0.0f;
};
struct DebuggerGraphLayout {
    std::vector<DebuggerGraphNodeBox> nodes;
    float width = 0.0f;
    float height = 0.0f;
    bool from_canvas = false;
};
DebuggerGraphLayout LayoutDebuggerGraph(
    const std::vector<DebuggerGraphNodeInput>& nodes,
    const std::vector<DebuggerGraphLinkInput>& links,
    float node_width, float node_height, float gap_x, float gap_y);

// Per-node aggregate for the graph view.
struct DebuggerNodeStatus {
    std::string worst_status;
    size_t trace_count = 0;
    size_t issue_count = 0;
    size_t recommendation_count = 0;
};
std::map<int, DebuggerNodeStatus> AggregateDebuggerNodeStatus(const StudioDebuggerSnapshot& snapshot);

// Run summary trace: carries mode, outcome, steps, domain and provenance so a
// saved run restores them (stored with the other traces).
DebugTraceRecord BuildStudioDebuggerRunSummaryTrace(const StudioDebuggerSnapshot& snapshot);
// Restores the fields above from a run summary trace; false if none present.
bool ApplyStudioDebuggerRunSummaryTrace(StudioDebuggerSnapshot& snapshot);

} // namespace cyxwiz
