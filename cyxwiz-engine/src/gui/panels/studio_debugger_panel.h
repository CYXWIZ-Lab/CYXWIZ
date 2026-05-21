#pragma once

#include "../panel.h"
#include "../../core/debug_session.h"
#include "../../core/debug_executor.h"
#include "../../core/crash_run_recorder.h"
#include "../../core/smoke_run_executor.h"
#include "../../core/debug_recommendation_engine.h"
#include "../../core/training_trace_collector.h"
#include "../../core/debug_run_store.h"
#include "../../core/async_task_manager.h"
#include "../icons.h"
#include <cstddef>
#include <imgui.h>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

enum class StudioDebuggerLens {
    Overview = 0,
    Preprocessing,
    Shapes,
    Values,
    Gradients,
    Runtime,
    StudioEvents,
    Recommendations
};

enum class StudioDebuggerRunMode {
    FullWorkflow = 0,
    Preflight,
    LocalDebug,
    SmokeRun,
    RuntimeTrace
};

struct StudioDebuggerSnapshot {
    bool success = false;
    bool has_debug_result = false;
    std::string run_id;

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
    std::vector<DebugRecommendation> recommendations;
    std::vector<DebugRunStoreSummary> run_history;
};

class StudioDebuggerPanel : public Panel {
public:
    StudioDebuggerPanel();
    ~StudioDebuggerPanel() override = default;

    void Render() override;
    const char* GetIcon() const override { return ICON_FA_BUG; }

    using RunDebugCallback = std::function<std::function<StudioDebuggerSnapshot()>(StudioDebuggerRunMode, int)>;
    using FocusNodeCallback = std::function<void(int)>;

    void SetRunDebugCallback(RunDebugCallback callback) { run_debug_callback_ = std::move(callback); }
    void SetFocusNodeCallback(FocusNodeCallback callback) { focus_node_callback_ = std::move(callback); }

    void SetSession(const StudioDebuggerSnapshot& session);
    void Clear();

    bool HasSession() const { return has_session_; }

private:
    struct AsyncRunState {
        std::mutex mutex;
        std::optional<StudioDebuggerSnapshot> result;
    };

    void RenderToolbar();
    void RenderTraceSettings();
    void RenderLensContent();
    void RenderRunHistory();
    void RenderRunComparison();
    void LoadStoredRun(const std::string& run_id);
    void RefreshLiveTrainingTrace();
    void RenderLiveTrainingStatus();
    void RenderOverview();
    void RenderGraphTraceView();
    void RenderLastRun();
    void RenderTrainingTrace();
    void RenderRuntimeTimeline(const TrainingTraceSummary& trace);
    void RenderMemoryTrace(const TrainingTraceSummary& trace);
    void RenderLayerTimingBreakdown(const TrainingTraceSummary& trace);
    void RenderTraceTimeline();
    void RenderStudioEvents();
    void RenderSelectedTraceDetails();
    void RenderTextPayloadInspector(const DebugTraceRecord& trace);
    void RenderTraceDiagnosis(const DebugTraceRecord& trace);
    void RenderIssueList();
    void RenderRecommendations();
    bool TraceMatchesActiveLens(const DebugTraceRecord& trace) const;
    const char* ActiveLensName() const;
    static std::string FormatShape(const std::vector<size_t>& shape);

    RunDebugCallback run_debug_callback_;
    FocusNodeCallback focus_node_callback_;
    std::shared_ptr<AsyncRunState> pending_run_state_;
    uint64_t pending_task_id_ = 0;

    StudioDebuggerSnapshot session_;
    StudioDebuggerSnapshot current_session_;
    bool has_session_ = false;
    bool has_current_session_ = false;
    std::string current_run_id_;
    int selected_trace_index_ = -1;
    bool trace_settings_initialized_ = false;
    bool trace_persist_enabled_ = true;
    int trace_persist_every_n_events_ = 10;
    int trace_max_recent_events_ = 200;
    StudioDebuggerLens active_lens_ = StudioDebuggerLens::Overview;
    StudioDebuggerRunMode run_mode_ = StudioDebuggerRunMode::FullWorkflow;
    int selected_sample_index_ = 0;
    std::string selected_runtime_event_key_;
    bool run_in_progress_ = false;
    std::string run_status_message_;
};

} // namespace cyxwiz
