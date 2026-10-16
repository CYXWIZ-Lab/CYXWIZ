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
#include "../../core/studio_debugger_presentation.h"
#include "../icons.h"
#include <atomic>
#include <cstddef>
#include <chrono>
#include <imgui.h>
#include <functional>
#include <map>
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

enum class StudioDebuggerSection {
    Overview = 0,
    Data,
    Model,
    Training,
    Runtime,
    Diagnostics
};

class StudioDebuggerPanel : public Panel {
public:
    StudioDebuggerPanel();
    ~StudioDebuggerPanel() override = default;

    void Render() override;
    const char* GetIcon() const override { return ICON_FA_BUG; }

    // Returns the work for one run (called on the UI thread with the frozen
    // graph); the returned function runs on a worker and reports through the
    // control it is given.
    using RunDebugCallback = std::function<std::function<StudioDebuggerSnapshot(
        const StudioDebuggerRunControl&)>(StudioDebuggerRunMode, int sample_index, int explain_node_id)>;
    using RunCompletedCallback = std::function<void(const StudioDebuggerSnapshot&)>;
    using FocusNodeCallback = std::function<void(int)>;
    // The current graph's compiled data domain, for Smoke capability.
    using GraphDomainCallback = std::function<std::string()>;

    void SetRunDebugCallback(RunDebugCallback callback) { run_debug_callback_ = std::move(callback); }
    void SetRunCompletedCallback(RunCompletedCallback callback) { run_completed_callback_ = std::move(callback); }
    void SetFocusNodeCallback(FocusNodeCallback callback) { focus_node_callback_ = std::move(callback); }
    void SetGraphDomainCallback(GraphDomainCallback callback) { graph_domain_callback_ = std::move(callback); }

    // One run path for Run, F6 and Explain Node. False if a run is active.
    bool StartRun(StudioDebuggerRunMode mode, int sample_index, int explain_node_id = -1);
    void RequestStop();
    bool IsRunning() const { return run_in_progress_; }
    // True once after the user asked the next run to rebuild prepared data.
    bool ConsumeRebuildPreparedDataRequest() {
        const bool requested = rebuild_prepared_data_next_run_;
        rebuild_prepared_data_next_run_ = false;
        return requested;
    }

    void SetSession(StudioDebuggerSnapshot session);
    void ShowRuntimeProfile();
    void ShowNodeExplanation(int node_id);
    void Clear();

    bool HasSession() const { return has_session_; }
    std::string GetSelectedTraceIdForAssistant() const;
    std::string BuildAssistantDebuggerContextJson() const;
    std::string BuildAssistantTrainingContextJson() const;

private:
    struct AsyncRunState {
        std::mutex mutex;
        std::optional<StudioDebuggerSnapshot> result;
        std::vector<StudioDebuggerStep> steps;
        std::string running_step;
        float progress = 0.0f;
        std::atomic<bool> stop{false};
    };
    struct AsyncLoadState {
        std::mutex mutex;
        std::optional<StudioDebuggerSnapshot> result;
        std::vector<DebugRunStoreSummary> history;
    };

    void RenderToolbar();
    void RenderTraceSettings();
    void RenderSectionRail();
    void RenderStepChecklist(const std::vector<StudioDebuggerStep>& steps);
    void RenderRunningBoard();
    void RenderSessionStatusStrip();
    void RenderSampleStepper();
    void RenderSummaryCards();
    void RenderInspectorHeader();
    // Rebuilds the graph layout and per-node / per-section status only when
    // the shown run changes, not every frame.
    void RebuildViewModelIfNeeded();
    void RenderWorkbenchBody();
    void RenderActiveWorkspace();
    void RenderInspectorPane();
    void RenderTraceDrawer(float height);
    void RenderLensContent();
    void RenderRunHistory();
    void RenderRunComparison();
    void LoadStoredRun(const std::string& run_id);
    void RequestRunHistoryRefresh();
    void PollRunProgress();
    void RefreshSmokeCapability();
    // The latest run of this session, wherever it is held; null if none.
    const StudioDebuggerSnapshot* LatestRun() const;
    void RefreshLiveTrainingTrace();
    void RenderLiveTrainingStatus();
    void RenderOverview();
    void RenderGraphTraceView(float height = 0.0f);
    void RenderLastRun();
    void RenderTrainingTrace();
    void RenderMaterializationTrace(const TrainingTraceSummary& trace);
    void RenderRuntimeTimeline(const TrainingTraceSummary& trace);
    void RenderMemoryTrace(const TrainingTraceSummary& trace);
    void RenderBatchInspector();
    void RenderModelConstructionTrace();
    void RenderShapeProphecyTrace();
    void RenderGradientHealth();
    void RenderLossMetricExplainer();
    void RenderBackendDecisionAudit();
    void RenderTensorLifecycle();
    void RenderLayerTimingBreakdown(const TrainingTraceSummary& trace);
    void RenderTraceTimeline();
    void RenderTraceFilters();
    void RenderStudioEvents();
    void RenderSelectedTraceDetails();
    void RenderTextPayloadInspector(const DebugTraceRecord& trace);
    void RenderTraceDiagnosis(const DebugTraceRecord& trace);
    void RenderIssueList();
    void RenderRecommendations();
    bool TraceMatchesActiveLens(const DebugTraceRecord& trace) const;
    bool TraceMatchesWorkflowFilter(const DebugTraceRecord& trace) const;
    void SelectSection(StudioDebuggerSection section);
    const char* ActiveLensName() const;
    static std::string FormatShape(const std::vector<size_t>& shape);

    RunDebugCallback run_debug_callback_;
    RunCompletedCallback run_completed_callback_;
    FocusNodeCallback focus_node_callback_;
    GraphDomainCallback graph_domain_callback_;
    std::shared_ptr<AsyncRunState> pending_run_state_;
    uint64_t pending_task_id_ = 0;
    std::shared_ptr<AsyncLoadState> pending_load_state_;
    bool load_in_progress_ = false;
    bool history_refresh_in_progress_ = false;
    bool history_loaded_ = false;
    // Live view of the running run, copied from the worker each frame.
    StudioDebuggerRunMode running_mode_ = StudioDebuggerRunMode::FullWorkflow;
    std::vector<StudioDebuggerStep> running_steps_;
    std::string running_step_;
    float running_progress_ = 0.0f;
    bool stop_requested_ = false;
    std::chrono::steady_clock::time_point run_started_{};
    int pending_explain_node_id_ = -1;
    SmokeCapability smoke_capability_;
    bool smoke_capability_known_ = false;

    // View model for the shown run (see RebuildViewModelIfNeeded).
    std::string view_key_;
    std::map<int, DebuggerNodeStatus> view_node_status_;
    DebuggerGraphLayout view_graph_layout_;
    std::vector<DebuggerGraphLinkInput> view_graph_links_;
    std::map<StudioDebuggerSection, DebuggerTone> view_section_tone_;
    int selected_graph_node_ = -1;
    // Selected sub-view per section (Summary / Runs / Compare, ...).
    int section_view_[6] = {0, 0, 0, 0, 0, 0};

    StudioDebuggerSnapshot session_;
    // The latest run, held here (moved, not copied) only while an older saved
    // run is shown; otherwise the latest run is session_ itself.
    std::optional<StudioDebuggerSnapshot> parked_latest_;
    bool has_session_ = false;
    std::string current_run_id_;
    int selected_trace_index_ = -1;
    bool trace_settings_initialized_ = false;
    bool trace_persist_enabled_ = true;
    int trace_persist_every_n_events_ = 1000;
    int trace_max_recent_events_ = 200;
    StudioDebuggerSection active_section_ = StudioDebuggerSection::Overview;
    bool section_selection_pending_ = false;
    StudioDebuggerLens active_lens_ = StudioDebuggerLens::Overview;
    StudioDebuggerRunMode run_mode_ = StudioDebuggerRunMode::FullWorkflow;
    int selected_sample_index_ = 0;
    std::string selected_runtime_event_key_;
    std::optional<DebugTraceRecord> run_comparison_trace_;
    std::string run_comparison_baseline_id_;
    std::string run_comparison_current_id_;
    bool comparison_loading_ = false;
    bool rebuild_prepared_data_next_run_ = false;
    char trace_search_[128] = {};
    bool trace_attention_only_ = false;
    bool trace_drawer_open_ = false;
    float trace_drawer_height_ = 220.0f;
    float inspector_width_ = 360.0f;
    bool inspector_expanded_ = false;
    bool run_in_progress_ = false;
    std::string run_status_message_;
    std::chrono::steady_clock::time_point next_training_trace_refresh_{};
};

} // namespace cyxwiz
