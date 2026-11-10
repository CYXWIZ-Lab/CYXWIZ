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
#include "../../core/layer_capture.h"
#include "../../core/layer_capture_store.h"
#include "../../core/training_snapshots_presentation.h"
#include "../../core/engine_config.h"
#include "../icons.h"
#include <atomic>
#include <cstddef>
#include <chrono>
#include <imgui.h>
#include <functional>
#include <utility>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace cyxwiz {

class LiveGraphCompile;
class DebugModelSession;
struct ItemsRun;
namespace occlusion {
enum class InputKind;
}

enum class StudioDebuggerLens {
    Overview = 0,
    Preprocessing,
    Shapes,
    Values,
    Gradients,
    Runtime,
    StudioEvents
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

    // Asked on the UI thread before a run starts: a message when the run
    // waits (its data loads first, as for Train) and is started again later.
    using RunWaitCallback =
        std::function<std::optional<std::string>(StudioDebuggerRunMode, int sample_index, int explain_node_id)>;

    void SetRunDebugCallback(RunDebugCallback callback) { run_debug_callback_ = std::move(callback); }
    // Items from the graph's data, prepared as validation sees them, in the
    // model's input layout: `count` of them spread over the data (Where it
    // looks' feature means, TOFIX178 6.3b; Top images, 6.3e). Called on the
    // UI thread with the graph frozen; the job runs on a worker (`why` set
    // when the items cannot be read). An empty job: the canvas does not
    // compile cleanly, or its data is being loaded first (`loading` names it:
    // a reopened project's Data Inputs load as for Train).
    struct DataItems {
        Tensor features;
        Tensor labels;
        std::vector<size_t> indices;           // each item's number in its source
        std::vector<std::string> names;        // "cat.10599.jpg"; empty when items have no name
        std::vector<std::string> class_names;  // in label order; empty when unknown
        std::string description;               // "320 images of cat_dog, every 78th from image 0"
        std::string why;
    };
    using DataItemsJob = std::function<DataItems(size_t count, const std::function<bool()>& stop)>;
    struct DataItemsStart {
        DataItemsJob job;
        std::string loading;  // "Cats and dogs": loading first; empty otherwise
        bool compiling = false;  // the canvas's compile is not up to date yet (just after a load)
    };
    using DataItemsCallback = std::function<DataItemsStart()>;
    // The view's message when the items cannot start now.
    static std::string DataItemsNotStarted(const DataItemsStart& start, const std::string& what);
    void SetDataItemsCallback(DataItemsCallback callback) { data_items_callback_ = std::move(callback); }
    // The Data Input loads a view waited for have ended (`failed` names what
    // could not be loaded, empty when all loaded): Top images starts then.
    void DataLoadsEnded(const std::string& failed);
    void SetRunWaitCallback(RunWaitCallback callback) { run_wait_callback_ = std::move(callback); }
    // The line under the run controls: a wait, or why a waiting run did not
    // start.
    void SetRunStatusMessage(std::string message) { run_status_message_ = std::move(message); }
    void SetRunCompletedCallback(RunCompletedCallback callback) { run_completed_callback_ = std::move(callback); }
    void SetFocusNodeCallback(FocusNodeCallback callback) { focus_node_callback_ = std::move(callback); }
    void SetGraphDomainCallback(GraphDomainCallback callback) { graph_domain_callback_ = std::move(callback); }
    // The way back to the canvas and on to training (TOFIX178 UI-1): the
    // canvas, its compile report, and Train once Local Debug passed on the
    // graph as it is now.
    struct CanvasLinks {
        std::function<void()> show_canvas;
        std::function<void()> show_compile_report;
        std::function<void()> train;
        // The last Local Debug against the graph now: its headline and
        // whether it is a pass on this graph.
        std::function<std::pair<std::string, bool>()> local_debug;
    };
    void SetCanvasLinks(CanvasLinks links) { canvas_links_ = std::move(links); }

    // One run path for Run, F6 and Explain Node. False if a run is active.
    bool StartRun(StudioDebuggerRunMode mode, int sample_index, int explain_node_id = -1);
    void RequestStop();
    bool IsRunning() const { return run_in_progress_; }
    // The sample the debugger runs on (Local Debug, training layer snapshots).
    int SelectedSampleIndex() const { return selected_sample_index_; }
    // What the next Local Debug or training run captures and how the layer
    // views colour maps: the user's defaults (Preferences > Studio Debugger)
    // until this session's capture control changes them.
    core::StudioDebuggerConfig CaptureSettings() const;
    // The canvas compile, for the capture control's size estimate.
    void SetLiveCompile(const LiveGraphCompile* live_compile) { live_compile_ = live_compile; }
    // Full size (TOFIX178 D1): the main window hides the side panels while
    // `on` and brings back the ones it hid when it turns off (the toolbar
    // button again, leaving the debugger's tab or closing it).
    void SetFullSizeCallback(std::function<void(bool on)> callback) { full_size_callback_ = std::move(callback); }
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
    void RenderCaptureControl();
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
    // Overview > Runs: Local Debug runs and training runs with layer
    // snapshots (studio_debugger_runs.cpp, TOFIX178 6.3d).
    void RenderRunHistory();
    void RenderRunComparison();
    // Overview > Compare for a shown training run: its snapshot table.
    void RenderSnapshotTable();
    // Shows a training run's snapshots in place of the debug run's capture.
    void ShowTrainingRun(const std::string& run_id);
    // The shown run's sample facts: the training run's when one is shown.
    const StudioDebuggerSampleFacts& ShownSampleFacts() const;
    // The shown training run's epochs (none for a debug run).
    const std::optional<TrainingRunRecord>& ShownTrainingRecord() const;
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
    // Model > Layers (TOFIX178 6.2, studio_debugger_layers.cpp): the shown
    // run's layer capture, its views and the Inspector's sample and channel.
    void RenderLayersView();
    // False when the Layers view has nothing for the Inspector.
    bool RenderLayersInspector();
    // The picked item's input picture and model output, read once per item.
    void ReadLayerSample();
    // Model > Layers > Where it looks (studio_debugger_where_it_looks.cpp):
    // the shown capture's model rebuilt in eval mode off the UI thread, its
    // prediction for the picked item, and occlusion.
    void UpdateLayerModelSession();
    void RenderWhereItLooks(uint32_t sample_texture, int sample_width, int sample_height);
    // The shown capture's rebuilt model; null while it is built or when it cannot be.
    std::shared_ptr<DebugModelSession> LayerModelSession() const;
    // Model > Layers > Top images (studio_debugger_top_images.cpp, 6.3e).
    void RenderTopImages();
    // Model > Layers > Contributions (studio_debugger_contributions.cpp, 6.4).
    void RenderContributions(const std::string& weight_name);
    // Model > Layers > Embedding (studio_debugger_embedding.cpp, 6.4b) and the
    // Inspector's card for its clicked point.
    void RenderEmbedding();
    bool RenderEmbeddingPick();
    // The Inspector's card for the picture picked in Top images; false when
    // none is picked (or Top images is not shown).
    bool RenderTopImagesPick();
    // A channel clicked in Feature maps or Units: Top images' "the clicked channel".
    void NoteClickedChannel(size_t layer, size_t channel);
    // Where it looks with Method Saliency (6.3c): Compute and the result for
    // the picked item and target. `item_values`: token ids or feature values.
    void RenderSaliency(occlusion::InputKind kind, const std::vector<float>& item_values, float pad_id,
                        const std::string& target_name, uint32_t sample_texture, int sample_width, int sample_height);
    // The model's output for the picked item from the evaluation pass; null
    // until read (or when the model cannot be rebuilt).
    const std::vector<float>* EvalModelOutput() const;
    // Why the evaluation pass is not available ("" while it is being read).
    std::string EvalModelUnavailable() const;
    // The capture of the shown run: the live one, else read from the run
    // store in the background.
    void RefreshLayerCaptureSource();
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
    DataItemsCallback data_items_callback_;
    RunWaitCallback run_wait_callback_;
    RunCompletedCallback run_completed_callback_;
    FocusNodeCallback focus_node_callback_;
    CanvasLinks canvas_links_;
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

    // Model > Layers. The cache holds what the shown view read from the
    // device and its textures (studio_debugger_layers.cpp).
    struct LayerViewCache;
    std::shared_ptr<const LayerCapture> layer_capture_;
    std::string layer_capture_run_id_;   // the run the capture belongs to
    std::string layer_capture_label_;    // "local_debug", "epoch_0005"
    std::vector<std::string> layer_capture_labels_;
    // Stepping snapshots (6.3d) keeps the picked layer: its node, matched in
    // the next capture read.
    std::string layer_keep_node_;
    // Reads another snapshot of the shown run, keeping the layer, tab and channel.
    void ShowLayerSnapshot(const std::string& label);
    // Compare with (6.3d): another snapshot of the same run, read in the
    // background and drawn beside the shown one.
    std::string compare_label_;  // empty = none
    std::string compare_read_;   // "<run>/<label>" read (or being read)
    std::shared_ptr<const LayerCapture> compare_capture_;
    std::shared_ptr<LayerViewCache> compare_view_cache_;
    bool layer_capture_loading_ = false;
    bool layer_capture_listed_ = false;  // the run store was asked for this run
    size_t selected_layer_ = 0;
    int selected_layer_tab_ = 0;
    size_t selected_layer_sample_ = 0;   // item within a captured batch
    size_t selected_channel_rank_ = 0;   // in strongest-first order
    bool layer_all_channels_ = false;
    size_t heatmap_row_begin_ = 0;
    int heatmap_kernel_position_ = -1;   // conv weights: one kernel position, -1 = all
    std::shared_ptr<LayerViewCache> layer_view_cache_;
    struct WhereItLooksState;
    std::shared_ptr<WhereItLooksState> where_it_looks_;
    struct TopImagesState;
    std::shared_ptr<TopImagesState> top_images_;
    struct EmbeddingState;
    std::shared_ptr<EmbeddingState> embedding_;
    // The views' computations over the graph's data (studio_debugger_items_run.h).
    std::shared_ptr<ItemsRun> top_images_run_;
    std::shared_ptr<ItemsRun> embedding_run_;
    size_t clicked_channel_layer_ = std::numeric_limits<size_t>::max();
    size_t clicked_channel_ = 0;
    std::string picked_from_top_images_;  // the item clicked last, named
    // Contributions: the explained output or unit, and the clicked input.
    std::string contributions_key_;  // capture / layer / item they are for
    int contributions_output_ = -1;
    std::optional<size_t> contributions_pick_;
    std::string contributions_inspector_;
    // A training run shown instead of the debug run (6.3d); empty = none.
    std::string shown_training_run_;
    std::optional<TrainingRunRecord> shown_training_record_;
    std::vector<LayerCaptureInfo> shown_training_captures_;
    StudioDebuggerSampleFacts shown_training_facts_;
    std::vector<training_snapshots::RunRow> training_run_rows_;  // listed with the saved runs
    std::set<std::string> open_run_details_;
    // This session's capture settings once changed here; empty = the user's
    // defaults. Save as my default writes them and empties it.
    std::optional<core::StudioDebuggerConfig> capture_session_;
    const LiveGraphCompile* live_compile_ = nullptr;
    bool full_size_ = false;
    std::function<void(bool)> full_size_callback_;
    void SetFullSize(bool on);

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
