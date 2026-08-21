#include "studio_debugger_panel.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <memory>

namespace cyxwiz {

namespace {

std::string CompactRunLabel(const std::string& run_id) {
    constexpr size_t kVisibleSuffix = 28;
    if (run_id.empty()) {
        return "No saved run selected";
    }
    if (run_id.size() <= kVisibleSuffix) {
        return run_id;
    }
    return "..." + run_id.substr(run_id.size() - kVisibleSuffix);
}

} // namespace

void StudioDebuggerPanel::SelectSection(StudioDebuggerSection section) {
    if (active_section_ == section) {
        return;
    }

    active_section_ = section;
    selected_trace_index_ = -1;
    switch (section) {
        case StudioDebuggerSection::Overview:
            active_lens_ = StudioDebuggerLens::Overview;
            break;
        case StudioDebuggerSection::Data:
            active_lens_ = StudioDebuggerLens::Preprocessing;
            break;
        case StudioDebuggerSection::Model:
            active_lens_ = StudioDebuggerLens::Shapes;
            break;
        case StudioDebuggerSection::Training:
            active_lens_ = StudioDebuggerLens::Gradients;
            break;
        case StudioDebuggerSection::Runtime:
            active_lens_ = StudioDebuggerLens::Runtime;
            break;
        case StudioDebuggerSection::Diagnostics:
            active_lens_ = StudioDebuggerLens::StudioEvents;
            break;
    }
}

void StudioDebuggerPanel::RenderToolbar() {
    const std::array<const char*, 5> run_modes = {
        "Full Workflow", "Preflight", "Local Debug", "Smoke Run", "Runtime Trace"
    };
    bool open_options_popup = false;
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8.0f, 6.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(6.0f, 4.0f));
    if (ImGui::BeginTable("StudioDebuggerCommandBar", 4,
                          ImGuiTableFlags_SizingStretchProp |
                          ImGuiTableFlags_NoSavedSettings)) {
        ImGui::TableSetupColumn("Mode", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("Run", ImGuiTableColumnFlags_WidthFixed, 92.0f);
        ImGui::TableSetupColumn("Saved run", ImGuiTableColumnFlags_WidthStretch, 1.35f);
        ImGui::TableSetupColumn("Options", ImGuiTableColumnFlags_WidthFixed, 36.0f);
        ImGui::TableNextRow();

        ImGui::TableSetColumnIndex(0);
        int run_mode_index = static_cast<int>(run_mode_);
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::Combo("##StudioDebuggerRunMode", &run_mode_index,
                         run_modes.data(), static_cast<int>(run_modes.size()))) {
            run_mode_ = static_cast<StudioDebuggerRunMode>(run_mode_index);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Debugger run mode");
        }

        ImGui::TableSetColumnIndex(1);
        if (run_in_progress_) {
            ImGui::BeginDisabled();
        }
        const ImVec4 accent = ImGui::GetStyleColorVec4(ImGuiCol_HeaderActive);
        ImGui::PushStyleColor(ImGuiCol_Button, accent);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                              ImGui::GetStyleColorVec4(ImGuiCol_HeaderHovered));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                              ImGui::GetStyleColorVec4(ImGuiCol_Header));
        const char* run_label = run_in_progress_
            ? ICON_FA_CLOCK " Running##StudioDebuggerRunAction"
            : ICON_FA_PLAY " Run##StudioDebuggerRunAction";
        if (ImGui::Button(run_label, ImVec2(-FLT_MIN, 0.0f))) {
            if (run_debug_callback_) {
                const StudioDebuggerRunMode mode = run_mode_;
                const int sample_index = selected_sample_index_;
                auto task = std::make_shared<std::function<StudioDebuggerSnapshot()>>(
                    run_debug_callback_(mode, sample_index));
                auto state = std::make_shared<AsyncRunState>();
                pending_run_state_ = state;
                run_status_message_ =
                    "Studio Debugger run is executing in the background.";
                run_in_progress_ = true;
                pending_task_id_ = AsyncTaskManager::Instance().RunAsync(
                    "Studio Debugger Run",
                    [task, state](LambdaTask& async_task) {
                        async_task.ReportProgress(
                            0.05f, "Preparing Studio Debugger run");
                        if (async_task.ShouldStop()) {
                            return;
                        }
                        StudioDebuggerSnapshot result = (*task)();
                        {
                            std::lock_guard<std::mutex> lock(state->mutex);
                            state->result = std::move(result);
                        }
                        async_task.ReportProgress(
                            1.0f, "Studio Debugger run completed");
                    },
                    nullptr,
                    [this, state](bool success, const std::string& error) {
                        run_in_progress_ = false;
                        pending_task_id_ = 0;
                        if (!success) {
                            StudioDebuggerSnapshot failed;
                            failed.success = false;
                            failed.failure_summary = error.empty()
                                ? "Studio Debugger task failed."
                                : error;
                            SetSession(failed);
                            run_status_message_ = failed.failure_summary;
                            return;
                        }

                        std::optional<StudioDebuggerSnapshot> result;
                        {
                            std::lock_guard<std::mutex> lock(state->mutex);
                            result = std::move(state->result);
                        }
                        if (result) {
                            SetSession(*result);
                            run_status_message_.clear();
                        } else {
                            run_status_message_ =
                                "Studio Debugger task completed without a result.";
                        }
                    });
            }
        }
        ImGui::PopStyleColor(3);
        if (run_in_progress_) {
            ImGui::EndDisabled();
        }

        ImGui::TableSetColumnIndex(2);
        const std::string run_preview = CompactRunLabel(session_.run_id);
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::BeginCombo("##StudioDebuggerSavedRun", run_preview.c_str())) {
            if (session_.run_history.empty()) {
                ImGui::TextDisabled("No saved debugger runs.");
            }
            for (const auto& run : session_.run_history) {
                const bool selected = run.run_id == session_.run_id;
                const std::string label = std::string(
                    run.success ? ICON_FA_CIRCLE_CHECK "  "
                                : ICON_FA_CIRCLE_EXCLAMATION "  ") +
                    CompactRunLabel(run.run_id) + "##" + run.run_id;
                if (ImGui::Selectable(label.c_str(), selected)) {
                    LoadStoredRun(run.run_id);
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%s\n%s", run.run_id.c_str(),
                                      run.timestamp.c_str());
                }
            }
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered() && !session_.run_id.empty()) {
            ImGui::SetTooltip("Loaded run: %s", session_.run_id.c_str());
        }

        ImGui::TableSetColumnIndex(3);
        if (ImGui::Button(ICON_FA_GEAR "##StudioDebuggerOptions",
                          ImVec2(-FLT_MIN, 0.0f))) {
            open_options_popup = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Debugger options");
        }
        ImGui::EndTable();
    }

    ImGui::PopStyleVar(2);

    if (open_options_popup) {
        ImGui::OpenPopup("StudioDebuggerOptionsPopup");
    }
    ImGui::SetNextWindowSizeConstraints(ImVec2(380.0f, 0.0f),
                                        ImVec2(520.0f, FLT_MAX));
    if (ImGui::BeginPopup("StudioDebuggerOptionsPopup")) {
        ImGui::TextUnformatted("Debugger options");
        ImGui::Separator();
        ImGui::TextDisabled("VIEW");
        ImGui::Checkbox("Show trace timeline", &trace_drawer_open_);
        if (ImGui::Button("Reset pane sizes", ImVec2(-FLT_MIN, 0.0f))) {
            inspector_width_ = 360.0f;
            inspector_expanded_ = false;
            trace_drawer_height_ = 220.0f;
            trace_drawer_open_ = false;
        }
        ImGui::Separator();
        RenderTraceSettings();
        ImGui::Separator();
        ImGui::TextDisabled("SESSION");
        if (run_in_progress_) {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("Clear current session", ImVec2(-FLT_MIN, 0.0f))) {
            Clear();
            ImGui::CloseCurrentPopup();
        }
        if (run_in_progress_) {
            ImGui::EndDisabled();
        }
        ImGui::EndPopup();
    }
}

void StudioDebuggerPanel::RenderSectionNavigation() {
    if (!ImGui::BeginTabBar("StudioDebuggerSections",
                            ImGuiTabBarFlags_FittingPolicyScroll)) {
        return;
    }

    const auto section = [this](const char* label,
                                StudioDebuggerSection value) {
        const ImGuiTabItemFlags flags = section_selection_pending_ &&
                active_section_ == value
            ? ImGuiTabItemFlags_SetSelected
            : ImGuiTabItemFlags_None;
        if (ImGui::BeginTabItem(label, nullptr, flags)) {
            if (active_section_ != value) {
                SelectSection(value);
            }
            ImGui::EndTabItem();
        }
    };
    section("Overview", StudioDebuggerSection::Overview);
    section("Data", StudioDebuggerSection::Data);
    section("Model", StudioDebuggerSection::Model);
    section("Training", StudioDebuggerSection::Training);
    section("Runtime", StudioDebuggerSection::Runtime);
    section("Diagnostics", StudioDebuggerSection::Diagnostics);
    ImGui::EndTabBar();
    section_selection_pending_ = false;
}

void StudioDebuggerPanel::RenderSessionStatusStrip() {
    const bool show_execution = ImGui::GetContentRegionAvail().x >= 700.0f;
    const char* state = run_in_progress_
        ? "Running"
        : has_session_ ? (session_.success ? "Passed" : "Needs attention")
                       : "Ready";
    ImGui::TextDisabled("%s  |  %zu traces  |  %zu issues  |  %zu fixes",
                        state,
                        session_.traces.size(),
                        session_.issues.size(),
                        session_.recommendations.size());

    if (show_execution) {
        ImGui::SameLine();
        if (session_.execution.available) {
            ImGui::TextDisabled("|  %s:%d  %s",
                session_.execution.effective_backend.empty()
                    ? "unknown" : session_.execution.effective_backend.c_str(),
                session_.execution.effective_device_id,
                session_.execution.residency_verdict.empty()
                    ? "residency unobserved"
                    : session_.execution.residency_verdict.c_str());
        } else if (!run_status_message_.empty()) {
            ImGui::TextDisabled("|  %s", run_status_message_.c_str());
        }
    }
}

void StudioDebuggerPanel::RenderPreprocessingSampleSelector() {
    ImGui::BeginGroup();
    ImGui::AlignTextToFramePadding();
    ImGui::TextDisabled("Preprocessing sample");
    ImGui::SameLine();
    if (run_in_progress_) {
        ImGui::BeginDisabled();
    }
    if (ImGui::SmallButton("-##StudioDebuggerPreprocessingSample")) {
        selected_sample_index_ = std::max(0, selected_sample_index_ - 1);
    }
    ImGui::SameLine();
    ImGui::Text("%d", selected_sample_index_);
    ImGui::SameLine();
    if (ImGui::SmallButton("+##StudioDebuggerPreprocessingSample")) {
        ++selected_sample_index_;
    }
    if (run_in_progress_) {
        ImGui::EndDisabled();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("trace only");
    ImGui::EndGroup();
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Selects the dataset row inspected by preprocessing traces.\n"
            "It does not change the Local Debug synthetic batch or the "
            "Smoke Run execution set.");
    }
}

void StudioDebuggerPanel::RenderActiveWorkspace() {
    const ImGuiTabBarFlags tab_flags = ImGuiTabBarFlags_FittingPolicyScroll;
    switch (active_section_) {
        case StudioDebuggerSection::Overview:
            if (ImGui::BeginTabBar("StudioDebuggerOverviewViews", tab_flags)) {
                if (ImGui::BeginTabItem("Summary")) {
                    active_lens_ = StudioDebuggerLens::Overview;
                    ImGui::BeginChild("StudioDebuggerOverviewSummary",
                                      ImVec2(0.0f, 0.0f), false);
                    RenderOverview();
                    ImGui::EndChild();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Graph")) {
                    active_lens_ = StudioDebuggerLens::Overview;
                    RenderGraphTraceView();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Runs")) {
                    active_lens_ = StudioDebuggerLens::Overview;
                    RenderRunHistory();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Comparison")) {
                    active_lens_ = StudioDebuggerLens::Overview;
                    RenderRunComparison();
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
            return;

        case StudioDebuggerSection::Data:
            RenderPreprocessingSampleSelector();
            ImGui::Spacing();
            if (ImGui::BeginTabBar("StudioDebuggerDataViews", tab_flags)) {
                if (ImGui::BeginTabItem("Pipeline")) {
                    active_lens_ = StudioDebuggerLens::Preprocessing;
                    RenderGraphTraceView();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Batch")) {
                    active_lens_ = StudioDebuggerLens::Preprocessing;
                    RenderBatchInspector();
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
            return;

        case StudioDebuggerSection::Model:
            if (ImGui::BeginTabBar("StudioDebuggerModelViews", tab_flags)) {
                if (ImGui::BeginTabItem("Construction")) {
                    active_lens_ = StudioDebuggerLens::Shapes;
                    RenderModelConstructionTrace();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Shape comparison")) {
                    active_lens_ = StudioDebuggerLens::Shapes;
                    RenderShapeProphecyTrace();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Graph path")) {
                    active_lens_ = StudioDebuggerLens::Shapes;
                    RenderGraphTraceView();
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
            return;

        case StudioDebuggerSection::Training:
            if (ImGui::BeginTabBar("StudioDebuggerTrainingViews", tab_flags)) {
                if (ImGui::BeginTabItem("Gradients")) {
                    active_lens_ = StudioDebuggerLens::Gradients;
                    RenderGradientHealth();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Loss & metrics")) {
                    active_lens_ = StudioDebuggerLens::Values;
                    RenderLossMetricExplainer();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Trace map")) {
                    active_lens_ = StudioDebuggerLens::Values;
                    RenderGraphTraceView();
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
            return;

        case StudioDebuggerSection::Runtime:
            RefreshLiveTrainingTrace();
            if (ImGui::BeginTabBar("StudioDebuggerRuntimeViews", tab_flags)) {
                if (ImGui::BeginTabItem("Execution")) {
                    active_lens_ = StudioDebuggerLens::Runtime;
                    RenderTrainingTrace();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Timeline")) {
                    active_lens_ = StudioDebuggerLens::Runtime;
                    RenderRuntimeTimeline(session_.training_trace);
                    ImGui::Spacing();
                    RenderLayerTimingBreakdown(session_.training_trace);
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Materialization")) {
                    active_lens_ = StudioDebuggerLens::Runtime;
                    RenderMaterializationTrace(session_.training_trace);
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Backend")) {
                    active_lens_ = StudioDebuggerLens::Runtime;
                    RenderBackendDecisionAudit();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Tensors")) {
                    active_lens_ = StudioDebuggerLens::Runtime;
                    RenderTensorLifecycle();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Memory")) {
                    active_lens_ = StudioDebuggerLens::Runtime;
                    RenderMemoryTrace(session_.training_trace);
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Crash")) {
                    active_lens_ = StudioDebuggerLens::Runtime;
                    RenderLastRun();
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
            return;

        case StudioDebuggerSection::Diagnostics:
            if (ImGui::BeginTabBar("StudioDebuggerDiagnosticViews", tab_flags)) {
                if (ImGui::BeginTabItem("Events")) {
                    active_lens_ = StudioDebuggerLens::StudioEvents;
                    RenderStudioEvents();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Replay & compare")) {
                    active_lens_ = StudioDebuggerLens::Overview;
                    RenderRunComparison();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Crash report")) {
                    active_lens_ = StudioDebuggerLens::Runtime;
                    RenderLastRun();
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
            return;
    }
}

void StudioDebuggerPanel::RenderInspectorPane() {
    if (ImGui::BeginTable("StudioDebuggerInspectorHeader", 2,
                          ImGuiTableFlags_SizingStretchProp |
                          ImGuiTableFlags_NoSavedSettings)) {
        ImGui::TableSetupColumn("Title", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Width", ImGuiTableColumnFlags_WidthFixed,
                                30.0f);
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextDisabled("INSPECTOR");
        ImGui::TableSetColumnIndex(1);
        const char* width_icon = inspector_expanded_
            ? ICON_FA_COMPRESS "##StudioDebuggerInspectorWidth"
            : ICON_FA_EXPAND "##StudioDebuggerInspectorWidth";
        if (ImGui::SmallButton(width_icon)) {
            inspector_expanded_ = !inspector_expanded_;
            inspector_width_ = inspector_expanded_ ? 560.0f : 360.0f;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s inspector (or drag its left edge)",
                              inspector_expanded_ ? "Narrow" : "Widen");
        }
        ImGui::EndTable();
    }
    if (!ImGui::BeginTabBar("StudioDebuggerInspectorTabs",
                            ImGuiTabBarFlags_FittingPolicyScroll)) {
        return;
    }

    if (ImGui::BeginTabItem("Trace")) {
        RenderSelectedTraceDetails();
        ImGui::EndTabItem();
    }
    const std::string issues = "Issues (" +
        std::to_string(session_.issues.size()) + ")";
    if (ImGui::BeginTabItem(issues.c_str())) {
        RenderIssueList();
        ImGui::EndTabItem();
    }
    const std::string fixes = "Fixes (" +
        std::to_string(session_.recommendations.size()) + ")";
    if (ImGui::BeginTabItem(fixes.c_str())) {
        RenderRecommendations();
        ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
}

void StudioDebuggerPanel::RenderWorkbenchBody() {
    const bool wide = ImGui::GetContentRegionAvail().x >= 920.0f;
    if (!wide) {
        if (ImGui::BeginTabBar("StudioDebuggerCompactPanes",
                               ImGuiTabBarFlags_FittingPolicyScroll)) {
            if (ImGui::BeginTabItem("Workspace")) {
                ImGui::BeginChild("StudioDebuggerCompactWorkspace",
                                  ImVec2(0.0f, 0.0f), false,
                                  ImGuiWindowFlags_NoScrollbar |
                                  ImGuiWindowFlags_NoScrollWithMouse);
                RenderActiveWorkspace();
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Inspector")) {
                ImGui::BeginChild("StudioDebuggerCompactInspector",
                                  ImVec2(0.0f, 0.0f), false,
                                  ImGuiWindowFlags_NoScrollbar |
                                  ImGuiWindowFlags_NoScrollWithMouse);
                RenderInspectorPane();
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        return;
    }

    constexpr float splitter_width = 6.0f;
    constexpr float minimum_workspace_width = 360.0f;
    constexpr float minimum_inspector_width = 280.0f;
    const ImVec2 available_size = ImGui::GetContentRegionAvail();
    const float available_width = available_size.x;
    const float maximum_inspector_width = std::max(
        minimum_inspector_width,
        available_width - minimum_workspace_width - splitter_width);
    inspector_width_ = std::clamp(inspector_width_,
                                  minimum_inspector_width,
                                  maximum_inspector_width);
    const float workspace_width = std::max(
        minimum_workspace_width,
        available_width - inspector_width_ - splitter_width);

    ImGui::BeginChild("StudioDebuggerPrimaryWorkspace",
                      ImVec2(workspace_width, available_size.y), false,
                      ImGuiWindowFlags_NoScrollbar |
                      ImGuiWindowFlags_NoScrollWithMouse);
    RenderActiveWorkspace();
    ImGui::EndChild();

    ImGui::SameLine(0.0f, 0.0f);
    ImGui::InvisibleButton("##StudioDebuggerInspectorSplitter",
                           ImVec2(splitter_width, available_size.y));
    if (ImGui::IsItemActive()) {
        inspector_width_ -= ImGui::GetIO().MouseDelta.x;
        inspector_width_ = std::clamp(inspector_width_,
                                      minimum_inspector_width,
                                      maximum_inspector_width);
        inspector_expanded_ = inspector_width_ > 440.0f;
    }
    if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    }

    ImGui::SameLine(0.0f, 0.0f);
    ImGui::BeginChild("StudioDebuggerInspectorPane",
                      ImVec2(inspector_width_, available_size.y), false,
                      ImGuiWindowFlags_NoScrollbar |
                      ImGuiWindowFlags_NoScrollWithMouse);
    RenderInspectorPane();
    ImGui::EndChild();
}

void StudioDebuggerPanel::RenderTraceDrawer(float height) {
    ImGui::BeginChild("StudioDebuggerTraceDrawer", ImVec2(0.0f, height), false,
                      ImGuiWindowFlags_NoScrollbar |
                      ImGuiWindowFlags_NoScrollWithMouse);
    const char* toggle = trace_drawer_open_
        ? ICON_FA_CHEVRON_DOWN
        : ICON_FA_CHEVRON_RIGHT;
    if (ImGui::SmallButton((std::string(toggle) +
                            "##StudioDebuggerTraceDrawerToggle").c_str())) {
        trace_drawer_open_ = !trace_drawer_open_;
    }
    ImGui::SameLine();
    ImGui::TextUnformatted("Trace timeline");
    ImGui::SameLine();
    ImGui::TextDisabled("%s  |  %zu records",
                        ActiveLensName(), session_.traces.size());
    if (trace_drawer_open_ && height > 60.0f) {
        ImGui::Spacing();
        RenderTraceTimeline();
    }
    ImGui::EndChild();
}

void StudioDebuggerPanel::RenderLensContent() {
    const float total_height = ImGui::GetContentRegionAvail().y;
    const float collapsed_height = 32.0f;
    const float minimum_body_height = 150.0f;
    float drawer_height = trace_drawer_open_
        ? std::clamp(trace_drawer_height_, 145.0f,
                     std::max(145.0f, total_height * 0.55f))
        : collapsed_height;
    if (total_height - drawer_height < minimum_body_height) {
        drawer_height = std::max(
            collapsed_height, total_height - minimum_body_height);
    }

    const float splitter_height = trace_drawer_open_ ? 5.0f : 0.0f;
    const float body_height = std::max(
        80.0f, total_height - drawer_height - splitter_height);
    ImGui::BeginChild("StudioDebuggerWorkbenchBody",
                      ImVec2(0.0f, body_height), false,
                      ImGuiWindowFlags_NoScrollbar |
                      ImGuiWindowFlags_NoScrollWithMouse);
    RenderWorkbenchBody();
    ImGui::EndChild();

    if (trace_drawer_open_) {
        ImGui::InvisibleButton("##StudioDebuggerTraceDrawerSplitter",
                               ImVec2(ImGui::GetContentRegionAvail().x,
                                      splitter_height));
        if (ImGui::IsItemActive()) {
            trace_drawer_height_ -= ImGui::GetIO().MouseDelta.y;
            trace_drawer_height_ = std::clamp(
                trace_drawer_height_, 145.0f,
                std::max(145.0f, total_height - minimum_body_height));
        }
        if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        }
    }
    RenderTraceDrawer(drawer_height);
}

void StudioDebuggerPanel::Render() {
    if (!visible_) {
        return;
    }

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6.0f, 4.0f));
    std::string title = std::string(ICON_FA_BUG) +
        " Studio Debugger###StudioDebuggerPanel";
    if (ImGui::Begin(title.c_str(), &visible_)) {
        RenderToolbar();
        RenderSectionNavigation();
        RenderSessionStatusStrip();
        RenderLensContent();
    }
    ImGui::End();
    ImGui::PopStyleVar();
}

} // namespace cyxwiz
