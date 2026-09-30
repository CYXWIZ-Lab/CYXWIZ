#include "studio_debugger_panel.h"
#include "studio_debugger_colors.h"
#include "../ui_buttons.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstdio>
#include <memory>

namespace cyxwiz {

namespace {

struct RunModeOption {
    StudioDebuggerRunMode mode;
    const char* description;
};

constexpr std::array<RunModeOption, 5> kRunModeOptions = {{
    {StudioDebuggerRunMode::FullWorkflow,
     "Runs Compile, Preflight, Smoke Run, Local Debug and collects runtime "
     "evidence. Every required step must pass; a Smoke Run the graph does not "
     "support is skipped, not failed."},
    {StudioDebuggerRunMode::Preflight,
     "Compiles the graph and checks dataset, preprocessing, shape, loss and "
     "training readiness. It does not execute model forward or backward passes."},
    {StudioDebuggerRunMode::LocalDebug,
     "Compiles and validates the graph, then executes one bounded synthetic "
     "batch through forward, loss, backward, gradient and optimizer checks. "
     "Timings are host wall-clock evidence; device kernel time is not claimed."},
    {StudioDebuggerRunMode::SmokeRun,
     "Exercises preprocessing and a bounded real-data training sample before "
     "full training. Supported for text graphs."},
    {StudioDebuggerRunMode::RuntimeTrace,
     "Inspects the latest training trace (runtime status, warnings, transfers, "
     "synchronizations, fallbacks) and crash evidence. It does not execute the "
     "graph."}
}};

const RunModeOption& GetRunModeOption(StudioDebuggerRunMode mode) {
    const auto option = std::find_if(
        kRunModeOptions.begin(), kRunModeOptions.end(),
        [mode](const RunModeOption& candidate) { return candidate.mode == mode; });
    return option == kRunModeOptions.end() ? kRunModeOptions.front() : *option;
}

std::string CompactRunLabel(const std::string& run_id) {
    constexpr size_t kVisibleSuffix = 24;
    if (run_id.size() <= kVisibleSuffix) {
        return run_id;
    }
    return "..." + run_id.substr(run_id.size() - kVisibleSuffix);
}

const char* StepIcon(StudioDebuggerStepState state) {
    switch (state) {
        case StudioDebuggerStepState::Passed: return ICON_FA_CIRCLE_CHECK;
        case StudioDebuggerStepState::Warning: return ICON_FA_TRIANGLE_EXCLAMATION;
        case StudioDebuggerStepState::Failed: return ICON_FA_CIRCLE_XMARK;
        case StudioDebuggerStepState::Running: return ICON_FA_SPINNER;
        case StudioDebuggerStepState::Stopped: return ICON_FA_CIRCLE_PAUSE;
        case StudioDebuggerStepState::Skipped:
        case StudioDebuggerStepState::Unsupported: return ICON_FA_MINUS;
        case StudioDebuggerStepState::Pending: return ICON_FA_CIRCLE;
    }
    return ICON_FA_CIRCLE;
}

ImVec4 StepColor(StudioDebuggerStepState state) {
    if (state == StudioDebuggerStepState::Pending) {
        return DebuggerFaint();
    }
    return DebuggerToneColor(StudioDebuggerStepStateTone(state));
}

std::string FormatSecondsShort(double seconds) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), seconds < 10.0 ? "%.1f s" : "%.0f s", seconds);
    return buffer;
}

// Sub-view selector: a segmented control when it fits, a combo otherwise,
// so no view depends on widening the panel.
void SubViewSelector(const char* id, const char* const* labels, int count, int* selected) {
    if (count <= 1) {
        return;
    }
    float needed = 0.0f;
    for (int i = 0; i < count; ++i) {
        needed += ImGui::CalcTextSize(labels[i]).x + 24.0f;
    }
    if (needed <= ImGui::GetContentRegionAvail().x) {
        ui::SegmentedControl(id, labels, count, selected);
        return;
    }
    ImGui::SetNextItemWidth(std::min(260.0f, ImGui::GetContentRegionAvail().x));
    ImGui::PushID(id);
    if (ImGui::BeginCombo("##view", labels[*selected])) {
        for (int i = 0; i < count; ++i) {
            if (ImGui::Selectable(labels[i], *selected == i)) {
                *selected = i;
            }
        }
        ImGui::EndCombo();
    }
    ImGui::PopID();
}

// Right-aligns the next item group of `width` on the current line when it
// fits; otherwise it starts on a new line.
void RightAlignOnLine(float width) {
    ImGui::SameLine();
    const float available = ImGui::GetContentRegionAvail().x;
    if (available >= width) {
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + available - width);
    } else {
        ImGui::NewLine();
    }
}

bool BeginCard(const char* id) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, DebuggerPanelBg());
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 8.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 10.0f));
    return ImGui::BeginChild(id, ImVec2(0.0f, 0.0f),
                             ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
}

void EndCard() {
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
}

void CardHeading(const char* text) {
    ImGui::PushStyleColor(ImGuiCol_Text, DebuggerBright());
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
}

} // namespace

void StudioDebuggerPanel::SelectSection(StudioDebuggerSection section) {
    if (active_section_ == section) {
        return;
    }

    active_section_ = section;
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

void StudioDebuggerPanel::RenderSampleStepper() {
    const bool running = run_in_progress_;
    ImGui::BeginGroup();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(DebuggerMuted(), "Sample");
    ImGui::SameLine(0.0f, 6.0f);
    if (ui::SecondaryButton(ICON_FA_MINUS "##StudioDebuggerSampleDown",
                            !running && selected_sample_index_ > 0,
                            running ? "Locked while a run is active" : "First sample")) {
        selected_sample_index_ = std::max(0, selected_sample_index_ - 1);
    }
    ImGui::SameLine(0.0f, 4.0f);
    ImGui::AlignTextToFramePadding();
    ImGui::Text("%d", selected_sample_index_);
    ImGui::SameLine(0.0f, 4.0f);
    if (ui::SecondaryButton(ICON_FA_PLUS "##StudioDebuggerSampleUp", !running,
                            "Locked while a run is active")) {
        ++selected_sample_index_;
    }
    ImGui::EndGroup();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip(
            "Dataset row inspected by preprocessing traces.\n"
            "It does not change the Local Debug synthetic batch or the Smoke Run set.");
    }
}

void StudioDebuggerPanel::RenderToolbar() {
    const bool running = run_in_progress_;
    bool open_options_popup = false;
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 6.0f));

    ImGui::AlignTextToFramePadding();
    ImGui::PushStyleColor(ImGuiCol_Text, DebuggerBright());
    ImGui::TextUnformatted("Studio Debugger");
    ImGui::PopStyleColor();
    ImGui::SameLine(0.0f, 16.0f);

    // Mode, with Smoke capability shown before anything is dispatched.
    const bool smoke_unavailable = smoke_capability_known_ && !smoke_capability_.supported;
    const auto mode_label = [&](StudioDebuggerRunMode mode) {
        std::string label = StudioDebuggerRunModeLabel(mode);
        if (mode == StudioDebuggerRunMode::SmokeRun && smoke_unavailable) {
            label += "  (not for this graph)";
        }
        return label;
    };
    const auto mode_tooltip = [&](StudioDebuggerRunMode mode) {
        ImGui::BeginTooltip();
        ImGui::TextUnformatted(StudioDebuggerRunModeLabel(mode));
        ImGui::Separator();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 30.0f);
        ImGui::TextUnformatted(GetRunModeOption(mode).description);
        if (smoke_unavailable && (mode == StudioDebuggerRunMode::SmokeRun ||
                                  mode == StudioDebuggerRunMode::FullWorkflow)) {
            ImGui::Spacing();
            ImGui::TextColored(DebuggerWarning(), "%s", smoke_capability_.reason.c_str());
            ImGui::TextColored(DebuggerMuted(), "%s", smoke_capability_.alternative.c_str());
        }
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    };
    ImGui::BeginDisabled(running);
    ImGui::SetNextItemWidth(std::clamp(ImGui::CalcTextSize(mode_label(run_mode_).c_str()).x + 40.0f,
                                       150.0f, 300.0f));
    if (ImGui::BeginCombo("##StudioDebuggerRunMode", mode_label(run_mode_).c_str())) {
        if (ImGui::IsWindowAppearing()) {
            RefreshSmokeCapability();
        }
        for (const auto& option : kRunModeOptions) {
            const bool selected = option.mode == run_mode_;
            if (ImGui::Selectable(mode_label(option.mode).c_str(), selected)) {
                run_mode_ = option.mode;
            }
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                mode_tooltip(option.mode);
            }
            if (selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_AllowWhenDisabled)) {
        mode_tooltip(run_mode_);
    }

    ImGui::SameLine();
    const bool smoke_blocked = run_mode_ == StudioDebuggerRunMode::SmokeRun && smoke_unavailable;
    const char* run_reason = running ? "A run is active"
        : !run_debug_callback_ ? "The graph editor is not available"
        : smoke_blocked ? smoke_capability_.reason.c_str()
        : nullptr;
    if (ui::PrimaryButton(running ? ICON_FA_SPINNER " Running##StudioDebuggerRun"
                                  : ICON_FA_PLAY " Run##StudioDebuggerRun",
                          !running && run_debug_callback_ && !smoke_blocked, run_reason,
                          ui::ButtonSize::Small)) {
        StartRun(run_mode_, selected_sample_index_);
    }
    ImGui::SameLine();
    if (ui::DangerButton(stop_requested_ ? ICON_FA_STOP " Stopping##StudioDebuggerStop"
                                         : ICON_FA_STOP " Stop##StudioDebuggerStop",
                         running && !stop_requested_,
                         running ? "Stopping after the current step" : "No run is active")) {
        RequestStop();
    }
    ImGui::SameLine(0.0f, 16.0f);
    RenderSampleStepper();

    // Saved runs and options, right-aligned when they fit.
    const float saved_width = 230.0f;
    const float gear_width = ui::ButtonWidth(ICON_FA_GEAR, ui::ButtonSize::Small);
    RightAlignOnLine(saved_width + gear_width + ImGui::GetStyle().ItemSpacing.x);
    const std::string saved_preview = session_.run_id.empty()
        ? std::string("Saved runs")
        : "Saved: " + CompactRunLabel(session_.run_id);
    ImGui::BeginDisabled(running || load_in_progress_);
    ImGui::SetNextItemWidth(saved_width);
    if (ImGui::BeginCombo("##StudioDebuggerSavedRun", saved_preview.c_str())) {
        if (ImGui::IsWindowAppearing() && !history_loaded_) {
            RequestRunHistoryRefresh();
        }
        if (history_refresh_in_progress_ && session_.run_history.empty()) {
            ImGui::TextColored(DebuggerMuted(), "Loading saved runs...");
        } else if (session_.run_history.empty()) {
            ImGui::TextColored(DebuggerMuted(), "No saved debugger runs.");
        }
        for (const auto& run : session_.run_history) {
            ImGui::PushID(run.run_id.c_str());
            const bool selected = run.run_id == session_.run_id;
            ImGui::PushStyleColor(ImGuiCol_Text, run.success ? DebuggerSuccess() : DebuggerWarning());
            ImGui::TextUnformatted(run.success ? ICON_FA_CIRCLE_CHECK : ICON_FA_CIRCLE_EXCLAMATION);
            ImGui::PopStyleColor();
            ImGui::SameLine();
            const std::string label = CompactRunLabel(run.run_id) +
                (run.run_id == current_run_id_ ? "  (latest)" : "");
            if (ImGui::Selectable(label.c_str(), selected)) {
                LoadStoredRun(run.run_id);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s\n%s\n%s", run.run_id.c_str(), run.timestamp.c_str(),
                                  run.summary.c_str());
            }
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled) && !session_.run_id.empty()) {
        ImGui::SetTooltip("Showing run %s", session_.run_id.c_str());
    }
    ImGui::SameLine();
    if (ui::SecondaryButton(ICON_FA_GEAR "##StudioDebuggerOptions")) {
        open_options_popup = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Debugger options");
    }
    ImGui::PopStyleVar();

    if (open_options_popup) {
        ImGui::OpenPopup("StudioDebuggerOptionsPopup");
    }
    ImGui::SetNextWindowSizeConstraints(ImVec2(380.0f, 0.0f), ImVec2(520.0f, FLT_MAX));
    if (ImGui::BeginPopup("StudioDebuggerOptionsPopup")) {
        CardHeading("Debugger options");
        ImGui::Separator();
        ImGui::TextColored(DebuggerFaint(), "VIEW");
        ImGui::Checkbox("Show trace timeline", &trace_drawer_open_);
        if (ImGui::Checkbox("Wide inspector", &inspector_expanded_)) {
            inspector_width_ = inspector_expanded_ ? 560.0f : 360.0f;
        }
        if (ui::SecondaryButton("Reset pane sizes")) {
            inspector_width_ = 360.0f;
            inspector_expanded_ = false;
            trace_drawer_height_ = 220.0f;
            trace_drawer_open_ = false;
        }
        ImGui::Separator();
        RenderTraceSettings();
        ImGui::Separator();
        ImGui::TextColored(DebuggerFaint(), "SESSION");
        if (ui::DangerButton("Clear current session", !running, "Locked while a run is active")) {
            Clear();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void StudioDebuggerPanel::RenderSessionStatusStrip() {
    const auto line = BuildStudioDebuggerStatusLine(
        session_, has_session_, run_in_progress_, running_step_, running_progress_);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(6.0f, 4.0f));
    DebuggerStatusPill(line.label.c_str(), line.tone);
    const auto same_line_text = [](const ImVec4& color, const std::string& text) {
        ImGui::SameLine();
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 2.0f);
        ImGui::TextColored(color, "%s", text.c_str());
    };
    same_line_text(DebuggerText(), line.summary);

    if (run_in_progress_) {
        ImGui::SameLine();
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 7.0f);
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, DebuggerInfo());
        ImGui::ProgressBar(running_progress_, ImVec2(140.0f, 6.0f), "");
        ImGui::PopStyleColor();
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - run_started_).count();
        same_line_text(DebuggerMuted(), FormatSecondsShort(elapsed));
        const char* locked = "Mode, sample and saved runs are locked while a run is active";
        RightAlignOnLine(ImGui::CalcTextSize(locked).x);
        ImGui::TextColored(DebuggerMuted(), "%s", locked);
    } else if (has_session_) {
        const auto& counts = line.counts;
        same_line_text(DebuggerFaint(), "·");
        same_line_text(DebuggerMuted(), std::to_string(counts.traces) + " traces");
        if (counts.errors > 0) {
            same_line_text(DebuggerFaint(), "·");
            same_line_text(DebuggerDanger(), std::to_string(counts.errors) +
                                                 (counts.errors == 1 ? " error" : " errors"));
        }
        if (counts.warnings > 0) {
            same_line_text(DebuggerFaint(), "·");
            same_line_text(DebuggerWarning(), std::to_string(counts.warnings) +
                                                  (counts.warnings == 1 ? " warning" : " warnings"));
        }
        same_line_text(DebuggerFaint(), "·");
        same_line_text(DebuggerMuted(), std::to_string(counts.fixes) +
                                            (counts.fixes == 1 ? " fix" : " fixes"));

        std::string right;
        if (!line.backend.empty()) {
            right = "Ran on " + line.backend;
            if (!line.provenance.empty()) {
                right += " · " + line.provenance;
            }
        } else {
            right = line.provenance;
        }
        if (!right.empty()) {
            RightAlignOnLine(ImGui::CalcTextSize(right.c_str()).x);
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 2.0f);
            ImGui::TextColored(line.backend.empty() ? DebuggerMuted() : DebuggerText(), "%s",
                               right.c_str());
        }
    }
    if (!run_status_message_.empty()) {
        ImGui::TextColored(DebuggerMuted(), "%s", run_status_message_.c_str());
    }
    ImGui::PopStyleVar();
}

void StudioDebuggerPanel::RenderStepChecklist(const std::vector<StudioDebuggerStep>& steps) {
    for (const auto& step : steps) {
        ImGui::PushID(step.id.c_str());
        ImGui::BeginGroup();
        ImGui::TextColored(StepColor(step.state), "%s", StepIcon(step.state));
        ImGui::SameLine(0.0f, 6.0f);
        const bool dim = step.state == StudioDebuggerStepState::Pending ||
                         step.state == StudioDebuggerStepState::Skipped ||
                         step.state == StudioDebuggerStepState::Unsupported;
        ImGui::TextColored(dim ? DebuggerMuted() : DebuggerText(), "%s", step.name.c_str());
        ImGui::EndGroup();
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::Text("%s: %s", step.name.c_str(), StudioDebuggerStepStateLabel(step.state));
            if (!step.required) {
                ImGui::TextColored(DebuggerMuted(), "Optional for this mode");
            }
            if (step.seconds > 0.0) {
                ImGui::TextColored(DebuggerMuted(), "%s", FormatSecondsShort(step.seconds).c_str());
            }
            if (!step.detail.empty()) {
                ImGui::PushTextWrapPos(ImGui::GetFontSize() * 26.0f);
                ImGui::TextUnformatted(step.detail.c_str());
                ImGui::PopTextWrapPos();
            }
            ImGui::EndTooltip();
        }
        ImGui::PopID();
    }
}

void StudioDebuggerPanel::RenderSectionRail() {
    struct SectionEntry {
        const char* label;
        StudioDebuggerSection section;
    };
    static constexpr SectionEntry kSections[] = {
        {"Overview", StudioDebuggerSection::Overview},
        {"Data", StudioDebuggerSection::Data},
        {"Model", StudioDebuggerSection::Model},
        {"Training", StudioDebuggerSection::Training},
        {"Runtime", StudioDebuggerSection::Runtime},
        {"Diagnostics", StudioDebuggerSection::Diagnostics},
    };
    const float width = ImGui::GetContentRegionAvail().x;
    ImDrawList* draw = ImGui::GetWindowDrawList();
    for (const auto& entry : kSections) {
        const bool active = active_section_ == entry.section;
        const ImVec2 min = ImGui::GetCursorScreenPos();
        const ImVec2 size(width, ImGui::GetTextLineHeight() + 12.0f);
        ImGui::PushID(entry.label);
        if (ImGui::InvisibleButton("##section", size)) {
            SelectSection(entry.section);
        }
        const bool hovered = ImGui::IsItemHovered();
        ImGui::PopID();
        if (active || hovered) {
            draw->AddRectFilled(min, ImVec2(min.x + size.x, min.y + size.y),
                                DebuggerU32(active ? DebuggerAccent() : DebuggerBorder(),
                                            active ? 0.28f : 0.6f),
                                6.0f);
        }
        draw->AddText(ImVec2(min.x + 10.0f, min.y + 6.0f),
                      DebuggerU32(active ? DebuggerBright() : DebuggerText()), entry.label);
        const auto tone = view_section_tone_.find(entry.section);
        if (tone != view_section_tone_.end()) {
            draw->AddCircleFilled(ImVec2(min.x + size.x - 12.0f, min.y + size.y * 0.5f), 3.5f,
                                  DebuggerU32(DebuggerToneColor(tone->second)));
        }
        if (hovered) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        }
    }

    const auto& steps = run_in_progress_ ? running_steps_ : session_.steps;
    if (!steps.empty()) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::TextColored(DebuggerFaint(), run_in_progress_ ? "Steps" : "Steps this run");
        RenderStepChecklist(steps);
    }
}

void StudioDebuggerPanel::RenderRunningBoard() {
    ImGui::BeginChild("StudioDebuggerRunningBoard", ImVec2(0.0f, 0.0f), false);
    if (BeginCard("StudioDebuggerRunningSteps")) {
        CardHeading(StudioDebuggerRunModeLabel(running_mode_));
        ImGui::TextColored(DebuggerMuted(),
                           "Each step stops the run early if it finds a blocking problem.");
        ImGui::Spacing();
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - run_started_).count();
        if (ImGui::BeginTable("StudioDebuggerRunningStepTable", 2,
                              ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings)) {
            ImGui::TableSetupColumn("Step", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthFixed, 60.0f);
            for (const auto& step : running_steps_) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextColored(StepColor(step.state), "%s", StepIcon(step.state));
                ImGui::SameLine(0.0f, 8.0f);
                const bool pending = step.state == StudioDebuggerStepState::Pending;
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextColored(pending ? DebuggerMuted() : DebuggerText(), "%s",
                                   step.name.c_str());
                if (!step.detail.empty()) {
                    ImGui::SameLine(0.0f, 6.0f);
                    ImGui::TextColored(DebuggerFaint(), "- %s", step.detail.c_str());
                }
                ImGui::PopTextWrapPos();
                ImGui::TableSetColumnIndex(1);
                if (step.state == StudioDebuggerStepState::Running) {
                    ImGui::TextColored(DebuggerInfo(), "running");
                } else if (step.seconds > 0.0) {
                    ImGui::TextColored(DebuggerFaint(), "%s", FormatSecondsShort(step.seconds).c_str());
                }
            }
            ImGui::EndTable();
        }
        ImGui::Spacing();
        ImGui::TextColored(DebuggerMuted(), "Elapsed %s", FormatSecondsShort(elapsed).c_str());
    }
    EndCard();
    ImGui::Spacing();

    const float half = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
    const bool side_by_side = half >= 260.0f;
    if (side_by_side) {
        ImGui::BeginChild("StudioDebuggerStopColumn", ImVec2(half, 0.0f),
                          ImGuiChildFlags_AutoResizeY);
    }
    if (BeginCard("StudioDebuggerStopCard")) {
        CardHeading("Stop");
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(DebuggerMuted(),
                           "Stops after the current step. Steps already finished keep their "
                           "results, and the run is saved as Stopped so you can inspect it.");
        ImGui::PopTextWrapPos();
        ImGui::Spacing();
        if (ui::DangerButton(stop_requested_ ? "Stopping...##StudioDebuggerStopCard"
                                             : "Stop run##StudioDebuggerStopCard",
                             !stop_requested_, "Stopping after the current step")) {
            RequestStop();
        }
    }
    EndCard();
    if (side_by_side) {
        ImGui::EndChild();
        ImGui::SameLine();
        ImGui::BeginChild("StudioDebuggerLiveColumn", ImVec2(0.0f, 0.0f),
                          ImGuiChildFlags_AutoResizeY);
    } else {
        ImGui::Spacing();
    }
    if (BeginCard("StudioDebuggerLiveCard")) {
        CardHeading("Live");
        if (ImGui::BeginTable("StudioDebuggerLiveTable", 2,
                              ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings)) {
            ImGui::TableSetupColumn("Key", ImGuiTableColumnFlags_WidthFixed, 110.0f);
            ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
            const auto row = [](const char* key, const std::string& value) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextColored(DebuggerMuted(), "%s", key);
                ImGui::TableSetColumnIndex(1);
                ImGui::TextUnformatted(value.c_str());
            };
            row("Mode", StudioDebuggerRunModeLabel(running_mode_));
            row("Current step", running_step_.empty() ? std::string("Preparing") : running_step_);
            row("Sample", std::to_string(selected_sample_index_));
            char progress[16];
            std::snprintf(progress, sizeof(progress), "%d%%",
                          static_cast<int>(running_progress_ * 100.0f + 0.5f));
            row("Progress", progress);
            ImGui::EndTable();
        }
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(DebuggerFaint(),
                           "Traces appear in the drawer and on the graph when the run finishes.");
        ImGui::PopTextWrapPos();
    }
    EndCard();
    if (side_by_side) {
        ImGui::EndChild();
    }
    ImGui::EndChild();
}

void StudioDebuggerPanel::RenderActiveWorkspace() {
    int& view = section_view_[static_cast<int>(active_section_)];
    const auto selector = [&view](const char* id, const char* const* labels, int count) {
        view = std::clamp(view, 0, count - 1);
        SubViewSelector(id, labels, count, &view);
        if (count > 1) {
            ImGui::Spacing();
        }
    };
    switch (active_section_) {
        case StudioDebuggerSection::Overview: {
            static const char* const kViews[] = {"Summary", "Runs", "Compare"};
            selector("StudioDebuggerOverviewViews", kViews, 3);
            active_lens_ = StudioDebuggerLens::Overview;
            if (view == 0) {
                ImGui::BeginChild("StudioDebuggerOverviewSummary", ImVec2(0.0f, 0.0f), false);
                RenderOverview();
                ImGui::EndChild();
            } else if (view == 1) {
                RenderRunHistory();
            } else {
                RenderRunComparison();
            }
            return;
        }
        case StudioDebuggerSection::Data:
            active_lens_ = StudioDebuggerLens::Preprocessing;
            RenderBatchInspector();
            return;
        case StudioDebuggerSection::Model: {
            static const char* const kViews[] = {"Construction", "Shape comparison", "Graph path"};
            selector("StudioDebuggerModelViews", kViews, 3);
            active_lens_ = StudioDebuggerLens::Shapes;
            if (view == 0) {
                RenderModelConstructionTrace();
            } else if (view == 1) {
                RenderShapeProphecyTrace();
            } else {
                RenderGraphTraceView();
            }
            return;
        }
        case StudioDebuggerSection::Training: {
            static const char* const kViews[] = {"Gradients", "Loss & metrics"};
            selector("StudioDebuggerTrainingViews", kViews, 2);
            if (view == 0) {
                active_lens_ = StudioDebuggerLens::Gradients;
                RenderGradientHealth();
            } else {
                active_lens_ = StudioDebuggerLens::Values;
                RenderLossMetricExplainer();
            }
            return;
        }
        case StudioDebuggerSection::Runtime: {
            static const char* const kViews[] = {"Execution", "Timeline", "Data preparation",
                                                 "Backend", "Tensors", "Memory", "Crash"};
            selector("StudioDebuggerRuntimeViews", kViews, 7);
            active_lens_ = StudioDebuggerLens::Runtime;
            RefreshLiveTrainingTrace();
            switch (view) {
                case 0: RenderTrainingTrace(); break;
                case 1:
                    ImGui::BeginChild("StudioDebuggerRuntimeTimelineView", ImVec2(0.0f, 0.0f), false);
                    RenderRuntimeTimeline(session_.training_trace);
                    ImGui::Spacing();
                    RenderLayerTimingBreakdown(session_.training_trace);
                    ImGui::EndChild();
                    break;
                case 2: RenderMaterializationTrace(session_.training_trace); break;
                case 3: RenderBackendDecisionAudit(); break;
                case 4: RenderTensorLifecycle(); break;
                case 5: RenderMemoryTrace(session_.training_trace); break;
                default: RenderLastRun(); break;
            }
            return;
        }
        case StudioDebuggerSection::Diagnostics:
            active_lens_ = StudioDebuggerLens::StudioEvents;
            RenderStudioEvents();
            return;
    }
}

void StudioDebuggerPanel::RenderInspectorHeader() {
    std::string title = "Nothing selected";
    std::string status;
    int node_id = -1;
    if (selected_trace_index_ >= 0 &&
        selected_trace_index_ < static_cast<int>(session_.traces.size())) {
        const auto& trace = session_.traces[selected_trace_index_];
        title = trace.node_name.empty() ? trace.phase : trace.node_name;
        status = trace.status;
        node_id = trace.node_id;
    } else if (selected_graph_node_ >= 0) {
        node_id = selected_graph_node_;
        title = "Node " + std::to_string(node_id);
        for (const auto& box : view_graph_layout_.nodes) {
            if (box.id == node_id) {
                title = box.name;
                break;
            }
        }
        if (const auto it = view_node_status_.find(node_id); it != view_node_status_.end()) {
            status = it->second.worst_status;
        }
    }
    ImGui::AlignTextToFramePadding();
    ImGui::PushStyleColor(ImGuiCol_Text, DebuggerBright());
    ImGui::TextUnformatted(title.c_str());
    ImGui::PopStyleColor();
    if (!status.empty()) {
        ImGui::SameLine();
        DebuggerStatusPill(TraceStatusLabel(status).c_str(), TraceStatusTone(status));
    }
    if (node_id >= 0 && focus_node_callback_) {
        const char* label = ICON_FA_CROSSHAIRS " Show on canvas";
        RightAlignOnLine(ui::ButtonWidth(label, ui::ButtonSize::Small));
        if (ui::SecondaryButton(label)) {
            focus_node_callback_(node_id);
        }
    }
    ImGui::Separator();
}

void StudioDebuggerPanel::RenderInspectorPane() {
    if (!ImGui::BeginTabBar("StudioDebuggerInspectorTabs", ImGuiTabBarFlags_FittingPolicyScroll)) {
        return;
    }
    if (ImGui::BeginTabItem("Inspector")) {
        RenderInspectorHeader();
        const bool has_trace = selected_trace_index_ >= 0 &&
            selected_trace_index_ < static_cast<int>(session_.traces.size());
        if (!has_trace && selected_graph_node_ >= 0) {
            ImGui::BeginChild("StudioDebuggerNodeFindings", ImVec2(0.0f, 0.0f), false);
            ImGui::PushTextWrapPos(0.0f);
            bool any = false;
            for (const auto& issue : session_.issues) {
                if (issue.node_id != selected_graph_node_) continue;
                any = true;
                const DebuggerTone tone = issue.level == IssueLevel::Error ? DebuggerTone::Danger
                    : issue.level == IssueLevel::Warning ? DebuggerTone::Warning
                                                         : DebuggerTone::Info;
                ImGui::TextColored(DebuggerToneColor(tone), ICON_FA_CIRCLE);
                ImGui::SameLine(0.0f, 6.0f);
                ImGui::TextUnformatted(issue.message.c_str());
                if (!issue.error_code.empty()) {
                    ImGui::TextColored(DebuggerFaint(), "%s", issue.error_code.c_str());
                }
                ImGui::Spacing();
            }
            for (const auto& rec : session_.recommendations) {
                if (rec.node_id != selected_graph_node_) continue;
                any = true;
                ImGui::TextColored(DebuggerAccentText(), "Fix: %s", rec.title.c_str());
                if (!rec.action.empty()) {
                    ImGui::TextColored(DebuggerMuted(), "%s", rec.action.c_str());
                }
                ImGui::Spacing();
            }
            if (!any) {
                ImGui::TextColored(DebuggerMuted(), "No traces or findings for this node in this run.");
            }
            ImGui::PopTextWrapPos();
            ImGui::EndChild();
        } else {
            RenderSelectedTraceDetails();
        }
        ImGui::EndTabItem();
    }
    const std::string issues = "Issues (" + std::to_string(session_.issues.size()) + ")###Issues";
    if (ImGui::BeginTabItem(issues.c_str())) {
        RenderIssueList();
        ImGui::EndTabItem();
    }
    const std::string fixes = "Fixes (" + std::to_string(session_.recommendations.size()) + ")###Fixes";
    if (ImGui::BeginTabItem(fixes.c_str())) {
        RenderRecommendations();
        ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
}

void StudioDebuggerPanel::RenderWorkbenchBody() {
    const ImVec2 available_size = ImGui::GetContentRegionAvail();
    const float rail_width = available_size.x >= 700.0f ? 150.0f : 118.0f;

    ImGui::PushStyleColor(ImGuiCol_ChildBg, DebuggerInputBg());
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 10.0f));
    ImGui::BeginChild("StudioDebuggerSectionRail", ImVec2(rail_width, available_size.y),
                      ImGuiChildFlags_AlwaysUseWindowPadding);
    RenderSectionRail();
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    ImGui::SameLine(0.0f, 0.0f);

    const auto workspace = [this]() {
        if (run_in_progress_) {
            RenderRunningBoard();
        } else {
            RenderActiveWorkspace();
        }
    };

    const float content_width = available_size.x - rail_width;
    const bool wide = content_width >= 780.0f;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 10.0f));
    if (!wide) {
        ImGui::BeginChild("StudioDebuggerCompactBody", ImVec2(0.0f, available_size.y),
                          ImGuiChildFlags_AlwaysUseWindowPadding);
        if (ImGui::BeginTabBar("StudioDebuggerCompactPanes", ImGuiTabBarFlags_FittingPolicyScroll)) {
            if (ImGui::BeginTabItem("Workspace")) {
                workspace();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Inspector")) {
                RenderInspectorPane();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        ImGui::EndChild();
        ImGui::PopStyleVar();
        return;
    }

    constexpr float splitter_width = 6.0f;
    constexpr float minimum_workspace_width = 420.0f;
    constexpr float minimum_inspector_width = 280.0f;
    const float maximum_inspector_width = std::max(
        minimum_inspector_width, content_width - minimum_workspace_width - splitter_width);
    inspector_width_ = std::clamp(inspector_width_, minimum_inspector_width, maximum_inspector_width);
    const float workspace_width = std::max(minimum_workspace_width,
                                           content_width - inspector_width_ - splitter_width);

    ImGui::BeginChild("StudioDebuggerPrimaryWorkspace", ImVec2(workspace_width, available_size.y),
                      ImGuiChildFlags_AlwaysUseWindowPadding);
    workspace();
    ImGui::EndChild();

    ImGui::SameLine(0.0f, 0.0f);
    ImGui::InvisibleButton("##StudioDebuggerInspectorSplitter",
                           ImVec2(splitter_width, available_size.y));
    if (ImGui::IsItemActive()) {
        inspector_width_ -= ImGui::GetIO().MouseDelta.x;
        inspector_width_ = std::clamp(inspector_width_, minimum_inspector_width,
                                      maximum_inspector_width);
        inspector_expanded_ = inspector_width_ > 440.0f;
    }
    if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    }

    ImGui::SameLine(0.0f, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, DebuggerPanelBg());
    ImGui::BeginChild("StudioDebuggerInspectorPane", ImVec2(inspector_width_, available_size.y),
                      ImGuiChildFlags_AlwaysUseWindowPadding);
    RenderInspectorPane();
    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

void StudioDebuggerPanel::RenderTraceDrawer(float height) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 6.0f));
    ImGui::BeginChild("StudioDebuggerTraceDrawer", ImVec2(0.0f, height),
                      ImGuiChildFlags_AlwaysUseWindowPadding,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    const std::string toggle = std::string(trace_drawer_open_ ? ICON_FA_CHEVRON_DOWN
                                                              : ICON_FA_CHEVRON_RIGHT) +
        " Trace timeline##StudioDebuggerTraceDrawerToggle";
    if (ui::LinkButton(toggle.c_str())) {
        trace_drawer_open_ = !trace_drawer_open_;
    }
    ImGui::SameLine();
    ImGui::TextColored(DebuggerFaint(), "%s · %zu records", ActiveLensName(),
                       session_.traces.size());
    if (trace_drawer_open_ && height > 60.0f) {
        ImGui::Spacing();
        RenderTraceTimeline();
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();
}

void StudioDebuggerPanel::RenderLensContent() {
    const float total_height = ImGui::GetContentRegionAvail().y;
    const float collapsed_height = 32.0f;
    const float minimum_body_height = 150.0f;
    float drawer_height = trace_drawer_open_
        ? std::clamp(trace_drawer_height_, 145.0f, std::max(145.0f, total_height * 0.55f))
        : collapsed_height;
    if (total_height - drawer_height < minimum_body_height) {
        drawer_height = std::max(collapsed_height, total_height - minimum_body_height);
    }

    const float splitter_height = trace_drawer_open_ ? 5.0f : 0.0f;
    const float body_height = std::max(80.0f, total_height - drawer_height - splitter_height);
    ImGui::BeginChild("StudioDebuggerWorkbenchBody", ImVec2(0.0f, body_height), false,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    RenderWorkbenchBody();
    ImGui::EndChild();

    if (trace_drawer_open_) {
        ImGui::InvisibleButton("##StudioDebuggerTraceDrawerSplitter",
                               ImVec2(ImGui::GetContentRegionAvail().x, splitter_height));
        if (ImGui::IsItemActive()) {
            trace_drawer_height_ -= ImGui::GetIO().MouseDelta.y;
            trace_drawer_height_ = std::clamp(trace_drawer_height_, 145.0f,
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

    PollRunProgress();
    RebuildViewModelIfNeeded();
    ImGui::SetNextWindowSize(ImVec2(1280.0f, 820.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(640.0f, 420.0f), ImVec2(FLT_MAX, FLT_MAX));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 6.0f));
    const std::string title = std::string(ICON_FA_BUG) + " Studio Debugger###StudioDebuggerPanel";
    if (ImGui::Begin(title.c_str(), &visible_)) {
        RenderToolbar();
        RenderSessionStatusStrip();
        ImGui::Separator();
        RenderLensContent();
    }
    ImGui::End();
    ImGui::PopStyleVar();
}

} // namespace cyxwiz
