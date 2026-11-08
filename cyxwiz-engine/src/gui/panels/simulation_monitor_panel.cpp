#include "simulation_monitor_panel.h"

#include "../icons.h"
#include "../node_editor.h"
#include "../separate_windows.h"
#include "../simulation_scope_plot.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../../core/engine_config.h"
#include "../../core/simulation_commands.h"
#include <imgui.h>
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

ImVec4 StateColour(const ui::Tokens& t, SimulationState state) {
    switch (state) {
        case SimulationState::Running: return t.running;
        case SimulationState::Paused: return t.pending;
        case SimulationState::Failed: return t.warning;
        case SimulationState::Stopped: break;
    }
    return t.text_dim;
}

// A card on the window: what is missing and what to do.
void Note(const ui::Tokens& t, const char* id, const char* title, const char* body) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_panel);
    if (ImGui::BeginChild(id, ImVec2(0, 0), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding)) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_bright, "%s", title);
        ImGui::TextColored(t.text_dim, "%s", body);
        ImGui::PopTextWrapPos();
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

}  // namespace

SimulationMonitorPanel::SimulationMonitorPanel() : Panel("Simulation Monitor", false) {}

const char* SimulationMonitorPanel::GetIcon() const { return ICON_FA_WAVE_SQUARE; }

void SimulationMonitorPanel::ShowAndFocus() {
    visible_ = true;
    focus_requested_ = true;
}

void SimulationMonitorPanel::Render() {
    if (!visible_) return;
    if (focus_requested_) {
        ImGui::SetNextWindowFocus();
        focus_requested_ = false;
    }
    ::gui::NextWindowMayLeave(GetName());
    const bool expanded = ImGui::Begin(GetName(), &visible_);
    ::gui::TabMenu(GetName(), &visible_);
    if (!expanded || !node_editor_) {
        focused_ = false;
        ImGui::End();
        return;
    }
    focused_ = ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);
    RenderBody();
    ImGui::End();
}

void SimulationMonitorPanel::RenderBody() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!node_editor_->HasSimulationNodes()) {
        Note(t, "##nothing", "The open graph has nothing to simulate.",
             "A graph run needs signal nodes (Sine Wave, Signal Slider, Signal Scope) or a plugin's simulation "
             "nodes. Open or build one, then press Run Sim. A plugin's own window runs its model on its own.");
        return;
    }

    using simulation::Command;
    auto& session = node_editor_->GetSimulationSession();
    const SimulationState state = session.GetState();
    const auto commands = node_editor_->GetSimulationCommands();

    // State, time, rate and run; then the commands, at the right while they fit
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(StateColour(t, state), ICON_FA_CIRCLE " %s", simulation::StateLabel(state));
    ImGui::SameLine(0.0f, 16.0f);
    ImGui::Text("t = %.2f s", session.GetSimTime());
    ImGui::SameLine(0.0f, 16.0f);
    const float run_rate = session.GetRateHz();
    const int rate = session.IsActive() && run_rate > 0.0f ? static_cast<int>(run_rate + 0.5f)
                                                           : core::EngineConfig::Instance().SimulationRateHz();
    ImGui::TextColored(t.text_dim, "%d Hz", rate);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Set the rate in Preferences > Simulation");
    if (session.GetRunId() > 0) {
        ImGui::SameLine(0.0f, 16.0f);
        ImGui::TextColored(t.text_dim, "Run %llu", static_cast<unsigned long long>(session.GetRunId()));
    }

    const bool running = state == SimulationState::Running;
    const auto& primary = running ? commands.pause : commands.run;
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float buttons = ui::ButtonWidth(primary.short_label.c_str(), ui::ButtonSize::Regular) +
                          ui::ButtonWidth(commands.step.short_label.c_str(), ui::ButtonSize::Regular) +
                          ui::ButtonWidth(commands.stop.short_label.c_str(), ui::ButtonSize::Regular) +
                          ui::ButtonWidth(commands.clear.short_label.c_str(), ui::ButtonSize::Regular) + 3.0f * spacing;
    ImGui::SameLine();
    const float room = ImGui::GetContentRegionAvail().x;
    if (room >= buttons) ImGui::SameLine(ImGui::GetCursorPosX() + room - buttons);
    else ImGui::NewLine();
    const auto button = [&](const simulation::CommandView& view, bool is_primary) {
        const char* reason = view.disabled_reason.c_str();
        const bool clicked = is_primary
            ? ui::PrimaryButton(view.short_label.c_str(), view.enabled, reason, ui::ButtonSize::Regular)
            : ui::SecondaryButton(view.short_label.c_str(), view.enabled, reason, ui::ButtonSize::Regular);
        if (view.enabled && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", view.hint.c_str());
        if (clicked) node_editor_->RunSimulationCommand(view.command);
    };
    button(primary, true);
    ImGui::SameLine();
    button(commands.step, false);
    ImGui::SameLine();
    button(commands.stop, false);
    ImGui::SameLine();
    button(commands.clear, false);
    ImGui::Spacing();

    if (state == SimulationState::Failed) {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_panel);
        if (ImGui::BeginChild("##failed", ImVec2(0, 0),
                              ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding)) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.warning, session.FailedBeforeRunning()
                                              ? ICON_FA_TRIANGLE_EXCLAMATION " The graph run did not start."
                                              : ICON_FA_TRIANGLE_EXCLAMATION " The graph run failed.");
            ImGui::SameLine();
            ImGui::TextUnformatted(session.GetError().c_str());
            ImGui::PopTextWrapPos();
        }
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::Spacing();
    }

    RenderScopes();
}

void SimulationMonitorPanel::RenderScopes() {
    const ui::Tokens& t = ui::CurrentTokens();
    const auto scopes = node_editor_->GetSignalScopes();
    if (scopes.empty()) {
        Note(t, "##no_scope", "This graph has no Signal Scope, so the run has nothing to plot here.",
             "Add a Signal Scope (Signal > Signal Scope) and connect a number output to it. A plugin's own "
             "window, such as its viewport, still shows its model.");
        return;
    }

    auto& session = node_editor_->GetSimulationSession();
    // Two cards a row while they have room, one below another when narrow
    const int columns = ImGui::GetContentRegionAvail().x >= 640.0f && scopes.size() > 1 ? 2 : 1;
    const int rows = static_cast<int>((scopes.size() + columns - 1) / columns);
    const float line = ImGui::GetTextLineHeightWithSpacing();
    const float row_height = std::max(
        150.0f, (ImGui::GetContentRegionAvail().y - ImGui::GetStyle().ItemSpacing.y * (rows - 1)) / rows);
    const float plot_height = row_height - 2.0f * line - 2.0f * ImGui::GetStyle().WindowPadding.y -
                              ImGui::GetStyle().ItemSpacing.y * 2.0f;

    if (!ImGui::BeginTable("##scopes", columns, ImGuiTableFlags_SizingStretchSame)) return;
    std::vector<float> times;
    std::vector<float> values;
    for (std::size_t i = 0; i < scopes.size(); ++i) {
        const auto& scope = *scopes[i];
        ImGui::TableNextColumn();
        ImGui::PushID(scope.id);
        const ImVec4 colour = t.series[i % ui::Tokens::kSeriesCount];
        times.clear();
        values.clear();
        session.CopyScope(scope.id, times, values);
        // Before its first run a Scope has no history: its own window then
        std::size_t capacity = session.GetScopeCapacity(scope.id);
        if (capacity == 0) capacity = GraphSimulationSession::ScopeWindow(scope);

        ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_panel);
        if (ImGui::BeginChild("##card", ImVec2(0, row_height), ImGuiChildFlags_AlwaysUseWindowPadding)) {
            ImGui::TextColored(colour, ICON_FA_CIRCLE);
            ImGui::SameLine();
            ImGui::TextUnformatted(scope.name.c_str());
            ImGui::SameLine();
            if (ui::InlineLink("Show in graph")) node_editor_->FocusNode(scope.id);
            if (!values.empty()) {
                char latest[32];
                std::snprintf(latest, sizeof(latest), "%.3f", values.back());
                ImGui::SameLine(ImGui::GetContentRegionMax().x - ImGui::CalcTextSize(latest).x);
                ImGui::TextUnformatted(latest);
            }
            if (times.empty()) {
                ImGui::Dummy(ImVec2(0.0f, std::max(0.0f, plot_height * 0.5f - line)));
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextColored(t.text_dim, "No samples yet. Run the graph to plot what reaches this Scope.");
                ImGui::PopTextWrapPos();
                ImGui::Dummy(ImVec2(0.0f, std::max(0.0f, plot_height * 0.5f - line)));
            } else {
                const bool auto_scale = [&]() {
                    const auto it = scope.parameters.find("auto_scale");
                    return it != scope.parameters.end() && it->second == "true";
                }();
                ::gui::DrawScopeTrace("##trace", scope.name.c_str(), times, values,
                                      ImVec2(-1.0f, std::max(60.0f, plot_height)), auto_scale, colour);
            }
            ImGui::TextColored(t.text_dim, "%zu of %zu samples", times.size(), capacity);
            if (times.size() > 1) {
                ImGui::SameLine(0.0f, 16.0f);
                ImGui::TextColored(t.text_dim, "Last %.1f s", times.back() - times.front());
            }
        }
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::PopID();
    }
    ImGui::EndTable();
}

}  // namespace cyxwiz
