#include "studio_debugger_panel.h"
#include <algorithm>

namespace cyxwiz {

namespace {

ImVec4 LevelColor(IssueLevel level) {
    switch (level) {
        case IssueLevel::Error:   return ImVec4(1.0f, 0.4f, 0.4f, 1.0f);
        case IssueLevel::Warning: return ImVec4(1.0f, 0.85f, 0.3f, 1.0f);
        case IssueLevel::Info:    return ImVec4(0.45f, 0.7f, 1.0f, 1.0f);
    }
    return ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
}

} // namespace

StudioDebuggerPanel::StudioDebuggerPanel()
    : Panel("Studio Debugger", false) {}

void StudioDebuggerPanel::SetSession(const StudioDebuggerSnapshot& session) {
    session_ = session;
    has_session_ = true;
    selected_trace_index_ = session_.debug_result.layer_traces.empty() ? -1 : 0;
}

void StudioDebuggerPanel::Clear() {
    session_ = StudioDebuggerSnapshot{};
    has_session_ = false;
    selected_trace_index_ = -1;
}

std::string StudioDebuggerPanel::FormatShape(const std::vector<size_t>& shape) {
    if (shape.empty()) return "[]";
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) out += ", ";
        out += std::to_string(shape[i]);
    }
    out += "]";
    return out;
}

void StudioDebuggerPanel::RenderToolbar() {
    if (ImGui::Button("Run Debug")) {
        if (run_debug_callback_) {
            SetSession(run_debug_callback_());
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear")) {
        Clear();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("POC: current DebugExecutor backend, synthetic sample only");
}

void StudioDebuggerPanel::RenderOverview() {
    if (!has_session_) {
        ImGui::TextDisabled("Run a debug session to capture a trace.");
        return;
    }

    ImGui::Text("Graph hash: 0x%016llx", static_cast<unsigned long long>(session_.graph_hash));
    ImGui::Text("Nodes: %zu  Links: %zu", session_.node_count, session_.link_count);
    ImGui::Text("Sample: %s", session_.sample_summary.c_str());
    ImGui::Text("Status: %s", session_.success ? "Success" : "Failed");
    if (!session_.failure_summary.empty()) {
        ImGui::TextWrapped("Failure: %s", session_.failure_summary.c_str());
    }

    if (!session_.graph_summary.empty()) {
        ImGui::Spacing();
        ImGui::BeginChild("StudioDebuggerSummary", ImVec2(0, 120), true);
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextWrapped("%s", session_.graph_summary.c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndChild();
    }
}

void StudioDebuggerPanel::RenderTraceTimeline() {
    const auto& traces = session_.debug_result.layer_traces;
    ImGui::Text("Trace timeline");
    ImGui::BeginChild("StudioDebuggerTraceTimeline", ImVec2(0, 220), true);

    if (traces.empty()) {
        ImGui::TextDisabled("No layer traces captured.");
        ImGui::EndChild();
        return;
    }

    for (int i = 0; i < static_cast<int>(traces.size()); ++i) {
        const auto& trace = traces[i];
        bool selected = (selected_trace_index_ == i);

        std::string label = trace.name + "  " + FormatShape(trace.actual_shape);
        if (trace.node_id >= 0) {
            label += "  [node " + std::to_string(trace.node_id) + "]";
        }

        ImVec4 row_color = ImVec4(0.85f, 0.85f, 0.85f, 1.0f);
        if (trace.has_nan || trace.has_inf) {
            row_color = ImVec4(1.0f, 0.45f, 0.45f, 1.0f);
        } else if (!trace.shape_matches && !trace.predicted_shape.empty()) {
            row_color = ImVec4(1.0f, 0.82f, 0.35f, 1.0f);
        } else if (trace.shape_matches) {
            row_color = ImVec4(0.45f, 0.95f, 0.55f, 1.0f);
        }

        ImGui::PushStyleColor(ImGuiCol_Text, row_color);
        if (ImGui::Selectable(label.c_str(), selected)) {
            selected_trace_index_ = i;
            if (focus_node_callback_ && trace.node_id >= 0) {
                focus_node_callback_(trace.node_id);
            }
        }
        ImGui::PopStyleColor();

        ImGui::SameLine();
        ImGui::TextDisabled("%.2f ms", trace.forward_ms);
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::Text("%s", trace.name.c_str());
            ImGui::Text("Predicted: %s", FormatShape(trace.predicted_shape).c_str());
            ImGui::Text("Actual: %s", FormatShape(trace.actual_shape).c_str());
            ImGui::Text("Shape match: %s", trace.shape_matches ? "yes" : "no");
            ImGui::EndTooltip();
        }
    }

    ImGui::EndChild();
}

void StudioDebuggerPanel::RenderSelectedTraceDetails() {
    const auto& traces = session_.debug_result.layer_traces;
    ImGui::Text("Selected trace");
    ImGui::BeginChild("StudioDebuggerTraceDetails", ImVec2(0, 160), true);

    if (traces.empty() || selected_trace_index_ < 0 ||
        selected_trace_index_ >= static_cast<int>(traces.size())) {
        ImGui::TextDisabled("Select a trace row to inspect it.");
        ImGui::EndChild();
        return;
    }

    const auto& trace = traces[selected_trace_index_];
    ImGui::Text("Name: %s", trace.name.c_str());
    ImGui::Text("Node id: %d", trace.node_id);
    ImGui::Text("Type: %d", static_cast<int>(trace.type));
    ImGui::Text("Predicted: %s", FormatShape(trace.predicted_shape).c_str());
    ImGui::Text("Actual: %s", FormatShape(trace.actual_shape).c_str());
    ImGui::Text("Duration: %.2f ms", trace.forward_ms);

    if (trace.shape_matches) {
        ImGui::TextColored(ImVec4(0.45f, 0.95f, 0.55f, 1.0f), "Shape match");
    } else if (!trace.predicted_shape.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.82f, 0.35f, 1.0f), "Shape mismatch");
    }

    if (trace.has_nan) {
        ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f), "NaN detected");
    }
    if (trace.has_inf) {
        ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f), "Inf detected");
    }

    if (trace.node_id >= 0 && focus_node_callback_) {
        if (ImGui::Button("Focus Node")) {
            focus_node_callback_(trace.node_id);
        }
    }

    ImGui::EndChild();
}

void StudioDebuggerPanel::RenderIssueList() {
    if (session_.issues.empty()) {
        ImGui::TextDisabled("No issues reported.");
        return;
    }

    ImGui::Text("Issues");
    ImGui::BeginChild("StudioDebuggerIssues", ImVec2(0, 160), true);
    for (const auto& issue : session_.issues) {
        ImGui::PushStyleColor(ImGuiCol_Text, LevelColor(issue.level));
        ImGui::TextUnformatted(issue.node_name.empty() ? "[issue]" : issue.node_name.c_str());
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::TextWrapped("%s", issue.message.c_str());
    }
    ImGui::EndChild();
}

void StudioDebuggerPanel::Render() {
    if (!visible_) {
        return;
    }

    std::string title = std::string(ICON_FA_BUG) + " Studio Debugger###StudioDebuggerPanel";
    if (ImGui::Begin(title.c_str(), &visible_)) {
        RenderToolbar();
        ImGui::Separator();
        RenderOverview();
        ImGui::Separator();
        RenderTraceTimeline();
        ImGui::Separator();
        RenderSelectedTraceDetails();
        ImGui::Separator();
        RenderIssueList();
    }
    ImGui::End();
}

} // namespace cyxwiz
