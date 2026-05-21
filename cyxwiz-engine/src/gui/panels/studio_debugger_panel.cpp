#include "studio_debugger_panel.h"
#include <algorithm>
#include <array>
#include <map>

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

ImVec4 RecommendationColor(DebugRecommendationSeverity severity) {
    switch (severity) {
        case DebugRecommendationSeverity::Critical:
            return ImVec4(1.0f, 0.45f, 0.45f, 1.0f);
        case DebugRecommendationSeverity::Warning:
            return ImVec4(1.0f, 0.82f, 0.35f, 1.0f);
        case DebugRecommendationSeverity::Info:
            return ImVec4(0.45f, 0.7f, 1.0f, 1.0f);
    }
    return ImVec4(0.85f, 0.85f, 0.85f, 1.0f);
}

ImVec4 TraceStatusColor(const std::string& status) {
    if (status == "ok" || status == "passed" || status == "ready" || status == "captured") {
        return ImVec4(0.45f, 0.95f, 0.55f, 1.0f);
    }
    if (status == "warning" || status == "zero" || status == "shape_mismatch" ||
        status == "blocked") {
        return ImVec4(1.0f, 0.82f, 0.35f, 1.0f);
    }
    if (status == "failed" || status == "nan") {
        return ImVec4(1.0f, 0.45f, 0.45f, 1.0f);
    }
    return ImVec4(0.65f, 0.7f, 0.78f, 1.0f);
}

int TraceStatusSeverity(const std::string& status) {
    if (status == "failed" || status == "nan") {
        return 4;
    }
    if (status == "shape_mismatch" || status == "blocked") {
        return 3;
    }
    if (status == "warning" || status == "zero") {
        return 2;
    }
    if (status == "ok" || status == "passed" || status == "ready" || status == "captured") {
        return 1;
    }
    return 0;
}

int CountTraceStatus(const StudioDebuggerSnapshot& session, const std::string& status) {
    int count = 0;
    for (const auto& trace : session.traces) {
        if (trace.status == status) {
            ++count;
        }
    }
    return count;
}

int CountRecommendationSeverity(const StudioDebuggerSnapshot& session,
                                DebugRecommendationSeverity severity) {
    int count = 0;
    for (const auto& rec : session.recommendations) {
        if (rec.severity == severity) {
            ++count;
        }
    }
    return count;
}

bool IsPreprocessingTrace(const DebugTraceRecord& trace) {
    return trace.role == DebugTraceRole::RawInput ||
        trace.role == DebugTraceRole::PreprocessingOutput ||
        trace.role == DebugTraceRole::FeatureTensor ||
        trace.phase.find("Text") != std::string::npos ||
        trace.phase.find("Preprocessing") != std::string::npos;
}

bool IsGradientTrace(const DebugTraceRecord& trace) {
    return trace.role == DebugTraceRole::Gradient ||
        trace.phase.find("Backward") != std::string::npos ||
        trace.status == "zero" ||
        trace.status == "nan";
}

bool IsShapeTrace(const DebugTraceRecord& trace) {
    return !trace.input_shape.empty() ||
        !trace.output_shape.empty() ||
        trace.status == "shape_mismatch" ||
        trace.payload.contains("predicted_shape") ||
        trace.payload.contains("actual_shape");
}

bool IsValueTrace(const DebugTraceRecord& trace) {
    return trace.role == DebugTraceRole::Activation ||
        trace.role == DebugTraceRole::Prediction ||
        trace.role == DebugTraceRole::Target ||
        trace.role == DebugTraceRole::Loss ||
        trace.payload.contains("loss") ||
        trace.payload.contains("average_loss") ||
        trace.payload.contains("token_ids_preview");
}

bool IsRuntimeTrace(const DebugTraceRecord& trace) {
    return trace.duration_ms > 0.0f ||
        trace.phase.find("SmokeRun") != std::string::npos ||
        trace.phase.find("Train") != std::string::npos ||
        trace.status == "failed" ||
        trace.status == "warning";
}

bool JsonHas(const nlohmann::json& payload, const char* key) {
    return payload.contains(key) && !payload.at(key).is_null();
}

std::string JsonString(const nlohmann::json& payload, const char* key,
                       const std::string& fallback = "") {
    if (!JsonHas(payload, key)) {
        return fallback;
    }
    const auto& value = payload.at(key);
    if (value.is_string()) {
        return value.get<std::string>();
    }
    return value.dump();
}

double JsonNumber(const nlohmann::json& payload, const char* key, double fallback = 0.0) {
    if (!JsonHas(payload, key) || !payload.at(key).is_number()) {
        return fallback;
    }
    return payload.at(key).get<double>();
}

bool JsonBool(const nlohmann::json& payload, const char* key, bool fallback = false) {
    if (!JsonHas(payload, key) || !payload.at(key).is_boolean()) {
        return fallback;
    }
    return payload.at(key).get<bool>();
}

std::string JsonArrayPreview(const nlohmann::json& payload, const char* key, size_t limit = 32) {
    if (!JsonHas(payload, key) || !payload.at(key).is_array()) {
        return "";
    }
    const auto& values = payload.at(key);
    std::string out;
    const size_t n = std::min(values.size(), limit);
    for (size_t i = 0; i < n; ++i) {
        if (i) {
            out += ", ";
        }
        if (values[i].is_string()) {
            out += values[i].get<std::string>();
        } else {
            out += values[i].dump();
        }
    }
    if (values.size() > limit) {
        out += ", ...";
    }
    return out;
}

} // namespace

StudioDebuggerPanel::StudioDebuggerPanel()
    : Panel("Studio Debugger", false) {}

void StudioDebuggerPanel::SetSession(const StudioDebuggerSnapshot& session) {
    session_ = session;
    current_session_ = session;
    has_current_session_ = true;
    current_run_id_ = session_.run_id;
    if (auto last_run = CrashRunRecorder::LoadLastRun()) {
        session_.last_run = *last_run;
    }
    if (auto training_trace = TrainingTraceCollector::LoadLastTrace()) {
        session_.training_trace = *training_trace;
        current_session_.training_trace = *training_trace;
    }
    session_.run_history = DebugRunStore::ListRecent(8);
    current_session_.run_history = session_.run_history;
    has_session_ = true;
    selected_trace_index_ = session_.debug_result.layer_traces.empty() ? -1 : 0;
}

void StudioDebuggerPanel::Clear() {
    if (run_in_progress_) {
        return;
    }
    session_ = StudioDebuggerSnapshot{};
    current_session_ = StudioDebuggerSnapshot{};
    has_session_ = false;
    has_current_session_ = false;
    current_run_id_.clear();
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
    const std::array<const char*, 5> run_modes = {
        "Full Workflow", "Preflight", "Local Debug", "Smoke Run", "Runtime Trace"
    };
    int run_mode_index = static_cast<int>(run_mode_);
    ImGui::SetNextItemWidth(145.0f);
    if (ImGui::Combo("Mode", &run_mode_index, run_modes.data(), static_cast<int>(run_modes.size()))) {
        run_mode_ = static_cast<StudioDebuggerRunMode>(run_mode_index);
    }
    ImGui::SameLine();

    if (run_in_progress_) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button(run_in_progress_ ? "Running..." : "Run")) {
        if (run_debug_callback_) {
            const StudioDebuggerRunMode mode = run_mode_;
            const int sample_index = selected_sample_index_;
            auto task = std::make_shared<std::function<StudioDebuggerSnapshot()>>(
                run_debug_callback_(mode, sample_index));
            auto state = std::make_shared<AsyncRunState>();
            pending_run_state_ = state;
            run_status_message_ = "Studio Debugger run is executing in the background.";
            run_in_progress_ = true;
            pending_task_id_ = AsyncTaskManager::Instance().RunAsync(
                "Studio Debugger Run",
                [task, state](LambdaTask& async_task) {
                    async_task.ReportProgress(0.05f, "Preparing Studio Debugger run");
                    if (async_task.ShouldStop()) {
                        return;
                    }
                    StudioDebuggerSnapshot result = (*task)();
                    {
                        std::lock_guard<std::mutex> lock(state->mutex);
                        state->result = std::move(result);
                    }
                    async_task.ReportProgress(1.0f, "Studio Debugger run completed");
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
                        run_status_message_ = "Studio Debugger task completed without a result.";
                    }
                });
        }
    }
    if (run_in_progress_) {
        ImGui::EndDisabled();
    }
    ImGui::SameLine();
    if (run_in_progress_) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button("Clear")) {
        Clear();
    }
    if (run_in_progress_) {
        ImGui::EndDisabled();
    }
    ImGui::SameLine();
    const std::array<const char*, 8> lenses = {
        "Overview", "Preprocessing", "Shapes", "Values",
        "Gradients", "Runtime", "Studio Events", "Recommendations"
    };
    int lens_index = static_cast<int>(active_lens_);
    ImGui::SetNextItemWidth(170.0f);
    if (ImGui::Combo("Lens", &lens_index, lenses.data(), static_cast<int>(lenses.size()))) {
        active_lens_ = static_cast<StudioDebuggerLens>(lens_index);
        selected_trace_index_ = -1;
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(90.0f);
    if (ImGui::InputInt("Sample", &selected_sample_index_)) {
        if (selected_sample_index_ < 0) {
            selected_sample_index_ = 0;
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Sample index used by preprocessing inspection. Smoke Run still runs a small subset.");
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Preflight + Local Debug + Smoke Run + Training Trace");
    if (run_in_progress_) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.45f, 0.7f, 1.0f, 1.0f),
                           "running in Task View #%llu",
                           static_cast<unsigned long long>(pending_task_id_));
    } else if (!run_status_message_.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled("%s", run_status_message_.c_str());
    }
}

const char* StudioDebuggerPanel::ActiveLensName() const {
    switch (active_lens_) {
        case StudioDebuggerLens::Overview: return "Overview";
        case StudioDebuggerLens::Preprocessing: return "Preprocessing";
        case StudioDebuggerLens::Shapes: return "Shapes";
        case StudioDebuggerLens::Values: return "Values";
        case StudioDebuggerLens::Gradients: return "Gradients";
        case StudioDebuggerLens::Runtime: return "Runtime";
        case StudioDebuggerLens::StudioEvents: return "Studio Events";
        case StudioDebuggerLens::Recommendations: return "Recommendations";
    }
    return "Overview";
}

bool StudioDebuggerPanel::TraceMatchesActiveLens(const DebugTraceRecord& trace) const {
    switch (active_lens_) {
        case StudioDebuggerLens::Overview:
            return true;
        case StudioDebuggerLens::Preprocessing:
            return IsPreprocessingTrace(trace);
        case StudioDebuggerLens::Shapes:
            return IsShapeTrace(trace);
        case StudioDebuggerLens::Values:
            return IsValueTrace(trace);
        case StudioDebuggerLens::Gradients:
            return IsGradientTrace(trace);
        case StudioDebuggerLens::Runtime:
            return IsRuntimeTrace(trace);
        case StudioDebuggerLens::StudioEvents:
            return trace.role == DebugTraceRole::StudioEvent;
        case StudioDebuggerLens::Recommendations:
            return trace.status == "warning" ||
                trace.status == "failed" ||
                trace.status == "shape_mismatch" ||
                trace.status == "zero" ||
                trace.status == "nan" ||
                trace.role == DebugTraceRole::Warning ||
                trace.role == DebugTraceRole::Error;
    }
    return true;
}

void StudioDebuggerPanel::RenderTraceSettings() {
    if (!trace_settings_initialized_) {
        const auto settings = TrainingTraceCollector::Instance().GetSettings();
        trace_persist_enabled_ = settings.persist_enabled;
        trace_persist_every_n_events_ = settings.persist_every_n_events;
        trace_max_recent_events_ = static_cast<int>(settings.max_recent_events);
        trace_settings_initialized_ = true;
    }

    if (!ImGui::CollapsingHeader("Trace Persistence Settings",
                                 ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    bool changed = false;
    changed |= ImGui::Checkbox("Persist training trace to disk", &trace_persist_enabled_);
    changed |= ImGui::InputInt("Write every N events", &trace_persist_every_n_events_);
    changed |= ImGui::InputInt("Keep recent events", &trace_max_recent_events_);

    if (trace_persist_every_n_events_ < 1) {
        trace_persist_every_n_events_ = 1;
        changed = true;
    }
    if (trace_max_recent_events_ < 20) {
        trace_max_recent_events_ = 20;
        changed = true;
    }
    if (trace_max_recent_events_ > 5000) {
        trace_max_recent_events_ = 5000;
        changed = true;
    }

    if (changed) {
        TrainingTraceSettings settings;
        settings.persist_enabled = trace_persist_enabled_;
        settings.persist_every_n_events = trace_persist_every_n_events_;
        settings.max_recent_events = static_cast<size_t>(trace_max_recent_events_);
        TrainingTraceCollector::Instance().Configure(settings);
    }

    ImGui::TextDisabled("Lower N gives better crash evidence; higher N reduces disk writes.");
}

void StudioDebuggerPanel::LoadStoredRun(const std::string& run_id) {
    auto record = DebugRunStore::Load(run_id);
    if (!record) {
        return;
    }

    const auto history = DebugRunStore::ListRecent(8);
    session_ = StudioDebuggerSnapshot{};
    session_.run_id = record->summary.run_id;
    session_.graph_hash = record->summary.graph_hash;
    session_.success = record->summary.success;
    session_.failure_summary = record->summary.success ? "" : record->summary.summary;
    session_.sample_summary = "Saved Studio Debugger run";
    session_.issues = std::move(record->issues);
    session_.traces = std::move(record->traces);
    session_.studio_events = std::move(record->studio_events);
    session_.recommendations = std::move(record->recommendations);
    session_.run_history = history;
    has_session_ = true;
    selected_trace_index_ = session_.traces.empty() ? -1 : 0;
}

void StudioDebuggerPanel::RenderRunComparison() {
    if (!has_session_) {
        return;
    }

    if (!has_current_session_ && !current_run_id_.empty()) {
        if (auto record = DebugRunStore::Load(current_run_id_)) {
            current_session_ = StudioDebuggerSnapshot{};
            current_session_.run_id = record->summary.run_id;
            current_session_.graph_hash = record->summary.graph_hash;
            current_session_.success = record->summary.success;
            current_session_.failure_summary = record->summary.success ? "" : record->summary.summary;
            current_session_.issues = std::move(record->issues);
            current_session_.traces = std::move(record->traces);
            current_session_.studio_events = std::move(record->studio_events);
            current_session_.recommendations = std::move(record->recommendations);
            has_current_session_ = true;
        }
    }

    ImGui::Text("Run Comparison");
    ImGui::BeginChild("StudioDebuggerRunComparison", ImVec2(0, 155), true);

    if (!has_current_session_ || current_run_id_.empty()) {
        ImGui::TextDisabled("Run a new debug session to establish a comparison baseline.");
        ImGui::EndChild();
        return;
    }

    if (session_.run_id == current_session_.run_id) {
        ImGui::TextDisabled("Viewing the current run. Select an older run to compare.");
        ImGui::EndChild();
        return;
    }

    ImGui::Text("Selected: %s", session_.run_id.c_str());
    ImGui::Text("Current:  %s", current_session_.run_id.c_str());
    ImGui::Separator();

    const int selected_errors = CountTraceStatus(session_, "failed") +
        CountTraceStatus(session_, "nan");
    const int current_errors = CountTraceStatus(current_session_, "failed") +
        CountTraceStatus(current_session_, "nan");
    const int selected_shape = CountTraceStatus(session_, "shape_mismatch");
    const int current_shape = CountTraceStatus(current_session_, "shape_mismatch");
    const int selected_warnings = CountTraceStatus(session_, "warning") +
        static_cast<int>(session_.issues.size());
    const int current_warnings = CountTraceStatus(current_session_, "warning") +
        static_cast<int>(current_session_.issues.size());
    const int selected_critical = CountRecommendationSeverity(
        session_, DebugRecommendationSeverity::Critical);
    const int current_critical = CountRecommendationSeverity(
        current_session_, DebugRecommendationSeverity::Critical);

    auto render_delta = [](const char* label, int selected, int current) {
        const int delta = current - selected;
        ImVec4 color = ImVec4(0.85f, 0.85f, 0.85f, 1.0f);
        if (delta > 0) {
            color = ImVec4(1.0f, 0.82f, 0.35f, 1.0f);
        } else if (delta < 0) {
            color = ImVec4(0.45f, 0.95f, 0.55f, 1.0f);
        }
        ImGui::Text("%s", label);
        ImGui::SameLine(170.0f);
        ImGui::Text("selected=%d current=%d", selected, current);
        ImGui::SameLine();
        ImGui::TextColored(color, "delta=%+d", delta);
    };

    render_delta("Error traces", selected_errors, current_errors);
    render_delta("Shape mismatches", selected_shape, current_shape);
    render_delta("Warnings/issues", selected_warnings, current_warnings);
    render_delta("Critical recommendations", selected_critical, current_critical);
    render_delta("Total traces",
                 static_cast<int>(session_.traces.size()),
                 static_cast<int>(current_session_.traces.size()));

    ImGui::EndChild();
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
    if (!session_.preflight.summary.empty()) {
        ImGui::Text("Preflight: %s", session_.preflight.ready ? "Ready" : "Blocked");
    }
    if (!session_.failure_summary.empty()) {
        ImGui::TextWrapped("Failure: %s", session_.failure_summary.c_str());
    }
    if (session_.smoke_result.supported) {
        ImVec4 smoke_color = session_.smoke_result.success
            ? ImVec4(0.45f, 0.95f, 0.55f, 1.0f)
            : ImVec4(1.0f, 0.82f, 0.35f, 1.0f);
        ImGui::TextColored(smoke_color, "Smoke Run: %s",
                           session_.smoke_result.success ? "Passed" : "Needs attention");
        ImGui::Text("Smoke samples: %d  batches: %d  avg loss: %.4f  last acc: %.2f%%",
                    session_.smoke_result.samples_seen,
                    session_.smoke_result.batches_seen,
                    session_.smoke_result.average_loss,
                    session_.smoke_result.last_accuracy * 100.0f);
    }

    if (!session_.preflight.summary.empty()) {
        ImGui::Spacing();
        ImGui::Text("Preflight summary");
        ImGui::BeginChild("StudioDebuggerPreflightSummary", ImVec2(0, 120), true);
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextWrapped("%s", session_.preflight.summary.c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndChild();
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

void StudioDebuggerPanel::RenderGraphTraceView() {
    ImGui::Text("Graph Trace");
    ImGui::BeginChild("StudioDebuggerGraphTrace", ImVec2(0, 280), true,
                      ImGuiWindowFlags_HorizontalScrollbar);

    const DebugTraceRecord* snapshot = nullptr;
    for (const auto& trace : session_.traces) {
        if (trace.phase == "GraphSnapshot") {
            snapshot = &trace;
            break;
        }
    }

    if (!snapshot || !snapshot->payload.contains("nodes") ||
        !snapshot->payload["nodes"].is_array()) {
        ImGui::TextDisabled("No frozen graph snapshot available for this run.");
        ImGui::TextDisabled("Run Studio Debugger again to capture a graph trace snapshot.");
        ImGui::EndChild();
        return;
    }

    struct NodeDrawInfo {
        int id = -1;
        std::string name;
        std::string status = "captured";
        int trace_count = 0;
        int issue_count = 0;
        int recommendation_count = 0;
        std::string detail;
        ImVec2 min;
        ImVec2 max;
    };

    std::map<int, NodeDrawInfo> node_draws;
    struct NodeAggregate {
        std::string status = "captured";
        int severity = 1;
        int trace_count = 0;
        int issue_count = 0;
        int recommendation_count = 0;
        std::vector<std::string> details;
    };
    std::map<int, NodeAggregate> node_status;
    for (const auto& trace : session_.traces) {
        if (trace.node_id < 0) {
            continue;
        }
        auto& agg = node_status[trace.node_id];
        const int severity = TraceStatusSeverity(trace.status);
        if (severity >= agg.severity) {
            agg.severity = severity;
            agg.status = trace.status.empty() ? "unknown" : trace.status;
        }
        agg.trace_count++;
        agg.issue_count += static_cast<int>(trace.issues.size());
        if (agg.details.size() < 4) {
            std::string detail = trace.phase + " / " +
                std::string(DebugTraceRoleName(trace.role)) + " / " +
                (trace.status.empty() ? "unknown" : trace.status);
            agg.details.push_back(std::move(detail));
        }
    }
    for (const auto& rec : session_.recommendations) {
        if (rec.node_id >= 0) {
            node_status[rec.node_id].recommendation_count++;
            if (node_status[rec.node_id].details.size() < 4) {
                node_status[rec.node_id].details.push_back(
                    "Recommendation: " + rec.title);
            }
        }
    }

    const ImVec2 canvas_origin = ImGui::GetCursorScreenPos();
    const float node_w = 150.0f;
    const float node_h = 48.0f;
    const float step_x = 190.0f;
    const float step_y = 82.0f;
    const int cols = 4;
    int index = 0;

    for (const auto& item : snapshot->payload["nodes"]) {
        const int id = item.value("id", -1);
        std::string name = item.value("name", "");
        if (name.empty()) {
            name = "Node " + std::to_string(id);
        }
        const int col = index % cols;
        const int row = index / cols;
        NodeDrawInfo info;
        info.id = id;
        info.name = name;
        if (auto it = node_status.find(id); it != node_status.end()) {
            info.status = it->second.status;
            info.trace_count = it->second.trace_count;
            info.issue_count = it->second.issue_count;
            info.recommendation_count = it->second.recommendation_count;
            for (const auto& detail : it->second.details) {
                if (!info.detail.empty()) {
                    info.detail += "\n";
                }
                info.detail += detail;
            }
        }
        info.min = ImVec2(canvas_origin.x + col * step_x, canvas_origin.y + row * step_y);
        info.max = ImVec2(info.min.x + node_w, info.min.y + node_h);
        node_draws[id] = std::move(info);
        ++index;
    }

    const int rows = std::max(1, (index + cols - 1) / cols);
    ImDrawList* draw = ImGui::GetWindowDrawList();

    if (snapshot->payload.contains("links") && snapshot->payload["links"].is_array()) {
        for (const auto& link : snapshot->payload["links"]) {
            const int from = link.value("from_node", -1);
            const int to = link.value("to_node", -1);
            auto from_it = node_draws.find(from);
            auto to_it = node_draws.find(to);
            if (from_it == node_draws.end() || to_it == node_draws.end()) {
                continue;
            }
            const ImVec2 p1(from_it->second.max.x,
                            (from_it->second.min.y + from_it->second.max.y) * 0.5f);
            const ImVec2 p2(to_it->second.min.x,
                            (to_it->second.min.y + to_it->second.max.y) * 0.5f);
            draw->AddLine(p1, p2, IM_COL32(120, 135, 155, 210), 2.0f);
            draw->AddTriangleFilled(
                ImVec2(p2.x, p2.y),
                ImVec2(p2.x - 7.0f, p2.y - 4.0f),
                ImVec2(p2.x - 7.0f, p2.y + 4.0f),
                IM_COL32(120, 135, 155, 210));
        }
    }

    for (auto& [id, info] : node_draws) {
        const ImVec4 color = TraceStatusColor(info.status);
        const ImU32 fill = ImGui::ColorConvertFloat4ToU32(
            ImVec4(color.x * 0.22f, color.y * 0.22f, color.z * 0.22f, 0.95f));
        const ImU32 border = ImGui::ColorConvertFloat4ToU32(color);
        draw->AddRectFilled(info.min, info.max, fill, 6.0f);
        draw->AddRect(info.min, info.max, border, 6.0f, 0, 2.0f);
        draw->AddText(ImVec2(info.min.x + 8.0f, info.min.y + 7.0f),
                      IM_COL32(235, 238, 245, 255),
                      info.name.c_str());
        draw->AddText(ImVec2(info.min.x + 8.0f, info.min.y + 27.0f),
                      IM_COL32(180, 188, 200, 255),
                      info.status.c_str());
        if (info.trace_count > 0) {
            const std::string count_text = std::to_string(info.trace_count) + " traces";
            draw->AddText(ImVec2(info.max.x - 68.0f, info.min.y + 27.0f),
                          IM_COL32(170, 178, 190, 255),
                          count_text.c_str());
        }
        if (info.issue_count > 0 || info.recommendation_count > 0) {
            const std::string badge = std::to_string(info.issue_count) + "i " +
                std::to_string(info.recommendation_count) + "r";
            draw->AddText(ImVec2(info.max.x - 54.0f, info.min.y + 7.0f),
                          IM_COL32(255, 215, 120, 255),
                          badge.c_str());
        }

        ImGui::SetCursorScreenPos(info.min);
        ImGui::PushID(id);
        if (ImGui::InvisibleButton("graph_node", ImVec2(node_w, node_h))) {
            for (int i = 0; i < static_cast<int>(session_.traces.size()); ++i) {
                if (session_.traces[i].node_id == id) {
                    selected_trace_index_ = i;
                    break;
                }
            }
            if (focus_node_callback_) {
                focus_node_callback_(id);
            }
        }
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::Text("%s", info.name.c_str());
            ImGui::Text("node %d", id);
            ImGui::Text("status: %s", info.status.c_str());
            ImGui::Text("traces: %d  issues: %d  recommendations: %d",
                        info.trace_count, info.issue_count, info.recommendation_count);
            if (!info.detail.empty()) {
                ImGui::Separator();
                ImGui::TextUnformatted(info.detail.c_str());
            }
            ImGui::EndTooltip();
        }
        ImGui::PopID();
    }

    ImGui::SetCursorScreenPos(canvas_origin);
    ImGui::Dummy(ImVec2(cols * step_x, rows * step_y + 12.0f));
    ImGui::EndChild();
}

void StudioDebuggerPanel::RenderRunHistory() {
    if (session_.run_history.empty()) {
        session_.run_history = DebugRunStore::ListRecent(8);
    }

    ImGui::Text("Run History");
    ImGui::BeginChild("StudioDebuggerRunHistory", ImVec2(0, 145), true);

    if (session_.run_history.empty()) {
        ImGui::TextDisabled("No saved debugger runs.");
        ImGui::EndChild();
        return;
    }

    for (int i = 0; i < static_cast<int>(session_.run_history.size()); ++i) {
        const auto& run = session_.run_history[i];
        ImGui::PushID(i);
        const bool current = !current_run_id_.empty() && run.run_id == current_run_id_;
        const ImVec4 status_color = run.success
            ? ImVec4(0.45f, 0.95f, 0.55f, 1.0f)
            : ImVec4(1.0f, 0.82f, 0.35f, 1.0f);

        ImGui::PushStyleColor(ImGuiCol_Text, status_color);
        const bool selected = run.run_id == session_.run_id;
        const std::string label = std::string(run.success ? "passed  " : "needs attention  ") +
            run.run_id + (current ? "  (current)" : "") + "##run_history_" + std::to_string(i);
        if (ImGui::Selectable(label.c_str(), selected)) {
            LoadStoredRun(run.run_id);
            ImGui::PopStyleColor();
            ImGui::PopID();
            continue;
        }
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::TextDisabled("%s", run.timestamp.c_str());
        ImGui::TextDisabled("issues=%zu traces=%zu events=%zu recommendations=%zu graph=0x%016llx",
                            run.issue_count, run.trace_count, run.event_count,
                            run.recommendation_count,
                            static_cast<unsigned long long>(run.graph_hash));
        if (!run.summary.empty()) {
            ImGui::TextWrapped("  %s", run.summary.c_str());
        }
        if (ImGui::IsItemHovered() && !run.file_path.empty()) {
            ImGui::SetTooltip("%s", run.file_path.c_str());
        }
        ImGui::Separator();
        ImGui::PopID();
    }

    ImGui::EndChild();
}

void StudioDebuggerPanel::RenderLastRun() {
    if (!session_.last_run.available) {
        if (auto last_run = CrashRunRecorder::LoadLastRun()) {
            session_.last_run = *last_run;
        }
    }

    ImGui::Text("Crash / Last Run");
    ImGui::BeginChild("StudioDebuggerLastRun", ImVec2(0, 240), true);

    const auto& run = session_.last_run;
    if (!run.available) {
        ImGui::TextDisabled("No training run heartbeat found.");
        ImGui::EndChild();
        return;
    }

    const ImVec4 status_color = run.suspected_crash
        ? ImVec4(1.0f, 0.45f, 0.45f, 1.0f)
        : ImVec4(0.45f, 0.95f, 0.55f, 1.0f);

    ImGui::Text("Run: %s", run.run_id.c_str());
    ImGui::SameLine();
    ImGui::TextColored(status_color, "Status: %s", run.status.c_str());
    ImGui::Text("Dataset: %s", run.dataset_name.empty() ? "(unknown)" : run.dataset_name.c_str());
    ImGui::Text("Backend: %s", run.backend.empty() ? "(unknown)" : run.backend.c_str());
    ImGui::Text("Last event: epoch %d/%d batch %d/%d",
                run.epoch, run.epochs, run.batch, run.total_batches);
    ImGui::Text("Last stage: %s", run.last_stage.c_str());
    ImGui::Text("Metrics: loss=%.4f acc=%.2f%%", run.loss, run.accuracy * 100.0f);
    ImGui::Text("Time: %s", run.last_event_time.c_str());
    if (!run.warning.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.82f, 0.35f, 1.0f), "Warning:");
        ImGui::SameLine();
        ImGui::TextWrapped("%s", run.warning.c_str());
    }
    ImGui::TextDisabled(
        "Hint: plot panel events help diagnose UI/training-thread races. "
        "If this run crashed, compare the last panel event with the last training stage.");
    ImGui::TextDisabled(
        "Windows crash details usually come from Event Viewer, Reliability Monitor, or WER ReportArchive.");

    if (run.windows_crash_available) {
        ImGui::Separator();
        ImGui::Text("Windows crash report");
        ImGui::Text("Fault module: %s",
                    run.windows_fault_module.empty() ? "(unknown)" : run.windows_fault_module.c_str());
        ImGui::Text("Exception code: %s",
                    run.windows_exception_code.empty() ? "(unknown)" : run.windows_exception_code.c_str());
        ImGui::Text("WER EventTime: %s",
                    run.windows_crash_time.empty() ? "(unknown)" : run.windows_crash_time.c_str());
        if (!run.windows_report_id.empty()) {
            ImGui::Text("Report id: %s", run.windows_report_id.c_str());
        }
        if (!run.windows_report_path.empty()) {
            ImGui::TextWrapped("Report path: %s", run.windows_report_path.c_str());
        }
    } else {
        ImGui::TextDisabled(
            "No local WER report attached. Open Reliability Monitor with 'perfmon /rel' "
            "or Event Viewer > Windows Logs > Application and search for cyxwiz-engine.exe.");
    }

    if (!run.panel_events.empty()) {
        ImGui::Separator();
        ImGui::Text("Recent run events");
        const int start = std::max(0, static_cast<int>(run.panel_events.size()) - 6);
        for (int i = start; i < static_cast<int>(run.panel_events.size()); ++i) {
            ImGui::TextWrapped("%s", run.panel_events[i].c_str());
        }
    }

    ImGui::EndChild();
}

void StudioDebuggerPanel::RenderTrainingTrace() {
    if (!session_.training_trace.available) {
        if (auto trace = TrainingTraceCollector::LoadLastTrace()) {
            session_.training_trace = *trace;
        }
    }

    ImGui::Text("Training Trace");
    ImGui::BeginChild("StudioDebuggerTrainingTrace", ImVec2(0, 170), true);

    const auto& trace = session_.training_trace;
    if (!trace.available) {
        ImGui::TextDisabled("No training trace found yet.");
        ImGui::EndChild();
        return;
    }

    ImGui::Text("Run: %s", trace.run_id.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("Status: %s", trace.status.c_str());
    ImGui::Text("Latest: epoch %d batch %d/%d  stage=%s",
                trace.latest_epoch, trace.latest_batch,
                trace.latest_total_batches, trace.latest_stage.c_str());
    ImGui::Text("Metrics: loss=%.4f acc=%.2f%%",
                trace.latest_loss, trace.latest_accuracy * 100.0f);

    if (!trace.warnings.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.82f, 0.35f, 1.0f),
                           "Warnings: %zu", trace.warnings.size());
    }

    ImGui::Separator();
    const int start = std::max(0, static_cast<int>(trace.recent_events.size()) - 8);
    for (int i = start; i < static_cast<int>(trace.recent_events.size()); ++i) {
        const auto& event = trace.recent_events[i];
        ImVec4 color = event.status == "ok"
            ? ImVec4(0.85f, 0.85f, 0.85f, 1.0f)
            : ImVec4(1.0f, 0.45f, 0.45f, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_Text, color);
        ImGui::Text("%s", event.stage.c_str());
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::TextDisabled("E%d B%d/%d loss=%.4f acc=%.2f%% %.2fms",
                            event.epoch, event.batch, event.total_batches,
                            event.loss, event.accuracy * 100.0f,
                            event.duration_ms);
        if (!event.message.empty()) {
            ImGui::TextWrapped("  %s", event.message.c_str());
        }
    }

    ImGui::EndChild();
}

void StudioDebuggerPanel::RenderTraceTimeline() {
    if (!session_.traces.empty()) {
        ImGui::Text("Trace timeline - %s lens", ActiveLensName());
        ImGui::BeginChild("StudioDebuggerUnifiedTraceTimeline", ImVec2(0, 220), true);
        int visible_count = 0;
        for (int i = 0; i < static_cast<int>(session_.traces.size()); ++i) {
            const auto& trace = session_.traces[i];
            if (!TraceMatchesActiveLens(trace)) {
                continue;
            }
            ++visible_count;
            const bool selected = (selected_trace_index_ == i);
            std::string label = trace.phase + "  " +
                std::string(DebugTraceRoleName(trace.role)) + "  " +
                (trace.node_name.empty() ? "(graph)" : trace.node_name);
            if (trace.node_id >= 0) {
                label += "  [node " + std::to_string(trace.node_id) + "]";
            }
            label += "##trace_" + std::to_string(i);

            ImVec4 row_color = ImVec4(0.85f, 0.85f, 0.85f, 1.0f);
            if (trace.status == "ok" || trace.status == "passed") {
                row_color = ImVec4(0.45f, 0.95f, 0.55f, 1.0f);
            } else if (trace.status == "warning" || trace.status == "zero" ||
                       trace.status == "shape_mismatch") {
                row_color = ImVec4(1.0f, 0.82f, 0.35f, 1.0f);
            } else if (trace.status == "failed" || trace.status == "nan") {
                row_color = ImVec4(1.0f, 0.45f, 0.45f, 1.0f);
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
            ImGui::TextDisabled("%s %.2f ms", trace.status.c_str(), trace.duration_ms);
        }
        if (visible_count == 0) {
            ImGui::TextDisabled("No traces match the active lens.");
        }
        ImGui::EndChild();
        return;
    }

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

void StudioDebuggerPanel::RenderStudioEvents() {
    ImGui::Text("Studio events");
    ImGui::BeginChild("StudioDebuggerStudioEvents", ImVec2(0, 130), true);

    if (session_.studio_events.empty()) {
        ImGui::TextDisabled("No Studio events captured for this run.");
        ImGui::EndChild();
        return;
    }

    for (const auto& event : session_.studio_events) {
        ImVec4 color = ImVec4(0.85f, 0.85f, 0.85f, 1.0f);
        if (event.status == "passed" || event.status == "ready" || event.status == "started") {
            color = ImVec4(0.45f, 0.7f, 1.0f, 1.0f);
        } else if (event.status == "failed" || event.status == "blocked") {
            color = ImVec4(1.0f, 0.45f, 0.45f, 1.0f);
        }

        ImGui::PushStyleColor(ImGuiCol_Text, color);
        ImGui::TextUnformatted(event.action.c_str());
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::TextDisabled("%s", event.status.c_str());
        if (!event.message.empty()) {
            ImGui::TextWrapped("  %s", event.message.c_str());
        }
    }

    ImGui::EndChild();
}

void StudioDebuggerPanel::RenderTextPayloadInspector(const DebugTraceRecord& trace) {
    if (!IsPreprocessingTrace(trace) || trace.payload.empty()) {
        return;
    }

    const auto& payload = trace.payload;
    ImGui::Separator();
    ImGui::Text("Text preprocessing");

    if (JsonHas(payload, "dataset")) {
        ImGui::Text("Dataset: %s", JsonString(payload, "dataset").c_str());
    }
    if (JsonHas(payload, "sample_index")) {
        ImGui::Text("Sample index: %.0f", JsonNumber(payload, "sample_index"));
    }

    if (JsonHas(payload, "raw_text_preview")) {
        ImGui::Text("Raw text");
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextWrapped("%s", JsonString(payload, "raw_text_preview").c_str());
        ImGui::PopTextWrapPos();
    }
    if (JsonHas(payload, "normalized_preview")) {
        ImGui::Text("Normalized / decoded");
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextWrapped("%s", JsonString(payload, "normalized_preview").c_str());
        ImGui::PopTextWrapPos();
    }

    if (JsonHas(payload, "token_count")) {
        ImGui::Text("Token count: %.0f", JsonNumber(payload, "token_count"));
    }
    const std::string tokens = JsonArrayPreview(payload, "tokens_preview", 28);
    if (!tokens.empty()) {
        ImGui::Text("Tokens");
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextWrapped("%s", tokens.c_str());
        ImGui::PopTextWrapPos();
    }

    if (JsonHas(payload, "vocab_size")) {
        ImGui::Text("Vocab size: %.0f", JsonNumber(payload, "vocab_size"));
    }
    if (JsonHas(payload, "vocab_file")) {
        const std::string vocab_file = JsonString(payload, "vocab_file");
        ImGui::Text("Vocab file: %s", vocab_file.empty() ? "(in-memory / generated)" : vocab_file.c_str());
    }
    if (JsonHas(payload, "vocab_hits") || JsonHas(payload, "vocab_misses")) {
        ImGui::Text("Vocab hits: %.0f  misses: %.0f",
                    JsonNumber(payload, "vocab_hits"),
                    JsonNumber(payload, "vocab_misses"));
    }
    if (JsonHas(payload, "unknown_token_count") || JsonHas(payload, "unknown_token_ratio")) {
        ImGui::Text("Unknown tokens: %.0f  ratio: %.2f%%",
                    JsonNumber(payload, "unknown_token_count"),
                    JsonNumber(payload, "unknown_token_ratio") * 100.0);
    }
    const std::string missing = JsonArrayPreview(payload, "missing_tokens_preview", 20);
    if (!missing.empty()) {
        ImGui::Text("Missing tokens");
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextWrapped("%s", missing.c_str());
        ImGui::PopTextWrapPos();
    }
    const std::string token_ids = JsonArrayPreview(payload, "token_ids_preview", 48);
    if (!token_ids.empty()) {
        ImGui::Text("Token ids");
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextWrapped("%s", token_ids.c_str());
        ImGui::PopTextWrapPos();
    }

    if (JsonHas(payload, "max_length") || JsonHas(payload, "final_sequence_length")) {
        ImGui::Text("Sequence length: %.0f / max %.0f",
                    JsonNumber(payload, "final_sequence_length"),
                    JsonNumber(payload, "max_length"));
    }
    if (JsonHas(payload, "pad_count") || JsonHas(payload, "pad_ratio")) {
        ImGui::Text("Padding: %.0f tokens  ratio: %.2f%%",
                    JsonNumber(payload, "pad_count"),
                    JsonNumber(payload, "pad_ratio") * 100.0);
    }
    if (JsonHas(payload, "padded") || JsonHas(payload, "truncated")) {
        const bool padded = JsonBool(payload, "padded");
        const bool truncated = JsonBool(payload, "truncated");
        ImVec4 trunc_color = truncated
            ? ImVec4(1.0f, 0.82f, 0.35f, 1.0f)
            : ImVec4(0.45f, 0.95f, 0.55f, 1.0f);
        ImGui::Text("Padded: %s", padded ? "yes" : "no");
        ImGui::SameLine();
        ImGui::TextColored(trunc_color, "Truncated: %s", truncated ? "yes" : "no");
    }
    if (JsonHas(payload, "note")) {
        ImGui::TextWrapped("Note: %s", JsonString(payload, "note").c_str());
    }
}

void StudioDebuggerPanel::RenderTraceDiagnosis(const DebugTraceRecord& trace) {
    std::string meaning;
    std::string likely_cause;
    std::string next_adjustment;

    const auto& payload = trace.payload;

    if (trace.status == "shape_mismatch") {
        meaning = "The runtime tensor shape does not match the graph/compiler prediction.";
        likely_cause = "A layer parameter, preprocessing output length, recurrent direction setting, or graph connection is producing a different shape than expected.";
        next_adjustment = "Inspect the input/output shapes here, then check the upstream node and the layer configuration that consumes this tensor.";
    } else if (trace.phase == "GraphSnapshot") {
        meaning = "The debugger captured an immutable copy of the Studio graph at run start.";
        likely_cause = "This prevents later UI edits from being confused with the graph that produced this debug run.";
        next_adjustment = "Use this snapshot as the source of truth when comparing trace rows, run history, or future graph-node execution.";
    } else if (trace.phase == "Compile") {
        meaning = "Compile checks whether the Studio graph can be converted into an executable training configuration.";
        if (trace.status == "failed") {
            likely_cause = "The graph has invalid wiring, missing required parameters, incompatible layer settings, or a compiler exception.";
            next_adjustment = "Inspect compile issues first; do not run Local Debug or Smoke Run until compile passes.";
        } else {
            likely_cause = "The graph compiled into a training configuration.";
            next_adjustment = "Continue to Preflight to verify dataset and preprocessing readiness.";
        }
    } else if (trace.phase == "Preflight") {
        meaning = "Preflight checks cheap graph/data readiness before model execution.";
        if (trace.status == "blocked") {
            likely_cause = "A required dataset, vocabulary, shape, label/loss setting, or preprocessing setting is not ready.";
            next_adjustment = "Fix the listed preflight issues before running Smoke Run or full training.";
        } else {
            likely_cause = "The graph and data path passed the lightweight readiness checks.";
            next_adjustment = "Run Smoke Run for real-data validation before long training.";
        }
    } else if (trace.status == "nan" || trace.status == "failed") {
        meaning = "This trace reached an invalid or failed execution state.";
        likely_cause = "The model may have non-finite values, invalid labels, an unsupported runtime path, or a backend exception.";
        next_adjustment = "Check the payload, loss trace, and Runtime lens before running a long training job again.";
    } else if (trace.status == "zero") {
        meaning = "A gradient trace reported zero gradient.";
        likely_cause = "The parameter may be disconnected, saturated by an activation, blocked by the loss path, or using bookkeeping that did not attach gradients.";
        next_adjustment = "Inspect adjacent forward traces and confirm the loss receives the expected prediction and target tensors.";
    } else if (trace.phase == "TextVocabulary") {
        const double unknown_ratio = JsonNumber(payload, "unknown_token_ratio", 0.0);
        meaning = "Vocabulary maps tokenizer output into integer ids for the embedding layer.";
        if (unknown_ratio > 0.20) {
            likely_cause = "Many tokens are not present in the vocabulary, so the embedding sees too many unknown ids.";
            next_adjustment = "Rebuild the vocabulary from the training corpus, increase vocab size, lower min_word_freq, or inspect tokenizer settings.";
        } else if (unknown_ratio > 0.05) {
            likely_cause = "Some tokens are missing from the vocabulary, but this single sample is not necessarily enough to change settings.";
            next_adjustment = "Use Smoke Run aggregate stats before changing vocab size or min_word_freq.";
        } else {
            likely_cause = "Vocabulary coverage looks acceptable for this selected sample.";
            next_adjustment = "Continue checking padding, sequence length, and gradient flow.";
        }
    } else if (trace.phase == "TextPadding") {
        const bool truncated = JsonBool(payload, "truncated", false);
        const double pad_ratio = JsonNumber(payload, "pad_ratio", 0.0);
        meaning = "Padding/truncation turns variable text lengths into a fixed sequence length.";
        if (truncated) {
            likely_cause = "The selected text is longer than max_length, so information was cut before embedding/GRU.";
            next_adjustment = "Increase max_length or inspect whether the important sentiment signal appears after the truncation point.";
        } else if (pad_ratio > 0.80) {
            likely_cause = "This selected sample is much shorter than max_length.";
            next_adjustment = "Do not change max_length from one sample alone; inspect aggregate smoke padding before reducing it.";
        } else {
            likely_cause = "Sequence length handling looks acceptable for this selected sample.";
            next_adjustment = "Continue to embedding/GRU shape and gradient traces.";
        }
    } else if (trace.phase == "TextTokenizer") {
        meaning = "Tokenizer converts raw text into tokens before vocabulary lookup.";
        likely_cause = "If tokens look wrong, the tokenizer type, lowercasing, punctuation handling, or source text column may be wrong.";
        next_adjustment = "Compare raw text to token preview and confirm the selected tokenizer matches the dataset language/style.";
    } else if (trace.phase == "SmokeRun.Loss") {
        const bool pred_bad = JsonBool(payload, "predictions_have_non_finite", false);
        meaning = "Smoke loss validates that real data reaches the loss function and produces finite scalar feedback.";
        if (pred_bad) {
            likely_cause = "Predictions contain NaN or Inf before or during loss computation.";
            next_adjustment = "Lower learning rate for train runs, inspect input ranges, and check the first layer producing non-finite values.";
        } else {
            likely_cause = "Loss is finite for the smoke batch.";
            next_adjustment = "Inspect gradients next; finite loss alone does not prove the model can learn.";
        }
    } else if (trace.phase == "SmokeRun.Backward") {
        const double grad_count = JsonNumber(payload, "gradient_tensor_count", 0.0);
        const double zero_grad_count = JsonNumber(payload, "zero_gradient_tensor_count", 0.0);
        meaning = "Backward trace checks whether trainable parameters receive gradients on real data.";
        if (grad_count <= 0.0) {
            likely_cause = "No gradient tensors were recorded, which suggests a disconnected loss path or gradient bookkeeping issue.";
            next_adjustment = "Check loss wiring, target dtype/shape, and parameter registration before full training.";
        } else if (zero_grad_count > 0.0) {
            likely_cause = "Some parameters have zero gradients on the smoke batch.";
            next_adjustment = "Inspect which layers are affected; if recurrent or embedding layers are zero, check sequence ids and loss connection.";
        } else {
            likely_cause = "Gradient coverage looks acceptable for this smoke batch.";
            next_adjustment = "Use a short training run to inspect loss trend before committing to long training.";
        }
    }

    if (meaning.empty()) {
        return;
    }

    ImGui::Separator();
    ImGui::Text("Diagnosis");
    ImGui::TextWrapped("What this means: %s", meaning.c_str());
    ImGui::TextWrapped("Likely cause: %s", likely_cause.c_str());
    ImGui::TextWrapped("Next adjustment: %s", next_adjustment.c_str());
}

void StudioDebuggerPanel::RenderSelectedTraceDetails() {
    if (!session_.traces.empty()) {
        ImGui::Text("Node Inspector");
        ImGui::BeginChild("StudioDebuggerUnifiedTraceDetails", ImVec2(0, 260), true);

        if (selected_trace_index_ < 0 ||
            selected_trace_index_ >= static_cast<int>(session_.traces.size())) {
            ImGui::TextDisabled("Select a trace row to inspect node details.");
            ImGui::EndChild();
            return;
        }

        const auto& trace = session_.traces[selected_trace_index_];
        ImGui::Text("Node: %s", trace.node_name.empty() ? "(graph)" : trace.node_name.c_str());
        ImGui::Text("Node id: %d", trace.node_id);
        ImGui::Text("Type: %s", trace.node_type.c_str());
        ImGui::Text("Phase: %s", trace.phase.c_str());
        ImGui::Text("Role: %s", DebugTraceRoleName(trace.role));
        ImGui::Text("Status: %s", trace.status.c_str());
        ImGui::Text("Input: %s", FormatShape(trace.input_shape).c_str());
        ImGui::Text("Output: %s", FormatShape(trace.output_shape).c_str());
        if (!trace.dtype.empty()) {
            ImGui::Text("DType: %s", trace.dtype.c_str());
        }
        ImGui::Text("Duration: %.2f ms", trace.duration_ms);

        if (trace.node_id >= 0 && focus_node_callback_) {
            if (ImGui::Button("Focus Node")) {
                focus_node_callback_(trace.node_id);
            }
        }

        if (!trace.issues.empty()) {
            ImGui::Separator();
            ImGui::Text("Trace issues");
            for (const auto& issue : trace.issues) {
                ImGui::PushStyleColor(ImGuiCol_Text, LevelColor(issue.level));
                ImGui::TextWrapped("%s", issue.message.c_str());
                ImGui::PopStyleColor();
            }
        }

        bool has_recommendation = false;
        for (const auto& rec : session_.recommendations) {
            if (rec.node_id != trace.node_id &&
                !(rec.node_id < 0 && trace.status != "ok" && trace.status != "passed")) {
                continue;
            }
            if (!has_recommendation) {
                ImGui::Separator();
                ImGui::Text("Related recommendations");
                has_recommendation = true;
            }
            ImGui::PushStyleColor(ImGuiCol_Text, RecommendationColor(rec.severity));
            ImGui::Text("%s", DebugRecommendationSeverityName(rec.severity));
            ImGui::PopStyleColor();
            ImGui::SameLine();
            ImGui::TextWrapped("[%s] %s", rec.category.c_str(), rec.title.c_str());
            if (!rec.action.empty()) {
                ImGui::TextWrapped("  Next: %s", rec.action.c_str());
            }
        }

        RenderTraceDiagnosis(trace);
        RenderTextPayloadInspector(trace);

        if (!trace.payload.empty()) {
            ImGui::Separator();
            if (ImGui::TreeNode("Raw payload JSON")) {
                const std::string payload = trace.payload.dump(2);
                ImGui::TextWrapped("%s", payload.c_str());
                ImGui::TreePop();
            }
        }

        ImGui::EndChild();
        return;
    }

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

void StudioDebuggerPanel::RenderRecommendations() {
    ImGui::Text("Recommendations");
    ImGui::BeginChild("StudioDebuggerRecommendations", ImVec2(0, 170), true);

    if (session_.recommendations.empty()) {
        ImGui::TextDisabled("No recommendations generated.");
        ImGui::EndChild();
        return;
    }

    for (int i = 0; i < static_cast<int>(session_.recommendations.size()); ++i) {
        const auto& rec = session_.recommendations[i];
        ImGui::PushID(i);
        ImGui::PushStyleColor(ImGuiCol_Text, RecommendationColor(rec.severity));
        ImGui::Text("%s", DebugRecommendationSeverityName(rec.severity));
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::Text("[%s] %s", rec.category.c_str(), rec.title.c_str());

        if (rec.node_id >= 0 && focus_node_callback_) {
            ImGui::SameLine();
            if (ImGui::SmallButton("Focus")) {
                focus_node_callback_(rec.node_id);
            }
        }

        if (!rec.detail.empty()) {
            ImGui::TextWrapped("  %s", rec.detail.c_str());
        }
        if (!rec.action.empty()) {
            ImGui::TextWrapped("  Next: %s", rec.action.c_str());
        }
        ImGui::Spacing();
        ImGui::PopID();
    }

    ImGui::EndChild();
}

void StudioDebuggerPanel::RenderLensContent() {
    switch (active_lens_) {
        case StudioDebuggerLens::Overview:
            RenderOverview();
            ImGui::Separator();
            RenderGraphTraceView();
            ImGui::Separator();
            RenderRunHistory();
            ImGui::Separator();
            RenderRunComparison();
            ImGui::Separator();
            RenderTraceTimeline();
            ImGui::Separator();
            RenderSelectedTraceDetails();
            ImGui::Separator();
            RenderRecommendations();
            return;

        case StudioDebuggerLens::Preprocessing:
            RenderOverview();
            ImGui::Separator();
            RenderGraphTraceView();
            ImGui::Separator();
            RenderTraceTimeline();
            ImGui::Separator();
            RenderSelectedTraceDetails();
            ImGui::Separator();
            RenderIssueList();
            ImGui::Separator();
            RenderRecommendations();
            return;

        case StudioDebuggerLens::Shapes:
            RenderGraphTraceView();
            ImGui::Separator();
            RenderTraceTimeline();
            ImGui::Separator();
            RenderSelectedTraceDetails();
            ImGui::Separator();
            RenderIssueList();
            ImGui::Separator();
            RenderRecommendations();
            return;

        case StudioDebuggerLens::Values:
            RenderGraphTraceView();
            ImGui::Separator();
            RenderTraceTimeline();
            ImGui::Separator();
            RenderSelectedTraceDetails();
            ImGui::Separator();
            RenderIssueList();
            return;

        case StudioDebuggerLens::Gradients:
            RenderGraphTraceView();
            ImGui::Separator();
            RenderTraceTimeline();
            ImGui::Separator();
            RenderSelectedTraceDetails();
            ImGui::Separator();
            RenderRecommendations();
            return;

        case StudioDebuggerLens::Runtime:
            RenderLastRun();
            ImGui::Separator();
            RenderTrainingTrace();
            ImGui::Separator();
            RenderTraceTimeline();
            ImGui::Separator();
            RenderStudioEvents();
            ImGui::Separator();
            RenderRecommendations();
            return;

        case StudioDebuggerLens::StudioEvents:
            RenderStudioEvents();
            ImGui::Separator();
            RenderTraceTimeline();
            ImGui::Separator();
            RenderSelectedTraceDetails();
            ImGui::Separator();
            RenderRunHistory();
            return;

        case StudioDebuggerLens::Recommendations:
            RenderRecommendations();
            ImGui::Separator();
            RenderIssueList();
            ImGui::Separator();
            RenderTraceTimeline();
            ImGui::Separator();
            RenderSelectedTraceDetails();
            return;
    }

    RenderOverview();
}

void StudioDebuggerPanel::Render() {
    if (!visible_) {
        return;
    }

    std::string title = std::string(ICON_FA_BUG) + " Studio Debugger###StudioDebuggerPanel";
    if (ImGui::Begin(title.c_str(), &visible_)) {
        RenderToolbar();
        ImGui::Separator();
        RenderTraceSettings();
        ImGui::Separator();
        RenderLensContent();
    }
    ImGui::End();
}

} // namespace cyxwiz
