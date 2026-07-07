#include "cyxwiz_assistant_plugin.h"
#include "knowledge_pack_backend.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iterator>

namespace cyxwiz::plugin::assistant {

namespace {

constexpr const char* kPanelId = "com.cyxwiz.assistant.panel";
constexpr const char* kContextModeLabels[] = {"General", "Trace", "Training"};
constexpr const char* kDefaultRuntimeEndpoint = "http://127.0.0.1:8768/completion";
constexpr const char* kDefaultPackPath = "docs/Data Studio/tofix42/knowledge_pack";

void TextWrappedOrMuted(const std::string& text, const char* fallback) {
    if (text.empty()) {
        ImGui::TextDisabled("%s", fallback);
        return;
    }
    ImGui::TextWrapped("%s", text.c_str());
}

} // namespace

CyxWizAssistantPlugin::CyxWizAssistantPlugin()
    : backend_(std::make_unique<KnowledgePackBackend>(std::filesystem::path(kDefaultPackPath))) {
    std::strncpy(question_buffer_.data(), "What source file defines DebugTraceRecord?",
                 question_buffer_.size() - 1);
    std::strncpy(knowledge_pack_path_buffer_.data(), kDefaultPackPath,
                 knowledge_pack_path_buffer_.size() - 1);
    std::strncpy(runtime_endpoint_buffer_.data(), kDefaultRuntimeEndpoint,
                 runtime_endpoint_buffer_.size() - 1);
}

std::filesystem::path CyxWizAssistantPlugin::ResolveDefaultPackPath(const PluginContext* context) {
    if (context) {
        const auto plugin_dir = context->GetPluginDir();
        if (!plugin_dir.empty()) {
            const auto repo_root = plugin_dir.parent_path().parent_path().parent_path();
            const auto repo_pack = repo_root / "docs" / "Data Studio" / "tofix42" / "knowledge_pack";
            if (std::filesystem::exists(repo_pack)) {
                return repo_pack;
            }

            const auto local_pack = plugin_dir / "knowledge_pack";
            if (std::filesystem::exists(local_pack)) {
                return local_pack;
            }
        }
    }

    const auto cwd_pack = std::filesystem::current_path() /
        "docs" / "Data Studio" / "tofix42" / "knowledge_pack";
    if (std::filesystem::exists(cwd_pack)) {
        return cwd_pack;
    }

    return std::filesystem::path(kDefaultPackPath);
}

bool CyxWizAssistantPlugin::OnLoad(PluginContext& ctx) {
    ctx.LogInfo("CyxWiz Assistant: OnLoad");
    context_ = &ctx;
    backend_ = std::make_unique<KnowledgePackBackend>(ResolveDefaultPackPath(context_));
    state_ = PluginState::Loaded;
    return true;
}

bool CyxWizAssistantPlugin::OnInitialize(PluginContext& ctx) {
    ctx.LogInfo("CyxWiz Assistant: OnInitialize");
    context_ = &ctx;
    auto* pack_backend = dynamic_cast<KnowledgePackBackend*>(backend_.get());
    if (!pack_backend || !pack_backend->IsLoaded()) {
        backend_ = std::make_unique<KnowledgePackBackend>(ResolveDefaultPackPath(context_));
    }
    state_ = PluginState::Active;
    return true;
}

void CyxWizAssistantPlugin::OnShutdown(PluginContext& ctx) {
    ctx.LogInfo("CyxWiz Assistant: OnShutdown");
    state_ = PluginState::Loaded;
}

void CyxWizAssistantPlugin::OnUnload(PluginContext& ctx) {
    ctx.LogInfo("CyxWiz Assistant: OnUnload");
    context_ = nullptr;
    state_ = PluginState::Unloaded;
}

void* CyxWizAssistantPlugin::QueryInterface(const char* name) {
    if (std::string(name) == "IPanelProvider") {
        return static_cast<IPanelProvider*>(this);
    }
    if (std::string(name) == "IAssistantProvider") {
        return static_cast<IAssistantProvider*>(this);
    }
    return nullptr;
}

std::string CyxWizAssistantPlugin::ContextModeToString(ContextMode mode) {
    switch (mode) {
        case ContextMode::Trace:
            return "Trace";
        case ContextMode::Training:
            return "Training";
        case ContextMode::General:
        default:
            return "General";
    }
}

std::vector<PluginPanelInfo> CyxWizAssistantPlugin::GetPanels() {
    PluginPanelInfo info;
    info.panel_id = kPanelId;
    info.title = "CyxWiz Assistant";
    info.category = "Tools";
    info.icon = "";
    info.show_by_default = false;
    return {info};
}

const AssistantContextSnapshot* CyxWizAssistantPlugin::CurrentSnapshot() const {
    if (!context_) {
        return nullptr;
    }
    return &context_->GetAssistantContextSnapshot();
}

bool CyxWizAssistantPlugin::HasSelectedNodeContext() const {
    const auto* snapshot = CurrentSnapshot();
    return snapshot && !snapshot->selected_node_id.empty();
}

bool CyxWizAssistantPlugin::HasSelectedTraceContext() const {
    const auto* snapshot = CurrentSnapshot();
    return snapshot && !snapshot->selected_trace_id.empty();
}

bool CyxWizAssistantPlugin::HasTrainingTerminalContext() const {
    const auto* snapshot = CurrentSnapshot();
    return snapshot &&
        snapshot->training_context_json.find("\"terminal_event\"") != std::string::npos;
}

void CyxWizAssistantPlugin::SetQuestionText(const std::string& question) {
    std::fill(question_buffer_.begin(), question_buffer_.end(), '\0');
    std::strncpy(question_buffer_.data(), question.c_str(), question_buffer_.size() - 1);
}

void CyxWizAssistantPlugin::SetMissingContextResponse(const std::string& message) {
    last_response_ = {};
    last_response_.error_code = "missing_context";
    last_response_.error_message = message;
    last_response_.unknowns = message;
    last_response_.unsupported_or_not_implemented =
        "This assistant action requires an active CyxWiz selection before it can run.";
}

void CyxWizAssistantPlugin::RenderPanel(const std::string& panel_id, bool* visible) {
    if (panel_id != kPanelId) return;

    ImGui::SetNextWindowSize(ImVec2(560, 520), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("CyxWiz Assistant", visible)) {
        ImGui::End();
        return;
    }

    RenderStatus();
    ImGui::Separator();
    RenderControls();
    ImGui::Separator();
    RenderResponse();

    ImGui::End();
}

AssistantRequest CyxWizAssistantPlugin::BuildRequest() const {
    AssistantRequest request;
    request.command_name = "ask";
    request.user_text = question_buffer_.data();
    request.engine_version = "unknown";
    request.build_id = "unknown";
    request.selected_panel = ContextModeToString(context_mode_);
    request.retrieval_only = retrieval_only_;
    request.top_k = std::clamp(top_k_, 1, 10);
    request.timeout_seconds = std::clamp(timeout_seconds_, 5, 300);
    request.runtime_endpoint = runtime_endpoint_buffer_.data();

    if (context_) {
        const auto& snapshot = context_->GetAssistantContextSnapshot();
        if (!snapshot.engine_version.empty()) request.engine_version = snapshot.engine_version;
        if (!snapshot.build_id.empty()) request.build_id = snapshot.build_id;
        request.workspace_root = snapshot.workspace_root;
        request.active_graph_path = snapshot.active_graph_path;
        request.selected_run_id = snapshot.selected_run_id;
        request.selected_node_id = snapshot.selected_node_id;
        request.selected_trace_id = snapshot.selected_trace_id;
        request.debugger_context_json = snapshot.debugger_context_json;
        request.training_context_json = snapshot.training_context_json;
    }

    if (context_mode_ == ContextMode::Trace) {
        request.command_name = "explain_trace";
    } else if (context_mode_ == ContextMode::Training) {
        request.command_name = "explain_training";
    }

    return request;
}

AssistantRequest CyxWizAssistantPlugin::BuildCommandRequest(
    const AssistantCommandRequest& command) const {
    AssistantRequest request = BuildRequest();
    request.command_name = command.command_name;
    request.user_text = command.user_text;
    request.selected_panel = "CommandWindow";
    if (command.command_name == "find_source") {
        request.retrieval_only = true;
    }
    return request;
}

std::string CyxWizAssistantPlugin::FormatCommandResponse(
    const AssistantResponse& response) {
    std::string output;
    output += "[Assistant]\n";
    if (!response.error_code.empty()) {
        output += "Backend state: " + response.error_code + "\n";
        if (!response.error_message.empty()) {
            output += response.error_message + "\n\n";
        }
    }
    output += "Answer:\n" + (response.answer.empty() ? "(none)" : response.answer) + "\n\n";
    output += "Evidence:\n" + (response.evidence.empty() ? "(none)" : response.evidence) + "\n\n";
    output += "Unknowns:\n" + (response.unknowns.empty() ? "(none)" : response.unknowns) + "\n\n";
    output += "Unsupported or not implemented:\n" +
        (response.unsupported_or_not_implemented.empty()
             ? "(none)"
             : response.unsupported_or_not_implemented) + "\n";

    if (!response.retrieval_hits.empty()) {
        output += "\nRetrieval hits:\n";
        for (const auto& hit : response.retrieval_hits) {
            output += "- #" + std::to_string(hit.rank) + " score=" +
                std::to_string(static_cast<int>(hit.score)) + " " +
                hit.citation.path + ":" +
                std::to_string(hit.citation.line_start) + "-" +
                std::to_string(hit.citation.line_end) + "\n";
        }
    }
    return output;
}

AssistantCommandResponse CyxWizAssistantPlugin::RunAssistantCommand(
    const AssistantCommandRequest& command) {
    AssistantCommandResponse result;
    result.handled = true;

    if (!backend_) {
        result.success = false;
        result.error = "Assistant backend is not available.";
        return result;
    }

    if (command.command_name == "explain_trace" && !HasSelectedTraceContext()) {
        result.success = false;
        result.error = "No debugger trace is selected. Open Studio Debugger and select a trace first.";
        return result;
    }
    if (command.command_name == "explain_training" && !HasTrainingTerminalContext()) {
        result.success = false;
        result.error = "No training terminal reason is available. Run training with tracing or load a terminal event first.";
        return result;
    }
    if (command.command_name == "find_source" &&
        command.user_text.empty() &&
        !HasSelectedNodeContext()) {
        result.success = false;
        result.error = "Provide a source query or select a graph node first.";
        return result;
    }
    if (command.command_name == "ask" && command.user_text.empty()) {
        result.success = false;
        result.error = "Usage: /ask <question>";
        return result;
    }

    const auto response = backend_->Run(BuildCommandRequest(command));
    result.success = response.success;
    result.output = FormatCommandResponse(response);
    if (!response.success && !response.error_message.empty()) {
        result.error = response.error_message;
    }
    return result;
}

void CyxWizAssistantPlugin::ReloadBackend() {
    backend_ = std::make_unique<KnowledgePackBackend>(
        std::filesystem::path(knowledge_pack_path_buffer_.data()));
    last_response_ = {};
}

void CyxWizAssistantPlugin::RunAsk() {
    if (!backend_) {
        last_response_ = {};
        last_response_.error_code = "assistant_unavailable";
        last_response_.error_message = "Assistant backend is not available.";
        return;
    }
    last_response_ = backend_->Run(BuildRequest());
}

void CyxWizAssistantPlugin::RunExplainSelectedTrace() {
    if (!HasSelectedTraceContext()) {
        SetMissingContextResponse("No debugger trace is selected. Open Studio Debugger and select a trace first.");
        return;
    }
    context_mode_ = ContextMode::Trace;
    SetQuestionText("Explain the selected CyxWiz debugger trace. Cite the source contracts and separate facts from inference.");
    RunAsk();
}

void CyxWizAssistantPlugin::RunExplainTrainingStopReason() {
    if (!HasTrainingTerminalContext()) {
        SetMissingContextResponse("No training terminal reason is available. Run training with tracing or load a run that has a terminal event.");
        return;
    }
    context_mode_ = ContextMode::Training;
    SetQuestionText("Explain why the selected CyxWiz training run stopped. Cite the training trace source contracts.");
    RunAsk();
}

void CyxWizAssistantPlugin::RunFindSelectedNodeSource() {
    if (!HasSelectedNodeContext()) {
        SetMissingContextResponse("No graph node is selected. Select a node on the graph canvas first.");
        return;
    }
    context_mode_ = ContextMode::General;
    SetQuestionText("Find the source implementation and docs related to the selected CyxWiz graph node.");
    RunAsk();
}

void CyxWizAssistantPlugin::RenderStatus() const {
    ImGui::Text("Status:");
    ImGui::SameLine();

    if (state_ == PluginState::Active) {
        ImGui::TextColored(ImVec4(0.40f, 0.75f, 0.45f, 1.0f), "Panel loaded");
    } else {
        ImGui::TextColored(ImVec4(0.85f, 0.55f, 0.20f, 1.0f), "Inactive");
    }

    ImGui::SameLine();
    if (auto* pack_backend = dynamic_cast<KnowledgePackBackend*>(backend_.get())) {
        ImGui::TextDisabled("| Backend: %s", pack_backend->Status().c_str());
    } else {
        ImGui::TextDisabled("| Backend: unavailable");
    }

    if (context_) {
        const auto& snapshot = context_->GetAssistantContextSnapshot();
        ImGui::TextDisabled("Context: graph=%s node=%s project=%s",
                            snapshot.active_graph_path.empty() ? "(unsaved)" : snapshot.active_graph_path.c_str(),
                            snapshot.selected_node_id.empty() ? "(none)" : snapshot.selected_node_id.c_str(),
                            snapshot.workspace_root.empty() ? "(none)" : snapshot.workspace_root.c_str());
        ImGui::TextDisabled("Actions: trace=%s training_stop=%s",
                            snapshot.selected_trace_id.empty() ? "(none)" : snapshot.selected_trace_id.c_str(),
                            HasTrainingTerminalContext() ? "available" : "(none)");
    } else {
        ImGui::TextDisabled("Context: unavailable");
    }
}

void CyxWizAssistantPlugin::RenderControls() {
    int current_mode = static_cast<int>(context_mode_);
    ImGui::SetNextItemWidth(180.0f);
    if (ImGui::Combo("Context", &current_mode, kContextModeLabels,
                     static_cast<int>(std::size(kContextModeLabels)))) {
        context_mode_ = static_cast<ContextMode>(current_mode);
    }

    ImGui::InputTextMultiline("Question", question_buffer_.data(), question_buffer_.size(),
                              ImVec2(-1.0f, 90.0f));

    ImGui::Checkbox("Retrieval only", &retrieval_only_);
    ImGui::SameLine();
    ImGui::Checkbox("Show citations", &show_citations_);

    ImGui::SetNextItemWidth(120.0f);
    ImGui::SliderInt("Top K", &top_k_, 1, 10);

    ImGui::SetNextItemWidth(120.0f);
    ImGui::SliderInt("Timeout", &timeout_seconds_, 5, 300, "%d sec");

    ImGui::InputText("Knowledge pack", knowledge_pack_path_buffer_.data(),
                     knowledge_pack_path_buffer_.size());
    if (ImGui::Button("Reload Pack")) {
        ReloadBackend();
    }

    ImGui::InputText("Runtime endpoint", runtime_endpoint_buffer_.data(),
                     runtime_endpoint_buffer_.size());

    if (ImGui::Button("Ask")) {
        RunAsk();
    }

    ImGui::Separator();
    ImGui::TextUnformatted("Context actions");
    if (ImGui::Button("Explain selected trace", ImVec2(-1.0f, 0.0f))) {
        RunExplainSelectedTrace();
    }
    if (ImGui::Button("Training stop reason", ImVec2(-1.0f, 0.0f))) {
        RunExplainTrainingStopReason();
    }
    if (ImGui::Button("Find selected node source", ImVec2(-1.0f, 0.0f))) {
        RunFindSelectedNodeSource();
    }
}

void CyxWizAssistantPlugin::RenderResponse() const {
    if (!last_response_.error_code.empty()) {
        ImGui::TextColored(ImVec4(0.90f, 0.55f, 0.20f, 1.0f), "Backend state: %s",
                           last_response_.error_code.c_str());
        TextWrappedOrMuted(last_response_.error_message, "No backend message.");
        ImGui::Spacing();
    }

    ImGui::TextUnformatted("Answer");
    TextWrappedOrMuted(last_response_.answer, "No answer yet.");
    ImGui::Spacing();

    ImGui::TextUnformatted("Evidence");
    TextWrappedOrMuted(last_response_.evidence, "No evidence returned.");
    ImGui::Spacing();

    ImGui::TextUnformatted("Unknowns");
    TextWrappedOrMuted(last_response_.unknowns, "No unknowns reported.");
    ImGui::Spacing();

    ImGui::TextUnformatted("Unsupported or not implemented");
    TextWrappedOrMuted(last_response_.unsupported_or_not_implemented,
                       "No unsupported items reported.");

    if (show_citations_) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextUnformatted("Retrieval Hits");
        if (last_response_.retrieval_hits.empty()) {
            ImGui::TextDisabled("No retrieval hits returned.");
        } else {
            for (const auto& hit : last_response_.retrieval_hits) {
                ImGui::BulletText("#%d score=%.0f %s:%d-%d", hit.rank, hit.score,
                                  hit.citation.path.c_str(), hit.citation.line_start,
                                  hit.citation.line_end);
                ImGui::TextWrapped("%s", hit.snippet.c_str());
            }
        }

        ImGui::Spacing();
        ImGui::TextUnformatted("Citations");
        if (last_response_.citations.empty()) {
            ImGui::TextDisabled("No citations returned.");
        } else {
            for (const auto& citation : last_response_.citations) {
                ImGui::BulletText("%s:%d-%d", citation.path.c_str(), citation.line_start,
                                  citation.line_end);
            }
        }
    }
}

} // namespace cyxwiz::plugin::assistant

CYXWIZ_PLUGIN_ENTRY(cyxwiz::plugin::assistant::CyxWizAssistantPlugin)
