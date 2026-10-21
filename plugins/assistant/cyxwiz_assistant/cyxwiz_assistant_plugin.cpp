#include "cyxwiz_assistant_plugin.h"
#include "knowledge_pack_backend.h"

#include <filesystem>

namespace cyxwiz::plugin::assistant {

namespace {

constexpr const char* kDefaultRuntimeEndpoint = "http://127.0.0.1:8768/completion";
constexpr const char* kDefaultPackPath =
    "internal/repository-private/engineering/Data Studio/tofix42/knowledge_pack";

} // namespace

CyxWizAssistantPlugin::CyxWizAssistantPlugin()
    : backend_(std::make_unique<KnowledgePackBackend>(
          std::filesystem::path(kDefaultPackPath))) {}

std::filesystem::path CyxWizAssistantPlugin::ResolveDefaultPackPath(const PluginContext* context) {
    if (context) {
        const auto plugin_dir = context->GetPluginDir();
        if (!plugin_dir.empty()) {
            const auto repo_root = plugin_dir.parent_path().parent_path().parent_path();
            const auto repo_pack = repo_root / "internal" /
                "repository-private" / "engineering" / "Data Studio" /
                "tofix42" / "knowledge_pack";
            if (std::filesystem::exists(repo_pack)) {
                return repo_pack;
            }

            const auto local_pack = plugin_dir / "knowledge_pack";
            if (std::filesystem::exists(local_pack)) {
                return local_pack;
            }
        }
    }

    const auto cwd_pack = std::filesystem::current_path() / "internal" /
        "repository-private" / "engineering" / "Data Studio" /
        "tofix42" / "knowledge_pack";
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
    if (std::string(name) == "IAssistantProvider") {
        return static_cast<IAssistantProvider*>(this);
    }
    return nullptr;
}

bool CyxWizAssistantPlugin::HasSelectedNodeContext() const {
    return context_ &&
        !context_->GetAssistantContextSnapshot().selected_node_id.empty();
}

bool CyxWizAssistantPlugin::HasSelectedTraceContext() const {
    return context_ &&
        !context_->GetAssistantContextSnapshot().selected_trace_id.empty();
}

bool CyxWizAssistantPlugin::HasTrainingTerminalContext() const {
    if (!context_) return false;
    const auto snapshot = context_->GetAssistantContextSnapshot();
    return snapshot.training_context_json.find("\"terminal_event\"") !=
        std::string::npos;
}

AssistantRequest CyxWizAssistantPlugin::BuildCommandRequest(
    const AssistantCommandRequest& command) const {
    AssistantRequest request;
    request.command_name = command.command_name;
    request.user_text = command.user_text;
    request.engine_version = "unknown";
    request.build_id = "unknown";
    request.selected_panel = "Console.AgentLlm";
    request.retrieval_only = command.command_name == "find_source";
    request.top_k = 3;
    request.timeout_seconds = 120;
    request.runtime_endpoint = kDefaultRuntimeEndpoint;

    if (context_) {
        const auto snapshot = context_->GetAssistantContextSnapshot();
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

    return request;
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

    const auto request = BuildCommandRequest(command);
    const auto response = backend_->Run(request);
    result.success = response.success;
    result.retrieval_requested = true;
    result.retrieval_ok = response.retrieval_ok;
    result.runtime_requested = !request.retrieval_only;
    result.runtime_ok = response.runtime_ok;
    result.backend_state = response.error_code;
    result.sections = {
        {"Answer", response.answer},
        {"Evidence", response.evidence},
        {"Unknowns", response.unknowns},
        {"Unsupported or not implemented",
         response.unsupported_or_not_implemented},
    };
    result.sources.reserve(response.retrieval_hits.size());
    for (const auto& hit : response.retrieval_hits) {
        result.sources.push_back({
            hit.rank,
            hit.score,
            hit.citation.path,
            hit.citation.line_start,
            hit.citation.line_end,
            hit.citation.title,
            hit.citation.source_type,
            hit.snippet,
        });
    }
    if (!response.success && !response.error_message.empty()) {
        result.error = response.error_message;
    }
    return result;
}

} // namespace cyxwiz::plugin::assistant

CYXWIZ_PLUGIN_ENTRY(cyxwiz::plugin::assistant::CyxWizAssistantPlugin)
