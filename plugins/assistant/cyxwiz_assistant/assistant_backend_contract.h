#pragma once

#include <string>
#include <vector>

namespace cyxwiz::plugin::assistant {

struct AssistantRequest {
    std::string command_name;
    std::string user_text;
    std::string engine_version;
    std::string build_id;
    std::string workspace_root;
    std::string active_graph_path;
    std::string selected_run_id;
    std::string selected_node_id;
    std::string selected_trace_id;
    std::string selected_panel;
    std::string debugger_context_json;
    std::string training_context_json;
    bool retrieval_only = false;
    int top_k = 3;
    int timeout_seconds = 120;
    std::string runtime_endpoint;
};

struct AssistantCitation {
    std::string path;
    int line_start = 0;
    int line_end = 0;
    std::string title;
    std::string source_type;
};

struct AssistantRetrievalHit {
    int rank = 0;
    double score = 0.0;
    AssistantCitation citation;
    std::string snippet;
};

struct AssistantResponse {
    bool success = false;
    bool retrieval_ok = false;
    bool runtime_ok = false;
    bool parsed = false;
    std::string answer;
    std::string evidence;
    std::string unknowns;
    std::string unsupported_or_not_implemented;
    std::vector<AssistantCitation> citations;
    std::vector<AssistantRetrievalHit> retrieval_hits;
    std::string raw_output;
    std::string error_code;
    std::string error_message;
};

class IAssistantBackend {
public:
    virtual ~IAssistantBackend() = default;
    virtual AssistantResponse Run(const AssistantRequest& request) = 0;
};

class PlaceholderAssistantBackend final : public IAssistantBackend {
public:
    AssistantResponse Run(const AssistantRequest& request) override {
        AssistantResponse response;

        if (request.user_text.empty() && request.command_name == "ask") {
            response.error_code = "invalid_request";
            response.error_message = "Enter a question before asking the assistant.";
            return response;
        }

        response.error_code = "knowledge_pack_missing";
        response.error_message =
            "Assistant backend scaffold is loaded, but no knowledge pack/runtime is wired yet.";
        response.unknowns =
            "The Agent LLM Console session is present, but retrieval and runtime calls are not connected.";
        response.unsupported_or_not_implemented =
            "Knowledge-pack loading, retrieval, model runtime calls, and Console routing are not implemented in this plugin scaffold.";
        return response;
    }
};

} // namespace cyxwiz::plugin::assistant
