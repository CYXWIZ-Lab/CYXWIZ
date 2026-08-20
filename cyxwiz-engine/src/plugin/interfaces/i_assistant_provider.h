#pragma once

#include <string>
#include <vector>

namespace cyxwiz::plugin {

struct AssistantCommandRequest {
    std::string command_name;
    std::string user_text;
};

struct AssistantCommandSection {
    std::string title;
    std::string content;
};

struct AssistantCommandSource {
    int rank = 0;
    double score = 0.0;
    std::string path;
    int line_start = 0;
    int line_end = 0;
    std::string title;
    std::string source_type;
    std::string snippet;
};

struct AssistantCommandResponse {
    bool handled = false;
    bool success = false;
    bool retrieval_requested = false;
    bool retrieval_ok = false;
    bool runtime_requested = false;
    bool runtime_ok = false;
    std::string backend_state;
    std::vector<AssistantCommandSection> sections;
    std::vector<AssistantCommandSource> sources;
    std::string output;
    std::string error;
};

class IAssistantProvider {
public:
    virtual ~IAssistantProvider() = default;

    virtual AssistantCommandResponse RunAssistantCommand(
        const AssistantCommandRequest& request) = 0;
};

} // namespace cyxwiz::plugin
