#pragma once

#include <string>

namespace cyxwiz::plugin {

struct AssistantCommandRequest {
    std::string command_name;
    std::string user_text;
};

struct AssistantCommandResponse {
    bool handled = false;
    bool success = false;
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
