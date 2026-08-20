#pragma once

#include "runtime_log_filter.h"
#include "runtime_truth_query.h"

#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cyxwiz {

enum class RuntimeConsoleOutputLevel : uint8_t {
    Info,
    Warning,
    Error,
    Success,
    Debug
};

struct RuntimeConsoleOutputLine {
    RuntimeConsoleOutputLevel level = RuntimeConsoleOutputLevel::Info;
    std::string text;
};

enum class RuntimeConsoleAction : uint8_t {
    None,
    Clear,
    ExecutePip
};

struct RuntimeConsoleCommandResult {
    bool success = true;
    RuntimeConsoleAction action = RuntimeConsoleAction::None;
    std::vector<std::string> action_arguments;
    std::vector<RuntimeConsoleOutputLine> lines;
};

struct RuntimeConsoleCommandDescriptor {
    std::string_view name;
    std::string_view usage;
    std::string_view description;
    std::string_view detailed_help;
};

class RuntimeConsoleCommandService {
public:
    static constexpr size_t kDefaultResultLimit = 100;
    static constexpr size_t kMaximumResultLimit = 1000;
    static constexpr size_t kDefaultBackendSupportLimit = 32;
    static constexpr size_t kMaximumBackendSupportLimit = 100;
    static constexpr size_t kCommandHistoryCapacity = 100;

    explicit RuntimeConsoleCommandService(
        const RuntimeLogStore& store,
        RuntimeTruthQueryProvider* truth_provider = nullptr)
        : store_(store), truth_provider_(truth_provider) {}

    RuntimeConsoleCommandResult Execute(std::string_view command);
    static const std::vector<RuntimeConsoleCommandDescriptor>& Descriptors();
    std::optional<std::string> PreviousCommand();
    std::optional<std::string> NextCommand();
    size_t CommandHistorySize() const { return command_history_.size(); }

    const std::string& ActiveFilterExpression() const {
        return active_filter_expression_;
    }

private:
    using CommandHandler = RuntimeConsoleCommandResult (
        RuntimeConsoleCommandService::*)(std::string_view);

    struct CommandRegistration {
        RuntimeConsoleCommandDescriptor descriptor;
        CommandHandler handler = nullptr;
        bool visible_in_help = true;
    };

    static const std::vector<CommandRegistration>& Registry();
    RuntimeConsoleCommandResult ExecuteHelp(std::string_view command);
    RuntimeConsoleCommandResult ExecuteClear(std::string_view command);
    RuntimeConsoleCommandResult ExecuteTest(std::string_view command);
    RuntimeConsoleCommandResult ExecutePip(std::string_view command);
    RuntimeConsoleCommandResult ExecuteShow(std::string_view command);
    RuntimeConsoleCommandResult ExecuteFilter(std::string_view command);
    RuntimeConsoleCommandResult ShowLogs(std::string_view arguments);
    RuntimeConsoleCommandResult ShowCode(std::string_view arguments);
    RuntimeConsoleCommandResult ShowCodes(std::string_view arguments);
    RuntimeConsoleCommandResult ShowTraining(std::string_view arguments);
    RuntimeConsoleCommandResult ShowDevice(std::string_view arguments);
    RuntimeConsoleCommandResult ShowBackend(std::string_view arguments);
    RuntimeConsoleCommandResult ShowRun(std::string_view arguments);
    RuntimeConsoleCommandResult ShowMaterialization(std::string_view arguments);
    void RecordCommand(std::string_view command);

    const RuntimeLogStore& store_;
    RuntimeTruthQueryProvider* truth_provider_ = nullptr;
    std::string active_filter_expression_;
    std::map<std::string, std::string, std::less<>> saved_filters_;
    std::deque<std::string> command_history_;
    size_t command_history_cursor_ = 0;
};

} // namespace cyxwiz
