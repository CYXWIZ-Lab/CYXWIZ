#include "runtime_console_commands.h"

#include "error_codes.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cctype>
#include <ctime>
#include <iomanip>
#include <optional>
#include <set>
#include <sstream>
#include <utility>

namespace cyxwiz {
namespace {

struct CommandToken {
    std::string text;
    size_t position = 0;
    size_t end = 0;
};

struct CommandTokens {
    std::vector<CommandToken> values;
    std::optional<RuntimeLogFilterParseError> error;
};

std::string_view Trim(std::string_view value) {
    while (!value.empty() &&
           std::isspace(static_cast<unsigned char>(value.front()))) {
        value.remove_prefix(1);
    }
    while (!value.empty() &&
           std::isspace(static_cast<unsigned char>(value.back()))) {
        value.remove_suffix(1);
    }
    return value;
}

std::string LowerAscii(std::string_view value) {
    std::string lowered(value);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return lowered;
}

CommandTokens TokenizeCommand(std::string_view input) {
    CommandTokens result;
    size_t position = 0;
    while (position < input.size()) {
        while (position < input.size() &&
               std::isspace(static_cast<unsigned char>(input[position]))) {
            ++position;
        }
        if (position == input.size()) break;

        CommandToken token;
        token.position = position;
        if (input[position] == '"') {
            ++position;
            bool closed = false;
            while (position < input.size()) {
                const char current = input[position++];
                if (current == '"') {
                    closed = true;
                    break;
                }
                if (current != '\\') {
                    token.text.push_back(current);
                    continue;
                }
                if (position == input.size()) {
                    result.error = RuntimeLogFilterParseError{
                        token.position, "unterminated escape sequence"};
                    return result;
                }
                const char escaped = input[position++];
                switch (escaped) {
                    case '"': token.text.push_back('"'); break;
                    case '\\': token.text.push_back('\\'); break;
                    case 'n': token.text.push_back('\n'); break;
                    case 't': token.text.push_back('\t'); break;
                    default:
                        token.text.push_back('\\');
                        token.text.push_back(escaped);
                        break;
                }
            }
            if (!closed) {
                result.error = RuntimeLogFilterParseError{
                    token.position, "unterminated quoted argument"};
                return result;
            }
            if (position < input.size() &&
                !std::isspace(static_cast<unsigned char>(input[position]))) {
                result.error = RuntimeLogFilterParseError{
                    position, "expected whitespace after quoted argument"};
                return result;
            }
        } else {
            const size_t start = position;
            while (position < input.size() &&
                   !std::isspace(static_cast<unsigned char>(input[position]))) {
                ++position;
            }
            token.text.assign(input.substr(start, position - start));
        }
        token.end = position;
        result.values.push_back(std::move(token));
    }
    return result;
}

std::string_view SliceFrom(std::string_view input, const CommandToken& token) {
    return Trim(input.substr(token.position));
}

RuntimeConsoleCommandResult ErrorResult(std::string message) {
    RuntimeConsoleCommandResult result;
    result.success = false;
    result.lines.push_back(
        {RuntimeConsoleOutputLevel::Error, std::move(message)});
    return result;
}

void AddLine(RuntimeConsoleCommandResult& result,
             RuntimeConsoleOutputLevel level, std::string text) {
    result.lines.push_back({level, std::move(text)});
}

void AddHelpLines(RuntimeConsoleCommandResult& result,
                  std::string_view text) {
    while (!text.empty()) {
        const size_t newline = text.find('\n');
        AddLine(result, RuntimeConsoleOutputLevel::Info,
                std::string(text.substr(0, newline)));
        if (newline == std::string_view::npos) break;
        text.remove_prefix(newline + 1);
    }
}

std::optional<size_t> ParseLimit(std::string_view value) {
    size_t parsed = 0;
    const auto conversion = std::from_chars(
        value.data(), value.data() + value.size(), parsed, 10);
    if (conversion.ec != std::errc{} ||
        conversion.ptr != value.data() + value.size() || parsed == 0 ||
        parsed > RuntimeConsoleCommandService::kMaximumResultLimit) {
        return std::nullopt;
    }
    return parsed;
}

bool IsValidFilterName(std::string_view name) {
    return !name.empty() && name.size() <= 64 &&
           std::all_of(name.begin(), name.end(), [](unsigned char c) {
               return std::isalnum(c) != 0 || c == '_' || c == '-';
           });
}

std::string EscapeFilterString(std::string_view value) {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const char current : value) {
        switch (current) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\n': escaped += "\\n"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped.push_back(current); break;
        }
    }
    escaped.push_back('"');
    return escaped;
}

std::string CombineFilters(std::string_view active,
                           std::string_view requested) {
    active = Trim(active);
    requested = Trim(requested);
    if (active.empty()) return std::string(requested);
    if (requested.empty()) return std::string(active);
    return "(" + std::string(active) + ") and (" +
           std::string(requested) + ")";
}

std::string FormatUtc(
    const std::chrono::system_clock::time_point& timestamp) {
    const auto time = std::chrono::system_clock::to_time_t(timestamp);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &time);
#else
    gmtime_r(&time, &utc);
#endif
    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            timestamp.time_since_epoch()) % 1000;
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << '.'
           << std::setfill('0') << std::setw(3) << milliseconds.count() << 'Z';
    return output.str();
}

std::string JoinedCodes(const RuntimeLogEvent& event) {
    std::ostringstream output;
    bool has_code = false;
    if (!event.primary_error_code.empty()) {
        output << event.primary_error_code;
        has_code = true;
    }
    for (const auto& code : event.issue_codes) {
        if (has_code) output << ',';
        output << code;
        has_code = true;
    }
    return output.str();
}

bool HasDiagnosticCode(const RuntimeLogEvent& event) {
    return !event.primary_error_code.empty() || !event.issue_codes.empty();
}

std::string FormatEvent(const RuntimeLogEvent& event) {
    std::ostringstream output;
    output << '#' << event.sequence << ' ' << FormatUtc(event.timestamp_utc)
           << " level=" << RuntimeLogLevelName(event.level)
           << " category=" << event.category;
    if (!event.source.empty()) output << " source=" << event.source;
    const auto codes = JoinedCodes(event);
    if (!codes.empty()) output << " code=" << codes;
    if (!event.run_id.empty()) output << " run=" << event.run_id;
    if (event.task_id != 0) output << " task=" << event.task_id;
    if (event.node_id >= 0) output << " node=" << event.node_id;
    if (!event.backend.empty()) output << " backend=" << event.backend;
    if (event.device_id >= 0) output << " device_id=" << event.device_id;
    if (!event.message.empty()) output << " | " << event.message;
    return output.str();
}

RuntimeConsoleCommandResult QueryEvents(
    const RuntimeLogStore& store, std::string_view active_filter,
    std::string_view requested_filter, size_t limit, bool require_code) {
    const auto expression = CombineFilters(active_filter, requested_filter);
    std::optional<RuntimeLogFilter> filter;
    if (!expression.empty()) {
        auto parsed = ParseRuntimeLogFilter(expression);
        if (!parsed.Ok()) {
            const auto& error = *parsed.error;
            return ErrorResult("Filter error at " +
                               std::to_string(error.position) + ": " +
                               error.message);
        }
        filter.emplace(std::move(*parsed.filter));
    }

    RuntimeLogQueryRequest request;
    request.filter = filter ? &*filter : nullptr;
    request.limit = store.GetStats().capacity;
    const RuntimeLogQueryService service(store);
    auto query = service.Query(request);

    std::vector<const RuntimeLogEvent*> matches;
    matches.reserve(query.events.size());
    for (const auto& event : query.events) {
        if (!require_code || HasDiagnosticCode(event)) {
            matches.push_back(&event);
        }
    }

    const size_t first = matches.size() > limit ? matches.size() - limit : 0;
    RuntimeConsoleCommandResult result;
    std::ostringstream summary;
    summary << "Logs: matched=" << matches.size()
            << " showing=" << (matches.size() - first)
            << " scanned=" << query.scanned_count
            << " retained=" << query.store_stats.size
            << " evicted=" << query.store_stats.evicted_count;
    AddLine(result, RuntimeConsoleOutputLevel::Info, summary.str());
    if (first > 0) {
        AddLine(result, RuntimeConsoleOutputLevel::Warning,
                "Result limit omitted " + std::to_string(first) +
                    " older matching event(s)");
    }
    for (size_t index = first; index < matches.size(); ++index) {
        AddLine(result, RuntimeConsoleOutputLevel::Info,
                FormatEvent(*matches[index]));
    }
    return result;
}

std::string FormatByteCount(uint64_t bytes) {
    std::ostringstream output;
    output << bytes << " B";
    if (bytes >= 1024) {
        output << " (" << std::fixed << std::setprecision(2)
               << static_cast<double>(bytes) / (1024.0 * 1024.0) << " MiB)";
    }
    return output.str();
}

std::string RecordedOr(std::string_view value) {
    return value.empty() ? "not_recorded" : std::string(value);
}

void AddTrainingSummary(RuntimeConsoleCommandResult& result,
                        const RuntimeTrainingTruth& truth) {
    AddLine(result, RuntimeConsoleOutputLevel::Info,
            "Training: source=" + RecordedOr(truth.source) +
                " active=" + (truth.active ? "true" : "false") +
                " run=" + RecordedOr(truth.run_id) +
                " status=" + RecordedOr(truth.status));

    std::ostringstream progress;
    progress << "Progress: epoch=" << truth.epoch;
    if (truth.total_epochs > 0) progress << '/' << truth.total_epochs;
    progress << " batch=" << truth.batch;
    if (truth.total_batches > 0) progress << '/' << truth.total_batches;
    progress << " stage=" << RecordedOr(truth.latest_stage);
    AddLine(result, RuntimeConsoleOutputLevel::Info, progress.str());

    std::ostringstream metrics;
    metrics << std::fixed << std::setprecision(6)
            << "Metrics: loss=" << truth.loss
            << " accuracy=" << truth.accuracy;
    AddLine(result, RuntimeConsoleOutputLevel::Info, metrics.str());

    AddLine(result, RuntimeConsoleOutputLevel::Info,
            "Execution: requested=" + RecordedOr(truth.requested_backend) +
                ':' + std::to_string(truth.requested_device_id) +
                " effective=" + RecordedOr(truth.effective_backend) + ':' +
                std::to_string(truth.effective_device_id) + " device='" +
                RecordedOr(truth.effective_device_name) + "' context=" +
                RecordedOr(truth.execution_context_id));
    AddLine(result, RuntimeConsoleOutputLevel::Info,
            "Policy: fallback=" + RecordedOr(truth.fallback_policy) +
                " residency=" + RecordedOr(truth.residency_verdict) +
                " reason=" + RecordedOr(truth.residency_reason));
    AddLine(result, RuntimeConsoleOutputLevel::Info,
            "Evidence: native_cpu_fallback=" +
                std::to_string(truth.native_cpu_fallback_count) +
                " host_sync=" + std::to_string(truth.host_sync_count) +
                " host_sync_bytes=" + std::to_string(truth.host_sync_bytes) +
                " transfer=" + std::to_string(truth.transfer_event_count) +
                " synchronization=" +
                std::to_string(truth.synchronization_event_count));
}

std::string FormatTrainingEvent(const RuntimeTrainingEventTruth& event) {
    std::ostringstream output;
    output << RecordedOr(event.timestamp)
           << " stage=" << RecordedOr(event.stage)
           << " status=" << RecordedOr(event.status)
           << " epoch=" << event.epoch << " batch=" << event.batch;
    if (event.total_batches > 0) output << '/' << event.total_batches;
    if (event.native_cpu_fallback) {
        output << " fallback_operation="
               << RecordedOr(event.fallback_operation)
               << " fallback_reason=" << RecordedOr(event.fallback_reason)
               << " fallback_policy=" << RecordedOr(event.fallback_policy);
    }
    if (event.host_sync_bytes > 0 || !event.host_sync_reason.empty()) {
        output << " host_sync_bytes=" << event.host_sync_bytes
               << " host_sync_category="
               << RecordedOr(event.host_sync_category)
               << " host_sync_reason=" << RecordedOr(event.host_sync_reason)
               << " host_sync_operation="
               << RecordedOr(event.host_sync_operation);
    }
    if (event.task_id != 0) {
        output << " task=" << event.task_id
               << " task_stage=" << RecordedOr(event.task_stage)
               << " progress=" << std::fixed << std::setprecision(3)
               << event.task_progress;
    }
    if (event.node_id >= 0) output << " node=" << event.node_id;
    if (!event.message.empty()) output << " | " << event.message;
    return output.str();
}

std::string FormatRunEvent(const RuntimeRunEventTruth& event) {
    std::ostringstream output;
    output << "source=" << RecordedOr(event.source)
           << " timestamp=" << RecordedOr(event.timestamp)
           << " stage=" << RecordedOr(event.stage)
           << " status=" << RecordedOr(event.status);
    if (event.node_id >= 0) output << " node=" << event.node_id;
    if (!event.message.empty()) output << " | " << event.message;
    return output.str();
}

std::string FormatRunIssue(const RuntimeRunIssueTruth& issue,
                           std::string_view run_id) {
    std::ostringstream output;
    output << "source=" << RecordedOr(issue.source)
           << " level=" << RecordedOr(issue.level)
           << " code=" << RecordedOr(issue.code)
           << " run=" << RecordedOr(run_id);
    if (issue.node_id >= 0) output << " node=" << issue.node_id;
    if (!issue.node_name.empty()) output << " node_name='" << issue.node_name << "'";
    if (!issue.message.empty()) output << " | " << issue.message;
    return output.str();
}

void AddRunSummary(RuntimeConsoleCommandResult& result,
                   const RuntimeRunTruth& run) {
    AddLine(result, RuntimeConsoleOutputLevel::Info,
            "Run: source=" + RecordedOr(run.source) +
                " run=" + RecordedOr(run.run_id) +
                " status=" + RecordedOr(run.status));
    if (!run.debug_run_id.empty() || !run.training_run_id.empty()) {
        AddLine(result, RuntimeConsoleOutputLevel::Info,
                "Identity: debug_run=" + RecordedOr(run.debug_run_id) +
                    " training_run=" + RecordedOr(run.training_run_id));
    }
    AddLine(result, RuntimeConsoleOutputLevel::Info,
            "Execution: requested=" + RecordedOr(run.requested_backend) +
                ':' + std::to_string(run.requested_device_id) +
                " effective=" + RecordedOr(run.effective_backend) + ':' +
                std::to_string(run.effective_device_id) + " device='" +
                RecordedOr(run.effective_device_name) + "' context=" +
                RecordedOr(run.execution_context_id));
    AddLine(result, RuntimeConsoleOutputLevel::Info,
            "Evidence: issues=" + std::to_string(run.issue_count) +
                " traces=" + std::to_string(run.trace_count) +
                " events=" + std::to_string(run.event_count) +
                " fallback=" +
                std::to_string(run.native_cpu_fallback_count) +
                " transfer=" + std::to_string(run.transfer_event_count) +
                " synchronization=" +
                std::to_string(run.synchronization_event_count));
    AddLine(result, RuntimeConsoleOutputLevel::Info,
            "Placement: fingerprint=" +
                RecordedOr(run.placement_fingerprint) + " residency=" +
                RecordedOr(run.residency_verdict));
    if (!run.summary.empty()) {
        AddLine(result, RuntimeConsoleOutputLevel::Info,
                "Summary: " + run.summary);
    }
}

RuntimeConsoleCommandResult Usage(std::string_view usage) {
    return ErrorResult("Usage: " + std::string(usage));
}

} // namespace

const std::vector<RuntimeConsoleCommandDescriptor>&
RuntimeConsoleCommandService::Descriptors() {
    static const std::vector<RuntimeConsoleCommandDescriptor> descriptors = [] {
        std::vector<RuntimeConsoleCommandDescriptor> visible;
        for (const auto& registration : Registry()) {
            if (registration.visible_in_help) {
                visible.push_back(registration.descriptor);
            }
        }
        return visible;
    }();
    return descriptors;
}

const std::vector<RuntimeConsoleCommandService::CommandRegistration>&
RuntimeConsoleCommandService::Registry() {
    static const std::vector<CommandRegistration> registrations{
        {{"help", "help [command]", "Show command help",
          "Usage:\n"
          "  help\n"
          "  help <command>\n"
          "Examples:\n"
          "  help show\n"
          "  help filter"},
         &RuntimeConsoleCommandService::ExecuteHelp, true},
        {{"clear", "clear", "Clear this Console view",
          "Clears only the visible Console transcript. Runtime event "
          "ingestion continues from a new view watermark.\n"
          "Example:\n"
          "  clear"},
         &RuntimeConsoleCommandService::ExecuteClear, true},
        {{"test", "test", "Emit one line at each Console severity",
          "Emits info, warning, error, and success lines to verify Console "
          "presentation.\n"
          "Example:\n"
          "  test"},
         &RuntimeConsoleCommandService::ExecuteTest, true},
        {{"pip", "pip <arguments>",
          "Run project-environment pip asynchronously",
          "Runs pip only from the active project's virtual environment. "
          "Output streams to the Console and the task can be cancelled.\n"
          "Examples:\n"
          "  pip list\n"
          "  pip show numpy\n"
          "  pip install numpy pandas"},
         &RuntimeConsoleCommandService::ExecutePip, true},
        {{"pip3", "pip3 <arguments>", "Alias for pip",
          "Alias for the project-environment pip command.\n"
          "Example:\n"
          "  pip3 list"},
         &RuntimeConsoleCommandService::ExecutePip, false},
        {{"show", "show logs|errors|code|codes|training|device|run|materialization ...",
          "Query bounded runtime diagnostics",
          "Log queries are read-only, bounded to 1-1000 displayed rows, and "
          "show the newest matching events in sequence order.\n"
          "Forms:\n"
          "  show logs last <n>\n"
          "  show logs errors\n"
          "  show logs warnings\n"
          "  show logs where <filter>\n"
          "  show logs grep <text>\n"
          "  show logs code <CW-X-NNNN>\n"
          "  show logs codes <CW-X-*>\n"
          "  show errors last <n>\n"
          "  show code <CW-X-NNNN>\n"
          "  show codes family <CW-X-*>\n"
          "  show codes last <n>\n"
          "  show codes where <filter>\n"
          "  show training current|last|trace|fallback|host-sync|placement|materialization\n"
          "  show device active|available|queued|backends|oneapi\n"
          "  show run current\n"
          "  show run <run_id> summary|events|codes|host-sync|fallback\n"
          "  show run <run_id> code <CW-X-NNNN>\n"
          "  show run <run_id> codes <CW-X-*>\n"
          "  show materialization last\n"
          "Examples:\n"
          "  show logs last 50\n"
          "  show logs where category=training and level>=warn\n"
          "  show logs grep \"native CPU fallback\"\n"
          "  show code CW-C-0101\n"
          "  show codes family CW-G-*\n"
          "  show training current\n"
          "  show device active\n"
          "  show run train-1786337176120 fallback\n"
          "  show materialization last"},
         &RuntimeConsoleCommandService::ExecuteShow, true},
        {{"filter", "filter set|clear|save|use ...",
          "Manage the session runtime-log filter",
          "The active session filter is combined with subsequent show-log "
          "queries. Saved filters remain in memory for this Console session.\n"
          "Forms:\n"
          "  filter set <expression>\n"
          "  filter clear\n"
          "  filter save <name> <expression>\n"
          "  filter use <name>\n"
          "Examples:\n"
          "  filter set category=training and level>=info\n"
          "  filter set error_code matches CW-G-*\n"
          "  filter save cuda_errors backend=arrayfire_cuda and level>=error\n"
          "  filter use cuda_errors"},
         &RuntimeConsoleCommandService::ExecuteFilter, true},
    };
    return registrations;
}

RuntimeConsoleCommandResult RuntimeConsoleCommandService::Execute(
    std::string_view command) {
    command = Trim(command);
    if (command.empty()) return ErrorResult("Command is empty");
    RecordCommand(command);

    const auto tokens = TokenizeCommand(command);
    if (tokens.error) {
        return ErrorResult("Command error at " +
                           std::to_string(tokens.error->position) + ": " +
                           tokens.error->message);
    }
    const auto name = LowerAscii(tokens.values.front().text);
    for (const auto& registration : Registry()) {
        if (name == registration.descriptor.name) {
            return (this->*registration.handler)(command);
        }
    }

    auto result = ErrorResult("Unknown command: " + tokens.values.front().text);
    AddLine(result, RuntimeConsoleOutputLevel::Info,
            "Type 'help' for available commands");
    return result;
}

void RuntimeConsoleCommandService::RecordCommand(std::string_view command) {
    if (command_history_.empty() || command_history_.back() != command) {
        command_history_.emplace_back(command);
        if (command_history_.size() > kCommandHistoryCapacity) {
            command_history_.pop_front();
        }
    }
    command_history_cursor_ = command_history_.size();
}

std::optional<std::string>
RuntimeConsoleCommandService::PreviousCommand() {
    if (command_history_.empty()) return std::nullopt;
    if (command_history_cursor_ > 0) --command_history_cursor_;
    return command_history_[command_history_cursor_];
}

std::optional<std::string> RuntimeConsoleCommandService::NextCommand() {
    if (command_history_.empty()) return std::nullopt;
    if (command_history_cursor_ + 1 < command_history_.size()) {
        ++command_history_cursor_;
        return command_history_[command_history_cursor_];
    }
    command_history_cursor_ = command_history_.size();
    return std::string{};
}

RuntimeConsoleCommandResult RuntimeConsoleCommandService::ExecuteHelp(
    std::string_view command) {
    const auto tokens = TokenizeCommand(command);
    if (tokens.error || tokens.values.size() > 2) {
        return Usage("help [command]");
    }

    if (tokens.values.size() == 2) {
        const auto topic = LowerAscii(tokens.values[1].text);
        for (const auto& registration : Registry()) {
            if (topic != registration.descriptor.name) continue;
            RuntimeConsoleCommandResult result;
            AddLine(result, RuntimeConsoleOutputLevel::Info,
                    "=== Help: " + std::string(registration.descriptor.name) +
                        " ===");
            AddLine(result, RuntimeConsoleOutputLevel::Info,
                    "Usage: " +
                        std::string(registration.descriptor.usage));
            AddLine(result, RuntimeConsoleOutputLevel::Info,
                    std::string(registration.descriptor.description));
            AddHelpLines(result, registration.descriptor.detailed_help);
            if (topic == "filter") {
                AddLine(result, RuntimeConsoleOutputLevel::Info,
                        "Expression syntax:");
                AddHelpLines(result, RuntimeLogFilterHelpText());
            }
            return result;
        }
        auto result = ErrorResult("Unknown help topic: " +
                                  tokens.values[1].text);
        AddLine(result, RuntimeConsoleOutputLevel::Info,
                "Use 'help' to list available commands");
        return result;
    }

    RuntimeConsoleCommandResult result;
    AddLine(result, RuntimeConsoleOutputLevel::Info,
            "=== CyxWiz Console Help ===");
    for (const auto& descriptor : Descriptors()) {
        AddLine(result, RuntimeConsoleOutputLevel::Info,
                "  " + std::string(descriptor.usage) + " - " +
                    std::string(descriptor.description));
    }
    AddLine(result, RuntimeConsoleOutputLevel::Info,
            "Use 'help <command>' for forms, behavior, and examples");
    return result;
}

RuntimeConsoleCommandResult RuntimeConsoleCommandService::ExecuteClear(
    std::string_view command) {
    const auto tokens = TokenizeCommand(command);
    if (tokens.error || tokens.values.size() != 1) return Usage("clear");
    RuntimeConsoleCommandResult result;
    result.action = RuntimeConsoleAction::Clear;
    return result;
}

RuntimeConsoleCommandResult RuntimeConsoleCommandService::ExecuteTest(
    std::string_view command) {
    const auto tokens = TokenizeCommand(command);
    if (tokens.error || tokens.values.size() != 1) return Usage("test");
    RuntimeConsoleCommandResult result;
    AddLine(result, RuntimeConsoleOutputLevel::Info,
            "This is an info message");
    AddLine(result, RuntimeConsoleOutputLevel::Warning,
            "This is a warning message");
    AddLine(result, RuntimeConsoleOutputLevel::Error,
            "This is an error message");
    AddLine(result, RuntimeConsoleOutputLevel::Success,
            "This is a success message");
    return result;
}

RuntimeConsoleCommandResult RuntimeConsoleCommandService::ExecutePip(
    std::string_view command) {
    const auto tokens = TokenizeCommand(command);
    if (tokens.error || tokens.values.empty()) return Usage("pip <arguments>");
    RuntimeConsoleCommandResult result;
    result.action = RuntimeConsoleAction::ExecutePip;
    for (size_t index = 1; index < tokens.values.size(); ++index) {
        result.action_arguments.push_back(tokens.values[index].text);
    }
    return result;
}

RuntimeConsoleCommandResult RuntimeConsoleCommandService::ExecuteShow(
    std::string_view command) {
    const auto tokens = TokenizeCommand(command);
    if (tokens.error || tokens.values.size() < 2) {
        return Usage("show logs|errors|code|codes|training|device|run|materialization ...");
    }
    const auto subject = LowerAscii(tokens.values[1].text);
    const auto arguments = tokens.values.size() > 2
                               ? SliceFrom(command, tokens.values[2])
                               : std::string_view{};
    if (subject == "logs") return ShowLogs(arguments);
    if (subject == "code") return ShowCode(arguments);
    if (subject == "codes") return ShowCodes(arguments);
    if (subject == "training") return ShowTraining(arguments);
    if (subject == "device") return ShowDevice(arguments);
    if (subject == "run") return ShowRun(arguments);
    if (subject == "materialization") {
        return ShowMaterialization(arguments);
    }
    if (subject == "errors") {
        size_t limit = kDefaultResultLimit;
        if (!arguments.empty()) {
            const auto parsed = TokenizeCommand(arguments);
            if (parsed.error || parsed.values.size() != 2 ||
                LowerAscii(parsed.values[0].text) != "last") {
                return Usage("show errors last <1-1000>");
            }
            const auto requested = ParseLimit(parsed.values[1].text);
            if (!requested) return Usage("show errors last <1-1000>");
            limit = *requested;
        }
        return QueryEvents(store_, active_filter_expression_,
                           "level>=error", limit, false);
    }
    return ErrorResult("Unknown show subject: " + tokens.values[1].text);
}

RuntimeConsoleCommandResult RuntimeConsoleCommandService::ShowLogs(
    std::string_view arguments) {
    arguments = Trim(arguments);
    if (arguments.empty()) {
        return QueryEvents(store_, active_filter_expression_, {},
                           kDefaultResultLimit, false);
    }
    const auto tokens = TokenizeCommand(arguments);
    if (tokens.error) {
        return ErrorResult("Command error at " +
                           std::to_string(tokens.error->position) + ": " +
                           tokens.error->message);
    }
    const auto mode = LowerAscii(tokens.values[0].text);
    if (mode == "last") {
        if (tokens.values.size() != 2) return Usage("show logs last <1-1000>");
        const auto limit = ParseLimit(tokens.values[1].text);
        if (!limit) return Usage("show logs last <1-1000>");
        return QueryEvents(store_, active_filter_expression_, {}, *limit,
                           false);
    }
    if (mode == "errors" || mode == "warnings") {
        if (tokens.values.size() != 1) {
            return Usage(mode == "errors" ? "show logs errors"
                                           : "show logs warnings");
        }
        return QueryEvents(store_, active_filter_expression_,
                           mode == "errors" ? "level>=error" : "level=warn",
                           kDefaultResultLimit, false);
    }
    if (mode == "where") {
        if (tokens.values.size() < 2) return Usage("show logs where <filter>");
        return QueryEvents(store_, active_filter_expression_,
                           SliceFrom(arguments, tokens.values[1]),
                           kDefaultResultLimit, false);
    }
    if (mode == "grep") {
        if (tokens.values.size() < 2) return Usage("show logs grep <text>");
        std::string text;
        for (size_t index = 1; index < tokens.values.size(); ++index) {
            if (!text.empty()) text.push_back(' ');
            text += tokens.values[index].text;
        }
        return QueryEvents(store_, active_filter_expression_,
                           "message contains " + EscapeFilterString(text),
                           kDefaultResultLimit, false);
    }
    if (mode == "code" || mode == "codes") {
        if (tokens.values.size() != 2) {
            return Usage(mode == "code" ? "show logs code <CW-X-NNNN>"
                                         : "show logs codes <CW-X-*>");
        }
        const auto normalized = mode == "code"
                                    ? NormalizeDiagnosticCode(tokens.values[1].text)
                                    : NormalizeDiagnosticFamily(tokens.values[1].text);
        if (!normalized) {
            return ErrorResult(mode == "code"
                                   ? "Invalid code; expected CW-C-0101"
                                   : "Invalid family; expected CW-C-*");
        }
        return QueryEvents(
            store_, active_filter_expression_,
            "error_code " + std::string(mode == "code" ? "= " : "matches ") +
                *normalized,
            kDefaultResultLimit, false);
    }
    return Usage("show logs last|errors|warnings|where|grep|code|codes ...");
}

RuntimeConsoleCommandResult RuntimeConsoleCommandService::ShowCode(
    std::string_view arguments) {
    const auto tokens = TokenizeCommand(arguments);
    if (tokens.error || tokens.values.size() != 1) {
        return Usage("show code <CW-X-NNNN>");
    }
    const auto code = NormalizeDiagnosticCode(tokens.values[0].text);
    if (!code) return ErrorResult("Invalid code; expected CW-C-0101");

    RuntimeConsoleCommandResult result;
    AddLine(result, RuntimeConsoleOutputLevel::Info, "Code: " + *code);
    const auto family_name = errors::DiagnosticFamilyName((*code)[3]);
    AddLine(result, RuntimeConsoleOutputLevel::Info,
            "Family: " + std::string(family_name) + " (CW-" + (*code)[3] +
                "-*)");
    if (const auto* descriptor = errors::FindDiagnosticCode(*code)) {
        AddLine(result, RuntimeConsoleOutputLevel::Info,
                "Symbolic name: " + std::string(descriptor->symbolic_name));
    } else {
        AddLine(result, RuntimeConsoleOutputLevel::Warning,
                "Symbolic name: unregistered");
    }

    auto recent = QueryEvents(store_, active_filter_expression_,
                              "error_code=" + *code, 20, false);
    result.success = recent.success;
    for (auto& line : recent.lines) {
        result.lines.push_back(std::move(line));
    }
    return result;
}

RuntimeConsoleCommandResult RuntimeConsoleCommandService::ShowCodes(
    std::string_view arguments) {
    const auto tokens = TokenizeCommand(arguments);
    if (tokens.error || tokens.values.empty()) {
        return Usage("show codes family|last|where ...");
    }
    const auto mode = LowerAscii(tokens.values[0].text);
    if (mode == "family") {
        if (tokens.values.size() != 2) {
            return Usage("show codes family <CW-X-*>");
        }
        const auto family = NormalizeDiagnosticFamily(tokens.values[1].text);
        if (!family) return ErrorResult("Invalid family; expected CW-C-*");

        RuntimeConsoleCommandResult result;
        AddLine(result, RuntimeConsoleOutputLevel::Info,
                "Registered codes in " + *family + ":");
        size_t count = 0;
        for (const auto& descriptor : errors::DiagnosticCodeCatalog) {
            if (std::string_view(descriptor.code).substr(0, 5) ==
                std::string_view(*family).substr(0, 5)) {
                AddLine(result, RuntimeConsoleOutputLevel::Info,
                        "  " + std::string(descriptor.code) + " " +
                            descriptor.symbolic_name);
                ++count;
            }
        }
        if (count == 0) {
            AddLine(result, RuntimeConsoleOutputLevel::Warning,
                    "No registered symbolic codes in this family");
        }
        return result;
    }
    if (mode == "last") {
        if (tokens.values.size() != 2) return Usage("show codes last <1-1000>");
        const auto limit = ParseLimit(tokens.values[1].text);
        if (!limit) return Usage("show codes last <1-1000>");
        return QueryEvents(store_, active_filter_expression_, {}, *limit, true);
    }
    if (mode == "where") {
        if (tokens.values.size() < 2) return Usage("show codes where <filter>");
        return QueryEvents(store_, active_filter_expression_,
                           SliceFrom(arguments, tokens.values[1]),
                           kDefaultResultLimit, true);
    }
    return Usage("show codes family|last|where ...");
}

RuntimeConsoleCommandResult RuntimeConsoleCommandService::ShowTraining(
    std::string_view arguments) {
    const auto tokens = TokenizeCommand(arguments);
    if (tokens.error || tokens.values.size() != 1) {
        return Usage(
            "show training current|last|trace|fallback|host-sync|placement|materialization");
    }
    if (!truth_provider_) {
        return ErrorResult("Runtime training truth provider is unavailable");
    }

    const auto mode = LowerAscii(tokens.values[0].text);
    RuntimeTrainingTruth truth;
    if (mode == "current") {
        truth = truth_provider_->GetCurrentTraining();
        if (!truth.available || !truth.active) {
            RuntimeConsoleCommandResult result;
            AddLine(result, RuntimeConsoleOutputLevel::Warning,
                    "No active training run; use 'show training last' for "
                    "the latest retained trace");
            return result;
        }
    } else if (mode == "last" || mode == "trace" || mode == "fallback" ||
               mode == "host-sync" || mode == "placement" ||
               mode == "materialization") {
        truth = truth_provider_->GetCurrentTraining();
        if (!truth.available || !truth.active) {
            truth = truth_provider_->GetLastTraining();
        }
        if (!truth.available) {
            RuntimeConsoleCommandResult result;
            AddLine(result, RuntimeConsoleOutputLevel::Warning,
                    "No training trace is available");
            return result;
        }
    } else {
        return Usage(
            "show training current|last|trace|fallback|host-sync|placement|materialization");
    }

    RuntimeConsoleCommandResult result;
    if (mode == "current" || mode == "last") {
        AddTrainingSummary(result, truth);
        return result;
    }
    if (mode == "trace") {
        AddTrainingSummary(result, truth);
        const size_t first = truth.recent_events.size() > 20
            ? truth.recent_events.size() - 20
            : 0;
        AddLine(result, RuntimeConsoleOutputLevel::Info,
                "Recent trace events: showing=" +
                    std::to_string(truth.recent_events.size() - first) +
                    " retained=" +
                    std::to_string(truth.recent_events.size()));
        for (size_t index = first; index < truth.recent_events.size(); ++index) {
            AddLine(result, RuntimeConsoleOutputLevel::Info,
                    FormatTrainingEvent(truth.recent_events[index]));
        }
        return result;
    }
    if (mode == "fallback") {
        AddLine(result, RuntimeConsoleOutputLevel::Info,
                "Native CPU fallback: run=" + RecordedOr(truth.run_id) +
                    " count=" +
                    std::to_string(truth.native_cpu_fallback_count));
        size_t shown = 0;
        for (const auto& event : truth.recent_events) {
            if (!event.native_cpu_fallback) continue;
            AddLine(result, RuntimeConsoleOutputLevel::Warning,
                    FormatTrainingEvent(event));
            ++shown;
        }
        if (truth.native_cpu_fallback_count > shown) {
            AddLine(result, RuntimeConsoleOutputLevel::Warning,
                    "Some fallback events are outside the retained trace "
                    "window");
        }
        return result;
    }
    if (mode == "host-sync") {
        AddLine(result, RuntimeConsoleOutputLevel::Info,
                "ArrayFire host sync: run=" + RecordedOr(truth.run_id) +
                    " count=" + std::to_string(truth.host_sync_count) +
                    " bytes=" + FormatByteCount(truth.host_sync_bytes));
        AddLine(result, RuntimeConsoleOutputLevel::Info,
                "Summary: " + RecordedOr(truth.host_sync_summary));
        for (const auto& event : truth.recent_events) {
            if (event.host_sync_bytes == 0 && event.host_sync_reason.empty()) {
                continue;
            }
            AddLine(result, RuntimeConsoleOutputLevel::Info,
                    FormatTrainingEvent(event));
        }
        return result;
    }
    if (mode == "placement") {
        AddLine(result, RuntimeConsoleOutputLevel::Info,
                "Placement: run=" + RecordedOr(truth.run_id) +
                    " fingerprint=" +
                    RecordedOr(truth.placement_fingerprint) + " entries=" +
                    std::to_string(truth.placement_entry_count));
        AddLine(result, RuntimeConsoleOutputLevel::Info,
                "Summary: " + RecordedOr(truth.placement_summary));
        AddLine(result, RuntimeConsoleOutputLevel::Info,
                "Residency: verdict=" +
                    RecordedOr(truth.residency_verdict) + " reason=" +
                    RecordedOr(truth.residency_reason));
        return result;
    }

    AddLine(result, RuntimeConsoleOutputLevel::Info,
            "Materialization: run=" + RecordedOr(truth.run_id) +
                " retained=" +
                std::to_string(truth.materialization_events.size()));
    for (const auto& event : truth.materialization_events) {
        AddLine(result, RuntimeConsoleOutputLevel::Info,
                FormatTrainingEvent(event));
    }
    return result;
}

RuntimeConsoleCommandResult RuntimeConsoleCommandService::ShowDevice(
    std::string_view arguments) {
    const auto tokens = TokenizeCommand(arguments);
    if (tokens.error || tokens.values.size() != 1) {
        return Usage("show device active|available|queued|backends|oneapi");
    }
    if (!truth_provider_) {
        return ErrorResult("Runtime device truth provider is unavailable");
    }
    const auto mode = LowerAscii(tokens.values[0].text);
    const bool include_inventory =
        mode == "available" || mode == "backends" || mode == "oneapi";
    if (mode != "active" && mode != "queued" && !include_inventory) {
        return Usage("show device active|available|queued|backends|oneapi");
    }
    const auto truth = truth_provider_->GetDeviceTruth(include_inventory);
    RuntimeConsoleCommandResult result;
    if (mode == "active") {
        if (!truth.active_available) {
            AddLine(result, RuntimeConsoleOutputLevel::Warning,
                    "Active device unavailable; source=" +
                        RecordedOr(truth.active_source));
            return result;
        }
        AddLine(result, RuntimeConsoleOutputLevel::Info,
                "Active device: source=" +
                    RecordedOr(truth.active_source) + " backend=" +
                    RecordedOr(truth.active_backend) + " device_id=" +
                    std::to_string(truth.active_device_id) + " name='" +
                    RecordedOr(truth.active_device_name) + "'");
        if (!truth.active_run_id.empty()) {
            AddLine(result, RuntimeConsoleOutputLevel::Info,
                    "Active run: " + truth.active_run_id);
        }
        return result;
    }
    if (mode == "queued") {
        if (truth.queued_available) {
            AddLine(result, RuntimeConsoleOutputLevel::Info,
                    "Queued device: source=next_run_selection backend=" +
                        truth.queued_backend + " device_id=" +
                        std::to_string(truth.queued_device_id));
        } else {
            AddLine(result, RuntimeConsoleOutputLevel::Info,
                    "Queued device: none");
        }
        AddLine(result, RuntimeConsoleOutputLevel::Info,
                "Next-run policy: source=" + truth.next_run_policy_source +
                    " value=" + truth.next_run_policy);
        return result;
    }

    AddLine(result, RuntimeConsoleOutputLevel::Info,
            "Device inventory: source=" +
                RecordedOr(truth.inventory_source) + " status=" +
                RecordedOr(truth.inventory_status) + " count=" +
                std::to_string(truth.available_devices.size()));
    if (mode == "backends") {
        std::set<std::string> backends;
        for (const auto& device : truth.available_devices) {
            backends.insert(device.backend);
        }
        for (const auto& backend : backends) {
            AddLine(result, RuntimeConsoleOutputLevel::Info,
                    "  " + backend);
        }
        return result;
    }
    size_t shown = 0;
    for (const auto& device : truth.available_devices) {
        if (mode == "oneapi" && device.backend != "arrayfire_oneapi") {
            continue;
        }
        AddLine(result, RuntimeConsoleOutputLevel::Info,
                "  backend=" + device.backend + " device_id=" +
                    std::to_string(device.device_id) + " name='" +
                    RecordedOr(device.name) + "' memory=" +
                    FormatByteCount(device.memory_total));
        ++shown;
    }
    if (mode == "oneapi" && shown == 0) {
        AddLine(result, RuntimeConsoleOutputLevel::Warning,
                "No ArrayFire oneAPI device was reported by the retained "
                "inventory");
    }
    return result;
}

RuntimeConsoleCommandResult RuntimeConsoleCommandService::ShowRun(
    std::string_view arguments) {
    const auto tokens = TokenizeCommand(arguments);
    if (tokens.error || tokens.values.empty() || tokens.values.size() > 3) {
        return Usage(
            "show run current | show run <run_id> summary|events|codes|host-sync|fallback | show run <run_id> code <CW-X-NNNN> | show run <run_id> codes <CW-X-*>");
    }
    if (!truth_provider_) {
        return ErrorResult("Runtime run truth provider is unavailable");
    }

    if (tokens.values.size() == 1 &&
        LowerAscii(tokens.values[0].text) == "current") {
        const auto run = truth_provider_->GetCurrentRun();
        if (!run.available) {
            RuntimeConsoleCommandResult result;
            AddLine(result, RuntimeConsoleOutputLevel::Warning,
                    "No active training run is available");
            return result;
        }
        RuntimeConsoleCommandResult result;
        AddRunSummary(result, run);
        return result;
    }
    if (tokens.values.size() < 2) {
        return Usage("show run current | show run <run_id> summary|events|codes|host-sync|fallback");
    }

    const std::string& requested_run_id = tokens.values[0].text;
    const auto mode = LowerAscii(tokens.values[1].text);
    const bool valid_two_token_mode =
        mode == "summary" || mode == "events" || mode == "codes" ||
        mode == "host-sync" || mode == "fallback";
    const bool valid_three_token_mode =
        mode == "code" || mode == "codes";
    if ((tokens.values.size() == 2 && !valid_two_token_mode) ||
        (tokens.values.size() == 3 && !valid_three_token_mode)) {
        return Usage(
            "show run <run_id> summary|events|codes|host-sync|fallback | show run <run_id> code <CW-X-NNNN> | show run <run_id> codes <CW-X-*>");
    }

    const auto run = truth_provider_->GetRun(requested_run_id);
    if (!run.available) {
        return ErrorResult("Run not found: " + requested_run_id);
    }
    if (mode == "summary") {
        RuntimeConsoleCommandResult result;
        AddRunSummary(result, run);
        return result;
    }

    const auto run_filter =
        "run_id=" + EscapeFilterString(requested_run_id);
    if (mode == "events") {
        auto result = QueryEvents(store_, active_filter_expression_, run_filter,
                                  kDefaultResultLimit, false);
        AddLine(result, RuntimeConsoleOutputLevel::Info,
                "Retained run-adapter events: " +
                    std::to_string(run.events.size()));
        for (const auto& event : run.events) {
            AddLine(result, RuntimeConsoleOutputLevel::Info,
                    FormatRunEvent(event));
        }
        return result;
    }

    if (mode == "fallback") {
        RuntimeConsoleCommandResult result;
        AddLine(result, RuntimeConsoleOutputLevel::Info,
                "Native CPU fallback: run=" + requested_run_id +
                    " count=" +
                    std::to_string(run.native_cpu_fallback_count) +
                    " source=" + RecordedOr(run.source));
        if (!run.training_evidence_available) {
            AddLine(result, RuntimeConsoleOutputLevel::Warning,
                    "Detailed fallback events were not retained for this run");
            return result;
        }
        for (const auto& event : run.training.recent_events) {
            if (!event.native_cpu_fallback) continue;
            AddLine(result, RuntimeConsoleOutputLevel::Warning,
                    FormatTrainingEvent(event));
        }
        return result;
    }

    if (mode == "host-sync") {
        RuntimeConsoleCommandResult result;
        if (!run.training_evidence_available) {
            AddLine(result, RuntimeConsoleOutputLevel::Warning,
                    "Host-sync evidence was not retained for run=" +
                        requested_run_id);
            return result;
        }
        AddLine(result, RuntimeConsoleOutputLevel::Info,
                "ArrayFire host sync: run=" + requested_run_id +
                    " count=" +
                    std::to_string(run.training.host_sync_count) +
                    " bytes=" +
                    FormatByteCount(run.training.host_sync_bytes));
        for (const auto& event : run.training.recent_events) {
            if (event.host_sync_bytes == 0 && event.host_sync_reason.empty()) {
                continue;
            }
            AddLine(result, RuntimeConsoleOutputLevel::Info,
                    FormatTrainingEvent(event));
        }
        return result;
    }

    std::optional<std::string> exact_code;
    std::optional<std::string> code_family;
    if (tokens.values.size() == 3) {
        if (mode == "code") {
            exact_code = NormalizeDiagnosticCode(tokens.values[2].text);
            if (!exact_code) return Usage("show run <run_id> code <CW-X-NNNN>");
        } else {
            code_family = NormalizeDiagnosticFamily(tokens.values[2].text);
            if (!code_family) return Usage("show run <run_id> codes <CW-X-*>");
        }
    }

    std::string log_filter = run_filter;
    if (exact_code) {
        log_filter += " and error_code=" + *exact_code;
    } else if (code_family) {
        log_filter += " and error_code matches " + *code_family;
    }
    auto result = QueryEvents(store_, active_filter_expression_, log_filter,
                              kDefaultResultLimit, true);
    size_t persisted_matches = 0;
    for (const auto& issue : run.issues) {
        if (issue.code.empty()) continue;
        if (exact_code && issue.code != *exact_code) continue;
        if (code_family &&
            (issue.code.size() < 5 || issue.code.compare(
                 0, 5, *code_family, 0, 5) != 0)) {
            continue;
        }
        AddLine(result, issue.level == "error"
                            ? RuntimeConsoleOutputLevel::Error
                            : RuntimeConsoleOutputLevel::Warning,
                FormatRunIssue(issue, requested_run_id));
        ++persisted_matches;
    }
    AddLine(result, RuntimeConsoleOutputLevel::Info,
            "Persisted run diagnostics: matched=" +
                std::to_string(persisted_matches) + " retained=" +
                std::to_string(run.issues.size()));
    return result;
}

RuntimeConsoleCommandResult RuntimeConsoleCommandService::ShowMaterialization(
    std::string_view arguments) {
    const auto tokens = TokenizeCommand(arguments);
    if (tokens.error || tokens.values.size() != 1 ||
        LowerAscii(tokens.values[0].text) != "last") {
        return Usage("show materialization last");
    }
    if (!truth_provider_) {
        return ErrorResult("Runtime materialization truth provider is unavailable");
    }

    auto truth = truth_provider_->GetCurrentTraining();
    if (!truth.available || truth.materialization_events.empty()) {
        truth = truth_provider_->GetLastTraining();
    }
    RuntimeConsoleCommandResult result;
    if (!truth.available || truth.materialization_events.empty()) {
        AddLine(result, RuntimeConsoleOutputLevel::Warning,
                "No retained materialization evidence is available");
        return result;
    }
    AddLine(result, RuntimeConsoleOutputLevel::Info,
            "Last materialization evidence: source=" +
                RecordedOr(truth.source) + " run=" +
                RecordedOr(truth.run_id) + " retained=" +
                std::to_string(truth.materialization_events.size()));
    for (const auto& event : truth.materialization_events) {
        AddLine(result, RuntimeConsoleOutputLevel::Info,
                FormatTrainingEvent(event));
    }
    return result;
}

RuntimeConsoleCommandResult RuntimeConsoleCommandService::ExecuteFilter(
    std::string_view command) {
    const auto tokens = TokenizeCommand(command);
    if (tokens.error || tokens.values.size() < 2) {
        return Usage("filter set|clear|save|use ...");
    }
    const auto mode = LowerAscii(tokens.values[1].text);
    if (mode == "clear") {
        if (tokens.values.size() != 2) return Usage("filter clear");
        active_filter_expression_.clear();
        RuntimeConsoleCommandResult result;
        AddLine(result, RuntimeConsoleOutputLevel::Success,
                "Session log filter cleared");
        return result;
    }
    if (mode == "set") {
        if (tokens.values.size() < 3) return Usage("filter set <expression>");
        const auto expression = std::string(SliceFrom(command, tokens.values[2]));
        auto parsed = ParseRuntimeLogFilter(expression);
        if (!parsed.Ok()) {
            return ErrorResult("Filter error at " +
                               std::to_string(parsed.error->position) + ": " +
                               parsed.error->message);
        }
        active_filter_expression_ = expression;
        RuntimeConsoleCommandResult result;
        AddLine(result, RuntimeConsoleOutputLevel::Success,
                "Session log filter set: " + expression);
        return result;
    }
    if (mode == "save") {
        if (tokens.values.size() < 4) {
            return Usage("filter save <name> <expression>");
        }
        const auto& name = tokens.values[2].text;
        if (!IsValidFilterName(name)) {
            return ErrorResult(
                "Filter name must use 1-64 letters, digits, '_' or '-'");
        }
        const auto expression = std::string(SliceFrom(command, tokens.values[3]));
        auto parsed = ParseRuntimeLogFilter(expression);
        if (!parsed.Ok()) {
            return ErrorResult("Filter error at " +
                               std::to_string(parsed.error->position) + ": " +
                               parsed.error->message);
        }
        saved_filters_[name] = expression;
        RuntimeConsoleCommandResult result;
        AddLine(result, RuntimeConsoleOutputLevel::Success,
                "Saved session filter '" + name + "'");
        return result;
    }
    if (mode == "use") {
        if (tokens.values.size() != 3) return Usage("filter use <name>");
        const auto found = saved_filters_.find(tokens.values[2].text);
        if (found == saved_filters_.end()) {
            return ErrorResult("Unknown saved filter: " + tokens.values[2].text);
        }
        active_filter_expression_ = found->second;
        RuntimeConsoleCommandResult result;
        AddLine(result, RuntimeConsoleOutputLevel::Success,
                "Using session filter '" + found->first + "': " +
                    found->second);
        return result;
    }
    return Usage("filter set|clear|save|use ...");
}

} // namespace cyxwiz
