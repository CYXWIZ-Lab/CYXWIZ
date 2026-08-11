#include "runtime_log_export.h"

#include "support_redaction.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <system_error>

namespace cyxwiz {
namespace {

std::string LowerAscii(std::string value) {
    std::transform(
        value.begin(), value.end(), value.begin(), [](unsigned char current) {
            return static_cast<char>(std::tolower(current));
        });
    return value;
}

bool ContainsLowered(std::string_view value, std::string_view expected) {
    return LowerAscii(std::string(value)).find(expected) != std::string::npos;
}

bool IsPathStart(const std::string& value, size_t index) {
    const bool boundary = index == 0 ||
        std::isspace(static_cast<unsigned char>(value[index - 1])) != 0 ||
        value[index - 1] == '=' || value[index - 1] == '"' ||
        value[index - 1] == '\'' || value[index - 1] == '(';
    if (!boundary) return false;
    if (index + 2 < value.size() &&
        std::isalpha(static_cast<unsigned char>(value[index])) != 0 &&
        value[index + 1] == ':' &&
        (value[index + 2] == '/' || value[index + 2] == '\\')) {
        return true;
    }
    if (index + 1 < value.size() && value[index] == '\\' &&
        value[index + 1] == '\\') {
        return true;
    }
    return value[index] == '/' &&
        index + 1 < value.size() && value[index + 1] != '/';
}

std::string RedactPaths(std::string value) {
    for (size_t index = 0; index < value.size(); ++index) {
        if (!IsPathStart(value, index)) continue;
        size_t end = index;
        while (end < value.size() && value[end] != '\r' &&
               value[end] != '\n' && value[end] != '"' &&
               value[end] != '\'' && value[end] != '<' &&
               value[end] != '>' && value[end] != '|') {
            ++end;
        }
        value.replace(index, end - index, "[REDACTED_PATH]");
        index += std::string_view("[REDACTED_PATH]").size() - 1;
    }
    return value;
}

bool IsQueryEvent(const RuntimeLogEvent& event) {
    return ContainsLowered(event.event_name, "query") ||
           ContainsLowered(event.event_name, "sql") ||
           ContainsLowered(event.source, "query") ||
           ContainsLowered(event.component, "query");
}

bool IsPythonEvent(const RuntimeLogEvent& event) {
    return event.category == "python" ||
           ContainsLowered(event.source, "python") ||
           ContainsLowered(event.source, "pip") ||
           ContainsLowered(event.component, "python") ||
           ContainsLowered(event.component, "package");
}

bool KeyContains(const std::string& key, std::string_view expected) {
    return LowerAscii(key).find(expected) != std::string::npos;
}

bool AnyRedaction(const RuntimeLogRedactionOptions& options) {
    return options.secrets || options.paths || options.dataset_names ||
           options.query_text || options.python_output;
}

std::string TimestampUtc(
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

nlohmann::json StoreStatsToJson(const RuntimeLogStoreStats& stats) {
    return {
        {"capacity", stats.capacity},
        {"size", stats.size},
        {"oldest_sequence", stats.oldest_sequence},
        {"newest_sequence", stats.newest_sequence},
        {"evicted_count", stats.evicted_count},
        {"dropped_count", stats.dropped_count},
        {"rejected_count", stats.rejected_count},
        {"suppressed_count", stats.suppressed_count}};
}

nlohmann::json RedactionToJson(const RuntimeLogRedactionOptions& options) {
    return {
        {"applied", AnyRedaction(options)},
        {"secrets", options.secrets},
        {"paths", options.paths},
        {"dataset_names", options.dataset_names},
        {"query_text", options.query_text},
        {"python_output", options.python_output}};
}

std::string RedactEffectiveFilter(
    const std::string& filter,
    const RuntimeLogRedactionOptions& options) {
    if (filter.empty()) return filter;
    if (options.query_text ||
        (options.dataset_names && ContainsLowered(filter, "dataset"))) {
        return "[REDACTED_FILTER]";
    }
    auto redacted = options.secrets
        ? SupportRedaction::RedactString(filter)
        : filter;
    if (options.paths) redacted = RedactPaths(std::move(redacted));
    return redacted;
}

nlohmann::json MetadataToJson(
    const RuntimeLogExportSnapshot& snapshot,
    const RuntimeLogRedactionOptions& options) {
    return {
        {"scope", snapshot.scope},
        {"effective_filter",
         RedactEffectiveFilter(snapshot.effective_filter, options)},
        {"after_sequence", snapshot.after_sequence},
        {"through_sequence", snapshot.through_sequence},
        {"matched_count", snapshot.matched_count},
        {"source_displayed_count", snapshot.source_displayed_count},
        {"exported_count", snapshot.events.size()},
        {"source_truncated", snapshot.source_truncated},
        {"store", StoreStatsToJson(snapshot.store_stats)},
        {"redaction", RedactionToJson(options)}};
}

} // namespace

RuntimeLogExportSnapshot RuntimeLogExportService::Freeze(
    const RuntimeLogInspectorResult& result,
    uint64_t after_sequence,
    std::optional<uint64_t> selected_sequence) {
    RuntimeLogExportSnapshot snapshot;
    snapshot.scope = selected_sequence ? "selected" : "filtered";
    snapshot.effective_filter = result.effective_filter;
    snapshot.after_sequence = after_sequence;
    snapshot.through_sequence = result.query.high_water_sequence;
    snapshot.matched_count = result.query.matched_count;
    snapshot.source_displayed_count = result.query.events.size();
    snapshot.source_truncated = result.query.truncated;
    snapshot.store_stats = result.query.store_stats;
    if (selected_sequence) {
        const auto selected = std::find_if(
            result.query.events.begin(), result.query.events.end(),
            [selected_sequence](const auto& event) {
                return event.sequence == *selected_sequence;
            });
        if (selected != result.query.events.end()) {
            snapshot.events.push_back(*selected);
        }
    } else {
        snapshot.events = result.query.events;
    }
    return snapshot;
}

RuntimeLogEvent RuntimeLogExportService::RedactEvent(
    const RuntimeLogEvent& event,
    const RuntimeLogRedactionOptions& options) {
    RuntimeLogEvent redacted = event;
    if (options.dataset_names && !redacted.dataset_name.empty()) {
        redacted.dataset_name = "[REDACTED_DATASET]";
    }

    if (options.query_text && IsQueryEvent(redacted)) {
        redacted.message = "[REDACTED_QUERY]";
    } else if (options.python_output && IsPythonEvent(redacted)) {
        redacted.message = "[REDACTED_PYTHON_OUTPUT]";
    } else {
        if (options.secrets) {
            redacted.message = SupportRedaction::RedactString(redacted.message);
        }
        if (options.paths) redacted.message = RedactPaths(redacted.message);
    }

    for (auto& [key, value] : redacted.details) {
        if (options.dataset_names && KeyContains(key, "dataset")) {
            value = "[REDACTED_DATASET]";
        } else if (options.query_text &&
                   (KeyContains(key, "query") || KeyContains(key, "sql"))) {
            value = "[REDACTED_QUERY]";
        } else if (options.python_output &&
                   (KeyContains(key, "python") || KeyContains(key, "pip") ||
                    KeyContains(key, "package"))) {
            value = "[REDACTED_PYTHON_OUTPUT]";
        } else if (options.paths &&
                   (KeyContains(key, "path") || KeyContains(key, "file"))) {
            value = "[REDACTED_PATH]";
        } else {
            if (options.secrets) value = SupportRedaction::RedactString(value);
            if (options.paths) value = RedactPaths(value);
        }
    }
    return redacted;
}

nlohmann::json RuntimeLogExportService::EventToJson(
    const RuntimeLogEvent& event,
    const RuntimeLogRedactionOptions& options) {
    const auto output = RedactEvent(event, options);
    nlohmann::json details = nlohmann::json::array();
    for (const auto& [key, value] : output.details) {
        details.push_back({{"key", key}, {"value", value}});
    }
    return {
        {"sequence", output.sequence},
        {"timestamp_utc", TimestampUtc(output.timestamp_utc)},
        {"level", std::string(RuntimeLogLevelName(output.level))},
        {"category", output.category},
        {"source", output.source},
        {"event_name", output.event_name},
        {"run_id", output.run_id},
        {"task_id", output.task_id},
        {"thread_id", output.thread_id},
        {"backend", output.backend},
        {"device_id", output.device_id},
        {"device_name", output.device_name},
        {"node_id", output.node_id},
        {"dataset_name", output.dataset_name},
        {"primary_error_code", output.primary_error_code},
        {"issue_codes", output.issue_codes},
        {"diagnostic_phase", output.diagnostic_phase},
        {"component", output.component},
        {"message", output.message},
        {"details", std::move(details)}};
}

nlohmann::json RuntimeLogExportService::SnapshotToJson(
    const RuntimeLogExportSnapshot& snapshot,
    const RuntimeLogRedactionOptions& options) {
    nlohmann::json events = nlohmann::json::array();
    for (const auto& event : snapshot.events) {
        events.push_back(EventToJson(event, options));
    }
    return {
        {"schema", kSchema},
        {"metadata", MetadataToJson(snapshot, options)},
        {"events", std::move(events)}};
}

std::string RuntimeLogExportService::FormatEventText(
    const RuntimeLogEvent& event,
    const RuntimeLogRedactionOptions& options) {
    const auto output = RedactEvent(event, options);
    std::ostringstream text;
    text << '#' << output.sequence << ' ' << TimestampUtc(output.timestamp_utc)
         << " level=" << RuntimeLogLevelName(output.level)
         << " category=" << output.category;
    if (!output.source.empty()) text << " source=" << output.source;
    if (!output.primary_error_code.empty()) {
        text << " code=" << output.primary_error_code;
    }
    if (!output.run_id.empty()) text << " run=" << output.run_id;
    if (output.task_id != 0) text << " task=" << output.task_id;
    if (!output.backend.empty()) text << " backend=" << output.backend;
    if (output.device_id >= 0) text << " device_id=" << output.device_id;
    if (!output.dataset_name.empty()) {
        text << " dataset=" << output.dataset_name;
    }
    text << " | " << output.message;
    for (const auto& [key, value] : output.details) {
        text << "\n  " << key << '=' << value;
    }
    return text.str();
}

std::string RuntimeLogExportService::Serialize(
    const RuntimeLogExportSnapshot& snapshot,
    RuntimeLogExportFormat format,
    const RuntimeLogRedactionOptions& options) {
    std::ostringstream output;
    if (format == RuntimeLogExportFormat::JsonLines) {
        output << nlohmann::json({
            {"record_type", "metadata"},
            {"schema", kSchema},
            {"metadata", MetadataToJson(snapshot, options)}}).dump() << '\n';
        for (const auto& event : snapshot.events) {
            output << nlohmann::json({
                {"record_type", "event"},
                {"event", EventToJson(event, options)}}).dump() << '\n';
        }
        return output.str();
    }

    output << "CyxWiz runtime log export\n"
           << "schema=" << kSchema << '\n'
           << "scope=" << snapshot.scope << '\n'
           << "effective_filter="
           << RedactEffectiveFilter(snapshot.effective_filter, options) << '\n'
           << "sequence_range=(" << snapshot.after_sequence << ", "
           << snapshot.through_sequence << "]\n"
           << "matched=" << snapshot.matched_count
           << " displayed=" << snapshot.source_displayed_count
           << " exported=" << snapshot.events.size()
           << " source_truncated="
           << (snapshot.source_truncated ? "true" : "false") << '\n'
           << "redaction=" << RedactionToJson(options).dump() << "\n\n";
    for (const auto& event : snapshot.events) {
        output << FormatEventText(event, options) << '\n';
    }
    return output.str();
}

RuntimeLogExportResult RuntimeLogExportService::Write(
    const RuntimeLogExportSnapshot& snapshot,
    const RuntimeLogExportRequest& request) {
    RuntimeLogExportResult result;
    result.destination = request.destination;
    if (request.destination.empty()) {
        result.error = "Export destination is empty";
        return result;
    }
    if (snapshot.events.empty()) {
        result.error = "Frozen log slice contains no events";
        return result;
    }

    std::error_code error;
    const auto parent = request.destination.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, error);
        if (error) {
            result.error = "Could not create export directory: " +
                error.message();
            return result;
        }
    }

    const auto payload = Serialize(snapshot, request.format, request.redaction);
    std::ofstream output(
        request.destination, std::ios::binary | std::ios::trunc);
    if (!output) {
        result.error = "Could not open export destination";
        return result;
    }
    output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    output.close();
    if (!output) {
        result.error = "Could not write complete runtime-log export";
        return result;
    }

    result.success = true;
    result.events_written = snapshot.events.size();
    return result;
}

} // namespace cyxwiz
