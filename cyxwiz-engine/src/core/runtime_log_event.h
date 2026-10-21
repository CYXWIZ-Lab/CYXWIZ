#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cyxwiz {

enum class RuntimeLogLevel : uint8_t {
    Trace = 0,
    Debug,
    Info,
    Warning,
    Error,
    Critical
};

std::string_view RuntimeLogLevelName(RuntimeLogLevel level);
bool IsCanonicalRuntimeLogCategory(std::string_view category);
bool IsCanonicalDiagnosticCode(std::string_view code);
bool IsCanonicalDiagnosticFamily(std::string_view family);
std::optional<std::string> NormalizeDiagnosticCode(std::string_view code);
std::optional<std::string> NormalizeDiagnosticFamily(std::string_view family);
std::optional<std::string> ExtractLegacyDiagnosticCode(
    std::string_view message);

struct RuntimeLogEvent {
    uint64_t sequence = 0;
    std::chrono::system_clock::time_point timestamp_utc{};
    RuntimeLogLevel level = RuntimeLogLevel::Info;
    std::string category = "system";
    std::string source;
    std::string event_name;
    std::string run_id;
    uint64_t task_id = 0;
    std::string thread_id;
    std::string backend;
    int device_id = -1;
    std::string device_name;
    int node_id = -1;
    std::string dataset_name;
    std::string primary_error_code;
    std::vector<std::string> issue_codes;
    std::string diagnostic_phase;
    std::string component;
    std::string message;
    std::vector<std::pair<std::string, std::string>> details;
};

} // namespace cyxwiz
