#pragma once

#include "runtime_log_inspector.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

enum class RuntimeLogExportFormat : uint8_t {
    JsonLines,
    ReadableText
};

struct RuntimeLogRedactionOptions {
    bool secrets = true;
    bool paths = true;
    bool dataset_names = true;
    bool query_text = true;
    bool python_output = true;
};

struct RuntimeLogExportSnapshot {
    std::string scope = "filtered";
    std::string effective_filter;
    uint64_t after_sequence = 0;
    uint64_t through_sequence = 0;
    size_t matched_count = 0;
    size_t source_displayed_count = 0;
    bool source_truncated = false;
    RuntimeLogStoreStats store_stats;
    std::vector<RuntimeLogEvent> events;
};

struct RuntimeLogExportRequest {
    std::filesystem::path destination;
    RuntimeLogExportFormat format = RuntimeLogExportFormat::JsonLines;
    RuntimeLogRedactionOptions redaction;
};

struct RuntimeLogExportResult {
    bool success = false;
    size_t events_written = 0;
    std::filesystem::path destination;
    std::string error;
};

class RuntimeLogExportService {
public:
    static constexpr const char* kSchema = "cyxwiz.runtime_log_export.v1";

    static RuntimeLogExportSnapshot Freeze(
        const RuntimeLogInspectorResult& result,
        uint64_t after_sequence,
        std::optional<uint64_t> selected_sequence = std::nullopt);

    static RuntimeLogEvent RedactEvent(
        const RuntimeLogEvent& event,
        const RuntimeLogRedactionOptions& options);
    static nlohmann::json EventToJson(
        const RuntimeLogEvent& event,
        const RuntimeLogRedactionOptions& options);
    static nlohmann::json SnapshotToJson(
        const RuntimeLogExportSnapshot& snapshot,
        const RuntimeLogRedactionOptions& options);
    static std::string FormatEventText(
        const RuntimeLogEvent& event,
        const RuntimeLogRedactionOptions& options);
    static std::string Serialize(
        const RuntimeLogExportSnapshot& snapshot,
        RuntimeLogExportFormat format,
        const RuntimeLogRedactionOptions& options);
    static RuntimeLogExportResult Write(
        const RuntimeLogExportSnapshot& snapshot,
        const RuntimeLogExportRequest& request);
};

} // namespace cyxwiz
