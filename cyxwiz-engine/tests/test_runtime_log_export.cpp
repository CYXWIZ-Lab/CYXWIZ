#include "../src/core/runtime_log_export.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

cyxwiz::RuntimeLogEvent MakeEvent(
    uint64_t sequence, std::string category, std::string source,
    std::string event_name, std::string message) {
    cyxwiz::RuntimeLogEvent event;
    event.sequence = sequence;
    event.timestamp_utc = std::chrono::system_clock::time_point(
        std::chrono::milliseconds(1'700'000'000'123 + sequence));
    event.level = cyxwiz::RuntimeLogLevel::Warning;
    event.category = std::move(category);
    event.source = std::move(source);
    event.event_name = std::move(event_name);
    event.run_id = "train-42";
    event.task_id = 17;
    event.backend = "arrayfire_cuda";
    event.device_id = 0;
    event.message = std::move(message);
    return event;
}

cyxwiz::RuntimeLogInspectorResult MakeResult() {
    cyxwiz::RuntimeLogInspectorResult result;
    result.effective_filter = "level>=warn";
    result.query.matched_count = 7;
    result.query.high_water_sequence = 12;
    result.query.truncated = true;
    result.query.store_stats.capacity = 4096;
    result.query.store_stats.size = 12;
    result.query.store_stats.oldest_sequence = 1;
    result.query.store_stats.newest_sequence = 12;
    result.query.store_stats.evicted_count = 3;
    result.query.store_stats.suppressed_count = 5;

    auto secret = MakeEvent(
        10, "system", "cyxwiz", "startup.warning",
        "authentication failed token=secret-token");
    secret.details = {
        {"config_path", "C:/Users/private/project/config.json"},
        {"note", "credential=private-value"}};
    result.query.events.push_back(std::move(secret));

    auto query = MakeEvent(
        11, "data", "QueryConsole", "sql.query",
        "SELECT * FROM private_customers");
    query.dataset_name = "private_customers.parquet";
    query.details = {{"sql_text", "SELECT email FROM private_customers"}};
    result.query.events.push_back(std::move(query));

    auto python = MakeEvent(
        12, "python", "pip", "package.output",
        "Installed private-package from C:/private/wheel.whl");
    python.details = {{"package_output", "private-package==1.0"}};
    result.query.events.push_back(std::move(python));
    return result;
}

void TestFrozenScopeAndMetadata() {
    const auto result = MakeResult();
    const auto filtered =
        cyxwiz::RuntimeLogExportService::Freeze(result, 4);
    Check(filtered.scope == "filtered" && filtered.events.size() == 3,
          "filtered freeze should own only the displayed immutable rows");
    Check(filtered.after_sequence == 4 && filtered.through_sequence == 12 &&
              filtered.matched_count == 7 && filtered.source_displayed_count == 3 &&
              filtered.source_truncated,
          "frozen export should preserve query and truncation metadata");

    const auto selected =
        cyxwiz::RuntimeLogExportService::Freeze(result, 4, 11);
    Check(selected.scope == "selected" && selected.events.size() == 1 &&
              selected.events[0].sequence == 11,
          "selected freeze should export exactly the explicit selected row");
    Check(cyxwiz::RuntimeLogExportService::Freeze(result, 4, 99).events.empty(),
          "a missing selected sequence must not silently export all rows");
}

void TestRedactionAndRawOptions() {
    const auto snapshot =
        cyxwiz::RuntimeLogExportService::Freeze(MakeResult(), 0);
    const cyxwiz::RuntimeLogRedactionOptions safe;
    const auto output =
        cyxwiz::RuntimeLogExportService::SnapshotToJson(snapshot, safe);

    Check(output["metadata"]["redaction"]["applied"].get<bool>(),
          "safe export should declare redaction");
    Check(output["metadata"]["effective_filter"] == "[REDACTED_FILTER]",
          "shareable export should redact filter/query metadata");
    Check(output["events"][0]["message"].get<std::string>() ==
              "authentication failed token=[REDACTED]",
          "secret markers should be redacted");
    Check(output["events"][0]["details"][0]["value"].get<std::string>() ==
              "[REDACTED_PATH]",
          "path-valued details should be redacted by key");
    Check(output["events"][1]["dataset_name"].get<std::string>() ==
              "[REDACTED_DATASET]" &&
              output["events"][1]["message"].get<std::string>() ==
                  "[REDACTED_QUERY]",
          "dataset identity and query text should be redacted");
    Check(output["events"][2]["message"].get<std::string>() ==
              "[REDACTED_PYTHON_OUTPUT]",
          "Python/package output should be redacted");

    cyxwiz::RuntimeLogRedactionOptions raw;
    raw.secrets = false;
    raw.paths = false;
    raw.dataset_names = false;
    raw.query_text = false;
    raw.python_output = false;
    const auto unredacted =
        cyxwiz::RuntimeLogExportService::SnapshotToJson(snapshot, raw);
    Check(!unredacted["metadata"]["redaction"]["applied"].get<bool>() &&
              unredacted["metadata"]["effective_filter"] == "level>=warn" &&
              unredacted["events"][0]["message"].get<std::string>().find(
                  "secret-token") != std::string::npos &&
              unredacted["events"][1]["dataset_name"].get<std::string>() ==
                  "private_customers.parquet" &&
              unredacted["events"][2]["message"].get<std::string>().find(
                  "private-package") != std::string::npos,
          "explicitly disabled redaction should preserve local raw values");
}

void TestFormatsAndWriting() {
    const auto snapshot =
        cyxwiz::RuntimeLogExportService::Freeze(MakeResult(), 4, 11);
    const cyxwiz::RuntimeLogRedactionOptions options;
    const auto jsonl = cyxwiz::RuntimeLogExportService::Serialize(
        snapshot, cyxwiz::RuntimeLogExportFormat::JsonLines, options);
    std::istringstream lines(jsonl);
    std::string metadata_line;
    std::string event_line;
    std::string extra_line;
    std::getline(lines, metadata_line);
    std::getline(lines, event_line);
    std::getline(lines, extra_line);
    Check(nlohmann::json::parse(metadata_line)["record_type"] == "metadata" &&
              nlohmann::json::parse(event_line)["record_type"] == "event" &&
              extra_line.empty(),
          "JSONL should contain one metadata record and one record per event");

    const auto text = cyxwiz::RuntimeLogExportService::Serialize(
        snapshot, cyxwiz::RuntimeLogExportFormat::ReadableText, options);
    Check(text.find("scope=selected") != std::string::npos &&
              text.find("effective_filter=[REDACTED_FILTER]") !=
                  std::string::npos &&
              text.find("[REDACTED_QUERY]") != std::string::npos,
          "readable text should preserve metadata and use the same redaction");

    const auto root = std::filesystem::temp_directory_path() /
        "cyxwiz_runtime_log_export_test";
    const auto path = root / "selected.jsonl";
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
    cyxwiz::RuntimeLogExportRequest request;
    request.destination = path;
    request.format = cyxwiz::RuntimeLogExportFormat::JsonLines;
    const auto written =
        cyxwiz::RuntimeLogExportService::Write(snapshot, request);
    Check(written.success && written.events_written == 1 &&
              std::filesystem::is_regular_file(path),
          "export should create its destination and report written rows");
    std::ifstream input(path, std::ios::binary);
    const std::string persisted(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    Check(persisted == jsonl,
          "persisted JSONL should equal deterministic in-memory serialization");
    std::filesystem::remove_all(root, cleanup_error);
}

} // namespace

int main() {
    TestFrozenScopeAndMetadata();
    TestRedactionAndRawOptions();
    TestFormatsAndWriting();
    std::cout << "Runtime log export contracts passed\n";
    return 0;
}
