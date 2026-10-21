#include "../src/core/runtime_log_presentation.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

bool HasField(const std::vector<std::pair<std::string, std::string>>& fields,
              const std::string& key, const std::string& value) {
    for (const auto& [k, v] : fields)
        if (k == key && v == value) return true;
    return false;
}

bool HasKey(const std::vector<std::pair<std::string, std::string>>& fields,
            const std::string& key) {
    for (const auto& field : fields)
        if (field.first == key) return true;
    return false;
}

// The export failure seen on the Dell run (2026-09-25).
cyxwiz::RuntimeLogEvent ExportFailure() {
    cyxwiz::RuntimeLogEvent e;
    e.sequence = 51237;
    e.timestamp_utc = std::chrono::system_clock::time_point(
        std::chrono::milliseconds(1790370828487LL));  // 2026-09-25T21:13:48.487Z
    e.level = cyxwiz::RuntimeLogLevel::Error;
    e.category = "system";
    e.source = "cyxwiz";
    e.event_name = "spdlog";
    e.thread_id = "14820";
    e.primary_error_code = "CW-S-0501";
    e.issue_codes = {"CW-S-0501"};
    e.message =
        "Model export failed: ArrayFire Exception (There was a mismatch between an "
        "array and the current backend:503): Error in af_retain_array\n"
        "In file src\\api\\unified\\array.cpp:50: Input array does not belong to "
        "current backend";
    e.details = {{"file", "export_dialog.cpp:457"}};
    return e;
}

void TestFormatting() {
    using namespace cyxwiz::logs;
    Check(FormatCount(0) == "0", "zero");
    Check(FormatCount(999) == "999", "no separator under 1000");
    Check(FormatCount(1184) == "1,184", "thousands");
    Check(FormatCount(51244) == "51,244", "tens of thousands");
    Check(FormatCount(1234567) == "1,234,567", "millions");
    const auto e = ExportFailure();
    Check(FormatUtcTimestamp(e.timestamp_utc) == "2026-09-25T21:13:48.487Z", "UTC timestamp");
    Check(FormatLocalTime(e.timestamp_utc).size() == 12, "local time HH:MM:SS.mmm");
    Check(FormatLocalTime(e.timestamp_utc).substr(8) == ".487", "milliseconds kept");
}

void TestLevelsAndDevice() {
    using namespace cyxwiz::logs;
    Check(std::string(LevelLabel(cyxwiz::RuntimeLogLevel::Warning)) == "Warn", "warn label");
    Check(IsProblemLevel(cyxwiz::RuntimeLogLevel::Critical), "critical is a problem");
    Check(!IsProblemLevel(cyxwiz::RuntimeLogLevel::Warning), "warning is not counted");
    cyxwiz::RuntimeLogEvent e;
    Check(DeviceLabel(e).empty(), "no device -> empty, not backend:-1");
    e.backend = "cuda";
    Check(DeviceLabel(e) == "cuda", "backend only");
    e.device_id = 0;
    Check(DeviceLabel(e) == "cuda \xC2\xB7 0", "backend and id");
    e.backend.clear();
    e.device_id = 2;
    Check(DeviceLabel(e) == "device 2", "id only");
}

void TestMessages() {
    using namespace cyxwiz::logs;
    const auto e = ExportFailure();
    const auto line = FirstLine(e.message);
    Check(line.more_lines, "multi-line message flagged");
    Check(line.text.rfind("Model export failed", 0) == 0 && line.text.find('\n') == std::string::npos,
          "first line only");
    Check(!FirstLine("single").more_lines, "single line");
    Check(!FirstLine("trailing\n").more_lines, "trailing newline is not more lines");
    const auto title = DetailsTitle(e);
    Check(title.size() <= 93 && title.substr(title.size() - 3) == "...", "long title cut");
}

void TestStatus() {
    using namespace cyxwiz::logs;
    StatusInput in;
    in.stats.size = 4096;
    in.stats.capacity = 4096;
    in.stats.evicted_count = 312;
    in.shown = 1000;
    in.matched = 1208;
    in.high_water = 51244;
    in.truncated = true;
    auto view = BuildStatusView(in);
    Check(view.live && view.state == "Live", "live state");
    Check(view.showing == "Showing 1,000 of 1,208 matched", "showing text");
    Check(view.retained == "Retained 4,096 / 4,096", "retained text");
    Check(view.evicted == "Evicted 312", "evicted text");
    Check(view.losses == "Dropped 0 \xC2\xB7 Rejected 0 \xC2\xB7 Suppressed 0" && !view.has_losses,
          "losses text");
    Check(view.high_water == "High-water #51,244", "high-water text");
    Check(view.truncation.rfind("Display limited to the newest 1,000 rows.", 0) == 0,
          "truncation notice");
    in.paused = true;
    in.truncated = false;
    in.hidden_through = 51200;
    in.stats.dropped_count = 3;
    view = BuildStatusView(in);
    Check(!view.live && view.state == "Paused at #51,244", "paused state");
    Check(view.truncation.empty(), "no notice without truncation");
    Check(view.hidden == "Cleared view hides events through #51,200", "cleared view text");
    Check(view.has_losses, "dropped events surfaced");
}

void TestDetails() {
    using namespace cyxwiz::logs;
    const auto e = ExportFailure();
    const auto fields = DetailFields(e);
    Check(HasField(fields, "Time (UTC)", "2026-09-25T21:13:48.487Z"), "utc field");
    Check(HasField(fields, "Sequence", "#51,237"), "sequence field");
    Check(HasField(fields, "Thread", "14820"), "thread shown");
    Check(HasField(fields, "Issue codes", "CW-S-0501"), "issue codes");
    Check(HasField(fields, "file", "export_dialog.cpp:457"), "detail pairs appended");
    Check(!HasKey(fields, "Run") && !HasKey(fields, "Device"), "empty fields skipped");
    const auto row = FormatRow(e);
    Check(row.find("#51237 2026-09-25T21:13:48.487Z level=Error") == 0, "row prefix");
    Check(row.find("thread=14820") != std::string::npos, "thread in copied row");
    Check(CountFieldFilters(true, false, true, false, false, true, false) == 3, "filter count");
}

}  // namespace

int main() {
    TestFormatting();
    TestLevelsAndDevice();
    TestMessages();
    TestStatus();
    TestDetails();
    if (failures) {
        std::cerr << failures << " runtime log presentation check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "Runtime log presentation tests passed\n";
    return EXIT_SUCCESS;
}
