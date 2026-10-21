#pragma once

// Presentation model for the Console's Logs session (tofix121).
// Pure data in, data out: no ImGui, no store access, so every renderer
// words, formats and groups runtime-log facts the same way.

#include "runtime_log_event.h"
#include "runtime_log_store.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::logs {

// ---- Severity ---------------------------------------------------------------

inline constexpr std::array<const char *, 6> kLevelLabels = {
    "Trace", "Debug", "Info", "Warn", "Error", "Critical"};

// Label used in the table and details ("Warn", "Error", ...).
const char *LevelLabel(RuntimeLogLevel level);
// Index 0..5 into kLevelLabels.
size_t LevelIndex(RuntimeLogLevel level);
// Error or Critical: counted in the tab badge and emphasised in rows.
bool IsProblemLevel(RuntimeLogLevel level);

// ---- Formatting -------------------------------------------------------------

// "1,184"
std::string FormatCount(uint64_t value);
// Table time: local "HH:MM:SS.mmm".
std::string FormatLocalTime(std::chrono::system_clock::time_point timestamp);
// "2026-09-25T21:13:48.487Z"
std::string FormatUtcTimestamp(std::chrono::system_clock::time_point timestamp);
// Local date and time with milliseconds: "2026-09-26 00:13:48.487".
std::string FormatLocalTimestamp(std::chrono::system_clock::time_point timestamp);

// Device column: "cuda · 0", "cuda", "device 2", or "" when unknown (never
// "backend:-1").
std::string DeviceLabel(const RuntimeLogEvent &event);

// First line of a message and whether more lines follow (rows stay one line
// high so the list clipper works).
struct MessageLine {
    std::string text;
    bool more_lines = false;
};
MessageLine FirstLine(const std::string &message);

// Short title for the details drawer: first line, cut at 90 characters.
std::string DetailsTitle(const RuntimeLogEvent &event);

// ---- Status line ------------------------------------------------------------

struct StatusInput {
    RuntimeLogStoreStats stats;
    size_t shown = 0;
    size_t matched = 0;
    uint64_t high_water = 0;
    bool truncated = false;       // more matches than the display shows
    bool paused = false;
    uint64_t hidden_through = 0;  // Clear view hides events up to this
    size_t display_limit = 1000;
};

struct StatusView {
    std::string state;        // "Live" / "Paused at #51,244"
    bool live = true;
    std::string showing;      // "Showing 1,000 of 1,208 matched"
    std::string retained;     // "Retained 4,096 / 4,096"
    std::string evicted;      // "Evicted 312"
    std::string losses;       // "Dropped 0 · Rejected 0 · Suppressed 0"
    bool has_losses = false;  // any of the three is non-zero
    std::string high_water;   // "High-water #51,244"
    std::string hidden;       // "Cleared view hides events through #51,200"
    std::string truncation;   // display-limit notice, empty when not limited
};

StatusView BuildStatusView(const StatusInput &input);

// ---- Details ----------------------------------------------------------------

// Every non-empty field of an event, in display order, for the details grid
// (two pairs per line) and for copying. Detail key/values are appended.
std::vector<std::pair<std::string, std::string>> DetailFields(
    const RuntimeLogEvent &event);

// One-line copy of a row: "#seq time level=.. category=.. ... | message".
std::string FormatRow(const RuntimeLogEvent &event);

// Number of field filters set (Filters (n) button).
size_t CountFieldFilters(bool category, bool source, bool code, bool run,
                         bool backend, bool task, bool device);

}  // namespace cyxwiz::logs
