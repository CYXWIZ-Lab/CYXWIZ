#pragma once

// Presentation model for the Console's Commands session (tofix121).
// Pure data in, data out: builds command suggestions from the command help
// text and turns the command service's output (text lines and event rows)
// into displayable pieces, so the renderer holds no parsing logic.

#include "runtime_log_event.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cyxwiz::commands {

// ---- Suggestions ------------------------------------------------------------

// What the command service publishes about one command.
struct CommandSource {
    std::string_view name;
    std::string_view usage;
    std::string_view description;
    std::string_view detailed_help;
};

struct CommandForm {
    std::string form;         // "show logs last <n>"
    std::string insert_text;  // "show logs last " (up to the first placeholder)
    std::string description;  // "Newest n events (1-1000)"
    std::string command;      // "show"
    bool complete = false;    // no placeholder: runs as inserted
};

// One form per line under "Forms:" / "Usage:" in the detailed help (a final
// "a|b|c" token expands to one form each), else the usage line.
std::vector<CommandForm> BuildCommandForms(const std::vector<CommandSource>& sources);

// Forms the input can still become, in list order. Empty input matches
// nothing; input that already runs past a form's placeholder matches none.
std::vector<size_t> MatchForms(const std::vector<CommandForm>& forms,
                               std::string_view input, size_t max_results = 8);

// The form whose command the input names ("show logs where x" -> the
// "show" usage), for the one-line hint beside the prompt; nullopt if unknown.
std::optional<size_t> UsageFormFor(const std::vector<CommandForm>& forms,
                                   std::string_view input);

// Lower-cased first word ("Show logs" -> "show").
std::string CommandName(std::string_view input);

// ---- Output -----------------------------------------------------------------

// An event row's fields besides time, level, category and message, on one
// line: "source cyxwiz · code CW-S-0501 · run train-1 · backend=... · key=value".
std::string EventFieldsLine(const RuntimeLogEvent& event);

// A JSON object or array that ends the message ("Generation sample: ... {...}"):
// the text before it, the JSON formatted with two-space indents, its size and
// how many keys or items it has. nullopt when the message carries none.
struct JsonPayload {
    std::string before;
    std::string pretty;
    size_t bytes = 0;
    size_t entries = 0;
};
std::optional<JsonPayload> FindJsonPayload(const std::string& message);

// "Logs: matched=167 showing=5 scanned=167 retained=167 evicted=0" ->
// "Matched 167 · showing 5 · scanned 167 · retained 167 · evicted 0".
std::optional<std::string> PrettySummary(std::string_view text);

// Status shown on a command block's header line.
struct BlockStatus {
    std::string text;  // "5 rows", "error", "Running · 12 s", "cancelled", ""
    enum class Kind { Ok, Error, Running, Cancelled } kind = Kind::Ok;
};
BlockStatus BuildBlockStatus(bool running, bool success, bool cancelled,
                             double running_seconds, size_t event_rows);

// "0.8 s", "12 s", "2 min 05 s".
std::string FormatElapsed(double seconds);

}  // namespace cyxwiz::commands
