#include "../src/core/console_commands_presentation.h"

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

// Excerpts of the real command help (runtime_console_commands.cpp).
std::vector<cyxwiz::commands::CommandSource> Sources() {
    return {
        {"help", "help [command]", "Show command help",
         "Usage:\n  help\n  help <command>\nExamples:\n  help show\n  help filter"},
        {"clear", "clear", "Clear this Console view",
         "Clears only the visible Console transcript.\nExample:\n  clear"},
        {"pip", "pip <arguments>", "Run project-environment pip asynchronously",
         "Runs pip only from the active project's virtual environment.\n"
         "Examples:\n  pip list\n  pip show numpy"},
        {"show", "show logs|errors|code|codes|training|device|backend|run|materialization ...",
         "Query bounded runtime diagnostics",
         "Log queries are read-only.\nForms:\n"
         "  show logs last <n>\n  show logs errors\n  show logs warnings\n"
         "  show logs where <filter>\n  show logs grep <text>\n"
         "  show training current|last|trace\n"
         "  show device active|available|queued\n"
         "  show device route <backend>:<id>\n"
         "  show backend support-bundle [1-100]\n"
         "  show backend packs  Installed GPU packs and their state\n"
         "Examples:\n  show logs last 50\n  show device active"},
    };
}

size_t Find(const std::vector<cyxwiz::commands::CommandForm>& forms, const std::string& form) {
    for (size_t i = 0; i < forms.size(); ++i)
        if (forms[i].form == form) return i;
    return forms.size();
}

void TestForms() {
    using namespace cyxwiz::commands;
    const auto forms = BuildCommandForms(Sources());
    Check(Find(forms, "help") < forms.size() && Find(forms, "help <command>") < forms.size(),
          "help forms from Usage:");
    Check(Find(forms, "clear") < forms.size(), "usage line when no forms listed");
    Check(Find(forms, "pip <arguments>") < forms.size(), "pip usage");
    Check(Find(forms, "show logs last <n>") < forms.size(), "show forms from Forms:");
    Check(Find(forms, "show training last") < forms.size() &&
              Find(forms, "show training trace") < forms.size(),
          "alternatives expanded");
    Check(Find(forms, "show device queued") < forms.size(), "device alternatives");
    Check(Find(forms, "show logs last 50") == forms.size(), "examples are not forms");
    Check(Find(forms, "show logs|errors|code|codes|training|device|backend|run|materialization ...") ==
              forms.size(),
          "summary usage skipped when forms exist");
    const auto& last = forms[Find(forms, "show logs last <n>")];
    Check(last.insert_text == "show logs last " && !last.complete, "insert up to placeholder");
    Check(last.description == "Newest n events (1-1000)", "known form description");
    const auto& errors = forms[Find(forms, "show logs errors")];
    Check(errors.complete && errors.insert_text == "show logs errors", "complete form");
    Check(forms[Find(forms, "show backend packs")].description ==
              "Installed GPU packs and their state",
          "description written beside the form wins");
    const auto& bundle = forms[Find(forms, "show backend support-bundle [1-100]")];
    Check(bundle.insert_text == "show backend support-bundle ", "optional placeholder");
    Check(forms[Find(forms, "show device route <backend>:<id>")].description ==
              "Query bounded runtime diagnostics",
          "fallback description");
}

void TestMatching() {
    using namespace cyxwiz::commands;
    const auto forms = BuildCommandForms(Sources());
    const auto matches = MatchForms(forms, "show logs w");
    Check(matches.size() == 2 && forms[matches[0]].form == "show logs warnings" &&
              forms[matches[1]].form == "show logs where <filter>",
          "prefix matches in order");
    Check(MatchForms(forms, "").empty(), "empty input matches nothing");
    Check(MatchForms(forms, "SHOW LOGS L").size() == 1, "case-insensitive");
    Check(MatchForms(forms, "show logs errors").empty(), "complete input closes the list");
    Check(MatchForms(forms, "show logs where level>=warn").empty(), "past placeholder");
    const auto usage = UsageFormFor(forms, "show logs where level>=warn");
    Check(usage && forms[*usage].form == "show logs where <filter>", "usage for typed form");
    const auto name_usage = UsageFormFor(forms, "show nonsense");
    Check(name_usage && forms[*name_usage].command == "show", "usage by command name");
    Check(!UsageFormFor(forms, "frobnicate"), "unknown command");
    Check(CommandName("  Show logs") == "show", "command name");
}

void TestOutput() {
    using namespace cyxwiz::commands;
    const auto event = ParseEventLine(
        "#167 2026-09-27T03:52:23.932Z level=error category=system source=cyxwiz "
        "code=CW-S-0501 run=train-1 | Python execution error: ValueError: boom | x");
    Check(event.has_value(), "event line parsed");
    if (event) {
        Check(event->sequence == 167, "sequence");
        Check(event->time == "03:52:23.932", "time part");
        Check(event->level == "error" && EventLevelIndex(event->level) == 4, "level");
        Check(event->category == "system" && event->source == "cyxwiz", "fields");
        Check(event->extra == "code=CW-S-0501 run=train-1", "extra fields");
        Check(event->message == "Python execution error: ValueError: boom | x",
              "message keeps later bars");
    }
    Check(!ParseEventLine("Logs: matched=1").has_value(), "not an event");
    Check(!ParseEventLine("#12 not-a-time").has_value(), "bad timestamp");
    const auto parsed_time = ParseUtcTimestamp("2026-09-25T21:13:48.487Z");
    Check(parsed_time && std::chrono::duration_cast<std::chrono::milliseconds>(
                             parsed_time->time_since_epoch())
                                 .count() == 1790370828487LL,
          "utc timestamp parsed");
    Check(!ParseUtcTimestamp("not a time").has_value(), "bad utc timestamp");
    Check(EventLevelIndex("warning") == 3 && EventLevelIndex("info") == 2, "level names");
    const auto summary =
        PrettySummary("Logs: matched=167 showing=5 scanned=167 retained=167 evicted=0");
    Check(summary && *summary ==
                         "Matched 167 \xC2\xB7 showing 5 \xC2\xB7 scanned 167 \xC2\xB7 "
                         "retained 167 \xC2\xB7 evicted 0",
          "summary");
    Check(!PrettySummary("Active device: x").has_value(), "not a summary");
}

void TestStatus() {
    using namespace cyxwiz::commands;
    auto status = BuildBlockStatus(false, true, false, 0, 5);
    Check(status.kind == BlockStatus::Kind::Ok && status.text == "5 rows", "rows");
    status = BuildBlockStatus(false, true, false, 0, 1);
    Check(status.text == "1 row", "one row");
    status = BuildBlockStatus(false, true, false, 0, 0);
    Check(status.text.empty(), "plain success");
    status = BuildBlockStatus(false, false, false, 0, 0);
    Check(status.kind == BlockStatus::Kind::Error && status.text == "error", "error");
    status = BuildBlockStatus(true, true, false, 12.4, 0);
    Check(status.kind == BlockStatus::Kind::Running && status.text == "Running \xC2\xB7 12 s",
          "running");
    status = BuildBlockStatus(false, false, true, 0, 0);
    Check(status.kind == BlockStatus::Kind::Cancelled, "cancelled");
    Check(FormatElapsed(125) == "2 min 05 s" && FormatElapsed(0.84) == "0.8 s", "elapsed");
}

}  // namespace

int main() {
    TestForms();
    TestMatching();
    TestOutput();
    TestStatus();
    if (failures) {
        std::cerr << failures << " console commands presentation check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "Console commands presentation tests passed\n";
    return EXIT_SUCCESS;
}
