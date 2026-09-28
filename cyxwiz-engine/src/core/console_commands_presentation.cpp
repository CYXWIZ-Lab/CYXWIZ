#include "console_commands_presentation.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <sstream>

namespace cyxwiz::commands {

namespace {

std::string Lower(std::string_view text) {
    std::string out(text);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

std::string_view TrimLeft(std::string_view text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())))
        text.remove_prefix(1);
    return text;
}

std::string_view Trim(std::string_view text) {
    text = TrimLeft(text);
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())))
        text.remove_suffix(1);
    return text;
}

bool StartsWith(std::string_view text, std::string_view prefix) {
    return text.substr(0, prefix.size()) == prefix;
}

// Short descriptions for the common forms; other forms use the command's own
// description from its help.
std::string FormDescription(std::string_view form, std::string_view fallback) {
    struct Known {
        std::string_view form;
        std::string_view description;
    };
    static const Known kKnown[] = {
        {"show logs last <n>", "Newest n events (1-1000)"},
        {"show logs errors", "Errors and critical events"},
        {"show logs warnings", "Warnings and worse"},
        {"show logs where <filter>", "Structured filter over retained events"},
        {"show logs grep <text>", "Text search in messages"},
        {"show logs code <CW-X-NNNN>", "Events with one diagnostic code"},
        {"show logs codes <CW-X-*>", "Events with a code family"},
        {"show errors last <n>", "Newest n errors"},
        {"show code <CW-X-NNNN>", "What a diagnostic code means"},
        {"show device active", "The device training uses now"},
        {"show device available", "Every detected compute device"},
        {"show training current", "The running training job"},
        {"show backend packs", "Installed GPU packs"},
        {"help", "List every command"},
        {"help <command>", "Forms and examples for one command"},
        {"clear", "Clear this transcript"},
        {"filter set <expression>", "Filter later show logs queries"},
        {"filter clear", "Remove the session filter"},
    };
    for (const auto& known : kKnown) {
        if (known.form == form) return std::string(known.description);
    }
    return std::string(fallback);
}

void AddForm(std::vector<CommandForm>& forms, std::string_view command,
             std::string_view description, std::string form_text, bool own_description) {
    // Expand a final "a|b|c" token without placeholders.
    const size_t last_space = form_text.rfind(' ');
    const std::string last =
        last_space == std::string::npos ? form_text : form_text.substr(last_space + 1);
    if (last.find('|') != std::string::npos && last.find('<') == std::string::npos &&
        last.find('.') == std::string::npos && last_space != std::string::npos) {
        const std::string head = form_text.substr(0, last_space + 1);
        size_t start = 0;
        while (start <= last.size()) {
            const size_t bar = last.find('|', start);
            const std::string alternative =
                last.substr(start, bar == std::string::npos ? std::string::npos : bar - start);
            if (!alternative.empty())
                AddForm(forms, command, description, head + alternative, own_description);
            if (bar == std::string::npos) break;
            start = bar + 1;
        }
        return;
    }
    // Forms like "show device active|available|..." with a placeholder after
    // them, or "show logs|errors|... ..." summaries, are left out: the
    // detailed forms cover them.
    if (form_text.find('|') != std::string::npos) return;
    for (const auto& existing : forms) {
        if (existing.form == form_text) return;
    }
    CommandForm form;
    form.form = form_text;
    const size_t placeholder = form_text.find_first_of("<[");
    form.complete = placeholder == std::string::npos;
    form.insert_text = form.complete ? form_text : form_text.substr(0, placeholder);
    form.description = own_description ? std::string(description)
                                       : FormDescription(form_text, description);
    form.command = std::string(command);
    forms.push_back(std::move(form));
}

}  // namespace

std::vector<CommandForm> BuildCommandForms(const std::vector<CommandSource>& sources) {
    std::vector<CommandForm> forms;
    for (const auto& source : sources) {
        bool found = false;
        bool in_forms = false;
        std::istringstream lines{std::string(source.detailed_help)};
        std::string line;
        while (std::getline(lines, line)) {
            if (line == "Forms:" || line == "Usage:") {
                in_forms = true;
                continue;
            }
            if (line.empty() || !std::isspace(static_cast<unsigned char>(line.front()))) {
                in_forms = false;
                continue;
            }
            if (!in_forms) continue;
            // "  show logs last <n>  Newest n events": two spaces start the
            // form's own description.
            const std::string_view entry = Trim(line);
            const size_t gap = entry.find("  ");
            const std::string_view form = Trim(entry.substr(0, gap));
            const std::string_view described =
                gap == std::string_view::npos ? source.description : Trim(entry.substr(gap));
            if (!StartsWith(form, source.name)) continue;
            AddForm(forms, source.name, described, std::string(form),
                    gap != std::string_view::npos);
            found = true;
        }
        if (!found)
            AddForm(forms, source.name, source.description, std::string(source.usage), false);
    }
    return forms;
}

std::vector<size_t> MatchForms(const std::vector<CommandForm>& forms, std::string_view input,
                               size_t max_results) {
    std::vector<size_t> matches;
    const std::string typed = Lower(TrimLeft(input));
    if (typed.empty()) return matches;
    for (size_t i = 0; i < forms.size() && matches.size() < max_results; ++i) {
        const std::string literal = Lower(forms[i].insert_text);
        // Still typing towards the form, and not already exactly it.
        if (StartsWith(literal, typed) && literal != typed) matches.push_back(i);
    }
    return matches;
}

std::optional<size_t> UsageFormFor(const std::vector<CommandForm>& forms,
                                   std::string_view input) {
    const std::string typed = Lower(TrimLeft(input));
    if (typed.empty()) return std::nullopt;
    // The longest form whose literal part the input has reached.
    std::optional<size_t> best;
    size_t best_length = 0;
    for (size_t i = 0; i < forms.size(); ++i) {
        std::string literal = Lower(forms[i].insert_text);
        while (!literal.empty() && literal.back() == ' ') literal.pop_back();
        if (literal.empty()) continue;
        const bool reached = StartsWith(typed, literal) &&
                             (typed.size() == literal.size() || typed[literal.size()] == ' ');
        if (reached && literal.size() > best_length) {
            best = i;
            best_length = literal.size();
        }
    }
    if (best) return best;
    const std::string name = CommandName(input);
    for (size_t i = 0; i < forms.size(); ++i) {
        if (forms[i].command == name) return i;
    }
    return std::nullopt;
}

std::string CommandName(std::string_view input) {
    const std::string_view trimmed = TrimLeft(input);
    const size_t end = trimmed.find_first_of(" \t");
    return Lower(trimmed.substr(0, end));
}

std::optional<EventLine> ParseEventLine(std::string_view text) {
    if (text.size() < 3 || text.front() != '#') return std::nullopt;
    EventLine event;
    size_t pos = 1;
    uint64_t sequence = 0;
    bool digits = false;
    while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) {
        sequence = sequence * 10 + static_cast<uint64_t>(text[pos] - '0');
        ++pos;
        digits = true;
    }
    if (!digits || pos >= text.size() || text[pos] != ' ') return std::nullopt;
    event.sequence = sequence;
    ++pos;
    const size_t ts_end = text.find(' ', pos);
    if (ts_end == std::string_view::npos) return std::nullopt;
    event.timestamp = std::string(text.substr(pos, ts_end - pos));
    if (event.timestamp.size() < 20 || event.timestamp.back() != 'Z') return std::nullopt;
    event.time = event.timestamp.substr(11, event.timestamp.size() - 12);
    pos = ts_end + 1;

    const size_t bar = text.find(" | ", pos);
    const std::string_view fields =
        text.substr(pos, bar == std::string_view::npos ? std::string_view::npos : bar - pos);
    if (bar != std::string_view::npos) event.message = std::string(text.substr(bar + 3));

    size_t start = 0;
    while (start < fields.size()) {
        size_t end = fields.find(' ', start);
        if (end == std::string_view::npos) end = fields.size();
        const std::string_view field = fields.substr(start, end - start);
        const size_t eq = field.find('=');
        if (eq != std::string_view::npos) {
            const std::string_view key = field.substr(0, eq);
            const std::string value(field.substr(eq + 1));
            if (key == "level") event.level = value;
            else if (key == "category") event.category = value;
            else if (key == "source") event.source = value;
            else {
                if (!event.extra.empty()) event.extra += ' ';
                event.extra += std::string(field);
            }
        }
        start = end + 1;
    }
    if (event.level.empty()) return std::nullopt;
    return event;
}

size_t EventLevelIndex(std::string_view level) {
    const std::string value = Lower(level);
    if (value == "trace") return 0;
    if (value == "debug") return 1;
    if (value == "warn" || value == "warning") return 3;
    if (value == "error" || value == "err") return 4;
    if (value == "critical") return 5;
    return 2;
}

std::optional<std::chrono::system_clock::time_point> ParseUtcTimestamp(std::string_view text) {
    // YYYY-MM-DDTHH:MM:SS[.mmm]Z
    if (text.size() < 20 || text[4] != '-' || text[7] != '-' || text[10] != 'T' ||
        text[13] != ':' || text[16] != ':' || text.back() != 'Z')
        return std::nullopt;
    const auto number = [&text](size_t pos, size_t len, int& out) {
        out = 0;
        for (size_t i = pos; i < pos + len; ++i) {
            if (!std::isdigit(static_cast<unsigned char>(text[i]))) return false;
            out = out * 10 + (text[i] - '0');
        }
        return true;
    };
    int year = 0, month = 0, day = 0, hour = 0, minute = 0, second = 0, millis = 0;
    if (!number(0, 4, year) || !number(5, 2, month) || !number(8, 2, day) ||
        !number(11, 2, hour) || !number(14, 2, minute) || !number(17, 2, second))
        return std::nullopt;
    if (text.size() >= 24 && text[19] == '.' && !number(20, 3, millis)) return std::nullopt;
    if (month < 1 || month > 12 || day < 1 || day > 31) return std::nullopt;
    // Days from civil (Howard Hinnant), valid for the proleptic Gregorian calendar.
    const int y = year - (month <= 2 ? 1 : 0);
    const int era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy =
        static_cast<unsigned>((153 * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1);
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    const long long days = static_cast<long long>(era) * 146097 + static_cast<long long>(doe) - 719468;
    const long long ms =
        ((days * 24 + hour) * 60 + minute) * 60000LL + second * 1000LL + millis;
    return std::chrono::system_clock::time_point(std::chrono::milliseconds(ms));
}

std::optional<std::string> PrettySummary(std::string_view text) {
    constexpr std::string_view kPrefix = "Logs: ";
    if (!StartsWith(text, kPrefix)) return std::nullopt;
    std::string out;
    std::string_view rest = text.substr(kPrefix.size());
    size_t start = 0;
    while (start < rest.size()) {
        size_t end = rest.find(' ', start);
        if (end == std::string_view::npos) end = rest.size();
        const std::string_view field = rest.substr(start, end - start);
        const size_t eq = field.find('=');
        if (eq == std::string_view::npos) return std::nullopt;
        std::string key(field.substr(0, eq));
        if (out.empty() && !key.empty())
            key[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(key[0])));
        if (!out.empty()) out += " \xC2\xB7 ";
        out += key + " " + std::string(field.substr(eq + 1));
        start = end + 1;
    }
    return out;
}

BlockStatus BuildBlockStatus(bool running, bool success, bool cancelled,
                             double running_seconds, size_t event_rows) {
    BlockStatus status;
    if (running) {
        status.kind = BlockStatus::Kind::Running;
        status.text = "Running \xC2\xB7 " + FormatElapsed(running_seconds);
    } else if (cancelled) {
        status.kind = BlockStatus::Kind::Cancelled;
        status.text = "cancelled";
    } else if (!success) {
        status.kind = BlockStatus::Kind::Error;
        status.text = "error";
    } else if (event_rows > 0) {
        status.text = std::to_string(event_rows) + (event_rows == 1 ? " row" : " rows");
    }
    return status;
}

std::string FormatElapsed(double seconds) {
    if (seconds < 0) seconds = 0;
    char buffer[32];
    if (seconds < 10.0) {
        std::snprintf(buffer, sizeof(buffer), "%.1f s", seconds);
    } else if (seconds < 60.0) {
        std::snprintf(buffer, sizeof(buffer), "%d s", static_cast<int>(seconds));
    } else {
        const int total = static_cast<int>(seconds);
        std::snprintf(buffer, sizeof(buffer), "%d min %02d s", total / 60, total % 60);
    }
    return buffer;
}

}  // namespace cyxwiz::commands
