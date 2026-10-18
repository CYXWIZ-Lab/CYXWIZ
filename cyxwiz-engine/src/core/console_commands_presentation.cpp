#include "console_commands_presentation.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <sstream>

#include <nlohmann/json.hpp>

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

std::string EventFieldsLine(const RuntimeLogEvent& event) {
    std::vector<std::string> parts;
    const auto add = [&parts](const std::string& key, const std::string& value) {
        if (!value.empty()) parts.push_back(key + " " + value);
    };
    if (event.source != "cyxwiz") add("source", event.source);  // The Engine's own logger says nothing.
    std::string codes = event.primary_error_code;
    for (const auto& code : event.issue_codes)
        if (code != event.primary_error_code) codes += (codes.empty() ? "" : ",") + code;
    add("code", codes);
    add("run", event.run_id);
    if (event.task_id != 0) add("task", std::to_string(event.task_id));
    if (event.node_id >= 0) add("node", std::to_string(event.node_id));
    add("backend", event.backend);
    if (event.device_id >= 0) add("device", std::to_string(event.device_id));
    add("dataset", event.dataset_name);
    for (const auto& [key, value] : event.details) add(key, value);
    std::string line;
    for (const auto& part : parts) line += (line.empty() ? "" : " \xC2\xB7 ") + part;
    return line;
}

namespace {
// Two-space indents; an array of plain values (token ids) stays on one line.
void PrettyJson(const nlohmann::json& value, int depth, std::string& out) {
    const std::string pad(static_cast<size_t>(depth) * 2, ' ');
    const std::string inner(static_cast<size_t>(depth + 1) * 2, ' ');
    const auto plain = [](const nlohmann::json& item) { return !item.is_object() && !item.is_array(); };
    if (value.is_object() && !value.empty()) {
        out += "{\n";
        size_t i = 0;
        for (auto it = value.begin(); it != value.end(); ++it, ++i) {
            out += inner + nlohmann::json(it.key()).dump() + ": ";
            PrettyJson(it.value(), depth + 1, out);
            out += i + 1 < value.size() ? ",\n" : "\n";
        }
        out += pad + "}";
    } else if (value.is_array() && !value.empty() && !std::all_of(value.begin(), value.end(), plain)) {
        out += "[\n";
        for (size_t i = 0; i < value.size(); ++i) {
            out += inner;
            PrettyJson(value[i], depth + 1, out);
            out += i + 1 < value.size() ? ",\n" : "\n";
        }
        out += pad + "]";
    } else if (value.is_array()) {
        out += "[";
        for (size_t i = 0; i < value.size(); ++i)
            out += (i ? ", " : "") + value[i].dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
        out += "]";
    } else {
        out += value.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
    }
}
}  // namespace

std::optional<JsonPayload> FindJsonPayload(const std::string& message) {
    // The payload ends the message: try each '{' or '[' from the first one on.
    for (size_t start = message.find_first_of("{["); start != std::string::npos;
         start = message.find_first_of("{[", start + 1)) {
        nlohmann::json document;
        try {
            document = nlohmann::json::parse(message.begin() + static_cast<std::ptrdiff_t>(start), message.end());
        } catch (const nlohmann::json::exception&) {
            continue;
        }
        if (!document.is_object() && !document.is_array()) continue;
        JsonPayload payload;
        payload.before = std::string(Trim(std::string_view(message).substr(0, start)));
        PrettyJson(document, 0, payload.pretty);
        payload.bytes = message.size() - start;
        payload.entries = document.size();
        return payload;
    }
    return std::nullopt;
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
