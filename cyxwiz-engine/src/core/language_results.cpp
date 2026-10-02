#include "language_results.h"

#include <nlohmann/json.hpp>

#include <cctype>

namespace cyxwiz::lang {

namespace {
using nlohmann::json;

json ParseOr(const std::string& text, json fallback) {
    try {
        return json::parse(text);
    } catch (...) {
        return fallback;
    }
}

std::string Str(const json& o, const char* key) {
    auto it = o.find(key);
    return it != o.end() && it->is_string() ? it->get<std::string>() : std::string();
}

int Int(const json& o, const char* key) {
    auto it = o.find(key);
    return it != o.end() && it->is_number_integer() ? it->get<int>() : 0;
}
}  // namespace

std::vector<Completion> ParseCompletions(const std::string& text) {
    std::vector<Completion> out;
    const json doc = ParseOr(text, json::array());
    if (!doc.is_array()) return out;
    for (const auto& item : doc) {
        if (!item.is_object()) continue;
        out.push_back({Str(item, "name"), Str(item, "complete"), Str(item, "kind"), Str(item, "detail")});
    }
    return out;
}

Description ParseDescription(const std::string& text) {
    const json o = ParseOr(text, json::object());
    if (!o.is_object()) return {};
    return {Str(o, "name"), Str(o, "kind"), Str(o, "signature"), Str(o, "doc"), Str(o, "module")};
}

Hover ParseHover(const std::string& text) {
    const json o = ParseOr(text, json::object());
    if (!o.is_object()) return {};
    Hover h;
    h.name = Str(o, "name");
    h.kind = Str(o, "kind");
    h.signature = Str(o, "signature");
    h.type = Str(o, "type");
    h.doc = Str(o, "doc");
    h.module = Str(o, "module");
    h.path = Str(o, "path");
    h.line = Int(o, "line");
    return h;
}

std::vector<Signature> ParseSignatures(const std::string& text) {
    std::vector<Signature> out;
    const json doc = ParseOr(text, json::array());
    if (!doc.is_array()) return out;
    for (const auto& item : doc) {
        if (!item.is_object()) continue;
        Signature s;
        s.name = Str(item, "name");
        s.doc = Str(item, "doc");
        auto it = item.find("index");
        s.index = it != item.end() && it->is_number_integer() ? it->get<int>() : -1;
        auto params = item.find("params");
        if (params != item.end() && params->is_array())
            for (const auto& p : *params)
                if (p.is_string()) s.params.push_back(p.get<std::string>());
        out.push_back(std::move(s));
    }
    return out;
}

std::vector<Location> ParseLocations(const std::string& text) {
    std::vector<Location> out;
    const json doc = ParseOr(text, json::array());
    if (!doc.is_array()) return out;
    for (const auto& item : doc) {
        if (!item.is_object()) continue;
        Location l;
        l.name = Str(item, "name");
        l.path = Str(item, "path");
        l.line = Int(item, "line");
        l.column = Int(item, "column");
        l.module = Str(item, "module");
        auto it = item.find("in_source");
        l.in_source = it != item.end() && it->is_boolean() && it->get<bool>();
        out.push_back(std::move(l));
    }
    return out;
}

std::vector<Problem> ParseProblems(const std::string& text) {
    std::vector<Problem> out;
    const json doc = ParseOr(text, json::array());
    if (!doc.is_array()) return out;
    for (const auto& item : doc) {
        if (!item.is_object()) continue;
        Problem p;
        p.line = Int(item, "line");
        p.column = Int(item, "column");
        p.error = Str(item, "severity") == "error";
        p.message = Str(item, "message");
        p.code = Str(item, "code");
        if (p.line > 0) out.push_back(std::move(p));
    }
    return out;
}

KindChip ChipFor(const std::string& kind) {
    if (kind == "function") return {"f", 0};
    if (kind == "class") return {"C", 1};
    if (kind == "module") return {"M", 2};
    if (kind == "variable" || kind == "property") return {"v", 3};
    if (kind == "keyword") return {"k", 4};
    if (kind == "path") return {"/", 5};
    return {"\xC2\xB7", 5};
}

std::string ProblemSummary(const std::vector<Problem>& problems) {
    int errors = 0, warnings = 0;
    for (const auto& p : problems) (p.error ? errors : warnings)++;
    if (errors == 0 && warnings == 0) return "No problems";
    std::string out;
    if (errors) out += std::to_string(errors) + (errors == 1 ? " error" : " errors");
    if (warnings) out += std::string(out.empty() ? "" : ", ") + std::to_string(warnings) + (warnings == 1 ? " warning" : " warnings");
    return out;
}

std::pair<int, int> ProblemRange(const std::string& line_text, const Problem& problem) {
    const int n = static_cast<int>(line_text.size());
    auto word = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; };
    int column = problem.column;
    if (problem.code == "UnusedImport" || problem.code == "RedefinedWhileUnused") {
        // "'pkg.mod.name' imported but unused": the last part of the quoted name.
        const size_t q1 = problem.message.find('\'');
        const size_t q2 = q1 == std::string::npos ? q1 : problem.message.find('\'', q1 + 1);
        if (q2 != std::string::npos) {
            std::string name = problem.message.substr(q1 + 1, q2 - q1 - 1);
            const size_t as = name.find(" as ");
            if (as != std::string::npos) name = name.substr(as + 4);
            const size_t dot = name.rfind('.');
            if (dot != std::string::npos) name = name.substr(dot + 1);
            for (size_t at = line_text.find(name); at != std::string::npos; at = line_text.find(name, at + 1)) {
                const bool left = at == 0 || !word(line_text[at - 1]);
                const size_t right_at = at + name.size();
                const bool right = right_at >= line_text.size() || !word(line_text[right_at]);
                if (left && right) return {static_cast<int>(at), static_cast<int>(right_at)};
            }
        }
    }
    if (column >= n) return {column, column + 1};
    int end = column;
    while (end < n && (word(line_text[static_cast<size_t>(end)]) || line_text[static_cast<size_t>(end)] == '.')) ++end;
    return {column, end > column ? end : column + 1};
}

}  // namespace cyxwiz::lang
