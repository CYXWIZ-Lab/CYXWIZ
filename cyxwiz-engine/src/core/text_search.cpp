#include "text_search.h"

#include <algorithm>
#include <cctype>
#include <regex>

namespace cyxwiz::textsearch {

namespace {
bool IsWordChar(char c) {
    const unsigned char u = static_cast<unsigned char>(c);
    return std::isalnum(u) || c == '_' || u >= 0x80;
}

bool IsWholeWord(const std::string& text, size_t pos, size_t len) {
    const bool start_ok = pos == 0 || !IsWordChar(text[pos - 1]);
    const bool end_ok = pos + len >= text.size() || !IsWordChar(text[pos + len]);
    return start_ok && end_ok;
}

std::string Lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::optional<std::regex> Compile(const std::string& pattern, const Options& options, std::string* error) {
    try {
        auto flags = std::regex::ECMAScript;
        if (!options.case_sensitive) flags |= std::regex::icase;
        return std::regex(pattern, flags);
    } catch (const std::regex_error& e) {
        if (error) *error = std::string("Invalid regular expression: ") + e.what();
        return std::nullopt;
    }
}

bool IsUtf8Continuation(char c) { return (static_cast<unsigned char>(c) & 0xC0) == 0x80; }
}  // namespace

std::vector<Match> FindAll(const std::string& text, const std::string& pattern, const Options& options,
                           std::string* error) {
    std::vector<Match> out;
    if (pattern.empty()) return out;
    if (options.regex) {
        const auto re = Compile(pattern, options, error);
        if (!re) return out;
        for (auto it = std::sregex_iterator(text.begin(), text.end(), *re); it != std::sregex_iterator(); ++it) {
            const size_t pos = static_cast<size_t>(it->position(0));
            const size_t len = static_cast<size_t>(it->length(0));
            if (len == 0) continue;
            if (options.whole_word && !IsWholeWord(text, pos, len)) continue;
            out.push_back({pos, len});
        }
        return out;
    }
    const std::string hay = options.case_sensitive ? text : Lower(text);
    const std::string needle = options.case_sensitive ? pattern : Lower(pattern);
    for (size_t pos = hay.find(needle); pos != std::string::npos; pos = hay.find(needle, pos + 1)) {
        if (options.whole_word && !IsWholeWord(text, pos, needle.size())) continue;
        if (!out.empty() && pos < out.back().pos + out.back().len) continue;  // no overlapping matches
        out.push_back({pos, needle.size()});
    }
    return out;
}

std::optional<Match> FindNext(const std::string& text, const std::string& pattern, size_t from, const Options& options,
                              std::string* error) {
    const auto all = FindAll(text, pattern, options, error);
    if (all.empty()) return std::nullopt;
    for (const auto& m : all)
        if (m.pos >= from) return m;
    return all.front();
}

std::optional<Match> FindPrevious(const std::string& text, const std::string& pattern, size_t before,
                                  const Options& options, std::string* error) {
    const auto all = FindAll(text, pattern, options, error);
    if (all.empty()) return std::nullopt;
    for (auto it = all.rbegin(); it != all.rend(); ++it)
        if (it->pos + it->len <= before) return *it;
    return all.back();
}

std::optional<std::string> ReplacementFor(const std::string& matched, const std::string& pattern,
                                          const std::string& replacement, const Options& options) {
    const auto all = FindAll(matched, pattern, options);
    if (all.size() != 1 || all.front().pos != 0 || all.front().len != matched.size()) return std::nullopt;
    if (!options.regex) return replacement;
    const auto re = Compile(pattern, options, nullptr);
    if (!re) return std::nullopt;
    return std::regex_replace(matched, *re, replacement, std::regex_constants::format_first_only);
}

std::string ReplaceAll(const std::string& text, const std::string& pattern, const std::string& replacement,
                       const Options& options, int* count, std::string* error) {
    const auto all = FindAll(text, pattern, options, error);
    if (count) *count = static_cast<int>(all.size());
    if (all.empty()) return text;
    std::optional<std::regex> re;
    if (options.regex) re = Compile(pattern, options, nullptr);
    std::string out;
    out.reserve(text.size());
    size_t last = 0;
    for (const auto& m : all) {
        out.append(text, last, m.pos - last);
        const std::string matched = text.substr(m.pos, m.len);
        out += re ? std::regex_replace(matched, *re, replacement, std::regex_constants::format_first_only) : replacement;
        last = m.pos + m.len;
    }
    out.append(text, last, std::string::npos);
    return out;
}

Position ToPosition(const std::string& text, size_t offset, int tab_size) {
    Position p;
    offset = std::min(offset, text.size());
    size_t line_start = 0;
    for (size_t i = 0; i < offset; ++i) {
        if (text[i] == '\n') {
            ++p.line;
            line_start = i + 1;
        }
    }
    int column = 0;
    for (size_t i = line_start; i < offset; ++i) {
        const char c = text[i];
        if (c == '\t')
            column = (column / tab_size + 1) * tab_size;
        else if (!IsUtf8Continuation(c))
            ++column;
    }
    p.column = column;
    return p;
}

size_t ToOffset(const std::string& text, Position position, int tab_size) {
    size_t i = 0;
    for (int line = 0; line < position.line && i < text.size(); ++i)
        if (text[i] == '\n') ++line;
    int column = 0;
    while (i < text.size() && text[i] != '\n') {
        int next = column;
        if (text[i] == '\t')
            next = (column / tab_size + 1) * tab_size;
        else
            ++next;
        if (next > position.column) break;
        column = next;
        ++i;
        while (i < text.size() && IsUtf8Continuation(text[i])) ++i;
    }
    return i;
}

}  // namespace cyxwiz::textsearch
