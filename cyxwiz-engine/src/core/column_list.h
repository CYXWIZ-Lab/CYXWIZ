#pragma once

// A list of column names held in one text parameter (a step's `columns`,
// `feature_cols`, `exclude_columns`): names separated by commas, spaces
// around each name ignored. A name that holds a comma, starts with a double
// quote or has spaces at its ends is written in double quotes, with a quote
// inside it doubled, as in SQL and CSV:
//
//   age, "income, net", " padded ", "say ""hi"""
//
// Every step reads its list with ParseColumnList and every writer that
// builds one from names uses FormatColumnList, so a column of any name can
// be listed.

#include <string>
#include <vector>

namespace cyxwiz {

namespace column_list_detail {
inline bool IsSpace(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

inline std::string Trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && IsSpace(s[b])) ++b;
    while (e > b && IsSpace(s[e - 1])) --e;
    return s.substr(b, e - b);
}
}  // namespace column_list_detail

// The names in `text`, in order; empty entries are skipped, but "" is a column
// with an empty name. A quoted name keeps its spaces and commas; a quote left
// open runs to the end of the text.
inline std::vector<std::string> ParseColumnList(const std::string& text) {
    using column_list_detail::IsSpace;
    std::vector<std::string> out;
    const size_t n = text.size();
    size_t i = 0;
    while (i < n) {
        while (i < n && IsSpace(text[i])) ++i;
        std::string name;
        const bool quoted = i < n && text[i] == '"';
        if (quoted) {
            ++i;
            while (i < n) {
                if (text[i] == '"') {
                    if (i + 1 < n && text[i + 1] == '"') {
                        name += '"';
                        i += 2;
                        continue;
                    }
                    ++i;
                    break;
                }
                name += text[i++];
            }
            // Text between the closing quote and the comma belongs to the name.
            const size_t comma = text.find(',', i);
            const size_t end = comma == std::string::npos ? n : comma;
            name += column_list_detail::Trim(text.substr(i, end - i));
            i = end;
        } else {
            const size_t comma = text.find(',', i);
            const size_t end = comma == std::string::npos ? n : comma;
            name = column_list_detail::Trim(text.substr(i, end - i));
            i = end;
        }
        if (!name.empty() || quoted) out.push_back(std::move(name));  // "" names an empty column
        if (i < n) ++i;  // the comma
    }
    return out;
}

// True when a name must be quoted to come back from ParseColumnList as itself.
inline bool ColumnNeedsQuotes(const std::string& name) {
    using column_list_detail::IsSpace;
    return name.empty() || name.find(',') != std::string::npos || name.front() == '"' || IsSpace(name.front()) ||
           IsSpace(name.back());
}

// The list text for `names` (separated by `separator`, a comma with or without
// spaces; quoted where needed), which ParseColumnList reads back as the same
// names.
inline std::string FormatColumnList(const std::vector<std::string>& names, const char* separator = ",") {
    std::string out;
    bool first = true;
    for (const auto& name : names) {
        if (!first) out += separator;
        first = false;
        if (!ColumnNeedsQuotes(name)) {
            out += name;
            continue;
        }
        out += '"';
        for (const char c : name) {
            if (c == '"') out += '"';
            out += c;
        }
        out += '"';
    }
    return out;
}

}  // namespace cyxwiz
