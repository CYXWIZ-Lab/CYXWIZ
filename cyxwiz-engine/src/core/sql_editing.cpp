#include "sql_editing.h"

#include <algorithm>
#include <cctype>
#include <set>

namespace cyxwiz::sqledit {

namespace {
// Reserved words: they cannot be unquoted column names, so upper-casing them
// never changes a result's column names (DuckDB keeps a name as written).
const std::set<std::string>& ReservedWords() {
    static const std::set<std::string> words = {
        "ALL", "AND", "AS", "ASC", "BETWEEN", "BY", "CASE", "CAST", "COLLATE", "CROSS", "DESC", "DISTINCT", "ELSE",
        "END", "EXCEPT", "EXCLUDE", "EXISTS", "FALSE", "FROM", "FULL", "GROUP", "HAVING", "ILIKE", "IN", "INNER",
        "INTERSECT", "INTERVAL", "IS", "JOIN", "LATERAL", "LIKE", "LIMIT", "MATERIALIZED", "NATURAL", "NOT", "NULL",
        "NULLS", "OFFSET", "ON", "OR", "ORDER", "ORDINALITY", "OUTER", "OVER", "PARTITION", "QUALIFY", "RECURSIVE",
        "SELECT", "SIMILAR", "THEN", "TRUE", "TRY_CAST", "UNION", "USING", "VALUES", "WHEN", "WHERE", "WINDOW",
        "WITH"};
    return words;
}

// Common DuckDB functions for completion (lower case, as usually written).
const std::vector<std::string>& Functions() {
    static const std::vector<std::string> names = {
        "abs", "any_value", "array_agg", "array_extract", "avg", "bool_and", "bool_or", "ceil", "chr", "coalesce",
        "concat", "concat_ws", "contains", "count", "count_star", "date_part", "date_trunc", "ends_with", "epoch",
        "first", "floor", "greatest", "hash", "ifnull", "instr", "last", "lcase", "least", "left", "len", "length",
        "levenshtein", "list_aggregate", "list_contains", "list_distinct", "list_filter", "list_sort",
        "list_transform", "list_value", "lower", "lpad", "ltrim", "max", "md5", "median", "min", "mode", "nullif",
        "ord", "position", "prefix", "quantile_cont", "random", "range", "regexp_extract", "regexp_extract_all",
        "regexp_full_match", "regexp_matches", "regexp_replace", "regexp_split_to_array", "repeat", "replace",
        "reverse", "right", "round", "row_number", "rpad", "rtrim", "sha256", "split_part", "sqrt", "starts_with",
        "string_agg", "string_split", "strip_accents", "strlen", "strpos", "struct_extract", "substring", "sum",
        "trim", "ucase", "unnest", "upper", "dense_rank", "rank", "lag", "lead", "ntile", "generate_series",
        "list_slice", "array_length", "translate", "format", "printf", "strftime", "strptime", "to_json",
        "json_extract", "json_extract_string"};
    return names;
}

std::string Lower(std::string_view text) {
    std::string out(text);
    for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

std::string Upper(std::string_view text) {
    std::string out(text);
    for (char& c : out) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return out;
}

bool IsWordChar(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }

bool StartsWithNoCase(std::string_view text, std::string_view prefix) {
    if (prefix.size() > text.size()) return false;
    for (size_t i = 0; i < prefix.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(text[i])) != std::tolower(static_cast<unsigned char>(prefix[i]))) {
            return false;
        }
    }
    return true;
}

// Names the query defines with "AS name" (outside strings and comments).
std::vector<std::string> DefinedNames(std::string_view sql) {
    std::vector<std::string> names;
    std::string previous;
    for (size_t i = 0; i < sql.size();) {
        const char c = sql[i];
        if (c == '-' && i + 1 < sql.size() && sql[i + 1] == '-') {
            while (i < sql.size() && sql[i] != '\n') ++i;
        } else if (c == '/' && i + 1 < sql.size() && sql[i + 1] == '*') {
            const size_t end = sql.find("*/", i + 2);
            i = end == std::string_view::npos ? sql.size() : end + 2;
        } else if (c == '\'' || c == '"') {
            size_t k = i + 1;
            while (k < sql.size() && !(sql[k] == c && !(k + 1 < sql.size() && sql[k + 1] == c))) k += sql[k] == c ? 2 : 1;
            i = k + 1;
            previous.clear();
        } else if (IsWordChar(c)) {
            size_t k = i;
            while (k < sql.size() && IsWordChar(sql[k])) ++k;
            const std::string word(sql.substr(i, k - i));
            if (Upper(previous) == "AS" && !std::isdigit(static_cast<unsigned char>(word[0])) &&
                !ReservedWords().count(Upper(word)) && std::find(names.begin(), names.end(), word) == names.end()) {
                names.push_back(word);
            }
            previous = word;
            i = k;
        } else {
            if (!std::isspace(static_cast<unsigned char>(c))) previous.clear();
            ++i;
        }
    }
    return names;
}
}  // namespace

std::string WordBefore(std::string_view line, size_t column) {
    column = std::min(column, line.size());
    size_t start = column;
    while (start > 0 && IsWordChar(line[start - 1])) --start;
    return std::string(line.substr(start, column - start));
}

std::vector<Completion> SqlCompletions(std::string_view prefix, const std::vector<std::string>& input_columns,
                                       std::string_view sql, const sqlstep::SqlOutline& outline, size_t limit) {
    std::vector<Completion> out;
    std::set<std::string> seen;
    const auto add = [&](const std::string& text, CompletionKind kind, const char* detail) {
        if (out.size() >= limit || !StartsWithNoCase(text, prefix) || Lower(text) == Lower(prefix)) return;
        if (!seen.insert(Lower(text)).second) return;
        out.push_back({text, kind, detail});
    };
    for (const auto& column : input_columns) add(column, CompletionKind::Column, "input column");
    for (const auto& part : outline.parts) {
        if (part.name != "result" && part.name != "query") add(part.name, CompletionKind::Part, "part of this query");
    }
    if (prefix.empty()) return out;
    for (const auto& name : DefinedNames(sql)) add(name, CompletionKind::Name, "named in this query");
    for (const auto& function : Functions()) add(function, CompletionKind::Function, "DuckDB function");
    for (const auto& keyword : ReservedWords()) add(keyword, CompletionKind::Keyword, "keyword");
    return out;
}

std::string FormatSqlKeywords(std::string_view sql) {
    std::string out;
    out.reserve(sql.size());
    for (size_t i = 0; i < sql.size();) {
        const char c = sql[i];
        if (c == '-' && i + 1 < sql.size() && sql[i + 1] == '-') {
            const size_t end = std::min(sql.find('\n', i), sql.size());
            out.append(sql.substr(i, end - i));
            i = end;
        } else if (c == '/' && i + 1 < sql.size() && sql[i + 1] == '*') {
            const size_t close = sql.find("*/", i + 2);
            const size_t end = close == std::string_view::npos ? sql.size() : close + 2;
            out.append(sql.substr(i, end - i));
            i = end;
        } else if (c == '\'' || c == '"') {
            size_t k = i + 1;
            while (k < sql.size()) {
                if (sql[k] == c) {
                    if (k + 1 < sql.size() && sql[k + 1] == c) {
                        k += 2;
                        continue;
                    }
                    break;
                }
                ++k;
            }
            const size_t end = std::min(k + 1, sql.size());
            out.append(sql.substr(i, end - i));
            i = end;
        } else if (IsWordChar(c) && (i == 0 || !IsWordChar(sql[i - 1]))) {
            size_t k = i;
            while (k < sql.size() && IsWordChar(sql[k])) ++k;
            const std::string word(sql.substr(i, k - i));
            const std::string upper = Upper(word);
            // Reserved words only; function names such as left( and right(
            // are not in the list, so a call is never upper-cased.
            out += ReservedWords().count(upper) ? upper : word;
            i = k;
        } else if (c == '\n' || c == '\r') {
            // Trailing spaces before the line break go.
            while (!out.empty() && (out.back() == ' ' || out.back() == '\t')) out.pop_back();
            out += c;
            ++i;
        } else {
            out += c;
            ++i;
        }
    }
    while (!out.empty() && (out.back() == ' ' || out.back() == '\t')) out.pop_back();
    return out;
}

std::vector<DiffLine> DiffLines(std::string_view before, std::string_view after) {
    const auto split = [](std::string_view text) {
        std::vector<std::string> lines;
        size_t start = 0;
        while (start <= text.size()) {
            size_t end = text.find('\n', start);
            if (end == std::string_view::npos) end = text.size();
            std::string line(text.substr(start, end - start));
            if (!line.empty() && line.back() == '\r') line.pop_back();
            lines.push_back(std::move(line));
            if (end == text.size()) break;
            start = end + 1;
        }
        return lines;
    };
    const auto a = split(before);
    const auto b = split(after);
    // Trim the common head and tail so the table stays small for edits.
    size_t head = 0;
    while (head < a.size() && head < b.size() && a[head] == b[head]) ++head;
    size_t tail = 0;
    while (tail < a.size() - head && tail < b.size() - head && a[a.size() - 1 - tail] == b[b.size() - 1 - tail]) ++tail;
    const size_t n = a.size() - head - tail;
    const size_t m = b.size() - head - tail;
    std::vector<std::vector<int>> lcs(n + 1, std::vector<int>(m + 1, 0));
    for (size_t i = n; i-- > 0;) {
        for (size_t j = m; j-- > 0;) {
            lcs[i][j] = a[head + i] == b[head + j] ? lcs[i + 1][j + 1] + 1 : std::max(lcs[i + 1][j], lcs[i][j + 1]);
        }
    }
    std::vector<DiffLine> out;
    for (size_t i = 0; i < head; ++i) {
        out.push_back({DiffKind::Same, static_cast<int>(i + 1), static_cast<int>(i + 1), a[i]});
    }
    size_t i = 0, j = 0;
    while (i < n || j < m) {
        if (i < n && j < m && a[head + i] == b[head + j]) {
            out.push_back({DiffKind::Same, static_cast<int>(head + i + 1), static_cast<int>(head + j + 1), a[head + i]});
            ++i;
            ++j;
        } else if (i < n && (j == m || lcs[i + 1][j] >= lcs[i][j + 1])) {
            // The saved line first, then what replaced it.
            out.push_back({DiffKind::Removed, static_cast<int>(head + i + 1), 0, a[head + i]});
            ++i;
        } else {
            out.push_back({DiffKind::Added, 0, static_cast<int>(head + j + 1), b[head + j]});
            ++j;
        }
    }
    for (size_t k = 0; k < tail; ++k) {
        const size_t ai = a.size() - tail + k;
        const size_t bi = b.size() - tail + k;
        out.push_back({DiffKind::Same, static_cast<int>(ai + 1), static_cast<int>(bi + 1), a[ai]});
    }
    return out;
}

}  // namespace cyxwiz::sqledit
