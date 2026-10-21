#include "sql_highlight.h"

#include <cctype>
#include <string>

namespace cyxwiz::editor {

namespace {
std::string Upper(std::string_view word) {
    std::string out(word);
    for (char& c : out) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return out;
}

bool In(const std::string& word, std::initializer_list<const char*> list) {
    for (const char* w : list)
        if (word == w) return true;
    return false;
}

bool IsKeyword(const std::string& w) {
    return In(w, {"ALL", "AND", "ANTI", "ANY", "AS", "ASC", "ASOF", "BETWEEN", "BY", "CASE", "CAST", "COLLATE",
                  "CROSS", "CUBE", "DESC", "DISTINCT", "ELSE", "END", "EXCEPT", "EXCLUDE", "EXISTS", "FILTER",
                  "FIRST", "FOLLOWING", "FOR", "FROM", "FULL", "GROUP", "GROUPING", "HAVING", "ILIKE", "IN",
                  "INNER", "INTERSECT", "INTERVAL", "IS", "JOIN", "LAST", "LATERAL", "LEFT", "LIKE", "LIMIT",
                  "MATERIALIZED", "NATURAL", "NOT", "NULLS", "OFFSET", "ON", "OR", "ORDER", "ORDINALITY", "OUTER",
                  "OVER", "PARTITION", "PIVOT", "POSITIONAL", "PRECEDING", "QUALIFY", "RANGE", "RECURSIVE",
                  "RENAME", "REPLACE", "RIGHT", "ROLLUP", "ROW", "ROWS", "SELECT", "SEMI", "SETS", "SIMILAR",
                  "TABLESAMPLE", "THEN", "TRY_CAST", "UNBOUNDED", "UNION", "UNNEST", "UNPIVOT", "USING",
                  "VALUES", "WHEN", "WHERE", "WINDOW", "WITH", "WITHIN"});
}

bool IsType(const std::string& w) {
    return In(w, {"BIGINT", "BLOB", "BOOLEAN", "BOOL", "DATE", "DECIMAL", "DOUBLE", "FLOAT", "HUGEINT",
                  "INTEGER", "INT", "INT8", "INT4", "INT2", "LIST", "MAP", "NUMERIC", "REAL", "SMALLINT",
                  "STRUCT", "TEXT", "TIME", "TIMESTAMP", "TINYINT", "UBIGINT", "UINTEGER", "UUID", "VARCHAR",
                  "STRING", "JSON"});
}

bool IsConstant(const std::string& w) { return In(w, {"TRUE", "FALSE", "NULL"}); }

bool IsWordChar(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }

void Push(std::vector<Span>& spans, size_t start, size_t end, TokenKind kind) {
    if (end > start) spans.push_back({static_cast<int>(start), static_cast<int>(end - start), kind});
}

// End of a '...' string from `from` (just after the opening quote; '' is an
// escaped quote), or npos when it stays open past the line.
size_t StringEnd(std::string_view line, size_t from) {
    for (size_t i = from; i < line.size(); ++i) {
        if (line[i] != '\'') continue;
        if (i + 1 < line.size() && line[i + 1] == '\'') {
            ++i;
            continue;
        }
        return i + 1;
    }
    return std::string_view::npos;
}
}  // namespace

LineState HighlightSqlLine(std::string_view line, LineState in, std::vector<Span>& spans) {
    spans.clear();
    size_t pos = 0;
    if (in.triple == '*') {
        const size_t close = line.find("*/");
        if (close == std::string_view::npos) {
            Push(spans, 0, line.size(), TokenKind::Comment);
            return in;
        }
        Push(spans, 0, close + 2, TokenKind::Comment);
        pos = close + 2;
    } else if (in.triple == '\'') {
        const size_t end = StringEnd(line, 0);
        if (end == std::string_view::npos) {
            Push(spans, 0, line.size(), TokenKind::String);
            return in;
        }
        Push(spans, 0, end, TokenKind::String);
        pos = end;
    }

    while (pos < line.size()) {
        const char c = line[pos];
        if (c == ' ' || c == '\t') {
            ++pos;
            continue;
        }
        if (c == '-' && pos + 1 < line.size() && line[pos + 1] == '-') {
            Push(spans, pos, line.size(), TokenKind::Comment);
            return {};
        }
        if (c == '/' && pos + 1 < line.size() && line[pos + 1] == '*') {
            const size_t close = line.find("*/", pos + 2);
            if (close == std::string_view::npos) {
                Push(spans, pos, line.size(), TokenKind::Comment);
                return {'*'};
            }
            Push(spans, pos, close + 2, TokenKind::Comment);
            pos = close + 2;
            continue;
        }
        if (c == '\'') {
            const size_t end = StringEnd(line, pos + 1);
            if (end == std::string_view::npos) {
                Push(spans, pos, line.size(), TokenKind::String);
                return {'\''};
            }
            Push(spans, pos, end, TokenKind::String);
            pos = end;
            continue;
        }
        if (c == '"') {  // a quoted name stays plain text
            const size_t end = line.find('"', pos + 1);
            pos = end == std::string_view::npos ? line.size() : end + 1;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(c)) && (pos == 0 || !IsWordChar(line[pos - 1]))) {
            size_t end = pos;
            while (end < line.size() && (std::isdigit(static_cast<unsigned char>(line[end])) || line[end] == '.')) ++end;
            Push(spans, pos, end, TokenKind::Number);
            pos = end;
            continue;
        }
        if (IsWordChar(c)) {
            size_t end = pos;
            while (end < line.size() && IsWordChar(line[end])) ++end;
            const std::string word = Upper(line.substr(pos, end - pos));
            size_t next = end;
            while (next < line.size() && (line[next] == ' ' || line[next] == '\t')) ++next;
            const bool call = next < line.size() && line[next] == '(';
            TokenKind kind = TokenKind::Default;
            // Words that are both keywords and functions: replace(...), left(...).
            if (call && In(word, {"REPLACE", "LEFT", "RIGHT", "FIRST", "LAST"})) kind = TokenKind::Function;
            else if (IsConstant(word)) kind = TokenKind::Constant;
            else if (IsKeyword(word)) kind = TokenKind::Keyword;
            else if (IsType(word)) kind = TokenKind::Builtin;
            else if (call) kind = TokenKind::Function;
            if (kind != TokenKind::Default) Push(spans, pos, end, kind);
            pos = end;
            continue;
        }
        if (std::ispunct(static_cast<unsigned char>(c))) {
            Push(spans, pos, pos + 1, TokenKind::Punctuation);
        }
        ++pos;
    }
    return {};
}

}  // namespace cyxwiz::editor
