#include "sql_step_presentation.h"

#include "sql_step_contract.h"

#include <algorithm>
#include <cctype>
#include <charconv>

namespace cyxwiz::sqlstep {
namespace {

enum class TokenKind { Word, Open, Close, Comma, Other };

struct Token {
    TokenKind kind = TokenKind::Other;
    std::string text;  // lower-cased for words
    int line = 1;
    size_t offset = 0;  // byte position of the token in the text
};

struct Scan {
    std::vector<Token> tokens;
    std::map<int, std::string> full_line_comments;  // line -> comment text without "--"
    int line_count = 0;
    bool balanced = true;
};

std::string Trim(std::string_view text) {
    size_t start = 0;
    while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start]))) ++start;
    size_t end = text.size();
    while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1]))) --end;
    return std::string(text.substr(start, end - start));
}

bool IsWordChar(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

// One pass over the text. Strings, quoted names and comments never produce
// tokens; brackets are counted only outside them.
Scan ScanSql(std::string_view sql) {
    Scan scan;
    int line = 1;
    int depth = 0;
    bool line_has_code = false;
    size_t i = 0;
    const auto newline = [&] {
        ++line;
        line_has_code = false;
    };
    while (i < sql.size()) {
        const char c = sql[i];
        if (c == '\n') {
            newline();
            ++i;
        } else if (c == '-' && i + 1 < sql.size() && sql[i + 1] == '-') {
            size_t end = sql.find('\n', i);
            if (end == std::string_view::npos) end = sql.size();
            if (!line_has_code) {
                std::string_view body = sql.substr(i + 2, end - i - 2);
                if (!body.empty() && body.back() == '\r') body.remove_suffix(1);
                if (!body.empty() && body.front() == ' ') body.remove_prefix(1);
                scan.full_line_comments[line] = std::string(body);
            }
            i = end;
        } else if (c == '/' && i + 1 < sql.size() && sql[i + 1] == '*') {
            size_t end = sql.find("*/", i + 2);
            const size_t stop = end == std::string_view::npos ? sql.size() : end + 2;
            if (end == std::string_view::npos) scan.balanced = false;
            for (size_t k = i; k < stop; ++k) {
                if (sql[k] == '\n') newline();
            }
            i = stop;
        } else if (c == '\'' || c == '"') {
            line_has_code = true;
            size_t k = i + 1;
            bool closed = false;
            while (k < sql.size()) {
                if (sql[k] == '\n') newline();
                if (sql[k] == c) {
                    if (k + 1 < sql.size() && sql[k + 1] == c) {
                        k += 2;  // doubled quote is part of the text
                        continue;
                    }
                    closed = true;
                    break;
                }
                ++k;
            }
            if (!closed) scan.balanced = false;
            scan.tokens.push_back({TokenKind::Other, c == '"' ? "\"name\"" : "'text'", line, i});
            i = closed ? k + 1 : sql.size();
        } else if (std::isspace(static_cast<unsigned char>(c))) {
            ++i;
        } else if (IsWordChar(c)) {
            line_has_code = true;
            size_t k = i;
            while (k < sql.size() && IsWordChar(sql[k])) ++k;
            std::string word(sql.substr(i, k - i));
            std::transform(word.begin(), word.end(), word.begin(),
                           [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
            scan.tokens.push_back({TokenKind::Word, word, line, i});
            i = k;
        } else {
            line_has_code = true;
            TokenKind kind = TokenKind::Other;
            if (c == '(') {
                kind = TokenKind::Open;
                ++depth;
            } else if (c == ')') {
                kind = TokenKind::Close;
                if (--depth < 0) scan.balanced = false;
            } else if (c == ',') {
                kind = TokenKind::Comma;
            }
            scan.tokens.push_back({kind, std::string(1, c), line, i});
            ++i;
        }
    }
    if (depth != 0) scan.balanced = false;
    scan.line_count = sql.empty() ? 0 : (sql.back() == '\n' ? line - 1 : line);
    return scan;
}

// Index of the bracket closing the one at `open`, or tokens.size().
size_t MatchingClose(const std::vector<Token>& tokens, size_t open) {
    int depth = 0;
    for (size_t k = open; k < tokens.size(); ++k) {
        if (tokens[k].kind == TokenKind::Open) ++depth;
        if (tokens[k].kind == TokenKind::Close && --depth == 0) return k;
    }
    return tokens.size();
}

// Names after FROM / JOIN in tokens [begin, end): not a function call
// (unnest(...)) or a subquery, not a keyword such as LATERAL.
std::vector<std::string> ReadsIn(const std::vector<Token>& tokens, size_t begin, size_t end) {
    std::vector<std::string> reads;
    for (size_t k = begin; k + 1 < end; ++k) {
        if (tokens[k].kind != TokenKind::Word || (tokens[k].text != "from" && tokens[k].text != "join")) continue;
        size_t n = k + 1;
        if (tokens[n].kind == TokenKind::Word && tokens[n].text == "lateral") ++n;
        if (n >= end || tokens[n].kind != TokenKind::Word) continue;
        if (n + 1 < end && tokens[n + 1].kind == TokenKind::Open) continue;
        const std::string& name = tokens[n].text;
        if (name == "select" || name == "values") continue;
        if (std::find(reads.begin(), reads.end(), name) == reads.end()) reads.push_back(name);
    }
    return reads;
}

std::string CommentAbove(const Scan& scan, int line, int header_end_line) {
    std::vector<std::string> lines;
    for (int l = line - 1; l > header_end_line; --l) {
        const auto found = scan.full_line_comments.find(l);
        if (found == scan.full_line_comments.end()) break;
        lines.push_back(Trim(found->second));
    }
    std::reverse(lines.begin(), lines.end());
    std::string joined;
    for (const auto& text : lines) {
        if (text.empty()) continue;
        if (!joined.empty()) joined += ' ';
        joined += text;
    }
    return joined;
}

bool ParseWholeNumber(const std::string& text, int64_t& value) {
    const std::string trimmed = Trim(text);
    if (trimmed.empty()) return false;
    const auto result = std::from_chars(trimmed.data(), trimmed.data() + trimmed.size(), value);
    return result.ec == std::errc{} && result.ptr == trimmed.data() + trimmed.size();
}

std::string CountText(const std::string& text) {
    int64_t value = 0;
    return ParseWholeNumber(text, value) ? FormatCount(value) : "'" + Trim(text) + "'";
}

std::string Param(const std::map<std::string, std::string>& params, const char* key) {
    const auto found = params.find(key);
    return found == params.end() ? std::string() : Trim(found->second);
}

}  // namespace

SqlOutline OutlineSqlQuery(std::string_view sql) {
    const Scan scan = ScanSql(sql);
    SqlOutline outline;
    outline.line_count = scan.line_count;
    const auto& tokens = scan.tokens;
    const int first_code_line = tokens.empty() ? scan.line_count + 1 : tokens.front().line;
    for (const auto& [line, text] : scan.full_line_comments) {
        if (line < first_code_line) outline.header.push_back(text);
    }
    const int header_end_line = first_code_line - 1;
    if (tokens.empty()) return outline;
    const int last_line = tokens.back().line;

    const auto whole_query = [&] {
        outline.structured = false;
        SqlPart query{"query", tokens.front().line, last_line, ""};
        query.begin_offset = tokens.front().offset;
        query.end_offset = sql.size();
        outline.parts = {std::move(query)};
    };
    if (!scan.balanced) {
        whole_query();
        return outline;
    }

    size_t k = 0;
    if (tokens[k].kind == TokenKind::Word && tokens[k].text == "with") {
        ++k;
        if (k < tokens.size() && tokens[k].kind == TokenKind::Word && tokens[k].text == "recursive") ++k;
        while (true) {
            if (k >= tokens.size() || tokens[k].kind != TokenKind::Word) {
                whole_query();
                return outline;
            }
            SqlPart part;
            part.name = tokens[k].text;
            part.first_line = tokens[k].line;
            part.begin_offset = tokens[k].offset;
            ++k;
            if (k < tokens.size() && tokens[k].kind == TokenKind::Open) {  // name(column, ...)
                k = MatchingClose(tokens, k) + 1;
            }
            if (k >= tokens.size() || tokens[k].text != "as") {
                whole_query();
                return outline;
            }
            ++k;
            while (k < tokens.size() && tokens[k].kind == TokenKind::Word &&
                   (tokens[k].text == "not" || tokens[k].text == "materialized")) {
                ++k;
            }
            if (k >= tokens.size() || tokens[k].kind != TokenKind::Open) {
                whole_query();
                return outline;
            }
            const size_t close = MatchingClose(tokens, k);
            if (close >= tokens.size()) {
                whole_query();
                return outline;
            }
            part.last_line = tokens[close].line;
            part.description = CommentAbove(scan, part.first_line, header_end_line);
            part.reads = ReadsIn(tokens, k + 1, close);
            part.end_offset = tokens[close].offset + 1;
            outline.parts.push_back(std::move(part));
            k = close + 1;
            if (k < tokens.size() && tokens[k].kind == TokenKind::Comma) {
                ++k;
                continue;
            }
            break;
        }
    }
    if (k < tokens.size()) {
        SqlPart result{"result", tokens[k].line, last_line, ""};
        result.description = CommentAbove(scan, result.first_line, header_end_line);
        result.reads = ReadsIn(tokens, k, tokens.size());
        result.begin_offset = tokens[k].offset;
        result.end_offset = sql.size();
        outline.parts.push_back(std::move(result));
    }
    return outline;
}

namespace {
// The final statement without trailing spaces and semicolons, so it can sit
// inside a subquery.
std::string FinalStatement(std::string_view sql, const SqlPart& part) {
    std::string text(sql.substr(part.begin_offset, part.end_offset - part.begin_offset));
    while (!text.empty() && (std::isspace(static_cast<unsigned char>(text.back())) || text.back() == ';')) {
        text.pop_back();
    }
    return text;
}
}  // namespace

std::string QueryThroughPart(std::string_view sql, const SqlOutline& outline, size_t index) {
    if (index >= outline.parts.size()) return {};
    const SqlPart& part = outline.parts[index];
    if (!outline.structured || index + 1 == outline.parts.size()) return std::string(sql);
    return std::string(sql.substr(0, part.end_offset)) + "\nSELECT * FROM " + part.name;
}

std::string PartCountsQuery(std::string_view sql, const SqlOutline& outline) {
    if (!outline.structured || outline.parts.size() < 2) return {};
    const SqlPart& last_cte = outline.parts[outline.parts.size() - 2];
    std::string query(sql.substr(0, last_cte.end_offset));
    query += "\n";
    for (size_t i = 0; i < outline.parts.size(); ++i) {
        const SqlPart& part = outline.parts[i];
        if (i > 0) query += "\nUNION ALL ";
        const std::string source = i + 1 == outline.parts.size()
                                       ? "(\n" + FinalStatement(sql, part) + "\n)"
                                       : part.name;
        query += "SELECT " + std::to_string(i) + " AS ord, '" + part.name + "' AS part, CAST(count(*) AS BIGINT) AS row_count FROM " + source;
    }
    return "SELECT part, row_count FROM (\n" + query + "\n) ORDER BY ord";
}

std::string FormatCount(int64_t value) {
    const bool negative = value < 0;
    std::string digits = std::to_string(negative ? -value : value);
    std::string out;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i > 0 && (digits.size() - i) % 3 == 0) out += ',';
        out += digits[i];
    }
    return negative ? "-" + out : out;
}

std::string DescribeRowCountRule(const std::map<std::string, std::string>& params) {
    const std::string expected = Param(params, "expected_rows");
    const std::string min_rows = Param(params, "min_rows");
    const std::string max_rows = Param(params, "max_rows");
    std::string rule;
    if (!expected.empty()) {
        rule = "exactly " + CountText(expected) + " rows";
    } else if (!min_rows.empty() && !max_rows.empty()) {
        rule = "between " + CountText(min_rows) + " and " + CountText(max_rows) + " rows";
    } else if (!min_rows.empty()) {
        rule = "at least " + CountText(min_rows) + " rows";
    } else if (!max_rows.empty()) {
        rule = "at most " + CountText(max_rows) + " rows";
    } else {
        return "no row count set";
    }
    const std::string true_column = Param(params, "count_true_column");
    const std::string column = Param(params, "count_column");
    if (!true_column.empty()) {
        rule += " where " + true_column + " is true";
    } else if (!column.empty()) {
        rule += " where " + column + " = " + Param(params, "count_value");
    }
    return rule;
}

std::vector<std::string> InputNames(const std::map<std::string, std::string>& params) {
    return SqlStepInputAliases(params);
}

}  // namespace cyxwiz::sqlstep
