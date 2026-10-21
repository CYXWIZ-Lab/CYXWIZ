// SQL Workbench editing helpers (TOFIX146 4b-2): completion candidates, the
// keyword formatter and the line diff. Main scenario: the Berean tier1 query.
#include "../src/core/sql_editing.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#ifndef CYXWIZ_SQL_FIXTURES
#define CYXWIZ_SQL_FIXTURES "tests/fixtures/sql"
#endif

namespace {
int checks = 0;
int failures = 0;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

std::string Read(const std::string& name) {
    std::ifstream file(std::string(CYXWIZ_SQL_FIXTURES) + "/" + name, std::ios::binary);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::string Texts(const std::vector<cyxwiz::sqledit::Completion>& items) {
    std::string out;
    for (const auto& item : items) out += (out.empty() ? "" : ",") + item.text;
    return out;
}

void TestCompletions() {
    using namespace cyxwiz;
    const std::string sql = Read("tier1_paragraphs.sql");
    const auto outline = sqlstep::OutlineSqlQuery(sql);
    const std::vector<std::string> columns = {"member_path", "member_sha256", "text"};
    // Columns first, then parts.
    auto items = sqledit::SqlCompletions("me", columns, sql, outline);
    Check(items.size() >= 3 && items[0].text == "member_path" && items[1].text == "member_sha256" &&
              items[2].text == "meta",
          "me: columns then the meta part: " + Texts(items));
    Check(items[2].kind == sqledit::CompletionKind::Part, "meta is a part");
    // Names the query defines, then functions, then keywords.
    items = sqledit::SqlCompletions("para", columns, sql, outline);
    Check(Texts(items).find("paragraph_index") != std::string::npos && Texts(items).find("paragraph_id") != std::string::npos &&
              Texts(items).find("paras") != std::string::npos && Texts(items).find("parts") == std::string::npos,
          "para: parts and defined names: " + Texts(items));
    items = sqledit::SqlCompletions("regexp_r", columns, sql, outline);
    Check(Texts(items) == "regexp_replace", "a function: " + Texts(items));
    items = sqledit::SqlCompletions("sel", columns, sql, outline);
    Check(Texts(items) == "SELECT", "a keyword: " + Texts(items));
    items = sqledit::SqlCompletions("text", columns, sql, outline);
    Check(Texts(items).find("text,") != 0 && Texts(items) != "text", "the word already typed is not offered again: " + Texts(items));
    // An empty prefix lists columns and parts only (Ctrl+Space on a blank).
    items = sqledit::SqlCompletions("", columns, sql, outline);
    Check(items.size() == columns.size() + 11, "blank: 3 columns + 11 parts, got " + std::to_string(items.size()));
    // Names inside strings and comments are not offered.
    items = sqledit::SqlCompletions("hid", {}, "SELECT 1 AS shown, 'x AS hidden' AS y -- AS hidden2", sqlstep::OutlineSqlQuery("SELECT 1"));
    Check(items.empty(), "no names from strings or comments: " + Texts(items));
    Check(sqledit::WordBefore("SELECT tex", 10) == "tex" && sqledit::WordBefore("a.b_c", 5) == "b_c" &&
              sqledit::WordBefore("x ", 2).empty(),
          "word before the cursor");
}

void TestFormat() {
    using namespace cyxwiz;
    const std::string in = "select text, left(t, 3) as l  \nfrom input -- select from where\nwhere t like 'from%' and \"select\" is not null   ";
    const std::string out = sqledit::FormatSqlKeywords(in);
    Check(out == "SELECT text, left(t, 3) AS l\nFROM input -- select from where\nWHERE t LIKE 'from%' AND \"select\" IS NOT NULL",
          "keywords up, strings/names/comments and functions kept, trailing spaces gone:\n" + out);
    Check(sqledit::FormatSqlKeywords(out) == out, "formatting twice changes nothing");
    const std::string tier1 = Read("tier1_paragraphs.sql");
    const std::string formatted = sqledit::FormatSqlKeywords(tier1);
    Check(sqlstep::OutlineSqlQuery(formatted).parts.size() == sqlstep::OutlineSqlQuery(tier1).parts.size(),
          "the tier1 query keeps its parts when formatted");
    Check(sqledit::FormatSqlKeywords(formatted) == formatted, "tier1 formatting is stable");
}

void TestDiff() {
    using namespace cyxwiz;
    const auto diff = sqledit::DiffLines("a\nb\nc\nd", "a\nB\nc\nd\ne");
    std::string shape;
    for (const auto& line : diff) {
        shape += line.kind == sqledit::DiffKind::Same ? "=" : line.kind == sqledit::DiffKind::Added ? "+" : "-";
        shape += line.text + " ";
    }
    Check(shape == "=a -b +B =c =d +e ", "diff shape (removed before added): " + shape);
    Check(diff.back().new_line == 5 && diff.back().old_line == 0, "added line numbers");
    const auto same = sqledit::DiffLines("x\ny", "x\ny");
    Check(same.size() == 2 && same[0].kind == sqledit::DiffKind::Same, "no change");
    const auto crlf = sqledit::DiffLines("x\r\ny", "x\ny");
    Check(crlf.size() == 2 && crlf[0].kind == sqledit::DiffKind::Same, "line endings do not count");
}
}  // namespace

void TestStepQuery() {
    using cyxwiz::sqledit::StepQueryFromScratch;
    Check(StepQueryFromScratch("SELECT author, count(*) FROM tier1_paragraphs GROUP BY author", "tier1_paragraphs") ==
              "WITH \"tier1_paragraphs\" AS (SELECT * FROM input)\nSELECT author, count(*) FROM tier1_paragraphs GROUP BY author",
          "a plain query gets a first WITH part");
    Check(StepQueryFromScratch("-- authors\nWITH a AS (SELECT 1) SELECT * FROM a, \"Tier 1\"", "Tier 1") ==
              "-- authors\nWITH \"Tier 1\" AS (SELECT * FROM input),\na AS (SELECT 1) SELECT * FROM a, \"Tier 1\"",
          "an existing WITH is joined, comments stay first, the name is quoted");
    Check(StepQueryFromScratch("with recursive r(n) AS (SELECT 1) SELECT * FROM r", "ds_sql_1") ==
              "WITH RECURSIVE \"ds_sql_1\" AS (SELECT * FROM input),\nr(n) AS (SELECT 1) SELECT * FROM r",
          "WITH RECURSIVE stays recursive");
    Check(StepQueryFromScratch("/* x */ SELECT 1", "a\"b") == "/* x */ WITH \"a\"\"b\" AS (SELECT * FROM input)\nSELECT 1",
          "a quote in the name is doubled");
    Check(StepQueryFromScratch("SELECT * FROM withdrawals", "withdrawals").rfind("WITH \"withdrawals\" AS", 0) == 0,
          "a table named withdrawals is not a WITH");
}

int main() {
    TestCompletions();
    TestStepQuery();
    TestFormat();
    TestDiff();
    if (failures) {
        std::cerr << failures << " of " << checks << " checks failed\n";
        return 1;
    }
    std::cout << "SQL editing helpers passed: " << checks << " checks\n";
    return 0;
}
