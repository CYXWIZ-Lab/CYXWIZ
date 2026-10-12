// SQL step presentation model (TOFIX146): query outline, check descriptions,
// input names. Main scenario: the real Berean tier1_paragraphs query.
// Usage: test_sql_step_presentation [path/to/tier1_paragraphs.sql]
#include "../src/core/sql_step_presentation.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef CYXWIZ_SQL_FIXTURE
#define CYXWIZ_SQL_FIXTURE "tests/fixtures/sql/tier1_paragraphs.sql"
#endif

namespace {

int failures = 0;
int checks = 0;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

using cyxwiz::sqlstep::OutlineSqlQuery;
using cyxwiz::sqlstep::SqlOutline;

std::string Names(const SqlOutline& outline) {
    std::string out;
    for (const auto& part : outline.parts) {
        if (!out.empty()) out += ',';
        out += part.name;
    }
    return out;
}

void TestRealQuery(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    Check(file.good(), "fixture readable: " + path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    const SqlOutline outline = OutlineSqlQuery(buffer.str());
    Check(outline.structured, "tier1 query is a structured WITH chain");
    Check(outline.line_count == 84, "84 lines, got " + std::to_string(outline.line_count));
    Check(outline.header.size() == 17, "17 header comment lines, got " + std::to_string(outline.header.size()));
    Check(!outline.header.empty() &&
              outline.header.front() == "base_corpus_v3, Tier 1 practical teaching books (owner D1, 2026-10-11).",
          "header title without the -- marker");
    Check(Names(outline) == "meta,src,body,lead_note,trail_note,checked,parts,stripped,kept,cleaned,result",
          "parts in order: " + Names(outline));
    const std::vector<std::pair<int, int>> lines = {
        {18, 26}, {26, 29}, {29, 37}, {37, 43}, {43, 49}, {49, 53},
        {53, 56}, {56, 58}, {58, 67}, {67, 75}, {76, 84}};
    for (size_t i = 0; i < lines.size() && i < outline.parts.size(); ++i) {
        const auto& part = outline.parts[i];
        Check(part.first_line == lines[i].first && part.last_line == lines[i].second,
              part.name + " lines " + std::to_string(part.first_line) + "-" + std::to_string(part.last_line));
        Check(part.description.empty(), part.name + " has no comment above it, so no description");
    }
}

void TestTrickyText() {
    // Brackets, commas and keywords inside strings, quoted names and comments
    // never split a part; a comment above a part becomes its description.
    const std::string sql =
        "-- Title line\n"
        "WITH RECURSIVE\n"
        "-- Numbers one to three.\n"
        "n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n WHERE x < 3),\n"
        "/* a block comment ), fake AS ( */\n"
        "s AS MATERIALIZED (SELECT '), y AS (' AS \"odd ) name\", 'it''s' AS t FROM n),\n"
        "-- Keep\n"
        "-- the small ones.\n"
        "k AS NOT MATERIALIZED (SELECT * FROM s WHERE x IN (1, (2)))\n"
        "SELECT * FROM k -- trailing note\n"
        "ORDER BY x\n";
    const SqlOutline outline = OutlineSqlQuery(sql);
    Check(outline.structured, "tricky query is structured");
    Check(Names(outline) == "n,s,k,result", "tricky parts: " + Names(outline));
    Check(outline.header.size() == 1 && outline.header[0] == "Title line", "single header line");
    if (outline.parts.size() == 4) {
        Check(outline.parts[0].description == "Numbers one to three.", "n description: " + outline.parts[0].description);
        Check(outline.parts[1].description.empty(), "block comments are not descriptions");
        Check(outline.parts[2].description == "Keep the small ones.", "k joins two comment lines");
        Check(outline.parts[3].first_line == 10 && outline.parts[3].last_line == 11, "result spans lines 10-11");
        Check(outline.parts[0].first_line == 4 && outline.parts[0].last_line == 4, "n on line 4");
    }

    const SqlOutline plain = OutlineSqlQuery("SELECT * FROM input");
    Check(plain.structured && Names(plain) == "result", "no WITH: one result part");

    const SqlOutline broken = OutlineSqlQuery("WITH a AS (SELECT 'unclosed FROM input)");
    Check(!broken.structured && Names(broken) == "query", "unclosed string: whole query, not structured");
    const SqlOutline unbalanced = OutlineSqlQuery("WITH a AS (SELECT (1 FROM input) SELECT 1");
    Check(!unbalanced.structured && Names(unbalanced) == "query", "unbalanced brackets: whole query");

    const SqlOutline empty = OutlineSqlQuery("-- only a comment\n");
    Check(empty.parts.empty() && empty.header.size() == 1, "comment-only text has a header and no parts");
}

void TestChecksAndInputs() {
    using cyxwiz::sqlstep::DescribeRowCountRule;
    using cyxwiz::sqlstep::FormatCount;
    Check(FormatCount(15349) == "15,349" && FormatCount(7) == "7" && FormatCount(1000000) == "1,000,000" &&
              FormatCount(-1234) == "-1,234",
          "thousand separators");
    Check(DescribeRowCountRule({{"expected_rows", "15349"}}) == "exactly 15,349 rows", "exact count");
    Check(DescribeRowCountRule({{"expected_rows", "3712"}, {"count_column", "source_id"}, {"count_value", "pg41633"}}) ==
              "exactly 3,712 rows where source_id = pg41633",
          "count with a column filter");
    Check(DescribeRowCountRule({{"expected_rows", "0"}, {"count_true_column", "training_admitted"}}) ==
              "exactly 0 rows where training_admitted is true",
          "count of true values");
    Check(DescribeRowCountRule({{"min_rows", "10"}, {"max_rows", "20"}}) == "between 10 and 20 rows", "range");
    Check(DescribeRowCountRule({{"min_rows", "5"}}) == "at least 5 rows", "minimum");
    Check(DescribeRowCountRule({{"max_rows", "2000"}}) == "at most 2,000 rows", "maximum");
    Check(DescribeRowCountRule({{"expected_rows", "12x"}}) == "exactly '12x' rows", "non-numbers are quoted");
    Check(DescribeRowCountRule({}) == "no row count set", "nothing set");

    using cyxwiz::sqlstep::InputNames;
    Check(InputNames({}) == std::vector<std::string>{"input"}, "default input name");
    Check(InputNames({{"input_aliases", "v1, practical ,ust"}}) == std::vector<std::string>{"v1", "practical", "ust"},
          "several inputs, trimmed, in pin order");
}

}  // namespace

int main(int argc, char** argv) {
    TestRealQuery(argc > 1 ? argv[1] : CYXWIZ_SQL_FIXTURE);
    TestTrickyText();
    TestChecksAndInputs();
    if (failures) {
        std::cerr << failures << " of " << checks << " checks failed\n";
        return 1;
    }
    std::cout << "SQL step presentation passed: " << checks << " checks\n";
    return 0;
}
