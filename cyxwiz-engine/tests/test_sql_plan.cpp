// SQL Workbench plan model (TOFIX146 board 4). Main scenario: DuckDB 1.4.3's
// EXPLAIN (ANALYZE, FORMAT JSON) and EXPLAIN (FORMAT JSON) of the real Berean
// tier1_paragraphs query on the 7 books.
#include "../src/core/sql_plan.h"

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

std::string Joined(const std::vector<std::string>& items) {
    std::string out;
    for (const auto& item : items) out += (out.empty() ? "" : ",") + item;
    return out;
}

void TestTimed() {
    using namespace cyxwiz;
    const std::string sql = Read("tier1_paragraphs.sql");
    auto plan = sqlplan::ParsePlanJson(Read("tier1_plan_timed.json"));
    Check(plan.error.empty() && plan.timed, "timed plan read: " + plan.error);
    sqlplan::ExplainPlan(plan, sql, sqlstep::OutlineSqlQuery(sql));
    Check(plan.ops.size() == 19, "19 operators, got " + std::to_string(plan.ops.size()));
    if (plan.ops.size() != 19) return;
    Check(plan.internal_count == 3, "3 internal projections");
    Check(plan.rows_out == 15349, "15,349 rows out");
    Check(plan.operator_seconds > 4.0 && plan.operator_seconds < 4.3, "about 4.15 s in operators");
    // Flow order: the join's inputs first, one level deeper.
    Check(plan.ops[0].name == "COLUMN_DATA_SCAN" && plan.ops[0].depth == 1, "meta values first");
    Check(plan.ops[1].name == "ARROW_SCAN" && plan.ops[1].depth == 1, "then the input scan");
    Check(plan.ops[2].name == "HASH_JOIN" && plan.ops[2].depth == 0 && plan.ops[2].rows_in == 14, "the join of 7 + 7");
    Check(plan.ops[18].name == "PROJECTION" && plan.ops[18].internal, "an internal projection last");
    // Parts, in the order of the query.
    const std::vector<std::string> parts = {"meta", "src", "src", "src", "body", "lead_note", "trail_note",
                                            "checked", "split", "parts", "stripped,kept,cleaned,result", "",
                                            "stripped", "kept", "cleaned", "result", "", "result", ""};
    for (size_t i = 0; i < parts.size(); ++i) {
        Check(Joined(plan.ops[i].parts) == parts[i],
              "operator " + std::to_string(i + 1) + " " + plan.ops[i].name + " part: " + Joined(plan.ops[i].parts) +
                  " (expected " + parts[i] + ")");
    }
    // The slowest operator and its notes.
    Check(plan.slowest == 10, "the merged FILTER is slowest, got " + std::to_string(plan.slowest));
    const auto& filter = plan.ops[10];
    Check(filter.rows_in == 17083 && filter.rows == 16999, "filter 17,083 -> 16,999");
    Check(filter.summary.rfind("keeps rows where ", 0) == 0, "filter summary: " + filter.summary);
    bool removes = false, repeated = false, estimate = false;
    for (const auto& note : filter.notes) {
        removes = removes || note.find("Removes 84 rows") != std::string::npos;
        repeated = repeated || (note.find("runs 6 times per row") != std::string::npos &&
                                note.find("regexp_replace(raw,") != std::string::npos);
        estimate = estimate || note.find("estimated 1 rows; 16,999 came") != std::string::npos;
    }
    Check(removes, "filter note: removes 84 rows");
    Check(repeated, "filter note: the stripped expression runs 6 times per row");
    Check(estimate, "filter note: estimate off");
    Check(plan.estimates_off == 5, "5 estimates 100x off, got " + std::to_string(plan.estimates_off));
    // The book-wide projection: long rows.
    bool per_row = false;
    for (const auto& note : plan.ops[4].notes) per_row = per_row || note.find("for 7 rows") != std::string::npos;
    Check(per_row, "body projection note: few rows, long text");
    Check(plan.ops[4].summary == "makes b", "body summary: " + plan.ops[4].summary);
    Check(plan.ops[2].summary.rfind("joins on source_id", 0) == 0, "join summary: " + plan.ops[2].summary);
    // An expression list is joined back with its newline.
    Check(plan.ops[13].details.front().second.find("string_split(p, '\n')") != std::string::npos,
          "the kept filter keeps its newline literal");
}

void TestEstimate() {
    using namespace cyxwiz;
    const std::string sql = Read("tier1_paragraphs.sql");
    auto plan = sqlplan::ParsePlanJson(Read("tier1_plan_estimate.json"));
    Check(plan.error.empty() && !plan.timed, "estimate plan read");
    sqlplan::ExplainPlan(plan, sql, sqlstep::OutlineSqlQuery(sql));
    Check(plan.ops.size() == 19, "estimate: 19 operators, got " + std::to_string(plan.ops.size()));
    Check(plan.slowest == -1 && plan.rows_out == -1 && plan.estimates_off == 0, "estimate: no times or rows");
    bool any_estimate = false;
    for (const auto& op : plan.ops) any_estimate = any_estimate || op.estimated >= 0;
    Check(any_estimate, "estimate: operators carry DuckDB's estimates");
    Check(!plan.ops.empty() && Joined(plan.ops[0].parts) == "meta", "estimate: parts too");
}

void TestOddInput() {
    using namespace cyxwiz;
    Check(!sqlplan::ParsePlanJson("not json").error.empty(), "bad JSON is reported");
    Check(sqlplan::ParsePlanJson("[]").ops.empty(), "an empty plan");
    Check(sqlplan::FormatSeconds(1.8134) == "1.81 s" && sqlplan::FormatSeconds(0.0546) == "54.6 ms" &&
              sqlplan::FormatSeconds(0.00001) == "<0.1 ms" && sqlplan::FormatSeconds(0.0) == "0 ms",
          "seconds format");
}
}  // namespace

int main() {
    TestTimed();
    TestEstimate();
    TestOddInput();
    if (failures) {
        std::cerr << failures << " of " << checks << " checks failed\n";
        return 1;
    }
    std::cout << "SQL plan model passed: " << checks << " checks\n";
    return 0;
}
