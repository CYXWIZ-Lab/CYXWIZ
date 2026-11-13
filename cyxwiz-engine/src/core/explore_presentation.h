#pragma once

// Words and choices of Data Studio's Explore tabs Missing, Outliers,
// Correlation and Distribution (TOFIX129 D1, approved mockup
// d1_explore_tabs_mockup.html): pure data in, text out, so the tabs and their
// tests agree. No ImGui, no queries.

#include "guided_sql.h"

#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz::explore_view {

// 2408 -> "2,408".
std::string Number(size_t n);
// "333 complete rows", "1 row", "2,408 cells".
std::string Count(size_t n, const char* one, const char* many);
// 3.2% (one decimal; "0%" for nothing of nothing).
std::string Share(size_t part, size_t whole);

// The advice for a column missing `share` percent of its values (the old
// Missing Values window's tiers: below 5, 20, 50, and 50 or more).
std::vector<std::string> MissingAdvice(double share);
// Advice for any column, by kind of data.
const std::vector<std::string>& GeneralMissingAdvice();

// The line under the summary: what counts as missing here.
// `texts_columns` = how many columns have texts marked as missing.
std::string MissingRule(size_t texts_columns);

// SQL for the Query tab listing the rows where `column` is missing in
// `table` (texts as quoted literals; the Query tab takes no parameters).
std::string MissingRowsSql(const std::string& table, const std::string& column, const std::vector<std::string>& texts);

// A figure in a few significant digits: 4201.754 -> "4202", 15021.6 ->
// "15022", 0.18772 -> "0.1877", 1437000 -> "1.437e+06"; "" for NaN.
std::string Figure(double v);

// SQL for the Query tab listing the rows of `column` outside lo .. hi.
std::string OutlierRowsSql(const std::string& table, const std::string& column, double lo, double hi);

// Words for |r|: "very strong" (>= .8), "strong" (.6), "moderate" (.4),
// "weak" (.2), "none"; "unknown" for NaN.
const char* CorrelationStrength(double r);
// "very strong positive: when one is larger, so is the other".
std::string CorrelationSentence(double r);
// The matrix as CSV: a header of the columns, then one row per column
// (row-major r, n x n; NaN as an empty cell).
std::string CorrelationCsv(const std::vector<std::string>& columns, const std::vector<double>& r);

// Skewness in words: "about symmetric" (|s| <= 0.2), "longer right tail",
// "longer left tail".
const char* SkewWords(double skewness);
// What a figure of the Distribution tab means (hover text); "" when unknown.
const char* FigureHint(const std::string& label);
// Rows of cells as CSV (quoted where needed).
std::string Csv(const std::vector<std::vector<std::string>>& rows);

// The Preparation Recipe step "Drop these rows": a guided Keep rows step
// keeping the rows where the column is not empty and (with texts marked as
// missing) is not one of them.
guided::GuidedStep KeepRowsWithValue(const std::string& column, const std::vector<std::string>& texts);

// Hover words of Explore's graph buttons (TOFIX129 D7a): what a step and a
// branch mean for training, so no button misleads. `node` is the node's name
// ("K-Means"), `gives` what a step adds ("a cluster_id column"; "" for none).
std::string StepTip(const std::string& node, const std::string& gives);
std::string BranchTip(const std::string& node);
// The tail of the message after a node was added: how training sees it.
const char* StepAddedNote();    // "Training reads its result."
const char* BranchAddedNote();  // "Training does not read a branch."

}  // namespace cyxwiz::explore_view
