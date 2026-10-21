#pragma once

// The queries behind Data Studio's Explore tabs Missing, Outliers and
// Correlation (TOFIX129 D1). They answer what the profile does not hold:
// which rows are incomplete and which columns go missing together, the rows
// missing a value, outliers by another rule than 1.5 x IQR with the rows
// they are in, and a correlation matrix of chosen columns. Same semantics as
// the profile (dataset_profiler.h): a value is missing when it is NULL or one
// of the texts the project marks as missing for that column; correlations
// use the rows where both values are present. Everything runs as SQL through
// a QueryRunner, so tables kept on disk are explored the same way.

#include "dataset_profiler.h"

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz::explore {

using MissingTexts = std::map<std::string, std::vector<std::string>>;

// SQL that is true when `column` is missing (NULL, or one of `texts`); the
// texts are appended to `params`.
std::string MissingCondition(const std::string& column, const std::vector<std::string>& texts,
                             std::vector<QueryParam>& params);

// Which rows are incomplete, and which columns go missing together.
struct MissingPattern {
    std::vector<std::string> columns;  // missing together in these rows
    size_t rows = 0;
};
struct MissingSummary {
    size_t complete_rows = 0;
    size_t incomplete_rows = 0;
    std::vector<MissingPattern> patterns;  // most rows first
    std::string error;
    bool ok() const { return error.empty(); }
};
constexpr size_t kPatterns = 10;          // patterns kept
constexpr size_t kPatternColumns = 40;    // columns a pattern is told apart by (the first with missing values)
MissingSummary SummariseMissing(const DatasetProfile& profile, const MissingTexts& texts, const QueryRunner& run);

// A page of rows: "row" is the 1-based row number in the table, then the
// table's own columns (and, for outliers, "score").
struct RowPage {
    std::shared_ptr<arrow::Table> rows;
    size_t offset = 0;
    std::string error;
    bool ok() const { return error.empty(); }
};
constexpr size_t kRowPage = 50;
RowPage MissingRows(const std::string& table, const std::string& column, const std::vector<std::string>& texts,
                    size_t offset, const QueryRunner& run, size_t limit = kRowPage);

enum class OutlierRule { Iqr, ZScore, ModifiedZ };
const char* OutlierRuleName(OutlierRule rule);   // "IQR", "Z-score", "Modified Z"
const char* OutlierScoreName(OutlierRule rule);  // what the score column measures
double DefaultOutlierFactor(OutlierRule rule);   // 1.5, 3, 3.5

struct OutlierBounds {
    double lo = 0, hi = 0;
    double mad = 0;          // Modified Z only
    std::string unavailable; // why the rule cannot be used on this column (no spread)
    bool ok() const { return unavailable.empty(); }
};
// The bounds from the profile's figures; Modified Z also needs the median
// absolute deviation (`mad`, from MedianAbsoluteDeviation).
OutlierBounds BoundsFor(const ProfiledColumn& column, OutlierRule rule, double factor, double mad = 0);
// median(|x - median|); NaN when it cannot be computed.
double MedianAbsoluteDeviation(const std::string& table, const ProfiledColumn& column, const QueryRunner& run,
                               std::string* error = nullptr);

struct OutlierCounts {
    size_t low = 0, high = 0;
    size_t total() const { return low + high; }
    std::string error;
};
OutlierCounts CountOutliers(const std::string& table, const ProfiledColumn& column, const OutlierBounds& bounds,
                            const QueryRunner& run);
// Rows outside the bounds, the largest |score| first.
RowPage OutlierRows(const std::string& table, const ProfiledColumn& column, OutlierRule rule,
                    const OutlierBounds& bounds, size_t offset, const QueryRunner& run, size_t limit = kRowPage);

}  // namespace cyxwiz::explore
