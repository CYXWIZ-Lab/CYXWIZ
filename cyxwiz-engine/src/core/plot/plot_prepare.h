#pragma once

// Builds the drawn data of a plot from table columns (TOFIX134 P1 step 1.3).
// Pure and thread-safe: run it off the UI thread for large tables; the UI
// only draws the result. Nothing is cut silently: the result's DataLabel says
// when values were reduced, sampled or read with a row limit.

#include "plot_model.h"

#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz::plot {

// One column of the source table. Numeric columns fill `numbers` (NaN for a
// missing value); other columns fill `text`.
struct SourceColumn {
    std::string name;
    bool numeric = true;
    std::vector<double> numbers;
    std::vector<std::string> text;
    size_t size() const { return numeric ? numbers.size() : text.size(); }
};

struct Source {
    std::vector<SourceColumn> columns;
    // The read was cut at this many rows (0: the whole table was read);
    // total_rows is the table's size when known.
    size_t row_limit = 0;
    size_t total_rows = 0;
    // After a row selection: each row's 0-based place in the table (the X of
    // a plot by row number). Empty: rows are in table order from 0.
    std::vector<double> row_index;
    size_t Rows() const;

    const SourceColumn* Find(const std::string& name) const;
};

// Limits: lines longer than this are reduced (min/max per step), scatters
// larger are sampled, categories beyond the cap are merged into "other".
constexpr size_t kMaxLinePoints = 4000;
constexpr size_t kMaxScatterPoints = 50000;
constexpr size_t kMaxColorGroups = 12;
constexpr size_t kMax3DPoints = 100000;   // Scatter 3D sample / Line 3D every Nth row beyond this
constexpr size_t kMaxSurfaceRows = 1000;  // Surface from grid columns: every Nth row beyond this
constexpr size_t kMaxGraphNodes = 5000;   // Network / Tree: more nodes are refused (choose fewer rows)
constexpr size_t kMaxMeshPoints = 5000;   // Mesh: a reproducible sample beyond this
constexpr size_t kMaxCategories = 200;
constexpr size_t kMaxPieSlices = 12;
constexpr int kViolinSteps = 64;
// A number column with more values than kMaxColorGroups is split into this
// many equal ranges when it colours groups.
constexpr int kColourRanges = 6;
// The legend lists this many series; hover shows all of them.
constexpr size_t kMaxLegendSeries = 12;

// Every column a spec reads: X, Y, colour and filter columns.
std::vector<std::string> ColumnsNeeded(const PlotSpec& spec);

// The rows the spec's row selection keeps (TOFIX134 P2, board 6), as a new
// source with only those rows (row_index keeps their place in the table).
struct RowSelection {
    bool all = false;       // every row: `source` is left empty, use the input
    Source source;
    size_t total = 0;       // rows of the source
    std::string text;       // "filtered · 7,293 of 70,000 rows"; "" for all rows
    std::string problem;    // "" when fine
};
RowSelection SelectRows(const PlotSpec& spec, const Source& source);

Prepared Prepare(const PlotSpec& spec, const Source& source);

// What the column picker shows of a column: its type, range, share of values
// that are not 0, and how many values it has (counted up to 13).
struct ColumnSummary {
    std::string name;
    bool numeric = false;
    bool has_stats = false;
    double min = 0, max = 0;
    double not_zero = 0;   // share 0..1 of the finite values (numbers)
    size_t distinct = 0;   // distinct values, at most kMaxColorGroups + 1
    bool OneValue() const { return has_stats && distinct <= 1; }
    // "0–255 · 66.3% not 0", "3 values"; "" without stats.
    std::string Text() const;
};
ColumnSummary SummarizeColumn(const SourceColumn& column);

}  // namespace cyxwiz::plot
