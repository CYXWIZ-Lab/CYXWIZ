#pragma once

// Data Studio > Prepare's view of a scale, power or log step (TOFIX129 D2,
// approved mockup d2_transform_mockup.html): which method a node is, what a
// method switch keeps, and the figures of the step's last run (before / after
// per column with histograms; a power step's lambda, likelihood curve,
// skewness and normality test; Compare methods). Tables in, figures and
// words out, so the view and its tests agree. No ImGui; the math is the
// backend's DataTransform (ArrayFire).

#include "graph_model.h"

#include <arrow/api.h>

#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz::transform_step {

enum class Kind { None, Scale, Power, Log };
Kind KindOf(gui::NodeType type);

// A scaling method: one node each (the method switch replaces the node).
struct Method {
    gui::NodeType type;
    const char* label;    // "Z-score"
    const char* formula;  // "(x - mean) / std"
    const char* advice;   // when to pick it (Compare methods)
};
// Z-score, Min-Max, Robust, Max-Abs, Quantile, in that order.
const std::vector<Method>& ScaleMethods();
const Method* MethodOf(gui::NodeType type);  // null for a node that is no scaling method

// What the step computes, for its header: "(x - mean) / std",
// "Yeo-Johnson, lambda found per column", "log(1 + x), base e".
std::string Formula(gui::NodeType type, const std::map<std::string, std::string>& params);

// The parameters a method switch keeps: the columns, label, excluded columns
// and whether it fits here or reuses saved values. The state file is not
// kept: saved values belong to the method that fitted them.
std::map<std::string, std::string> CarriedParameters(const std::map<std::string, std::string>& from);

// The columns a step transforms: its `columns`, else every number column of
// `schema` except `label_col` and `exclude_columns` (the operators' rule).
std::vector<std::string> StepColumns(const std::map<std::string, std::string>& params, const arrow::Schema& schema);
// The number columns of `schema`, for Pick columns.
std::vector<std::string> NumberColumns(const arrow::Schema& schema);

// A column's values as doubles; NaN where missing. Empty when the column is
// missing or not a number column.
std::vector<double> ColumnValues(const arrow::Table& table, const std::string& name);

struct Stats {
    bool ok = false;
    int64_t count = 0;    // values present
    int64_t missing = 0;
    double min = NAN, max = NAN, mean = NAN, std = NAN, skew = NAN;
    std::vector<int64_t> bins;  // a histogram over [min, max]
    int64_t beyond_z3 = 0;      // |value| > 3 (a Z-score step's after)
};
inline constexpr int kBins = 24;
Stats Describe(const std::vector<double>& values, int bins = kBins);

struct PowerFigures {
    bool box_cox = false;
    bool auto_lambda = true;
    double lambda = NAN;                    // the one used
    std::vector<double> lambdas, log_likelihoods;  // the likelihood curve, -5 .. 5
    double best = NAN;                      // the best lambda on the curve
    double p_before = NAN, p_after = NAN;   // D'Agostino-Pearson; NaN under 20 values
    std::string note;                       // why something is missing (Box-Cox on 0 or less)
};

struct ColumnFigures {
    std::string name;
    Stats before, after;
    bool has_power = false;
    PowerFigures power;
};

struct StepFigures {
    std::vector<ColumnFigures> columns;
    int64_t rows_in = 0, rows_out = 0;
    bool sampled = false;  // the tables were read up to their first kFigureRows rows
    std::string error;
    bool ok() const { return error.empty(); }
};
inline constexpr int64_t kFigureRows = 200000;

// The figures of a step from its input and output (`saved_lambdas`: a power
// step's lambda per column when it used saved values).
StepFigures ComputeFigures(gui::NodeType type, const std::map<std::string, std::string>& params, const arrow::Table& input,
                           const arrow::Table& output, const std::map<std::string, double>& saved_lambdas = {});

// Compare methods: one column through every scaling method (preview only).
struct CompareFigures {
    std::string column;
    Stats original;
    std::vector<Stats> methods;  // in ScaleMethods() order
    std::string error;
    bool ok() const { return error.empty(); }
};
CompareFigures CompareMethods(const std::string& column, const std::vector<double>& values);

// Words.
std::string Range(const Stats& s);   // "32.1 .. 59.6"
std::string Value(double v);         // a figure in a few digits (never "-0"); "" for NaN
std::string PValue(double p);        // "1.3e-08", "0.21"; "-" for NaN
std::string RunLine(int64_t rows_in, int64_t rows_out);  // "Last run passed · 344 rows in, 344 out"
// Under a power step's figures: what the transform did to the shape.
std::string PowerSentence(const ColumnFigures& c);
// A Box-Cox step's note on its columns: "Box-Cox needs values above 0; all 4 are."
std::string BoxCoxNote(const StepFigures& f);
// A Z-score step's outlier line: "Values beyond |z| 3: none in these columns."
std::string BeyondZ3Line(const StepFigures& f);

}  // namespace cyxwiz::transform_step
