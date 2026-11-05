#pragma once

// A linear map fitted on the Train rows and applied to every row: the
// number columns x become ((x - shift) / divisor) W, W of d inputs × k
// outputs. PCA, Truncated SVD and Whitening fit it their own way
// (TOFIX129 D9); this is what they share: the column moments and the
// product on the device (the backend's LinearAlgebra), the output table
// (the outputs beside the input, or alone with the label and the split),
// and the fitted state (one feature per input column: its shift, divisor
// and row of W).

#include "../preprocessing_state.h"

#include <cyxwiz/linear_algebra.h>
#include "feature_rows.h"

#include <arrow/api.h>

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

using LinearMatrix = std::vector<std::vector<double>>;

struct LinearMap {
    std::vector<std::string> inputs;   // d number columns
    std::vector<double> shift;         // d (0: none)
    std::vector<double> divisor;       // d (1: none)
    LinearMatrix w;                    // d × k
    std::vector<std::string> outputs;  // k column names
};

// A comma-separated column list (spaces around names dropped).
std::vector<std::string> ParseColumnList(const std::string& text);
// A "true" / "false" parameter (absent or empty: `fallback`); false + error otherwise.
bool ReadBoolParameter(const std::map<std::string, std::string>& params, const char* key, bool fallback, bool& out,
                       const std::string& op_name, std::string& error);
// The number columns to map: `wanted` (each must be a number column), or
// every number column but the label and the internal "__" ones.
arrow::Status ResolveNumberFeatures(const std::shared_ptr<arrow::Table>& input, const std::vector<std::string>& wanted,
                                    const std::string& label_col, const std::string& op_name, std::vector<std::string>& out);

// The Train rows' second moments from their column summary (one upload,
// LinearAlgebra::ColumnSummary): the cross products about the means, or
// about 0 when `uncentred`, each divided by divisor_i divisor_j (empty: 1)
// and by `denominator` (n - 1: the covariance).
LinearMatrix SecondMoments(const ColumnSummaryResult& summary, bool uncentred, const std::vector<double>& divisor, double denominator);

// The k leading directions of a symmetric positive semi-definite M (d × d)
// as rows, each signed so its largest weight is positive, and their values:
// a randomized subspace iteration for a few of many (d above 256, k at most
// d / 4), else the full SVD of M; on the device.
bool TopDirections(const LinearMatrix& m, size_t k, LinearMatrix& vt, std::vector<double>& values, std::string& error);

// The rows of a table with every feature (null or not finite: left out),
// their number columns in one column-major buffer: what goes to the device
// as it is (LinearAlgebra's DenseView), with no row-by-row repacking.
struct NumberColumns {
    int64_t total = 0;            // rows in the table
    std::vector<int64_t> rows;    // each kept row's index in the table
    size_t cols = 0;
    std::vector<double> data;     // column j: data[j * rows.size()] ..
    DenseView View() const { return DenseView{static_cast<int64_t>(rows.size()), static_cast<int>(cols), data.data()}; }
};
arrow::Status ReadNumberColumns(const std::shared_ptr<arrow::Table>& table, const std::vector<std::string>& names,
                                const std::string& op_name, NumberColumns& out, const std::function<bool()>& cancelled = {});
// The Train rows of `all` (all of them when the table carries no split).
NumberColumns TrainColumns(const std::shared_ptr<arrow::Table>& table, const NumberColumns& all);

// ((x - shift) / divisor) W for each row, on the device (one pass); `out`
// column-major (rows × outputs).
bool ApplyLinearMap(const LinearMap& map, const NumberColumns& rows, std::vector<double>& out, std::string& error);

// The output: with `keep_columns`, the input with the outputs beside it
// (Float64, null on the rows left out); else the outputs alone (Float32) for
// the rows read, with `y` from `label_col` (int32 codes) and the split carried.
arrow::Result<std::shared_ptr<arrow::Table>> LinearMapTable(const std::shared_ptr<arrow::Table>& input, const LinearMap& map,
                                                            const NumberColumns& rows, const std::vector<double>& mapped,
                                                            bool keep_columns, const std::string& label_col, const std::string& op_name);

// The fitted state of `map` (fit_rows, the schema and the settings that must match on reuse).
FittedPreprocessingState LinearMapState(const LinearMap& map, const std::string& op_name, int64_t fit_rows,
                                        const std::shared_ptr<arrow::Schema>& schema,
                                        const std::map<std::string, std::string>& configuration);
// `map` from a fitted state; false + error when it is not one.
bool LinearMapFromState(const FittedPreprocessingState& state, LinearMap& map, std::string& error);

}  // namespace cyxwiz
