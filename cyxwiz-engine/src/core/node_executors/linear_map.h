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

// The column means and population standard deviations (divides by n) of x
// [n × d], on the device. False + error when the device fails.
bool ColumnMoments(const LinearMatrix& x, std::vector<double>& mean, std::vector<double>& sd, std::string& error);

// x [n × d] centred by `mean`, on the device.
bool Centred(const LinearMatrix& x, const std::vector<double>& mean, LinearMatrix& out, std::string& error);

// ((x - shift) / divisor) W for each row, on the device.
bool ApplyLinearMap(const LinearMap& map, const LinearMatrix& rows, LinearMatrix& out, std::string& error);

// The rows of `fit` that are Train rows (all of them when the table carries no split).
LinearMatrix TrainSubset(const std::shared_ptr<arrow::Table>& table, const FeatureRows& fit);

// The output: with `keep_columns`, the input with the outputs beside it
// (null on the rows left out); else the outputs alone for the rows read,
// with `y` from `label_col` (int32 codes) and the split column carried.
arrow::Result<std::shared_ptr<arrow::Table>> LinearMapTable(const std::shared_ptr<arrow::Table>& input, const LinearMap& map,
                                                            const FeatureRows& fit, const LinearMatrix& projected,
                                                            bool keep_columns, const std::string& label_col, const std::string& op_name);

// The fitted state of `map` (fit_rows, the schema and the settings that must match on reuse).
FittedPreprocessingState LinearMapState(const LinearMap& map, const std::string& op_name, int64_t fit_rows,
                                        const std::shared_ptr<arrow::Schema>& schema,
                                        const std::map<std::string, std::string>& configuration);
// `map` from a fitted state; false + error when it is not one.
bool LinearMapFromState(const FittedPreprocessingState& state, LinearMap& map, std::string& error);

}  // namespace cyxwiz
