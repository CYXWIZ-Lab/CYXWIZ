#pragma once

// The linear-algebra steps of a training graph (TOFIX129 D9): each fits a
// linear map (linear_map.h) on the Train rows of its number columns, then
// maps every row, with the backend's LinearAlgebra on the device.
//
//   Truncated SVD: x V_k, the k strongest directions of the uncentred
//     columns (scikit-learn's TruncatedSVD): svd_0 .. svd_{k-1}.
//   Whitening: the columns centred, decorrelated and scaled to unit
//     variance: PCA whitening (scikit-learn's PCA(whiten=True)), white_0 ..,
//     or ZCA (the whitened data rotated back onto the columns), <column>_white.
//
// A component's sign: its largest-magnitude weight is positive (as PCA's).

#include "linear_map.h"
#include "pipeline_operator.h"

#include <map>
#include <string>
#include <vector>

namespace cyxwiz {

class FittedLinearOperator : public IPipelineOperator {
public:
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }
    // The output is kept unless the step saves its fitted values (a hit would not write them).
    bool IsCacheable() const override { return options_.IsTransformOnly() || !options_.save_state; }
    bool CollectCacheDependencies(std::vector<PipelineOperatorCacheDependency>& dependencies, std::string&) const override {
        if (options_.IsTransformOnly()) dependencies.push_back({"fitted_state", options_.state_path});
        return true;
    }
    bool Configure(const std::map<std::string, std::string>& params, std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(const std::shared_ptr<arrow::Table>& input) override;
    void SetProgressCallback(PipelineOperatorProgressCallback callback) override { progress_callback_ = std::move(callback); }

protected:
    virtual bool ConfigureOwn(const std::map<std::string, std::string>& params, std::string& error) = 0;
    // The settings the saved values must have been fitted with.
    virtual std::map<std::string, std::string> Settings() const = 0;
    // The map from the Train rows `train` [n × d] of `features`.
    virtual arrow::Status Fit(const LinearMatrix& train, const std::vector<std::string>& features, LinearMap& map) = 0;

private:
    std::vector<std::string> feature_cols_;  // empty: every number column but the label
    std::string label_col_;
    bool keep_columns_ = false;
    FittedPreprocessingOptions options_;
    PipelineOperatorProgressCallback progress_callback_;
};

class TruncatedSVDOperator : public FittedLinearOperator {
public:
    std::string GetName() const override { return "TruncatedSVD"; }

protected:
    bool ConfigureOwn(const std::map<std::string, std::string>& params, std::string& error) override;
    std::map<std::string, std::string> Settings() const override;
    arrow::Status Fit(const LinearMatrix& train, const std::vector<std::string>& features, LinearMap& map) override;

private:
    int n_components_ = 2;
};

class WhiteningOperator : public FittedLinearOperator {
public:
    std::string GetName() const override { return "Whitening"; }

protected:
    bool ConfigureOwn(const std::map<std::string, std::string>& params, std::string& error) override;
    std::map<std::string, std::string> Settings() const override;
    arrow::Status Fit(const LinearMatrix& train, const std::vector<std::string>& features, LinearMap& map) override;

private:
    std::string method_ = "pca";  // "pca" or "zca"
    int n_components_ = 0;        // PCA whitening: the directions kept; 0: every one
    double epsilon_ = 0.0;        // added to each variance before its root
};

// A x = b from table columns (TOFIX129 D9): A's columns from the Data
// input, b's from the b input (a second table, row by row) or from the same
// table. Square A of full rank: the one solution; otherwise least squares,
// the shortest of the best x (the backend's LinearAlgebra). Fitted on the
// Train rows when the run marks them. Output: one row per column of A
// ("column", then x_<b column> per column of b).
class LinearSolveOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "LinearSolve"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }
    int SecondTablePin() const override { return 1; }
    void SetSecondTable(std::shared_ptr<arrow::Table> table) override { b_table_ = std::move(table); }
    bool Configure(const std::map<std::string, std::string>& params, std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(const std::shared_ptr<arrow::Table>& input) override;

private:
    std::vector<std::string> a_cols_, b_cols_;
    std::string method_ = "auto";  // "auto", "solve" or "least_squares"
    std::shared_ptr<arrow::Table> b_table_;
};

// Every row times a matrix from the Matrix input (TOFIX129 D9): the matrix
// table's number columns are the outputs, its rows the data's columns in
// order (or named by its text column, matched by name). Output as the
// fitted steps: the outputs alone (y, the split) or beside the input.
class LinearProjectionOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "LinearProjection"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }
    int SecondTablePin() const override { return 1; }
    void SetSecondTable(std::shared_ptr<arrow::Table> table) override { matrix_ = std::move(table); }
    bool Configure(const std::map<std::string, std::string>& params, std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(const std::shared_ptr<arrow::Table>& input) override;

private:
    std::vector<std::string> feature_cols_;
    std::string label_col_;
    std::string prefix_;
    bool keep_columns_ = false;
    std::shared_ptr<arrow::Table> matrix_;
};

}  // namespace cyxwiz
