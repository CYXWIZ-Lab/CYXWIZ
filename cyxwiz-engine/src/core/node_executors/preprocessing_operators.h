#pragma once

#include "pipeline_operator.h"
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

/**
 * Shared base for all Phase 4 data preprocessing Cat-1 operators.
 * Most subclasses operate on a set of numeric feature columns
 * (auto-detected by default, or explicitly listed) and replace each
 * column in-place. Encoders replace string columns with int columns.
 *
 * Closes the "Data preprocessing (Phase 4 block)" dead NodeTypes from
 * the Tool-to-Node Migration (StandardScaler, MinMaxScaler,
 * RobustScaler, LabelEncoder, OrdinalEncoder, TargetEncoder,
 * OutlierDetector).
 */

/**
 * StandardScalerOperator — z-score standardization per column.
 *
 * Applies `(x - mean) / std` column-by-column. When with_mean=false
 * it skips centering; when with_std=false it skips scaling.
 *
 * Params:
 *   columns    (optional, csv) — empty = auto-detect numeric.
 *   label_col  (optional)      — excluded from auto-detect.
 *   with_mean  (default true)
 *   with_std   (default true)
 */
class StandardScalerOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "StandardScaler"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }
    // Fit writes external state; Transform Only reads a mutable artifact.
    // Re-enable caching only after artifact content identity joins cache keys.
    bool IsCacheable() const override { return false; }

    bool Configure(const std::map<std::string, std::string>& params,
                   std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

    void SetProgressCallback(PipelineOperatorProgressCallback callback) override {
        progress_callback_ = std::move(callback);
    }

private:
    std::vector<std::string> columns_;
    std::vector<std::string> exclude_columns_;
    std::string label_col_;
    bool with_mean_ = true;
    bool with_std_ = true;
    std::string operation_mode_ = "fit_transform";
    std::string state_path_;
    bool save_state_ = false;
    bool state_overwrite_ = false;
    PipelineOperatorProgressCallback progress_callback_;
};

/**
 * MinMaxScalerOperator — scale each column to a user-specified range.
 *
 * `x' = (x - data_min) / (data_max - data_min) * (range_max - range_min) + range_min`
 *
 * Params:
 *   columns    (optional, csv)
 *   label_col  (optional)
 *   min        (default 0.0)   — target range minimum.
 *   max        (default 1.0)   — target range maximum.
 */
class MinMaxScalerOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "MinMaxScaler"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params,
                   std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

    void SetProgressCallback(PipelineOperatorProgressCallback callback) override {
        progress_callback_ = std::move(callback);
    }

private:
    std::vector<std::string> columns_;
    std::string label_col_;
    double range_min_ = 0.0;
    double range_max_ = 1.0;
    PipelineOperatorProgressCallback progress_callback_;
};

/**
 * RobustScalerOperator — median/IQR scaling, outlier-resistant.
 *
 * `x' = (x - median) / (Q_high - Q_low)`. Unlike StandardScaler, this
 * uses percentiles so heavy outliers don't dominate the scaling.
 *
 * Params:
 *   columns         (optional, csv)
 *   label_col       (optional)
 *   with_centering  (default true)
 *   with_scaling    (default true)
 *   quantile_min    (default 25.0) — lower percentile.
 *   quantile_max    (default 75.0) — upper percentile.
 */
class RobustScalerOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "RobustScaler"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params,
                   std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

    void SetProgressCallback(PipelineOperatorProgressCallback callback) override {
        progress_callback_ = std::move(callback);
    }

private:
    std::vector<std::string> columns_;
    std::string label_col_;
    bool with_centering_ = true;
    bool with_scaling_ = true;
    double quantile_min_ = 25.0;
    double quantile_max_ = 75.0;
    PipelineOperatorProgressCallback progress_callback_;
};

/**
 * LabelEncoderOperator — string → int32, single column.
 *
 * Encodes a string column to stable integer codes using alphabetical
 * ordering. Column type changes string → int32. Typical use: target
 * columns for classification.
 *
 * Params:
 *   column  (required)  — single column to encode.
 */
class LabelEncoderOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "LabelEncoder"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params,
                   std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

    void SetProgressCallback(PipelineOperatorProgressCallback callback) override {
        progress_callback_ = std::move(callback);
    }

private:
    std::string column_;
    PipelineOperatorProgressCallback progress_callback_;
};

/**
 * OrdinalEncoderOperator — string → int32, multiple columns.
 *
 * Like LabelEncoder but accepts a comma-separated list of columns.
 * Categories use alphabetical auto-ordering. Custom per-column
 * ordering is deferred to tofix (requires a nested param schema).
 *
 * Params:
 *   columns   (required, csv)  — columns to encode.
 *   categories (default "auto") — only "auto" supported in v1.
 */
class OrdinalEncoderOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "OrdinalEncoder"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params,
                   std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

    void SetProgressCallback(PipelineOperatorProgressCallback callback) override {
        progress_callback_ = std::move(callback);
    }

private:
    std::vector<std::string> columns_;
    PipelineOperatorProgressCallback progress_callback_;
};

/**
 * TargetEncoderOperator — category column → smoothed target mean.
 *
 * For each categorical column, replaces each category value with a
 * smoothed mean of the target column for that category:
 *   encoded = (count * cat_mean + smoothing * global_mean)
 *           / (count + smoothing)
 *
 * Smoothing regularizes low-count categories toward the global mean.
 * Works with numeric target (mean encoding) and int/bool target
 * (frequency encoding for class 1).
 *
 * Params:
 *   columns    (required, csv) — categorical columns to encode.
 *   target_col (required)      — numeric target column.
 *   smoothing  (default 1.0)   — pseudo-count toward global mean.
 */
class TargetEncoderOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "TargetEncoder"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params,
                   std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

    void SetProgressCallback(PipelineOperatorProgressCallback callback) override {
        progress_callback_ = std::move(callback);
    }

private:
    std::vector<std::string> columns_;
    std::string target_col_;
    double smoothing_ = 1.0;
    PipelineOperatorProgressCallback progress_callback_;
};

/**
 * OutlierDetectorOperator — annotate rows with is_outlier bool.
 *
 * Adds an `is_outlier` int32 column (0/1) flagging rows where ANY
 * of the selected feature columns trip the IQR or Z-score threshold.
 * Does NOT filter rows — filtering would break alignment with other
 * pipeline data. Users who want to drop outliers should chain a
 * DataSplit node keyed on `is_outlier`.
 *
 * The existing node has action="remove"/"clip"/"flag"; this
 * operator ships only "flag" semantics. Row-deletion and value-
 * clipping variants are deferred to tofix.
 *
 * Params:
 *   columns    (optional, csv) — empty = auto-detect numeric.
 *   label_col  (optional)      — excluded from auto-detect.
 *   method     (default "iqr") — "iqr" or "zscore".
 *   threshold  (default 1.5)   — IQR multiplier or Z-score threshold.
 */
class OutlierDetectorOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "OutlierDetector"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params,
                   std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

    void SetProgressCallback(PipelineOperatorProgressCallback callback) override {
        progress_callback_ = std::move(callback);
    }

private:
    std::vector<std::string> columns_;
    std::string label_col_;
    std::string method_ = "iqr";
    double threshold_ = 1.5;
    PipelineOperatorProgressCallback progress_callback_;
};

} // namespace cyxwiz
