#pragma once

#include "pipeline_operator.h"
#include "../preprocessing_state.h"
#include <cyxwiz/data_transform.h>
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
 * Fitted column transforms (TOFIX129 D2): the scalers and the quantile and
 * power transforms. One base does what they share:
 *   - columns: `columns` (csv), or every number column except `label_col`
 *     and `exclude_columns`;
 *   - Fit here (and optionally save the fitted values to `state_path`), or
 *     reuse saved values (`operation_mode` fit_transform / transform_only,
 *     `save_state`, `state_overwrite`); saved values must come from the same
 *     settings;
 *   - a missing value stays missing in its row; the fit uses the values
 *     present (scikit-learn's behaviour);
 *   - the streamed path (data bigger than memory) for transforms whose fit
 *     merges across batches (Standard, Min-Max, Max-Abs).
 * Each transform supplies its settings, its fit and its apply.
 */
class ColumnTransformOperator : public IPipelineOperator {
public:
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }
    // The fit learns from rows the cache key covers (the source, the steps
    // before and the Split's Train rows), so the output is kept, unless the
    // step saves its fitted values: a hit would not write them (TOFIX178 0.5).
    bool IsCacheable() const override { return options_.IsTransformOnly() || !options_.save_state; }
    // Use saved values: the values file is in the key.
    bool CollectCacheDependencies(std::vector<PipelineOperatorCacheDependency>& dependencies,
                                  std::string&) const override {
        if (options_.IsTransformOnly()) dependencies.push_back({"fitted_state", options_.state_path});
        return true;
    }
    bool Configure(const std::map<std::string, std::string>& params,
                   std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;
    void SetProgressCallback(PipelineOperatorProgressCallback callback) override {
        progress_callback_ = std::move(callback);
    }
    bool SupportsStreaming() const override { return Streams(); }
    bool NeedsFitPass() const override { return !options_.IsTransformOnly(); }
    arrow::Status BeginStream(const std::shared_ptr<arrow::Schema>& schema) override;
    arrow::Status FitBatch(const std::shared_ptr<arrow::Table>& batch) override;
    arrow::Status EndFit() override;
    arrow::Result<std::shared_ptr<arrow::Table>> ApplyBatch(const std::shared_ptr<arrow::Table>& batch) override;
    arrow::Status EndStream() override;

protected:
    // The transform's own parameters.
    virtual bool ConfigureOwn(const std::map<std::string, std::string>& params, std::string& error) = 0;
    // Settings saved with the fitted values; reusing them needs the same.
    virtual std::map<std::string, std::string> Settings() const = 0;
    virtual bool Streams() const = 0;
    // A fit over the present values of each column, batch by batch.
    virtual void BeginFit(size_t columns) = 0;
    virtual arrow::Status FitValues(size_t column, const std::vector<double>& present) = 0;
    // The fitted values of a column into `feature.numeric_values`.
    virtual arrow::Status EndColumnFit(size_t column, PreprocessingFeatureState& feature) = 0;
    // Saved values of a column back (Use saved values).
    virtual arrow::Status LoadColumn(size_t column, const PreprocessingFeatureState& feature) = 0;
    virtual arrow::Result<std::vector<double>> TransformValues(size_t column, std::vector<double> present) = 0;
    // "Standard scaling", for progress and the log.
    virtual std::string Describe() const = 0;

    const std::string& ColumnName(size_t column) const { return resolved_[column]; }

private:
    std::vector<std::string> columns_;
    std::vector<std::string> exclude_columns_;
    std::string label_col_;
    FittedPreprocessingOptions options_;
    PipelineOperatorProgressCallback progress_callback_;
    std::shared_ptr<arrow::Schema> schema_;
    std::vector<std::string> resolved_;
    FittedPreprocessingState state_;
    int64_t fit_rows_ = 0;
};

/**
 * StandardScalerOperator: (x - mean) / std (population std, scikit-learn).
 * Params: with_mean, with_std (default true). Streams.
 */
class StandardScalerOperator : public ColumnTransformOperator {
public:
    std::string GetName() const override { return "StandardScaler"; }

protected:
    bool ConfigureOwn(const std::map<std::string, std::string>& params, std::string& error) override;
    std::map<std::string, std::string> Settings() const override;
    bool Streams() const override { return true; }
    void BeginFit(size_t columns) override;
    arrow::Status FitValues(size_t column, const std::vector<double>& present) override;
    arrow::Status EndColumnFit(size_t column, PreprocessingFeatureState& feature) override;
    arrow::Status LoadColumn(size_t column, const PreprocessingFeatureState& feature) override;
    arrow::Result<std::vector<double>> TransformValues(size_t column, std::vector<double> present) override;
    std::string Describe() const override { return "Standard scaling"; }

private:
    bool with_mean_ = true;
    bool with_std_ = true;
    std::vector<DataTransform::ColumnMoments> moments_;
    std::vector<double> centers_, scales_;
};

/**
 * MinMaxScalerOperator: (x - min) / (max - min) into min .. max (default 0 .. 1).
 * Streams.
 */
class MinMaxScalerOperator : public ColumnTransformOperator {
public:
    std::string GetName() const override { return "MinMaxScaler"; }

protected:
    bool ConfigureOwn(const std::map<std::string, std::string>& params, std::string& error) override;
    std::map<std::string, std::string> Settings() const override;
    bool Streams() const override { return true; }
    void BeginFit(size_t columns) override;
    arrow::Status FitValues(size_t column, const std::vector<double>& present) override;
    arrow::Status EndColumnFit(size_t column, PreprocessingFeatureState& feature) override;
    arrow::Status LoadColumn(size_t column, const PreprocessingFeatureState& feature) override;
    arrow::Result<std::vector<double>> TransformValues(size_t column, std::vector<double> present) override;
    std::string Describe() const override { return "Min-max scaling"; }

private:
    double range_min_ = 0.0;
    double range_max_ = 1.0;
    std::vector<DataTransform::ColumnMoments> moments_;
    std::vector<double> lows_, highs_;
};

/**
 * RobustScalerOperator: (x - median) / (Q_high - Q_low), outlier-resistant.
 * Params: with_centering, with_scaling (default true), quantile_min /
 * quantile_max (default 25 / 75). Fits on the whole column.
 */
class RobustScalerOperator : public ColumnTransformOperator {
public:
    std::string GetName() const override { return "RobustScaler"; }

protected:
    bool ConfigureOwn(const std::map<std::string, std::string>& params, std::string& error) override;
    std::map<std::string, std::string> Settings() const override;
    bool Streams() const override { return false; }
    void BeginFit(size_t columns) override;
    arrow::Status FitValues(size_t column, const std::vector<double>& present) override;
    arrow::Status EndColumnFit(size_t column, PreprocessingFeatureState& feature) override;
    arrow::Status LoadColumn(size_t column, const PreprocessingFeatureState& feature) override;
    arrow::Result<std::vector<double>> TransformValues(size_t column, std::vector<double> present) override;
    std::string Describe() const override { return "Robust scaling"; }

private:
    bool with_centering_ = true;
    bool with_scaling_ = true;
    double quantile_min_ = 25.0;
    double quantile_max_ = 75.0;
    std::vector<std::vector<double>> values_;
    std::vector<double> centers_, scales_;
};

/**
 * MaxAbsScalerOperator: x / max |x|, into -1 .. 1 (keeps zeros: sparse data).
 * A column of zeros is left as it is. Streams.
 */
class MaxAbsScalerOperator : public ColumnTransformOperator {
public:
    std::string GetName() const override { return "MaxAbsScaler"; }

protected:
    bool ConfigureOwn(const std::map<std::string, std::string>&, std::string&) override { return true; }
    std::map<std::string, std::string> Settings() const override { return {}; }
    bool Streams() const override { return true; }
    void BeginFit(size_t columns) override;
    arrow::Status FitValues(size_t column, const std::vector<double>& present) override;
    arrow::Status EndColumnFit(size_t column, PreprocessingFeatureState& feature) override;
    arrow::Status LoadColumn(size_t column, const PreprocessingFeatureState& feature) override;
    arrow::Result<std::vector<double>> TransformValues(size_t column, std::vector<double> present) override;
    std::string Describe() const override { return "Max-abs scaling"; }

private:
    std::vector<DataTransform::ColumnMoments> moments_;
    std::vector<double> scales_;
};

/**
 * QuantileTransformOperator: each column through its quantiles onto a
 * uniform 0 .. 1 or a standard normal shape (scikit-learn
 * QuantileTransformer). Params: output_distribution (uniform / normal,
 * default uniform), n_quantiles (default 1000). Fits on the whole column.
 */
class QuantileTransformOperator : public ColumnTransformOperator {
public:
    std::string GetName() const override { return "QuantileTransform"; }

protected:
    bool ConfigureOwn(const std::map<std::string, std::string>& params, std::string& error) override;
    std::map<std::string, std::string> Settings() const override;
    bool Streams() const override { return false; }
    void BeginFit(size_t columns) override;
    arrow::Status FitValues(size_t column, const std::vector<double>& present) override;
    arrow::Status EndColumnFit(size_t column, PreprocessingFeatureState& feature) override;
    arrow::Status LoadColumn(size_t column, const PreprocessingFeatureState& feature) override;
    arrow::Result<std::vector<double>> TransformValues(size_t column, std::vector<double> present) override;
    std::string Describe() const override { return "Quantile transform"; }

private:
    std::string distribution_ = "uniform";
    int n_quantiles_ = 1000;
    std::vector<std::vector<double>> values_;
    std::vector<std::vector<double>> quantiles_;
};

/**
 * PowerTransformOperator: Box-Cox (values above 0) or Yeo-Johnson (any
 * values), lambda found by maximum likelihood per column or typed
 * (scipy.stats.boxcox / yeojohnson). Params: family (box-cox / yeo-johnson,
 * default yeo-johnson), lambda_mode (auto / fixed, default auto), lambda.
 * Fits on the whole column.
 */
class PowerTransformOperator : public ColumnTransformOperator {
public:
    std::string GetName() const override { return "PowerTransform"; }

protected:
    bool ConfigureOwn(const std::map<std::string, std::string>& params, std::string& error) override;
    std::map<std::string, std::string> Settings() const override;
    bool Streams() const override { return false; }
    void BeginFit(size_t columns) override;
    arrow::Status FitValues(size_t column, const std::vector<double>& present) override;
    arrow::Status EndColumnFit(size_t column, PreprocessingFeatureState& feature) override;
    arrow::Status LoadColumn(size_t column, const PreprocessingFeatureState& feature) override;
    arrow::Result<std::vector<double>> TransformValues(size_t column, std::vector<double> present) override;
    std::string Describe() const override { return box_cox_ ? "Box-Cox transform" : "Yeo-Johnson transform"; }

private:
    bool box_cox_ = false;
    bool auto_lambda_ = true;
    double lambda_ = 0.0;
    std::vector<std::vector<double>> values_;
    std::vector<double> lambdas_;
};

/**
 * LogTransformOperator: log(x) or log(1 + x), in base e, 10 or 2 (TOFIX129 D2).
 * Params: columns (csv; empty = every number column except label_col and
 * exclude_columns), base (natural / log10 / log2, default natural),
 * use_log1p (default true). Nothing is fitted; a missing value stays missing
 * in its row. Streams (row by row).
 */
class LogTransformOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "LogTransform"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }
    void SetProgressCallback(PipelineOperatorProgressCallback callback) override {
        progress_callback_ = std::move(callback);
    }
    bool Configure(const std::map<std::string, std::string>& params, std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(const std::shared_ptr<arrow::Table>& input) override;
    bool SupportsStreaming() const override { return true; }
    arrow::Status BeginStream(const std::shared_ptr<arrow::Schema>& schema) override;
    arrow::Result<std::shared_ptr<arrow::Table>> ApplyBatch(const std::shared_ptr<arrow::Table>& batch) override;

private:
    std::vector<std::string> columns_;
    std::vector<std::string> exclude_columns_;
    std::string label_col_;
    std::string base_ = "natural";
    bool log1p_ = true;
    std::vector<std::string> resolved_;
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
    bool LearnsFromEveryRow() const override { return true; }

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
