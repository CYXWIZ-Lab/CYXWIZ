#pragma once

#include "pipeline_operator.h"
#include <utility>
#include <vector>

namespace cyxwiz {

/**
 * PCAOperator — Cat-1 Band 1 pipeline operator.
 *
 * Principal Component Analysis dimensionality reduction. Transforms an
 * Arrow table of numeric features into a lower-dimensional projection:
 *   `pc_0, pc_1, ..., pc_{n_components-1}, y`
 *
 * Wraps `cyxwiz::DimensionalityReduction::ComputePCA` (SVD-based PCA
 * with optional centering and unit-variance scaling). Schema mirrors
 * TFIDFVectorizer / CountVectorizer / TextTokenizer — wide float
 * columns + optional `y` int label column.
 *
 * Useful as a preprocessing step BEFORE training: reduces 1000-dim
 * TF-IDF vectors to 50-dim dense vectors, makes downstream Dense
 * layers smaller and training faster, and acts as feature
 * decorrelation for classical ML.
 *
 * Closes the PCANode dead NodeType from tofix.md "Tool-to-Node
 * Migration" linear algebra block.
 *
 * Params:
 *   feature_cols  (optional, comma-sep)  — columns to use as features.
 *                                          Empty = auto-detect (all numeric
 *                                          columns except label_col).
 *   label_col     (optional)             — label column passed through to
 *                                          output as `y` int32.
 *   n_components  (default 2)            — dimensionality of projection.
 *   center        (default true)         — subtract per-feature mean.
 *   scale         (default false)        — divide by per-feature std (unit
 *                                          variance).
 */
class PCAOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "PCA"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(
        const std::map<std::string, std::string>& params,
        std::string& error) override;

    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

    void SetProgressCallback(PipelineOperatorProgressCallback callback) override {
        progress_callback_ = std::move(callback);
    }

private:
    std::vector<std::string> feature_cols_;  // empty = auto-detect
    std::string label_col_;
    int n_components_ = 2;
    bool center_ = true;
    bool scale_ = false;
    PipelineOperatorProgressCallback progress_callback_;
};

} // namespace cyxwiz
