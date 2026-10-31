#pragma once

// The t-SNE and UMAP nodes (TOFIX129 D5): a 2-D or 3-D embedding of the chosen
// numeric columns, added beside the input (tsne_0, tsne_1 / umap_0, umap_1;
// null on rows missing a feature) or alone. They fit on every row with all
// features (Explore > Projection samples 5,000 rows for a quick look; the node
// is the full run), standardised by default as the Projection tab does.
// t-SNE stops when the run is cancelled.
//
// Params (both): feature_cols (comma-separated; empty = every numeric column
// but label_col), label_col, standardise (true), keep_columns (true),
// n_components (2 or 3).
// t-SNE: perplexity (30), learning_rate (200), iterations (1000).
// UMAP: n_neighbors (15), min_dist (0.1).

#include "pipeline_operator.h"

#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

class ProjectionOperatorBase : public IPipelineOperator {
public:
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }
    void SetProgressCallback(PipelineOperatorProgressCallback callback) override { progress_callback_ = std::move(callback); }
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(const std::shared_ptr<arrow::Table>& input) override;

protected:
    bool ConfigureBase(const std::map<std::string, std::string>& params, std::string& error);
    // The embedding of `data` [rows][features], one row per input row.
    virtual arrow::Result<std::vector<std::vector<double>>> Embed(const std::vector<std::vector<double>>& data) = 0;
    virtual std::string Prefix() const = 0;
    void Report(const std::string& message, double progress) const;

    std::vector<std::string> feature_cols_;
    std::string label_col_;
    bool standardise_ = true;
    bool keep_columns_ = true;
    int n_components_ = 2;
    PipelineOperatorProgressCallback progress_callback_;
};

class TSNEOperator : public ProjectionOperatorBase {
public:
    std::string GetName() const override { return "TSNE"; }
    bool Configure(const std::map<std::string, std::string>& params, std::string& error) override;

private:
    arrow::Result<std::vector<std::vector<double>>> Embed(const std::vector<std::vector<double>>& data) override;
    std::string Prefix() const override { return "tsne_"; }

    int perplexity_ = 30;
    double learning_rate_ = 200.0;
    int iterations_ = 1000;
};

class UMAPOperator : public ProjectionOperatorBase {
public:
    std::string GetName() const override { return "UMAP"; }
    bool Configure(const std::map<std::string, std::string>& params, std::string& error) override;

private:
    arrow::Result<std::vector<std::vector<double>>> Embed(const std::vector<std::vector<double>>& data) override;
    std::string Prefix() const override { return "umap_"; }

    int n_neighbors_ = 15;
    double min_dist_ = 0.1;
};

}  // namespace cyxwiz
