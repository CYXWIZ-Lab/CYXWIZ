#pragma once

#include "pipeline_operator.h"
#include "text_feature_matrix.h"
#include "../preprocessing_state.h"

namespace cyxwiz {

/**
 * TFIDFVectorizerOperator — Cat-1 Band 1 pipeline operator.
 *
 * Transforms an Arrow table containing raw text + (optional) label
 * columns into a wide TF-IDF feature table ready for classical ML
 * downstream (Dense + softmax, logistic regression, etc.):
 *   `tfidf_0, tfidf_1, ..., tfidf_{max_features-1}, y`
 *
 * Applies sklearn-compatible raw term count multiplied by optional
 * smoothed IDF, followed by l1/l2/none row normalization. Vocabulary
 * capping via `max_features` keeps the output column count
 * bounded on large corpora. For a sentiment dataset with ~10k
 * unique words and 50k docs, an unbounded TF-IDF matrix would be
 * 2 GB of floats — `max_features=2000` keeps the top-N terms by
 * IDF score (rarest = most-discriminating words first), matching
 * sklearn's TfidfVectorizer(max_features=N) semantics.
 *
 * Schema parallels TextTokenizerOperator: `tok_*` becomes `tfidf_*`,
 * label column is `y` int32. ArrowDatasetBatcher handles the wide
 * output unchanged. Same caveats: this operator only fires when the
 * source dataset is registered as an Arrow table; legacy
 * `RegisterTextDataset` graphs skip it.
 *
 * Params:
 *   text_col          (required)        — string column with raw text
 *   label_col         (optional)        — column to use as label
 *   max_features      (default 2000)    — top-N terms by IDF (>= 1)
 *   min_df            (default 1)       — keep terms seen in >= N docs
 *   use_idf           (default true)    — multiply TF by IDF
 *   smooth_idf        (default true)    — add 1 to doc frequencies
 *   norm              (default "l2")    — "l1" / "l2" / "none"
 */
class TFIDFVectorizerOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "TFIDFVectorizer"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }
    bool IsCacheable() const override {
        return state_options_.IsTransformOnly() ||
               !state_options_.save_state;
    }
    bool CollectCacheDependencies(
        std::vector<PipelineOperatorCacheDependency>& dependencies,
        std::string& error) const override;

    bool Configure(
        const std::map<std::string, std::string>& params,
        std::string& error) override;

    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

    arrow::Result<std::shared_ptr<SparseFeatureDataset>> ApplySparse(
        const std::shared_ptr<arrow::Table>& input,
        const std::string& dataset_name) override;
    bool SupportsSparseFeatureOutput() const override { return true; }

    void SetProgressCallback(PipelineOperatorProgressCallback callback) override;

private:
    std::map<std::string, std::string> BuildFittedConfiguration() const;
    arrow::Result<TextVectorizerMaterialization> ApplyConfigured(
        const std::shared_ptr<arrow::Table>& input,
        const std::string& sparse_dataset_name);

    std::string text_col_;
    std::string label_col_;
    int max_features_ = 2000;
    int min_df_ = 1;
    int ngram_min_ = 1;
    int ngram_max_ = 1;
    bool use_idf_ = true;
    bool smooth_idf_ = true;
    std::string norm_ = "l2";
    std::string stop_words_ = "english";
    std::string output_format_ = "dense";
    FittedPreprocessingOptions state_options_;
    PipelineOperatorProgressCallback progress_callback_;
};

} // namespace cyxwiz
