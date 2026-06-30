#pragma once

#include "pipeline_operator.h"

namespace cyxwiz {

/**
 * CountVectorizerOperator — Cat-1 Band 1 pipeline operator.
 *
 * Sibling of TFIDFVectorizerOperator with IDF disabled. Produces a
 * bag-of-words (term frequency) matrix:
 *   `count_0, count_1, ..., count_{max_features-1}, y`
 *
 * Each column is the term-frequency value for one vocabulary term
 * (count divided by document total tokens). Use this when you want
 * BoW features without the IDF reweighting — common baseline for
 * classical ML and a useful contrast against TFIDFVectorizer.
 *
 * Implementation note: backed by the same `TextProcessing::ComputeTFIDF`
 * call as TFIDFVectorizer with `use_idf=false`. Schema parallels
 * TFIDFVectorizer (`tfidf_*` → `count_*`); same `max_features` capping
 * by document frequency. Future: a true raw-count variant (no
 * normalization) would be a `mode="raw"` switch.
 *
 * Params:
 *   text_col          (required)        — string column with raw text
 *   label_col         (optional)        — column to use as label
 *   max_features      (default 2000)    — top-N terms by document frequency
 *   norm              (default "l2")    — "l1" / "l2" / "none"
 */
class CountVectorizerOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "CountVectorizer"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(
        const std::map<std::string, std::string>& params,
        std::string& error) override;

    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

    void SetProgressCallback(PipelineOperatorProgressCallback callback) override;

private:
    std::string text_col_;
    std::string label_col_;
    int max_features_ = 2000;
    std::string norm_ = "l2";
    PipelineOperatorProgressCallback progress_callback_;
};

} // namespace cyxwiz
