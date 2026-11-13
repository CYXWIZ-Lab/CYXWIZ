#pragma once

#include "pipeline_operator.h"
#include <utility>

namespace cyxwiz {

/**
 * SentimentAnalyzerOperator — Cat-1 Band 1 pipeline operator.
 *
 * VADER sentiment (TextProcessing::Vader, vaderSentiment 3.3.2's rules and
 * lexicon) of a text column, added beside the input's columns so the label
 * and the split stay (TOFIX129 D8):
 *   `<text_col>_compound`  (f64, -1 .. 1: the valences summed and normalised)
 *   `<text_col>_positive`, `_neutral`, `_negative` (f64, the shares of the
 *                          text's weight, summing to 1)
 *   `<text_col>_sentiment` (string: "positive" from compound 0.05,
 *                          "negative" at -0.05 and below, else "neutral")
 * A row without text gets nulls. Training reads the columns as features;
 * `<text_col>_sentiment` can be a target through a Label Encoder.
 *
 * Params:
 *   text_col   (required)  — column containing text.
 */
class SentimentAnalyzerOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "SentimentAnalyzer"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    void SetProgressCallback(PipelineOperatorProgressCallback callback) override {
        progress_callback_ = std::move(callback);
    }

    bool Configure(
        const std::map<std::string, std::string>& params,
        std::string& error) override;

    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

private:
    std::string text_col_;
    PipelineOperatorProgressCallback progress_callback_;
};

} // namespace cyxwiz
