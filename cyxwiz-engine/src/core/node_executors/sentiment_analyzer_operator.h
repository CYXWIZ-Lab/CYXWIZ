#pragma once

#include "pipeline_operator.h"
#include <utility>

namespace cyxwiz {

/**
 * SentimentAnalyzerOperator — Cat-1 Band 1 pipeline operator.
 *
 * Lexicon-based sentiment analysis. Reads a text column, runs each
 * document through `TextProcessing::AnalyzeSentiment` using one of the
 * built-in lexicons ("simple", "vader", "afinn"), and emits a wide
 * Arrow table:
 *   `polarity (f32), subjectivity (f32), sentiment_label (i32: 0=neg,
 *    1=neu, 2=pos), confidence (f32), y (i32, optional)`
 *
 * No external model files or pretrained weights — the backend ships the
 * lexicons in-binary. Useful both as a classification output itself
 * (`y = sentiment_label` for sentiment tasks) or as a feature block
 * feeding into a downstream classifier when combined with TFIDF/
 * CountVectorizer.
 *
 * Closes the SentimentAnalyzer dead NodeType from the "Tool-to-Node
 * Migration — text analytics" block in tofix.md. Pretrained BERT-style
 * sentiment models are intentionally deferred.
 *
 * Params:
 *   text_col   (required)                     — column containing text.
 *   label_col  (optional)                     — passthrough to `y` int32.
 *   method     (default "vader")              — "simple" / "vader" / "afinn".
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
    std::string label_col_;
    std::string method_ = "vader";
    PipelineOperatorProgressCallback progress_callback_;
};

} // namespace cyxwiz
