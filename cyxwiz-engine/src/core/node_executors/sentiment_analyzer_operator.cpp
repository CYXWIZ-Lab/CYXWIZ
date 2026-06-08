#include "sentiment_analyzer_operator.h"
#include "text_column_utils.h"

#include <cyxwiz/text_processing.h>

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

// Map the backend's string label ("positive"/"negative"/"neutral") to
// a stable int code so downstream Dense/Loss layers can use it as a
// classification target without a LabelEncoder node.
int SentimentLabelToInt(const std::string& label) {
    if (label == "positive") return 2;
    if (label == "negative") return 0;
    return 1;  // "neutral" or anything unrecognized
}

} // namespace

bool SentimentAnalyzerOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {

    text_col_.clear();
    label_col_.clear();
    method_ = "vader";

    auto it = params.find("text_col");
    if (it == params.end() || it->second.empty()) {
        error = "SentimentAnalyzer: 'text_col' parameter is required";
        return false;
    }
    text_col_ = it->second;

    auto lc = params.find("label_col");
    if (lc != params.end()) label_col_ = lc->second;

    auto m = params.find("method");
    if (m != params.end() && !m->second.empty()) {
        method_ = m->second;
        if (method_ != "simple" && method_ != "vader" && method_ != "afinn") {
            error = "SentimentAnalyzer: 'method' must be 'simple' / 'vader' / "
                    "'afinn' (got '" + method_ + "')";
            return false;
        }
    }

    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
SentimentAnalyzerOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    if (!input) {
        return arrow::Status::Invalid("SentimentAnalyzer: input table is null");
    }

    auto text_column = input->GetColumnByName(text_col_);
    if (!text_column) {
        return arrow::Status::KeyError(
            "SentimentAnalyzer: text column '" + text_col_ + "' not found");
    }

    std::vector<std::string> texts;
    std::string bad_type;
    if (!ReadColumnAsStrings(text_column, texts, bad_type)) {
        return arrow::Status::TypeError(
            "SentimentAnalyzer: text column '" + text_col_ +
            "' must be string/large_string, got '" + bad_type + "'");
    }

    const size_t n = texts.size();
    if (n == 0) {
        return arrow::Status::Invalid("SentimentAnalyzer: empty corpus");
    }

    // Optional label passthrough (resolved before the hot loop so we fail
    // fast on a bad label column).
    std::vector<int> labels;
    std::vector<std::string> class_names;
    if (!label_col_.empty()) {
        auto label_column = input->GetColumnByName(label_col_);
        if (!label_column) {
            return arrow::Status::KeyError(
                "SentimentAnalyzer: label column '" + label_col_ + "' not found");
        }
        std::string lbad;
        if (!ReadLabelColumnAsInt(label_column, labels, class_names, lbad)) {
            return arrow::Status::TypeError(
                "SentimentAnalyzer: label column '" + label_col_ +
                "' has unsupported type '" + lbad + "'");
        }
        if (labels.size() != n) {
            return arrow::Status::Invalid(
                "SentimentAnalyzer: label count (" + std::to_string(labels.size()) +
                ") differs from text count (" + std::to_string(n) + ")");
        }
    }

    arrow::MemoryPool* pool = arrow::default_memory_pool();
    arrow::FloatBuilder polarity_builder(pool);
    arrow::FloatBuilder subjectivity_builder(pool);
    arrow::Int32Builder sentiment_label_builder(pool);
    arrow::FloatBuilder confidence_builder(pool);
    arrow::Int32Builder label_builder(pool);

    ARROW_RETURN_NOT_OK(polarity_builder.Reserve(static_cast<int64_t>(n)));
    ARROW_RETURN_NOT_OK(subjectivity_builder.Reserve(static_cast<int64_t>(n)));
    ARROW_RETURN_NOT_OK(sentiment_label_builder.Reserve(static_cast<int64_t>(n)));
    ARROW_RETURN_NOT_OK(confidence_builder.Reserve(static_cast<int64_t>(n)));
    if (!labels.empty()) {
        ARROW_RETURN_NOT_OK(label_builder.Reserve(static_cast<int64_t>(n)));
    }

    int pos_count = 0, neg_count = 0, neu_count = 0, failed = 0;
    for (size_t r = 0; r < n; ++r) {
        auto result = TextProcessing::AnalyzeSentiment(texts[r], method_);
        if (!result.success) {
            // Failed analysis gets neutral zeroes — don't abort the whole
            // pipeline for one bad row. Tracked in the final summary.
            ARROW_RETURN_NOT_OK(polarity_builder.Append(0.0f));
            ARROW_RETURN_NOT_OK(subjectivity_builder.Append(0.0f));
            ARROW_RETURN_NOT_OK(sentiment_label_builder.Append(1));
            ARROW_RETURN_NOT_OK(confidence_builder.Append(0.0f));
            failed++;
        } else {
            const int sentiment_int = SentimentLabelToInt(result.label);
            ARROW_RETURN_NOT_OK(polarity_builder.Append(static_cast<float>(result.polarity)));
            ARROW_RETURN_NOT_OK(subjectivity_builder.Append(static_cast<float>(result.subjectivity)));
            ARROW_RETURN_NOT_OK(sentiment_label_builder.Append(sentiment_int));
            ARROW_RETURN_NOT_OK(confidence_builder.Append(static_cast<float>(result.confidence)));
            if (sentiment_int == 2) pos_count++;
            else if (sentiment_int == 0) neg_count++;
            else neu_count++;
        }
        if (!labels.empty()) {
            ARROW_RETURN_NOT_OK(label_builder.Append(labels[r]));
        }
    }

    std::vector<std::shared_ptr<arrow::Array>> arrays;
    std::vector<std::shared_ptr<arrow::Field>> fields;
    arrays.reserve(labels.empty() ? 4 : 5);
    fields.reserve(labels.empty() ? 4 : 5);

    std::shared_ptr<arrow::Array> arr;
    ARROW_RETURN_NOT_OK(polarity_builder.Finish(&arr));
    arrays.push_back(std::move(arr));
    fields.push_back(arrow::field("polarity", arrow::float32()));

    ARROW_RETURN_NOT_OK(subjectivity_builder.Finish(&arr));
    arrays.push_back(std::move(arr));
    fields.push_back(arrow::field("subjectivity", arrow::float32()));

    ARROW_RETURN_NOT_OK(sentiment_label_builder.Finish(&arr));
    arrays.push_back(std::move(arr));
    fields.push_back(arrow::field("sentiment_label", arrow::int32()));

    ARROW_RETURN_NOT_OK(confidence_builder.Finish(&arr));
    arrays.push_back(std::move(arr));
    fields.push_back(arrow::field("confidence", arrow::float32()));

    if (!labels.empty()) {
        ARROW_RETURN_NOT_OK(label_builder.Finish(&arr));
        arrays.push_back(std::move(arr));
        fields.push_back(arrow::field("y", arrow::int32()));
    }

    auto out_schema = arrow::schema(fields);
    auto out_table = arrow::Table::Make(out_schema, arrays, static_cast<int64_t>(n));

    spdlog::info("SentimentAnalyzer: {} docs, method={}, pos={} neu={} neg={}"
                 "{}",
                 n, method_, pos_count, neu_count, neg_count,
                 failed > 0 ? (", failed=" + std::to_string(failed)) : std::string());
    return out_table;
}

} // namespace cyxwiz
