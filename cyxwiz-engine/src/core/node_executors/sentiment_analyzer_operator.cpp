#include "sentiment_analyzer_operator.h"
#include "../profiler_trace.h"
#include "materialization_memory_preflight.h"
#include "text_column_utils.h"

#include <cyxwiz/text_processing.h>

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

namespace {

bool IsStringColumn(const std::shared_ptr<arrow::ChunkedArray>& column) {
    return column &&
           (column->type()->id() == arrow::Type::STRING ||
            column->type()->id() == arrow::Type::LARGE_STRING);
}

bool IsSupportedLabelColumn(
    const std::shared_ptr<arrow::ChunkedArray>& column) {
    if (!column) return false;
    switch (column->type()->id()) {
    case arrow::Type::STRING:
    case arrow::Type::LARGE_STRING:
    case arrow::Type::INT64:
    case arrow::Type::INT32:
    case arrow::Type::INT16:
    case arrow::Type::INT8:
    case arrow::Type::UINT8:
    case arrow::Type::UINT16:
    case arrow::Type::UINT32:
    case arrow::Type::FLOAT:
    case arrow::Type::DOUBLE:
        return true;
    default:
        return false;
    }
}

uint64_t StringPayloadBufferBytes(
    const std::shared_ptr<arrow::ChunkedArray>& column) {
    uint64_t total = 0;
    if (!IsStringColumn(column)) return total;
    for (const auto& chunk : column->chunks()) {
        const auto& buffers = chunk->data()->buffers;
        if (buffers.size() <= 2 || !buffers[2]) continue;
        uint64_t next = 0;
        if (!CheckedAddU64(
                total, static_cast<uint64_t>(buffers[2]->size()), next)) {
            return (std::numeric_limits<uint64_t>::max)();
        }
        total = next;
    }
    return total;
}

void AddSentimentTemporaryBytes(
    MaterializationMemoryEstimate& estimate,
    uint64_t bytes) {
    uint64_t temporary = 0;
    uint64_t peak = 0;
    if (!CheckedAddU64(estimate.temporary_bytes, bytes, temporary) ||
        !CheckedAddU64(estimate.estimated_peak_bytes, bytes, peak)) {
        estimate.overflow = true;
        estimate.temporary_bytes = (std::numeric_limits<uint64_t>::max)();
        estimate.estimated_peak_bytes =
            (std::numeric_limits<uint64_t>::max)();
        return;
    }
    estimate.temporary_bytes = temporary;
    estimate.estimated_peak_bytes = peak;
}

void ReportProgress(const PipelineOperatorProgressCallback& callback,
                    std::string stage,
                    std::string message,
                    float progress,
                    uint64_t processed_items = 0,
                    uint64_t total_items = 0,
                    uint64_t estimated_memory_bytes = 0) {
    if (!callback) return;
    PipelineOperatorProgress event;
    event.stage = std::move(stage);
    event.message = std::move(message);
    event.progress = progress;
    event.processed_items = processed_items;
    event.total_items = total_items;
    event.estimated_memory_bytes = estimated_memory_bytes;
    callback(event);
}

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
        method_ = NormalizeTextParameterChoice(m->second);
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
    CYXWIZ_PROFILE_ZONE("CyxWiz SentimentAnalyzer Materializer");
    if (!input) {
        return arrow::Status::Invalid("SentimentAnalyzer: input table is null");
    }

    auto text_column = input->GetColumnByName(text_col_);
    if (!text_column) {
        return arrow::Status::KeyError(
            "SentimentAnalyzer: text column '" + text_col_ + "' not found");
    }
    if (!IsStringColumn(text_column)) {
        return arrow::Status::TypeError(
            "SentimentAnalyzer: text column '" + text_col_ +
            "' must be string/large_string, got '" +
            text_column->type()->ToString() + "'");
    }

    std::shared_ptr<arrow::ChunkedArray> label_column;
    if (!label_col_.empty()) {
        label_column = input->GetColumnByName(label_col_);
        if (!label_column) {
            return arrow::Status::KeyError(
                "SentimentAnalyzer: label column '" + label_col_ + "' not found");
        }
        if (!IsSupportedLabelColumn(label_column)) {
            return arrow::Status::TypeError(
                "SentimentAnalyzer: label column '" + label_col_ +
                "' has unsupported type '" + label_column->type()->ToString() +
                "'");
        }
    }

    const uint64_t planned_rows = static_cast<uint64_t>(
        std::max<int64_t>(0, input->num_rows()));
    const uint64_t output_columns = label_column ? 5ULL : 4ULL;
    auto sentiment_estimate = EstimateDenseMaterializationMemory(
        planned_rows,
        output_columns,
        static_cast<uint64_t>(sizeof(float)));

    uint64_t text_objects = 0;
    if (!CheckedMulU64(
            planned_rows,
            static_cast<uint64_t>(sizeof(std::string)),
            text_objects)) {
        sentiment_estimate.overflow = true;
        text_objects = (std::numeric_limits<uint64_t>::max)();
    }
    uint64_t text_copy_bytes = 0;
    if (!CheckedAddU64(
            text_objects,
            StringPayloadBufferBytes(text_column),
            text_copy_bytes)) {
        sentiment_estimate.overflow = true;
        text_copy_bytes = (std::numeric_limits<uint64_t>::max)();
    }
    AddSentimentTemporaryBytes(sentiment_estimate, text_copy_bytes);

    if (label_column) {
        uint64_t label_bytes = 0;
        if (!CheckedMulU64(
                planned_rows,
                static_cast<uint64_t>(sizeof(int)),
                label_bytes)) {
            sentiment_estimate.overflow = true;
            label_bytes = (std::numeric_limits<uint64_t>::max)();
        }
        if (IsStringColumn(label_column)) {
            // Worst case: every label is unique, so class_names and the map
            // each own a string copy in addition to the integer label vector.
            uint64_t label_string_objects = 0;
            uint64_t label_payload_copies = 0;
            uint64_t label_string_bytes = 0;
            if (!CheckedMulU64(
                    planned_rows,
                    2ULL * static_cast<uint64_t>(sizeof(std::string)),
                    label_string_objects) ||
                !CheckedMulU64(
                    StringPayloadBufferBytes(label_column),
                    2,
                    label_payload_copies) ||
                !CheckedAddU64(
                    label_string_objects,
                    label_payload_copies,
                    label_string_bytes) ||
                !CheckedAddU64(label_bytes, label_string_bytes, label_bytes)) {
                sentiment_estimate.overflow = true;
                label_bytes = (std::numeric_limits<uint64_t>::max)();
            }
        }
        AddSentimentTemporaryBytes(sentiment_estimate, label_bytes);
    }

    ARROW_ASSIGN_OR_RAISE(
        auto preflight_estimate,
        EmitMaterializationMemoryPreflight(
            sentiment_estimate,
            "SentimentAnalyzer",
            "planned_output_columns",
            "Reduce corpus rows or analyze a smaller text sample first.",
            GetMaterializationMemoryContext(),
            progress_callback_,
            SaturatingMaterializationItemCount(planned_rows, output_columns),
            0.15f));
    ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));

    ReportProgress(progress_callback_, "read_text",
                   "Reading text column for sentiment analysis", 0.20f, 0,
                   planned_rows,
                   preflight_estimate.estimated_peak_bytes);

    std::vector<std::string> texts;
    std::string bad_type;
    if (!ReadColumnAsStrings(
            text_column, texts, bad_type, GetCancellationQuery())) {
        ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
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
        ReportProgress(progress_callback_, "read_labels",
                       "Reading sentiment label passthrough column", 0.20f, 0,
                       static_cast<uint64_t>(n));
        std::string lbad;
        if (!ReadLabelColumnAsInt(
                label_column, labels, class_names, lbad,
                GetCancellationQuery())) {
            ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
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

    ReportProgress(progress_callback_, "analyze",
                   "Analyzing sentiment documents", 0.35f, 0,
                   static_cast<uint64_t>(n),
                    preflight_estimate.estimated_peak_bytes);
    int pos_count = 0, neg_count = 0, neu_count = 0, failed = 0;
    for (size_t r = 0; r < n; ++r) {
        if ((r & 1023) == 0) {
            ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
        }
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
        if (r + 1 == n || (r + 1) % 1000 == 0) {
            const float progress =
                0.35f + (0.45f * static_cast<float>(r + 1) /
                         static_cast<float>(n));
            ReportProgress(progress_callback_, "analyze",
                           "Analyzing sentiment documents", progress,
                           static_cast<uint64_t>(r + 1),
                           static_cast<uint64_t>(n),
                            preflight_estimate.estimated_peak_bytes);
        }
    }

    ReportProgress(progress_callback_, "finalize",
                   "Finalizing sentiment output table", 0.90f,
                   static_cast<uint64_t>(n), static_cast<uint64_t>(n),
                    preflight_estimate.estimated_peak_bytes);
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
    ReportProgress(progress_callback_, "complete",
                   "Sentiment analysis complete", 1.0f,
                   static_cast<uint64_t>(n), static_cast<uint64_t>(n),
                    preflight_estimate.estimated_peak_bytes);
    return out_table;
}

} // namespace cyxwiz
