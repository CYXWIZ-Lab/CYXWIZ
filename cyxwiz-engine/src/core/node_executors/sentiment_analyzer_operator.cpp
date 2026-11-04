#include "sentiment_analyzer_operator.h"
#include "../profiler_trace.h"
#include "materialization_memory_preflight.h"

#include <cyxwiz/text_processing.h>

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

namespace {

constexpr double kPositiveFrom = 0.05;   // VADER's own thresholds
constexpr double kNegativeBelow = -0.05;

bool IsStringColumn(const std::shared_ptr<arrow::ChunkedArray>& column) {
    return column &&
           (column->type()->id() == arrow::Type::STRING ||
            column->type()->id() == arrow::Type::LARGE_STRING);
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

} // namespace

bool SentimentAnalyzerOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {
    text_col_.clear();
    auto it = params.find("text_col");
    if (it == params.end() || it->second.empty()) {
        error = "SentimentAnalyzer: 'text_col' parameter is required";
        return false;
    }
    text_col_ = it->second;
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
    const std::array<std::string, 5> names = {
        text_col_ + "_compound", text_col_ + "_positive", text_col_ + "_neutral",
        text_col_ + "_negative", text_col_ + "_sentiment"};
    for (const std::string& name : names) {
        if (input->schema()->GetFieldIndex(name) >= 0) {
            return arrow::Status::Invalid(
                "SentimentAnalyzer: the table already has a column '" + name +
                "'; rename it or remove the earlier Sentiment Analysis node");
        }
    }

    const int64_t n = input->num_rows();
    const uint64_t rows = static_cast<uint64_t>(std::max<int64_t>(0, n));
    // Four float64 columns and a short label per row, beside the input.
    auto estimate = EstimateDenseMaterializationMemory(rows, 5ULL, static_cast<uint64_t>(sizeof(double)));
    ARROW_ASSIGN_OR_RAISE(
        auto preflight_estimate,
        EmitMaterializationMemoryPreflight(
            estimate,
            "SentimentAnalyzer",
            "planned_output_columns",
            "Reduce corpus rows or analyze a smaller text sample first.",
            GetMaterializationMemoryContext(),
            progress_callback_,
            SaturatingMaterializationItemCount(rows, 5ULL),
            0.15f));
    ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));

    arrow::MemoryPool* pool = arrow::default_memory_pool();
    std::array<arrow::DoubleBuilder, 4> scores = {arrow::DoubleBuilder(pool), arrow::DoubleBuilder(pool),
                                                  arrow::DoubleBuilder(pool), arrow::DoubleBuilder(pool)};
    arrow::StringBuilder label(pool);
    for (auto& b : scores) ARROW_RETURN_NOT_OK(b.Reserve(n));
    ARROW_RETURN_NOT_OK(label.Reserve(n));

    ReportProgress(progress_callback_, "analyze", "Scoring sentiment (VADER)", 0.20f, 0, rows,
                   preflight_estimate.estimated_peak_bytes);
    int64_t done = 0, positive = 0, negative = 0, neutral = 0, empty = 0;
    for (const auto& chunk : text_column->chunks()) {
        const bool large = chunk->type_id() == arrow::Type::LARGE_STRING;
        for (int64_t i = 0; i < chunk->length(); ++i, ++done) {
            if ((done & 1023) == 0) {
                ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
            }
            if (chunk->IsNull(i)) {
                for (auto& b : scores) ARROW_RETURN_NOT_OK(b.AppendNull());
                ARROW_RETURN_NOT_OK(label.AppendNull());
                ++empty;
                continue;
            }
            const std::string text = large
                ? std::string(static_cast<const arrow::LargeStringArray&>(*chunk).GetView(i))
                : std::string(static_cast<const arrow::StringArray&>(*chunk).GetView(i));
            const VaderScores s = TextProcessing::Vader(text);
            ARROW_RETURN_NOT_OK(scores[0].Append(s.compound));
            ARROW_RETURN_NOT_OK(scores[1].Append(s.positive));
            ARROW_RETURN_NOT_OK(scores[2].Append(s.neutral));
            ARROW_RETURN_NOT_OK(scores[3].Append(s.negative));
            const char* word = "neutral";
            if (s.compound >= kPositiveFrom) {
                word = "positive";
                ++positive;
            } else if (s.compound <= kNegativeBelow) {
                word = "negative";
                ++negative;
            } else {
                ++neutral;
            }
            ARROW_RETURN_NOT_OK(label.Append(word));
            if ((done + 1) % 1000 == 0) {
                ReportProgress(progress_callback_, "analyze", "Scoring sentiment (VADER)",
                               0.20f + 0.70f * static_cast<float>(done + 1) / static_cast<float>(std::max<int64_t>(1, n)),
                               static_cast<uint64_t>(done + 1), rows, preflight_estimate.estimated_peak_bytes);
            }
        }
    }

    std::shared_ptr<arrow::Table> out = input;
    for (size_t k = 0; k < 4; ++k) {
        std::shared_ptr<arrow::Array> array;
        ARROW_RETURN_NOT_OK(scores[k].Finish(&array));
        ARROW_ASSIGN_OR_RAISE(out, out->AddColumn(out->num_columns(), arrow::field(names[k], arrow::float64()),
                                                  std::make_shared<arrow::ChunkedArray>(array)));
    }
    std::shared_ptr<arrow::Array> labels;
    ARROW_RETURN_NOT_OK(label.Finish(&labels));
    ARROW_ASSIGN_OR_RAISE(out, out->AddColumn(out->num_columns(), arrow::field(names[4], arrow::utf8()),
                                              std::make_shared<arrow::ChunkedArray>(labels)));

    spdlog::info("SentimentAnalyzer: {} rows (VADER): {} positive, {} neutral, {} negative{}", n, positive, neutral, negative,
                 empty > 0 ? ", " + std::to_string(empty) + " without text" : std::string());
    ReportProgress(progress_callback_, "complete", "Sentiment analysis complete", 1.0f, rows, rows,
                   preflight_estimate.estimated_peak_bytes);
    return out;
}

} // namespace cyxwiz
