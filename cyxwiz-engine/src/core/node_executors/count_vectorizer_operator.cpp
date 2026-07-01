#include "count_vectorizer_operator.h"
#include "text_column_utils.h"

#include "../profiler_trace.h"

#include <cyxwiz/text_processing.h>

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace cyxwiz {

void CountVectorizerOperator::SetProgressCallback(
    PipelineOperatorProgressCallback callback) {
    progress_callback_ = std::move(callback);
}

namespace {

struct CountTermStats {
    std::string term;
    int doc_freq = 0;
    int corpus_count = 0;
};

void NormalizeCountRow(std::vector<float>& values, const std::string& norm) {
    if (norm == "none") {
        return;
    }
    double denom = 0.0;
    if (norm == "l1") {
        for (float value : values) {
            denom += std::abs(static_cast<double>(value));
        }
    } else {
        for (float value : values) {
            denom += static_cast<double>(value) * value;
        }
        denom = std::sqrt(denom);
    }
    if (denom <= 0.0) {
        return;
    }
    for (float& value : values) {
        value = static_cast<float>(static_cast<double>(value) / denom);
    }
}

std::string JoinNGram(const std::vector<std::string>& tokens,
                      size_t start,
                      size_t n) {
    std::string out;
    for (size_t i = 0; i < n; ++i) {
        if (i > 0) {
            out += " ";
        }
        out += tokens[start + i];
    }
    return out;
}

std::vector<std::string> BuildNGramFeatures(
    const std::vector<std::string>& tokens,
    int ngram_min,
    int ngram_max) {
    std::vector<std::string> features;
    if (tokens.empty()) {
        return features;
    }
    const int safe_min = std::max(1, ngram_min);
    const int safe_max = std::max(safe_min, ngram_max);
    size_t total = 0;
    for (int n = safe_min; n <= safe_max; ++n) {
        if (tokens.size() >= static_cast<size_t>(n)) {
            total += tokens.size() - static_cast<size_t>(n) + 1;
        }
    }
    features.reserve(total);
    for (int n = safe_min; n <= safe_max; ++n) {
        const size_t width = static_cast<size_t>(n);
        if (tokens.size() < width) {
            continue;
        }
        for (size_t i = 0; i + width <= tokens.size(); ++i) {
            features.push_back(width == 1
                ? tokens[i]
                : JoinNGram(tokens, i, width));
        }
    }
    return features;
}

bool ParseNGramRange(const std::map<std::string, std::string>& params,
                     int& ngram_min,
                     int& ngram_max,
                     std::string& error) {
    ngram_min = 1;
    ngram_max = 1;

    auto range = params.find("ngram_range");
    const bool has_range = range != params.end() && !range->second.empty();
    if (has_range) {
        std::string value = range->second;
        std::replace(value.begin(), value.end(), ';', ',');
        const size_t comma = value.find(',');
        if (comma == std::string::npos) {
            error = "CountVectorizer: ngram_range must be formatted as "
                    "'min,max' (got '" + range->second + "')";
            return false;
        }
        try {
            ngram_min = std::stoi(value.substr(0, comma));
            ngram_max = std::stoi(value.substr(comma + 1));
        } catch (...) {
            error = "CountVectorizer: ngram_range must contain integer values "
                    "(got '" + range->second + "')";
            return false;
        }
    }

    auto parse_positive = [&](const char* key, int& out) -> bool {
        auto it = params.find(key);
        if (it == params.end() || it->second.empty()) {
            return true;
        }
        try {
            out = std::stoi(it->second);
        } catch (...) {
            error = std::string("CountVectorizer: '") + key +
                    "' is not a valid integer: " + it->second;
            return false;
        }
        if (out < 1) {
            error = std::string("CountVectorizer: '") + key +
                    "' must be >= 1 (got " + std::to_string(out) + ")";
            return false;
        }
        return true;
    };

    if (!has_range) {
        if (!parse_positive("ngram_min", ngram_min)) return false;
        if (!parse_positive("ngram_max", ngram_max)) return false;
    }
    if (ngram_min > ngram_max) {
        error = "CountVectorizer: ngram_min must be <= ngram_max (got " +
                std::to_string(ngram_min) + "," +
                std::to_string(ngram_max) + ")";
        return false;
    }
    if (ngram_max > 3) {
        error = "CountVectorizer: ngram_max > 3 is not supported yet (got " +
                std::to_string(ngram_max) + ")";
        return false;
    }
    return true;
}

} // namespace

bool CountVectorizerOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {

    text_col_.clear();
    label_col_.clear();
    max_features_ = 2000;
    ngram_min_ = 1;
    ngram_max_ = 1;
    binary_ = false;
    norm_ = "l2";
    stop_words_ = "english";
    output_format_ = "dense";

    auto it = params.find("text_col");
    if (it == params.end() || it->second.empty()) {
        error = "CountVectorizer: 'text_col' parameter is required";
        return false;
    }
    text_col_ = it->second;

    auto lc = params.find("label_col");
    if (lc != params.end()) label_col_ = lc->second;

    auto p = params.find("max_features");
    if (p != params.end() && !p->second.empty()) {
        try { max_features_ = std::stoi(p->second); }
        catch (...) {
            error = std::string("CountVectorizer: 'max_features' is not a valid integer: ") + p->second;
            return false;
        }
    }
    if (max_features_ < 1) {
        error = "CountVectorizer: max_features must be >= 1 (got " +
                std::to_string(max_features_) + ")";
        return false;
    }

    auto nrm = params.find("norm");
    if (nrm != params.end() && !nrm->second.empty()) {
        norm_ = NormalizeTextParameterChoice(nrm->second);
        if (norm_ != "l1" && norm_ != "l2" && norm_ != "none") {
            error = "CountVectorizer: 'norm' must be 'l1' / 'l2' / 'none' (got '" +
                    norm_ + "')";
            return false;
        }
    } else {
        norm_ = "l2";
    }

    auto binary = params.find("binary");
    if (binary != params.end() && !binary->second.empty()) {
        if (binary->second == "true") {
            binary_ = true;
        } else if (binary->second == "false") {
            binary_ = false;
        } else {
            error = "CountVectorizer: 'binary' must be 'true' or 'false' (got '" +
                    binary->second + "')";
            return false;
        }
    }

    if (!ParseNGramRange(params, ngram_min_, ngram_max_, error)) return false;

    auto sw = params.find("stop_words");
    if (sw != params.end() && !sw->second.empty()) {
        stop_words_ = NormalizeTextParameterChoice(sw->second);
        if (stop_words_ != "english" && stop_words_ != "none") {
            error = "CountVectorizer: 'stop_words' must be 'english' / 'none' (got '" +
                    stop_words_ + "')";
            return false;
        }
    }

    auto output_format = params.find("output_format");
    if (output_format != params.end() && !output_format->second.empty()) {
        output_format_ = NormalizeTextParameterChoice(output_format->second);
        if (output_format_ != "dense") {
            error = "CountVectorizer: output_format='" + output_format_ +
                    "' is not supported yet; current engine supports dense output only";
            return false;
        }
    }

    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
CountVectorizerOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz CountVectorizer Materializer");
    if (!input) {
        return arrow::Status::Invalid("CountVectorizer: input table is null");
    }

    auto report_progress = [this](const std::string& stage,
                                  const std::string& message,
                                  float progress,
                                  uint64_t estimated_memory_bytes = 0,
                                  uint64_t processed_items = 0,
                                  uint64_t total_items = 0) {
        if (!progress_callback_) {
            return;
        }
        PipelineOperatorProgress event;
        event.stage = stage;
        event.message = message;
        event.progress = progress;
        event.estimated_memory_bytes = estimated_memory_bytes;
        event.processed_items = processed_items;
        event.total_items = total_items;
        progress_callback_(event);
    };

    auto text_column = input->GetColumnByName(text_col_);
    if (!text_column) {
        return arrow::Status::KeyError(
            "CountVectorizer: text column '" + text_col_ + "' not found");
    }

    std::vector<std::string> texts;
    std::string bad_type;
    if (!ReadColumnAsStrings(text_column, texts, bad_type)) {
        return arrow::Status::TypeError(
            "CountVectorizer: text column '" + text_col_ +
            "' must be string/large_string, got '" + bad_type + "'");
    }
    const uint64_t planned_memory_estimate =
        static_cast<uint64_t>(texts.size()) *
        static_cast<uint64_t>(std::max(1, max_features_)) *
        static_cast<uint64_t>(sizeof(float));
    report_progress(
        "Computing term frequencies",
        "Computing CountVectorizer term-frequency matrix...",
        0.10f,
        planned_memory_estimate,
        0,
        static_cast<uint64_t>(texts.size()));

    // Build a bounded term-frequency matrix directly so CountVectorizer
    // follows its own text-feature contract: stop-word mode, n-gram range,
    // vocabulary capping, and optional l1/l2 row normalization.
    const size_t n = texts.size();
    std::vector<std::unordered_map<std::string, int>> doc_counts;
    std::vector<size_t> doc_token_counts;
    std::unordered_map<std::string, CountTermStats> term_stats_by_name;
    doc_counts.reserve(n);
    doc_token_counts.reserve(n);

    for (size_t row = 0; row < texts.size(); ++row) {
        auto tokenized = TextProcessing::Tokenize(
            texts[row], "word", 2, /*lowercase=*/true,
            /*remove_punctuation=*/true);
        if (!tokenized.success) {
            return arrow::Status::ExecutionError(
                "CountVectorizer: tokenization failed: " +
                tokenized.error_message);
        }
        auto base_tokens = stop_words_ == "english"
            ? TextProcessing::RemoveStopwords(tokenized.tokens, "english")
            : tokenized.tokens;
        auto tokens = BuildNGramFeatures(
            base_tokens, ngram_min_, ngram_max_);
        doc_token_counts.push_back(tokens.size());

        std::unordered_map<std::string, int> counts;
        for (const auto& token : tokens) {
            counts[token]++;
        }
        for (const auto& pair : counts) {
            auto& stats = term_stats_by_name[pair.first];
            stats.term = pair.first;
            stats.doc_freq++;
            stats.corpus_count += pair.second;
        }
        doc_counts.push_back(std::move(counts));
    }

    const size_t full_vocab = term_stats_by_name.size();
    if (n == 0 || full_vocab == 0) {
        return arrow::Status::Invalid(
            "CountVectorizer: empty corpus or empty vocabulary "
            "(n=" + std::to_string(n) + ", vocab=" + std::to_string(full_vocab) + ")");
    }
    report_progress(
        "Selecting vocabulary",
        "Selecting bounded count vocabulary from " +
            std::to_string(full_vocab) + " terms...",
        0.45f,
        planned_memory_estimate,
        static_cast<uint64_t>(full_vocab),
        static_cast<uint64_t>(full_vocab));

    // Cap by document frequency. Without IDF scores we approximate
    // "most useful terms" as "most frequent terms" — count how many
    // documents each vocab term appears in, keep top-N.
    std::vector<CountTermStats> all_terms;
    all_terms.reserve(full_vocab);
    for (auto& pair : term_stats_by_name) {
        all_terms.push_back(std::move(pair.second));
    }
    std::sort(all_terms.begin(), all_terms.end(),
              [](const CountTermStats& a, const CountTermStats& b) {
                  return a.term < b.term;
              });

    const size_t kept =
        std::min(full_vocab, static_cast<size_t>(max_features_));
    if (full_vocab > kept) {
        std::partial_sort(
            all_terms.begin(),
            all_terms.begin() + kept,
            all_terms.end(),
            [](const CountTermStats& a, const CountTermStats& b) {
                if (a.doc_freq != b.doc_freq) {
                    return a.doc_freq > b.doc_freq;
                }
                if (a.corpus_count != b.corpus_count) {
                    return a.corpus_count > b.corpus_count;
                }
                return a.term < b.term;
            });
        all_terms.resize(kept);
        std::sort(all_terms.begin(), all_terms.end(),
                  [](const CountTermStats& a, const CountTermStats& b) {
                      return a.term < b.term;
                  });
    } else {
        all_terms.resize(kept);
    }

    std::unordered_map<std::string, size_t> kept_index_by_term;
    kept_index_by_term.reserve(kept);
    for (size_t i = 0; i < kept; ++i) {
        kept_index_by_term[all_terms[i].term] = i;
    }
    const uint64_t bounded_memory_estimate =
        static_cast<uint64_t>(n) *
        static_cast<uint64_t>(std::max<size_t>(1, kept)) *
        static_cast<uint64_t>(sizeof(float));
    report_progress(
        "Planning count matrix",
        "Planning " + std::to_string(n) + " rows x " +
            std::to_string(kept) + " count features",
        0.55f,
        bounded_memory_estimate,
        0,
        static_cast<uint64_t>(n) * static_cast<uint64_t>(kept));

    std::vector<int> labels;
    std::vector<std::string> class_names;
    if (!label_col_.empty()) {
        auto label_column = input->GetColumnByName(label_col_);
        if (!label_column) {
            return arrow::Status::KeyError(
                "CountVectorizer: label column '" + label_col_ + "' not found");
        }
        std::string lbad;
        if (!ReadLabelColumnAsInt(label_column, labels, class_names, lbad)) {
            return arrow::Status::TypeError(
                "CountVectorizer: label column '" + label_col_ +
                "' has unsupported type '" + lbad + "'");
        }
        if (labels.size() != n) {
            return arrow::Status::Invalid(
                "CountVectorizer: label count (" + std::to_string(labels.size()) +
                ") differs from text count (" + std::to_string(n) + ")");
        }
    }

    arrow::MemoryPool* pool = arrow::default_memory_pool();
    report_progress(
        "Building Arrow columns",
        "Allocating Arrow builders for CountVectorizer output...",
        0.62f,
        bounded_memory_estimate,
        0,
        static_cast<uint64_t>(n) * static_cast<uint64_t>(kept));
    std::vector<std::unique_ptr<arrow::FloatBuilder>> count_builders;
    count_builders.reserve(kept);
    for (size_t i = 0; i < kept; ++i) {
        count_builders.push_back(std::make_unique<arrow::FloatBuilder>(pool));
        ARROW_RETURN_NOT_OK(count_builders.back()->Reserve(static_cast<int64_t>(n)));
    }
    arrow::Int32Builder label_builder(pool);
    if (!labels.empty()) {
        ARROW_RETURN_NOT_OK(label_builder.Reserve(static_cast<int64_t>(n)));
    }

    for (size_t r = 0; r < n; ++r) {
        std::vector<float> row_values(kept, 0.0f);
        const auto& counts = doc_counts[r];
        const size_t token_count = doc_token_counts[r];
        if (token_count > 0) {
            for (const auto& pair : counts) {
                auto kept_it = kept_index_by_term.find(pair.first);
                if (kept_it == kept_index_by_term.end()) {
                    continue;
                }
                row_values[kept_it->second] = binary_
                    ? 1.0f
                    : static_cast<float>(pair.second) /
                          static_cast<float>(token_count);
            }
            NormalizeCountRow(row_values, norm_);
        }

        for (size_t c = 0; c < kept; ++c) {
            ARROW_RETURN_NOT_OK(count_builders[c]->Append(row_values[c]));
        }
        if (!labels.empty()) {
            ARROW_RETURN_NOT_OK(label_builder.Append(labels[r]));
        }
        if ((r + 1) % 5000 == 0 || r + 1 == n) {
            const float p = 0.65f + 0.25f *
                (static_cast<float>(r + 1) / static_cast<float>(n));
            report_progress(
                "Building count rows",
                "Built " + std::to_string(r + 1) +
                    "/" + std::to_string(n) + " count rows",
                p,
                bounded_memory_estimate,
                static_cast<uint64_t>(r + 1) * static_cast<uint64_t>(kept),
                static_cast<uint64_t>(n) * static_cast<uint64_t>(kept));
        }
    }

    report_progress(
        "Finishing Arrow table",
        "Finalizing CountVectorizer Arrow table...",
        0.95f,
        bounded_memory_estimate,
        static_cast<uint64_t>(n) * static_cast<uint64_t>(kept),
        static_cast<uint64_t>(n) * static_cast<uint64_t>(kept));

    std::vector<std::shared_ptr<arrow::Array>> arrays;
    std::vector<std::shared_ptr<arrow::Field>> fields;
    arrays.reserve(kept + (labels.empty() ? 0 : 1));
    fields.reserve(kept + (labels.empty() ? 0 : 1));
    for (size_t i = 0; i < kept; ++i) {
        std::shared_ptr<arrow::Array> arr;
        ARROW_RETURN_NOT_OK(count_builders[i]->Finish(&arr));
        arrays.push_back(std::move(arr));
        fields.push_back(arrow::field("count_" + std::to_string(i), arrow::float32()));
    }
    if (!labels.empty()) {
        std::shared_ptr<arrow::Array> arr;
        ARROW_RETURN_NOT_OK(label_builder.Finish(&arr));
        arrays.push_back(std::move(arr));
        fields.push_back(arrow::field("y", arrow::int32()));
    }

    auto out_schema = arrow::schema(fields);
    auto out_table = arrow::Table::Make(out_schema, arrays, static_cast<int64_t>(n));

    spdlog::info("CountVectorizer: {} docs x {} features (capped from {}), "
                 "binary={}, norm={}, stop_words={}, ngram_range={},{} classes={}",
                 n, kept, full_vocab, binary_, norm_, stop_words_,
                 ngram_min_, ngram_max_, class_names.size());
    report_progress(
        "CountVectorizer materialization complete",
        "Materialized " + std::to_string(n) + " rows x " +
            std::to_string(kept) + " count features",
        1.0f,
        bounded_memory_estimate,
        static_cast<uint64_t>(n) * static_cast<uint64_t>(kept),
        static_cast<uint64_t>(n) * static_cast<uint64_t>(kept));
    return out_table;
}

} // namespace cyxwiz
