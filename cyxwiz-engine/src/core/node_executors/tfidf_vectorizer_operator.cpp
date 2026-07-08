#include "tfidf_vectorizer_operator.h"
#include "text_column_utils.h"

#include "../materialization_memory_guard.h"
#include "../profiler_trace.h"

#include <cyxwiz/text_processing.h>

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace cyxwiz {

void TFIDFVectorizerOperator::SetProgressCallback(
    PipelineOperatorProgressCallback callback) {
    progress_callback_ = std::move(callback);
}

namespace {

struct TFIDFTermStats {
    std::string term;
    int doc_freq = 0;
    int corpus_count = 0;
    double idf = 1.0;
};

void NormalizeDenseRow(std::vector<float>& values, const std::string& norm) {
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
            if (width == 1) {
                features.push_back(tokens[i]);
            } else {
                features.push_back(JoinNGram(tokens, i, width));
            }
        }
    }
    return features;
}

bool ParseNGramRange(const std::map<std::string, std::string>& params,
                     const std::string& operator_name,
                     int& ngram_min,
                     int& ngram_max,
                     std::string& error) {
    ngram_min = 1;
    ngram_max = 1;

    auto parse_positive = [&](const char* key, int& out) -> bool {
        auto it = params.find(key);
        if (it == params.end() || it->second.empty()) {
            return true;
        }
        try {
            out = std::stoi(it->second);
        } catch (...) {
            error = operator_name + ": '" + key +
                    "' is not a valid integer: " + it->second;
            return false;
        }
        if (out < 1) {
            error = operator_name + ": '" + key +
                    "' must be >= 1 (got " + std::to_string(out) + ")";
            return false;
        }
        return true;
    };

    auto range = params.find("ngram_range");
    const bool has_range = range != params.end() && !range->second.empty();
    if (has_range) {
        std::string value = range->second;
        std::replace(value.begin(), value.end(), ';', ',');
        const size_t comma = value.find(',');
        if (comma == std::string::npos) {
            error = operator_name +
                    ": ngram_range must be formatted as 'min,max' (got '" +
                    range->second + "')";
            return false;
        }
        try {
            ngram_min = std::stoi(value.substr(0, comma));
            ngram_max = std::stoi(value.substr(comma + 1));
        } catch (...) {
            error = operator_name +
                    ": ngram_range must contain integer values (got '" +
                    range->second + "')";
            return false;
        }
    }

    if (!has_range) {
        if (!parse_positive("ngram_min", ngram_min)) return false;
        if (!parse_positive("ngram_max", ngram_max)) return false;
    }
    if (ngram_min > ngram_max) {
        error = operator_name + ": ngram_min must be <= ngram_max (got " +
                std::to_string(ngram_min) + "," +
                std::to_string(ngram_max) + ")";
        return false;
    }
    if (ngram_max > 3) {
        error = operator_name + ": ngram_max > 3 is not supported yet (got " +
                std::to_string(ngram_max) + ")";
        return false;
    }
    return true;
}

bool IsStringLikeColumn(const std::shared_ptr<arrow::ChunkedArray>& column,
                        std::string& bad_type) {
    if (!column) {
        return false;
    }
    for (int c = 0; c < column->num_chunks(); ++c) {
        auto chunk = column->chunk(c);
        if (chunk->type_id() != arrow::Type::STRING &&
            chunk->type_id() != arrow::Type::LARGE_STRING) {
            bad_type = chunk->type()->ToString();
            return false;
        }
    }
    return true;
}

std::string BuildTfidfMemoryPreflightMessage(
    const MaterializationMemoryEstimate& estimate,
    const MaterializationMemoryDecision& decision) {
    std::ostringstream ss;
    ss << "TF-IDF memory preflight: risk="
       << MaterializationMemoryRiskName(decision.risk)
       << ", rows=" << estimate.rows
       << ", max_features=" << estimate.output_features
       << ", raw=" << FormatMaterializationBytes(estimate.raw_output_bytes)
       << ", estimated_peak="
       << FormatMaterializationBytes(estimate.estimated_peak_bytes)
       << ", available="
       << FormatMaterializationBytes(decision.available_bytes)
       << ", safe_budget="
       << FormatMaterializationBytes(decision.safe_budget_bytes)
       << ". " << decision.reason
       << ". Suggestion: " << decision.suggestion;
    return ss.str();
}

} // namespace

bool TFIDFVectorizerOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {

    text_col_.clear();
    label_col_.clear();
    max_features_ = 2000;
    min_df_ = 1;
    ngram_min_ = 1;
    ngram_max_ = 1;
    use_idf_ = true;
    smooth_idf_ = true;
    norm_ = "l2";
    stop_words_ = "english";
    output_format_ = "dense";

    auto it = params.find("text_col");
    if (it == params.end() || it->second.empty()) {
        error = "TFIDFVectorizer: 'text_col' parameter is required";
        return false;
    }
    text_col_ = it->second;

    auto lc = params.find("label_col");
    if (lc != params.end()) label_col_ = lc->second;

    auto read_int = [&](const char* key, int default_value, int& out) -> bool {
        auto p = params.find(key);
        if (p == params.end() || p->second.empty()) {
            out = default_value;
            return true;
        }
        try { out = std::stoi(p->second); }
        catch (...) {
            error = std::string("TFIDFVectorizer: '") + key +
                    "' is not a valid integer: " + p->second;
            return false;
        }
        return true;
    };

    if (!read_int("max_features", 2000, max_features_)) return false;
    if (max_features_ < 1) {
        error = "TFIDFVectorizer: max_features must be >= 1 (got " +
                std::to_string(max_features_) + ")";
        return false;
    }
    if (!read_int("min_df", 1, min_df_)) return false;
    if (min_df_ < 1) {
        error = "TFIDFVectorizer: min_df must be >= 1 (got " +
                std::to_string(min_df_) + ")";
        return false;
    }

    auto read_bool = [&](const char* key, bool default_value, bool& out) -> bool {
        auto p = params.find(key);
        if (p == params.end() || p->second.empty()) {
            out = default_value;
            return true;
        }
        if (p->second == "true") {
            out = true;
            return true;
        }
        if (p->second == "false") {
            out = false;
            return true;
        }
        error = std::string("TFIDFVectorizer: '") + key +
                "' must be 'true' or 'false' (got '" + p->second + "')";
        return false;
    };

    if (!read_bool("use_idf", true, use_idf_)) return false;
    if (!read_bool("smooth_idf", true, smooth_idf_)) return false;

    auto nrm = params.find("norm");
    if (nrm != params.end() && !nrm->second.empty()) {
        norm_ = NormalizeTextParameterChoice(nrm->second);
        if (norm_ != "l1" && norm_ != "l2" && norm_ != "none") {
            error = "TFIDFVectorizer: 'norm' must be 'l1' / 'l2' / 'none' (got '" +
                    norm_ + "')";
            return false;
        }
    } else {
        norm_ = "l2";
    }

    auto sw = params.find("stop_words");
    if (sw != params.end() && !sw->second.empty()) {
        stop_words_ = NormalizeTextParameterChoice(sw->second);
        if (stop_words_ != "english" && stop_words_ != "none") {
            error = "TFIDFVectorizer: 'stop_words' must be 'english' / 'none' (got '" +
                    stop_words_ + "')";
            return false;
        }
    }

    auto output_format = params.find("output_format");
    if (output_format != params.end() && !output_format->second.empty()) {
        output_format_ = NormalizeTextParameterChoice(output_format->second);
        if (output_format_ != "dense") {
            error = "TFIDFVectorizer: output_format='" + output_format_ +
                    "' is not supported yet; current engine supports dense output only";
            return false;
        }
    }

    if (!ParseNGramRange(params, "TFIDFVectorizer",
                         ngram_min_, ngram_max_, error)) return false;

    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
TFIDFVectorizerOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz TF-IDF Materializer");
    if (!input) {
        return arrow::Status::Invalid("TFIDFVectorizer: input table is null");
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
            "TFIDFVectorizer: text column '" + text_col_ + "' not found");
    }

    std::string bad_type;
    if (!IsStringLikeColumn(text_column, bad_type)) {
        return arrow::Status::TypeError(
            "TFIDFVectorizer: text column '" + text_col_ +
            "' must be string/large_string, got '" + bad_type + "'");
    }

    const uint64_t planned_rows =
        static_cast<uint64_t>(std::max<int64_t>(0, text_column->length()));
    if (planned_rows == 0) {
        return arrow::Status::Invalid(
            "TFIDFVectorizer: empty corpus");
    }
    const uint64_t planned_features =
        static_cast<uint64_t>(std::max(1, max_features_));
    const auto preflight_estimate = EstimateDenseMaterializationMemory(
        planned_rows, planned_features, static_cast<uint64_t>(sizeof(float)));
    const auto preflight_decision = EvaluateMaterializationMemory(
        preflight_estimate, DetectMaterializationMemorySnapshot());
    const std::string preflight_message =
        BuildTfidfMemoryPreflightMessage(
            preflight_estimate, preflight_decision);
    uint64_t planned_cells = 0;
    if (!CheckedMulU64(planned_rows, planned_features, planned_cells)) {
        planned_cells = std::numeric_limits<uint64_t>::max();
    }
    report_progress(
        "TF-IDF memory preflight",
        preflight_message,
        0.03f,
        preflight_estimate.estimated_peak_bytes,
        0,
        planned_cells);
    if (preflight_decision.blocked) {
        return arrow::Status::CapacityError(
            "Materialization blocked: " + preflight_message);
    }

    std::vector<std::string> texts;
    if (!ReadColumnAsStrings(text_column, texts, bad_type)) {
        return arrow::Status::TypeError(
            "TFIDFVectorizer: text column '" + text_col_ +
            "' must be string/large_string, got '" + bad_type + "'");
    }

    const size_t n = texts.size();
    if (n == 0) {
        return arrow::Status::Invalid(
            "TFIDFVectorizer: empty corpus");
    }
    const uint64_t initial_memory_estimate =
        preflight_estimate.estimated_peak_bytes;
    report_progress(
        "Tokenizing text",
        "Tokenizing text and building term counts...",
        0.10f,
        initial_memory_estimate,
        0,
        static_cast<uint64_t>(n));

    std::vector<std::unordered_map<std::string, int>> doc_counts;
    std::vector<size_t> doc_token_counts;
    doc_counts.reserve(n);
    doc_token_counts.reserve(n);

    std::unordered_map<std::string, TFIDFTermStats> term_stats_by_name;
    try {
        for (size_t row = 0; row < texts.size(); ++row) {
            const auto& text = texts[row];
            auto tokenized = TextProcessing::Tokenize(
                text, "word", 2, /*lowercase=*/true,
                /*remove_punctuation=*/true);
            if (!tokenized.success) {
                return arrow::Status::ExecutionError(
                    "TFIDFVectorizer: tokenization failed: " +
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
            if ((row + 1) % 5000 == 0 || row + 1 == texts.size()) {
                const float p = 0.10f + 0.25f *
                    (static_cast<float>(row + 1) /
                     static_cast<float>(texts.size()));
                report_progress(
                    "Tokenizing text",
                    "Tokenized " + std::to_string(row + 1) +
                        "/" + std::to_string(texts.size()) + " rows",
                    p,
                    initial_memory_estimate,
                    static_cast<uint64_t>(row + 1),
                    static_cast<uint64_t>(texts.size()));
            }
        }
    } catch (const std::bad_alloc&) {
        return arrow::Status::CapacityError(
            "TFIDFVectorizer: insufficient memory while building bounded "
            "term-frequency maps");
    } catch (const std::exception& e) {
        return arrow::Status::ExecutionError(
            std::string("TFIDFVectorizer: token counting failed: ") + e.what());
    }

    const size_t full_vocab = term_stats_by_name.size();
    if (full_vocab == 0) {
        return arrow::Status::Invalid(
            "TFIDFVectorizer: empty vocabulary after tokenization and stopword removal");
    }

    std::vector<TFIDFTermStats> all_terms;
    report_progress(
        "Selecting vocabulary",
        "Selecting bounded vocabulary from " + std::to_string(full_vocab) +
            " terms with min_df=" + std::to_string(min_df_) + "...",
        0.45f,
        initial_memory_estimate,
        static_cast<uint64_t>(full_vocab),
        static_cast<uint64_t>(full_vocab));
    all_terms.reserve(full_vocab);
    for (auto& pair : term_stats_by_name) {
        auto stats = std::move(pair.second);
        if (stats.doc_freq < min_df_) {
            continue;
        }
        if (use_idf_) {
            const double df = smooth_idf_
                ? static_cast<double>(stats.doc_freq + 1)
                : static_cast<double>(stats.doc_freq);
            const double total = smooth_idf_
                ? static_cast<double>(n + 1)
                : static_cast<double>(n);
            stats.idf = std::log(total / df) + 1.0;
        } else {
            stats.idf = 1.0;
        }
        all_terms.push_back(std::move(stats));
    }
    if (all_terms.empty()) {
        return arrow::Status::Invalid(
            "TFIDFVectorizer: empty vocabulary after applying min_df=" +
            std::to_string(min_df_));
    }

    std::sort(all_terms.begin(), all_terms.end(),
              [](const TFIDFTermStats& a, const TFIDFTermStats& b) {
                  return a.term < b.term;
              });

    const size_t filtered_vocab = all_terms.size();
    const size_t kept =
        std::min(filtered_vocab, static_cast<size_t>(max_features_));
    const auto bounded_memory_plan = EstimateDenseMaterializationMemory(
        static_cast<uint64_t>(n),
        static_cast<uint64_t>(std::max<size_t>(1, kept)),
        static_cast<uint64_t>(sizeof(float)));
    const uint64_t bounded_memory_estimate =
        bounded_memory_plan.estimated_peak_bytes;
    if (filtered_vocab > kept) {
        std::partial_sort(
            all_terms.begin(),
            all_terms.begin() + kept,
            all_terms.end(),
            [](const TFIDFTermStats& a, const TFIDFTermStats& b) {
                if (a.corpus_count != b.corpus_count) {
                    return a.corpus_count > b.corpus_count;
                }
                if (a.doc_freq != b.doc_freq) {
                    return a.doc_freq > b.doc_freq;
                }
                if (a.idf != b.idf) return a.idf > b.idf;
                return a.term < b.term;
            });
        all_terms.resize(kept);
        std::sort(all_terms.begin(), all_terms.end(),
                  [](const TFIDFTermStats& a, const TFIDFTermStats& b) {
                      return a.term < b.term;
                  });
    } else {
        all_terms.resize(kept);
    }

    std::unordered_map<std::string, size_t> kept_index_by_term;
    report_progress(
        "Planning TF-IDF matrix",
        "Planning " + std::to_string(n) + " rows x " +
            std::to_string(kept) + " TF-IDF features",
        0.55f,
        bounded_memory_estimate,
        0,
        static_cast<uint64_t>(n) * static_cast<uint64_t>(kept));
    kept_index_by_term.reserve(kept);
    std::vector<double> kept_idf;
    kept_idf.reserve(kept);
    for (size_t i = 0; i < kept; ++i) {
        kept_index_by_term[all_terms[i].term] = i;
        kept_idf.push_back(all_terms[i].idf);
    }

    // Read label column if specified.
    std::vector<int> labels;
    std::vector<std::string> class_names;
    if (!label_col_.empty()) {
        auto label_column = input->GetColumnByName(label_col_);
        if (!label_column) {
            return arrow::Status::KeyError(
                "TFIDFVectorizer: label column '" + label_col_ + "' not found");
        }
        std::string lbad;
        if (!ReadLabelColumnAsInt(label_column, labels, class_names, lbad)) {
            return arrow::Status::TypeError(
                "TFIDFVectorizer: label column '" + label_col_ +
                "' has unsupported type '" + lbad + "'");
        }
        if (labels.size() != n) {
            return arrow::Status::Invalid(
                "TFIDFVectorizer: label count (" + std::to_string(labels.size()) +
                ") differs from text count (" + std::to_string(n) + ")");
        }
    }

    // Build wide output: tfidf_0..tfidf_{kept-1}, y.
    report_progress(
        "Building Arrow columns",
        "Allocating Arrow builders for TF-IDF output...",
        0.62f,
        bounded_memory_estimate,
        0,
        static_cast<uint64_t>(n) * static_cast<uint64_t>(kept));
    arrow::MemoryPool* pool = arrow::default_memory_pool();
    std::vector<std::unique_ptr<arrow::FloatBuilder>> tfidf_builders;
    tfidf_builders.reserve(kept);
    for (size_t i = 0; i < kept; ++i) {
        tfidf_builders.push_back(std::make_unique<arrow::FloatBuilder>(pool));
        ARROW_RETURN_NOT_OK(tfidf_builders.back()->Reserve(static_cast<int64_t>(n)));
    }
    arrow::Int32Builder label_builder(pool);
    if (!labels.empty()) {
        ARROW_RETURN_NOT_OK(label_builder.Reserve(static_cast<int64_t>(n)));
    }

    for (size_t r = 0; r < n; ++r) {
        std::vector<float> dense_row(kept, 0.0f);
        const auto& counts = doc_counts[r];
        const size_t token_count = doc_token_counts[r];
        if (token_count > 0) {
            for (const auto& pair : counts) {
                auto kept_it = kept_index_by_term.find(pair.first);
                if (kept_it == kept_index_by_term.end()) {
                    continue;
                }
                const size_t col = kept_it->second;
                const double tf =
                    static_cast<double>(pair.second) /
                    static_cast<double>(token_count);
                dense_row[col] =
                    static_cast<float>(tf * kept_idf[col]);
            }
            NormalizeDenseRow(dense_row, norm_);
        }

        for (size_t c = 0; c < kept; ++c) {
            ARROW_RETURN_NOT_OK(tfidf_builders[c]->Append(dense_row[c]));
        }
        if (!labels.empty()) {
            ARROW_RETURN_NOT_OK(label_builder.Append(labels[r]));
        }
        if ((r + 1) % 5000 == 0 || r + 1 == n) {
            const float p = 0.65f + 0.25f *
                (static_cast<float>(r + 1) / static_cast<float>(n));
            report_progress(
                "Building TF-IDF rows",
                "Built " + std::to_string(r + 1) +
                    "/" + std::to_string(n) + " TF-IDF rows",
                p,
                bounded_memory_estimate,
                static_cast<uint64_t>(r + 1) * static_cast<uint64_t>(kept),
                static_cast<uint64_t>(n) * static_cast<uint64_t>(kept));
        }
    }

    report_progress(
        "Finishing Arrow table",
        "Finalizing TF-IDF Arrow table...",
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
        ARROW_RETURN_NOT_OK(tfidf_builders[i]->Finish(&arr));
        arrays.push_back(std::move(arr));
        fields.push_back(arrow::field("tfidf_" + std::to_string(i), arrow::float32()));
    }
    if (!labels.empty()) {
        std::shared_ptr<arrow::Array> arr;
        ARROW_RETURN_NOT_OK(label_builder.Finish(&arr));
        arrays.push_back(std::move(arr));
        fields.push_back(arrow::field("y", arrow::int32()));
    }

    auto out_schema = arrow::schema(fields);
    auto out_table = arrow::Table::Make(out_schema, arrays, static_cast<int64_t>(n));

    spdlog::info("TFIDFVectorizer: {} docs x {} features (capped from {}, "
                 "filtered from {} with min_df={}), use_idf={}, "
                 "smooth_idf={}, norm={}, stop_words={}, ngram_range={},{} classes={}, bounded=true",
                 n, kept, filtered_vocab, full_vocab, min_df_,
                 use_idf_, smooth_idf_, norm_,
                 stop_words_,
                 ngram_min_, ngram_max_,
                 class_names.size());
    report_progress(
        "TF-IDF materialization complete",
        "Materialized " + std::to_string(n) + " rows x " +
            std::to_string(kept) + " TF-IDF features",
        1.0f,
        bounded_memory_estimate,
        static_cast<uint64_t>(n) * static_cast<uint64_t>(kept),
        static_cast<uint64_t>(n) * static_cast<uint64_t>(kept));
    return out_table;
}

} // namespace cyxwiz
