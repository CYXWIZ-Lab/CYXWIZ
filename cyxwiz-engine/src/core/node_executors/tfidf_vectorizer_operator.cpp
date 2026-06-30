#include "tfidf_vectorizer_operator.h"
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
#include <new>
#include <numeric>
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

} // namespace

bool TFIDFVectorizerOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {

    text_col_.clear();
    label_col_.clear();
    max_features_ = 2000;
    use_idf_ = true;
    smooth_idf_ = true;
    norm_ = "l2";

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

    auto ngram_range = params.find("ngram_range");
    if (ngram_range != params.end() && !ngram_range->second.empty() &&
        ngram_range->second != "1,1") {
        error = "TFIDFVectorizer: ngram_range values other than '1,1' "
                "are not supported by this operator";
        return false;
    }

    auto ngram_min = params.find("ngram_min");
    if (ngram_min != params.end() && !ngram_min->second.empty() &&
        ngram_min->second != "1") {
        error = "TFIDFVectorizer: ngram_min values other than 1 "
                "are not supported by this operator";
        return false;
    }

    auto ngram_max = params.find("ngram_max");
    if (ngram_max != params.end() && !ngram_max->second.empty() &&
        ngram_max->second != "1") {
        error = "TFIDFVectorizer: ngram_max values other than 1 "
                "are not supported by this operator";
        return false;
    }

    auto min_df = params.find("min_df");
    if (min_df != params.end() && !min_df->second.empty() &&
        min_df->second != "1") {
        error = "TFIDFVectorizer: min_df values other than 1 "
                "are not supported by this operator";
        return false;
    }

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

    std::vector<std::string> texts;
    std::string bad_type;
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
        static_cast<uint64_t>(n) *
        static_cast<uint64_t>(std::max(1, max_features_)) *
        static_cast<uint64_t>(sizeof(float));
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

            auto tokens =
                TextProcessing::RemoveStopwords(tokenized.tokens, "english");
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
            " terms...",
        0.45f,
        initial_memory_estimate,
        static_cast<uint64_t>(full_vocab),
        static_cast<uint64_t>(full_vocab));
    all_terms.reserve(full_vocab);
    for (auto& pair : term_stats_by_name) {
        auto stats = std::move(pair.second);
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

    std::sort(all_terms.begin(), all_terms.end(),
              [](const TFIDFTermStats& a, const TFIDFTermStats& b) {
                  return a.term < b.term;
              });

    const size_t kept =
        std::min(full_vocab, static_cast<size_t>(max_features_));
    const uint64_t bounded_memory_estimate =
        static_cast<uint64_t>(n) *
        static_cast<uint64_t>(std::max<size_t>(1, kept)) *
        static_cast<uint64_t>(sizeof(float));
    if (full_vocab > kept) {
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

    spdlog::info("TFIDFVectorizer: {} docs x {} features (capped from {}), "
                 "use_idf={}, smooth_idf={}, norm={}, classes={}, bounded=true",
                 n, kept, full_vocab, use_idf_, smooth_idf_, norm_,
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
