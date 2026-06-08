#include "tfidf_vectorizer_operator.h"
#include "text_column_utils.h"

#include <cyxwiz/text_processing.h>

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace cyxwiz {

bool TFIDFVectorizerOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {

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

    auto idf = params.find("use_idf");
    use_idf_ = (idf == params.end()) ? true : (idf->second == "true");

    auto sidf = params.find("smooth_idf");
    smooth_idf_ = (sidf == params.end()) ? true : (sidf->second == "true");

    auto nrm = params.find("norm");
    if (nrm != params.end() && !nrm->second.empty()) {
        norm_ = nrm->second;
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
    if (!input) {
        return arrow::Status::Invalid("TFIDFVectorizer: input table is null");
    }

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

    // Compute TF-IDF over the entire corpus.
    auto result = TextProcessing::ComputeTFIDF(texts, use_idf_, smooth_idf_, norm_);
    if (!result.success) {
        return arrow::Status::ExecutionError(
            "TFIDFVectorizer: ComputeTFIDF failed: " + result.error_message);
    }

    const size_t n = static_cast<size_t>(result.num_documents);
    const size_t full_vocab = static_cast<size_t>(result.vocab_size);
    if (n == 0 || full_vocab == 0) {
        return arrow::Status::Invalid(
            "TFIDFVectorizer: empty corpus or empty vocabulary "
            "(n=" + std::to_string(n) + ", vocab=" + std::to_string(full_vocab) + ")");
    }

    // Cap to top-N most discriminating terms by IDF score (highest IDF =
    // rarest = most discriminative). Matches sklearn TfidfVectorizer's
    // max_features semantics. If full_vocab is already <= max_features,
    // keep everything.
    std::vector<size_t> kept_indices;
    if (full_vocab <= static_cast<size_t>(max_features_)) {
        kept_indices.resize(full_vocab);
        std::iota(kept_indices.begin(), kept_indices.end(), size_t{0});
    } else {
        kept_indices.resize(full_vocab);
        std::iota(kept_indices.begin(), kept_indices.end(), size_t{0});
        // Partial sort by IDF descending.
        std::partial_sort(
            kept_indices.begin(),
            kept_indices.begin() + max_features_,
            kept_indices.end(),
            [&](size_t a, size_t b) {
                return result.idf_scores[a] > result.idf_scores[b];
            });
        kept_indices.resize(max_features_);
        // Re-sort kept indices in original order so column ordering is
        // stable across runs (otherwise it's IDF-rank order which would
        // change across corpora).
        std::sort(kept_indices.begin(), kept_indices.end());
    }

    const size_t kept = kept_indices.size();

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
        const auto& doc_row = result.tfidf_matrix[r];
        for (size_t c = 0; c < kept; ++c) {
            const size_t src_col = kept_indices[c];
            const float v = (src_col < doc_row.size())
                ? static_cast<float>(doc_row[src_col]) : 0.0f;
            ARROW_RETURN_NOT_OK(tfidf_builders[c]->Append(v));
        }
        if (!labels.empty()) {
            ARROW_RETURN_NOT_OK(label_builder.Append(labels[r]));
        }
    }

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
                 "use_idf={}, norm={}, classes={}",
                 n, kept, full_vocab, use_idf_, norm_, class_names.size());
    return out_table;
}

} // namespace cyxwiz
