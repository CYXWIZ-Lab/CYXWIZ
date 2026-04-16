#include "count_vectorizer_operator.h"
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

bool CountVectorizerOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {

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
        norm_ = nrm->second;
        if (norm_ != "l1" && norm_ != "l2" && norm_ != "none") {
            error = "CountVectorizer: 'norm' must be 'l1' / 'l2' / 'none' (got '" +
                    norm_ + "')";
            return false;
        }
    } else {
        norm_ = "l2";
    }

    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
CountVectorizerOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    if (!input) {
        return arrow::Status::Invalid("CountVectorizer: input table is null");
    }

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

    // ComputeTFIDF with use_idf=false reduces to TF (term frequency)
    // matrix — each row is the per-document term-frequency vector,
    // optionally l1/l2 normalized. That's BoW semantics; we reuse the
    // backend rather than reimplementing.
    auto result = TextProcessing::ComputeTFIDF(
        texts, /*use_idf=*/false, /*smooth_idf=*/false, norm_);
    if (!result.success) {
        return arrow::Status::ExecutionError(
            "CountVectorizer: ComputeTFIDF failed: " + result.error_message);
    }

    const size_t n = static_cast<size_t>(result.num_documents);
    const size_t full_vocab = static_cast<size_t>(result.vocab_size);
    if (n == 0 || full_vocab == 0) {
        return arrow::Status::Invalid(
            "CountVectorizer: empty corpus or empty vocabulary "
            "(n=" + std::to_string(n) + ", vocab=" + std::to_string(full_vocab) + ")");
    }

    // Cap by document frequency. Without IDF scores we approximate
    // "most useful terms" as "most frequent terms" — count how many
    // documents each vocab term appears in, keep top-N.
    std::vector<size_t> kept_indices;
    if (full_vocab <= static_cast<size_t>(max_features_)) {
        kept_indices.resize(full_vocab);
        std::iota(kept_indices.begin(), kept_indices.end(), size_t{0});
    } else {
        std::vector<int> doc_freq(full_vocab, 0);
        for (size_t r = 0; r < n; ++r) {
            const auto& row = result.tfidf_matrix[r];
            for (size_t c = 0; c < full_vocab && c < row.size(); ++c) {
                if (row[c] > 0.0) doc_freq[c]++;
            }
        }
        kept_indices.resize(full_vocab);
        std::iota(kept_indices.begin(), kept_indices.end(), size_t{0});
        std::partial_sort(
            kept_indices.begin(),
            kept_indices.begin() + max_features_,
            kept_indices.end(),
            [&](size_t a, size_t b) { return doc_freq[a] > doc_freq[b]; });
        kept_indices.resize(max_features_);
        std::sort(kept_indices.begin(), kept_indices.end());
    }

    const size_t kept = kept_indices.size();

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
        const auto& doc_row = result.tfidf_matrix[r];
        for (size_t c = 0; c < kept; ++c) {
            const size_t src_col = kept_indices[c];
            const float v = (src_col < doc_row.size())
                ? static_cast<float>(doc_row[src_col]) : 0.0f;
            ARROW_RETURN_NOT_OK(count_builders[c]->Append(v));
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
                 "norm={}, classes={}",
                 n, kept, full_vocab, norm_, class_names.size());
    return out_table;
}

} // namespace cyxwiz
