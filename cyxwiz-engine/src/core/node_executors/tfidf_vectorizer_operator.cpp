#include "tfidf_vectorizer_operator.h"
#include "text_column_utils.h"
#include "text_vectorizer_contract.h"

#include "../materialization_memory_guard.h"
#include "../profiler_trace.h"

#include <cyxwiz/text_processing.h>

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <limits>
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
    state_options_ = {};

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
        if (output_format_ != "dense" && output_format_ != "sparse") {
            error = "TFIDFVectorizer: output_format='" + output_format_ +
                    "' must be 'dense' or 'sparse'";
            return false;
        }
    }

    if (!text_vectorizer_contract::ParseNGramRange(
            params, "TFIDFVectorizer", ngram_min_, ngram_max_, error)) {
        return false;
    }

    return ParseFittedPreprocessingOptions(
        params, GetName(), state_options_, error);
}

std::map<std::string, std::string>
TFIDFVectorizerOperator::BuildFittedConfiguration() const {
    return {
        {"text_col", text_col_},
        {"max_features", std::to_string(max_features_)},
        {"min_df", std::to_string(min_df_)},
        {"use_idf", use_idf_ ? "true" : "false"},
        {"smooth_idf", smooth_idf_ ? "true" : "false"},
        {"norm", norm_},
        {"ngram_range", std::to_string(ngram_min_) + "," +
                            std::to_string(ngram_max_)},
        {"stop_words", stop_words_},
        {"value_semantics", "sklearn_raw_count_v1"},
    };
}

bool TFIDFVectorizerOperator::CollectCacheDependencies(
    std::vector<PipelineOperatorCacheDependency>& dependencies,
    std::string& error) const {
    if (!state_options_.IsTransformOnly()) {
        return true;
    }
    FittedTextVectorizerState state;
    if (!LoadFittedTextVectorizerState(
            state_options_.state_path, GetName(), BuildFittedConfiguration(),
            static_cast<size_t>(max_features_), state, error)) {
        return false;
    }
    dependencies.push_back({"fitted_state", state_options_.state_path});
    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
TFIDFVectorizerOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    if (output_format_ != "dense") {
        return arrow::Status::Invalid(
            "TFIDFVectorizer: sparse output requires typed materializer "
            "publication; the Arrow table path remains fail-closed");
    }
    ARROW_ASSIGN_OR_RAISE(auto output, ApplyConfigured(input, {}));
    return output.dense_table;
}

arrow::Result<std::shared_ptr<SparseFeatureDataset>>
TFIDFVectorizerOperator::ApplySparse(
    const std::shared_ptr<arrow::Table>& input,
    const std::string& dataset_name) {
    if (output_format_ != "sparse") {
        return arrow::Status::Invalid(
            "TFIDFVectorizer: ApplySparse requires output_format=sparse");
    }
    if (dataset_name.empty()) {
        return arrow::Status::Invalid(
            "TFIDFVectorizer: sparse dataset name must not be empty");
    }
    ARROW_ASSIGN_OR_RAISE(auto output,
                          ApplyConfigured(input, dataset_name));
    return output.sparse_dataset;
}

arrow::Result<TextVectorizerMaterialization>
TFIDFVectorizerOperator::ApplyConfigured(
    const std::shared_ptr<arrow::Table>& input,
    const std::string& sparse_dataset_name) {
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
        event.status = "running";
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

    const auto fitted_configuration = BuildFittedConfiguration();
    FittedTextVectorizerState fitted_state;
    std::vector<TFIDFTermStats> fitted_terms;
    if (state_options_.IsTransformOnly()) {
        std::string state_error;
        if (!LoadFittedTextVectorizerState(
                state_options_.state_path, GetName(), fitted_configuration,
                static_cast<size_t>(max_features_), fitted_state,
                state_error)) {
            return arrow::Status::Invalid(state_error);
        }
        fitted_terms.reserve(fitted_state.features.size());
        for (const auto& feature : fitted_state.features) {
            TFIDFTermStats term;
            term.term = feature.term;
            term.idf = feature.weight;
            fitted_terms.push_back(std::move(term));
        }
        spdlog::info(
            "TFIDFVectorizer: Transform Only loaded '{}' (fit_rows={}, "
            "features={}, schema={})",
            state_options_.state_path, fitted_state.fit_rows,
            fitted_terms.size(), fitted_state.input_schema_fingerprint);
    }

    const uint64_t planned_rows =
        static_cast<uint64_t>(std::max<int64_t>(0, text_column->length()));
    if (planned_rows == 0) {
        return arrow::Status::Invalid(
            "TFIDFVectorizer: empty corpus");
    }
    const uint64_t planned_features = state_options_.IsTransformOnly()
        ? static_cast<uint64_t>(std::max<size_t>(1, fitted_terms.size()))
        : static_cast<uint64_t>(std::max(1, max_features_));
    const bool sparse_output = output_format_ == "sparse";
    const auto preflight_estimate = sparse_output
        ? EstimateSparseTextFeatureMemory(
              planned_rows, 0, !label_col_.empty())
        : EstimateDenseMaterializationMemory(
              planned_rows, planned_features,
              static_cast<uint64_t>(sizeof(float)));
    const auto preflight_decision = EvaluateMaterializationMemory(
        preflight_estimate, GetMaterializationMemoryContext());
    const std::string preflight_message =
        BuildMaterializationMemoryPreflightMessage(
            "TF-IDF", "max_features", preflight_estimate, preflight_decision,
            sparse_output
                ? "This is a CSR lower-bound estimate; exact nnz memory is "
                  "checked after vocabulary selection."
                : "Reduce TF-IDF max_features, sample fewer rows first, or "
                  "select sparse output once its training path is enabled.");
    uint64_t planned_cells = 0;
    if (!CheckedMulU64(planned_rows, planned_features, planned_cells)) {
        planned_cells = std::numeric_limits<uint64_t>::max();
    }
    if (progress_callback_) {
        PipelineOperatorProgress event;
        event.stage = sparse_output
            ? "TF-IDF sparse memory preflight"
            : "TF-IDF memory preflight";
        event.message = preflight_message;
        event.status = MaterializationMemoryRiskToProgressStatus(
            preflight_decision.risk);
        event.progress = 0.03f;
        event.estimated_memory_bytes = preflight_estimate.estimated_peak_bytes;
        event.memory_risk_level = MaterializationMemoryRiskName(
            preflight_decision.risk);
        event.processed_items = 0;
        event.total_items = sparse_output ? planned_rows : planned_cells;
        progress_callback_(event);
    }
    if (preflight_decision.blocked) {
        return arrow::Status::CapacityError(
            "Materialization blocked: " + preflight_message);
    }
    ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));

    std::vector<std::string> texts;
    if (!ReadColumnAsStrings(
            text_column, texts, bad_type, GetCancellationQuery())) {
        ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
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
    doc_counts.reserve(n);

    std::unordered_map<std::string, TFIDFTermStats> term_stats_by_name;
    try {
        for (size_t row = 0; row < texts.size(); ++row) {
            if ((row & 1023) == 0) {
                ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
            }
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
            auto tokens = text_vectorizer_contract::BuildNGramFeatures(
                base_tokens, ngram_min_, ngram_max_);
            std::unordered_map<std::string, int> counts;
            for (const auto& token : tokens) {
                counts[token]++;
            }

            if (!state_options_.IsTransformOnly()) {
                for (const auto& pair : counts) {
                    auto& stats = term_stats_by_name[pair.first];
                    stats.term = pair.first;
                    stats.doc_freq++;
                    stats.corpus_count += pair.second;
                }
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

    size_t full_vocab = fitted_terms.size();
    size_t filtered_vocab = fitted_terms.size();
    std::vector<TFIDFTermStats> all_terms = std::move(fitted_terms);
    if (!state_options_.IsTransformOnly()) {
        full_vocab = term_stats_by_name.size();
        if (full_vocab == 0) {
            return arrow::Status::Invalid(
                "TFIDFVectorizer: empty vocabulary after tokenization and stopword removal");
        }
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
        filtered_vocab = all_terms.size();
    } else {
        report_progress(
            "Loading fitted vocabulary",
            "Using " + std::to_string(all_terms.size()) +
                " fitted TF-IDF features",
            0.45f,
            initial_memory_estimate,
            static_cast<uint64_t>(all_terms.size()),
            static_cast<uint64_t>(all_terms.size()));
    }

    const size_t kept = state_options_.IsTransformOnly()
        ? all_terms.size()
        : std::min(filtered_vocab, static_cast<size_t>(max_features_));
    if (!state_options_.IsTransformOnly() && filtered_vocab > kept) {
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
    } else if (!state_options_.IsTransformOnly()) {
        all_terms.resize(kept);
    }

    std::unordered_map<std::string, size_t> kept_index_by_term;
    kept_index_by_term.reserve(kept);
    std::vector<double> kept_idf;
    kept_idf.reserve(kept);
    for (size_t i = 0; i < kept; ++i) {
        kept_index_by_term[all_terms[i].term] = i;
        kept_idf.push_back(all_terms[i].idf);
    }

    uint64_t expected_nnz = 0;
    for (const auto& counts : doc_counts) {
        for (const auto& [term, _] : counts) {
            if (kept_index_by_term.find(term) != kept_index_by_term.end()) {
                if (expected_nnz == (std::numeric_limits<uint64_t>::max)()) {
                    return arrow::Status::CapacityError(
                        "TFIDFVectorizer: sparse nnz count overflow");
                }
                ++expected_nnz;
            }
        }
    }
    constexpr uint64_t kSparseIndexMax = static_cast<uint64_t>(
        (std::numeric_limits<int32_t>::max)());
    if (sparse_output &&
        (static_cast<uint64_t>(n) > kSparseIndexMax ||
         static_cast<uint64_t>(kept) > kSparseIndexMax ||
         expected_nnz > kSparseIndexMax)) {
        return arrow::Status::CapacityError(
            "TFIDFVectorizer: output exceeds the int32 CSR contract");
    }
    const auto bounded_memory_plan = sparse_output
        ? EstimateSparseTextFeatureMemory(
              static_cast<uint64_t>(n), expected_nnz,
              !label_col_.empty())
        : EstimateDenseMaterializationMemory(
              static_cast<uint64_t>(n),
              static_cast<uint64_t>(std::max<size_t>(1, kept)),
              static_cast<uint64_t>(sizeof(float)));
    const auto bounded_memory_decision = EvaluateMaterializationMemory(
        bounded_memory_plan, GetMaterializationMemoryContext());
    const uint64_t bounded_memory_estimate =
        bounded_memory_plan.estimated_peak_bytes;
    if (sparse_output && bounded_memory_decision.blocked) {
        return arrow::Status::CapacityError(
            "TFIDFVectorizer: sparse CSR materialization blocked: " +
            bounded_memory_decision.reason);
    }
    report_progress(
        "Planning TF-IDF matrix",
        "Planning " + std::to_string(n) + " rows x " +
            std::to_string(kept) + " TF-IDF features",
        0.55f,
        bounded_memory_estimate,
        0,
        sparse_output
            ? expected_nnz
            : static_cast<uint64_t>(n) * static_cast<uint64_t>(kept));

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
        if (!ReadLabelColumnAsInt(
                label_column, labels, class_names, lbad,
                GetCancellationQuery())) {
            ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
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

    // Assemble canonical nonzero rows once. Dense Arrow and sparse CSR output
    // are both derived from this matrix, so representation choice cannot
    // change vocabulary order, normalization, or numerical values.
    report_progress(
        sparse_output ? "Building sparse TF-IDF rows"
                      : "Building TF-IDF feature rows",
        sparse_output
            ? "Building canonical CSR values for TF-IDF output..."
            : "Building canonical TF-IDF values for dense output...",
        0.62f,
        bounded_memory_estimate,
        0,
        sparse_output
            ? expected_nnz
            : static_cast<uint64_t>(n) * static_cast<uint64_t>(kept));

    TextFeatureMatrix feature_matrix;
    feature_matrix.num_rows = static_cast<int64_t>(n);
    feature_matrix.num_features = static_cast<int64_t>(kept);
    feature_matrix.row_offsets.reserve(n + 1);
    feature_matrix.column_indices.reserve(
        static_cast<size_t>(expected_nnz));
    feature_matrix.values.reserve(static_cast<size_t>(expected_nnz));
    feature_matrix.feature_names.reserve(kept);
    for (const auto& term : all_terms) {
        feature_matrix.feature_names.push_back(term.term);
    }
    if (!labels.empty()) {
        feature_matrix.labels.reserve(labels.size());
        for (int label : labels) {
            feature_matrix.labels.push_back(static_cast<int32_t>(label));
        }
        feature_matrix.label_name = "y";
    }

    for (size_t r = 0; r < n; ++r) {
        if ((r & 1023) == 0) {
            ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
        }
        std::vector<TextFeatureEntry> row_entries;
        const auto& counts = doc_counts[r];
        row_entries.reserve(std::min(counts.size(), kept));
        for (const auto& pair : counts) {
            auto kept_it = kept_index_by_term.find(pair.first);
            if (kept_it == kept_index_by_term.end()) {
                continue;
            }
            const size_t column = kept_it->second;
            row_entries.push_back({
                static_cast<int32_t>(column),
                static_cast<float>(
                    static_cast<double>(pair.second) * kept_idf[column])});
        }
        ARROW_RETURN_NOT_OK(AppendNormalizedTextFeatureRow(
            feature_matrix, std::move(row_entries), norm_));
        if ((r + 1) % 5000 == 0 || r + 1 == n) {
            const float p = 0.65f + 0.25f *
                (static_cast<float>(r + 1) / static_cast<float>(n));
            report_progress(
                state_options_.IsTransformOnly()
                    ? "Transforming with fitted TF-IDF state"
                    : "Building TF-IDF rows",
                (state_options_.IsTransformOnly()
                     ? "Transformed "
                     : "Built ") +
                    std::to_string(r + 1) + "/" + std::to_string(n) +
                    " TF-IDF rows",
                p,
                bounded_memory_estimate,
                sparse_output
                    ? feature_matrix.values.size()
                    : static_cast<uint64_t>(r + 1) *
                          static_cast<uint64_t>(kept),
                sparse_output
                    ? expected_nnz
                    : static_cast<uint64_t>(n) *
                          static_cast<uint64_t>(kept));
        }
    }

    report_progress(
        sparse_output ? "Finishing sparse dataset"
                      : "Finishing Arrow table",
        sparse_output ? "Finalizing TF-IDF CSR dataset..."
                      : "Finalizing TF-IDF Arrow table...",
        0.95f,
        bounded_memory_estimate,
        sparse_output ? expected_nnz
                      : static_cast<uint64_t>(n) *
                            static_cast<uint64_t>(kept),
        sparse_output ? expected_nnz
                      : static_cast<uint64_t>(n) *
                            static_cast<uint64_t>(kept));

    TextVectorizerMaterialization output;
    if (sparse_output) {
        ARROW_ASSIGN_OR_RAISE(
            output.sparse_dataset,
            BuildSparseTextFeatureDataset(
                std::move(feature_matrix), sparse_dataset_name));
    } else {
        ARROW_ASSIGN_OR_RAISE(
            output.dense_table,
            BuildDenseTextFeatureTable(feature_matrix, "tfidf_"));
    }

    if (!state_options_.IsTransformOnly() && state_options_.save_state) {
        std::vector<FittedTextVectorizerFeature> state_features;
        state_features.reserve(all_terms.size());
        for (const auto& term : all_terms) {
            state_features.push_back({term.term, term.idf});
        }
        const auto schema_fingerprint =
            FingerprintPreprocessingSchema(input->schema());
        std::string state_error;
        if (!SaveFittedTextVectorizerState(
                state_options_.state_path, GetName(),
                static_cast<int64_t>(n), schema_fingerprint,
                fitted_configuration, state_features,
                state_options_.state_overwrite, state_error)) {
            return arrow::Status::IOError(
                GetName() + ": failed to save fitted state: " + state_error);
        }
        spdlog::info(
            "TFIDFVectorizer: saved fitted state '{}' (fit_rows={}, "
            "features={}, schema={})",
            state_options_.state_path, n, state_features.size(),
            schema_fingerprint);
    }

    spdlog::info("TFIDFVectorizer: {} docs x {} features (capped from {}, "
                 "filtered from {} with min_df={}), use_idf={}, "
                 "smooth_idf={}, norm={}, stop_words={}, ngram_range={},{} "
                 "classes={}, mode={}, output_format={}, nnz={}, bounded=true",
                 n, kept, filtered_vocab, full_vocab, min_df_,
                 use_idf_, smooth_idf_, norm_,
                 stop_words_,
                 ngram_min_, ngram_max_,
                 class_names.size(), state_options_.operation_mode,
                 output_format_, expected_nnz);
    report_progress(
        "TF-IDF materialization complete",
        "Materialized " + std::to_string(n) + " rows x " +
            std::to_string(kept) + " TF-IDF features",
        1.0f,
        bounded_memory_estimate,
        sparse_output ? expected_nnz
                      : static_cast<uint64_t>(n) *
                            static_cast<uint64_t>(kept),
        sparse_output ? expected_nnz
                      : static_cast<uint64_t>(n) *
                            static_cast<uint64_t>(kept));
    return output;
}

} // namespace cyxwiz
