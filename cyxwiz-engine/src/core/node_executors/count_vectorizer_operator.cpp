#include "count_vectorizer_operator.h"
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
#include <numeric>
#include <sstream>
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
    state_options_ = {};

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

    if (!text_vectorizer_contract::ParseNGramRange(
            params, "CountVectorizer", ngram_min_, ngram_max_, error)) {
        return false;
    }

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

    return ParseFittedPreprocessingOptions(
        params, GetName(), state_options_, error);
}

std::map<std::string, std::string>
CountVectorizerOperator::BuildFittedConfiguration() const {
    return {
        {"text_col", text_col_},
        {"max_features", std::to_string(max_features_)},
        {"norm", norm_},
        {"ngram_range", std::to_string(ngram_min_) + "," +
                            std::to_string(ngram_max_)},
        {"stop_words", stop_words_},
        {"binary", binary_ ? "true" : "false"},
        {"output_format", output_format_},
        {"value_semantics", "sklearn_raw_count_v1"},
    };
}

bool CountVectorizerOperator::CollectCacheDependencies(
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
            "CountVectorizer: text column '" + text_col_ + "' not found");
    }

    std::string bad_type;
    if (!IsStringLikeColumn(text_column, bad_type)) {
        return arrow::Status::TypeError(
            "CountVectorizer: text column '" + text_col_ +
            "' must be string/large_string, got '" + bad_type + "'");
    }

    const auto fitted_configuration = BuildFittedConfiguration();
    FittedTextVectorizerState fitted_state;
    std::vector<CountTermStats> fitted_terms;
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
            CountTermStats term;
            term.term = feature.term;
            fitted_terms.push_back(std::move(term));
        }
        spdlog::info(
            "CountVectorizer: Transform Only loaded '{}' (fit_rows={}, "
            "features={}, schema={})",
            state_options_.state_path, fitted_state.fit_rows,
            fitted_terms.size(), fitted_state.input_schema_fingerprint);
    }

    const uint64_t planned_rows =
        static_cast<uint64_t>(std::max<int64_t>(0, text_column->length()));
    if (planned_rows == 0) {
        return arrow::Status::Invalid("CountVectorizer: empty corpus");
    }
    const uint64_t planned_features = state_options_.IsTransformOnly()
        ? static_cast<uint64_t>(std::max<size_t>(1, fitted_terms.size()))
        : static_cast<uint64_t>(std::max(1, max_features_));
    const auto preflight_estimate = EstimateDenseMaterializationMemory(
        planned_rows, planned_features, static_cast<uint64_t>(sizeof(float)));
    const auto preflight_decision = EvaluateMaterializationMemory(
        preflight_estimate, GetMaterializationMemoryContext());
    const std::string preflight_message =
        BuildMaterializationMemoryPreflightMessage(
            "CountVectorizer", "max_features",
            preflight_estimate, preflight_decision,
            "Reduce CountVectorizer max_features, sample fewer rows first, "
            "or use a sparse/chunked materialization path when supported.");
    uint64_t planned_cells = 0;
    if (!CheckedMulU64(planned_rows, planned_features, planned_cells)) {
        planned_cells = std::numeric_limits<uint64_t>::max();
    }
    if (progress_callback_) {
        PipelineOperatorProgress event;
        event.stage = "CountVectorizer memory preflight";
        event.message = preflight_message;
        event.status = MaterializationMemoryRiskToProgressStatus(
            preflight_decision.risk);
        event.progress = 0.03f;
        event.estimated_memory_bytes = preflight_estimate.estimated_peak_bytes;
        event.memory_risk_level = MaterializationMemoryRiskName(
            preflight_decision.risk);
        event.processed_items = 0;
        event.total_items = planned_cells;
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
            "CountVectorizer: text column '" + text_col_ +
            "' must be string/large_string, got '" + bad_type + "'");
    }
    const uint64_t planned_memory_estimate =
        preflight_estimate.estimated_peak_bytes;
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
        if ((row & 1023) == 0) {
            ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
        }
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
        auto tokens = text_vectorizer_contract::BuildNGramFeatures(
            base_tokens, ngram_min_, ngram_max_);
        doc_token_counts.push_back(tokens.size());

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
    }

    if (state_options_.IsTransformOnly()) {
        for (auto& term : fitted_terms) {
            term_stats_by_name.emplace(term.term, std::move(term));
        }
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

    // Match sklearn max_features selection: keep the terms with the highest
    // corpus counts, then use document frequency and term name as stable
    // tie-breakers.
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
                if (a.corpus_count != b.corpus_count) {
                    return a.corpus_count > b.corpus_count;
                }
                if (a.doc_freq != b.doc_freq) {
                    return a.doc_freq > b.doc_freq;
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
    const auto bounded_memory_plan = EstimateDenseMaterializationMemory(
        static_cast<uint64_t>(n),
        static_cast<uint64_t>(std::max<size_t>(1, kept)),
        static_cast<uint64_t>(sizeof(float)));
    const uint64_t bounded_memory_estimate =
        bounded_memory_plan.estimated_peak_bytes;
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
        if (!ReadLabelColumnAsInt(
                label_column, labels, class_names, lbad,
                GetCancellationQuery())) {
            ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
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
        if ((r & 1023) == 0) {
            ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
        }
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
                    : static_cast<float>(pair.second);
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
                state_options_.IsTransformOnly()
                    ? "Transforming with fitted Count state"
                    : "Building count rows",
                (state_options_.IsTransformOnly()
                     ? "Transformed "
                     : "Built ") +
                    std::to_string(r + 1) + "/" + std::to_string(n) +
                    " count rows",
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

    if (!state_options_.IsTransformOnly() && state_options_.save_state) {
        std::vector<FittedTextVectorizerFeature> state_features;
        state_features.reserve(all_terms.size());
        for (const auto& term : all_terms) {
            state_features.push_back({term.term, 1.0});
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
            "CountVectorizer: saved fitted state '{}' (fit_rows={}, "
            "features={}, schema={})",
            state_options_.state_path, n, state_features.size(),
            schema_fingerprint);
    }

    spdlog::info("CountVectorizer: {} docs x {} features (capped from {}), "
                 "binary={}, norm={}, stop_words={}, ngram_range={},{} "
                 "classes={}, mode={}",
                 n, kept, full_vocab, binary_, norm_, stop_words_,
                 ngram_min_, ngram_max_, class_names.size(),
                 state_options_.operation_mode);
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
