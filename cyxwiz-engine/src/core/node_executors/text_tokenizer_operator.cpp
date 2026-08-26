#include "text_tokenizer_operator.h"
#include "../materialization_memory_guard.h"
#include "../profiler_trace.h"
#include "text_column_utils.h"

#include <cyxwiz/tokenizer.h>

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

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

std::string BuildTokenizerMemoryPreflightMessage(
    const MaterializationMemoryEstimate& estimate,
    const MaterializationMemoryDecision& decision) {
    std::ostringstream ss;
    ss << "TextTokenizer memory preflight: risk="
       << MaterializationMemoryRiskName(decision.risk)
       << ", rows=" << estimate.rows
       << ", output_columns=" << estimate.output_features
       << ", raw=" << FormatMaterializationBytes(estimate.raw_output_bytes)
       << ", estimated_peak="
       << FormatMaterializationBytes(estimate.estimated_peak_bytes)
       << ", available="
       << FormatMaterializationBytes(decision.available_bytes)
       << ", safe_budget="
       << FormatMaterializationBytes(decision.safe_budget_bytes)
       << ". " << decision.reason
       << ". Suggestion: reduce max_length, source rows, or use a future "
          "chunked tokenization path.";
    return ss.str();
}

} // namespace

bool TextTokenizerOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {

    text_col_.clear();
    label_col_.clear();
    vocab_file_.clear();
    max_length_ = 256;
    tokenizer_type_ = 1;
    lowercase_ = true;
    min_word_freq_ = 2;
    max_vocab_size_ = 10000;
    pad_value_ = 0;
    vocab_build_if_missing_ = false;
    last_vocab_size_ = 0;

    auto it = params.find("text_col");
    if (it == params.end() || it->second.empty()) {
        error = "TextTokenizer: 'text_col' parameter is required";
        return false;
    }
    text_col_ = it->second;

    auto lc = params.find("label_col");
    if (lc != params.end()) label_col_ = lc->second;
    auto vf = params.find("vocab_file");
    if (vf != params.end()) vocab_file_ = vf->second;
    auto vb = params.find("vocab_build_if_missing");
    if (vb != params.end() && !vb->second.empty()) {
        if (vb->second == "true" || vb->second == "1") {
            vocab_build_if_missing_ = true;
        } else if (vb->second == "false" || vb->second == "0") {
            vocab_build_if_missing_ = false;
        } else {
            error = "TextTokenizer: 'vocab_build_if_missing' must be true/false";
            return false;
        }
    }

    auto read_int = [&](const char* key, int default_value, int& out) -> bool {
        auto p = params.find(key);
        if (p == params.end() || p->second.empty()) {
            out = default_value;
            return true;
        }
        try { out = std::stoi(p->second); }
        catch (...) {
            error = std::string("TextTokenizer: '") + key +
                    "' is not a valid integer: " + p->second;
            return false;
        }
        return true;
    };

    if (!read_int("max_length",     256,   max_length_))     return false;
    if (!read_int("tokenizer_type", 1,     tokenizer_type_)) return false;
    if (!read_int("min_word_freq",  2,     min_word_freq_))  return false;
    if (!read_int("max_vocab_size", 10000, max_vocab_size_)) return false;
    if (!read_int("pad_value",      0,     pad_value_))      return false;

    auto lcase = params.find("lowercase");
    if (lcase == params.end() || lcase->second.empty()) {
        lowercase_ = true;
    } else if (lcase->second == "true") {
        lowercase_ = true;
    } else if (lcase->second == "false") {
        lowercase_ = false;
    } else {
        error = "TextTokenizer: 'lowercase' must be 'true' or 'false' (got '" +
                lcase->second + "')";
        return false;
    }

    if (max_length_ < 1) {
        error = "TextTokenizer: max_length must be >= 1 (got " +
                std::to_string(max_length_) + ")";
        return false;
    }
    if (tokenizer_type_ < 0 || tokenizer_type_ > 2) {
        error = "TextTokenizer: tokenizer_type must be 0..2 (got " +
                std::to_string(tokenizer_type_) + ")";
        return false;
    }
    if (pad_value_ < 0) {
        error = "TextTokenizer: pad_value must be >= 0 (got " +
                std::to_string(pad_value_) + ")";
        return false;
    }

    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
TextTokenizerOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz TextTokenizer Materializer");

    last_vocab_size_ = 0;
    if (!input) {
        return arrow::Status::Invalid("TextTokenizer: input table is null");
    }

    auto report_progress = [&](std::string stage,
                               std::string message,
                               double progress,
                               uint64_t processed = 0,
                               uint64_t total = 0,
                               uint64_t memory = 0) {
        if (!progress_callback_) {
            return;
        }
        PipelineOperatorProgress event;
        event.stage = std::move(stage);
        event.message = std::move(message);
        event.status = "running";
        event.progress = static_cast<float>(progress);
        event.processed_items = processed;
        event.total_items = total;
        event.estimated_memory_bytes = memory;
        progress_callback_(event);
    };

    auto text_column = input->GetColumnByName(text_col_);
    if (!text_column) {
        return arrow::Status::KeyError(
            "TextTokenizer: text column '" + text_col_ + "' not found");
    }

    std::string bad_type;
    if (!IsStringLikeColumn(text_column, bad_type)) {
        return arrow::Status::TypeError(
            "TextTokenizer: text column '" + text_col_ +
            "' must be string/large_string, got '" + bad_type + "'");
    }

    const uint64_t planned_rows =
        static_cast<uint64_t>(std::max<int64_t>(0, text_column->length()));
    const uint64_t planned_output_columns =
        static_cast<uint64_t>(max_length_ + (label_col_.empty() ? 0 : 1));
    const auto preflight_estimate = EstimateDenseMaterializationMemory(
        planned_rows,
        planned_output_columns,
        static_cast<uint64_t>(sizeof(float)));
    const auto preflight_decision = EvaluateMaterializationMemory(
        preflight_estimate, GetMaterializationMemoryContext());
    const std::string preflight_message = BuildTokenizerMemoryPreflightMessage(
        preflight_estimate, preflight_decision);
    uint64_t planned_cells = 0;
    if (!CheckedMulU64(planned_rows, planned_output_columns, planned_cells)) {
        planned_cells = std::numeric_limits<uint64_t>::max();
    }
    if (progress_callback_) {
        PipelineOperatorProgress event;
        event.stage = "TextTokenizer memory preflight";
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

    report_progress("Reading text",
                    "Reading text column '" + text_col_ + "'",
                    0.05,
                    0,
                    planned_rows,
                    preflight_estimate.estimated_peak_bytes);

    std::vector<std::string> texts;
    if (!ReadColumnAsStrings(
            text_column, texts, bad_type, GetCancellationQuery())) {
        ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
        return arrow::Status::TypeError(
            "TextTokenizer: text column '" + text_col_ +
            "' must be string/large_string, got '" + bad_type + "'");
    }
    const uint64_t total_rows = static_cast<uint64_t>(texts.size());
    const uint64_t estimated_token_matrix_bytes =
        preflight_estimate.estimated_peak_bytes;
    report_progress("Planning token matrix",
                    "Planning " + std::to_string(total_rows) +
                    " rows x " + std::to_string(max_length_) +
                    " token columns",
                    0.15,
                    0,
                    total_rows,
                    estimated_token_matrix_bytes);

    // Build tokenizer + vocab from the corpus.
    TokenizerType tt = TokenizerType::Word;
    switch (tokenizer_type_) {
        case 0: tt = TokenizerType::Whitespace; break;
        case 2: tt = TokenizerType::Character; break;
        default: tt = TokenizerType::Word; break;
    }
    Tokenizer tokenizer(tt);
    tokenizer.SetLowercase(lowercase_);
    tokenizer.SetMaxLength(max_length_);
    tokenizer.SetPadding(true);
    tokenizer.SetTruncation(true);

    if (!vocab_file_.empty() && std::filesystem::exists(vocab_file_)) {
        report_progress("Loading vocabulary",
                        "Loading tokenizer vocabulary from file",
                        0.20,
                        0,
                        total_rows,
                        estimated_token_matrix_bytes);
        if (!tokenizer.GetVocabulary().LoadFromFile(vocab_file_)) {
            return arrow::Status::Invalid(
                "TextTokenizer: failed to load vocab_file '" + vocab_file_ + "'");
        }
    } else if (!vocab_file_.empty() && vocab_build_if_missing_) {
        report_progress("Training vocabulary",
                        "Training tokenizer vocabulary before saving",
                        0.20,
                        0,
                        total_rows,
                        estimated_token_matrix_bytes);
        tokenizer.Train(texts, min_word_freq_, max_vocab_size_);
        ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
        const std::filesystem::path path(vocab_file_);
        if (path.has_parent_path()) {
            std::error_code ec;
            std::filesystem::create_directories(path.parent_path(), ec);
            if (ec) {
                return arrow::Status::IOError(
                    "TextTokenizer: failed to create vocabulary directory '" +
                    path.parent_path().string() + "': " + ec.message());
            }
        }
        if (!tokenizer.GetVocabulary().SaveToFile(vocab_file_)) {
            return arrow::Status::IOError(
                "TextTokenizer: failed to save built vocab_file '" +
                vocab_file_ + "'");
        }
        spdlog::info("TextTokenizer: built and saved vocabulary '{}' with {} entries",
                     vocab_file_, tokenizer.GetVocabulary().Size());
    } else if (!vocab_file_.empty()) {
        return arrow::Status::Invalid(
            "TextTokenizer: vocab_file '" + vocab_file_ +
            "' does not exist. Enable vocab_build_if_missing, build it from the TextTokenizer dialog, or remove vocab_file to train in memory.");
    } else {
        report_progress("Training vocabulary",
                        "Training tokenizer vocabulary in memory",
                        0.20,
                        0,
                        total_rows,
                        estimated_token_matrix_bytes);
        tokenizer.Train(texts, min_word_freq_, max_vocab_size_);
        ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
    }
    const size_t trained_vocab_size = tokenizer.GetVocabulary().Size();
    report_progress("Vocabulary ready",
                    "Tokenizer vocabulary ready with " +
                    std::to_string(trained_vocab_size) + " entries",
                    0.35,
                    total_rows,
                    total_rows,
                    estimated_token_matrix_bytes);

    // Encode + pad. EncodeBatch then PadBatch produces the final
    // [num_samples, max_length] int matrix.
    report_progress("Tokenizing rows",
                    "Encoding and padding text rows",
                    0.40,
                    0,
                    total_rows,
                    estimated_token_matrix_bytes);
    auto encoded = tokenizer.EncodeBatch(texts);
    ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
    auto padded = tokenizer.PadBatch(encoded, max_length_);
    ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
    if (pad_value_ != tokenizer.GetVocabulary().PadIndex()) {
        const int tokenizer_pad = tokenizer.GetVocabulary().PadIndex();
        size_t row_index = 0;
        for (auto& row : padded) {
            if ((row_index & 1023) == 0) {
                ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
            }
            for (int& id : row) {
                if (id == tokenizer_pad) {
                    id = pad_value_;
                }
            }
            ++row_index;
        }
    }

    const size_t n = padded.size();
    report_progress("Rows tokenized",
                    "Encoded and padded " + std::to_string(n) +
                    " text rows",
                    0.55,
                    static_cast<uint64_t>(n),
                    total_rows,
                    estimated_token_matrix_bytes);

    // Read label column if specified.
    std::vector<int> labels;
    std::vector<std::string> class_names;
    if (!label_col_.empty()) {
        auto label_column = input->GetColumnByName(label_col_);
        if (!label_column) {
            return arrow::Status::KeyError(
                "TextTokenizer: label column '" + label_col_ + "' not found");
        }
        std::string lbad;
        if (!ReadLabelColumnAsInt(
                label_column, labels, class_names, lbad,
                GetCancellationQuery())) {
            ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
            return arrow::Status::TypeError(
                "TextTokenizer: label column '" + label_col_ +
                "' has unsupported type '" + lbad + "'");
        }
        if (labels.size() != n) {
            return arrow::Status::Invalid(
                "TextTokenizer: label count (" + std::to_string(labels.size()) +
                ") differs from text count (" + std::to_string(n) + ")");
        }
    }

    // Build wide output columns: tok_0 .. tok_{max-1}, y.
    report_progress("Building Arrow columns",
                    "Allocating token output columns",
                    0.60,
                    0,
                    static_cast<uint64_t>(n),
                    estimated_token_matrix_bytes);
    arrow::MemoryPool* pool = arrow::default_memory_pool();
    std::vector<std::unique_ptr<arrow::FloatBuilder>> tok_builders;
    tok_builders.reserve(max_length_);
    for (int i = 0; i < max_length_; ++i) {
        tok_builders.push_back(std::make_unique<arrow::FloatBuilder>(pool));
        ARROW_RETURN_NOT_OK(tok_builders.back()->Reserve(static_cast<int64_t>(n)));
    }
    arrow::Int32Builder label_builder(pool);
    if (!labels.empty()) {
        ARROW_RETURN_NOT_OK(label_builder.Reserve(static_cast<int64_t>(n)));
    }

    for (size_t r = 0; r < n; ++r) {
        if ((r & 1023) == 0) {
            ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
        }
        const auto& row = padded[r];
        for (int i = 0; i < max_length_; ++i) {
            const float v = (i < static_cast<int>(row.size()))
                ? static_cast<float>(row[i]) : 0.0f;
            ARROW_RETURN_NOT_OK(tok_builders[i]->Append(v));
        }
        if (!labels.empty()) {
            ARROW_RETURN_NOT_OK(label_builder.Append(labels[r]));
        }
        if ((r + 1) == n || ((r + 1) % 1024) == 0) {
            const double row_progress =
                n == 0 ? 0.90 : 0.60 + (0.30 * static_cast<double>(r + 1) /
                                        static_cast<double>(n));
            report_progress("Building token rows",
                            "Writing token rows to Arrow columns",
                            row_progress,
                            static_cast<uint64_t>(r + 1),
                            static_cast<uint64_t>(n),
                            estimated_token_matrix_bytes);
        }
    }

    report_progress("Finishing Arrow table",
                    "Finalizing tokenized Arrow table",
                    0.95,
                    static_cast<uint64_t>(n),
                    static_cast<uint64_t>(n),
                    estimated_token_matrix_bytes);
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    std::vector<std::shared_ptr<arrow::Field>> fields;
    arrays.reserve(max_length_ + (labels.empty() ? 0 : 1));
    fields.reserve(max_length_ + (labels.empty() ? 0 : 1));
    for (int i = 0; i < max_length_; ++i) {
        std::shared_ptr<arrow::Array> arr;
        ARROW_RETURN_NOT_OK(tok_builders[i]->Finish(&arr));
        arrays.push_back(std::move(arr));
        fields.push_back(arrow::field("tok_" + std::to_string(i), arrow::float32()));
    }
    if (!labels.empty()) {
        std::shared_ptr<arrow::Array> arr;
        ARROW_RETURN_NOT_OK(label_builder.Finish(&arr));
        arrays.push_back(std::move(arr));
        fields.push_back(arrow::field("y", arrow::int32()));
    }

    auto out_schema = arrow::schema(fields);
    auto out_table = arrow::Table::Make(out_schema, arrays, static_cast<int64_t>(n));

    spdlog::info("TextTokenizer: {} samples tokenized, vocab_size={}, "
                 "max_length={}, classes={}",
                 n, trained_vocab_size, max_length_,
                 class_names.size());
    last_vocab_size_ = trained_vocab_size;
    report_progress("Complete",
                    "TextTokenizer materialization complete",
                    1.0,
                    static_cast<uint64_t>(n),
                    static_cast<uint64_t>(n),
                    estimated_token_matrix_bytes);
    return out_table;
}

} // namespace cyxwiz
