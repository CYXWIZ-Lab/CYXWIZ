#include "text_tokenizer_operator.h"
#include "../materialization_memory_guard.h"
#include "../profiler_trace.h"
#include "text_column_utils.h"

#include <cyxwiz/tokenizer.h>

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cmath>
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

bool ParseTokenIdList(const std::string& text, std::vector<int>& ids) {
    ids.clear();
    const char* current = text.data();
    const char* end = text.data() + text.size();
    while (current < end) {
        while (current < end) {
            const unsigned char c = static_cast<unsigned char>(*current);
            if (std::isspace(c) || *current == ',' || *current == '[' ||
                *current == ']') {
                ++current;
            } else {
                break;
            }
        }
        if (current >= end) break;
        int value = 0;
        const auto parsed = std::from_chars(current, end, value);
        if (parsed.ec != std::errc{} || parsed.ptr == current) {
            return false;
        }
        ids.push_back(value);
        current = parsed.ptr;
    }
    return true;
}

std::string JoinTokenIds(const std::vector<int>& ids) {
    std::ostringstream out;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) out << ' ';
        out << ids[i];
    }
    return out.str();
}

arrow::Result<double> NumericValueAt(
    const std::shared_ptr<arrow::ChunkedArray>& column,
    int64_t row) {
    int64_t offset = row;
    for (int chunk_index = 0; chunk_index < column->num_chunks(); ++chunk_index) {
        const auto& chunk = column->chunk(chunk_index);
        if (offset >= chunk->length()) {
            offset -= chunk->length();
            continue;
        }
        if (chunk->IsNull(offset)) {
            return arrow::Status::Invalid("Token id column contains null");
        }
        switch (chunk->type_id()) {
            case arrow::Type::INT8:
                return static_cast<double>(
                    std::static_pointer_cast<arrow::Int8Array>(chunk)->Value(offset));
            case arrow::Type::INT16:
                return static_cast<double>(
                    std::static_pointer_cast<arrow::Int16Array>(chunk)->Value(offset));
            case arrow::Type::INT32:
                return static_cast<double>(
                    std::static_pointer_cast<arrow::Int32Array>(chunk)->Value(offset));
            case arrow::Type::INT64:
                return static_cast<double>(
                    std::static_pointer_cast<arrow::Int64Array>(chunk)->Value(offset));
            case arrow::Type::UINT8:
                return static_cast<double>(
                    std::static_pointer_cast<arrow::UInt8Array>(chunk)->Value(offset));
            case arrow::Type::UINT16:
                return static_cast<double>(
                    std::static_pointer_cast<arrow::UInt16Array>(chunk)->Value(offset));
            case arrow::Type::UINT32:
                return static_cast<double>(
                    std::static_pointer_cast<arrow::UInt32Array>(chunk)->Value(offset));
            case arrow::Type::UINT64:
                return static_cast<double>(
                    std::static_pointer_cast<arrow::UInt64Array>(chunk)->Value(offset));
            case arrow::Type::FLOAT:
                return static_cast<double>(
                    std::static_pointer_cast<arrow::FloatArray>(chunk)->Value(offset));
            case arrow::Type::DOUBLE:
                return std::static_pointer_cast<arrow::DoubleArray>(chunk)->Value(offset);
            default:
                return arrow::Status::TypeError(
                    "Token id column must be integer or floating-point, got " +
                    chunk->type()->ToString());
        }
    }
    return arrow::Status::IndexError("Token id row out of range");
}

bool IsNumericTokenIdType(arrow::Type::type type_id) {
    switch (type_id) {
        case arrow::Type::INT8:
        case arrow::Type::INT16:
        case arrow::Type::INT32:
        case arrow::Type::INT64:
        case arrow::Type::UINT8:
        case arrow::Type::UINT16:
        case arrow::Type::UINT32:
        case arrow::Type::UINT64:
        case arrow::Type::FLOAT:
        case arrow::Type::DOUBLE:
            return true;
        default:
            return false;
    }
}

bool IsTokenIdColumnType(const std::shared_ptr<arrow::ChunkedArray>& column) {
    if (!column || column->num_chunks() == 0) return false;
    for (int chunk_index = 0; chunk_index < column->num_chunks(); ++chunk_index) {
        if (!IsNumericTokenIdType(column->chunk(chunk_index)->type_id())) {
            return false;
        }
    }
    return true;
}

arrow::Result<double> NumericArrayValueAt(
    const std::shared_ptr<arrow::Array>& array,
    int64_t index) {
    if (!array) {
        return arrow::Status::Invalid("Token id list values array is null");
    }
    if (index < 0 || index >= array->length()) {
        return arrow::Status::IndexError("Token id list value index out of range");
    }
    if (array->IsNull(index)) {
        return arrow::Status::Invalid("Token id list contains null value");
    }
    switch (array->type_id()) {
        case arrow::Type::INT8:
            return static_cast<double>(
                std::static_pointer_cast<arrow::Int8Array>(array)->Value(index));
        case arrow::Type::INT16:
            return static_cast<double>(
                std::static_pointer_cast<arrow::Int16Array>(array)->Value(index));
        case arrow::Type::INT32:
            return static_cast<double>(
                std::static_pointer_cast<arrow::Int32Array>(array)->Value(index));
        case arrow::Type::INT64:
            return static_cast<double>(
                std::static_pointer_cast<arrow::Int64Array>(array)->Value(index));
        case arrow::Type::UINT8:
            return static_cast<double>(
                std::static_pointer_cast<arrow::UInt8Array>(array)->Value(index));
        case arrow::Type::UINT16:
            return static_cast<double>(
                std::static_pointer_cast<arrow::UInt16Array>(array)->Value(index));
        case arrow::Type::UINT32:
            return static_cast<double>(
                std::static_pointer_cast<arrow::UInt32Array>(array)->Value(index));
        case arrow::Type::UINT64:
            return static_cast<double>(
                std::static_pointer_cast<arrow::UInt64Array>(array)->Value(index));
        case arrow::Type::FLOAT:
            return static_cast<double>(
                std::static_pointer_cast<arrow::FloatArray>(array)->Value(index));
        case arrow::Type::DOUBLE:
            return std::static_pointer_cast<arrow::DoubleArray>(array)->Value(index);
        default:
            return arrow::Status::TypeError(
                "Token id list values must be integer or floating-point, got " +
                array->type()->ToString());
    }
}

arrow::Result<int> TokenIdFromDouble(double value, const std::string& context) {
    const double rounded = std::round(value);
    if (std::fabs(value - rounded) > 1.0e-6) {
        return arrow::Status::Invalid(context + " value must be an integer");
    }
    if (rounded < static_cast<double>(std::numeric_limits<int>::min()) ||
        rounded > static_cast<double>(std::numeric_limits<int>::max())) {
        return arrow::Status::Invalid(context + " value is outside int range");
    }
    return static_cast<int>(rounded);
}

bool IsTokenIdListColumnType(const std::shared_ptr<arrow::ChunkedArray>& column) {
    if (!column || column->num_chunks() == 0) return false;
    for (int chunk_index = 0; chunk_index < column->num_chunks(); ++chunk_index) {
        const auto& chunk = column->chunk(chunk_index);
        std::shared_ptr<arrow::DataType> value_type;
        switch (chunk->type_id()) {
            case arrow::Type::LIST:
                value_type = std::static_pointer_cast<arrow::ListType>(chunk->type())->value_type();
                break;
            case arrow::Type::LARGE_LIST:
                value_type = std::static_pointer_cast<arrow::LargeListType>(chunk->type())->value_type();
                break;
            case arrow::Type::FIXED_SIZE_LIST:
                value_type = std::static_pointer_cast<arrow::FixedSizeListType>(chunk->type())->value_type();
                break;
            default:
                return false;
        }
        if (!value_type || !IsNumericTokenIdType(value_type->id())) {
            return false;
        }
    }
    return true;
}

arrow::Result<std::vector<std::vector<int>>> ReadTokenIdListColumn(
    const std::shared_ptr<arrow::ChunkedArray>& column) {
    std::vector<std::vector<int>> rows;
    if (!column) {
        return arrow::Status::Invalid("Token id list column is null");
    }
    rows.reserve(static_cast<size_t>(column->length()));
    for (int chunk_index = 0; chunk_index < column->num_chunks(); ++chunk_index) {
        const auto& chunk = column->chunk(chunk_index);
        for (int64_t row = 0; row < chunk->length(); ++row) {
            if (chunk->IsNull(row)) {
                return arrow::Status::Invalid("Token id list column contains null row");
            }
            std::shared_ptr<arrow::Array> values;
            int64_t offset = 0;
            int64_t length = 0;
            switch (chunk->type_id()) {
                case arrow::Type::LIST: {
                    auto list = std::static_pointer_cast<arrow::ListArray>(chunk);
                    values = list->values();
                    offset = list->value_offset(row);
                    length = list->value_length(row);
                    break;
                }
                case arrow::Type::LARGE_LIST: {
                    auto list = std::static_pointer_cast<arrow::LargeListArray>(chunk);
                    values = list->values();
                    offset = list->value_offset(row);
                    length = list->value_length(row);
                    break;
                }
                case arrow::Type::FIXED_SIZE_LIST: {
                    auto list = std::static_pointer_cast<arrow::FixedSizeListArray>(chunk);
                    values = list->values();
                    length = list->list_type()->list_size();
                    offset = row * length;
                    break;
                }
                default:
                    return arrow::Status::TypeError(
                        "Token id list column must be list, large_list, or fixed_size_list");
            }
            std::vector<int> ids;
            ids.reserve(static_cast<size_t>(length));
            for (int64_t item = 0; item < length; ++item) {
                ARROW_ASSIGN_OR_RAISE(
                    const double value,
                    NumericArrayValueAt(values, offset + item));
                ARROW_ASSIGN_OR_RAISE(
                    const int id,
                    TokenIdFromDouble(value, "Token id list"));
                ids.push_back(id);
            }
            rows.push_back(std::move(ids));
        }
    }
    return rows;
}

} // namespace

bool TextTokenizerOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {

    output_mode_ = "wide";
    document_id_col_.clear();
    split_col_ = "split";
    token_ids_col_ = "token_ids";
    if (auto p=params.find("output_mode");p!=params.end()) output_mode_=p->second;
    if (auto p=params.find("document_id_col");p!=params.end()) document_id_col_=p->second;
    if (auto p=params.find("split_col");p!=params.end()) split_col_=p->second;
    if (auto p=params.find("token_ids_col");p!=params.end()) token_ids_col_=p->second;
    if (output_mode_!="wide" && output_mode_!="causal_windows" &&
        output_mode_!="decode" && output_mode_!="roundtrip") {
        error="TextTokenizer: output_mode must be wide, causal_windows, decode, or roundtrip"; return false;
    }
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
    if ((output_mode_ != "decode") && (it == params.end() || it->second.empty())) {
        error = "TextTokenizer: 'text_col' parameter is required";
        return false;
    }
    if (it != params.end()) {
        text_col_ = it->second;
    }
    if (output_mode_ == "decode" && token_ids_col_.empty()) {
        error = "TextTokenizer: decode output_mode requires token_ids_col";
        return false;
    }

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
        lowercase_ = tokenizer_type_ != 3;
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
    if (tokenizer_type_ < 0 || tokenizer_type_ > 6) {
        error = "TextTokenizer: tokenizer_type must be 0..6 (got " +
                std::to_string(tokenizer_type_) + ")";
        return false;
    }
    if (tokenizer_type_ == 5 || tokenizer_type_ == 6) {
        error = "TextTokenizer: SentencePiece tokenizer support is not enabled in this build; install/build the optional provider or choose a native tokenizer family";
        return false;
    }
    if (tokenizer_type_ == 3 && (lowercase_ || min_word_freq_ < 1 || max_vocab_size_ < 260)) {
        error = "TextTokenizer: Byte BPE requires lowercase=false, min_word_freq>=1 and max_vocab_size>=260";
        return false;
    }
    if (output_mode_=="causal_windows" && (document_id_col_.empty() || split_col_.empty() ||
        document_id_col_==text_col_ || split_col_==text_col_ || document_id_col_==split_col_ || !label_col_.empty())) {
        error="Causal windows require distinct text/document_id_col/split_col and no classification label_col"; return false;
    }
    if ((output_mode_ == "decode" || output_mode_ == "roundtrip") &&
        !label_col_.empty()) {
        error = "TextTokenizer: decode/roundtrip modes do not support label_col";
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

    auto make_tokenizer = [&]() {
        TokenizerType tt = TokenizerType::Word;
        switch (tokenizer_type_) {
            case 0: tt = TokenizerType::Whitespace; break;
            case 2: tt = TokenizerType::Character; break;
            case 3: tt = TokenizerType::ByteBPE; break;
            case 4: tt = TokenizerType::WordPiece; break;
            case 5: tt = TokenizerType::SentencePieceBPE; break;
            case 6: tt = TokenizerType::SentencePieceUnigram; break;
            default: tt = TokenizerType::Word; break;
        }
        Tokenizer tokenizer(tt);
        tokenizer.SetCancellationQuery(GetCancellationQuery());
        tokenizer.SetLowercase(lowercase_);
        tokenizer.SetMaxLength(max_length_);
        tokenizer.SetPadding(true);
        tokenizer.SetTruncation(true);
        return tokenizer;
    };

    if (output_mode_ == "decode") {
        if (vocab_file_.empty()) {
            return arrow::Status::Invalid(
                "TextTokenizer: decode output_mode requires vocab_file");
        }
        Tokenizer tokenizer = make_tokenizer();
        if (!std::filesystem::exists(vocab_file_) ||
            !tokenizer.GetVocabulary().LoadFromFile(vocab_file_)) {
            return arrow::Status::Invalid(
                "TextTokenizer: failed to load vocab_file '" + vocab_file_ +
                "' for decode");
        }
        try {
            tokenizer.ValidateVocabulary();
        } catch (const std::exception& e) {
            return arrow::Status::Invalid("TextTokenizer: ", e.what());
        }
        last_vocab_size_ = tokenizer.GetVocabulary().Size();
        return DecodeTokenRows(input, tokenizer);
    }

    if (output_mode_=="causal_windows") ARROW_RETURN_NOT_OK(ValidateWindowInput(input));
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

    // Build tokenizer + vocab from the corpus. Round-trip inspection checks
    // encode/decode identity for the artifact itself, so it must not pad or
    // truncate source text by the training context length.
    Tokenizer tokenizer = make_tokenizer();
    if (output_mode_ == "roundtrip") {
        tokenizer.SetPadding(false);
        tokenizer.SetTruncation(false);
    }

    try {
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
    tokenizer.ValidateVocabulary();
    } catch (const std::exception& e) {
        ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
        return arrow::Status::Invalid("TextTokenizer: ", e.what());
    }
    const size_t trained_vocab_size = tokenizer.GetVocabulary().Size();
    report_progress("Vocabulary ready",
                    "Tokenizer vocabulary ready with " +
                    std::to_string(trained_vocab_size) + " entries",
                    0.35,
                    total_rows,
                    total_rows,
                    estimated_token_matrix_bytes);

    if (output_mode_ == "roundtrip") {
        last_vocab_size_ = trained_vocab_size;
        return RoundTripTextRows(texts, tokenizer);
    }

    if (output_mode_=="causal_windows") return BuildTokenWindows(input,texts,tokenizer);

    // Encode + pad. EncodeBatch then PadBatch produces the final
    // [num_samples, max_length] int matrix.
    report_progress("Tokenizing rows",
                    "Encoding and padding text rows",
                    0.40,
                    0,
                    total_rows,
                    estimated_token_matrix_bytes);
    std::vector<std::vector<int>> encoded;
    try {
        encoded = tokenizer.EncodeBatch(texts);
    } catch (const std::exception& e) {
        ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
        return arrow::Status::Invalid("TextTokenizer: ", e.what());
    }
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

arrow::Result<std::shared_ptr<arrow::Table>>
TextTokenizerOperator::DecodeTokenRows(
    const std::shared_ptr<arrow::Table>& input,
    Tokenizer& tokenizer) {
    if (!input) {
        return arrow::Status::Invalid("TextTokenizer: input table is null");
    }

    std::vector<std::vector<int>> rows;
    if (auto id_column = input->GetColumnByName(token_ids_col_)) {
        std::string bad_type;
        if (IsStringLikeColumn(id_column, bad_type)) {
            std::vector<std::string> id_texts;
            if (!ReadColumnAsStrings(
                    id_column, id_texts, bad_type, GetCancellationQuery())) {
                return arrow::Status::TypeError(
                    "TextTokenizer: token_ids_col '" + token_ids_col_ +
                    "' must be string/large_string, got '" + bad_type + "'");
            }
            rows.reserve(id_texts.size());
            for (size_t row = 0; row < id_texts.size(); ++row) {
                if ((row & 1023) == 0) {
                    ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
                }
                std::vector<int> ids;
                if (!ParseTokenIdList(id_texts[row], ids)) {
                    return arrow::Status::Invalid(
                        "TextTokenizer: failed to parse token_ids_col '" +
                        token_ids_col_ + "' at row " + std::to_string(row));
                }
                rows.push_back(std::move(ids));
            }
        } else if (IsTokenIdListColumnType(id_column)) {
            ARROW_ASSIGN_OR_RAISE(rows, ReadTokenIdListColumn(id_column));
        } else if (IsTokenIdColumnType(id_column)) {
            rows.reserve(static_cast<size_t>(input->num_rows()));
            for (int64_t row = 0; row < input->num_rows(); ++row) {
                ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
                ARROW_ASSIGN_OR_RAISE(const double value,
                                      NumericValueAt(id_column, row));
                ARROW_ASSIGN_OR_RAISE(
                    const int id,
                    TokenIdFromDouble(value,
                                      "TextTokenizer: token id at row " +
                                          std::to_string(row)));
                rows.push_back({id});
            }
        } else {
            return arrow::Status::TypeError(
                "TextTokenizer: token_ids_col '" + token_ids_col_ +
                "' must be string, numeric, or list<numeric>, got '" +
                id_column->type()->ToString() + "'");
        }
    } else {
        std::vector<std::shared_ptr<arrow::ChunkedArray>> token_columns;
        for (int index = 0; index < max_length_; ++index) {
            auto column = input->GetColumnByName("tok_" + std::to_string(index));
            if (!column) {
                break;
            }
            if (!IsTokenIdColumnType(column)) {
                return arrow::Status::TypeError(
                    "TextTokenizer: tok_" + std::to_string(index) +
                    " must be numeric for decode");
            }
            token_columns.push_back(std::move(column));
        }
        if (token_columns.empty()) {
            return arrow::Status::KeyError(
                "TextTokenizer: decode requires token_ids_col '" + token_ids_col_ +
                "' or wide tok_0..tok_n columns");
        }
        rows.reserve(static_cast<size_t>(input->num_rows()));
        for (int64_t row = 0; row < input->num_rows(); ++row) {
            if ((row & 1023) == 0) {
                ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
            }
            std::vector<int> ids;
            ids.reserve(token_columns.size());
            for (size_t column_index = 0; column_index < token_columns.size();
                 ++column_index) {
                ARROW_ASSIGN_OR_RAISE(const double value,
                                      NumericValueAt(token_columns[column_index], row));
                ARROW_ASSIGN_OR_RAISE(
                    const int id,
                    TokenIdFromDouble(value,
                                      "TextTokenizer: tok_" +
                                          std::to_string(column_index) +
                                          " at row " + std::to_string(row)));
                ids.push_back(id);
            }
            rows.push_back(std::move(ids));
        }
    }

    arrow::StringBuilder decoded_builder;
    arrow::StringBuilder id_builder;
    ARROW_RETURN_NOT_OK(decoded_builder.Reserve(static_cast<int64_t>(rows.size())));
    ARROW_RETURN_NOT_OK(id_builder.Reserve(static_cast<int64_t>(rows.size())));
    for (size_t row = 0; row < rows.size(); ++row) {
        if ((row & 1023) == 0) {
            ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
        }
        ARROW_RETURN_NOT_OK(decoded_builder.Append(tokenizer.Decode(rows[row])));
        ARROW_RETURN_NOT_OK(id_builder.Append(JoinTokenIds(rows[row])));
    }
    std::shared_ptr<arrow::Array> decoded_array;
    std::shared_ptr<arrow::Array> id_array;
    ARROW_RETURN_NOT_OK(decoded_builder.Finish(&decoded_array));
    ARROW_RETURN_NOT_OK(id_builder.Finish(&id_array));
    auto schema = arrow::schema({
        arrow::field("decoded_text", arrow::utf8()),
        arrow::field("token_ids", arrow::utf8()),
    });
    return arrow::Table::Make(schema, {decoded_array, id_array},
                              static_cast<int64_t>(rows.size()));
}

arrow::Result<std::shared_ptr<arrow::Table>>
TextTokenizerOperator::RoundTripTextRows(
    const std::vector<std::string>& texts,
    Tokenizer& tokenizer) {
    arrow::StringBuilder input_builder;
    arrow::StringBuilder encoded_builder;
    arrow::StringBuilder decoded_builder;
    arrow::BooleanBuilder ok_builder;
    ARROW_RETURN_NOT_OK(input_builder.Reserve(static_cast<int64_t>(texts.size())));
    ARROW_RETURN_NOT_OK(encoded_builder.Reserve(static_cast<int64_t>(texts.size())));
    ARROW_RETURN_NOT_OK(decoded_builder.Reserve(static_cast<int64_t>(texts.size())));
    ARROW_RETURN_NOT_OK(ok_builder.Reserve(static_cast<int64_t>(texts.size())));

    for (size_t row = 0; row < texts.size(); ++row) {
        if ((row & 1023) == 0) {
            ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
        }
        const std::vector<int> ids = tokenizer.Encode(texts[row]);
        const std::string decoded = tokenizer.Decode(ids);
        ARROW_RETURN_NOT_OK(input_builder.Append(texts[row]));
        ARROW_RETURN_NOT_OK(encoded_builder.Append(JoinTokenIds(ids)));
        ARROW_RETURN_NOT_OK(decoded_builder.Append(decoded));
        ARROW_RETURN_NOT_OK(ok_builder.Append(decoded == texts[row]));
    }

    std::shared_ptr<arrow::Array> input_array;
    std::shared_ptr<arrow::Array> encoded_array;
    std::shared_ptr<arrow::Array> decoded_array;
    std::shared_ptr<arrow::Array> ok_array;
    ARROW_RETURN_NOT_OK(input_builder.Finish(&input_array));
    ARROW_RETURN_NOT_OK(encoded_builder.Finish(&encoded_array));
    ARROW_RETURN_NOT_OK(decoded_builder.Finish(&decoded_array));
    ARROW_RETURN_NOT_OK(ok_builder.Finish(&ok_array));

    auto schema = arrow::schema({
        arrow::field("input_text", arrow::utf8()),
        arrow::field("encoded_ids", arrow::utf8()),
        arrow::field("decoded_text", arrow::utf8()),
        arrow::field("roundtrip_ok", arrow::boolean()),
    });
    return arrow::Table::Make(
        schema,
        {input_array, encoded_array, decoded_array, ok_array},
        static_cast<int64_t>(texts.size()));
}

} // namespace cyxwiz
