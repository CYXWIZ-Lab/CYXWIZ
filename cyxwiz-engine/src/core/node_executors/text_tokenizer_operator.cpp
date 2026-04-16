#include "text_tokenizer_operator.h"

#include <cyxwiz/tokenizer.h>

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

// Read all string values from a column into std::vector<std::string>.
// Null values become empty strings (tokenize to no-tokens, pad to all 0).
bool ReadColumnAsStrings(
    const std::shared_ptr<arrow::ChunkedArray>& column,
    std::vector<std::string>& out,
    std::string& error_type_name) {

    out.clear();
    out.reserve(static_cast<size_t>(column->length()));

    for (int c = 0; c < column->num_chunks(); ++c) {
        auto chunk = column->chunk(c);
        const int64_t chunk_len = chunk->length();
        if (chunk->type_id() != arrow::Type::STRING &&
            chunk->type_id() != arrow::Type::LARGE_STRING) {
            error_type_name = chunk->type()->ToString();
            return false;
        }
        if (chunk->type_id() == arrow::Type::STRING) {
            auto arr = std::static_pointer_cast<arrow::StringArray>(chunk);
            for (int64_t i = 0; i < chunk_len; ++i) {
                if (chunk->IsNull(i)) {
                    out.emplace_back();
                } else {
                    out.emplace_back(arr->GetString(i));
                }
            }
        } else {
            auto arr = std::static_pointer_cast<arrow::LargeStringArray>(chunk);
            for (int64_t i = 0; i < chunk_len; ++i) {
                if (chunk->IsNull(i)) {
                    out.emplace_back();
                } else {
                    out.emplace_back(arr->GetString(i));
                }
            }
        }
    }
    return true;
}

// Read a label column. If the column is numeric, copies as int. If
// the column is string, builds a string→int mapping (returned via
// `class_names`) and emits the per-row class index.
bool ReadLabelColumnAsInt(
    const std::shared_ptr<arrow::ChunkedArray>& column,
    std::vector<int>& out,
    std::vector<std::string>& class_names,
    std::string& error_type_name) {

    out.clear();
    out.reserve(static_cast<size_t>(column->length()));
    class_names.clear();

    if (column->num_chunks() == 0) return true;
    auto first_type = column->chunk(0)->type_id();

    if (first_type == arrow::Type::STRING ||
        first_type == arrow::Type::LARGE_STRING) {
        // Build string→int map by first-seen order. This matches what
        // TextDataset's class_names construction does.
        std::map<std::string, int> name_to_idx;
        for (int c = 0; c < column->num_chunks(); ++c) {
            auto chunk = column->chunk(c);
            const int64_t chunk_len = chunk->length();
            for (int64_t i = 0; i < chunk_len; ++i) {
                std::string s;
                if (!chunk->IsNull(i)) {
                    if (chunk->type_id() == arrow::Type::STRING) {
                        s = std::static_pointer_cast<arrow::StringArray>(chunk)->GetString(i);
                    } else {
                        s = std::static_pointer_cast<arrow::LargeStringArray>(chunk)->GetString(i);
                    }
                }
                auto it = name_to_idx.find(s);
                int idx;
                if (it == name_to_idx.end()) {
                    idx = static_cast<int>(class_names.size());
                    class_names.push_back(s);
                    name_to_idx[s] = idx;
                } else {
                    idx = it->second;
                }
                out.push_back(idx);
            }
        }
        return true;
    }

    // Numeric label column — copy values cast to int.
    for (int c = 0; c < column->num_chunks(); ++c) {
        auto chunk = column->chunk(c);
        const int64_t chunk_len = chunk->length();
        for (int64_t i = 0; i < chunk_len; ++i) {
            if (chunk->IsNull(i)) { out.push_back(0); continue; }
            switch (chunk->type_id()) {
                case arrow::Type::INT64:
                    out.push_back(static_cast<int>(
                        std::static_pointer_cast<arrow::Int64Array>(chunk)->Value(i)));
                    break;
                case arrow::Type::INT32:
                    out.push_back(std::static_pointer_cast<arrow::Int32Array>(chunk)->Value(i));
                    break;
                case arrow::Type::INT16:
                    out.push_back(std::static_pointer_cast<arrow::Int16Array>(chunk)->Value(i));
                    break;
                case arrow::Type::INT8:
                    out.push_back(std::static_pointer_cast<arrow::Int8Array>(chunk)->Value(i));
                    break;
                case arrow::Type::UINT8:
                    out.push_back(std::static_pointer_cast<arrow::UInt8Array>(chunk)->Value(i));
                    break;
                case arrow::Type::UINT16:
                    out.push_back(std::static_pointer_cast<arrow::UInt16Array>(chunk)->Value(i));
                    break;
                case arrow::Type::UINT32:
                    out.push_back(static_cast<int>(
                        std::static_pointer_cast<arrow::UInt32Array>(chunk)->Value(i)));
                    break;
                case arrow::Type::FLOAT:
                    out.push_back(static_cast<int>(
                        std::static_pointer_cast<arrow::FloatArray>(chunk)->Value(i)));
                    break;
                case arrow::Type::DOUBLE:
                    out.push_back(static_cast<int>(
                        std::static_pointer_cast<arrow::DoubleArray>(chunk)->Value(i)));
                    break;
                default:
                    error_type_name = chunk->type()->ToString();
                    return false;
            }
        }
    }
    return true;
}

} // namespace

bool TextTokenizerOperator::Configure(
    const std::map<std::string, std::string>& params,
    std::string& error) {

    auto it = params.find("text_col");
    if (it == params.end() || it->second.empty()) {
        error = "TextTokenizer: 'text_col' parameter is required";
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

    auto lcase = params.find("lowercase");
    lowercase_ = (lcase == params.end()) ? true : (lcase->second == "true");

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

    return true;
}

arrow::Result<std::shared_ptr<arrow::Table>>
TextTokenizerOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    if (!input) {
        return arrow::Status::Invalid("TextTokenizer: input table is null");
    }

    auto text_column = input->GetColumnByName(text_col_);
    if (!text_column) {
        return arrow::Status::KeyError(
            "TextTokenizer: text column '" + text_col_ + "' not found");
    }

    std::vector<std::string> texts;
    std::string bad_type;
    if (!ReadColumnAsStrings(text_column, texts, bad_type)) {
        return arrow::Status::TypeError(
            "TextTokenizer: text column '" + text_col_ +
            "' must be string/large_string, got '" + bad_type + "'");
    }

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

    tokenizer.Train(texts, min_word_freq_, max_vocab_size_);

    // Encode + pad. EncodeBatch then PadBatch produces the final
    // [num_samples, max_length] int matrix.
    auto encoded = tokenizer.EncodeBatch(texts);
    auto padded = tokenizer.PadBatch(encoded, max_length_);

    const size_t n = padded.size();

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
        if (!ReadLabelColumnAsInt(label_column, labels, class_names, lbad)) {
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
        const auto& row = padded[r];
        for (int i = 0; i < max_length_; ++i) {
            const float v = (i < static_cast<int>(row.size()))
                ? static_cast<float>(row[i]) : 0.0f;
            ARROW_RETURN_NOT_OK(tok_builders[i]->Append(v));
        }
        if (!labels.empty()) {
            ARROW_RETURN_NOT_OK(label_builder.Append(labels[r]));
        }
    }

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
                 n, tokenizer.GetVocabulary().Size(), max_length_,
                 class_names.size());
    return out_table;
}

} // namespace cyxwiz
