#include "text_tokenizer_operator.h"
#include "text_column_utils.h"

#include <cyxwiz/tokenizer.h>

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

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
    last_vocab_size_ = 0;
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
    const size_t trained_vocab_size = tokenizer.GetVocabulary().Size();

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
                 n, trained_vocab_size, max_length_,
                 class_names.size());
    last_vocab_size_ = trained_vocab_size;
    return out_table;
}

} // namespace cyxwiz
