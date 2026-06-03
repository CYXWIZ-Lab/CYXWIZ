#include "../src/core/node_executors/text_tokenizer_operator.h"

#include <arrow/api.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

std::shared_ptr<arrow::Array> FinishStringArray(
    const std::vector<std::string>& values) {
    arrow::StringBuilder builder;
    for (const auto& value : values) {
        auto st = builder.Append(value);
        Check(st.ok(), st.ToString());
    }
    std::shared_ptr<arrow::Array> array;
    auto st = builder.Finish(&array);
    Check(st.ok(), st.ToString());
    return array;
}

float ReadFloatValue(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& column_name,
    int64_t row) {

    auto column = table->GetColumnByName(column_name);
    Check(column != nullptr, "missing float column " + column_name);
    Check(column->num_chunks() == 1, "expected one float chunk");
    auto array = std::static_pointer_cast<arrow::FloatArray>(column->chunk(0));
    return array->Value(row);
}

} // namespace

int main() {
    auto text = FinishStringArray({
        "Small text sample",
        "Another small sample",
        "Text pipelines should tokenize",
    });
    auto label = FinishStringArray({"positive", "positive", "negative"});

    auto schema = arrow::schema({
        arrow::field("text", arrow::utf8()),
        arrow::field("label", arrow::utf8()),
    });
    auto input = arrow::Table::Make(schema, {text, label}, 3);

    cyxwiz::TextTokenizerOperator op;
    std::map<std::string, std::string> params = {
        {"text_col", "text"},
        {"label_col", "label"},
        {"tokenizer_type", "1"},
        {"max_length", "4"},
        {"lowercase", "true"},
        {"min_word_freq", "1"},
        {"max_vocab_size", "100"},
    };

    std::string error;
    Check(op.Configure(params, error), error);

    auto result = op.Apply(input);
    Check(result.ok(), result.status().ToString());
    Check(op.GetLastVocabSize() > 0,
          "operator should report trained vocabulary size");

    auto output = result.ValueOrDie();
    Check(output != nullptr, "output table is null");
    Check(output->num_rows() == 3, "expected 3 output rows");
    Check(output->num_columns() == 5, "expected 4 token columns plus y");

    for (int i = 0; i < 4; ++i) {
        const std::string name = "tok_" + std::to_string(i);
        auto column = output->GetColumnByName(name);
        Check(column != nullptr, "missing column " + name);
        Check(column->type()->id() == arrow::Type::FLOAT,
              name + " should be float32");
    }

    auto y = output->GetColumnByName("y");
    Check(y != nullptr, "missing y column");
    Check(y->type()->id() == arrow::Type::INT32, "y should be int32");

    const auto vocab_file =
        std::filesystem::temp_directory_path() /
        "cyxwiz_text_tokenizer_operator_vocab.txt";
    {
        std::ofstream out(vocab_file);
        Check(out.good(), "failed to create vocab file");
        out << "[PAD]\n[UNK]\n[BOS]\n[EOS]\nsmall\n";
    }

    cyxwiz::TextTokenizerOperator file_vocab_op;
    params["vocab_file"] = vocab_file.string();
    Check(file_vocab_op.Configure(params, error), error);
    auto file_vocab_result = file_vocab_op.Apply(input);
    Check(file_vocab_result.ok(), file_vocab_result.status().ToString());
    auto file_vocab_output = file_vocab_result.ValueOrDie();
    Check(file_vocab_op.GetLastVocabSize() == 5,
          "operator should report loaded vocabulary size");
    Check(ReadFloatValue(file_vocab_output, "tok_0", 0) == 4.0f,
          "known vocab token should use loaded vocabulary index");
    Check(ReadFloatValue(file_vocab_output, "tok_1", 0) == 1.0f,
          "unknown vocab token should use loaded UNK index");
    std::filesystem::remove(vocab_file);

    std::cout << "TextTokenizerOperator Arrow path passed\n";
    return 0;
}
