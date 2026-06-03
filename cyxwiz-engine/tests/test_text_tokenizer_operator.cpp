#include "../src/core/node_executors/text_tokenizer_operator.h"

#include <arrow/api.h>

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

    std::cout << "TextTokenizerOperator Arrow path passed\n";
    return 0;
}
