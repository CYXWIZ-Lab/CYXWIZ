#include "../src/core/node_executors/text_tokenizer_operator.h"

#include <arrow/api.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

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

    std::vector<cyxwiz::PipelineOperatorProgress> progress_events;
    op.SetProgressCallback(
        [&](const cyxwiz::PipelineOperatorProgress& event) {
            progress_events.push_back(event);
        });
    auto result = op.Apply(input);
    Check(result.ok(), result.status().ToString());
    Check(!progress_events.empty(),
          "TextTokenizer should emit materialization progress events");
    Check(progress_events.front().stage == "TextTokenizer memory preflight",
          "TextTokenizer first progress event should be memory preflight");
    Check(progress_events.front().status == "running",
          "safe TextTokenizer preflight should stay in running status");
    Check(progress_events.front().memory_risk_level == "safe",
          "safe TextTokenizer preflight should report safe risk");
    Check(progress_events.front().estimated_memory_bytes >
              3ULL * 5ULL * static_cast<uint64_t>(sizeof(float)),
          "TextTokenizer preflight should include peak allocation overhead");
    Check(progress_events.front().message.find("Suggestion:") !=
              std::string::npos,
          "TextTokenizer preflight message should include mitigation guidance");
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

    cyxwiz::TextTokenizerOperator pad_value_op;
    params.erase("vocab_file");
    params["max_length"] = "5";
    params["pad_value"] = "9";
    Check(pad_value_op.Configure(params, error), error);
    auto pad_value_result = pad_value_op.Apply(input);
    Check(pad_value_result.ok(), pad_value_result.status().ToString());
    auto pad_value_output = pad_value_result.ValueOrDie();
    Check(ReadFloatValue(pad_value_output, "tok_3", 0) == 9.0f,
          "custom pad_value should replace tokenizer PAD ids");
    Check(ReadFloatValue(pad_value_output, "tok_4", 0) == 9.0f,
          "custom pad_value should fill all padded positions");

    const auto missing_vocab_file =
        std::filesystem::temp_directory_path() /
        "cyxwiz_text_tokenizer_operator_built_vocab.txt";
    std::filesystem::remove(missing_vocab_file);

    cyxwiz::TextTokenizerOperator strict_missing_op;
    params["vocab_file"] = missing_vocab_file.string();
    params.erase("vocab_build_if_missing");
    Check(strict_missing_op.Configure(params, error), error);
    auto strict_missing_result = strict_missing_op.Apply(input);
    Check(!strict_missing_result.ok(),
          "missing strict vocab_file should fail without build-if-missing");

    cyxwiz::TextTokenizerOperator build_vocab_op;
    params["vocab_build_if_missing"] = "true";
    Check(build_vocab_op.Configure(params, error), error);
    auto build_vocab_result = build_vocab_op.Apply(input);
    Check(build_vocab_result.ok(), build_vocab_result.status().ToString());
    Check(std::filesystem::exists(missing_vocab_file),
          "build-if-missing should write vocab file");
    Check(build_vocab_op.GetLastVocabSize() > 4,
          "built vocabulary should include corpus tokens");
    std::filesystem::remove(missing_vocab_file);

    cyxwiz::TextTokenizerOperator character_op;
    params.erase("vocab_file");
    params.erase("vocab_build_if_missing");
    params.erase("pad_value");
    params["tokenizer_type"] = "2";
    params["max_length"] = "3";
    params["max_vocab_size"] = "20";
    Check(character_op.Configure(params, error), error);
    auto character_result = character_op.Apply(input);
    Check(character_result.ok(), character_result.status().ToString());
    auto character_output = character_result.ValueOrDie();
    Check(character_op.GetLastVocabSize() == 20,
          "character vocabulary should honor total max vocab size including specials");
    Check(ReadFloatValue(character_output, "tok_0", 0) != 1.0f,
          "character tokenizer should train character tokens, not word-only UNKs");

    std::cout << "TextTokenizerOperator Arrow path passed\n";
    return 0;
}
