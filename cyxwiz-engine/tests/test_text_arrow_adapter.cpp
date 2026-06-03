#include "../src/core/formats/text_dataset.h"
#include "../src/core/text_arrow_adapter.h"
#include "../src/core/node_executors/text_tokenizer_operator.h"

#include <arrow/api.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

void WriteFile(const std::filesystem::path& path, const std::string& content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    Check(out.good(), "failed to open " + path.string());
    out << content;
}

std::shared_ptr<arrow::Table> BuildTableOrFail(
    const cyxwiz::TextDataset& dataset,
    const std::string& text_col,
    const std::string& label_col) {
    auto result = cyxwiz::BuildRawTextArrowTable(dataset, text_col, label_col);
    Check(result.ok(), result.status().ToString());
    auto table = result.ValueOrDie();
    Check(table != nullptr, "adapter returned null table");
    return table;
}

void CheckTextValue(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& column_name,
    int64_t row,
    const std::string& expected) {
    auto column = table->GetColumnByName(column_name);
    Check(column != nullptr, "missing text column " + column_name);
    Check(column->num_chunks() == 1, "expected one text chunk");
    auto arr = std::static_pointer_cast<arrow::StringArray>(column->chunk(0));
    Check(arr->GetString(row) == expected,
          "unexpected text value in " + column_name);
}

int32_t ReadInt32(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& column_name,
    int64_t row) {
    auto column = table->GetColumnByName(column_name);
    Check(column != nullptr, "missing int32 column " + column_name);
    Check(column->num_chunks() == 1, "expected one int32 chunk");
    auto arr = std::static_pointer_cast<arrow::Int32Array>(column->chunk(0));
    return arr->Value(row);
}

} // namespace

int main() {
    const auto root =
        std::filesystem::temp_directory_path() /
        "cyxwiz_text_arrow_adapter_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);

    {
        const auto jsonl = root / "samples.jsonl";
        WriteFile(
            jsonl,
            "{\"body\":\"json positive\",\"label\":\"pos\"}\n"
            "{\"body\":\"json negative\",\"label\":\"neg\"}\n");

        cyxwiz::TextDatasetConfig cfg;
        cfg.text_column = "body";
        cfg.label_column = "label";
        cfg.has_labels = true;
        cfg.max_length = 4;
        cyxwiz::TextDataset dataset(jsonl.string(), cfg);

        auto table = BuildTableOrFail(dataset, "body", "label");
        Check(table->num_rows() == 2, "JSONL table row count");
        Check(table->num_columns() == 2, "JSONL table column count");
        CheckTextValue(table, "body", 0, "json positive");
        Check(ReadInt32(table, "label", 0) == 0, "JSONL label 0");
        Check(ReadInt32(table, "label", 1) == 1, "JSONL label 1");
    }

    {
        const auto txt = root / "lines.txt";
        WriteFile(txt, "first plain line\n\nsecond plain line\n");

        cyxwiz::TextDatasetConfig cfg;
        cfg.text_column = "text";
        cfg.label_column = "label";
        cfg.has_labels = false;
        cfg.max_length = 4;
        cyxwiz::TextDataset dataset(txt.string(), cfg);

        auto table = BuildTableOrFail(dataset, "text", "label");
        Check(table->num_rows() == 2, "TXT table row count");
        Check(table->num_columns() == 1, "TXT table should be unlabeled");
        CheckTextValue(table, "text", 0, "first plain line");
        Check(table->GetColumnByName("label") == nullptr,
              "TXT table should not emit label column");
    }

    {
        const auto corpus = root / "corpus";
        WriteFile(corpus / "neg" / "a.txt", "folder negative");
        WriteFile(corpus / "pos" / "b.txt", "folder positive");

        cyxwiz::TextDatasetConfig cfg;
        cfg.text_column = "text";
        cfg.label_column = "label";
        cfg.max_length = 4;
        cyxwiz::TextDataset dataset(corpus.string(), cfg);

        auto table = BuildTableOrFail(dataset, "text", "label");
        Check(table->num_rows() == 2, "folder table row count");
        Check(table->num_columns() == 2, "folder table column count");
        Check(ReadInt32(table, "label", 0) == 0, "folder label 0");
        Check(ReadInt32(table, "label", 1) == 1, "folder label 1");

        cyxwiz::TextTokenizerOperator tokenizer;
        std::string error;
        Check(tokenizer.Configure({
            {"text_col", "text"},
            {"label_col", "label"},
            {"tokenizer_type", "1"},
            {"max_length", "4"},
            {"lowercase", "true"},
            {"min_word_freq", "1"},
            {"max_vocab_size", "100"},
        }, error), error);
        auto tokenized = tokenizer.Apply(table);
        Check(tokenized.ok(), tokenized.status().ToString());
        auto out = tokenized.ValueOrDie();
        Check(out != nullptr, "tokenizer returned null table");
        Check(out->GetColumnByName("tok_0") != nullptr, "missing tok_0");
        Check(out->GetColumnByName("y") != nullptr, "missing y");
        Check(out->num_columns() == 5, "expected 4 token columns plus y");
    }

    std::filesystem::remove_all(root);
    std::cout << "Text Arrow adapter tests passed\n";
    return 0;
}
