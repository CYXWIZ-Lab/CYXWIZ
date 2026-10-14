// Word Embeddings node operator (TOFIX152): pretrained vectors lined up with a
// vocabulary, the saved .vec file, the Rows output (each row's mean vector),
// word2vec training on the text column, and the refusals.

#include "../src/core/node_executors/word_embeddings_operator.h"

#include <cyxwiz/word_embeddings.h>

#include <arrow/api.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

std::shared_ptr<arrow::Table> TextTable(const std::vector<std::string>& texts) {
    arrow::StringBuilder builder;
    Check(builder.AppendValues(texts).ok(), "append texts");
    std::shared_ptr<arrow::Array> array;
    Check(builder.Finish(&array).ok(), "finish texts");
    return arrow::Table::Make(arrow::schema({arrow::field("text", arrow::utf8())}), {array});
}

float Value(const std::shared_ptr<arrow::Table>& table, const std::string& column, int64_t row) {
    const auto chunked = table->GetColumnByName(column);
    Check(chunked != nullptr, "column " + column);
    return std::static_pointer_cast<arrow::FloatArray>(chunked->chunk(0))->Value(row);
}

std::string Word(const std::shared_ptr<arrow::Table>& table, int64_t row) {
    return std::static_pointer_cast<arrow::StringArray>(table->GetColumnByName("word")->chunk(0))->GetString(row);
}

// A small .vec file: three words, dim 2.
fs::path WriteVectors(const fs::path& dir) {
    const fs::path path = dir / "small.vec";
    std::ofstream(path) << "3 2\nthe 1 2\nparis 3 5\nlondon -1 0.5\n";
    return path;
}

fs::path WriteVocabulary(const fs::path& dir) {
    const fs::path path = dir / "vocab.txt";
    std::ofstream(path) << "[PAD]\n[UNK]\n[BOS]\n[EOS]\nthe\nParis\nunicorn\n";
    return path;
}

void CheckPretrainedAligned(const fs::path& dir) {
    cyxwiz::WordEmbeddingsOperator op;
    std::string error;
    // A folder that does not exist yet, like the run's artifacts folder.
    const fs::path saved = dir / "out" / "run" / "aligned.vec";
    fs::remove_all(dir / "out");
    Check(op.Configure({{"source", "pretrained"}, {"vectors_file", WriteVectors(dir).string()},
                        {"vocab_file", WriteVocabulary(dir).string()}, {"vectors_path", saved.string()},
                        {"text_col", "text"}, {"row_vectors", "true"}},
                       error),
          "configures pretrained: " + error);
    auto result = op.Apply(TextTable({"The Paris!", "unicorn", "", "the the"}));
    Check(result.ok(), "applies pretrained: " + result.status().ToString());
    const auto table = *result;
    Check(table->num_rows() == 7 && table->num_columns() == 3, "one row per vocabulary token, word + 2 values");
    Check(Word(table, 0) == "[PAD]" && Word(table, 4) == "the" && Word(table, 5) == "Paris", "vocabulary order");
    Check(Value(table, "v_0", 0) == 0.0f && Value(table, "v_1", 0) == 0.0f, "PAD is zeros");
    Check(std::abs(Value(table, "v_0", 1) - 1.0f) < 1e-6f && std::abs(Value(table, "v_1", 1) - 2.5f) < 1e-6f,
          "UNK is the mean vector");
    Check(Value(table, "v_0", 4) == 1.0f && Value(table, "v_1", 5) == 5.0f, "the / Paris (lowercase match) rows");
    Check(Value(table, "v_0", 6) == 0.0f, "a missing word is zeros");
    Check(op.OutcomeSummary().find("1 found, 1 in lowercase, 3 missing") != std::string::npos,
          "coverage counts: " + op.OutcomeSummary());

    const auto saved_vectors = cyxwiz::WordEmbeddings::Load(saved.string());
    Check(saved_vectors.success && saved_vectors.words.size() == 7 && saved_vectors.dim == 2 &&
              saved_vectors.values[5 * 2 + 1] == 5.0f,
          "the saved .vec holds the aligned table");

    // Rows: the input plus each row's mean word vector (by the table's rows).
    const auto extras = op.ExtraOutputs();
    Check(extras.size() == 1 && extras[0] != nullptr, "row vectors on pin 1");
    const auto rows = extras[0];
    Check(rows->num_rows() == 4 && rows->GetColumnByName("text") && rows->GetColumnByName("emb_1"),
          "rows keep the input and add emb_ columns");
    // "the paris": tokens the (row 4) and paris -> not in the aligned words
    // ("Paris" is), so only "the" counts: mean (1, 2).
    Check(Value(rows, "emb_0", 0) == 1.0f && Value(rows, "emb_1", 0) == 2.0f, "row 0 mean");
    Check(Value(rows, "emb_0", 1) == 0.0f, "unicorn has a zero row");
    Check(Value(rows, "emb_0", 2) == 0.0f && Value(rows, "emb_1", 2) == 0.0f, "an empty row is zeros");
    Check(Value(rows, "emb_0", 3) == 1.0f, "repeated words average to themselves");
}

void CheckPretrainedWithoutVocabulary(const fs::path& dir) {
    cyxwiz::WordEmbeddingsOperator op;
    std::string error;
    Check(op.Configure({{"vectors_file", WriteVectors(dir).string()}, {"max_words", "2"}}, error), error);
    auto result = op.Apply(TextTable({"x"}));
    Check(result.ok() && (*result)->num_rows() == 2 && Word(*result, 1) == "paris", "the file's own words, max_words");
    Check(op.ExtraOutputs().size() == 1 && op.ExtraOutputs()[0] == nullptr, "no Rows without row vectors");
}

// Long rows of wide vectors: the rows are averaged in several device blocks
// (each within the operator's budget), and each row still gets its own mean.
void CheckRowVectorBlocks(const fs::path& dir) {
    const int dim = 1000;
    const fs::path path = dir / "wide.vec";
    {
        std::ofstream out(path);
        out << "3 " << dim << "\n";
        for (const char* word : {"the", "paris", "london"}) {
            out << word;
            for (int d = 0; d < dim; ++d) out << ' ' << (word[0] == 't' ? 1 : word[0] == 'p' ? 3 : -1) * (d % 7 + 1);
            out << "\n";
        }
    }
    std::string the, london;
    for (int i = 0; i < 20000; ++i) the += "the ";
    for (int i = 0; i < 30000; ++i) london += "london ";
    cyxwiz::WordEmbeddingsOperator op;
    std::string error;
    Check(op.Configure({{"vectors_file", path.string()}, {"text_col", "text"}, {"row_vectors", "true"}}, error), error);
    auto result = op.Apply(TextTable({the, "paris", london, "the paris unknown"}));
    Check(result.ok(), "averages long rows: " + result.status().ToString());
    const auto rows = op.ExtraOutputs()[0];
    for (const int d : {0, 3, 999}) {
        const float unit = static_cast<float>(d % 7 + 1);
        const std::string column = "emb_" + std::to_string(d);
        // A mean may be computed as sum x (1 / count): within float rounding.
        const auto close_to = [](float got, float want) { return std::abs(got - want) <= 1e-6f * std::abs(want); };
        Check(close_to(Value(rows, column, 0), unit) && close_to(Value(rows, column, 1), 3.0f * unit) &&
                  close_to(Value(rows, column, 2), -unit) && close_to(Value(rows, column, 3), 2.0f * unit),
              "each row's mean, whatever block it was in (" + column + ")");
    }
}

void CheckTraining() {
    std::vector<std::string> texts;
    for (int i = 0; i < 300; ++i) {
        texts.push_back(i % 2 ? "cat dog cat mouse dog cat" : "red blue green red blue red");
    }
    const std::map<std::string, std::string> params = {
        {"source", "train"}, {"text_col", "text"}, {"vector_size", "8"}, {"window", "2"},
        {"min_count", "2"}, {"epochs", "3"}, {"train_seed", "4"}, {"architecture", "cbow"}};
    cyxwiz::WordEmbeddingsOperator first, second;
    std::string error;
    Check(first.Configure(params, error) && second.Configure(params, error), "configures training: " + error);
    auto a = first.Apply(TextTable(texts));
    auto b = second.Apply(TextTable(texts));
    Check(a.ok() && b.ok(), "trains: " + (a.ok() ? b.status().ToString() : a.status().ToString()));
    Check((*a)->num_rows() == 6 && (*a)->num_columns() == 9, "6 words x 8 values");
    for (int64_t r = 0; r < 6; ++r)
        Check(Value(*a, "v_3", r) == Value(*b, "v_3", r), "a training run repeats for one seed");
    Check(first.OutcomeSummary().find("CBOW") != std::string::npos, "outcome names the model: " + first.OutcomeSummary());
    const auto losses = cyxwiz::WordEmbeddingsOperator::EpochLosses(first.OutcomeSummary());
    Check(losses.size() == 3 && losses[0] > 0.0, "the outcome carries the loss of each epoch: " + first.OutcomeSummary());
    Check(cyxwiz::WordEmbeddingsOperator::EpochLosses("read 3 words x 2").empty(), "no losses for pretrained vectors");
}

void CheckRefusals() {
    cyxwiz::WordEmbeddingsOperator op;
    std::string error;
    Check(!op.Configure({}, error) && error.find("vector file") != std::string::npos, "pretrained needs a file");
    Check(!op.Configure({{"source", "train"}}, error) && error.find("text column") != std::string::npos,
          "training needs the text column");
    Check(!op.Configure({{"source", "glove"}}, error), "an unknown source is refused");
    Check(!op.Configure({{"vectors_file", "x.vec"}, {"row_vectors", "true"}}, error), "row vectors need the text");
    Check(op.Configure({{"vectors_file", "absent.vec"}}, error), "configures");
    Check(!op.Apply(TextTable({"x"})).ok(), "a missing file is refused at run time");
    Check(op.Configure({{"vectors_file", "absent.vec"}, {"text_col", "nope"}}, error), "configures");
    Check(!op.Apply(TextTable({"x"})).ok(), "a missing text column is refused");
}

}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / "cyxwiz_word_embeddings_operator";
    fs::create_directories(dir);
    CheckPretrainedAligned(dir);
    CheckPretrainedWithoutVocabulary(dir);
    CheckRowVectorBlocks(dir);
    CheckTraining();
    CheckRefusals();
    fs::remove_all(dir);
    std::cout << "word embeddings operator tests passed\n";
    return 0;
}
