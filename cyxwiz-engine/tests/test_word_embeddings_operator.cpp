// Word Embeddings node operator (TOFIX152): pretrained vectors lined up with a
// vocabulary, the saved .vec file, the Rows output (each row's mean vector),
// word2vec, GloVe and fastText training on the text column, the saved vectors
// reused while nothing changed, a fastText .bin, and the refusals.

#include "../src/core/node_executors/word_embeddings_operator.h"
#include "../src/core/fasttext_model_store.h"
#include "../src/core/word_vectors_training_record.h"
#include "../src/core/word_embeddings_presentation.h"

#include <cyxwiz/word_embeddings.h>

#include <arrow/api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
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
        {"source", "word2vec"}, {"text_col", "text"}, {"vector_size", "8"}, {"window", "2"},
        {"min_count", "2"}, {"epochs", "3"}, {"train_seed", "4"}, {"architecture", "cbow"}};
    cyxwiz::WordEmbeddingsOperator first, second;
    std::vector<cyxwiz::OperatorFigures> figures;
    first.SetProgressCallback([&](const cyxwiz::PipelineOperatorProgress& event) {
        if (!event.figures.empty()) figures.push_back(event.Figures());
    });
    std::string error;
    Check(first.Configure(params, error) && second.Configure(params, error), "configures training: " + error);
    auto a = first.Apply(TextTable(texts));
    auto b = second.Apply(TextTable(texts));
    Check(a.ok() && b.ok(), "trains: " + (a.ok() ? b.status().ToString() : a.status().ToString()));
    Check((*a)->num_rows() == 6 && (*a)->num_columns() == 9, "6 words x 8 values");
    for (int64_t r = 0; r < 6; ++r)
        Check(Value(*a, "v_3", r) == Value(*b, "v_3", r), "a training run repeats for one seed");
    Check(first.OutcomeSummary().find("CBOW") != std::string::npos, "outcome names the model: " + first.OutcomeSummary());
    // The Training Dashboard's figures: the settings, the vocabulary, each
    // epoch's loss and examples, then the end.
    Check(!figures.empty() && figures.front().Note("method") == "word2vec CBOW" &&
              figures.front().Figure("dims") == 8 && figures.front().Figure("window") == 2 &&
              figures.front().Figure("epochs") == 3 && figures.front().Figure("min_count") == 2,
          "the first figures name the run's settings");
    std::vector<double> losses;
    double last_elapsed = -1.0;
    bool words = false;
    for (const auto& event : figures) {
        Check(event.Figure("elapsed") >= last_elapsed, "every figure carries the seconds so far, in order");
        last_elapsed = event.Figure("elapsed");
        words = words || event.Figure("words") == 6;
        if (!event.Has("epoch")) continue;
        Check(event.Figure("epoch") == static_cast<double>(losses.size() + 1) && event.Figure("examples") > 0,
              "epochs in order, with their examples");
        losses.push_back(event.Figure("loss"));
    }
    Check(words && losses.size() == 3 && losses[0] > 0.0, "the vocabulary and the loss of each epoch");
    char last[32];
    std::snprintf(last, sizeof(last), "%.4g)", losses.back());
    Check(first.OutcomeSummary().find(last) != std::string::npos,
          "the last loss is the outcome's: " + first.OutcomeSummary());
    Check(figures.back().stage == "Done", "the last figure ends the run");
}

// A second run with the same text, settings and vocabulary reads the saved
// vectors; a change to any of them, or to the saved file, trains again.
void CheckTrainingReuse(const fs::path& dir) {
    std::vector<std::string> texts;
    for (int i = 0; i < 300; ++i) texts.push_back(i % 2 ? "cat dog cat mouse dog cat" : "red blue green red blue red");
    const fs::path vocab = dir / "reuse_vocab.txt";
    std::ofstream(vocab) << "[PAD]\n[UNK]\ncat\ndog\nred\nblue\n";
    const fs::path saved = dir / "reuse.vec";
    std::map<std::string, std::string> params = {
        {"source", "word2vec"}, {"text_col", "text"}, {"vector_size", "4"},
        {"window", "2"},        {"min_count", "2"},   {"epochs", "2"},
        {"vocab_file", vocab.string()},               {"vectors_path", saved.string()}};
    // Runs the step; true when it trained (published figures).
    const auto run = [&](const std::vector<std::string>& rows, std::shared_ptr<arrow::Table>* table = nullptr,
                         std::string* outcome = nullptr) {
        cyxwiz::WordEmbeddingsOperator op;
        size_t figures = 0;
        op.SetProgressCallback([&](const cyxwiz::PipelineOperatorProgress& event) {
            if (!event.figures.empty()) ++figures;
        });
        std::string error;
        Check(op.Configure(params, error), "configures: " + error);
        auto result = op.Apply(TextTable(rows));
        Check(result.ok(), "runs: " + result.status().ToString());
        if (table) *table = *result;
        if (outcome) *outcome = op.OutcomeSummary();
        Check((figures > 0) != cyxwiz::ReusedVectorsOutcome(op.OutcomeSummary()),
              "a trained run publishes figures, a reused one none: " + op.OutcomeSummary());
        return figures > 0;
    };
    Check(!cyxwiz::ReadWordVectorsTrainingRecord(saved.string()), "no record before training");
    std::shared_ptr<arrow::Table> trained, reused;
    std::string first, second;
    Check(run(texts, &trained, &first), "the first run trains");
    const auto record = cyxwiz::ReadWordVectorsTrainingRecord(saved.string());
    Check(record && record->text_sha256.size() == 64 && record->vocab_sha256.size() == 64 &&
              record->settings.find("epochs=2\n") != std::string::npos &&
              record->trained_at.size() == 16,
          "the record beside the saved file");
    Check(!run(texts, &reused, &second), "the same text, settings and vocabulary reuse the saved vectors");
    Check(second.find("unchanged since " + record->trained_at) != std::string::npos &&
              second.find(record->summary) != std::string::npos &&
              second.find("vocabulary 6 tokens") != std::string::npos,
          "the outcome names the reuse, the training and the line-up: " + second);
    Check(reused->num_rows() == trained->num_rows() && reused->num_columns() == trained->num_columns(),
          "the same table");
    for (int64_t r = 0; r < trained->num_rows(); ++r) {
        Check(Word(reused, r) == Word(trained, r), "the same words");
        for (int d = 0; d < 4; ++d) {
            const std::string column = "v_" + std::to_string(d);
            Check(Value(reused, column, r) == Value(trained, column, r), "the saved values are exact");
        }
    }
    params["row_vectors"] = "true";
    Check(!run(texts), "the Rows output does not change the vectors");
    params["negative"] = "5";
    params["glove_window"] = "7";
    Check(!run(texts), "a default written out and GloVe's settings do not change word2vec vectors");
    params["epochs"] = "3";
    Check(run(texts), "another setting trains again");
    Check(!run(texts), "and is reused on the next run");
    auto changed = texts;
    changed.push_back("cat dog red");
    Check(run(changed), "changed text trains again");
    std::ofstream(vocab, std::ios::app) << "mouse\n";
    Check(run(changed), "a changed vocabulary file trains again");
    std::ofstream(saved, std::ios::app) << "\n";
    Check(!cyxwiz::ReadWordVectorsTrainingRecord(saved.string()), "a replaced vectors file has no record");
    Check(run(changed), "and trains again");
}

// A trained run lined up with a vocabulary and saved: the line-up's figures.
void CheckTrainingLineUpFigures(const fs::path& dir) {
    std::vector<std::string> texts;
    for (int i = 0; i < 300; ++i) texts.push_back(i % 2 ? "cat dog cat mouse dog cat" : "red blue green red blue red");
    const fs::path vocab = dir / "line_up_vocab.txt";
    std::ofstream(vocab) << "[PAD]\n[UNK]\ncat\nDog\nzebra\n";
    const fs::path saved = dir / "line_up.vec";
    cyxwiz::WordEmbeddingsOperator op;
    std::vector<cyxwiz::OperatorFigures> figures;
    op.SetProgressCallback([&](const cyxwiz::PipelineOperatorProgress& event) {
        if (!event.figures.empty()) figures.push_back(event.Figures());
    });
    std::string error;
    Check(op.Configure({{"source", "word2vec"}, {"text_col", "text"}, {"vector_size", "4"}, {"window", "2"},
                        {"min_count", "2"}, {"epochs", "1"}, {"vocab_file", vocab.string()},
                        {"vectors_path", saved.string()}},
                       error),
          "configures a lined-up run: " + error);
    Check(op.Apply(TextTable(texts)).ok(), "trains and lines up");
    Check(figures.front().Note("saved") == saved.string() && figures.front().Note("vocabulary_file") == vocab.string(),
          "the first figures name the saved file and the vocabulary");
    const auto line_up = std::find_if(figures.begin(), figures.end(),
                                      [](const cyxwiz::OperatorFigures& event) { return event.Has("tokens"); });
    Check(line_up != figures.end() && line_up->Figure("tokens") == 5 && line_up->Figure("found") == 1 &&
              line_up->Figure("lowercase") == 1 && line_up->Figure("missing") == 1,
          "5 tokens: cat found, Dog in lowercase, zebra missing");
    Check(line_up->Note("special") == "[PAD] zeros \xC2\xB7 [UNK] mean", "special tokens: " + line_up->Note("special"));
    Check(figures.back().stage == "Done" && figures.back().message == "Saved the vectors" && fs::exists(saved),
          "the run ends saved");
}

void CheckGloVeTraining() {
    std::vector<std::string> texts;
    for (int i = 0; i < 300; ++i) {
        texts.push_back(i % 2 ? "cat dog cat mouse dog cat" : "red blue green red blue red");
    }
    const std::map<std::string, std::string> params = {
        {"source", "glove"}, {"text_col", "text"}, {"vector_size", "8"}, {"glove_window", "3"},
        {"min_count", "2"}, {"glove_epochs", "4"}, {"train_seed", "4"}, {"x_max", "10"}, {"glove_batch", "16"},
        // word2vec's settings play no part.
        {"window", "1"}, {"epochs", "1"}};
    cyxwiz::WordEmbeddingsOperator first, second;
    std::vector<std::string> stages;
    std::vector<cyxwiz::OperatorFigures> figures;
    first.SetProgressCallback([&](const cyxwiz::PipelineOperatorProgress& event) {
        if (stages.empty() || stages.back() != event.stage) stages.push_back(event.stage);
        if (!event.figures.empty()) figures.push_back(event.Figures());
    });
    std::string error;
    Check(first.Configure(params, error) && second.Configure(params, error), "configures GloVe: " + error);
    auto a = first.Apply(TextTable(texts));
    auto b = second.Apply(TextTable(texts));
    Check(a.ok() && b.ok(), "trains GloVe: " + (a.ok() ? b.status().ToString() : a.status().ToString()));
    Check((*a)->num_rows() == 6 && (*a)->num_columns() == 9, "GloVe: 6 words x 8 values");
    for (int64_t r = 0; r < 6; ++r)
        Check(Value(*a, "v_3", r) == Value(*b, "v_3", r), "a GloVe run repeats for one seed");
    const auto counting = std::find(stages.begin(), stages.end(), "Counting");
    Check(counting != stages.end() && std::find(counting, stages.end(), "Training") != stages.end(),
          "GloVe reports counting, then training");
    const std::string outcome = first.OutcomeSummary();
    Check(outcome.find("GloVe") != std::string::npos && outcome.find("word pairs") != std::string::npos,
          "outcome names GloVe and its pairs: " + outcome);
    std::vector<double> costs;
    double pairs = -1.0;
    for (const auto& event : figures) {
        if (event.Has("pairs")) pairs = event.Figure("pairs");
        if (event.Has("epoch")) costs.push_back(event.Figure("cost"));
    }
    Check(figures.front().Note("method") == "GloVe" && figures.front().Figure("window") == 3 &&
              figures.front().Figure("epochs") == 4,
          "the GloVe run's settings");
    Check(pairs > 0.0 && outcome.find(cyxwiz::GroupedCount(static_cast<size_t>(pairs)) + " word pairs") !=
                             std::string::npos,
          "the counted pairs are the outcome's: " + outcome);
    Check(costs.size() == 4 && costs.back() < costs.front(), "the cost of each epoch falls: " + outcome);
    Check(figures.back().stage == "Done" && !figures.back().Has("tokens"), "no vocabulary file, no line-up figures");
    cyxwiz::WordEmbeddingsOperator bad;
    Check(!bad.Configure({{"source", "glove"}, {"text_col", "text"}, {"x_max", "0"}}, error) &&
              error.find("x_max") != std::string::npos,
          "a zero count cap is refused");
}

// fastText trained on the text and lined up: a token the text lacks is built
// from its pieces (or missing with unseen_words = missing); the model is kept
// for the dialog's unseen words.
void CheckFastTextTraining(const fs::path& dir) {
    std::vector<std::string> texts;
    for (int i = 0; i < 300; ++i) texts.push_back(i % 2 ? "cat dog cat mouse dog cat" : "red blue green red blue red");
    const fs::path vocab = dir / "fasttext_vocab.txt";
    std::ofstream(vocab) << "[PAD]\n[UNK]\ncat\nmouse\nmice\n!!\n";
    const fs::path saved = dir / "fasttext.vec";
    const auto params = [&](const std::string& unseen) {
        return std::map<std::string, std::string>{
            {"source", "fasttext"}, {"text_col", "text"}, {"vector_size", "8"}, {"window", "2"},
            {"min_count", "2"}, {"epochs", "2"}, {"bucket", "200"}, {"min_n", "3"}, {"max_n", "4"},
            {"vocab_file", vocab.string()}, {"vectors_path", saved.string()}, {"unseen_words", unseen}};
    };
    cyxwiz::WordEmbeddingsOperator op;
    std::vector<cyxwiz::OperatorFigures> figures;
    op.SetProgressCallback([&](const cyxwiz::PipelineOperatorProgress& event) {
        if (!event.figures.empty()) figures.push_back(event.Figures());
    });
    std::string error;
    Check(op.Configure(params("pieces"), error), "configures fastText: " + error);
    auto result = op.Apply(TextTable(texts));
    Check(result.ok(), "trains fastText: " + result.status().ToString());
    Check((*result)->num_rows() == 6 && (*result)->num_columns() == 9, "fastText: 6 tokens x 8 values");
    Check(figures.front().Note("method") == "fastText skip-gram" && figures.front().Figure("bucket") == 200 &&
              figures.front().Figure("min_n") == 3 && figures.front().Figure("max_n") == 4,
          "fastText's first figures name its settings");
    const auto per_word = std::find_if(figures.begin(), figures.end(), [](const cyxwiz::OperatorFigures& event) {
        return event.Has("pieces_per_word");
    });
    Check(per_word != figures.end() && per_word->Figure("pieces_per_word") > 1.0, "pieces per word reported");
    const auto line_up = std::find_if(figures.begin(), figures.end(),
                                      [](const cyxwiz::OperatorFigures& event) { return event.Has("tokens"); });
    Check(line_up != figures.end() && line_up->Figure("found") == 2 && line_up->Figure("pieces") == 1 &&
              line_up->Figure("missing") == 1,
          "cat and mouse found, mice built from pieces, !! missing");
    Check(op.OutcomeSummary().find("fastText skip-gram") != std::string::npos &&
              op.OutcomeSummary().find("1 from pieces") != std::string::npos,
          "outcome names fastText and the pieces: " + op.OutcomeSummary());
    // The row of "mice" (id 4) is not zeros; "!!" (id 5) is.
    float mice = 0.0f, bangs = 0.0f;
    for (int d = 0; d < 8; ++d) {
        mice += std::abs(Value(*result, "v_" + std::to_string(d), 4));
        bangs += std::abs(Value(*result, "v_" + std::to_string(d), 5));
    }
    Check(mice > 0.0f && bangs == 0.0f, "mice has a vector from its pieces, !! zeros");
    const auto kept = cyxwiz::FindFastTextModel(saved.string());
    Check(kept && kept->bucket == 200 && kept->words.size() == 6, "the model is kept for the dialog");

    cyxwiz::WordEmbeddingsOperator missing_op;
    figures.clear();
    missing_op.SetProgressCallback([&](const cyxwiz::PipelineOperatorProgress& event) {
        if (!event.figures.empty()) figures.push_back(event.Figures());
    });
    Check(missing_op.Configure(params("missing"), error) && missing_op.Apply(TextTable(texts)).ok(),
          "fastText with unseen words as missing: " + error);
    const auto as_missing = std::find_if(figures.begin(), figures.end(),
                                         [](const cyxwiz::OperatorFigures& event) { return event.Has("tokens"); });
    Check(as_missing->Figure("pieces") == 0 && as_missing->Figure("missing") == 2, "mice missing with the rule");
    Check(!missing_op.Configure({{"source", "fasttext"}, {"text_col", "text"}, {"min_n", "5"}, {"max_n", "4"}},
                                error) &&
              error.find("max_n") != std::string::npos,
          "max_n below min_n is refused");
}

// A fastText .bin read for a vocabulary: only its words' rows, the rest from
// pieces.
void CheckFastTextBin(const fs::path& dir, const fs::path& model) {
    const fs::path vocab = dir / "fasttext_bin_vocab.txt";
    std::ofstream(vocab) << "[PAD]\n[UNK]\nsleep\nWorry\nsleepyx\n!!\n";
    cyxwiz::WordEmbeddingsOperator op;
    std::string error;
    Check(op.Configure({{"vectors_file", model.string()}, {"vocab_file", vocab.string()}}, error),
          "configures a fastText .bin: " + error);
    auto result = op.Apply(TextTable({"x"}));
    Check(result.ok(), "reads a fastText .bin: " + result.status().ToString());
    Check((*result)->num_rows() == 6 && (*result)->num_columns() == 7, "6 tokens x 6 values");
    Check(op.OutcomeSummary().find("fastText model") != std::string::npos &&
              op.OutcomeSummary().find("1 found, 1 in lowercase, 1 from pieces, 1 missing") != std::string::npos,
          "sleep, Worry in lowercase, sleepyx from pieces, !! missing: " + op.OutcomeSummary());
}

void CheckRefusals() {
    cyxwiz::WordEmbeddingsOperator op;
    std::string error;
    Check(!op.Configure({}, error) && error.find("vector file") != std::string::npos, "pretrained needs a file");
    Check(!op.Configure({{"source", "word2vec"}}, error) && error.find("text column") != std::string::npos,
          "training needs the text column");
    Check(!op.Configure({{"source", "glove"}}, error) && error.find("text column") != std::string::npos,
          "GloVe needs the text column");
    Check(!op.Configure({{"source", "train"}}, error), "an unknown source is refused");
    Check(!op.Configure({{"vectors_file", "x.vec"}, {"row_vectors", "true"}}, error), "row vectors need the text");
    Check(op.Configure({{"vectors_file", "absent.vec"}}, error), "configures");
    Check(!op.Apply(TextTable({"x"})).ok(), "a missing file is refused at run time");
    Check(op.Configure({{"vectors_file", "absent.vec"}, {"text_col", "nope"}}, error), "configures");
    Check(!op.Apply(TextTable({"x"})).ok(), "a missing text column is refused");
}

}  // namespace

int main(int, char** argv) {
    // The gensim fastText .bin: beside the test (a copied build) or in the source tree.
    fs::path model = fs::path(argv[0]).parent_path() / "small_fasttext.bin";
    if (!fs::exists(model)) model = CYXWIZ_FASTTEXT_FIXTURE;
    const fs::path dir = fs::temp_directory_path() / "cyxwiz_word_embeddings_operator";
    fs::remove_all(dir);  // a failed earlier run's files (a saved record would be reused)
    fs::create_directories(dir);
    CheckPretrainedAligned(dir);
    CheckPretrainedWithoutVocabulary(dir);
    CheckRowVectorBlocks(dir);
    CheckTraining();
    CheckTrainingLineUpFigures(dir);
    CheckTrainingReuse(dir);
    CheckGloVeTraining();
    CheckFastTextTraining(dir);
    CheckFastTextBin(dir, model);
    CheckRefusals();
    fs::remove_all(dir);
    std::cout << "word embeddings operator tests passed\n";
    return 0;
}
