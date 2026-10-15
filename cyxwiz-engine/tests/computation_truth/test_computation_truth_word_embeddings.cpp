// Word vectors (TOFIX152):
// - files and vocabulary alignment against gensim 4: fixtures/word_embeddings_gensim.json
//   and fixtures/word_embeddings/ (reference/generate_word_embeddings_fixtures.py);
// - word2vec steps against PyTorch and training quality against gensim:
//   fixtures/word2vec_reference.json (reference/generate_word2vec_fixtures.py);
// - GloVe counts and steps against numpy / PyTorch and training quality against the reference GloVe:
//   fixtures/glove_reference.json (reference/generate_glove_fixtures.py);
// - fastText hash, pieces, steps and vectors against gensim / PyTorch and training quality against gensim:
//   fixtures/fasttext_reference.json (reference/generate_fasttext_fixtures.py). Runs on the active ArrayFire
// backend (CYXWIZ_TEST_ARRAYFIRE_BACKEND=cuda|opencl|oneapi|cpu).
#include "test_device_selection.h"

#include <cyxwiz/word_embeddings.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;
using cyxwiz::WordEmbeddings;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

// Text files hold decimal strings: gensim parses them to float64 then float32,
// we parse straight to float32, so they may differ by one float32 step.
void CheckValues(const std::vector<float>& actual, const json& expected, double tolerance, const std::string& what) {
    const auto values = expected.get<std::vector<double>>();
    Check(actual.size() == values.size(),
          what + ": " + std::to_string(actual.size()) + " values, expected " + std::to_string(values.size()));
    for (size_t i = 0; i < values.size(); ++i)
        Check(std::abs(actual[i] - values[i]) <= tolerance * (1.0 + std::abs(values[i])),
              what + " [" + std::to_string(i) + "]: " + std::to_string(actual[i]) + ", expected " +
                  std::to_string(values[i]));
}

fs::path FixtureRoot(const char* argv0) {
    const auto beside = fs::path(argv0).parent_path() / "computation_truth_fixtures";
    if (fs::exists(beside / "word_embeddings_gensim.json")) return beside;
    return fs::path(CYXWIZ_WORD_EMBEDDINGS_FIXTURE_DIR);
}

void RunLoad(const json& c, const fs::path& root) {
    const std::string name = "load " + c.at("file").get<std::string>() + " max " + c.at("max_words").dump();
    const auto vectors = WordEmbeddings::Load((root / "word_embeddings" / c.at("file").get<std::string>()).string(),
                                              c.at("max_words").get<size_t>());
    Check(vectors.success, name + ": " + vectors.error_message);
    Check(vectors.words == c.at("words").get<std::vector<std::string>>(), name + ": words differ from gensim");
    Check(vectors.dim == c.at("dim").get<int>(), name + ": dim");
    const bool binary = c.at("file").get<std::string>().ends_with(".bin");
    CheckValues(vectors.values, c.at("values"), binary ? 0.0 : 1e-7, name);
}

void RunAlign(const json& c, const fs::path& root) {
    const std::string name = "align " + c.at("label").get<std::string>();
    const auto vectors = WordEmbeddings::Load((root / "word_embeddings" / c.at("file").get<std::string>()).string());
    Check(vectors.success, name + " load: " + vectors.error_message);
    const auto& o = c.at("options");
    cyxwiz::WordAlignmentOptions options;
    options.lowercase_fallback = o.at("lowercase_fallback").get<bool>();
    options.unknown = o.at("unknown").get<std::string>();
    options.pad_id = o.at("pad_id").get<int>();
    options.unknown_id = o.at("unknown_id").get<int>();
    const auto aligned = WordEmbeddings::Align(vectors, c.at("vocabulary").get<std::vector<std::string>>(), options);
    Check(aligned.success, name + ": " + aligned.error_message);
    Check(aligned.exact == c.at("exact").get<size_t>() && aligned.lowercase == c.at("lowercase").get<size_t>() &&
              aligned.missing == c.at("missing").get<size_t>(),
          name + ": counts " + std::to_string(aligned.exact) + " / " + std::to_string(aligned.lowercase) + " / " +
              std::to_string(aligned.missing));
    Check(aligned.missing_words == c.at("missing_words").get<std::vector<std::string>>(), name + ": missing words");
    CheckValues(aligned.values, c.at("values"), 1e-6, name);
}

// The same SGNS steps as PyTorch (float32) on the same matrices and batches.
void RunSteps(const json& c) {
    auto input = c.at("input_vectors").get<std::vector<float>>();
    auto output = c.at("output_vectors").get<std::vector<float>>();
    std::vector<cyxwiz::Word2VecBatch> batches;
    for (const auto& b : c.at("batches")) {
        cyxwiz::Word2VecBatch batch;
        batch.inputs = b.at("inputs").get<std::vector<int>>();
        batch.context_width = b.at("context_width").get<int>();
        batch.outputs = b.at("outputs").get<std::vector<int>>();
        batch.negatives = b.at("negatives").get<std::vector<int>>();
        batch.negative = b.at("negative").get<int>();
        batch.alpha = b.at("alpha").get<double>();
        batches.push_back(std::move(batch));
    }
    std::vector<double> losses;
    std::string error;
    Check(WordEmbeddings::Word2VecSteps(input, output, c.at("vocab_size").get<int>(), c.at("dim").get<int>(), batches,
                                        &losses, &error),
          "word2vec steps: " + error);
    const auto expected_losses = c.at("losses").get<std::vector<double>>();
    Check(losses.size() == expected_losses.size(), "word2vec steps: loss count");
    for (size_t i = 0; i < losses.size(); ++i)
        Check(std::abs(losses[i] - expected_losses[i]) <= 1e-5 * (1.0 + std::abs(expected_losses[i])),
              "word2vec step " + std::to_string(i) + " loss " + std::to_string(losses[i]) + ", PyTorch " +
                  std::to_string(expected_losses[i]));
    CheckValues(input, c.at("expected_input"), 1e-5, "word2vec input vectors");
    CheckValues(output, c.at("expected_output"), 1e-5, "word2vec output vectors");
}

// How often a word's nearest neighbour (cosine) is in its own group.
double GroupAccuracy(const cyxwiz::WordVectors& vectors, const std::vector<std::vector<std::string>>& groups) {
    std::vector<std::vector<double>> rows;
    std::vector<int> group_of;
    for (size_t g = 0; g < groups.size(); ++g) {
        for (const auto& name : groups[g]) {
            const auto it = std::find(vectors.words.begin(), vectors.words.end(), name);
            Check(it != vectors.words.end(), "trained vocabulary lacks " + name);
            const size_t r = static_cast<size_t>(it - vectors.words.begin());
            const size_t dim = static_cast<size_t>(vectors.dim);
            std::vector<double> row(vectors.values.begin() + static_cast<long long>(r * dim),
                                    vectors.values.begin() + static_cast<long long>((r + 1) * dim));
            double norm = 0.0;
            for (const double x : row) norm += x * x;
            for (double& x : row) x /= std::sqrt(norm);
            rows.push_back(std::move(row));
            group_of.push_back(static_cast<int>(g));
        }
    }
    size_t same = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        double best = -2.0;
        size_t best_j = i;
        for (size_t j = 0; j < rows.size(); ++j) {
            if (j == i) continue;
            double dot = 0.0;
            for (size_t k = 0; k < rows[i].size(); ++k) dot += rows[i][k] * rows[j][k];
            if (dot > best) {
                best = dot;
                best_j = j;
            }
        }
        same += group_of[i] == group_of[best_j] ? 1 : 0;
    }
    return static_cast<double>(same) / static_cast<double>(rows.size());
}

// Trained on the device with gensim's settings: the groups are as separable as
// gensim makes them (within 0.1), the loss falls, and a run repeats for a seed.
void RunQuality(const json& c) {
    const auto& s = c.at("settings");
    cyxwiz::Word2VecOptions options;
    options.cbow = c.at("cbow").get<bool>();
    options.vector_size = s.at("vector_size").get<int>();
    options.window = s.at("window").get<int>();
    options.min_count = s.at("min_count").get<int>();
    options.negative = s.at("negative").get<int>();
    options.sample = s.at("sample").get<double>();
    options.epochs = s.at("epochs").get<int>();
    options.alpha = s.at("alpha").get<double>();
    options.min_alpha = s.at("min_alpha").get<double>();
    const auto sentences = c.at("sentences").get<std::vector<std::vector<std::string>>>();
    const auto groups = c.at("groups").get<std::vector<std::vector<std::string>>>();
    const std::string name = "word2vec " + c.at("label").get<std::string>();
    const auto model = WordEmbeddings::TrainWord2Vec(sentences, options);
    Check(model.success, name + ": " + model.error_message);
    std::string losses;
    for (const double loss : model.epoch_loss) losses += " " + std::to_string(loss);
    Check(model.epoch_loss.size() == static_cast<size_t>(options.epochs) &&
              model.epoch_loss.back() < model.epoch_loss.front(),
          name + ": the loss falls; per epoch:" + losses + "; examples " + std::to_string(model.examples));
    const double accuracy = GroupAccuracy(model.vectors, groups);
    const double reference = c.at("gensim_accuracy").get<double>();
    std::cout << name << ": nearest neighbour in its own group " << accuracy << " (gensim " << reference
              << "), loss per example " << model.epoch_loss.front() << " -> " << model.epoch_loss.back() << std::endl;
    Check(accuracy >= reference - 0.1,
          name + ": group accuracy " + std::to_string(accuracy) + ", gensim " + std::to_string(reference));
    const auto again = WordEmbeddings::TrainWord2Vec(sentences, options);
    Check(again.success && again.vectors.values == model.vectors.values, name + ": a run repeats for one seed");
}

// The same co-occurrence counts as numpy (cooccur.c's rule).
void RunGloVeCounts(const json& c) {
    std::string error;
    const auto counts = WordEmbeddings::CountCooccurrences(
        c.at("tokens").get<std::vector<int>>(), c.at("sentence_of").get<std::vector<int>>(),
        c.at("vocab_size").get<int>(), c.at("window").get<int>(), &error);
    Check(counts.success, "glove counts: " + error);
    Check(counts.rows == c.at("rows").get<std::vector<int>>() && counts.cols == c.at("cols").get<std::vector<int>>(),
          "glove counts: the pairs differ from numpy (" + std::to_string(counts.rows.size()) + " pairs)");
    CheckValues(counts.values, c.at("values"), 1e-6, "glove counts");
}

// The same AdaGrad steps as PyTorch (float32) on the same state and batches.
void RunGloVeSteps(const json& c) {
    const auto& s = c.at("state");
    cyxwiz::GloVeState state;
    state.word = s.at("word").get<std::vector<float>>();
    state.context = s.at("context").get<std::vector<float>>();
    state.word_bias = s.at("word_bias").get<std::vector<float>>();
    state.context_bias = s.at("context_bias").get<std::vector<float>>();
    state.word_sq = s.at("word_sq").get<std::vector<float>>();
    state.context_sq = s.at("context_sq").get<std::vector<float>>();
    state.word_bias_sq = s.at("word_bias_sq").get<std::vector<float>>();
    state.context_bias_sq = s.at("context_bias_sq").get<std::vector<float>>();
    std::vector<cyxwiz::GloVeBatch> batches;
    for (const auto& b : c.at("batches")) {
        cyxwiz::GloVeBatch batch;
        batch.rows = b.at("rows").get<std::vector<int>>();
        batch.cols = b.at("cols").get<std::vector<int>>();
        batch.counts = b.at("counts").get<std::vector<float>>();
        batches.push_back(std::move(batch));
    }
    const std::string name = "glove steps " + c.at("label").get<std::string>();
    std::vector<double> costs;
    std::string error;
    Check(WordEmbeddings::GloVeSteps(state, c.at("vocab_size").get<int>(), c.at("dim").get<int>(), batches,
                                     c.at("x_max").get<double>(), c.at("alpha").get<double>(),
                                     c.at("learning_rate").get<double>(), &costs, &error),
          name + ": " + error);
    const auto expected_costs = c.at("costs").get<std::vector<double>>();
    Check(costs.size() == expected_costs.size(), name + ": cost count");
    for (size_t i = 0; i < costs.size(); ++i)
        Check(std::abs(costs[i] - expected_costs[i]) <= 1e-4 * (1.0 + std::abs(expected_costs[i])),
              name + " cost " + std::to_string(i) + ": " + std::to_string(costs[i]) + ", PyTorch " +
                  std::to_string(expected_costs[i]));
    const auto& e = c.at("expected");
    CheckValues(state.word, e.at("word"), 1e-5, name + " word vectors");
    CheckValues(state.context, e.at("context"), 1e-5, name + " context vectors");
    CheckValues(state.word_bias, e.at("word_bias"), 1e-5, name + " word biases");
    CheckValues(state.context_bias, e.at("context_bias"), 1e-5, name + " context biases");
    CheckValues(state.word_sq, e.at("word_sq"), 1e-5, name + " word AdaGrad sums");
    CheckValues(state.context_sq, e.at("context_sq"), 1e-5, name + " context AdaGrad sums");
    CheckValues(state.word_bias_sq, e.at("word_bias_sq"), 1e-5, name + " word bias AdaGrad sums");
    CheckValues(state.context_bias_sq, e.at("context_bias_sq"), 1e-5, name + " context bias AdaGrad sums");
}

// Trained on the device with the reference's settings: the groups are as
// separable as the reference makes them (within 0.1), the cost falls, and a run
// repeats for a seed.
void RunGloVeQuality(const json& c) {
    const auto& s = c.at("settings");
    cyxwiz::GloVeOptions options;
    options.vector_size = s.at("vector_size").get<int>();
    options.window = s.at("window").get<int>();
    options.min_count = s.at("min_count").get<int>();
    options.x_max = s.at("x_max").get<double>();
    options.alpha = s.at("alpha").get<double>();
    options.learning_rate = s.at("learning_rate").get<double>();
    options.epochs = s.at("epochs").get<int>();
    options.batch_size = s.at("batch_size").get<int>();
    const auto sentences = c.at("sentences").get<std::vector<std::vector<std::string>>>();
    const auto groups = c.at("groups").get<std::vector<std::vector<std::string>>>();
    int counting_calls = 0;
    int last_epoch = 0;
    options.on_progress = [&](double, int epoch) {
        if (epoch == 0) ++counting_calls;
        last_epoch = epoch;
        return true;
    };
    const auto model = WordEmbeddings::TrainGloVe(sentences, options);
    Check(model.success, "glove quality: " + model.error_message);
    Check(counting_calls > 0 && last_epoch == options.epochs, "glove quality: progress reports both stages");
    std::string costs;
    for (const double cost : model.epoch_cost) costs += " " + std::to_string(cost);
    Check(model.epoch_cost.size() == static_cast<size_t>(options.epochs) &&
              model.epoch_cost.back() < model.epoch_cost.front(),
          "glove quality: the cost falls; per epoch:" + costs);
    const double accuracy = GroupAccuracy(model.vectors, groups);
    const double reference = c.at("reference_accuracy").get<double>();
    std::cout << "glove: nearest neighbour in its own group " << accuracy << " (reference " << reference << "), "
              << model.pairs << " pairs, cost per pair " << model.epoch_cost.front() << " -> "
              << model.epoch_cost.back() << std::endl;
    Check(accuracy >= reference - 0.1,
          "glove quality: group accuracy " + std::to_string(accuracy) + ", reference " + std::to_string(reference));
    options.on_progress = nullptr;
    const auto again = WordEmbeddings::TrainGloVe(sentences, options);
    Check(again.success && again.vectors.values == model.vectors.values,
          "glove quality: a run repeats for one seed");
    options.on_progress = [](double, int epoch) { return epoch < 2; };
    Check(!WordEmbeddings::TrainGloVe(sentences, options).success, "glove: a stop refuses the model");
}

// fastText's hash and pieces, as gensim computes them.
void RunFastTextHash(const json& c) {
    const auto texts = c.at("texts").get<std::vector<std::string>>();
    const auto hashes = c.at("hashes").get<std::vector<uint32_t>>();
    for (size_t i = 0; i < texts.size(); ++i)
        Check(WordEmbeddings::FastTextHash(texts[i]) == hashes[i],
              "fastText hash of '" + texts[i] + "': " + std::to_string(WordEmbeddings::FastTextHash(texts[i])) +
                  ", gensim " + std::to_string(hashes[i]));
}

void RunFastTextNgrams(const json& c) {
    for (const auto& row : c.at("rows")) {
        const auto word = row.at("word").get<std::string>();
        const int min_n = row.at("min_n").get<int>(), max_n = row.at("max_n").get<int>();
        const std::string name =
            "fastText n-grams of '" + word + "' " + std::to_string(min_n) + ".." + std::to_string(max_n);
        Check(WordEmbeddings::FastTextNgrams(word, min_n, max_n) == row.at("ngrams").get<std::vector<std::string>>(),
              name + " differ from gensim");
        Check(WordEmbeddings::FastTextBuckets(word, min_n, max_n, 2000000) ==
                  row.at("buckets").get<std::vector<int>>(),
              name + ": buckets differ from gensim");
    }
}

cyxwiz::FastTextModel FastTextFixtureModel(const json& c) {
    cyxwiz::FastTextModel model;
    model.words = c.at("words").get<std::vector<std::string>>();
    model.min_n = c.at("min_n").get<int>();
    model.max_n = c.at("max_n").get<int>();
    model.bucket = c.at("bucket").get<int>();
    model.dim = c.at("dim").get<int>();
    model.word_rows = c.at("word_rows").get<std::vector<float>>();
    model.ngram_rows = c.at("ngram_rows").get<std::vector<float>>();
    if (c.contains("output_vectors")) model.output_vectors = c.at("output_vectors").get<std::vector<float>>();
    return model;
}

void RunFastTextSteps(const json& c) {
    auto model = FastTextFixtureModel(c);
    std::vector<cyxwiz::Word2VecBatch> batches;
    for (const auto& b : c.at("batches")) {
        cyxwiz::Word2VecBatch batch;
        batch.inputs = b.at("inputs").get<std::vector<int>>();
        batch.context_width = b.at("context_width").get<int>();
        batch.outputs = b.at("outputs").get<std::vector<int>>();
        batch.negatives = b.at("negatives").get<std::vector<int>>();
        batch.negative = b.at("negative").get<int>();
        batch.alpha = b.at("alpha").get<double>();
        batches.push_back(std::move(batch));
    }
    const std::string name = "fastText steps (" + c.at("label").get<std::string>() + ")";
    std::vector<double> losses;
    std::string error;
    Check(WordEmbeddings::FastTextSteps(model, batches, &losses, &error), name + ": " + error);
    const auto expected_losses = c.at("losses").get<std::vector<double>>();
    Check(losses.size() == expected_losses.size(), name + ": loss count");
    for (size_t i = 0; i < losses.size(); ++i)
        Check(std::abs(losses[i] - expected_losses[i]) <= 1e-5 * (1.0 + std::abs(expected_losses[i])),
              name + " " + std::to_string(i) + ": loss " + std::to_string(losses[i]) + ", PyTorch " +
                  std::to_string(expected_losses[i]));
    CheckValues(model.word_rows, c.at("expected_word_rows"), 1e-5, name + ": word rows");
    CheckValues(model.ngram_rows, c.at("expected_ngram_rows"), 1e-5, name + ": n-gram rows");
    CheckValues(model.output_vectors, c.at("expected_output"), 1e-5, name + ": output rows");
}

void RunFastTextVectors(const json& c) {
    const auto model = FastTextFixtureModel(c);
    const auto wanted = c.at("wanted").get<std::vector<std::string>>();
    std::vector<cyxwiz::FastTextSource> sources;
    const auto vectors = WordEmbeddings::FastTextVectors(model, &wanted, &sources);
    Check(vectors.success && vectors.words == wanted, "fastText vectors: " + vectors.error_message);
    std::vector<double> expected;
    for (const auto& row : c.at("expected"))
        for (const auto& x : row) expected.push_back(x.get<double>());
    CheckValues(vectors.values, json(expected), 1e-6, "fastText vectors (word, pieces, none)");
    const auto expected_sources = c.at("sources").get<std::vector<int>>();
    for (size_t i = 0; i < wanted.size(); ++i)
        Check(static_cast<int>(sources[i]) == expected_sources[i], "fastText source of '" + wanted[i] + "'");
}

// How often each word's nearest known word (cosine, not itself) is in its
// group; `vectors` row i is `vectors.words[i]`, `known` the vocabulary's vectors.
double NearestKnownAccuracy(const cyxwiz::WordVectors& vectors, const cyxwiz::WordVectors& known,
                            const std::map<std::string, int>& group_of) {
    const size_t dim = static_cast<size_t>(known.dim);
    const auto unit = [&](const std::vector<float>& values, size_t r) {
        std::vector<double> row(values.begin() + static_cast<long long>(r * dim),
                                values.begin() + static_cast<long long>((r + 1) * dim));
        double norm = 0.0;
        for (const double x : row) norm += x * x;
        for (double& x : row) x /= std::sqrt(std::max(norm, 1e-30));
        return row;
    };
    size_t same = 0;
    for (size_t i = 0; i < vectors.words.size(); ++i) {
        const auto row = unit(vectors.values, i);
        double best = -2.0;
        size_t best_j = 0;
        for (size_t j = 0; j < known.words.size(); ++j) {
            if (known.words[j] == vectors.words[i]) continue;
            const auto other = unit(known.values, j);
            double dot = 0.0;
            for (size_t k = 0; k < dim; ++k) dot += row[k] * other[k];
            if (dot > best) {
                best = dot;
                best_j = j;
            }
        }
        same += group_of.at(vectors.words[i]) == group_of.at(known.words[best_j]) ? 1 : 0;
    }
    return static_cast<double>(same) / static_cast<double>(vectors.words.size());
}

// Trained on the device with gensim's settings: the groups (and unseen
// variants built from pieces) are as separable as gensim makes them (within
// 0.1), the loss falls, and a run repeats for a seed.
void RunFastTextQuality(const json& c) {
    const auto& s = c.at("settings");
    cyxwiz::FastTextOptions options;
    options.cbow = c.at("cbow").get<bool>();
    options.vector_size = s.at("vector_size").get<int>();
    options.window = s.at("window").get<int>();
    options.min_count = s.at("min_count").get<int>();
    options.negative = s.at("negative").get<int>();
    options.sample = s.at("sample").get<double>();
    options.epochs = s.at("epochs").get<int>();
    options.alpha = s.at("alpha").get<double>();
    options.min_alpha = s.at("min_alpha").get<double>();
    options.min_n = s.at("min_n").get<int>();
    options.max_n = s.at("max_n").get<int>();
    options.bucket = s.at("bucket").get<int>();
    const auto sentences = c.at("sentences").get<std::vector<std::vector<std::string>>>();
    const auto groups = c.at("groups").get<std::vector<std::vector<std::string>>>();
    const auto unseen = c.at("unseen").get<std::vector<std::string>>();
    const std::string name = "fastText " + c.at("label").get<std::string>();
    const auto model = WordEmbeddings::TrainFastText(sentences, options);
    Check(model.success, name + ": " + model.error_message);
    Check(model.epoch_loss.size() == static_cast<size_t>(options.epochs) &&
              model.epoch_loss.back() < model.epoch_loss.front(),
          name + ": the loss falls; per epoch:" + [&] {
              std::string text;
              for (const double loss : model.epoch_loss) text += " " + std::to_string(loss);
              return text;
          }());
    std::map<std::string, int> group_of;
    std::vector<std::string> names;
    for (size_t g = 0; g < groups.size(); ++g) {
        for (const auto& w : groups[g]) {
            group_of[w] = static_cast<int>(g);
            group_of[w + "x"] = static_cast<int>(g);
            names.push_back(w);
        }
    }
    std::vector<cyxwiz::FastTextSource> sources;
    const auto known = WordEmbeddings::FastTextVectors(model, &names);
    const auto built = WordEmbeddings::FastTextVectors(model, &unseen, &sources);
    Check(known.success && built.success, name + ": vectors");
    Check(std::all_of(sources.begin(), sources.end(),
                      [](cyxwiz::FastTextSource source) { return source == cyxwiz::FastTextSource::Pieces; }),
          name + ": unseen words are built from pieces");
    const double accuracy = NearestKnownAccuracy(known, known, group_of);
    const double unseen_accuracy = NearestKnownAccuracy(built, known, group_of);
    const double reference = c.at("gensim_accuracy").get<double>();
    const double unseen_reference = c.at("gensim_unseen_accuracy").get<double>();
    std::cout << name << ": nearest neighbour in its own group " << accuracy << " (gensim " << reference
              << "), unseen words " << unseen_accuracy << " (gensim " << unseen_reference << "), loss per example "
              << model.epoch_loss.front() << " -> " << model.epoch_loss.back() << std::endl;
    Check(accuracy >= reference - 0.1 && unseen_accuracy >= unseen_reference - 0.1,
          name + ": group accuracy " + std::to_string(accuracy) + " / unseen " + std::to_string(unseen_accuracy));
    const auto again = WordEmbeddings::TrainFastText(sentences, options);
    Check(again.success && again.word_rows == model.word_rows && again.ngram_rows == model.ngram_rows,
          name + ": a run repeats for one seed");
}

// A fastText .bin written by gensim's save_facebook_model: its header and
// words, its vectors (words and unseen words) as load_facebook_model gives
// them, and a vocabulary lined up with n-gram vectors for what it lacks.
void RunFastTextBin(const json& c, const fs::path& root) {
    const std::string path = (root / "word_embeddings" / c.at("file").get<std::string>()).string();
    const auto words = c.at("words").get<std::vector<std::string>>();
    Check(WordEmbeddings::IsFastTextBin(path) &&
              !WordEmbeddings::IsFastTextBin((root / "word_embeddings" / "vectors.bin").string()),
          "fastText .bin told from word2vec binary");
    const auto info = WordEmbeddings::ReadFastTextInfo(path);
    Check(info.success, "fastText .bin header: " + info.error_message);
    Check(info.dim == c.at("dim").get<int>() && info.min_n == c.at("min_n").get<int>() &&
              info.max_n == c.at("max_n").get<int>() && info.bucket == c.at("bucket").get<int>() &&
              info.model == c.at("model").get<std::string>() && info.words == words,
          "fastText .bin header and words");

    const auto all = WordEmbeddings::Load(path);
    Check(all.success && all.words == words, "Load reads a fastText .bin: " + all.error_message);
    CheckValues(std::vector<float>(all.values.begin(), all.values.begin() + 3 * info.dim), c.at("first_vectors"), 1e-5,
                "fastText .bin: the first words' vectors");
    const auto first_two = WordEmbeddings::Load(path, 2);
    Check(first_two.success && first_two.words.size() == 2, "fastText .bin: read at most 2 words");

    const auto wanted = c.at("wanted").get<std::vector<std::string>>();
    std::vector<cyxwiz::FastTextSource> sources;
    const auto vectors = WordEmbeddings::LoadFastTextBin(path, &wanted, &sources);
    Check(vectors.success && vectors.words == wanted, "fastText .bin vectors: " + vectors.error_message);
    std::vector<double> expected;
    for (const auto& row : c.at("expected"))
        for (const auto& x : row) expected.push_back(x.get<double>());
    CheckValues(vectors.values, json(expected), 1e-5, "fastText .bin vectors (words, pieces, none) vs gensim");
    const auto expected_sources = c.at("sources").get<std::vector<int>>();
    for (size_t i = 0; i < wanted.size(); ++i)
        Check(static_cast<int>(sources[i]) == expected_sources[i], "fastText .bin source of '" + wanted[i] + "'");

    const auto vocabulary = c.at("vocabulary").get<std::vector<std::string>>();
    const auto found = WordEmbeddings::LoadFastTextBinFor(path, vocabulary, true);
    Check(found.success && found.words == std::vector<std::string>({"sleep", "worry"}),
          "fastText .bin: only the vocabulary's words are read");
    cyxwiz::WordAlignmentOptions options;
    options.pieces = [&](const std::vector<std::string>& tokens) {
        return WordEmbeddings::LoadFastTextBin(path, &tokens);
    };
    const auto aligned = WordEmbeddings::Align(found, vocabulary, options);
    Check(aligned.success, "fastText line-up: " + aligned.error_message);
    Check(aligned.exact == 1 && aligned.lowercase == 1 && aligned.pieces == 1 && aligned.missing == 2 &&
              aligned.piece_words == std::vector<std::string>({"sleepyx"}),
          "fastText line-up: 1 found, 1 in lowercase, 1 from pieces, 2 missing (punctuation, emoji)");
    for (const auto& [id, row] : c.at("aligned_rows").items()) {
        const size_t r = static_cast<size_t>(std::stoi(id));
        CheckValues(std::vector<float>(aligned.values.begin() + static_cast<long long>(r * 6),
                                       aligned.values.begin() + static_cast<long long>((r + 1) * 6)),
                    row, 1e-5, "fastText line-up row " + id);
    }
    // A model cut short is refused with a reason.
    const fs::path cut = fs::temp_directory_path() / "cyxwiz_fasttext_cut.bin";
    {
        std::ifstream in(path, std::ios::binary);
        std::string head(200, '\0');
        in.read(head.data(), static_cast<std::streamsize>(head.size()));
        std::ofstream(cut, std::ios::binary) << head;
    }
    const auto refused = WordEmbeddings::ReadFastTextInfo(cut.string());
    Check(!refused.success && !refused.error_message.empty(), "a cut fastText .bin is refused");
    fs::remove(cut);
}

// Nearest words by cosine, and row means of token vectors, against numpy.
void RunNearest(const json& c, const fs::path& root) {
    const auto vectors = WordEmbeddings::Load((root / "word_embeddings" / c.at("file").get<std::string>()).string());
    const auto nearest = WordEmbeddings::Nearest(vectors, c.at("word").get<std::string>(), c.at("k").get<int>());
    const auto words = c.at("words").get<std::vector<std::string>>();
    const auto similarity = c.at("similarity").get<std::vector<double>>();
    Check(nearest.size() == words.size(), "nearest " + c.at("word").get<std::string>() + ": count");
    for (size_t i = 0; i < words.size(); ++i) {
        Check(nearest[i].first == words[i], "nearest " + c.at("word").get<std::string>() + " [" + std::to_string(i) +
                                                "]: " + nearest[i].first + ", numpy " + words[i]);
        Check(std::abs(nearest[i].second - similarity[i]) <= 1e-5, "nearest similarity " + std::to_string(i));
    }
    Check(WordEmbeddings::Nearest(vectors, "not-a-word").empty(), "a word without a vector has no neighbours");
}

void RunRowMeans(const json& c, const fs::path& root) {
    const auto vectors = WordEmbeddings::Load((root / "word_embeddings" / c.at("file").get<std::string>()).string());
    std::string error;
    const auto means = WordEmbeddings::RowMeans(vectors.values, vectors.dim, c.at("tokens").get<std::vector<int>>(),
                                                c.at("tokens_per_row").get<int>(), &error);
    Check(!means.empty(), "row means: " + error);
    CheckValues(means, c.at("values"), 1e-6, "row means");
    Check(WordEmbeddings::RowMeans(vectors.values, vectors.dim, {99}, 1, &error).empty(),
          "a token outside the table is refused");
    // A block past 32-bit device offsets is refused before it reaches the device.
    const int wide = 1 << 21;
    Check(WordEmbeddings::RowMeans(std::vector<float>(wide, 0.0f), wide, std::vector<int>(1024, 0), 1024, &error)
                  .empty() &&
              error.find("too large") != std::string::npos,
          "an oversized block is refused: " + error);
}

// Saved .vec text reads back as the same float32 values.
void CheckSaveRoundTrip(const fs::path& root, const fs::path& dir) {
    const auto vectors = WordEmbeddings::Load((root / "word_embeddings" / "vectors.bin").string());
    std::string error;
    const fs::path out = dir / "saved.vec";
    Check(WordEmbeddings::SaveText(vectors, out.string(), &error), "save: " + error);
    const auto again = WordEmbeddings::Load(out.string());
    Check(again.success && again.words == vectors.words && again.dim == vectors.dim && again.values == vectors.values,
          "saved vectors read back exactly");
}

void WriteBytes(const fs::path& path, const std::string& bytes) {
    std::ofstream(path, std::ios::binary) << bytes;
}

// What gensim cannot show: seeded random rows, a byte order mark, a word with a
// space, and the refusals.
void CheckOwnRules(const fs::path& root) {
    const fs::path dir = fs::temp_directory_path() / "cyxwiz_word_embeddings_test";
    fs::create_directories(dir);

    const auto vectors = WordEmbeddings::Load((root / "word_embeddings" / "vectors.vec").string());
    cyxwiz::WordAlignmentOptions random;
    random.missing = "random";
    random.seed = 7;
    const std::vector<std::string> vocab = {"[PAD]", "[UNK]", "the", "unicorn", "pegasus", "griffin"};
    const auto first = WordEmbeddings::Align(vectors, vocab, random);
    const auto again = WordEmbeddings::Align(vectors, vocab, random);
    Check(first.success && again.success && first.values == again.values, "random rows repeat for one seed");
    random.seed = 8;
    const auto other = WordEmbeddings::Align(vectors, vocab, random);
    Check(other.success && other.values != first.values, "another seed gives other random rows");
    bool nonzero = false;
    for (int d = 0; d < first.dim; ++d) nonzero = nonzero || first.values[3 * static_cast<size_t>(first.dim) + d] != 0.0f;
    Check(nonzero && first.missing == 3, "missing words get random rows");
    for (int d = 0; d < first.dim; ++d) Check(first.values[static_cast<size_t>(d)] == 0.0f, "PAD stays zeros");

    WriteBytes(dir / "bom.txt", "\xEF\xBB\xBFhello 1 2\nnew york 3 4\n");
    const auto bom = WordEmbeddings::Load((dir / "bom.txt").string());
    Check(bom.success && bom.words.size() == 2 && bom.words[0] == "hello" && bom.words[1] == "new york" &&
              bom.dim == 2 && bom.values[2] == 3.0f,
          "a byte order mark is skipped and a word may contain a space: " + bom.error_message);

    Check(!WordEmbeddings::Load((dir / "absent.vec").string()).success, "a missing file is refused");
    WriteBytes(dir / "bad.txt", "a 1 2\nb 3 x\n");
    const auto bad = WordEmbeddings::Load((dir / "bad.txt").string());
    Check(!bad.success && bad.error_message.find("Line 2") != std::string::npos &&
              bad.error_message.find("'x'") != std::string::npos,
          "a bad number names its line: " + bad.error_message);
    WriteBytes(dir / "short.txt", "a 1 2\nb 3\n");
    const auto short_row = WordEmbeddings::Load((dir / "short.txt").string());
    Check(!short_row.success && short_row.error_message.find("Line 2") != std::string::npos, "a short row is refused");
    WriteBytes(dir / "cut.bin", std::string("2 3\nword ") + std::string(5, '\0'));
    const auto cut = WordEmbeddings::Load((dir / "cut.bin").string());
    Check(!cut.success && cut.error_message.find("word") != std::string::npos, "a cut binary file is refused");
    Check(!WordEmbeddings::Align(vectors, {}).success, "an empty vocabulary is refused");
    cyxwiz::WordAlignmentOptions wrong;
    wrong.missing = "ones";
    Check(!WordEmbeddings::Align(vectors, vocab, wrong).success, "an unknown missing rule is refused");
    CheckSaveRoundTrip(root, dir);
    fs::remove_all(dir);
}

}  // namespace

int main(int, char** argv) {
    Check(cyxwiz::test::SelectTestDeviceFromEnvironment(), "requested test device");
    const fs::path root = FixtureRoot(argv[0]);
    std::ifstream in(root / "word_embeddings_gensim.json");
    Check(static_cast<bool>(in), "cannot open the word embeddings fixture");
    const json fixture = json::parse(in);
    size_t passed = 0;
    for (const auto& c : fixture.at("cases")) {
        if (c.at("op") == "load") RunLoad(c, root);
        else if (c.at("op") == "nearest") RunNearest(c, root);
        else if (c.at("op") == "row_means") RunRowMeans(c, root);
        else RunAlign(c, root);
        ++passed;
    }
    CheckOwnRules(root);

    std::ifstream w2v_in(root / "word2vec_reference.json");
    Check(static_cast<bool>(w2v_in), "cannot open the word2vec fixture");
    const json w2v = json::parse(w2v_in);
    for (const auto& c : w2v.at("cases")) {
        if (c.at("op") == "steps") RunSteps(c);
        else RunQuality(c);
        ++passed;
    }
    std::ifstream glove_in(root / "glove_reference.json");
    Check(static_cast<bool>(glove_in), "cannot open the GloVe fixture");
    const json glove = json::parse(glove_in);
    for (const auto& c : glove.at("cases")) {
        if (c.at("op") == "counts") RunGloVeCounts(c);
        else if (c.at("op") == "steps") RunGloVeSteps(c);
        else RunGloVeQuality(c);
        ++passed;
    }
    std::ifstream fasttext_in(root / "fasttext_reference.json");
    Check(static_cast<bool>(fasttext_in), "cannot open the fastText fixture");
    const json fasttext = json::parse(fasttext_in);
    for (const auto& c : fasttext.at("cases")) {
        if (c.at("op") == "hash") RunFastTextHash(c);
        else if (c.at("op") == "ngrams") RunFastTextNgrams(c);
        else if (c.at("op") == "steps") RunFastTextSteps(c);
        else if (c.at("op") == "vectors") RunFastTextVectors(c);
        else if (c.at("op") == "bin") RunFastTextBin(c, root);
        else RunFastTextQuality(c);
        ++passed;
    }
    std::string error;
    Check(!WordEmbeddings::TrainFastText({{"one"}}).success, "fastText: fewer than two words is refused");
    cyxwiz::FastTextOptions bad_pieces;
    bad_pieces.min_n = 4;
    bad_pieces.max_n = 3;
    Check(!WordEmbeddings::TrainFastText({{"a", "b", "a", "b", "a", "b", "a", "b", "a", "b"}}, bad_pieces).success,
          "fastText: max_n below min_n is refused");
    Check(!WordEmbeddings::CountCooccurrences({0, 7}, {0, 0}, 3, 2, &error).success,
          "a GloVe token outside the vocabulary is refused");
    cyxwiz::GloVeState two_words;
    two_words.word = two_words.context = two_words.word_sq = two_words.context_sq = std::vector<float>(4, 1.0f);
    two_words.word_bias = two_words.context_bias = std::vector<float>(2, 1.0f);
    two_words.word_bias_sq = two_words.context_bias_sq = std::vector<float>(2, 1.0f);
    cyxwiz::GloVeBatch zero;
    zero.rows = {0};
    zero.cols = {1};
    zero.counts = {0.0f};
    Check(!WordEmbeddings::GloVeSteps(two_words, 2, 2, {zero}, 100.0, 0.75, 0.05, nullptr, &error),
          "a zero GloVe count is refused");
    Check(!WordEmbeddings::TrainGloVe({{"one"}}).success, "GloVe: fewer than two words above min_count is refused");
    std::vector<float> a(4, 0.0f), b(4, 0.0f);
    cyxwiz::Word2VecBatch outside;
    outside.inputs = {5};
    outside.outputs = {0};
    Check(!WordEmbeddings::Word2VecSteps(a, b, 2, 2, {outside}, nullptr, &error),
          "a word outside the vocabulary is refused");
    Check(!WordEmbeddings::TrainWord2Vec({{"one"}}).success, "fewer than two words above min_count is refused");
    std::cout << "word vectors match gensim " << fixture.at("gensim").get<std::string>() << ": " << passed << " cases"
              << std::endl;
    return 0;
}
