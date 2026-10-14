// Word vectors (TOFIX152):
// - files and vocabulary alignment against gensim 4: fixtures/word_embeddings_gensim.json
//   and fixtures/word_embeddings/ (reference/generate_word_embeddings_fixtures.py);
// - word2vec steps against PyTorch and training quality against gensim:
//   fixtures/word2vec_reference.json (reference/generate_word2vec_fixtures.py). Runs on the active ArrayFire
// backend (CYXWIZ_TEST_ARRAYFIRE_BACKEND=cuda|opencl|oneapi|cpu).
#include "test_device_selection.h"

#include <cyxwiz/word_embeddings.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
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
    std::string error;
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
