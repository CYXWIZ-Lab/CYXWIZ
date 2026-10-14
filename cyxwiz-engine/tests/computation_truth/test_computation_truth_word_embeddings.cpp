// Word vector files and vocabulary alignment (TOFIX152) against gensim 4:
// fixtures/word_embeddings_gensim.json and fixtures/word_embeddings/ from
// reference/generate_word_embeddings_fixtures.py. Runs on the active ArrayFire
// backend (CYXWIZ_TEST_ARRAYFIRE_BACKEND=cuda|opencl|oneapi|cpu).
#include "test_device_selection.h"

#include <cyxwiz/word_embeddings.h>

#include <nlohmann/json.hpp>

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
    std::cout << "word vectors match gensim " << fixture.at("gensim").get<std::string>() << ": " << passed << " cases"
              << std::endl;
    return 0;
}
