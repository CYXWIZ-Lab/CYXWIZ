// Word Embeddings dialog wording (TOFIX152). The main scenario is the real
// sentiment dataset: its 5,000-token vocabulary lined up with vectors trained
// on it (gensim, min count 5): 4,824 found, 172 missing.

#include "../src/core/word_embeddings_presentation.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

const std::string kDot = " \xC2\xB7 ";

}  // namespace

int main() {
    Check(cyxwiz::GroupedCount(0) == "0" && cyxwiz::GroupedCount(999) == "999" &&
              cyxwiz::GroupedCount(4824) == "4,824" && cyxwiz::GroupedCount(400000) == "400,000" &&
              cyxwiz::GroupedCount(5990000) == "5,990,000",
          "grouped counts");

    Check(cyxwiz::VectorFileFormat("D:/v/GoogleNews.BIN", "") == "word2vec binary", "a .bin file is binary");
    Check(cyxwiz::VectorFileFormat("wiki.en.vec", "2519370 300") == "word2vec / fastText text", "a header line");
    Check(cyxwiz::VectorFileFormat("glove.6B.100d.txt", "the -0.038194 -0.24487 0.72812") == "GloVe text (no header)",
          "GloVe has no header");
    Check(cyxwiz::VectorFileFormat("odd.txt", "3 2 1") == "GloVe text (no header)", "three numbers are a row");
    Check(cyxwiz::VectorFileSummary("GloVe text (no header)", 400000, 100, 2.44) ==
              "GloVe text (no header)" + kDot + "400,000 words" + kDot + "100 values each" + kDot + "read 2.4 s",
          "file summary");

    const std::vector<std::string> missing = {".", ",", "?", "-", "(", ")", "\"", "/", "!", "*", ":", ";", "2", "3",
                                              "x2", "caf\xC3\xA9"};
    auto view = cyxwiz::BuildWordCoverageView(4824, 0, 172, missing, "[PAD]", "[UNK]", "zeros", "mean");
    Check(view.found == "4,824 found as written" && view.lowercase == "0 found in lowercase" &&
              view.missing == "172 missing",
          "coverage counts");
    Check(view.special == "[PAD] zeros" + kDot + "[UNK] mean", "special tokens: " + view.special);
    Check(std::abs(view.found_share - 4824.0f / 4996.0f) < 1e-6f && view.lowercase_share == 0.0f, "bar shares");
    Check(view.missing_preview == "Missing, first of 172:" && view.preview_words.size() == 13 &&
              view.preview_words.front() == "." && view.preview_words.back() == "2",
          "the first 13 missing words");
    Check(view.missing_kind == "mostly punctuation, digits and symbols", "missing kind");

    view = cyxwiz::BuildWordCoverageView(10, 2, 1, {"caf\xC3\xA9"}, "", "<unk>", "random", "missing");
    Check(view.special == "<unk> random", "unknown as missing follows the missing rows: " + view.special);
    Check(view.missing_kind.empty(), "a word of another script has letters");
    view = cyxwiz::BuildWordCoverageView(10, 0, 3, {"\xF0\x9F\x98\x8A", "\xE2\x9D\xA4", "\xD0\xB4\xD0\xB0"}, "", "",
                                         "zeros", "mean");
    Check(view.missing_kind == "mostly punctuation, digits and symbols", "emoji are symbols, Cyrillic is letters");
    Check(cyxwiz::Utf8CodePoints("a\xC3\xA9\xF0\x9F\x98\x8A") == std::vector<char32_t>{U'a', 0xE9, 0x1F60A},
          "code points");
    Check(cyxwiz::Utf8CodePoints("\xC3") == std::vector<char32_t>{0xFFFD}, "a cut sequence");
    Check(cyxwiz::CodePointNames("\xF0\x9F\x98\x8A\xE2\x9D\xA4") == "U+1F60A U+2764", "code point names");
    view = cyxwiz::BuildWordCoverageView(3, 0, 0, {}, "[PAD]", "", "zeros", "mean");
    Check(view.missing_preview.empty() && view.special == "[PAD] zeros" && view.found_share == 1.0f, "all found");

    Check(std::abs(cyxwiz::StepShareFromStatus("Vectors: Training word2vec, epoch 3 (62%)") - 0.62f) < 1e-6f,
          "share from the status");
    Check(cyxwiz::StepShareFromStatus("Executing node 3 of 4") < 0.0f, "no share");
    Check(cyxwiz::StepMessageFromStatus("Vectors: Training word2vec, epoch 3 (62%)") == "Training word2vec, epoch 3",
          "message from the status");
    Check(cyxwiz::StepMessageFromStatus("Loading") == "Loading", "a plain status");
    std::cout << "word embeddings presentation tests passed\n";
    return 0;
}
