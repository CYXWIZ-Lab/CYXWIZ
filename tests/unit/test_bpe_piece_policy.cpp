#include <catch2/catch_test_macros.hpp>
#include <cyxwiz/tokenizer.h>
#include <algorithm>
#include <sstream>

using namespace cyxwiz;
namespace {
Tokenizer Fit(const std::vector<std::string>& docs) {
    Tokenizer t(TokenizerType::ByteBPE);
    t.SetPadding(false); t.SetTruncation(false);
    t.SetBPEFitPiecePolicy(ByteBPEPiecePolicy::LeadingSpaceV2);
    t.Train(docs, 1, 400);
    return t;
}
std::string Artifact(const Tokenizer& t) {
    std::ostringstream out;
    REQUIRE(t.GetVocabulary().SaveToStream(out));
    return out.str();
}
}
TEST_CASE("Leading-space BPE merges boundaries without changing the v1 default") {
    auto t = Fit({" word word word", " word", "truth mercy"});
    REQUIRE(t.Encode(" word").size() == 1);
    REQUIRE(t.Encode("word").front() != t.Encode(" word").front());
    REQUIRE(t.GetVocabulary().GetBPEPiecePolicy() == ByteBPEPiecePolicy::LeadingSpaceV2);
    Tokenizer old(TokenizerType::ByteBPE);
    old.SetPadding(false); old.SetTruncation(false);
    old.Train({" word word word", " word", "truth mercy"},1,400);
    REQUIRE(old.Encode(" word").size() == 2);
    REQUIRE(old.Encode(" word").front() == 36);
    REQUIRE(Artifact(old).starts_with("#cyxwiz-vocabulary-byte-bpe-v1\n"));
    auto reordered = Fit({"truth mercy", " word", " word word word"});
    REQUIRE(Artifact(reordered) == Artifact(t));
    // Encoding uses the saved artifact policy, never the fit option.
    t.SetBPEFitPiecePolicy(ByteBPEPiecePolicy::WhitespaceV1);
    REQUIRE(t.Encode(" word").size() == 1);
}
TEST_CASE("Leading-space artifacts survive reload and reject mismatched boundaries atomically") {
    auto original = Fit({" word word word", "truth mercy"});
    const auto saved = Artifact(original);
    REQUIRE(saved.starts_with("#cyxwiz-vocabulary-byte-bpe-v2\nleading_space_v2\n"));
    Tokenizer loaded(TokenizerType::ByteBPE);
    loaded.SetPadding(false); loaded.SetTruncation(false);
    std::istringstream in(saved);
    REQUIRE(loaded.GetVocabulary().LoadFromStream(in));
    REQUIRE(loaded.Encode("word word\tmercy") == original.Encode("word word\tmercy"));
    REQUIRE(Artifact(loaded) == saved);
    auto wrong = saved;
    wrong.replace(wrong.find("leading_space_v2"), 16, "unknown_policy");
    std::istringstream unsupported(wrong);
    REQUIRE_FALSE(loaded.GetVocabulary().LoadFromStream(unsupported));
    REQUIRE(Artifact(loaded) == saved);
    REQUIRE_FALSE(loaded.GetVocabulary().SetByteBPE(original.GetVocabulary().GetWords(),
        original.GetVocabulary().GetBPEMerges())); // v1 must reject v2 merges
    REQUIRE(Artifact(loaded) == saved);
    auto words = original.GetVocabulary().GetWords();
    auto merges = original.GetVocabulary().GetBPEMerges();
    const int word = original.Encode(" word").front();
    words.push_back(" word word");
    merges.push_back({word,word,static_cast<int>(words.size()-1)});
    REQUIRE_FALSE(loaded.GetVocabulary().SetByteBPE(words,merges,ByteBPEPiecePolicy::LeadingSpaceV2));
    REQUIRE(Artifact(loaded) == saved);
}
TEST_CASE("Leading-space BPE preserves whitespace all bytes special literals and long pieces") {
    auto t = Fit({"  mercy mercy\ttruth\ntruth", " [PAD] [UNK] [BOS] [EOS]", std::string(600,'a')});
    std::string all;
    for (int i=0;i<256;++i) all.push_back(static_cast<char>(i));
    for (const std::string& text : std::vector<std::string>{all, "", " ", "  mercy   ",
            "\tmercy\ntruth\r\n", "caf\xc3\xa9 \xce\xb1", "[PAD] [UNK] [BOS] [EOS]",
            " " + std::string(4097,'a')}) {
        const auto ids=t.Encode(text);
        REQUIRE(t.Decode(ids)==text);
        REQUIRE(std::all_of(ids.begin(),ids.end(),[](int id){return id>=4;}));
    }
    for (const auto& token:t.GetVocabulary().GetWords()) REQUIRE(token.size()<=256);
    const auto saved=Artifact(t);
    int checks=0;
    t.SetCancellationQuery([&]{return ++checks>3;});
    REQUIRE_THROWS(t.Train({"changed corpus with new words"},1,500));
    REQUIRE(Artifact(t)==saved);
    REQUIRE_THROWS(t.SetBPEFitPiecePolicy(static_cast<ByteBPEPiecePolicy>(99)));
    Tokenizer other(TokenizerType::Word);
    REQUIRE_THROWS(other.SetBPEFitPiecePolicy(ByteBPEPiecePolicy::LeadingSpaceV2));
}

// Staged test fragment, appended to the existing BPE policy suite in the worktree.
namespace {
Tokenizer CharacterFit(const std::vector<std::string>& docs, int frequency = 1, int cap = 200) {
    Tokenizer t(TokenizerType::ByteBPE);
    t.SetPadding(false); t.SetTruncation(false);
    t.SetBPEFitInitialUnit(BPEInitialUnit::UnicodeCharacter);
    t.SetBPEFitPiecePolicy(ByteBPEPiecePolicy::LeadingSpaceV2);
    t.Train(docs, frequency, cap);
    return t;
}
}
TEST_CASE("BPE character initial alphabet is distinct from byte initialization") {
    const std::string accented = "\xc3\xa9";
    auto t = CharacterFit({accented + " " + accented, "a"}, 100, 7);
    REQUIRE(t.GetVocabulary().Size() == 7);
    REQUIRE(t.GetVocabulary().GetBPEAlphabetSize() == 3);
    REQUIRE(t.GetVocabulary().GetBPEInitialUnit() == BPEInitialUnit::UnicodeCharacter);
    REQUIRE(t.GetVocabulary().GetBPEMerges().empty());
    REQUIRE(t.Encode(accented).size() == 1);
    REQUIRE(t.Decode(t.Encode(accented + " a")) == accented + " a");
    REQUIRE(t.Decode(t.Encode("z")) == "[UNK]");
    REQUIRE(t.Encode("z").front() == 1);
    Tokenizer bytes(TokenizerType::ByteBPE);
    bytes.SetPadding(false); bytes.SetTruncation(false);
    bytes.Train({accented}, 100, 260);
    REQUIRE(bytes.Encode(accented).size() == 2);
    REQUIRE(bytes.GetVocabulary().GetBPEAlphabetSize() == 256);
    REQUIRE_THROWS(CharacterFit({"abc"}, 1, 6));
    auto reordered = CharacterFit({"a", accented + " " + accented}, 100, 7);
    REQUIRE(Artifact(reordered) == Artifact(t));
    auto empty = CharacterFit({}, 1, 4);
    REQUIRE(empty.GetVocabularySize() == 4);
    REQUIRE(empty.Encode("").empty());
    REQUIRE(empty.Decode(empty.Encode("a")) == "[UNK]");
}
TEST_CASE("Character BPE uses ranked merges with a manually specified oracle") {
    Vocabulary v;
    const std::vector<std::string> words{"[PAD]", "[UNK]", "[BOS]", "[EOS]",
        "a", "b", "\xc3\xa9", "ab", "ab\xc3\xa9"};
    REQUIRE(v.SetBPE(words, {{4,5,7}, {7,6,8}}, ByteBPEPiecePolicy::WhitespaceV1,
        BPEInitialUnit::UnicodeCharacter, 3));
    Tokenizer t(TokenizerType::ByteBPE);
    t.SetVocabulary(v); t.SetPadding(false); t.SetTruncation(false);
    REQUIRE(t.Encode("ab\xc3\xa9") == std::vector<int>{8});
    REQUIRE(t.Encode("aba") == std::vector<int>{7,4});
    REQUIRE(t.Decode({8}) == "ab\xc3\xa9");
    auto fitted = CharacterFit({"ab\xc3\xa9", "ab\xc3\xa9"}, 1, 9);
    REQUIRE(fitted.GetVocabulary().GetBPEMerges() == v.GetBPEMerges());
    REQUIRE(fitted.Encode("ab\xc3\xa9") == t.Encode("ab\xc3\xa9"));
}
TEST_CASE("Character BPE preserves multilingual text boundaries and artifact identity") {
    const std::string scalar = "\xf0\x9f\x98\x80";
    const std::string text = " caf\xc3\xa9 \xce\xb1\xce\xb2 " + scalar + "\t\r\n";
    const std::string boundary = std::string(254, 'a') + scalar + std::string(270, 'a');
    const std::string literals = "[PAD] [UNK] [BOS] [EOS]";
    const std::string nul("a\0b", 3);
    auto t = CharacterFit({text, text, boundary, literals, nul});
    for (const auto& sample : {text, boundary, literals, nul, std::string("")}) {
        const auto ids = t.Encode(sample);
        REQUIRE(t.Decode(ids) == sample);
        REQUIRE(std::all_of(ids.begin(), ids.end(), [](int id) {return id >= 4;}));
    }
    const auto saved = Artifact(t);
    REQUIRE(saved.starts_with("#cyxwiz-vocabulary-character-bpe-v1\nleading_space_v2\n"));
    Tokenizer loaded(TokenizerType::ByteBPE);
    loaded.SetPadding(false); loaded.SetTruncation(false);
    std::istringstream in(saved);
    REQUIRE(loaded.GetVocabulary().LoadFromStream(in));
    REQUIRE(Artifact(loaded) == saved);
    REQUIRE(loaded.Encode(text) == t.Encode(text));
    loaded.SetBPEFitInitialUnit(BPEInitialUnit::Byte); // Fitting choices cannot change saved IDs.
    REQUIRE(loaded.Encode(text) == t.Encode(text));
    for (const auto& token : t.GetVocabulary().GetWords()) REQUIRE(token.size() <= 256);
    t.SetBPEFitPiecePolicy(ByteBPEPiecePolicy::WhitespaceV1);
    t.Train({text}, 1, 200);
    REQUIRE(t.Decode(t.Encode(text)) == text);
    REQUIRE(Artifact(t).starts_with("#cyxwiz-vocabulary-character-bpe-v1\nwhitespace_v1\n"));
}
TEST_CASE("Character BPE rejects invalid UTF8 and preserves vocabulary on failed fit") {
    auto t = CharacterFit({"valid valid"});
    const auto saved = Artifact(t);
    for (const auto& bad : std::vector<std::string>{"\x80", "\xc0\xaf", "\xc2", "\xe0\x80\xaf",
            "\xed\xa0\x80", "\xf4\x90\x80\x80", "\xf0\x9f", "\xf5\x80\x80\x80"}) {
        REQUIRE_THROWS(t.Encode(bad));
        REQUIRE_THROWS(t.Train({bad}, 1, 50));
        REQUIRE(Artifact(t) == saved);
    }
    REQUIRE_THROWS(t.Train({"too many characters"}, 1, 5));
    REQUIRE(Artifact(t) == saved);
    int checks = 0;
    t.SetCancellationQuery([&] { return ++checks > 3; });
    REQUIRE_THROWS(t.Train({"different words repeated repeatedly"}, 1, 100));
    REQUIRE(Artifact(t) == saved);
    REQUIRE_THROWS(t.SetBPEFitInitialUnit(static_cast<BPEInitialUnit>(99)));
    Tokenizer word(TokenizerType::Word);
    REQUIRE_THROWS(word.SetBPEFitInitialUnit(BPEInitialUnit::UnicodeCharacter));
}
TEST_CASE("Character BPE rejects malformed persisted alphabets and merges atomically") {
    auto t = CharacterFit({"abab abab"});
    const auto saved = Artifact(t);
    const auto unchanged = [&] { REQUIRE(Artifact(t) == saved); };
    auto words = t.GetVocabulary().GetWords();
    auto merges = t.GetVocabulary().GetBPEMerges();
    const auto alphabet = t.GetVocabulary().GetBPEAlphabetSize();
    words[4] = "ab"; // Alphabet entries must be one scalar each.
    REQUIRE_FALSE(t.GetVocabulary().SetBPE(words, merges, ByteBPEPiecePolicy::LeadingSpaceV2,
        BPEInitialUnit::UnicodeCharacter, alphabet)); unchanged();
    words[4] = "\xed\xa0\x80";
    REQUIRE_FALSE(t.GetVocabulary().SetBPE(words, merges, ByteBPEPiecePolicy::LeadingSpaceV2,
        BPEInitialUnit::UnicodeCharacter, alphabet)); unchanged();
    for (const auto& bad : {std::string("#cyxwiz-vocabulary-character-bpe-v1\nother\n0\n4\n"),
            std::string("#cyxwiz-vocabulary-character-bpe-v1\nleading_space_v2\n9999999999999999999999\n"),
            saved + "unexpected\n", saved.substr(0, saved.size()/2)}) {
        std::istringstream in(bad);
        REQUIRE_FALSE(t.GetVocabulary().LoadFromStream(in)); unchanged();
    }
    words = t.GetVocabulary().GetWords();
    merges.front()[0] = static_cast<int>(words.size());
    REQUIRE_FALSE(t.GetVocabulary().SetBPE(words, merges, ByteBPEPiecePolicy::LeadingSpaceV2,
        BPEInitialUnit::UnicodeCharacter, alphabet)); unchanged();
}
