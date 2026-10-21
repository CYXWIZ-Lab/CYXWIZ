#include <catch2/catch_test_macros.hpp>
#include <cyxwiz/tokenizer.h>
#include <sstream>
#include <random>
#include <algorithm>

using namespace cyxwiz;
static Tokenizer MakeBPE(const std::vector<std::string>& docs, int cap = 300, int min_freq = 1) {
    Tokenizer t(TokenizerType::ByteBPE);
    t.SetPadding(false); t.SetTruncation(false);
    t.Train(docs, min_freq, cap);
    return t;
}

TEST_CASE("BPE learns ranked merges with deterministic byte IDs") {
    auto t = MakeBPE({"abab", "abab"}, 262);
    REQUIRE(t.GetVocabulary().Size() == 262);
    REQUIRE(t.Encode("abab") == std::vector<int>{261});
    REQUIRE(t.GetVocabulary().GetBPEMerges() ==
            std::vector<std::array<int,3>>{{101,102,260},{260,260,261}});
    auto reversed = MakeBPE({"ac ab", "abab"});
    auto other = MakeBPE({"abab", "ac ab"});
    REQUIRE(reversed.GetVocabulary().GetWords() == other.GetVocabulary().GetWords());
    REQUIRE(reversed.GetVocabulary().GetBPEMerges() == other.GetVocabulary().GetBPEMerges());
}

TEST_CASE("BPE preserves unseen bytes case spaces UTF8 and literal special spellings") {
    auto t = MakeBPE({"AB ab ab [PAD] [UNK] [BOS] [EOS]"}, 350);
    std::string bytes;
    for (int i=0;i<256;++i) bytes.push_back(static_cast<char>(i));
    for (const auto& text : std::vector<std::string>{bytes, "A\nB\r\t C", "", "caf\xc3\xa9", "[PAD][UNK][BOS][EOS]", std::string(4097,'a')}) {
        const auto ids = t.Encode(text);
        REQUIRE(t.Decode(ids) == text);
        REQUIRE(std::all_of(ids.begin(),ids.end(),[](int id){return id>=4;}));
    }
    t.SetAddBos(true); t.SetAddEos(true); t.SetPadding(true); t.SetMaxLength(30);
    REQUIRE(t.Decode(t.Encode("AB ab")) == "AB ab");
}

TEST_CASE("BPE pieces do not cross whitespace or document boundaries") {
    auto t = MakeBPE({"a", "b", "a b"}, 300);
    REQUIRE(t.GetVocabulary().Size() == 260);
    auto cap = MakeBPE({"abab"}, 260);
    REQUIRE(cap.Encode("ab") == std::vector<int>{101,102});
    auto rare = MakeBPE({"ab"},300,2);
    REQUIRE(rare.GetVocabulary().Size() == 260);
    REQUIRE_THROWS(MakeBPE({"ab"},259));
    REQUIRE_THROWS(MakeBPE({"ab"},300,0));
}

TEST_CASE("BPE artifact preserves merge order and rejects malformed artifacts atomically") {
    auto original = MakeBPE({"abab ac [PAD]"});
    std::ostringstream output;
    REQUIRE(original.GetVocabulary().SaveToStream(output));
    const auto artifact=output.str();
    Tokenizer restored(TokenizerType::ByteBPE);
    restored.SetPadding(false); restored.SetTruncation(false);
    std::istringstream input(artifact);
    REQUIRE(restored.GetVocabulary().LoadFromStream(input));
    REQUIRE(restored.Encode("abab ac") == original.Encode("abab ac"));
    REQUIRE(restored.GetVocabulary().GetBPEMerges() == original.GetVocabulary().GetBPEMerges());
    auto words=restored.GetVocabulary().GetWords();
    auto merges=restored.GetVocabulary().GetBPEMerges();
    auto invalid=merges; invalid[0][0]=999999;
    REQUIRE_FALSE(restored.GetVocabulary().SetByteBPE(words,invalid));
    REQUIRE(restored.GetVocabulary().GetBPEMerges() == merges);
    invalid=merges; invalid.push_back(merges.front());
    REQUIRE_FALSE(restored.GetVocabulary().SetByteBPE(words,invalid));
    for (auto bad : {artifact+"extra\n", artifact.substr(0,artifact.size()-4), std::string("#cyxwiz-vocabulary-byte-bpe-v2\n")}) {
        std::istringstream broken(bad);
        REQUIRE_FALSE(restored.GetVocabulary().LoadFromStream(broken));
        REQUIRE(restored.GetVocabulary().GetBPEMerges() == merges);
    }
    REQUIRE_THROWS(restored.GetVocabulary().AddWord("new"));
    Tokenizer wrong(TokenizerType::Word); wrong.SetVocabulary(original.GetVocabulary());
    REQUIRE_THROWS(wrong.Encode("abab"));
    REQUIRE_THROWS(wrong.Decode({260}));
    restored.SetLowercase(true);
    REQUIRE_THROWS(restored.Encode("ab"));
}

TEST_CASE("BPE cancellation leaves the previous vocabulary intact") {
    auto t=MakeBPE({"ab ab"});
    auto words=t.GetVocabulary().GetWords();
    int checks=0;
    t.SetCancellationQuery([&checks]{return ++checks>4;});
    REQUIRE_THROWS(t.Train({"abab", "abcabc", "hello world"},1,300));
    REQUIRE(t.GetVocabulary().GetWords() == words);
}

TEST_CASE("BPE rank encoder agrees with independent sequential merge replay") {
    std::mt19937 rng(53);
    std::vector<std::string> docs;
    for(int n=0;n<80;++n) {
        std::string doc;
        for(int j=0;j<40;++j) doc.push_back(static_cast<char>('a'+rng()%5));
        docs.push_back(doc);
    }
    auto t=MakeBPE(docs,340);
    for(const auto& doc:docs) {
        std::vector<int> reference;
        for(unsigned char c:doc) reference.push_back(c+4);
        for(const auto& m:t.GetVocabulary().GetBPEMerges()) {
            std::vector<int> next;
            for(size_t i=0;i<reference.size();++i) {
                if(i+1<reference.size() && reference[i]==m[0] && reference[i+1]==m[1]) {
                    next.push_back(m[2]); ++i;
                } else next.push_back(reference[i]);
            }
            reference=std::move(next);
        }
        REQUIRE(t.Encode(doc) == reference);
        REQUIRE(t.Decode(reference) == doc);
    }
}
