#include <catch2/catch_test_macros.hpp>

#include "../../cyxwiz-engine/src/inference/text_inference_input.h"
#include <cyxwiz/tokenizer.h>

#ifndef CYXWIZ_HAS_SENTENCEPIECE
#define CYXWIZ_HAS_SENTENCEPIECE 0
#endif

#include <sstream>
#include <string>
#include <vector>

TEST_CASE("Text inference input loads packaged tokenizer assets",
          "[inference][text]") {
    const std::string config =
        R"({"effective":{"tokenizer_type":"1","max_length":"4","lowercase":"true"}})";
    const std::string vocab =
        "[PAD]\n[UNK]\n[BOS]\n[EOS]\nhello\nworld\n";

    cyxwiz::TextTokenizerPackage package;
    std::string error;
    REQUIRE(cyxwiz::LoadTextTokenizerPackage(config, vocab, package, error));
    REQUIRE(package.tokenizer != nullptr);
    REQUIRE(package.has_vocabulary);
    REQUIRE(package.tokenizer->GetMaxLength() == 4);
    REQUIRE(package.tokenizer->GetLowercase());

    const auto input =
        cyxwiz::EncodeTextForInference(*package.tokenizer, "Hello world");

    REQUIRE(input.size() == 4);
    REQUIRE(input[0] == 4.0f);
    REQUIRE(input[1] == 5.0f);
    REQUIRE(input[2] == 0.0f);
    REQUIRE(input[3] == 0.0f);
}

TEST_CASE("Text generation input strips trailing padding tokens",
          "[inference][text][generation]") {
    const std::string config =
        R"({"effective":{"tokenizer_type":"1","max_length":"4","lowercase":"true"}})";
    const std::string vocab =
        "[PAD]\n[UNK]\n[BOS]\n[EOS]\nhello\nworld\n";

    cyxwiz::TextTokenizerPackage package;
    std::string error;
    REQUIRE(cyxwiz::LoadTextTokenizerPackage(config, vocab, package, error));

    const auto token_ids =
        cyxwiz::EncodeTextTokenIdsForGeneration(*package.tokenizer,
                                                "Hello world");

    REQUIRE(token_ids == std::vector<int64_t>{4, 5});
}

TEST_CASE("Text generation output decodes generated token ids",
          "[inference][text][generation]") {
    const std::string config =
        R"({"effective":{"tokenizer_type":"1","max_length":"4","lowercase":"true"}})";
    const std::string vocab =
        "[PAD]\n[UNK]\n[BOS]\n[EOS]\nhello\nworld\n";

    cyxwiz::TextTokenizerPackage package;
    std::string error;
    REQUIRE(cyxwiz::LoadTextTokenizerPackage(config, vocab, package, error));

    const std::string text = cyxwiz::DecodeGeneratedTokenIds(
        *package.tokenizer,
        {4, 5, 3, 0});

    REQUIRE(text == "hello world");
}

TEST_CASE("Text inference input rejects invalid tokenizer config",
          "[inference][text]") {
    cyxwiz::TextTokenizerPackage package;
    std::string error;

    REQUIRE_FALSE(cyxwiz::LoadTextTokenizerPackage(
        "{not-json",
        "[PAD]\n[UNK]\n",
        package,
        error));
    REQUIRE(!error.empty());
}

TEST_CASE("Packaged character vocabulary preserves control bytes and UTF-8",
          "[inference][text][tokenizer]") {
    cyxwiz::Tokenizer source(cyxwiz::TokenizerType::Character);
    source.SetLowercase(false);
    source.SetPadding(false);
    source.SetTruncation(false);
    const std::string text = std::string("A\nB\t") + "\xC3\xA9";
    source.Train({text}, 1, -1);
    std::ostringstream artifact;
    REQUIRE(source.GetVocabulary().SaveToStream(artifact));
    REQUIRE(cyxwiz::ParseVocabularyWords(artifact.str()) == source.GetVocabulary().GetWords());
    cyxwiz::TextTokenizerPackage package;
    std::string error;
    REQUIRE(cyxwiz::LoadTextTokenizerPackage(
        R"({"effective":{"tokenizer_type":"2","max_length":"32","lowercase":"false"}})",
        artifact.str(), package, error));
    const auto ids = cyxwiz::EncodeTextTokenIdsForGeneration(*package.tokenizer, text);
    const auto expected = source.Encode(text);
    REQUIRE(ids == std::vector<int64_t>(expected.begin(), expected.end()));
    REQUIRE(cyxwiz::DecodeGeneratedTokenIds(*package.tokenizer, ids) == text);
}

TEST_CASE("WordPiece package preserves continuation pieces", "[inference][text][tokenizer]") {
    cyxwiz::Vocabulary vocab;
    vocab.SetVocabulary({"play", "##ing", "##ed"});
    std::ostringstream saved;
    REQUIRE(vocab.SaveToStream(saved));

    cyxwiz::TextTokenizerPackage package;
    std::string error;
    REQUIRE(cyxwiz::LoadTextTokenizerPackage(
        R"({"effective":{"method":"wordpiece","max_length":"16","lowercase":"true"}})",
        saved.str(), package, error));
    REQUIRE(package.tokenizer->GetType() == cyxwiz::TokenizerType::WordPiece);

    const auto ids = cyxwiz::EncodeTextTokenIdsForGeneration(*package.tokenizer,
                                                             "PLAYING played");
    REQUIRE(cyxwiz::DecodeGeneratedTokenIds(*package.tokenizer, ids) ==
            "playing played");
}

TEST_CASE("Packaged SentencePiece tokenizer requires model artifact and validates provider", "[inference][text][tokenizer]") {
    REQUIRE(cyxwiz::IsSentencePieceTokenizerType(cyxwiz::TokenizerType::SentencePieceBPE));
    REQUIRE(cyxwiz::IsSentencePieceTokenizerType(cyxwiz::TokenizerType::SentencePieceUnigram));
#if CYXWIZ_HAS_SENTENCEPIECE
    REQUIRE(cyxwiz::IsSentencePieceTokenizerAvailable());
#else
    REQUIRE_FALSE(cyxwiz::IsSentencePieceTokenizerAvailable());
#endif
    const std::string expected = cyxwiz::SentencePieceTokenizerUnavailableMessage();

    cyxwiz::TextTokenizerPackage package;
    std::string error;
    REQUIRE_FALSE(cyxwiz::LoadTextTokenizerPackage(
        R"({"effective":{"method":"sentencepiece_bpe","max_length":"16"}})",
        "", package, error));
    REQUIRE(error == "SentencePiece tokenizer package requires tokenizer/model.spm");
    REQUIRE_FALSE(cyxwiz::LoadTextTokenizerPackage(
        R"({"effective":{"method":"sentencepiece_bpe","max_length":"16"}})",
        "", "fake spm model bytes", package, error));
#if CYXWIZ_HAS_SENTENCEPIECE
    REQUIRE(error.find("Invalid SentencePiece model artifact") != std::string::npos);
#else
    REQUIRE(error == expected);
#endif
    REQUIRE_FALSE(cyxwiz::LoadTextTokenizerPackage(
        R"({"effective":{"tokenizer_type":6,"max_length":"16"}})",
        "", package, error));
    REQUIRE(error == "SentencePiece tokenizer package requires tokenizer/model.spm");
    REQUIRE_FALSE(cyxwiz::LoadTextTokenizerPackage(
        R"({"effective":{"tokenizer_type":6,"max_length":"16"}})",
        "", "fake spm model bytes", package, error));
#if CYXWIZ_HAS_SENTENCEPIECE
    REQUIRE(error.find("Invalid SentencePiece model artifact") != std::string::npos);
#else
    REQUIRE(error == expected);
#endif
}

TEST_CASE("Packaged tokenizer rejects malformed versioned vocabulary",
          "[inference][text][tokenizer]") {
    cyxwiz::TextTokenizerPackage package;
    std::string error;
    REQUIRE_FALSE(cyxwiz::LoadTextTokenizerPackage("{}",
        "#cyxwiz-vocabulary-hex-v1\n2\n61\n", package, error));
    REQUIRE(!error.empty());
    REQUIRE_FALSE(package.has_vocabulary);
}

TEST_CASE("Byte BPE package preserves merges and strategy", "[inference][text][bpe]") {
    cyxwiz::Tokenizer fitted(cyxwiz::TokenizerType::ByteBPE);
    fitted.Train({"abab AB AB"},1,280);
    std::ostringstream saved;
    REQUIRE(fitted.GetVocabulary().SaveToStream(saved));
    cyxwiz::TextTokenizerPackage package;
    std::string error;
    REQUIRE(cyxwiz::LoadTextTokenizerPackage(R"({"tokenizer_type":3,"max_length":32})", saved.str(), package, error));
    REQUIRE(package.tokenizer->GetType() == cyxwiz::TokenizerType::ByteBPE);
    REQUIRE(package.tokenizer->GetVocabulary().GetBPEMerges() == fitted.GetVocabulary().GetBPEMerges());
    REQUIRE(cyxwiz::DecodeGeneratedTokenIds(*package.tokenizer,
        cyxwiz::EncodeTextTokenIdsForGeneration(*package.tokenizer,"abab AB\n")) == "abab AB\n");
    REQUIRE_FALSE(cyxwiz::LoadTextTokenizerPackage(R"({"tokenizer_type":1})", saved.str(), package, error));
    REQUIRE_FALSE(cyxwiz::LoadTextTokenizerPackage(R"({"tokenizer_type":3,"lowercase":true})", saved.str(), package, error));
    REQUIRE_FALSE(cyxwiz::LoadTextTokenizerPackage(R"({"method":"byte_bpe"})", "", package, error));
    REQUIRE_FALSE(cyxwiz::LoadTextTokenizerPackage(R"({"method":"byte_bpe"})", "[PAD]\n[UNK]\n[BOS]\n[EOS]\nab\n", package, error));
}
