#include <catch2/catch_test_macros.hpp>

#include "../../cyxwiz-engine/src/inference/text_inference_input.h"
#include <cyxwiz/tokenizer.h>

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
