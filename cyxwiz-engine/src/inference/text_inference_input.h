#pragma once

#include <memory>
#include <cstdint>
#include <string>
#include <vector>

namespace cyxwiz {

class Tokenizer;

struct TextTokenizerPackage {
    TextTokenizerPackage();
    ~TextTokenizerPackage();

    TextTokenizerPackage(TextTokenizerPackage&&) noexcept;
    TextTokenizerPackage& operator=(TextTokenizerPackage&&) noexcept;

    TextTokenizerPackage(const TextTokenizerPackage&) = delete;
    TextTokenizerPackage& operator=(const TextTokenizerPackage&) = delete;

    std::unique_ptr<Tokenizer> tokenizer;
    bool has_vocabulary = false;
};

bool LoadTextTokenizerPackage(
    const std::string& config_json,
    const std::string& vocab_text,
    TextTokenizerPackage& out,
    std::string& error);

std::vector<float> EncodeTextForInference(
    const Tokenizer& tokenizer,
    const std::string& text);

std::vector<int64_t> EncodeTextTokenIdsForGeneration(
    const Tokenizer& tokenizer,
    const std::string& text);

std::string DecodeGeneratedTokenIds(
    const Tokenizer& tokenizer,
    const std::vector<int64_t>& token_ids);

} // namespace cyxwiz
