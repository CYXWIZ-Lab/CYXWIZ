#include "text_inference_input.h"

#include <cyxwiz/tokenizer.h>
#include <nlohmann/json.hpp>

#include <sstream>

namespace cyxwiz {

namespace {

using json = nlohmann::json;

TokenizerType ParseTokenizerType(const json& config) {
    const auto effective = config.contains("effective") && config["effective"].is_object()
        ? config["effective"]
        : config;

    if (effective.contains("method") && effective["method"].is_string()) {
        const std::string method = effective["method"].get<std::string>();
        if (method == "whitespace") return TokenizerType::Whitespace;
        if (method == "character") return TokenizerType::Character;
        return TokenizerType::Word;
    }

    if (effective.contains("tokenizer_type")) {
        int value = 1;
        if (effective["tokenizer_type"].is_string()) {
            try {
                value = std::stoi(effective["tokenizer_type"].get<std::string>());
            } catch (...) {
                value = 1;
            }
        } else if (effective["tokenizer_type"].is_number_integer()) {
            value = effective["tokenizer_type"].get<int>();
        }

        if (value == 0) return TokenizerType::Whitespace;
        if (value == 2) return TokenizerType::Character;
    }

    return TokenizerType::Word;
}

int ReadIntConfig(const json& effective, const char* key, int fallback) {
    if (!effective.contains(key)) {
        return fallback;
    }
    if (effective[key].is_number_integer()) {
        return effective[key].get<int>();
    }
    if (effective[key].is_string()) {
        try {
            return std::stoi(effective[key].get<std::string>());
        } catch (...) {
            return fallback;
        }
    }
    return fallback;
}

bool ReadBoolConfig(const json& effective, const char* key, bool fallback) {
    if (!effective.contains(key)) {
        return fallback;
    }
    if (effective[key].is_boolean()) {
        return effective[key].get<bool>();
    }
    if (effective[key].is_string()) {
        const std::string value = effective[key].get<std::string>();
        if (value == "true" || value == "1") return true;
        if (value == "false" || value == "0") return false;
    }
    return fallback;
}

std::vector<std::string> ParseVocabularyWordsImpl(const std::string& vocab_text) {
    // Parse one vocabulary token per line, trimming CRLF line endings.
    std::vector<std::string> words;
    std::istringstream stream(vocab_text);
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            words.push_back(line);
        }
    }
    return words;
}

} // namespace

std::vector<std::string> ParseVocabularyWords(const std::string& vocab_text) {
    return ParseVocabularyWordsImpl(vocab_text);
}

TextTokenizerPackage::TextTokenizerPackage() = default;
TextTokenizerPackage::~TextTokenizerPackage() = default;
TextTokenizerPackage::TextTokenizerPackage(TextTokenizerPackage&&) noexcept = default;
TextTokenizerPackage& TextTokenizerPackage::operator=(TextTokenizerPackage&&) noexcept = default;

bool LoadTextTokenizerPackage(
    const std::string& config_json,
    const std::string& vocab_text,
    TextTokenizerPackage& out,
    std::string& error) {

    out = TextTokenizerPackage{};
    error.clear();

    json tokenizer_config = json::object();
    if (!config_json.empty()) {
        try {
            tokenizer_config = json::parse(config_json);
        } catch (const json::exception& e) {
            error = std::string("Invalid tokenizer config JSON: ") + e.what();
            return false;
        }
    }

    auto tokenizer = std::make_unique<Tokenizer>(
        ParseTokenizerType(tokenizer_config));

    const auto effective =
        tokenizer_config.contains("effective") &&
        tokenizer_config["effective"].is_object()
            ? tokenizer_config["effective"]
            : tokenizer_config;

    tokenizer->SetLowercase(ReadBoolConfig(effective, "lowercase", true));
    tokenizer->SetMaxLength(ReadIntConfig(effective, "max_length", 512));
    tokenizer->SetPadding(true);
    tokenizer->SetTruncation(true);

    bool has_vocabulary = false;
    if (!vocab_text.empty()) {
        const auto words = ParseVocabularyWords(vocab_text);
        if (!words.empty()) {
            tokenizer->GetVocabulary().SetVocabulary(words);
            has_vocabulary = true;
        }
    }

    out.tokenizer = std::move(tokenizer);
    out.has_vocabulary = has_vocabulary;
    return true;
}

std::vector<float> EncodeTextForInference(
    const Tokenizer& tokenizer,
    const std::string& text) {

    const auto token_ids = tokenizer.Encode(text);
    std::vector<float> input_data;
    input_data.reserve(token_ids.size());
    for (const int token_id : token_ids) {
        input_data.push_back(static_cast<float>(token_id));
    }
    return input_data;
}

std::vector<int64_t> EncodeTextTokenIdsForGeneration(
    const Tokenizer& tokenizer,
    const std::string& text) {

    const auto encoded_ids = tokenizer.Encode(text);
    std::vector<int64_t> token_ids;
    token_ids.reserve(encoded_ids.size());
    for (const int token_id : encoded_ids) {
        token_ids.push_back(static_cast<int64_t>(token_id));
    }

    const int pad_id = tokenizer.GetVocabulary().PadIndex();
    while (!token_ids.empty() && token_ids.back() == pad_id) {
        token_ids.pop_back();
    }
    if (token_ids.empty()) {
        token_ids.push_back(static_cast<int64_t>(tokenizer.GetVocabulary().UnkIndex()));
    }

    return token_ids;
}

std::string DecodeGeneratedTokenIds(
    const Tokenizer& tokenizer,
    const std::vector<int64_t>& token_ids) {

    std::vector<int> decode_ids;
    decode_ids.reserve(token_ids.size());
    for (const int64_t token_id : token_ids) {
        decode_ids.push_back(static_cast<int>(token_id));
    }
    return tokenizer.Decode(decode_ids);
}

} // namespace cyxwiz
