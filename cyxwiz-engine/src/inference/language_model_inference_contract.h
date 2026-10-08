#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace cyxwiz {

class Tensor;
class Tokenizer;
struct ProbeResult;
struct TextTokenizerPackage;

// Semantic identity, independent of vocabulary filename and generation length.
// A missing expected tokenizer is explicitly unverified, never a verified match.
struct LanguageModelTokenizerIdentity {
    bool compatible = false;
    bool verified = false;
    std::string message;
};

LanguageModelTokenizerIdentity ValidateLanguageModelTokenizerIdentity(
    const Tokenizer& selected,
    const Tokenizer* expected,
    std::string_view selected_model_artifact = {},
    std::string_view expected_model_artifact = {});

struct LanguageModelPackageContract {
    bool compatible = false;
    std::string error;

    std::string package_path;
    std::string model_family;
    bool supports_generation = false;
    std::string generation_output_contract;
    bool has_tokenizer = false;
    bool has_vocabulary = false;

    size_t tokenizer_vocabulary_size = 0;
    size_t max_sequence_length = 0;
    int64_t eos_token_id = -1;
};

struct LanguageModelPromptContract {
    bool compatible = false;
    std::string error;
    size_t batch_size = 1;
    size_t sequence_length = 0;
};

struct LanguageModelRuntimeOutputContract {
    bool compatible = false;
    std::string error;
    std::vector<size_t> output_shape;
    size_t batch_size = 0;
    size_t sequence_length = 0;
    size_t vocab_size = 0;
};

LanguageModelPackageContract ValidateLanguageModelPackageContract(
    const ProbeResult& probe,
    const TextTokenizerPackage* tokenizer_package,
    const std::string& package_path = {});

LanguageModelPromptContract ValidateLanguageModelPromptIds(
    const std::vector<int64_t>& prompt_ids,
    size_t max_sequence_length = 0);

LanguageModelRuntimeOutputContract ValidateLanguageModelRuntimeOutput(
    const Tensor& logits,
    size_t expected_sequence_length,
    size_t tokenizer_vocabulary_size = 0);

} // namespace cyxwiz
