#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cyxwiz {

class Tensor;
struct ProbeResult;
struct TextTokenizerPackage;

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
