#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cyxwiz {

class Tensor;
class Tokenizer;
struct ProbeResult;
struct TextTokenizerPackage;

struct BertEncoderPackageContract {
    bool compatible = false;
    std::string error;

    std::string package_path;
    std::string model_family;
    bool supports_bert_encoder = false;
    std::string task;
    std::string input_kind;
    std::string output_contract;
    bool has_attention_mask = false;
    bool requires_token_type_ids = false;
    bool has_tokenizer = false;
    bool has_vocabulary = false;

    size_t tokenizer_vocabulary_size = 0;
    size_t max_sequence_length = 0;
};

struct BertEncoderTextInputContract {
    bool compatible = false;
    std::string error;
    size_t batch_size = 1;
    size_t sequence_length = 0;
    bool has_attention_mask = false;
};

BertEncoderPackageContract ValidateBertEncoderPackageContract(
    const ProbeResult& probe,
    const TextTokenizerPackage* tokenizer_package,
    const std::string& package_path = {});

BertEncoderTextInputContract ValidateBertEncoderTextInputIds(
    const std::vector<int64_t>& token_ids,
    size_t max_sequence_length = 0,
    bool has_attention_mask = false);

std::vector<int64_t> EncodeTextTokenIdsForBertEncoder(
    const Tokenizer& tokenizer,
    const std::string& text);

std::vector<int64_t> BuildBertEncoderAttentionMask(
    const std::vector<int64_t>& token_ids,
    int64_t pad_token_id);

} // namespace cyxwiz
