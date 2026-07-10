#include "bert_encoder_inference_contract.h"

#include "core/model_format.h"
#include "inference/text_inference_input.h"

#include <cyxwiz/tokenizer.h>

#include <algorithm>
#include <cctype>
#include <sstream>
#include <utility>

namespace cyxwiz {

namespace {

constexpr const char* kBertEncoderFamily = "bert_encoder";
constexpr const char* kBertTokenInputKind = "token_ids";
constexpr const char* kBertSequenceClassifierTask = "sequence_classification";
constexpr const char* kBertTokenClassifierTask = "token_classification";
constexpr const char* kBertSequenceClassifierOutput = "Float32[batch,classes]";
constexpr const char* kBertTokenClassifierOutput = "Float32[batch,seq,classes]";

std::string NormalizeContract(std::string value) {
    value.erase(
        std::remove_if(
            value.begin(),
            value.end(),
            [](unsigned char ch) { return std::isspace(ch) != 0; }),
        value.end());
    return value;
}

void AddIssue(std::vector<std::string>& issues, std::string issue) {
    issues.push_back(std::move(issue));
}

std::string JoinIssues(const std::vector<std::string>& issues) {
    std::ostringstream stream;
    for (size_t i = 0; i < issues.size(); ++i) {
        if (i > 0) {
            stream << "; ";
        }
        stream << issues[i];
    }
    return stream.str();
}

void FinishContract(bool& compatible,
                    std::string& error,
                    const std::vector<std::string>& issues) {
    compatible = issues.empty();
    error = compatible ? std::string{} : JoinIssues(issues);
}

} // namespace

BertEncoderPackageContract ValidateBertEncoderPackageContract(
    const ProbeResult& probe,
    const TextTokenizerPackage* tokenizer_package,
    const std::string& package_path) {

    BertEncoderPackageContract contract;
    contract.package_path = package_path;
    contract.model_family = probe.model_family;
    contract.supports_bert_encoder = probe.supports_bert_encoder;
    contract.task = probe.bert_encoder_task;
    contract.input_kind = probe.bert_encoder_input_kind;
    contract.output_contract = probe.bert_encoder_output_contract;
    contract.has_attention_mask = probe.bert_encoder_has_attention_mask;
    contract.requires_token_type_ids = probe.bert_encoder_requires_token_type_ids;
    contract.has_tokenizer = probe.has_tokenizer;
    contract.has_vocabulary = probe.has_vocabulary;
    contract.max_sequence_length = probe.sequence_max_sequence_length;

    std::vector<std::string> issues;
    if (!probe.valid) {
        std::string issue = "model probe is invalid";
        if (!probe.error_message.empty()) {
            issue += ": " + probe.error_message;
        }
        AddIssue(issues, std::move(issue));
    }
    if (probe.model_family != kBertEncoderFamily) {
        AddIssue(issues,
                 "model_family must be bert_encoder for BERT encoder inference");
    }
    if (!probe.supports_bert_encoder) {
        AddIssue(issues, "package does not declare BERT encoder support");
    }
    if (probe.bert_encoder_input_kind != kBertTokenInputKind) {
        AddIssue(issues,
                 "bert_encoder_input_kind must be token_ids for raw text inference");
    }
    if (probe.bert_encoder_requires_token_type_ids) {
        AddIssue(issues,
                 "BERT token_type/segment IDs are not supported yet");
    }
    if (probe.bert_encoder_task == kBertSequenceClassifierTask) {
        if (NormalizeContract(probe.bert_encoder_output_contract) !=
            kBertSequenceClassifierOutput) {
            AddIssue(issues,
                     "bert_encoder_output_contract must be Float32[batch,classes]");
        }
    } else if (probe.bert_encoder_task == kBertTokenClassifierTask) {
        if (NormalizeContract(probe.bert_encoder_output_contract) !=
            kBertTokenClassifierOutput) {
            AddIssue(issues,
                     "bert_encoder_output_contract must be Float32[batch,seq,classes]");
        }
    } else {
        AddIssue(issues,
                 "bert_encoder_task must be sequence_classification or token_classification");
    }
    if (!probe.has_tokenizer) {
        AddIssue(issues, "package is missing tokenizer/config.json");
    }
    if (!probe.has_vocabulary) {
        AddIssue(issues, "package is missing tokenizer/vocab.txt");
    }

    if (tokenizer_package == nullptr || !tokenizer_package->tokenizer) {
        AddIssue(issues, "tokenizer package is not loaded");
    } else {
        const auto& vocabulary =
            tokenizer_package->tokenizer->GetVocabulary();
        contract.tokenizer_vocabulary_size = vocabulary.Size();
        contract.max_sequence_length = static_cast<size_t>(
            std::max(0, tokenizer_package->tokenizer->GetMaxLength()));

        if (!tokenizer_package->has_vocabulary) {
            AddIssue(issues, "tokenizer package has no usable vocabulary");
        }
        if (contract.tokenizer_vocabulary_size == 0) {
            AddIssue(issues, "tokenizer vocabulary is empty");
        }
    }

    FinishContract(contract.compatible, contract.error, issues);
    return contract;
}

BertEncoderTextInputContract ValidateBertEncoderTextInputIds(
    const std::vector<int64_t>& token_ids,
    size_t max_sequence_length,
    bool has_attention_mask) {

    BertEncoderTextInputContract contract;
    contract.sequence_length = token_ids.size();
    contract.has_attention_mask = has_attention_mask;

    std::vector<std::string> issues;
    if (token_ids.empty()) {
        AddIssue(issues, "text input must encode to at least one token");
    }
    for (const int64_t token_id : token_ids) {
        if (token_id < 0) {
            AddIssue(issues, "BERT text token ids must be non-negative");
            break;
        }
    }
    if (max_sequence_length > 0 && token_ids.size() > max_sequence_length) {
        AddIssue(issues,
                 "BERT text input exceeds tokenizer max sequence length");
    }

    FinishContract(contract.compatible, contract.error, issues);
    return contract;
}

std::vector<int64_t> EncodeTextTokenIdsForBertEncoder(
    const Tokenizer& tokenizer,
    const std::string& text) {

    const auto encoded_ids = tokenizer.Encode(text);
    std::vector<int64_t> token_ids;
    token_ids.reserve(encoded_ids.size());
    for (const int token_id : encoded_ids) {
        token_ids.push_back(static_cast<int64_t>(token_id));
    }
    return token_ids;
}

std::vector<int64_t> BuildBertEncoderAttentionMask(
    const std::vector<int64_t>& token_ids,
    int64_t pad_token_id) {

    std::vector<int64_t> mask;
    mask.reserve(token_ids.size());
    for (const int64_t token_id : token_ids) {
        mask.push_back(token_id == pad_token_id ? 0 : 1);
    }
    return mask;
}

} // namespace cyxwiz
