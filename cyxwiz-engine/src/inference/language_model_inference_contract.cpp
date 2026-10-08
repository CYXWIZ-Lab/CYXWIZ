#include "language_model_inference_contract.h"

#include "core/model_format.h"
#include "inference/text_inference_input.h"

#include <cyxwiz/tensor.h>
#include <cyxwiz/tokenizer.h>

#include <algorithm>
#include <cctype>
#include <sstream>
#include <utility>

namespace cyxwiz {

namespace {

constexpr const char* kCausalLanguageModelFamily = "causal_lm";
constexpr const char* kGenerationOutputContract = "Float32[1,seq,vocab]";

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

LanguageModelTokenizerIdentity ValidateLanguageModelTokenizerIdentity(
    const Tokenizer& selected,
    const Tokenizer* expected,
    std::string_view selected_model_artifact,
    std::string_view expected_model_artifact) {
    if (!expected) {
        return {true, false,
            "Tokenizer identity unverified: the active model has no prepared tokenizer metadata."};
    }
    const auto mismatch = [](const std::string& reason) {
        return LanguageModelTokenizerIdentity{false, false,
            "Tokenizer mismatch: " + reason +
            ". Select the tokenizer used with the active model. Equal vocabulary sizes do not imply equal token IDs."};
    };
    if (selected.GetType() != expected->GetType())
        return mismatch("tokenizer families differ");
    if (selected.GetLowercase() != expected->GetLowercase())
        return mismatch("lowercase settings differ");
    if (IsSentencePieceTokenizerType(selected.GetType())) {
        if (selected_model_artifact.empty() || expected_model_artifact.empty())
            return {false, false, "Cannot verify tokenizer identity: SentencePiece model bytes are missing."};
        if (selected_model_artifact != expected_model_artifact)
            return mismatch("SentencePiece model artifacts differ");
    } else {
        // Canonical persistence includes token order, special tokens, BPE merge
        // ranks, piece policy and initial alphabet. Filename/newline changes do
        // not cause a false mismatch. Work occurs only on an explicit UI action.
        std::ostringstream selected_artifact, expected_artifact;
        if (!selected.GetVocabulary().SaveToStream(selected_artifact) ||
            !expected->GetVocabulary().SaveToStream(expected_artifact))
            return {false, false, "Cannot verify tokenizer identity: vocabulary serialization failed."};
        if (selected_artifact.str() != expected_artifact.str())
            return mismatch("vocabulary token mapping or BPE encoding rules differ");
    }
    return {true, true, "Tokenizer identity matches the active model's prepared tokenizer."};
}

LanguageModelPackageContract ValidateLanguageModelPackageContract(
    const ProbeResult& probe,
    const TextTokenizerPackage* tokenizer_package,
    const std::string& package_path) {

    LanguageModelPackageContract contract;
    contract.package_path = package_path;
    contract.model_family = probe.model_family;
    contract.supports_generation = probe.supports_generation;
    contract.generation_output_contract = probe.generation_output_contract;
    contract.has_tokenizer = probe.has_tokenizer;
    contract.has_vocabulary = probe.has_vocabulary ||
        (tokenizer_package != nullptr && tokenizer_package->has_model_artifact);
    contract.max_sequence_length = probe.sequence_max_sequence_length;

    std::vector<std::string> issues;
    if (!probe.valid) {
        std::string issue = "model probe is invalid";
        if (!probe.error_message.empty()) {
            issue += ": " + probe.error_message;
        }
        AddIssue(issues, std::move(issue));
    }
    if (probe.model_family != kCausalLanguageModelFamily) {
        AddIssue(issues,
                 "model_family must be causal_lm for text generation");
    }
    if (!probe.supports_generation) {
        AddIssue(issues, "package does not declare generation support");
    }
    if (NormalizeContract(probe.generation_output_contract) !=
        kGenerationOutputContract) {
        AddIssue(issues,
                 "generation_output_contract must be Float32[1,seq,vocab]");
    }
    if (!probe.has_tokenizer) {
        AddIssue(issues, "package is missing tokenizer/config.json");
    }
    if (!probe.has_vocabulary &&
        (tokenizer_package == nullptr || !tokenizer_package->has_model_artifact)) {
        AddIssue(issues, "package is missing tokenizer/vocab.txt or tokenizer/model.spm");
    }

    if (tokenizer_package == nullptr || !tokenizer_package->tokenizer) {
        AddIssue(issues, "tokenizer package is not loaded");
    } else {
        contract.tokenizer_vocabulary_size =
            tokenizer_package->tokenizer->GetVocabularySize();
        contract.eos_token_id =
            static_cast<int64_t>(tokenizer_package->tokenizer->GetEosId());
        contract.max_sequence_length = static_cast<size_t>(
            std::max(0, tokenizer_package->tokenizer->GetMaxLength()));

        if (!tokenizer_package->has_vocabulary &&
            !tokenizer_package->has_model_artifact) {
            AddIssue(issues, "tokenizer package has no usable vocabulary or model artifact");
        }
        if (contract.tokenizer_vocabulary_size == 0) {
            AddIssue(issues, "tokenizer vocabulary is empty");
        }
        if (contract.eos_token_id < 0 ||
            static_cast<size_t>(contract.eos_token_id) >=
                contract.tokenizer_vocabulary_size) {
            AddIssue(issues, "tokenizer EOS token id is outside the vocabulary");
        }
    }

    FinishContract(contract.compatible, contract.error, issues);
    return contract;
}

LanguageModelPromptContract ValidateLanguageModelPromptIds(
    const std::vector<int64_t>& prompt_ids,
    size_t max_sequence_length) {

    LanguageModelPromptContract contract;
    contract.sequence_length = prompt_ids.size();

    std::vector<std::string> issues;
    if (prompt_ids.empty()) {
        AddIssue(issues, "prompt must encode to at least one token");
    }
    for (const int64_t token_id : prompt_ids) {
        if (token_id < 0) {
            AddIssue(issues, "prompt token ids must be non-negative");
            break;
        }
    }
    if (max_sequence_length > 0 && prompt_ids.size() > max_sequence_length) {
        AddIssue(issues, "prompt length exceeds tokenizer max sequence length");
    }

    FinishContract(contract.compatible, contract.error, issues);
    return contract;
}

LanguageModelRuntimeOutputContract ValidateLanguageModelRuntimeOutput(
    const Tensor& logits,
    size_t expected_sequence_length,
    size_t tokenizer_vocabulary_size) {

    LanguageModelRuntimeOutputContract contract;
    contract.output_shape = logits.Shape();
    if (contract.output_shape.size() == 3) {
        contract.batch_size = contract.output_shape[0];
        contract.sequence_length = contract.output_shape[1];
        contract.vocab_size = contract.output_shape[2];
    }

    std::vector<std::string> issues;
    if (expected_sequence_length == 0) {
        AddIssue(issues, "expected prompt sequence length must be greater than 0");
    }
    if (logits.GetDataType() != DataType::Float32) {
        AddIssue(issues, "model output must be Float32 logits");
    }
    if (contract.output_shape.size() != 3) {
        AddIssue(issues, "model output must have rank 3 [1, seq, vocab]");
    } else {
        if (contract.batch_size != 1) {
            AddIssue(issues, "model output batch size must be 1");
        }
        if (contract.sequence_length != expected_sequence_length) {
            AddIssue(issues,
                     "model output sequence length must match prompt length");
        }
        if (contract.vocab_size == 0) {
            AddIssue(issues, "model output vocab size must be greater than 0");
        }
        if (tokenizer_vocabulary_size > 0 &&
            contract.vocab_size < tokenizer_vocabulary_size) {
            AddIssue(issues,
                     "model output vocab size is smaller than tokenizer vocabulary");
        }
    }

    FinishContract(contract.compatible, contract.error, issues);
    return contract;
}

} // namespace cyxwiz
