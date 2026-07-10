#include "inference/bert_encoder_inference_contract.h"
#include "inference/language_model_inference_contract.h"

#include "core/model_format.h"
#include "inference/text_inference_input.h"

#include <cyxwiz/tensor.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

cyxwiz::ProbeResult MakeCausalLanguageModelProbe() {
    cyxwiz::ProbeResult probe;
    probe.valid = true;
    probe.format = cyxwiz::ModelFormat::CyxModel;
    probe.model_family = "causal_lm";
    probe.supports_generation = true;
    probe.generation_output_contract = "Float32 [1, seq, vocab]";
    probe.has_tokenizer = true;
    probe.has_vocabulary = true;
    probe.sequence_max_sequence_length = 6;
    return probe;
}

cyxwiz::ProbeResult MakeBertEncoderProbe() {
    cyxwiz::ProbeResult probe;
    probe.valid = true;
    probe.format = cyxwiz::ModelFormat::CyxModel;
    probe.model_family = "bert_encoder";
    probe.supports_bert_encoder = true;
    probe.bert_encoder_task = "sequence_classification";
    probe.bert_encoder_input_kind = "token_ids";
    probe.bert_encoder_output_contract = "Float32 [batch, classes]";
    probe.bert_encoder_has_attention_mask = true;
    probe.has_tokenizer = true;
    probe.has_vocabulary = true;
    probe.sequence_max_sequence_length = 6;
    return probe;
}

cyxwiz::TextTokenizerPackage LoadFixtureTokenizer() {
    cyxwiz::TextTokenizerPackage tokenizer_package;
    std::string error;
    const bool loaded = cyxwiz::LoadTextTokenizerPackage(
        R"({"method":"word","lowercase":true,"max_length":6})",
        "[PAD]\n[UNK]\nhello\nworld\n",
        tokenizer_package,
        error);
    Check(loaded, "fixture tokenizer should load: " + error);
    return tokenizer_package;
}

void TestPackageContractAcceptsBertEncoder() {
    auto tokenizer_package = LoadFixtureTokenizer();
    const auto contract = cyxwiz::ValidateBertEncoderPackageContract(
        MakeBertEncoderProbe(),
        &tokenizer_package,
        "bert_fixture.cyxmodel");

    Check(contract.compatible,
          "BERT encoder package contract should be compatible: " +
              contract.error);
    Check(contract.package_path == "bert_fixture.cyxmodel",
          "BERT package path should be surfaced");
    Check(contract.model_family == "bert_encoder",
          "BERT model family should be surfaced");
    Check(contract.task == "sequence_classification",
          "BERT task should be surfaced");
    Check(contract.input_kind == "token_ids",
          "BERT input kind should be surfaced");
    Check(contract.output_contract == "Float32 [batch, classes]",
          "BERT output contract should be surfaced");
    Check(contract.has_attention_mask,
          "BERT attention-mask capability should be surfaced");
    Check(contract.tokenizer_vocabulary_size == 6,
          "BERT tokenizer vocabulary size should include special tokens");
    Check(contract.max_sequence_length == 6,
          "BERT max sequence length should come from tokenizer config");
}

void TestPackageContractRejectsMissingBertTokenizer() {
    auto probe = MakeBertEncoderProbe();
    probe.has_tokenizer = false;
    probe.has_vocabulary = false;

    const auto contract =
        cyxwiz::ValidateBertEncoderPackageContract(probe, nullptr);

    Check(!contract.compatible,
          "missing BERT tokenizer package should fail contract");
    Check(contract.error.find("tokenizer/config.json") != std::string::npos,
          "missing BERT tokenizer config should be named in error");
    Check(contract.error.find("tokenizer/vocab.txt") != std::string::npos,
          "missing BERT tokenizer vocab should be named in error");
}

void TestPackageContractRejectsBertTokenTypeIds() {
    auto tokenizer_package = LoadFixtureTokenizer();
    auto probe = MakeBertEncoderProbe();
    probe.bert_encoder_requires_token_type_ids = true;

    const auto contract = cyxwiz::ValidateBertEncoderPackageContract(
        probe,
        &tokenizer_package);

    Check(!contract.compatible,
          "BERT token_type IDs should fail closed until supported");
    Check(contract.error.find("token_type/segment IDs") != std::string::npos,
          "BERT token_type failure should be named");
}

void TestBertTextInputContractAndMask() {
    auto tokenizer_package = LoadFixtureTokenizer();
    const auto token_ids = cyxwiz::EncodeTextTokenIdsForBertEncoder(
        *tokenizer_package.tokenizer,
        "hello world");
    const auto contract = cyxwiz::ValidateBertEncoderTextInputIds(
        token_ids,
        6,
        true);

    Check(contract.compatible,
          "encoded BERT text input should pass: " + contract.error);
    Check(contract.batch_size == 1,
          "BERT text input should describe one batch row");
    Check(contract.sequence_length == token_ids.size(),
          "BERT text input should surface encoded sequence length");
    Check(contract.has_attention_mask,
          "BERT text input should surface requested attention mask");

    const auto too_long = cyxwiz::ValidateBertEncoderTextInputIds(
        {1, 2, 3},
        2,
        false);
    Check(!too_long.compatible,
          "BERT text input longer than max sequence length should fail");

    const auto negative = cyxwiz::ValidateBertEncoderTextInputIds(
        {1, -1},
        6,
        false);
    Check(!negative.compatible,
          "negative BERT token IDs should fail");

    const auto mask = cyxwiz::BuildBertEncoderAttentionMask({2, 0, 3}, 0);
    Check(mask == std::vector<int64_t>({1, 0, 1}),
          "BERT attention mask should mark non-pad tokens");
}

void TestPackageContractAcceptsCausalLanguageModel() {
    auto tokenizer_package = LoadFixtureTokenizer();
    const auto contract = cyxwiz::ValidateLanguageModelPackageContract(
        MakeCausalLanguageModelProbe(),
        &tokenizer_package,
        "fixture.cyxmodel");

    Check(contract.compatible,
          "causal LM package contract should be compatible: " +
              contract.error);
    Check(contract.package_path == "fixture.cyxmodel",
          "package path should be surfaced");
    Check(contract.model_family == "causal_lm",
          "model family should be surfaced");
    Check(contract.generation_output_contract == "Float32 [1, seq, vocab]",
          "output contract should be surfaced");
    Check(contract.tokenizer_vocabulary_size == 6,
          "tokenizer vocabulary size should include required special tokens");
    Check(contract.max_sequence_length == 6,
          "max sequence length should come from loaded tokenizer config");
    Check(contract.eos_token_id == 3,
          "EOS token id should come from packaged vocabulary");
}

void TestPackageContractRejectsMissingTokenizer() {
    auto probe = MakeCausalLanguageModelProbe();
    probe.has_tokenizer = false;
    probe.has_vocabulary = false;

    const auto contract =
        cyxwiz::ValidateLanguageModelPackageContract(probe, nullptr);

    Check(!contract.compatible,
          "missing tokenizer package should fail contract");
    Check(contract.error.find("tokenizer/config.json") != std::string::npos,
          "missing tokenizer config should be named in error");
    Check(contract.error.find("tokenizer/vocab.txt") != std::string::npos,
          "missing tokenizer vocab should be named in error");
}

void TestPackageContractRejectsNonGenerationModel() {
    auto tokenizer_package = LoadFixtureTokenizer();
    auto probe = MakeCausalLanguageModelProbe();
    probe.model_family = "classifier";
    probe.supports_generation = false;
    probe.generation_output_contract = "Float32[1,classes]";

    const auto contract =
        cyxwiz::ValidateLanguageModelPackageContract(probe,
                                                     &tokenizer_package);

    Check(!contract.compatible,
          "classifier package should fail language-model contract");
    Check(contract.error.find("model_family") != std::string::npos,
          "model family failure should be named");
    Check(contract.error.find("generation support") != std::string::npos,
          "generation support failure should be named");
    Check(contract.error.find("Float32[1,seq,vocab]") != std::string::npos,
          "generation output contract failure should be named");
}

void TestPromptContract() {
    const auto ok =
        cyxwiz::ValidateLanguageModelPromptIds({4, 5}, 6);
    Check(ok.compatible,
          "non-empty prompt inside max sequence length should pass");
    Check(ok.batch_size == 1 && ok.sequence_length == 2,
          "prompt contract should describe [1, seq] shape");

    const auto empty = cyxwiz::ValidateLanguageModelPromptIds({}, 6);
    Check(!empty.compatible, "empty prompt should fail");

    const auto negative =
        cyxwiz::ValidateLanguageModelPromptIds({4, -1}, 6);
    Check(!negative.compatible, "negative prompt token id should fail");

    const auto too_long =
        cyxwiz::ValidateLanguageModelPromptIds({1, 2, 3}, 2);
    Check(!too_long.compatible,
          "prompt longer than tokenizer max sequence length should fail");
}

void TestRuntimeOutputContractAcceptsCompatibleVocab() {
    const std::vector<float> logits(1 * 2 * 8, 0.0f);
    const cyxwiz::Tensor output({1, 2, 8},
                                logits.data(),
                                cyxwiz::DataType::Float32);

    const auto contract =
        cyxwiz::ValidateLanguageModelRuntimeOutput(output, 2, 6);

    Check(contract.compatible,
          "runtime output with wider model vocab should pass: " +
              contract.error);
    Check(contract.batch_size == 1,
          "runtime batch size should be surfaced");
    Check(contract.sequence_length == 2,
          "runtime sequence length should be surfaced");
    Check(contract.vocab_size == 8,
          "runtime vocab size should be surfaced");
}

void TestRuntimeOutputContractRejectsClassifierOutput() {
    const std::vector<float> logits(3, 0.0f);
    const cyxwiz::Tensor output({1, 3},
                                logits.data(),
                                cyxwiz::DataType::Float32);

    const auto contract =
        cyxwiz::ValidateLanguageModelRuntimeOutput(output, 2, 6);

    Check(!contract.compatible,
          "classifier output should fail language-model runtime contract");
    Check(contract.error.find("rank 3") != std::string::npos,
          "runtime rank error should be clear");
}

void TestRuntimeOutputContractRejectsTooSmallVocab() {
    const std::vector<float> logits(1 * 2 * 5, 0.0f);
    const cyxwiz::Tensor output({1, 2, 5},
                                logits.data(),
                                cyxwiz::DataType::Float32);

    const auto contract =
        cyxwiz::ValidateLanguageModelRuntimeOutput(output, 2, 6);

    Check(!contract.compatible,
          "model vocab smaller than tokenizer vocab should fail");
    Check(contract.error.find("smaller than tokenizer vocabulary") !=
              std::string::npos,
          "vocab compatibility error should be clear");
}

} // namespace

int main() {
    TestPackageContractAcceptsBertEncoder();
    TestPackageContractRejectsMissingBertTokenizer();
    TestPackageContractRejectsBertTokenTypeIds();
    TestBertTextInputContractAndMask();
    TestPackageContractAcceptsCausalLanguageModel();
    TestPackageContractRejectsMissingTokenizer();
    TestPackageContractRejectsNonGenerationModel();
    TestPromptContract();
    TestRuntimeOutputContractAcceptsCompatibleVocab();
    TestRuntimeOutputContractRejectsClassifierOutput();
    TestRuntimeOutputContractRejectsTooSmallVocab();

    std::cout << "Language model inference contract test passed\n";
    return 0;
}
