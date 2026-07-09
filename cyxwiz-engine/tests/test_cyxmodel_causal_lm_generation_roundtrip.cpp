#include "../src/core/graph_compiler.h"
#include "../src/core/model_exporter.h"
#include "../src/core/model_importer.h"
#include "../src/core/language_model_generation.h"
#include "../src/gui/loaders/data_loader.h"
#include "../src/inference/text_inference_input.h"
#include "../src/inference/language_model_inference_contract.h"

#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>
#include <cyxwiz/tokenizer.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

namespace cyxwiz {

std::string ReadTreeModelArtifactType(const std::string&, std::string*) {
    return {};
}

} // namespace cyxwiz

namespace cyxwiz::loaders {

DataLoader* GetByCategory(FileCategory) {
    return nullptr;
}

DataLoader* GetByRegisteredDataset(const std::string&) {
    return nullptr;
}

DataLoader* GetByBackendTag(int) {
    return nullptr;
}

const std::vector<DataLoader*>& All() {
    static const std::vector<DataLoader*> loaders;
    return loaders;
}

FileCategory FileCategoryFromString(const std::string&) {
    return FileCategory::Tabular;
}

} // namespace cyxwiz::loaders

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

void CheckNear(float actual,
               float expected,
               float tolerance,
               const std::string& message) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << message << ": expected=" << expected
                  << " actual=" << actual << "\n";
        std::exit(1);
    }
}

void WriteTextFile(const std::filesystem::path& path, const std::string& text) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::binary);
    Check(file.is_open(), "could not create " + path.string());
    file << text;
}

std::string BuildCausalLmGraphJson() {
    using json = nlohmann::json;
    json graph;
    graph["nodes"] = json::array();
    graph["links"] = json::array();

    graph["nodes"].push_back({
        {"id", 1},
        {"type", static_cast<int>(gui::NodeType::DatasetInput)},
        {"name", "Token IDs"},
        {"parameters", {
            {"dataset_name", "causal_lm_generation_roundtrip_dataset"},
            {"dataset", "causal_lm_generation_roundtrip_dataset"},
            {"shape", "[4]"},
            {"create_causal_lm_targets", "true"},
            {"max_sequence_length", "4"}
        }}
    });
    graph["nodes"].push_back({
        {"id", 2},
        {"type", static_cast<int>(gui::NodeType::Embedding)},
        {"name", "Token Embedding"},
        {"parameters", {
            {"num_embeddings", "6"},
            {"embedding_dim", "4"},
            {"padding_idx", "0"}
        }}
    });
    graph["nodes"].push_back({
        {"id", 3},
        {"type", static_cast<int>(gui::NodeType::TransformerDecoder)},
        {"name", "Causal Decoder"},
        {"parameters", {
            {"d_model", "4"},
            {"num_heads", "2"},
            {"dim_feedforward", "8"},
            {"dropout", "0"},
            {"norm_first", "false"}
        }}
    });
    graph["nodes"].push_back({
        {"id", 4},
        {"type", static_cast<int>(gui::NodeType::TimeDistributed)},
        {"name", "Token Logit Head"},
        {"parameters", {{"units", "6"}}}
    });
    graph["nodes"].push_back({
        {"id", 5},
        {"type", static_cast<int>(gui::NodeType::Output)},
        {"name", "Sequence Logits"},
        {"parameters", {{"num_classes", "6"}}}
    });
    graph["nodes"].push_back({
        {"id", 6},
        {"type", static_cast<int>(gui::NodeType::CrossEntropyLoss)},
        {"name", "Token Cross Entropy"},
        {"parameters", json::object()}
    });
    graph["nodes"].push_back({
        {"id", 7},
        {"type", static_cast<int>(gui::NodeType::Adam)},
        {"name", "Adam"},
        {"parameters", {{"learning_rate", "0.001"}}}
    });

    graph["links"].push_back({{"id", 101}, {"from_node", 1}, {"to_node", 2}});
    graph["links"].push_back({{"id", 102}, {"from_node", 2}, {"to_node", 3}});
    graph["links"].push_back({{"id", 103}, {"from_node", 3}, {"to_node", 4}});
    graph["links"].push_back({{"id", 104}, {"from_node", 4}, {"to_node", 5}});
    graph["links"].push_back({{"id", 105}, {"from_node", 4}, {"to_node", 6}});
    graph["links"].push_back({{"id", 106}, {"from_node", 1}, {"to_node", 6}});
    graph["links"].push_back({{"id", 107}, {"from_node", 7}, {"to_node", 6}});

    return graph.dump();
}
void CheckTensorValues(const cyxwiz::Tensor& tensor,
                       const cyxwiz::Tensor& expected,
                       const std::string& name) {
    Check(tensor.Shape() == expected.Shape(), name + " shape mismatch");
    const float* actual_data = tensor.Data<float>();
    const float* expected_data = expected.Data<float>();
    for (size_t i = 0; i < tensor.NumElements(); ++i) {
        CheckNear(actual_data[i], expected_data[i], 1e-6f, name);
    }
}

} // namespace

int main() {
    namespace fs = std::filesystem;

    const fs::path root =
        fs::temp_directory_path() /
        "cyxwiz_cyxmodel_causal_lm_generation_roundtrip";
    const fs::path package_path = root / "causal_lm_generation.cyxmodel";
    const fs::path vocab_path = root / "vocab.txt";
    fs::remove_all(root);
    fs::create_directories(root);
    WriteTextFile(vocab_path, "[PAD]\n[UNK]\nhello\nworld\n");

    cyxwiz::SequentialModel source;
    source.Add<cyxwiz::EmbeddingModule>(6, 4, 0);
    source.Add<cyxwiz::TransformerDecoderModule>(4, 2, 8, 0.0f, false);
    source.Add<cyxwiz::TimeDistributedDenseModule>(4, 6, true);

    const auto source_params = source.GetParameters();
    Check(!source_params.empty(),
          "source causal LM model should expose parameters");

    cyxwiz::ExportOptions export_options;
    export_options.format = cyxwiz::ModelFormat::CyxModel;
    export_options.model_name = "Causal LM generation roundtrip";
    export_options.include_graph = true;
    export_options.include_training_history = false;
    export_options.include_optimizer_state = false;
    export_options.include_tokenizer_assets = true;
    export_options.text_tokenizer_config_json =
        R"({"method":"word","lowercase":true,"max_length":6})";
    export_options.text_tokenizer_vocab_path = vocab_path.string();
    export_options.include_sequence_assets = true;
    export_options.sequence_create_causal_lm_targets = true;
    export_options.sequence_max_sequence_length = 4;

    cyxwiz::ModelExporter exporter;
    const cyxwiz::ExportResult exported = exporter.ExportCyxModel(
        source,
        nullptr,
        nullptr,
        BuildCausalLmGraphJson(),
        package_path.string(),
        export_options);
    Check(exported.success,
          "causal LM .cyxmodel export failed: " + exported.error_message);

    cyxwiz::ModelImporter importer;
    const cyxwiz::ProbeResult probe = importer.ProbeFile(package_path.string());
    Check(probe.valid, "probe should be valid: " + probe.error_message);
    Check(probe.model_family == "causal_lm",
          "causal LM package should declare model family");
    Check(probe.supports_generation,
          "causal LM package should declare generation support");
    Check(probe.generation_output_contract == "Float32[1,seq,vocab]",
          "causal LM package should declare generation output contract");
    Check(probe.has_tokenizer,
          "causal LM package should declare tokenizer config asset");
    Check(probe.has_vocabulary,
          "causal LM package should declare tokenizer vocabulary asset");

    cyxwiz::formats::CyxModelFormat package_format;
    std::string tokenizer_config_json;
    std::string tokenizer_vocab_text;
    const bool tokenizer_extracted =
        package_format.ExtractTextTokenizerAssets(package_path.string(),
                                                  tokenizer_config_json,
                                                  tokenizer_vocab_text);
    Check(tokenizer_extracted,
          "causal LM package should extract tokenizer assets: " +
              package_format.GetLastError());
    Check(tokenizer_config_json.find("\"method\":\"word\"") !=
              std::string::npos,
          "tokenizer config JSON should round-trip");
    Check(tokenizer_vocab_text == "[PAD]\n[UNK]\nhello\nworld\n",
          "tokenizer vocabulary text should round-trip");

    cyxwiz::TextTokenizerPackage tokenizer_package;
    std::string tokenizer_error;
    Check(cyxwiz::LoadTextTokenizerPackage(tokenizer_config_json,
                                           tokenizer_vocab_text,
                                           tokenizer_package,
                                           tokenizer_error),
          "tokenizer package should load: " + tokenizer_error);
    Check(tokenizer_package.has_vocabulary && tokenizer_package.tokenizer,
          "tokenizer package should contain a usable vocabulary");
    const auto prompt_ids = cyxwiz::EncodeTextTokenIdsForGeneration(
        *tokenizer_package.tokenizer,
        "hello world");
    const auto& vocab = tokenizer_package.tokenizer->GetVocabulary();
    Check(prompt_ids == std::vector<int64_t>({
              static_cast<int64_t>(vocab.WordToIndex("hello")),
              static_cast<int64_t>(vocab.WordToIndex("world"))}),
          "packaged tokenizer should encode known prompt tokens");
    Check(cyxwiz::DecodeGeneratedTokenIds(*tokenizer_package.tokenizer,
                                          prompt_ids) == "hello world",
          "packaged tokenizer should decode generated token ids");

    const auto package_contract = cyxwiz::ValidateLanguageModelPackageContract(
        probe,
        &tokenizer_package,
        package_path.string());
    Check(package_contract.compatible,
          "causal LM package contract should be compatible: " +
              package_contract.error);
    Check(package_contract.tokenizer_vocabulary_size == 6,
          "package contract should surface tokenizer vocabulary size");
    Check(package_contract.max_sequence_length == 6,
          "package contract should surface tokenizer max sequence length");
    Check(package_contract.eos_token_id == vocab.EosIndex(),
          "package contract should surface tokenizer EOS token id");

    cyxwiz::SequentialModel imported;
    cyxwiz::ImportOptions import_options;
    import_options.strict_mode = true;

    const cyxwiz::ImportResult imported_result = importer.ImportCyxModel(
        package_path.string(),
        imported,
        import_options);
    Check(imported_result.success,
          "causal LM .cyxmodel import failed: " +
              imported_result.error_message);
    Check(imported.Size() == 3,
          "causal LM import should rebuild embedding, decoder, and token head");

    const auto imported_params = imported.GetParameters();
    Check(imported_params.size() == source_params.size(),
          "causal LM parameter count should round-trip");
    for (const auto& [name, tensor] : source_params) {
        Check(imported_params.count(name) == 1,
              "imported causal LM parameter missing: " + name);
        CheckTensorValues(imported_params.at(name), tensor, name);
    }

    const std::vector<int64_t> input_token_ids = {
        static_cast<int64_t>(vocab.WordToIndex("hello")),
        static_cast<int64_t>(vocab.WordToIndex("world")),
        static_cast<int64_t>(vocab.EosIndex()),
        static_cast<int64_t>(vocab.PadIndex()),
    };
    const cyxwiz::Tensor input({1, input_token_ids.size()},
                               input_token_ids.data(),
                               cyxwiz::DataType::Int64);
    const cyxwiz::Tensor source_output = source.Forward(input);
    const cyxwiz::Tensor imported_output = imported.Forward(input);
    Check(source_output.Shape() == std::vector<size_t>({1, 4, 6}),
          "source causal LM output should be [batch, seq, vocab]");
    Check(imported_output.Shape() == std::vector<size_t>({1, 4, 6}),
          "imported causal LM output should be [batch, seq, vocab]");
    CheckTensorValues(imported_output,
                      source_output,
                      "causal LM imported generation logits");

    const auto runtime_contract = cyxwiz::ValidateLanguageModelRuntimeOutput(
        imported_output,
        input_token_ids.size(),
        package_contract.tokenizer_vocabulary_size);
    Check(runtime_contract.compatible,
          "imported causal LM runtime output should satisfy contract: " +
              runtime_contract.error);

    cyxwiz::LanguageModelGenerationConfig generation_config;
    generation_config.max_new_tokens = 2;
    generation_config.eos_token_id = -1;
    generation_config.include_prompt = true;
    const std::vector<int64_t> generated_ids = cyxwiz::GenerateTokenIdsWithConfig(
        imported,
        prompt_ids,
        generation_config,
        7u);
    Check(generated_ids.size() == prompt_ids.size() + 2,
          "imported causal LM generation should append requested tokens");
    Check(std::equal(prompt_ids.begin(),
                     prompt_ids.end(),
                     generated_ids.begin()),
          "generated token IDs should preserve prompt prefix");
    const std::string generated_text = cyxwiz::DecodeGeneratedTokenIds(
        *tokenizer_package.tokenizer,
        generated_ids);
    Check(!generated_text.empty(),
          "generated token IDs should decode through packaged tokenizer");
    fs::remove_all(root);
    std::cout << "CyxModel causal LM generation round-trip test passed\n";
    return 0;
}
