#include "../src/core/training_export_metadata.h"
#include <sstream>
#include "../src/core/graph_compiler.h"
#include "../src/core/model_exporter.h"
#include "../src/core/model_importer.h"
#include "../src/core/formats/cyxmodel_archive.h"
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

std::string BuildCausalLmGraphJson(bool supplied_roles = false) {
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

    if (supplied_roles) {
        graph["data_boundary_version"] = 2;
        graph["nodes"][0]["type"] = static_cast<int>(gui::NodeType::DataInput);
        auto validation = graph["nodes"][0];
        validation["id"] = 10;
        validation["name"] = "Unloaded validation dataset";
        validation["parameters"]["dataset_name"] = "unloaded_validation";
        validation["parameters"]["dataset"] = "unloaded_validation";
        graph["nodes"].push_back(validation);
        graph["nodes"].push_back({{"id",8},{"type",static_cast<int>(gui::NodeType::DataSplit)},
            {"name","Supplied roles"},{"parameters",{{"train_ratio","1.0"},
            {"val_ratio","0.0"},{"test_ratio","0.0"},{"data_boundary_pin_contract","dataset.v2"}}}});
        graph["nodes"].push_back({{"id",9},{"type",static_cast<int>(gui::NodeType::DataLoader)},
            {"name","Training loader with preview"},{"parameters",{
            {"create_causal_lm_targets","true"},{"max_sequence_length","32"},
            {"generation_preview_enabled","true"},{"generation_preview_every_epochs","1"},
            {"generation_preview_prompts","hello"},{"generation_preview_max_new_tokens","2"},
            {"data_boundary_pin_contract","dataset.v2"}}}});
        graph["links"] = json::array();
        auto connect = [&](int from, int output, int to, int input) {
            graph["links"].push_back({{"id",static_cast<int>(graph["links"].size()+1)},
                {"from_node",from},{"from_pin_index",output},{"to_node",to},{"to_pin_index",input}});
        };
        connect(1,0,8,0); connect(10,0,8,1); connect(8,0,9,0);
        connect(9,0,2,0); connect(2,0,3,0); connect(3,0,4,0);
        connect(4,0,5,0); connect(5,0,6,0); connect(9,1,6,1); connect(6,0,7,0);
    }
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


void CheckBpeSnapshotRoundtrip(const std::filesystem::path& root) {
    cyxwiz::Tokenizer tokenizer(cyxwiz::TokenizerType::ByteBPE);
    tokenizer.SetLowercase(false);
    tokenizer.SetPadding(false);
    tokenizer.Train({"In the beginning", "And God said"}, 1, 280);
    std::ostringstream artifact;
    Check(tokenizer.GetVocabulary().SaveToStream(artifact), "serialize BPE snapshot");
    const auto width = tokenizer.GetVocabulary().Size();
    cyxwiz::TrainingConfiguration config;
    config.batch_size=1; config.input_shape={32}; config.output_size=width;
    config.sequence_batch.enabled=true; config.sequence_batch.create_causal_lm_targets=true;
    config.sequence_batch.max_sequence_length=32;
    config.sequence_batch.tokenizer_config_json=R"({"effective":{"tokenizer_type":"3","lowercase":"false","max_length":32}})";
    config.sequence_batch.tokenizer_vocabulary_artifact=artifact.str();
    auto options=cyxwiz::TrainingExportMetadata(config);
    options.include_optimizer_state=false; options.include_training_history=false;
    auto graph=nlohmann::json::parse(BuildCausalLmGraphJson(true));
    graph["nodes"][0]["parameters"]["shape"]="[32]";
    graph["nodes"][0]["parameters"]["max_sequence_length"]="32";
    graph["nodes"][1]["parameters"]["num_embeddings"]=std::to_string(width);
    graph["nodes"][3]["parameters"]["units"]=std::to_string(width);
    graph["nodes"][4]["parameters"]["num_classes"]=std::to_string(width);
    cyxwiz::SequentialModel source;
    source.Add<cyxwiz::EmbeddingModule>(width,4,0);
    source.Add<cyxwiz::TransformerDecoderModule>(4,2,8,0.0f,false);
    source.Add<cyxwiz::TimeDistributedDenseModule>(4,width,true);
    source.SetTraining(false);
    cyxwiz::ModelExporter exporter;
    const auto path=root/"bpe_snapshot.cyxmodel";
    const auto exported=exporter.ExportCyxModel(source,nullptr,nullptr,graph.dump(),path.string(),options);
    Check(exported.success,"BPE snapshot export: "+exported.error_message);
    Check(std::filesystem::is_regular_file(path),"native export must create a binary file");
    cyxwiz::formats::CyxModelFormat format;
    std::string cfg,vocab,error;
    Check(format.ExtractTextTokenizerAssets(path.string(),cfg,vocab),"BPE extraction");
    Check(vocab==artifact.str(),"BPE artifact byte identity");
    cyxwiz::TextTokenizerPackage packaged;
    Check(cyxwiz::LoadTextTokenizerPackage(cfg,vocab,packaged,error),"BPE tokenizer reload: "+error);
    const auto ids=cyxwiz::EncodeTextTokenIdsForGeneration(*packaged.tokenizer,"In the beginning");
    const auto expected=tokenizer.Encode("In the beginning");
    Check(ids==std::vector<int64_t>(expected.begin(),expected.end()),"BPE prompt ID parity");
    auto contract=cyxwiz::ValidateLanguageModelPackageContract(format.Probe(path.string()),&packaged,path.string());
    Check(contract.compatible && contract.tokenizer_vocabulary_size==width,"BPE inference package contract: "+contract.error);
    cyxwiz::SequentialModel imported;
    cyxwiz::ModelImporter importer;
    cyxwiz::ImportOptions load;load.strict_mode=true;
    const auto result=importer.ImportCyxModel(path.string(),imported,load);
    Check(result.success,"BPE model import: "+result.error_message);
    Check(imported.Size()==3,"supplied-role graph must retain exactly three model layers");
    imported.SetTraining(false);
    cyxwiz::Tensor input({1,ids.size()},ids.data(),cyxwiz::DataType::Int64);
    CheckTensorValues(imported.Forward(input),source.Forward(input),"BPE imported logits");
    cyxwiz::LanguageModelGenerationConfig generation;
    generation.max_new_tokens=2; generation.eos_token_id=-1;
    auto before=cyxwiz::GenerateTokenIdsWithConfig(source,ids,generation,7u);
    auto after=cyxwiz::GenerateTokenIdsWithConfig(imported,ids,generation,7u);
    Check(before==after,"BPE generated-token parity after package reload");
    Check(cyxwiz::DecodeGeneratedTokenIds(tokenizer,before)==
          cyxwiz::DecodeGeneratedTokenIds(*packaged.tokenizer,after),"BPE decoded-output parity");
    // An invalid saved graph must fail with its original cause, never guess Dense layers.
    auto invalid = graph;
    invalid["links"][0]["to_pin_index"] = 999;
    auto assets=cyxwiz::formats::CyxModelArchive::Read(path);
    const auto invalid_json=invalid.dump();
    assets["graph.cyxgraph"]={invalid_json.begin(),invalid_json.end()};
    cyxwiz::formats::CyxModelArchive::WriteBinary(path,assets);
    cyxwiz::SequentialModel rejected;
    const auto failed=importer.Import(path.string(),rejected,{});
    Check(!failed.success && failed.error_message.find("Invalid serialized model graph link")!=std::string::npos,
          "invalid graph must preserve the import error: "+failed.error_message);
    Check(rejected.Size()==0,"invalid graph must not be reconstructed from weight shapes");
}

} // namespace

int main(int argc, char** argv) {
    // Optional real-artifact oracle: no training dataset or project-specific code.
    if (argc == 3) {
        const std::filesystem::path package_path(argv[1]);
        std::ifstream preview_file(argv[2]);
        const auto preview = nlohmann::json::parse(preview_file);
        cyxwiz::ModelImporter importer;
        cyxwiz::SequentialModel model;
        const auto result = importer.Import(package_path.string(), model, {});
        Check(result.success, "actual package import: " + result.error_message);
        model.SetTraining(false);
        cyxwiz::formats::CyxModelFormat format;
        std::string config, vocabulary, error;
        Check(format.ExtractTextTokenizerAssets(package_path.string(), config, vocabulary),
              "actual tokenizer extraction");
        cyxwiz::TextTokenizerPackage tokenizer;
        Check(cyxwiz::LoadTextTokenizerPackage(config, vocabulary, tokenizer, error), error);
        Check(tokenizer.tokenizer != nullptr, "actual tokenizer must be present");
        for (const auto& sample : preview.at("samples")) {
            const auto prompt = sample.at("prompt_token_ids").get<std::vector<int64_t>>();
            const auto expected = sample.at("new_token_ids").get<std::vector<int64_t>>();
            cyxwiz::LanguageModelGenerationConfig generation;
            generation.max_new_tokens = preview.at("max_new_tokens").get<size_t>();
            generation.max_context_tokens = preview.at("context").get<size_t>();
            generation.eos_token_id = tokenizer.tokenizer->GetVocabulary().EosIndex();
            generation.include_prompt = false;
            generation.sampling_mode = cyxwiz::LanguageModelSamplingMode::Greedy;
            const auto report = cyxwiz::GenerateTokenIdsWithReport(model, prompt, generation, 52u);
            Check(report.new_token_ids == expected, "actual package generated-token parity");
            const auto decoded = cyxwiz::DecodeGeneratedTokenIds(*tokenizer.tokenizer, report.new_token_ids);
            Check(decoded == sample.at("generated_text").get<std::string>(),
                  "actual package decoded-text parity");
            std::cout << "Verified prompt " << sample.at("prompt") << ": " << decoded << '\n';
        }
        std::cout << "Actual package import and preview parity passed; layers=" << model.Size() << '\n';
        return 0;
    }
    Check(argc == 1, "usage: test [package_path preview_json]");
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
    CheckBpeSnapshotRoundtrip(root);
    fs::remove_all(root);
    std::cout << "CyxModel causal LM generation round-trip test passed\n";
    return 0;
}
