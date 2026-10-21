#include "../src/core/training_export_metadata.h"
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <cyxwiz/tokenizer.h>
#include "../src/core/formats/cyxmodel_format.h"
#include "../src/core/formats/cyxmodel_archive.h"
#include "../src/core/model_exporter.h"

#include <cyxwiz/sequential.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace cyxwiz {

std::string ReadTreeModelArtifactType(const std::string&, std::string* error) {
    if (error) {
        error->clear();
    }
    return {};
}

} // namespace cyxwiz

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

} // namespace

int main() {
    namespace fs = std::filesystem;

    const fs::path root =
        fs::temp_directory_path() /
        "cyxwiz_cyxmodel_exporter_generation_metadata_test";
    const fs::path package_path = root / "exported_causal_lm.cyxmodel";

    fs::remove_all(root);
    fs::create_directories(root);

    cyxwiz::SequentialModel model;
    model.Add<cyxwiz::LinearModule>(4, 8, true);

    cyxwiz::ExportOptions options;
    options.format = cyxwiz::ModelFormat::CyxModel;
    options.model_name = "Exporter causal LM";
    options.include_graph = true;
    options.include_training_history = false;
    options.include_optimizer_state = false;
    options.include_sequence_assets = true;
    options.sequence_create_causal_lm_targets = true;
    options.sequence_max_sequence_length = 8;

    cyxwiz::ModelExporter exporter;
    const cyxwiz::ExportResult exported = exporter.ExportCyxModel(
        model,
        nullptr,
        nullptr,
        "{\"nodes\":[]}",
        package_path.string(),
        options);
    Check(exported.success,
          "causal LM export failed: " + exported.error_message);

    cyxwiz::formats::CyxModelFormat format;
    const cyxwiz::ProbeResult probe = format.Probe(package_path.string());
    Check(probe.valid, "probe should be valid: " + probe.error_message);
    Check(probe.model_name == "Exporter causal LM",
          "probe should preserve exported model name");
    Check(probe.model_family == "causal_lm",
          "exporter should mark causal LM model family");
    Check(probe.supports_generation,
          "exporter should mark causal LM generation support");
    Check(probe.generation_output_contract == "Float32[1,seq,vocab]",
          "exporter should mark causal LM generation output contract");
    Check(probe.sequence_create_causal_lm_targets,
          "probe should preserve causal LM target creation flag");

    cyxwiz::Tokenizer tokenizer(cyxwiz::TokenizerType::ByteBPE);
    tokenizer.Train({"In the beginning", "And God said"}, 1, 260);
    std::ostringstream vocabulary;
    Check(tokenizer.GetVocabulary().SaveToStream(vocabulary), "BPE snapshot serialization");
    cyxwiz::TrainingConfiguration trained;
    trained.batch_size=1;trained.epochs=20;trained.dataset_name="actual_training";
    trained.input_shape={256};trained.output_size=tokenizer.GetVocabulary().Size();
    trained.learning_rate=.0003f;
    trained.sequence_batch.enabled=true;trained.sequence_batch.create_causal_lm_targets=true;
    trained.sequence_batch.max_sequence_length=256;
    trained.sequence_batch.tokenizer_config_json=R"({"effective":{"tokenizer_type":3,"lowercase":false,"max_length":256}})";
    trained.sequence_batch.tokenizer_vocabulary_artifact=vocabulary.str();
    auto captured=cyxwiz::TrainingExportMetadata(trained);
    trained.batch_size=99;trained.sequence_batch.tokenizer_vocabulary_artifact.clear();
    captured.include_optimizer_state=false;captured.include_training_history=false;
    const auto snapshot_path=root/"snapshot.cyxmodel";
    auto snapshot=exporter.ExportCyxModel(model,nullptr,nullptr,"{\"nodes\":[]}",snapshot_path.string(),captured);
    Check(snapshot.success,"snapshot export: "+snapshot.error_message);
    std::string saved_config,saved_vocabulary;
    Check(format.ExtractTextTokenizerAssets(snapshot_path.string(),saved_config,saved_vocabulary),"extract captured tokenizer");
    Check(saved_vocabulary==vocabulary.str(),"exact BPE bytes must survive export without a source file");
    cyxwiz::Tokenizer loaded(cyxwiz::TokenizerType::ByteBPE);
    std::istringstream input(saved_vocabulary);
    Check(loaded.GetVocabulary().LoadFromStream(input),"load BPE artifact");
    Check(loaded.Encode("In the beginning")==tokenizer.Encode("In the beginning"),"BPE token IDs must roundtrip");
    const auto assets=cyxwiz::formats::CyxModelArchive::Read(snapshot_path);
    const auto& config_bytes=assets.at("config.json");
    auto saved=nlohmann::json::parse(config_bytes.begin(),config_bytes.end());
    Check(saved["training"]["batch_size"]==1,"export must use captured actual batch size");
    Check(saved["data"]["dataset_name"]=="actual_training","export must retain dataset identity");
    Check(saved["training"]["loss_function"]=="CrossEntropy","export must retain loss identity");
    captured.text_tokenizer_vocab_path="must_not_be_read.vocab";
    auto conflict=exporter.ExportCyxModel(model,nullptr,nullptr,"{\"nodes\":[]}",(root/"conflict.cyxmodel").string(),captured);
    Check(!conflict.success,"ambiguous tokenizer source must fail");
    fs::remove_all(root);
    std::cout << "CyxModel exporter generation metadata test passed\n";
    return 0;
}
