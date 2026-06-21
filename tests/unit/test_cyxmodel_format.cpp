#include <catch2/catch_test_macros.hpp>

#include "../../cyxwiz-engine/src/core/formats/cyxmodel_format.h"

#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;

TEST_CASE("CyxModel packages tokenizer assets", "[cyxmodel][tokenizer]") {
    const fs::path root = fs::temp_directory_path() / "cyxwiz_cyxmodel_tokenizer_test";
    const fs::path package_path = root / "text_model.cyxmodel";
    const fs::path vocab_path = root / "vocab.txt";

    fs::remove_all(root);
    fs::create_directories(root);

    {
        std::ofstream vocab(vocab_path, std::ios::binary);
        vocab << "<pad>\n<unk>\nhello\nworld\n";
    }

    cyxwiz::ModelManifest manifest;
    manifest.model_name = "text-model";
    manifest.model_type = "SequentialModel";
    manifest.has_tokenizer = true;
    manifest.has_vocabulary = true;

    cyxwiz::TrainingConfig config;

    cyxwiz::ExportOptions options;
    options.include_graph = false;
    options.include_training_history = false;
    options.include_optimizer_state = false;
    options.include_tokenizer_assets = true;
    options.text_tokenizer_config_json =
        R"({"version":"1.0","effective":{"tokenizer_type":"1","max_length":"8"}})";
    options.text_tokenizer_vocab_path = vocab_path.string();

    cyxwiz::formats::CyxModelFormat format;
    const bool created = format.Create(
        package_path.string(),
        manifest,
        "",
        config,
        nullptr,
        {},
        {},
        nullptr,
        options);

    REQUIRE(created);
    REQUIRE(fs::exists(package_path / "tokenizer" / "config.json"));
    REQUIRE(fs::exists(package_path / "tokenizer" / "vocab.txt"));

    const auto probe = format.Probe(package_path.string());
    REQUIRE(probe.valid);
    REQUIRE(probe.has_tokenizer);
    REQUIRE(probe.has_vocabulary);

    std::string extracted_config;
    std::string extracted_vocab;
    REQUIRE(format.ExtractTextTokenizerAssets(
        package_path.string(),
        extracted_config,
        extracted_vocab));

    REQUIRE(extracted_config.find("\"tokenizer_type\":\"1\"") != std::string::npos);
    REQUIRE(extracted_vocab.find("hello") != std::string::npos);

    fs::remove_all(root);
}
