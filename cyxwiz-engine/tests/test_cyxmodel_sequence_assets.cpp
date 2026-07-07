#include "../src/core/formats/cyxmodel_format.h"
#include "../src/core/sequence_inference_response.h"
#include "../src/core/sequence_model_input.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

void WriteTextFile(const std::filesystem::path& path, const std::string& text) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::binary);
    Check(file.is_open(), "could not create " + path.string());
    file << text;
}

std::vector<int64_t> TensorToInt64Vector(const cyxwiz::Tensor& tensor) {
    std::vector<int64_t> values(tensor.NumElements());
    const int64_t* data = tensor.Data<int64_t>();
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = data[i];
    }
    return values;
}

} // namespace

int main() {
    namespace fs = std::filesystem;

    const fs::path root =
        fs::temp_directory_path() / "cyxwiz_cyxmodel_sequence_assets_test";
    const fs::path package_path = root / "sequence_model.cyxmodel";
    const fs::path token_vocab = root / "token_vocab.txt";
    const fs::path pos_vocab = root / "pos_vocab.txt";
    const fs::path tag_vocab = root / "tag_vocab.txt";

    fs::remove_all(root);
    fs::create_directories(root);

    WriteTextFile(token_vocab, "[PAD]\n[UNK]\nLondon\n");
    WriteTextFile(pos_vocab, "[PAD]\n[UNK]\nNNP\n");
    WriteTextFile(tag_vocab, "[PAD]\n[UNK]\nO\nB-geo\n");

    cyxwiz::ModelManifest manifest;
    manifest.model_name = "NER smoke model";
    manifest.model_type = "SequenceTagger";
    manifest.has_graph = true;
    manifest.has_sequence = true;
    manifest.has_sequence_token_vocabulary = true;
    manifest.has_sequence_pos_vocabulary = true;
    manifest.has_sequence_tag_vocabulary = true;
    manifest.sequence_batch_first = true;
    manifest.sequence_create_attention_mask = true;
    manifest.sequence_create_causal_lm_targets = false;
    manifest.sequence_max_sequence_length = 96;
    manifest.sequence_word_pad_id = 0;
    manifest.sequence_pos_pad_id = 0;
    manifest.sequence_tag_ignore_index = -100;
    manifest.sequence_target_ignore_index = -100;
    manifest.sequence_token_vocabulary_path = "sequence/token_vocab.txt";
    manifest.sequence_pos_vocabulary_path = "sequence/pos_vocab.txt";
    manifest.sequence_tag_vocabulary_path = "sequence/tag_vocab.txt";

    cyxwiz::ExportOptions options;
    options.include_graph = true;
    options.include_sequence_assets = true;
    options.sequence_token_vocabulary_path = token_vocab.string();
    options.sequence_pos_vocabulary_path = pos_vocab.string();
    options.sequence_tag_vocabulary_path = tag_vocab.string();

    cyxwiz::TrainingConfig config;
    config.dataset_name = "ner_sequence_smoke";
    config.input_shape = {96};
    config.num_classes = 4;

    cyxwiz::formats::CyxModelFormat format;
    const bool created = format.Create(
        package_path.string(),
        manifest,
        "{\"nodes\":[]}",
        config,
        nullptr,
        {},
        {},
        nullptr,
        options);
    Check(created, "failed to create cyxmodel: " + format.GetLastError());

    const cyxwiz::ProbeResult probe = format.Probe(package_path.string());
    Check(probe.valid, "probe should be valid: " + probe.error_message);
    Check(probe.has_sequence, "probe should report sequence content");
    Check(probe.has_sequence_token_vocabulary,
          "probe should report token vocabulary");
    Check(probe.has_sequence_pos_vocabulary,
          "probe should report POS vocabulary");
    Check(probe.has_sequence_tag_vocabulary,
          "probe should report tag vocabulary");
    Check(probe.sequence_max_sequence_length == 96,
          "probe should preserve max sequence length");
    Check(probe.sequence_tag_ignore_index == -100,
          "probe should preserve tag ignore index");
    Check(probe.sequence_token_vocabulary_path == "sequence/token_vocab.txt",
          "probe should preserve token vocab package path");
    Check(probe.sequence_pos_vocabulary_path == "sequence/pos_vocab.txt",
          "probe should preserve POS vocab package path");
    Check(probe.sequence_tag_vocabulary_path == "sequence/tag_vocab.txt",
          "probe should preserve tag vocab package path");

    std::string token_text;
    std::string pos_text;
    std::string tag_text;
    const bool extracted = format.ExtractSequenceVocabularyAssets(
        package_path.string(), token_text, pos_text, tag_text);
    Check(extracted,
          "failed to extract sequence vocabularies: " + format.GetLastError());
    Check(token_text == "[PAD]\n[UNK]\nLondon\n",
          "token vocabulary text did not round-trip");
    Check(pos_text == "[PAD]\n[UNK]\nNNP\n",
          "POS vocabulary text did not round-trip");
    Check(tag_text == "[PAD]\n[UNK]\nO\nB-geo\n",
          "tag vocabulary text did not round-trip");

    const std::vector<int64_t> predicted_data = {
        2, 3, 0, 0,
        3, 2, 2, 0,
    };
    const cyxwiz::Tensor predicted_ids(
        {2, 4}, predicted_data.data(), cyxwiz::DataType::Int64);
    const std::vector<std::string> labels = {"[PAD]", "[UNK]", "O", "B-geo"};
    const auto decoded = cyxwiz::DecodeSequenceTagIdsForInference(
        predicted_ids, labels, {2, 3});
    Check(decoded.tag_ids.size() == 2, "decode should preserve batch rows");
    Check(decoded.effective_lengths == std::vector<size_t>({2, 3}),
          "decode should preserve clipped effective lengths");
    Check(decoded.tag_ids[0] == std::vector<int64_t>({2, 3}),
          "decode should trim first padded row");
    Check(decoded.tag_ids[1] == std::vector<int64_t>({3, 2, 2}),
          "decode should trim second padded row");
    Check(decoded.tag_labels[0] == std::vector<std::string>({"O", "B-geo"}),
          "decode should map first row IDs to labels");
    Check(decoded.tag_labels[1] ==
              std::vector<std::string>({"B-geo", "O", "O"}),
          "decode should map second row IDs to labels");

    bool mismatched_lengths_failed = false;
    try {
        (void)cyxwiz::DecodeSequenceTagIdsForInference(
            predicted_ids, labels, {1});
    } catch (const std::exception& e) {
        mismatched_lengths_failed =
            std::string(e.what()).find("sequence_lengths") != std::string::npos;
    }
    Check(mismatched_lengths_failed,
          "decode should reject sequence length count mismatches");

    bool declared_missing_tag_failed = false;
    try {
        cyxwiz::RequireDeclaredSequenceTagVocabulary(true, {});
    } catch (const std::exception& e) {
        declared_missing_tag_failed =
            std::string(e.what()).find("declared but missing or empty") !=
            std::string::npos;
    }
    Check(declared_missing_tag_failed,
          "declared sequence tag vocabulary should fail when missing");
    cyxwiz::RequireDeclaredSequenceTagVocabulary(true, labels);
    cyxwiz::RequireDeclaredSequenceTagVocabulary(false, {});

    cyxwiz::SequentialModel fusion_model;
    fusion_model.Add<cyxwiz::SequenceFeatureFusionModule>(8, 3, 5, 2, 0, 0);
    Check(cyxwiz::ModelUsesSequenceFeatureFusion(fusion_model),
          "fusion detector should recognize sequence feature fusion module");

    cyxwiz::SequentialModel word_only_model;
    word_only_model.Add<cyxwiz::EmbeddingModule>(8, 3, 0);
    Check(!cyxwiz::ModelUsesSequenceFeatureFusion(word_only_model),
          "fusion detector should leave word-only sequence models unchanged");

    const std::vector<int64_t> word_ids = {2, 3, 0, 4};
    const std::vector<int64_t> pos_ids = {1, 2, 0, 3};
    const cyxwiz::Tensor word_tensor(
        {2, 2}, word_ids.data(), cyxwiz::DataType::Int64);
    const cyxwiz::Tensor pos_tensor(
        {2, 2}, pos_ids.data(), cyxwiz::DataType::Int64);
    const cyxwiz::Tensor packed =
        cyxwiz::BuildPackedWordPosSequenceInput(word_tensor, pos_tensor);
    Check(packed.Shape() == std::vector<size_t>({2, 2, 2}),
          "packed sequence inference input should be [batch, seq, 2]");
    const auto* packed_ids = packed.Data<int64_t>();
    Check(std::vector<int64_t>(packed_ids, packed_ids + packed.NumElements()) ==
              std::vector<int64_t>({2, 1, 3, 2, 0, 0, 4, 3}),
          "packed sequence inference input should interleave word/POS ids");

    const std::vector<int64_t> mask_data = {1, 0, 0, 1};
    cyxwiz::SequenceBatch masked_batch;
    masked_batch.word_ids = word_tensor;
    masked_batch.pos_ids = pos_tensor;
    masked_batch.attention_mask = cyxwiz::Tensor(
        {2, 2}, mask_data.data(), cyxwiz::DataType::Int64);
    masked_batch.size = 2;
    masked_batch.sequence_length = 2;

    cyxwiz::TrainingConfiguration word_only_config;
    word_only_config.sequence_batch.word_pad_id = 99;
    const cyxwiz::Tensor masked_words =
        cyxwiz::BuildSequenceModelInput(masked_batch, word_only_config);
    Check(TensorToInt64Vector(masked_words) ==
              std::vector<int64_t>({2, 99, 99, 4}),
          "sequence attention mask should normalize word-only padded tokens");
    Check(TensorToInt64Vector(masked_batch.word_ids) == word_ids,
          "sequence attention mask should not mutate the source word tensor");

    cyxwiz::TrainingConfiguration fused_config;
    fused_config.sequence_batch.word_pad_id = 9;
    fused_config.sequence_batch.pos_pad_id = 7;
    cyxwiz::CompiledLayer fusion_layer;
    fusion_layer.type = gui::NodeType::Concatenate;
    fusion_layer.parameters["sequence_feature_fusion"] = "true";
    fused_config.layers.push_back(fusion_layer);
    const cyxwiz::Tensor masked_packed =
        cyxwiz::BuildSequenceModelInput(masked_batch, fused_config);
    Check(masked_packed.Shape() == std::vector<size_t>({2, 2, 2}),
          "masked fused sequence input should keep [batch, seq, 2] shape");
    Check(TensorToInt64Vector(masked_packed) ==
              std::vector<int64_t>({2, 1, 9, 7, 9, 7, 4, 3}),
          "sequence attention mask should normalize word/POS padded tokens");

    const std::vector<int64_t> bad_mask_data = {1, 0, 1};
    cyxwiz::SequenceBatch bad_mask_batch = masked_batch;
    bad_mask_batch.attention_mask = cyxwiz::Tensor(
        {1, 3}, bad_mask_data.data(), cyxwiz::DataType::Int64);
    bool mask_shape_failed = false;
    try {
        (void)cyxwiz::BuildSequenceModelInput(
            bad_mask_batch, word_only_config);
    } catch (const std::exception& e) {
        mask_shape_failed =
            std::string(e.what()).find("attention mask shape") !=
            std::string::npos;
    }
    Check(mask_shape_failed,
          "sequence attention mask should reject shape mismatches");

    bool pos_shape_failed = false;
    try {
        const cyxwiz::Tensor bad_pos(
            {1, 4}, pos_ids.data(), cyxwiz::DataType::Int64);
        (void)cyxwiz::BuildPackedWordPosSequenceInput(word_tensor, bad_pos);
    } catch (const std::exception& e) {
        pos_shape_failed =
            std::string(e.what()).find("POS ids shape") != std::string::npos;
    }
    Check(pos_shape_failed,
          "packed sequence inference input should reject POS shape mismatch");

    fs::remove_all(root);
    std::cout << "CyxModel sequence asset packaging test passed\n";
    return 0;
}
