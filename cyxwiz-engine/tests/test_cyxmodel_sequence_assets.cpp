#include "../src/core/formats/cyxmodel_format.h"

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

    fs::remove_all(root);
    std::cout << "CyxModel sequence asset packaging test passed\n";
    return 0;
}
