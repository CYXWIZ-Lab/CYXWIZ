// The tokenizer kept in every checkpoint (TOFIX159): written with the
// weights, read back as it was, applied to a graph's configuration the way
// the training data would, and absent for runs without a tokenizer.
#include "core/checkpoint_manager.h"
#include "core/checkpoint_tokenizer.h"
#include "core/model_builder.h"
#include "core/sequence_arrow_batcher.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace cyxwiz;
namespace fs = std::filesystem;

namespace {
int checks = 0;
void Require(bool ok, const std::string& why) {
    ++checks;
    if (!ok) throw std::runtime_error(why);
}

TrainingConfiguration TabularConfig() {
    TrainingConfiguration cfg;
    cfg.input_size = 4;
    cfg.output_size = 2;
    cfg.preprocessing_domain = PreprocessingDomain::Tabular;
    CompiledLayer a;
    a.type = gui::NodeType::Dense; a.node_id = 1; a.name = "Hidden"; a.units = 3;
    a.input_shape = {4}; a.output_shape = {3};
    CompiledLayer b;
    b.type = gui::NodeType::Dense; b.node_id = 2; b.name = "Output"; b.units = 2;
    b.input_shape = {3}; b.output_shape = {2};
    cfg.layers = {a, b};
    cfg.loss_type = gui::NodeType::CrossEntropyLoss;
    cfg.optimizer_type = gui::NodeType::Adam;
    return cfg;
}

// A causal language model's config as training leaves it: an Embedding, a
// TimeDistributed head and the sequence fields from the token windows.
TrainingConfiguration LanguageModelConfig() {
    TrainingConfiguration cfg;
    cfg.sequence_batch.enabled = true;
    cfg.sequence_batch.create_causal_lm_targets = true;
    cfg.sequence_batch.tokenizer_config_json =
        R"({"effective":{"lowercase":"true","max_length":8,"tokenizer_type":"3"},"source":"training_token_windows","version":"1.0"})";
    cfg.sequence_batch.tokenizer_vocabulary_artifact = "#cyxwiz-vocabulary-hex-v1\n3c7061643e\n61\n62\n63\n";
    cfg.sequence_batch.expected_token_vocabulary = {"<pad>", "a", "b", "c"};
    cfg.sequence_batch.word_pad_id = 0;
    cfg.input_size = 8;
    cfg.output_size = 4;
    CompiledLayer embedding;
    embedding.type = gui::NodeType::Embedding;
    embedding.parameters["num_embeddings"] = "4";
    CompiledLayer head;
    head.type = gui::NodeType::TimeDistributed;
    head.units = 4;
    head.parameters["units"] = "4";
    cfg.layers = {embedding, head};
    return cfg;
}
}  // namespace

int main() {
    try {
        const fs::path root = fs::temp_directory_path() / "cyxwiz_checkpoint_tokenizer_test";
        fs::remove_all(root);

        // What training keeps: the tokenizer and what it read from the data.
        const CheckpointTokenizer kept = CheckpointTokenizerFromConfig(LanguageModelConfig());
        Require(!kept.empty() && kept.sequence_length == 8 && kept.token_vocabulary_size == 4 &&
                    kept.labels.size() == 4 && kept.tag_vocabulary_size == 0,
                "a language model's tokenizer and sequence facts");
        Require(CheckpointTokenizerFromConfig(TabularConfig()).empty(), "a run without a tokenizer keeps none");

        // Saved with every checkpoint by the manager.
        auto built = BuildSequentialFromConfig(TabularConfig());
        Require(built.ok(), "model build");
        const fs::path run = root / "train-1";
        {
            CheckpointManager manager(run.string());
            manager.SetProvenance(kept.Sha256(), 8, "");
            manager.SetTokenizer(kept);
            CheckpointDataSplit split;
            split.known = true;
            split.train = 0.7f;
            split.val = 0.1f;
            split.test = 0.2f;
            split.stratified = true;
            split.seed = 7;
            split.excluded_files = {"D:/pets/cat/blurry.jpg"};
            manager.SetDataSplit(split);
            Require(!manager.SaveCheckpoint(*built.model, nullptr, TrainingMetrics{}, "best").empty(), "save best");
            TrainingMetrics later;
            later.current_epoch = 2;
            Require(!manager.SaveCheckpoint(*built.model, nullptr, later, "epoch_0002").empty(), "save epoch 2");
        }
        for (const char* name : {"best", "epoch_0002"}) {
            for (const char* file : {"config.json", "vocab.txt", "sequence.json"})
                Require(fs::exists(run / name / "tokenizer" / file), std::string(name) + " keeps tokenizer/" + file);
        }

        // The Data Split it trained with (Run Test reads its test part from it).
        {
            CheckpointManager reader(run.string());
            const auto metadata = reader.LoadCheckpoint(*built.model, nullptr, "best");
            Require(metadata && metadata->split.known && metadata->split.train == 0.7f &&
                        metadata->split.val == 0.1f && metadata->split.test == 0.2f && metadata->split.stratified &&
                        metadata->split.seed == 7 &&
                        metadata->split.excluded_files == std::vector<std::string>{"D:/pets/cat/blurry.jpg"},
                    "the Data Split read back");
        }

        // Read back as it was.
        std::string error;
        const auto read = ReadCheckpointTokenizer(run / "best", error);
        Require(read && error.empty(), "read back: " + error);
        Require(read->config_json == kept.config_json && read->vocabulary == kept.vocabulary &&
                    read->labels == kept.labels && read->sequence_length == 8 && read->token_vocabulary_size == 4 &&
                    read->Sha256() == kept.Sha256(),
                "the same tokenizer");

        // Applied to a freshly compiled graph (no vocabulary yet, placeholder
        // sizes): what the training data would have filled in.
        TrainingConfiguration compiled = LanguageModelConfig();
        compiled.sequence_batch.expected_token_vocabulary.clear();
        compiled.sequence_batch.tokenizer_vocabulary_artifact.clear();
        compiled.sequence_batch.tokenizer_config_json.clear();
        compiled.layers[0].parameters["num_embeddings"] = "1";
        compiled.layers[1].units = 1;
        compiled.input_size = 0;
        ApplyCheckpointTokenizer(*read, compiled);
        Require(compiled.sequence_batch.expected_token_vocabulary == kept.labels, "vocabulary applied");
        Require(compiled.sequence_batch.tokenizer_vocabulary_artifact == kept.vocabulary, "tokenizer applied");
        Require(compiled.layers[0].parameters["num_embeddings"] == "4" && compiled.layers[1].units == 4,
                "layer sizes follow the vocabulary");
        Require(compiled.input_size == 8, "sequence length applied");
        Require(compiled.generation_preview.vocabulary_artifact == kept.vocabulary &&
                    compiled.generation_preview.tokenizer_type == 3 && compiled.generation_preview.lowercase &&
                    compiled.generation_preview.context == 8,
                "generation previews decode with the same tokenizer");

        // A checkpoint without one (older, or a tabular run) says so quietly.
        {
            CheckpointManager manager((root / "tabular").string());
            Require(!manager.SaveCheckpoint(*built.model, nullptr, TrainingMetrics{}, "best").empty(), "save tabular");
        }
        Require(!ReadCheckpointTokenizer(root / "tabular" / "best", error) && error.empty(), "none stored: no error");
        {
            CheckpointManager reader((root / "tabular").string());
            const auto metadata = reader.LoadCheckpoint(*built.model, nullptr, "best");
            Require(metadata && !metadata->split.known, "a checkpoint without a Data Split says so");
        }
        Require(!fs::exists(root / "tabular" / "best" / "tokenizer"), "no tokenizer folder");

        // A damaged one is reported, not ignored.
        { std::ofstream(run / "best" / "tokenizer" / "sequence.json") << "{not json"; }
        Require(!ReadCheckpointTokenizer(run / "best", error) && !error.empty(), "a damaged tokenizer is an error");

        fs::remove_all(root);
        std::cout << "checkpoint tokenizer: kept with every checkpoint, read back, applied. " << checks
                  << " checks OK\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
