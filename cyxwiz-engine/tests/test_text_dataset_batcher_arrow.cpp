#include "../src/core/text_dataset_batcher.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

void WriteFile(const std::filesystem::path& path, const std::string& content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    Check(out.good(), "failed to open " + path.string());
    out << content;
}

} // namespace

int main() {
    const auto root =
        std::filesystem::temp_directory_path() /
        "cyxwiz_text_dataset_batcher_arrow_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);

    const auto jsonl = root / "samples.jsonl";
    WriteFile(
        jsonl,
        "{\"text\":\"alpha positive\",\"label\":\"pos\"}\n"
        "{\"text\":\"beta positive\",\"label\":\"pos\"}\n"
        "{\"text\":\"gamma negative\",\"label\":\"neg\"}\n"
        "{\"text\":\"delta negative\",\"label\":\"neg\"}\n");

    cyxwiz::DataRegistry::TextDatasetEntry entry;
    entry.source_path = jsonl.string();
    entry.text_column = "text";
    entry.label_column = "label";
    entry.has_labels = true;
    entry.tokenizer_type = 1;
    entry.max_length = 4;
    entry.lowercase = true;
    entry.do_padding = true;
    entry.do_truncation = true;
    entry.min_word_freq = 1;
    entry.max_vocab_size = 100;
    entry.num_samples = 4;
    entry.num_classes = 2;

    cyxwiz::TextPreprocessingConfig preprocessing;
    cyxwiz::TextDatasetBatcher batcher(
        entry,
        preprocessing,
        /*batch_size=*/2,
        /*train_split=*/0.5f,
        /*val_split=*/0.25f,
        /*test_split=*/0.25f,
        /*shuffle=*/false,
        /*num_workers=*/0);
    batcher.SetOneHotEncoding(2);

    Check(batcher.GetMaxLength() == 4, "max_length should come from entry");
    Check(batcher.GetVocabSize() > 0, "vocab should be built");
    Check(batcher.GetNumSamples() == 2, "train split should have two rows");
    Check(batcher.GetNumValSamples() == 1, "val split should have one row");
    Check(batcher.GetNumTestSamples() == 1, "test split should have one row");
    Check(batcher.GetNumBatches() == 1, "train split should have one batch");

    auto train = batcher.GetNextBatch();
    Check(train.IsValid(), "train batch should be valid");
    Check(train.size == 2, "train batch size");
    Check(train.data.Shape().size() == 2, "train data rank");
    Check(train.data.Shape()[0] == 2, "train data batch dim");
    Check(train.data.Shape()[1] == 4, "train data token width");
    Check(train.labels.Shape().size() == 2, "train labels rank");
    Check(train.labels.Shape()[0] == 2, "train labels batch dim");
    Check(train.labels.Shape()[1] == 2, "train labels one-hot width");

    batcher.SetPhase(cyxwiz::BatcherPhase::Val);
    batcher.Reset();
    Check(batcher.GetNumSamples() == 1, "val split should have one row");

    auto val = batcher.GetNextBatch();
    Check(val.IsValid(), "val batch should be valid");
    Check(val.size == 1, "val batch size");
    Check(val.data.Shape()[1] == 4, "val data token width");
    Check(val.labels.Shape()[1] == 2, "val labels one-hot width");

    batcher.SetPhase(cyxwiz::BatcherPhase::Test);
    batcher.Reset();
    Check(batcher.GetNumSamples() == 1, "test split should have one row");

    auto test = batcher.GetNextBatch();
    Check(test.IsValid(), "test batch should be valid");
    Check(test.size == 1, "test batch size");
    Check(test.data.Shape()[1] == 4, "test data token width");
    Check(test.labels.Shape()[1] == 2, "test labels one-hot width");

    std::filesystem::remove_all(root);
    std::cout << "TextDatasetBatcher Arrow delegation passed\n";
    return 0;
}
