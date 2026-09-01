#include "../src/core/text_dataset_batcher.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace cyxwiz {

bool TryApplyBalancedClassWeightsFromArrowTable(
    TrainingConfiguration&,
    const std::shared_ptr<arrow::Table>&,
    const std::string&,
    const std::string&,
    const std::string&) {
    return false;
}

} // namespace cyxwiz

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

    batcher.SetBatchInspectionEnabled(true);
    auto train = batcher.GetNextBatch();
    Check(train.IsValid(), "train batch should be valid");
    Check(train.size == 2, "train batch size");
    Check(train.data.Shape().size() == 2, "train data rank");
    Check(train.data.Shape()[0] == 2, "train data batch dim");
    Check(train.data.Shape()[1] == 4, "train data token width");
    Check(train.labels.Shape().size() == 2, "train labels rank");
    Check(train.labels.Shape()[0] == 2, "train labels batch dim");
    Check(train.labels.Shape()[1] == 2, "train labels one-hot width");
    Check(train.inspection.available, "inspection metadata should be available");
    Check(train.inspection.row_count == 2, "inspection row count");
    Check(train.inspection.feature_column_count == 4,
          "inspection feature column count");
    Check(train.inspection.label_column_count == 1,
          "inspection label column count");
    Check(train.inspection.feature_columns_preview.size() == 4,
          "inspection feature preview size");
    Check(train.inspection.label_columns_preview.size() == 1,
          "inspection label preview size");
    Check(train.inspection.feature_columns_preview[0].name == "tok_0",
          "inspection should preserve token source columns");
    Check(train.inspection.label_columns_preview[0].name == "y",
          "inspection should preserve label source column");
    Check(train.inspection.token_sequence_columns,
          "token slot columns should be recognized");
    Check(train.inspection.null_summary_available,
          "source null summary should be available");
    Check(train.inspection.inspected_value_count == 10,
          "source null scan should cover selected source cells");
    Check(train.inspection.feature_null_count == 0,
          "feature source cells should not be null");
    Check(train.inspection.label_null_count == 0,
          "label source cells should not be null");
    batcher.SetBatchInspectionEnabled(false);

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

    cyxwiz::TextDatasetBatcher drop_last_batcher(
        entry,
        preprocessing,
        /*batch_size=*/2,
        /*train_split=*/0.75f,
        /*val_split=*/0.25f,
        /*test_split=*/0.0f,
        /*shuffle=*/false,
        /*num_workers=*/0);
    drop_last_batcher.SetDropLast(true);
    Check(drop_last_batcher.GetNumSamples() == 3,
          "drop_last should not change the Text Train role sample count");
    Check(drop_last_batcher.GetNumBatches() == 1,
          "drop_last should floor the Text Train batch count");
    auto kept_train = drop_last_batcher.GetNextBatch();
    Check(kept_train.IsValid() && kept_train.size == 2,
          "drop_last should retain the complete Text Train batch");
    Check(drop_last_batcher.IsEpochComplete(),
          "drop_last should suppress the partial Text Train batch");

    drop_last_batcher.SetPhase(cyxwiz::BatcherPhase::Val);
    drop_last_batcher.Reset();
    Check(drop_last_batcher.GetNumBatches() == 1,
          "drop_last should not remove the partial Text validation batch");
    auto partial_val = drop_last_batcher.GetNextBatch();
    Check(partial_val.IsValid() && partial_val.size == 1,
          "Text validation should retain its partial batch");

    std::filesystem::remove_all(root);
    std::cout << "TextDatasetBatcher Arrow delegation passed\n";
    return 0;
}
