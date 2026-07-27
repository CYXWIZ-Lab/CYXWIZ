#include "../src/core/arrow_dataset.h"
#include "../src/core/classification_decision.h"
#include "../src/core/model_builder.h"
#include "../src/core/parquet_backed_dataset.h"
#include "../src/core/test_dataset_selection.h"
#include "../src/core/training_batcher_setup.h"
#include "../src/core/training_executor.h"
#include "../src/core/worker_defaults.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>
#include <cyxwiz/losses/probability.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

std::shared_ptr<arrow::Array> FinishFloatArray(const std::vector<float>& values) {
    arrow::FloatBuilder builder;
    for (float value : values) {
        auto st = builder.Append(value);
        Check(st.ok(), st.ToString());
    }
    std::shared_ptr<arrow::Array> array;
    auto st = builder.Finish(&array);
    Check(st.ok(), st.ToString());
    return array;
}

std::shared_ptr<arrow::Array> FinishIntArray(const std::vector<int32_t>& values) {
    arrow::Int32Builder builder;
    for (int32_t value : values) {
        auto st = builder.Append(value);
        Check(st.ok(), st.ToString());
    }
    std::shared_ptr<arrow::Array> array;
    auto st = builder.Finish(&array);
    Check(st.ok(), st.ToString());
    return array;
}

std::shared_ptr<arrow::Array> FinishInt8Array(const std::vector<int8_t>& values) {
    arrow::Int8Builder builder;
    for (int8_t value : values) {
        auto st = builder.Append(value);
        Check(st.ok(), st.ToString());
    }
    std::shared_ptr<arrow::Array> array;
    auto st = builder.Finish(&array);
    Check(st.ok(), st.ToString());
    return array;
}

std::shared_ptr<cyxwiz::ArrowDataset> MakeDataset() {
    auto schema = arrow::schema({
        arrow::field("x0", arrow::float32()),
        arrow::field("x1", arrow::float32()),
        arrow::field("label", arrow::int32()),
    });
    auto table = arrow::Table::Make(schema, {
        FinishFloatArray({1.0f, 2.0f, 3.0f, 4.0f}),
        FinishFloatArray({10.0f, 20.0f, 30.0f, 40.0f}),
        FinishIntArray({0, 1, 0, 1}),
    }, 4);
    return std::make_shared<cyxwiz::ArrowDataset>(table, "batcher_setup");
}

std::shared_ptr<cyxwiz::ArrowDataset> MakeMultiGroupDataset() {
    auto schema = arrow::schema({
        arrow::field("x0", arrow::float32()),
        arrow::field("x1", arrow::float32()),
        arrow::field("label", arrow::int32()),
    });
    auto table = arrow::Table::Make(schema, {
        FinishFloatArray({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}),
        FinishFloatArray({10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f}),
        FinishIntArray({0, 1, 0, 1, 0, 1}),
    }, 6);
    return std::make_shared<cyxwiz::ArrowDataset>(table, "batcher_setup_multi");
}

std::shared_ptr<cyxwiz::ArrowDataset> MakeStratifiedDataset() {
    auto schema = arrow::schema({
        arrow::field("x0", arrow::float32()),
        arrow::field("x1", arrow::float32()),
        arrow::field("label", arrow::int32()),
    });
    auto table = arrow::Table::Make(schema, {
        FinishFloatArray({1.0f, 2.0f, 3.0f, 4.0f,
                          5.0f, 6.0f, 7.0f, 8.0f,
                          9.0f, 10.0f, 11.0f, 12.0f}),
        FinishFloatArray({10.0f, 20.0f, 30.0f, 40.0f,
                          50.0f, 60.0f, 70.0f, 80.0f,
                          90.0f, 100.0f, 110.0f, 120.0f}),
        FinishIntArray({0, 0, 0, 0, 0, 0, 0, 0,
                        1, 1, 1, 1}),
    }, 12);
    return std::make_shared<cyxwiz::ArrowDataset>(
        table, "batcher_setup_stratified");
}

std::shared_ptr<cyxwiz::ArrowDataset> MakeWideStratifiedDataset(
    int64_t rows = 64,
    size_t features = 8000) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> columns;
    fields.reserve(features + 1);
    columns.reserve(features + 1);

    std::vector<float> values(static_cast<size_t>(rows), 0.0f);
    for (size_t feature = 0; feature < features; ++feature) {
        for (int64_t row = 0; row < rows; ++row) {
            values[static_cast<size_t>(row)] =
                static_cast<float>((row + static_cast<int64_t>(feature)) % 17) /
                17.0f;
        }
        fields.push_back(arrow::field(
            "x" + std::to_string(feature), arrow::float32()));
        columns.push_back(FinishFloatArray(values));
    }

    std::vector<int32_t> labels(static_cast<size_t>(rows), 0);
    for (int64_t row = 0; row < rows; ++row) {
        labels[static_cast<size_t>(row)] = row % 4 == 0 ? 1 : 0;
    }
    fields.push_back(arrow::field("label", arrow::int32()));
    columns.push_back(FinishIntArray(labels));

    auto table = arrow::Table::Make(
        arrow::schema(fields), columns, rows);
    return std::make_shared<cyxwiz::ArrowDataset>(
        table, "batcher_setup_wide_stratified");
}

std::shared_ptr<cyxwiz::ArrowDataset> MakeTimeSeriesDataset() {
    auto schema = arrow::schema({
        arrow::field("x0", arrow::float32()),
        arrow::field("x1", arrow::float32()),
        arrow::field("y", arrow::float32()),
        arrow::field("__partition__", arrow::int8()),
    });
    auto table = arrow::Table::Make(schema, {
        FinishFloatArray({1.0f, 2.0f, 3.0f, 4.0f}),
        FinishFloatArray({10.0f, 20.0f, 30.0f, 40.0f}),
        FinishFloatArray({1.5f, 2.5f, 3.5f, 4.5f}),
        FinishInt8Array({0, 0, 0, 1}),
    }, 4);
    return std::make_shared<cyxwiz::ArrowDataset>(table, "batcher_setup_ts");
}

std::shared_ptr<cyxwiz::ArrowDataset> MakeMultiGroupTimeSeriesDataset() {
    auto schema = arrow::schema({
        arrow::field("x0", arrow::float32()),
        arrow::field("x1", arrow::float32()),
        arrow::field("y", arrow::float32()),
        arrow::field("__partition__", arrow::int8()),
    });
    auto table = arrow::Table::Make(schema, {
        FinishFloatArray({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}),
        FinishFloatArray({10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f}),
        FinishFloatArray({1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f}),
        FinishInt8Array({0, 1, 0, 1, 0, 1}),
    }, 6);
    return std::make_shared<cyxwiz::ArrowDataset>(
        table, "batcher_setup_ts_multi");
}

cyxwiz::TrainingConfiguration MakeConfig() {
    cyxwiz::TrainingConfiguration config;
    config.output_size = 2;
    config.train_ratio = 0.75f;
    config.shuffle = false;
    config.num_workers = 0;
    return config;
}

cyxwiz::TrainingConfiguration MakeTimeSeriesConfig() {
    auto config = MakeConfig();
    config.is_time_series = true;
    config.input_size = 2;
    config.output_size = 1;
    return config;
}

cyxwiz::TrainingConfiguration MakeTrainingLoopConfig(
    const std::filesystem::path& checkpoint_dir) {
    auto config = MakeConfig();
    config.input_size = 2;
    config.input_shape = {2};
    config.output_size = 2;
    config.loss_type = gui::NodeType::CrossEntropyLoss;
    config.optimizer_type = gui::NodeType::Adam;
    config.learning_rate = 0.001f;
    config.batch_size = 2;
    config.epochs = 1;
    config.save_best_checkpoint = false;
    config.early_stopping_patience = 0;
    config.checkpoint_dir = checkpoint_dir.string();

    cyxwiz::CompiledLayer dense;
    dense.type = gui::NodeType::Dense;
    dense.units = 2;
    config.layers.push_back(dense);
    return config;
}

void CheckFeatureAndLabelShapes(const cyxwiz::Batch& batch,
                                size_t batch_rows,
                                size_t feature_width,
                                size_t label_width,
                                const std::string& label) {
    Check(batch.IsValid(), label + " batch should be valid");
    Check(batch.data.Shape().size() == 2, label + " feature tensor should be 2D");
    Check(batch.data.Shape()[0] == batch_rows, label + " batch dimension should match");
    Check(batch.data.Shape()[1] == feature_width, label + " feature width should match");
    Check(batch.labels.Shape().size() == 2, label + " label tensor should be 2D");
    Check(batch.labels.Shape()[0] == batch_rows, label + " label rows should match");
    Check(batch.labels.Shape()[1] == label_width, label + " label width should match");
}

std::vector<size_t> CountOneHotLabels(cyxwiz::IBatcher& batcher,
                                      size_t num_classes,
                                      const std::string& label) {
    std::vector<size_t> counts(num_classes, 0);
    batcher.Reset();
    while (!batcher.IsEpochComplete()) {
        auto batch = batcher.GetNextBatch();
        if (!batch.IsValid()) break;
        Check(batch.labels.Shape().size() == 2,
              label + " labels should be one-hot 2D");
        Check(batch.labels.Shape()[1] == num_classes,
              label + " one-hot width should match class count");
        const float* values = batch.labels.Data<float>();
        for (size_t row = 0; row < batch.labels.Shape()[0]; ++row) {
            size_t best = 0;
            float best_value = values[row * num_classes];
            for (size_t cls = 1; cls < num_classes; ++cls) {
                const float value = values[row * num_classes + cls];
                if (value > best_value) {
                    best = cls;
                    best_value = value;
                }
            }
            ++counts[best];
        }
    }
    return counts;
}

std::vector<size_t> CountScalarBinaryLabels(cyxwiz::IBatcher& batcher,
                                            const std::string& label) {
    std::vector<size_t> counts(2, 0);
    batcher.Reset();
    while (!batcher.IsEpochComplete()) {
        auto batch = batcher.GetNextBatch();
        if (!batch.IsValid()) break;
        Check(batch.labels.Shape().size() == 2,
              label + " labels should be 2D");
        Check(batch.labels.Shape()[1] == 1,
              label + " labels should be scalar [batch, 1]");
        const float* values = batch.labels.Data<float>();
        for (size_t row = 0; row < batch.labels.Shape()[0]; ++row) {
            const float value = values[row];
            Check(value == 0.0f || value == 1.0f,
                  label + " labels should contain encoded 0/1 values");
            ++counts[static_cast<size_t>(value)];
        }
    }
    return counts;
}

std::vector<size_t> CollectOneHotLabels(cyxwiz::IBatcher& batcher,
                                        size_t num_classes,
                                        const std::string& label) {
    std::vector<size_t> labels;
    batcher.Reset();
    while (!batcher.IsEpochComplete()) {
        auto batch = batcher.GetNextBatch();
        if (!batch.IsValid()) break;
        Check(batch.labels.Shape().size() == 2,
              label + " labels should be one-hot 2D");
        Check(batch.labels.Shape()[1] == num_classes,
              label + " one-hot width should match class count");
        const float* values = batch.labels.Data<float>();
        for (size_t row = 0; row < batch.labels.Shape()[0]; ++row) {
            size_t best = 0;
            float best_value = values[row * num_classes];
            for (size_t cls = 1; cls < num_classes; ++cls) {
                const float value = values[row * num_classes + cls];
                if (value > best_value) {
                    best = cls;
                    best_value = value;
                }
            }
            labels.push_back(best);
        }
    }
    return labels;
}

void WriteParquetWithRowGroupSize(const cyxwiz::ArrowDataset& dataset,
                                  const std::string& path,
                                  int64_t row_group_size) {
    auto table = dataset.GetArrowTable();
    Check(table != nullptr, "source table should exist for row-group parquet write");
    auto output = arrow::io::FileOutputStream::Open(path);
    Check(output.ok(), output.status().ToString());
    auto status = parquet::arrow::WriteTable(*table,
                                             arrow::default_memory_pool(),
                                             *output,
                                             row_group_size);
    Check(status.ok(), status.ToString());
}

void RunModelTrainValidationSmoke(cyxwiz::IBatcher& train,
                                  cyxwiz::IBatcher& val,
                                  const cyxwiz::TrainingConfiguration& config,
                                  const std::string& label) {
    auto built = cyxwiz::BuildSequentialFromConfig(config);
    Check(built.ok(), label + " model/loss/optimizer should build");
    Check(built.model != nullptr, label + " model should exist");
    Check(built.loss != nullptr, label + " loss should exist");
    Check(built.optimizer != nullptr, label + " optimizer should exist");

    built.model->SetTraining(true);
    train.Reset();
    int train_batches = 0;
    while (!train.IsEpochComplete()) {
        auto batch = train.GetNextBatch();
        if (!batch.IsValid()) break;
        auto predictions = built.model->Forward(batch.data);
        auto loss = built.loss->Forward(predictions, batch.labels);
        Check(loss.NumElements() == 1, label + " train loss should be scalar");
        Check(std::isfinite(loss.Data<float>()[0]),
              label + " train loss should be finite");
        auto grad = built.loss->Backward(predictions, batch.labels);
        built.model->Backward(grad);
        built.model->UpdateParameters(built.optimizer.get());
        ++train_batches;
    }
    Check(train_batches == 2, label + " train pass should consume two batches");

    built.model->SetTraining(false);
    val.Reset();
    int val_batches = 0;
    while (!val.IsEpochComplete()) {
        auto batch = val.GetNextBatch();
        if (!batch.IsValid()) break;
        auto predictions = built.model->Forward(batch.data);
        auto loss = built.loss->Forward(predictions, batch.labels);
        Check(loss.NumElements() == 1, label + " validation loss should be scalar");
        Check(std::isfinite(loss.Data<float>()[0]),
              label + " validation loss should be finite");
        ++val_batches;
    }
    Check(val_batches == 1, label + " validation pass should consume one batch");
}

} // namespace

int main() {
    namespace fs = std::filesystem;

    Check(cyxwiz::ClampNumWorkersToPlatform(-3) == 0,
          "negative workers should normalize to single-threaded");
    Check(cyxwiz::ClampNumWorkersToPlatform(0) == 0,
          "zero workers should remain single-threaded");
    Check(cyxwiz::ClampNumWorkersToPlatform(cyxwiz::GetDefaultNumWorkers() + 64) ==
              cyxwiz::GetDefaultNumWorkers(),
          "oversized workers should clamp to platform default");

    auto tabular_resolution =
        cyxwiz::ResolveTabularTrainingInputSize(MakeConfig(), 5);
    Check(tabular_resolution.input_size == 4,
          "tabular input size should reserve one label column");
    Check(!tabular_resolution.used_compiled_override,
          "tabular input size should not use compiled override");
    Check(tabular_resolution.has_separate_label_column,
          "tabular multi-column data should report a separate label column");

    auto single_col_resolution =
        cyxwiz::ResolveTabularTrainingInputSize(MakeConfig(), 1);
    Check(single_col_resolution.input_size == 1,
          "single-column input size should use the only column");
    Check(!single_col_resolution.has_separate_label_column,
          "single-column data should not report a separate label column");

    auto ts_config = MakeConfig();
    ts_config.is_time_series = true;
    ts_config.input_size = 12;
    auto ts_resolution =
        cyxwiz::ResolveTabularTrainingInputSize(ts_config, 99);
    Check(ts_resolution.input_size == 12,
          "time-series input size should preserve GraphCompiler override");
    Check(ts_resolution.used_compiled_override,
          "time-series input size should mark compiled override");

    auto batchers = cyxwiz::BuildArrowTrainingBatchers(
        MakeConfig(),
        MakeDataset(),
        "label",
        /*batch_size=*/2);

    Check(batchers.arrow_train != nullptr, "train Arrow batcher should be owned");
    Check(batchers.arrow_val != nullptr, "val Arrow batcher should be owned");
    Check(batchers.train == batchers.arrow_train.get(), "train pointer should target train owner");
    Check(batchers.val == batchers.arrow_val.get(), "val pointer should target val owner");
    Check(batchers.num_train_samples == 3, "train split should contain 3 samples");

    auto batch = batchers.train->GetNextBatch();
    Check(batch.IsValid(), "train batch should be valid");
    Check(batch.data.Shape().size() == 2, "feature tensor should be 2D");
    Check(batch.data.Shape()[0] == 2, "batch dimension should be 2");
    Check(batch.data.Shape()[1] == 2, "feature width should be 2");
    Check(batch.labels.Shape().size() == 2, "label tensor should be 2D");
    Check(batch.labels.Shape()[1] == 2, "labels should be one-hot by output_size");

    auto prefetch_config = MakeConfig();
    prefetch_config.prefetch_factor = 2;
    auto prefetch_batchers = cyxwiz::BuildArrowTrainingBatchers(
        prefetch_config,
        MakeDataset(),
        "label",
        /*batch_size=*/2);
    Check(prefetch_batchers.prefetch_train != nullptr,
          "prefetch train wrapper should be owned when prefetch_factor is positive");
    Check(prefetch_batchers.prefetch_val != nullptr,
          "prefetch val wrapper should be owned when prefetch_factor is positive");
    Check(prefetch_batchers.train == prefetch_batchers.prefetch_train.get(),
          "train pointer should target prefetch wrapper");
    Check(prefetch_batchers.train != prefetch_batchers.arrow_train.get(),
          "prefetch wrapper should sit in front of Arrow train batcher");
    Check(prefetch_batchers.num_train_samples == 3,
          "prefetch train split should preserve sample count");
    auto prefetch_first = prefetch_batchers.train->GetNextBatch();
    CheckFeatureAndLabelShapes(prefetch_first, 2, 2, 2, "prefetch Arrow first");
    auto prefetch_second = prefetch_batchers.train->GetNextBatch();
    CheckFeatureAndLabelShapes(prefetch_second, 1, 2, 2, "prefetch Arrow second");
    auto prefetch_end = prefetch_batchers.train->GetNextBatch();
    Check(!prefetch_end.IsValid(),
          "prefetch train should return an invalid batch after epoch end");
    Check(prefetch_batchers.train->IsEpochComplete(),
          "prefetch train should complete after queued batches are consumed");
    prefetch_batchers.train->Reset();
    auto prefetch_after_reset = prefetch_batchers.train->GetNextBatch();
    CheckFeatureAndLabelShapes(prefetch_after_reset, 2, 2, 2,
                               "prefetch Arrow after reset");

    auto explicit_split_config = MakeConfig();
    explicit_split_config.has_data_split = true;
    explicit_split_config.train_ratio = 0.50f;
    explicit_split_config.val_ratio = 0.25f;
    explicit_split_config.test_ratio = 0.25f;
    auto explicit_split_batchers = cyxwiz::BuildArrowTrainingBatchers(
        explicit_split_config,
        MakeMultiGroupDataset(),
        "label",
        /*batch_size=*/2);
    Check(explicit_split_batchers.num_train_samples == 3,
          "explicit Arrow DataSplit train split should contain 3 samples");
    Check(explicit_split_batchers.num_val_samples == 1,
          "explicit Arrow DataSplit val split should contain 1 sample");
    Check(explicit_split_batchers.num_test_samples == 2,
          "explicit Arrow DataSplit test split should contain 2 held-out samples");

    auto stratified_split_config = MakeConfig();
    stratified_split_config.has_data_split = true;
    stratified_split_config.train_ratio = 0.50f;
    stratified_split_config.val_ratio = 0.25f;
    stratified_split_config.test_ratio = 0.25f;
    stratified_split_config.stratified = true;
    stratified_split_config.split_seed = 7;
    stratified_split_config.shuffle = false;
    auto stratified_batchers = cyxwiz::BuildArrowTrainingBatchers(
        stratified_split_config,
        MakeStratifiedDataset(),
        "label",
        /*batch_size=*/3);
    Check(stratified_batchers.num_train_samples == 6,
          "stratified Arrow DataSplit train split should contain 6 samples");
    Check(stratified_batchers.num_val_samples == 3,
          "stratified Arrow DataSplit val split should contain 3 samples");
    Check(stratified_batchers.num_test_samples == 3,
          "stratified Arrow DataSplit test split should contain 3 samples");
    auto stratified_train_counts =
        CountOneHotLabels(*stratified_batchers.train, 2, "stratified train");
    auto stratified_val_counts =
        CountOneHotLabels(*stratified_batchers.val, 2, "stratified val");
    auto stratified_test_counts =
        CountOneHotLabels(*stratified_batchers.test, 2, "stratified test");
    Check(stratified_train_counts == std::vector<size_t>({4, 2}),
          "stratified train split should preserve 2:1 class ratio");
    Check(stratified_val_counts == std::vector<size_t>({2, 1}),
          "stratified val split should preserve 2:1 class ratio");
    Check(stratified_test_counts == std::vector<size_t>({2, 1}),
          "stratified test split should preserve 2:1 class ratio");

    auto balanced_loader_config = MakeConfig();
    balanced_loader_config.train_ratio = 0.75f;
    balanced_loader_config.shuffle = false;
    balanced_loader_config.balance_classes = true;
    balanced_loader_config.balance_mode = "oversample";
    balanced_loader_config.balance_target = "max";
    balanced_loader_config.balance_seed = 11;
    auto balanced_batchers = cyxwiz::BuildArrowTrainingBatchers(
        balanced_loader_config,
        MakeStratifiedDataset(),
        "label",
        /*batch_size=*/4);
    Check(balanced_batchers.num_train_samples == 16,
          "balanced Arrow oversampling should expand train split to max class count per class");
    Check(balanced_batchers.num_val_samples == 3,
          "balanced Arrow oversampling should not alter validation split");
    Check(balanced_batchers.num_test_samples == 0,
          "balanced Arrow oversampling should not create test samples");
    auto balanced_train_counts =
        CountOneHotLabels(*balanced_batchers.train, 2, "balanced train");
    Check(balanced_train_counts == std::vector<size_t>({8, 8}),
          "balanced Arrow oversampling should equalize train class counts");

    auto weighted_sampler_config = MakeConfig();
    weighted_sampler_config.train_ratio = 0.75f;
    weighted_sampler_config.shuffle = false;
    weighted_sampler_config.balance_classes = true;
    weighted_sampler_config.balance_mode = "weighted_sampler";
    weighted_sampler_config.balance_target = "max";
    weighted_sampler_config.balance_seed = 19;
    auto weighted_sampler_batchers = cyxwiz::BuildArrowTrainingBatchers(
        weighted_sampler_config,
        MakeStratifiedDataset(),
        "label",
        /*batch_size=*/4);
    Check(weighted_sampler_batchers.num_train_samples == 16,
          "weighted sampler should draw target_count * num_classes train samples");
    Check(weighted_sampler_batchers.num_val_samples == 3,
          "weighted sampler should not alter validation split");
    Check(weighted_sampler_batchers.num_test_samples == 0,
          "weighted sampler should not create test samples");
    auto weighted_labels = CollectOneHotLabels(
        *weighted_sampler_batchers.train, 2, "weighted sampler train");
    auto weighted_sampler_batchers_repeat = cyxwiz::BuildArrowTrainingBatchers(
        weighted_sampler_config,
        MakeStratifiedDataset(),
        "label",
        /*batch_size=*/4);
    auto weighted_labels_repeat = CollectOneHotLabels(
        *weighted_sampler_batchers_repeat.train, 2, "weighted sampler repeat");
    Check(weighted_labels == weighted_labels_repeat,
          "weighted sampler should be deterministic for a fixed balance_seed");
    size_t weighted_minority_count = 0;
    for (size_t label : weighted_labels) {
        if (label == 1) {
            ++weighted_minority_count;
        }
    }
    Check(weighted_minority_count > 1,
          "weighted sampler should sample minority class rows with replacement");

    auto wide_sampler_config = MakeConfig();
    wide_sampler_config.train_ratio = 0.8f;
    wide_sampler_config.val_ratio = 0.1f;
    wide_sampler_config.test_ratio = 0.1f;
    wide_sampler_config.has_data_split = true;
    wide_sampler_config.stratified = true;
    wide_sampler_config.shuffle = true;
    wide_sampler_config.split_seed = 42;
    wide_sampler_config.balance_classes = true;
    wide_sampler_config.balance_mode = "weighted_sampler";
    wide_sampler_config.balance_target = "median";
    wide_sampler_config.balance_seed = 42;
    wide_sampler_config.prefetch_factor = 0;
    auto wide_batchers = cyxwiz::BuildArrowTrainingBatchers(
        wide_sampler_config,
        MakeWideStratifiedDataset(),
        "label",
        /*batch_size=*/8);
    Check(wide_batchers.num_train_samples > 0,
          "wide weighted sampler should create train samples");
    Check(wide_batchers.num_val_samples > 0,
          "wide weighted sampler should create validation samples");
    auto wide_first_batch = wide_batchers.train->GetNextBatch();
    CheckFeatureAndLabelShapes(wide_first_batch, 8, 8000, 2,
                               "wide weighted sampler first batch");

    auto balanced_loss_config = MakeConfig();
    balanced_loss_config.train_ratio = 0.75f;
    balanced_loss_config.output_size = 2;
    balanced_loss_config.loss_type = gui::NodeType::CrossEntropyLoss;
    balanced_loss_config.loss_params["class_weight"] = "balanced";
    const bool applied_balanced_weights =
        cyxwiz::TryApplyBalancedClassWeightsFromArrowTable(
            balanced_loss_config,
            MakeStratifiedDataset()->GetArrowTable(),
            "label",
            "",
            "test balanced loss weights");
    Check(applied_balanced_weights,
          "balanced CrossEntropy weights should be computed from Arrow train split");
    Check(balanced_loss_config.loss_params["class_weight"] == "manual",
          "balanced class_weight should be resolved to manual weights");
    Check(balanced_loss_config.loss_params["class_weights"] == "[0.5625, 4.5]",
          "balanced class_weights should use n_samples / (n_classes * class_count)");

    auto stratified_loss_config = MakeConfig();
    stratified_loss_config.has_data_split = true;
    stratified_loss_config.train_ratio = 0.50f;
    stratified_loss_config.val_ratio = 0.25f;
    stratified_loss_config.test_ratio = 0.25f;
    stratified_loss_config.stratified = true;
    stratified_loss_config.split_seed = 7;
    stratified_loss_config.shuffle = false;
    stratified_loss_config.output_size = 2;
    stratified_loss_config.loss_type = gui::NodeType::CrossEntropyLoss;
    stratified_loss_config.loss_params["class_weight"] = "balanced";
    const bool applied_stratified_weights =
        cyxwiz::TryApplyBalancedClassWeightsFromArrowTable(
            stratified_loss_config,
            MakeStratifiedDataset()->GetArrowTable(),
            "label",
            "",
            "test stratified balanced loss weights");
    Check(applied_stratified_weights,
          "balanced CrossEntropy weights should support stratified train splits");
    Check(stratified_loss_config.loss_params["class_weights"] == "[0.75, 1.5]",
          "stratified balanced class_weights should use stratified train counts");

    auto high_worker_config = MakeConfig();
    high_worker_config.num_workers = cyxwiz::GetDefaultNumWorkers() + 64;
    auto high_worker_batchers = cyxwiz::BuildArrowTrainingBatchers(
        high_worker_config,
        MakeDataset(),
        "label",
        /*batch_size=*/2);
    auto high_worker_batch = high_worker_batchers.train->GetNextBatch();
    Check(high_worker_batch.IsValid(),
          "oversized worker config should still produce a valid batch");

    const fs::path parquet_path =
        fs::temp_directory_path() / "cyxwiz_training_batcher_setup.parquet";
    const fs::path ts_parquet_path =
        fs::temp_directory_path() / "cyxwiz_training_batcher_setup_ts.parquet";
    const fs::path multi_parquet_path =
        fs::temp_directory_path() / "cyxwiz_training_batcher_setup_multi.parquet";
    const fs::path multi_ts_parquet_path =
        fs::temp_directory_path() / "cyxwiz_training_batcher_setup_ts_multi.parquet";
    fs::remove(parquet_path);
    fs::remove(ts_parquet_path);
    fs::remove(multi_parquet_path);
    fs::remove(multi_ts_parquet_path);

    auto parquet_source = MakeDataset();
    Check(parquet_source->ExportParquet(parquet_path.string()),
          "tabular Arrow table should export to Parquet");
    auto parquet_dataset = cyxwiz::ParquetBackedDataset::Open(
        parquet_path.string(), "batcher_setup_parquet");
    Check(parquet_dataset != nullptr, "tabular Parquet dataset should open");
    auto parquet_batchers = cyxwiz::BuildParquetTrainingBatchers(
        MakeConfig(),
        parquet_dataset,
        "label",
        /*batch_size=*/2);
    Check(parquet_batchers.parquet_train != nullptr,
          "train Parquet batcher should be owned");
    Check(parquet_batchers.parquet_val != nullptr,
          "val Parquet batcher should be owned");
    Check(parquet_batchers.train == parquet_batchers.parquet_train.get(),
          "train pointer should target Parquet train owner");
    auto parquet_batch = parquet_batchers.train->GetNextBatch();
    CheckFeatureAndLabelShapes(parquet_batch, 2, 2, 2, "tabular Parquet");

    auto parquet_prefetch_config = MakeConfig();
    parquet_prefetch_config.prefetch_factor = 2;
    auto parquet_prefetch_batchers = cyxwiz::BuildParquetTrainingBatchers(
        parquet_prefetch_config,
        parquet_dataset,
        "label",
        /*batch_size=*/2);
    Check(parquet_prefetch_batchers.prefetch_train != nullptr,
          "prefetch train wrapper should be owned for Parquet");
    Check(parquet_prefetch_batchers.train == parquet_prefetch_batchers.prefetch_train.get(),
          "Parquet train pointer should target prefetch wrapper");
    Check(parquet_prefetch_batchers.train != parquet_prefetch_batchers.parquet_train.get(),
          "Parquet prefetch wrapper should sit in front of train batcher");
    Check(parquet_prefetch_batchers.num_train_samples == 4,
          "Parquet prefetch train split should preserve sample count");
    auto parquet_prefetch_first = parquet_prefetch_batchers.train->GetNextBatch();
    CheckFeatureAndLabelShapes(parquet_prefetch_first, 2, 2, 2,
                               "prefetch Parquet first");
    auto parquet_prefetch_second = parquet_prefetch_batchers.train->GetNextBatch();
    CheckFeatureAndLabelShapes(parquet_prefetch_second, 2, 2, 2,
                               "prefetch Parquet second");
    auto parquet_prefetch_end = parquet_prefetch_batchers.train->GetNextBatch();
    Check(!parquet_prefetch_end.IsValid(),
          "Parquet prefetch train should return invalid batch after epoch end");
    Check(parquet_prefetch_batchers.train->IsEpochComplete(),
          "Parquet prefetch train should complete after queued batches are consumed");
    parquet_prefetch_batchers.train->Reset();
    auto parquet_prefetch_after_reset = parquet_prefetch_batchers.train->GetNextBatch();
    CheckFeatureAndLabelShapes(parquet_prefetch_after_reset, 2, 2, 2,
                               "prefetch Parquet after reset");

    auto resolved_prefetch_source = cyxwiz::BuildArrowTrainingBatchers(
        parquet_prefetch_config,
        MakeDataset(),
        "label",
        /*batch_size=*/2);
    auto resolved_prefetch = cyxwiz::TakeResolvedExternalBatchers(
        std::move(resolved_prefetch_source));
    Check(resolved_prefetch.train != nullptr &&
              resolved_prefetch.train->GetNumSamples() == 3,
          "resolved-role handoff must retain the prefetch source lifetime");
    auto resolved_prefetch_batch = resolved_prefetch.train->GetNextBatch();
    CheckFeatureAndLabelShapes(resolved_prefetch_batch, 2, 2, 2,
                               "resolved-role prefetch handoff");

    auto external_role_config = MakeConfig();
    external_role_config.prefetch_factor = 2;
    auto external_role_assembly_config = external_role_config;
    external_role_assembly_config.prefetch_factor = 0;
    auto external_role_batchers = cyxwiz::BuildArrowTrainingBatchers(
        external_role_assembly_config,
        MakeDataset(),
        "label",
        /*batch_size=*/2);
    external_role_batchers.arrow_test =
        std::make_unique<cyxwiz::ArrowDatasetBatcher>(
            MakeMultiGroupDataset(), "label", 2, false, 1.0f, true, "", 0, 0,
            cyxwiz::BatcherPhase::Train, 0.0f, 42);
    external_role_batchers.parquet_test.reset();
    external_role_batchers.test = external_role_batchers.arrow_test.get();
    external_role_batchers.test->SetOneHotEncoding(
        external_role_config.output_size);
    external_role_batchers.num_test_samples =
        external_role_batchers.test->GetNumSamples();
    cyxwiz::AttachTrainingBatcherPrefetchWrappers(
        external_role_batchers,
        external_role_config,
        "explicit tabular roles");
    Check(external_role_batchers.prefetch_test != nullptr,
          "external Test replacement should receive a fresh prefetch wrapper");
    auto resolved_external_role = cyxwiz::TakeResolvedExternalBatchers(
        std::move(external_role_batchers));
    Check(resolved_external_role.test != nullptr &&
              resolved_external_role.test->GetNumSamples() == 6,
          "external Test handoff must retain the replacement source");
    auto resolved_external_test_batch =
        resolved_external_role.test->GetNextBatch();
    CheckFeatureAndLabelShapes(resolved_external_test_batch, 2, 2, 2,
                               "external Test prefetch handoff");

    auto binary_config = MakeConfig();
    binary_config.output_size = 1;
    binary_config.loss_type = gui::NodeType::BCEWithLogits;
    auto binary_batchers = cyxwiz::BuildArrowTrainingBatchers(
        binary_config, MakeDataset(), "label", /*batch_size=*/2);
    auto binary_batch = binary_batchers.train->GetNextBatch();
    Check(binary_batch.labels.Shape() == std::vector<size_t>({2, 1}),
          "BCEWithLogits labels must be scalar float [batch, 1], not one-hot");
    const float* binary_labels = binary_batch.labels.Data<float>();
    Check(binary_labels[0] == 0.0f && binary_labels[1] == 1.0f,
          "binary scalar labels must preserve encoded 0/1 values");

    auto balanced_binary_config = balanced_loader_config;
    balanced_binary_config.output_size = 1;
    balanced_binary_config.loss_type = gui::NodeType::BCEWithLogits;
    auto balanced_binary_batchers = cyxwiz::BuildArrowTrainingBatchers(
        balanced_binary_config,
        MakeStratifiedDataset(),
        "label",
        /*batch_size=*/4);
    const auto balanced_binary_counts = CountScalarBinaryLabels(
        *balanced_binary_batchers.train, "balanced binary train");
    Check(balanced_binary_counts == std::vector<size_t>({8, 8}),
          "binary scalar-label mode must preserve class balancing after reset");

    auto test_role_config = binary_config;
    test_role_config.dataset_name = "train_dataset";
    test_role_config.data_source_node_id = 1;
    test_role_config.dataset_roles.train = {
        "train_dataset", "label", 1, true};
    test_role_config.dataset_roles.test = {
        "external_test_dataset", "label", 14, true};
    const auto test_selection =
        cyxwiz::ResolveGraphTestDataset(test_role_config);
    Check(test_selection.dataset_name == "external_test_dataset" &&
              test_selection.label_column == "label" &&
              test_selection.source_node_id == 14 &&
              test_selection.scope ==
                  cyxwiz::TestDatasetScope::EntireProvidedDataset,
          "Tools > Test must select the supplied Test role, not Train");

    const auto entire_test_config = cyxwiz::ConfigureTestDatasetScope(
        test_role_config, test_selection.scope);
    Check(entire_test_config.train_ratio == 1.0f &&
              entire_test_config.val_ratio == 0.0f &&
              entire_test_config.test_ratio == 0.0f &&
              !entire_test_config.shuffle &&
              !entire_test_config.balance_classes,
          "supplied Test role must consume every row without reshuffling or balancing");
    auto entire_test_batchers = cyxwiz::BuildArrowTrainingBatchers(
        entire_test_config, MakeDataset(), "label", /*batch_size=*/2);
    Check(entire_test_batchers.train->GetNumSamples() == 4,
          "entire provided Test dataset must retain all rows");

    test_role_config.loss_params["pos_weight"] = "59";
    auto configured_binary_loss =
        cyxwiz::BuildLossFromConfig(test_role_config);
    auto* weighted_binary_loss =
        dynamic_cast<cyxwiz::BCEWithLogitsLoss*>(
            configured_binary_loss.get());
    Check(weighted_binary_loss &&
              weighted_binary_loss->GetPosWeight() == 59.0f,
          "Tools > Test must reuse BCEWithLogits pos_weight from training");

    {
        auto binary_parquet_batchers =
            cyxwiz::BuildParquetTrainingBatchers(
                binary_config, parquet_dataset, "label", /*batch_size=*/2);
        auto binary_parquet_batch =
            binary_parquet_batchers.train->GetNextBatch();
        Check(binary_parquet_batch.labels.Shape() ==
                  std::vector<size_t>({2, 1}),
              "Parquet BCEWithLogits labels must be scalar float [batch, 1]");
        const float* binary_parquet_labels =
            binary_parquet_batch.labels.Data<float>();
        Check(binary_parquet_labels[0] == 0.0f &&
                  binary_parquet_labels[1] == 1.0f,
              "Parquet binary labels must preserve encoded 0/1 values");
    }

    const float binary_logits[] = {-2.0f, 2.0f, 0.1f, -0.1f};
    const float binary_targets[] = {0.0f, 1.0f, 0.0f, 1.0f};
    const auto binary_accuracy = cyxwiz::CountClassificationDecisions(
        binary_logits, binary_targets, 4, 1,
        cyxwiz::ClassificationDecisionMode::BinaryLogit);
    Check(binary_accuracy.correct == 2 && binary_accuracy.total == 4,
          "binary-logit accuracy must threshold logits instead of using argmax");

    const float multiclass_scores[] = {0.1f, 0.8f, 0.1f, 0.7f, 0.2f, 0.1f};
    const float multiclass_targets[] = {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    const auto multiclass_accuracy = cyxwiz::CountClassificationDecisions(
        multiclass_scores, multiclass_targets, 2, 3,
        cyxwiz::ClassificationDecisionMode::MulticlassScores);
    Check(multiclass_accuracy.correct == 2 && multiclass_accuracy.total == 2,
          "multiclass accuracy must retain argmax behavior");

    auto ts_arrow_dataset = MakeTimeSeriesDataset();
    auto ts_arrow_batchers = cyxwiz::BuildArrowTrainingBatchers(
        MakeTimeSeriesConfig(),
        ts_arrow_dataset,
        "ignored_label",
        /*batch_size=*/2);
    Check(ts_arrow_batchers.num_train_samples == 3,
          "time-series Arrow train split should use partition 0 rows");
    Check(ts_arrow_batchers.val->GetNumSamples() == 1,
          "time-series Arrow val split should use partition 1 rows");
    auto ts_arrow_batch = ts_arrow_batchers.train->GetNextBatch();
    CheckFeatureAndLabelShapes(ts_arrow_batch, 2, 2, 1, "time-series Arrow");

    Check(ts_arrow_dataset->ExportParquet(ts_parquet_path.string()),
          "time-series Arrow table should export to Parquet");
    auto ts_parquet_dataset = cyxwiz::ParquetBackedDataset::Open(
        ts_parquet_path.string(), "batcher_setup_ts_parquet");
    Check(ts_parquet_dataset != nullptr, "time-series Parquet dataset should open");
    auto ts_parquet_batchers = cyxwiz::BuildParquetTrainingBatchers(
        MakeTimeSeriesConfig(),
        ts_parquet_dataset,
        "ignored_label",
        /*batch_size=*/2);
    Check(ts_parquet_batchers.num_train_samples == 3,
          "time-series Parquet train split should use partition 0 rows");
    Check(ts_parquet_batchers.val->GetNumSamples() == 1,
          "time-series Parquet val split should use partition 1 rows");
    auto ts_parquet_batch = ts_parquet_batchers.train->GetNextBatch();
    CheckFeatureAndLabelShapes(ts_parquet_batch, 2, 2, 1, "time-series Parquet");

    auto multi_source = MakeMultiGroupDataset();
    WriteParquetWithRowGroupSize(*multi_source, multi_parquet_path.string(), 2);
    auto multi_parquet_dataset = cyxwiz::ParquetBackedDataset::Open(
        multi_parquet_path.string(), "batcher_setup_multi_parquet");
    Check(multi_parquet_dataset != nullptr, "multi-row-group Parquet dataset should open");
    Check(multi_parquet_dataset->GetNumRowGroups() == 3,
          "tabular Parquet test fixture should have three row groups");
    auto multi_parquet_batchers = cyxwiz::BuildParquetTrainingBatchers(
        MakeConfig(),
        multi_parquet_dataset,
        "label",
        /*batch_size=*/3);
    Check(multi_parquet_batchers.num_train_samples == 4,
          "multi-row-group Parquet train split should use first two row groups");
    Check(multi_parquet_batchers.val->GetNumSamples() == 2,
          "multi-row-group Parquet val split should use remaining row group");
    auto multi_parquet_first = multi_parquet_batchers.train->GetNextBatch();
    CheckFeatureAndLabelShapes(multi_parquet_first, 3, 2, 2,
                               "multi-row-group tabular Parquet first");
    auto multi_parquet_second = multi_parquet_batchers.train->GetNextBatch();
    CheckFeatureAndLabelShapes(multi_parquet_second, 1, 2, 2,
                               "multi-row-group tabular Parquet second");

    auto multi_ts_source = MakeMultiGroupTimeSeriesDataset();
    WriteParquetWithRowGroupSize(*multi_ts_source,
                                 multi_ts_parquet_path.string(),
                                 2);
    auto multi_ts_parquet_dataset = cyxwiz::ParquetBackedDataset::Open(
        multi_ts_parquet_path.string(), "batcher_setup_ts_multi_parquet");
    Check(multi_ts_parquet_dataset != nullptr,
          "multi-row-group time-series Parquet dataset should open");
    Check(multi_ts_parquet_dataset->GetNumRowGroups() == 3,
          "time-series Parquet test fixture should have three row groups");
    auto multi_ts_parquet_batchers = cyxwiz::BuildParquetTrainingBatchers(
        MakeTimeSeriesConfig(),
        multi_ts_parquet_dataset,
        "ignored_label",
        /*batch_size=*/2);
    Check(multi_ts_parquet_batchers.num_train_samples == 3,
          "multi-row-group time-series Parquet train split should count partition 0 rows");
    Check(multi_ts_parquet_batchers.val->GetNumSamples() == 3,
          "multi-row-group time-series Parquet val split should count partition 1 rows");
    auto multi_ts_parquet_first = multi_ts_parquet_batchers.train->GetNextBatch();
    CheckFeatureAndLabelShapes(multi_ts_parquet_first, 2, 2, 1,
                               "multi-row-group time-series Parquet first");
    auto multi_ts_parquet_second = multi_ts_parquet_batchers.train->GetNextBatch();
    CheckFeatureAndLabelShapes(multi_ts_parquet_second, 1, 2, 1,
                               "multi-row-group time-series Parquet second");

    const fs::path work_dir =
        fs::temp_directory_path() / "cyxwiz_training_batcher_setup_work";
    fs::remove_all(work_dir);
    fs::create_directories(work_dir);
    auto training_config = MakeTrainingLoopConfig(work_dir / "checkpoints");
    auto loop_arrow_batchers = cyxwiz::BuildArrowTrainingBatchers(
        training_config,
        MakeMultiGroupDataset(),
        "label",
        /*batch_size=*/2);
    RunModelTrainValidationSmoke(
        *loop_arrow_batchers.train,
        *loop_arrow_batchers.val,
        training_config,
        "Arrow model-step");
    auto loop_parquet_batchers = cyxwiz::BuildParquetTrainingBatchers(
        training_config,
        multi_parquet_dataset,
        "label",
        /*batch_size=*/2);
    RunModelTrainValidationSmoke(
        *loop_parquet_batchers.train,
        *loop_parquet_batchers.val,
        training_config,
        "Parquet model-step");

    auto external_test = std::make_unique<cyxwiz::ArrowDatasetBatcher>(
        MakeMultiGroupDataset(), "label", 2, false, 1.0f, true, "", 0, 0,
        cyxwiz::BatcherPhase::Train, 0.0f, 42);
    Check(external_test->GetNumSamples() == 6,
          "external test batcher must preserve every supplied row");

    auto external_parquet_test = std::make_unique<cyxwiz::ParquetArrowBatcher>(
        multi_parquet_dataset, "label", 2, false, 1.0f, true, "", 0, 0,
        cyxwiz::BatcherPhase::Train, 0.0f, 42);
    Check(external_parquet_test->GetNumSamples() == 6,
          "external Parquet test batcher must preserve every supplied row group");
    external_test.reset();
    external_parquet_test.reset();

    parquet_batchers = cyxwiz::TrainingBatcherSet{};
    prefetch_batchers = cyxwiz::TrainingBatcherSet{};
    parquet_prefetch_batchers = cyxwiz::TrainingBatcherSet{};
    explicit_split_batchers = cyxwiz::TrainingBatcherSet{};
    stratified_batchers = cyxwiz::TrainingBatcherSet{};
    balanced_batchers = cyxwiz::TrainingBatcherSet{};
    ts_parquet_batchers = cyxwiz::TrainingBatcherSet{};
    multi_parquet_batchers = cyxwiz::TrainingBatcherSet{};
    multi_ts_parquet_batchers = cyxwiz::TrainingBatcherSet{};
    loop_arrow_batchers = cyxwiz::TrainingBatcherSet{};
    loop_parquet_batchers = cyxwiz::TrainingBatcherSet{};
    parquet_dataset.reset();
    ts_parquet_dataset.reset();
    multi_parquet_dataset.reset();
    multi_ts_parquet_dataset.reset();
    parquet_source.reset();
    ts_arrow_dataset.reset();
    multi_source.reset();
    multi_ts_source.reset();

    fs::remove(parquet_path);
    fs::remove(ts_parquet_path);
    fs::remove(multi_parquet_path);
    fs::remove(multi_ts_parquet_path);
    fs::remove_all(work_dir);

    std::cout << "Training batcher setup passed\n";
    return 0;
}
