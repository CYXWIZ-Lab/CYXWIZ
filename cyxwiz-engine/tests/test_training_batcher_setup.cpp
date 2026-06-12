#include "../src/core/arrow_dataset.h"
#include "../src/core/model_builder.h"
#include "../src/core/parquet_backed_dataset.h"
#include "../src/core/training_batcher_setup.h"
#include "../src/core/worker_defaults.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

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

    parquet_batchers = cyxwiz::TrainingBatcherSet{};
    prefetch_batchers = cyxwiz::TrainingBatcherSet{};
    explicit_split_batchers = cyxwiz::TrainingBatcherSet{};
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
