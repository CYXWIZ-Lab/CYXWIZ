#include "core/prefetch_batcher.h"
#include "core/sparse_feature_dataset.h"
#include "core/sparse_feature_dataset_batcher.h"

#include <arrow/api.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::shared_ptr<cyxwiz::SparseFeatureDataset> MakeDataset(
    bool null_second_label = false) {
    arrow::Int32Builder labels_builder;
    Check(labels_builder.Append(0).ok(), "label append failed");
    if (null_second_label) {
        Check(labels_builder.AppendNull().ok(), "null label append failed");
    } else {
        Check(labels_builder.Append(0).ok(), "label append failed");
    }
    Check(labels_builder.AppendValues({0, 0, 1, 1}).ok(),
          "label append failed");
    auto labels = labels_builder.Finish();
    Check(labels.ok(), labels.status().ToString());

    cyxwiz::SparseFeatureDataset::Contents contents;
    contents.name = null_second_label ? "sparse_null_label" : "sparse_batch";
    contents.num_rows = 6;
    contents.num_features = 4;
    contents.row_offsets = {0, 1, 2, 2, 3, 4, 6};
    contents.column_indices = {0, 1, 2, 3, 0, 3};
    contents.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    contents.feature_names = {"alpha", "beta", "delta", "gamma"};
    contents.labels = std::make_shared<arrow::ChunkedArray>(
        labels.ValueOrDie());
    contents.label_name = "y";
    auto dataset = cyxwiz::SparseFeatureDataset::Create(std::move(contents));
    Check(dataset.ok(), dataset.status().ToString());
    return dataset.ValueOrDie();
}

void CheckNear(float actual, float expected, const std::string& message) {
    Check(std::fabs(actual - expected) < 1e-6f, message);
}

void TestDenseBatchAndLabelModes() {
    auto dataset = MakeDataset();
    cyxwiz::SparseFeatureDatasetBatcher batcher(
        dataset, 2, false, 1.0f, true);
    Check(batcher.GetNumSamples() == 6 && batcher.GetNumBatches() == 3,
          "full split shape mismatch");
    Check(batcher.GetDenseBatchMemoryEstimate().raw_output_bytes == 32 &&
              batcher.GetDenseBatchMemoryEstimate().estimated_peak_bytes == 64,
          "batch memory estimate must cover host and Tensor copies");
    batcher.SetBatchInspectionEnabled(true);
    auto batch = batcher.GetNextBatch();
    Check(batch.IsValid() && batch.size == 2,
          "first sparse batch should be valid");
    Check(batch.data.Shape() == std::vector<size_t>({2, 4}),
          "dense batch shape mismatch");
    const auto* data = batch.data.ReadData<float>();
    const std::vector<float> expected = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 0.0f, 0.0f,
    };
    for (size_t index = 0; index < expected.size(); ++index) {
        CheckNear(data[index], expected[index], "dense batch value mismatch");
    }
    Check(batch.inspection.available &&
              batch.inspection.feature_column_count == 4 &&
              batch.inspection.feature_columns_preview[0].name == "alpha" &&
              batch.inspection.feature_columns_preview[0].source_dtype ==
                  "float32 (CSR)",
          "sparse batch inspection truth mismatch");
    Check(batch.labels.Shape() == std::vector<size_t>({2}) &&
              batch.labels.GetDataType() == cyxwiz::DataType::Float32,
          "default label mode mismatch");

    batcher.SetOneHotEncoding(2);
    batcher.Reset();
    batch = batcher.GetNextBatch();
    Check(batch.labels.Shape() == std::vector<size_t>({2, 2}),
          "one-hot label shape mismatch");
    const auto* one_hot = batch.labels.ReadData<float>();
    CheckNear(one_hot[0], 1.0f, "one-hot class zero mismatch");
    CheckNear(one_hot[1], 0.0f, "one-hot class one mismatch");

    batcher.SetClassIndexLabelMode(true);
    batcher.Reset();
    batch = batcher.GetNextBatch();
    Check(batch.labels.GetDataType() == cyxwiz::DataType::Int32,
          "class-index labels must be int32");

    batcher.SetScalarLabelMode(true);
    batcher.Reset();
    batch = batcher.GetNextBatch();
    Check(batch.labels.Shape() == std::vector<size_t>({2, 1}) &&
              batch.labels.GetDataType() == cyxwiz::DataType::Float32,
          "scalar label mode mismatch");

    batcher.SetNormalization(1.0f, 2.0f);
    batcher.Reset();
    batch = batcher.GetNextBatch();
    data = batch.data.ReadData<float>();
    CheckNear(data[0], 0.0f, "nonzero normalization mismatch");
    CheckNear(data[1], -0.5f, "implicit zero normalization mismatch");

    cyxwiz::SparseFeatureDatasetBatcher sparse_output(
        dataset, 2, false, 1.0f, true);
    sparse_output.SetSparseFeatureOutput(true);
    auto sparse_batch = sparse_output.GetNextBatch();
    Check(sparse_batch.HasSparseFeatures(),
          "direct sparse output must carry a typed CSR batch");
    Check(sparse_batch.data.Shape().empty(),
          "direct sparse output must not allocate a dense input Tensor");
    const auto& sparse = *sparse_batch.sparse_features;
    Check(sparse.rows == 2 && sparse.columns == 4 &&
              sparse.row_offsets == std::vector<int32_t>({0, 1, 2}) &&
              sparse.column_indices == std::vector<int32_t>({0, 1}) &&
              sparse.values == std::vector<float>({1.0f, 2.0f}),
          "direct sparse output CSR contents mismatch");

    cyxwiz::SparseFeatureDatasetBatcher zero_mean_normalized(
        dataset, 2, false, 1.0f, true);
    zero_mean_normalized.SetNormalization(0.0f, 2.0f);
    zero_mean_normalized.SetSparseFeatureOutput(true);
    sparse_batch = zero_mean_normalized.GetNextBatch();
    Check(sparse_batch.HasSparseFeatures() &&
              sparse_batch.sparse_features->values ==
                  std::vector<float>({0.5f, 1.0f}),
          "zero-mean normalization must preserve CSR sparsity");

    cyxwiz::SparseFeatureDatasetBatcher nonzero_mean_normalized(
        dataset, 2, false, 1.0f, true);
    nonzero_mean_normalized.SetNormalization(1.0f, 2.0f);
    nonzero_mean_normalized.SetSparseFeatureOutput(true);
    sparse_batch = nonzero_mean_normalized.GetNextBatch();
    Check(!sparse_batch.HasSparseFeatures() &&
              sparse_batch.data.Shape() == std::vector<size_t>({2, 4}),
          "non-zero mean normalization must preserve numerical truth via dense output");
}

void TestSplitsDropLastShuffleAndBalancing() {
    auto dataset = MakeDataset();
    cyxwiz::SparseFeatureDatasetBatcher train(
        dataset, 2, false, 0.5f, true, cyxwiz::BatcherPhase::Train, 0.33f);
    cyxwiz::SparseFeatureDatasetBatcher val(
        dataset, 2, false, 0.5f, false, cyxwiz::BatcherPhase::Val, 0.33f);
    cyxwiz::SparseFeatureDatasetBatcher test(
        dataset, 2, false, 0.5f, false, cyxwiz::BatcherPhase::Test, 0.33f);
    Check(train.GetNumSamples() == 3 && val.GetNumSamples() == 1 &&
              test.GetNumSamples() == 2,
          "train/val/test split semantics mismatch");

    cyxwiz::SparseFeatureDatasetBatcher drop_last(
        dataset, 4, false, 1.0f, true);
    drop_last.SetDropLast(true);
    Check(drop_last.GetNumBatches() == 1, "drop-last batch count mismatch");
    Check(drop_last.GetNextBatch().size == 4 && drop_last.IsEpochComplete(),
          "drop-last completion mismatch");

    cyxwiz::SparseFeatureDatasetBatcher first(
        dataset, 3, true, 1.0f, true,
        cyxwiz::BatcherPhase::Train, 0.0f, 77);
    cyxwiz::SparseFeatureDatasetBatcher second(
        dataset, 3, true, 1.0f, true,
        cyxwiz::BatcherPhase::Train, 0.0f, 77);
    const auto first_batch = first.GetNextBatch();
    const auto second_batch = second.GetNextBatch();
    const auto* first_data = first_batch.data.ReadData<float>();
    const auto* second_data = second_batch.data.ReadData<float>();
    for (size_t index = 0; index < 12; ++index) {
        CheckNear(first_data[index], second_data[index],
                  "equal shuffle seeds must reproduce order");
    }

    cyxwiz::SparseFeatureDatasetBatcher oversample(
        dataset, 2, false, 1.0f, true,
        cyxwiz::BatcherPhase::Train, 0.0f, 42,
        true, "oversample", "max", 91);
    Check(oversample.GetNumSamples() == 8,
          "oversampling should match the majority class");
    cyxwiz::SparseFeatureDatasetBatcher weighted(
        dataset, 2, false, 1.0f, true,
        cyxwiz::BatcherPhase::Train, 0.0f, 42,
        true, "weighted_sampler", "max", 91);
    Check(weighted.GetNumSamples() == 6,
          "weighted sampling must preserve epoch length");

    cyxwiz::SparseFeatureDatasetBatcher stratified(
        dataset, 6, true, 0.5f, true,
        cyxwiz::BatcherPhase::Train, 0.25f, 123,
        false, "none", "max", 42, {}, true);
    const auto stratified_batch = stratified.GetNextBatch();
    Check(stratified_batch.size == 3,
          "stratified train split size mismatch");
    const auto* stratified_labels =
        stratified_batch.labels.ReadData<float>();
    size_t zeros = 0;
    size_t ones = 0;
    for (size_t row = 0; row < stratified_batch.size; ++row) {
        zeros += stratified_labels[row] == 0.0f ? 1 : 0;
        ones += stratified_labels[row] == 1.0f ? 1 : 0;
    }
    Check(zeros == 2 && ones == 1,
          "stratified split must preserve per-class proportions");
}

void TestMemoryAndFailurePropagation() {
    auto dataset = MakeDataset();
    cyxwiz::MaterializationMemoryContext context;
    context.snapshot_override = cyxwiz::MaterializationMemorySnapshot{
        1024, 1024, true};
    context.policy.hard_limit_bytes = 63;
    bool blocked = false;
    try {
        cyxwiz::SparseFeatureDatasetBatcher batcher(
            dataset, 2, false, 1.0f, true,
            cyxwiz::BatcherPhase::Train, 0.0f, 42,
            false, "none", "max", 42, context);
    } catch (const std::length_error& error) {
        blocked = std::string(error.what()).find("hard limit") !=
            std::string::npos;
    }
    Check(blocked, "unsafe dense batch memory must fail before allocation");

    cyxwiz::SparseFeatureDatasetBatcher direct_sparse(
        dataset, 2, false, 1.0f, true,
        cyxwiz::BatcherPhase::Train, 0.0f, 42,
        false, "none", "max", 42, context, false, false, true);
    Check(direct_sparse.GetNextBatch().HasSparseFeatures(),
          "direct sparse batches must not be blocked by dense memory limits");

    auto prefetched_source =
        std::make_shared<cyxwiz::SparseFeatureDatasetBatcher>(
            dataset, 2, false, 1.0f, true);
    cyxwiz::PrefetchBatcher sparse_prefetch(
        prefetched_source, 1, "sparse_direct");
    sparse_prefetch.SetSparseFeatureOutput(true);
    Check(sparse_prefetch.GetNextBatch().HasSparseFeatures(),
          "prefetch must propagate direct sparse output configuration");

    auto null_labels = MakeDataset(true);
    cyxwiz::SparseFeatureDatasetBatcher failing(
        null_labels, 2, false, 1.0f, true);
    auto source = std::make_shared<cyxwiz::SparseFeatureDatasetBatcher>(
        std::move(failing));
    cyxwiz::PrefetchBatcher prefetch(source, 1, "sparse_failure");
    bool propagated = false;
    try {
        (void)prefetch.GetNextBatch();
    } catch (const std::runtime_error& error) {
        propagated = std::string(error.what()).find("null label") !=
            std::string::npos;
    }
    Check(propagated,
          "prefetch must propagate sparse batch errors instead of ending epoch");
}

} // namespace

int main() {
    TestDenseBatchAndLabelModes();
    TestSplitsDropLastShuffleAndBalancing();
    TestMemoryAndFailurePropagation();
    std::cout << "Sparse feature dataset batcher tests passed\n";
    return 0;
}
