// A batcher refuses what it cannot train on, with the reason, instead of
// returning something that trains silently wrong (TOFIX223 0.2 and 0.4):
// - a batch that cannot be built throws; it is never an empty batch, which
//   the training loop reads as the end of the epoch;
// - a class label outside the one-hot width throws; it is never an all-zero
//   target row (labels 1..K with K outputs, or -1 / +1).
//
// CYXWIZ_TEST_ARRAYFIRE_BACKEND=cuda|opencl|cpu picks the device.

#include "core/arrow_dataset.h"
#include "core/dataset_batcher.h"
#include "computation_truth/test_device_selection.h"

#include <arrow/api.h>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void Check(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

std::shared_ptr<arrow::ChunkedArray> Ints(const std::vector<int64_t>& values) {
    arrow::Int64Builder builder;
    (void)builder.AppendValues(values);
    std::shared_ptr<arrow::Array> out;
    (void)builder.Finish(&out);
    return std::make_shared<arrow::ChunkedArray>(out);
}

std::shared_ptr<arrow::ChunkedArray> Strings(const std::vector<std::string>& values) {
    arrow::StringBuilder builder;
    (void)builder.AppendValues(values);
    std::shared_ptr<arrow::Array> out;
    (void)builder.Finish(&out);
    return std::make_shared<arrow::ChunkedArray>(out);
}

// What GetNextBatch does: the error text when it throws, "" when it returns.
std::string FirstBatchError(cyxwiz::ArrowDatasetBatcher& batcher, size_t& batch_size) {
    batcher.Reset();
    try {
        batch_size = batcher.GetNextBatch().size;
        return "";
    } catch (const std::exception& e) {
        return e.what();
    }
}

}  // namespace

int main() {
    if (!cyxwiz::test::SelectTestDeviceFromEnvironment()) {
        std::cerr << "the requested device could not be activated\n";
        return 1;
    }

    // No numeric feature column: the batch cannot be built.
    {
        auto schema = arrow::schema({arrow::field("text", arrow::utf8()), arrow::field("label", arrow::int64())});
        auto table = arrow::Table::Make(schema, {Strings({"a", "b", "c"}), Ints({0, 1, 0})});
        auto dataset = std::make_shared<cyxwiz::ArrowDataset>(table, "refusals_no_features");
        cyxwiz::ArrowDatasetBatcher batcher(dataset, "label", 2, false, 1.0f, true);
        size_t size = 99;
        const std::string error = FirstBatchError(batcher, size);
        Check(!error.empty() && error.find("no numeric feature columns") != std::string::npos,
              "a table without feature columns throws with the reason (got: '" + error + "', batch size " +
                  std::to_string(size) + ")");
    }

    // Class labels against a one-hot width of 2.
    const auto run_labels = [](const std::vector<int64_t>& labels, size_t& size) {
        auto schema = arrow::schema({arrow::field("x", arrow::int64()), arrow::field("label", arrow::int64())});
        auto table = arrow::Table::Make(schema, {Ints({1, 2, 3, 4}), Ints(labels)});
        auto dataset = std::make_shared<cyxwiz::ArrowDataset>(table, "refusals_labels");
        cyxwiz::ArrowDatasetBatcher batcher(dataset, "label", 4, false, 1.0f, true);
        batcher.SetOneHotEncoding(2);
        return FirstBatchError(batcher, size);
    };
    {
        size_t size = 0;
        const std::string error = run_labels({0, 1, 1, 0}, size);
        Check(error.empty() && size == 4, "labels 0 and 1 with two outputs make a batch (" + error + ")");
    }
    {
        size_t size = 0;
        const std::string error = run_labels({1, 2, 1, 2}, size);
        Check(error.find("class label 2 is outside the model's 2 outputs") != std::string::npos,
              "labels 1..2 with two outputs are refused, naming the label (got: '" + error + "')");
    }
    {
        size_t size = 0;
        const std::string error = run_labels({-1, 1, -1, 1}, size);
        Check(error.find("class label -1") != std::string::npos,
              "labels -1 / +1 are refused (got: '" + error + "')");
    }

    if (failures == 0) std::cout << "test_batcher_refusals: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
