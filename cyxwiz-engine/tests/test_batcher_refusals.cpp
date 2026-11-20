// A batcher refuses what it cannot train on, with the reason, instead of
// returning something that trains silently wrong (TOFIX223 0.2 and 0.4):
// - a batch that cannot be built throws; it is never an empty batch, which
//   the training loop reads as the end of the epoch;
// - a class label outside the one-hot width throws; it is never an all-zero
//   target row (labels 1..K with K outputs, or -1 / +1);
// - a missing feature or label value (an empty cell, or NaN in a float
//   column) throws, naming the column; it is never trained on as 0 or class
//   0 (TOFIX223 0.3), in memory and from a Parquet file.
//
// CYXWIZ_TEST_ARRAYFIRE_BACKEND=cuda|opencl|cpu picks the device.

#include "core/arrow_dataset.h"
#include "core/dataset_batcher.h"
#include "core/parquet_arrow_batcher.h"
#include "core/parquet_backed_dataset.h"
#include "computation_truth/test_device_selection.h"

#include <arrow/api.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
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

// Doubles; a NaN becomes an empty cell when `empty_for_nan` is set.
std::shared_ptr<arrow::ChunkedArray> Doubles(const std::vector<double>& values, bool empty_for_nan) {
    arrow::DoubleBuilder builder;
    for (const double v : values) {
        if (std::isnan(v) && empty_for_nan) (void)builder.AppendNull();
        else (void)builder.Append(v);
    }
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

    // Missing values (TOFIX223 0.3).
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const auto missing_error = [&](const std::shared_ptr<arrow::ChunkedArray>& x,
                                   const std::shared_ptr<arrow::ChunkedArray>& label,
                                   const std::vector<std::string>& excluded) {
        auto schema = arrow::schema({arrow::field("x", arrow::float64()), arrow::field("note", arrow::float64()),
                                     arrow::field("label", label->type())});
        auto table = arrow::Table::Make(schema, {x, Doubles({nan, 1, 2, 3}, true), label});
        auto dataset = std::make_shared<cyxwiz::ArrowDataset>(table, "refusals_missing");
        cyxwiz::ArrowDatasetBatcher batcher(dataset, "label", 4, false, 1.0f, true);
        batcher.ExcludeFeatureColumns(excluded);
        size_t size = 0;
        return FirstBatchError(batcher, size);
    };
    {
        const std::string error = missing_error(Doubles({1, nan, 3, 4}, true), Ints({0, 1, 0, 1}), {"note"});
        Check(error.find("missing values") != std::string::npos && error.find("'x' 1 row") != std::string::npos,
              "an empty feature cell is refused, naming the column (got: '" + error + "')");
    }
    {
        const std::string error = missing_error(Doubles({1, nan, 3, 4}, false), Ints({0, 1, 0, 1}), {"note"});
        Check(error.find("'x' 1 row") != std::string::npos, "a NaN feature value is refused (got: '" + error + "')");
    }
    {
        arrow::Int64Builder labels;
        (void)labels.AppendValues({0, 1});
        (void)labels.AppendNull();
        (void)labels.Append(1);
        std::shared_ptr<arrow::Array> label_array;
        (void)labels.Finish(&label_array);
        const std::string error = missing_error(Doubles({1, 2, 3, 4}, true),
                                                std::make_shared<arrow::ChunkedArray>(label_array), {"note"});
        Check(error.find("label 'label' 1 row") != std::string::npos && error.find("IS NOT EMPTY") != std::string::npos,
              "an empty label is refused with the Row Filter hint (got: '" + error + "')");
    }
    {
        // The only empty cell is in a column the model does not use (a role
        // the Data Input excluded): the batch builds.
        const std::string error = missing_error(Doubles({1, 2, 3, 4}, true), Ints({0, 1, 0, 1}), {"note"});
        Check(error.empty(), "an empty cell in an excluded column is not refused (got: '" + error + "')");
    }
    {
        // From a Parquet file: the file's statistics hold the empty cell.
        const auto path = std::filesystem::temp_directory_path() / "cyxwiz_refusals_missing.parquet";
        auto schema = arrow::schema({arrow::field("x", arrow::float64()), arrow::field("label", arrow::int64())});
        auto table = arrow::Table::Make(schema, {Doubles({1, nan, 3, 4}, true), Ints({0, 1, 0, 1})});
        cyxwiz::ArrowDataset source(table, "refusals_missing_source");
        Check(source.ExportParquet(path.string()), "the Parquet fixture is written");
        auto dataset = cyxwiz::ParquetBackedDataset::Open(path.string(), "refusals_missing_parquet");
        Check(dataset != nullptr, "the Parquet fixture opens");
        if (dataset) {
            Check(dataset->NullCount("x") == 1 && dataset->NullCount("label") == 0,
                  "the file's statistics count the empty cell");
            cyxwiz::ParquetArrowBatcher batcher(dataset, "label", 4, false, 1.0f, true);
            std::string error;
            try {
                batcher.Reset();
                (void)batcher.GetNextBatch();
            } catch (const std::exception& e) {
                error = e.what();
            }
            Check(error.find("missing values") != std::string::npos && error.find("'x' 1 row") != std::string::npos,
                  "a Parquet file with an empty feature cell is refused (got: '" + error + "')");
        }
        dataset.reset();
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }

    if (failures == 0) std::cout << "test_batcher_refusals: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
