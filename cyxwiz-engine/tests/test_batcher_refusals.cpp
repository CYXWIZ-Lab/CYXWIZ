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
#include "core/missing_values.h"
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
        // The refusal carries its columns as data, for the fix the Engine
        // offers: name, count, NaN among them, text or number, label or not,
        // and the rows they are out of. Columns without a missing cell are
        // not listed.
        arrow::Int64Builder labels;
        (void)labels.AppendValues({0, 1});
        (void)labels.AppendNull();
        (void)labels.Append(0);
        std::shared_ptr<arrow::Array> label_array;
        (void)labels.Finish(&label_array);
        auto schema = arrow::schema({arrow::field("x", arrow::float64()), arrow::field("y", arrow::float64()),
                                     arrow::field("label", arrow::int64())});
        auto table = arrow::Table::Make(
            schema, {Doubles({1, nan, nan, 4}, false), Doubles({1, 2, 3, 4}, true),
                     std::make_shared<arrow::ChunkedArray>(label_array)});
        auto dataset = std::make_shared<cyxwiz::ArrowDataset>(table, "refusals_typed");
        bool typed = false;
        try {
            cyxwiz::ArrowDatasetBatcher batcher(dataset, "label", 4, false, 1.0f, true);
            batcher.Reset();
            (void)batcher.GetNextBatch();
        } catch (const cyxwiz::MissingValuesError& e) {
            const auto& c = e.Columns();
            typed = c.size() == 2 && c[0].name == "x" && c[0].rows == 2 && c[0].nan_rows == 2 && !c[0].label &&
                    !c[0].text && c[1].name == "label" && c[1].rows == 1 && c[1].nan_rows == 0 && c[1].label &&
                    !c[1].text && e.TotalRows() == 4 && e.Source() == "ArrowDatasetBatcher";
            if (!typed)
                std::cerr << "  columns " << c.size() << ", total " << e.TotalRows() << ", source " << e.Source()
                          << "\n";
        } catch (const std::exception& e) {
            std::cerr << "  untyped: " << e.what() << "\n";
        }
        Check(typed, "the refusal is a MissingValuesError holding the columns, counts and kinds");
        const auto text = cyxwiz::CountMissingColumn("city", *Strings({"a", "b"}), false);
        Check(text.text && text.rows == 0, "a text column is counted as text");

        // The check before training counts what the batcher refuses: the same
        // columns and counts, the label last; a column a role keeps out, a
        // "__" column and a text column (not a feature) are not counted.
        auto wider = arrow::Table::Make(
            arrow::schema({arrow::field("x", arrow::float64()), arrow::field("y", arrow::float64()),
                           arrow::field("label", arrow::int64()), arrow::field("kept_out", arrow::float64()),
                           arrow::field("__partition__", arrow::float64()), arrow::field("city", arrow::utf8())}),
            {Doubles({1, nan, nan, 4}, false), Doubles({1, 2, 3, 4}, true),
             std::make_shared<arrow::ChunkedArray>(label_array), Doubles({nan, 1, 2, 3}, true),
             Doubles({nan, 0, 0, 0}, true), Strings({"a", "b", "c", "d"})});
        const auto counted = cyxwiz::CountMissingTrainingColumns(*wider, "label", {"kept_out"});
        Check(counted.size() == 2 && counted[0].name == "x" && counted[0].rows == 2 && counted[0].nan_rows == 2 &&
                  counted[1].name == "label" && counted[1].label && counted[1].rows == 1,
              "the check before training counts the batcher's columns (" + std::to_string(counted.size()) + ")");
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

    // From a Parquet file, the rest of TOFIX223 0.3b (TAB-18): a uint64 feature
    // column is read (it was all zeros), and a text label the vocabulary does
    // not hold is refused, naming it (it was trained as class 0).
    const auto parquet_batcher_of = [](const std::shared_ptr<arrow::Table>& table, const std::string& file,
                                       std::shared_ptr<const cyxwiz::LabelVocabulary> vocabulary,
                                       std::shared_ptr<cyxwiz::ParquetBackedDataset>& dataset) {
        const auto path = std::filesystem::temp_directory_path() / file;
        cyxwiz::ArrowDataset source(table, file + "_source");
        Check(source.ExportParquet(path.string()), file + ": the Parquet fixture is written");
        dataset = cyxwiz::ParquetBackedDataset::Open(path.string(), file);
        Check(dataset != nullptr, file + ": the Parquet fixture opens");
        return dataset ? std::make_unique<cyxwiz::ParquetArrowBatcher>(dataset, "label", 4, false, 1.0f, true, "", 0,
                                                                       0, cyxwiz::BatcherPhase::Train, 0.0f, 42,
                                                                       std::move(vocabulary))
                       : nullptr;
    };
    {
        arrow::UInt64Builder wide;
        (void)wide.AppendValues({3, 5, 7, 9});
        std::shared_ptr<arrow::Array> wide_array;
        (void)wide.Finish(&wide_array);
        auto schema = arrow::schema({arrow::field("x", arrow::uint64()), arrow::field("label", arrow::int64())});
        auto table = arrow::Table::Make(schema, {std::make_shared<arrow::ChunkedArray>(wide_array), Ints({0, 1, 0, 1})});
        std::shared_ptr<cyxwiz::ParquetBackedDataset> dataset;
        auto batcher = parquet_batcher_of(table, "cyxwiz_refusals_uint64.parquet", nullptr, dataset);
        if (batcher) {
            batcher->Reset();
            const cyxwiz::Batch batch = batcher->GetNextBatch();
            std::string got;
            if (batch.size == 4 && batch.data.Shape() == std::vector<size_t>({4, 1})) {
                const float* data = batch.data.ReadData<float>();
                for (size_t b = 0; b < 4; ++b) got += (b ? "," : "") + std::to_string(static_cast<int>(data[b]));
            }
            Check(got == "3,5,7,9", "a uint64 feature column from Parquet is read (got '" + got + "')");
        }
    }
    {
        auto schema = arrow::schema({arrow::field("x", arrow::float64()), arrow::field("label", arrow::utf8())});
        auto table = arrow::Table::Make(schema, {Doubles({1, 2, 3, 4}, true), Strings({"cat", "dog", "bird", "cat"})});
        auto vocabulary = std::make_shared<cyxwiz::LabelVocabulary>(
            cyxwiz::BuildLabelVocabulary(std::vector<std::string>{"cat", "dog"}));
        std::shared_ptr<cyxwiz::ParquetBackedDataset> dataset;
        auto batcher = parquet_batcher_of(table, "cyxwiz_refusals_unknown_label.parquet", vocabulary, dataset);
        if (batcher) {
            std::string error;
            try {
                batcher->Reset();
                (void)batcher->GetNextBatch();
            } catch (const std::exception& e) {
                error = e.what();
            }
            Check(error.find("label 'bird'") != std::string::npos && error.find("not one of the training classes") !=
                                                                         std::string::npos,
                  "a text label outside the vocabulary is refused, naming it (got: '" + error + "')");
        }
    }

    // In memory, every number type and bool reaches the label as it is: a
    // uint64 class label and a bool label read in scalar mode were 0 for
    // every row (the label reader had no case for them).
    {
        arrow::UInt64Builder wide;
        (void)wide.AppendValues({0, 1, 1, 0});
        std::shared_ptr<arrow::Array> wide_array;
        (void)wide.Finish(&wide_array);
        auto schema = arrow::schema({arrow::field("x", arrow::float64()), arrow::field("label", arrow::uint64())});
        auto table = arrow::Table::Make(schema, {Doubles({1, 2, 3, 4}, true),
                                                 std::make_shared<arrow::ChunkedArray>(wide_array)});
        auto dataset = std::make_shared<cyxwiz::ArrowDataset>(table, "refusals_uint64_label");
        cyxwiz::ArrowDatasetBatcher batcher(dataset, "label", 4, false, 1.0f, true);
        batcher.SetOneHotEncoding(2);
        batcher.Reset();
        const cyxwiz::Batch batch = batcher.GetNextBatch();
        std::string got;
        if (batch.size == 4 && batch.labels.Shape() == std::vector<size_t>({4, 2})) {
            const float* labels = batch.labels.ReadData<float>();
            for (size_t b = 0; b < 4; ++b) got += (b ? "," : "") + std::to_string(labels[b * 2 + 1] > 0.5f ? 1 : 0);
        }
        Check(got == "0,1,1,0", "a uint64 class label in memory is read (got '" + got + "')");
    }
    {
        arrow::BooleanBuilder flags;
        (void)flags.AppendValues(std::vector<bool>{true, false, true, true});
        std::shared_ptr<arrow::Array> flag_array;
        (void)flags.Finish(&flag_array);
        auto schema = arrow::schema({arrow::field("x", arrow::float64()), arrow::field("label", arrow::boolean())});
        auto table = arrow::Table::Make(schema, {Doubles({1, 2, 3, 4}, true),
                                                 std::make_shared<arrow::ChunkedArray>(flag_array)});
        auto dataset = std::make_shared<cyxwiz::ArrowDataset>(table, "refusals_bool_label");
        cyxwiz::ArrowDatasetBatcher batcher(dataset, "label", 4, false, 1.0f, true);
        batcher.SetScalarLabelMode(true);
        batcher.Reset();
        const cyxwiz::Batch batch = batcher.GetNextBatch();
        std::string got;
        if (batch.size == 4 && batch.labels.NumElements() == 4) {
            const float* labels = batch.labels.ReadData<float>();
            for (size_t b = 0; b < 4; ++b) got += (b ? "," : "") + std::to_string(static_cast<int>(labels[b]));
        }
        Check(got == "1,0,1,1", "a bool label read as a scalar target in memory is 0 / 1 (got '" + got + "')");
    }

    // class_weight=balanced counts the Train classes before the first batch:
    // an empty label is refused there with the missing-values message, not as
    // classes that "cannot be counted".
    {
        arrow::Int64Builder labels;
        (void)labels.AppendValues({0, 1});
        (void)labels.AppendNull();
        (void)labels.Append(1);
        std::shared_ptr<arrow::Array> label_array;
        (void)labels.Finish(&label_array);
        auto schema = arrow::schema({arrow::field("x", arrow::float64()), arrow::field("label", arrow::int64())});
        auto table = arrow::Table::Make(schema, {Doubles({1, 2, 3, 4}, true),
                                                 std::make_shared<arrow::ChunkedArray>(label_array)});
        auto dataset = std::make_shared<cyxwiz::ArrowDataset>(table, "refusals_count_missing");
        cyxwiz::ArrowDatasetBatcher batcher(dataset, "label", 4, false, 1.0f, true);
        std::string error;
        try {
            (void)batcher.TrainClassCounts();
        } catch (const std::exception& e) {
            error = e.what();
        }
        Check(error.find("label 'label' 1 row") != std::string::npos,
              "counting the Train classes refuses an empty label with its message (got: '" + error + "')");
    }

    // From a Parquet file a NaN is a missing value too: the file's statistics
    // do not count it, so the batch refuses it as it is read (it trained as a
    // NaN value until the loss became NaN).
    {
        auto schema = arrow::schema({arrow::field("x", arrow::float64()), arrow::field("label", arrow::int64())});
        auto table = arrow::Table::Make(schema, {Doubles({1, nan, 3, 4}, false), Ints({0, 1, 0, 1})});
        std::shared_ptr<cyxwiz::ParquetBackedDataset> dataset;
        auto batcher = parquet_batcher_of(table, "cyxwiz_refusals_nan.parquet", nullptr, dataset);
        if (batcher) {
            std::string error;
            try {
                batcher->Reset();
                (void)batcher->GetNextBatch();
            } catch (const std::exception& e) {
                error = e.what();
            }
            Check(error.find("missing values") != std::string::npos && error.find("'x' at least 1 row") != std::string::npos,
                  "a NaN feature value from Parquet is refused, naming the column (got: '" + error + "')");
        }
    }

    if (failures == 0) std::cout << "test_batcher_refusals: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
