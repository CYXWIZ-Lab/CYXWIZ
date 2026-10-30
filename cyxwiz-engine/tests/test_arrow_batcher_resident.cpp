// Device-resident training features (TOFIX178 S0-b): a batcher that keeps its
// converted, normalised rows on the device gives exactly the batches of one
// that prepares every batch on the host - same rows, same order, same values,
// same labels - over shuffled epochs, for training and validation splits, with
// class balancing, mixed column types and a column in several chunks.
//
// CYXWIZ_TEST_ARRAYFIRE_BACKEND=cuda|opencl|cpu picks the device.

#include "core/arrow_dataset.h"
#include "core/dataset_batcher.h"
#include "core/prefetch_batcher.h"
#include "computation_truth/test_device_selection.h"

#include <arrow/api.h>

#include <cstdio>
#include <iostream>
#include <memory>
#include <random>
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

template <typename Builder, typename T>
std::shared_ptr<arrow::Array> Column(const std::vector<T>& values) {
    Builder builder;
    for (const T& v : values) (void)builder.Append(v);
    std::shared_ptr<arrow::Array> out;
    (void)builder.Finish(&out);
    return out;
}

std::shared_ptr<arrow::Table> MakeTable(int64_t rows) {
    std::mt19937 rng(1782);
    std::uniform_int_distribution<int> pixel(0, 255), offset(-50, 50);
    std::normal_distribution<double> normal(0.0, 3.0);
    std::vector<uint8_t> a(rows);
    std::vector<int64_t> b(rows), label(rows);
    std::vector<float> c(rows);
    std::vector<double> d(rows);
    for (int64_t r = 0; r < rows; ++r) {
        a[r] = static_cast<uint8_t>(pixel(rng));
        b[r] = offset(rng);
        c[r] = static_cast<float>(normal(rng));
        d[r] = normal(rng) * 10.0;
        label[r] = r % 7 == 0 ? 2 : (r % 3 == 0 ? 1 : 0);  // imbalanced classes
    }
    // Column `d` in three chunks (the CSV reader splits large files).
    const int64_t third = rows / 3;
    std::vector<std::shared_ptr<arrow::Array>> d_chunks;
    for (int part = 0; part < 3; ++part) {
        const int64_t begin = part * third, end = part == 2 ? rows : begin + third;
        d_chunks.push_back(Column<arrow::DoubleBuilder>(std::vector<double>(d.begin() + begin, d.begin() + end)));
    }
    auto schema = arrow::schema({arrow::field("a", arrow::uint8()), arrow::field("b", arrow::int64()),
                                 arrow::field("c", arrow::float32()), arrow::field("d", arrow::float64()),
                                 arrow::field("label", arrow::int64())});
    return arrow::Table::Make(schema, {std::make_shared<arrow::ChunkedArray>(Column<arrow::UInt8Builder>(a)),
                                       std::make_shared<arrow::ChunkedArray>(Column<arrow::Int64Builder>(b)),
                                       std::make_shared<arrow::ChunkedArray>(Column<arrow::FloatBuilder>(c)),
                                       std::make_shared<arrow::ChunkedArray>(d_chunks),
                                       std::make_shared<arrow::ChunkedArray>(Column<arrow::Int64Builder>(label))});
}

void CompareTensors(const cyxwiz::Tensor& resident, const cyxwiz::Tensor& host, const std::string& what) {
    Check(resident.Shape() == host.Shape(), what + ": shape");
    if (resident.Shape() != host.Shape()) return;
    const float* r = resident.ReadData<float>();
    const float* h = host.ReadData<float>();
    size_t differ = 0;
    for (size_t i = 0; i < host.NumElements(); ++i) differ += r[i] != h[i] ? 1 : 0;  // bit-identical
    Check(differ == 0, what + ": " + std::to_string(differ) + " values differ");
}

void CompareRuns(const std::shared_ptr<cyxwiz::ArrowDataset>& dataset, bool training, float val_split,
                 bool balance, const std::string& name) {
    const auto make = [&](bool resident) {
        auto batcher = std::make_unique<cyxwiz::ArrowDatasetBatcher>(
            dataset, "label", 64, true, 0.7f, training, "", 0, 0,
            training ? cyxwiz::BatcherPhase::Train : cyxwiz::BatcherPhase::Val, val_split, 1234u, balance,
            balance ? "oversample" : "none", "max", 99u);
        batcher->SetNormalization(3.5f, 17.25f);
        batcher->SetOneHotEncoding(3);
        batcher->SetDeviceResidentFeatures(resident);
        return batcher;
    };
    auto resident = make(true);
    auto host = make(false);
    for (int epoch = 0; epoch < 2; ++epoch) {
        resident->Reset();
        host->Reset();
        size_t batches = 0;
        while (!host->IsEpochComplete()) {
            Check(!resident->IsEpochComplete(), name + ": the resident batcher ended early");
            const cyxwiz::Batch r = resident->GetNextBatch();
            const cyxwiz::Batch h = host->GetNextBatch();
            const std::string what = name + " epoch " + std::to_string(epoch) + " batch " + std::to_string(batches);
            Check(r.size == h.size && r.size > 0, what + ": size");
            Check(r.row_ids == h.row_ids, what + ": rows and order");
            CompareTensors(r.data, h.data, what + " data");
            CompareTensors(r.labels, h.labels, what + " labels");
            ++batches;
        }
        Check(resident->IsEpochComplete(), name + ": the resident batcher has more batches");
        Check(batches > 1, name + ": several batches per epoch");
    }
}

}  // namespace

int main() {
    if (!cyxwiz::test::SelectTestDeviceFromEnvironment()) {
        std::cerr << "the requested device could not be activated\n";
        return 1;
    }
    auto dataset = std::make_shared<cyxwiz::ArrowDataset>(MakeTable(1000), "resident_test");
    CompareRuns(dataset, true, 0.0f, false, "train");
    CompareRuns(dataset, true, 0.15f, false, "train with a validation share");
    CompareRuns(dataset, false, 0.15f, false, "validation");
    CompareRuns(dataset, true, 0.0f, true, "train, classes balanced");

    // Training prefetches batches on a worker thread (PrefetchBatcher): the
    // resident features are built and looked up there, on the training device.
    {
        auto resident = std::make_shared<cyxwiz::ArrowDatasetBatcher>(dataset, "label", 64, true, 0.7f, true, "", 0, 0,
                                                                      cyxwiz::BatcherPhase::Train, 0.0f, 77u);
        auto host = std::make_unique<cyxwiz::ArrowDatasetBatcher>(dataset, "label", 64, true, 0.7f, true, "", 0, 0,
                                                                  cyxwiz::BatcherPhase::Train, 0.0f, 77u);
        for (cyxwiz::ArrowDatasetBatcher* b : {resident.get(), host.get()}) {
            b->SetNormalization(3.5f, 17.25f);
            b->SetOneHotEncoding(3);
        }
        host->SetDeviceResidentFeatures(false);
        cyxwiz::PrefetchBatcher prefetched(resident, 3, "resident test");
        for (int epoch = 0; epoch < 2; ++epoch) {
            prefetched.Reset();
            host->Reset();
            size_t batches = 0;
            while (!host->IsEpochComplete()) {
                const cyxwiz::Batch r = prefetched.GetNextBatch();
                const cyxwiz::Batch h = host->GetNextBatch();
                const std::string what = "prefetched epoch " + std::to_string(epoch) + " batch " + std::to_string(batches++);
                Check(r.row_ids == h.row_ids, what + ": rows and order");
                CompareTensors(r.data, h.data, what + " data");
                CompareTensors(r.labels, h.labels, what + " labels");
            }
            Check(batches > 1, "prefetched: several batches");
        }
    }
    if (failures) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "resident features give the host batches exactly: train, validation, balanced, prefetched, 2 epochs each\n";
    return 0;
}
