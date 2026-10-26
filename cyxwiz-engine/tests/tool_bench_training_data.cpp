// Training data-path benchmark (TOFIX178 Stage 0, S0-a).
//
// How much of an epoch is spent getting batches to the model, and how much on
// the model itself? Builds an MNIST-sized table (uint8 pixel columns, as the
// CSV loader compacts them, and an int label), then times with the training
// batcher (ArrowDatasetBatcher):
//   data:  one epoch of GetNextBatch only (gather, float conversion, host
//          copies, upload), synchronised on the device;
//   train: one epoch of GetNextBatch + forward, loss, backward, update of a
//          784-128-10 MLP;
//   model: the same training steps on one batch already on the device.
//
//   cyxwiz-bench-training-data [--rows N] [--features F] [--batch B] [--epochs E]
//
// CYXWIZ_TEST_ARRAYFIRE_BACKEND=cuda|opencl|cpu picks the device. Prints one
// JSON line per measurement.

#include "core/arrow_dataset.h"
#include "core/dataset_batcher.h"
#include "computation_truth/test_device_selection.h"

#include <cyxwiz/losses/classification.h>
#include <cyxwiz/optimizers/optimizer_base.h>
#include <cyxwiz/sequential.h>

#include <arrayfire.h>
#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

std::string Arg(int argc, char** argv, const std::string& name, const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i)
        if (name == argv[i]) return argv[i + 1];
    return fallback;
}

std::shared_ptr<arrow::Table> MakeTable(int64_t rows, int features, int classes) {
    std::mt19937 rng(178);
    std::uniform_int_distribution<int> pixel(0, 255), label(0, classes - 1);
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> columns;
    for (int f = 0; f < features; ++f) {
        arrow::UInt8Builder builder;
        (void)builder.Reserve(rows);
        for (int64_t r = 0; r < rows; ++r) builder.UnsafeAppend(static_cast<uint8_t>(pixel(rng)));
        std::shared_ptr<arrow::Array> column;
        (void)builder.Finish(&column);
        fields.push_back(arrow::field("p" + std::to_string(f), arrow::uint8()));
        columns.push_back(column);
    }
    arrow::Int64Builder labels;
    (void)labels.Reserve(rows);
    for (int64_t r = 0; r < rows; ++r) labels.UnsafeAppend(label(rng));
    std::shared_ptr<arrow::Array> label_column;
    (void)labels.Finish(&label_column);
    fields.push_back(arrow::field("label", arrow::int64()));
    columns.push_back(label_column);
    return arrow::Table::Make(arrow::schema(fields), columns);
}

double Seconds(Clock::time_point since) {
    return std::chrono::duration<double>(Clock::now() - since).count();
}

}  // namespace

int main(int argc, char** argv) {
    if (!cyxwiz::test::SelectTestDeviceFromEnvironment()) {
        std::cerr << "the requested device could not be activated\n";
        return 1;
    }
    const int64_t rows = std::stoll(Arg(argc, argv, "--rows", "60000"));
    const int features = std::stoi(Arg(argc, argv, "--features", "784"));
    const size_t batch = static_cast<size_t>(std::stoul(Arg(argc, argv, "--batch", "128")));
    const int epochs = std::stoi(Arg(argc, argv, "--epochs", "2"));
    constexpr int kClasses = 10;

    const auto table = MakeTable(rows, features, kClasses);
    auto dataset = std::make_shared<cyxwiz::ArrowDataset>(table, "bench_training_data");
    cyxwiz::ArrowDatasetBatcher batcher(dataset, "label", batch, true, 1.0f, true);
    batcher.SetNormalization(0.1307f * 255.0f, 0.3081f * 255.0f);
    batcher.SetOneHotEncoding(kClasses);

    cyxwiz::SequentialModel model;
    model.Add<cyxwiz::LinearModule>(static_cast<size_t>(features), 128);
    model.Add<cyxwiz::ReLUModule>();
    model.Add<cyxwiz::LinearModule>(128, kClasses);
    cyxwiz::CrossEntropyLoss loss;
    auto optimizer = cyxwiz::CreateOptimizer(cyxwiz::OptimizerType::Adam, 0.001);

    const auto step = [&](const cyxwiz::Batch& b) {
        const cyxwiz::Tensor logits = model.Forward(b.data);
        (void)loss.Forward(logits, b.labels);
        model.Backward(loss.Backward(logits, b.labels));
        model.UpdateParameters(optimizer.get());
    };

    const auto report = [&](const char* what, int epoch, double seconds, size_t batches) {
        nlohmann::json line = {{"measure", what}, {"epoch", epoch}, {"seconds", seconds}, {"batches", batches},
                               {"rows", rows}, {"features", features}, {"batch", batch},
                               {"backend", af::getActiveBackend() == AF_BACKEND_CUDA     ? "cuda"
                                           : af::getActiveBackend() == AF_BACKEND_OPENCL ? "opencl"
                                                                                         : "cpu"}};
        std::cout << line.dump() << std::endl;
    };

    for (int e = 0; e < epochs; ++e) {
        batcher.Reset();
        af::sync();
        const auto start = Clock::now();
        size_t batches = 0;
        while (!batcher.IsEpochComplete()) {
            const cyxwiz::Batch b = batcher.GetNextBatch();
            if (b.size == 0) break;
            (void)b.data.GetArrayRowMajor2D();  // the upload the model's first layer does
            ++batches;
        }
        af::sync();
        report("data", e, Seconds(start), batches);
    }
    for (int e = 0; e < epochs; ++e) {
        batcher.Reset();
        af::sync();
        const auto start = Clock::now();
        size_t batches = 0;
        while (!batcher.IsEpochComplete()) {
            const cyxwiz::Batch b = batcher.GetNextBatch();
            if (b.size == 0) break;
            step(b);
            ++batches;
        }
        af::sync();
        report("train", e, Seconds(start), batches);
    }
    {
        batcher.Reset();
        const cyxwiz::Batch b = batcher.GetNextBatch();
        (void)b.data.GetArrayRowMajor2D();
        const size_t batches = (static_cast<size_t>(rows) + batch - 1) / batch;
        step(b);  // warm the kernels
        af::sync();
        const auto start = Clock::now();
        for (size_t i = 0; i < batches; ++i) step(b);
        af::sync();
        report("model", 0, Seconds(start), batches);
    }
    return 0;
}
