// Unit test for Commit 1 of the Local Debug Mode plan:
// - BuildSequentialFromConfig extraction (model_builder.cpp)
// - SyntheticBatch helper for Tabular + Text domains
//
// Builds a minimal Dense(10->32) -> ReLU -> Dense(32->4) with CrossEntropy +
// Adam, runs the builder, generates a synthetic batch, asserts nothing
// throws and the shapes are what the domain spec promises.

#include "../src/core/model_builder.h"
#include "../src/core/synthetic_batch.h"
#include "../src/core/graph_compiler.h"

#include <cyxwiz/tensor.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <iostream>
#include <stdexcept>

using namespace cyxwiz;

namespace {

TrainingConfiguration MakeTabularConfig() {
    TrainingConfiguration cfg;
    cfg.input_size  = 10;
    cfg.output_size = 4;
    cfg.preprocessing_domain = PreprocessingDomain::Tabular;

    CompiledLayer l1;
    l1.type  = gui::NodeType::Dense;
    l1.units = 32;
    cfg.layers.push_back(l1);

    CompiledLayer l2;
    l2.type = gui::NodeType::ReLU;
    cfg.layers.push_back(l2);

    CompiledLayer l3;
    l3.type  = gui::NodeType::Dense;
    l3.units = 4;
    cfg.layers.push_back(l3);

    cfg.loss_type      = gui::NodeType::CrossEntropyLoss;
    cfg.optimizer_type = gui::NodeType::Adam;
    cfg.learning_rate  = 0.001f;
    return cfg;
}

TrainingConfiguration MakeTextConfig() {
    TrainingConfiguration cfg;
    cfg.input_size  = 16;   // seq_len
    cfg.output_size = 3;
    cfg.preprocessing_domain = PreprocessingDomain::Text;

    CompiledLayer emb;
    emb.type = gui::NodeType::Embedding;
    emb.parameters["num_embeddings"] = "500";
    emb.parameters["embedding_dim"]  = "8";
    cfg.layers.push_back(emb);

    CompiledLayer flatten;
    flatten.type = gui::NodeType::Flatten;
    cfg.layers.push_back(flatten);

    CompiledLayer dense;
    dense.type  = gui::NodeType::Dense;
    dense.units = 3;
    cfg.layers.push_back(dense);

    cfg.loss_type      = gui::NodeType::CrossEntropyLoss;
    cfg.optimizer_type = gui::NodeType::Adam;
    cfg.learning_rate  = 0.001f;
    return cfg;
}

void ExpectEq(size_t a, size_t b, const char* what) {
    if (a != b) {
        std::cerr << "FAIL: " << what << ": expected " << b
                  << " got " << a << "\n";
        std::exit(1);
    }
}

void TestBuildSequentialTabular() {
    spdlog::info("--- TestBuildSequentialTabular ---");
    auto cfg = MakeTabularConfig();
    auto built = BuildSequentialFromConfig(cfg);
    if (!built.ok()) {
        std::cerr << "FAIL: BuildSequentialFromConfig returned !ok()\n";
        std::exit(1);
    }
    assert(built.model && built.loss && built.optimizer);
    // Layer count: Dense, ReLU, Dense = 3 modules. Output node / losses
    // are filtered out by the builder.
    ExpectEq(built.model->Size(), 3, "model.Size()");
    spdlog::info("  OK: model has {} modules", built.model->Size());
}

void TestSyntheticBatchTabular() {
    spdlog::info("--- TestSyntheticBatchTabular ---");
    auto cfg = MakeTabularConfig();
    auto batch = MakeSyntheticBatch(cfg, /*seed=*/1337);

    ExpectEq(batch.features.Shape().size(), 2, "features ndim");
    ExpectEq(batch.features.Shape()[0], 1,  "features dim0 (batch)");
    ExpectEq(batch.features.Shape()[1], 10, "features dim1 (input_size)");
    assert(batch.features.GetDataType() == DataType::Float32);

    ExpectEq(batch.labels.Shape().size(), 1, "labels ndim (CE)");
    ExpectEq(batch.labels.Shape()[0], 1, "labels dim0");
    assert(batch.labels.GetDataType() == DataType::Int64);
    spdlog::info("  OK: features=[1,10] float32, labels=[1] int64");

    // Reproducibility: same seed -> identical bytes.
    auto batch2 = MakeSyntheticBatch(cfg, /*seed=*/1337);
    const float* a = batch.features.Data<float>();
    const float* b = batch2.features.Data<float>();
    for (size_t i = 0; i < batch.features.NumElements(); ++i) {
        if (a[i] != b[i]) {
            std::cerr << "FAIL: non-reproducible features at i=" << i
                      << " " << a[i] << " vs " << b[i] << "\n";
            std::exit(1);
        }
    }
    spdlog::info("  OK: reproducible across calls with same seed");
}

void TestSyntheticBatchText() {
    spdlog::info("--- TestSyntheticBatchText ---");
    auto cfg = MakeTextConfig();
    auto batch = MakeSyntheticBatch(cfg, /*seed=*/1337);

    ExpectEq(batch.features.Shape().size(), 2, "features ndim");
    ExpectEq(batch.features.Shape()[0], 1,  "features dim0");
    ExpectEq(batch.features.Shape()[1], 16, "features dim1 (seq_len)");
    assert(batch.features.GetDataType() == DataType::Int64);

    // Every token ID must fall inside [0, num_embeddings).
    const int64_t* ids = batch.features.Data<int64_t>();
    for (size_t i = 0; i < batch.features.NumElements(); ++i) {
        if (ids[i] < 0 || ids[i] >= 500) {
            std::cerr << "FAIL: token id " << ids[i]
                      << " out of [0, 500) at i=" << i << "\n";
            std::exit(1);
        }
    }
    spdlog::info("  OK: text batch ids within vocab range");
}

void TestMseLossLabelsAreFloat() {
    spdlog::info("--- TestMseLossLabelsAreFloat ---");
    auto cfg = MakeTabularConfig();
    cfg.loss_type = gui::NodeType::MSELoss;
    auto batch = MakeSyntheticBatch(cfg, /*seed=*/7);
    assert(batch.labels.GetDataType() == DataType::Float32);
    ExpectEq(batch.labels.Shape()[0], 1, "mse labels dim0");
    ExpectEq(batch.labels.Shape()[1], 4, "mse labels dim1 (output_size)");
    spdlog::info("  OK: MSE labels are float [1, output_size]");
}

} // namespace

int main() {
    spdlog::set_level(spdlog::level::info);
    try {
        TestBuildSequentialTabular();
        TestSyntheticBatchTabular();
        TestSyntheticBatchText();
        TestMseLossLabelsAreFloat();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: exception: " << e.what() << "\n";
        return 1;
    }
    std::cout << "ALL TESTS PASSED\n";
    return 0;
}
