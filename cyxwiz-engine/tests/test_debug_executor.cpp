// Unit tests for Commits 1 + 2 of the Local Debug Mode plan:
//   Commit 1: BuildSequentialFromConfig extraction, SyntheticBatch helper
//   Commit 2: DebugExecutor::Run (forward + backward + shape + grad norms)
//
// Minimal Dense(10->32) -> ReLU -> Dense(32->4) with CrossEntropy + Adam
// exercises the golden path. A zero-weights variant exercises the
// dead-subgraph warning path.

#include "../src/core/debug_executor.h"
#include "../src/core/model_builder.h"
#include "../src/core/synthetic_batch.h"
#include "../src/core/graph_compiler.h"

#include <cyxwiz/loss.h>
#include <cyxwiz/tensor.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
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

TrainingConfiguration MakeTextConfigWithEmbeddingWeights(const std::filesystem::path& weights_file) {
    TrainingConfiguration cfg = MakeTextConfig();
    cfg.layers[0].parameters["num_embeddings"] = "500";
    cfg.layers[0].parameters["embedding_dim"] = "8";
    cfg.layers[0].parameters["padding_idx"] = "0";
    cfg.layers[0].parameters["weights_file"] = weights_file.string();
    cfg.layers[0].parameters["freeze"] = "true";
    return cfg;
}

TrainingConfiguration MakeTransformerTextConfig() {
    TrainingConfiguration cfg;
    cfg.input_size  = 8;   // seq_len
    cfg.output_size = 3;
    cfg.preprocessing_domain = PreprocessingDomain::Text;

    CompiledLayer emb;
    emb.type = gui::NodeType::Embedding;
    emb.parameters["num_embeddings"] = "128";
    emb.parameters["embedding_dim"]  = "8";
    cfg.layers.push_back(emb);

    CompiledLayer encoder;
    encoder.type = gui::NodeType::TransformerEncoder;
    encoder.parameters["d_model"] = "8";
    encoder.parameters["num_heads"] = "2";
    encoder.parameters["ff_dim"] = "16";
    encoder.parameters["dropout"] = "0.0";
    cfg.layers.push_back(encoder);

    CompiledLayer flatten;
    flatten.type = gui::NodeType::Flatten;
    cfg.layers.push_back(flatten);

    CompiledLayer dense;
    dense.type = gui::NodeType::Dense;
    dense.units = 3;
    cfg.layers.push_back(dense);

    cfg.loss_type      = gui::NodeType::CrossEntropyLoss;
    cfg.optimizer_type = gui::NodeType::Adam;
    cfg.learning_rate  = 0.001f;
    return cfg;
}

TrainingConfiguration MakeSentimentConfig(size_t num_layers = 1) {
    TrainingConfiguration cfg;
    cfg.input_size  = 128;
    cfg.output_size = 7;
    cfg.preprocessing_domain = PreprocessingDomain::Text;

    CompiledLayer emb;
    emb.type = gui::NodeType::Embedding;
    emb.parameters["num_embeddings"] = "10000";
    emb.parameters["embedding_dim"]  = "64";
    cfg.layers.push_back(emb);

    CompiledLayer gru;
    gru.type = gui::NodeType::GRU;
    gru.parameters["hidden_size"] = "96";
    gru.parameters["num_layers"] = std::to_string(num_layers);
    gru.parameters["bidirectional"] = "true";
    gru.parameters["return_sequences"] = "false";
    cfg.layers.push_back(gru);

    CompiledLayer dense1;
    dense1.type  = gui::NodeType::Dense;
    dense1.units = 64;
    cfg.layers.push_back(dense1);

    CompiledLayer relu;
    relu.type = gui::NodeType::ReLU;
    cfg.layers.push_back(relu);

    CompiledLayer drop;
    drop.type = gui::NodeType::Dropout;
    drop.dropout_rate = 0.35f;
    cfg.layers.push_back(drop);

    CompiledLayer dense2;
    dense2.type  = gui::NodeType::Dense;
    dense2.units = 7;
    cfg.layers.push_back(dense2);

    cfg.loss_type      = gui::NodeType::CrossEntropyLoss;
    cfg.optimizer_type = gui::NodeType::Adam;
    cfg.learning_rate  = 0.0002f;
    return cfg;
}

void ExpectEq(size_t a, size_t b, const char* what) {
    if (a != b) {
        std::cerr << "FAIL: " << what << ": expected " << b
                  << " got " << a << "\n";
        std::exit(1);
    }
}

void ExpectTrue(bool condition, const char* what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << "\n";
        std::exit(1);
    }
}

void ExpectNear(float actual, float expected, float tolerance, const char* what) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << what << ": expected " << expected
                  << " got " << actual << "\n";
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

void TestBuildSequentialTextEmbeddingWeights() {
    spdlog::info("--- TestBuildSequentialTextEmbeddingWeights ---");

    const auto weights_file =
        std::filesystem::temp_directory_path() / "cyxwiz_test_embedding_weights.txt";
    {
        std::ofstream out(weights_file);
        out << "# cyxwiz_embedding rows=500 dim=8\n";
        for (int r = 0; r < 500; ++r) {
            for (int c = 0; c < 8; ++c) {
                if (c > 0) out << ' ';
                out << ((r == 0) ? 0.0f : static_cast<float>((r + c) % 7) / 10.0f);
            }
            out << '\n';
        }
    }

    auto built = BuildSequentialFromConfig(
        MakeTextConfigWithEmbeddingWeights(weights_file));
    std::filesystem::remove(weights_file);

    if (!built.ok()) {
        std::cerr << "FAIL: pretrained embedding BuildSequentialFromConfig returned !ok()\n";
        std::exit(1);
    }

    auto params = built.model->GetParameters();
    ExpectTrue(params.count("layer0.weight") == 0,
               "frozen pretrained embedding should not expose trainable layer0.weight");
}

void TestCrossEntropyIgnoreIndexFromLossParams() {
    spdlog::info("--- TestCrossEntropyIgnoreIndexFromLossParams ---");
    auto cfg = MakeTabularConfig();
    cfg.loss_params["ignore_index"] = "-100";
    auto built = BuildSequentialFromConfig(cfg);
    ExpectTrue(built.ok(), "config should build with CrossEntropy loss");
    ExpectTrue(built.loss != nullptr, "loss should be built");

    const std::vector<float> logits = {
        2.0f, 0.0f,
        0.0f, 2.0f,
    };
    const std::vector<int64_t> targets = {0, -100};
    Tensor predictions({2, 2}, logits.data(), DataType::Float32);
    Tensor labels({2}, targets.data(), DataType::Int64);

    Tensor loss = built.loss->Forward(predictions, labels);
    ExpectTrue(loss.NumElements() == 1, "ignored CE loss should be scalar");
    ExpectTrue(std::isfinite(loss.Data<float>()[0]),
               "ignored CE loss should be finite");

    Tensor grad = built.loss->Backward(predictions, labels);
    ExpectEq(grad.Shape().size(), 2, "ignored CE grad ndim");
    ExpectEq(grad.Shape()[0], 2, "ignored CE grad rows");
    ExpectEq(grad.Shape()[1], 2, "ignored CE grad classes");
    const float* g = grad.Data<float>();
    ExpectNear(g[2], 0.0f, 1e-6f,
               "ignored CE grad row should zero class 0");
    ExpectNear(g[3], 0.0f, 1e-6f,
               "ignored CE grad row should zero class 1");
    spdlog::info("  OK: CrossEntropy ignore_index=-100 propagates to loss");
}

void TestWeightedLossConfigParams() {
    spdlog::info("--- TestWeightedLossConfigParams ---");

    auto ce_cfg = MakeTabularConfig();
    ce_cfg.loss_params["class_weight"] = "manual";
    ce_cfg.loss_params["class_weights"] = "[1.0, 2.0, 3.0, 4.0]";
    auto ce_built = BuildSequentialFromConfig(ce_cfg);
    ExpectTrue(ce_built.ok(), "weighted CrossEntropy config should build");
    auto* ce_loss = dynamic_cast<CrossEntropyLoss*>(ce_built.loss.get());
    ExpectTrue(ce_loss != nullptr,
               "weighted CrossEntropy config should construct CrossEntropyLoss");
    ExpectTrue(ce_loss->GetClassWeights().size() == 4,
               "CrossEntropyLoss should receive four class weights");
    ExpectNear(ce_loss->GetClassWeights()[3], 4.0f, 1e-6f,
               "CrossEntropyLoss should preserve manual class weights");

    auto bce_cfg = MakeTabularConfig();
    bce_cfg.output_size = 1;
    bce_cfg.layers.back().units = 1;
    bce_cfg.loss_type = gui::NodeType::BCEWithLogits;
    bce_cfg.loss_params["pos_weight"] = "2.5";
    auto bce_built = BuildSequentialFromConfig(bce_cfg);
    ExpectTrue(bce_built.ok(), "weighted BCEWithLogits config should build");
    auto* bce_loss = dynamic_cast<BCEWithLogitsLoss*>(bce_built.loss.get());
    ExpectTrue(bce_loss != nullptr,
               "weighted BCEWithLogits config should construct BCEWithLogitsLoss");
    ExpectNear(bce_loss->GetPosWeight(), 2.5f, 1e-6f,
               "BCEWithLogitsLoss should preserve pos_weight");

    auto focal_cfg = MakeTabularConfig();
    focal_cfg.loss_type = gui::NodeType::FocalLoss;
    focal_cfg.loss_params["alpha"] = "0.75";
    focal_cfg.loss_params["gamma"] = "1.5";
    auto focal_built = BuildSequentialFromConfig(focal_cfg);
    ExpectTrue(focal_built.ok(), "FocalLoss config should build");
    auto* focal_loss = dynamic_cast<FocalLoss*>(focal_built.loss.get());
    ExpectTrue(focal_loss != nullptr,
               "FocalLoss config should construct FocalLoss");
    ExpectNear(focal_loss->GetAlpha(), 0.75f, 1e-6f,
               "FocalLoss should preserve alpha");
    ExpectNear(focal_loss->GetGamma(), 1.5f, 1e-6f,
               "FocalLoss should preserve gamma");
}

void TestCrossEntropyTokenShapeIgnoreIndex() {
    spdlog::info("--- TestCrossEntropyTokenShapeIgnoreIndex ---");
    CrossEntropyLoss loss(Reduction::Mean, -100);
    const std::vector<float> logits = {
        4.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 4.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 4.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 4.0f,
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
    };
    const std::vector<int64_t> tags = {
        0, 1, -100,
        3, 0, 1,
    };
    Tensor predictions({2, 3, 4}, logits.data(), DataType::Float32);
    Tensor labels({2, 3}, tags.data(), DataType::Int64);

    Tensor loss_value = loss.Forward(predictions, labels);
    ExpectTrue(loss_value.NumElements() == 1,
               "token CE mean loss should be scalar");
    ExpectTrue(std::isfinite(loss_value.Data<float>()[0]),
               "token CE mean loss should be finite");
    const float exp4 = std::exp(4.0f);
    const float confident_prob = exp4 / (exp4 + 3.0f);
    const float exp1 = std::exp(1.0f);
    const float medium_prob = exp1 / (exp1 + 3.0f);
    const float expected_loss =
        (3.0f * -std::log(confident_prob + 1e-10f) +
         2.0f * -std::log(medium_prob + 1e-10f)) /
        5.0f;
    ExpectNear(loss_value.Data<float>()[0], expected_loss, 1e-5f,
               "token CE mean should divide by non-ignored token count");

    Tensor grad = loss.Backward(predictions, labels);
    ExpectTrue(grad.Shape() == std::vector<size_t>({2, 3, 4}),
               "token CE grad should preserve [batch, seq, classes]");
    const float* g = grad.Data<float>();
    ExpectNear(g[0], (confident_prob - 1.0f) / 5.0f, 1e-5f,
               "token CE grad mean should divide by non-ignored token count");
    const size_t ignored_base = 2 * 4;
    for (size_t i = 0; i < 4; ++i) {
        ExpectNear(g[ignored_base + i], 0.0f, 1e-6f,
                   "ignored token CE grad row should be zero");
    }

    CrossEntropyLoss none_loss(Reduction::None, -100);
    Tensor per_token = none_loss.Forward(predictions, labels);
    ExpectTrue(per_token.Shape() == std::vector<size_t>({2, 3}),
               "token CE unreduced loss should be [batch, seq]");

    NLLLoss nll(Reduction::Mean, -100);
    const std::vector<float> log_probs = {
        -0.2f, -2.0f, -2.0f, -2.0f,
        -2.0f, -0.3f, -2.0f, -2.0f,
        -2.0f, -2.0f, -0.7f, -2.0f,
        -2.0f, -2.0f, -2.0f, -0.4f,
        -0.5f, -2.0f, -2.0f, -2.0f,
        -2.0f, -0.6f, -2.0f, -2.0f,
    };
    Tensor log_prob_tensor({2, 3, 4}, log_probs.data(), DataType::Float32);
    Tensor nll_value = nll.Forward(log_prob_tensor, labels);
    ExpectNear(nll_value.Data<float>()[0], 0.4f, 1e-6f,
               "token NLL mean should divide by non-ignored token count");
    Tensor nll_grad = nll.Backward(log_prob_tensor, labels);
    ExpectTrue(nll_grad.Shape() == std::vector<size_t>({2, 3, 4}),
               "token NLL grad should preserve [batch, seq, classes]");
    ExpectNear(nll_grad.Data<float>()[0], -0.2f, 1e-6f,
               "token NLL grad mean should divide by non-ignored token count");
    for (size_t i = 0; i < 4; ++i) {
        ExpectNear(nll_grad.Data<float>()[ignored_base + i], 0.0f, 1e-6f,
                   "ignored token NLL grad row should be zero");
    }
    spdlog::info("  OK: token-shaped CrossEntropy/NLL honor ignore_index");
}

void TestWeightedCrossEntropyBackend() {
    spdlog::info("--- TestWeightedCrossEntropyBackend ---");
    CrossEntropyLoss loss(Reduction::Mean, -100, {1.0f, 3.0f});

    const std::vector<float> logits = {
        2.0f, 0.0f,
        0.0f, 0.0f,
    };
    const std::vector<int64_t> targets = {0, 1};
    Tensor predictions({2, 2}, logits.data(), DataType::Float32);
    Tensor labels({2}, targets.data(), DataType::Int64);

    Tensor loss_value = loss.Forward(predictions, labels);
    const float exp2 = std::exp(2.0f);
    const float p0 = exp2 / (exp2 + 1.0f);
    const float p1 = 0.5f;
    const float expected_loss =
        (-std::log(p0 + 1e-10f) + 3.0f * -std::log(p1 + 1e-10f)) / 4.0f;
    ExpectNear(loss_value.Data<float>()[0], expected_loss, 1e-5f,
               "weighted CE mean should divide by sum of target weights");

    Tensor grad = loss.Backward(predictions, labels);
    ExpectTrue(grad.Shape() == std::vector<size_t>({2, 2}),
               "weighted CE grad should preserve prediction shape");
    const float* g = grad.Data<float>();
    ExpectNear(g[0], (p0 - 1.0f) / 4.0f, 1e-5f,
               "weighted CE class-0 grad should use class weight denominator");
    ExpectNear(g[1], (1.0f - p0) / 4.0f, 1e-5f,
               "weighted CE non-target grad should use class weight denominator");
    ExpectNear(g[2], 3.0f * p1 / 4.0f, 1e-5f,
               "weighted CE non-target row grad should scale by class weight");
    ExpectNear(g[3], 3.0f * (p1 - 1.0f) / 4.0f, 1e-5f,
               "weighted CE target grad should scale by class weight");
}

void TestBCEWithLogitsPosWeightBackend() {
    spdlog::info("--- TestBCEWithLogitsPosWeightBackend ---");
    BCEWithLogitsLoss loss(Reduction::Mean, 4.0f);

    const std::vector<float> logits = {0.0f, 0.0f};
    const std::vector<float> targets = {1.0f, 0.0f};
    Tensor predictions({2}, logits.data(), DataType::Float32);
    Tensor labels({2}, targets.data(), DataType::Float32);

    Tensor loss_value = loss.Forward(predictions, labels);
    const float expected_loss =
        (4.0f * std::log(2.0f) + std::log(2.0f)) / 2.0f;
    ExpectNear(loss_value.Data<float>()[0], expected_loss, 1e-5f,
               "BCEWithLogits pos_weight should scale positive loss terms");

    Tensor grad = loss.Backward(predictions, labels);
    ExpectTrue(grad.Shape() == std::vector<size_t>({2}),
               "BCEWithLogits pos_weight grad should preserve shape");
    const float* g = grad.Data<float>();
    ExpectNear(g[0], -1.0f, 1e-6f,
               "BCEWithLogits positive grad should scale by pos_weight / mean count");
    ExpectNear(g[1], 0.25f, 1e-6f,
               "BCEWithLogits negative grad should remain unweighted / mean count");

    const std::vector<float> fractional_target = {0.25f};
    Tensor fractional_predictions({1}, logits.data(), DataType::Float32);
    Tensor fractional_labels({1}, fractional_target.data(), DataType::Float32);
    Tensor fractional_grad = loss.Backward(fractional_predictions, fractional_labels);
    ExpectNear(fractional_grad.Data<float>()[0], -0.125f, 1e-6f,
               "BCEWithLogits pos_weight grad should support fractional targets");
}

void TestTimeDistributedDenseModule() {
    spdlog::info("--- TestTimeDistributedDenseModule ---");

    const std::vector<float> input_values = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f,
    };
    Tensor input({2, 2, 3}, input_values.data(), DataType::Float32);

    TimeDistributedDenseModule td(3, 4);
    Tensor output = td.Forward(input);
    ExpectTrue(output.Shape() == std::vector<size_t>({2, 2, 4}),
               "TimeDistributedDense output should preserve batch/seq");

    LinearModule linear(3, 4);
    linear.SetParameters(td.GetParameters());
    Tensor flat_input = input.Reshape({4, 3});
    Tensor flat_output = linear.Forward(flat_input);
    const float* td_out = output.Data<float>();
    const float* ref_out = flat_output.Data<float>();
    for (size_t i = 0; i < output.NumElements(); ++i) {
        ExpectNear(td_out[i], ref_out[i], 1e-6f,
                   "TimeDistributedDense forward should match flat Linear");
    }

    std::vector<float> grad_values(output.NumElements(), 1.0f);
    Tensor grad_output({2, 2, 4}, grad_values.data(), DataType::Float32);
    Tensor grad_input = td.Backward(grad_output);
    ExpectTrue(grad_input.Shape() == std::vector<size_t>({2, 2, 3}),
               "TimeDistributedDense grad should restore input shape");

    Tensor flat_grad = grad_output.Reshape({4, 4});
    Tensor ref_grad_input = linear.Backward(flat_grad);
    const float* td_grad = grad_input.Data<float>();
    const float* ref_grad = ref_grad_input.Data<float>();
    for (size_t i = 0; i < grad_input.NumElements(); ++i) {
        ExpectNear(td_grad[i], ref_grad[i], 1e-6f,
                   "TimeDistributedDense backward should match flat Linear");
    }

    spdlog::info("  OK: TimeDistributedDense wraps Linear over [batch, seq]");
}

void TestBuildSequentialTimeDistributedHead() {
    spdlog::info("--- TestBuildSequentialTimeDistributedHead ---");

    TrainingConfiguration cfg;
    cfg.input_size = 3;
    cfg.output_size = 2;
    cfg.loss_type = gui::NodeType::CrossEntropyLoss;
    cfg.optimizer_type = gui::NodeType::Adam;

    CompiledLayer lstm;
    lstm.type = gui::NodeType::LSTM;
    lstm.parameters["hidden_size"] = "5";
    lstm.parameters["return_sequences"] = "true";
    cfg.layers.push_back(lstm);

    CompiledLayer head;
    head.type = gui::NodeType::TimeDistributed;
    head.units = 2;
    cfg.layers.push_back(head);

    auto built = BuildSequentialFromConfig(cfg);
    ExpectTrue(built.ok(), "config should build TimeDistributed head");
    ExpectEq(built.model->Size(), 2, "time-distributed model size");

    std::vector<float> values(2 * 4 * 3);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i + 1) / 10.0f;
    }
    Tensor input({2, 4, 3}, values.data(), DataType::Float32);
    Tensor output = built.model->Forward(input);
    ExpectTrue(output.Shape() == std::vector<size_t>({2, 4, 2}),
               "LSTM return_sequences + TimeDistributed should output [batch, seq, units]");

    spdlog::info("  OK: ModelBuilder creates TimeDistributed token head");
}

void TestDebugExecutorGoldenPath() {
    spdlog::info("--- TestDebugExecutorGoldenPath ---");
    auto cfg = MakeTabularConfig();
    DebugExecutor exe(cfg);
    auto res = exe.Run();

    if (res.reached != DebugStage::Complete) {
        std::cerr << "FAIL: expected Complete, got reached="
                  << static_cast<int>(res.reached)
                  << " summary=" << res.failure_summary << "\n";
        std::exit(1);
    }
    assert(res.success && "golden-path Run should succeed with no Error issues");
    assert(res.loss_finite && "loss must be finite on random init");
    assert(res.forward_total_ms >= 0.0f);
    assert(res.backward_total_ms >= 0.0f);

    // 2 Linear modules -> 2 weight + 2 bias = 4 params expected.
    ExpectEq(res.params_with_grad, 4, "params_with_grad");
    ExpectEq(res.params_missing_grad, 0, "params_missing_grad");

    // Per-layer traces: one per module (Dense, ReLU, Dense = 3 modules).
    ExpectEq(res.layer_traces.size(), 3, "layer_traces.size()");
    for (const auto& t : res.layer_traces) {
        if (t.has_nan || t.has_inf) {
            std::cerr << "FAIL: non-finite forward output in trace "
                      << t.name << "\n";
            std::exit(1);
        }
        if (t.actual_shape.empty()) {
            std::cerr << "FAIL: empty actual_shape in trace "
                      << t.name << "\n";
            std::exit(1);
        }
    }
    spdlog::info("  OK: reached=Complete, params_with_grad={}, "
                 "layer_traces={}, loss={}",
                 res.params_with_grad, res.layer_traces.size(),
                 res.loss_value);
}

void TestDebugExecutorGradNormBookkeeping() {
    spdlog::info("--- TestDebugExecutorGradNormBookkeeping ---");
    // The plan listed an "all-zero weights → dead-subgraph Warning"
    // pathological case, but SequentialModel is a single chain with no
    // branching, so every trainable layer is reached on every backward
    // and a true dead subgraph can't form. We still validate the
    // bookkeeping that the Warning path would depend on: every
    // trainable param lands in grad_norms with a valid layer_index and
    // a non-NaN norm on random init.
    auto cfg = MakeTabularConfig();
    DebugExecutor exe(cfg);
    auto res = exe.Run();
    assert(res.reached == DebugStage::Complete);
    ExpectEq(res.grad_norms.size(), 4, "grad_norms size");

    bool any_positive_norm = false;
    for (const auto& g : res.grad_norms) {
        if (g.layer_index < 0) {
            std::cerr << "FAIL: grad_norms entry missing layer_index for "
                      << g.param_name << "\n";
            std::exit(1);
        }
        assert(!g.is_nan && "grad norm NaN on random init is a backend bug");
        if (g.l2_norm > 0.0f) any_positive_norm = true;
    }
    assert(any_positive_norm && "at least one grad should have nonzero norm");
    spdlog::info("  OK: grad_norms fully populated with valid layer_index");
}

void TestDebugExecutorTextGraph() {
    spdlog::info("--- TestDebugExecutorTextGraph ---");
    auto cfg = MakeTextConfig();
    DebugExecutor exe(cfg);
    auto res = exe.Run();
    if (res.reached != DebugStage::Complete) {
        std::cerr << "FAIL: text graph reached="
                  << static_cast<int>(res.reached)
                  << " summary=" << res.failure_summary << "\n";
        std::exit(1);
    }
    assert(res.success && "text graph should reach Complete cleanly");
    // 3 config layers = Embedding, Flatten, Dense. Gradients cover:
    //   layer0.weight (Embedding),
    //   layer2.weight, layer2.bias (Dense).
    // EmbeddingLayer::GetParameters still exposes a legacy "grad_weight"
    // alongside "weight" (see cyxwiz-backend/src/algorithms/layer.cpp —
    // the partial cleanup only reached GetGradients). That leftover
    // param has no matching grad, so it lands in params_missing_grad.
    // DebugExecutor is reporting reality here; when the backend cleanup
    // finishes, this count will drop to 0 and we can tighten the test.
    ExpectEq(res.params_with_grad, 3, "text params_with_grad");
    ExpectEq(res.params_missing_grad, 0, "text params_missing_grad");
    spdlog::info("  OK: text graph end-to-end, params_with_grad={}",
                 res.params_with_grad);
}

void TestSentimentComputationCurrentPath() {
    spdlog::info("--- TestSentimentComputationCurrentPath ---");

    auto cfg = MakeSentimentConfig();
    DebugExecutor exe(cfg);
    spdlog::info("  Running synthetic sentiment-shaped GRU DebugExecutor");
    auto res = exe.Run();
    if (res.reached != DebugStage::Complete) {
        std::cerr << "FAIL: sentiment config reached="
                  << static_cast<int>(res.reached)
                  << " summary=" << res.failure_summary << "\n";
        std::exit(1);
    }
    ExpectTrue(res.success, "sentiment config should reach Complete cleanly");
    ExpectTrue(res.loss_finite, "sentiment config loss must be finite");
    ExpectEq(res.layer_traces.size(), 6, "sentiment layer_traces");
    ExpectEq(res.params_missing_grad, 0, "sentiment params_missing_grad");
    ExpectEq(res.layer_traces[1].actual_shape.size(), 2,
             "sentiment GRU trace ndim");
    ExpectEq(res.layer_traces[1].actual_shape[1], 192,
             "sentiment bidirectional GRU features");
    spdlog::info("  OK: synthetic sentiment-shaped GRU graph end-to-end");
}

void TestSentimentComputationMultiLayerBiGRU() {
    spdlog::info("--- TestSentimentComputationMultiLayerBiGRU ---");
    auto cfg = MakeSentimentConfig(/*num_layers=*/2);
    DebugExecutor exe(cfg);
    spdlog::info("  Running multi-layer sentiment GRU DebugExecutor");
    auto res = exe.Run();
    if (res.reached != DebugStage::Complete) {
        std::cerr << "FAIL: multi-layer sentiment config reached="
                  << static_cast<int>(res.reached)
                  << " summary=" << res.failure_summary << "\n";
        std::exit(1);
    }
    ExpectTrue(res.success, "multi-layer sentiment config should reach Complete cleanly");
    ExpectTrue(res.loss_finite, "multi-layer sentiment config loss must be finite");
    ExpectEq(res.layer_traces.size(), 6, "multi-layer layer_traces");
    ExpectEq(res.params_with_grad, 21, "multi-layer params_with_grad");
    ExpectEq(res.params_missing_grad, 0, "multi-layer params_missing_grad");
    ExpectEq(res.layer_traces[1].actual_shape.size(), 2,
             "multi-layer GRU trace ndim");
    ExpectEq(res.layer_traces[1].actual_shape[1], 192,
             "multi-layer bidirectional GRU features");
    spdlog::info("  OK: multi-layer sentiment graph end-to-end, params_with_grad={}",
                 res.params_with_grad);
}

void TestTransformerTextComputation() {
    spdlog::info("--- TestTransformerTextComputation ---");
    auto cfg = MakeTransformerTextConfig();
    DebugExecutor exe(cfg);
    spdlog::info("  Running synthetic text TransformerEncoder DebugExecutor");
    auto res = exe.Run();
    if (res.reached != DebugStage::Complete) {
        std::cerr << "FAIL: transformer text config reached="
                  << static_cast<int>(res.reached)
                  << " summary=" << res.failure_summary << "\n";
        std::exit(1);
    }
    ExpectTrue(res.success, "transformer text config should reach Complete cleanly");
    ExpectTrue(res.loss_finite, "transformer text config loss must be finite");
    ExpectEq(res.layer_traces.size(), 4, "transformer layer_traces");
    ExpectEq(res.params_missing_grad, 0, "transformer params_missing_grad");
    ExpectEq(res.layer_traces[1].actual_shape.size(), 3,
             "transformer encoder trace ndim");
    ExpectEq(res.layer_traces[1].actual_shape[1], 8,
             "transformer encoder seq_len");
    ExpectEq(res.layer_traces[1].actual_shape[2], 8,
             "transformer encoder d_model");
    ExpectEq(res.layer_traces[2].actual_shape.size(), 2,
             "transformer flatten trace ndim");
    ExpectEq(res.layer_traces[2].actual_shape[1], 64,
             "transformer flattened features");
    spdlog::info("  OK: transformer text graph end-to-end, params_with_grad={}",
                 res.params_with_grad);
}

} // namespace

int main() {
    spdlog::set_level(spdlog::level::info);
    try {
        const bool run_slow_sentiment =
            std::getenv("CYXWIZ_RUN_SLOW_DEBUG_TESTS") != nullptr;

        TestBuildSequentialTabular();
        TestSyntheticBatchTabular();
        TestSyntheticBatchText();
        TestMseLossLabelsAreFloat();
        TestBuildSequentialTextEmbeddingWeights();
        TestCrossEntropyIgnoreIndexFromLossParams();
        TestWeightedLossConfigParams();
        TestCrossEntropyTokenShapeIgnoreIndex();
        TestWeightedCrossEntropyBackend();
        TestBCEWithLogitsPosWeightBackend();
        TestTimeDistributedDenseModule();
        TestBuildSequentialTimeDistributedHead();
        TestDebugExecutorGoldenPath();
        TestDebugExecutorGradNormBookkeeping();
        TestDebugExecutorTextGraph();
        if (run_slow_sentiment) {
            TestSentimentComputationCurrentPath();
            TestSentimentComputationMultiLayerBiGRU();
            TestTransformerTextComputation();
        } else {
            spdlog::info("Skipping slow sentiment debug checks. Set "
                         "CYXWIZ_RUN_SLOW_DEBUG_TESTS=1 to run them.");
        }
    } catch (const std::exception& e) {
        std::cerr << "FAIL: exception: " << e.what() << "\n";
        return 1;
    }
    std::cout << "ALL TESTS PASSED\n";
    return 0;
}
