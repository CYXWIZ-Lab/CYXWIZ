// Unit tests for Commits 1 + 2 of the Local Debug Mode plan:
//   Commit 1: BuildSequentialFromConfig extraction, SyntheticBatch helper
//   Commit 2: DebugExecutor::Run (forward + backward + shape + grad norms)
//
// Minimal Dense(10->32) -> ReLU -> Dense(32->4) with CrossEntropy + Adam
// exercises the golden path. A zero-weights variant exercises the
// dead-subgraph warning path.

#include "../src/core/debug_executor.h"
#include "../src/core/debug_model_session.h"
#include "../src/core/occlusion.h"
#include "../src/core/saliency.h"
#include "../src/core/top_images.h"
#include "../src/core/dense_contributions.h"
#include "../src/core/layer_embedding.h"
#include "../src/core/unit_weights.h"
#include "../src/core/dataset_batcher.h"
#include "../src/core/training_snapshots_presentation.h"
#include "../src/core/where_it_looks_presentation.h"
#include "../src/core/debug_numerics.h"
#include "../src/core/debug_loss_metric_explainer.h"
#include "../src/core/model_builder.h"
#include "../src/core/synthetic_batch.h"
#include "../src/core/graph_compiler.h"
#include "../src/core/graph_compiler_dataset_hooks.h"
#include "../src/core/graph_document.h"
#include "../src/core/checkpoint_manifest.h"
#include "../src/core/checkpoint_payload_io.h"
#include "../src/core/checkpoint_manager.h"
#include "../src/core/debug_run_paths.h"
#include "../src/core/layer_capture_store.h"
#include "../src/core/layer_tensor_reader.h"
#include "../src/core/tensor_file.h"
#include "algorithms/arrayfire_backend_utils.h"

#include <cyxwiz/loss.h>
#include <cyxwiz/optimizers/sgd.h>
#include <cyxwiz/tensor.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>

using namespace cyxwiz;

namespace {

TrainingConfiguration MakeTabularConfig() {
    TrainingConfiguration cfg;
    cfg.input_size  = 10;
    cfg.output_size = 4;
    cfg.preprocessing_domain = PreprocessingDomain::Tabular;

    CompiledLayer l1;
    l1.type  = gui::NodeType::Dense;
    l1.node_id = 101;
    l1.name = "Input Dense";
    l1.units = 32;
    l1.input_shape = {10};
    l1.output_shape = {32};
    cfg.layers.push_back(l1);

    CompiledLayer l2;
    l2.type = gui::NodeType::ReLU;
    l2.node_id = 102;
    l2.name = "Hidden ReLU";
    l2.input_shape = {32};
    l2.output_shape = {32};
    cfg.layers.push_back(l2);

    CompiledLayer l3;
    l3.type  = gui::NodeType::Dense;
    l3.node_id = 103;
    l3.name = "Output Dense";
    l3.units = 4;
    l3.input_shape = {32};
    l3.output_shape = {4};
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

TrainingConfiguration MakeLayerNormConfig() {
    TrainingConfiguration cfg;
    cfg.input_size = 4;
    cfg.output_size = 2;
    cfg.preprocessing_domain = PreprocessingDomain::Tabular;

    CompiledLayer norm;
    norm.type = gui::NodeType::LayerNorm;
    norm.parameters["normalized_shape"] = "4";
    norm.parameters["epsilon"] = "1e-5";
    norm.parameters["elementwise_affine"] = "true";
    cfg.layers.push_back(norm);

    CompiledLayer dense;
    dense.type = gui::NodeType::Dense;
    dense.units = 2;
    cfg.layers.push_back(dense);

    cfg.loss_type = gui::NodeType::MSELoss;
    cfg.optimizer_type = gui::NodeType::Adam;
    cfg.learning_rate = 0.001f;
    return cfg;
}

TrainingConfiguration MakeMultiHeadAttentionConfig() {
    TrainingConfiguration cfg;
    cfg.input_size = 4;
    cfg.output_size = 4;
    cfg.preprocessing_domain = PreprocessingDomain::Text;

    CompiledLayer mha;
    mha.type = gui::NodeType::MultiHeadAttention;
    mha.parameters["embed_dim"] = "4";
    mha.parameters["num_heads"] = "2";
    mha.parameters["dropout"] = "0";
    cfg.layers.push_back(mha);

    cfg.loss_type = gui::NodeType::MSELoss;
    cfg.optimizer_type = gui::NodeType::Adam;
    cfg.learning_rate = 0.001f;
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
    drop.parameters["rate"] = "0.35";
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

void ExpectEq(size_t a, size_t b, const std::string& what) {
    if (a != b) {
        std::cerr << "FAIL: " << what << ": expected " << b
                  << " got " << a << "\n";
        std::exit(1);
    }
}

void ExpectTrue(bool condition, const std::string& what) {
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
    ExpectEq(built.module_provenance.size(), 3,
             "module_provenance.size()");
    for (size_t i = 0; i < built.module_provenance.size(); ++i) {
        const auto& provenance = built.module_provenance[i];
        ExpectTrue(provenance.created(),
                   "tabular layer should create a module");
        ExpectEq(*provenance.module_index, i,
                 "module provenance index");
        ExpectEq(provenance.compiled_layer_index, i,
                 "compiled layer provenance index");
        ExpectEq(provenance.node_id, 101 + static_cast<int>(i),
                 "module provenance node id");
    }

    auto marker_cfg = MakeTabularConfig();
    CompiledLayer output_marker;
    output_marker.type = gui::NodeType::Output;
    output_marker.node_id = 104;
    output_marker.name = "Output";
    output_marker.input_shape = {4};
    output_marker.output_shape = {4};
    marker_cfg.layers.push_back(output_marker);
    auto marker_built = BuildSequentialFromConfig(marker_cfg);
    ExpectTrue(marker_built.ok(),
               "Output marker should not prevent model construction");
    ExpectEq(marker_built.model->Size(), 3,
             "Output marker must not add another projection");
    ExpectEq(marker_built.module_provenance.size(), 4,
             "Output marker should retain provenance without a module");
    ExpectTrue(!marker_built.module_provenance.back().created(),
               "Output marker provenance should not own a module");

    auto skipped_cfg = MakeTabularConfig();
    // A layer type the builder has no module for (Lambda is still
    // a template node). Conv2D used to be that type; since c5433f95 (TOFIX140
    // A1) it builds a real module and needs the spatial head's input shape.
    CompiledLayer unsupported;
    unsupported.type = gui::NodeType::Lambda;
    unsupported.node_id = 99;
    unsupported.name = "Unsupported Lambda";
    skipped_cfg.layers.insert(skipped_cfg.layers.begin(), unsupported);
    auto skipped = BuildSequentialFromConfig(skipped_cfg);
    ExpectTrue(skipped.ok(),
               "supported layers after a skipped layer should still build");
    ExpectEq(skipped.module_provenance.size(), 4,
             "skipped module provenance size");
    ExpectTrue(!skipped.module_provenance[0].created(),
               "unsupported compiled layer should be recorded as skipped");
    ExpectEq(skipped.module_provenance[0].node_id, 99,
             "skipped provenance node id");
    ExpectEq(*skipped.module_provenance[1].module_index, 0,
             "first created module index after skipped layer");
    ExpectEq(skipped.module_provenance[1].compiled_layer_index, 1,
             "compiled index should not be inferred from module index");
    spdlog::info("  OK: model has {} modules", built.model->Size());
}

void TestDenseActivationConfigurationContract() {
    spdlog::info("--- TestDenseActivationConfigurationContract ---");

    auto config = MakeTabularConfig();
    config.layers.resize(1);
    config.layers[0].parameters["units"] = "0";
    auto built = BuildSequentialFromConfig(config);
    ExpectTrue(!built.ok() &&
                   built.error_message.find("Dense units") !=
                       std::string::npos,
               "ModelBuilder must reject an invalid persisted Dense width");

    config = MakeTabularConfig();
    config.layers.resize(1);
    config.layers[0].units = -1;
    built = BuildSequentialFromConfig(config);
    ExpectTrue(!built.ok() &&
                   built.error_message.find("Dense units") !=
                       std::string::npos,
               "ModelBuilder must reject an invalid direct Dense width");

    config = MakeTabularConfig();
    config.layers.clear();
    CompiledLayer leaky;
    leaky.type = gui::NodeType::LeakyReLU;
    leaky.parameters["negative_slope"] = "0";
    config.layers.push_back(leaky);
    built = BuildSequentialFromConfig(config);
    ExpectTrue(built.ok() && built.model->Size() == 1,
               "LeakyReLU negative_slope=0 must remain an exact valid value");
    ExpectTrue(built.model->GetModule(0)->GetName().find("0.000000") !=
                   std::string::npos,
               "LeakyReLU constructor must receive the exact zero slope");

    config.layers[0].parameters["negative_slope"] = "nan";
    built = BuildSequentialFromConfig(config);
    ExpectTrue(!built.ok() &&
                   built.error_message.find("finite number") !=
                       std::string::npos,
               "LeakyReLU must reject non-finite persisted slopes");

    config.layers[0].type = gui::NodeType::ELU;
    config.layers[0].parameters.clear();
    config.layers[0].parameters["alpha"] = "0.25";
    built = BuildSequentialFromConfig(config);
    ExpectTrue(built.ok() &&
                   built.model->GetModule(0)->GetName().find("0.250000") !=
                       std::string::npos,
               "ELU constructor must receive the exact persisted alpha");

    config.layers[0].parameters["alpha"] = "0";
    built = BuildSequentialFromConfig(config);
    ExpectTrue(!built.ok() &&
                   built.error_message.find("ELU alpha") !=
                       std::string::npos,
               "ELU must reject a non-positive persisted alpha");

    spdlog::info("  OK: Dense and parameterized activation construction is exact and fail-closed");
}

TrainingConfiguration MakeBatchNormConfig() {
    TrainingConfiguration config;
    config.input_size = 2;
    config.output_size = 2;
    CompiledLayer norm;
    norm.type = gui::NodeType::BatchNorm;
    norm.parameters = {{"eps", "0.001"}, {"momentum", "0.1"}};
    config.layers.push_back(norm);
    return config;
}

void TestRecurrentConfigurationFailClosed() {
    spdlog::info("--- TestRecurrentConfigurationFailClosed ---");
    auto cfg = MakeTabularConfig();
    CompiledLayer recurrent;
    recurrent.type = gui::NodeType::LSTM;
    recurrent.node_id = 801;
    recurrent.name = "Unsupported BiLSTM";
    recurrent.parameters = {
        {"hidden_size", "8"},
        {"num_layers", "1"},
        {"bidirectional", "true"},
        {"return_sequences", "false"},
        {"dropout", "0.0"},
    };
    cfg.layers = {recurrent};

    auto built = BuildSequentialFromConfig(cfg);
    ExpectTrue(built.ok(),
               "ModelBuilder must build bidirectional LSTM through the split "
               "forward/reverse path");

    recurrent.type = gui::NodeType::GRU;
    recurrent.name = "Unsupported GRU Dropout";
    recurrent.parameters["bidirectional"] = "false";
    recurrent.parameters["dropout"] = "0.25";
    cfg.layers = {recurrent};
    built = BuildSequentialFromConfig(cfg);
    ExpectTrue(!built.ok() &&
                   built.error_message.find("not wired") != std::string::npos,
               "ModelBuilder must reject ignored recurrent dropout if compiler validation is bypassed");
}

void TestTransformerConfigurationFailClosed() {
    spdlog::info("--- TestTransformerConfigurationFailClosed ---");

    auto config = MakeMultiHeadAttentionConfig();
    config.layers[0].parameters["d_model"] = "8";
    auto built = BuildSequentialFromConfig(config);
    ExpectTrue(!built.ok() &&
                   built.error_message.find("conflicts") != std::string::npos,
               "ModelBuilder must reject conflicting attention width aliases");

    config = MakeMultiHeadAttentionConfig();
    config.layers[0].parameters["num_heads"] = "3";
    built = BuildSequentialFromConfig(config);
    ExpectTrue(!built.ok() &&
                   built.error_message.find("divisible") != std::string::npos,
               "ModelBuilder must reject non-divisible attention heads");

    config = MakeTransformerTextConfig();
    config.layers.resize(2);
    config.layers[1].parameters["num_layers"] = "6";
    built = BuildSequentialFromConfig(config);
    ExpectTrue(!built.ok() &&
                   built.error_message.find("exactly one block") !=
                       std::string::npos,
               "ModelBuilder must reject fictional internal transformer depth");

    config = MakeTransformerTextConfig();
    config.layers.resize(2);
    config.layers[1].parameters["d_model"] = "16";
    built = BuildSequentialFromConfig(config);
    ExpectTrue(!built.ok() &&
                   built.error_message.find("incoming feature width") !=
                       std::string::npos,
               "ModelBuilder must reject transformer width drift");

    TrainingConfiguration positional;
    positional.input_size = 4;
    positional.output_size = 4;
    CompiledLayer encoding;
    encoding.type = gui::NodeType::PositionalEncoding;
    encoding.parameters = {{"d_model", "4"},
                           {"max_len", "32"},
                           {"max_length", "32"}};
    positional.layers = {encoding};
    built = BuildSequentialFromConfig(positional);
    ExpectTrue(built.ok() && built.model->Size() == 1,
               "equal-valued positional aliases should remain executable");

    positional.layers[0].parameters["max_length"] = "64";
    built = BuildSequentialFromConfig(positional);
    ExpectTrue(!built.ok() &&
                   built.error_message.find("conflicts") != std::string::npos,
               "ModelBuilder must reject conflicting positional aliases");
}

void TestBuildSequentialLayerNorm() {
    spdlog::info("--- TestBuildSequentialLayerNorm ---");
    auto built = BuildSequentialFromConfig(MakeLayerNormConfig());
    if (!built.ok()) {
        std::cerr << "FAIL: LayerNorm BuildSequentialFromConfig returned !ok()\n";
        std::exit(1);
    }
    ExpectEq(built.model->Size(), 2, "LayerNorm model.Size()");

    const float x_values[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        -1.0f, 0.0f, 1.0f, 2.0f,
    };
    Tensor input({2, 4}, x_values, DataType::Float32);
    Tensor output = built.model->Forward(input);
    ExpectEq(output.Shape().size(), 2, "LayerNorm output ndim");
    ExpectEq(output.Shape()[0], 2, "LayerNorm output batch");
    ExpectEq(output.Shape()[1], 2, "LayerNorm output features");

    const float grad_values[] = {
        0.1f, -0.2f,
        0.3f, 0.4f,
    };
    Tensor grad({2, 2}, grad_values, DataType::Float32);
    (void)built.model->Backward(grad);

    const auto grads = built.model->GetGradients();
    ExpectTrue(grads.count("layer0.gamma") == 1,
               "LayerNorm gamma gradient is exposed");
    ExpectTrue(grads.count("layer0.beta") == 1,
               "LayerNorm beta gradient is exposed");
    spdlog::info("  OK: LayerNorm builds, runs, and exposes gradients");
}

void TestNormalizationRegularizationConfigurationContract() {
    spdlog::info("--- TestNormalizationRegularizationConfigurationContract ---");

    TrainingConfiguration dropout_config;
    dropout_config.input_size = 2;
    dropout_config.output_size = 2;
    CompiledLayer dropout;
    dropout.type = gui::NodeType::Dropout;
    dropout.parameters["rate"] = "0";
    dropout_config.layers.push_back(dropout);
    auto built = BuildSequentialFromConfig(dropout_config);
    ExpectTrue(built.ok() && built.model->Size() == 1,
               "Dropout rate=0 should build as an exact executable value");
    const float values[] = {1.0f, -2.0f, 3.0f, -4.0f};
    Tensor input({2, 2}, values, DataType::Float32);
    Tensor output = built.model->Forward(input);
    const float* output_data = output.ReadData<float>();
    for (size_t i = 0; i < 4; ++i) {
        ExpectNear(output_data[i], values[i], 1e-7f,
                   "Dropout rate=0 should preserve training input");
    }

    dropout_config.layers[0].parameters["rate"] = "1";
    built = BuildSequentialFromConfig(dropout_config);
    ExpectTrue(built.ok() && built.model->Size() == 1,
               "Dropout rate=1 should build as the PyTorch all-zero boundary");
    output = built.model->Forward(input);
    output_data = output.ReadData<float>();
    for (size_t i = 0; i < 4; ++i) {
        ExpectNear(output_data[i], 0.0f, 0.0f,
                   "Dropout rate=1 should zero every training activation");
    }

    TrainingConfiguration batch_norm_config;
    batch_norm_config.input_size = 2;
    batch_norm_config.output_size = 2;
    CompiledLayer batch_norm;
    batch_norm.type = gui::NodeType::BatchNorm;
    batch_norm.parameters = {{"eps", "0.001"}, {"momentum", "0"}};
    batch_norm_config.layers.push_back(batch_norm);
    built = BuildSequentialFromConfig(batch_norm_config);
    ExpectTrue(built.ok() && built.model->Size() == 1,
               "BatchNorm momentum=0 should build as an exact executable value");
    (void)built.model->Forward(input);
    built.model->SetTraining(false);
    output = built.model->Forward(input);
    output_data = output.ReadData<float>();
    const float expected_scale = 1.0f / std::sqrt(1.001f);
    for (size_t i = 0; i < 4; ++i) {
        ExpectNear(output_data[i], values[i] * expected_scale, 1e-5f,
                   "BatchNorm momentum=0 should retain initial running statistics");
    }

    const auto state = built.model->GetParameters();
    ExpectTrue(state.count("layer0.running_mean") == 1 &&
                   state.count("layer0.running_var") == 1,
               "BatchNorm running statistics should participate in model state persistence");

    batch_norm_config.layers[0].parameters["epsilon"] = "0.01";
    built = BuildSequentialFromConfig(batch_norm_config);
    ExpectTrue(!built.ok() &&
                   built.error_message.find("conflicts") != std::string::npos,
               "BatchNorm conflicting epsilon aliases should fail closed");

    TrainingConfiguration layer_norm_config = MakeLayerNormConfig();
    layer_norm_config.layers[0].parameters["eps"] = "1e-5";
    built = BuildSequentialFromConfig(layer_norm_config);
    ExpectTrue(built.ok(),
               "LayerNorm equal canonical and legacy epsilon values should remain executable");
    layer_norm_config.layers[0].parameters["normalized_shape"] = "4,-1";
    built = BuildSequentialFromConfig(layer_norm_config);
    ExpectTrue(!built.ok() &&
                   built.error_message.find("normalized_shape") != std::string::npos,
               "LayerNorm invalid trailing dimensions should fail closed");

    bool direct_dropout_accepted = true;
    try {
        DropoutModule boundary(1.0f);
        (void)boundary;
    } catch (...) {
        direct_dropout_accepted = false;
    }
    ExpectTrue(direct_dropout_accepted,
               "direct DropoutModule construction should preserve PyTorch p=1");

    bool direct_batch_norm_rejected = false;
    try {
        BatchNormModule invalid(2, 0.0f, 0.1f);
        (void)invalid;
    } catch (const std::invalid_argument&) {
        direct_batch_norm_rejected = true;
    }
    ExpectTrue(direct_batch_norm_rejected,
               "direct BatchNormModule construction should reject eps=0");

    spdlog::info(
        "  OK: normalization/regularization exact values and invalid states are enforced");
}

void TestBuildSequentialMultiHeadAttention() {
    spdlog::info("--- TestBuildSequentialMultiHeadAttention ---");
    auto built = BuildSequentialFromConfig(MakeMultiHeadAttentionConfig());
    if (!built.ok()) {
        std::cerr << "FAIL: MultiHeadAttention BuildSequentialFromConfig returned !ok(): "
                  << built.error_message << "\n";
        std::exit(1);
    }
    ExpectEq(built.model->Size(), 1, "MultiHeadAttention model.Size()");

    const float x_values[] = {
        0.1f, -0.2f, 0.3f, 0.4f,
        -0.5f, 0.6f, -0.7f, 0.8f,
    };
    Tensor input({1, 2, 4}, x_values, DataType::Float32);
    Tensor output = built.model->Forward(input);
    ExpectEq(output.Shape().size(), 3, "MultiHeadAttention output ndim");
    ExpectEq(output.Shape()[0], 1, "MultiHeadAttention output batch");
    ExpectEq(output.Shape()[1], 2, "MultiHeadAttention output seq_len");
    ExpectEq(output.Shape()[2], 4, "MultiHeadAttention output embed_dim");

    const float grad_values[] = {
        0.01f, -0.02f, 0.03f, -0.04f,
        0.05f, -0.06f, 0.07f, -0.08f,
    };
    Tensor grad({1, 2, 4}, grad_values, DataType::Float32);
    Tensor input_grad = built.model->Backward(grad);
    ExpectEq(input_grad.Shape().size(), 3, "MultiHeadAttention grad_input ndim");
    ExpectEq(input_grad.Shape()[0], 1, "MultiHeadAttention grad_input batch");
    ExpectEq(input_grad.Shape()[1], 2, "MultiHeadAttention grad_input seq_len");
    ExpectEq(input_grad.Shape()[2], 4, "MultiHeadAttention grad_input embed_dim");

    const auto grads = built.model->GetGradients();
    ExpectTrue(grads.count("layer0.W_q") == 1,
               "MultiHeadAttention W_q gradient is exposed");
    ExpectTrue(grads.count("layer0.W_k") == 1,
               "MultiHeadAttention W_k gradient is exposed");
    ExpectTrue(grads.count("layer0.W_v") == 1,
               "MultiHeadAttention W_v gradient is exposed");
    ExpectTrue(grads.count("layer0.W_o") == 1,
               "MultiHeadAttention W_o gradient is exposed");
    spdlog::info("  OK: MultiHeadAttention builds, runs, and exposes gradients");
}

void TestLayerNormCheckpointRoundTrip() {
    spdlog::info("--- TestLayerNormCheckpointRoundTrip ---");
    auto source = BuildSequentialFromConfig(MakeLayerNormConfig());
    auto target = BuildSequentialFromConfig(MakeLayerNormConfig());
    ExpectTrue(source.ok() && source.model != nullptr,
               "source LayerNorm model should build");
    ExpectTrue(target.ok() && target.model != nullptr,
               "target LayerNorm model should build");

    const float gamma_values[] = {0.5f, 1.5f, -0.25f, 2.0f};
    const float beta_values[] = {0.1f, -0.2f, 0.3f, -0.4f};
    std::map<std::string, Tensor> source_params = source.model->GetParameters();
    source_params["layer0.gamma"] = Tensor({4}, gamma_values, DataType::Float32);
    source_params["layer0.beta"] = Tensor({4}, beta_values, DataType::Float32);
    source.model->SetParameters(source_params);

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        "cyxwiz_layernorm_checkpoint_test";
    std::filesystem::remove_all(root);
    CheckpointManager manager(root.string());

    TrainingMetrics metrics;
    metrics.current_epoch = 3;
    metrics.optimizer_step_count = 7;
    metrics.train_loss = 0.25f;
    metrics.val_loss = 0.2f;
    metrics.has_validation_metrics = true;

    const std::string saved =
        manager.SaveCheckpoint(*source.model, nullptr, metrics, "layernorm");
    ExpectTrue(!saved.empty(), "LayerNorm checkpoint should save");

    const auto loaded =
        manager.LoadCheckpoint(*target.model, nullptr, "layernorm");
    ExpectTrue(loaded.has_value(), "LayerNorm checkpoint should load");
    ExpectEq(static_cast<size_t>(loaded->epoch), 3,
             "LayerNorm checkpoint epoch");

    const auto target_params = target.model->GetParameters();
    ExpectTrue(target_params.count("layer0.gamma") == 1,
               "loaded LayerNorm gamma should exist");
    ExpectTrue(target_params.count("layer0.beta") == 1,
               "loaded LayerNorm beta should exist");
    const float* loaded_gamma = target_params.at("layer0.gamma").Data<float>();
    const float* loaded_beta = target_params.at("layer0.beta").Data<float>();
    for (size_t i = 0; i < 4; ++i) {
        ExpectNear(loaded_gamma[i], gamma_values[i], 1e-6f,
                   "LayerNorm checkpoint gamma round-trip");
        ExpectNear(loaded_beta[i], beta_values[i], 1e-6f,
                   "LayerNorm checkpoint beta round-trip");
    }

    std::filesystem::remove_all(root);
    spdlog::info("  OK: LayerNorm checkpoint parameters round-trip");
}

class CheckpointParameterModule final : public Module {
public:
    explicit CheckpointParameterModule(Tensor parameter)
        : parameter_(std::move(parameter)) {}

    Tensor Forward(const Tensor& input) override { return input; }
    Tensor Backward(const Tensor& grad_output) override { return grad_output; }
    std::map<std::string, Tensor> GetParameters() override {
        return {{"value", parameter_}};
    }
    void SetParameters(const std::map<std::string, Tensor>& params) override {
        const auto found = params.find("value");
        if (found != params.end()) {
            parameter_ = found->second;
        }
    }
    bool HasParameters() const override { return true; }
    std::string GetName() const override { return "CheckpointParameter"; }

private:
    Tensor parameter_;
};

template <typename T>
void CheckCheckpointParameterRoundTrip(
    DataType dtype,
    const std::vector<T>& source_values,
    const std::string& label) {
    const std::vector<T> target_values(source_values.size(), T{});
    SequentialModel source;
    source.AddModule(std::make_unique<CheckpointParameterModule>(
        Tensor({source_values.size()}, source_values.data(), dtype)));
    SequentialModel target;
    target.AddModule(std::make_unique<CheckpointParameterModule>(
        Tensor({target_values.size()}, target_values.data(), dtype)));

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        ("cyxwiz_checkpoint_" + label + "_compatibility_test");
    std::filesystem::remove_all(root);
    CheckpointManager manager(root.string());

    TrainingMetrics metrics;
    metrics.current_epoch = 4;
    const std::string saved =
        manager.SaveCheckpoint(source, nullptr, metrics, label);
    ExpectTrue(!saved.empty(),
               (label + " checkpoint should save").c_str());
    const auto tensor_path =
        root / label / "model" / "layer0_value.bin";
    const uintmax_t expected_file_size =
        sizeof(uint32_t) + sizeof(uint64_t) + sizeof(uint32_t) +
        source_values.size() * sizeof(T);
    ExpectTrue(std::filesystem::file_size(tensor_path) == expected_file_size,
               (label + " checkpoint must write the exact payload size").c_str());

    const auto loaded = manager.LoadCheckpoint(target, nullptr, label);
    ExpectTrue(loaded.has_value(),
               (label + " checkpoint should load").c_str());
    const auto params = target.GetParameters();
    const auto found = params.find("layer0.value");
    ExpectTrue(found != params.end(),
               (label + " parameter should exist after load").c_str());
    ExpectTrue(found->second.GetDataType() == dtype,
               (label + " checkpoint must preserve dtype").c_str());
    ExpectTrue(found->second.Shape() ==
                   std::vector<size_t>{source_values.size()},
               (label + " checkpoint must preserve shape").c_str());
    ExpectTrue(std::memcmp(found->second.ReadData(), source_values.data(),
                           source_values.size() * sizeof(T)) == 0,
               (label + " checkpoint must preserve exact bytes").c_str());

    std::filesystem::remove_all(root);
}

void TestCheckpointTensorSerializationCompatibility() {
    spdlog::info("--- TestCheckpointTensorSerializationCompatibility ---");
    CheckCheckpointParameterRoundTrip<float>(
        DataType::Float32, {1.25f, -2.5f, 3.75f}, "float32");
    CheckCheckpointParameterRoundTrip<double>(
        DataType::Float64, {1.25, -2.5, 3.75}, "float64");
    CheckCheckpointParameterRoundTrip<int32_t>(
        DataType::Int32, {1, -2, 300000}, "int32");
    CheckCheckpointParameterRoundTrip<int64_t>(
        DataType::Int64, {1, -2, INT64_C(5000000000)}, "int64");
    CheckCheckpointParameterRoundTrip<uint8_t>(
        DataType::UInt8, {0, 127, 255}, "uint8");

    const std::vector<double> source_values = {1.25, -2.5, 3.75};
    const std::vector<double> target_values = {9.0, 8.0, 7.0};
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        "cyxwiz_checkpoint_tensor_corruption_test";
    std::filesystem::remove_all(root);
    CheckpointManager manager(root.string());
    SequentialModel source;
    source.AddModule(std::make_unique<CheckpointParameterModule>(
        Tensor({source_values.size()}, source_values.data(), DataType::Float64)));
    TrainingMetrics metrics;

    const auto make_target = [&]() {
        SequentialModel target;
        target.AddModule(std::make_unique<CheckpointParameterModule>(
            Tensor({target_values.size()}, target_values.data(),
                   DataType::Float64)));
        return target;
    };
    const auto assert_target_unchanged = [&](SequentialModel& target,
                                              const char* message) {
        const auto params = target.GetParameters();
        const auto found = params.find("layer0.value");
        ExpectTrue(found != params.end(), message);
        ExpectTrue(std::memcmp(found->second.ReadData(), target_values.data(),
                               target_values.size() * sizeof(double)) == 0,
                   message);
    };

    ExpectTrue(!manager.SaveCheckpoint(
                    source, nullptr, metrics, "unsupported-dtype").empty(),
               "unsupported-dtype fixture should save");
    const auto unsupported_path =
        root / "unsupported-dtype" / "model" / "layer0_value.bin";
    {
        std::fstream file(unsupported_path,
                          std::ios::binary | std::ios::in | std::ios::out);
        ExpectTrue(file.is_open(), "dtype fixture should open for corruption");
        file.seekp(sizeof(uint32_t) + sizeof(uint64_t));
        const uint32_t unsupported_dtype = UINT32_MAX;
        file.write(reinterpret_cast<const char*>(&unsupported_dtype),
                   sizeof(unsupported_dtype));
    }
    auto unsupported_target = make_target();
    ExpectTrue(!manager.LoadCheckpoint(
                    unsupported_target, nullptr, "unsupported-dtype").has_value(),
               "unsupported checkpoint dtype must fail closed");
    ExpectTrue(manager.GetLastError().find("unsupported data type") !=
                   std::string::npos,
               "unsupported checkpoint dtype should be diagnosed");
    assert_target_unchanged(
        unsupported_target,
        "unsupported dtype must not mutate the active model");

    ExpectTrue(!manager.SaveCheckpoint(
                    source, nullptr, metrics, "truncated").empty(),
               "truncated-payload fixture should save");
    const auto truncated_path =
        root / "truncated" / "model" / "layer0_value.bin";
    std::filesystem::resize_file(
        truncated_path, std::filesystem::file_size(truncated_path) - 1);
    auto truncated_target = make_target();
    ExpectTrue(!manager.LoadCheckpoint(
                    truncated_target, nullptr, "truncated").has_value(),
               "truncated checkpoint payload must fail closed");
    ExpectTrue(manager.GetLastError().find("truncated") != std::string::npos,
               "truncated checkpoint payload should be diagnosed");
    assert_target_unchanged(
        truncated_target,
        "truncated payload must not mutate the active model");

    std::filesystem::remove_all(root);
    spdlog::info(
        "  OK: checkpoint tensor dtype matrix and corruption guards");
}

void TestBatchNormCheckpointRoundTrip() {
    spdlog::info("--- TestBatchNormCheckpointRoundTrip ---");
    auto source = BuildSequentialFromConfig(MakeBatchNormConfig());
    auto target = BuildSequentialFromConfig(MakeBatchNormConfig());
    ExpectTrue(source.ok() && target.ok(),
               "BatchNorm checkpoint fixture models should build");

    const float gamma[] = {0.5f, 1.5f};
    const float beta[] = {0.1f, -0.2f};
    const float running_mean[] = {2.0f, -3.0f};
    const float running_var[] = {4.0f, 9.0f};
    auto state = source.model->GetParameters();
    state["layer0.gamma"] = Tensor({2}, gamma, DataType::Float32);
    state["layer0.beta"] = Tensor({2}, beta, DataType::Float32);
    state["layer0.running_mean"] =
        Tensor({2}, running_mean, DataType::Float32);
    state["layer0.running_var"] = Tensor({2}, running_var, DataType::Float32);
    source.model->SetParameters(state);

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        "cyxwiz_batchnorm_checkpoint_test";
    std::filesystem::remove_all(root);
    CheckpointManager manager(root.string());
    TrainingMetrics metrics;
    metrics.current_epoch = 4;
    const std::string saved =
        manager.SaveCheckpoint(*source.model, nullptr, metrics, "batchnorm");
    ExpectTrue(!saved.empty(), "BatchNorm checkpoint should save");
    const auto loaded =
        manager.LoadCheckpoint(*target.model, nullptr, "batchnorm");
    ExpectTrue(loaded.has_value(), "BatchNorm checkpoint should load");

    const auto restored = target.model->GetParameters();
    for (const auto& name : {"layer0.gamma", "layer0.beta",
                             "layer0.running_mean", "layer0.running_var"}) {
        const std::string presence_message =
            std::string("restored BatchNorm state should include ") + name;
        ExpectTrue(restored.count(name) == 1,
                   presence_message.c_str());
        const float* expected = state.at(name).ReadData<float>();
        const float* actual = restored.at(name).ReadData<float>();
        for (size_t i = 0; i < 2; ++i) {
            const std::string value_message =
                std::string("BatchNorm checkpoint round-trip for ") + name;
            ExpectNear(actual[i], expected[i], 1e-6f,
                       value_message.c_str());
        }
    }

    auto legacy_target = BuildSequentialFromConfig(MakeBatchNormConfig());
    ExpectTrue(legacy_target.ok(),
               "legacy BatchNorm checkpoint target should build");
    const auto manifest_path =
        root / "batchnorm" / "model" / "manifest.json";
    {
        std::ifstream input_stream(manifest_path);
        nlohmann::json manifest;
        input_stream >> manifest;
        for (auto it = manifest.begin(); it != manifest.end();) {
            const std::string parameter_name = it.value().get<std::string>();
            if (parameter_name.find(".running_mean") != std::string::npos ||
                parameter_name.find(".running_var") != std::string::npos) {
                it = manifest.erase(it);
            } else {
                ++it;
            }
        }
        std::ofstream output_stream(manifest_path, std::ios::trunc);
        output_stream << manifest.dump(2);
    }
    const auto legacy_loaded =
        manager.LoadCheckpoint(*legacy_target.model, nullptr, "batchnorm");
    ExpectTrue(legacy_loaded.has_value(),
               "legacy BatchNorm checkpoint without running statistics should load");
    const auto legacy_state = legacy_target.model->GetParameters();
    const float* legacy_mean =
        legacy_state.at("layer0.running_mean").ReadData<float>();
    const float* legacy_var =
        legacy_state.at("layer0.running_var").ReadData<float>();
    for (size_t i = 0; i < 2; ++i) {
        ExpectNear(legacy_mean[i], 0.0f, 1e-6f,
                   "legacy BatchNorm checkpoint should initialize running mean");
        ExpectNear(legacy_var[i], 1.0f, 1e-6f,
                   "legacy BatchNorm checkpoint should initialize running variance");
    }

    std::filesystem::remove_all(root);
    spdlog::info(
        "  OK: BatchNorm state round-trips and legacy checkpoints migrate");
}

void TestCheckpointRejectsIncompatibleModelWithoutMutation() {
    spdlog::info("--- TestCheckpointRejectsIncompatibleModelWithoutMutation ---");
    auto source = BuildSequentialFromConfig(MakeLayerNormConfig());
    auto incompatible_config = MakeLayerNormConfig();
    incompatible_config.input_size = 5;
    incompatible_config.layers[0].parameters["normalized_shape"] = "5";
    auto target = BuildSequentialFromConfig(incompatible_config);
    ExpectTrue(source.ok() && target.ok(),
               "checkpoint compatibility fixture models should build");

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        "cyxwiz_checkpoint_shape_guard_test";
    std::filesystem::remove_all(root);
    CheckpointManager manager(root.string());

    TrainingMetrics metrics;
    metrics.current_epoch = 2;
    const std::string saved =
        manager.SaveCheckpoint(*source.model, nullptr, metrics, "source");
    ExpectTrue(!saved.empty(), "shape-guard checkpoint should save");

    const auto before = target.model->GetParameters();
    const auto loaded =
        manager.LoadCheckpoint(*target.model, nullptr, "source");
    ExpectTrue(!loaded.has_value(),
               "incompatible checkpoint shape must be rejected");
    ExpectTrue(manager.GetLastError().find("shape mismatch") !=
                   std::string::npos,
               "shape rejection should explain the compatibility failure");

    const auto after = target.model->GetParameters();
    ExpectEq(after.size(), before.size(),
             "failed checkpoint load must preserve parameter count");
    for (const auto& [name, tensor] : before) {
        ExpectTrue(after.count(name) == 1,
                   "failed checkpoint load must preserve parameter names");
        ExpectTrue(after.at(name).Shape() == tensor.Shape(),
                   "failed checkpoint load must preserve parameter shapes");
    }

    std::filesystem::remove_all(root);
    spdlog::info("  OK: incompatible checkpoint rejected transactionally");
}

void TestCheckpointFormatCapabilitiesFailClosed() {
    spdlog::info("--- TestCheckpointFormatCapabilitiesFailClosed ---");
    auto source = BuildSequentialFromConfig(MakeLayerNormConfig());
    auto target = BuildSequentialFromConfig(MakeLayerNormConfig());
    ExpectTrue(source.ok() && target.ok(),
               "checkpoint format fixture models should build");

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        "cyxwiz_checkpoint_format_capabilities_test";
    std::filesystem::remove_all(root);
    CheckpointManager manager(root.string());

    TrainingMetrics metrics;
    metrics.current_epoch = 2;
    const std::string saved =
        manager.SaveCheckpoint(*source.model, nullptr, metrics, "format");
    ExpectTrue(!saved.empty(), "format checkpoint should save");

    const auto v1 = manager.InspectCheckpoint("format");
    ExpectTrue(v1.valid, "v1 checkpoint metadata should be valid");
    ExpectTrue(v1.format_version == "1.0",
               "v1 checkpoint format should be explicit");
    ExpectTrue(v1.can_load_for_testing,
               "v1 should support loading for testing");
    ExpectTrue(v1.can_warm_start, "v1 should support warm start");
    ExpectTrue(!v1.can_exact_resume,
               "v1 must not claim exact-resume capability");
    ExpectTrue(v1.exact_resume_reason.find("optimizer tensors") !=
                   std::string::npos,
               "v1 inspection should explain missing exact-resume state");

    const auto before = target.model->GetParameters();
    const auto metadata_path = root / "format" / "metadata.json";
    {
        std::ifstream input(metadata_path);
        nlohmann::json metadata;
        input >> metadata;
        metadata["version"] = "2.0";
        std::ofstream output(metadata_path, std::ios::trunc);
        output << metadata.dump(2);
    }

    const auto unsupported = manager.InspectCheckpoint("format");
    ExpectTrue(!unsupported.valid,
               "unimplemented future checkpoint format must be invalid");
    ExpectTrue(!unsupported.can_load_for_testing &&
                   !unsupported.can_warm_start &&
                   !unsupported.can_exact_resume,
               "unsupported checkpoint format must expose no capabilities");

    const auto loaded =
        manager.LoadCheckpoint(*target.model, nullptr, "format");
    ExpectTrue(!loaded.has_value(),
               "future checkpoint format must fail closed before loading");
    ExpectTrue(manager.GetLastError().find("Unsupported checkpoint format") !=
                   std::string::npos,
               "future format rejection should be corrective");

    const auto after = target.model->GetParameters();
    ExpectEq(after.size(), before.size(),
             "future format rejection must preserve parameter count");
    for (const auto& [name, tensor] : before) {
        ExpectTrue(after.count(name) == 1,
                   "future format rejection must preserve parameter names");
        ExpectTrue(after.at(name).Shape() == tensor.Shape(),
                   "future format rejection must preserve parameter shapes");
    }

    std::filesystem::remove_all(root);
    spdlog::info("  OK: v1 capabilities are truthful and future formats fail closed");
}

CheckpointManifestV2 MakeCompleteCheckpointManifestV2() {
    CheckpointManifestV2 manifest;
    manifest.checkpoint_id = "checkpoint-1";
    manifest.run_id = "run-1";
    manifest.created_at = "2026-07-29T17:00:00Z";
    manifest.engine_version = "test-engine";
    manifest.backend_version = "test-backend";
    manifest.graph_fingerprint = "graph-fingerprint";
    manifest.dataset_fingerprint = "dataset-fingerprint";
    manifest.partition_fingerprint = "partition-fingerprint";
    manifest.model_type = "Sequential";
    manifest.optimizer_type = "Adam";
    manifest.loss_type = "MSE";
    manifest.precision = "float32";
    manifest.completed_epoch = 2;
    manifest.next_batch = 3;
    manifest.optimizer_step = 11;
    manifest.rng_state_present = true;
    manifest.sampler_state_present = true;

    const std::string hash(64, 'a');
    const auto add_payload = [&](CheckpointPayloadKind kind,
                                 const std::string& path) {
        manifest.payloads.push_back({kind, path, 10, hash, true});
    };
    add_payload(CheckpointPayloadKind::ModelParameters, "model/state.bin");
    add_payload(CheckpointPayloadKind::OptimizerState, "optimizer/state.bin");
    add_payload(CheckpointPayloadKind::RuntimeState, "runtime/state.json");
    add_payload(CheckpointPayloadKind::GraphSnapshot, "graph.cyxgraph");
    add_payload(CheckpointPayloadKind::DatasetManifest, "dataset.json");
    return manifest;
}

void TestCheckpointManifestV2AtomicContract() {
    spdlog::info("--- TestCheckpointManifestV2AtomicContract ---");
    auto manifest = MakeCompleteCheckpointManifestV2();
    const auto validation = ValidateCheckpointManifestV2(manifest);
    ExpectTrue(validation.valid, "complete v2 manifest should be valid");
    ExpectTrue(validation.declares_exact_resume_state,
               "complete v2 inventory should declare exact-resume state");

    auto incomplete = manifest;
    incomplete.payloads.erase(
        std::remove_if(incomplete.payloads.begin(), incomplete.payloads.end(),
                       [](const CheckpointPayloadDescriptor& payload) {
                           return payload.kind ==
                                  CheckpointPayloadKind::OptimizerState;
                       }),
        incomplete.payloads.end());
    const auto incomplete_validation = ValidateCheckpointManifestV2(incomplete);
    ExpectTrue(incomplete_validation.valid,
               "incomplete resume inventory may remain structurally valid");
    ExpectTrue(!incomplete_validation.declares_exact_resume_state,
               "missing optimizer payload must disable exact resume");

    auto optional_optimizer = manifest;
    for (auto& payload : optional_optimizer.payloads) {
        if (payload.kind == CheckpointPayloadKind::OptimizerState) {
            payload.required = false;
        }
    }
    ExpectTrue(
        !ValidateCheckpointManifestV2(optional_optimizer)
             .declares_exact_resume_state,
        "optional optimizer payload must not satisfy exact-resume inventory");

    auto unsafe = manifest;
    unsafe.payloads.front().relative_path = "../escape.bin";
    ExpectTrue(!ValidateCheckpointManifestV2(unsafe).valid,
               "v2 manifest must reject escaping payload paths");

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        "cyxwiz_checkpoint_manifest_v2_test";
    std::filesystem::remove_all(root);
    std::string error;
    ExpectTrue(SaveCheckpointManifestV2Atomic(root, manifest, error),
               "v2 manifest should publish atomically");
    ExpectTrue(error.empty(), "successful v2 manifest save should clear error");
    ExpectTrue(!SaveCheckpointManifestV2Atomic(root, manifest, error),
               "published v2 manifest must be immutable");

    error.clear();
    const auto loaded = LoadCheckpointManifestV2(root, error);
    ExpectTrue(loaded.has_value(), "published v2 manifest should load");
    ExpectTrue(loaded->checkpoint_id == manifest.checkpoint_id &&
                   loaded->payloads.size() == manifest.payloads.size(),
               "v2 manifest identity and inventory should round-trip");

    const auto manifest_path = root / "manifest.json";
    {
        std::ifstream input(manifest_path);
        nlohmann::json value;
        input >> value;
        value["schema_version"] = 3;
        std::ofstream output(manifest_path, std::ios::trunc);
        output << value.dump(2);
    }
    error.clear();
    ExpectTrue(!LoadCheckpointManifestV2(root, error).has_value(),
               "future v2 manifest schema must fail closed");
    ExpectTrue(error.find("schema_version must be 2") != std::string::npos,
               "future schema rejection should identify the version contract");

    std::filesystem::remove_all(root);
    spdlog::info("  OK: v2 manifest validates, publishes atomically, and fails closed");
}

void TestCheckpointPayloadV2RoundTripAndCorruptionGuard() {
    spdlog::info("--- TestCheckpointPayloadV2RoundTripAndCorruptionGuard ---");
    auto source = BuildSequentialFromConfig(MakeLayerNormConfig());
    auto target = BuildSequentialFromConfig(MakeLayerNormConfig());
    ExpectTrue(source.ok() && target.ok(),
               "v2 model payload fixture models should build");

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        "cyxwiz_checkpoint_payload_v2_test";
    std::filesystem::remove_all(root);
    std::string error;

    CheckpointPayloadDescriptor model_descriptor;
    if (!SaveModelPayloadV2(root, "model/state.bin", *source.model,
                            model_descriptor, error)) {
        throw std::runtime_error("v2 model payload save failed: " + error);
    }
    ExpectTrue(model_descriptor.sha256.size() == 64 &&
                   VerifyCheckpointPayloadFile(root, model_descriptor, error),
               "v2 model payload should have a verified SHA-256");
    auto uppercase_hash_descriptor = model_descriptor;
    std::transform(uppercase_hash_descriptor.sha256.begin(),
                   uppercase_hash_descriptor.sha256.end(),
                   uppercase_hash_descriptor.sha256.begin(),
                   [](unsigned char value) {
                       return static_cast<char>(std::toupper(value));
                   });
    ExpectTrue(VerifyCheckpointPayloadFile(
                   root, uppercase_hash_descriptor, error),
               "v2 payload verification should accept hexadecimal hash case");
    if (!LoadModelPayloadV2(root, model_descriptor, *target.model, error)) {
        throw std::runtime_error("v2 model payload load failed: " + error);
    }

    const auto source_parameters = source.model->GetParameters();
    const auto loaded_parameters = target.model->GetParameters();
    ExpectEq(loaded_parameters.size(), source_parameters.size(),
             "v2 model payload parameter count");
    for (const auto& [name, expected] : source_parameters) {
        const auto& actual = loaded_parameters.at(name);
        ExpectTrue(actual.Shape() == expected.Shape(),
                   "v2 model payload parameter shape");
        const float* expected_data = expected.Data<float>();
        const float* actual_data = actual.Data<float>();
        for (size_t index = 0; index < expected.NumElements(); ++index) {
            ExpectNear(actual_data[index], expected_data[index], 1e-7f,
                       "v2 model payload parameter value");
        }
    }

    float parameter_data[] = {1.0f, -2.0f};
    float gradient_data[] = {0.5f, -1.0f};
    std::map<std::string, Tensor> original_parameters;
    std::map<std::string, Tensor> gradients;
    original_parameters.emplace(
        "w", Tensor({2}, parameter_data, DataType::Float32));
    gradients.emplace("w", Tensor({2}, gradient_data, DataType::Float32));
    AdamOptimizer original_optimizer(0.001, 0.9, 0.999, 1e-8);
    original_optimizer.Step(original_parameters, gradients);

    CheckpointPayloadDescriptor optimizer_descriptor;
    if (!SaveOptimizerPayloadV2(root, "optimizer/state.bin",
                                original_optimizer, optimizer_descriptor,
                                error)) {
        throw std::runtime_error("v2 Adam payload save failed: " + error);
    }
    AdamOptimizer resumed_optimizer(0.001, 0.9, 0.999, 1e-8);
    if (!LoadOptimizerPayloadV2(root, optimizer_descriptor,
                                resumed_optimizer, error)) {
        throw std::runtime_error("v2 Adam payload load failed: " + error);
    }
    auto resumed_parameters = original_parameters;
    original_optimizer.Step(original_parameters, gradients);
    resumed_optimizer.Step(resumed_parameters, gradients);
    const float* expected_next = original_parameters.at("w").Data<float>();
    const float* actual_next = resumed_parameters.at("w").Data<float>();
    ExpectNear(actual_next[0], expected_next[0], 1e-7f,
               "v2 Adam payload exact next step value 0");
    ExpectNear(actual_next[1], expected_next[1], 1e-7f,
               "v2 Adam payload exact next step value 1");

    AdamOptimizer unstepped_optimizer(0.001, 0.9, 0.999, 1e-8);
    CheckpointPayloadDescriptor unstepped_descriptor;
    if (!SaveOptimizerPayloadV2(root, "optimizer/step-zero.bin",
                                unstepped_optimizer, unstepped_descriptor,
                                error)) {
        throw std::runtime_error("v2 step-zero Adam payload save failed: " +
                                 error);
    }
    AdamOptimizer restored_unstepped_optimizer(0.001, 0.9, 0.999, 1e-8);
    if (!LoadOptimizerPayloadV2(root, unstepped_descriptor,
                                restored_unstepped_optimizer, error)) {
        throw std::runtime_error("v2 step-zero Adam payload load failed: " +
                                 error);
    }
    OptimizerState restored_unstepped_state;
    ExpectTrue(restored_unstepped_optimizer.ExportState(
                   restored_unstepped_state, error) &&
                   restored_unstepped_state.step_count == 0 &&
                   restored_unstepped_state.tensors.empty(),
               "v2 Adam payload should preserve valid step-zero state");

    SGDOptimizer scheduler_source_optimizer(0.1);
    ReduceLROnPlateau scheduler_source(
        &scheduler_source_optimizer, "max", 0.5, 1, 0.05, 0.01);
    scheduler_source.Step(1, 0.50f);
    scheduler_source.Step(2, 0.54f);
    scheduler_source.Step(3, 0.56f);
    scheduler_source.Step(4, 0.55f);
    scheduler_source.Step(5, 0.54f);
    CheckpointPayloadDescriptor scheduler_descriptor;
    if (!SaveSchedulerPayloadV2(root, "scheduler/state.bin",
                                scheduler_source, scheduler_descriptor,
                                error)) {
        throw std::runtime_error("v2 scheduler payload save failed: " + error);
    }
    ExpectTrue(scheduler_descriptor.kind ==
                   CheckpointPayloadKind::SchedulerState &&
                   VerifyCheckpointPayloadFile(
                       root, scheduler_descriptor, error),
               "v2 scheduler payload should have a verified SHA-256");

    SGDOptimizer scheduler_resumed_optimizer(0.1);
    ReduceLROnPlateau scheduler_resumed(
        &scheduler_resumed_optimizer, "max", 0.5, 1, 0.05, 0.01);
    if (!LoadSchedulerPayloadV2(root, scheduler_descriptor,
                                scheduler_resumed, error)) {
        throw std::runtime_error("v2 scheduler payload load failed: " + error);
    }
    ExpectNear(
        static_cast<float>(scheduler_resumed.GetLR()),
        static_cast<float>(scheduler_source.GetLR()), 0.0f,
        "v2 scheduler payload current learning rate");
    ExpectEq(scheduler_resumed.GetBadEpochs(), scheduler_source.GetBadEpochs(),
             "v2 scheduler payload plateau counter");
    scheduler_source.Step(6, 0.53f);
    scheduler_resumed.Step(6, 0.53f);
    scheduler_source.Step(7, 0.52f);
    scheduler_resumed.Step(7, 0.52f);
    ExpectNear(
        static_cast<float>(scheduler_resumed.GetLR()),
        static_cast<float>(scheduler_source.GetLR()), 0.0f,
        "v2 scheduler payload exact continuation");

    SGDOptimizer incompatible_scheduler_optimizer(0.1);
    ReduceLROnPlateau incompatible_scheduler(
        &incompatible_scheduler_optimizer, "max", 0.25, 1, 0.05, 0.01);
    ExpectTrue(!LoadSchedulerPayloadV2(
                   root, scheduler_descriptor, incompatible_scheduler, error),
               "incompatible v2 scheduler payload must be rejected");
    ExpectTrue(error.find("configuration does not match") != std::string::npos,
               "incompatible scheduler payload should identify config drift");
    ExpectNear(
        static_cast<float>(incompatible_scheduler.GetLR()), 0.1f, 0.0f,
        "incompatible scheduler payload must not mutate active state");

    float warmup_parameter_value = 1.0f;
    float warmup_gradient_value = 0.25f;
    std::map<std::string, Tensor> warmup_source_parameters{
        {"w", Tensor({1}, &warmup_parameter_value, DataType::Float32)}};
    const std::map<std::string, Tensor> warmup_gradients{
        {"w", Tensor({1}, &warmup_gradient_value, DataType::Float32)}};
    auto warmup_source_optimizer =
        std::make_unique<SGDOptimizer>(0.1, 0.9);
    LRWarmup warmup_source(
        std::move(warmup_source_optimizer), 4, WarmupType::Linear, 0.1);
    warmup_source.Step(warmup_source_parameters, warmup_gradients);
    warmup_source.ZeroGrad();
    warmup_source.Step(warmup_source_parameters, warmup_gradients);
    warmup_source.ZeroGrad();

    CheckpointPayloadDescriptor warmup_descriptor;
    if (!SaveLRWarmupPayloadV2(root, "scheduler/warmup.bin", warmup_source,
                               warmup_descriptor, error)) {
        throw std::runtime_error("v2 LRWarmup payload save failed: " + error);
    }
    ExpectTrue(
        warmup_descriptor.kind == CheckpointPayloadKind::SchedulerState &&
            VerifyCheckpointPayloadFile(root, warmup_descriptor, error),
        "v2 LRWarmup payload should have a verified SHA-256");

    auto warmup_resumed_optimizer =
        std::make_unique<SGDOptimizer>(0.1, 0.9);
    LRWarmup warmup_resumed(
        std::move(warmup_resumed_optimizer), 4, WarmupType::Linear, 0.1);
    if (!LoadLRWarmupPayloadV2(
            root, warmup_descriptor, warmup_resumed, error)) {
        throw std::runtime_error("v2 LRWarmup payload load failed: " + error);
    }
    auto warmup_resumed_parameters = warmup_source_parameters;
    warmup_source.Step(warmup_source_parameters, warmup_gradients);
    warmup_resumed.Step(warmup_resumed_parameters, warmup_gradients);
    ExpectNear(
        warmup_resumed_parameters.at("w").ReadData<float>()[0],
        warmup_source_parameters.at("w").ReadData<float>()[0], 1.0e-7f,
        "v2 LRWarmup payload exact nested-SGD continuation");
    ExpectNear(
        static_cast<float>(warmup_resumed.GetCurrentLR()),
        static_cast<float>(warmup_source.GetCurrentLR()), 0.0f,
        "v2 LRWarmup payload exact learning-rate continuation");

    auto incompatible_warmup_optimizer =
        std::make_unique<SGDOptimizer>(0.1, 0.9);
    LRWarmup incompatible_warmup(
        std::move(incompatible_warmup_optimizer), 5,
        WarmupType::Linear, 0.1);
    ExpectTrue(
        !LoadLRWarmupPayloadV2(
            root, warmup_descriptor, incompatible_warmup, error),
        "incompatible v2 LRWarmup payload must be rejected");
    ExpectTrue(error.find("configuration does not match") != std::string::npos,
               "incompatible LRWarmup payload should identify config drift");
    ExpectNear(
        static_cast<float>(incompatible_warmup.GetCurrentLR()), 0.0f, 0.0f,
        "incompatible LRWarmup payload must not mutate active state");

    const auto before_corruption = target.model->GetParameters();
    const auto model_path = root / model_descriptor.relative_path;
    {
        std::fstream file(model_path,
                          std::ios::binary | std::ios::in | std::ios::out);
        file.seekg(-1, std::ios::end);
        char byte = 0;
        file.read(&byte, 1);
        byte ^= 0x5a;
        file.seekp(-1, std::ios::end);
        file.write(&byte, 1);
    }
    ExpectTrue(!LoadModelPayloadV2(
                   root, model_descriptor, *target.model, error),
               "corrupted v2 model payload must be rejected");
    ExpectTrue(error.find("SHA-256 mismatch") != std::string::npos,
               "corrupted v2 payload should identify hash mismatch");
    const auto after_corruption = target.model->GetParameters();
    for (const auto& [name, before] : before_corruption) {
        const float* before_data = before.Data<float>();
        const float* after_data = after_corruption.at(name).Data<float>();
        for (size_t index = 0; index < before.NumElements(); ++index) {
            ExpectNear(after_data[index], before_data[index], 0.0f,
                       "corrupted payload must not mutate active model");
        }
    }

    std::filesystem::remove_all(root);
    spdlog::info(
        "  OK: v2 model/Adam/scheduler/LRWarmup payloads round-trip and "
        "corruption is transactional");
}

void TestTransformerDecoderCheckpointRoundTrip() {
    spdlog::info("--- TestTransformerDecoderCheckpointRoundTrip ---");

    SequentialModel source;
    source.Add<TransformerDecoderModule>(4, 2, 8, 0.0f, false);
    source.Add<LinearModule>(16, 2, true);

    SequentialModel target;
    target.Add<TransformerDecoderModule>(4, 2, 8, 0.0f, false);
    target.Add<LinearModule>(16, 2, true);

    auto source_params = source.GetParameters();
    ExpectTrue(!source_params.empty(),
               "TransformerDecoder checkpoint source params should exist");

    const float wq_values[] = {
        0.10f, 0.20f, -0.10f, 0.00f,
        0.00f, 0.15f, 0.25f, -0.05f,
        -0.20f, 0.05f, 0.30f, 0.10f,
        0.05f, -0.10f, 0.20f, 0.25f,
    };
    const float norm_values[] = {1.0f, 1.1f, 0.9f, 1.2f};
    const float head_values[] = {
        0.20f, -0.10f, 0.05f, 0.30f,
        -0.15f, 0.25f, 0.10f, -0.05f,
        0.05f, 0.15f, -0.20f, 0.10f,
        0.30f, -0.05f, 0.25f, -0.10f,
        -0.10f, 0.20f, 0.15f, 0.05f,
        0.12f, -0.08f, 0.06f, 0.14f,
        -0.04f, 0.11f, -0.13f, 0.07f,
    };

    source_params["layer0.self_attn.W_q"] =
        Tensor({4, 4}, wq_values, DataType::Float32);
    source_params["layer0.norm1.gamma"] =
        Tensor({4}, norm_values, DataType::Float32);
    source_params["layer1.weight"] =
        Tensor({2, 16}, head_values, DataType::Float32);
    source.SetParameters(source_params);
    source_params = source.GetParameters();

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        "cyxwiz_transformer_decoder_checkpoint_test";
    std::filesystem::remove_all(root);
    CheckpointManager manager(root.string());

    TrainingMetrics metrics;
    metrics.current_epoch = 4;
    metrics.current_batch = 11;
    metrics.train_loss = 0.33f;
    metrics.val_loss = 0.29f;
    metrics.has_validation_metrics = true;

    const std::string saved =
        manager.SaveCheckpoint(source, nullptr, metrics, "decoder");
    ExpectTrue(!saved.empty(), "TransformerDecoder checkpoint should save");

    const auto loaded =
        manager.LoadCheckpoint(target, nullptr, "decoder");
    ExpectTrue(loaded.has_value(),
               "TransformerDecoder checkpoint should load");
    ExpectEq(static_cast<size_t>(loaded->epoch), 4,
             "TransformerDecoder checkpoint epoch");

    const auto target_params = target.GetParameters();
    ExpectEq(target_params.size(), source_params.size(),
             "TransformerDecoder checkpoint parameter count");
    for (const auto& [name, expected] : source_params) {
        ExpectTrue(target_params.count(name) == 1,
                   ("loaded TransformerDecoder parameter missing: " + name).c_str());
        const Tensor& actual = target_params.at(name);
        ExpectTrue(actual.Shape() == expected.Shape(),
                   ("TransformerDecoder checkpoint shape mismatch: " + name).c_str());
        const float* actual_data = actual.Data<float>();
        const float* expected_data = expected.Data<float>();
        for (size_t i = 0; i < expected.NumElements(); ++i) {
            ExpectNear(actual_data[i], expected_data[i], 1e-6f,
                       "TransformerDecoder checkpoint parameter round-trip");
        }
    }

    std::filesystem::remove_all(root);
    spdlog::info("  OK: TransformerDecoder checkpoint parameters round-trip");
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

// TOFIX152: the layer also reads word vectors as text (a Word Embeddings
// node's .vec, GloVe .txt), a row per line in file order, word dropped.
void TestEmbeddingWeightsWordVectorFiles() {
    spdlog::info("--- TestEmbeddingWeightsWordVectorFiles ---");
    const auto dir = std::filesystem::temp_directory_path() / "cyxwiz_test_embedding_word_vectors";
    std::filesystem::create_directories(dir);
    const auto shape = [](const std::filesystem::path& path, size_t& rows, size_t& cols) {
        std::string error;
        const bool ok = cyxwiz::InspectEmbeddingWeightsFile(path.string(), rows, cols, error);
        if (!ok) std::cerr << "  " << error << "\n";
        return ok;
    };
    size_t rows = 0, cols = 0;
    // word2vec text with its header; "1990" is a word, not a value.
    std::ofstream(dir / "w2v.vec") << "3 2\n[PAD] 0 0\nthe 1 2\n1990 3 4\n";
    ExpectTrue(shape(dir / "w2v.vec", rows, cols) && rows == 3 && cols == 2, ".vec: header skipped, words dropped");
    // Punctuation tokens of a real vocabulary: "," and "#" are words, not
    // separators or comments (the sentiment vocabulary has both).
    std::ofstream(dir / "punct.vec") << "4 2\n[PAD] 0 0\n, 1 2\n# 3 4\nthe 5 6\n";
    ExpectTrue(shape(dir / "punct.vec", rows, cols) && rows == 4 && cols == 2, ".vec: ',' and '#' are words");
    std::ofstream(dir / "commas.txt") << "# a matrix\n1,2,3\n4, 5, 6\n";
    ExpectTrue(shape(dir / "commas.txt", rows, cols) && rows == 2 && cols == 3, "a comma matrix with a comment");
    std::ofstream(dir / "glove.txt") << "the 1 2\nparis 3 5\n";
    ExpectTrue(shape(dir / "glove.txt", rows, cols) && rows == 2 && cols == 2, "GloVe: a word first on each line");
    // Two whole numbers first, then numbers: a matrix row, not a header.
    std::ofstream(dir / "matrix.txt") << "1 0\n0 1\n";
    ExpectTrue(shape(dir / "matrix.txt", rows, cols) && rows == 2 && cols == 2, "a 2-column matrix keeps its first row");
    std::ofstream(dir / "single.txt") << "4 5\n";
    ExpectTrue(shape(dir / "single.txt", rows, cols) && rows == 1 && cols == 2, "a one-row matrix");
    std::ofstream(dir / "ragged.vec") << "2 2\nthe 1 2\nparis 3\n";
    ExpectTrue(!shape(dir / "ragged.vec", rows, cols), "a short row is refused");

    // A 500 x 8 .vec builds the layer.
    const auto vec = dir / "vectors.vec";
    {
        std::ofstream out(vec);
        out << "500 8\n";
        for (int r = 0; r < 500; ++r) {
            out << "w" << r;
            for (int c = 0; c < 8; ++c) out << ' ' << ((r == 0) ? 0.0f : static_cast<float>((r + c) % 7) / 10.0f);
            out << '\n';
        }
    }
    ExpectTrue(BuildSequentialFromConfig(MakeTextConfigWithEmbeddingWeights(vec)).ok(),
               "an Embedding layer loads a .vec of its shape");
    auto wrong = MakeTextConfigWithEmbeddingWeights(vec);
    wrong.layers[0].parameters["embedding_dim"] = "9";
    ExpectTrue(!BuildSequentialFromConfig(wrong).ok(), "a .vec of another width is refused");
    std::filesystem::remove_all(dir);
}

void TestSequenceProjectionConfigurationFailClosed() {
    spdlog::info("--- TestSequenceProjectionConfigurationFailClosed ---");

    auto embedding = MakeTextConfig();
    embedding.layers[0].parameters["padding_idx"] = "500";
    auto built = BuildSequentialFromConfig(embedding);
    ExpectTrue(!built.ok() &&
                   built.error_message.find("padding_idx") != std::string::npos,
               "Embedding padding outside the vocabulary should fail model construction");

    embedding = MakeTextConfig();
    embedding.layers[0].parameters["max_norm"] = "-0.5";
    built = BuildSequentialFromConfig(embedding);
    ExpectTrue(!built.ok() &&
                   built.error_message.find("max_norm") != std::string::npos,
               "Embedding negative max_norm should fail model construction");

    embedding = MakeTextConfig();
    embedding.layers[0].parameters["weights_file"] = "canonical.bin";
    embedding.layers[0].parameters["embedding_weights_file"] = "legacy.bin";
    built = BuildSequentialFromConfig(embedding);
    ExpectTrue(!built.ok() &&
                   built.error_message.find("conflict") != std::string::npos,
               "Embedding conflicting weights aliases should fail before file loading");

    TrainingConfiguration head_config;
    head_config.input_size = 3;
    head_config.output_size = 2;
    head_config.loss_type = gui::NodeType::MSELoss;
    head_config.optimizer_type = gui::NodeType::Adam;
    CompiledLayer head;
    head.type = gui::NodeType::TimeDistributed;
    head.parameters["units"] = "4";
    head.parameters["out_features"] = "5";
    head_config.layers.push_back(head);
    built = BuildSequentialFromConfig(head_config);
    ExpectTrue(!built.ok() &&
                   built.error_message.find("conflict") != std::string::npos,
               "TimeDistributed conflicting width aliases should fail model construction");

    spdlog::info("  OK: sequence projection configuration rejects invalid and conflicting values");
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
    ce_cfg.loss_params["label_smoothing"] = "0.1";
    ce_cfg.loss_params["reduction"] = "sum";
    auto ce_built = BuildSequentialFromConfig(ce_cfg);
    ExpectTrue(ce_built.ok(), "weighted CrossEntropy config should build");
    auto* ce_loss = dynamic_cast<CrossEntropyLoss*>(ce_built.loss.get());
    ExpectTrue(ce_loss != nullptr,
               "weighted CrossEntropy config should construct CrossEntropyLoss");
    ExpectTrue(ce_loss->GetReduction() == Reduction::Sum,
               "CrossEntropyLoss should preserve reduction");
    ExpectTrue(ce_loss->GetClassWeights().size() == 4,
               "CrossEntropyLoss should receive four class weights");
    ExpectNear(ce_loss->GetClassWeights()[3], 4.0f, 1e-6f,
               "CrossEntropyLoss should preserve manual class weights");
    ExpectNear(ce_loss->GetLabelSmoothing(), 0.1f, 1e-6f,
               "CrossEntropyLoss should preserve label_smoothing");

    auto bce_cfg = MakeTabularConfig();
    bce_cfg.output_size = 1;
    bce_cfg.layers.back().units = 1;
    bce_cfg.loss_type = gui::NodeType::BCEWithLogits;
    bce_cfg.loss_params["pos_weight"] = "2.5";
    bce_cfg.loss_params["label_smoothing"] = "0.1";
    bce_cfg.loss_params["reduction"] = "sum";
    auto bce_built = BuildSequentialFromConfig(bce_cfg);
    ExpectTrue(bce_built.ok(), "weighted BCEWithLogits config should build");
    auto* bce_loss = dynamic_cast<BCEWithLogitsLoss*>(bce_built.loss.get());
    ExpectTrue(bce_loss != nullptr,
               "weighted BCEWithLogits config should construct BCEWithLogitsLoss");
    ExpectTrue(bce_loss->GetReduction() == Reduction::Sum,
               "BCEWithLogitsLoss should preserve reduction");
    ExpectNear(bce_loss->GetPosWeight(), 2.5f, 1e-6f,
               "BCEWithLogitsLoss should preserve pos_weight");
    ExpectNear(bce_loss->GetLabelSmoothing(), 0.1f, 1e-6f,
               "BCEWithLogitsLoss should preserve label_smoothing");

    auto focal_cfg = MakeTabularConfig();
    focal_cfg.loss_type = gui::NodeType::FocalLoss;
    focal_cfg.loss_params["alpha"] = "0.75";
    focal_cfg.loss_params["gamma"] = "1.5";
    focal_cfg.loss_params["reduction"] = "none";
    auto focal_built = BuildSequentialFromConfig(focal_cfg);
    ExpectTrue(focal_built.ok(), "FocalLoss config should build");
    auto* focal_loss = dynamic_cast<FocalLoss*>(focal_built.loss.get());
    ExpectTrue(focal_loss != nullptr,
               "FocalLoss config should construct FocalLoss");
    ExpectTrue(focal_loss->GetReduction() == Reduction::None,
               "FocalLoss should preserve reduction");
    ExpectNear(focal_loss->GetAlpha(), 0.75f, 1e-6f,
               "FocalLoss should preserve alpha");
    ExpectNear(focal_loss->GetGamma(), 1.5f, 1e-6f,
               "FocalLoss should preserve gamma");

    auto smooth_cfg = MakeTabularConfig();
    smooth_cfg.loss_type = gui::NodeType::SmoothL1Loss;
    smooth_cfg.loss_params["beta"] = "0.5";
    smooth_cfg.loss_params["reduction"] = "sum";
    auto smooth_built = BuildSequentialFromConfig(smooth_cfg);
    ExpectTrue(smooth_built.ok(), "SmoothL1 config should build");
    auto* smooth_loss = dynamic_cast<SmoothL1Loss*>(smooth_built.loss.get());
    ExpectTrue(smooth_loss != nullptr,
               "SmoothL1 config should construct SmoothL1Loss");
    ExpectTrue(smooth_loss->GetReduction() == Reduction::Sum,
               "SmoothL1Loss should preserve reduction");
    ExpectNear(smooth_loss->GetDelta(), 0.5f, 1e-6f,
               "SmoothL1Loss should preserve beta/delta");

    auto huber_cfg = MakeTabularConfig();
    huber_cfg.loss_type = gui::NodeType::HuberLoss;
    huber_cfg.loss_params["beta"] = "0.5";
    huber_cfg.loss_params["reduction"] = "sum";
    auto huber_built = BuildSequentialFromConfig(huber_cfg);
    ExpectTrue(huber_built.ok(), "Huber config should build");
    auto* huber_loss = dynamic_cast<HuberLoss*>(huber_built.loss.get());
    ExpectTrue(huber_loss != nullptr,
               "Huber config must construct HuberLoss, not SmoothL1Loss");
    ExpectTrue(huber_loss->GetReduction() == Reduction::Sum,
               "HuberLoss should preserve reduction");
    ExpectNear(huber_loss->GetDelta(), 0.5f, 1e-6f,
               "HuberLoss should preserve delta");

    auto dice_cfg = MakeTabularConfig();
    dice_cfg.loss_type = gui::NodeType::SoftDiceLoss;
    dice_cfg.loss_params["smooth"] = "0.5";
    dice_cfg.loss_params["reduction"] = "sum";
    auto dice_built = BuildSequentialFromConfig(dice_cfg);
    ExpectTrue(dice_built.ok(), "SoftDice config should build");
    auto* dice_loss = dynamic_cast<SoftDiceLoss*>(dice_built.loss.get());
    ExpectTrue(dice_loss != nullptr,
               "SoftDice config should construct SoftDiceLoss");
    ExpectTrue(dice_loss->GetReduction() == Reduction::Sum,
               "SoftDiceLoss should preserve reduction");
    ExpectNear(dice_loss->GetSmooth(), 0.5f, 1e-6f,
               "SoftDiceLoss should preserve smooth");

    auto tversky_cfg = MakeTabularConfig();
    tversky_cfg.loss_type = gui::NodeType::TverskyLoss;
    tversky_cfg.loss_params["alpha"] = "0.3";
    tversky_cfg.loss_params["beta"] = "0.7";
    tversky_cfg.loss_params["smooth"] = "0.5";
    tversky_cfg.loss_params["reduction"] = "sum";
    auto tversky_built = BuildSequentialFromConfig(tversky_cfg);
    ExpectTrue(tversky_built.ok(), "Tversky config should build");
    auto* tversky_loss = dynamic_cast<TverskyLoss*>(tversky_built.loss.get());
    ExpectTrue(tversky_loss != nullptr,
               "Tversky config should construct TverskyLoss");
    ExpectTrue(tversky_loss->GetReduction() == Reduction::Sum,
               "TverskyLoss should preserve reduction");
    ExpectNear(tversky_loss->GetAlpha(), 0.3f, 1e-6f,
               "TverskyLoss should preserve alpha");
    ExpectNear(tversky_loss->GetBeta(), 0.7f, 1e-6f,
               "TverskyLoss should preserve beta");
    ExpectNear(tversky_loss->GetSmooth(), 0.5f, 1e-6f,
               "TverskyLoss should preserve smooth");

    auto jaccard_cfg = MakeTabularConfig();
    jaccard_cfg.loss_type = gui::NodeType::JaccardLoss;
    jaccard_cfg.loss_params["smooth"] = "0.5";
    jaccard_cfg.loss_params["reduction"] = "sum";
    auto jaccard_built = BuildSequentialFromConfig(jaccard_cfg);
    ExpectTrue(jaccard_built.ok(), "Jaccard config should build");
    auto* jaccard_loss = dynamic_cast<JaccardLoss*>(jaccard_built.loss.get());
    ExpectTrue(jaccard_loss != nullptr,
               "Jaccard config should construct JaccardLoss");
    ExpectTrue(jaccard_loss->GetReduction() == Reduction::Sum,
               "JaccardLoss should preserve reduction");
    ExpectNear(jaccard_loss->GetSmooth(), 0.5f, 1e-6f,
               "JaccardLoss should preserve smooth");
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

    const float singleton_logit = 0.0f;
    const float singleton_target = 1.0f;
    Tensor singleton_predictions(
        {1, 1}, &singleton_logit, DataType::Float32);
    Tensor singleton_labels(
        {1, 1}, &singleton_target, DataType::Float32);
    Tensor singleton_grad = loss.Backward(
        singleton_predictions, singleton_labels);
    ExpectTrue(singleton_grad.Shape() == std::vector<size_t>({1, 1}),
               "BCEWithLogits grad should preserve singleton 2D rank");
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

void TestDebugNumericsContract() {
    spdlog::info("--- TestDebugNumericsContract ---");
    const float values[] = {
        -2.0f,
        0.0f,
        2.0f,
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
    };
    Tensor tensor({5}, values, DataType::Float32);
    const auto summary = ScanDebugTensorNumerics(
        tensor, gui::NodeType::Dense);
    ExpectTrue(summary.available, "Float32 numeric summary availability");
    ExpectEq(summary.element_count, 5, "numeric element count");
    ExpectEq(summary.finite_count, 3, "numeric finite count");
    ExpectEq(summary.nan_count, 1, "numeric NaN count");
    ExpectEq(summary.inf_count, 1, "numeric Inf count");
    ExpectEq(summary.zero_count, 1, "numeric zero count");
    ExpectNear(static_cast<float>(summary.minimum), -2.0f, 1e-6f,
               "numeric minimum");
    ExpectNear(static_cast<float>(summary.maximum), 2.0f, 1e-6f,
               "numeric maximum");
    ExpectNear(static_cast<float>(summary.mean), 0.0f, 1e-6f,
               "numeric mean");
    ExpectNear(static_cast<float>(summary.l2_norm), std::sqrt(8.0f), 1e-6f,
               "numeric L2 norm");
    auto trace = DebugNodeTraceContract::Make(
        "numeric-run", -1, "Numerics", "Diagnostic", "Numerics",
        DebugTraceRole::Activation, {}, {}, "float32", "LocalDebug", "ok");
    AttachDebugNumericsPayload(trace, summary, "activation");
    ExpectTrue(trace.payload["numerics_schema"].get<std::string>() ==
                   kDebugNumericsSchema,
               "numeric trace schema");
    ExpectTrue(trace.payload["numeric_host_read_performed"].get<bool>(),
               "numeric trace should disclose its bounded host read");
    ExpectTrue(!trace.payload["numeric_values_included"].get<bool>(),
               "numeric trace should not include raw tensor values");

    std::vector<float> dead_values(100, 0.0f);
    dead_values.back() = 1.0f;
    Tensor dead_relu({10, 10}, dead_values.data(), DataType::Float32);
    const auto relu = ScanDebugTensorNumerics(
        dead_relu, gui::NodeType::ReLU);
    ExpectTrue(relu.saturation_summary_available,
               "ReLU saturation summary availability");
    ExpectTrue(relu.dead_relu_candidate,
               "99 percent zero ReLU should be a dead-output candidate");
    ExpectNear(static_cast<float>(relu.saturation_candidate_ratio),
               0.99f, 1e-6f, "ReLU zero-output ratio");

    const float softmax_values[] = {
        0.9995f, 0.0003f, 0.0002f,
        0.9997f, 0.0002f, 0.0001f,
    };
    Tensor softmax({2, 3}, softmax_values, DataType::Float32);
    const auto softmax_summary = ScanDebugTensorNumerics(
        softmax, gui::NodeType::Softmax);
    ExpectTrue(softmax_summary.softmax_saturation_candidate,
               "confident Softmax rows should be a saturation candidate");
    ExpectEq(softmax_summary.saturation_candidate_count, 2,
             "Softmax saturated row count");
    ExpectEq(softmax_summary.saturation_observation_count, 2,
             "Softmax observed row count");

    const auto token_targets = CheckDebugPredictionTargetShapes(
        {2, 3, 4}, {2, 3}, gui::NodeType::CrossEntropyLoss);
    ExpectTrue(token_targets.available && token_targets.compatible,
               "token class-index targets should match logits prefix");
    const auto one_hot_targets = CheckDebugPredictionTargetShapes(
        {8, 4}, {8, 4}, gui::NodeType::CrossEntropyLoss);
    ExpectTrue(one_hot_targets.compatible,
               "one-hot targets should match logits exactly");
    const auto mismatch = CheckDebugPredictionTargetShapes(
        {8, 4}, {7}, gui::NodeType::CrossEntropyLoss);
    ExpectTrue(mismatch.available && !mismatch.compatible,
               "wrong target batch should report a shape mismatch");
    spdlog::info("  OK: bounded summaries and shape contracts");
}

// The device scan against a plain host pass over the same values: random
// data with NaN, Inf, zeros and large values, a ReLU, a Sigmoid, a Tanh and
// a Softmax-shaped tensor, so every count and every reduction is compared.
void TestDebugNumericsMatchHostReference() {
    spdlog::info("--- TestDebugNumericsMatchHostReference ---");
    std::mt19937 random(20261106);
    std::normal_distribution<float> normal(0.0f, 1.5f);
    std::vector<float> values(6000);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = normal(random);
        if (i % 97 == 0) values[i] = 0.0f;
        if (i % 211 == 0) values[i] = std::numeric_limits<float>::quiet_NaN();
        if (i % 307 == 0) values[i] = -std::numeric_limits<float>::infinity();
        if (i % 401 == 0) values[i] = 2.0e6f;
        if (i % 13 == 0) values[i] = std::abs(values[i]) < 1.0f ? 0.995f : values[i];
    }
    for (const auto type : {gui::NodeType::Dense, gui::NodeType::ReLU, gui::NodeType::Sigmoid,
                            gui::NodeType::Tanh, gui::NodeType::Softmax}) {
        // [rows, 6] so the Softmax case has rows of 6 classes.
        Tensor tensor({values.size() / 6, 6}, values.data(), DataType::Float32);
        const auto summary = ScanDebugTensorNumerics(tensor, type);
        size_t nan = 0, inf = 0, finite = 0, zero = 0, exploding = 0, saturated = 0;
        long double sum = 0.0L, squares = 0.0L;
        double low = std::numeric_limits<double>::infinity(), high = -low, largest = 0.0;
        for (const float value : values) {
            if (std::isnan(value)) { ++nan; continue; }
            if (std::isinf(value)) { ++inf; continue; }
            ++finite;
            sum += value;
            squares += static_cast<long double>(value) * value;
            low = std::min(low, static_cast<double>(value));
            high = std::max(high, static_cast<double>(value));
            largest = std::max(largest, std::abs(static_cast<double>(value)));
            if (value == 0.0f) ++zero;
            if (std::abs(value) >= kDebugExplodingAbsoluteThreshold) ++exploding;
            if ((type == gui::NodeType::Sigmoid && (value <= 0.01f || value >= 0.99f)) ||
                (type == gui::NodeType::Tanh && std::abs(value) >= 0.99f) ||
                (type == gui::NodeType::ReLU && value <= 0.0f)) {
                ++saturated;
            }
        }
        if (type == gui::NodeType::Softmax) {
            for (size_t row = 0; row < values.size() / 6; ++row) {
                double row_high = -std::numeric_limits<double>::infinity();
                for (size_t column = 0; column < 6; ++column) {
                    const float value = values[row * 6 + column];
                    if (std::isfinite(value)) row_high = std::max(row_high, static_cast<double>(value));
                }
                if (row_high >= kDebugSoftmaxProbabilityThreshold) ++saturated;
            }
        }
        const std::string kind = std::to_string(static_cast<int>(type));
        ExpectEq(summary.element_count, values.size(), "device numerics element count " + kind);
        ExpectEq(summary.nan_count, nan, "device numerics NaN count " + kind);
        ExpectEq(summary.inf_count, inf, "device numerics Inf count " + kind);
        ExpectEq(summary.finite_count, finite, "device numerics finite count " + kind);
        ExpectEq(summary.zero_count, zero, "device numerics zero count " + kind);
        ExpectEq(summary.exploding_count, exploding, "device numerics exploding count " + kind);
        ExpectEq(summary.saturation_candidate_count, saturated, "device numerics saturation count " + kind);
        const double mean = static_cast<double>(sum / static_cast<long double>(finite));
        const double l2 = std::sqrt(static_cast<double>(squares));
        ExpectTrue(std::abs(summary.mean - mean) <= 1e-5 * std::max(1.0, std::abs(mean)) + 1e-3,
                   "device numerics mean " + kind);
        ExpectTrue(std::abs(summary.l2_norm - l2) <= 1e-5 * l2, "device numerics L2 norm " + kind);
        ExpectTrue(summary.minimum == low && summary.maximum == high && summary.maximum_absolute == largest,
                   "device numerics range " + kind);
    }
    spdlog::info("  OK: device numerics equal the host reference");
}

// The layer view reads (TOFIX178 6.2c) against host arithmetic on tensors
// whose every value is known: one sample of [H, W, C, N], its channels'
// means, maxima and pictures, flat units, a histogram, and heatmap rows of
// conv [kh, kw, cin, cout] and Dense [out, in] weights.
void TestLayerTensorReader() {
    spdlog::info("--- TestLayerTensorReader ---");
    using namespace cyxwiz::layer_view;
    const size_t H = 3, W = 4, C = 5, N = 2;
    std::vector<float> spatial(H * W * C * N);
    for (size_t i = 0; i < spatial.size(); ++i) spatial[i] = static_cast<float>(i % 23) - 7.0f;
    const auto at = [&](size_t h, size_t w, size_t c, size_t n) { return spatial[((h * W + w) * C + c) * N + n]; };
    const Tensor activation({H, W, C, N}, spatial.data(), DataType::Float32);

    const Tensor sample = SampleOf(activation, 1);
    ExpectTrue(sample.Shape() == std::vector<size_t>({H, W, C}), "a spatial sample is [H, W, C]");
    const std::vector<float> sample_values = ReadValues(sample);
    bool same = sample_values.size() == H * W * C;
    for (size_t h = 0; same && h < H; ++h)
        for (size_t w = 0; w < W; ++w)
            for (size_t c = 0; c < C; ++c) same = same && sample_values[(h * W + w) * C + c] == at(h, w, c, 1);
    ExpectTrue(same, "sample 1 holds the batch's second item, row-major");

    const auto summary = ReadChannelSummary(sample);
    bool sums = summary.means.size() == C && summary.maxima.size() == C;
    for (size_t c = 0; sums && c < C; ++c) {
        double sum = 0.0;
        float high = -1e30f;
        for (size_t h = 0; h < H; ++h)
            for (size_t w = 0; w < W; ++w) {
                sum += at(h, w, c, 1);
                high = std::max(high, at(h, w, c, 1));
            }
        sums = sums && std::abs(summary.means[c] - static_cast<float>(sum / (H * W))) < 1e-5f &&
               summary.maxima[c] == high;
    }
    ExpectTrue(sums, "channel means and maxima");

    const auto pictures = ReadChannelImages(sample, {4, 1});
    bool pictured = pictures.size() == 2 * H * W;
    for (size_t h = 0; pictured && h < H; ++h)
        for (size_t w = 0; w < W; ++w)
            pictured = pictured && pictures[h * W + w] == at(h, w, 4, 1) &&
                       pictures[H * W + h * W + w] == at(h, w, 1, 1);
    ExpectTrue(pictured, "the shown channels' pictures, row by row, in the asked order");

    std::vector<float> rows(3 * 6);
    for (size_t i = 0; i < rows.size(); ++i) rows[i] = static_cast<float>(i) * 0.5f;
    const Tensor flat({3, 6}, rows.data(), DataType::Float32);
    const auto units = ReadValues(SampleOf(flat, 2));
    ExpectTrue(units.size() == 6 && units[0] == 6.0f && units[5] == 8.5f, "a flat sample's units");

    const float histogram_values[] = {0.0f, 0.1f, 0.9f, 1.0f, 2.5f, 3.9f, std::numeric_limits<float>::quiet_NaN(),
                                      4.0f};
    const Tensor for_histogram({8}, histogram_values, DataType::Float32);
    const auto histogram = ReadHistogram(for_histogram, 0.0f, 4.0f, 4);
    ExpectTrue(histogram.counts == std::vector<unsigned>({3, 1, 1, 2}),
               "histogram of the finite values over [low, high] (the top value in the last bin)");

    const size_t kh = 2, kw = 2, cin = 3, cout = 4;
    std::vector<float> conv(kh * kw * cin * cout);
    for (size_t i = 0; i < conv.size(); ++i) conv[i] = static_cast<float>(i);
    const Tensor conv_weights({kh, kw, cin, cout}, conv.data(), DataType::Float32);
    const TensorInfo conv_info{CapturedTensorRole::Parameter, "weights", {kh, kw, cin, cout}};
    const auto page = ReadMatrixRows(conv_info, conv_weights, 1, 2);
    const size_t columns = kh * kw * cin;
    bool conv_rows = page.size() == 2 * columns;
    for (size_t r = 0; conv_rows && r < 2; ++r)
        for (size_t col = 0; col < columns; ++col)
            conv_rows = conv_rows && page[r * columns + col] == conv[MatrixFlatIndex(conv_info, 1 + r, col)];
    ExpectTrue(conv_rows, "conv heatmap rows are output channels over (ky, kx, cin)");
    for (size_t position = 0; position < kh * kw; ++position) {
        const auto slice = ReadMatrixRows(conv_info, conv_weights, 0, cout, position);
        bool slice_ok = slice.size() == cout * cin;
        for (size_t r = 0; slice_ok && r < cout; ++r)
            for (size_t c = 0; c < cin; ++c)
                slice_ok = slice_ok && slice[r * cin + c] == conv[MatrixFlatIndex(conv_info, r, c, position)];
        ExpectTrue(slice_ok, "a kernel position's slice is output channels over input channels");
    }

    std::vector<float> dense(3 * 5);
    for (size_t i = 0; i < dense.size(); ++i) dense[i] = static_cast<float>(i) - 4.0f;
    const Tensor dense_weights({3, 5}, dense.data(), DataType::Float32);
    const TensorInfo dense_info{CapturedTensorRole::Parameter, "weight", {3, 5}};
    const auto dense_page = ReadMatrixRows(dense_info, dense_weights, 2, 9);
    ExpectTrue(dense_page.size() == 5 && dense_page[0] == dense[10] && dense_page[4] == dense[14],
               "Dense heatmap rows, clipped to the matrix");
    spdlog::info("  OK: layer view reads equal host arithmetic");
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
    assert(res.optimizer_step_ms >= 0.0f);

    // 2 Linear modules -> 2 weight + 2 bias = 4 params expected.
    ExpectEq(res.params_with_grad, 4, "params_with_grad");
    ExpectEq(res.params_missing_grad, 0, "params_missing_grad");

    // Per-layer traces: one per module (Dense, ReLU, Dense = 3 modules).
    ExpectEq(res.layer_traces.size(), 3, "layer_traces.size()");
    ExpectEq(res.model_build_traces.size(), 3,
             "model_build_traces.size()");
    for (size_t i = 0; i < res.model_build_traces.size(); ++i) {
        const auto& build_trace = res.model_build_traces[i];
        ExpectTrue(build_trace.status == "ok",
                   "created module build trace should be successful");
        ExpectTrue(build_trace.payload["module_created"].get<bool>(),
                   "build trace should identify a created module");
        ExpectEq(build_trace.payload["module_index"].get<size_t>(), i,
                 "build trace module index");
        ExpectEq(build_trace.payload["compiled_layer_index"].get<size_t>(), i,
                 "build trace compiled layer index");
        ExpectTrue(
            build_trace.payload["parameter_summary_available"].get<bool>(),
            "created module should expose bounded parameter metadata");
    }
    ExpectEq(
        res.model_build_traces[0].payload["parameter_tensor_count"].get<size_t>(),
        2,
        "input Dense parameter tensor count");
    ExpectEq(
        res.model_build_traces[0].payload["parameter_numel"].get<size_t>(),
        352,
        "input Dense parameter element count");

    for (size_t i = 0; i < res.layer_traces.size(); ++i) {
        const auto& t = res.layer_traces[i];
        ExpectEq(t.node_id, 101 + static_cast<int>(i),
                 "forward trace graph node id");
        ExpectEq(t.module_index, i, "forward trace module index");
        ExpectEq(t.compiled_layer_index, i,
                 "forward trace compiled layer index");
        ExpectTrue(t.input_shape_matches,
                   "forward input should match its exact compiled layer");
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
        ExpectTrue(t.shape_matches,
                   "forward trace should match its exact compiled layer");
        ExpectTrue(!t.has_shape_mismatch(),
                   "golden path should have no shape divergence");
        ExpectTrue(t.numerics.available,
                   "forward trace should expose bounded numerics");
        ExpectEq(t.numerics.finite_count, t.numerics.element_count,
                 "golden path forward values should all be finite");
        ExpectTrue(t.numerics.minimum <= t.numerics.mean &&
                       t.numerics.mean <= t.numerics.maximum,
                   "forward mean should remain within min/max");
    }
    ExpectTrue(res.layer_traces[1].numerics.saturation_summary_available,
               "ReLU layer should expose zero-output saturation evidence");
    ExpectTrue(res.loss_target_compatibility.available &&
                   res.loss_target_compatibility.compatible,
               "golden path prediction/target shape contract");
    spdlog::info("  OK: reached=Complete, params_with_grad={}, "
                 "layer_traces={}, loss={}",
                 res.params_with_grad, res.layer_traces.size(),
                 res.loss_value);
}

// TOFIX178 6.0b: Local Debug runs the sample it is given (the picked item)
// and names it; without one it runs and names a synthetic sample.
void TestDebugExecutorInputSample() {
    spdlog::info("--- TestDebugExecutorInputSample ---");
    auto cfg = MakeTabularConfig();
    DebugExecutor synthetic(cfg);
    const auto plain = synthetic.Run();
    ExpectTrue(plain.success && plain.input_description == "a synthetic sample (seed 1337)",
               "no sample given: a synthetic one, named: " + plain.input_description);

    const auto given = MakeSyntheticBatch(cfg, /*seed=*/7);
    DebugExecutor picked(cfg);
    picked.SetInputSample(given.features, given.labels, "row 3 of rows, validation");
    const auto res = picked.Run();
    ExpectTrue(res.success && res.input_description == "row 3 of rows, validation",
               "the given sample runs and is named: " + res.input_description);
    ExpectTrue(res.loss_value != plain.loss_value, "the given sample, not the synthetic one, reached the loss");
}

// TOFIX178 6.0a: Local Debug runs a branched graph the way training does (its
// graph plan), not as a straight chain. The 1.8 ResNet example (skips into
// Add, stride-2 blocks with 1x1 projections) used to stop at the first
// stride-2 block with "Conv2D forward input channel mismatch".
TrainingConfiguration CompileResNetExample() {
    cyxwiz::GraphCompilerDatasetHooks hooks;
    hooks.is_dataset_registered = [](const std::string&) { return false; };
    hooks.labels_from_structure = [](const std::string& category) -> std::optional<bool> {
        return category == "image";
    };
    hooks.preprocessing_domain = [](const std::string& category) -> std::optional<cyxwiz::PreprocessingDomain> {
        return category == "image" ? cyxwiz::PreprocessingDomain::Image : cyxwiz::PreprocessingDomain::Tabular;
    };
    cyxwiz::SetGraphCompilerDatasetHooks(std::move(hooks));

    const std::filesystem::path path = std::filesystem::path(CYXWIZ_SOURCE_ROOT) / "examples" / "cyxgraph" /
                                       "cats_dogs_resnet_scratch.cyxgraph";
    std::ifstream in(path, std::ios::binary);
    ExpectTrue(static_cast<bool>(in), "ResNet example opens: " + path.string());
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    cyxwiz::GraphDocument graph;
    std::string error;
    ExpectTrue(cyxwiz::ParseGraphDocument(text, graph, error), "ResNet example loads: " + error);
    auto config = cyxwiz::GraphCompiler{}.Compile(graph.nodes, graph.links, /*allow_unloaded_data=*/true);
    ExpectTrue(config.is_valid, "ResNet example compiles without its data");
    return config;
}

void TestDebugExecutorResNetExample() {
    spdlog::info("--- TestDebugExecutorResNetExample ---");
    const auto config = CompileResNetExample();
    DebugExecutor exe(config);
    const auto res = exe.Run();
    ExpectTrue(res.reached == DebugStage::Complete && res.success,
               "ResNet Local Debug completes: " + res.failure_summary);
    // The 12 BatchNorm layers' running mean and variance are state, not
    // parameters missing a gradient; every trained parameter has one.
    ExpectEq(res.state_tensors, 24, "BatchNorm running statistics counted as state");
    ExpectEq(res.params_missing_grad, 0, "ResNet params_missing_grad");
    for (const auto& g : res.grad_norms)
        ExpectTrue(g.param_name.find("running_") == std::string::npos,
                   "no running statistic among the gradients: " + g.param_name);
    // BN stem (32 channels): gamma and beta trained, running mean and
    // variance its state.
    const auto& bn_stem = res.model_build_traces.at(1).payload;
    ExpectEq(bn_stem["parameter_tensor_count"].get<size_t>(), 2, "BN stem trained tensors");
    ExpectEq(bn_stem["parameter_numel"].get<size_t>(), 64, "BN stem trained elements");
    ExpectEq(bn_stem["state_tensor_count"].get<size_t>(), 2, "BN stem state tensors");
    ExpectEq(bn_stem["state_numel"].get<size_t>(), 64, "BN stem state elements");
    ExpectEq(res.layer_traces.size(), config.layers.size(), "one forward trace per layer");
    ExpectEq(res.model_build_traces.size(), config.layers.size(), "one build trace per layer");
    for (const auto& t : res.layer_traces) {
        const auto shape_text = [](const std::vector<size_t>& shape) {
            std::string out;
            for (size_t d : shape) out += (out.empty() ? "" : ",") + std::to_string(d);
            return "[" + out + "]";
        };
        ExpectTrue(t.shape_matches && t.input_shape_matches && !t.has_nan && !t.has_inf,
                   "layer " + t.name + " runs with its compiled shapes: input " + shape_text(t.actual_input_shape) +
                       " (compiled " + shape_text(t.predicted_input_shape) + "), output " +
                       shape_text(t.actual_shape) + " (compiled " + shape_text(t.predicted_shape) + ")");
    }
    // The ReLU after a block is fed by the block's Add, not by the layer run
    // before it (the projection's BatchNorm).
    int relus_after_add = 0;
    for (const auto& t : res.layer_traces) {
        if (t.name.rfind("ReLU s", 0) == 0 && t.name.size() == 7) {
            ExpectTrue(t.upstream_node_name == "Add " + t.name.substr(5),
                       t.name + " is fed by its block's Add, got " + t.upstream_node_name);
            ++relus_after_add;
        }
    }
    ExpectEq(relus_after_add, 4, "four blocks end in Add -> ReLU");
    // Every weight gets a gradient. What comes back without one is BatchNorm's
    // running statistics, which GetParameters lists beside the weights (state,
    // not trained; TOFIX178 6.0e separates them).
    for (const auto& g : res.grad_norms) {
        const bool running_stat = g.param_name.size() > 13 &&
            (g.param_name.compare(g.param_name.size() - 13, 13, ".running_mean") == 0 ||
             g.param_name.compare(g.param_name.size() - 12, 12, ".running_var") == 0);
        ExpectTrue(g.has_gradient || running_stat, g.param_name + " gets a gradient");
    }
    // A picked image arrives as the image batcher hands it out, batch first
    // [1, H, W, C] (TOFIX178 6.0b), and enters the model the way training's
    // Forward converts it.
    std::vector<float> pixels(64 * 64 * 3);
    for (size_t i = 0; i < pixels.size(); ++i) pixels[i] = static_cast<float>(i % 251) / 251.0f;
    const auto synthetic = MakeSyntheticBatch(config, 3);
    DebugExecutor picked(config);
    picked.SetInputSample(Tensor({1, 64, 64, 3}, pixels.data(), DataType::Float32), synthetic.labels,
                          "image 0 of cat_dog, train");
    const auto picked_res = picked.Run();
    ExpectTrue(picked_res.reached == DebugStage::Complete && picked_res.success,
               "a batch-first picked image runs the ResNet: " + picked_res.failure_summary);
    spdlog::info("  OK: ResNet layer_traces={}, params_with_grad={}, loss={}",
                 res.layer_traces.size(), res.params_with_grad, res.loss_value);
}

// TOFIX178 6.1a: Local Debug keeps what every layer held.
std::vector<float> HostValues(const Tensor& t) {
    const float* data = t.ReadData<float>();
    return std::vector<float>(data, data + t.NumElements());
}

float MaxAbsDifference(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return std::numeric_limits<float>::infinity();
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(a[i] - b[i]));
    return worst;
}

const CapturedLayer& CapturedNamed(const LayerCapture& capture, const std::string& name) {
    for (const auto& layer : capture.Layers())
        if (layer.node_name == name) return layer;
    std::cerr << "FAIL: no captured layer named " << name << "\n";
    std::exit(1);
}

const Tensor& CapturedTensorOf(const CapturedLayer& layer, CapturedTensorRole role, const std::string& name = {}) {
    const CapturedTensor* t = layer.Find(role, name);
    ExpectTrue(t != nullptr, layer.node_name + " captured its " + CapturedTensorRoleName(role) + " " + name);
    return t->tensor;
}

void TestLayerCaptureTabular() {
    spdlog::info("--- TestLayerCaptureTabular ---");
    const auto cfg = MakeTabularConfig();

    DebugExecutor off(cfg);
    const auto plain = off.Run();
    ExpectTrue(plain.success && !plain.layer_capture, "capture off: nothing is kept");

    DebugExecutor exe(cfg);
    exe.SetLayerCapture({});
    const auto res = exe.Run();
    ExpectTrue(res.success && res.layer_capture, "capture on: the run keeps its layers");
    const LayerCapture& capture = *res.layer_capture;
    ExpectEq(capture.Layers().size(), 3, "one captured layer per module");
    ExpectTrue(!capture.Truncated(), "a small model fits the default budget");
    size_t bytes = 0;
    for (size_t i = 0; i < capture.Layers().size(); ++i) {
        const auto& layer = capture.Layers()[i];
        ExpectEq(layer.module_index, i, "captured layers in run order");
        ExpectEq(static_cast<size_t>(layer.node_id), static_cast<size_t>(101 + i), "captured layer node id");
        ExpectTrue(layer.node_name == res.layer_traces[i].name, "captured layer named as its trace");
        const Tensor& output = CapturedTensorOf(layer, CapturedTensorRole::Output);
        ExpectTrue(output.Shape() == res.layer_traces[i].actual_shape, layer.node_name + " output shape");
        for (const auto& t : layer.tensors) bytes += t.tensor.NumBytes();
        if (i + 1 < capture.Layers().size()) {
            const Tensor& next_input = CapturedTensorOf(capture.Layers()[i + 1], CapturedTensorRole::Input);
            ExpectTrue(MaxAbsDifference(HostValues(output), HostValues(next_input)) == 0.0f,
                       layer.node_name + " output is the next layer's input");
        }
    }
    ExpectEq(capture.BytesHeld(), bytes, "bytes held is what the layers keep");

    // The kept weights are the ones that ran: Dense recomputed on the host
    // from the captured input, weight and bias gives the captured output.
    const auto& dense = capture.Layers()[0];
    const auto x = HostValues(CapturedTensorOf(dense, CapturedTensorRole::Input));
    const Tensor& w_tensor = CapturedTensorOf(dense, CapturedTensorRole::Parameter, "weight");
    const auto w = HostValues(w_tensor);
    const auto b = HostValues(CapturedTensorOf(dense, CapturedTensorRole::Parameter, "bias"));
    const auto y = HostValues(CapturedTensorOf(dense, CapturedTensorRole::Output));
    const size_t in = 10, out = 32, rows = x.size() / in;
    const bool out_by_in = w_tensor.Shape() == std::vector<size_t>{out, in};
    ExpectTrue(out_by_in || w_tensor.Shape() == std::vector<size_t>{in, out}, "Dense weight is [out, in] or [in, out]");
    float worst = 0.0f;
    for (size_t r = 0; r < rows; ++r)
        for (size_t o = 0; o < out; ++o) {
            double sum = b[o];
            for (size_t k = 0; k < in; ++k) sum += x[r * in + k] * w[out_by_in ? o * in + k : k * out + o];
            worst = std::max(worst, std::fabs(static_cast<float>(sum) - y[r * out + o]));
        }
    ExpectTrue(worst < 1e-4f, "Dense output recomputed from the captured tensors, worst " + std::to_string(worst));

    // Parameters and gradients are the model's, before the optimizer step:
    // the same norms Local Debug reports.
    size_t gradients = 0;
    for (const auto& entry : res.grad_norms) {
        const auto& layer = capture.Layers().at(static_cast<size_t>(entry.layer_index));
        const auto field = entry.param_name.substr(entry.param_name.find('.') + 1);
        const auto p = HostValues(CapturedTensorOf(layer, CapturedTensorRole::Parameter, field));
        const auto g = HostValues(CapturedTensorOf(layer, CapturedTensorRole::Gradient, field));
        double p2 = 0, g2 = 0;
        for (float v : p) p2 += double(v) * v;
        for (float v : g) g2 += double(v) * v;
        ExpectNear(static_cast<float>(std::sqrt(p2)), entry.parameter_l2_norm, 1e-4f * (1 + entry.parameter_l2_norm),
                   "captured parameter is the model's");
        ExpectNear(static_cast<float>(std::sqrt(g2)), entry.l2_norm, 1e-4f * (1 + entry.l2_norm),
                   "captured gradient is the model's");
        ++gradients;
    }
    ExpectEq(gradients, 4, "two Dense layers: weight and bias gradients");

    // What to keep.
    DebugExecutor weights_only(cfg);
    weights_only.SetLayerCapture({/*activations=*/false, /*parameters=*/true, /*gradients=*/false});
    const auto w_res = weights_only.Run();
    ExpectEq(w_res.layer_capture->Layers().size(), 2, "weights only: the two Dense layers");
    for (const auto& layer : w_res.layer_capture->Layers())
        for (const auto& t : layer.tensors)
            ExpectTrue(t.role == CapturedTensorRole::Parameter, "weights only keeps parameters");

    // The budget: tensors past it are skipped and counted, none is cut.
    LayerCaptureOptions small;
    small.byte_budget = bytes / 2;
    DebugExecutor bounded(cfg);
    bounded.SetLayerCapture(small);
    const auto b_res = bounded.Run();
    const LayerCapture& kept = *b_res.layer_capture;
    ExpectTrue(b_res.success && kept.Truncated(), "a budget below the model's tensors truncates");
    ExpectTrue(kept.BytesHeld() <= small.byte_budget, "held within the budget");
    ExpectEq(kept.BytesHeld() + kept.SkippedBytes(), bytes, "held + skipped is everything");
    size_t held_tensors = 0, all_tensors = 0;
    for (const auto& layer : kept.Layers()) held_tensors += layer.tensors.size();
    for (const auto& layer : capture.Layers()) all_tensors += layer.tensors.size();
    ExpectEq(held_tensors + kept.SkippedTensors(), all_tensors, "held + skipped tensors is every tensor");
    spdlog::info("  OK: capture {} bytes, Dense recompute worst {}", bytes, worst);
}

// The graph model (branches into Add): each layer is captured with its
// compiled name, BatchNorm keeps its running statistics as state, and the
// ReLU after a block is fed the sum of the block's two branches.
void TestLayerCaptureResNetExample() {
    spdlog::info("--- TestLayerCaptureResNetExample ---");
    const auto config = CompileResNetExample();
    DebugExecutor exe(config);
    exe.SetLayerCapture({});
    const auto res = exe.Run();
    ExpectTrue(res.success && res.layer_capture, "ResNet runs with capture: " + res.failure_summary);
    const LayerCapture& capture = *res.layer_capture;
    ExpectTrue(!capture.Truncated(), "the ResNet's one-image step fits the default budget");
    ExpectEq(capture.Layers().size(), res.layer_traces.size(), "one captured layer per layer that ran");
    for (const auto& trace : res.layer_traces) {
        const auto& layer = CapturedNamed(capture, trace.name);
        ExpectTrue(CapturedTensorOf(layer, CapturedTensorRole::Output).Shape() == trace.actual_shape,
                   trace.name + " captured output shape");
        ExpectTrue(CapturedTensorOf(layer, CapturedTensorRole::Input).Shape() == trace.actual_input_shape,
                   trace.name + " captured input shape");
    }
    size_t state = 0;
    for (const auto& layer : capture.Layers())
        for (const auto& t : layer.tensors)
            if (t.role == CapturedTensorRole::State) {
                ++state;
                ExpectTrue(t.name.rfind("running_", 0) == 0, layer.node_name + " state is a running statistic");
                ExpectTrue(!layer.Find(CapturedTensorRole::Gradient, t.name), "no gradient for state " + t.name);
            }
    ExpectEq(state, res.state_tensors, "every running statistic kept as state");
    const auto& stem = CapturedNamed(capture, "Conv 32 stem");
    ExpectTrue(CapturedTensorOf(stem, CapturedTensorRole::Gradient, "weights").Shape() ==
                   CapturedTensorOf(stem, CapturedTensorRole::Parameter, "weights").Shape(),
               "the stem's weight gradient has the weight's shape");
    ExpectTrue(MaxAbsDifference(HostValues(CapturedTensorOf(stem, CapturedTensorRole::Output)),
                                HostValues(CapturedTensorOf(CapturedNamed(capture, "BN stem"),
                                                            CapturedTensorRole::Input))) == 0.0f,
               "the stem's output is BN stem's input");

    const auto add_input = [&](const std::string& relu, const std::string& main, const std::string& skip) {
        const auto in = HostValues(CapturedTensorOf(CapturedNamed(capture, relu), CapturedTensorRole::Input));
        const auto a = HostValues(CapturedTensorOf(CapturedNamed(capture, main), CapturedTensorRole::Output));
        const auto b = HostValues(CapturedTensorOf(CapturedNamed(capture, skip), CapturedTensorRole::Output));
        ExpectTrue(in.size() == a.size() && a.size() == b.size(), relu + " input and branch sizes agree");
        float worst = 0.0f;
        for (size_t i = 0; i < in.size(); ++i) worst = std::max(worst, std::fabs(in[i] - (a[i] + b[i])));
        ExpectTrue(worst < 1e-5f, relu + " input is " + main + " + " + skip + ", worst " + std::to_string(worst));
        const auto out = HostValues(CapturedTensorOf(CapturedNamed(capture, relu), CapturedTensorRole::Output));
        float relu_worst = 0.0f;
        for (size_t i = 0; i < in.size(); ++i) relu_worst = std::max(relu_worst, std::fabs(out[i] - std::max(0.0f, in[i])));
        ExpectTrue(relu_worst == 0.0f, relu + " output is max(0, input)");
    };
    add_input("ReLU s1", "BN s1b", "ReLU stem");  // identity skip
    add_input("ReLU s2", "BN s2b", "BN s2s");     // 1x1 projection skip
    spdlog::info("  OK: ResNet capture {} layers, {} MB", capture.Layers().size(), capture.BytesHeld() >> 20);
}

// TOFIX202: a debug run's model rebuilt from its capture answers again with
// the weights the run used; its input gradient matches central differences.
std::vector<float> RowMajor(const Tensor& t) {
    const af::array flat = t.GetRowMajorFlatArray();
    std::vector<float> values(static_cast<size_t>(flat.elements()));
    if (!values.empty()) flat.as(f32).host(values.data());
    return values;
}

Tensor FromRowMajor(const std::vector<float>& values, const std::vector<size_t>& shape) {
    return Tensor::FromRowMajorFlatArray(af::array(static_cast<dim_t>(values.size()), values.data()), shape);
}

// sum(output x target) for an input.
double Score(DebugModelSession& session, const Tensor& input, const std::vector<float>& target) {
    const auto out = RowMajor(session.Forward(input));
    double s = 0.0;
    for (size_t i = 0; i < out.size(); ++i) s += double(out[i]) * target[i];
    return s;
}

// The input gradient at `indices` against central differences.
void CheckInputGradient(DebugModelSession& session, const Tensor& input, const std::vector<size_t>& indices,
                        float eps, const std::string& what) {
    const Tensor output = session.Forward(input);
    std::vector<float> target(output.NumElements(), 0.0f);
    target[target.size() - 1] = 1.0f;  // the last output of the last sample
    const auto grad = RowMajor(session.InputGradient(input, FromRowMajor(target, output.Shape())));
    ExpectEq(grad.size(), input.NumElements(), what + ": the gradient has the input's size");
    const auto x = RowMajor(input);
    for (size_t k : indices) {
        auto up = x, down = x;
        up[k] += eps;
        down[k] -= eps;
        const double numeric = (Score(session, FromRowMajor(up, input.Shape()), target) -
                                Score(session, FromRowMajor(down, input.Shape()), target)) / (2.0 * eps);
        const double analytic = grad[k];
        const double tolerance = 0.05 * std::max(std::fabs(numeric), std::fabs(analytic)) + 2e-3;
        ExpectTrue(std::fabs(numeric - analytic) <= tolerance,
                   what + ": input " + std::to_string(k) + " gradient " + std::to_string(analytic) +
                       " vs central difference " + std::to_string(numeric));
    }
}

// TOFIX204: the layer backward observer reports the gradient that reached
// each layer's output, summed over its consumers, and changes nothing.
std::map<size_t, Tensor> BackwardThroughModel(IExecutableModel& model, const Tensor& input, Tensor* input_grad) {
    std::map<size_t, Tensor> grads;
    model.SetLayerBackwardObserver([&](size_t module_index, const Tensor& grad) {
        ExpectTrue(grads.find(module_index) == grads.end(), "each layer's output gradient is reported once");
        grads.emplace(module_index, grad.Clone());
    });
    const Tensor output = model.Forward(input);
    std::vector<float> target(output.NumElements(), 0.0f);
    target[target.size() - 1] = 1.0f;
    *input_grad = model.Backward(FromRowMajor(target, output.Shape()));
    model.SetLayerBackwardObserver(nullptr);
    return grads;
}

void TestLayerBackwardObserver() {
    spdlog::info("--- TestLayerBackwardObserver ---");
    // An MLP: each layer's backward of its output gradient is the gradient
    // at the layer before it, and the first gives the input gradient.
    auto mlp = BuildExecutableFromConfig(MakeTabularConfig());
    ExpectTrue(mlp.ok(), "the MLP builds: " + mlp.error_message);
    mlp.model->SetTraining(false);
    std::vector<float> x(3 * 10);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(0.7f * float(i)) * 2.0f;
    const Tensor input = FromRowMajor(x, {3, 10});
    Tensor input_grad;
    auto grads = BackwardThroughModel(*mlp.model, input, &input_grad);
    ExpectEq(grads.size(), 3, "one output gradient per MLP layer");
    const Tensor plain_grad = mlp.model->Backward(grads.at(2));
    ExpectTrue(MaxAbsDifference(RowMajor(plain_grad), RowMajor(input_grad)) == 0.0f,
               "without the observer the backward is unchanged");
    std::vector<float> last(12, 0.0f);
    last.back() = 1.0f;
    ExpectTrue(RowMajor(grads.at(2)) == last, "the last layer's output gradient is the target");
    for (size_t i = 2; i > 0; --i) {
        const float worst = MaxAbsDifference(RowMajor(mlp.model->LayerModule(i)->Backward(grads.at(i))),
                                             RowMajor(grads.at(i - 1)));
        ExpectTrue(worst < 1e-6f, "MLP layer " + std::to_string(i) + " backward gives layer " +
                                      std::to_string(i - 1) + "'s output gradient, worst " + std::to_string(worst));
    }
    ExpectTrue(MaxAbsDifference(RowMajor(mlp.model->LayerModule(0)->Backward(grads.at(0))), RowMajor(input_grad)) <
                   1e-6f,
               "the first MLP layer's backward gives the input gradient");

    // Embedding, Flatten, Dense: the gradient at each token position is what
    // the Embedding table's gradient sums over the positions of that id.
    auto text = BuildExecutableFromConfig(MakeTextConfig());
    ExpectTrue(text.ok(), "the text model builds: " + text.error_message);
    text.model->SetTraining(false);
    std::vector<float> ids(2 * 16);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = float((i * 7) % 11 + 1);  // ids repeat
    Tensor ids_grad;
    grads = BackwardThroughModel(*text.model, FromRowMajor(ids, {2, 16}), &ids_grad);
    const Tensor& at_embedding = grads.at(0);
    ExpectTrue(at_embedding.Shape() == std::vector<size_t>({2, 16, 8}),
               "the gradient at the Embedding output is [N, T, D]");
    const auto per_position = RowMajor(at_embedding);
    std::vector<double> summed(500 * 8, 0.0);
    for (size_t p = 0; p < ids.size(); ++p)
        for (size_t d = 0; d < 8; ++d) summed[size_t(ids[p]) * 8 + d] += per_position[p * 8 + d];
    const auto table_grads = text.model->GetGradients();
    const Tensor* table = nullptr;
    for (const auto& [key, g] : table_grads)
        if (g.Shape() == std::vector<size_t>({500, 8})) table = &g;
    ExpectTrue(table != nullptr, "the Embedding table has a [500, 8] gradient");
    const auto table_values = RowMajor(*table);
    double worst = 0.0;
    for (size_t i = 0; i < summed.size(); ++i) worst = std::max(worst, std::fabs(summed[i] - table_values[i]));
    ExpectTrue(worst < 1e-5, "per-position gradients sum to the Embedding table's gradient, worst " +
                                 std::to_string(worst));

    // The ResNet: "ReLU stem" feeds the first block and its identity skip,
    // so its output gradient is the sum of both; back through the stem it
    // gives the model's input gradient.
    const auto resnet = CompileResNetExample();
    auto r = BuildExecutableFromConfig(resnet);
    ExpectTrue(r.ok(), "the ResNet builds: " + r.error_message);
    r.model->SetTraining(false);
    std::map<std::string, size_t> module_of;
    for (const auto& p : r.module_provenance)
        if (p.module_index) module_of[p.node_name] = *p.module_index;
    ExpectTrue(module_of.count("Conv 32 stem") && module_of.count("BN stem") && module_of.count("ReLU stem"),
               "the ResNet's stem layers are modules");
    DebugExecutor r_exe(resnet);
    r_exe.SetLayerCapture({});
    const auto r_res = r_exe.Run();
    ExpectTrue(r_res.success && r_res.layer_capture, "the ResNet runs with capture: " + r_res.failure_summary);
    const Tensor& image = CapturedTensorOf(r_res.layer_capture->Layers().front(), CapturedTensorRole::Input);
    Tensor image_grad;
    grads = BackwardThroughModel(*r.model, image, &image_grad);
    Tensor back = grads.at(module_of.at("ReLU stem"));
    for (const char* name : {"ReLU stem", "BN stem", "Conv 32 stem"})
        back = r.model->LayerModule(module_of.at(name))->Backward(back);
    const float r_worst = MaxAbsDifference(RowMajor(back), RowMajor(image_grad));
    ExpectTrue(MaxAbsDifference(RowMajor(image_grad), std::vector<float>(image_grad.NumElements(), 0.0f)) > 0.0f,
               "the ResNet's input gradient is not all zero");
    ExpectTrue(r_worst < 1e-4f, "the stem's summed output gradient leads to the input gradient, worst " +
                                    std::to_string(r_worst));
    spdlog::info("  OK: layer backward observer (MLP, Embedding table worst {}, ResNet stem worst {})", worst, r_worst);
}

// TOFIX178 6.3c: saliency against PyTorch (computation_truth/fixtures/
// saliency_pytorch.json, generate_saliency_fixtures.py): the output weighting,
// the gradient (at the input, or at the Embedding's output for tokens) and
// the per-element size and gradient x input, for an image CNN, a one-logit
// table MLP (both classes) and an Embedding text model, in eval mode.
Tensor FixtureTensor(const nlohmann::json& value) {
    return FromRowMajor(value.at("values").get<std::vector<float>>(), value.at("shape").get<std::vector<size_t>>());
}

void ExpectCloseTo(const std::vector<float>& actual, const nlohmann::json& expected, float atol, float rtol,
                   const std::string& what) {
    const auto values = expected.at("values").get<std::vector<float>>();
    ExpectEq(actual.size(), values.size(), what + ": size");
    for (size_t i = 0; i < values.size(); ++i)
        ExpectTrue(std::fabs(actual[i] - values[i]) <= atol + rtol * std::fabs(values[i]),
                   what + ": element " + std::to_string(i) + " " + std::to_string(actual[i]) + ", PyTorch " +
                       std::to_string(values[i]));
}

std::vector<float> AsFloats(const std::vector<double>& v) {
    std::vector<float> out;
    out.reserve(v.size());
    for (double x : v) out.push_back(static_cast<float>(x));
    return out;
}

TrainingConfiguration SaliencyFixtureConfig(const std::string& kind) {
    TrainingConfiguration cfg;
    cfg.loss_type = gui::NodeType::CrossEntropyLoss;
    cfg.optimizer_type = gui::NodeType::Adam;
    const auto layer = [](gui::NodeType type, std::vector<size_t> in, std::vector<size_t> out) {
        CompiledLayer l;
        l.type = type;
        l.input_shape = std::move(in);
        l.output_shape = std::move(out);
        return l;
    };
    if (kind == "image") {
        cfg.preprocessing_domain = PreprocessingDomain::Image;
        cfg.input_size = 75;
        cfg.input_shape = {5, 5, 3};
        cfg.output_size = 3;
        CompiledLayer conv = layer(gui::NodeType::Conv2D, {5, 5, 3}, {5, 5, 4});
        conv.parameters = {{"filters", "4"}, {"kernel_size", "3"}, {"stride", "1"}, {"padding", "1"}};
        cfg.layers = {conv, layer(gui::NodeType::ReLU, {5, 5, 4}, {5, 5, 4}),
                      layer(gui::NodeType::Flatten, {5, 5, 4}, {100}), layer(gui::NodeType::Dense, {100}, {3})};
        cfg.layers.back().units = 3;
    } else if (kind == "features") {
        cfg.preprocessing_domain = PreprocessingDomain::Tabular;
        cfg.input_size = 4;
        cfg.output_size = 1;
        cfg.loss_type = gui::NodeType::BCEWithLogits;
        cfg.layers = {layer(gui::NodeType::Dense, {4}, {5}), layer(gui::NodeType::ReLU, {5}, {5}),
                      layer(gui::NodeType::Dense, {5}, {1})};
        cfg.layers[0].units = 5;
        cfg.layers[2].units = 1;
    } else {
        cfg.preprocessing_domain = PreprocessingDomain::Text;
        cfg.input_size = 5;
        cfg.output_size = 3;
        CompiledLayer emb;
        emb.type = gui::NodeType::Embedding;
        emb.parameters["num_embeddings"] = "20";
        emb.parameters["embedding_dim"] = "4";
        CompiledLayer flatten;
        flatten.type = gui::NodeType::Flatten;
        CompiledLayer dense;
        dense.type = gui::NodeType::Dense;
        dense.units = 3;
        cfg.layers = {emb, flatten, dense};
    }
    return cfg;
}

void TestSaliencyAgainstPyTorch() {
    spdlog::info("--- TestSaliencyAgainstPyTorch ---");
    namespace sal = saliency;
    const auto path = std::filesystem::path(CYXWIZ_SOURCE_ROOT) / "cyxwiz-engine" / "tests" / "computation_truth" /
                      "fixtures" / "saliency_pytorch.json";
    std::ifstream in(path);
    ExpectTrue(in.good(), "the saliency fixture opens: " + path.string());
    const auto fixture = nlohmann::json::parse(in);
    const float atol = fixture.at("tolerance").at("atol").get<float>();
    const float rtol = fixture.at("tolerance").at("rtol").get<float>();
    for (const auto& c : fixture.at("cases")) {
        const std::string name = c.at("name").get<std::string>();
        const std::string kind_key = c.at("kind").get<std::string>();
        const auto kind = kind_key == "image"      ? occlusion::InputKind::Image
                          : kind_key == "features" ? occlusion::InputKind::Features
                                                   : occlusion::InputKind::Tokens;
        auto built = BuildExecutableFromConfig(SaliencyFixtureConfig(kind_key));
        ExpectTrue(built.ok(), name + ": the model builds: " + built.error_message);
        std::map<std::string, Tensor> parameters;
        for (const auto& [key, value] : c.at("layers").items()) parameters[key] = FixtureTensor(value);
        built.model->SetParameters(parameters);
        built.model->SetTraining(false);

        const Tensor input = FixtureTensor(c.at("input"));
        const Tensor output = built.model->Forward(input);
        ExpectCloseTo(RowMajor(output), c.at("output"), atol, rtol, name + " output");
        const auto rule = layer_view::OutputRuleFromKey(c.at("rule").get<std::string>());
        const auto weights = sal::TargetWeights(output.NumElements(), rule, c.at("target").get<int>());
        ExpectCloseTo(weights, c.at("weights"), 0.0f, 0.0f, name + " output weighting");

        Tensor gradient, values = input;
        if (kind == occlusion::InputKind::Tokens) {
            built.model->SetLayerForwardObserver([&](size_t i, const Tensor&, const Tensor& out, float) {
                if (i == 0) values = out.Clone();
            });
            built.model->SetLayerBackwardObserver([&](size_t i, const Tensor& g) {
                if (i == 0) gradient = g.Clone();
            });
            built.model->Forward(input);
            built.model->Backward(FromRowMajor(weights, output.Shape()));
            ExpectCloseTo(RowMajor(values), c.at("embedded"), atol, rtol, name + " embedding output");
        } else {
            built.model->Forward(input);
            gradient = built.model->Backward(FromRowMajor(weights, output.Shape()));
        }
        ExpectCloseTo(RowMajor(gradient), c.at("gradient"), atol, rtol, name + " gradient");
        const auto reduced = sal::Reduce(kind, gradient, values);
        ExpectCloseTo(AsFloats(reduced.size), c.at("size"), atol, rtol, name + " size");
        ExpectCloseTo(AsFloats(reduced.times_input), c.at("times_input"), atol, rtol, name + " gradient x input");
    }

    // Run on a rebuilt model gives Reduce of the session's own gradient.
    const auto cfg = MakeTabularConfig();
    DebugExecutor exe(cfg);
    exe.SetLayerCapture({});
    const auto res = exe.Run();
    std::string error;
    auto session = DebugModelSession::FromCapture(cfg, *res.layer_capture, &error);
    ExpectTrue(session != nullptr, "the MLP is rebuilt: " + error);
    const Tensor all = CapturedTensorOf(res.layer_capture->Layers().front(), CapturedTensorRole::Input);
    const auto row = RowMajor(all);
    const std::vector<float> first(row.begin(), row.begin() + 10);
    const Tensor one = FromRowMajor(first, {1, 10});
    const auto run = sal::Run(*session, one, occlusion::InputKind::Features, layer_view::OutputRule::Softmax, 1);
    std::vector<float> w(4, 0.0f);
    w[1] = 1.0f;
    const auto direct = sal::Reduce(occlusion::InputKind::Features, session->InputGradient(one, FromRowMajor(w, {1, 4})), one);
    ExpectTrue(run.size == direct.size && run.times_input == direct.times_input,
               "Run is Reduce of the session's input gradient");

    // Tokens through Run: one value per token position.
    const auto text_cfg = MakeTextConfig();
    DebugExecutor text_exe(text_cfg);
    text_exe.SetLayerCapture({});
    const auto text_res = text_exe.Run();
    auto text_session = DebugModelSession::FromCapture(text_cfg, *text_res.layer_capture, &error);
    ExpectTrue(text_session != nullptr, "the text model is rebuilt: " + error);
    const auto ids = RowMajor(CapturedTensorOf(text_res.layer_capture->Layers().front(), CapturedTensorRole::Input));
    const Tensor sentence = FromRowMajor(std::vector<float>(ids.begin(), ids.begin() + 16), {1, 16});
    const auto tokens = sal::Run(*text_session, sentence, occlusion::InputKind::Tokens, layer_view::OutputRule::Softmax, 0, 0);
    ExpectEq(tokens.size.size(), 16, "one saliency value per token");
    ExpectEq(tokens.times_input.size(), 16, "one gradient x embedding per token");
    spdlog::info("  OK: saliency matches PyTorch ({} cases), Run == Reduce", fixture.at("cases").size());
}

// TOFIX178 6.3d: training snapshots in words (approved mockup board Snapshots).
void TestTrainingSnapshotsPresentation() {
    spdlog::info("--- TestTrainingSnapshotsPresentation ---");
    namespace ts = training_snapshots;
    const std::string dot = " \xC2\xB7 ";
    ExpectTrue(ts::WhenFromRunId("train-1793417384228").size() == 16, "a run id's time: " + ts::WhenFromRunId("train-1793417384228"));
    ExpectTrue(ts::WhenFromRunId("run-a").empty() && ts::WhenFromRunId("train-12").empty(), "no time in the id: none");
    ExpectTrue(ts::SizeText(19293798) == "18.4 MB" && ts::SizeText(2048) == "2.0 KB", "sizes");
    ExpectTrue(ts::SnapshotName("epoch_0020") == "epoch 20" && ts::SnapshotName("local_debug") == "local_debug",
               "snapshot names");

    // The approved board's run: 20 epochs, snapshots every 5.
    const float val[] = {0.6425f, 0.6258f, 0.7501f, 0.8003f, 0.8259f, 0.8259f, 0.8292f, 0.8623f, 0.8828f, 0.8949f,
                         0.8912f, 0.9108f, 0.9083f, 0.9117f, 0.9234f, 0.9217f, 0.928f, 0.9297f, 0.923f, 0.9296f};
    TrainingRunRecord record;
    record.run_id = "train-1793417384228";
    record.graph_name = "cats_dogs_resnet_scratch";
    record.total_epochs = 20;
    for (int e = 1; e <= 20; ++e) record.epochs.push_back({e, 1.0f / e, 0.5f, 0.9f / e, val[e - 1]});
    std::vector<LayerCaptureInfo> captures;
    for (int e : {5, 10, 15, 20}) captures.push_back({EpochCaptureLabel(e), e, 19293798});
    const auto row = ts::TrainingRunRow(record.run_id, record, captures);
    ExpectTrue(row.training && row.kind == "Training" &&
                   row.title == "train-1793417384228" + dot + "cats_dogs_resnet_scratch",
               "a training run's title names its graph: " + row.title);
    ExpectTrue(row.detail == "20 epochs" + dot + "best validation 92.97 % (epoch 18)" + dot +
                                 "4 layer snapshots (epochs 5, 10, 15, 20)",
               "a training run's line: " + row.detail);
    const auto bare = ts::TrainingRunRow("train-1793427851890", std::nullopt, {{EpochCaptureLabel(2), 2, 10}});
    ExpectTrue(bare.detail == "1 layer snapshot (epoch 2)" + dot + "epochs not recorded",
               "snapshots saved before the epochs were kept: " + bare.detail);

    const auto table = ts::SnapshotTableFor(record.run_id, record, captures);
    ExpectTrue(table.heading == "Snapshots of train-1793417384228" + dot + "cats_dogs_resnet_scratch (20 epochs)",
               "table heading: " + table.heading);
    ExpectTrue(table.metric_header == "Validation accuracy" && table.rows.size() == 4, "four snapshot rows");
    ExpectTrue(table.rows[3].epoch == 20 && table.rows[3].label == "epoch_0020" && table.rows[3].val_metric == "92.96 %" &&
                   table.rows[3].train_loss == "0.050" && table.rows[3].val_loss == "0.045" && table.rows[3].size == "18.4 MB",
               "the epoch 20 row");
    const auto old = ts::SnapshotTableFor("train-1", std::nullopt, captures);
    ExpectTrue(old.rows[0].val_metric == "not recorded" && old.heading == "Snapshots of train-1", "not recorded");
    TrainingRunRecord mae = record;
    mae.regression = true;
    for (auto& e : mae.epochs) e.val_metric = 1.0f - *e.val_metric;
    ExpectTrue(ts::SnapshotTableFor(mae.run_id, mae, captures).metric_header == "Validation MAE" &&
                   ts::TrainingRunRow(mae.run_id, mae, captures).detail.find("best validation MAE 0.070 (epoch 18)") !=
                       std::string::npos,
               "a regression run: the lowest MAE is the best");
    ExpectTrue(ts::StepperText(3, 4, 20, record) == "4 of 4 snapshots" + dot + "validation 92.96 %", "stepper text");
    ExpectTrue(ts::StepperText(0, 1, 2, std::nullopt) == "1 of 1 snapshot", "stepper without a record");
    ExpectTrue(ts::SnapshotTitle("epoch_0020", record) == "Epoch 20" + dot + "validation 92.96 %" &&
                   ts::SnapshotTitle("epoch_0002", std::nullopt) == "Epoch 2" &&
                   ts::SnapshotTitle("local_debug", record) == "local_debug",
               "a compared side's title");

    DebugRunStoreSummary debug;
    debug.run_id = "local-debug-a2c705e68029f801-1794247562482";
    debug.timestamp = "2026-11-09 22:06";
    debug.success = true;
    debug.trace_count = 72;
    debug.file_path = "C:/runs/session.json";
    debug.summary = "Local Debug: Passed";
    const auto d = ts::DebugRunRow(debug);
    ExpectTrue(!d.training && d.kind == "Local Debug" && d.detail == "Passed", "a debug run's line: " + d.detail);
    ExpectTrue(d.when == ts::WhenFromRunId(debug.run_id), "a debug run's time from its id, as a training run's");
    bool traces = false, saved = false, summary = false;
    for (const auto& [k, v] : d.details) {
        traces |= k == "Traces" && v == "72";
        saved |= k == "Saved at" && v == "C:/runs/session.json";
        summary |= k == "Summary" && v == "Local Debug: Passed";
    }
    ExpectTrue(traces && saved && summary, "a debug run keeps its facts, and its summary, under Details");
    spdlog::info("  OK: training snapshots presentation");
}

// TOFIX178 6.3e: top images. Strengths per channel and item against host
// arithmetic, item ranges in both layouts, a batched run equal to items one
// at a time on the ResNet, and the Dense head's votes.
void TestTopImages() {
    spdlog::info("--- TestTopImages ---");
    namespace ti = top_images;
    // [H=2, W=2, C=3, N=2] row-major: ((h * 2 + w) * 3 + c) * 2 + n.
    std::vector<float> maps(24);
    for (size_t i = 0; i < maps.size(); ++i) maps[i] = std::sin(1.3f * float(i)) * 3.0f;
    const Tensor spatial = FromRowMajor(maps, {2, 2, 3, 2});
    for (const auto rank : {ti::Rank::Maximum, ti::Rank::Mean}) {
        const auto s = ti::ChannelStrengths(spatial, rank);
        ExpectTrue(s.channels == 3 && s.items == 2, "a spatial output's channels and items");
        for (size_t c = 0; c < 3; ++c)
            for (size_t n = 0; n < 2; ++n) {
                float best = -1e30f, sum = 0.0f;
                for (size_t pos = 0; pos < 4; ++pos) {
                    const float v = maps[(pos * 3 + c) * 2 + n];
                    best = std::max(best, v);
                    sum += v;
                }
                const float want = rank == ti::Rank::Maximum ? best : sum / 4.0f;
                ExpectTrue(std::fabs(s.At(c, n) - want) < 1e-5f, "channel " + std::to_string(c) + " item " +
                                                                       std::to_string(n) + " strength");
            }
    }
    // [N=3, C=2] row-major: n * 2 + c.
    const auto flat = ti::ChannelStrengths(FromRowMajor({1, 2, 3, 4, 5, 6}, {3, 2}), ti::Rank::Maximum);
    ExpectTrue(flat.channels == 2 && flat.items == 3 && flat.At(0, 2) == 5.0f && flat.At(1, 0) == 2.0f,
               "a flat output's value per channel and item");
    ExpectTrue(ti::TopItems(flat, 1, 2) == std::vector<size_t>({2, 1}), "top items strongest first");
    ExpectTrue(ti::TopItems(flat, 5, 2).empty(), "no channel: none");

    // Item ranges: batch last and batch first.
    const auto images = RowMajor(ti::ItemRange(spatial, 1, 2));
    for (size_t s = 0; s < 12; ++s) ExpectTrue(images[s] == maps[s * 2 + 1], "item 1 of a batch-last batch");
    ExpectTrue(RowMajor(ti::ItemRange(FromRowMajor({1, 2, 3, 4, 5, 6}, {3, 2}), 1, 3)) ==
                   std::vector<float>({3, 4, 5, 6}),
               "items 1-2 of a batch-first batch");

    // The ResNet: five items, batched in chunks of 2 = each item alone.
    const auto resnet = CompileResNetExample();
    DebugExecutor exe(resnet);
    exe.SetLayerCapture({});
    const auto res = exe.Run();
    ExpectTrue(res.success && res.layer_capture, "the ResNet runs with capture: " + res.failure_summary);
    std::string error;
    auto session = DebugModelSession::FromCapture(resnet, *res.layer_capture, &error);
    ExpectTrue(session != nullptr, "the ResNet is rebuilt: " + error);
    const Tensor& image = CapturedTensorOf(res.layer_capture->Layers().front(), CapturedTensorRole::Input);
    const auto pixels = RowMajor(image);  // [H, W, C, 1]
    const size_t n = 5;
    std::vector<float> batch(pixels.size() * n);
    for (size_t s = 0; s < pixels.size(); ++s)
        for (size_t i = 0; i < n; ++i) batch[s * n + i] = pixels[s] * (1.0f - 0.15f * float(i)) + 0.05f * float(i);
    auto shape = image.Shape();
    shape.back() = n;
    const Tensor items = FromRowMajor(batch, shape);
    const CapturedLayer* pooled = nullptr;
    const CapturedLayer* conv = nullptr;
    for (const auto& l : res.layer_capture->Layers()) {
        const auto& out = CapturedTensorOf(l, CapturedTensorRole::Output).Shape();
        if (!pooled && out.size() == 2 && out[1] > 1) pooled = &l;
        if (!conv && out.size() == 4) conv = &l;
    }
    ExpectTrue(pooled && conv, "the ResNet has a pooled and a conv layer");
    size_t progress_calls = 0;
    for (const CapturedLayer* layer : {pooled, conv}) {
        const auto batched = ti::Run(*session, items, layer->module_index, ti::Rank::Maximum, 2,
                                     [&](size_t, size_t) { ++progress_calls; });
        ExpectTrue(!batched.cancelled && batched.items == n && batched.channels > 1, layer->node_name + " strengths");
        float worst = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            const auto alone = ti::ChannelStrengths(session->LayerOutput(ti::ItemRange(items, i, i + 1),
                                                                         layer->module_index),
                                                    ti::Rank::Maximum);
            for (size_t c = 0; c < batched.channels; ++c) worst = std::max(worst, std::fabs(alone.At(c, 0) - batched.At(c, i)));
        }
        ExpectTrue(worst < 1e-4f, layer->node_name + ": batched == one at a time, worst " + std::to_string(worst));
    }
    ExpectEq(progress_calls, 6, "progress after each chunk (3 chunks per layer)");
    const auto stopped = ti::Run(*session, items, pooled->module_index, ti::Rank::Maximum, 2, {}, [] { return true; });
    ExpectTrue(stopped.cancelled, "Stop ends the run");
    bool refused = false;
    try {
        session->LayerOutput(image, 999);
    } catch (const std::exception&) {
        refused = true;
    }
    ExpectTrue(refused, "a layer not in the model is refused");

    // Votes: one logit votes both classes; two outputs vote each.
    const auto one = ti::StrongestVotes({0.3f, -0.5f, 0.1f, -0.2f, 0.4f}, 1, 5, 2);
    ExpectTrue(one.size() == 4 && one[0].channel == 4 && one[0].towards == 1 && one[1].channel == 0 &&
                   one[2].channel == 1 && one[2].towards == 0 && one[3].channel == 3,
               "one logit: the largest positive weights vote class 1, the largest negative class 0");
    const auto two = ti::StrongestVotes({0.1f, 0.9f, 0.2f, 0.7f, 0.0f, -0.4f}, 2, 3, 1);
    ExpectTrue(two.size() == 2 && two[0].channel == 1 && two[0].towards == 0 && two[1].channel == 0 &&
                   two[1].towards == 1,
               "two outputs: each votes with its largest weight");

    // Items spread over the data (6.3e): every k-th, and a stand-in batcher
    // handing back exactly those numbers.
    const auto spread = SpreadItems(25000, 0, 320);
    ExpectTrue(spread.count == 320 && spread.stride == 78, "320 of 25,000: every 78th");
    ExpectTrue(SpreadItems(100, 0, 320).count == 100 && SpreadItems(100, 0, 320).stride == 1,
               "more asked than there are: all of them, consecutive");
    ExpectTrue(SpreadItems(10, 10, 4).count == 0, "nothing past the end");
    struct Numbered : IBatcher {
        Batch GetNextBatch() override { return {}; }
        void Reset() override {}
        bool IsEpochComplete() const override { return true; }
        size_t GetNumBatches() const override { return 0; }
        size_t GetNumSamples() const override { return 0; }
        void SetNormalization(float, float) override {}
        void SetOneHotEncoding(size_t) override {}
        void SetFlatten(bool) override {}
        size_t SourceSampleCount() const override { return 40; }
        std::optional<Batch> GetSourceSamples(const std::vector<size_t>& indices) override {
            std::vector<float> v;
            for (size_t i : indices) {
                if (i >= 40) throw std::out_of_range("past the end");
                v.push_back(static_cast<float>(i));
            }
            Batch b;
            b.data = FromRowMajor(v, {v.size(), 1});
            return b;
        }
    } numbered;
    size_t taken = 0;
    const auto every = GetSourceSampleRange(numbered, 2, 4, &taken, 9);
    ExpectTrue(every && taken == 4 && RowMajor(every->data) == std::vector<float>({2, 11, 20, 29}),
               "items 2, 11, 20, 29 by number");
    const auto past = GetSourceSampleRange(numbered, 30, 4, &taken, 9);
    ExpectTrue(past && taken == 1 && RowMajor(past->data) == std::vector<float>({30}),
               "running past the end falls back to the first item alone");
    // Words for the result.
    ExpectTrue(ti::ValueText(4.2134f) == "4.21" && ti::ValueText(0.0123f) == "0.0123", "values, 3 digits");
    ExpectTrue(ti::Caption(1, "dog") == "#1 \xC2\xB7 dog" && ti::Caption(2, "") == "#2",
               "captions");
    ExpectTrue(ti::RowSummary({"dog", "cat", "dog", "", "dog"}, {4.21f, 4.0f, 3.5f, 3.2f, 3.12f}) ==
                   "3 \xC3\x97 dog, 1 \xC3\x97 cat \xC2\xB7 4.21 to 3.12",
               "a row's summary: labels counted, the value range");
    ExpectTrue(ti::RowSummary({}, {2.0f, 1.0f}) == "2 to 1", "no labels: the range alone");
    spdlog::info("  OK: top images");
}

// TOFIX178 6.4: Contributions. Terms by hand; the MLP's output layer
// rebuilt from its captured input, weights and bias equals its captured
// output; a row read from the device; an input's place.
void TestContributions() {
    spdlog::info("--- TestContributions ---");
    namespace ct = contributions;
    const auto e = ct::Explain({0.5f, -2.0f, 0.1f, 1.0f}, {2.0f, 1.0f, 3.0f, -0.25f}, 0.5, 2);
    ExpectTrue(e.top.size() == 2 && e.top[0].input == 1 && e.top[0].wx == -2.0f && e.top[1].input == 0,
               "the largest |w x x| first");
    ExpectTrue(std::fabs(e.sum - (1.0 - 2.0 + 0.3 - 0.25)) < 1e-6 && std::fabs(e.value - (e.sum + 0.5)) < 1e-9,
               "the sum and the value");
    ExpectTrue(e.rest == 2 && std::fabs(e.rest_sum - 0.05) < 1e-7, "the rest of the inputs and their share");
    ExpectTrue(ct::InputPlace(4 * 8 * 128 + 6 * 128 + 65, {8, 8, 128}) == "row 4, column 6, channel 65",
               "an input's place in an [H, W, C] layer");
    ExpectTrue(ct::InputPlace(3, {8}).empty(), "no place in a flat layer");
    ExpectTrue(ct::Signed(0.06811) == "+0.0681" && ct::Signed(-0.08333) == "-0.0833" && ct::Signed(0.00001) == "0.0000",
               "signed values");
    ExpectTrue(ct::MatrixRow(FromRowMajor({1, 2, 3, 4, 5, 6}, {2, 3}), 1) == std::vector<float>({4, 5, 6}),
               "a row of a matrix");

    // The MLP's output Dense: sum of w x x + b == its captured output.
    DebugExecutor exe(MakeTabularConfig());
    exe.SetLayerCapture({});
    const auto res = exe.Run();
    ExpectTrue(res.success && res.layer_capture, "the MLP runs with capture: " + res.failure_summary);
    const CapturedLayer& dense = CapturedNamed(*res.layer_capture, "Output Dense");
    const Tensor& input = CapturedTensorOf(dense, CapturedTensorRole::Input);
    const Tensor& weight = CapturedTensorOf(dense, CapturedTensorRole::Parameter, "weight");
    const Tensor& bias = CapturedTensorOf(dense, CapturedTensorRole::Parameter, "bias");
    const auto out = RowMajor(CapturedTensorOf(dense, CapturedTensorRole::Output));
    const auto x_all = RowMajor(input);
    const auto b = RowMajor(bias);
    const size_t in = input.Shape()[1], outputs = weight.Shape()[0];
    float worst = 0.0f;
    for (size_t item = 0; item < input.Shape()[0]; ++item)
        for (size_t o = 0; o < outputs; ++o) {
            const std::vector<float> x(x_all.begin() + static_cast<std::ptrdiff_t>(item * in),
                                       x_all.begin() + static_cast<std::ptrdiff_t>((item + 1) * in));
            const auto ex = ct::Explain(ct::MatrixRow(weight, o), x, b[o], 8);
            worst = std::max(worst, std::fabs(static_cast<float>(ex.value) - out[item * outputs + o]));
        }
    ExpectTrue(worst < 1e-4f, "sum of w x x + b == the captured output, worst " + std::to_string(worst));
    spdlog::info("  OK: contributions (MLP output rebuilt, worst {})", worst);
}

// TOFIX178 6.4b: Embedding. Two clouds in 8-D stay apart in 2-D with each
// method; unit scaling, the nearest point and the class counts.
void TestLayerEmbedding() {
    spdlog::info("--- TestLayerEmbedding ---");
    namespace le = layer_embedding;
    const size_t n = 40, f = 8;
    std::vector<float> rows(n * f);
    std::vector<int> labels(n);
    for (size_t i = 0; i < n; ++i) {
        labels[i] = static_cast<int>(i % 2);
        for (size_t k = 0; k < f; ++k)
            rows[i * f + k] = (labels[i] ? 5.0f : -5.0f) + 0.3f * std::sin(1.7f * float(i * f + k));
    }
    for (const auto method : {le::Method::PCA, le::Method::TSNE, le::Method::UMAP}) {
        const auto p = le::Project(rows, n, f, method);
        ExpectTrue(p.error.empty() && p.x.size() == n && p.y.size() == n,
                   std::string(le::MethodName(method)) + " projects every item: " + p.error);
        // The two classes' centres are further apart than either cloud is wide.
        double cx[2] = {0, 0}, cy[2] = {0, 0}, spread[2] = {0, 0};
        for (size_t i = 0; i < n; ++i) {
            cx[labels[i]] += p.x[i] / (n / 2.0);
            cy[labels[i]] += p.y[i] / (n / 2.0);
        }
        for (size_t i = 0; i < n; ++i)
            spread[labels[i]] = std::max(spread[labels[i]], std::hypot(p.x[i] - cx[labels[i]], p.y[i] - cy[labels[i]]));
        const double apart = std::hypot(cx[0] - cx[1], cy[0] - cy[1]);
        ExpectTrue(apart > spread[0] && apart > spread[1],
                   std::string(le::MethodName(method)) + " keeps the two clouds apart");
        ExpectTrue(!p.about.empty(), "the projection says how it was made");
    }
    const auto few = le::Project({1, 2}, 2, 1, le::Method::PCA);
    ExpectTrue(!few.error.empty(), "too few items: said so");

    le::Projection small;
    small.x = {0.0f, 10.0f, 5.0f};
    small.y = {2.0f, 2.0f, 2.0f};
    const auto unit = le::ToUnit(small);
    ExpectTrue(unit.x == std::vector<float>({0.0f, 1.0f, 0.5f}) && unit.y == std::vector<float>({0.5f, 0.5f, 0.5f}),
               "unit scaling; a constant axis sits at 0.5");
    ExpectTrue(le::Nearest(unit, 0.95f, 0.5f, 0.1f, 1.0f) == std::optional<size_t>(1), "the nearest point");
    ExpectTrue(!le::Nearest(unit, 0.25f, 0.9f, 0.1f, 1.0f), "nothing near: none");
    const std::vector<float> three = {0.0f, 0.0f, 3.0f, 4.0f, 10.0f, -2.0f};
    const auto twin = le::SameRow(three, 3, 2, {3.0f + 1e-6f, 4.0f});
    ExpectTrue(twin && *twin == 1, "a sample that is one of the items is found (float noise allowed)");
    ExpectTrue(!le::SameRow(three, 3, 2, {3.0f, 4.1f}), "a sample near an item is not taken for it");
    const auto counts = le::ClassCounts({1, 0, 1, -1, 1}, {"cat", "dog"});
    ExpectTrue(counts.size() == 2 && counts[0].label == 0 && counts[0].name == "cat" && counts[0].count == 1 &&
                   counts[1].label == 1 && counts[1].name == "dog" && counts[1].count == 3,
               "class counts, items without a class left out");
    spdlog::info("  OK: layer embedding (PCA, t-SNE, UMAP)");
}

// TOFIX178 6.4c: unit weights as images. A Flatten hands the Dense the image
// [H, W, C] row-major, so weight k of a unit is that pixel and colour; norms
// and rows read on the device match the host; order, drawing, the image rule.
void TestUnitWeights() {
    spdlog::info("--- TestUnitWeights ---");
    namespace uw = unit_weights;
    // Flatten order on a built model: the Dense's input is the sample read row-major.
    TrainingConfiguration cfg;
    cfg.preprocessing_domain = PreprocessingDomain::Image;
    cfg.input_shape = {4, 3, 2};
    cfg.input_size = 24;
    cfg.output_size = 5;
    cfg.loss_type = gui::NodeType::CrossEntropyLoss;
    cfg.optimizer_type = gui::NodeType::Adam;
    CompiledLayer flatten;
    flatten.type = gui::NodeType::Flatten;
    flatten.input_shape = {4, 3, 2};
    flatten.output_shape = {24};
    CompiledLayer dense;
    dense.type = gui::NodeType::Dense;
    dense.units = 5;
    dense.input_shape = {24};
    dense.output_shape = {5};
    cfg.layers = {flatten, dense};
    auto built = BuildExecutableFromConfig(cfg);
    ExpectTrue(built.ok(), "the image MLP builds: " + built.error_message);
    std::vector<float> pixels(24);
    for (size_t k = 0; k < pixels.size(); ++k) pixels[k] = 0.25f * static_cast<float>(k) - 1.0f;
    // As the image batcher hands it out: batch first [1, H, W, C].
    const Tensor image = FromRowMajor(pixels, {1, 4, 3, 2});
    Tensor seen, flat;
    built.model->SetLayerForwardObserver([&](size_t i, const Tensor& in, const Tensor& out, float) {
        if (i == 0) {
            seen = in.Clone();
            flat = out.Clone();
        }
    });
    built.model->Forward(image);
    ExpectTrue(seen.Shape() == std::vector<size_t>({1, 4, 3, 2}), "the Flatten takes the image batch first");
    ExpectTrue(RowMajor(flat) == pixels, "a Flatten hands the Dense the image [H, W, C] row-major");

    // Norms and rows on the device == the host.
    const std::vector<float> w = {3, 4, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1};  // [3 units, 4 inputs]
    const Tensor weights = FromRowMajor(w, {3, 4});
    const auto norms = uw::RowNorms(weights);
    ExpectTrue(norms.size() == 3 && std::fabs(norms[0] - 5.0f) < 1e-6f && std::fabs(norms[1] - std::sqrt(3.0f)) < 1e-6f &&
                   std::fabs(norms[2] - 2.0f) < 1e-6f,
               "each unit's weight norm");
    ExpectTrue(uw::Order(norms, true, 0) == std::vector<size_t>({0, 2, 1}), "the largest norm first");
    ExpectTrue(uw::Order(norms, true, 2) == std::vector<size_t>({0, 2}), "the first N by norm");
    ExpectTrue(uw::Order(norms, false, 2) == std::vector<size_t>({0, 1}), "unit order");
    ExpectTrue(uw::Rows(weights, {2, 0}) == std::vector<float>({1, 1, 1, 1, 3, 4, 0, 0}), "rows gathered in order");

    // Drawing: grey keeps 0 at 0 and the sign; colour spans 0..1.
    const float grey[4] = {-2.0f, 1.0f, 0.0f, 4.0f};
    const auto g = uw::Draw(grey, {2, 2, 1});
    ExpectTrue(g.pixels == std::vector<float>({-0.5f, 0.25f, 0.0f, 1.0f}) && g.low == -2.0f && g.high == 4.0f,
               "a grey unit over its largest |weight|");
    const float colour[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12};
    const auto c = uw::Draw(colour, {2, 2, 3});
    ExpectTrue(c.pixels.front() == 0.0f && c.pixels.back() == 1.0f && std::fabs(c.pixels[6] - 0.5f) < 1e-6f,
               "a colour unit over its own range");

    // The image rule.
    ExpectTrue(uw::ReadsImage(784, 28, 28, 1, true).has_value(), "a first Dense on a 28 x 28 image");
    ExpectTrue(uw::ReadsImage(12288, 64, 64, 3, true).has_value(), "a first Dense on a 64 x 64 colour image");
    ExpectTrue(!uw::ReadsImage(784, 28, 28, 1, false), "not after a layer with parameters");
    ExpectTrue(!uw::ReadsImage(100, 28, 28, 1, true), "not with another input size");
    ExpectTrue(!uw::ReadsImage(8, 2, 2, 2, true), "not for 2 channels");
    ExpectTrue(uw::Lead({28, 28, 1}, 128, 32, true).find("784 input weights drawn as the 28 \xC3\x97 28 image") !=
                   std::string::npos,
               "the lead names the picture");
    spdlog::info("  OK: unit weights (Flatten order, norms, rows, drawing)");
}

void TestDebugModelSession() {
    spdlog::info("--- TestDebugModelSession ---");
    // An MLP (no dropout or BatchNorm: the run's training forward equals
    // eval): the rebuilt model gives the run's output for the run's input.
    const auto cfg = MakeTabularConfig();
    DebugExecutor exe(cfg);
    exe.SetLayerCapture({});
    const auto res = exe.Run();
    ExpectTrue(res.success && res.layer_capture, "the MLP runs with capture: " + res.failure_summary);
    const LayerCapture& capture = *res.layer_capture;
    std::string error;
    auto session = DebugModelSession::FromCapture(cfg, capture, &error);
    ExpectTrue(session != nullptr, "the MLP's model is rebuilt from its capture: " + error);
    const Tensor& input = CapturedTensorOf(capture.Layers().front(), CapturedTensorRole::Input);
    const Tensor& output = CapturedTensorOf(capture.Layers().back(), CapturedTensorRole::Output);
    const float worst = MaxAbsDifference(RowMajor(session->Forward(input)), RowMajor(output));
    ExpectTrue(worst < 1e-5f, "the rebuilt MLP gives the run's output, worst " + std::to_string(worst));
    CheckInputGradient(*session, input, {0, 3, 7, 9}, 1e-2f, "MLP");

    // A layer's output and the gradient at it (TOFIX204): the last layer's
    // gradient is the target; the hidden ReLU's output is the run's; the
    // first layer's backward of its gradient is the input gradient.
    {
        const Tensor out = session->Forward(input);
        std::vector<float> target(out.NumElements(), 0.0f);
        target[1] = 1.0f;
        const Tensor weights = FromRowMajor(target, out.Shape());
        const auto last = session->LayerOutputGradient(input, weights, 2);
        ExpectTrue(RowMajor(last.gradient) == target, "the last layer's output gradient is the target");
        const auto hidden = session->LayerOutputGradient(input, weights, 1);
        const float hidden_worst = MaxAbsDifference(
            RowMajor(hidden.output), RowMajor(CapturedTensorOf(CapturedNamed(capture, "Hidden ReLU"),
                                                               CapturedTensorRole::Output)));
        ExpectTrue(hidden_worst < 1e-5f, "the hidden layer's output is the run's, worst " + std::to_string(hidden_worst));
        ExpectTrue(hidden.gradient.Shape() == hidden.output.Shape(), "the gradient has the layer output's shape");
        const auto first = session->LayerOutputGradient(input, weights, 0);
        bool not_reached = false;
        try {
            session->LayerOutputGradient(input, weights, 7);
        } catch (const std::exception&) {
            not_reached = true;
        }
        ExpectTrue(not_reached, "a layer that is not in the model is refused");
        // The observers are cleared: the session answers as before.
        const float again = MaxAbsDifference(RowMajor(session->Forward(input)), RowMajor(out));
        ExpectTrue(again == 0.0f, "the session's forward is unchanged after a layer gradient");
        ExpectTrue(first.gradient.Shape() == std::vector<size_t>({input.Shape()[0], 32}),
                   "the first Dense's output gradient is [N, 32]");
    }

    // A text model: the gradient at the Embedding output, one row per token.
    {
        const auto text_cfg = MakeTextConfig();
        DebugExecutor text_exe(text_cfg);
        text_exe.SetLayerCapture({});
        const auto text_res = text_exe.Run();
        ExpectTrue(text_res.success && text_res.layer_capture, "the text model runs with capture: " + text_res.failure_summary);
        error.clear();
        auto text_session = DebugModelSession::FromCapture(text_cfg, *text_res.layer_capture, &error);
        ExpectTrue(text_session != nullptr, "the text model is rebuilt from its capture: " + error);
        const Tensor& ids = CapturedTensorOf(text_res.layer_capture->Layers().front(), CapturedTensorRole::Input);
        const Tensor text_out = text_session->Forward(ids);
        std::vector<float> target(text_out.NumElements(), 0.0f);
        target[0] = 1.0f;
        const auto at_embedding = text_session->LayerOutputGradient(ids, FromRowMajor(target, text_out.Shape()), 0);
        const auto& id_shape = ids.Shape();
        ExpectTrue(at_embedding.gradient.Shape() == std::vector<size_t>({id_shape[0], id_shape[1], 8}),
                   "the gradient at the Embedding output is [N, T, D]");
        ExpectTrue(at_embedding.output.Shape() == at_embedding.gradient.Shape(), "the Embedding output is [N, T, D]");
        const auto g = RowMajor(at_embedding.gradient);
        ExpectTrue(std::any_of(g.begin(), g.end(), [](float v) { return v != 0.0f; }),
                   "the first item's tokens have a gradient");
    }

    // A capture without the weights cannot rebuild the model, and says why.
    DebugExecutor activations_only(cfg);
    activations_only.SetLayerCapture({/*activations=*/true, /*parameters=*/false, /*gradients=*/false});
    const auto a_res = activations_only.Run();
    error.clear();
    ExpectTrue(!DebugModelSession::FromCapture(cfg, *a_res.layer_capture, &error) &&
                   error.find("did not keep") != std::string::npos,
               "a capture without weights is refused: " + error);

    // A target of the wrong shape is refused.
    bool refused = false;
    try {
        session->InputGradient(input, FromRowMajor({1.0f}, {1}));
    } catch (const std::exception&) {
        refused = true;
    }
    ExpectTrue(refused, "a target that is not the output's shape is refused");

    // The ResNet example (branches into Add, BatchNorm kept state): rebuilt
    // in eval mode, its input gradient matches central differences.
    const auto resnet = CompileResNetExample();
    DebugExecutor r_exe(resnet);
    r_exe.SetLayerCapture({});
    const auto r_res = r_exe.Run();
    ExpectTrue(r_res.success && r_res.layer_capture, "the ResNet runs with capture: " + r_res.failure_summary);
    error.clear();
    auto r_session = DebugModelSession::FromCapture(resnet, *r_res.layer_capture, &error);
    ExpectTrue(r_session != nullptr, "the ResNet's model is rebuilt from its capture: " + error);
    const Tensor& image = CapturedTensorOf(r_res.layer_capture->Layers().front(), CapturedTensorRole::Input);
    const Tensor r_out = r_session->Forward(image);
    ExpectTrue(r_out.Shape() == CapturedTensorOf(r_res.layer_capture->Layers().back(), CapturedTensorRole::Output).Shape(),
               "the rebuilt ResNet's output has the run's output shape");
    for (float v : RowMajor(r_out)) ExpectTrue(std::isfinite(v), "the rebuilt ResNet's output is finite");
    const size_t pixels = image.NumElements();
    CheckInputGradient(*r_session, image, {pixels / 7, pixels / 2 + 13, pixels - 5}, 1e-2f, "ResNet");
    spdlog::info("  OK: MLP rebuilt output worst {}", worst);
}

// TOFIX178 6.3b: occlusion covers the right elements in every layout, and a
// batch scores each piece exactly as covering it alone does.
void TestOcclusion() {
    spdlog::info("--- TestOcclusion ---");
    namespace oc = cyxwiz::occlusion;
    using layer_view::OutputRule;

    // Kinds from the input shape.
    ExpectTrue(oc::KindOf({64, 64, 3, 1}, false) == oc::InputKind::Image, "an image");
    ExpectTrue(oc::KindOf({1, 40}, true) == oc::InputKind::Tokens, "token ids into an Embedding");
    ExpectTrue(oc::KindOf({1, 8}, false) == oc::InputKind::Features, "features");
    ExpectTrue(!oc::KindOf({1, 10, 4}, false), "a sequence of vectors is none of the three");

    // An image [4, 5, 2, 1], patch 2, step 2: rows 0, 2; columns 0, 2, 3 (flush).
    oc::Plan image;
    image.kind = oc::InputKind::Image;
    image.sample_shape = {4, 5, 2, 1};
    image.patch = 2;
    image.step = 2;
    image.fill = {-1.0f, -2.0f};
    const auto patches = oc::PiecesOf(image);
    ExpectEq(patches.size(), 6, "2 x 3 patch positions");
    ExpectTrue(oc::PatchGrid(image) == std::make_pair<size_t, size_t>(2, 3), "the patch grid");
    ExpectTrue(patches.back().row == 2 && patches.back().column == 3, "the last patch is flush with the edges");
    std::vector<float> pixels(40);
    for (size_t i = 0; i < pixels.size(); ++i) pixels[i] = static_cast<float>(i);
    const Tensor sample = FromRowMajor(pixels, {4, 5, 2, 1});
    const Tensor batch = oc::CoveredBatch(sample, image, patches);
    ExpectTrue(batch.Shape() == std::vector<size_t>({4, 5, 2, 6}), "one image per patch, batch last");
    const auto got = RowMajor(batch);
    size_t wrong = 0;
    for (size_t h = 0; h < 4; ++h)
        for (size_t w = 0; w < 5; ++w)
            for (size_t c = 0; c < 2; ++c)
                for (size_t k = 0; k < patches.size(); ++k) {
                    const auto& p = patches[k];
                    const bool covered = h >= p.row && h < p.row + p.rows && w >= p.column && w < p.column + p.columns;
                    const float want = covered ? image.fill[c] : pixels[(h * 5 + w) * 2 + c];
                    if (got[((h * 5 + w) * 2 + c) * patches.size() + k] != want) ++wrong;
                }
    ExpectEq(wrong, 0, "each image has exactly its patch filled with its channel's value");

    // Tokens [1, 6] with two pads (0): four pieces; each copy has one token padded.
    oc::Plan tokens;
    tokens.kind = oc::InputKind::Tokens;
    tokens.sample_shape = {1, 6};
    tokens.fill = {0.0f};
    const std::vector<float> ids = {5, 9, 2, 7, 0, 0};
    const auto positions = oc::PiecesOf(tokens, ids, 0.0f);
    ExpectEq(positions.size(), 4, "padding is not covered");
    const auto token_batch = RowMajor(oc::CoveredBatch(FromRowMajor(ids, {1, 6}), tokens, positions));
    wrong = 0;
    for (size_t k = 0; k < positions.size(); ++k)
        for (size_t t = 0; t < 6; ++t)
            if (token_batch[k * 6 + t] != (t == positions[k].position ? 0.0f : ids[t])) ++wrong;
    ExpectEq(wrong, 0, "each copy has one token replaced by the pad id, batch first");

    // Scores through the output rule.
    ExpectNear(static_cast<float>(oc::TargetScore({0.0f, std::log(3.0f)}, OutputRule::Softmax, 1)), 0.75f, 1e-6f,
               "softmax probability of class 1");
    ExpectNear(static_cast<float>(oc::TargetScore({0.0f}, OutputRule::Sigmoid, 0)), 0.5f, 1e-6f, "sigmoid class 0");
    ExpectNear(static_cast<float>(oc::TargetScore({std::log(3.0f)}, OutputRule::Sigmoid, 0)), 0.25f, 1e-6f,
               "one logit: class 0 is 1 - sigmoid");
    ExpectNear(static_cast<float>(oc::TargetScore({2.5f, -1.0f}, OutputRule::Value, 1)), -1.0f, 0.0f, "a value");

    // The MLP: features replaced by a fill; a batch of 4 scores each feature
    // exactly as covering it alone does.
    const auto cfg = MakeTabularConfig();
    DebugExecutor exe(cfg);
    exe.SetLayerCapture({});
    const auto res = exe.Run();
    std::string error;
    auto session = DebugModelSession::FromCapture(cfg, *res.layer_capture, &error);
    ExpectTrue(session != nullptr, "the MLP session: " + error);
    const Tensor& rows = CapturedTensorOf(res.layer_capture->Layers().front(), CapturedTensorRole::Input);
    const auto row_values = RowMajor(rows);
    const Tensor one_row = FromRowMajor(std::vector<float>(row_values.begin(), row_values.begin() + 10), {1, 10});
    oc::Plan features;
    features.kind = oc::InputKind::Features;
    features.sample_shape = {1, 10};
    features.fill = std::vector<float>(10, 0.25f);
    const auto all = oc::PiecesOf(features);
    const auto batched = oc::Run(*session, one_row, features, all, OutputRule::Softmax, 2, 4);
    const auto alone = oc::Run(*session, one_row, features, all, OutputRule::Softmax, 2, 1);
    ExpectEq(batched.change.size(), 10, "one score per feature");
    double worst = 0.0;
    for (size_t i = 0; i < all.size(); ++i) worst = std::max(worst, std::fabs(batched.change[i] - alone.change[i]));
    ExpectTrue(worst < 1e-6, "MLP: batched scores equal one at a time, worst " + std::to_string(worst));
    auto covered = RowMajor(one_row);
    covered[3] = 0.25f;
    const double by_hand = oc::TargetScore(RowMajor(session->Forward(FromRowMajor(covered, {1, 10}))), OutputRule::Softmax, 2) -
                           batched.baseline;
    ExpectTrue(std::fabs(by_hand - batched.change[3]) < 1e-6, "MLP: feature 3's change is covering it by hand");

    // The ResNet (batch-last images, BatchNorm in eval): 16 patches a forward
    // score each patch as one at a time does; a cancel stops between forwards.
    const auto resnet = CompileResNetExample();
    DebugExecutor r_exe(resnet);
    r_exe.SetLayerCapture({});
    const auto r_res = r_exe.Run();
    auto r_session = DebugModelSession::FromCapture(resnet, *r_res.layer_capture, &error);
    ExpectTrue(r_session != nullptr, "the ResNet session: " + error);
    const Tensor& img = CapturedTensorOf(r_res.layer_capture->Layers().front(), CapturedTensorRole::Input);
    oc::Plan picture;
    picture.kind = oc::InputKind::Image;
    picture.sample_shape = img.Shape();
    picture.fill = std::vector<float>(picture.sample_shape[2], 0.0f);
    auto some = oc::PiecesOf(picture);
    ExpectEq(some.size(), 196, "a 64 x 64 image: 14 x 14 patches of 12, step 4");
    some.resize(20);
    const auto r_batched = oc::Run(*r_session, img, picture, some, OutputRule::Sigmoid, 0, 16);
    const auto r_alone = oc::Run(*r_session, img, picture, some, OutputRule::Sigmoid, 0, 1);
    worst = 0.0;
    double moved = 0.0;
    for (size_t i = 0; i < some.size(); ++i) {
        worst = std::max(worst, std::fabs(r_batched.change[i] - r_alone.change[i]));
        moved = std::max(moved, std::fabs(r_alone.change[i]));
    }
    ExpectTrue(worst < 1e-5, "ResNet: batched patches equal one at a time, worst " + std::to_string(worst));
    ExpectTrue(moved > 0.0, "ResNet: covering patches moves the score");
    size_t forwards = 0;
    const auto stopped = oc::Run(*r_session, img, picture, some, OutputRule::Sigmoid, 0, 4, {},
                                 [&] { return ++forwards > 2; });
    ExpectTrue(stopped.cancelled && stopped.change.size() == 8, "a cancel stops between forwards");
    spdlog::info("  OK: occlusion, ResNet batched vs alone worst {}, largest change {}", worst, moved);
}

// TOFIX178 6.3b: the Where it looks view's words.
void TestWhereItLooksPresentation() {
    spdlog::info("--- TestWhereItLooksPresentation ---");
    namespace wl = cyxwiz::where_it_looks;
    namespace oc = cyxwiz::occlusion;
    using layer_view::OutputRule;
    // One logit through a sigmoid: two classes.
    const auto two = wl::OutputNames(OutputRule::Sigmoid, 1, {"cat", "dog"});
    ExpectTrue(two == std::vector<std::string>({"cat", "dog"}), "one logit explains two classes");
    const auto targets = wl::Targets(two, 0, 1);
    ExpectTrue(targets.size() == 2 && targets[0].label == "Predicted: cat" && targets[1].label == "True: dog",
               "predicted, then true");
    ExpectTrue(wl::Targets(two, 0, 0).size() == 2 && wl::Targets(two, 0, 0)[1].label == "dog",
               "true left out when it is the predicted class; the other class follows");
    ExpectTrue(wl::OutputNames(OutputRule::Value, 2, {})[1] == "output 1", "values are outputs");

    oc::Plan image;
    image.kind = oc::InputKind::Image;
    image.sample_shape = {64, 64, 3, 1};
    ExpectTrue(wl::CoverText(image, "the image's mean per channel") ==
                   "Patch 12 \xC3\x97 12 px, every 4 px, filled with the image's mean per channel",
               "cover text: " + wl::CoverText(image, "the image's mean per channel"));
    ExpectTrue(wl::PassesText(image, 196, 64) ==
                   "196 forward passes (14 \xC3\x97 14 patch positions), 64 at a time on the device, and one on the "
                   "unchanged input.",
               "passes text: " + wl::PassesText(image, 196, 64));
    const auto pieces = oc::PiecesOf(image);
    oc::Result result;
    result.baseline = 0.94;
    result.change.assign(pieces.size(), 0.0);
    result.change[45] = -0.409;
    result.change[3] = 0.12;
    const std::string headline = wl::Headline(image, pieces, result, "cat", {});
    ExpectTrue(headline == "Covering rows 12-23, columns 12-23 lowers the cat score most: 0.940 to 0.531.",
               "headline: " + headline);
    const auto scaled = wl::Scaled(result.change);
    ExpectNear(scaled[45], -1.0f, 1e-6f, "the largest change scales to -1");
    ExpectNear(scaled[3], static_cast<float>(0.12 / 0.409), 1e-6f, "others in proportion");
    oc::Result flat;
    flat.baseline = 0.5;
    flat.change = {0.0, 0.01};
    ExpectTrue(wl::Headline(image, pieces, flat, "cat", {}).rfind("No covered piece lowers", 0) == 0,
               "nothing lowers the score: said so");
    oc::Plan features;
    features.kind = oc::InputKind::Features;
    features.sample_shape = {1, 3};
    ExpectTrue(wl::PieceText(features, oc::PiecesOf(features)[1], {"preg", "plas", "pres"}) == "plas",
               "a feature by its name");
    ExpectTrue(wl::ChangeText(-0.4091) == "-0.409" && wl::ChangeText(0.0001) == "0.000", "change text");
    ExpectTrue(!wl::InspectorHint().empty(), "the view has its Inspector note");

    // Saliency (6.3c).
    using layer_view::OutputRule;
    ExpectTrue(wl::SaliencyOfText(OutputRule::Softmax, 2, 1, "dog") ==
                   "the dog logit (the score before the softmax, which does not saturate)",
               "softmax: the class's logit");
    ExpectTrue(wl::SaliencyOfText(OutputRule::Sigmoid, 1, 1, "positive") == "the logit (the output before the sigmoid)",
               "one sigmoid logit, class 1: the logit");
    ExpectTrue(wl::SaliencyOfText(OutputRule::Sigmoid, 1, 0, "negative") ==
                   "minus the logit: negative rises as the logit falls",
               "one sigmoid logit, class 0: minus the logit");
    ExpectTrue(wl::ScaledSize({0.5, -2.0, 0.0}) == std::vector<float>({0.25f, 1.0f, 0.0f}), "sizes scale by the largest");
    ExpectTrue(wl::ScaledSize({0.0, 0.0}) == std::vector<float>({0.0f, 0.0f}), "no gradient: all zero");
    saliency::Result sal_image;
    sal_image.rows = 2;
    sal_image.columns = 3;
    sal_image.size = {0.1, 0.2, 0.9, 0.0, 0.3, 0.1};
    sal_image.times_input = {0.1, -0.5, 0.2, 0.0, 0.3, 0.1};
    ExpectTrue(wl::SaliencyHeadline(oc::InputKind::Image, sal_image, false, {}, "dog", {}) ==
                   "Most sensitive: the pixel at row 0, column 2.",
               "image size headline: " + wl::SaliencyHeadline(oc::InputKind::Image, sal_image, false, {}, "dog", {}));
    ExpectTrue(wl::SaliencyHeadline(oc::InputKind::Image, sal_image, true, {}, "dog", {}) ==
                   "The pixel at row 0, column 1 pulls the dog score down most (-0.500).",
               "image gradient x input headline");
    saliency::Result sal_tokens;
    sal_tokens.size = {0.9, 0.4, 0.6, 0.0};
    sal_tokens.times_input = {0.2, 0.1, 0.3, 0.0};
    ExpectTrue(wl::SaliencyHeadline(oc::InputKind::Tokens, sal_tokens, false, {1, 2}, "Anxiety", {"#3", "#7", "#12", "#0"}) ==
                   "Most sensitive: \"#12\", then \"#7\".",
               "token headline skips what is not considered (padding)");
    ExpectTrue(wl::SaliencyHeadline(oc::InputKind::Features, sal_tokens, true, {}, "positive", {"preg", "plas", "pres", "skin"}) ==
                   "pres pushes the positive score up most (+0.300).",
               "feature gradient x input headline");
    saliency::Result sal_none;
    sal_none.size = {0.0, 0.0};
    sal_none.times_input = {0.0, 0.0};
    ExpectTrue(wl::SaliencyHeadline(oc::InputKind::Features, sal_none, false, {}, "positive", {}) ==
                   "No element moves the positive score.",
               "no gradient: said so");
    ExpectTrue(wl::SaliencyValueText(0.2141, false) == "0.214" && wl::SaliencyValueText(0.2141, true) == "+0.214" &&
                   wl::SaliencyValueText(-0.0001, true) == "0.000",
               "saliency value text");
    ExpectTrue(!wl::SaliencyInspectorHint().empty() && !wl::SaliencyPassesText().empty(), "saliency's notes");
    spdlog::info("  OK: where it looks presentation");
}

// TOFIX178 6.1b: a training snapshot runs in eval mode, so BatchNorm's
// running statistics and every weight are left exactly as they were.
void TestCaptureStepLeavesTheModel() {
    spdlog::info("--- TestCaptureStepLeavesTheModel ---");
    const auto config = CompileResNetExample();
    auto built = BuildExecutableFromConfig(config);
    ExpectTrue(built.ok(), "ResNet builds: " + built.error_message);
    built.model->SetTraining(true);
    const auto batch = MakeSyntheticBatch(config, 5);
    const Tensor& input = batch.features;  // synthetic batches are model-ready
    std::map<std::string, std::vector<float>> before;
    for (const auto& [name, t] : built.model->GetParameters()) before[name] = HostValues(t);
    const auto capture = CaptureStep(*built.model, *built.loss, input, batch.labels, {}, built.module_provenance);
    ExpectEq(capture->Layers().size(), config.layers.size(), "the snapshot holds every layer");
    size_t changed = 0;
    for (const auto& [name, t] : built.model->GetParameters())
        if (HostValues(t) != before.at(name)) ++changed;
    ExpectEq(changed, 0, "a snapshot changes no weight and no running statistic");
    const auto& stem = CapturedNamed(*capture, "Conv 32 stem");
    ExpectTrue(stem.Find(CapturedTensorRole::Gradient, "weights") != nullptr, "the snapshot holds gradients");
    spdlog::info("  OK: snapshot of {} layers, model unchanged", capture->Layers().size());
}

// TOFIX178 6.1c: a capture saved with its run reads back as it was kept, a
// damaged one is refused, and only the last runs keep their captures.
void TestLayerCaptureStore() {
    spdlog::info("--- TestLayerCaptureStore ---");
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "cyxwiz_layer_capture_store";
    fs::remove_all(root);
    const ScopedDebugRunRootOverrideForTesting debug_root(root);

    DebugExecutor exe(MakeTabularConfig());
    LayerCaptureOptions options;
    options.byte_budget = 4096;  // some tensors left out, so the counts travel too
    exe.SetLayerCapture(options);
    const auto res = exe.Run();
    ExpectTrue(res.success && res.layer_capture && res.layer_capture->Truncated(), "a capture over its budget");
    const LayerCapture& kept = *res.layer_capture;

    std::string error;
    ExpectTrue(SaveLayerCapture(kept, "run-a", kLocalDebugCaptureLabel, &error), "the capture saves: " + error);
    const auto loaded = LoadLayerCapture("run-a", kLocalDebugCaptureLabel, &error);
    ExpectTrue(loaded != nullptr, "the capture reads back: " + error);
    ExpectEq(loaded->Layers().size(), kept.Layers().size(), "every layer reads back");
    ExpectEq(loaded->BytesHeld(), kept.BytesHeld(), "bytes held read back");
    ExpectEq(loaded->SkippedTensors(), kept.SkippedTensors(), "skipped tensors read back");
    ExpectEq(loaded->SkippedBytes(), kept.SkippedBytes(), "skipped bytes read back");
    ExpectEq(loaded->Options().byte_budget, options.byte_budget, "the budget reads back");
    for (size_t l = 0; l < kept.Layers().size(); ++l) {
        const auto& a = kept.Layers()[l];
        const auto& b = loaded->Layers()[l];
        ExpectTrue(a.module_index == b.module_index && a.node_id == b.node_id && a.node_name == b.node_name &&
                       a.tensors.size() == b.tensors.size(),
                   a.node_name + " keeps its module, node and tensors");
        for (size_t t = 0; t < a.tensors.size(); ++t) {
            const auto& x = a.tensors[t];
            const auto& y = b.tensors[t];
            ExpectTrue(x.role == y.role && x.name == y.name && x.tensor.Shape() == y.tensor.Shape() &&
                           HostValues(x.tensor) == HostValues(y.tensor),
                       a.node_name + " " + x.name + " reads back bit for bit");
        }
    }
    ExpectTrue(ListLayerCaptures("run-a") == std::vector<std::string>{kLocalDebugCaptureLabel},
               "the run lists its capture");
    ExpectTrue(EpochCaptureLabel(5) == "epoch_0005", "epoch labels sort by epoch");

    // Not a folder name: refused, nothing written outside the store.
    ExpectTrue(!SaveLayerCapture(kept, "../run", kLocalDebugCaptureLabel, &error), "a run id with a path is refused");
    ExpectTrue(!SaveLayerCapture(kept, "run-a", "", &error), "an empty label is refused");

    // A damaged tensor file: the load says why instead of returning a part.
    const fs::path dir = LayerCaptureDir("run-a", kLocalDebugCaptureLabel);
    fs::path victim;
    for (const auto& entry : fs::directory_iterator(dir))
        if (entry.path().extension() == ".bin") victim = entry.path();
    ExpectTrue(!victim.empty(), "the capture wrote tensor files");
    fs::resize_file(victim, fs::file_size(victim) - 1);
    ExpectTrue(LoadLayerCapture("run-a", kLocalDebugCaptureLabel, &error) == nullptr &&
                   error.find("truncated") != std::string::npos,
               "a truncated tensor file is refused: " + error);
    // Saving again replaces it whole.
    ExpectTrue(SaveLayerCapture(kept, "run-a", kLocalDebugCaptureLabel, &error), "the capture saves again");
    ExpectTrue(LoadLayerCapture("run-a", kLocalDebugCaptureLabel, &error) != nullptr, "the new save reads back");
    // An unfinished save (no capture.json) is not listed.
    fs::create_directories(LayerCaptureDir("run-a", "epoch_0001"));
    ExpectEq(ListLayerCaptures("run-a").size(), 1, "an unfinished save is not listed");

    // Tensor files keep their data type.
    const uint8_t bytes[6] = {0, 1, 2, 250, 254, 255};
    const Tensor u8({2, 3}, bytes, DataType::UInt8);
    ExpectTrue(WriteTensorFile(root / "u8.bin", u8, ArrayFireHostSyncCategory::DebugSampleDump, "test", &error),
               "a uint8 tensor writes: " + error);
    const auto u8_back = ReadTensorFile(root / "u8.bin", &error);
    ExpectTrue(u8_back && u8_back->GetDataType() == DataType::UInt8 && u8_back->Shape() == u8.Shape() &&
                   std::memcmp(u8_back->ReadData(), bytes, sizeof(bytes)) == 0,
               "a uint8 tensor reads back as it was");
    ExpectTrue(!ReadTensorFile(root / "missing.bin", &error) && error.find("unreadable") != std::string::npos,
               "a missing tensor file says so");

    // Only the last runs keep their captures; their folders stay.
    for (const char* run : {"run-b", "run-c"}) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));  // distinct folder times
        ExpectTrue(SaveLayerCapture(kept, run, kLocalDebugCaptureLabel, &error), std::string(run) + " saves");
    }
    ExpectEq(PruneLayerCaptures(2), 1, "one run past the last two is cleared");
    ExpectTrue(ListLayerCaptures("run-a").empty() && fs::exists(GetDebugRunStoreRoot() / "run-a"),
               "the oldest run's captures are gone, its folder stays");
    ExpectEq(ListLayerCaptures("run-b").size() + ListLayerCaptures("run-c").size(), 2, "the last two keep theirs");
    ExpectEq(PruneLayerCaptures(2), 0, "nothing more to clear");

    // 6.3d: runs newest first; a capture's epoch and size; a training run's
    // epochs round trip, and a run without them has none.
    ExpectTrue(ListRunsWithLayerCaptures() == std::vector<std::string>({"run-c", "run-b"}),
               "runs with captures, newest first; a pruned run is not listed");
    ExpectTrue(SaveLayerCapture(kept, "run-c", EpochCaptureLabel(5), &error), "a snapshot saves");
    const auto infos = ListLayerCaptureInfo("run-c");
    ExpectTrue(infos.size() == 2 && infos[0].label == "epoch_0005" && infos[0].epoch == 5 && infos[0].bytes > 0 &&
                   infos[1].label == kLocalDebugCaptureLabel && infos[1].epoch == 0,
               "captures listed with their epoch and size");
    TrainingRunRecord record;
    record.run_id = "run-c";
    record.graph_name = "cats_dogs";
    record.total_epochs = 3;
    record.epochs = {{1, 0.9f, 0.5f, 0.8f, 0.55f}, {2, 0.7f, 0.6f, std::nullopt, std::nullopt}};
    ExpectTrue(SaveTrainingRunRecord(record, &error), "a training record saves: " + error);
    const auto back = LoadTrainingRunRecord("run-c");
    ExpectTrue(back && back->graph_name == "cats_dogs" && back->total_epochs == 3 && !back->regression &&
                   back->epochs.size() == 2 && back->epochs[0].val_metric == 0.55f && !back->epochs[1].val_loss &&
                   back->epochs[1].train_loss == 0.7f,
               "a training record reads back, an epoch without validation stays without");
    ExpectTrue(!LoadTrainingRunRecord("run-b") && !LoadTrainingRunRecord("../run-c"),
               "no record for a Local Debug run or a path outside the store");
    fs::remove_all(root);
    spdlog::info("  OK: {} layers saved and read back, damaged files refused, old runs pruned",
                 kept.Layers().size());
}

void TestDebugExecutorBinaryHeadBackward() {
    spdlog::info("--- TestDebugExecutorBinaryHeadBackward ---");
    auto cfg = MakeTabularConfig();
    cfg.output_size = 1;
    cfg.layers.back().units = 1;
    cfg.layers.back().output_shape = {1};
    cfg.loss_type = gui::NodeType::BCEWithLogits;

    DebugExecutor exe(cfg);
    const auto res = exe.Run();
    ExpectTrue(res.reached == DebugStage::Complete,
               "binary-head Local Debug should complete backward");
    ExpectTrue(res.success,
               "binary-head Local Debug should succeed");
    ExpectTrue(res.loss_target_compatibility.available &&
                   res.loss_target_compatibility.compatible,
               "binary-head prediction/target shapes should be compatible");
    ExpectEq(res.params_with_grad, 4,
             "binary-head parameters with gradients");
    ExpectEq(res.params_missing_grad, 0,
             "binary-head parameters missing gradients");
    spdlog::info("  OK: binary head completes forward/loss/backward/update");
}

void TestDebugExecutorFirstShapeMismatch() {
    spdlog::info("--- TestDebugExecutorFirstShapeMismatch ---");
    auto cfg = MakeTabularConfig();
    cfg.layers[0].output_shape = {31};
    cfg.layers[1].input_shape = {31};

    DebugExecutor exe(cfg);
    auto res = exe.Run();
    ExpectTrue(res.reached == DebugStage::Complete,
               "shape prophecy mismatch should remain observational");
    ExpectEq(res.layer_traces.size(), 3,
             "shape mismatch layer trace count");

    const auto& first = res.layer_traces[0];
    ExpectTrue(first.shape_mismatch == ShapeMismatchKind::Output,
               "first divergence should identify the output prediction");
    ExpectTrue(first.is_first_shape_mismatch,
               "first divergence should be marked once");
    ExpectTrue(first.upstream_node_id == -1,
               "first module should not claim an upstream graph node");

    const auto& second = res.layer_traces[1];
    ExpectTrue(second.shape_mismatch == ShapeMismatchKind::Input,
               "downstream trace should identify its input mismatch");
    ExpectTrue(!second.is_first_shape_mismatch,
               "later divergence should not replace the first one");
    ExpectEq(second.upstream_node_id, 101,
             "downstream mismatch upstream graph node");
    ExpectTrue(second.upstream_node_name == "Input Dense",
               "downstream mismatch upstream graph name");
    spdlog::info("  OK: first shape divergence and upstream node are exact");
}

void TestDebugExecutorSkippedLayerMapping() {
    spdlog::info("--- TestDebugExecutorSkippedLayerMapping ---");
    auto cfg = MakeTabularConfig();

    // A layer type the builder has no module for (Lambda is still
    // a template node). Conv2D used to be that type; since c5433f95 (TOFIX140
    // A1) it builds a real module and needs the spatial head's input shape.
    CompiledLayer unsupported;
    unsupported.type = gui::NodeType::Lambda;
    unsupported.node_id = 99;
    unsupported.name = "Unsupported Lambda";
    cfg.layers.insert(cfg.layers.begin(), unsupported);

    DebugExecutor exe(cfg);
    auto res = exe.Run();
    ExpectTrue(res.reached == DebugStage::Complete,
               "supported modules after a skipped layer should execute");
    ExpectEq(res.model_build_traces.size(), 4,
             "skipped layer model build traces");

    const auto& skipped = res.model_build_traces.front();
    ExpectTrue(skipped.status == "skipped",
               "unsupported compiled layer should emit a skipped build trace");
    ExpectTrue(!skipped.payload["module_created"].get<bool>(),
               "skipped build trace should not claim module creation");
    ExpectTrue(skipped.payload["module_index"].is_null(),
               "skipped build trace should not have a module index");
    ExpectTrue(!skipped.warnings.empty(),
               "skipped build trace should explain the omission");

    ExpectEq(
        res.model_build_traces[1].payload["module_index"].get<size_t>(),
        0,
        "first created module index after skipped layer");
    ExpectEq(
        res.model_build_traces[1].payload["compiled_layer_index"].get<size_t>(),
        1,
        "first created module compiled index after skipped layer");
    ExpectEq(res.layer_traces.front().node_id, 101,
             "forward trace should map to the actual first graph module");
    ExpectEq(res.layer_traces.front().compiled_layer_index, 1,
             "forward trace should retain the non-positional compiled index");
    spdlog::info("  OK: skipped layers cannot shift module trace identity");
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
    bool any_positive_update = false;
    for (const auto& g : res.grad_norms) {
        if (g.layer_index < 0) {
            std::cerr << "FAIL: grad_norms entry missing layer_index for "
                      << g.param_name << "\n";
            std::exit(1);
        }
        assert(g.has_gradient && "golden path grad norm should know the gradient exists");
        ExpectTrue(!g.parameter_shape.empty(),
                   "gradient health should retain parameter shape metadata");
        ExpectTrue(g.gradient_shape == g.parameter_shape,
                   "gradient shape should match its parameter on the golden path");
        ExpectTrue(g.node_id == 101 || g.node_id == 103,
                   "gradient should resolve to its graph layer node");
        ExpectTrue(!g.node_name.empty(),
                   "gradient should preserve graph layer name");
        ExpectTrue(std::isfinite(g.parameter_l2_norm),
                   "parameter norm should be a bounded scalar");
        ExpectTrue(std::isfinite(g.grad_parameter_ratio),
                   "gradient/parameter ratio should be finite");
        ExpectTrue(g.update_observed,
                   "completed optimizer step should expose its update");
        ExpectTrue(std::isfinite(g.update_l2_norm),
                   "update norm should be a bounded scalar");
        ExpectTrue(std::isfinite(g.update_parameter_ratio),
                   "update/parameter ratio should be finite");
        assert(!g.is_nan && "grad norm NaN on random init is a backend bug");
        assert(!g.is_inf && "grad norm Inf on random init is a backend bug");
        ExpectTrue(g.numerics.available,
                   "gradient should expose bounded numerics");
        ExpectNear(static_cast<float>(g.numerics.l2_norm), g.l2_norm,
                   1e-5f, "gradient numeric summary should own the L2 scan");
        if (g.l2_norm > 0.0f) any_positive_norm = true;
        if (g.update_l2_norm > 0.0f) any_positive_update = true;
    }
    assert(any_positive_norm && "at least one grad should have nonzero norm");
    assert(any_positive_update && "at least one parameter should be updated");
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

void TestLossMetricExplainerContract() {
    spdlog::info("--- TestLossMetricExplainerContract ---");
    TrainingConfiguration config;
    config.output_size = 1;
    config.preprocessing.num_classes = 2;
    config.loss_node_id = 12;
    config.loss_type = gui::NodeType::BCEWithLogits;
    config.loss_params = {
        {"pos_weight", "59"},
        {"reduction", "sum"},
    };

    gui::MLNode loss_node;
    loss_node.id = 12;
    loss_node.type = gui::NodeType::BCEWithLogits;
    loss_node.name = "Weighted Binary BCE";
    loss_node.parameters = config.loss_params;

    gui::MLNode metric_node;
    metric_node.id = 13;
    metric_node.type = gui::NodeType::ClassificationMetricsNode;
    metric_node.name = "Class Metrics";
    metric_node.parameters = {
        {"metrics", "accuracy,precision,recall,f1"},
    };

    DebugTraceRecord stale_loss = DebugNodeTraceContract::Make(
        "other-run", 12, loss_node.name, "BCEWithLogits", "Loss",
        DebugTraceRole::Loss, {}, {}, "float32", "LocalDebug", "ok");
    stale_loss.payload["prediction_shape"] = {9, 9};
    stale_loss.payload["target_shape"] = {9, 9};

    DebugTraceRecord observed_loss = DebugNodeTraceContract::Make(
        "loss-explainer-run", 12, loss_node.name, "BCEWithLogits", "Loss",
        DebugTraceRole::Loss, {}, {}, "float32", "LocalDebug", "ok");
    observed_loss.payload["prediction_shape"] = {1, 1};
    observed_loss.payload["target_shape"] = {1, 1};
    observed_loss.payload["prediction_target_shape_check_available"] = true;
    observed_loss.payload["prediction_target_shapes_compatible"] = true;
    observed_loss.payload["prediction_target_shape_reason"] = "exact match";
    observed_loss.payload["loss_finite"] = true;

    const ResolvedLossConfiguration resolved =
        ResolveLossConfiguration(config);
    ExpectTrue(resolved.reduction == Reduction::Sum,
               "shared resolver should expose effective reduction");
    ExpectTrue(resolved.pos_weight.has_value(),
               "shared resolver should expose BCE pos_weight");
    ExpectNear(*resolved.pos_weight, 59.0f, 1e-6f,
               "shared resolver BCE pos_weight");

    DebugLossMetricExplainer explainer;
    const DebugTraceRecord summary = explainer.BuildTrace(
        "loss-explainer-run",
        config,
        {loss_node, metric_node},
        {stale_loss, observed_loss});
    ExpectTrue(summary.phase == "LossMetricExplanation",
               "loss explainer phase should be stable");
    ExpectTrue(summary.node_id == 12,
               "loss explainer should retain selected graph node id");
    ExpectTrue(summary.node_name == "Weighted Binary BCE",
               "loss explainer should retain selected graph node name");
    ExpectTrue(summary.payload["loss_metric_explanation_schema"] ==
                   DebugLossMetricExplainer::kSchema,
               "loss explainer schema should be stable");
    ExpectTrue(summary.payload["tensor_reads_added"] == false,
               "loss explainer should not add Tensor reads");
    ExpectEq(summary.payload["rows"].size(), 3,
             "selected loss, runtime policy, configured metric rows");

    const auto& loss = summary.payload["rows"][0];
    ExpectTrue(loss["node_id"] == 12,
               "selected loss row should retain node id");
    ExpectTrue(loss["reduction"] == "sum",
               "selected loss row should expose effective reduction");
    ExpectNear(loss["pos_weight"].get<float>(), 59.0f, 1e-6f,
               "selected loss row pos_weight");
    ExpectTrue(loss["class_count"] == 2,
               "binary loss should report two decision classes");
    ExpectTrue(loss["actual_prediction_shape"] ==
                   nlohmann::json::array({1, 1}),
               "same-run actual prediction shape should be used");
    ExpectTrue(loss["shapes_compatible"] == true,
               "shape compatibility should be retained");

    const auto& policy = summary.payload["rows"][1];
    ExpectTrue(policy["threshold"] == 0.0f &&
                   policy["equivalent_probability_threshold"] == 0.5f,
               "BCEWithLogits policy should expose logit threshold truth");
    const auto& metric = summary.payload["rows"][2];
    ExpectTrue(metric["evidence_state"] == "configured_only" &&
                   metric["result_observed"] == false,
               "graph metric must not be presented as executed evidence");

    std::vector<gui::MLNode> many_nodes = {loss_node};
    for (size_t i = 0; i < DebugLossMetricExplainer::kMaxRows + 10; ++i) {
        gui::MLNode node;
        node.id = 100 + static_cast<int>(i);
        node.type = gui::NodeType::RegressionMetricsNode;
        node.name = "Metric " + std::to_string(i);
        many_nodes.push_back(std::move(node));
    }
    const auto bounded = explainer.BuildTrace(
        "loss-explainer-run", config, many_nodes, {observed_loss});
    ExpectEq(bounded.payload["rows"].size(),
             DebugLossMetricExplainer::kMaxRows,
             "loss explainer rows should be bounded");
    ExpectTrue(bounded.payload["rows_truncated"] == true,
               "loss explainer should report row truncation");
    spdlog::info("  OK: loss/metric explanation is truthful and bounded");
}

void TestOptimizerConfigurationBuildContract() {
    spdlog::info("--- TestOptimizerConfigurationBuildContract ---");

    TrainingConfiguration config;
    config.learning_rate = 0.0123f;
    config.momentum = 0.42f;
    config.beta1 = 0.81f;
    config.beta2 = 0.93f;
    config.epsilon = 2e-7f;
    config.rmsprop_alpha = 0.88f;
    config.weight_decay = 0.07f;

    const auto check_learning_rate = [&](const Optimizer& optimizer) {
        ExpectNear(static_cast<float>(optimizer.GetLearningRate()),
                   config.learning_rate, 1e-7f,
                   "configured optimizer learning rate");
    };

    config.optimizer_type = gui::NodeType::SGD;
    auto resolved = ResolveOptimizerConfiguration(config);
    auto optimizer = BuildOptimizerFromConfig(config);
    ExpectTrue(dynamic_cast<SGDOptimizer*>(optimizer.get()) != nullptr,
               "SGD config should build SGDOptimizer");
    ExpectTrue(resolved.momentum.has_value(),
               "SGD resolver should retain momentum");
    ExpectNear(*resolved.momentum, config.momentum, 1e-7f,
               "SGD resolved momentum");
    check_learning_rate(*optimizer);

    config.optimizer_type = gui::NodeType::Adam;
    resolved = ResolveOptimizerConfiguration(config);
    optimizer = BuildOptimizerFromConfig(config);
    ExpectTrue(dynamic_cast<AdamOptimizer*>(optimizer.get()) != nullptr,
               "Adam config should build AdamOptimizer");
    ExpectNear(*resolved.beta1, config.beta1, 1e-7f,
               "Adam resolved beta1");
    ExpectNear(*resolved.beta2, config.beta2, 1e-7f,
               "Adam resolved beta2");
    ExpectNear(*resolved.epsilon, config.epsilon, 1e-10f,
               "Adam resolved epsilon");

    config.optimizer_type = gui::NodeType::AdamW;
    resolved = ResolveOptimizerConfiguration(config);
    optimizer = BuildOptimizerFromConfig(config);
    ExpectTrue(dynamic_cast<AdamWOptimizer*>(optimizer.get()) != nullptr,
               "AdamW config should build AdamWOptimizer");
    ExpectNear(*resolved.weight_decay, config.weight_decay, 1e-7f,
               "AdamW resolved weight decay");

    config.optimizer_type = gui::NodeType::RMSprop;
    resolved = ResolveOptimizerConfiguration(config);
    optimizer = BuildOptimizerFromConfig(config);
    ExpectTrue(dynamic_cast<RMSpropOptimizer*>(optimizer.get()) != nullptr,
               "RMSprop config should build RMSpropOptimizer");
    ExpectNear(*resolved.rmsprop_alpha, config.rmsprop_alpha, 1e-7f,
               "RMSprop resolved alpha");
    ExpectNear(*resolved.momentum, config.momentum, 1e-7f,
               "RMSprop resolved momentum");

    config.optimizer_type = gui::NodeType::Adagrad;
    resolved = ResolveOptimizerConfiguration(config);
    optimizer = BuildOptimizerFromConfig(config);
    ExpectTrue(dynamic_cast<AdaGradOptimizer*>(optimizer.get()) != nullptr,
               "Adagrad config should build AdaGradOptimizer");
    ExpectNear(*resolved.epsilon, config.epsilon, 1e-10f,
               "Adagrad resolved epsilon");

    config.optimizer_type = gui::NodeType::NAdam;
    resolved = ResolveOptimizerConfiguration(config);
    optimizer = BuildOptimizerFromConfig(config);
    ExpectTrue(dynamic_cast<NAdamOptimizer*>(optimizer.get()) != nullptr,
               "NAdam config should build NAdamOptimizer");
    ExpectNear(*resolved.beta1, config.beta1, 1e-7f,
               "NAdam resolved beta1");
    ExpectNear(*resolved.beta2, config.beta2, 1e-7f,
               "NAdam resolved beta2");
    ExpectNear(*resolved.epsilon, config.epsilon, 1e-10f,
               "NAdam resolved epsilon");

    spdlog::info("  OK: all optimizer settings reach the backend constructor path");
}

} // namespace

int main(int argc, char** argv) {
    spdlog::set_level(spdlog::level::info);
    try {
        if (argc == 2 &&
            std::string(argv[1]) == "--checkpoint-serialization-only") {
            TestCheckpointTensorSerializationCompatibility();
            std::cout << "Checkpoint serialization compatibility passed\n";
            return 0;
        }

        const bool run_slow_sentiment =
            std::getenv("CYXWIZ_RUN_SLOW_DEBUG_TESTS") != nullptr;

        TestBuildSequentialTabular();
        TestDenseActivationConfigurationContract();
        TestRecurrentConfigurationFailClosed();
        TestTransformerConfigurationFailClosed();
        TestBuildSequentialLayerNorm();
        TestNormalizationRegularizationConfigurationContract();
        TestBuildSequentialMultiHeadAttention();
        TestLayerNormCheckpointRoundTrip();
        TestCheckpointTensorSerializationCompatibility();
        TestBatchNormCheckpointRoundTrip();
        TestCheckpointRejectsIncompatibleModelWithoutMutation();
        TestCheckpointFormatCapabilitiesFailClosed();
        TestCheckpointManifestV2AtomicContract();
        TestCheckpointPayloadV2RoundTripAndCorruptionGuard();
        TestTransformerDecoderCheckpointRoundTrip();
        TestSyntheticBatchTabular();
        TestSyntheticBatchText();
        TestMseLossLabelsAreFloat();
        TestBuildSequentialTextEmbeddingWeights();
        TestEmbeddingWeightsWordVectorFiles();
        TestSequenceProjectionConfigurationFailClosed();
        TestCrossEntropyIgnoreIndexFromLossParams();
        TestWeightedLossConfigParams();
        TestCrossEntropyTokenShapeIgnoreIndex();
        TestWeightedCrossEntropyBackend();
        TestBCEWithLogitsPosWeightBackend();
        TestTimeDistributedDenseModule();
        TestBuildSequentialTimeDistributedHead();
        TestDebugNumericsContract();
        TestDebugNumericsMatchHostReference();
        TestLayerTensorReader();
        TestDebugExecutorGoldenPath();
        TestDebugExecutorResNetExample();
        TestDebugExecutorInputSample();
        TestLayerCaptureTabular();
        TestLayerCaptureResNetExample();
        TestCaptureStepLeavesTheModel();
        TestLayerCaptureStore();
        TestLayerBackwardObserver();
        TestDebugModelSession();
        TestOcclusion();
        TestSaliencyAgainstPyTorch();
        TestTrainingSnapshotsPresentation();
        TestTopImages();
        TestContributions();
        TestLayerEmbedding();
        TestUnitWeights();
        TestWhereItLooksPresentation();
        TestDebugExecutorBinaryHeadBackward();
        TestDebugExecutorFirstShapeMismatch();
        TestDebugExecutorSkippedLayerMapping();
        TestDebugExecutorGradNormBookkeeping();
        TestDebugExecutorTextGraph();
        TestLossMetricExplainerContract();
        TestOptimizerConfigurationBuildContract();
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
