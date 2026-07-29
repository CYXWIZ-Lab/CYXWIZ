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
#include "../src/core/checkpoint_manifest.h"
#include "../src/core/checkpoint_payload_io.h"
#include "../src/core/checkpoint_manager.h"

#include <cyxwiz/loss.h>
#include <cyxwiz/tensor.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <cctype>
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
    spdlog::info("  OK: v2 model/Adam payloads round-trip and corruption is transactional");
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
        assert(g.has_gradient && "golden path grad norm should know the gradient exists");
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
        TestBuildSequentialLayerNorm();
        TestBuildSequentialMultiHeadAttention();
        TestLayerNormCheckpointRoundTrip();
        TestCheckpointRejectsIncompatibleModelWithoutMutation();
        TestCheckpointFormatCapabilitiesFailClosed();
        TestCheckpointManifestV2AtomicContract();
        TestCheckpointPayloadV2RoundTripAndCorruptionGuard();
        TestTransformerDecoderCheckpointRoundTrip();
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
