#include "../src/core/graph_compiler.h"
#include "../src/core/backend_placement_capabilities.h"
#include "../src/core/execution_placement_plan.h"
#include "../src/gui/loaders/data_loader.h"
#include "../../cyxwiz-backend/src/algorithms/arrayfire_backend_utils.h"
#include "cyxwiz/backend_placement_observation.h"
#include "cyxwiz/recurrent_cuda_placement.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::loaders {

DataLoader* GetByCategory(FileCategory) {
    return nullptr;
}

DataLoader* GetByRegisteredDataset(const std::string&) {
    return nullptr;
}

FileCategory FileCategoryFromString(const std::string&) {
    return FileCategory::Tabular;
}

} // namespace cyxwiz::loaders

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

gui::NodePin Pin(int id,
                 gui::PinType type,
                 const std::string& name,
                 bool is_input,
                 bool required = true) {
    gui::NodePin pin;
    pin.id = id;
    pin.type = type;
    pin.name = name;
    pin.is_input = is_input;
    pin.is_required = required;
    return pin;
}

gui::MLNode Node(int id,
                 gui::NodeType type,
                 const std::string& name,
                 std::vector<gui::NodePin> inputs,
                 std::vector<gui::NodePin> outputs) {
    gui::MLNode node;
    node.id = id;
    node.type = type;
    node.name = name;
    node.inputs = std::move(inputs);
    node.outputs = std::move(outputs);
    return node;
}

gui::NodeLink Link(int id, int from_node, int from_pin, int to_node, int to_pin) {
    gui::NodeLink link;
    link.id = id;
    link.from_node = from_node;
    link.from_pin = from_pin;
    link.to_node = to_node;
    link.to_pin = to_pin;
    return link;
}

bool HasWarningText(const cyxwiz::TrainingConfiguration& config,
                    const std::string& text) {
    for (const auto& issue : config.issues) {
        if (issue.level == cyxwiz::IssueLevel::Warning &&
            issue.message.find(text) != std::string::npos) {
            return true;
        }
    }
    return false;
}

const cyxwiz::BackendPlacementEntry* FindPlacement(
    const cyxwiz::TrainingConfiguration& config,
    int node_id) {
    for (const auto& placement : config.backend_placements) {
        if (placement.node_id == node_id) {
            return &placement;
        }
    }
    return nullptr;
}

cyxwiz::CompiledLayer TensorLayer(gui::NodeType type,
                                  int node_id,
                                  std::vector<size_t> input_shape,
                                  std::vector<size_t> output_shape) {
    cyxwiz::CompiledLayer layer;
    layer.type = type;
    layer.node_id = node_id;
    layer.input_shape = std::move(input_shape);
    layer.output_shape = std::move(output_shape);
    return layer;
}

cyxwiz::TrainingConfiguration CompileRecurrentGraph(gui::NodeType recurrent_type,
                                                    int hidden_size,
                                                    bool bidirectional,
                                                    const std::string& placement_cache_path = {}) {
    auto data = Node(1,
                     gui::NodeType::DataInput,
                     "Data",
                     {},
                     {Pin(101, gui::PinType::Tensor, "Data", false),
                      Pin(102, gui::PinType::Labels, "Labels", false)});
    data.parameters["dataset_name"] = "placement_test_dataset";
    data.parameters["data_loaded"] = "true";
    data.parameters["file_category"] = "tabular";
    data.parameters["label_column"] = "label";
    data.parameters["shape"] = "[64]";

    auto loader = Node(2,
                       gui::NodeType::DataLoader,
                       "Loader",
                       {Pin(201, gui::PinType::Tensor, "Data", true),
                        Pin(202, gui::PinType::Labels, "Labels", true)},
                       {Pin(203, gui::PinType::Tensor, "Data", false),
                        Pin(204, gui::PinType::Labels, "Labels", false)});
    loader.parameters["batch_size"] = "64";
    loader.parameters["epochs"] = "1";

    auto embedding = Node(3,
                          gui::NodeType::Embedding,
                          "Embedding",
                          {Pin(301, gui::PinType::Tensor, "Indices", true)},
                          {Pin(302, gui::PinType::Tensor, "Embeddings", false)});
    embedding.parameters["num_embeddings"] = "1000";
    embedding.parameters["embedding_dim"] = "64";

    auto recurrent = Node(4,
                          recurrent_type,
                          recurrent_type == gui::NodeType::GRU ? "GRU 32" : "LSTM 8",
                          {Pin(401, gui::PinType::Tensor, "Input", true)},
                          {Pin(402, gui::PinType::Tensor, "Output", false),
                           Pin(403, gui::PinType::Tensor, "Hidden", false, false)});
    recurrent.parameters["input_size"] = "64";
    recurrent.parameters["hidden_size"] = std::to_string(hidden_size);
    recurrent.parameters["num_layers"] = "1";
    recurrent.parameters["bidirectional"] = bidirectional ? "true" : "false";
    recurrent.parameters["return_sequences"] = "false";

    auto dense = Node(5,
                      gui::NodeType::Dense,
                      "Classifier",
                      {Pin(501, gui::PinType::Tensor, "Input", true)},
                      {Pin(502, gui::PinType::Tensor, "Output", false)});
    dense.parameters["units"] = "2";

    auto loss = Node(6,
                     gui::NodeType::CrossEntropyLoss,
                     "Loss",
                     {Pin(601, gui::PinType::Tensor, "Predictions", true),
                      Pin(602, gui::PinType::Labels, "Targets", true)},
                     {Pin(603, gui::PinType::Loss, "Loss", false)});

    auto optimizer = Node(7,
                          gui::NodeType::Adam,
                          "Adam",
                          {Pin(701, gui::PinType::Loss, "Loss", true)},
                          {});

    std::vector<gui::MLNode> nodes = {
        data, loader, embedding, recurrent, dense, loss, optimizer,
    };
    std::vector<gui::NodeLink> links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 1, 102, 2, 202),
        Link(3, 2, 203, 3, 301),
        Link(4, 3, 302, 4, 401),
        Link(5, 4, 402, 5, 501),
        Link(6, 5, 502, 6, 601),
        Link(7, 2, 204, 6, 602),
        Link(8, 6, 603, 7, 701),
    };

    cyxwiz::GraphCompiler compiler;
    return compiler.Compile(nodes, links, true, placement_cache_path);
}

cyxwiz::TrainingConfiguration CompileUnclassifiedLayerGraph() {
    auto data = Node(1,
                     gui::NodeType::DataInput,
                     "Sequence Data",
                     {},
                     {Pin(101, gui::PinType::Tensor, "Data", false),
                      Pin(102, gui::PinType::Labels, "Labels", false)});
    data.parameters["dataset_name"] = "placement_test_dataset";
    data.parameters["data_loaded"] = "true";
    data.parameters["file_category"] = "tabular";
    data.parameters["label_column"] = "label";
    data.parameters["shape"] = "[8, 4]";

    auto time_distributed = Node(
        8,
        gui::NodeType::TimeDistributed,
        "Token Head",
        {Pin(801, gui::PinType::Tensor, "Input", true)},
        {Pin(802, gui::PinType::Tensor, "Output", false)});
    time_distributed.parameters["units"] = "2";

    auto loss = Node(6,
                     gui::NodeType::CrossEntropyLoss,
                     "Loss",
                     {Pin(601, gui::PinType::Tensor, "Predictions", true),
                      Pin(602, gui::PinType::Labels, "Targets", true)},
                     {Pin(603, gui::PinType::Loss, "Loss", false)});

    auto optimizer = Node(7,
                          gui::NodeType::Adam,
                          "Adam",
                          {Pin(701, gui::PinType::Loss, "Loss", true)},
                          {});

    std::vector<gui::MLNode> nodes = {
        data, time_distributed, loss, optimizer,
    };
    std::vector<gui::NodeLink> links = {
        Link(1, 1, 101, 8, 801),
        Link(2, 8, 802, 6, 601),
        Link(3, 1, 102, 6, 602),
        Link(4, 6, 603, 7, 701),
    };

    cyxwiz::GraphCompiler compiler;
    return compiler.Compile(nodes, links, true);
}

} // namespace

int main() {
    Check(cyxwiz::ClassifyArrayFireBackendFallbackReason(
              "Formal parameter space overflowed") ==
              cyxwiz::BackendFallbackReason::CudaJitParamOverflow,
          "fallback classifier should preserve CUDA formal-parameter overflow");
    Check(cyxwiz::ClassifyArrayFireBackendFallbackReason(
              "CUDA_ERROR_MEMORY_ALLOCATION: out of memory") ==
              cyxwiz::BackendFallbackReason::GpuOutOfMemory,
          "fallback classifier should detect GPU out-of-memory");
    Check(cyxwiz::ClassifyArrayFireBackendFallbackReason(
              "unsupported dtype for this ArrayFire kernel") ==
              cyxwiz::BackendFallbackReason::UnsupportedDtype,
          "fallback classifier should detect unsupported dtype");
    Check(cyxwiz::ClassifyArrayFireBackendFallbackReason(
              "invalid dimension / shape not supported") ==
              cyxwiz::BackendFallbackReason::UnsupportedShape,
          "fallback classifier should detect unsupported shape");
    Check(cyxwiz::ClassifyArrayFireBackendFallbackReason(
              "NVRTC compile timeout while building generated kernel") ==
              cyxwiz::BackendFallbackReason::BackendCompileTimeout,
          "fallback classifier should detect backend compile timeout");
    Check(cyxwiz::ClassifyArrayFireBackendFallbackReason(
              "NVRTC compilation failed") ==
              cyxwiz::BackendFallbackReason::ArrayFireJitCompileFailure,
          "fallback classifier should detect generic JIT compile failures");
    Check(cyxwiz::ClassifyArrayFireBackendFallbackReason(
              "driver returned an unexpected backend error") ==
              cyxwiz::BackendFallbackReason::BackendInternalError,
          "fallback classifier should classify unknown backend errors");
    Check(std::string(cyxwiz::BackendFallbackReasonName(
              cyxwiz::BackendFallbackReason::BackendInternalError)) ==
              cyxwiz::BackendPlacementObservationReason::BackendInternalError,
          "fallback reason names should align with observation reason codes");
    Check(std::string(cyxwiz::BackendPlacementProbeOutcomeName(
              cyxwiz::BackendPlacementProbeOutcome::Unsupported)) ==
              "unsupported",
          "probe outcome names should expose stable strings");
    Check(std::string(cyxwiz::BackendPlacementProbeOutcomeName(
              cyxwiz::BackendPlacementProbeOutcome::Timeout)) ==
              "timeout",
          "timeout probe outcome name should expose a stable string");
    cyxwiz::ClearBackendPlacementObservationCacheForTesting();
    cyxwiz::RecurrentCudaPlacementRequest timeout_lstm_probe;
    timeout_lstm_probe.kind = cyxwiz::RecurrentLayerKind::LSTM;
    timeout_lstm_probe.batch_size = 16;
    timeout_lstm_probe.seq_len = 8;
    timeout_lstm_probe.input_size = 4;
    timeout_lstm_probe.hidden_size = 4;
    timeout_lstm_probe.deep_preflight = true;
    timeout_lstm_probe.preflight_timeout_ms = 0;
    cyxwiz::BackendPlacementObservation timeout_observation;
    Check(cyxwiz::TryRunRecurrentCudaPreflightProbe(
              timeout_lstm_probe,
              timeout_observation),
          "zero-budget preflight timeout should surface through legacy wrapper");
    Check(timeout_observation.reason_code ==
              cyxwiz::BackendPlacementObservationReason::BackendCompileTimeout,
          "zero-budget preflight timeout should record timeout reason");
    Check(timeout_observation.source ==
              cyxwiz::BackendPlacementObservationSource::PreflightProbe,
          "zero-budget preflight timeout should record preflight source");
    Check(timeout_observation.probe_outcome == "timeout",
          "zero-budget preflight timeout should record timeout outcome");
    Check(timeout_observation.probe_scope ==
              cyxwiz::BackendPlacementProbeScope::DeepPreflight,
          "zero-budget deep preflight should record deep preflight scope");
    cyxwiz::ClearBackendPlacementObservationCacheForTesting();
    cyxwiz::RecurrentCudaPlacementRequest unsupported_bidirectional_gru_probe;
    unsupported_bidirectional_gru_probe.kind = cyxwiz::RecurrentLayerKind::GRU;
    unsupported_bidirectional_gru_probe.batch_size = 16;
    unsupported_bidirectional_gru_probe.seq_len = 8;
    unsupported_bidirectional_gru_probe.input_size = 4;
    unsupported_bidirectional_gru_probe.hidden_size = 4;
    unsupported_bidirectional_gru_probe.bidirectional = true;
    const auto gru_probe_result = cyxwiz::RunRecurrentCudaPreflightProbe(
        unsupported_bidirectional_gru_probe);
    Check(gru_probe_result.outcome ==
              cyxwiz::BackendPlacementProbeOutcome::Unsupported,
          "bidirectional GRU preflight should remain unsupported");
    Check(gru_probe_result.reason_code ==
              cyxwiz::BackendPlacementObservationReason::UnsupportedShape,
          "unsupported GRU preflight should use a structured reason");
    Check(!gru_probe_result.has_observation,
          "unsupported GRU preflight should not create a failure observation");
    cyxwiz::ClearBackendPlacementObservationCacheForTesting();
    cyxwiz::RecordBackendPlacementObservationForActiveDevice(
        "Dense",
        "cuda",
        "float32",
        cyxwiz::BuildDensePlacementShapeSignature({64, 128}, 2),
        cyxwiz::BackendPlacementObservationReason::GpuOutOfMemory,
        cyxwiz::BackendPlacementObservationSource::RuntimeFallback,
        "dense matmul allocation failed");
    cyxwiz::BackendPlacementObservation generic_observation;
    Check(cyxwiz::TryGetBackendPlacementObservationForActiveDevice(
              "Dense",
              "cuda",
              "float32",
              cyxwiz::BuildDensePlacementShapeSignature({64, 128}, 2),
              generic_observation),
          "generic fallback observation should be retrievable by active device");
    Check(generic_observation.reason_code ==
              cyxwiz::BackendPlacementObservationReason::GpuOutOfMemory,
          "generic fallback observation should preserve structured reason");
    Check(generic_observation.source ==
              cyxwiz::BackendPlacementObservationSource::RuntimeFallback,
          "generic fallback observation should preserve source");
    const auto generic_snapshot =
        cyxwiz::SnapshotBackendPlacementObservations();
    Check(generic_snapshot.size() == 1,
          "placement observation snapshot should expose recorded observations");
    Check(generic_snapshot.front().op_type == "Dense",
          "placement observation snapshot should preserve op type");
    Check(generic_snapshot.front().reason_code ==
              cyxwiz::BackendPlacementObservationReason::GpuOutOfMemory,
          "placement observation snapshot should preserve reason code");
    Check(generic_snapshot.front().source ==
              cyxwiz::BackendPlacementObservationSource::RuntimeFallback,
          "placement observation snapshot should preserve source");
    Check(generic_snapshot.front().detail ==
              "dense matmul allocation failed",
          "placement observation snapshot should preserve detail text");

    const std::string embedding_signature =
        cyxwiz::BuildEmbeddingPlacementShapeSignature(500, 32, {16}, "int32");
    Check(embedding_signature.find("num_embeddings=500") != std::string::npos,
          "embedding signature should include vocabulary size");
    Check(embedding_signature.find("embedding_dim=32") != std::string::npos,
          "embedding signature should include embedding dimension");
    Check(embedding_signature.find("input_rank=1") != std::string::npos,
          "embedding signature should include input rank");
    Check(embedding_signature.find("index_dtype=int32") != std::string::npos,
          "embedding signature should include index dtype");
    const std::string activation_signature =
        cyxwiz::BuildActivationPlacementShapeSignature({4, 8}, "float32");
    Check(activation_signature == "input=[4x8];dtype=float32",
          "activation signature should include input shape and dtype");
    const std::string linear_signature =
        cyxwiz::BuildLinearPlacementShapeSignature(
            {4, 8}, {2, 8}, {4, 2}, "float32", true);
    Check(linear_signature ==
              "lhs=[4x8];rhs=[2x8];output=[4x2];dtype=float32;bias=true",
          "linear signature should include lhs, rhs, output, dtype, and bias");

    cyxwiz::ClearBackendPlacementObservationCacheForTesting();
    cyxwiz::RecordBackendPlacementObservationForActiveDevice(
        "Linear",
        "cuda",
        "float32",
        linear_signature,
        cyxwiz::BackendPlacementObservationReason::BackendInternalError,
        cyxwiz::BackendPlacementObservationSource::RuntimeFallback,
        "simulated Linear fallback observation");
    cyxwiz::BackendPlacementObservation linear_observation;
    Check(cyxwiz::TryGetBackendPlacementObservationForActiveDevice(
              "Linear",
              "cuda",
              "float32",
              linear_signature,
              linear_observation),
          "Linear fallback observation should be retrievable by active device");
    Check(linear_observation.reason_code ==
              cyxwiz::BackendPlacementObservationReason::BackendInternalError,
          "Linear fallback observation should preserve structured reason");
    const std::string loss_signature =
        cyxwiz::BuildLossPlacementShapeSignature(
            {4, 3}, {4}, "mean", "float32");
    Check(loss_signature ==
              "prediction=[4x3];target=[4];reduction=mean;dtype=float32",
          "loss signature should include prediction, target, reduction, and dtype");
    const std::string tensor_op_signature =
        cyxwiz::BuildTensorOpPlacementShapeSignature(
            {{2, 3}, {2, 3}}, {2, 6}, "float32", "dim=1");
    Check(tensor_op_signature ==
              "inputs=[[2x3],[2x3]];output=[2x6];dtype=float32;dim=1",
          "tensor-op signature should include inputs, output, dtype, and attributes");

    cyxwiz::ClearBackendPlacementObservationCacheForTesting();
    cyxwiz::RecordBackendPlacementObservationForActiveDevice(
        "CrossEntropyLoss::Forward",
        "cuda",
        "float32",
        loss_signature,
        cyxwiz::BackendPlacementObservationReason::ArrayFireJitCompileFailure,
        cyxwiz::BackendPlacementObservationSource::RuntimeFallback,
        "simulated loss fallback observation");
    cyxwiz::BackendPlacementObservation loss_observation;
    Check(cyxwiz::TryGetBackendPlacementObservationForActiveDevice(
              "CrossEntropyLoss::Forward",
              "cuda",
              "float32",
              loss_signature,
              loss_observation),
          "Loss fallback observation should be retrievable by active device");
    Check(loss_observation.reason_code ==
              cyxwiz::BackendPlacementObservationReason::ArrayFireJitCompileFailure,
          "Loss fallback observation should preserve structured reason");
    cyxwiz::ClearBackendPlacementObservationCacheForTesting();
    cyxwiz::RecordBackendPlacementObservationForActiveDevice(
        "Tensor::Cat",
        "cuda",
        "float32",
        tensor_op_signature,
        cyxwiz::BackendPlacementObservationReason::BackendInternalError,
        cyxwiz::BackendPlacementObservationSource::RuntimeFallback,
        "simulated tensor concat fallback observation");
    cyxwiz::BackendPlacementObservation tensor_op_observation;
    Check(cyxwiz::TryGetBackendPlacementObservationForActiveDevice(
              "Tensor::Cat",
              "cuda",
              "float32",
              tensor_op_signature,
              tensor_op_observation),
          "Tensor op fallback observation should be retrievable by active device");
    Check(tensor_op_observation.reason_code ==
              cyxwiz::BackendPlacementObservationReason::BackendInternalError,
          "Tensor op fallback observation should preserve structured reason");
    Check(!tensor_op_observation.timestamp.empty(),
          "Tensor op fallback observation should preserve timestamp metadata");
    const auto cache_path = std::filesystem::temp_directory_path() /
        "cyxwiz_backend_placement_cache_test.json";
    std::string cache_error;
    Check(cyxwiz::SaveBackendPlacementObservationCache(
              cache_path.string(),
              &cache_error),
          "Placement observation cache should save: " + cache_error);
    cyxwiz::ClearBackendPlacementObservationCacheForTesting();
    Check(!cyxwiz::TryGetBackendPlacementObservationForActiveDevice(
              "Tensor::Cat",
              "cuda",
              "float32",
              tensor_op_signature,
              tensor_op_observation),
          "Cleared placement cache should not return saved observation before load");
    cache_error.clear();
    Check(cyxwiz::LoadBackendPlacementObservationCache(
              cache_path.string(),
              &cache_error),
          "Placement observation cache should load: " + cache_error);
    Check(cyxwiz::TryGetBackendPlacementObservationForActiveDevice(
              "Tensor::Cat",
              "cuda",
              "float32",
              tensor_op_signature,
              tensor_op_observation),
          "Loaded placement cache should restore active-device observation");
    Check(tensor_op_observation.source ==
              cyxwiz::BackendPlacementObservationSource::RuntimeFallback,
          "Loaded placement cache should preserve observation source");
    std::error_code remove_error;
    std::filesystem::remove(cache_path, remove_error);

    cyxwiz::ClearBackendPlacementObservationCacheForTesting();
    auto embedding_layer = TensorLayer(
        gui::NodeType::Embedding, 30, {16}, {16, 32});
    embedding_layer.parameters["num_embeddings"] = "500";
    embedding_layer.parameters["embedding_dim"] = "32";
    cyxwiz::RecordBackendPlacementObservationForActiveDevice(
        "Embedding",
        "cuda",
        "int32",
        cyxwiz::BuildEmbeddingPlacementShapeSignature(500, 32, {16}, "int32"),
        cyxwiz::BackendPlacementObservationReason::GpuBackendException,
        cyxwiz::BackendPlacementObservationSource::RuntimeFallback,
        "simulated Embedding fallback observation");
    const auto embedding_cached_placement =
        cyxwiz::backend_placement::BuildArrayFireTensorPlacement(
            embedding_layer);
    Check(embedding_cached_placement.status == cyxwiz::BackendPlacementStatus::Cpu,
          "cached Embedding fallback should route exact shape to CPU");
    Check(embedding_cached_placement.reason_code ==
              cyxwiz::BackendPlacementObservationReason::GpuBackendException,
          "cached Embedding fallback should preserve structured reason");

    cyxwiz::ClearBackendPlacementObservationCacheForTesting();
    auto relu_layer = TensorLayer(gui::NodeType::ReLU, 31, {4, 8}, {4, 8});
    cyxwiz::RecordBackendPlacementObservationForActiveDevice(
        "ReLU",
        "cuda",
        "float32",
        cyxwiz::BuildActivationPlacementShapeSignature({4, 8}, "float32"),
        cyxwiz::BackendPlacementObservationReason::ArrayFireJitCompileFailure,
        cyxwiz::BackendPlacementObservationSource::RuntimeFallback,
        "simulated ReLU fallback observation");
    const auto relu_cached_placement =
        cyxwiz::backend_placement::BuildArrayFireTensorPlacement(relu_layer);
    Check(relu_cached_placement.status == cyxwiz::BackendPlacementStatus::Cpu,
          "cached ReLU fallback should route exact activation shape to CPU");

    cyxwiz::ClearBackendPlacementObservationCacheForTesting();
    for (const auto activation_case : {
             std::pair<gui::NodeType, const char*>{gui::NodeType::Sigmoid, "Sigmoid"},
             std::pair<gui::NodeType, const char*>{gui::NodeType::Tanh, "Tanh"}}) {
        cyxwiz::RecordBackendPlacementObservationForActiveDevice(
            activation_case.second,
            "cuda",
            "float32",
            cyxwiz::BuildActivationPlacementShapeSignature({4, 8}, "float32"),
            cyxwiz::BackendPlacementObservationReason::BackendInternalError,
            cyxwiz::BackendPlacementObservationSource::RuntimeFallback,
            "simulated activation fallback observation");
        const auto placement =
            cyxwiz::backend_placement::BuildArrayFireTensorPlacement(
                TensorLayer(activation_case.first, 32, {4, 8}, {4, 8}));
        Check(placement.status == cyxwiz::BackendPlacementStatus::Cpu,
              std::string("cached ") + activation_case.second +
                  " fallback should route exact activation shape to CPU");
        cyxwiz::ClearBackendPlacementObservationCacheForTesting();
    }

    Check(cyxwiz::backend_placement::IsKnownArrayFireTensorLayer(
              gui::NodeType::Embedding),
          "Embedding should be classified as ArrayFire tensor-capable");
    Check(!cyxwiz::backend_placement::IsKnownArrayFireTensorLayer(
              gui::NodeType::TimeDistributed),
          "TimeDistributed should not be classified as direct ArrayFire tensor-capable");
    Check(cyxwiz::backend_placement::IsRecurrentLayer(gui::NodeType::GRU),
          "GRU should be classified as recurrent placement layer");
    Check(cyxwiz::backend_placement::ClassifyLayer(gui::NodeType::Embedding).kind ==
              cyxwiz::backend_placement::LayerCapabilityKind::ArrayFireTensor,
          "Embedding capability kind should be ArrayFireTensor");
    Check(cyxwiz::backend_placement::ClassifyLayer(gui::NodeType::GRU).kind ==
              cyxwiz::backend_placement::LayerCapabilityKind::Recurrent,
          "GRU capability kind should be Recurrent");
    Check(!cyxwiz::backend_placement::IsKnownArrayFireTensorLayer(
              gui::NodeType::LayerNorm),
          "LayerNorm should not be classified as direct ArrayFire tensor-capable");
    Check(cyxwiz::backend_placement::IsKnownCpuBackedModelLayer(
              gui::NodeType::LayerNorm),
          "LayerNorm should be classified as a known CPU-backed model layer");
    Check(cyxwiz::backend_placement::ClassifyLayer(gui::NodeType::LayerNorm).kind ==
              cyxwiz::backend_placement::LayerCapabilityKind::CpuBackedModelLayer,
          "LayerNorm capability kind should be CpuBackedModelLayer");
    Check(!cyxwiz::backend_placement::IsKnownArrayFireTensorLayer(
              gui::NodeType::MultiHeadAttention),
          "MultiHeadAttention should not be classified as direct ArrayFire tensor-capable");
    Check(cyxwiz::backend_placement::IsKnownCpuBackedModelLayer(
              gui::NodeType::MultiHeadAttention),
          "MultiHeadAttention should be classified as a known CPU-backed model layer");
    Check(cyxwiz::backend_placement::ClassifyLayer(
              gui::NodeType::MultiHeadAttention).kind ==
              cyxwiz::backend_placement::LayerCapabilityKind::CpuBackedModelLayer,
          "MultiHeadAttention capability kind should be CpuBackedModelLayer");
    Check(!cyxwiz::backend_placement::IsKnownArrayFireTensorLayer(
              gui::NodeType::TransformerEncoder),
          "TransformerEncoder should not be classified as direct ArrayFire tensor-capable");
    Check(cyxwiz::backend_placement::IsKnownCpuBackedModelLayer(
              gui::NodeType::TransformerEncoder),
          "TransformerEncoder should be classified as a known CPU-backed model layer");
    Check(cyxwiz::backend_placement::ClassifyLayer(
              gui::NodeType::TransformerEncoder).kind ==
              cyxwiz::backend_placement::LayerCapabilityKind::CpuBackedModelLayer,
          "TransformerEncoder capability kind should be CpuBackedModelLayer");
    Check(!cyxwiz::backend_placement::IsKnownArrayFireTensorLayer(
              gui::NodeType::TransformerDecoder),
          "TransformerDecoder should not be classified as direct ArrayFire tensor-capable");
    Check(cyxwiz::backend_placement::IsKnownCpuBackedModelLayer(
              gui::NodeType::TransformerDecoder),
          "TransformerDecoder should be classified as a known CPU-backed model layer");
    Check(cyxwiz::backend_placement::ClassifyLayer(
              gui::NodeType::TransformerDecoder).kind ==
              cyxwiz::backend_placement::LayerCapabilityKind::CpuBackedModelLayer,
          "TransformerDecoder capability kind should be CpuBackedModelLayer");
    Check(cyxwiz::backend_placement::ClassifyLayer(
              gui::NodeType::TimeDistributed).kind ==
              cyxwiz::backend_placement::LayerCapabilityKind::TimeDistributedSequenceWrapper,
          "TimeDistributed capability kind should be an explicit sequence wrapper");

    auto gru_config = CompileRecurrentGraph(gui::NodeType::GRU, 32, false);
    Check(gru_config.is_valid, "GRU placement graph should compile");
    Check(!gru_config.compiler_placement_fingerprint.empty(),
          "compiler should produce a placement capability fingerprint");
    Check(gru_config.compiler_placement_fingerprint ==
              cyxwiz::FingerprintPlacementEntries(
                  gru_config.backend_placements),
          "compiler placement fingerprint should match its emitted entries");
    Check(gru_config.backend_placements.size() == 3,
          "GRU graph should produce placement entries for Embedding, GRU, Dense");
    const auto gru_summary = gru_config.SummarizeBackendPlacements();
    Check(gru_summary.total == 3, "GRU placement summary should count all entries");
    Check(gru_summary.gpu == 2, "GRU placement summary should count Embedding and Dense as GPU");
    Check(gru_summary.cpu == 1, "GRU placement summary should count GRU as CPU");
    Check(gru_summary.unknown == 0, "GRU placement summary should have no unknown entries");

    cyxwiz::ExecutionDeviceContext cpu_context;
    cpu_context.requested_backend = "arrayfire_cpu";
    cpu_context.effective_backend = "arrayfire_cpu";
    cpu_context.device_name = "Contract CPU";
    cpu_context.valid = true;
    cpu_context.fallback_policy =
        cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback;
    const auto gru_cpu_plan =
        cyxwiz::BuildExecutionPlacementPlan(gru_config, cpu_context);
    const auto gru_cpu_plan_repeat =
        cyxwiz::BuildExecutionPlacementPlan(gru_config, cpu_context);
    Check(gru_cpu_plan.fingerprint == gru_cpu_plan_repeat.fingerprint,
          "context-resolved placement fingerprint should be deterministic");
    Check(gru_cpu_plan.compiler_fingerprint ==
              gru_config.compiler_placement_fingerprint,
          "executable plan should retain the compiler capability identity");
    Check(!gru_cpu_plan.IsStrictlyExecutable(),
          "strict preflight should reject the compiler-known native CPU GRU path");
    Check(gru_cpu_plan.StrictBlockerSummary().find("GRU") !=
              std::string::npos,
          "strict preflight blocker should identify the GRU stage");

    cyxwiz::TrainingConfiguration dense_config;
    cyxwiz::CompiledLayer dense_layer;
    dense_layer.node_id = 50;
    dense_layer.name = "DenseContract";
    dense_layer.type = gui::NodeType::Dense;
    dense_layer.input_shape = {8};
    dense_layer.output_shape = {2};
    dense_layer.units = 2;
    dense_config.layers.push_back(dense_layer);
    cyxwiz::BackendPlacementEntry dense_capability;
    dense_capability.node_id = dense_layer.node_id;
    dense_capability.node_name = dense_layer.name;
    dense_capability.node_type = "Dense";
    dense_capability.status = cyxwiz::BackendPlacementStatus::Gpu;
    dense_capability.reason_code =
        cyxwiz::BackendPlacementReason::ArrayFireTensorOpCapable;
    dense_config.backend_placements.push_back(dense_capability);
    dense_config.compiler_placement_fingerprint =
        cyxwiz::FingerprintPlacementEntries(dense_config.backend_placements);

    const auto dense_cpu_plan =
        cyxwiz::BuildExecutionPlacementPlan(dense_config, cpu_context);
    Check(dense_cpu_plan.IsExecutable(),
          "known dense loss/metric/optimizer stages should pass executable preflight");
    Check(dense_cpu_plan.IsStrictlyExecutable(),
          "ArrayFire CPU dense plan should not be classified as native CPU fallback");
    size_t dense_forward_entries = 0;
    bool saw_gradient_accumulation = false;
    bool saw_gradient_averaging = false;
    bool saw_optimizer_state = false;
    bool saw_optimizer_update = false;
    for (const auto& entry : dense_cpu_plan.entries) {
        Check(entry.fallback_backend.empty(),
              "executable plan entries must not predict an unobserved fallback");
        if (entry.node_id == dense_layer.node_id) {
            ++dense_forward_entries;
            Check(entry.node_type == "ModelForward.Dense",
                  "resolved dense entry should carry its executable stage name");
            Check(entry.expected_backend == "arrayfire_cpu",
                  "resolved dense entry should bind ArrayFire CPU exactly");
            Check(entry.status == cyxwiz::ExecutionPlacementStatus::ArrayFire,
                  "ArrayFire CPU should use the ArrayFire execution status");
        }
        saw_gradient_accumulation = saw_gradient_accumulation ||
            entry.node_type == "GradientTransform.Accumulate";
        saw_gradient_averaging = saw_gradient_averaging ||
            entry.node_type == "GradientTransform.Average";
        saw_optimizer_state = saw_optimizer_state ||
            entry.node_type == "OptimizerState.Adam";
        saw_optimizer_update = saw_optimizer_update ||
            entry.node_type == "OptimizerUpdate.Adam";
    }
    Check(dense_forward_entries == 1,
          "executable plan should not duplicate compiler and runtime Dense entries");
    Check(saw_gradient_accumulation,
          "executable plan should declare gradient accumulation");
    Check(saw_gradient_averaging,
          "executable plan should declare gradient averaging");
    Check(saw_optimizer_state,
          "executable plan should declare optimizer state placement");
    Check(saw_optimizer_update,
          "executable plan should declare optimizer update placement");

    auto native_loss_config = dense_config;
    native_loss_config.loss_type = gui::NodeType::SoftDiceLoss;
    const auto native_loss_plan =
        cyxwiz::BuildExecutionPlacementPlan(native_loss_config, cpu_context);
    Check(native_loss_plan.IsExecutable(),
          "declared native CPU loss should remain executable in compatibility mode");
    Check(!native_loss_plan.IsStrictlyExecutable(),
          "declared native CPU loss should block strict ArrayFire residency");
    Check(native_loss_plan.StrictBlockerSummary().find(
              "loss_native_cpu_compatibility") != std::string::npos,
          "strict blocker should identify the native CPU loss reason");

    auto sequence_metrics_config = dense_config;
    sequence_metrics_config.sequence_batch.enabled = true;
    sequence_metrics_config.target.value_kind =
        cyxwiz::TargetValueKind::TokenIds;
    const auto sequence_metrics_plan =
        cyxwiz::BuildExecutionPlacementPlan(
            sequence_metrics_config, cpu_context);
    Check(sequence_metrics_plan.IsExecutable(),
          "sequence metrics should remain executable in compatibility mode");
    Check(!sequence_metrics_plan.IsStrictlyExecutable(),
          "host sequence metrics should block strict ArrayFire residency");
    Check(sequence_metrics_plan.StrictBlockerSummary().find(
              "metrics_sequence_native_cpu_compatibility") !=
              std::string::npos,
          "strict blocker should identify sequence metric host materialization");
    bool saw_sequence_metrics = false;
    for (const auto& entry : sequence_metrics_plan.entries) {
        if (entry.node_type != "Metrics.SequenceTokenAccuracy") {
            continue;
        }
        saw_sequence_metrics = true;
        Check(entry.status == cyxwiz::BackendPlacementStatus::Cpu,
              "sequence token metrics should be declared native CPU");
        Check(entry.expected_backend == "native_cpu",
              "sequence token metrics should not claim the ArrayFire device");
    }
    Check(saw_sequence_metrics,
          "sequence plan should include the token-accuracy metric stage");

    auto unsupported_config = dense_config;
    unsupported_config.loss_type = gui::NodeType::Output;
    unsupported_config.optimizer_type = gui::NodeType::Output;
    const auto unsupported_plan =
        cyxwiz::BuildExecutionPlacementPlan(unsupported_config, cpu_context);
    Check(!unsupported_plan.IsExecutable(),
          "unknown loss and optimizer node types should fail executable preflight");
    Check(unsupported_plan.FatalBlockerSummary().find("loss_unsupported") !=
              std::string::npos,
          "fatal preflight should identify an unsupported loss");
    Check(unsupported_plan.FatalBlockerSummary().find("optimizer_unsupported") !=
              std::string::npos,
          "fatal preflight should identify an unsupported optimizer");

    auto metric_mismatch_config = dense_config;
    metric_mismatch_config.target.value_kind =
        cyxwiz::TargetValueKind::Continuous;
    const auto metric_mismatch_plan =
        cyxwiz::BuildExecutionPlacementPlan(metric_mismatch_config, cpu_context);
    Check(!metric_mismatch_plan.IsExecutable(),
          "continuous targets with classification loss should fail metric preflight");
    Check(metric_mismatch_plan.FatalBlockerSummary().find(
              "metrics_loss_target_contract_unsupported") !=
              std::string::npos,
          "fatal preflight should identify the loss/metric target mismatch");

    const auto* gru_embedding_placement = FindPlacement(gru_config, 3);
    Check(gru_embedding_placement != nullptr,
          "GRU graph should report Embedding placement");
    Check(gru_embedding_placement->node_type == "Embedding",
          "Embedding placement should name the layer");
    Check(gru_embedding_placement->status == cyxwiz::BackendPlacementStatus::Gpu,
          "Embedding should be marked GPU-capable");
    Check(gru_embedding_placement->reason_code ==
              cyxwiz::BackendPlacementReason::ArrayFireTensorOpCapable,
          "Embedding should use the generic tensor placement reason");

    const auto* gru_placement = FindPlacement(gru_config, 4);
    Check(gru_placement != nullptr, "GRU placement entry should reference node 4");
    Check(gru_placement->node_type == "GRU", "GRU placement should name the layer");
    Check(gru_placement->expected_backend == "CPU",
          "GRU should be conservatively placed on CPU");
    Check(gru_placement->status == cyxwiz::BackendPlacementStatus::Cpu,
          "GRU placement status should be cpu");
    Check(gru_placement->reason_code ==
              cyxwiz::RecurrentCudaPlacementReason::GruArrayFireCudaProbeRequired,
          "GRU placement should use the shared reason code");
    Check(gru_placement->explanation.find("batch_size=64") != std::string::npos,
          "GRU placement explanation should include compiled batch size");
    Check(gru_placement->explanation.find("seq_len=64") != std::string::npos,
          "GRU placement explanation should include inferred sequence length");
    Check(HasWarningText(
              gru_config,
              cyxwiz::RecurrentCudaPlacementReason::GruArrayFireCudaProbeRequired),
          "GRU CPU placement should surface as a compiler warning");

    const auto* gru_dense_placement = FindPlacement(gru_config, 5);
    Check(gru_dense_placement != nullptr,
          "GRU graph should report Dense placement");
    Check(gru_dense_placement->node_type == "Dense",
          "Dense placement should name the layer");
    Check(gru_dense_placement->status == cyxwiz::BackendPlacementStatus::Gpu,
          "Dense should be marked GPU-capable");

    auto lstm_config = CompileRecurrentGraph(gui::NodeType::LSTM, 8, false);
    Check(lstm_config.is_valid, "small LSTM placement graph should compile");
    Check(lstm_config.backend_placements.size() == 3,
          "LSTM graph should produce placement entries for Embedding, LSTM, Dense");
    const auto lstm_summary = lstm_config.SummarizeBackendPlacements();
    Check(lstm_summary.total == 3, "LSTM placement summary should count all entries");
    Check(lstm_summary.gpu == 3, "LSTM placement summary should count all layers as GPU");
    Check(lstm_summary.cpu == 0, "LSTM placement summary should have no CPU entries");

    const auto* lstm_placement = FindPlacement(lstm_config, 4);
    Check(lstm_placement != nullptr, "LSTM placement entry should reference node 4");
    Check(lstm_placement->node_type == "LSTM", "LSTM placement should name the layer");
    Check(lstm_placement->expected_backend == "ArrayFire CUDA",
          "small single-direction LSTM should remain GPU-eligible");
    Check(lstm_placement->status == cyxwiz::BackendPlacementStatus::Gpu,
          "LSTM placement status should be gpu");
    Check(lstm_placement->reason_code ==
              cyxwiz::RecurrentCudaPlacementReason::ArrayFireCudaAllowedByEstimator,
          "LSTM placement should use the shared allow reason code");
    Check(!HasWarningText(
              lstm_config,
              cyxwiz::RecurrentCudaPlacementReason::ArrayFireCudaAllowedByEstimator),
          "GPU-eligible LSTM placement should not create a warning");

    cyxwiz::RecurrentCudaPlacementRequest observed_lstm;
    observed_lstm.kind = cyxwiz::RecurrentLayerKind::LSTM;
    observed_lstm.batch_size = 64;
    observed_lstm.seq_len = 64;
    observed_lstm.input_size = 64;
    observed_lstm.hidden_size = 8;
    observed_lstm.num_layers = 1;
    observed_lstm.bidirectional = false;
    observed_lstm.return_sequences = false;
    cyxwiz::RecordRecurrentCudaPlacementObservation(
        observed_lstm,
        cyxwiz::BackendPlacementObservationReason::CudaJitParamOverflow,
        cyxwiz::BackendPlacementObservationSource::Test,
        "test observation");
    cyxwiz::BackendPlacementObservation direct_observation;
    Check(cyxwiz::TryGetBackendPlacementObservation(
              "LSTM",
              "cuda",
              cyxwiz::CurrentBackendPlacementDeviceSignature(),
              "float32",
              cyxwiz::BuildRecurrentCudaPlacementShapeSignature(observed_lstm),
              direct_observation),
          "recurrent observation should be keyed by active device signature");
    Check(!cyxwiz::TryGetBackendPlacementObservation(
              "LSTM",
              "cuda",
              "different-device",
              "float32",
              cyxwiz::BuildRecurrentCudaPlacementShapeSignature(observed_lstm),
              direct_observation),
          "recurrent observation should not match a different device signature");

    auto cached_lstm_config =
        CompileRecurrentGraph(gui::NodeType::LSTM, 8, false);
    const auto* cached_lstm_placement =
        FindPlacement(cached_lstm_config, 4);
    Check(cached_lstm_placement != nullptr,
          "cached LSTM placement entry should reference node 4");
    Check(cached_lstm_placement->status == cyxwiz::BackendPlacementStatus::Cpu,
          "cached CUDA overflow should route previously GPU-eligible LSTM to CPU");
    Check(cached_lstm_placement->reason_code ==
              cyxwiz::RecurrentCudaPlacementReason::CudaJitParamOverflowRisk,
          "cached CUDA overflow should use compiler placement overflow reason");
    Check(cached_lstm_placement->explanation.find(
              "previous runtime/probe observation") != std::string::npos,
          "test-source LSTM placement should explain generic cache feedback");
    Check(cached_lstm_placement->explanation.find("source=test") !=
              std::string::npos,
          "cached LSTM placement should include observation source");
    Check(cached_lstm_placement->explanation.find("Device:") !=
              std::string::npos,
          "cached LSTM placement should include observation device");
    Check(cached_lstm_placement->explanation.find("separate from VRAM") !=
              std::string::npos,
          "cached LSTM placement should distinguish kernel overflow from VRAM");
    Check(cached_lstm_placement->observation_source ==
              cyxwiz::BackendPlacementObservationSource::Test,
          "cached LSTM placement should carry observation source metadata");
    Check(cached_lstm_placement->observation_shape_signature.find("kind=LSTM") !=
              std::string::npos,
          "cached LSTM placement should carry observation shape metadata");
    Check(HasWarningText(
              cached_lstm_config,
              cyxwiz::RecurrentCudaPlacementReason::CudaJitParamOverflowRisk),
          "cached CUDA overflow should surface as a compiler warning");
    cyxwiz::ClearBackendPlacementObservationCacheForTesting();

    cyxwiz::RecordRecurrentCudaPreflightProbeFailure(
        observed_lstm,
        cyxwiz::BackendPlacementObservationReason::CudaJitParamOverflow,
        "simulated probe observation");
    auto probed_lstm_config =
        CompileRecurrentGraph(gui::NodeType::LSTM, 8, false);
    const auto* probed_lstm_placement =
        FindPlacement(probed_lstm_config, 4);
    Check(probed_lstm_placement != nullptr,
          "preflight-probed LSTM placement entry should reference node 4");
    Check(probed_lstm_placement->status == cyxwiz::BackendPlacementStatus::Cpu,
          "preflight probe observation should route GPU-eligible LSTM to CPU");
    Check(probed_lstm_placement->explanation.find("source=preflight_probe") !=
              std::string::npos,
          "preflight probe observation source should be visible");
    Check(probed_lstm_placement->explanation.find(
              "previous preflight probe observation") != std::string::npos,
          "preflight probe placement should use source-specific wording");
    Check(HasWarningText(probed_lstm_config, "source=preflight_probe"),
          "preflight probe observation should surface through compiler warnings");
    cyxwiz::ClearBackendPlacementObservationCacheForTesting();

    cyxwiz::RecordRecurrentCudaPreflightProbeFailure(
        observed_lstm,
        cyxwiz::BackendPlacementObservationReason::BackendCompileTimeout,
        "simulated recurrent preflight timeout");
    auto timeout_lstm_config =
        CompileRecurrentGraph(gui::NodeType::LSTM, 8, false);
    const auto* timeout_lstm_placement =
        FindPlacement(timeout_lstm_config, 4);
    Check(timeout_lstm_placement != nullptr,
          "timeout LSTM placement entry should reference node 4");
    Check(timeout_lstm_placement->status == cyxwiz::BackendPlacementStatus::Cpu,
          "timeout preflight observation should route GPU-eligible LSTM to CPU");
    Check(timeout_lstm_placement->reason_code ==
              cyxwiz::BackendPlacementObservationReason::BackendCompileTimeout,
          "timeout preflight placement should preserve timeout reason");
    Check(timeout_lstm_placement->observation_source ==
              cyxwiz::BackendPlacementObservationSource::PreflightProbe,
          "timeout preflight placement should carry source metadata");
    Check(timeout_lstm_placement->observation_detail.find("timeout") !=
              std::string::npos,
          "timeout preflight placement should carry detail metadata");
    Check(timeout_lstm_placement->observation_probe_outcome == "timeout",
          "timeout preflight placement should carry timeout outcome metadata");
    Check(timeout_lstm_placement->observation_probe_scope ==
              cyxwiz::BackendPlacementProbeScope::NormalCompile,
          "timeout preflight placement should carry normal compile scope metadata");
    Check(HasWarningText(
              timeout_lstm_config,
              cyxwiz::BackendPlacementObservationReason::BackendCompileTimeout),
          "timeout preflight observation should surface as compiler warning");
    cyxwiz::ClearBackendPlacementObservationCacheForTesting();

    cyxwiz::RecordBackendPlacementObservationForActiveDevice(
        "Dense",
        "cuda",
        "float32",
        cyxwiz::BuildDensePlacementShapeSignature({8}, 2),
        cyxwiz::BackendPlacementObservationReason::GpuOutOfMemory,
        cyxwiz::BackendPlacementObservationSource::RuntimeFallback,
        "simulated Dense fallback observation");
    auto dense_cached_config =
        CompileRecurrentGraph(gui::NodeType::LSTM, 8, false);
    const auto* dense_cached_placement =
        FindPlacement(dense_cached_config, 5);
    Check(dense_cached_placement != nullptr,
          "cached Dense fallback placement entry should reference node 5");
    Check(dense_cached_placement->status == cyxwiz::BackendPlacementStatus::Cpu,
          "cached tensor-layer fallback should route exact Dense shape to CPU");
    Check(dense_cached_placement->reason_code ==
              cyxwiz::BackendPlacementObservationReason::GpuOutOfMemory,
          "cached Dense fallback should preserve structured fallback reason");
    Check(dense_cached_placement->explanation.find("source=runtime_fallback") !=
              std::string::npos,
          "cached Dense fallback should include observation source");
    Check(dense_cached_placement->observation_source ==
              cyxwiz::BackendPlacementObservationSource::RuntimeFallback,
          "cached Dense fallback should carry observation source metadata");
    Check(dense_cached_placement->observation_shape_signature ==
              cyxwiz::BuildDensePlacementShapeSignature({8}, 2),
          "cached Dense fallback should carry observation shape metadata");
    Check(HasWarningText(dense_cached_config,
                         cyxwiz::BackendPlacementObservationReason::GpuOutOfMemory),
          "cached Dense fallback should surface as a compiler warning");
    const auto compiler_cache_path = std::filesystem::temp_directory_path() /
        "cyxwiz_backend_placement_compiler_cache_test.json";
    cache_error.clear();
    Check(cyxwiz::SaveBackendPlacementObservationCache(
              compiler_cache_path.string(),
              &cache_error),
          "Compiler placement observation cache should save: " + cache_error);
    cyxwiz::ClearBackendPlacementObservationCacheForTesting();
    auto persistent_dense_cached_config =
        CompileRecurrentGraph(
            gui::NodeType::LSTM,
            8,
            false,
            compiler_cache_path.string());
    const auto* persistent_dense_cached_placement =
        FindPlacement(persistent_dense_cached_config, 5);
    Check(persistent_dense_cached_placement != nullptr,
          "persistent Dense fallback placement entry should reference node 5");
    Check(persistent_dense_cached_placement->status ==
              cyxwiz::BackendPlacementStatus::Cpu,
          "persistent tensor-layer fallback should route exact Dense shape to CPU");
    Check(persistent_dense_cached_placement->reason_code ==
              cyxwiz::BackendPlacementObservationReason::GpuOutOfMemory,
          "persistent Dense fallback should preserve structured fallback reason");
    Check(HasWarningText(
              persistent_dense_cached_config,
              cyxwiz::BackendPlacementObservationReason::GpuOutOfMemory),
          "persistent Dense fallback should surface as a compiler warning");
    std::error_code compiler_cache_remove_error;
    std::filesystem::remove(compiler_cache_path, compiler_cache_remove_error);
    cyxwiz::ClearBackendPlacementObservationCacheForTesting();

    cyxwiz::TrainingConfiguration unknown_config;
    cyxwiz::BackendPlacementEntry unknown_placement;
    unknown_placement.status = cyxwiz::BackendPlacementStatus::Unknown;
    unknown_config.backend_placements.push_back(unknown_placement);
    const auto unknown_summary = unknown_config.SummarizeBackendPlacements();
    Check(unknown_summary.total == 1,
          "unknown placement summary should count the entry");
    Check(unknown_summary.unknown == 1,
          "unknown placement summary should classify unknown status");
    Check(unknown_summary.HasNonGpu(),
          "unknown placement summary should be treated as non-GPU");
    Check(unknown_config.backend_placements.front().NeedsUserAttention(),
          "unknown placement should require user attention");

    auto unclassified_config = CompileUnclassifiedLayerGraph();
    Check(unclassified_config.is_valid,
          "unclassified layer graph should still compile");
    Check(unclassified_config.backend_placements.size() == 1,
          "unclassified graph should produce one backend placement");
    const auto* unclassified_placement =
        FindPlacement(unclassified_config, 8);
    Check(unclassified_placement != nullptr,
          "unclassified placement should reference TimeDistributed node");
    Check(unclassified_placement->node_type == "TimeDistributed",
          "unclassified placement should name the layer");
    Check(unclassified_placement->status == cyxwiz::BackendPlacementStatus::Unknown,
          "TimeDistributed wrapper placement should be unknown");
    Check(unclassified_placement->reason_code ==
              cyxwiz::BackendPlacementReason::TimeDistributedSequenceWrapper,
          "TimeDistributed wrapper placement should use the wrapper reason code");
    Check(unclassified_placement->NeedsUserAttention(),
          "TimeDistributed wrapper placement should require user attention");
    Check(HasWarningText(
              unclassified_config,
              cyxwiz::BackendPlacementReason::TimeDistributedSequenceWrapper),
          "TimeDistributed wrapper placement should surface as a compiler warning");
    const auto unclassified_summary =
        unclassified_config.SummarizeBackendPlacements();
    Check(unclassified_summary.unknown == 1,
          "compiled unclassified layer should count as unknown");

    std::cout << "Recurrent backend placement tests passed\n";
    return 0;
}
