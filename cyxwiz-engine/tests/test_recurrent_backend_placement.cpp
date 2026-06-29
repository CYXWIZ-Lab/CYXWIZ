#include "../src/core/graph_compiler.h"
#include "../src/core/backend_placement_capabilities.h"
#include "../src/gui/loaders/data_loader.h"
#include "../../cyxwiz-backend/src/algorithms/arrayfire_backend_utils.h"
#include "cyxwiz/backend_placement_observation.h"
#include "cyxwiz/recurrent_cuda_placement.h"

#include <cstdlib>
#include <iostream>
#include <string>
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

cyxwiz::TrainingConfiguration CompileRecurrentGraph(gui::NodeType recurrent_type,
                                                    int hidden_size,
                                                    bool bidirectional) {
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
    return compiler.Compile(nodes, links, true);
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
    Check(cyxwiz::backend_placement::ClassifyLayer(
              gui::NodeType::TimeDistributed).kind ==
              cyxwiz::backend_placement::LayerCapabilityKind::TimeDistributedSequenceWrapper,
          "TimeDistributed capability kind should be an explicit sequence wrapper");

    auto gru_config = CompileRecurrentGraph(gui::NodeType::GRU, 32, false);
    Check(gru_config.is_valid, "GRU placement graph should compile");
    Check(gru_config.backend_placements.size() == 3,
          "GRU graph should produce placement entries for Embedding, GRU, Dense");
    const auto gru_summary = gru_config.SummarizeBackendPlacements();
    Check(gru_summary.total == 3, "GRU placement summary should count all entries");
    Check(gru_summary.gpu == 2, "GRU placement summary should count Embedding and Dense as GPU");
    Check(gru_summary.cpu == 1, "GRU placement summary should count GRU as CPU");
    Check(gru_summary.unknown == 0, "GRU placement summary should have no unknown entries");

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
    Check(HasWarningText(dense_cached_config,
                         cyxwiz::BackendPlacementObservationReason::GpuOutOfMemory),
          "cached Dense fallback should surface as a compiler warning");
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
