#include "../src/core/debug_recommendation_engine.h"
#include "../src/core/debug_export_correlation_tracer.h"
#include "../src/core/debug_graph_trace_executor.h"
#include "../src/core/debug_memory_ownership_tracer.h"
#include "../src/core/debug_node_inspector.h"
#include "../src/core/debug_operator_trace_adapter.h"
#include "../src/core/debug_run_store.h"
#include "../src/core/debug_runtime_backend_classifier.h"
#include "../src/core/debug_session_manager.h"
#include "../src/core/debug_smoke_sample_selector.h"
#include "../src/core/debug_support_bundle_builder.h"
#include "../src/core/debug_windows_crash_importer.h"
#include "../src/core/annotation_manager.h"
#include "../src/core/data_registry.h"
#include "../src/core/error_codes.h"
#include "../src/core/node_executors/text_tokenizer_operator.h"
#include "../src/core/preflight_validator.h"
#include "../src/core/text_preprocessing_tracer.h"

#include <arrow/api.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace cyxwiz {

DataRegistry& DataRegistry::Instance() {
    static DataRegistry registry;
    return registry;
}

void DataRegistry::RegisterTextDataset(const std::string& name,
                                       const TextDatasetEntry& entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    text_dataset_entries_[name] = entry;
}

const DataRegistry::TextDatasetEntry* DataRegistry::GetTextDatasetEntry(
    const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = text_dataset_entries_.find(name);
    return it == text_dataset_entries_.end() ? nullptr : &it->second;
}

void DataRegistry::UnregisterTextDataset(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    text_dataset_entries_.erase(name);
}

} // namespace cyxwiz

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

gui::MLNode MakeNode(int id, gui::NodeType type, const std::string& name) {
    gui::MLNode node;
    node.id = id;
    node.type = type;
    node.name = name;
    return node;
}

std::shared_ptr<arrow::Array> FinishStringArray(
    const std::vector<std::string>& values) {
    arrow::StringBuilder builder;
    for (const auto& value : values) {
        auto st = builder.Append(value);
        Check(st.ok(), st.ToString());
    }
    std::shared_ptr<arrow::Array> array;
    auto st = builder.Finish(&array);
    Check(st.ok(), st.ToString());
    return array;
}

bool HasRecommendation(const std::vector<cyxwiz::DebugRecommendation>& recs,
                       const std::string& title) {
    for (const auto& rec : recs) {
        if (rec.title == title) {
            return true;
        }
    }
    return false;
}

bool HasIssueCode(const std::vector<cyxwiz::ValidationIssue>& issues,
                  const std::string& code) {
    for (const auto& issue : issues) {
        if (issue.error_code == code) {
            return true;
        }
    }
    return false;
}

void TestDebugSessionSnapshotContract() {
    auto data = MakeNode(1, gui::NodeType::DataInput, "Data Input");
    data.parameters = {{"dataset", "debug_text"}};
    auto tokenizer = MakeNode(2, gui::NodeType::TextTokenizer, "Tokenizer");
    tokenizer.parameters = {{"lowercase", "true"}};

    gui::NodeLink link;
    link.id = 10;
    link.from_node = 1;
    link.from_pin = 0;
    link.to_node = 2;
    link.to_pin = 0;

    const std::vector<gui::MLNode> nodes = {data, tokenizer};
    const std::vector<gui::NodeLink> links = {link};

    cyxwiz::DebugSession session = cyxwiz::DebugSessionManager::StartSession(
        "debug-contract-run",
        "FullWorkflow",
        0xBEEF,
        nodes,
        links,
        3);

    Check(session.run_id == "debug-contract-run", "run id should be preserved");
    Check(session.mode_name == "FullWorkflow", "mode name should be preserved");
    Check(session.graph_hash == 0xBEEF, "graph hash should be preserved");
    Check(session.node_count == 2, "node count should match snapshot");
    Check(session.link_count == 1, "link count should match snapshot");
    Check(session.selected_sample_index == 3, "selected sample should be preserved");
    Check(session.graph_nodes.size() == 2, "node snapshots should be captured");
    Check(session.graph_nodes[1].id == 2, "node id should be captured");
    Check(session.graph_nodes[1].name == "Tokenizer", "node name should be captured");
    Check(session.graph_nodes[1].parameters.size() == 1, "node params should be captured");
    Check(session.graph_links.size() == 1, "link snapshots should be captured");
    Check(session.graph_links[0].from_node == 1, "link source should be captured");
    Check(session.graph_links[0].to_node == 2, "link target should be captured");
    Check(session.studio_events.size() == 1, "session start Studio event should be emitted");
    Check(session.studio_events[0].action == "DebugSession.Start",
          "session start event action should be stable");
    Check(session.traces.size() == 1, "graph snapshot trace should be emitted");

    const cyxwiz::DebugTraceRecord& trace = session.traces[0];
    Check(trace.run_id == session.run_id, "snapshot trace should carry run id");
    Check(trace.node_name == "GraphSnapshot", "snapshot trace name should be stable");
    Check(trace.phase == "GraphSnapshot", "snapshot trace phase should be stable");
    Check(trace.role == cyxwiz::DebugTraceRole::CompileArtifact,
          "snapshot trace should be a compile artifact");
    Check(trace.status == "captured", "snapshot trace status should be captured");
    Check(trace.payload["mode"].get<std::string>() == "FullWorkflow",
          "snapshot payload should include mode");
    Check(trace.payload["graph_hash"].get<uint64_t>() == 0xBEEF,
          "snapshot payload should include graph hash");
    Check(trace.payload["node_count"].get<size_t>() == 2,
          "snapshot payload should include node count");
    Check(trace.payload["link_count"].get<size_t>() == 1,
          "snapshot payload should include link count");
    Check(trace.payload["selected_sample_index"].get<size_t>() == 3,
          "snapshot payload should include selected sample");
    Check(trace.payload["nodes"].is_array() && trace.payload["nodes"].size() == 2,
          "snapshot payload should include node array");
    Check(trace.payload["links"].is_array() && trace.payload["links"].size() == 1,
          "snapshot payload should include link array");
}

void TestNodeTraceContract() {
    cyxwiz::DebugTraceRecord trace = cyxwiz::DebugNodeTraceContract::Make(
        "node-contract-run",
        42,
        "DenseHead",
        "Dense",
        "Forward",
        cyxwiz::DebugTraceRole::Activation,
        {2, 3},
        {2, 4},
        "float32",
        "CPU",
        "ok");

    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(trace),
          "node trace should advertise canonical schema");
    Check(trace.run_id == "node-contract-run",
          "node trace should preserve run id");
    Check(trace.node_id == 42, "node trace should preserve node id");
    Check(trace.node_name == "DenseHead",
          "node trace should preserve node name");
    Check(trace.node_type == "Dense",
          "node trace should preserve node type");
    Check(trace.phase == "Forward", "node trace should preserve phase");
    Check(trace.role == cyxwiz::DebugTraceRole::Activation,
          "node trace should preserve role");
    Check(trace.input_shape == std::vector<size_t>{2, 3},
          "node trace should preserve input shape");
    Check(trace.output_shape == std::vector<size_t>{2, 4},
          "node trace should preserve output shape");
    Check(trace.dtype == "float32", "node trace should preserve dtype");
    Check(trace.status == "ok", "node trace should preserve status");
    Check(trace.payload["schema"].get<std::string>() ==
              cyxwiz::DebugNodeTraceContract::kSchema,
          "node trace payload should include schema");
    Check(trace.payload["backend"].get<std::string>() == "CPU",
          "node trace payload should include backend");
    Check(trace.payload["input_rank"].get<size_t>() == 2,
          "node trace payload should include input rank");
    Check(trace.payload["output_rank"].get<size_t>() == 2,
          "node trace payload should include output rank");
    Check(trace.payload["input_numel"].get<size_t>() == 6,
          "node trace payload should include input element count");
    Check(trace.payload["output_numel"].get<size_t>() == 8,
          "node trace payload should include output element count");
    Check(trace.payload["warning_count"].get<size_t>() == 0,
          "node trace should start with zero warnings");
    Check(trace.payload["error_count"].get<size_t>() == 0,
          "node trace should start with zero errors");

    cyxwiz::DebugNodeTraceContract::AddWarning(
        trace,
        "CPU fallback used",
        cyxwiz::errors::Gpu::KernelExecutionFailed);
    Check(trace.issues.size() == 1,
          "node trace warning should append an issue");
    Check(trace.issues[0].error_code ==
              cyxwiz::errors::Gpu::KernelExecutionFailed,
          "node trace warning should preserve error code");
    Check(trace.payload["warning_count"].get<size_t>() == 1,
          "node trace warning should update warning count");
    Check(trace.status == "ok",
          "node trace warning should not fail the trace");

    cyxwiz::DebugNodeTraceContract::AddError(
        trace,
        "Shape mismatch",
        cyxwiz::errors::Compiler::TensorShapeMismatch);
    Check(trace.issues.size() == 2,
          "node trace error should append an issue");
    Check(trace.issues[1].error_code ==
              cyxwiz::errors::Compiler::TensorShapeMismatch,
          "node trace error should preserve error code");
    Check(trace.payload["error_count"].get<size_t>() == 1,
          "node trace error should update error count");
    Check(trace.status == "failed",
          "node trace error should fail the trace");
}

void TestGraphTraceExecutionSlice() {
    cyxwiz::DebugGraphTraceExecutor executor;
    std::vector<cyxwiz::DebugGraphTraceStep> steps;

    cyxwiz::DebugGraphTraceStep input;
    input.node_id = 1;
    input.node_name = "Input";
    input.node_type = "DataInput";
    input.phase = "Load";
    input.role = cyxwiz::DebugTraceRole::RawInput;
    input.output_shape = {3, 2};
    input.dtype = "float32";
    input.backend = "CPU";
    input.payload["rows"] = 3;
    input.payload["columns"] = 2;
    steps.push_back(std::move(input));

    cyxwiz::DebugGraphTraceStep scaler;
    scaler.node_id = 2;
    scaler.node_name = "Scale";
    scaler.node_type = "StandardScaler";
    scaler.phase = "Transform";
    scaler.role = cyxwiz::DebugTraceRole::FeatureTensor;
    scaler.input_shape = {3, 2};
    scaler.output_shape = {3, 2};
    scaler.dtype = "float32";
    scaler.backend = "CPU";
    scaler.duration_ms = 0.25f;
    scaler.warnings.push_back("CPU operator path used");
    scaler.payload["operator"] = "StandardScaler";
    steps.push_back(std::move(scaler));

    cyxwiz::DebugGraphTraceStep output;
    output.node_id = 3;
    output.node_name = "Output";
    output.node_type = "DataOutput";
    output.phase = "Save";
    output.role = cyxwiz::DebugTraceRole::Prediction;
    output.input_shape = {3, 2};
    output.output_shape = {3, 2};
    output.dtype = "float32";
    output.backend = "CPU";
    steps.push_back(std::move(output));

    const auto traces = executor.TraceSteps("graph-trace-run", steps);
    Check(traces.size() == 3,
          "graph trace executor should emit one trace per step");

    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(traces[0]),
          "graph trace step should use canonical node trace schema");
    Check(traces[0].run_id == "graph-trace-run",
          "graph trace should preserve run id");
    Check(traces[0].node_id == 1,
          "graph trace should preserve first node id");
    Check(traces[0].role == cyxwiz::DebugTraceRole::RawInput,
          "graph trace should preserve first node role");
    Check(traces[0].payload["rows"].get<int>() == 3,
          "graph trace should preserve custom payload");

    Check(traces[1].node_id == 2,
          "graph trace should preserve transform node id");
    Check(traces[1].phase == "Transform",
          "graph trace should preserve transform phase");
    Check(traces[1].payload["operator"].get<std::string>() == "StandardScaler",
          "graph trace should preserve operator payload");
    Check(traces[1].payload["warning_count"].get<size_t>() == 1,
          "graph trace should count warnings");
    Check(traces[1].status == "ok",
          "warning-only graph trace should remain ok");
    Check(traces[1].duration_ms == 0.25f,
          "graph trace should preserve duration");

    Check(traces[2].node_type == "DataOutput",
          "graph trace should preserve output node type");
    Check(traces[2].input_shape == std::vector<size_t>{3, 2},
          "graph trace should preserve output input shape");
    Check(traces[2].output_shape == std::vector<size_t>{3, 2},
          "graph trace should preserve output output shape");
}

void TestRuntimeBackendClassificationContract() {
    cyxwiz::BackendPlacementEntry gpu;
    gpu.node_id = 42;
    gpu.node_name = "DenseHead";
    gpu.node_type = "Dense";
    gpu.requested_backend = "auto";
    gpu.expected_backend = "ArrayFire active backend";
    gpu.fallback_backend = "CPU";
    gpu.status = cyxwiz::BackendPlacementStatus::Gpu;
    gpu.reason_code =
        cyxwiz::BackendPlacementReason::ArrayFireTensorOpCapable;
    gpu.explanation = "Dense is ArrayFire capable.";
    gpu.suggested_action = "No action needed.";

    cyxwiz::DebugRuntimeBackendClassifier classifier;
    const auto gpu_classification = classifier.Classify(gpu);
    Check(gpu_classification.proven,
          "GPU backend placement should be marked proven");
    Check(gpu_classification.status == cyxwiz::BackendPlacementStatus::Gpu,
          "GPU backend placement should preserve status");
    Check(gpu_classification.fallback_possible,
          "GPU backend placement should preserve fallback path");
    Check(!gpu_classification.needs_attention,
          "GPU backend placement should not need user attention");

    cyxwiz::DebugTraceRecord trace = cyxwiz::DebugNodeTraceContract::Make(
        "backend-classification-run",
        42,
        "DenseHead",
        "Dense",
        "Forward",
        cyxwiz::DebugTraceRole::Activation,
        {2, 3},
        {2, 4},
        "float32",
        "ArrayFire active backend",
        "ok");

    classifier.AttachToTrace(trace, gpu);
    Check(trace.payload["backend_status"].get<std::string>() ==
              cyxwiz::BackendPlacementStatus::Gpu,
          "backend trace payload should include placement status");
    Check(trace.payload["backend_reason_code"].get<std::string>() ==
              cyxwiz::BackendPlacementReason::ArrayFireTensorOpCapable,
          "backend trace payload should include reason code");
    Check(trace.payload["backend_proven"].get<bool>(),
          "backend trace payload should expose proven status");
    Check(trace.payload["backend_fallback_possible"].get<bool>(),
          "backend trace payload should expose fallback path");
    Check(!trace.payload["backend_needs_attention"].get<bool>(),
          "backend trace payload should expose attention flag");
    Check(trace.payload["warning_count"].get<size_t>() == 0,
          "healthy backend placement should not add warnings");

    cyxwiz::BackendPlacementEntry mha_cpu;
    mha_cpu.node_id = 88;
    mha_cpu.node_name = "Self MHA";
    mha_cpu.node_type = "MultiHeadAttention";
    mha_cpu.requested_backend = "auto";
    mha_cpu.expected_backend = "CPU";
    mha_cpu.fallback_backend = "CPU";
    mha_cpu.status = cyxwiz::BackendPlacementStatus::Cpu;
    mha_cpu.reason_code = cyxwiz::BackendPlacementReason::GraphRuntimeCpuBacked;
    mha_cpu.explanation =
        "MultiHeadAttention is supported as CPU-backed self-attention.";
    mha_cpu.suggested_action =
        "No correctness action needed for single-input self-attention.";
    mha_cpu.observation_source = "preflight_probe";
    mha_cpu.observation_device = "af_device=0;name=test";
    mha_cpu.observation_dtype = "float32";
    mha_cpu.observation_shape_signature = "kind=LSTM;batch=4";
    mha_cpu.observation_detail = "simulated preflight timeout";
    mha_cpu.observation_timestamp = "2026-07-08T00:00:00Z";
    mha_cpu.observation_probe_outcome = "timeout";
    mha_cpu.observation_probe_scope = "deep_preflight";

    const auto mha_classification = classifier.Classify(mha_cpu);
    Check(mha_classification.proven,
          "MHA CPU-backed placement should be marked proven");
    Check(mha_classification.status == cyxwiz::BackendPlacementStatus::Cpu,
          "MHA CPU-backed placement should preserve CPU status");
    Check(mha_classification.reason_code ==
              cyxwiz::BackendPlacementReason::GraphRuntimeCpuBacked,
          "MHA CPU-backed placement should preserve CPU-backed reason");
    Check(mha_classification.observation_source == "preflight_probe",
          "MHA CPU-backed placement should preserve observation source");
    Check(mha_classification.observation_shape_signature == "kind=LSTM;batch=4",
          "MHA CPU-backed placement should preserve observation shape");
    Check(mha_classification.observation_probe_outcome == "timeout",
          "MHA CPU-backed placement should preserve probe outcome");
    Check(mha_classification.observation_probe_scope == "deep_preflight",
          "MHA CPU-backed placement should preserve probe scope");
    Check(mha_classification.needs_attention,
          "MHA CPU-backed self-attention should require debugger attention");

    cyxwiz::DebugTraceRecord mha_trace =
        cyxwiz::DebugNodeTraceContract::Make(
            "backend-classification-run",
            88,
            "Self MHA",
            "MultiHeadAttention",
            "Forward",
            cyxwiz::DebugTraceRole::Activation,
            {1, 4, 4},
            {1, 4, 4},
            "float32",
            "CPU",
            "ok");

    classifier.AttachToTrace(mha_trace, mha_cpu);
    Check(mha_trace.payload["backend_status"].get<std::string>() ==
              cyxwiz::BackendPlacementStatus::Cpu,
          "MHA backend trace payload should expose CPU status");
    Check(mha_trace.payload["backend_reason_code"].get<std::string>() ==
              cyxwiz::BackendPlacementReason::GraphRuntimeCpuBacked,
          "MHA backend trace payload should expose CPU-backed reason");
    Check(mha_trace.payload["backend_observation_source"].get<std::string>() ==
              "preflight_probe",
          "MHA backend trace payload should expose observation source");
    Check(mha_trace.payload["backend_observation_device"].get<std::string>() ==
              "af_device=0;name=test",
          "MHA backend trace payload should expose observation device");
    Check(mha_trace.payload["backend_observation_dtype"].get<std::string>() ==
              "float32",
          "MHA backend trace payload should expose observation dtype");
    Check(mha_trace.payload["backend_observation_shape_signature"].get<std::string>() ==
              "kind=LSTM;batch=4",
          "MHA backend trace payload should expose observation shape");
    Check(mha_trace.payload["backend_observation_detail"].get<std::string>() ==
              "simulated preflight timeout",
          "MHA backend trace payload should expose observation detail");
    Check(mha_trace.payload["backend_observation_timestamp"].get<std::string>() ==
              "2026-07-08T00:00:00Z",
          "MHA backend trace payload should expose observation timestamp");
    Check(mha_trace.payload["backend_observation_probe_outcome"].get<std::string>() ==
              "timeout",
          "MHA backend trace payload should expose probe outcome");
    Check(mha_trace.payload["backend_observation_probe_scope"].get<std::string>() ==
              "deep_preflight",
          "MHA backend trace payload should expose probe scope");
    Check(mha_trace.payload["backend_proven"].get<bool>(),
          "MHA backend trace payload should expose proven status");
    Check(mha_trace.payload["backend_needs_attention"].get<bool>(),
          "MHA backend trace payload should request attention");
    Check(mha_trace.payload["warning_count"].get<size_t>() == 1,
          "MHA CPU-backed placement should add one debugger warning");
    Check(mha_trace.status == "ok",
          "MHA CPU-backed attention warning should not fail execution trace");

    cyxwiz::BackendPlacementEntry unknown;
    unknown.node_id = 7;
    unknown.node_name = "CustomNode";
    unknown.node_type = "Custom";
    unknown.expected_backend = cyxwiz::BackendPlacementStatus::Unknown;
    unknown.fallback_backend = "CPU";
    unknown.status = cyxwiz::BackendPlacementStatus::Unknown;
    unknown.reason_code =
        cyxwiz::BackendPlacementReason::BackendCapabilityUnclassified;
    unknown.explanation = "Backend capability is not classified.";
    unknown.suggested_action = "Classify this node before relying on GPU.";

    cyxwiz::DebugTraceRecord unknown_trace =
        cyxwiz::DebugNodeTraceContract::Make(
            "backend-classification-run",
            7,
            "CustomNode",
            "Custom",
            "Forward",
            cyxwiz::DebugTraceRole::Activation,
            {1},
            {1},
            "float32",
            "unknown",
            "ok");

    classifier.AttachToTrace(unknown_trace, unknown);
    Check(!unknown_trace.payload["backend_proven"].get<bool>(),
          "unknown backend placement should not be marked proven");
    Check(unknown_trace.payload["backend_needs_attention"].get<bool>(),
          "unknown backend placement should need attention");
    Check(unknown_trace.payload["warning_count"].get<size_t>() == 1,
          "unknown backend placement should add a warning");
    Check(unknown_trace.status == "ok",
          "backend attention warning should not fail execution trace");
}

void TestMemoryOwnershipTraceContract() {
    cyxwiz::TrainingTraceEvent before;
    before.cpu_allocated_bytes = 1024;
    before.cpu_peak_bytes = 2048;
    before.af_allocated_bytes = 4096;
    before.af_locked_bytes = 1024;
    before.af_alloc_buffers = 2;
    before.af_lock_buffers = 1;

    cyxwiz::TrainingTraceEvent after;
    after.cpu_allocated_bytes = 4096;
    after.cpu_peak_bytes = 9216;
    after.af_allocated_bytes = 8192;
    after.af_locked_bytes = 9216;
    after.af_alloc_buffers = 4;
    after.af_lock_buffers = 3;

    cyxwiz::DebugMemoryOwnershipInput input;
    input.node_id = 33;
    input.node_name = "DenseHead";
    input.node_type = "Dense";
    input.phase = "Forward.Memory";
    input.role = cyxwiz::DebugTraceRole::Activation;
    input.output_shape = {2, 4};
    input.dtype = "float32";
    input.backend = "ArrayFire active backend";
    input.bytes_per_element = 4;
    input.host_budget_bytes = 10 * 1024;
    input.device_budget_bytes = 10 * 1024;
    input.before = before;
    input.after = after;

    cyxwiz::DebugMemoryOwnershipTracer tracer;
    const auto trace = tracer.BuildTrace("memory-trace-run", input);

    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(trace),
          "memory trace should use canonical node trace schema");
    Check(trace.node_id == 33,
          "memory trace should preserve node id");
    Check(trace.phase == "Forward.Memory",
          "memory trace should preserve phase");
    Check(trace.payload["memory_schema"].get<std::string>() ==
              cyxwiz::DebugMemoryOwnershipTracer::kSchema,
          "memory trace should expose memory schema");
    Check(!trace.payload["ownership_proven"].get<bool>(),
          "memory trace should not claim allocator-proven ownership");
    Check(trace.payload["estimated_tensor_bytes"].get<uint64_t>() == 32,
          "memory trace should estimate output tensor bytes");
    Check(trace.payload["cpu_allocated_delta_bytes"].get<int64_t>() == 3072,
          "memory trace should include CPU allocation delta");
    Check(trace.payload["af_allocated_delta_bytes"].get<int64_t>() == 4096,
          "memory trace should include ArrayFire allocation delta");
    Check(trace.payload["af_locked_delta_bytes"].get<int64_t>() == 8192,
          "memory trace should include ArrayFire locked delta");
    Check(trace.payload["cpu_peak_increased"].get<bool>(),
          "memory trace should mark CPU peak increase");
    Check(trace.payload["device_locked_increased"].get<bool>(),
          "memory trace should mark device locked increase");
    Check(trace.payload["host_oom_risk"].get<bool>(),
          "memory trace should mark host OOM risk near budget");
    Check(trace.payload["device_oom_risk"].get<bool>(),
          "memory trace should mark device OOM risk near budget");
    Check(trace.payload["warning_count"].get<size_t>() == 2,
          "memory trace should warn for host and device risk");

    Check(cyxwiz::DebugMemoryOwnershipTracer::EstimateTensorBytes({2, 3}, 8) == 48,
          "tensor byte estimate should multiply shape by element size");
    Check(cyxwiz::DebugMemoryOwnershipTracer::EstimateTensorBytes({}, 4) == 0,
          "empty tensor shape should estimate zero bytes");
}

void TestExportCorrelationTraceContract() {
    cyxwiz::DebugExportCorrelationInput input;
    input.artifact_kind = "python";
    input.artifact_path = "exports/debug_model.py";
    input.exporter_name = "PythonCodeGenerator";
    input.graph_hash = 0xBEEF;
    input.compile_success = true;
    input.compile_status = "compiled";
    input.source_node_ids = {1, 2, 3};
    input.generated_content = "model = Sequential()\\nmodel.add(Dense(4))\\n";
    input.message = "Generated Python model code.";

    cyxwiz::DebugExportCorrelationTracer tracer;
    const auto trace = tracer.BuildTrace("export-correlation-run", input);

    Check(trace.run_id == "export-correlation-run",
          "export correlation trace should preserve run id");
    Check(trace.role == cyxwiz::DebugTraceRole::GeneratedCode,
          "export correlation trace should use GeneratedCode role");
    Check(trace.phase == "ExportCorrelation",
          "export correlation phase should be stable");
    Check(trace.status == "ok",
          "successful export correlation should be ok");
    Check(trace.payload["schema"].get<std::string>() ==
              cyxwiz::DebugExportCorrelationTracer::kSchema,
          "export correlation trace should expose schema");
    Check(trace.payload["artifact_kind"].get<std::string>() == "python",
          "export correlation trace should include artifact kind");
    Check(trace.payload["artifact_path"].get<std::string>() ==
              "exports/debug_model.py",
          "export correlation trace should include artifact path");
    Check(trace.payload["graph_hash"].get<uint64_t>() == 0xBEEF,
          "export correlation trace should include graph hash");
    Check(trace.payload["compile_success"].get<bool>(),
          "export correlation trace should include compile success");
    Check(trace.payload["source_node_ids"].is_array() &&
              trace.payload["source_node_ids"].size() == 3,
          "export correlation trace should include source node ids");
    Check(trace.payload["content_bytes"].get<size_t>() ==
              input.generated_content.size(),
          "export correlation trace should include content byte count");
    Check(trace.payload["content_fingerprint"].get<uint64_t>() ==
              cyxwiz::DebugExportCorrelationTracer::Fingerprint(
                  input.generated_content),
          "export correlation trace should include deterministic fingerprint");

    cyxwiz::DebugExportCorrelationInput failed;
    failed.artifact_kind = "onnx";
    failed.exporter_name = "ONNXExporter";
    failed.graph_hash = 0xCAFE;
    failed.compile_success = false;
    failed.compile_status = "compile failed before export";
    failed.generated_content = "";

    const auto failed_trace = tracer.BuildTrace(
        "export-correlation-run",
        failed);

    Check(failed_trace.status == "failed",
          "failed export correlation should fail the trace");
    Check(failed_trace.payload["warning_count"].get<size_t>() == 1,
          "missing export artifact path should produce a warning");
    Check(failed_trace.payload["error_count"].get<size_t>() == 1,
          "failed compile correlation should produce an error");
}

void TestWindowsCrashImportContract() {
    const std::string wer =
        "Version=1\n"
        "EventType=APPCRASH\n"
        "AppName=cyxwiz-engine.exe\n"
        "Sig[0].Name=Application Name\n"
        "Sig[0].Value=cyxwiz-engine.exe\n"
        "Sig[3].Name=Fault Module Name\n"
        "Sig[3].Value=arrayfire.dll\n"
        "Sig[6].Name=Exception Code\n"
        "Sig[6].Value=c0000005\n"
        "ReportIdentifier=wer-report-123\n"
        "EventTime=133636000000000000\n"
        "RunId=train-123\n";

    cyxwiz::DebugWindowsCrashImporter importer;
    const auto report = importer.ParseWerText(
        wer,
        "C:/ProgramData/Microsoft/Windows/WER/ReportArchive/app.wer");

    Check(report.available,
          "WER parser should mark populated reports available");
    Check(report.process_name == "cyxwiz-engine.exe",
          "WER parser should capture process name");
    Check(report.fault_module == "arrayfire.dll",
          "WER parser should capture fault module");
    Check(report.exception_code == "c0000005",
          "WER parser should capture exception code");
    Check(report.report_id == "wer-report-123",
          "WER parser should capture report id");

    cyxwiz::CrashRunSummary run;
    run.available = true;
    run.suspected_crash = true;
    run.run_id = "train-123";
    run.status = "suspected crash";
    run.last_stage = "Forward";
    run.last_event_time = "2026-06-18 13:30:00";

    const auto correlation = importer.Correlate(run, report);
    Check(correlation.matched,
          "WER correlation should match run id or CyxWiz process");

    const auto trace = importer.BuildTrace("train-123", run, report);
    Check(trace.role == cyxwiz::DebugTraceRole::Error,
          "Windows crash import trace should use Error role");
    Check(trace.phase == "WindowsCrashImport",
          "Windows crash import trace phase should be stable");
    Check(trace.status == "captured",
          "available Windows crash report should be captured");
    Check(trace.payload["schema"].get<std::string>() ==
              cyxwiz::DebugWindowsCrashImporter::kSchema,
          "Windows crash import trace should expose schema");
    Check(trace.payload["error_code"].get<std::string>() ==
              cyxwiz::DebugWindowsCrashImporter::kCrashErrorCode,
          "Windows crash import trace should include stable error code");
    Check(trace.payload["matched"].get<bool>(),
          "Windows crash import trace should include match status");
    Check(trace.payload["fault_module"].get<std::string>() ==
              "arrayfire.dll",
          "Windows crash import trace should include fault module");
    Check(trace.payload["exception_code"].get<std::string>() ==
              "c0000005",
          "Windows crash import trace should include exception code");

    const auto empty_report = importer.ParseWerText("", "");
    const auto missing_trace = importer.BuildTrace(
        "train-123",
        run,
        empty_report);
    Check(missing_trace.status == "missing",
          "missing Windows crash report should be explicit");
    Check(missing_trace.payload["warning_count"].get<size_t>() == 1,
          "missing Windows crash report should add warning");
}

void TestSupportBundleContract() {
    cyxwiz::DebugRunStoreRecord record;
    record.summary.run_id = "support-run";
    record.summary.timestamp = "2026-06-18T13:50:00";
    record.summary.graph_hash = 0xFEED;
    record.summary.success = false;
    record.summary.issue_count = 1;
    record.summary.trace_count = 1;
    record.summary.event_count = 1;
    record.summary.recommendation_count = 1;
    record.summary.summary = "Support bundle contract";
    record.summary.file_path = "C:/Users/private/.cyxwiz/debug_runs/support-run.json";

    record.issues.push_back({
        cyxwiz::IssueLevel::Error,
        5,
        "Tokenizer",
        "[CW-D-0101] required column missing",
        "CW-D-0101"
    });

    cyxwiz::DebugTraceRecord trace =
        cyxwiz::DebugNodeTraceContract::Make(
            "support-run",
            5,
            "Tokenizer",
            "TextTokenizer",
            "TextTokenizer",
            cyxwiz::DebugTraceRole::PreprocessingOutput,
            {2},
            {2, 4},
            "int64",
            "CPU",
            "failed");
    trace.payload["raw_text_preview"] = "private dataset row";
    trace.payload["source_path"] = "C:/Users/private/data.csv";
    trace.payload["error_code"] = "CW-D-0101";
    trace.issues.push_back({
        cyxwiz::IssueLevel::Error,
        5,
        "Tokenizer",
        "required column missing",
        "CW-D-0101"
    });
    record.traces.push_back(std::move(trace));

    record.studio_events.push_back({
        "support-run",
        "2026-06-18T13:50:01",
        0xFEED,
        5,
        "StudioDebugger.SelectTrace",
        "ok",
        "Selected failing trace"
    });

    record.recommendations.push_back({
        cyxwiz::DebugRecommendationSeverity::Critical,
        5,
        "Data",
        "Missing required column",
        "The text column was not found.",
        "Select a dataset with the configured text column."
    });

    cyxwiz::CrashRunSummary crash;
    crash.available = true;
    crash.suspected_crash = true;
    crash.run_id = "support-run";
    crash.status = "suspected crash";
    crash.dataset_name = "private_dataset";
    crash.backend = "CUDA";
    crash.last_stage = "Forward";
    crash.file_path = "C:/Users/private/.cyxwiz/debug_runs/current_run.json";
    crash.windows_crash_available = true;
    crash.windows_exception_code = "c0000005";
    crash.windows_report_path = "C:/ProgramData/Microsoft/Windows/WER/app.wer";

    cyxwiz::TrainingTraceSummary training;
    training.available = true;
    training.run_id = "support-run";
    training.status = "failed";
    training.latest_stage = "Forward";
    cyxwiz::TrainingTraceEvent event;
    event.run_id = "support-run";
    event.stage = "Forward";
    event.node_id = 19;
    event.node_name = "Sequence DataLoader token=secret-token";
    event.message = "operator failed token=secret-token";
    event.cpu_allocated_bytes = 1024;
    event.pin_memory_requested = true;
    event.transfer_mode =
        cyxwiz::PinMemoryTransferMode::PinnedRequestedButUnsupported;
    event.transfer_reason =
        cyxwiz::PinMemoryTransferReason::BackendUnavailable;
    event.transfer_backend = "auto";
    event.transfer_batch_size = 32;
    training.recent_events.push_back(event);

    cyxwiz::DebugSupportBundleInput input;
    input.request_id = "support-request-1";
    input.reason = "app requested engine log for HQ diagnostics";
    input.debug_run = record;
    input.crash_run = crash;
    input.training_trace = training;
    input.environment = {
        {"os", "Windows"},
        {"dataset_path", "C:/Users/private/data.csv"},
        {"backend", "CUDA"}
    };
    input.recent_logs = {
        "failed with token=secret-token",
        "normal warning"
    };
    cyxwiz::BackendPlacementObservation placement_observation;
    placement_observation.op_type = "LSTM";
    placement_observation.backend = "CUDA";
    placement_observation.device = "af_device=0;name=NVIDIA";
    placement_observation.dtype = "float32";
    placement_observation.shape_signature = "kind=LSTM;batch=64;seq=8";
    placement_observation.reason_code =
        cyxwiz::BackendPlacementObservationReason::BackendCompileTimeout;
    placement_observation.source =
        cyxwiz::BackendPlacementObservationSource::PreflightProbe;
    placement_observation.detail = "probe timed out token=secret-token";
    placement_observation.timestamp = "2026-06-18T13:50:02";
    placement_observation.probe_outcome = "timeout";
    placement_observation.probe_scope =
        cyxwiz::BackendPlacementProbeScope::DeepPreflight;
    input.placement_observations.push_back(std::move(placement_observation));
    input.allow_hq_upload = true;

    cyxwiz::DebugSupportBundleBuilder builder;
    const auto bundle = builder.Build(input);

    Check(bundle["schema"].get<std::string>() ==
              cyxwiz::DebugSupportBundleBuilder::kSchema,
          "support bundle should expose schema");
    Check(bundle["local_first"].get<bool>(),
          "support bundle should be local-first");
    Check(bundle["hq_upload_allowed"].get<bool>(),
          "support bundle should preserve explicit upload permission");
    Check(!bundle["hq_upload_performed"].get<bool>(),
          "support bundle builder should not upload");
    Check(bundle["redaction_applied"].get<bool>(),
          "support bundle should mark redaction");
    Check(bundle["debug_run"]["summary"]["file_path"].get<std::string>() ==
              "[REDACTED]",
          "support bundle should redact debug run file path");
    Check(bundle["debug_run"]["traces"][0]["payload"]["raw_text_preview"].get<std::string>() ==
              "[REDACTED]",
          "support bundle should redact dataset row previews");
    Check(bundle["debug_run"]["traces"][0]["payload"]["source_path"].get<std::string>() ==
              "[REDACTED]",
          "support bundle should redact source paths");
    Check(bundle["debug_run"]["traces"][0]["payload"]["error_code"].get<std::string>() ==
              "CW-D-0101",
          "support bundle should keep structured error codes");
    Check(bundle["debug_run"]["issues"][0]["error_code"].get<std::string>() ==
              "CW-D-0101",
          "support bundle should keep record issue error codes");
    Check(bundle["debug_run"]["traces"][0]["issues"][0]["error_code"].get<std::string>() ==
              "CW-D-0101",
          "support bundle should keep trace issue error codes");
    Check(bundle["placement_observations"].is_array() &&
              bundle["placement_observations"].size() == 1,
          "support bundle should export placement observations");
    Check(bundle["placement_observations"][0]["op_type"].get<std::string>() ==
              "LSTM",
          "support bundle should keep placement op type");
    Check(bundle["placement_observations"][0]["reason_code"].get<std::string>() ==
              cyxwiz::BackendPlacementObservationReason::BackendCompileTimeout,
          "support bundle should keep placement reason code");
    Check(bundle["placement_observations"][0]["source"].get<std::string>() ==
              cyxwiz::BackendPlacementObservationSource::PreflightProbe,
          "support bundle should keep placement source");
    Check(bundle["placement_observations"][0]["detail"].get<std::string>() ==
              "probe timed out token=[REDACTED]",
          "support bundle should redact placement observation detail");
    Check(bundle["placement_observations"][0]["probe_outcome"].get<std::string>() ==
              "timeout",
          "support bundle should keep placement probe outcome");
    Check(bundle["placement_observations"][0]["probe_scope"].get<std::string>() ==
              cyxwiz::BackendPlacementProbeScope::DeepPreflight,
          "support bundle should keep placement probe scope");
    Check(bundle["crash_run"]["dataset_name"].get<std::string>() ==
              "[REDACTED]",
          "support bundle should redact dataset names");
    Check(bundle["crash_run"]["windows_report_path"].get<std::string>() ==
              "[REDACTED]",
          "support bundle should redact WER report paths");
    Check(bundle["environment"]["dataset_path"].get<std::string>() ==
              "[REDACTED]",
          "support bundle should redact sensitive environment fields");
    Check(bundle["training_trace"]["recent_events"][0]["message"].get<std::string>() ==
              "operator failed token=[REDACTED]",
          "support bundle should redact tokens in event messages");
    Check(bundle["training_trace"]["recent_events"][0]["node_id"].get<int>() ==
              19,
          "support bundle should export training trace node id");
    Check(bundle["training_trace"]["recent_events"][0]["node_name"].get<std::string>() ==
              "Sequence DataLoader token=[REDACTED]",
          "support bundle should export redacted training trace node name");
    Check(bundle["training_trace"]["recent_events"][0]["pin_memory_requested"].get<bool>(),
          "support bundle should export pin_memory request state");
    Check(bundle["training_trace"]["recent_events"][0]["transfer_mode"].get<std::string>() ==
              cyxwiz::PinMemoryTransferMode::PinnedRequestedButUnsupported,
          "support bundle should export transfer mode");
    Check(bundle["training_trace"]["recent_events"][0]["transfer_reason"].get<std::string>() ==
              cyxwiz::PinMemoryTransferReason::BackendUnavailable,
          "support bundle should export transfer reason");
    Check(bundle["training_trace"]["recent_events"][0]["transfer_backend"].get<std::string>() ==
              "auto",
          "support bundle should export transfer backend");
    Check(bundle["training_trace"]["recent_events"][0]["transfer_batch_size"].get<int>() ==
              32,
          "support bundle should export transfer batch size");
    Check(bundle["recent_logs"][0].get<std::string>() ==
              "failed with token=[REDACTED]",
          "support bundle should redact tokens in recent logs");
}

void TestNodeInspectorSummaryContract() {
    cyxwiz::DebugTraceRecord trace = cyxwiz::DebugNodeTraceContract::Make(
        "inspector-run",
        9,
        "Scale",
        "StandardScaler",
        "Transform",
        cyxwiz::DebugTraceRole::FeatureTensor,
        {4, 2},
        {4, 2},
        "float32",
        "CPU",
        "ok");
    trace.duration_ms = 1.25f;
    cyxwiz::DebugNodeTraceContract::AddWarning(trace, "CPU fallback used");

    std::vector<cyxwiz::DebugRecommendation> recommendations = {
        {
            cyxwiz::DebugRecommendationSeverity::Warning,
            9,
            "Runtime",
            "CPU fallback",
            "The node executed on CPU.",
            "Inspect backend placement."
        },
        {
            cyxwiz::DebugRecommendationSeverity::Info,
            99,
            "Other",
            "Unrelated",
            "Different node.",
            "Ignore for this node."
        }
    };

    cyxwiz::DebugNodeInspector inspector;
    const auto summary = inspector.BuildSummary(trace, recommendations);

    Check(summary.available, "node inspector summary should be available");
    Check(summary.node_id == 9, "node inspector should preserve node id");
    Check(summary.node_name == "Scale",
          "node inspector should preserve node name");
    Check(summary.node_type == "StandardScaler",
          "node inspector should preserve node type");
    Check(summary.phase == "Transform",
          "node inspector should preserve phase");
    Check(summary.role == "FeatureTensor",
          "node inspector should expose role name");
    Check(summary.status == "ok",
          "node inspector should preserve status");
    Check(summary.dtype == "float32",
          "node inspector should preserve dtype");
    Check(summary.backend == "CPU",
          "node inspector should expose backend");
    Check(summary.input_shape == std::vector<size_t>{4, 2},
          "node inspector should preserve input shape");
    Check(summary.output_shape == std::vector<size_t>{4, 2},
          "node inspector should preserve output shape");
    Check(summary.input_rank == 2 && summary.output_rank == 2,
          "node inspector should expose ranks");
    Check(summary.input_numel == 8 && summary.output_numel == 8,
          "node inspector should expose element counts");
    Check(summary.duration_ms == 1.25f,
          "node inspector should preserve duration");
    Check(summary.issues.size() == 1,
          "node inspector should include trace issues");
    Check(summary.recommendations.size() == 1,
          "node inspector should include only related recommendations");
    Check(summary.recommendations[0].title == "CPU fallback",
          "node inspector should preserve recommendation details");
}

void TestOperatorBackedPreprocessingTraceContract() {
    auto text = FinishStringArray({
        "Small text sample",
        "Another small sample",
        "Text pipelines should tokenize",
    });
    auto label = FinishStringArray({"positive", "positive", "negative"});

    auto schema = arrow::schema({
        arrow::field("text", arrow::utf8()),
        arrow::field("label", arrow::utf8()),
    });
    auto input = arrow::Table::Make(schema, {text, label}, 3);

    cyxwiz::TextTokenizerOperator op;
    std::map<std::string, std::string> params = {
        {"text_col", "text"},
        {"label_col", "label"},
        {"tokenizer_type", "1"},
        {"max_length", "4"},
        {"lowercase", "true"},
        {"min_word_freq", "1"},
        {"max_vocab_size", "100"},
    };

    std::string error;
    Check(op.Configure(params, error), error);
    auto result = op.Apply(input);
    Check(result.ok(), result.status().ToString());
    auto output = result.ValueOrDie();

    cyxwiz::DebugOperatorTraceAdapter adapter;
    cyxwiz::DebugGraphTraceStep step = adapter.BuildStep(
        21,
        "Tokenize",
        "TextTokenizer",
        input,
        output,
        0.5f);

    cyxwiz::DebugGraphTraceExecutor executor;
    const auto traces = executor.TraceSteps("operator-trace-run", {step});
    Check(traces.size() == 1,
          "operator-backed adapter should produce one trace step");

    const auto& trace = traces[0];
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(trace),
          "operator-backed trace should use canonical schema");
    Check(trace.node_id == 21,
          "operator-backed trace should preserve node id");
    Check(trace.node_type == "TextTokenizer",
          "operator-backed trace should preserve node type");
    Check(trace.phase == "OperatorTransform",
          "operator-backed trace should use operator transform phase");
    Check(trace.role == cyxwiz::DebugTraceRole::PreprocessingOutput,
          "operator-backed trace should be preprocessing output");
    Check(trace.input_shape == std::vector<size_t>{3, 2},
          "operator-backed trace should capture input table shape");
    Check(trace.output_shape == std::vector<size_t>{3, 5},
          "operator-backed trace should capture output table shape");
    Check(trace.dtype == "arrow::Table",
          "operator-backed trace should expose Arrow table dtype");
    Check(trace.payload["backend"].get<std::string>() == "CPU",
          "operator-backed trace should expose CPU backend for Arrow operator");
    Check(trace.payload["operator"].get<std::string>() == "TextTokenizer",
          "operator-backed trace should include operator name");
    Check(trace.payload["input_rows"].get<int64_t>() == 3,
          "operator-backed trace should include input rows");
    Check(trace.payload["input_columns"].get<int64_t>() == 2,
          "operator-backed trace should include input columns");
    Check(trace.payload["output_rows"].get<int64_t>() == 3,
          "operator-backed trace should include output rows");
    Check(trace.payload["output_columns"].get<int64_t>() == 5,
          "operator-backed trace should include output columns");
    Check(trace.payload["output_schema"].get<std::string>().find("tok_0") != std::string::npos,
          "operator-backed trace should include output schema summary");
}

void TestSmokeSampleSelectionContract() {
    cyxwiz::DebugSmokeSampleSelector selector;

    const auto first = selector.SelectDeterministic(10, 4);
    Check(!first.stratified,
          "deterministic smoke sample selection should not claim stratification");
    Check(first.indices == std::vector<size_t>{0, 1, 2, 3},
          "deterministic smoke sample selection should pick stable first indexes");

    const auto capped = selector.SelectDeterministic(3, 10);
    Check(capped.indices == std::vector<size_t>{0, 1, 2},
          "deterministic smoke sample selection should cap at sample count");

    const std::vector<int> labels = {0, 0, 1, 1, 2, 2, 2};
    const auto stratified = selector.SelectStratified(labels, 5);
    Check(stratified.stratified,
          "multi-label smoke sample selection should claim stratification");
    Check(stratified.indices == std::vector<size_t>{0, 2, 4, 1, 3},
          "stratified smoke sample selection should round-robin labels deterministically");

    const auto single_label = selector.SelectStratified({7, 7, 7}, 2);
    Check(!single_label.stratified,
          "single-label smoke sample selection should fall back to deterministic");
    Check(single_label.indices == std::vector<size_t>{0, 1},
          "single-label smoke sample selection should be deterministic");

    const auto empty = selector.SelectStratified({}, 5);
    Check(empty.indices.empty(),
          "empty smoke sample selection should produce no indexes");
}

void TestRecommendationContract() {
    std::vector<cyxwiz::DebugTraceRecord> traces;

    cyxwiz::DebugTraceRecord vocab;
    vocab.node_id = 2;
    vocab.node_name = "TextVocabulary";
    vocab.phase = "TextVocabulary";
    vocab.payload["unknown_token_ratio"] = 0.25;
    traces.push_back(std::move(vocab));

    cyxwiz::DebugTraceRecord padding;
    padding.node_id = 3;
    padding.node_name = "TextPadding";
    padding.phase = "TextPadding";
    padding.payload["truncated"] = true;
    padding.payload["pad_ratio"] = 0.0;
    traces.push_back(std::move(padding));

    cyxwiz::DebugTraceRecord loss;
    loss.node_id = 4;
    loss.node_name = "SmokeRun";
    loss.phase = "SmokeRun.Loss";
    loss.payload["predictions_have_non_finite"] = true;
    loss.payload["loss"] = std::numeric_limits<double>::quiet_NaN();
    traces.push_back(std::move(loss));

    cyxwiz::DebugTraceRecord backward;
    backward.node_id = 4;
    backward.node_name = "SmokeRun";
    backward.phase = "SmokeRun.Backward";
    backward.payload["gradient_tensor_count"] = 0;
    backward.payload["zero_gradient_tensor_count"] = 0;
    traces.push_back(std::move(backward));

    cyxwiz::SmokeRunResult smoke;
    smoke.supported = true;
    smoke.success = false;
    smoke.summary = "Smoke Run failed for debugger contract test.";

    cyxwiz::CrashRunSummary last_run;
    cyxwiz::TrainingTraceSummary training_trace;

    cyxwiz::DebugRecommendationEngine engine;
    const auto recs = engine.Build(
        traces,
        {},
        smoke,
        last_run,
        training_trace);

    Check(HasRecommendation(recs, "High unknown-token ratio"),
          "high unknown-token ratio should produce recommendation");
    Check(HasRecommendation(recs, "Text sample was truncated"),
          "truncation should produce recommendation");
    Check(HasRecommendation(recs, "Smoke Run produced invalid values"),
          "invalid smoke loss should produce recommendation");
    Check(HasRecommendation(recs, "No gradients observed"),
          "missing gradients should produce recommendation");
    Check(HasRecommendation(recs, "Smoke Run needs attention"),
          "failed supported smoke run should produce recommendation");
}

void TestSmokeRunResultValueContract() {
    cyxwiz::SmokeRunResult result;

    Check(!result.supported, "default smoke result should be unsupported");
    Check(!result.success, "default smoke result should not be successful");
    Check(result.requested_samples == 100,
          "default smoke result should request 100 samples");
    Check(result.samples_seen == 0,
          "default smoke result should start with zero samples");
    Check(result.batches_seen == 0,
          "default smoke result should start with zero batches");
    Check(result.average_loss == 0.0f,
          "default smoke result should start with zero average loss");
    Check(result.last_accuracy == 0.0f,
          "default smoke result should start with zero accuracy");
    Check(result.issues.empty(), "default smoke result should have no issues");
    Check(result.traces.empty(), "default smoke result should have no traces");

    result.supported = true;
    result.success = false;
    result.summary = "Smoke Run currently supports text graphs in this slice.";
    result.issues.push_back({
        cyxwiz::IssueLevel::Error,
        -1,
        "SmokeRun",
        result.summary,
        cyxwiz::errors::Runtime::UnsupportedNode
    });

    Check(result.supported, "supported smoke result should expose support flag");
    Check(!result.success, "blocked smoke result should expose failed status");
    Check(!result.summary.empty(), "blocked smoke result should explain status");
    Check(result.issues.size() == 1,
          "blocked smoke result should carry issue details");
    Check(result.issues[0].error_code ==
              cyxwiz::errors::Runtime::UnsupportedNode,
          "blocked smoke result should preserve issue code");
}

void TestPreflightIssueCodeContract() {
    cyxwiz::TrainingConfiguration config;
    std::vector<gui::MLNode> nodes;
    std::vector<gui::NodeLink> links;

    cyxwiz::PreflightValidator preflight;
    const auto preflight_result = preflight.Validate(config, nodes, links, 0xBADC0DE);

    Check(!preflight_result.ready,
          "empty preflight should be blocked");
    Check(HasIssueCode(preflight_result.issues,
                       cyxwiz::errors::Compiler::MissingTrainingPathNode),
          "empty preflight should expose missing training-path code");
    Check(HasIssueCode(preflight_result.issues,
                       cyxwiz::errors::Runtime::InputDatasetMissing),
          "empty preflight should expose missing dataset code");
    Check(HasIssueCode(preflight_result.issues,
                       cyxwiz::errors::Compiler::TensorShapeMismatch),
          "empty preflight should expose unknown input-shape code");
    Check(HasIssueCode(preflight_result.issues,
                       cyxwiz::errors::Compiler::LabelOutputShapeMismatch),
          "empty preflight should expose unknown output-shape code");
}

void TestDebugRunStoreContract() {
    const auto original_cwd = std::filesystem::current_path();
    const auto test_root =
        std::filesystem::temp_directory_path() / "cyxwiz_debugger_contract_store";
    std::filesystem::remove_all(test_root);
    std::filesystem::create_directories(test_root);
    std::filesystem::current_path(test_root);

    cyxwiz::DebugRunStoreRecord record;
    record.summary.run_id = "store-contract-run";
    record.summary.timestamp = "2026-06-18T00:00:00";
    record.summary.graph_hash = 0xCAFE;
    record.summary.success = true;
    record.summary.summary = "Debugger store contract run";

    record.issues.push_back({
        cyxwiz::IssueLevel::Warning,
        7,
        "DebugNode",
        "Contract warning",
        "CW-C-0103"
    });

    cyxwiz::DebugTraceRecord trace;
    trace.run_id = record.summary.run_id;
    trace.node_id = 7;
    trace.node_name = "DebugNode";
    trace.node_type = "Dense";
    trace.phase = "Forward";
    trace.role = cyxwiz::DebugTraceRole::Activation;
    trace.input_shape = {2, 3};
    trace.output_shape = {2, 4};
    trace.dtype = "float32";
    trace.duration_ms = 1.5f;
    trace.status = "ok";
    trace.payload["backend"] = "CPU";
    trace.payload["max_abs"] = 0.75;
    trace.issues.push_back({
        cyxwiz::IssueLevel::Warning,
        7,
        "DebugNode",
        "Trace warning",
        "CW-C-0103"
    });
    record.traces.push_back(std::move(trace));

    record.studio_events.push_back({
        record.summary.run_id,
        "2026-06-18T00:00:01",
        0xCAFE,
        7,
        "StudioDebugger.SelectTrace",
        "ok",
        "Selected trace row"
    });

    record.recommendations.push_back({
        cyxwiz::DebugRecommendationSeverity::Warning,
        7,
        "Contract",
        "Debugger store recommendation",
        "Persisted recommendation detail",
        "Inspect persisted run"
    });

    Check(cyxwiz::DebugRunStore::Save(record),
          "debug run store should save valid records");

    auto loaded = cyxwiz::DebugRunStore::Load(record.summary.run_id);
    Check(loaded.has_value(), "debug run store should load saved records");
    Check(loaded->summary.run_id == record.summary.run_id,
          "loaded run id should round-trip");
    Check(loaded->summary.graph_hash == 0xCAFE,
          "loaded graph hash should round-trip");
    Check(loaded->summary.success, "loaded success should round-trip");
    Check(loaded->summary.issue_count == 1,
          "loaded issue count should match persisted issue count");
    Check(loaded->summary.trace_count == 1,
          "loaded trace count should match persisted trace count");
    Check(loaded->summary.event_count == 1,
          "loaded event count should match persisted event count");
    Check(loaded->summary.recommendation_count == 1,
          "loaded recommendation count should match persisted recommendation count");
    Check(loaded->issues.size() == 1 &&
              loaded->issues[0].message == "Contract warning" &&
              loaded->issues[0].error_code == "CW-C-0103",
          "issue payload and error code should round-trip");
    Check(loaded->traces.size() == 1 &&
              loaded->traces[0].role == cyxwiz::DebugTraceRole::Activation &&
              loaded->traces[0].input_shape == std::vector<size_t>{2, 3} &&
              loaded->traces[0].output_shape == std::vector<size_t>{2, 4} &&
              loaded->traces[0].payload["backend"].get<std::string>() == "CPU" &&
              loaded->traces[0].issues.size() == 1 &&
              loaded->traces[0].issues[0].error_code == "CW-C-0103",
          "trace payload and issue error code should round-trip");
    Check(loaded->studio_events.size() == 1 &&
              loaded->studio_events[0].action == "StudioDebugger.SelectTrace",
          "Studio event payload should round-trip");
    Check(loaded->recommendations.size() == 1 &&
              loaded->recommendations[0].title == "Debugger store recommendation",
          "recommendation payload should round-trip");

    const auto recent = cyxwiz::DebugRunStore::ListRecent(1);
    Check(recent.size() == 1, "ListRecent should return saved run");
    Check(recent[0].run_id == record.summary.run_id,
          "ListRecent should preserve run id");

    cyxwiz::DebugRunStoreRecord invalid;
    Check(!cyxwiz::DebugRunStore::Save(invalid),
          "debug run store should reject empty run ids");

    std::filesystem::current_path(original_cwd);
    std::filesystem::remove_all(test_root);
}

void TestTextPreprocessingTraceContract() {
    const auto test_root =
        std::filesystem::temp_directory_path() / "cyxwiz_debugger_text_trace";
    std::filesystem::remove_all(test_root);
    std::filesystem::create_directories(test_root);

    const auto csv_path = test_root / "debug_text.csv";
    {
        std::ofstream file(csv_path);
        file << "text,label\n";
        file << "Hello stable debugger trace,positive\n";
        file << "Short row,negative\n";
    }

    const std::string dataset_name = "debugger_contract_text_dataset";
    cyxwiz::DataRegistry::TextDatasetEntry entry;
    entry.source_path = csv_path.string();
    entry.text_column = "text";
    entry.label_column = "label";
    entry.has_labels = true;
    entry.tokenizer_type = 1;
    entry.max_length = 6;
    entry.lowercase = true;
    entry.do_padding = true;
    entry.do_truncation = true;
    entry.min_word_freq = 1;
    entry.max_vocab_size = -1;

    cyxwiz::DataRegistry::Instance().RegisterTextDataset(dataset_name, entry);

    cyxwiz::TrainingConfiguration config;
    config.dataset_name = dataset_name;
    config.preprocessing_domain = cyxwiz::PreprocessingDomain::Text;
    config.text_preprocessing.has_tokenizer_node = true;
    config.text_preprocessing.has_vocabulary_node = true;
    config.text_preprocessing.has_padding_node = true;
    config.text_preprocessing.tokenizer_type = 1;
    config.text_preprocessing.lowercase = true;
    config.text_preprocessing.do_padding = true;
    config.text_preprocessing.do_truncation = true;
    config.text_preprocessing.max_length = 6;
    config.text_preprocessing.min_word_freq = 1;
    config.text_preprocessing.max_vocab_size = -1;

    std::vector<gui::MLNode> nodes = {
        MakeNode(11, gui::NodeType::TextTokenizer, "Text Tokenizer"),
        MakeNode(12, gui::NodeType::TextVocabulary, "Text Vocabulary"),
        MakeNode(13, gui::NodeType::TextPadding, "Text Padding"),
    };

    cyxwiz::TextPreprocessingTracer tracer;
    const auto traces = tracer.TraceSample(config, nodes, "text-trace-run", 0);

    Check(traces.size() == 3,
          "text preprocessing tracer should emit tokenizer, vocab, and padding traces");

    const cyxwiz::DebugTraceRecord& tokenizer = traces[0];
    Check(tokenizer.run_id == "text-trace-run",
          "tokenizer trace should carry run id");
    Check(tokenizer.node_id == 11, "tokenizer trace should bind to tokenizer node");
    Check(tokenizer.phase == "TextTokenizer",
          "tokenizer trace phase should be stable");
    Check(tokenizer.role == cyxwiz::DebugTraceRole::PreprocessingOutput,
          "tokenizer trace should be preprocessing output");
    Check(tokenizer.status == "ok", "tokenizer trace should succeed");
    Check(tokenizer.payload["dataset"].get<std::string>() == dataset_name,
          "tokenizer payload should include dataset");
    Check(tokenizer.payload["sample_index"].get<size_t>() == 0,
          "tokenizer payload should include sample index");
    Check(tokenizer.payload["raw_text_preview"].get<std::string>().find("Hello") != std::string::npos,
          "tokenizer payload should include raw text preview");
    Check(tokenizer.payload["token_count"].get<size_t>() >= 4,
          "tokenizer payload should include token count");
    Check(tokenizer.payload["tokens_preview"].is_array(),
          "tokenizer payload should include token preview");

    const cyxwiz::DebugTraceRecord& vocab = traces[1];
    Check(vocab.node_id == 12, "vocab trace should bind to vocab node");
    Check(vocab.phase == "TextVocabulary", "vocab trace phase should be stable");
    Check(vocab.payload["vocab_size"].get<size_t>() > 0,
          "vocab payload should include vocab size");
    Check(vocab.payload["vocab_hits"].get<int>() > 0,
          "vocab payload should include hits");
    Check(vocab.payload["unknown_token_count"].get<int>() >= 0,
          "vocab payload should include unknown-token count");
    Check(vocab.payload["unknown_token_ratio"].get<float>() >= 0.0f,
          "vocab payload should include unknown-token ratio");
    Check(vocab.payload["token_ids_preview"].is_array(),
          "vocab payload should include token id preview");

    const cyxwiz::DebugTraceRecord& padding = traces[2];
    Check(padding.node_id == 13, "padding trace should bind to padding node");
    Check(padding.phase == "TextPadding", "padding trace phase should be stable");
    Check(padding.output_shape == std::vector<size_t>{6},
          "padding output shape should reflect configured max length");
    Check(padding.payload["max_length"].get<int>() == 6,
          "padding payload should include max length");
    Check(padding.payload["final_sequence_length"].get<size_t>() == 6,
          "padding payload should include final sequence length");
    Check(padding.payload["pad_count"].get<int>() >= 0,
          "padding payload should include pad count");
    Check(padding.payload["pad_ratio"].get<float>() >= 0.0f,
          "padding payload should include pad ratio");
    Check(padding.payload.contains("truncated"),
          "padding payload should include truncation flag");

    cyxwiz::DataRegistry::Instance().UnregisterTextDataset(dataset_name);
    const auto missing_traces =
        tracer.TraceSample(config, nodes, "missing-text-trace-run", 0);
    Check(missing_traces.size() == 1,
          "text preprocessing tracer should report missing dataset trace");
    Check(missing_traces[0].payload["error_code"].get<std::string>() ==
              cyxwiz::errors::Runtime::InputDatasetMissing,
          "missing text dataset trace should expose runtime input code");
    Check(!missing_traces[0].issues.empty() &&
              missing_traces[0].issues[0].error_code ==
                  cyxwiz::errors::Runtime::InputDatasetMissing,
          "missing text dataset issue should expose runtime input code");
    std::filesystem::remove_all(test_root);
}

} // namespace

int main() {
    TestDebugSessionSnapshotContract();
    TestNodeTraceContract();
    TestGraphTraceExecutionSlice();
    TestRuntimeBackendClassificationContract();
    TestMemoryOwnershipTraceContract();
    TestExportCorrelationTraceContract();
    TestWindowsCrashImportContract();
    TestSupportBundleContract();
    TestNodeInspectorSummaryContract();
    TestOperatorBackedPreprocessingTraceContract();
    TestSmokeSampleSelectionContract();
    TestRecommendationContract();
    TestSmokeRunResultValueContract();
    TestPreflightIssueCodeContract();
    TestDebugRunStoreContract();
    TestTextPreprocessingTraceContract();
    std::cout << "Debugger contract tests passed\n";
    return 0;
}
