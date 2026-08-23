#include "../src/core/debug_recommendation_engine.h"
#include "../src/core/debug_batch_inspector.h"
#include "../src/core/debug_error_code_timeline.h"
#include "../src/core/debug_gradient_health_analyzer.h"
#include "../src/core/debug_slow_path_detector.h"
#include "../src/core/debug_training_stall_detector.h"
#include "../src/core/debug_training_graph_diff.h"
#include "../src/core/debug_export_correlation_tracer.h"
#include "../src/core/debug_graph_trace_executor.h"
#include "../src/core/debug_memory_ownership_tracer.h"
#include "../src/core/debug_node_inspector.h"
#include "../src/core/debug_operator_trace_adapter.h"
#include "../src/core/debug_operator_trace_producer.h"
#include "../src/core/debug_run_paths.h"
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

int CountRecommendationsWithDetail(
    const std::vector<cyxwiz::DebugRecommendation>& recs,
    const std::string& title,
    const std::string& detail) {
    int count = 0;
    for (const auto& rec : recs) {
        if (rec.title == title && rec.detail == detail) {
            ++count;
        }
    }
    return count;
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
    Check(trace.payload["success"].get<bool>(),
          "graph snapshot trace should mark success");
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(trace),
          "snapshot trace should use canonical node trace schema");
    Check(trace.payload["diagnostic_phase"].get<std::string>() ==
              "graph_snapshot",
          "snapshot trace should expose diagnostic phase");
    Check(trace.payload["component"].get<std::string>() ==
              "DebugSessionManager",
          "snapshot trace should expose diagnostic component");
    Check(trace.payload["source_file"].get<std::string>().find(
              "debug_session_manager.cpp") != std::string::npos,
          "snapshot trace should expose diagnostic source file");
    Check(trace.payload["source_symbol"].get<std::string>() ==
              "cyxwiz::DebugSessionManager::BuildGraphSnapshotTrace",
          "snapshot trace should expose diagnostic source symbol");
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
    Check(trace.payload["node_trace_schema"].get<std::string>() ==
              cyxwiz::DebugNodeTraceContract::kSchema,
          "node trace payload should include canonical node schema marker");
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
    Check(trace.payload["issue_count"].get<size_t>() == 1,
          "node trace warning should expose issue count");
    Check(trace.payload["primary_warning_code"].get<std::string>() ==
              cyxwiz::errors::Gpu::KernelExecutionFailed,
          "node trace warning should expose primary warning code");
    Check(trace.payload["issue_codes"][0].get<std::string>() ==
              cyxwiz::errors::Gpu::KernelExecutionFailed,
          "node trace warning should expose issue code summary");
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
    Check(trace.payload["issue_count"].get<size_t>() == 2,
          "node trace error should update aggregate issue count");
    Check(trace.payload["primary_error_code"].get<std::string>() ==
              cyxwiz::errors::Compiler::TensorShapeMismatch,
          "node trace error should expose primary error code");
    Check(trace.payload["issue_codes"].size() == 2,
          "node trace error should preserve distinct issue codes");
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
    output.errors.push_back("Output sink rejected shape");
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
    Check(traces[0].payload["success"].get<bool>(),
          "ok graph trace should mark success");
    Check(traces[0].payload["diagnostic_phase"].get<std::string>() ==
              "graph_trace_step",
          "graph trace should expose executor diagnostic phase");
    Check(traces[0].payload["component"].get<std::string>() ==
              "DebugGraphTraceExecutor",
          "graph trace should expose executor component");
    Check(traces[0].payload["source_file"].get<std::string>().find(
              "debug_graph_trace_executor.cpp") != std::string::npos,
          "graph trace should expose executor source file");
    Check(traces[0].payload["source_symbol"].get<std::string>() ==
              "cyxwiz::DebugGraphTraceExecutor::TraceSteps",
          "graph trace should expose executor source symbol");

    Check(traces[1].node_id == 2,
          "graph trace should preserve transform node id");
    Check(traces[1].phase == "Transform",
          "graph trace should preserve transform phase");
    Check(traces[1].payload["operator"].get<std::string>() == "StandardScaler",
          "graph trace should preserve operator payload");
    Check(traces[1].payload["warning_count"].get<size_t>() == 1,
          "graph trace should count warnings");
    Check(traces[1].payload["issue_count"].get<size_t>() == 1,
          "warning graph trace should expose issue count");
    Check(traces[1].payload["primary_warning_code"].get<std::string>() ==
              cyxwiz::errors::Runtime::ExecutionFailed,
          "warning graph trace should expose primary warning code");
    Check(traces[1].payload["issue_codes"][0].get<std::string>() ==
              cyxwiz::errors::Runtime::ExecutionFailed,
          "warning graph trace should expose issue code summary");
    Check(traces[1].status == "ok",
          "warning-only graph trace should remain ok");
    Check(traces[1].payload["success"].get<bool>(),
          "warning-only ok graph trace should preserve success");
    Check(traces[1].duration_ms == 0.25f,
          "graph trace should preserve duration");

    Check(traces[2].node_type == "DataOutput",
          "graph trace should preserve output node type");
    Check(traces[2].input_shape == std::vector<size_t>{3, 2},
          "graph trace should preserve output input shape");
    Check(traces[2].output_shape == std::vector<size_t>{3, 2},
          "graph trace should preserve output output shape");
    Check(traces[2].status == "failed",
          "error graph trace should fail the trace");
    Check(!traces[2].payload["success"].get<bool>(),
          "failed graph trace should mark unsuccessful");
    Check(traces[2].payload["issue_count"].get<size_t>() == 1,
          "error graph trace should expose issue count");
    Check(traces[2].payload["error_count"].get<size_t>() == 1,
          "error graph trace should expose error count");
    Check(traces[2].payload["primary_error_code"].get<std::string>() ==
              cyxwiz::errors::Runtime::ExecutionFailed,
          "error graph trace should expose primary error code");
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
    Check(gpu_classification.evidence_scope == "compiler_capability",
          "GPU placement without observations should remain capability evidence");
    Check(!gpu_classification.actual_backend_observed &&
              gpu_classification.actual_backend == "unobserved",
          "compiler placement must not claim an actual runtime backend");

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
    Check(trace.payload["diagnostic_phase"].get<std::string>() ==
              "backend_placement",
          "backend trace should expose diagnostic phase");
    Check(trace.payload["component"].get<std::string>() ==
              "DebugRuntimeBackendClassifier",
          "backend trace should expose diagnostic component");
    Check(trace.payload["source_file"].get<std::string>().find(
              "debug_runtime_backend_classifier.cpp") != std::string::npos,
          "backend trace should expose diagnostic source file");
    Check(trace.payload["source_symbol"].get<std::string>() ==
              "cyxwiz::DebugRuntimeBackendClassifier::AttachToTrace",
          "backend trace should expose diagnostic source symbol");
    Check(trace.payload["backend_proven"].get<bool>(),
          "backend trace payload should expose proven status");
    Check(trace.payload["backend_fallback_possible"].get<bool>(),
          "backend trace payload should expose fallback path");
    Check(!trace.payload["backend_needs_attention"].get<bool>(),
          "backend trace payload should expose attention flag");
    Check(trace.payload["success"].get<bool>(),
          "healthy backend trace should mark success");
    Check(trace.payload["warning_count"].get<size_t>() == 0,
          "healthy backend placement should not add warnings");
    Check(trace.payload["backend_intended"].get<std::string>() ==
              "ArrayFire active backend",
          "backend trace should expose intended placement");
    Check(trace.payload["backend_actual"].get<std::string>() == "unobserved" &&
              !trace.payload["backend_actual_observed"].get<bool>(),
          "backend trace should not relabel placement as runtime fact");
    Check(!trace.payload["backend_cost_estimate_available"].get<bool>(),
          "backend trace should not invent an unavailable cost estimate");

    const auto placement_trace = classifier.BuildPlacementTrace(
        "backend-audit-run", 12345, gpu);
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(placement_trace),
          "placement audit should use the canonical node trace schema");
    Check(placement_trace.run_id == "backend-audit-run" &&
              placement_trace.node_id == gpu.node_id,
          "placement audit should preserve run and graph-node identity");
    Check(placement_trace.phase == "BackendPlacement" &&
              placement_trace.payload["graph_hash"].get<uint64_t>() == 12345,
          "placement audit should expose phase and graph correlation");
    Check(placement_trace.payload["trace_producer"].get<std::string>() ==
              "DebugRuntimeBackendClassifier",
          "placement audit should expose its existing classifier producer");

    auto prior_fallback = gpu;
    prior_fallback.expected_backend = "CPU";
    prior_fallback.status = cyxwiz::BackendPlacementStatus::Cpu;
    prior_fallback.observation_source =
        cyxwiz::BackendPlacementObservationSource::RuntimeFallback;
    prior_fallback.observation_device = "af_device=0;name=prior-device";
    const auto prior_fallback_trace = classifier.BuildPlacementTrace(
        "backend-audit-run", 12345, prior_fallback);
    Check(prior_fallback_trace.payload[
              "backend_prior_runtime_fallback_observed"].get<bool>(),
          "cached runtime fallback should be identified as prior evidence");
    Check(prior_fallback_trace.payload["backend_evidence_scope"].get<std::string>() ==
              "prior_runtime_fallback",
          "cached runtime fallback should expose its historical evidence scope");
    Check(!prior_fallback_trace.payload[
              "backend_fallback_observed_this_run"].get<bool>() &&
              !prior_fallback_trace.payload["backend_actual_observed"].get<bool>(),
          "cached fallback must not be claimed as current-run execution fact");

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
    Check(mha_classification.evidence_scope == "preflight_probe",
          "preflight observation should remain preflight-scoped evidence");

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
    Check(mha_trace.payload["issue_count"].get<size_t>() == 1,
          "MHA CPU-backed placement should expose issue count");
    Check(mha_trace.payload["primary_warning_code"].get<std::string>() ==
              cyxwiz::errors::Runtime::ExecutionFailed,
          "MHA CPU-backed placement should expose primary warning code");
    Check(mha_trace.status == "ok",
          "MHA CPU-backed attention warning should not fail execution trace");
    Check(mha_trace.payload["success"].get<bool>(),
          "MHA CPU-backed attention warning should preserve trace success");

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
    Check(unknown_trace.payload["success"].get<bool>(),
          "unknown backend attention warning should preserve trace success");

    const auto unknown_placement_trace = classifier.BuildPlacementTrace(
        "backend-audit-run", 12345, unknown);
    cyxwiz::DebugTraceRecord observed_actual =
        cyxwiz::DebugNodeTraceContract::Make(
            "backend-audit-run",
            42,
            "DenseHead",
            "Dense",
            "Forward",
            cyxwiz::DebugTraceRole::Activation,
            {2, 3}, {2, 4}, "float32", "LocalDebug", "ok");
    observed_actual.duration_ms = 4.5f;
    observed_actual.payload["backend_actual"] = "arrayfire_cuda";
    observed_actual.payload["backend_actual_observed"] = true;
    observed_actual.payload["backend_fallback_observed_this_run"] = true;

    auto stale_actual = observed_actual;
    stale_actual.run_id = "another-run";
    stale_actual.payload["backend_actual"] = "native_cpu";

    cyxwiz::DebugRunExecutionSummary linked_execution;
    linked_execution.available = true;
    linked_execution.training_run_id = "linked-training-run";
    linked_execution.requested_backend = "arrayfire_cuda";
    linked_execution.effective_backend = "arrayfire_cuda";
    linked_execution.effective_device_id = 0;
    linked_execution.effective_device_name = "Test CUDA Device";
    linked_execution.residency_verdict =
        "compatibility_mode_no_fallback_observed";

    const auto audit = classifier.BuildAuditTrace(
        "backend-audit-run",
        {placement_trace, unknown_placement_trace,
         stale_actual, observed_actual},
        linked_execution);
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(audit) &&
              audit.phase == "BackendDecisionAudit",
          "backend audit should use the canonical trace contract");
    Check(audit.payload["backend_decision_audit_schema"].get<std::string>() ==
              cyxwiz::DebugRuntimeBackendClassifier::kAuditSchema,
          "backend audit schema should be stable");
    Check(audit.payload["placement_count"].get<size_t>() == 2 &&
              audit.payload["retained_row_count"].get<size_t>() == 2,
          "backend audit should retain each placement once");
    Check(audit.payload["actual_backend_observed_count"].get<size_t>() == 1 &&
              audit.payload["same_run_fallback_count"].get<size_t>() == 1,
          "backend audit should count explicit same-run runtime evidence");
    Check(audit.payload["unknown_count"].get<size_t>() == 1 &&
              audit.payload["attention_count"].get<size_t>() == 1,
          "backend audit should count unknown attention placements");
    Check(audit.payload["linked_execution_scope"].get<std::string>() ==
              "linked_training_run" &&
              !audit.payload["linked_context_is_node_actual_evidence"].get<bool>(),
          "linked execution context must remain separate from node actual evidence");
    Check(!audit.payload["cost_estimate_available"].get<bool>() &&
              !audit.payload["tensor_reads_added"].get<bool>() &&
              !audit.payload["raw_tensor_values_included"].get<bool>(),
          "backend audit should not invent cost or add Tensor reads/values");

    const auto& audit_rows = audit.payload["rows"];
    const auto dense_row = std::find_if(
        audit_rows.begin(), audit_rows.end(),
        [](const nlohmann::json& row) {
            return row.value("node_id", -1) == 42;
        });
    Check(dense_row != audit_rows.end(),
          "backend audit should retain Dense placement row");
    Check((*dense_row)["intended_backend"] == "ArrayFire active backend" &&
              (*dense_row)["actual_backend"] == "arrayfire_cuda" &&
              (*dense_row)["actual_backend_observed"].get<bool>(),
          "backend audit should distinguish intended and observed backend");
    Check((*dense_row)["fallback_observed_this_run"].get<bool>() &&
              (*dense_row)["observed_duration_available"].get<bool>() &&
              std::abs((*dense_row)["observed_duration_ms"].get<float>() -
                       4.5f) < 1.0e-6f,
          "backend audit should retain same-run fallback and timing evidence");
    Check(!(*dense_row)["cost_estimate_available"].get<bool>() &&
              (*dense_row)["cost_estimate"] == "unavailable",
          "observed duration must not be relabeled as estimated cost");

    std::vector<cyxwiz::DebugTraceRecord> many_placements;
    for (size_t i = 0;
         i < cyxwiz::DebugRuntimeBackendClassifier::kMaxAuditRows + 10;
         ++i) {
        auto placement = gpu;
        placement.node_id = 1000 + static_cast<int>(i);
        placement.node_name = "Layer " + std::to_string(i);
        many_placements.push_back(classifier.BuildPlacementTrace(
            "bounded-backend-audit", 12345, placement));
    }
    const cyxwiz::DebugRunExecutionSummary no_execution;
    const auto bounded_audit = classifier.BuildAuditTrace(
        "bounded-backend-audit", many_placements, no_execution);
    Check(bounded_audit.payload["placement_count"].get<size_t>() ==
              cyxwiz::DebugRuntimeBackendClassifier::kMaxAuditRows + 10 &&
              bounded_audit.payload["rows"].size() ==
                  cyxwiz::DebugRuntimeBackendClassifier::kMaxAuditRows &&
              bounded_audit.payload["rows_truncated"].get<bool>(),
          "backend audit should bound row detail while keeping exact counts");
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
    Check(trace.payload["success"].get<bool>(),
          "memory warning trace should preserve successful observation status");
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
    Check(trace.payload["diagnostic_phase"].get<std::string>() ==
              "memory_ownership",
          "memory trace should expose diagnostic phase");
    Check(trace.payload["component"].get<std::string>() ==
              "DebugMemoryOwnershipTracer",
          "memory trace should expose diagnostic component");
    Check(trace.payload["source_file"].get<std::string>().find(
              "debug_memory_ownership_tracer.cpp") != std::string::npos,
          "memory trace should expose diagnostic source file");
    Check(trace.payload["source_symbol"].get<std::string>() ==
              "cyxwiz::DebugMemoryOwnershipTracer::BuildTrace",
          "memory trace should expose diagnostic source symbol");
    Check(trace.payload["issue_count"].get<size_t>() == 2,
          "memory trace should expose warning issue count");
    Check(trace.payload["primary_warning_code"].get<std::string>() ==
              cyxwiz::errors::Runtime::ExecutionFailed,
          "memory trace should expose primary warning code");

    Check(cyxwiz::DebugMemoryOwnershipTracer::EstimateTensorBytes({2, 3}, 8) == 48,
          "tensor byte estimate should multiply shape by element size");
    Check(cyxwiz::DebugMemoryOwnershipTracer::EstimateTensorBytes({}, 4) == 0,
          "empty tensor shape should estimate zero bytes");

    auto source = MakeNode(1, gui::NodeType::DataInput, "Source");
    auto dense = MakeNode(2, gui::NodeType::Dense, "Dense");
    auto loss = MakeNode(
        3, gui::NodeType::CrossEntropyLoss, "Cross Entropy");
    gui::NodeLink source_to_dense;
    source_to_dense.from_node = 1;
    source_to_dense.to_node = 2;
    gui::NodeLink dense_to_loss;
    dense_to_loss.from_node = 2;
    dense_to_loss.to_node = 3;
    const auto debug_session = cyxwiz::DebugSessionManager::StartSession(
        "tensor-lifecycle-run", "FullWorkflow", 0x5151,
        {source, dense, loss}, {source_to_dense, dense_to_loss}, 0);
    std::vector<cyxwiz::DebugTraceRecord> lifecycle_sources =
        debug_session.traces;

    auto activation = cyxwiz::DebugNodeTraceContract::Make(
        "tensor-lifecycle-run", 2, "Dense", "Dense", "Forward",
        cyxwiz::DebugTraceRole::Activation,
        {2, 3}, {2, 4}, "float32", "LocalDebug", "ok");
    activation.payload["numeric_element_count"] = 8;
    activation.payload["numeric_host_read_performed"] = true;
    lifecycle_sources.push_back(std::move(activation));

    auto loss_trace = cyxwiz::DebugNodeTraceContract::Make(
        "tensor-lifecycle-run", 3, "Cross Entropy", "Loss", "Loss",
        cyxwiz::DebugTraceRole::Loss,
        {}, {}, "float32", "LocalDebug", "ok");
    loss_trace.payload["prediction_shape"] = {2, 4};
    loss_trace.payload["target_shape"] = {2};
    lifecycle_sources.push_back(std::move(loss_trace));

    auto gradient = cyxwiz::DebugNodeTraceContract::Make(
        "tensor-lifecycle-run", 2, "Dense", "Parameter", "Backward",
        cyxwiz::DebugTraceRole::Gradient,
        {}, {}, "float32", "LocalDebug", "ok");
    gradient.payload["numeric_element_count"] = 12;
    gradient.payload["numeric_host_read_performed"] = true;
    gradient.payload["parameter_name"] = "layer0.weight";
    lifecycle_sources.push_back(std::move(gradient));

    lifecycle_sources.push_back(cyxwiz::DebugNodeTraceContract::Make(
        "tensor-lifecycle-run", -1, "Optimizer", "Optimizer", "OptimizerStep",
        cyxwiz::DebugTraceRole::OptimizerStep,
        {}, {}, "optimizer", "LocalDebug", "ok"));

    cyxwiz::DebugRunExecutionSummary execution;
    execution.available = true;
    execution.training_run_id = "tensor-lifecycle-run";
    execution.effective_backend = "CUDA";
    execution.effective_device_id = 0;
    execution.effective_device_name = "Test GPU";
    execution.execution_context_id = "ctx-test";
    const auto lifecycle = tracer.BuildTensorLifecycleTrace(
        "tensor-lifecycle-run", lifecycle_sources, execution);
    const auto& lifecycle_payload = lifecycle.payload;
    Check(lifecycle.phase == "TensorLifecycle" &&
              lifecycle.status == "captured" &&
              lifecycle_payload["tensor_lifecycle_schema"] ==
                  cyxwiz::DebugMemoryOwnershipTracer::kLifecycleSchema,
          "tensor lifecycle should use a stable captured trace contract");
    Check(lifecycle_payload["entry_count"].get<size_t>() == 5 &&
              lifecycle_payload["retained_entry_count"].get<size_t>() == 5,
          "tensor lifecycle should derive model input, prediction, target, loss, and gradient");
    Check(lifecycle_payload["relation_counts"]["model_input"].get<size_t>() == 1 &&
              lifecycle_payload["relation_counts"]["prediction"].get<size_t>() == 1 &&
              lifecycle_payload["relation_counts"]["target"].get<size_t>() == 1 &&
              lifecycle_payload["relation_counts"]["loss"].get<size_t>() == 1 &&
              lifecycle_payload["relation_counts"]["gradient"].get<size_t>() == 1,
          "tensor lifecycle should classify training relations");
    Check(lifecycle_payload["estimated_total_bytes"].get<uint64_t>() == 116,
          "tensor lifecycle should estimate bounded metadata bytes without tensor reads");
    Check(lifecycle_payload["runtime_backend_observed"].get<bool>() &&
              lifecycle_payload["effective_backend"] == "CUDA" &&
              lifecycle_payload["effective_device"] == "Test GPU",
          "tensor lifecycle should reuse observed execution backend truth");
    Check(!lifecycle_payload["allocator_lifecycle_observed"].get<bool>() &&
              !lifecycle_payload["retained_freed_state_observed"].get<bool>() &&
              !lifecycle_payload["host_reads_added"].get<bool>() &&
              !lifecycle_payload["raw_tensor_values_included"].get<bool>(),
          "derived lifecycle should not claim allocator truth or add tensor reads");
    Check(lifecycle_payload["entries"][0]["origin_node_name"] ==
              "Local Debug batch" &&
              lifecycle_payload["entries"][0]["consumers"][0]["node_id"] == 2,
          "model input should originate at the batch boundary and feed the first layer");
    Check(lifecycle_payload["entries"][1]["consumer_evidence"] ==
              "graph_snapshot_topology" &&
              lifecycle_payload["entries"][1]["consumers"][0]["node_id"] == 3,
          "activation lifecycle should derive consumers from frozen topology");
    Check(lifecycle_payload["entries"][4]["parameter_name"] ==
              "layer0.weight" &&
              lifecycle_payload["entries"][4]["consumer_evidence"] ==
                  "canonical_trace_sequence",
          "gradient lifecycle should identify its parameter and optimizer consumer");

    cyxwiz::DebugRunExecutionSummary unobserved_execution;
    const auto unobserved_lifecycle = tracer.BuildTensorLifecycleTrace(
        "tensor-lifecycle-empty", {}, unobserved_execution);
    Check(unobserved_lifecycle.status == "unobserved" &&
              unobserved_lifecycle.payload["entry_count"].get<size_t>() == 0 &&
              !unobserved_lifecycle.payload["runtime_backend_observed"].get<bool>(),
          "missing lifecycle evidence should remain explicitly unobserved");

    auto mismatched_execution = execution;
    mismatched_execution.training_run_id = "different-training-run";
    const auto mismatched_lifecycle = tracer.BuildTensorLifecycleTrace(
        "tensor-lifecycle-run", lifecycle_sources, mismatched_execution);
    Check(!mismatched_lifecycle.payload["runtime_backend_observed"].get<bool>() &&
              mismatched_lifecycle.payload["effective_backend"] == "unobserved" &&
              mismatched_lifecycle.payload["runtime_backend_evidence_scope"] ==
                  "unobserved",
          "lifecycle should reject backend evidence from a different run");

    std::vector<cyxwiz::DebugTraceRecord> many_lifecycle_sources;
    many_lifecycle_sources.reserve(140);
    for (int node_id = 0; node_id < 140; ++node_id) {
        many_lifecycle_sources.push_back(cyxwiz::DebugNodeTraceContract::Make(
            "tensor-lifecycle-bounded", node_id,
            "Layer " + std::to_string(node_id), "Dense", "Forward",
            cyxwiz::DebugTraceRole::Activation,
            {}, {1}, "float32", "LocalDebug", "ok"));
    }
    const auto bounded_lifecycle = tracer.BuildTensorLifecycleTrace(
        "tensor-lifecycle-bounded", many_lifecycle_sources,
        unobserved_execution);
    Check(bounded_lifecycle.payload["entry_count"].get<size_t>() == 140 &&
              bounded_lifecycle.payload["retained_entry_count"].get<size_t>() == 128 &&
              bounded_lifecycle.payload["entries_truncated"].get<bool>(),
          "tensor lifecycle should retain exact counts with bounded detail");
}

void TestGradientHealthAnalyzerContract() {
    auto make_gradient = [](
        const std::string& run_id,
        int node_id,
        const std::string& node_name,
        const std::string& parameter_name,
        double parameter_norm,
        bool has_gradient,
        double gradient_norm,
        bool is_zero,
        bool is_nan,
        bool is_inf,
        bool update_observed,
        double update_norm,
        const std::string& missing_reason = std::string{}) {
        auto trace = cyxwiz::DebugNodeTraceContract::Make(
            run_id, node_id, node_name, "Parameter", "Backward",
            cyxwiz::DebugTraceRole::Gradient,
            {}, {}, "float32", "LocalDebug",
            !has_gradient ? "missing_gradient"
                : (is_nan ? "nan" : (is_zero ? "zero" : "ok")));
        trace.payload["parameter_name"] = parameter_name;
        trace.payload["module_index"] = node_id;
        trace.payload["compiled_layer_index"] = node_id;
        trace.payload["parameter_l2_norm"] = parameter_norm;
        trace.payload["has_gradient"] = has_gradient;
        trace.payload["gradient_l2_norm"] = gradient_norm;
        trace.payload["l2_norm"] = gradient_norm;
        trace.payload["is_zero"] = is_zero;
        trace.payload["is_nan"] = is_nan;
        trace.payload["is_inf"] = is_inf;
        trace.payload["update_observed"] = update_observed;
        trace.payload["update_l2_norm"] = update_norm;
        trace.payload["missing_gradient_reason"] = missing_reason;
        return trace;
    };

    std::vector<cyxwiz::DebugTraceRecord> sources;
    sources.push_back(make_gradient(
        "gradient-health-run", 10, "Dense A", "layer0.weight",
        3.0, true, 0.3, false, false, false, true, 0.03));
    sources.push_back(make_gradient(
        "gradient-health-run", 10, "Dense A", "layer0.bias",
        4.0, true, 0.4, false, false, false, true, 0.04));
    sources.push_back(make_gradient(
        "gradient-health-run", 20, "Dense Missing", "layer1.weight",
        2.0, false, 0.0, true, false, false, false, 0.0,
        "No gradient tensor matched this trainable parameter after backward."));
    sources.push_back(make_gradient(
        "gradient-health-run", 30, "Dense Partial", "layer2.weight",
        2.0, true, 0.2, false, false, false, true, 0.02));
    sources.push_back(make_gradient(
        "gradient-health-run", 30, "Dense Partial", "layer2.bias",
        1.0, true, 0.0, true, false, false, true, 0.0));
    sources.push_back(make_gradient(
        "gradient-health-run", 40, "Dense Zero", "layer3.weight",
        1.0, true, 0.0, true, false, false, true, 0.0));
    sources.push_back(make_gradient(
        "gradient-health-run", 50, "Dense NaN", "layer4.weight",
        1.0, true, 0.0, false, true, false, false, 0.0));
    sources.push_back(make_gradient(
        "gradient-health-run", 60, "Dense No Update", "layer5.weight",
        1.0, true, 0.5, false, false, false, true, 0.0));
    sources.push_back(make_gradient(
        "gradient-health-run", 70, "Dense Partial Evidence", "layer6.weight",
        1.0, true, 0.5, false, false, false, false, 0.0));
    sources.push_back(make_gradient(
        "different-run", 80, "Foreign Layer", "layer7.weight",
        1.0, true, 0.5, false, false, false, true, 0.1));

    cyxwiz::DebugGradientHealthAnalyzer analyzer;
    const auto summary = analyzer.BuildTrace(
        "gradient-health-run", sources);
    const auto& payload = summary.payload;
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(summary) &&
              summary.phase == "GradientHealth" &&
              payload["gradient_health_schema"] ==
                  cyxwiz::DebugGradientHealthAnalyzer::kSchema,
          "gradient health should use canonical and stable schemas");
    Check(summary.status == "needs_attention" &&
              summary.role == cyxwiz::DebugTraceRole::Gradient &&
              !payload["success"].get<bool>(),
          "attention rows should make the aggregate visibly need attention");
    Check(payload["layer_count"].get<size_t>() == 7 &&
              payload["retained_layer_count"].get<size_t>() == 7 &&
              payload["parameter_trace_count"].get<size_t>() == 9,
          "gradient health should aggregate only same-run parameter traces per layer");
    Check(payload["healthy_layer_count"].get<size_t>() == 1 &&
              payload["attention_layer_count"].get<size_t>() == 5 &&
              payload["unobserved_layer_count"].get<size_t>() == 1,
          "gradient health should retain exact layer outcome counts");

    const auto& healthy = payload["layers"][0];
    Check(healthy["status"] == "healthy" &&
              healthy["parameter_tensor_count"].get<size_t>() == 2 &&
              healthy["gradient_tensor_count"].get<size_t>() == 2 &&
              healthy["update_observation_count"].get<size_t>() == 2,
          "healthy layer should expose complete parameter, gradient, and update coverage");
    Check(std::abs(healthy["parameter_l2_norm"].get<double>() - 5.0) < 1e-12 &&
              std::abs(healthy["gradient_l2_norm"].get<double>() - 0.5) < 1e-12 &&
              std::abs(healthy["grad_parameter_ratio"].get<double>() - 0.1) < 1e-12 &&
              std::abs(healthy["update_l2_norm"].get<double>() - 0.05) < 1e-12 &&
              std::abs(healthy["update_parameter_ratio"].get<double>() - 0.01) < 1e-12,
          "layer norms and ratios should aggregate by squared L2 composition");

    const auto& missing = payload["layers"][1];
    Check(missing["status"] == "missing_gradient" &&
              missing["missing_gradient_count"].get<size_t>() == 1 &&
              missing["missing_gradient_explanations"][0]["parameter_name"] ==
                  "layer1.weight" &&
              !missing["gradient_norm_complete"].get<bool>(),
          "missing gradients should retain a parameter-specific explanation");
    Check(payload["layers"][2]["status"] == "partial_zero_gradient" &&
              payload["layers"][3]["status"] == "zero_gradient" &&
              payload["layers"][4]["status"] == "non_finite" &&
              payload["layers"][5]["status"] == "zero_update" &&
              payload["layers"][6]["status"] == "partial_evidence",
          "gradient health should distinguish partial-zero, all-zero, non-finite, zero-update, and partial-evidence states");
    Check(!payload["tensor_reads_added"].get<bool>() &&
              !payload["raw_tensor_values_included"].get<bool>(),
          "gradient aggregation should add no Tensor read or raw value capture");

    const auto empty = analyzer.BuildTrace("gradient-health-empty", {});
    Check(empty.status == "unobserved" &&
              empty.payload["layer_count"].get<size_t>() == 0 &&
              !empty.payload.contains("success"),
          "missing gradient evidence should remain unobserved, not failed");

    std::vector<cyxwiz::DebugTraceRecord> partial_sources;
    partial_sources.push_back(make_gradient(
        "gradient-health-partial", 1, "Partial", "layer0.weight",
        1.0, true, 0.5, false, false, false, false, 0.0));
    const auto partial = analyzer.BuildTrace(
        "gradient-health-partial", partial_sources);
    Check(partial.status == "partial_evidence" &&
              !partial.payload["success"].get<bool>() &&
              partial.payload["attention_layer_count"].get<size_t>() == 0 &&
              partial.payload["unobserved_layer_count"].get<size_t>() == 1,
          "incomplete optimizer evidence should remain partial, not healthy");

    std::vector<cyxwiz::DebugTraceRecord> many_sources;
    many_sources.reserve(140);
    for (int node_id = 0; node_id < 140; ++node_id) {
        many_sources.push_back(make_gradient(
            "gradient-health-bounded", node_id,
            "Layer " + std::to_string(node_id),
            "layer" + std::to_string(node_id) + ".weight",
            1.0, true, 1.0, false, false, false, true, 0.1));
    }
    const auto bounded = analyzer.BuildTrace(
        "gradient-health-bounded", many_sources);
    Check(bounded.payload["layer_count"].get<size_t>() == 140 &&
              bounded.payload["retained_layer_count"].get<size_t>() ==
                  cyxwiz::DebugGradientHealthAnalyzer::kMaxLayerRows &&
              bounded.payload["layers"].size() ==
                  cyxwiz::DebugGradientHealthAnalyzer::kMaxLayerRows &&
              bounded.payload["layers_truncated"].get<bool>() &&
              bounded.payload["healthy_layer_count"].get<size_t>() == 140,
          "gradient health should retain exact counts while bounding row detail");
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
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(trace),
          "export correlation trace should use canonical node trace schema");
    Check(trace.payload["node_trace_schema"].get<std::string>() ==
              cyxwiz::DebugNodeTraceContract::kSchema,
          "export correlation trace should expose canonical node schema marker");
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
    Check(trace.payload["success"].get<bool>(),
          "successful export correlation trace should mark success");
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
    Check(trace.payload["diagnostic_phase"].get<std::string>() ==
              "export_correlation",
          "export correlation trace should expose diagnostic phase");
    Check(trace.payload["component"].get<std::string>() ==
              "DebugExportCorrelationTracer",
          "export correlation trace should expose diagnostic component");
    Check(trace.payload["source_file"].get<std::string>().find(
              "debug_export_correlation_tracer.cpp") != std::string::npos,
          "export correlation trace should expose diagnostic source file");
    Check(trace.payload["source_symbol"].get<std::string>() ==
              "cyxwiz::DebugExportCorrelationTracer::BuildTrace",
          "export correlation trace should expose diagnostic source symbol");

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
    Check(!failed_trace.payload["success"].get<bool>(),
          "failed export correlation trace should mark unsuccessful");
    Check(failed_trace.payload["warning_count"].get<size_t>() == 1,
          "missing export artifact path should produce a warning");
    Check(failed_trace.payload["error_count"].get<size_t>() == 1,
          "failed compile correlation should produce an error");
    Check(failed_trace.payload["issue_count"].get<size_t>() == 2,
          "failed export correlation should expose total issue count");
    Check(failed_trace.payload["primary_warning_code"].get<std::string>() ==
              cyxwiz::errors::Serialization::ArtifactPathMissing,
          "failed export correlation should expose primary warning code");
    Check(failed_trace.payload["primary_error_code"].get<std::string>() ==
              cyxwiz::errors::Runtime::ExecutionFailed,
          "failed export correlation should expose primary error code");
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
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(trace),
          "Windows crash import trace should use canonical node trace schema");
    Check(trace.payload["node_trace_schema"].get<std::string>() ==
              cyxwiz::DebugNodeTraceContract::kSchema,
          "Windows crash import trace should expose canonical node schema marker");
    Check(trace.payload["schema"].get<std::string>() ==
              cyxwiz::DebugWindowsCrashImporter::kSchema,
          "Windows crash import trace should expose schema");
    Check(trace.payload["error_code"].get<std::string>() ==
              cyxwiz::DebugWindowsCrashImporter::kCrashErrorCode,
          "Windows crash import trace should include stable error code");
    Check(trace.payload["matched"].get<bool>(),
          "Windows crash import trace should include match status");
    Check(trace.payload["success"].get<bool>(),
          "matched Windows crash import trace should mark success");
    Check(trace.payload["fault_module"].get<std::string>() ==
              "arrayfire.dll",
          "Windows crash import trace should include fault module");
    Check(trace.payload["exception_code"].get<std::string>() ==
              "c0000005",
          "Windows crash import trace should include exception code");
    Check(trace.payload["diagnostic_phase"].get<std::string>() ==
              "windows_crash_import",
          "Windows crash import trace should expose diagnostic phase");
    Check(trace.payload["component"].get<std::string>() ==
              "DebugWindowsCrashImporter",
          "Windows crash import trace should expose diagnostic component");
    Check(trace.payload["source_file"].get<std::string>().find(
              "debug_windows_crash_importer.cpp") != std::string::npos,
          "Windows crash import trace should expose diagnostic source file");
    Check(trace.payload["source_symbol"].get<std::string>() ==
              "cyxwiz::DebugWindowsCrashImporter::BuildTrace",
          "Windows crash import trace should expose diagnostic source symbol");

    const auto empty_report = importer.ParseWerText("", "");
    const auto missing_trace = importer.BuildTrace(
        "train-123",
        run,
        empty_report);
    Check(missing_trace.status == "missing",
          "missing Windows crash report should be explicit");
    Check(!missing_trace.payload["success"].get<bool>(),
          "missing Windows crash report should mark unsuccessful");
    Check(missing_trace.payload["warning_count"].get<size_t>() == 1,
          "missing Windows crash report should add warning");
    Check(missing_trace.payload["issue_count"].get<size_t>() == 1,
          "missing Windows crash report should expose issue count");
    Check(missing_trace.payload["primary_warning_code"].get<std::string>() ==
              cyxwiz::DebugWindowsCrashImporter::kCrashErrorCode,
          "missing Windows crash report should expose primary warning code");
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
    record.summary.summary = "Support bundle contract token=secret-token";
    record.summary.file_path = "C:/Users/private/.cyxwiz/debug_runs/support-run.json";
    record.replay_capsule.available = true;
    record.replay_capsule.mode = "SmokeRun";
    record.replay_capsule.graph_hash = record.summary.graph_hash;
    record.replay_capsule.graph_snapshot_trace_available = true;
    record.replay_capsule.dataset_reference = "private_dataset";
    record.replay_capsule.selected_sample_index = 4;
    record.replay_capsule.smoke_sample_limit = 100;
    record.replay_capsule.smoke_batch_size_limit = 32;
    record.replay_capsule.compiled_config.available = true;
    record.replay_capsule.compiled_config.valid = true;
    record.replay_capsule.compiled_config.batch_size = 64;
    record.replay_capsule.compiled_config.loss = "CrossEntropy";
    record.replay_capsule.split_seed = 17;
    record.replay_capsule.dataloader_seed = 19;
    record.replay_capsule.balance_seed = 23;
    record.replay_capsule.environment = {
        {"platform", "windows"},
        {"dataset_path", "C:/Users/private/data.csv"}
    };

    record.issues.push_back({
        cyxwiz::IssueLevel::Error,
        5,
        "Tokenizer token=secret-token",
        "[CW-D-0101] required column missing token=secret-token",
        "CW-D-0101"
    });

    cyxwiz::DebugTraceRecord trace =
        cyxwiz::DebugNodeTraceContract::Make(
            "support-run",
            5,
            "Tokenizer token=secret-token",
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
        "Tokenizer token=secret-token",
        "required column missing token=secret-token",
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
        "Selected failing trace token=secret-token"
    });

    record.recommendations.push_back({
        cyxwiz::DebugRecommendationSeverity::Critical,
        5,
        "Data",
        "Missing required column token=secret-token",
        "The text column was not found token=secret-token.",
        "Select a dataset with the configured text column token=secret-token."
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
    event.estimated_memory_bytes = 8192;
    event.available_memory_bytes = 16384;
    event.safe_memory_budget_bytes = 12288;
    event.memory_risk_level = "risky";
    event.process_memory_detected = true;
    event.process_resident_memory_bytes = 4096;
    event.process_private_memory_bytes = 3072;
    event.process_resident_growth_bytes = 1024;
    event.process_private_memory_name = "private commit";
    event.process_memory_source = "deterministic test";
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
    input.reason = "app requested engine log for HQ diagnostics token=secret-token";
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
    cyxwiz::RuntimeLogExportSnapshot runtime_log_slice;
    runtime_log_slice.scope = "selected";
    runtime_log_slice.effective_filter =
        "run_id=train-42 and level>=warn";
    runtime_log_slice.after_sequence = 40;
    runtime_log_slice.through_sequence = 50;
    runtime_log_slice.matched_count = 3;
    runtime_log_slice.source_displayed_count = 3;
    runtime_log_slice.store_stats.capacity = 4096;
    runtime_log_slice.store_stats.size = 50;
    cyxwiz::RuntimeLogEvent runtime_event;
    runtime_event.sequence = 48;
    runtime_event.timestamp_utc = std::chrono::system_clock::time_point(
        std::chrono::milliseconds(1'700'000'000'000));
    runtime_event.level = cyxwiz::RuntimeLogLevel::Warning;
    runtime_event.category = "data";
    runtime_event.source = "QueryConsole";
    runtime_event.event_name = "sql.query";
    runtime_event.run_id = "train-42";
    runtime_event.dataset_name = "private.parquet";
    runtime_event.message = "SELECT email FROM private_customers";
    runtime_event.details = {
        {"source_path", "C:/Users/private/private.parquet"}};
    runtime_log_slice.events.push_back(std::move(runtime_event));
    input.runtime_log_slice = std::move(runtime_log_slice);
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
    Check(bundle["reason"].get<std::string>() ==
              "app requested engine log for HQ diagnostics token=[REDACTED]",
          "support bundle should redact top-level reason text");
    Check(bundle["debug_run"]["summary"]["summary"].get<std::string>() ==
              "Support bundle contract token=[REDACTED]",
          "support bundle should redact debug run summary text");
    Check(bundle["debug_run"]["traces"][0]["node_name"].get<std::string>() ==
              "Tokenizer token=[REDACTED]",
          "support bundle should redact trace node names");
    Check(bundle["debug_run"]["summary"]["file_path"].get<std::string>() ==
              "[REDACTED]",
          "support bundle should redact debug run file path");
    Check(bundle["debug_run"]["replay_capsule"]["schema"].get<std::string>() ==
              cyxwiz::DebugRunReplayCapsule::kSchema,
          "support bundle should preserve replay capsule schema");
    Check(bundle["debug_run"]["replay_capsule"]["dataset_reference"].get<std::string>() ==
              "[REDACTED]",
          "support bundle should redact replay dataset references");
    Check(bundle["debug_run"]["replay_capsule"]["environment"]
                ["dataset_path"].get<std::string>() == "[REDACTED]",
          "support bundle should redact replay environment paths");
    Check(!bundle["debug_run"]["replay_capsule"]
                  ["data_values_included"].get<bool>() &&
              !bundle["debug_run"]["replay_capsule"]
                   ["exact_replay_claimed"].get<bool>(),
          "support bundle should preserve replay limitation facts");
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
    Check(bundle["debug_run"]["issues"][0]["node_name"].get<std::string>() ==
              "Tokenizer token=[REDACTED]",
          "support bundle should redact record issue node names");
    Check(bundle["debug_run"]["issues"][0]["message"].get<std::string>() ==
              "[CW-D-0101] required column missing token=[REDACTED]",
          "support bundle should redact record issue messages");
    Check(bundle["debug_run"]["traces"][0]["issues"][0]["node_name"].get<std::string>() ==
              "Tokenizer token=[REDACTED]",
          "support bundle should redact trace issue node names");
    Check(bundle["debug_run"]["traces"][0]["issues"][0]["message"].get<std::string>() ==
              "required column missing token=[REDACTED]",
          "support bundle should redact trace issue messages");
    Check(bundle["debug_run"]["studio_events"][0]["message"].get<std::string>() ==
              "Selected failing trace token=[REDACTED]",
          "support bundle should redact Studio event messages");
    Check(bundle["debug_run"]["recommendations"][0]["title"].get<std::string>() ==
              "Missing required column token=[REDACTED]",
          "support bundle should redact recommendation titles");
    Check(bundle["debug_run"]["recommendations"][0]["detail"].get<std::string>() ==
              "The text column was not found token=[REDACTED]",
          "support bundle should redact recommendation details");
    Check(bundle["debug_run"]["recommendations"][0]["action"].get<std::string>() ==
              "Select a dataset with the configured text column token=[REDACTED]",
          "support bundle should redact recommendation actions");
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
    Check(bundle["training_trace"]["recent_events"][0]["memory_risk_level"].get<std::string>() ==
              "risky",
          "support bundle should export materialization memory risk level");
    Check(bundle["training_trace"]["recent_events"][0]["estimated_memory_bytes"].get<uint64_t>() ==
              8192 &&
              bundle["training_trace"]["recent_events"][0]["available_memory_bytes"].get<uint64_t>() ==
                  16384 &&
              bundle["training_trace"]["recent_events"][0]["safe_memory_budget_bytes"].get<uint64_t>() ==
                  12288,
          "support bundle should export estimate and system memory budgets");
    Check(bundle["training_trace"]["recent_events"][0]["process_memory_detected"].get<bool>() &&
              bundle["training_trace"]["recent_events"][0]["process_resident_memory_bytes"].get<uint64_t>() ==
                  4096 &&
              bundle["training_trace"]["recent_events"][0]["process_private_memory_bytes"].get<uint64_t>() ==
                  3072 &&
              bundle["training_trace"]["recent_events"][0]["process_resident_growth_bytes"].get<uint64_t>() ==
                  1024,
          "support bundle should export actual process memory evidence");
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
    Check(bundle["runtime_log_slice"]["included"].get<bool>() &&
              bundle["runtime_log_slice"]["metadata"]["scope"] ==
                  "selected" &&
              bundle["runtime_log_slice"]["metadata"]["effective_filter"] ==
                  "[REDACTED_FILTER]" &&
              bundle["runtime_log_slice"]["events"].size() == 1,
          "support bundle should include only the explicit frozen log slice");
    Check(bundle["runtime_log_slice"]["events"][0]["dataset_name"] ==
              "[REDACTED_DATASET]" &&
              bundle["runtime_log_slice"]["events"][0]["message"] ==
                  "[REDACTED_QUERY]" &&
              bundle["runtime_log_slice"]["events"][0]["details"][0]
                    ["value"] == "[REDACTED_PATH]",
          "support-bundle runtime slices should always use shareable redaction");
}

void TestTrainingGraphDiffContract() {
    auto data = MakeNode(1, gui::NodeType::DataInput, "Data");
    data.parameters = {{"label_column", "target"}};
    auto dense_before = MakeNode(2, gui::NodeType::Dense, "Classifier");
    dense_before.parameters = {{"units", "4"}};
    auto loss = MakeNode(
        3, gui::NodeType::CrossEntropyLoss, "Cross Entropy");

    gui::NodeLink data_to_dense;
    data_to_dense.id = 10;
    data_to_dense.from_node = 1;
    data_to_dense.from_pin = 100;
    data_to_dense.to_node = 2;
    data_to_dense.to_pin = 200;
    gui::NodeLink dense_to_loss;
    dense_to_loss.id = 11;
    dense_to_loss.from_node = 2;
    dense_to_loss.from_pin = 201;
    dense_to_loss.to_node = 3;
    dense_to_loss.to_pin = 300;

    auto dense_after = dense_before;
    dense_after.parameters["units"] = "8";
    auto relu = MakeNode(4, gui::NodeType::ReLU, "ReLU");
    gui::NodeLink dense_to_relu = dense_to_loss;
    dense_to_relu.id = 99;
    dense_to_relu.to_node = 4;
    dense_to_relu.to_pin = 400;

    const auto baseline_session = cyxwiz::DebugSessionManager::StartSession(
        "graph-diff-baseline", "FullWorkflow", 0x1111,
        {data, dense_before, loss}, {data_to_dense, dense_to_loss}, 0);
    const auto current_session = cyxwiz::DebugSessionManager::StartSession(
        "graph-diff-current", "FullWorkflow", 0x2222,
        {data, dense_after, relu}, {data_to_dense, dense_to_relu}, 1);

    cyxwiz::DebugRunStoreRecord baseline;
    baseline.summary.run_id = baseline_session.run_id;
    baseline.summary.graph_hash = baseline_session.graph_hash;
    baseline.traces = baseline_session.traces;
    baseline.replay_capsule.available = true;
    baseline.replay_capsule.dataset_reference = "dataset_a";
    baseline.replay_capsule.selected_sample_index = 0;
    baseline.replay_capsule.compiled_config.available = true;
    baseline.replay_capsule.compiled_config.valid = true;
    baseline.replay_capsule.compiled_config.layer_count = 1;
    baseline.replay_capsule.compiled_config.loss = "CrossEntropy";
    baseline.replay_capsule.compiled_config.optimizer = "Adam";
    baseline.replay_capsule.compiled_config.learning_rate = 0.001f;
    baseline.summary.execution.available = true;
    baseline.summary.execution.requested_backend = "auto";
    baseline.summary.execution.effective_backend = "CPU";
    baseline.summary.execution.placement_fingerprint = "placement-a";

    cyxwiz::DebugRunStoreRecord current;
    current.summary.run_id = current_session.run_id;
    current.summary.graph_hash = current_session.graph_hash;
    current.traces = current_session.traces;
    current.replay_capsule = baseline.replay_capsule;
    current.replay_capsule.dataset_reference = "dataset_b";
    current.replay_capsule.selected_sample_index = 1;
    current.replay_capsule.compiled_config.optimizer = "AdamW";
    current.replay_capsule.compiled_config.learning_rate = 0.0005f;
    current.summary.execution = baseline.summary.execution;
    current.summary.execution.effective_backend = "CUDA";
    current.summary.execution.effective_device_name = "Test GPU";
    current.summary.execution.placement_fingerprint = "placement-b";

    cyxwiz::DebugTrainingGraphDiff differ;
    const auto changed = differ.BuildTrace(baseline, current);
    const auto& payload = changed.payload;
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(changed),
          "training graph diff should use the canonical trace contract");
    Check(changed.phase == "TrainingGraphDiff" &&
              changed.status == "changed" &&
              payload["training_graph_diff_schema"] ==
                  cyxwiz::DebugTrainingGraphDiff::kSchema,
          "training graph diff should expose a stable changed outcome");
    Check(payload["added_node_count"].get<size_t>() == 1 &&
              payload["removed_node_count"].get<size_t>() == 1 &&
              payload["changed_node_count"].get<size_t>() == 1,
          "graph diff should classify added, removed, and changed nodes");
    Check(payload["added_link_count"].get<size_t>() == 1 &&
              payload["removed_link_count"].get<size_t>() == 1,
          "graph diff should compare semantic link endpoints");
    Check(payload["changed_nodes"][0]["changed_parameter_keys"][0] ==
              "units" &&
              !payload["raw_parameter_values_included"].get<bool>(),
          "node parameter diff should retain keys without raw values");
    Check(!payload["structural_details_truncated"].get<bool>() &&
              payload["structural_detail_limit"].get<size_t>() == 128,
          "small graph diffs should retain details within a stable bound");
    Check(payload["dataset_changed"].get<bool>() &&
              payload["selected_sample_changed"].get<bool>(),
          "graph diff should compare replay dataset and sample selection");
    Check(payload["compiled_config_change_count"].get<size_t>() == 2,
          "graph diff should report changed optimizer and learning rate");
    Check(payload["backend_change_count"].get<size_t>() == 3,
          "graph diff should report backend, device name, and placement changes");
    Check(payload["differences_found"].get<bool>() &&
              !payload["raw_graph_content_included"].get<bool>(),
          "graph diff should report bounded differences without raw graph data");

    auto identical = current;
    identical.summary.run_id = "graph-diff-identical";
    identical.traces[0].payload["links"][0]["id"] = 123456;
    const auto unchanged = differ.BuildTrace(current, identical);
    Check(unchanged.status == "unchanged" &&
              unchanged.payload["structural_change_count"].get<size_t>() == 0,
          "link ids alone should not create a structural graph difference");

    cyxwiz::DebugRunStoreRecord unavailable_baseline;
    unavailable_baseline.summary.run_id = "graph-diff-empty-a";
    cyxwiz::DebugRunStoreRecord unavailable_current;
    unavailable_current.summary.run_id = "graph-diff-empty-b";
    const auto unavailable = differ.BuildTrace(
        unavailable_baseline, unavailable_current);
    Check(unavailable.status == "unobserved" &&
              !unavailable.payload["comparison_available"].get<bool>(),
          "missing persisted evidence should remain unobserved");

    std::vector<gui::MLNode> many_nodes;
    many_nodes.reserve(140);
    for (int id = 1; id <= 140; ++id) {
        many_nodes.push_back(MakeNode(
            id, gui::NodeType::ReLU, "Bounded node " + std::to_string(id)));
    }
    const auto empty_session = cyxwiz::DebugSessionManager::StartSession(
        "graph-diff-bounded-empty", "FullWorkflow", 0x3333, {}, {}, 0);
    const auto many_session = cyxwiz::DebugSessionManager::StartSession(
        "graph-diff-bounded-many", "FullWorkflow", 0x4444,
        many_nodes, {}, 0);
    cyxwiz::DebugRunStoreRecord empty_record;
    empty_record.summary.run_id = empty_session.run_id;
    empty_record.summary.graph_hash = empty_session.graph_hash;
    empty_record.traces = empty_session.traces;
    cyxwiz::DebugRunStoreRecord many_record;
    many_record.summary.run_id = many_session.run_id;
    many_record.summary.graph_hash = many_session.graph_hash;
    many_record.traces = many_session.traces;
    const auto bounded = differ.BuildTrace(empty_record, many_record);
    Check(bounded.payload["added_node_count"].get<size_t>() == 140 &&
              bounded.payload["added_nodes"].size() == 128 &&
              bounded.payload["retained_structural_detail_count"]
                      .get<size_t>() == 128 &&
              bounded.payload["structural_details_truncated"].get<bool>(),
          "large graph diffs should retain exact totals and bounded details");
}

void TestArtifactConsistencyTraceContract() {
    cyxwiz::DebugArtifactConsistencyInput input;
    input.action = "export";
    input.artifact_kind = "CyxWiz Model";
    input.artifact_path = "exports/debug_model.cyxmodel";
    input.producer_name = "ModelExporter";
    input.graph_hash = 0xA551;
    input.operation_success = true;
    input.operation_status = "export completed";
    input.source_graph_content = R"({"nodes":[1,2],"links":[[1,2]]})";
    input.artifact_graph_content = input.source_graph_content;
    input.expected.graph_expected = true;
    input.expected.manifest_expected = true;
    input.expected.training_config_expected = true;
    input.expected.weights_manifest_expected = true;
    input.expected.tokenizer_config_required = true;
    input.expected.tokenizer_vocabulary_required = true;
    input.expected.optimizer_state_expected = true;
    input.expected.training_history_expected = true;
    input.expected.parameter_count = 33;
    input.expected.layer_count = 2;
    input.expected.input_contract = "token_ids+attention_mask";
    input.expected.output_contract = "Float32[1,seq,vocab]";
    input.observed.available = true;
    input.observed.manifest_valid = true;
    input.observed.evidence_scope = "cyxmodel_directory_inventory";
    input.observed.format_version = "1.0";
    input.observed.model_family = "causal_lm";
    input.observed.manifest_present = true;
    input.observed.training_config_present = true;
    input.observed.weights_manifest_present = true;
    input.observed.graph_present = true;
    input.observed.tokenizer_config_present = true;
    input.observed.tokenizer_vocabulary_present = true;
    input.observed.optimizer_state_present = true;
    input.observed.training_history_present = true;
    input.observed.sequence_assets_present = true;
    input.observed.parameter_count = 33;
    input.observed.layer_count = 2;
    input.observed.input_contract = "token_ids+attention_mask";
    input.observed.output_contract = "Float32[1,seq,vocab]";
    input.observed.package_warnings = {"bounded package warning"};

    cyxwiz::DebugExportCorrelationTracer tracer;
    const auto compatible = tracer.BuildConsistencyTrace(
        "artifact-consistency-run", input);

    Check(compatible.phase == "ArtifactConsistency" &&
              compatible.status == "ok",
          "matching artifact consistency evidence should be ok");
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(compatible),
          "artifact consistency should use the canonical node trace contract");
    Check(compatible.payload["schema"].get<std::string>() ==
              cyxwiz::DebugExportCorrelationTracer::kConsistencySchema,
          "artifact consistency should expose its stable schema");
    Check(compatible.payload["consistency_outcome"].get<std::string>() ==
              "compatible" &&
              compatible.payload["consistency_verified"].get<bool>(),
          "matching observed artifact facts should verify compatibility");
    Check(compatible.payload["graph_fingerprints_match"].get<bool>() &&
              compatible.payload["parameter_counts_match"].get<bool>() &&
              compatible.payload["layer_counts_match"].get<bool>(),
          "artifact consistency should compare graph and model inventory");
    Check(!compatible.payload["raw_artifact_content_included"].get<bool>(),
          "artifact consistency must not persist raw artifact content");
    Check(compatible.payload["package_warnings"].is_array() &&
              compatible.payload["package_warnings"].size() == 1,
          "artifact consistency should retain bounded package warnings");

    auto mismatch_input = input;
    mismatch_input.artifact_graph_content = R"({"nodes":[1]})";
    mismatch_input.observed.parameter_count = 32;
    mismatch_input.observed.tokenizer_vocabulary_present = false;
    mismatch_input.observed.optimizer_state_present = false;
    const auto mismatch = tracer.BuildConsistencyTrace(
        "artifact-consistency-mismatch", mismatch_input);

    Check(mismatch.status == "failed" &&
              mismatch.payload["consistency_outcome"].get<std::string>() ==
                  "mismatch",
          "required asset or content mismatches should fail consistency");
    Check(mismatch.payload["error_count"].get<size_t>() == 3,
          "graph, parameter, and required tokenizer mismatches should be errors");
    Check(mismatch.payload["warning_count"].get<size_t>() == 1,
          "missing requested optimizer state should remain a warning");
    Check(mismatch.payload["primary_error_code"].get<std::string>() ==
              cyxwiz::errors::Serialization::ModelSaveFailed,
          "artifact mismatches should use the serialization diagnostic code");

    auto invalid_export_input = input;
    invalid_export_input.observed.manifest_valid = false;
    const auto invalid_export = tracer.BuildConsistencyTrace(
        "artifact-consistency-invalid-export", invalid_export_input);
    Check(invalid_export.payload["primary_error_code"].get<std::string>() ==
              cyxwiz::errors::Serialization::ModelSaveFailed,
          "invalid exported headers should use the model-save diagnostic code");

    cyxwiz::DebugArtifactConsistencyInput unobserved_input;
    unobserved_input.action = "export";
    unobserved_input.artifact_kind = "ONNX";
    unobserved_input.artifact_path = "exports/debug_model.onnx";
    unobserved_input.operation_success = true;
    unobserved_input.operation_status = "export completed; probe unsupported";
    const auto unobserved = tracer.BuildConsistencyTrace(
        "artifact-consistency-unobserved", unobserved_input);
    Check(unobserved.status == "unobserved" &&
              unobserved.payload["consistency_outcome"].get<std::string>() ==
                  "unobserved" &&
              unobserved.payload["success"].get<bool>(),
          "unsupported inspection must remain unobserved without failing a successful export");

    cyxwiz::DebugArtifactConsistencyInput binary_input;
    binary_input.action = "import_inspection";
    binary_input.artifact_kind = "CyxWiz Model";
    binary_input.artifact_path = "models/aps.cyxmodel";
    binary_input.producer_name = "ModelImporter";
    binary_input.operation_success = true;
    binary_input.operation_status = "binary model inspection completed";
    binary_input.observed.available = true;
    binary_input.observed.manifest_valid = true;
    binary_input.observed.evidence_scope = "cyxw_binary_header";
    binary_input.observed.format_version = "CYXW v2";
    binary_input.observed.parameter_count = 30209;
    binary_input.observed.layer_count = 7;
    const auto binary = tracer.BuildConsistencyTrace(
        "artifact-consistency-binary", binary_input);
    Check(binary.status == "ok" &&
              binary.payload["consistency_outcome"].get<std::string>() ==
                  "compatible" &&
              binary.payload["consistency_verified"].get<bool>(),
          "validated CYXW header should be compatible without directory assets");
    Check(binary.payload["consistency_check_count"].get<size_t>() == 1 &&
              binary.payload["error_count"].get<size_t>() == 0,
          "CYXW consistency should validate its header without requiring manifests");
}

void TestBatchInspectionContract() {
    cyxwiz::Batch batch;
    const std::vector<float> feature_values{
        1.0f, 2.0f, 0.0f, 0.0f,
        3.0f, 4.0f, 5.0f, 0.0f};
    const std::vector<float> label_values{1.0f, 0.0f, 0.0f, 1.0f};
    batch.data = cyxwiz::Tensor(
        {2, 4}, feature_values.data(), cyxwiz::DataType::Float32);
    batch.labels = cyxwiz::Tensor(
        {2, 2}, label_values.data(), cyxwiz::DataType::Float32);
    batch.size = 2;
    batch.inspection.available = true;
    batch.inspection.row_count = 2;
    batch.inspection.feature_column_count = 4;
    batch.inspection.label_column_count = 1;
    batch.inspection.token_sequence_columns = true;
    batch.inspection.null_summary_available = true;
    batch.inspection.inspected_value_count = 10;
    for (size_t i = 0; i < 4; ++i) {
        batch.inspection.feature_columns_preview.push_back(
            {"tok_" + std::to_string(i), "float"});
    }
    batch.inspection.label_columns_preview.push_back({"label", "int32"});

    cyxwiz::TrainingConfiguration config;
    config.loss_type = gui::NodeType::CrossEntropyLoss;
    config.text_preprocessing.pad_value = 0;
    auto trace = cyxwiz::DebugNodeTraceContract::Make(
        "batch-inspection-run",
        -1,
        "First Smoke Batch",
        "DatasetBatch",
        "SmokeRun.BatchInput",
        cyxwiz::DebugTraceRole::Activation,
        batch.data.Shape(),
        batch.labels.Shape(),
        "float32",
        "ArrayFire active backend",
        "captured");

    cyxwiz::AttachDebugBatchInspection(
        trace, batch, config, "bounded-dataset", "ArrowDatasetBatcher", 32);

    const auto& payload = trace.payload;
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(trace),
          "batch inspection should preserve canonical trace schema");
    Check(payload["batch_inspector"].get<bool>(),
          "batch inspection marker");
    Check(payload["batch_inspection_scope"] == "first_smoke_batch",
          "batch inspection scope");
    Check(payload["actual_row_count"].get<size_t>() == 2,
          "batch inspection actual rows");
    Check(!payload["feature_values_captured"].get<bool>() &&
              !payload["label_values_captured"].get<bool>(),
          "batch inspection should not capture raw values");
    Check(payload["feature_columns_preview"].size() == 4,
          "batch inspection feature provenance");
    Check(payload["null_summary_available"].get<bool>() &&
              payload["inspected_source_value_count"].get<size_t>() == 10,
          "batch inspection exact source null scope");
    Check(payload["padding_summary_available"].get<bool>(),
          "batch inspection padding summary");
    Check(payload["pad_count"].get<size_t>() == 3,
          "batch inspection padding count");
    Check(payload["sequence_lengths"][0].get<size_t>() == 2 &&
              payload["sequence_lengths"][1].get<size_t>() == 3,
          "batch inspection row lengths");
    Check(payload["sequence_length_mean"].get<double>() == 2.5,
          "batch inspection mean row length");
    Check(payload["padding_summary_scalar_read_count"].get<size_t>() == 2,
          "batch padding host read should be row bounded");
    Check(payload["class_balance_available"].get<bool>(),
          "batch inspection class balance");
    Check(payload["class_counts"].size() == 2 &&
              payload["class_counts"][0]["count"].get<size_t>() == 1 &&
              payload["class_counts"][1]["count"].get<size_t>() == 1,
          "batch inspection class counts");
    Check(payload["class_balance_scalar_read_count"].get<size_t>() == 2,
          "batch class balance host read should be class bounded");
    Check(!payload["post_conversion_non_finite_summary_available"].get<bool>(),
          "batch inspection should state unavailable tensor scan");
}

void TestSlowPathDetectorContract() {
    std::vector<cyxwiz::DebugTraceRecord> traces;
    auto preprocessing = cyxwiz::DebugNodeTraceContract::Make(
        "slow-path-run", 7, "Scale", "Scale", "OperatorTransform",
        cyxwiz::DebugTraceRole::PreprocessingOutput,
        {4, 2}, {4, 2}, "arrow::Table", "CPU", "ok");
    preprocessing.duration_ms = 12.0f;
    preprocessing.payload["trace_producer"] = "DebugOperatorTraceProducer";
    traces.push_back(preprocessing);

    auto export_trace = cyxwiz::DebugNodeTraceContract::Make(
        "slow-path-run", -1, "Export", "Export", "ExportCorrelation",
        cyxwiz::DebugTraceRole::GeneratedCode,
        {}, {}, "artifact", "CPU", "ok");
    export_trace.duration_ms = 7.0f;
    traces.push_back(export_trace);

    cyxwiz::TrainingTraceSummary training;
    training.available = true;
    training.run_id = "training-source-run";
    training.status = "completed";
    training.effective_backend = "arrayfire_cpu";
    training.native_cpu_fallback_count = 1;
    const auto add_event = [&](const std::string& stage, float duration_ms) {
        cyxwiz::TrainingTraceEvent event;
        event.run_id = training.run_id;
        event.stage = stage;
        event.duration_ms = duration_ms;
        training.recent_events.push_back(event);
    };
    add_event("GetNextBatch", 8.0f);
    add_event("GetNextBatch", 12.0f);
    add_event("Forward", 20.0f);
    add_event("Forward", 30.0f);
    add_event("Backward", 35.0f);
    add_event("Backward", 45.0f);
    add_event("UpdateParameters", 5.0f);
    add_event("UpdateParameters", 6.0f);

    cyxwiz::DebugSlowPathDetector detector;
    const auto summary = detector.BuildTrace(
        "slow-path-run", traces, training);
    const auto& payload = summary.payload;

    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(summary),
          "slow path summary should use canonical node trace schema");
    Check(summary.phase == "RuntimeSlowPath",
          "slow path summary phase");
    Check(summary.duration_ms == 0.0f,
          "summary trace should not claim the candidate duration as its own");
    Check(payload["slow_path_schema"] ==
              cyxwiz::DebugSlowPathDetector::kSchema,
          "slow path payload schema");
    Check(payload["measurement_semantics"] ==
              "host_wall_clock_no_forced_arrayfire_sync",
          "slow path should state host wall-clock semantics");
    Check(!payload["device_kernel_time_proven"].get<bool>(),
          "slow path should not claim device kernel timing");
    Check(payload["timing_category_count"].get<size_t>() == 9,
          "slow path should keep a bounded category set");
    Check(payload["available_timing_category_count"].get<size_t>() == 6,
          "slow path available category count");
    Check(payload["slow_path_candidate_category"] == "backward" &&
              payload["slow_path_candidate_max_ms"].get<double>() == 45.0,
          "slow path should rank the largest observed max duration");
    Check(!payload["candidate_is_regression_verdict"].get<bool>(),
          "slow path candidate should not be a regression verdict");
    Check(payload["source_training_run_id"] == "training-source-run",
          "slow path should preserve training-run provenance");

    const auto find_stage = [&](const std::string& category)
        -> const nlohmann::json* {
        for (const auto& stage : payload["stage_timings"]) {
            if (stage["category"] == category) {
                return &stage;
            }
        }
        return nullptr;
    };
    const auto* batch_fetch = find_stage("batch_fetch");
    Check(batch_fetch && (*batch_fetch)["timing_available"].get<bool>() &&
              (*batch_fetch)["event_count"].get<size_t>() == 2 &&
              (*batch_fetch)["mean_ms"].get<double>() == 10.0,
          "slow path should aggregate all timed batch-fetch events");
    const auto* batch_creation = find_stage("batch_creation");
    Check(batch_creation &&
              !(*batch_creation)["timing_available"].get<bool>() &&
              (*batch_creation)["unavailable_reason"].get<std::string>().find(
                  "combines") != std::string::npos,
          "slow path should not fabricate separate batch-creation timing");
    const auto* fallback = find_stage("native_cpu_fallback");
    Check(fallback && !(*fallback)["timing_available"].get<bool>() &&
              (*fallback)["occurrence_count"].get<uint64_t>() == 1,
          "slow path should separate fallback occurrence from unavailable duration");
}

void TestTrainingStallDetectorContract() {
    std::vector<cyxwiz::DebugTraceRecord> traces;
    for (int batch = 1; batch <= 3; ++batch) {
        auto loss = cyxwiz::DebugNodeTraceContract::Make(
            "stall-run", 8, "Loss", "Loss", "SmokeRun.Loss",
            cyxwiz::DebugTraceRole::Loss,
            {}, {}, "float32", "LocalDebug", "ok");
        loss.payload["batch"] = batch;
        loss.payload["loss"] = 0.75;
        traces.push_back(std::move(loss));
    }

    auto smoke_gradient = cyxwiz::DebugNodeTraceContract::Make(
        "stall-run", 7, "Dense", "Dense", "SmokeRun.Backward",
        cyxwiz::DebugTraceRole::Gradient,
        {}, {}, "float32", "LocalDebug", "warning");
    smoke_gradient.payload["gradient_tensor_count"] = 2;
    smoke_gradient.payload["zero_gradient_tensor_count"] = 2;
    traces.push_back(std::move(smoke_gradient));

    auto activation = cyxwiz::DebugNodeTraceContract::Make(
        "stall-run", 6, "Sigmoid", "Sigmoid", "Forward",
        cyxwiz::DebugTraceRole::Activation,
        {}, {}, "float32", "LocalDebug", "ok");
    activation.payload["saturation_summary_available"] = true;
    activation.payload["saturation_candidate_ratio"] = 0.98;
    traces.push_back(std::move(activation));

    auto batch = cyxwiz::DebugNodeTraceContract::Make(
        "stall-run", 1, "Dataset", "DataInput", "SmokeRun.BatchInput",
        cyxwiz::DebugTraceRole::ModelInput,
        {}, {}, "float32", "LocalDebug", "ok");
    batch.payload["batch"] = 1;
    batch.payload["batch_inspector"] = true;
    batch.payload["class_balance_available"] = true;
    batch.payload["class_counts"] = nlohmann::json::array({
        {{"class_index", 0}, {"count", 10}},
        {{"class_index", 1}, {"count", 0}},
    });
    batch.payload["source_metadata_available"] = true;
    batch.payload["label_column_count"] = 1;
    batch.payload["label_null_count"] = 1;
    traces.push_back(std::move(batch));

    auto update = cyxwiz::DebugNodeTraceContract::Make(
        "stall-run", 7, "Dense", "Parameter", "Backward",
        cyxwiz::DebugTraceRole::Gradient,
        {}, {}, "float32", "LocalDebug", "zero");
    update.payload["has_gradient"] = true;
    update.payload["is_zero"] = true;
    update.payload["update_observed"] = true;
    update.payload["update_l2_norm"] = 0.0;
    traces.push_back(std::move(update));

    cyxwiz::TrainingTraceSummary training;
    training.available = true;
    training.run_id = "linked-training-run";
    training.status = "running";
    for (int repeat = 0; repeat < 4; ++repeat) {
        cyxwiz::TrainingTraceEvent event;
        event.run_id = training.run_id;
        event.stage = "GetNextBatch";
        event.status = "ok";
        event.epoch = 2;
        event.batch = 5;
        training.recent_events.push_back(std::move(event));
    }

    cyxwiz::DebugTrainingStallConfigEvidence config;
    config.learning_rate_available = true;
    config.learning_rate = 1.0e-9;

    cyxwiz::DebugTrainingStallDetector detector;
    const auto summary = detector.BuildTrace(
        "stall-run", traces, training, config);
    const auto& payload = summary.payload;
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(summary),
          "stall analysis should use canonical trace schema");
    Check(summary.phase == "TrainingStallAnalysis" &&
              payload["training_stall_schema"] ==
                  cyxwiz::DebugTrainingStallDetector::kSchema,
          "stall analysis schema and phase");
    Check(payload["finding_count"].get<size_t>() == 9,
          "stall detector should keep the requested bounded cause set");
    Check(payload["suspected_count"].get<size_t>() == 8 &&
              payload["not_detected_count"].get<size_t>() == 1 &&
              payload["unobserved_count"].get<size_t>() == 0,
          "stall detector should classify observed evidence without unknowns");
    Check(payload["source_training_run_id"] == "linked-training-run",
          "stall analysis should preserve linked training provenance");

    const auto find_cause = [&payload](const std::string& cause)
        -> const nlohmann::json* {
        for (const auto& finding : payload["findings"]) {
            if (finding["cause"] == cause) {
                return &finding;
            }
        }
        return nullptr;
    };
    const auto* flat_loss = find_cause("loss_not_changing");
    Check(flat_loss && (*flat_loss)["status"] == "suspected" &&
              (*flat_loss)["evidence"]["observation_count"] == 3,
          "flat loss should be suspected from three finite observations");
    const auto* high_lr = find_cause("learning_rate_too_high");
    Check(high_lr && (*high_lr)["status"] == "not_detected",
          "low learning rate should not also be classified as high");
    const auto* batcher = find_cause("batcher_not_advancing");
    Check(batcher && (*batcher)["status"] == "suspected" &&
              (*batcher)["evidence"]["repeat_threshold"] == 4,
          "repeated completed fetch cursor should be suspected");
    const auto* class_balance = find_cause("class_imbalance");
    Check(class_balance && (*class_balance)["status"] == "suspected" &&
              !(*class_balance)["evidence"]["dataset_level_conclusion"].get<bool>(),
          "first-batch imbalance should not claim dataset-wide proof");

    const auto unavailable = detector.BuildTrace(
        "stall-run-empty", {}, cyxwiz::TrainingTraceSummary{});
    Check(unavailable.payload["suspected_count"].get<size_t>() == 0 &&
              unavailable.payload["unobserved_count"].get<size_t>() == 9,
          "missing evidence should remain unobserved, not become a diagnosis");
}

void TestErrorCodeTimelineContract() {
    std::vector<cyxwiz::DebugTraceRecord> traces;
    auto compile = cyxwiz::DebugNodeTraceContract::Make(
        "timeline-run", 4, "Dense", "Dense", "Compile",
        cyxwiz::DebugTraceRole::CompileArtifact,
        {}, {}, "graph", "GraphCompiler", "warning");
    cyxwiz::DebugNodeTraceContract::AddWarning(
        compile, "Dense parameter needs review.",
        cyxwiz::errors::Compiler::InvalidParameter);
    traces.push_back(compile);

    auto preflight = cyxwiz::DebugNodeTraceContract::Make(
        "timeline-run", 1, "Dataset", "DataInput", "Preflight",
        cyxwiz::DebugTraceRole::Error,
        {}, {}, "graph", "PreflightValidator", "failed");
    cyxwiz::DebugNodeTraceContract::AddError(
        preflight, "Required label column is missing.",
        cyxwiz::errors::Data::RequiredLabelColumnMissing);
    traces.push_back(preflight);

    cyxwiz::RuntimeLogEvent runtime_event;
    runtime_event.sequence = 42;
    runtime_event.timestamp_utc = std::chrono::system_clock::now();
    runtime_event.level = cyxwiz::RuntimeLogLevel::Error;
    runtime_event.run_id = "training-source-run";
    runtime_event.event_name = "Training.Loss";
    runtime_event.diagnostic_phase = "training_loss";
    runtime_event.component = "TrainingExecutor";
    runtime_event.primary_error_code =
        cyxwiz::errors::Training::TrainingExecutionFailed;
    runtime_event.message = "Loss computation failed.";

    cyxwiz::CrashRunSummary crash;
    crash.available = true;
    crash.suspected_crash = true;
    crash.run_id = "crashed-training-run";
    crash.status = "failed";
    crash.last_stage = "Backward";
    crash.failure_reason = "Training process terminated unexpectedly.";

    cyxwiz::DebugErrorCodeTimeline timeline;
    const auto summary = timeline.BuildTrace(
        "timeline-run", traces, {runtime_event}, crash);
    const auto& payload = summary.payload;
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(summary),
          "error-code timeline should use canonical trace schema");
    Check(summary.phase == "ErrorCodeTimeline" &&
              payload["error_code_timeline_schema"] ==
                  cyxwiz::DebugErrorCodeTimeline::kSchema,
          "error-code timeline schema and phase");
    Check(payload["entry_count"].get<size_t>() == 4 &&
              payload["observed_entry_count"].get<size_t>() == 4,
          "error-code timeline should avoid trace issue/payload duplicates");
    Check(!payload["wall_clock_complete"].get<bool>() &&
              payload["ordering_semantics"] ==
                  "canonical_trace_capture_order_then_runtime_sequence",
          "error-code timeline should state ordering limitations");

    const auto find_entry = [&](const std::string& code)
        -> const nlohmann::json* {
        for (const auto& entry : payload["entries"]) {
            if (entry["code"] == code) {
                return &entry;
            }
        }
        return nullptr;
    };
    const auto* compiler = find_entry(
        cyxwiz::errors::Compiler::InvalidParameter);
    Check(compiler && (*compiler)["phase"] == "compile" &&
              (*compiler)["subsystem"] == "compiler" &&
              (*compiler)["symbolic_name"] ==
                  "Compiler.InvalidParameter",
          "compiler code timeline metadata");
    const auto* data = find_entry(
        cyxwiz::errors::Data::RequiredLabelColumnMissing);
    Check(data && (*data)["phase"] == "preflight" &&
              (*data)["node_id"].get<int>() == 1,
          "preflight code should preserve phase and node");
    const auto* loss = find_entry(
        cyxwiz::errors::Training::TrainingExecutionFailed);
    Check(loss && (*loss)["phase"] == "loss" &&
              (*loss)["source_run_id"] == "training-source-run" &&
              (*loss)["runtime_sequence"].get<uint64_t>() == 42,
          "runtime code should preserve phase, run, and sequence");
    const auto* runtime = find_entry(
        cyxwiz::errors::Runtime::ExecutionFailed);
    Check(runtime && (*runtime)["phase"] == "crash_runtime" &&
              (*runtime)["source_run_id"] == "crashed-training-run",
          "crash summary should map to a coded runtime entry");

    std::vector<cyxwiz::DebugTraceRecord> many_traces;
    for (size_t i = 0; i < cyxwiz::DebugErrorCodeTimeline::kMaxEntries + 2;
         ++i) {
        auto trace = cyxwiz::DebugNodeTraceContract::Make(
            "bounded-timeline-run", static_cast<int>(i), "Layer", "Dense",
            "Forward", cyxwiz::DebugTraceRole::Warning,
            {}, {}, "tensor", "ArrayFire", "warning");
        cyxwiz::DebugNodeTraceContract::AddWarning(
            trace,
            "Forward warning " + std::to_string(i),
            cyxwiz::errors::Runtime::ExecutionFailed);
        many_traces.push_back(std::move(trace));
    }
    const auto bounded = timeline.BuildTrace(
        "bounded-timeline-run", many_traces);
    Check(bounded.payload["timeline_truncated"].get<bool>() &&
              bounded.payload["entry_count"].get<size_t>() ==
                  cyxwiz::DebugErrorCodeTimeline::kMaxEntries &&
              bounded.payload["observed_entry_count"].get<size_t>() ==
                  cyxwiz::DebugErrorCodeTimeline::kMaxEntries + 2,
          "error-code timeline should cap persisted UI entries");
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

void TestNodeExplanationTraceContract() {
    const auto data = MakeNode(1, gui::NodeType::DataInput, "Data");
    const auto dense = MakeNode(2, gui::NodeType::Dense, "Dense");
    const auto unused = MakeNode(3, gui::NodeType::ReLU, "Unused ReLU");
    const std::vector<gui::MLNode> nodes = {data, dense, unused};

    gui::NodeLink link;
    link.id = 10;
    link.from_node = 1;
    link.to_node = 2;
    const std::vector<gui::NodeLink> links = {link};

    cyxwiz::TrainingConfiguration configuration;
    configuration.data_source_node_id = 1;
    configuration.graph_plan.available = true;
    configuration.graph_plan.nodes = {
        {gui::NodeType::DataInput, 1, "Data"},
        {gui::NodeType::Dense, 2, "Dense"}
    };
    cyxwiz::CompiledLayer layer;
    layer.node_id = 2;
    layer.name = "Dense";
    layer.input_shape = {4, 8};
    layer.output_shape = {4, 3};
    configuration.layers.push_back(layer);

    cyxwiz::BackendPlacementEntry placement;
    placement.node_id = 2;
    placement.node_name = "Dense";
    placement.node_type = "Dense";
    placement.requested_backend = "auto";
    placement.expected_backend = "gpu";
    placement.status = cyxwiz::BackendPlacementStatus::Gpu;
    placement.reason_code = "arrayfire_tensor_op_capable";
    cyxwiz::DebugRuntimeBackendClassifier classifier;
    std::vector<cyxwiz::DebugTraceRecord> traces = {
        classifier.BuildPlacementTrace("explain-run", 0x1234, placement)
    };

    const cyxwiz::ValidationIssue issue = {
        cyxwiz::IssueLevel::Warning,
        2,
        "Dense",
        "Check the Dense input width.",
        "Compiler.ShapeWarning"
    };
    traces[0].issues.push_back(issue);
    const std::vector<cyxwiz::DebugRecommendation> recommendations = {{
        cyxwiz::DebugRecommendationSeverity::Warning,
        2,
        "Shapes",
        "Inspect Dense width",
        "The input width needs review.",
        "Compare the upstream output width with Dense input width."
    }};

    cyxwiz::DebugNodeInspector inspector;
    const auto predicted = inspector.BuildExplanationTrace(
        "explain-run", 0x1234, dense, "Dense", &configuration,
        nodes, links, traces, {issue}, recommendations);

    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(predicted),
          "node explanation should retain the canonical trace schema");
    Check(predicted.phase == "NodeExplanation",
          "node explanation phase should be stable");
    Check(predicted.payload["explanation_schema"] ==
              "cyxwiz.debug.node_explanation.v1",
          "node explanation should expose its payload schema");
    Check(predicted.payload["graph_role"] == "model_layer" &&
              predicted.payload["training_path_status"] == "selected",
          "compiled layer should be identified on the selected path");
    Check(predicted.input_shape == std::vector<size_t>{4, 8} &&
              predicted.output_shape == std::vector<size_t>{4, 3} &&
              predicted.payload["data_evidence_scope"] ==
                  "compiler_prediction",
          "explanation should label compiler-predicted shapes");
    Check(predicted.payload["backend_expected"] == "gpu" &&
              predicted.payload["backend_actual"] == "unobserved" &&
              !predicted.payload["backend_actual_observed"].get<bool>(),
          "expected placement must not be presented as actual execution");
    Check(predicted.issues.size() == 1,
          "node explanation should deduplicate matching issues");
    Check(predicted.payload["recommendations"].size() == 1 &&
              predicted.payload["next_inspection_action"] ==
                  recommendations[0].action,
          "node explanation should reuse the existing recommendation action");
    Check(predicted.payload["upstream_nodes"].size() == 1 &&
              predicted.payload["upstream_nodes"][0]["node_id"] == 1,
          "node explanation should expose direct upstream wiring");

    auto observed = cyxwiz::DebugNodeTraceContract::Make(
        "explain-run", 2, "Dense", "Dense", "Forward",
        cyxwiz::DebugTraceRole::Activation,
        {2, 8}, {2, 3}, "float32", "ArrayFire", "ok");
    observed.payload["backend_actual"] = "ArrayFire/CUDA";
    observed.payload["backend_actual_observed"] = true;
    traces.push_back(std::move(observed));
    const auto runtime = inspector.BuildExplanationTrace(
        "explain-run", 0x1234, dense, "Dense", &configuration,
        nodes, links, traces, {}, {});
    Check(runtime.input_shape == std::vector<size_t>{2, 8} &&
              runtime.payload["data_evidence_scope"] == "same_run_trace" &&
              runtime.payload["data_evidence_phase"] == "Forward",
          "same-run trace data should take precedence over prediction");
    Check(runtime.payload["backend_actual"] == "ArrayFire/CUDA" &&
              runtime.payload["backend_actual_observed"].get<bool>() &&
              runtime.payload["backend_evidence_scope"] ==
                  "same_run_runtime",
          "observed runtime backend should be identified explicitly");

    const auto outside = inspector.BuildExplanationTrace(
        "explain-run", 0x1234, unused, "ReLU", &configuration,
        nodes, links, traces, {}, {});
    Check(outside.payload["graph_role"] ==
              "outside_selected_training_path" &&
              outside.payload["training_path_status"] == "outside",
          "unselected node should be explicitly outside the training path");
    Check(outside.payload["data_evidence_scope"] == "unobserved" &&
              outside.payload["backend_evidence_scope"] == "unobserved",
          "outside node should not invent data or backend evidence");
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

void TestOperatorTraceProducerContract() {
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

    auto data = MakeNode(1, gui::NodeType::DataInput, "Data Input");
    auto tokenizer = MakeNode(2, gui::NodeType::TextTokenizer, "Tokenizer");
    tokenizer.parameters = {
        {"text_col", "text"},
        {"label_col", "label"},
        {"tokenizer_type", "1"},
        {"max_length", "4"},
        {"lowercase", "true"},
        {"min_word_freq", "1"},
        {"max_vocab_size", "100"},
    };

    gui::NodeLink link;
    link.id = 101;
    link.from_node = data.id;
    link.to_node = tokenizer.id;

    cyxwiz::DebugOperatorTraceProducer producer;
    const auto traces = producer.TracePreprocessingGraph(
        "operator-producer-run",
        {data, tokenizer},
        {link},
        input);

    Check(traces.size() == 1,
          "operator producer should emit one TextTokenizer trace");
    const auto& trace = traces[0];
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(trace),
          "operator producer trace should use canonical node schema");
    Check(trace.run_id == "operator-producer-run",
          "operator producer trace should preserve run id");
    Check(trace.node_id == tokenizer.id,
          "operator producer trace should preserve tokenizer node id");
    Check(trace.node_name == "Tokenizer",
          "operator producer trace should preserve tokenizer name");
    Check(trace.node_type == "TextTokenizer",
          "operator producer trace should preserve tokenizer type");
    Check(trace.phase == "OperatorTransform",
          "operator producer should use operator transform phase");
    Check(trace.role == cyxwiz::DebugTraceRole::PreprocessingOutput,
          "operator producer should emit preprocessing output role");
    Check(trace.input_shape == std::vector<size_t>{3, 2},
          "operator producer should capture input table shape");
    Check(trace.output_shape == std::vector<size_t>{3, 5},
          "operator producer should capture tokenized output table shape");
    Check(trace.payload["trace_producer"].get<std::string>() ==
              "DebugOperatorTraceProducer",
          "operator producer trace should name its producer");
    Check(trace.payload["operator_backed"].get<bool>(),
          "operator producer should mark real operator-backed traces");
    Check(trace.payload["success"].get<bool>(),
          "operator producer should mark successful operator-backed traces");
    Check(trace.payload["diagnostic_phase"].get<std::string>() ==
              "operator_transform",
          "operator producer trace should expose operator-transform diagnostic phase");
    Check(trace.payload["component"].get<std::string>() ==
              "DebugOperatorTraceProducer",
          "operator producer trace should expose diagnostic component");
    Check(trace.payload["source_file"].get<std::string>().find(
              "debug_operator_trace_producer.cpp") != std::string::npos,
          "operator producer trace should expose source file");
    Check(trace.payload["source_symbol"].get<std::string>() ==
              "cyxwiz::DebugOperatorTraceProducer::TraceTextTokenizer",
          "operator producer trace should expose source symbol");
    Check(trace.payload["input_schema"].get<std::string>().find("text") !=
              std::string::npos,
          "operator producer should include input schema");
    Check(trace.payload["output_schema"].get<std::string>().find("tok_0") !=
              std::string::npos,
          "operator producer should include output schema");

    Check(trace.payload["effective_text_tokenizer_config"]["max_length"].get<std::string>() ==
              "4",
          "operator producer should expose effective tokenizer max_length");
    Check(trace.payload["effective_text_tokenizer_config"]["text_col"].get<std::string>() ==
              "text",
          "operator producer should expose effective tokenizer text column");
    Check(!trace.payload["effective_text_tokenizer_config"]["vocab_file_configured"].get<bool>(),
          "operator producer should avoid storing an unconfigured vocab path");
    Check(!trace.payload["folded_text_config_applied"].get<bool>(),
          "plain tokenizer trace should mark that no folded config was applied");
    Check(trace.payload["folded_text_config_nodes"].empty(),
          "plain tokenizer trace should expose no folded config provenance");
    Check(trace.payload["source_node_id"].get<int>() == data.id,
          "operator producer trace should preserve source node id");
    Check(trace.payload["source_node_name"].get<std::string>() == data.name,
          "operator producer trace should preserve source node name");
    Check(trace.payload["source_node_type"].get<std::string>() == "DataInput",
          "operator producer trace should preserve source node type");
    const auto bounded_traces = producer.TracePreprocessingGraph(
        "operator-producer-bounded-run",
        {data, tokenizer},
        {link},
        input,
        {},
        1,
        1);
    Check(bounded_traces.size() == 1,
          "bounded operator producer should emit tokenizer trace");
    Check(bounded_traces[0].input_shape == std::vector<size_t>{1, 2},
          "bounded operator producer should trace only the selected row window");
    Check(bounded_traces[0].output_shape == std::vector<size_t>{1, 5},
          "bounded operator producer should emit output shape for selected row window");
    Check(bounded_traces[0].payload["source_rows"].get<size_t>() == 3,
          "bounded operator producer should preserve original source row count");
    Check(bounded_traces[0].payload["selected_sample_index"].get<size_t>() == 1,
          "bounded operator producer should preserve selected sample index");
    Check(bounded_traces[0].payload["debug_row_offset"].get<size_t>() == 1,
          "bounded operator producer should expose debug row offset");
    Check(bounded_traces[0].payload["debug_row_count"].get<size_t>() == 1,
          "bounded operator producer should expose debug row count");
    Check(bounded_traces[0].payload["bounded_debug_table"].get<bool>(),
          "bounded operator producer should mark bounded debug source tables");
    Check(!bounded_traces[0].payload["selected_sample_clamped"].get<bool>(),
          "bounded operator producer should not clamp an in-range selected sample");
    Check(bounded_traces[0].payload["selected_sample_available"].get<bool>(),
          "bounded operator producer should mark an in-range selected sample as available");

    const auto clamped_traces = producer.TracePreprocessingGraph(
        "operator-producer-clamped-run",
        {data, tokenizer},
        {link},
        input,
        {},
        99,
        1);
    Check(clamped_traces.size() == 1,
          "clamped operator producer should still emit tokenizer trace");
    Check(clamped_traces[0].input_shape == std::vector<size_t>{1, 2},
          "clamped operator producer should trace one available row");
    Check(clamped_traces[0].payload["selected_sample_index"].get<size_t>() == 99,
          "clamped operator producer should preserve requested selected sample");
    Check(clamped_traces[0].payload["debug_row_offset"].get<size_t>() == 2,
          "clamped operator producer should expose actual clamped row offset");
    Check(clamped_traces[0].payload["selected_sample_clamped"].get<bool>(),
          "clamped operator producer should mark selected sample clamping");
    Check(!clamped_traces[0].payload["selected_sample_available"].get<bool>(),
          "clamped operator producer should mark the requested sample unavailable");

    auto empty_text = FinishStringArray(std::vector<std::string>{});
    auto empty_label = FinishStringArray(std::vector<std::string>{});
    auto empty_input = arrow::Table::Make(schema, {empty_text, empty_label}, 0);
    const auto empty_source_traces = producer.TracePreprocessingGraph(
        "operator-producer-empty-source-run",
        {data, tokenizer},
        {link},
        empty_input,
        {},
        10,
        1);
    Check(empty_source_traces.size() == 1,
          "empty source operator producer should still emit tokenizer trace");
    Check(empty_source_traces[0].input_shape == std::vector<size_t>{0, 2},
          "empty source operator producer should preserve empty input shape");
    Check(empty_source_traces[0].payload["source_rows"].get<size_t>() == 0,
          "empty source operator producer should preserve zero source rows");
    Check(empty_source_traces[0].payload["debug_row_count"].get<size_t>() == 0,
          "empty source operator producer should expose zero debug rows");
    Check(!empty_source_traces[0].payload["selected_sample_available"].get<bool>(),
          "empty source operator producer should mark selected sample unavailable");
    Check(!empty_source_traces[0].payload["selected_sample_clamped"].get<bool>(),
          "empty source operator producer should not claim a row clamp occurred");

    auto vocabulary = MakeNode(4, gui::NodeType::TextVocabulary, "Vocabulary");
    vocabulary.parameters = {
        {"min_freq", "2"},
        {"max_vocab_size", "100"},
    };
    auto padding = MakeNode(5, gui::NodeType::TextPadding, "Padding");
    padding.parameters = {
        {"max_length", "2"},
        {"pad_value", "0"},
    };
    gui::NodeLink vocab_link;
    vocab_link.id = 104;
    vocab_link.from_node = tokenizer.id;
    vocab_link.to_node = vocabulary.id;
    gui::NodeLink padding_link;
    padding_link.id = 105;
    padding_link.from_node = vocabulary.id;
    padding_link.to_node = padding.id;
    const auto folded_config_traces = producer.TracePreprocessingGraph(
        "operator-producer-folded-config-run",
        {data, tokenizer, vocabulary, padding},
        {link, vocab_link, padding_link},
        input);
    Check(folded_config_traces.size() == 1,
          "folded text config nodes should not emit separate operator traces");
    Check(folded_config_traces[0].output_shape == std::vector<size_t>{3, 3},
          "folded TextPadding max_length should shape the tokenizer trace");
    Check(folded_config_traces[0].payload["vocab_size"].get<size_t>() <
              trace.payload["vocab_size"].get<size_t>(),
          "folded TextVocabulary min_freq should shape tokenizer vocabulary");

    Check(folded_config_traces[0].payload["effective_text_tokenizer_config"]["max_length"].get<std::string>() ==
              "2",
          "folded tokenizer trace should expose effective folded max_length");
    Check(folded_config_traces[0].payload["effective_text_tokenizer_config"]["min_word_freq"].get<std::string>() ==
              "2",
          "folded tokenizer trace should expose effective folded min_word_freq");
    Check(folded_config_traces[0].payload["effective_text_tokenizer_config"]["pad_value"].get<std::string>() ==
              "0",
          "folded tokenizer trace should expose effective folded pad_value");
    Check(folded_config_traces[0].payload["folded_text_config_applied"].get<bool>(),
          "folded tokenizer trace should mark that folded config was applied");
    Check(folded_config_traces[0].payload["folded_text_config_nodes"].size() == 2,
          "folded tokenizer trace should expose folded config node provenance");
    Check(folded_config_traces[0].payload["folded_text_config_nodes"][0]["node_id"].get<int>() ==
              vocabulary.id,
          "folded tokenizer trace should preserve vocabulary config node id");
    Check(folded_config_traces[0].payload["folded_text_config_nodes"][0]["node_type"].get<std::string>() ==
              "TextVocabulary",
          "folded tokenizer trace should preserve vocabulary config node type");
    Check(folded_config_traces[0].payload["folded_text_config_nodes"][0]["contributed_keys"].size() == 2,
          "folded tokenizer trace should expose vocabulary contributed keys");
    Check(folded_config_traces[0].payload["folded_text_config_nodes"][0]["contributed_keys"][0].get<std::string>() ==
              "min_word_freq",
          "folded tokenizer trace should expose vocabulary min frequency contribution");
    Check(folded_config_traces[0].payload["folded_text_config_nodes"][0]["contributed_keys"][1].get<std::string>() ==
              "max_vocab_size",
          "folded tokenizer trace should expose vocabulary max size contribution");
    Check(folded_config_traces[0].payload["folded_text_config_nodes"][1]["node_id"].get<int>() ==
              padding.id,
          "folded tokenizer trace should preserve padding config node id");
    Check(folded_config_traces[0].payload["folded_text_config_nodes"][1]["contributed_keys"].size() == 2,
          "folded tokenizer trace should expose padding contributed keys");
    Check(folded_config_traces[0].payload["folded_text_config_nodes"][1]["contributed_keys"][0].get<std::string>() ==
              "max_length",
          "folded tokenizer trace should expose padding max_length contribution");
    Check(folded_config_traces[0].payload["folded_text_config_nodes"][1]["contributed_keys"][1].get<std::string>() ==
              "pad_value",
          "folded tokenizer trace should expose padding value contribution");

    auto empty_padding = MakeNode(14, gui::NodeType::TextPadding, "Empty Padding");
    gui::NodeLink empty_padding_link;
    empty_padding_link.id = 115;
    empty_padding_link.from_node = tokenizer.id;
    empty_padding_link.to_node = empty_padding.id;
    const auto empty_config_traces = producer.TracePreprocessingGraph(
        "operator-producer-empty-folded-config-run",
        {data, tokenizer, empty_padding},
        {link, empty_padding_link},
        input);
    Check(empty_config_traces.size() == 1,
          "empty folded config nodes should still allow tokenizer trace");
    Check(empty_config_traces[0].output_shape == std::vector<size_t>{3, 5},
          "empty folded config nodes should not alter tokenizer output shape");
    Check(!empty_config_traces[0].payload["folded_text_config_applied"].get<bool>(),
          "empty folded config nodes should not mark folded config as applied");
    Check(empty_config_traces[0].payload["folded_text_config_nodes"].empty(),
          "empty folded config nodes should not appear in folded config provenance");

    gui::NodeLink folded_only_link;
    folded_only_link.id = 112;
    folded_only_link.from_node = data.id;
    folded_only_link.to_node = padding.id;
    const auto folded_only_traces = producer.TracePreprocessingGraph(
        "operator-producer-folded-only-run",
        {data, padding},
        {folded_only_link},
        input);
    Check(folded_only_traces.size() == 1,
          "folded text config without tokenizer should emit one warning trace");
    Check(folded_only_traces[0].node_id == data.id,
          "folded-only warning should attach to the source node");
    Check(folded_only_traces[0].role == cyxwiz::DebugTraceRole::Warning,
          "folded-only trace should be warning-only");
    Check(folded_only_traces[0].payload["diagnostic_phase"].get<std::string>() ==
              "graph_walk",
          "folded-only warning should identify graph-walk validation");
    Check(folded_only_traces[0].payload["message"].get<std::string>().find(
              "TextTokenizer") != std::string::npos,
          "folded-only warning should explain the missing tokenizer operator");

    auto other_data = MakeNode(10, gui::NodeType::DataInput, "Other Data");
    other_data.parameters["dataset_name"] = "other_dataset";
    auto named_data = MakeNode(11, gui::NodeType::DataInput, "Named Data");
    named_data.parameters["dataset_name"] = "wanted_dataset";
    gui::NodeLink named_link;
    named_link.id = 103;
    named_link.from_node = named_data.id;
    named_link.to_node = tokenizer.id;
    const auto named_traces = producer.TracePreprocessingGraph(
        "operator-producer-named-run",
        {other_data, named_data, tokenizer},
        {named_link},
        input,
        "wanted_dataset");
    Check(named_traces.size() == 1,
          "operator producer should start from the data input matching dataset name");
    Check(named_traces[0].payload["source_dataset_name"].get<std::string>() ==
              "wanted_dataset",
          "operator producer should annotate named source dataset");
    Check(named_traces[0].payload["source_node_id"].get<int>() == named_data.id,
          "operator producer should preserve selected named source node id");
    Check(named_traces[0].payload["source_node_name"].get<std::string>() ==
              named_data.name,
          "operator producer should preserve selected named source node name");
    Check(named_traces[0].payload["source_node_dataset_name"].get<std::string>() ==
              "wanted_dataset",
          "operator producer should preserve selected source node dataset name");
    const auto unavailable_source_traces = producer.TracePreprocessingGraph(
        "operator-producer-unavailable-source-run",
        {named_data, tokenizer},
        {named_link},
        {},
        "wanted_dataset");
    Check(unavailable_source_traces.size() == 1,
          "unavailable Arrow source table should emit one warning trace");
    Check(unavailable_source_traces[0].node_id == -1,
          "unavailable Arrow source table warning should be graph-level");
    Check(unavailable_source_traces[0].role == cyxwiz::DebugTraceRole::Warning,
          "unavailable Arrow source table should be warning-only");
    Check(unavailable_source_traces[0].payload["diagnostic_phase"].get<std::string>() ==
              "data_source",
          "unavailable Arrow source table warning should identify data-source validation");
    Check(unavailable_source_traces[0].payload["error_code"].get<std::string>() ==
              cyxwiz::errors::Runtime::InputDatasetMissing,
          "unavailable Arrow source table warning should expose missing-input code");
    Check(!unavailable_source_traces[0].issues.empty() &&
              unavailable_source_traces[0].issues[0].error_code ==
                  cyxwiz::errors::Runtime::InputDatasetMissing,
          "unavailable Arrow source table issue should expose missing-input code");
    Check(unavailable_source_traces[0].payload["issue_count"].get<size_t>() == 1,
          "unavailable Arrow source table warning should expose issue count");
    Check(unavailable_source_traces[0].payload["warning_count"].get<size_t>() == 1,
          "unavailable Arrow source table warning should expose warning count");
    Check(unavailable_source_traces[0].payload["primary_warning_code"].get<std::string>() ==
              cyxwiz::errors::Runtime::InputDatasetMissing,
          "unavailable Arrow source table warning should expose primary warning code");
    Check(unavailable_source_traces[0].payload["component"].get<std::string>() ==
              "DebugOperatorTraceProducer",
          "unavailable Arrow source table warning should expose diagnostic component");
    Check(unavailable_source_traces[0].payload["source_file"].get<std::string>().find(
              "debug_operator_trace_producer.cpp") != std::string::npos,
          "unavailable Arrow source table warning should expose source file");
    Check(unavailable_source_traces[0].payload["source_symbol"].get<std::string>() ==
              "cyxwiz::DebugOperatorTraceProducer::TracePreprocessingGraph",
          "unavailable Arrow source table warning should expose source symbol");
    Check(unavailable_source_traces[0].payload["source_dataset_name"].get<std::string>() ==
              "wanted_dataset",
          "unavailable Arrow source table warning should preserve requested dataset");
    Check(unavailable_source_traces[0].payload["source_node_id"].get<int>() ==
              named_data.id,
          "unavailable Arrow source table warning should preserve source node id");
    Check(unavailable_source_traces[0].payload["source_rows"].get<size_t>() == 0,
          "unavailable Arrow source table warning should expose zero source rows");
    Check(unavailable_source_traces[0].payload["message"].get<std::string>().find(
              "source table") != std::string::npos,
          "unavailable Arrow source table warning should explain skipped trace");
    const auto missing_source_traces = producer.TracePreprocessingGraph(
        "operator-producer-missing-source-run",
        {tokenizer},
        {},
        input);
    Check(missing_source_traces.size() == 1,
          "missing data source should emit one warning trace");
    Check(missing_source_traces[0].node_id == -1,
          "missing data source warning should be graph-level");
    Check(missing_source_traces[0].role == cyxwiz::DebugTraceRole::Warning,
          "missing data source should be warning-only");
    Check(missing_source_traces[0].payload["diagnostic_phase"].get<std::string>() ==
              "graph_walk",
          "missing data source warning should identify graph-walk validation");
    Check(missing_source_traces[0].payload["error_code"].get<std::string>() ==
              cyxwiz::errors::Runtime::InputDatasetMissing,
          "missing data source warning should expose missing-input code");
    Check(!missing_source_traces[0].issues.empty() &&
              missing_source_traces[0].issues[0].error_code ==
                  cyxwiz::errors::Runtime::InputDatasetMissing,
          "missing data source issue should expose missing-input code");
    Check(missing_source_traces[0].payload["message"].get<std::string>().find(
              "no DataInput or DatasetInput") != std::string::npos,
          "missing data source warning should explain skipped trace");

    const auto mismatched_source_traces = producer.TracePreprocessingGraph(
        "operator-producer-mismatched-source-run",
        {other_data, tokenizer},
        {link},
        input,
        "wanted_dataset");
    Check(mismatched_source_traces.size() == 1,
          "mismatched dataset source should emit one warning trace");
    Check(mismatched_source_traces[0].node_id == -1,
          "mismatched dataset source warning should be graph-level");
    Check(mismatched_source_traces[0].role == cyxwiz::DebugTraceRole::Warning,
          "mismatched dataset source should be warning-only");
    Check(mismatched_source_traces[0].payload["diagnostic_phase"].get<std::string>() ==
              "graph_walk",
          "mismatched dataset source warning should identify graph-walk validation");
    Check(mismatched_source_traces[0].payload["error_code"].get<std::string>() ==
              cyxwiz::errors::Runtime::InputDatasetMissing,
          "mismatched dataset source warning should expose missing-input code");
    Check(!mismatched_source_traces[0].issues.empty() &&
              mismatched_source_traces[0].issues[0].error_code ==
                  cyxwiz::errors::Runtime::InputDatasetMissing,
          "mismatched dataset source issue should expose missing-input code");
    Check(mismatched_source_traces[0].payload["source_dataset_name"].get<std::string>() ==
              "wanted_dataset",
          "mismatched dataset source warning should preserve requested dataset");
    Check(mismatched_source_traces[0].payload["message"].get<std::string>().find(
              "does not match any DataInput/DatasetInput") != std::string::npos,
          "mismatched dataset source warning should explain skipped trace");
    auto second_tokenizer = tokenizer;
    second_tokenizer.id = 6;
    second_tokenizer.name = "Second Tokenizer";
    gui::NodeLink second_tokenizer_link;
    second_tokenizer_link.id = 106;
    second_tokenizer_link.from_node = data.id;
    second_tokenizer_link.to_node = second_tokenizer.id;
    const auto branched_traces = producer.TracePreprocessingGraph(
        "operator-producer-branched-run",
        {data, tokenizer, second_tokenizer},
        {link, second_tokenizer_link},
        input);
    Check(branched_traces.size() == 1,
          "branched supported trace paths should emit one warning trace");
    Check(branched_traces[0].role == cyxwiz::DebugTraceRole::Warning,
          "branched supported trace paths should be warning-only");
    Check(!branched_traces[0].payload["operator_backed"].get<bool>(),
          "branched warning should not claim operator-backed execution");
    Check(branched_traces[0].payload["diagnostic_phase"].get<std::string>() ==
              "graph_walk",
          "branched warning should identify graph-walk validation");
    Check(branched_traces[0].payload["message"].get<std::string>().find(
              "branched operator trace paths") != std::string::npos,
          "branched warning should explain unsupported trace topology");

    gui::NodeLink cycle_link;
    cycle_link.id = 107;
    cycle_link.from_node = padding.id;
    cycle_link.to_node = tokenizer.id;
    const auto cycle_traces = producer.TracePreprocessingGraph(
        "operator-producer-cycle-run",
        {data, tokenizer, vocabulary, padding},
        {link, vocab_link, padding_link, cycle_link},
        input);
    Check(cycle_traces.size() == 1,
          "cyclic trace paths should emit one warning trace");
    Check(cycle_traces[0].role == cyxwiz::DebugTraceRole::Warning,
          "cyclic trace paths should be warning-only");
    Check(!cycle_traces[0].payload["operator_backed"].get<bool>(),
          "cycle warning should not claim operator-backed execution");
    Check(cycle_traces[0].payload["diagnostic_phase"].get<std::string>() ==
              "graph_walk",
          "cycle warning should identify graph-walk validation");
    Check(cycle_traces[0].payload["message"].get<std::string>().find(
              "cyclic graph path") != std::string::npos,
          "cycle warning should explain unsupported trace topology");
    auto unsupported_cycle = MakeNode(7, gui::NodeType::Dense, "Unsupported Cycle");
    gui::NodeLink unsupported_cycle_in;
    unsupported_cycle_in.id = 108;
    unsupported_cycle_in.from_node = data.id;
    unsupported_cycle_in.to_node = unsupported_cycle.id;
    gui::NodeLink unsupported_cycle_back;
    unsupported_cycle_back.id = 109;
    unsupported_cycle_back.from_node = unsupported_cycle.id;
    unsupported_cycle_back.to_node = unsupported_cycle.id;
    const auto unsupported_cycle_traces = producer.TracePreprocessingGraph(
        "operator-producer-unsupported-cycle-run",
        {data, unsupported_cycle},
        {unsupported_cycle_in, unsupported_cycle_back},
        input);
    Check(unsupported_cycle_traces.size() == 1,
          "unsupported-only cycles should emit the unsupported node warning");
    Check(unsupported_cycle_traces[0].node_id == unsupported_cycle.id,
          "unsupported-only cycle warning should attach to the unsupported node");
    Check(unsupported_cycle_traces[0].payload["diagnostic_phase"].get<std::string>() ==
              "unsupported_operator",
          "unsupported-only cycle warning should identify unsupported-operator phase");
    Check(unsupported_cycle_traces[0].payload["error_code"].get<std::string>() ==
              cyxwiz::errors::Runtime::UnsupportedNode,
          "unsupported-only cycle warning should expose unsupported-node code");
    Check(!unsupported_cycle_traces[0].issues.empty() &&
              unsupported_cycle_traces[0].issues[0].error_code ==
                  cyxwiz::errors::Runtime::UnsupportedNode,
          "unsupported-only cycle issue should expose unsupported-node code");
    Check(unsupported_cycle_traces[0].payload["message"].get<std::string>().find(
              "supports TextTokenizer only") != std::string::npos,
          "unsupported-only cycle should not be masked by topology validation");

    const auto mixed_cycle_traces = producer.TracePreprocessingGraph(
        "operator-producer-mixed-unsupported-cycle-run",
        {data, tokenizer, unsupported_cycle},
        {link, unsupported_cycle_in, unsupported_cycle_back},
        input);
    Check(mixed_cycle_traces.size() == 2,
          "unsupported side cycles should not block a valid tokenizer trace");
    Check(mixed_cycle_traces[0].node_id == tokenizer.id,
          "mixed unsupported cycle should still emit the tokenizer trace");
    Check(mixed_cycle_traces[0].payload["operator_backed"].get<bool>(),
          "mixed unsupported cycle should preserve operator-backed tokenizer trace");
    Check(mixed_cycle_traces[1].node_id == unsupported_cycle.id,
          "mixed unsupported cycle should attach warning to unsupported node");
    Check(mixed_cycle_traces[1].role == cyxwiz::DebugTraceRole::Warning,
          "mixed unsupported cycle should emit unsupported-node warning");
    Check(mixed_cycle_traces[1].payload["diagnostic_phase"].get<std::string>() ==
              "unsupported_operator",
          "mixed unsupported cycle warning should identify unsupported-operator phase");
    Check(mixed_cycle_traces[1].payload["error_code"].get<std::string>() ==
              cyxwiz::errors::Runtime::UnsupportedNode,
          "mixed unsupported cycle warning should expose unsupported-node code");
    Check(!mixed_cycle_traces[1].issues.empty() &&
              mixed_cycle_traces[1].issues[0].error_code ==
                  cyxwiz::errors::Runtime::UnsupportedNode,
          "mixed unsupported cycle issue should expose unsupported-node code");
    Check(mixed_cycle_traces[1].payload["message"].get<std::string>().find(
              "supports TextTokenizer only") != std::string::npos,
          "mixed unsupported cycle should not report graph-walk topology failure");

    auto unsupported = MakeNode(3, gui::NodeType::Dense, "Dense Head");
    gui::NodeLink unsupported_link;
    unsupported_link.id = 102;
    unsupported_link.from_node = tokenizer.id;
    unsupported_link.to_node = unsupported.id;
    const auto unsupported_traces = producer.TracePreprocessingGraph(
        "operator-producer-warning-run",
        {data, tokenizer, unsupported},
        {link, unsupported_link},
        input);
    Check(unsupported_traces.size() == 2,
          "unsupported operator chain should keep tokenizer trace and warning trace");
    Check(unsupported_traces[1].role == cyxwiz::DebugTraceRole::Warning,
          "unsupported operator trace should use warning role");
    Check(unsupported_traces[1].node_type == "Dense",
          "unsupported operator trace should preserve readable node type");
    Check(unsupported_traces[1].input_shape == std::vector<size_t>{3, 5},
          "unsupported operator warning should receive tokenized table shape");
    Check(!unsupported_traces[1].issues.empty(),
          "unsupported operator trace should carry an issue");
    Check(unsupported_traces[1].payload["error_code"].get<std::string>() ==
              cyxwiz::errors::Runtime::UnsupportedNode,
          "unsupported operator trace should expose unsupported-node code");
    Check(unsupported_traces[1].issues[0].error_code ==
              cyxwiz::errors::Runtime::UnsupportedNode,
          "unsupported operator issue should preserve unsupported-node code");
    Check(unsupported_traces[1].payload["issue_count"].get<size_t>() == 1,
          "unsupported operator trace should expose issue count");
    Check(unsupported_traces[1].payload["warning_count"].get<size_t>() == 1,
          "unsupported operator trace should expose warning count");
    Check(unsupported_traces[1].payload["error_count"].get<size_t>() == 0,
          "unsupported operator trace should expose zero error count");
    Check(unsupported_traces[1].payload["primary_warning_code"].get<std::string>() ==
              cyxwiz::errors::Runtime::UnsupportedNode,
          "unsupported operator trace should expose primary warning code");
    Check(unsupported_traces[1].payload["issue_codes"][0].get<std::string>() ==
              cyxwiz::errors::Runtime::UnsupportedNode,
          "unsupported operator trace should expose issue code summary");
    Check(unsupported_traces[1].payload["component"].get<std::string>() ==
              "DebugOperatorTraceProducer",
          "unsupported operator trace should expose diagnostic component");
    Check(unsupported_traces[1].payload["source_file"].get<std::string>().find(
              "debug_operator_trace_producer.cpp") != std::string::npos,
          "unsupported operator trace should expose source file");
    Check(unsupported_traces[1].payload["source_symbol"].get<std::string>() ==
              "cyxwiz::DebugOperatorTraceProducer::BuildWarningTrace",
          "unsupported operator trace should expose source symbol");
    Check(!unsupported_traces[1].payload["operator_backed"].get<bool>(),
          "unsupported operator trace should not claim operator-backed execution");
    Check(!unsupported_traces[1].payload["success"].get<bool>(),
          "unsupported operator trace should mark unsuccessful");
    Check(unsupported_traces[1].payload["diagnostic_phase"].get<std::string>() ==
              "unsupported_operator",
          "unsupported operator warning should identify unsupported-operator phase");

    auto unsupported_before_tokenizer = MakeNode(
        9, gui::NodeType::Dense, "Unsupported Before Tokenizer");
    gui::NodeLink unsupported_before_tokenizer_link;
    unsupported_before_tokenizer_link.id = 113;
    unsupported_before_tokenizer_link.from_node = data.id;
    unsupported_before_tokenizer_link.to_node = unsupported_before_tokenizer.id;
    gui::NodeLink tokenizer_after_unsupported_link;
    tokenizer_after_unsupported_link.id = 114;
    tokenizer_after_unsupported_link.from_node = unsupported_before_tokenizer.id;
    tokenizer_after_unsupported_link.to_node = tokenizer.id;
    const auto tokenizer_after_unsupported_traces = producer.TracePreprocessingGraph(
        "operator-producer-tokenizer-after-unsupported-run",
        {data, unsupported_before_tokenizer, tokenizer},
        {unsupported_before_tokenizer_link, tokenizer_after_unsupported_link},
        input);
    Check(tokenizer_after_unsupported_traces.size() == 1,
          "unsupported operators should stop downstream tokenizer tracing");
    Check(tokenizer_after_unsupported_traces[0].node_id ==
              unsupported_before_tokenizer.id,
          "unsupported boundary warning should attach to the unsupported node");
    Check(tokenizer_after_unsupported_traces[0].role == cyxwiz::DebugTraceRole::Warning,
          "unsupported boundary trace should be warning-only");
    Check(tokenizer_after_unsupported_traces[0].node_type == "Dense",
          "unsupported boundary warning should preserve readable node type");
    Check(!tokenizer_after_unsupported_traces[0].payload["operator_backed"].get<bool>(),
          "unsupported boundary warning should not claim operator-backed execution");
    Check(tokenizer_after_unsupported_traces[0].payload["diagnostic_phase"].get<std::string>() ==
              "unsupported_operator",
          "unsupported boundary warning should identify unsupported-operator phase");
    Check(tokenizer_after_unsupported_traces[0].payload["error_code"].get<std::string>() ==
              cyxwiz::errors::Runtime::UnsupportedNode,
          "unsupported boundary warning should expose unsupported-node code");
    Check(!tokenizer_after_unsupported_traces[0].issues.empty() &&
              tokenizer_after_unsupported_traces[0].issues[0].error_code ==
                  cyxwiz::errors::Runtime::UnsupportedNode,
          "unsupported boundary issue should expose unsupported-node code");
    Check(tokenizer_after_unsupported_traces[0].payload["message"].get<std::string>().find(
              "supports TextTokenizer only") != std::string::npos,
          "unsupported boundary warning should explain skipped downstream trace");

    const auto direct_and_unsupported_tokenizer_traces = producer.TracePreprocessingGraph(
        "operator-producer-direct-and-unsupported-tokenizer-run",
        {data, tokenizer, unsupported_before_tokenizer},
        {link, unsupported_before_tokenizer_link, tokenizer_after_unsupported_link},
        input);
    Check(direct_and_unsupported_tokenizer_traces.size() == 2,
          "unsupported tokenizer side paths should not make valid tokenizer paths look branched");
    Check(direct_and_unsupported_tokenizer_traces[0].node_id == tokenizer.id,
          "valid direct tokenizer path should still emit tokenizer trace");
    Check(direct_and_unsupported_tokenizer_traces[0].payload["operator_backed"].get<bool>(),
          "valid direct tokenizer path should remain operator-backed");
    Check(direct_and_unsupported_tokenizer_traces[1].node_id ==
              unsupported_before_tokenizer.id,
          "unsupported side path should still emit unsupported-node warning");
    Check(direct_and_unsupported_tokenizer_traces[1].role ==
              cyxwiz::DebugTraceRole::Warning,
          "unsupported side path should be warning-only");
    Check(direct_and_unsupported_tokenizer_traces[1].payload["error_code"].get<std::string>() ==
              cyxwiz::errors::Runtime::UnsupportedNode,
          "unsupported side path warning should expose unsupported-node code");
    Check(!direct_and_unsupported_tokenizer_traces[1].issues.empty() &&
              direct_and_unsupported_tokenizer_traces[1].issues[0].error_code ==
                  cyxwiz::errors::Runtime::UnsupportedNode,
          "unsupported side path issue should expose unsupported-node code");
    Check(direct_and_unsupported_tokenizer_traces[1].payload["message"].get<std::string>().find(
              "supports TextTokenizer only") != std::string::npos,
          "unsupported side path should not be reported as branched topology");

    auto invalid_tokenizer = tokenizer;
    invalid_tokenizer.id = 8;
    invalid_tokenizer.name = "Invalid Tokenizer";
    invalid_tokenizer.parameters["max_length"] = "0";
    gui::NodeLink invalid_tokenizer_link;
    invalid_tokenizer_link.id = 110;
    invalid_tokenizer_link.from_node = data.id;
    invalid_tokenizer_link.to_node = invalid_tokenizer.id;
    gui::NodeLink invalid_downstream_link;
    invalid_downstream_link.id = 111;
    invalid_downstream_link.from_node = invalid_tokenizer.id;
    invalid_downstream_link.to_node = unsupported.id;
    const auto failed_tokenizer_traces = producer.TracePreprocessingGraph(
        "operator-producer-failed-tokenizer-run",
        {data, invalid_tokenizer, unsupported},
        {invalid_tokenizer_link, invalid_downstream_link},
        input);
    Check(failed_tokenizer_traces.size() == 1,
          "failed supported operators should stop downstream debugger tracing");
    Check(failed_tokenizer_traces[0].node_id == invalid_tokenizer.id,
          "failed tokenizer warning should attach to the tokenizer node");
    Check(failed_tokenizer_traces[0].role == cyxwiz::DebugTraceRole::Warning,
          "failed tokenizer trace should be warning-only");
    Check(failed_tokenizer_traces[0].payload["diagnostic_phase"].get<std::string>() ==
              "configure",
          "failed tokenizer trace should identify configure phase");
    Check(failed_tokenizer_traces[0].payload["message"].get<std::string>().find(
              "max_length") != std::string::npos,
          "failed tokenizer warning should preserve operator error text");
    Check(failed_tokenizer_traces[0].payload["effective_text_tokenizer_config"]["max_length"].get<std::string>() ==
              "0",
          "failed tokenizer warning should expose effective invalid config");
    Check(failed_tokenizer_traces[0].payload["folded_text_config_nodes"].empty(),
          "failed plain tokenizer warning should expose no folded config provenance");
    Check(!failed_tokenizer_traces[0].payload["operator_backed"].get<bool>(),
          "failed tokenizer warning should not claim operator-backed execution");
    Check(!failed_tokenizer_traces[0].payload["success"].get<bool>(),
          "failed tokenizer warning should mark unsuccessful");
    Check(failed_tokenizer_traces[0].payload["issue_count"].get<size_t>() == 1,
          "failed tokenizer warning should expose issue count");
    Check(failed_tokenizer_traces[0].payload["primary_warning_code"].get<std::string>() ==
              cyxwiz::errors::Runtime::ExecutionFailed,
          "failed tokenizer warning should expose execution-failed warning code");
    Check(failed_tokenizer_traces[0].payload["source_symbol"].get<std::string>() ==
              "cyxwiz::DebugOperatorTraceProducer::BuildWarningTrace",
          "failed tokenizer warning should expose source symbol");
    const auto test_root = std::filesystem::temp_directory_path() /
        "cyxwiz_debugger_operator_trace_producer_store";
    std::filesystem::remove_all(test_root);
    std::filesystem::create_directories(test_root);
    const cyxwiz::ScopedDebugRunRootOverrideForTesting debug_root(
        test_root / "debug_runs");

    cyxwiz::DebugRunStoreRecord record;
    record.summary.run_id = "operator-producer-store-run";
    record.summary.timestamp = "2026-07-09T00:00:00";
    record.summary.graph_hash = 0x320032;
    record.summary.success = true;
    record.summary.summary = "Operator producer trace persistence";
    record.traces.push_back(trace);

    Check(cyxwiz::DebugRunStore::Save(record),
          "operator producer traces should persist through DebugRunStore");
    auto loaded = cyxwiz::DebugRunStore::Load(record.summary.run_id);
    Check(loaded.has_value(),
          "operator producer persisted run should load");
    Check(loaded->traces.size() == 1,
          "operator producer persisted run should contain trace");
    Check(loaded->traces[0].payload["trace_producer"].get<std::string>() ==
              "DebugOperatorTraceProducer",
          "operator producer payload should round-trip through store");
    Check(std::filesystem::exists(
              cyxwiz::GetDebugRunRoot() / "studio" /
              record.summary.run_id / "session.json"),
          "operator producer store should use the injected debug-run root");

    std::filesystem::remove_all(test_root);
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

    cyxwiz::DebugTraceRecord operator_warning = cyxwiz::DebugNodeTraceContract::Make(
        "recommendation-run",
        11,
        "UnsupportedPreprocess",
        "UnsupportedPreprocess",
        "OperatorTransform",
        cyxwiz::DebugTraceRole::Warning,
        {2, 3},
        {2, 3},
        "arrow",
        "Arrow",
        "warning");
    cyxwiz::DebugNodeTraceContract::AddWarning(
        operator_warning,
        "Operator-backed trace skipped unsupported preprocessing operator.",
        cyxwiz::errors::Runtime::UnsupportedNode);
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(operator_warning),
          "trace issue recommendation fixture should use canonical trace schema");
    traces.push_back(std::move(operator_warning));

    cyxwiz::DebugTraceRecord duplicate_warning = cyxwiz::DebugNodeTraceContract::Make(
        "recommendation-run",
        12,
        "DuplicateWarning",
        "Preflight",
        "Preflight",
        cyxwiz::DebugTraceRole::Warning,
        {},
        {},
        "graph",
        "PreflightValidator",
        "blocked");
    cyxwiz::DebugNodeTraceContract::AddWarning(
        duplicate_warning,
        "Duplicate warning should be recommended only once.",
        cyxwiz::errors::Compiler::InvalidParameter);
    const cyxwiz::ValidationIssue duplicate_issue = duplicate_warning.issues[0];
    traces.push_back(std::move(duplicate_warning));

    cyxwiz::DebugTraceRecord loss = cyxwiz::DebugNodeTraceContract::Make(
        "recommendation-run",
        4,
        "SmokeRun",
        "SmokeRun",
        "SmokeRun.Loss",
        cyxwiz::DebugTraceRole::Loss,
        {2, 3},
        {1},
        "float32",
        "Runtime",
        "failed");
    cyxwiz::DebugNodeTraceContract::AttachDiagnosticContext(
        loss,
        "smoke_run",
        "SmokeRunExecutor",
        "cyxwiz-engine/src/core/smoke_run_executor.cpp",
        "cyxwiz::SmokeRunExecutor::RunTextSmoke");
    loss.payload["predictions_have_non_finite"] = true;
    loss.payload["loss"] = std::numeric_limits<double>::quiet_NaN();
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(loss),
          "smoke loss recommendation fixture should use canonical trace schema");
    traces.push_back(std::move(loss));

    cyxwiz::DebugTraceRecord backward = cyxwiz::DebugNodeTraceContract::Make(
        "recommendation-run",
        4,
        "SmokeRun",
        "SmokeRun",
        "SmokeRun.Backward",
        cyxwiz::DebugTraceRole::Gradient,
        {},
        {},
        "float32",
        "Runtime",
        "warning");
    cyxwiz::DebugNodeTraceContract::AttachDiagnosticContext(
        backward,
        "smoke_run",
        "SmokeRunExecutor",
        "cyxwiz-engine/src/core/smoke_run_executor.cpp",
        "cyxwiz::SmokeRunExecutor::RunTextSmoke");
    backward.payload["gradient_tensor_count"] = 0;
    backward.payload["zero_gradient_tensor_count"] = 0;
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(backward),
          "smoke backward recommendation fixture should use canonical trace schema");
    Check(backward.payload["diagnostic_phase"].get<std::string>() == "smoke_run",
          "smoke backward recommendation fixture should expose diagnostic phase");
    traces.push_back(std::move(backward));

    cyxwiz::DebugTraceRecord zero_backward = cyxwiz::DebugNodeTraceContract::Make(
        "recommendation-run",
        5,
        "SmokeRun",
        "SmokeRun",
        "SmokeRun.Backward",
        cyxwiz::DebugTraceRole::Gradient,
        {},
        {},
        "float32",
        "Runtime",
        "warning");
    cyxwiz::DebugNodeTraceContract::AttachDiagnosticContext(
        zero_backward,
        "smoke_run",
        "SmokeRunExecutor",
        "cyxwiz-engine/src/core/smoke_run_executor.cpp",
        "cyxwiz::SmokeRunExecutor::RunTextSmoke");
    zero_backward.payload["gradient_tensor_count"] = 3;
    zero_backward.payload["zero_gradient_tensor_count"] = 3;
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(zero_backward),
          "smoke all-zero-gradient recommendation fixture should use canonical trace schema");
    traces.push_back(std::move(zero_backward));

    cyxwiz::DebugTraceRecord local_build = cyxwiz::DebugNodeTraceContract::Make(
        "recommendation-run",
        -1,
        "LocalDebugBuildModel",
        "BuildModel",
        "BuildModel",
        cyxwiz::DebugTraceRole::Error,
        {},
        {},
        "model",
        "LocalDebug",
        "failed");
    cyxwiz::DebugNodeTraceContract::AttachDiagnosticContext(
        local_build,
        "local_debug_build_model",
        "DebugExecutor",
        "cyxwiz-engine/src/core/debug_executor.cpp",
        "cyxwiz::DebugExecutor::Run");
    local_build.payload["build_model_reached"] = true;
    local_build.payload["success"] = false;
    cyxwiz::DebugNodeTraceContract::AddError(
        local_build,
        "Model build failed (invalid config)",
        cyxwiz::errors::Training::ModelBuildFailed);
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(local_build),
          "local debug build-model recommendation fixture should use canonical trace schema");
    traces.push_back(std::move(local_build));

    cyxwiz::DebugTraceRecord local_loss = cyxwiz::DebugNodeTraceContract::Make(
        "recommendation-run",
        -1,
        "LocalDebugLoss",
        "Loss",
        "Loss",
        cyxwiz::DebugTraceRole::Loss,
        {},
        {},
        "float32",
        "LocalDebug",
        "failed");
    cyxwiz::DebugNodeTraceContract::AttachDiagnosticContext(
        local_loss,
        "local_debug_loss",
        "DebugExecutor",
        "cyxwiz-engine/src/core/debug_executor.cpp",
        "cyxwiz::DebugExecutor::Run");
    local_loss.payload["loss"] = std::numeric_limits<double>::quiet_NaN();
    local_loss.payload["loss_finite"] = false;
    local_loss.payload["loss_stage_failed"] = false;
    local_loss.payload["success"] = false;
    cyxwiz::DebugNodeTraceContract::AddError(
        local_loss,
        "Local Debug loss was not finite.",
        cyxwiz::errors::Training::TrainingExecutionFailed);
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(local_loss),
          "local debug loss recommendation fixture should use canonical trace schema");
    Check(!local_loss.payload["success"].get<bool>(),
          "local debug non-finite loss fixture should mark trace unsuccessful");
    Check(!local_loss.payload["loss_stage_failed"].get<bool>(),
          "local debug non-finite loss fixture should stay distinct from loss-stage failure");
    traces.push_back(std::move(local_loss));

    cyxwiz::DebugTraceRecord local_loss_failed = cyxwiz::DebugNodeTraceContract::Make(
        "recommendation-run",
        -1,
        "LocalDebugLoss",
        "Loss",
        "Loss",
        cyxwiz::DebugTraceRole::Loss,
        {},
        {},
        "float32",
        "LocalDebug",
        "failed");
    cyxwiz::DebugNodeTraceContract::AttachDiagnosticContext(
        local_loss_failed,
        "local_debug_loss",
        "DebugExecutor",
        "cyxwiz-engine/src/core/debug_executor.cpp",
        "cyxwiz::DebugExecutor::Run");
    local_loss_failed.payload["loss"] = std::numeric_limits<double>::quiet_NaN();
    local_loss_failed.payload["loss_finite"] = false;
    local_loss_failed.payload["loss_stage_failed"] = true;
    local_loss_failed.payload["success"] = false;
    cyxwiz::DebugNodeTraceContract::AddError(
        local_loss_failed,
        "Exception during Loss: target shape mismatch",
        cyxwiz::errors::Training::TrainingExecutionFailed);
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(local_loss_failed),
          "local debug failed-loss recommendation fixture should use canonical trace schema");
    traces.push_back(std::move(local_loss_failed));

    cyxwiz::DebugTraceRecord local_optimizer = cyxwiz::DebugNodeTraceContract::Make(
        "recommendation-run",
        -1,
        "LocalDebugOptimizerStep",
        "OptimizerStep",
        "OptimizerStep",
        cyxwiz::DebugTraceRole::OptimizerStep,
        {},
        {},
        "optimizer",
        "LocalDebug",
        "failed");
    cyxwiz::DebugNodeTraceContract::AttachDiagnosticContext(
        local_optimizer,
        "local_debug_optimizer",
        "DebugExecutor",
        "cyxwiz-engine/src/core/debug_executor.cpp",
        "cyxwiz::DebugExecutor::Run");
    local_optimizer.payload["optimizer_step_reached"] = true;
    local_optimizer.payload["success"] = false;
    cyxwiz::DebugNodeTraceContract::AddError(
        local_optimizer,
        "Exception during OptimizerStep: optimizer update failed",
        cyxwiz::errors::Training::TrainingExecutionFailed);
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(local_optimizer),
          "local debug optimizer recommendation fixture should use canonical trace schema");
    traces.push_back(std::move(local_optimizer));

    cyxwiz::DebugTraceRecord local_forward = cyxwiz::DebugNodeTraceContract::Make(
        "recommendation-run",
        7,
        "Dense_7",
        "Dense",
        "Forward",
        cyxwiz::DebugTraceRole::Activation,
        {},
        {2, 4},
        "float32",
        "LocalDebug",
        "warning");
    cyxwiz::DebugNodeTraceContract::AttachDiagnosticContext(
        local_forward,
        "local_debug_forward",
        "DebugExecutor",
        "cyxwiz-engine/src/core/debug_executor.cpp",
        "cyxwiz::DebugExecutor::Run");
    local_forward.payload["has_nan"] = true;
    local_forward.payload["has_inf"] = false;
    local_forward.payload["success"] = false;
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(local_forward),
          "local debug forward recommendation fixture should use canonical trace schema");
    Check(!local_forward.payload["success"].get<bool>(),
          "local debug non-finite forward fixture should mark trace unsuccessful");
    traces.push_back(std::move(local_forward));

    cyxwiz::DebugTraceRecord local_forward_shape = cyxwiz::DebugNodeTraceContract::Make(
        "recommendation-run",
        11,
        "Dense_11",
        "Dense",
        "Forward",
        cyxwiz::DebugTraceRole::Activation,
        {2, 8},
        {2, 4},
        "float32",
        "LocalDebug",
        "shape_mismatch");
    cyxwiz::DebugNodeTraceContract::AttachDiagnosticContext(
        local_forward_shape,
        "local_debug_forward",
        "DebugExecutor",
        "cyxwiz-engine/src/core/debug_executor.cpp",
        "cyxwiz::DebugExecutor::Run");
    local_forward_shape.payload["shape_matches"] = false;
    local_forward_shape.payload["success"] = false;
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(local_forward_shape),
          "local debug shape-mismatch recommendation fixture should use canonical trace schema");
    Check(!local_forward_shape.payload["success"].get<bool>(),
          "local debug shape-mismatch fixture should mark trace unsuccessful");
    traces.push_back(std::move(local_forward_shape));

    cyxwiz::DebugTraceRecord runtime_shape = cyxwiz::DebugNodeTraceContract::Make(
        "recommendation-run",
        12,
        "RuntimeDense_12",
        "Dense",
        "Forward",
        cyxwiz::DebugTraceRole::Activation,
        {2, 8},
        {2, 4},
        "float32",
        "Runtime",
        "shape_mismatch");
    runtime_shape.payload["shape_matches"] = false;
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(runtime_shape),
          "runtime shape-mismatch recommendation fixture should use canonical trace schema");
    traces.push_back(std::move(runtime_shape));

    cyxwiz::DebugTraceRecord runtime_forward = cyxwiz::DebugNodeTraceContract::Make(
        "recommendation-run",
        17,
        "RuntimeDense_17",
        "Dense",
        "Forward",
        cyxwiz::DebugTraceRole::Activation,
        {},
        {2, 4},
        "float32",
        "Runtime",
        "warning");
    runtime_forward.payload["has_nan"] = true;
    runtime_forward.payload["has_inf"] = false;
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(runtime_forward),
          "runtime forward recommendation fixture should use canonical trace schema");
    traces.push_back(std::move(runtime_forward));

    cyxwiz::DebugTraceRecord local_forward_failed = cyxwiz::DebugNodeTraceContract::Make(
        "recommendation-run",
        -1,
        "LocalDebugForward",
        "Forward",
        "Forward",
        cyxwiz::DebugTraceRole::Activation,
        {},
        {},
        "float32",
        "LocalDebug",
        "failed");
    cyxwiz::DebugNodeTraceContract::AttachDiagnosticContext(
        local_forward_failed,
        "local_debug_forward",
        "DebugExecutor",
        "cyxwiz-engine/src/core/debug_executor.cpp",
        "cyxwiz::DebugExecutor::Run");
    local_forward_failed.payload["forward_reached"] = true;
    local_forward_failed.payload["success"] = false;
    cyxwiz::DebugNodeTraceContract::AddError(
        local_forward_failed,
        "Exception during Forward: layer input shape mismatch",
        cyxwiz::errors::Training::TrainingExecutionFailed);
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(local_forward_failed),
          "local debug failed-forward recommendation fixture should use canonical trace schema");
    traces.push_back(std::move(local_forward_failed));

    cyxwiz::DebugTraceRecord local_backward_failed = cyxwiz::DebugNodeTraceContract::Make(
        "recommendation-run",
        -1,
        "LocalDebugBackward",
        "Backward",
        "Backward",
        cyxwiz::DebugTraceRole::Error,
        {},
        {},
        "float32",
        "LocalDebug",
        "failed");
    cyxwiz::DebugNodeTraceContract::AttachDiagnosticContext(
        local_backward_failed,
        "local_debug_backward",
        "DebugExecutor",
        "cyxwiz-engine/src/core/debug_executor.cpp",
        "cyxwiz::DebugExecutor::Run");
    local_backward_failed.payload["backward_reached"] = true;
    local_backward_failed.payload["success"] = false;
    cyxwiz::DebugNodeTraceContract::AddError(
        local_backward_failed,
        "Exception during Backward: tensor shape mismatch",
        cyxwiz::errors::Training::TrainingExecutionFailed);
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(local_backward_failed),
          "local debug failed-backward recommendation fixture should use canonical trace schema");
    traces.push_back(std::move(local_backward_failed));

    cyxwiz::DebugTraceRecord local_gradient = cyxwiz::DebugNodeTraceContract::Make(
        "recommendation-run",
        8,
        "Dense_8.weight",
        "Parameter",
        "Backward",
        cyxwiz::DebugTraceRole::Gradient,
        {},
        {},
        "float32",
        "LocalDebug",
        "nan");
    cyxwiz::DebugNodeTraceContract::AttachDiagnosticContext(
        local_gradient,
        "local_debug_backward",
        "DebugExecutor",
        "cyxwiz-engine/src/core/debug_executor.cpp",
        "cyxwiz::DebugExecutor::Run");
    local_gradient.payload["has_gradient"] = true;
    local_gradient.payload["is_nan"] = true;
    local_gradient.payload["is_zero"] = false;
    local_gradient.payload["success"] = false;
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(local_gradient),
          "local debug gradient recommendation fixture should use canonical trace schema");
    Check(!local_gradient.payload["success"].get<bool>(),
          "local debug NaN-gradient fixture should mark trace unsuccessful");
    traces.push_back(std::move(local_gradient));

    cyxwiz::DebugTraceRecord runtime_gradient = cyxwiz::DebugNodeTraceContract::Make(
        "recommendation-run",
        18,
        "RuntimeDense_18.weight",
        "Parameter",
        "Backward",
        cyxwiz::DebugTraceRole::Gradient,
        {},
        {},
        "float32",
        "Runtime",
        "nan");
    runtime_gradient.payload["has_gradient"] = true;
    runtime_gradient.payload["is_nan"] = true;
    runtime_gradient.payload["is_zero"] = false;
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(runtime_gradient),
          "runtime gradient recommendation fixture should use canonical trace schema");
    traces.push_back(std::move(runtime_gradient));

    cyxwiz::DebugTraceRecord local_missing_gradient = cyxwiz::DebugNodeTraceContract::Make(
        "recommendation-run",
        10,
        "Dense_10.weight",
        "Parameter",
        "Backward",
        cyxwiz::DebugTraceRole::Gradient,
        {},
        {},
        "float32",
        "LocalDebug",
        "missing_gradient");
    cyxwiz::DebugNodeTraceContract::AttachDiagnosticContext(
        local_missing_gradient,
        "local_debug_backward",
        "DebugExecutor",
        "cyxwiz-engine/src/core/debug_executor.cpp",
        "cyxwiz::DebugExecutor::Run");
    local_missing_gradient.payload["has_gradient"] = false;
    local_missing_gradient.payload["is_nan"] = false;
    local_missing_gradient.payload["is_zero"] = true;
    local_missing_gradient.payload["success"] = false;
    cyxwiz::DebugNodeTraceContract::AddWarning(
        local_missing_gradient,
        "Local Debug gradient was missing.",
        cyxwiz::errors::Training::TrainingExecutionFailed);
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(local_missing_gradient),
          "local debug missing-gradient recommendation fixture should use canonical trace schema");
    Check(!local_missing_gradient.payload["success"].get<bool>(),
          "local debug missing-gradient fixture should mark trace unsuccessful");
    traces.push_back(std::move(local_missing_gradient));

    cyxwiz::DebugTraceRecord local_zero_gradient = cyxwiz::DebugNodeTraceContract::Make(
        "recommendation-run",
        9,
        "Dense_9.bias",
        "Parameter",
        "Backward",
        cyxwiz::DebugTraceRole::Gradient,
        {},
        {},
        "float32",
        "LocalDebug",
        "zero");
    cyxwiz::DebugNodeTraceContract::AttachDiagnosticContext(
        local_zero_gradient,
        "local_debug_backward",
        "DebugExecutor",
        "cyxwiz-engine/src/core/debug_executor.cpp",
        "cyxwiz::DebugExecutor::Run");
    local_zero_gradient.payload["has_gradient"] = true;
    local_zero_gradient.payload["is_nan"] = false;
    local_zero_gradient.payload["is_zero"] = true;
    local_zero_gradient.payload["success"] = false;
    cyxwiz::DebugNodeTraceContract::AddWarning(
        local_zero_gradient,
        "Local Debug gradient norm was zero.",
        cyxwiz::errors::Training::TrainingExecutionFailed);
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(local_zero_gradient),
          "local debug zero-gradient recommendation fixture should use canonical trace schema");
    Check(local_zero_gradient.payload["warning_count"].get<size_t>() == 1,
          "local debug zero-gradient fixture should expose issue summary");
    Check(!local_zero_gradient.payload["success"].get<bool>(),
          "local debug zero-gradient fixture should mark trace unsuccessful");
    traces.push_back(std::move(local_zero_gradient));

    cyxwiz::DebugExportCorrelationInput export_input;
    export_input.artifact_kind = "onnx";
    export_input.exporter_name = "ONNXExporter";
    export_input.compile_success = false;
    export_input.compile_status = "compile failed before export";
    export_input.generated_content = "";
    cyxwiz::DebugExportCorrelationTracer export_tracer;
    auto export_trace = export_tracer.BuildTrace(
        "recommendation-run",
        export_input);
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(export_trace),
          "export recommendation fixture should use canonical trace schema");
    traces.push_back(std::move(export_trace));

    cyxwiz::DebugArtifactConsistencyInput artifact_input;
    artifact_input.action = "export";
    artifact_input.artifact_kind = "CyxWiz Model";
    artifact_input.artifact_path = "exports/incomplete.cyxmodel";
    artifact_input.operation_success = true;
    artifact_input.operation_status = "export completed";
    artifact_input.expected.manifest_expected = true;
    artifact_input.observed.available = true;
    artifact_input.observed.manifest_valid = true;
    artifact_input.observed.evidence_scope =
        "cyxmodel_directory_manifest_and_inventory";
    artifact_input.observed.manifest_present = false;
    traces.push_back(export_tracer.BuildConsistencyTrace(
        "recommendation-run", artifact_input));

    cyxwiz::CrashRunSummary crash_run;
    crash_run.available = true;
    crash_run.suspected_crash = true;
    crash_run.run_id = "recommendation-run";
    crash_run.status = "suspected crash";
    cyxwiz::DebugWindowsCrashImporter crash_importer;
    auto missing_crash_report = crash_importer.ParseWerText("", "");
    auto crash_trace = crash_importer.BuildTrace(
        "recommendation-run",
        crash_run,
        missing_crash_report);
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(crash_trace),
          "crash recommendation fixture should use canonical trace schema");
    traces.push_back(std::move(crash_trace));

    cyxwiz::SmokeRunResult smoke;
    smoke.supported = true;
    smoke.success = false;
    smoke.summary = "Smoke Run failed for debugger contract test.";

    cyxwiz::CrashRunSummary last_run;
    cyxwiz::TrainingTraceSummary training_trace;

    cyxwiz::DebugRecommendationEngine engine;
    const auto recs = engine.Build(
        traces,
        {duplicate_issue},
        smoke,
        last_run,
        training_trace);

    Check(HasRecommendation(recs, "High unknown-token ratio"),
          "high unknown-token ratio should produce recommendation");
    Check(HasRecommendation(recs, "Text sample was truncated"),
          "truncation should produce recommendation");
    Check(CountRecommendationsWithDetail(
              recs,
              "Trace warning reported",
              "Operator-backed trace skipped unsupported preprocessing operator.") == 1,
          "unique trace warning issue should produce recommendation");
    Check(CountRecommendationsWithDetail(
              recs,
              "Trace warning reported",
              "Duplicate warning should be recommended only once.") == 0,
          "duplicate trace warning issue should not produce duplicate recommendation");
    Check(HasRecommendation(recs, "Smoke Run produced invalid values"),
          "invalid smoke loss should produce recommendation");
    Check(HasRecommendation(recs, "No gradients observed"),
          "missing gradients should produce recommendation");
    Check(HasRecommendation(recs, "All gradients are zero"),
          "all-zero smoke gradients should produce recommendation");
    Check(HasRecommendation(recs, "Local Debug model build failed"),
          "local debug model build failure should produce recommendation");
    Check(HasRecommendation(recs, "Local Debug loss is not finite"),
          "local debug non-finite loss should produce recommendation");
    Check(HasRecommendation(recs, "Local Debug loss stage failed"),
          "local debug loss stage failure should produce recommendation");
    Check(HasRecommendation(recs, "Local Debug optimizer step failed"),
          "local debug optimizer failure should produce recommendation");
    Check(HasRecommendation(recs, "Local Debug forward shape mismatch"),
          "local debug forward shape mismatch should produce focused recommendation");
    Check(HasRecommendation(recs, "Shape mismatch detected"),
          "non-local shape mismatch should keep generic recommendation");
    Check(HasRecommendation(recs, "Local Debug produced non-finite activation"),
          "local debug non-finite activation should produce recommendation");
    Check(CountRecommendationsWithDetail(
              recs,
              "Local Debug produced non-finite activation",
              "A Local Debug forward trace reported NaN or Inf output values.") == 1,
          "runtime forward non-finite fixture should not produce Local Debug recommendation");
    Check(HasRecommendation(recs, "Local Debug forward pass failed"),
          "local debug forward failure should produce recommendation");
    Check(HasRecommendation(recs, "Local Debug backward pass failed"),
          "local debug backward failure should produce recommendation");
    Check(HasRecommendation(recs, "Local Debug gradient is NaN"),
          "local debug NaN gradient should produce recommendation");
    Check(CountRecommendationsWithDetail(
              recs,
              "Local Debug gradient is NaN",
              "A Local Debug gradient trace reported a NaN parameter norm.") == 1,
          "runtime gradient NaN fixture should not produce Local Debug recommendation");
    Check(HasRecommendation(recs, "Local Debug gradient is missing"),
          "local debug missing gradient should produce recommendation");
    Check(HasRecommendation(recs, "Local Debug gradient is zero"),
          "local debug zero gradient should produce recommendation");
    Check(HasRecommendation(recs, "Export correlation failed"),
          "failed export correlation trace should produce recommendation");
    Check(HasRecommendation(recs, "Export artifact path missing"),
          "missing export artifact path should produce recommendation");
    Check(HasRecommendation(recs, "Export/import consistency failed"),
          "artifact consistency mismatch should produce recommendation");
    Check(HasRecommendation(recs, "Windows crash report unavailable"),
          "missing Windows crash report should produce recommendation");
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
    cyxwiz::DebugTraceRecord trace;
    trace.issues = preflight_result.issues;
    trace.payload["success"] = preflight_result.ready;
    cyxwiz::DebugNodeTraceContract::AttachIssueSummary(trace, trace.issues);
    Check(!trace.payload["success"].get<bool>(),
          "blocked preflight diagnostic should mark unsuccessful");
    Check(trace.payload["issue_count"].get<size_t>() == trace.issues.size(),
          "issue summary should preserve total issue count");
    Check(trace.payload["error_count"].get<size_t>() == trace.issues.size(),
          "issue summary should count preflight errors");
    Check(trace.payload["warning_count"].get<size_t>() == 0,
          "issue summary should count preflight warnings");
    Check(trace.payload["primary_error_code"].get<std::string>() ==
              cyxwiz::errors::Compiler::MissingTrainingPathNode,
          "issue summary should expose the first error code");
    Check(trace.payload["issue_codes"].size() == 4,
          "issue summary should expose unique issue codes");
    Check(trace.payload["issue_codes"][1].get<std::string>() ==
              cyxwiz::errors::Runtime::InputDatasetMissing,
          "issue summary should preserve first-seen code order");

    cyxwiz::DebugNodeTraceContract::AttachDiagnosticContext(
        trace,
        "preflight",
        "PreflightValidator",
        "cyxwiz-engine/src/core/preflight_validator.cpp",
        "cyxwiz::PreflightValidator::Validate");
    Check(trace.payload["diagnostic_phase"].get<std::string>() == "preflight",
          "diagnostic context should preserve phase");
    Check(trace.payload["component"].get<std::string>() == "PreflightValidator",
          "diagnostic context should preserve component");
    Check(trace.payload["source_file"].get<std::string>().find("preflight_validator.cpp") !=
              std::string::npos,
          "diagnostic context should preserve source file");
    Check(trace.payload["source_symbol"].get<std::string>() ==
              "cyxwiz::PreflightValidator::Validate",
          "diagnostic context should preserve source symbol");
}

void TestDebugRunStoreContract() {
    const auto test_root =
        std::filesystem::temp_directory_path() / "cyxwiz_debugger_contract_store";
    std::filesystem::remove_all(test_root);
    std::filesystem::create_directories(test_root);
    const cyxwiz::ScopedDebugRunRootOverrideForTesting debug_root(
        test_root / "debug_runs");

    cyxwiz::DebugRunStoreRecord record;
    record.summary.run_id = "store-contract-run";
    record.summary.timestamp = "2026-06-18T00:00:00";
    record.summary.graph_hash = 0xCAFE;
    record.summary.success = true;
    record.summary.summary = "Debugger store contract run";

    cyxwiz::TrainingTraceSummary execution_trace;
    execution_trace.available = true;
    execution_trace.run_id = "training-store-contract-run";
    execution_trace.status = "completed";
    execution_trace.requested_backend = "arrayfire_cuda";
    execution_trace.requested_device_id = 2;
    execution_trace.effective_backend = "arrayfire_cuda";
    execution_trace.effective_device_id = 2;
    execution_trace.effective_device_name = "Contract CUDA Device";
    execution_trace.execution_context_id = "arrayfire:arrayfire_cuda:2";
    execution_trace.placement_fingerprint = "placement-contract";
    execution_trace.residency_verdict = "device_resident";
    execution_trace.native_cpu_fallback_count = 0;
    execution_trace.transfer_event_count = 3;
    execution_trace.transfer_known_bytes = 4096;
    execution_trace.synchronization_event_count = 2;
    execution_trace.synchronization_known_bytes = 1024;
    record.summary.execution =
        cyxwiz::MakeDebugRunExecutionSummary(execution_trace);

    auto replay_data = MakeNode(
        1, gui::NodeType::DataInput, "Replay Data Input");
    replay_data.parameters = {{"dataset_name", "replay_dataset"}};
    auto replay_dense = MakeNode(
        2, gui::NodeType::Dense, "Replay Dense");
    gui::NodeLink replay_link;
    replay_link.id = 101;
    replay_link.from_node = 1;
    replay_link.from_pin = 0;
    replay_link.to_node = 2;
    replay_link.to_pin = 0;
    const auto replay_session = cyxwiz::DebugSessionManager::StartSession(
        record.summary.run_id,
        "SmokeRun",
        record.summary.graph_hash,
        {replay_data, replay_dense},
        {replay_link},
        9);
    cyxwiz::TrainingConfiguration replay_config;
    replay_config.is_valid = true;
    replay_config.dataset_name = "replay_dataset";
    replay_config.input_shape = {8};
    replay_config.input_size = 8;
    replay_config.output_size = 3;
    replay_config.batch_size = 64;
    replay_config.epochs = 5;
    replay_config.shuffle = true;
    replay_config.drop_last = true;
    replay_config.num_workers = 2;
    replay_config.prefetch_factor = 3;
    replay_config.log_interval = 4;
    replay_config.validation_freq = 2;
    replay_config.grad_accum_steps = 2;
    replay_config.train_ratio = 0.7f;
    replay_config.val_ratio = 0.2f;
    replay_config.test_ratio = 0.1f;
    replay_config.stratified = true;
    replay_config.split_seed = 17;
    replay_config.dataloader_seed = 19;
    replay_config.balance_seed = 23;
    replay_config.learning_rate = 0.005f;
    replay_config.compiler_placement_fingerprint = "replay-placement";
    replay_config.forbid_native_cpu_fallback = true;
    record.replay_capsule = cyxwiz::MakeDebugRunReplayCapsule(
        replay_session, &replay_config, record.summary.execution, 100);

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

    cyxwiz::BackendPlacementEntry stored_placement;
    stored_placement.node_id = 8;
    stored_placement.node_name = "StoredBackendNode";
    stored_placement.node_type = "Dense";
    stored_placement.expected_backend = "ArrayFire active backend";
    stored_placement.status = cyxwiz::BackendPlacementStatus::Gpu;
    stored_placement.reason_code =
        cyxwiz::BackendPlacementReason::ArrayFireTensorOpCapable;
    cyxwiz::DebugRuntimeBackendClassifier backend_classifier;
    record.traces.push_back(backend_classifier.BuildPlacementTrace(
        record.summary.run_id, record.summary.graph_hash, stored_placement));
    cyxwiz::DebugErrorCodeTimeline error_timeline;
    record.traces.push_back(error_timeline.BuildTrace(
        record.summary.run_id, record.traces));

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
    Check(loaded->summary.trace_count == 3,
          "loaded trace count should match persisted trace count");
    Check(loaded->summary.event_count == 1,
          "loaded event count should match persisted event count");
    Check(loaded->summary.recommendation_count == 1,
          "loaded recommendation count should match persisted recommendation count");
    Check(loaded->summary.execution.available &&
              loaded->summary.execution.training_run_id ==
                  execution_trace.run_id &&
              loaded->summary.execution.effective_backend ==
                  execution_trace.effective_backend &&
              loaded->summary.execution.effective_device_id == 2 &&
              loaded->summary.execution.effective_device_name ==
                  execution_trace.effective_device_name &&
              loaded->summary.execution.residency_verdict ==
                  execution_trace.residency_verdict &&
              loaded->summary.execution.transfer_event_count == 3 &&
              loaded->summary.execution.transfer_known_bytes == 4096 &&
              loaded->summary.execution.synchronization_event_count == 2 &&
              loaded->summary.execution.synchronization_known_bytes == 1024,
          "linked training execution summary should round-trip");
    Check(loaded->replay_capsule.available &&
              loaded->replay_capsule.mode == "SmokeRun" &&
              loaded->replay_capsule.graph_hash == 0xCAFE &&
              loaded->replay_capsule.graph_snapshot_trace_available &&
              loaded->replay_capsule.graph_snapshot_trace_index == 0,
          "replay capsule should reference the persisted graph snapshot trace");
    Check(loaded->replay_capsule.dataset_reference == "replay_dataset" &&
              loaded->replay_capsule.selected_sample_index == 9 &&
              loaded->replay_capsule.smoke_sample_limit == 100 &&
              loaded->replay_capsule.smoke_batch_size_limit == 32,
          "replay capsule should preserve dataset and bounded sample selection");
    Check(loaded->replay_capsule.compiled_config.available &&
              loaded->replay_capsule.compiled_config.valid &&
              loaded->replay_capsule.compiled_config.input_shape ==
                  std::vector<size_t>{8} &&
              loaded->replay_capsule.compiled_config.batch_size == 64 &&
              loaded->replay_capsule.compiled_config.grad_accum_steps == 2 &&
              loaded->replay_capsule.compiled_config.learning_rate == 0.005f &&
              loaded->replay_capsule.compiled_config
                      .compiler_placement_fingerprint == "replay-placement" &&
              loaded->replay_capsule.compiled_config
                  .forbid_native_cpu_fallback,
          "replay capsule should round-trip reconstruction-critical config");
    Check(loaded->replay_capsule.split_seed == 17 &&
              loaded->replay_capsule.dataloader_seed == 19 &&
              loaded->replay_capsule.balance_seed == 23,
          "replay capsule should preserve all run-affecting seeds");
    Check(loaded->replay_capsule.backend_evidence_scope ==
                  "linked_training_run" &&
              loaded->replay_capsule.backend_source_run_id ==
                  execution_trace.run_id &&
              loaded->replay_capsule.requested_backend ==
                  execution_trace.requested_backend &&
              loaded->replay_capsule.effective_backend ==
                  execution_trace.effective_backend,
          "replay capsule should preserve scoped backend selection evidence");
    Check(loaded->replay_capsule.environment.count("platform") == 1 &&
              loaded->replay_capsule.environment.count("architecture") == 1 &&
              loaded->replay_capsule.environment.count(
                  "build_configuration") == 1 &&
              !loaded->replay_capsule.raw_dataset_values_included &&
              !loaded->replay_capsule.exact_replay_claimed,
          "replay capsule should preserve bounded environment and limitations");
    Check(loaded->issues.size() == 1 &&
              loaded->issues[0].message == "Contract warning" &&
              loaded->issues[0].error_code == "CW-C-0103",
          "issue payload and error code should round-trip");
    Check(loaded->traces.size() == 3 &&
              loaded->traces[0].role == cyxwiz::DebugTraceRole::Activation &&
              loaded->traces[0].input_shape == std::vector<size_t>{2, 3} &&
              loaded->traces[0].output_shape == std::vector<size_t>{2, 4} &&
              loaded->traces[0].payload["backend"].get<std::string>() == "CPU" &&
              loaded->traces[0].issues.size() == 1 &&
              loaded->traces[0].issues[0].error_code == "CW-C-0103",
          "trace payload and issue error code should round-trip");
    Check(loaded->traces[1].phase == "BackendPlacement" &&
              loaded->traces[1].payload["backend_actual"].get<std::string>() ==
                  "unobserved" &&
              loaded->traces[1].payload["graph_hash"].get<uint64_t>() == 0xCAFE,
          "canonical backend placement audit should persist without claiming runtime fact");
    Check(loaded->traces[2].phase == "ErrorCodeTimeline" &&
              loaded->traces[2].payload["entry_count"].get<size_t>() == 1 &&
              loaded->traces[2].payload["entries"][0]["code"] ==
                  "CW-C-0103",
          "canonical error-code timeline should persist through the generic trace store");
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
    Check(recent[0].execution.available &&
              recent[0].execution.execution_context_id ==
                  execution_trace.execution_context_id,
          "ListRecent should expose the linked execution record");

    cyxwiz::DebugRunStoreRecord invalid;
    Check(!cyxwiz::DebugRunStore::Save(invalid),
          "debug run store should reject empty run ids");
    Check(std::filesystem::exists(
              cyxwiz::GetDebugRunRoot() / "studio" /
              record.summary.run_id / "session.json"),
          "debug run store should use the injected root without changing CWD");

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
    Check(tokenizer.payload["success"].get<bool>(),
          "tokenizer trace should mark success");
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(tokenizer),
          "tokenizer trace should use canonical node trace schema");
    Check(tokenizer.payload["output_rank"].get<size_t>() == 1,
          "tokenizer trace should expose output rank");
    Check(tokenizer.payload["output_numel"].get<size_t>() ==
              tokenizer.output_shape[0],
          "tokenizer trace should expose output element count");
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
    Check(tokenizer.payload["diagnostic_phase"].get<std::string>() == "TextTokenizer",
          "tokenizer trace should expose text diagnostic phase");
    Check(tokenizer.payload["component"].get<std::string>() == "TextTokenizer",
          "tokenizer trace should expose diagnostic component");
    Check(tokenizer.payload["source_file"].get<std::string>().find(
              "text_preprocessing_tracer.cpp") != std::string::npos,
          "tokenizer trace should expose source file");
    Check(tokenizer.payload["source_symbol"].get<std::string>() ==
              "cyxwiz::TextPreprocessingTracer::TraceSample",
          "tokenizer trace should expose source symbol");

    const cyxwiz::DebugTraceRecord& vocab = traces[1];
    Check(vocab.node_id == 12, "vocab trace should bind to vocab node");
    Check(vocab.phase == "TextVocabulary", "vocab trace phase should be stable");
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(vocab),
          "vocab trace should use canonical node trace schema");
    Check(vocab.payload["success"].get<bool>(),
          "vocab trace should mark success");
    Check(vocab.payload["output_rank"].get<size_t>() == 1,
          "vocab trace should expose output rank");
    Check(vocab.payload["output_numel"].get<size_t>() == vocab.output_shape[0],
          "vocab trace should expose output element count");
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
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(padding),
          "padding trace should use canonical node trace schema");
    Check(padding.payload["success"].get<bool>(),
          "padding trace should mark success");
    Check(padding.output_shape == std::vector<size_t>{6},
          "padding output shape should reflect configured max length");
    Check(padding.payload["output_rank"].get<size_t>() == 1,
          "padding trace should expose output rank");
    Check(padding.payload["output_numel"].get<size_t>() == 6,
          "padding trace should expose output element count");
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

    auto truncated_config = config;
    truncated_config.text_preprocessing.max_length = 2;
    const auto truncated_traces =
        tracer.TraceSample(truncated_config, nodes, "text-trace-truncated-run", 0);
    Check(truncated_traces.size() == 3,
          "truncated text preprocessing trace should preserve trace count");
    const cyxwiz::DebugTraceRecord& truncated_padding = truncated_traces[2];
    Check(truncated_padding.status == "warning",
          "truncated padding trace should warn");
    Check(!truncated_padding.payload["success"].get<bool>(),
          "truncated padding trace should mark unsuccessful");
    Check(truncated_padding.payload["issue_count"].get<size_t>() == 1,
          "truncated padding trace should expose issue count");
    Check(truncated_padding.payload["warning_count"].get<size_t>() == 1,
          "truncated padding trace should expose warning count");
    Check(truncated_padding.payload["error_count"].get<size_t>() == 0,
          "truncated padding trace should expose zero error count");
    Check(truncated_padding.payload["primary_warning_code"].get<std::string>() ==
              cyxwiz::errors::Compiler::InvalidParameter,
          "truncated padding trace should expose primary warning code");
    Check(truncated_padding.payload["issue_codes"][0].get<std::string>() ==
              cyxwiz::errors::Compiler::InvalidParameter,
          "truncated padding trace should expose warning issue code");

    cyxwiz::DataRegistry::Instance().UnregisterTextDataset(dataset_name);
    const auto missing_traces =
        tracer.TraceSample(config, nodes, "missing-text-trace-run", 0);
    Check(missing_traces.size() == 1,
          "text preprocessing tracer should report missing dataset trace");
    Check(cyxwiz::DebugNodeTraceContract::IsNodeTrace(missing_traces[0]),
          "missing text dataset trace should use canonical node trace schema");
    Check(!missing_traces[0].payload["success"].get<bool>(),
          "missing text dataset trace should mark unsuccessful");
    Check(missing_traces[0].payload["error_code"].get<std::string>() ==
              cyxwiz::errors::Runtime::InputDatasetMissing,
          "missing text dataset trace should expose runtime input code");
    Check(!missing_traces[0].issues.empty() &&
              missing_traces[0].issues[0].error_code ==
                  cyxwiz::errors::Runtime::InputDatasetMissing,
          "missing text dataset issue should expose runtime input code");
    Check(missing_traces[0].payload["issue_count"].get<size_t>() == 1,
          "missing text dataset trace should expose issue count");
    Check(missing_traces[0].payload["error_count"].get<size_t>() == 1,
          "missing text dataset trace should expose error count");
    Check(missing_traces[0].payload["warning_count"].get<size_t>() == 0,
          "missing text dataset trace should expose zero warning count");
    Check(missing_traces[0].payload["primary_error_code"].get<std::string>() ==
              cyxwiz::errors::Runtime::InputDatasetMissing,
          "missing text dataset trace should expose primary error code");
    Check(missing_traces[0].payload["issue_codes"][0].get<std::string>() ==
              cyxwiz::errors::Runtime::InputDatasetMissing,
          "missing text dataset trace should expose issue code summary");
    Check(missing_traces[0].payload["diagnostic_phase"].get<std::string>() ==
              "data_source",
          "missing text dataset trace should expose data-source phase");
    Check(missing_traces[0].payload["component"].get<std::string>() ==
              "TextDataset",
          "missing text dataset trace should expose component");
    Check(missing_traces[0].payload["source_file"].get<std::string>().find(
              "text_preprocessing_tracer.cpp") != std::string::npos,
          "missing text dataset trace should expose source file");
    Check(missing_traces[0].payload["source_symbol"].get<std::string>() ==
              "cyxwiz::TextPreprocessingTracer::TraceSample",
          "missing text dataset trace should expose source symbol");
    std::filesystem::remove_all(test_root);
}

} // namespace

int main() {
    TestDebugSessionSnapshotContract();
    TestTrainingGraphDiffContract();
    TestNodeTraceContract();
    TestGraphTraceExecutionSlice();
    TestRuntimeBackendClassificationContract();
    TestMemoryOwnershipTraceContract();
    TestGradientHealthAnalyzerContract();
    TestBatchInspectionContract();
    TestSlowPathDetectorContract();
    TestTrainingStallDetectorContract();
    TestErrorCodeTimelineContract();
    TestExportCorrelationTraceContract();
    TestArtifactConsistencyTraceContract();
    TestWindowsCrashImportContract();
    TestSupportBundleContract();
    TestNodeInspectorSummaryContract();
    TestNodeExplanationTraceContract();
    TestOperatorBackedPreprocessingTraceContract();
    TestOperatorTraceProducerContract();
    TestSmokeSampleSelectionContract();
    TestRecommendationContract();
    TestSmokeRunResultValueContract();
    TestPreflightIssueCodeContract();
    TestDebugRunStoreContract();
    TestTextPreprocessingTraceContract();
    std::cout << "Debugger contract tests passed\n";
    return 0;
}
