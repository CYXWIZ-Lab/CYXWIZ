#include "debug_memory_ownership_tracer.h"
#include "debug_run_store.h"

#include <algorithm>
#include <limits>
#include <map>
#include <string>

namespace cyxwiz {

namespace {

int64_t Delta(uint64_t after, uint64_t before) {
    if (after >= before) {
        const uint64_t diff = after - before;
        if (diff > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            return std::numeric_limits<int64_t>::max();
        }
        return static_cast<int64_t>(diff);
    }

    const uint64_t diff = before - after;
    if (diff > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return std::numeric_limits<int64_t>::min();
    }
    return -static_cast<int64_t>(diff);
}

bool NearBudget(uint64_t value, uint64_t budget) {
    return budget > 0 && value >= ((budget / 10) * 9);
}

constexpr size_t kMaxLifecycleEntries = 128;
constexpr size_t kMaxConsumersPerEntry = 16;

uint64_t BytesPerElement(const std::string& dtype) {
    if (dtype == "float64" || dtype == "double" || dtype == "int64" ||
        dtype == "uint64") {
        return 8;
    }
    if (dtype == "float16" || dtype == "half" || dtype == "int16" ||
        dtype == "uint16") {
        return 2;
    }
    if (dtype == "bool" || dtype == "int8" || dtype == "uint8") {
        return 1;
    }
    return 4;
}

uint64_t SaturatingMultiply(uint64_t lhs, uint64_t rhs, bool& overflowed) {
    if (lhs == 0 || rhs == 0) {
        return 0;
    }
    if (lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        overflowed = true;
        return std::numeric_limits<uint64_t>::max();
    }
    return lhs * rhs;
}

uint64_t SaturatingAdd(uint64_t lhs, uint64_t rhs, bool& overflowed) {
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        overflowed = true;
        return std::numeric_limits<uint64_t>::max();
    }
    return lhs + rhs;
}

std::vector<size_t> JsonShape(const nlohmann::json& payload,
                              const char* key) {
    std::vector<size_t> shape;
    const auto it = payload.find(key);
    if (it == payload.end() || !it->is_array()) {
        return shape;
    }
    shape.reserve(it->size());
    for (const auto& dim : *it) {
        if (!dim.is_number_unsigned() && !dim.is_number_integer()) {
            return {};
        }
        if (dim.is_number_unsigned()) {
            const uint64_t value = dim.get<uint64_t>();
            if (value > static_cast<uint64_t>(
                    std::numeric_limits<size_t>::max())) {
                return {};
            }
            shape.push_back(static_cast<size_t>(value));
        } else {
            const int64_t value = dim.get<int64_t>();
            if (value < 0) {
                return {};
            }
            shape.push_back(static_cast<size_t>(value));
        }
    }
    return shape;
}

bool IsLifecycleTensorRole(DebugTraceRole role) {
    switch (role) {
        case DebugTraceRole::RawInput:
        case DebugTraceRole::PreprocessingOutput:
        case DebugTraceRole::FeatureTensor:
        case DebugTraceRole::ModelInput:
        case DebugTraceRole::Activation:
        case DebugTraceRole::Parameter:
        case DebugTraceRole::Gradient:
        case DebugTraceRole::Prediction:
        case DebugTraceRole::Target:
        case DebugTraceRole::Loss:
            return true;
        default:
            return false;
    }
}

const char* LifecycleRelation(DebugTraceRole role) {
    switch (role) {
        case DebugTraceRole::RawInput: return "batch_input";
        case DebugTraceRole::PreprocessingOutput: return "preprocessing_output";
        case DebugTraceRole::FeatureTensor: return "feature_tensor";
        case DebugTraceRole::ModelInput: return "model_input";
        case DebugTraceRole::Activation: return "model_activation";
        case DebugTraceRole::Parameter: return "parameter";
        case DebugTraceRole::Gradient: return "gradient";
        case DebugTraceRole::Prediction: return "prediction";
        case DebugTraceRole::Target: return "target";
        case DebugTraceRole::Loss: return "loss";
        default: return "tensor";
    }
}

struct LifecycleTopology {
    std::map<int, std::string> node_names;
    std::map<int, std::vector<int>> consumers;
};

LifecycleTopology ReadLifecycleTopology(
    const std::vector<DebugTraceRecord>& traces) {
    LifecycleTopology topology;
    const auto snapshot = std::find_if(
        traces.begin(), traces.end(), [](const DebugTraceRecord& trace) {
            return trace.phase == "GraphSnapshot";
        });
    if (snapshot == traces.end()) {
        return topology;
    }

    const auto nodes = snapshot->payload.find("nodes");
    if (nodes != snapshot->payload.end() && nodes->is_array()) {
        for (const auto& node : *nodes) {
            if (!node.is_object() || !node.contains("id") ||
                !node.at("id").is_number_integer()) {
                continue;
            }
            topology.node_names[node.at("id").get<int>()] =
                node.value("name", std::string{});
        }
    }

    const auto links = snapshot->payload.find("links");
    if (links != snapshot->payload.end() && links->is_array()) {
        for (const auto& link : *links) {
            if (!link.is_object()) {
                continue;
            }
            const int from_node = link.value("from_node", -1);
            const int to_node = link.value("to_node", -1);
            if (from_node < 0 || to_node < 0) {
                continue;
            }
            auto& consumers = topology.consumers[from_node];
            if (std::find(consumers.begin(), consumers.end(), to_node) ==
                consumers.end()) {
                consumers.push_back(to_node);
            }
        }
    }
    return topology;
}

} // namespace

DebugTraceRecord DebugMemoryOwnershipTracer::BuildTrace(
    const std::string& run_id,
    const DebugMemoryOwnershipInput& input) const {
    const uint64_t estimated_tensor_bytes =
        EstimateTensorBytes(input.output_shape, input.bytes_per_element);
    const bool cpu_peak_increased =
        input.after.cpu_peak_bytes > input.before.cpu_peak_bytes;
    const bool device_locked_increased =
        input.after.af_locked_bytes > input.before.af_locked_bytes;
    const bool host_oom_risk =
        NearBudget(input.after.cpu_peak_bytes, input.host_budget_bytes);
    const bool device_oom_risk =
        NearBudget(input.after.af_locked_bytes, input.device_budget_bytes);

    DebugTraceRecord trace = DebugNodeTraceContract::Make(
        run_id,
        input.node_id,
        input.node_name,
        input.node_type,
        input.phase,
        input.role,
        {},
        input.output_shape,
        input.dtype,
        input.backend,
        "ok");
    DebugNodeTraceContract::AttachDiagnosticContext(
        trace,
        "memory_ownership",
        "DebugMemoryOwnershipTracer",
        "cyxwiz-engine/src/core/debug_memory_ownership_tracer.cpp",
        "cyxwiz::DebugMemoryOwnershipTracer::BuildTrace");

    trace.payload["memory_schema"] = kSchema;
    trace.payload["memory_observation"] = "training_trace_delta";
    trace.payload["ownership_proven"] = false;
    trace.payload["ownership_note"] =
        "Per-node memory is inferred from before/after training snapshots; "
        "allocator-level tensor ownership is not proven in this slice.";
    trace.payload["estimated_tensor_bytes"] = estimated_tensor_bytes;
    trace.payload["bytes_per_element"] = input.bytes_per_element;
    trace.payload["cpu_allocated_before_bytes"] =
        input.before.cpu_allocated_bytes;
    trace.payload["cpu_allocated_after_bytes"] =
        input.after.cpu_allocated_bytes;
    trace.payload["cpu_allocated_delta_bytes"] = Delta(
        input.after.cpu_allocated_bytes,
        input.before.cpu_allocated_bytes);
    trace.payload["cpu_peak_before_bytes"] = input.before.cpu_peak_bytes;
    trace.payload["cpu_peak_after_bytes"] = input.after.cpu_peak_bytes;
    trace.payload["cpu_peak_increased"] = cpu_peak_increased;
    trace.payload["af_allocated_before_bytes"] =
        input.before.af_allocated_bytes;
    trace.payload["af_allocated_after_bytes"] =
        input.after.af_allocated_bytes;
    trace.payload["af_allocated_delta_bytes"] = Delta(
        input.after.af_allocated_bytes,
        input.before.af_allocated_bytes);
    trace.payload["af_locked_before_bytes"] = input.before.af_locked_bytes;
    trace.payload["af_locked_after_bytes"] = input.after.af_locked_bytes;
    trace.payload["af_locked_delta_bytes"] = Delta(
        input.after.af_locked_bytes,
        input.before.af_locked_bytes);
    trace.payload["af_alloc_buffers_before"] = input.before.af_alloc_buffers;
    trace.payload["af_alloc_buffers_after"] = input.after.af_alloc_buffers;
    trace.payload["af_lock_buffers_before"] = input.before.af_lock_buffers;
    trace.payload["af_lock_buffers_after"] = input.after.af_lock_buffers;
    trace.payload["device_locked_increased"] = device_locked_increased;
    trace.payload["host_budget_bytes"] = input.host_budget_bytes;
    trace.payload["device_budget_bytes"] = input.device_budget_bytes;
    trace.payload["host_oom_risk"] = host_oom_risk;
    trace.payload["device_oom_risk"] = device_oom_risk;

    if (host_oom_risk) {
        DebugNodeTraceContract::AddWarning(
            trace,
            "Host memory peak is close to the configured debug budget.");
    }
    if (device_oom_risk) {
        DebugNodeTraceContract::AddWarning(
            trace,
            "Device locked memory is close to the configured debug budget.");
    }
    trace.payload["success"] = trace.status == "ok";

    return trace;
}

DebugTraceRecord DebugMemoryOwnershipTracer::BuildTensorLifecycleTrace(
    const std::string& run_id,
    const std::vector<DebugTraceRecord>& traces,
    const DebugRunExecutionSummary& execution) const {
    const bool backend_observed = execution.available &&
        execution.training_run_id == run_id &&
        !execution.effective_backend.empty();
    const std::string backend = backend_observed
        ? execution.effective_backend
        : "unobserved";
    const std::string device = backend_observed
        ? (execution.effective_device_name.empty()
            ? std::string("device ") +
                std::to_string(execution.effective_device_id)
            : execution.effective_device_name)
        : "unobserved";

    DebugTraceRecord lifecycle = DebugNodeTraceContract::Make(
        run_id,
        -1,
        "Tensor Lifecycle",
        "TrainingDiagnostics",
        "TensorLifecycle",
        DebugTraceRole::CompileArtifact,
        {}, {}, "tensor_metadata", backend, "unobserved");
    DebugNodeTraceContract::AttachDiagnosticContext(
        lifecycle,
        "tensor_lifecycle",
        "DebugMemoryOwnershipTracer",
        "cyxwiz-engine/src/core/debug_memory_ownership_tracer.cpp",
        "cyxwiz::DebugMemoryOwnershipTracer::BuildTensorLifecycleTrace");

    auto& payload = lifecycle.payload;
    payload["tensor_lifecycle_schema"] = kLifecycleSchema;
    payload["observation_scope"] =
        "derived_from_canonical_debug_traces_and_graph_snapshot";
    payload["allocator_lifecycle_observed"] = false;
    payload["retained_freed_state_observed"] = false;
    payload["lifecycle_note"] =
        "Tensor origin, shape, relation, and consumers are derived from "
        "canonical traces and graph topology. Retained/freed state remains "
        "unobserved until allocator hooks provide direct evidence.";
    payload["runtime_backend_observed"] = backend_observed;
    payload["runtime_backend_evidence_scope"] = backend_observed
        ? "same_run_execution_summary"
        : "unobserved";
    payload["effective_backend"] = backend;
    payload["effective_device"] = device;
    payload["execution_context_id"] = backend_observed
        ? execution.execution_context_id
        : std::string{};
    payload["native_cpu_fallback_count"] = backend_observed
        ? execution.native_cpu_fallback_count
        : 0;
    payload["host_reads_added"] = false;
    payload["raw_tensor_values_included"] = false;
    payload["entry_limit"] = kMaxLifecycleEntries;

    const LifecycleTopology topology = ReadLifecycleTopology(traces);
    const bool optimizer_observed = std::any_of(
        traces.begin(), traces.end(), [](const DebugTraceRecord& trace) {
            return trace.role == DebugTraceRole::OptimizerStep &&
                trace.status != "failed";
        });
    const bool backward_observed = std::any_of(
        traces.begin(), traces.end(), [](const DebugTraceRecord& trace) {
            return trace.role == DebugTraceRole::Gradient;
        });
    const bool explicit_model_input = std::any_of(
        traces.begin(), traces.end(), [](const DebugTraceRecord& trace) {
            return trace.role == DebugTraceRole::ModelInput;
        });
    const bool explicit_prediction = std::any_of(
        traces.begin(), traces.end(), [](const DebugTraceRecord& trace) {
            return trace.role == DebugTraceRole::Prediction;
        });
    const bool explicit_target = std::any_of(
        traces.begin(), traces.end(), [](const DebugTraceRecord& trace) {
            return trace.role == DebugTraceRole::Target;
        });

    size_t last_forward_index = traces.size();
    for (size_t i = 0; i < traces.size(); ++i) {
        if (traces[i].phase == "Forward" &&
            traces[i].role == DebugTraceRole::Activation) {
            last_forward_index = i;
        }
    }

    nlohmann::json entries = nlohmann::json::array();
    nlohmann::json relation_counts = nlohmann::json::object();
    size_t entry_count = 0;
    uint64_t estimated_total_bytes = 0;
    bool estimated_total_overflow = false;

    auto append_entry = [&](nlohmann::json entry) {
        const std::string relation = entry.value(
            "relation", std::string{"tensor"});
        relation_counts[relation] =
            relation_counts.value(relation, static_cast<size_t>(0)) + 1;
        const uint64_t bytes = entry.value(
            "estimated_bytes", static_cast<uint64_t>(0));
        estimated_total_bytes = SaturatingAdd(
            estimated_total_bytes, bytes, estimated_total_overflow);
        ++entry_count;
        if (entries.size() < kMaxLifecycleEntries) {
            entries.push_back(std::move(entry));
        }
    };

    auto consumer_payload = [&](int node_id,
                                const char* fallback_name,
                                bool fallback_observed,
                                int fallback_node_id = -1) {
        nlohmann::json consumers = nlohmann::json::array();
        size_t total = 0;
        std::string source = "unobserved";
        const auto consumer_it = topology.consumers.find(node_id);
        if (node_id >= 0 && consumer_it != topology.consumers.end()) {
            total = consumer_it->second.size();
            source = "graph_snapshot_topology";
            for (const int consumer_id : consumer_it->second) {
                if (consumers.size() >= kMaxConsumersPerEntry) {
                    break;
                }
                const auto name_it = topology.node_names.find(consumer_id);
                consumers.push_back({
                    {"node_id", consumer_id},
                    {"node_name", name_it == topology.node_names.end()
                        ? std::string{} : name_it->second},
                });
            }
        } else if (fallback_observed) {
            total = 1;
            source = "canonical_trace_sequence";
            consumers.push_back({
                {"node_id", fallback_node_id},
                {"node_name", fallback_name},
            });
        }
        return nlohmann::json{
            {"consumers", std::move(consumers)},
            {"consumer_count", total},
            {"consumers_truncated", total > kMaxConsumersPerEntry},
            {"consumer_evidence", source},
        };
    };

    auto make_entry = [&](const std::string& tensor_id,
                          const std::string& relation,
                          int node_id,
                          const std::string& node_name,
                          const std::string& node_type,
                          const std::string& phase,
                          const std::vector<size_t>& shape,
                          uint64_t observed_elements,
                          const std::string& dtype,
                          size_t source_trace_index,
                          nlohmann::json consumer_info,
                          bool host_read_observed) {
        const uint64_t bytes_per_element = BytesPerElement(dtype);
        uint64_t element_count = observed_elements;
        if (!shape.empty()) {
            const uint64_t estimated = EstimateTensorBytes(shape, 1);
            element_count = estimated;
            if (estimated == std::numeric_limits<uint64_t>::max()) {
                estimated_total_overflow = true;
            }
        }
        nlohmann::json entry = {
            {"tensor_id", tensor_id},
            {"relation", relation},
            {"origin_node_id", node_id},
            {"origin_node_name", node_name},
            {"origin_node_type", node_type},
            {"origin_phase", phase},
            {"shape", shape},
            {"shape_observed", !shape.empty()},
            {"element_count", element_count},
            {"element_count_observed", element_count > 0},
            {"dtype", dtype.empty() ? "unobserved" : dtype},
            {"bytes_per_element", bytes_per_element},
            {"estimated_bytes", SaturatingMultiply(
                element_count, bytes_per_element,
                estimated_total_overflow)},
            {"backend", backend},
            {"device", device},
            {"runtime_backend_observed", backend_observed},
            {"retention_state", "unobserved"},
            {"allocator_lifecycle_observed", false},
            {"host_read_observed", host_read_observed},
            {"host_read_scope", host_read_observed
                ? "bounded_debug_numeric_summary" : "unobserved"},
            {"source_trace_index", source_trace_index},
            {"raw_values_included", false},
        };
        for (auto it = consumer_info.begin(); it != consumer_info.end(); ++it) {
            entry[it.key()] = std::move(it.value());
        }
        return entry;
    };

    if (!explicit_model_input) {
        for (size_t i = 0; i < traces.size(); ++i) {
            const auto& trace = traces[i];
            if (trace.phase != "Forward" || trace.input_shape.empty()) {
                continue;
            }
            const int upstream_node_id = trace.payload.value(
                "upstream_node_id", -1);
            const std::string upstream_node_name = trace.payload.value(
                "upstream_node_name", std::string{});
            append_entry(make_entry(
                "model_input",
                "model_input",
                upstream_node_id,
                upstream_node_name.empty()
                    ? "Local Debug batch" : upstream_node_name,
                "ModelInput",
                "Forward.Input",
                trace.input_shape,
                0,
                trace.dtype,
                i,
                consumer_payload(
                    -1, trace.node_name.c_str(), true, trace.node_id),
                false));
            break;
        }
    }

    bool synthesized_prediction = false;
    bool synthesized_target = false;
    for (size_t i = 0; i < traces.size(); ++i) {
        const auto& trace = traces[i];
        if (!IsLifecycleTensorRole(trace.role) ||
            trace.phase == "TensorLifecycle") {
            continue;
        }

        if (trace.role == DebugTraceRole::Loss) {
            if (!explicit_prediction && last_forward_index == traces.size() &&
                !synthesized_prediction) {
                const auto prediction_shape = JsonShape(
                    trace.payload, "prediction_shape");
                if (!prediction_shape.empty()) {
                    append_entry(make_entry(
                        "prediction",
                        "prediction",
                        trace.node_id,
                        trace.node_name,
                        trace.node_type,
                        "Loss.Prediction",
                        prediction_shape,
                        0,
                        trace.dtype,
                        i,
                        consumer_payload(
                            -1, "Loss", true, trace.node_id),
                        false));
                    synthesized_prediction = true;
                }
            }
            if (!explicit_target && !synthesized_target) {
                const auto target_shape = JsonShape(trace.payload, "target_shape");
                if (!target_shape.empty()) {
                    append_entry(make_entry(
                        "target",
                        "target",
                        -1,
                        "Local Debug target",
                        "Target",
                        "Loss.Target",
                        target_shape,
                        0,
                        trace.dtype,
                        i,
                        consumer_payload(
                            -1, "Loss", true, trace.node_id),
                        false));
                    synthesized_target = true;
                }
            }
        }

        std::vector<size_t> shape = trace.output_shape;
        if (shape.empty()) {
            shape = JsonShape(trace.payload, "actual_shape");
        }
        uint64_t observed_elements = 0;
        const auto numeric_count = trace.payload.find("numeric_element_count");
        if (numeric_count != trace.payload.end() &&
            numeric_count->is_number_unsigned()) {
            observed_elements = numeric_count->get<uint64_t>();
        } else if (numeric_count != trace.payload.end() &&
                   numeric_count->is_number_integer()) {
            const int64_t value = numeric_count->get<int64_t>();
            if (value > 0) {
                observed_elements = static_cast<uint64_t>(value);
            }
        } else if (trace.role == DebugTraceRole::Loss) {
            shape = {1};
        }

        std::string relation = LifecycleRelation(trace.role);
        if (trace.role == DebugTraceRole::Activation &&
            i == last_forward_index) {
            relation = "prediction";
        }
        const std::string tensor_id = relation + ":" +
            std::to_string(trace.node_id) + ":" + std::to_string(i);

        nlohmann::json consumer_info;
        if (trace.role == DebugTraceRole::Gradient) {
            consumer_info = consumer_payload(
                -1, "Optimizer step", optimizer_observed);
        } else if (trace.role == DebugTraceRole::Loss) {
            consumer_info = consumer_payload(
                -1, "Backward pass", backward_observed);
        } else {
            consumer_info = consumer_payload(trace.node_id, "", false);
        }
        nlohmann::json entry = make_entry(
            tensor_id,
            relation,
            trace.node_id,
            trace.node_name,
            trace.node_type,
            trace.phase,
            shape,
            observed_elements,
            trace.dtype,
            i,
            std::move(consumer_info),
            trace.payload.value("numeric_host_read_performed", false));
        const auto parameter_name = trace.payload.find("parameter_name");
        if (parameter_name != trace.payload.end() &&
            parameter_name->is_string()) {
            entry["parameter_name"] = *parameter_name;
        }
        append_entry(std::move(entry));
    }

    payload["entries"] = std::move(entries);
    payload["entry_count"] = entry_count;
    payload["retained_entry_count"] = payload["entries"].size();
    payload["entries_truncated"] =
        entry_count > payload["entries"].size();
    payload["relation_counts"] = std::move(relation_counts);
    payload["estimated_total_bytes"] = estimated_total_bytes;
    payload["estimated_total_bytes_overflowed"] = estimated_total_overflow;
    payload["success"] = true;
    lifecycle.status = entry_count > 0 ? "captured" : "unobserved";
    return lifecycle;
}

uint64_t DebugMemoryOwnershipTracer::EstimateTensorBytes(
    const std::vector<size_t>& shape,
    uint64_t bytes_per_element) {
    if (shape.empty() || bytes_per_element == 0) {
        return 0;
    }

    uint64_t elements = 1;
    for (size_t dim : shape) {
        if (dim == 0) {
            return 0;
        }
        if (elements > std::numeric_limits<uint64_t>::max() /
                           static_cast<uint64_t>(dim)) {
            return std::numeric_limits<uint64_t>::max();
        }
        elements *= static_cast<uint64_t>(dim);
    }

    if (elements > std::numeric_limits<uint64_t>::max() /
                       bytes_per_element) {
        return std::numeric_limits<uint64_t>::max();
    }
    return elements * bytes_per_element;
}

} // namespace cyxwiz
