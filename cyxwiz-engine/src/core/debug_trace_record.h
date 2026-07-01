#pragma once

#include "error_codes.h"
#include "graph_compiler.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <string>
#include <vector>

namespace cyxwiz {

enum class DebugTraceRole {
    RawInput,
    PreprocessingOutput,
    FeatureTensor,
    ModelInput,
    Activation,
    Parameter,
    Gradient,
    Prediction,
    Target,
    Loss,
    OptimizerStep,
    CompileArtifact,
    GeneratedCode,
    StudioEvent,
    Warning,
    Error
};

struct DebugTraceRecord {
    std::string run_id;
    int node_id = -1;
    std::string node_name;
    std::string node_type;
    std::string phase;
    DebugTraceRole role = DebugTraceRole::Activation;
    std::vector<size_t> input_shape;
    std::vector<size_t> output_shape;
    std::string dtype;
    float duration_ms = 0.0f;
    std::string status;
    std::vector<ValidationIssue> issues;
    nlohmann::json payload = nlohmann::json::object();
};

struct StudioEventRecord {
    std::string run_id;
    std::string timestamp;
    uint64_t graph_hash = 0;
    int selected_node_id = -1;
    std::string action;
    std::string status;
    std::string message;
};

struct DebugNodeTraceContract {
    static constexpr const char* kSchema = "cyxwiz.debug.node_trace.v1";

    static DebugTraceRecord Make(const std::string& run_id,
                                 int node_id,
                                 const std::string& node_name,
                                 const std::string& node_type,
                                 const std::string& phase,
                                 DebugTraceRole role,
                                 const std::vector<size_t>& input_shape,
                                 const std::vector<size_t>& output_shape,
                                 const std::string& dtype,
                                 const std::string& backend,
                                 const std::string& status) {
        DebugTraceRecord trace;
        trace.run_id = run_id;
        trace.node_id = node_id;
        trace.node_name = node_name;
        trace.node_type = node_type;
        trace.phase = phase;
        trace.role = role;
        trace.input_shape = input_shape;
        trace.output_shape = output_shape;
        trace.dtype = dtype;
        trace.status = status;
        trace.payload["schema"] = kSchema;
        trace.payload["backend"] = backend;
        trace.payload["input_rank"] = input_shape.size();
        trace.payload["output_rank"] = output_shape.size();
        trace.payload["input_numel"] = NumElements(input_shape);
        trace.payload["output_numel"] = NumElements(output_shape);
        trace.payload["warning_count"] = 0;
        trace.payload["error_count"] = 0;
        return trace;
    }

    static bool IsNodeTrace(const DebugTraceRecord& trace) {
        auto it = trace.payload.find("schema");
        return it != trace.payload.end() &&
               it->is_string() &&
               it->get<std::string>() == kSchema;
    }

    static void AddWarning(DebugTraceRecord& trace,
                           const std::string& message,
                           const std::string& error_code = "") {
        trace.issues.push_back({
            IssueLevel::Warning,
            trace.node_id,
            trace.node_name,
            message,
            error_code.empty()
                ? errors::Runtime::ExecutionFailed
                : error_code
        });
        trace.payload["warning_count"] = CountIssues(trace, IssueLevel::Warning);
    }

    static void AddError(DebugTraceRecord& trace,
                         const std::string& message,
                         const std::string& error_code = "") {
        trace.issues.push_back({
            IssueLevel::Error,
            trace.node_id,
            trace.node_name,
            message,
            error_code.empty()
                ? errors::Runtime::ExecutionFailed
                : error_code
        });
        trace.status = "failed";
        trace.payload["error_count"] = CountIssues(trace, IssueLevel::Error);
    }

private:
    static size_t NumElements(const std::vector<size_t>& shape) {
        if (shape.empty()) {
            return 0;
        }
        size_t n = 1;
        for (size_t dim : shape) {
            n *= dim;
        }
        return n;
    }

    static size_t CountIssues(const DebugTraceRecord& trace,
                              IssueLevel level) {
        size_t count = 0;
        for (const auto& issue : trace.issues) {
            if (issue.level == level) {
                ++count;
            }
        }
        return count;
    }
};

inline const char* DebugTraceRoleName(DebugTraceRole role) {
    switch (role) {
        case DebugTraceRole::RawInput: return "RawInput";
        case DebugTraceRole::PreprocessingOutput: return "PreprocessingOutput";
        case DebugTraceRole::FeatureTensor: return "FeatureTensor";
        case DebugTraceRole::ModelInput: return "ModelInput";
        case DebugTraceRole::Activation: return "Activation";
        case DebugTraceRole::Parameter: return "Parameter";
        case DebugTraceRole::Gradient: return "Gradient";
        case DebugTraceRole::Prediction: return "Prediction";
        case DebugTraceRole::Target: return "Target";
        case DebugTraceRole::Loss: return "Loss";
        case DebugTraceRole::OptimizerStep: return "OptimizerStep";
        case DebugTraceRole::CompileArtifact: return "CompileArtifact";
        case DebugTraceRole::GeneratedCode: return "GeneratedCode";
        case DebugTraceRole::StudioEvent: return "StudioEvent";
        case DebugTraceRole::Warning: return "Warning";
        case DebugTraceRole::Error: return "Error";
    }
    return "Unknown";
}

} // namespace cyxwiz
