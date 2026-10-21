#pragma once

#include "error_codes.h"
#include "graph_compiler.h"
#include <nlohmann/json.hpp>
#include <algorithm>
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
        trace.payload["node_trace_schema"] = kSchema;
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
        if (it != trace.payload.end() &&
            it->is_string() &&
            it->get<std::string>() == kSchema) {
            return true;
        }
        auto node_trace_it = trace.payload.find("node_trace_schema");
        return node_trace_it != trace.payload.end() &&
               node_trace_it->is_string() &&
               node_trace_it->get<std::string>() == kSchema;
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
        AttachIssueSummary(trace, trace.issues);
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
        AttachIssueSummary(trace, trace.issues);
    }

    static void AttachIssueSummary(DebugTraceRecord& trace,
                                   const std::vector<ValidationIssue>& issues) {
        trace.payload["issue_count"] = issues.size();
        trace.payload["error_count"] = CountIssues(issues, IssueLevel::Error);
        trace.payload["warning_count"] = CountIssues(issues, IssueLevel::Warning);
        trace.payload["info_count"] = CountIssues(issues, IssueLevel::Info);

        nlohmann::json issue_codes = nlohmann::json::array();
        std::vector<std::string> seen_codes;
        for (const auto& issue : issues) {
            if (issue.error_code.empty()) {
                continue;
            }
            const bool already_seen = std::find(
                seen_codes.begin(), seen_codes.end(), issue.error_code) !=
                seen_codes.end();
            if (!already_seen) {
                seen_codes.push_back(issue.error_code);
                issue_codes.push_back(issue.error_code);
            }
            if (issue.level == IssueLevel::Error &&
                !trace.payload.contains("primary_error_code")) {
                trace.payload["primary_error_code"] = issue.error_code;
            } else if (issue.level == IssueLevel::Warning &&
                       !trace.payload.contains("primary_warning_code")) {
                trace.payload["primary_warning_code"] = issue.error_code;
            }
        }
        trace.payload["issue_codes"] = std::move(issue_codes);
    }

    static void AttachDiagnosticContext(DebugTraceRecord& trace,
                                        const std::string& diagnostic_phase,
                                        const std::string& component,
                                        const std::string& source_file = "",
                                        const std::string& source_symbol = "") {
        if (!diagnostic_phase.empty()) {
            trace.payload["diagnostic_phase"] = diagnostic_phase;
        }
        if (!component.empty()) {
            trace.payload["component"] = component;
        }
        if (!source_file.empty()) {
            trace.payload["source_file"] = source_file;
        }
        if (!source_symbol.empty()) {
            trace.payload["source_symbol"] = source_symbol;
        }
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
        return CountIssues(trace.issues, level);
    }

    static size_t CountIssues(const std::vector<ValidationIssue>& issues,
                              IssueLevel level) {
        size_t count = 0;
        for (const auto& issue : issues) {
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
