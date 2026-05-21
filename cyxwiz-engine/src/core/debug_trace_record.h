#pragma once

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
