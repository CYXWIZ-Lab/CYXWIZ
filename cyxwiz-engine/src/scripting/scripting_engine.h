#pragma once

#include "python_engine.h"
#include "python_sandbox.h"
#include <string>
#include <memory>
#include <functional>

// Forward declarations
namespace cyxwiz {
    class TrainingPlotPanel;
}

namespace scripting {

/**
 * Execution result from ScriptingEngine
 */
struct ExecutionResult {
    bool success;
    std::string output;        // stdout/return value
    std::string error_message; // stderr/exception message

    // Security/resource info (from sandbox)
    bool timeout_exceeded{false};
    bool memory_exceeded{false};
    bool security_violation{false};
    std::string violation_reason;
};

/**
 * ScriptingEngine - High-level wrapper around PythonEngine
 * Adds output capture, error handling, sandbox security, and additional features
 */
class ScriptingEngine {
public:
    ScriptingEngine();
    ~ScriptingEngine();

    // Execute single command (REPL-style)
    ExecutionResult ExecuteCommand(const std::string& command);

    // Execute multi-line script
    ExecutionResult ExecuteScript(const std::string& script);

    // Execute script file
    ExecutionResult ExecuteFile(const std::string& filepath);

    // Output callback (for real-time output streaming)
    using OutputCallback = std::function<void(const std::string&)>;
    void SetOutputCallback(OutputCallback callback);

    // Sandbox configuration
    void EnableSandbox(bool enable);
    bool IsSandboxEnabled() const { return sandbox_enabled_; }
    void SetSandboxConfig(const PythonSandbox::Config& config);
    PythonSandbox::Config GetSandboxConfig() const;

    // Check if engine is initialized
    bool IsInitialized() const;

    // Register Training Dashboard with Python module
    void RegisterTrainingDashboard(cyxwiz::TrainingPlotPanel* panel);

private:
    std::unique_ptr<PythonEngine> python_engine_;
    std::unique_ptr<PythonSandbox> sandbox_;
    OutputCallback output_callback_;
    bool sandbox_enabled_;

    // Helper to capture stdout/stderr
    std::string CaptureOutput(const std::function<bool()>& execution_func);

    // Convert sandbox result to engine result
    ExecutionResult ConvertSandboxResult(const PythonSandbox::ExecutionResult& sandbox_result);
};

} // namespace scripting
