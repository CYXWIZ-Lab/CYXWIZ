#pragma once

#include "python_engine.h"
#include <string>
#include <memory>
#include <functional>

namespace scripting {

/**
 * Execution result from ScriptingEngine
 */
struct ExecutionResult {
    bool success;
    std::string output;        // stdout/return value
    std::string error_message; // stderr/exception message
};

/**
 * ScriptingEngine - High-level wrapper around PythonEngine
 * Adds output capture, error handling, and additional features
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

    // Check if engine is initialized
    bool IsInitialized() const;

private:
    std::unique_ptr<PythonEngine> python_engine_;
    OutputCallback output_callback_;

    // Helper to capture stdout/stderr
    std::string CaptureOutput(const std::function<bool()>& execution_func);
};

} // namespace scripting
