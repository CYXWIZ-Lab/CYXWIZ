#include "scripting_engine.h"
#include <pybind11/embed.h>
#include <spdlog/spdlog.h>
#include <fstream>

namespace py = pybind11;

namespace scripting {

ScriptingEngine::ScriptingEngine()
    : sandbox_enabled_(false)
{
    python_engine_ = std::make_unique<PythonEngine>();
    sandbox_ = std::make_unique<PythonSandbox>();
    spdlog::info("ScriptingEngine initialized (sandbox disabled by default)");
}

ScriptingEngine::~ScriptingEngine() = default;

void ScriptingEngine::EnableSandbox(bool enable) {
    sandbox_enabled_ = enable;
    spdlog::info("Sandbox {}", enable ? "enabled" : "disabled");
}

void ScriptingEngine::SetSandboxConfig(const PythonSandbox::Config& config) {
    if (sandbox_) {
        sandbox_->SetConfig(config);
    }
}

PythonSandbox::Config ScriptingEngine::GetSandboxConfig() const {
    if (sandbox_) {
        return sandbox_->GetConfig();
    }
    return PythonSandbox::Config();
}

ExecutionResult ScriptingEngine::ConvertSandboxResult(const PythonSandbox::ExecutionResult& sandbox_result) {
    ExecutionResult result;
    result.success = sandbox_result.success;
    result.output = sandbox_result.output;
    result.error_message = sandbox_result.error_message;
    result.timeout_exceeded = sandbox_result.timeout_exceeded;
    result.memory_exceeded = sandbox_result.memory_exceeded;
    result.security_violation = sandbox_result.security_violation;
    result.violation_reason = sandbox_result.violation_reason;
    return result;
}

bool ScriptingEngine::IsInitialized() const {
    return python_engine_ != nullptr;
}

void ScriptingEngine::SetOutputCallback(OutputCallback callback) {
    output_callback_ = callback;
}

ExecutionResult ScriptingEngine::ExecuteCommand(const std::string& command) {
    ExecutionResult result;
    result.success = false;

    if (!IsInitialized()) {
        result.error_message = "Scripting engine not initialized";
        return result;
    }

    try {
        // Redirect stdout/stderr to capture output
        py::object sys = py::module_::import("sys");
        py::object io = py::module_::import("io");

        // Create StringIO objects for stdout and stderr
        py::object stdout_capture = io.attr("StringIO")();
        py::object stderr_capture = io.attr("StringIO")();

        // Save original stdout/stderr
        py::object original_stdout = sys.attr("stdout");
        py::object original_stderr = sys.attr("stderr");

        // Redirect to our captures
        sys.attr("stdout") = stdout_capture;
        sys.attr("stderr") = stderr_capture;

        // Execute the command
        py::object py_result;
        bool has_result = false;

        try {
            // Try exec first (for statements)
            py::exec(command);
        } catch (const py::error_already_set& e) {
            // If exec fails, try eval (for expressions)
            try {
                py_result = py::eval(command);
                has_result = true;
            } catch (const py::error_already_set& eval_error) {
                // Restore stdout/stderr before throwing
                sys.attr("stdout") = original_stdout;
                sys.attr("stderr") = original_stderr;
                throw;
            }
        }

        // Get captured output
        stdout_capture.attr("seek")(0);
        stderr_capture.attr("seek")(0);

        std::string stdout_str = py::str(stdout_capture.attr("read")());
        std::string stderr_str = py::str(stderr_capture.attr("read")());

        // Restore original stdout/stderr
        sys.attr("stdout") = original_stdout;
        sys.attr("stderr") = original_stderr;

        // Build result
        result.success = true;

        // Include eval result if present
        if (has_result && !py_result.is_none()) {
            result.output = py::str(py_result);
            result.output += "\n";
        }

        // Append stdout
        if (!stdout_str.empty()) {
            result.output += stdout_str;
        }

        // Include stderr as error if present
        if (!stderr_str.empty()) {
            result.error_message = stderr_str;
            result.success = false; // Mark as failure if there's stderr output
        }

        // Call output callback if set
        if (output_callback_ && !result.output.empty()) {
            output_callback_(result.output);
        }

    } catch (const py::error_already_set& e) {
        result.success = false;
        result.error_message = e.what();
        spdlog::error("Python execution error: {}", e.what());
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
        spdlog::error("Execution error: {}", e.what());
    }

    return result;
}

ExecutionResult ScriptingEngine::ExecuteScript(const std::string& script) {
    // If sandbox is enabled, use it
    if (sandbox_enabled_ && sandbox_) {
        auto sandbox_result = sandbox_->Execute(script);
        return ConvertSandboxResult(sandbox_result);
    }

    // Otherwise, use normal execution
    return ExecuteCommand(script);
}

ExecutionResult ScriptingEngine::ExecuteFile(const std::string& filepath) {
    ExecutionResult result;
    result.success = false;

    if (!IsInitialized()) {
        result.error_message = "Scripting engine not initialized";
        return result;
    }

    try {
        // Read file content
        std::ifstream file(filepath);
        if (!file.is_open()) {
            result.error_message = "Failed to open file: " + filepath;
            return result;
        }

        std::string script((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());
        file.close();

        // Execute the script content
        result = ExecuteScript(script);

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
        spdlog::error("File execution error: {}", e.what());
    }

    return result;
}

} // namespace scripting
