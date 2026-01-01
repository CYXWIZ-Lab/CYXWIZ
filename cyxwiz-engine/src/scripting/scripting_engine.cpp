#include "scripting_engine.h"
#include "../gui/panels/training_plot_panel.h"
#include "../core/project_manager.h"
#include <Python.h>  // For PyThreadState_SetAsyncExc, PyThread_get_thread_ident
#include <pybind11/embed.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <optional>
#include <filesystem>
#include <algorithm>

namespace py = pybind11;

namespace scripting {

// Define the static cancellation flag
std::atomic<int> ScriptingEngine::shared_cancel_flag_{0};

ScriptingEngine::ScriptingEngine()
    : sandbox_enabled_(false)
{
    python_engine_ = std::make_unique<PythonEngine>();
    sandbox_ = std::make_unique<PythonSandbox>();
    spdlog::info("ScriptingEngine initialized (sandbox disabled by default)");
}

ScriptingEngine::~ScriptingEngine() {
    spdlog::info("~ScriptingEngine: starting destruction");

    // Stop any running script before destruction
    if (script_running_) {
        spdlog::info("~ScriptingEngine: stopping running script");
        StopScript();
    }
    // Wait for script thread to finish
    if (script_thread_ && script_thread_->joinable()) {
        spdlog::info("~ScriptingEngine: joining script thread");
        script_thread_->join();
    }

    // Stop any running command before destruction
    if (command_running_) {
        spdlog::info("~ScriptingEngine: stopping running command");
        StopCommand();
    }
    // Wait for command thread to finish
    if (command_thread_ && command_thread_->joinable()) {
        spdlog::info("~ScriptingEngine: joining command thread");
        command_thread_->join();
    }

    // Explicitly destroy members before implicit destruction
    // to control the order and log progress
    spdlog::info("~ScriptingEngine: destroying sandbox_");
    sandbox_.reset();
    spdlog::info("~ScriptingEngine: destroying python_engine_");
    python_engine_.reset();
    spdlog::info("~ScriptingEngine: destruction complete");
}

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

    // Initialize MATLAB-style aliases on first command
    if (!matlab_aliases_initialized_) {
        InitializeMatlabAliases();
    }

    // Debug: log commands (optionally skip internal Variable Explorer commands)
    bool is_internal = command.find("_cyxwiz_") != std::string::npos;
    if (verbose_logging_ || !is_internal) {
        spdlog::info("Executing command: [{}] (length={})", command, command.length());
    }

    // Skip timeout for internal commands (fast operations)
    if (is_internal || console_timeout_seconds_ <= 0) {
        return ExecuteCommandDirect(command);
    }

    // Use Python's own threading for timeout (safer than C++ threading with pybind11)
    return ExecuteCommandWithPythonTimeout(command);
}

ExecutionResult ScriptingEngine::ExecuteCommandDirect(const std::string& command) {
    ExecutionResult result;
    result.success = false;

    try {
        // Acquire GIL for this execution
        py::gil_scoped_acquire acquire;

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
        } catch (const py::error_already_set&) {
            // If exec fails, try eval (for expressions)
            try {
                py_result = py::eval(command);
                has_result = true;
            } catch (const py::error_already_set&) {
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

ExecutionResult ScriptingEngine::ExecuteCommandWithPythonTimeout(const std::string& command) {
    ExecutionResult result;
    result.success = false;

    try {
        py::gil_scoped_acquire acquire;

        // Escape the command for embedding in Python triple-quoted string
        std::string escaped_command = command;
        // Replace backslashes first
        size_t pos = 0;
        while ((pos = escaped_command.find('\\', pos)) != std::string::npos) {
            escaped_command.replace(pos, 1, "\\\\");
            pos += 2;
        }
        // Escape triple quotes if they exist
        pos = 0;
        while ((pos = escaped_command.find("'''", pos)) != std::string::npos) {
            escaped_command.replace(pos, 3, "\\'\\'\\'");
            pos += 6;
        }

        int timeout_ms = static_cast<int>(console_timeout_seconds_ * 1000);

        // Python code that runs the command with timeout using threading + trace
        std::string timeout_code = R"(
import sys
import threading
import io
import ctypes

_cmd_result = {'output': '', 'error': '', 'success': False, 'timeout': False}
_cmd_cancel = [False]

def _cmd_trace(frame, event, arg):
    if _cmd_cancel[0]:
        raise KeyboardInterrupt("Command timeout")
    return _cmd_trace

def _run_command():
    global _cmd_result
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    sys.settrace(_cmd_trace)
    try:
        exec(''')" + escaped_command + R"(''')
        _cmd_result['success'] = True
    except KeyboardInterrupt:
        _cmd_result['error'] = 'Command interrupted (timeout)'
        _cmd_result['timeout'] = True
    except Exception as e:
        _cmd_result['error'] = str(e)
    finally:
        sys.settrace(None)
        _cmd_result['output'] = sys.stdout.getvalue()
        if sys.stderr.getvalue():
            _cmd_result['error'] = sys.stderr.getvalue()
        sys.stdout = old_stdout
        sys.stderr = old_stderr

_cmd_thread = threading.Thread(target=_run_command)
_cmd_thread.start()
_cmd_thread.join(timeout=)" + std::to_string(timeout_ms / 1000.0) + R"()

if _cmd_thread.is_alive():
    _cmd_cancel[0] = True
    _cmd_thread.join(timeout=2.0)
    if _cmd_thread.is_alive():
        _cmd_result['error'] = 'Command timed out and could not be stopped'
        _cmd_result['timeout'] = True
    else:
        _cmd_result['error'] = 'Command interrupted (timeout)'
        _cmd_result['timeout'] = True
)";

        py::exec(timeout_code);

        // Get the result from Python
        py::dict cmd_result = py::eval("_cmd_result").cast<py::dict>();

        result.success = cmd_result["success"].cast<bool>();
        result.output = cmd_result["output"].cast<std::string>();
        result.error_message = cmd_result["error"].cast<std::string>();
        result.timeout_exceeded = cmd_result["timeout"].cast<bool>();

        if (result.timeout_exceeded) {
            spdlog::warn("Command timed out after {} seconds", console_timeout_seconds_);
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

void ScriptingEngine::ExecuteCommandWorker(const std::string& command) {
    ExecutionResult result;
    result.success = false;

    try {
        // Acquire GIL for this execution
        py::gil_scoped_acquire acquire;

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
            // Check if this was a KeyboardInterrupt (from timeout)
            if (e.matches(PyExc_KeyboardInterrupt)) {
                result.success = false;
                result.timeout_exceeded = true;
                result.error_message = "Command interrupted (timeout)";
                spdlog::info("Command interrupted via KeyboardInterrupt");

                // Restore stdout/stderr
                sys.attr("stdout") = original_stdout;
                sys.attr("stderr") = original_stderr;

                // Store result and signal completion
                {
                    std::lock_guard<std::mutex> lock(command_mutex_);
                    command_result_ = result;
                    command_finished_ = true;
                }
                command_cv_.notify_one();
                return;
            }

            // If exec fails, try eval (for expressions)
            try {
                py_result = py::eval(command);
                has_result = true;
            } catch (const py::error_already_set& eval_e) {
                // Check for KeyboardInterrupt again
                if (eval_e.matches(PyExc_KeyboardInterrupt)) {
                    result.success = false;
                    result.timeout_exceeded = true;
                    result.error_message = "Command interrupted (timeout)";

                    sys.attr("stdout") = original_stdout;
                    sys.attr("stderr") = original_stderr;

                    {
                        std::lock_guard<std::mutex> lock(command_mutex_);
                        command_result_ = result;
                        command_finished_ = true;
                    }
                    command_cv_.notify_one();
                    return;
                }

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
            result.success = false;
        }

        // Call output callback if set
        if (output_callback_ && !result.output.empty()) {
            output_callback_(result.output);
        }

    } catch (const py::error_already_set& e) {
        // Check for KeyboardInterrupt
        if (e.matches(PyExc_KeyboardInterrupt)) {
            result.success = false;
            result.timeout_exceeded = true;
            result.error_message = "Command interrupted (timeout)";
            spdlog::info("Command interrupted via KeyboardInterrupt");
        } else {
            result.success = false;
            result.error_message = e.what();
            spdlog::error("Python execution error: {}", e.what());
        }
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
        spdlog::error("Execution error: {}", e.what());
    }

    // Store result and signal completion
    {
        std::lock_guard<std::mutex> lock(command_mutex_);
        command_result_ = result;
        command_finished_ = true;
    }
    command_cv_.notify_one();
}

// ========== Async Command Execution ==========

void ScriptingEngine::ExecuteCommandAsync(const std::string& command) {
    // Don't start if already running
    if (command_running_) {
        spdlog::warn("ExecuteCommandAsync: command already running");
        return;
    }

    // Wait for previous thread to finish
    if (command_thread_ && command_thread_->joinable()) {
        command_thread_->join();
    }

    // Reset state
    command_stop_requested_ = false;
    command_running_ = true;
    {
        std::lock_guard<std::mutex> lock(command_result_mutex_);
        async_command_result_.reset();
    }

    spdlog::info("Starting async command execution");

    // Start worker thread
    command_thread_ = std::make_unique<std::thread>(&ScriptingEngine::CommandAsyncWorker, this, command);
}

void ScriptingEngine::StopCommand() {
    if (!command_running_) {
        return;
    }

    spdlog::info("Requesting command stop");
    command_stop_requested_ = true;

    // Set the shared cancel flag that Python trace function checks
    shared_cancel_flag_.store(1);

    // Also send interrupt signal
    {
        py::gil_scoped_acquire acquire;
        PyErr_SetInterrupt();
    }
}

std::optional<ExecutionResult> ScriptingEngine::GetCommandResult() {
    std::lock_guard<std::mutex> lock(command_result_mutex_);
    auto result = async_command_result_;
    if (result) {
        async_command_result_.reset();  // Clear after reading
    }
    return result;
}

void ScriptingEngine::CommandAsyncWorker(const std::string& command) {
    spdlog::info("CommandAsyncWorker: starting for command length {}", command.length());

    // Initialize MATLAB aliases if needed (must be done with GIL)
    if (!matlab_aliases_initialized_) {
        InitializeMatlabAliases();
    }

    // Reset cancel flag
    shared_cancel_flag_.store(0);

    ExecutionResult result;
    result.success = false;

    try {
        py::gil_scoped_acquire acquire;

        // Escape the command for embedding in Python triple-quoted string
        std::string escaped_command = command;
        size_t pos = 0;
        while ((pos = escaped_command.find('\\', pos)) != std::string::npos) {
            escaped_command.replace(pos, 1, "\\\\");
            pos += 2;
        }
        pos = 0;
        while ((pos = escaped_command.find("'''", pos)) != std::string::npos) {
            escaped_command.replace(pos, 3, "\\'\\'\\'");
            pos += 6;
        }

        double timeout_secs = console_timeout_seconds_;

        // Python code that runs the command with timeout using threading + trace
        std::string timeout_code = R"(
import sys
import threading
import io

_cmd_result = {'output': '', 'error': '', 'success': False, 'timeout': False}
_cmd_cancel = [False]

def _cmd_trace(frame, event, arg):
    if _cmd_cancel[0]:
        raise KeyboardInterrupt("Command timeout")
    return _cmd_trace

def _run_command():
    global _cmd_result
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    sys.settrace(_cmd_trace)
    try:
        exec(''')" + escaped_command + R"(''')
        _cmd_result['success'] = True
    except KeyboardInterrupt:
        _cmd_result['error'] = 'Command interrupted (timeout)'
        _cmd_result['timeout'] = True
    except Exception as e:
        _cmd_result['error'] = str(e)
    finally:
        sys.settrace(None)
        _cmd_result['output'] = sys.stdout.getvalue()
        if sys.stderr.getvalue():
            _cmd_result['error'] = sys.stderr.getvalue()
        sys.stdout = old_stdout
        sys.stderr = old_stderr

_cmd_thread = threading.Thread(target=_run_command)
_cmd_thread.start()
_cmd_thread.join(timeout=)" + std::to_string(timeout_secs) + R"()

if _cmd_thread.is_alive():
    _cmd_cancel[0] = True
    _cmd_thread.join(timeout=2.0)
    if _cmd_thread.is_alive():
        _cmd_result['error'] = 'Command timed out and could not be stopped'
        _cmd_result['timeout'] = True
    else:
        _cmd_result['error'] = 'Command interrupted (timeout)'
        _cmd_result['timeout'] = True
)";

        py::exec(timeout_code);

        // Get the result from Python
        py::dict cmd_result = py::eval("_cmd_result").cast<py::dict>();

        result.success = cmd_result["success"].cast<bool>();
        result.output = cmd_result["output"].cast<std::string>();
        result.error_message = cmd_result["error"].cast<std::string>();
        result.timeout_exceeded = cmd_result["timeout"].cast<bool>();

        if (result.timeout_exceeded) {
            spdlog::warn("Async command timed out after {} seconds", timeout_secs);
        }

        // Call output callback if set
        if (output_callback_ && !result.output.empty()) {
            output_callback_(result.output);
        }

    } catch (const py::error_already_set& e) {
        result.success = false;
        result.error_message = e.what();
        spdlog::error("Async command Python error: {}", e.what());
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
        spdlog::error("Async command error: {}", e.what());
    }

    // Check if cancelled
    if (command_stop_requested_) {
        result.was_cancelled = true;
        result.error_message = "Command cancelled by user";
        spdlog::info("Async command was cancelled");
    }

    // Store result
    {
        std::lock_guard<std::mutex> lock(command_result_mutex_);
        async_command_result_ = result;
    }

    command_running_ = false;
    shared_cancel_flag_.store(0);  // Reset cancel flag

    spdlog::info("CommandAsyncWorker: completed, success={}", result.success);
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

void ScriptingEngine::RegisterTrainingDashboard(cyxwiz::TrainingPlotPanel* panel) {
    if (!IsInitialized()) {
        spdlog::error("Cannot register Training Dashboard: scripting engine not initialized");
        return;
    }

    try {
        // Acquire GIL for Python operations
        py::gil_scoped_acquire acquire;

        // Import the cyxwiz_plotting module
        py::module_ plotting_module = py::module_::import("cyxwiz_plotting");

        // Get the set_training_plot_panel function
        py::object set_func = plotting_module.attr("set_training_plot_panel");

        // Call it with the panel pointer (pybind11 will handle the pointer conversion)
        set_func(py::cast(panel, py::return_value_policy::reference));

        spdlog::info("Training Dashboard registered with Python successfully");
    } catch (const py::error_already_set& e) {
        spdlog::error("Failed to register Training Dashboard with Python: {}", e.what());
    } catch (const std::exception& e) {
        spdlog::error("Exception while registering Training Dashboard: {}", e.what());
    }
}

// ========== Async Execution Implementation ==========

void ScriptingEngine::SetCompletionCallback(CompletionCallback callback) {
    completion_callback_ = callback;
}

void ScriptingEngine::ExecuteScriptAsync(const std::string& script) {
    // Don't start if already running
    if (script_running_) {
        spdlog::warn("Script already running, ignoring new execution request");
        return;
    }

    // Wait for previous thread to finish if it exists
    if (script_thread_ && script_thread_->joinable()) {
        script_thread_->join();
    }

    // Clear previous result
    {
        std::lock_guard<std::mutex> lock(result_mutex_);
        async_result_.reset();
    }

    // Clear output queue
    {
        std::lock_guard<std::mutex> lock(output_mutex_);
        std::queue<std::string> empty;
        std::swap(output_queue_, empty);
    }

    // Reset flags
    cancel_requested_ = false;
    script_running_ = true;

    // Start worker thread
    script_thread_ = std::make_unique<std::thread>(&ScriptingEngine::ScriptWorker, this, script);

    spdlog::info("Script execution started in background thread");
}

void ScriptingEngine::StopScript() {
    if (!script_running_) {
        return;
    }

    spdlog::info("Requesting script cancellation...");
    cancel_requested_ = true;

    // Set the shared atomic flag - the trace function will check this
    shared_cancel_flag_.store(1);
    spdlog::info("Set shared_cancel_flag_ = 1");

    // NOTE: We deliberately do NOT use PyThreadState_SetAsyncExc anymore.
    // While it can stop tight loops, it corrupts Python's internal state
    // and causes crashes when pybind11 tries to clean up.
    //
    // Instead, we rely on:
    // 1. The trace function checking shared_cancel_flag_ on each line
    // 2. The output write() function checking the flag
    // 3. Cooperative cancellation for well-behaved scripts
    //
    // For truly uncooperative scripts (like "while True: pass"), users
    // will need to wait or force-close the application.
    spdlog::info("Cancellation flag set. Script will stop at next cooperative check point.");
}

bool ScriptingEngine::IsScriptRunning() const {
    return script_running_;
}

std::optional<ExecutionResult> ScriptingEngine::GetAsyncResult() {
    std::lock_guard<std::mutex> lock(result_mutex_);
    return async_result_;
}

std::string ScriptingEngine::GetPendingOutput() {
    std::lock_guard<std::mutex> lock(output_mutex_);
    std::string result;

    while (!output_queue_.empty()) {
        result += output_queue_.front();
        output_queue_.pop();
    }

    return result;
}

void ScriptingEngine::QueueOutput(const std::string& output) {
    std::lock_guard<std::mutex> lock(output_mutex_);
    output_queue_.push(output);
}

void ScriptingEngine::QueuePlot(const CapturedPlot& plot) {
    std::lock_guard<std::mutex> lock(plot_mutex_);
    plot_queue_.push_back(plot);
}

std::vector<CapturedPlot> ScriptingEngine::GetPendingPlots() {
    std::lock_guard<std::mutex> lock(plot_mutex_);
    std::vector<CapturedPlot> plots;
    std::swap(plots, plot_queue_);
    return plots;
}

void ScriptingEngine::ScriptWorker(const std::string& script) {
    spdlog::debug("Script worker thread started");

    ExecutionResult result = ExecuteWithStreaming(script);

    // Check if cancelled
    if (cancel_requested_) {
        result.was_cancelled = true;
        result.error_message = "Script execution cancelled by user";
    }

    // Store result
    {
        std::lock_guard<std::mutex> lock(result_mutex_);
        async_result_ = result;
    }

    // Mark as not running
    script_running_ = false;

    // Call completion callback if set
    if (completion_callback_) {
        try {
            completion_callback_(result);
        } catch (const std::exception& e) {
            spdlog::error("Exception in completion callback: {}", e.what());
        }
    }

    spdlog::debug("Script worker thread finished");
}

ExecutionResult ScriptingEngine::ExecuteWithStreaming(const std::string& script) {
    ExecutionResult result;
    result.success = false;

    if (!IsInitialized()) {
        result.error_message = "Scripting engine not initialized";
        return result;
    }

    // Reset the cancellation flag at start
    shared_cancel_flag_.store(0);
    python_thread_id_.store(0);

    // Clear any pending plots from previous execution
    {
        std::lock_guard<std::mutex> lock(plot_mutex_);
        plot_queue_.clear();
    }

    try {
        // Acquire GIL for this thread
        py::gil_scoped_acquire acquire;

        // Store the Python thread ID for async exception injection
        unsigned long tid = PyThread_get_thread_ident();
        python_thread_id_.store(tid);
        spdlog::info("Python thread ID: {}", tid);

        py::object sys = py::module_::import("sys");
        py::object os = py::module_::import("os");

        // Set working directory to project root if a project is open
        std::string project_root;
        if (cyxwiz::ProjectManager::Instance().HasActiveProject()) {
            project_root = cyxwiz::ProjectManager::Instance().GetProjectRoot();
            // Normalize path separators for Python (use forward slashes)
            std::replace(project_root.begin(), project_root.end(), '\\', '/');

            try {
                // Change Python's working directory
                os.attr("chdir")(project_root);
                spdlog::info("Python working directory set to project root: {}", project_root);

                // Add project root to sys.path if not already present
                py::list sys_path = sys.attr("path").cast<py::list>();
                bool found = false;
                for (size_t i = 0; i < sys_path.size(); ++i) {
                    std::string path_entry = py::str(sys_path[i]);
                    std::replace(path_entry.begin(), path_entry.end(), '\\', '/');
                    if (path_entry == project_root) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    sys_path.insert(0, project_root);
                    spdlog::info("Added project root to sys.path");
                }

                // Also add the scripts subfolder if it exists
                std::string scripts_path = cyxwiz::ProjectManager::Instance().GetScriptsPath();
                std::replace(scripts_path.begin(), scripts_path.end(), '\\', '/');
                if (std::filesystem::exists(scripts_path)) {
                    bool scripts_found = false;
                    for (size_t i = 0; i < sys_path.size(); ++i) {
                        std::string path_entry = py::str(sys_path[i]);
                        std::replace(path_entry.begin(), path_entry.end(), '\\', '/');
                        if (path_entry == scripts_path) {
                            scripts_found = true;
                            break;
                        }
                    }
                    if (!scripts_found) {
                        sys_path.insert(0, scripts_path);
                        spdlog::info("Added scripts folder to sys.path: {}", scripts_path);
                    }
                }
            } catch (const py::error_already_set& e) {
                spdlog::warn("Failed to set Python working directory: {}", e.what());
            }
        }

        // Create output callback wrapper
        auto queue_func = [this](const std::string& text) {
            QueueOutput(text);
            if (output_callback_) {
                output_callback_(text);
            }
        };

        // Create plot capture callback wrapper
        auto plot_capture_func = [this](py::bytes png_data, int width, int height, const std::string& label) {
            CapturedPlot plot;
            std::string data_str = png_data;  // Convert py::bytes to std::string
            plot.png_data = std::vector<unsigned char>(data_str.begin(), data_str.end());
            plot.width = width;
            plot.height = height;
            plot.label = label;
            QueuePlot(plot);
            spdlog::debug("Captured plot: {}x{}, {} bytes, label: {}", width, height, plot.png_data.size(), label);
        };

        // Get the address of our atomic flag
        void* flag_addr = (void*)&shared_cancel_flag_;
        uintptr_t flag_addr_int = reinterpret_cast<uintptr_t>(flag_addr);

        // Create setup code that uses ctypes to read the flag directly from memory
        // This doesn't need the GIL for reading!
        // Also includes matplotlib capture setup
        std::string setup_code = R"(
import sys
import ctypes

# Memory address of the C++ atomic flag (passed from C++)
_cyxwiz_cancel_flag_addr = )" + std::to_string(flag_addr_int) + R"(

# Create a ctypes pointer to read the flag
_cyxwiz_cancel_ptr = ctypes.cast(_cyxwiz_cancel_flag_addr, ctypes.POINTER(ctypes.c_int))

def _cyxwiz_is_cancelled():
    """Check if script should be cancelled by reading C++ memory directly"""
    return _cyxwiz_cancel_ptr[0] != 0

class _CyxWizOutput:
    def __init__(self, callback):
        self._callback = callback
        self._buffer = ""

    def write(self, text):
        # Safety check - callback might be None after cancellation
        if self._callback is None:
            return
        # Check cancellation on every write
        if _cyxwiz_is_cancelled():
            raise KeyboardInterrupt("Script cancelled")
        self._buffer += text
        if '\n' in self._buffer:
            lines = self._buffer.split('\n')
            for line in lines[:-1]:
                if self._callback is not None:
                    self._callback(line + '\n')
            self._buffer = lines[-1]

    def flush(self):
        if self._buffer and self._callback is not None:
            try:
                self._callback(self._buffer)
            except:
                pass  # Ignore errors during cleanup
            self._buffer = ""

    def getvalue(self):
        return ""

def _cyxwiz_trace(frame, event, arg):
    """Trace function that checks cancellation at every line"""
    if _cyxwiz_is_cancelled():
        raise KeyboardInterrupt("Script cancelled by user")
    return _cyxwiz_trace

# Matplotlib capture setup
_cyxwiz_plot_capture_callback = None
_cyxwiz_captured_plots = []

def _cyxwiz_setup_matplotlib_capture(capture_callback):
    """Setup matplotlib to capture plots instead of showing windows"""
    global _cyxwiz_plot_capture_callback
    _cyxwiz_plot_capture_callback = capture_callback

    try:
        import matplotlib
        # Use non-interactive backend
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Store original show function
        _original_show = plt.show

        def _cyxwiz_show(*args, **kwargs):
            """Capture all figures and send to C++"""
            global _cyxwiz_plot_capture_callback, _cyxwiz_captured_plots
            import io

            # Get all figure numbers
            fig_nums = plt.get_fignums()

            for fig_num in fig_nums:
                fig = plt.figure(fig_num)

                # Get figure size in pixels
                dpi = fig.dpi
                width = int(fig.get_figwidth() * dpi)
                height = int(fig.get_figheight() * dpi)

                # Get title if available
                title = ""
                if fig._suptitle:
                    title = fig._suptitle.get_text()
                elif len(fig.axes) > 0 and fig.axes[0].get_title():
                    title = fig.axes[0].get_title()

                # Render to PNG bytes
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
                buf.seek(0)
                png_data = buf.read()
                buf.close()

                # Send to C++ callback
                if _cyxwiz_plot_capture_callback is not None:
                    _cyxwiz_plot_capture_callback(png_data, width, height, title)
                else:
                    # Store locally if no callback
                    _cyxwiz_captured_plots.append({
                        'data': png_data,
                        'width': width,
                        'height': height,
                        'label': title
                    })

            # Close all figures after capturing
            plt.close('all')

        # Replace plt.show with our capture function
        plt.show = _cyxwiz_show

    except ImportError:
        # matplotlib not installed, silently skip
        pass

# ============================================================================
# MATLAB-Style Command Window Functions (Flat Namespace)
# ============================================================================
# Import pycyxwiz and create convenient aliases
try:
    import pycyxwiz
    cyx = pycyxwiz  # Short alias for grouped namespace

    # Linear Algebra - Flat namespace aliases
    svd = pycyxwiz.linalg.svd
    eig = pycyxwiz.linalg.eig
    qr = pycyxwiz.linalg.qr
    chol = pycyxwiz.linalg.chol
    lu = pycyxwiz.linalg.lu
    det = pycyxwiz.linalg.det
    rank = pycyxwiz.linalg.rank
    trace = pycyxwiz.linalg.trace
    norm = pycyxwiz.linalg.norm
    cond = pycyxwiz.linalg.cond
    inv = pycyxwiz.linalg.inv
    transpose = pycyxwiz.linalg.transpose
    solve = pycyxwiz.linalg.solve
    lstsq = pycyxwiz.linalg.lstsq
    matmul = pycyxwiz.linalg.matmul
    eye = pycyxwiz.linalg.eye
    zeros = pycyxwiz.linalg.zeros
    ones = pycyxwiz.linalg.ones

    # Signal Processing - Flat namespace aliases
    fft = pycyxwiz.signal.fft
    ifft = pycyxwiz.signal.ifft
    conv = pycyxwiz.signal.conv
    conv2 = pycyxwiz.signal.conv2
    spectrogram = pycyxwiz.signal.spectrogram
    lowpass = pycyxwiz.signal.lowpass
    highpass = pycyxwiz.signal.highpass
    bandpass = pycyxwiz.signal.bandpass
    filter = pycyxwiz.signal.filter
    findpeaks = pycyxwiz.signal.findpeaks
    sine = pycyxwiz.signal.sine
    square = pycyxwiz.signal.square
    noise = pycyxwiz.signal.noise

    # Statistics/Clustering - Flat namespace aliases
    kmeans = pycyxwiz.stats.kmeans
    dbscan = pycyxwiz.stats.dbscan
    gmm = pycyxwiz.stats.gmm
    pca = pycyxwiz.stats.pca
    tsne = pycyxwiz.stats.tsne
    silhouette = pycyxwiz.stats.silhouette
    confusion_matrix = pycyxwiz.stats.confusion_matrix
    roc = pycyxwiz.stats.roc

    # Time Series - Flat namespace aliases
    acf = pycyxwiz.timeseries.acf
    pacf = pycyxwiz.timeseries.pacf
    decompose = pycyxwiz.timeseries.decompose
    stationarity = pycyxwiz.timeseries.stationarity
    arima = pycyxwiz.timeseries.arima
    diff = pycyxwiz.timeseries.diff
    rolling_mean = pycyxwiz.timeseries.rolling_mean
    rolling_std = pycyxwiz.timeseries.rolling_std

except ImportError as e:
    # pycyxwiz not available, skip MATLAB-style functions
    print(f"[CyxWiz] pycyxwiz not found: {e}")
except AttributeError as e:
    # submodule not found (linalg, signal, etc.)
    print(f"[CyxWiz] MATLAB functions error: {e}")
except Exception as e:
    # Any other error
    print(f"[CyxWiz] Error loading MATLAB functions: {e}")
)";
        py::exec(setup_code);

        // Setup matplotlib capture with our callback
        py::object setup_matplotlib = py::eval("_cyxwiz_setup_matplotlib_capture");
        setup_matplotlib(py::cpp_function(plot_capture_func));

        // Create output object
        py::object output_class = py::eval("_CyxWizOutput");
        py::object output_obj = output_class(py::cpp_function(queue_func));

        // Save original stdout/stderr
        py::object original_stdout = sys.attr("stdout");
        py::object original_stderr = sys.attr("stderr");

        // Redirect
        sys.attr("stdout") = output_obj;
        sys.attr("stderr") = output_obj;

        // Set the trace function
        py::exec("sys.settrace(_cyxwiz_trace)");

        try {
            // Execute the user script
            py::exec(script);
            output_obj.attr("flush")();
            result.success = true;
        } catch (const py::error_already_set& e) {
            if (e.matches(PyExc_KeyboardInterrupt)) {
                result.success = false;
                result.was_cancelled = true;
                result.error_message = "Script cancelled";
                spdlog::info("Script cancelled via KeyboardInterrupt (cooperative)");
            } else {
                result.success = false;
                result.error_message = e.what();
                spdlog::error("Python execution error: {}", e.what());
            }
        }

        // Normal cleanup - safe because we didn't use PyThreadState_SetAsyncExc
        try {
            output_obj.attr("_callback") = py::none();
            output_obj.attr("_buffer") = "";
        } catch (...) {
            spdlog::warn("Error clearing output callback, ignoring");
        }

        // Restore stdout/stderr and remove trace
        try {
            sys.attr("stdout") = original_stdout;
            sys.attr("stderr") = original_stderr;
            PyEval_SetTrace(nullptr, nullptr);
        } catch (...) {
            spdlog::warn("Error during Python cleanup, ignoring");
        }

        // Clear any pending Python errors
        PyErr_Clear();

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
        spdlog::error("Execution error: {}", e.what());
        // Clear any pending Python errors
        PyErr_Clear();
    }

    // Clear the thread ID when done
    python_thread_id_.store(0);

    // Collect any captured plots
    {
        std::lock_guard<std::mutex> lock(plot_mutex_);
        result.plots = std::move(plot_queue_);
        plot_queue_.clear();
    }

    if (!result.plots.empty()) {
        spdlog::info("Script execution captured {} plot(s)", result.plots.size());
    }

    return result;
}

void ScriptingEngine::InitializeMatlabAliases() {
    if (matlab_aliases_initialized_) return;

    spdlog::info("Initializing MATLAB-style aliases...");

    try {
        py::gil_scoped_acquire acquire;

        // MATLAB-style aliases setup code
        std::string matlab_setup = R"PYTHON(
# ============================================================================
# MATLAB-Style Command Window Functions (Flat Namespace)
# ============================================================================
# Import pycyxwiz and create convenient aliases
try:
    import pycyxwiz
    cyx = pycyxwiz  # Short alias for grouped namespace

    # Linear Algebra - Flat namespace aliases
    svd = pycyxwiz.linalg.svd
    eig = pycyxwiz.linalg.eig
    qr = pycyxwiz.linalg.qr
    chol = pycyxwiz.linalg.chol
    lu = pycyxwiz.linalg.lu
    det = pycyxwiz.linalg.det
    rank = pycyxwiz.linalg.rank
    trace = pycyxwiz.linalg.trace
    norm = pycyxwiz.linalg.norm
    cond = pycyxwiz.linalg.cond
    inv = pycyxwiz.linalg.inv
    transpose = pycyxwiz.linalg.transpose
    solve = pycyxwiz.linalg.solve
    lstsq = pycyxwiz.linalg.lstsq
    matmul = pycyxwiz.linalg.matmul
    eye = pycyxwiz.linalg.eye
    zeros = pycyxwiz.linalg.zeros
    ones = pycyxwiz.linalg.ones

    # Signal Processing - Flat namespace aliases
    fft = pycyxwiz.signal.fft
    ifft = pycyxwiz.signal.ifft
    conv = pycyxwiz.signal.conv
    conv2 = pycyxwiz.signal.conv2
    spectrogram = pycyxwiz.signal.spectrogram
    lowpass = pycyxwiz.signal.lowpass
    highpass = pycyxwiz.signal.highpass
    bandpass = pycyxwiz.signal.bandpass
    filter = pycyxwiz.signal.filter
    findpeaks = pycyxwiz.signal.findpeaks
    sine = pycyxwiz.signal.sine
    square = pycyxwiz.signal.square
    noise = pycyxwiz.signal.noise

    # Statistics/Clustering - Flat namespace aliases
    kmeans = pycyxwiz.stats.kmeans
    dbscan = pycyxwiz.stats.dbscan
    gmm = pycyxwiz.stats.gmm
    pca = pycyxwiz.stats.pca
    tsne = pycyxwiz.stats.tsne
    silhouette = pycyxwiz.stats.silhouette
    confusion_matrix = pycyxwiz.stats.confusion_matrix
    roc = pycyxwiz.stats.roc

    # Time Series - Flat namespace aliases
    acf = pycyxwiz.timeseries.acf
    pacf = pycyxwiz.timeseries.pacf
    decompose = pycyxwiz.timeseries.decompose
    stationarity = pycyxwiz.timeseries.stationarity
    arima = pycyxwiz.timeseries.arima
    diff = pycyxwiz.timeseries.diff
    rolling_mean = pycyxwiz.timeseries.rolling_mean
    rolling_std = pycyxwiz.timeseries.rolling_std

    # Matrix printing helper
    def printmat(matrix, precision=4, suppress_small=True):
        """Print a matrix in MATLAB-style format.

        Args:
            matrix: 2D list or nested list
            precision: Number of decimal places (default 4)
            suppress_small: Replace very small values with 0 (default True)
        """
        if not matrix:
            print("[]")
            return

        # Handle 1D arrays
        if not isinstance(matrix[0], (list, tuple)):
            matrix = [matrix]

        # Find the maximum width needed for formatting
        threshold = 10 ** (-precision) if suppress_small else 0
        formatted = []
        max_width = 0

        for row in matrix:
            row_formatted = []
            for val in row:
                if isinstance(val, (int, float)):
                    if suppress_small and abs(val) < threshold:
                        val = 0.0
                    if isinstance(val, float):
                        s = f"{val:.{precision}f}".rstrip('0').rstrip('.')
                        if '.' not in s:
                            s = f"{val:.1f}"
                    else:
                        s = str(val)
                else:
                    s = str(val)
                row_formatted.append(s)
                max_width = max(max_width, len(s))
            formatted.append(row_formatted)

        # Print with alignment
        for row in formatted:
            print("  " + "  ".join(s.rjust(max_width) for s in row))

    # Short alias
    pm = printmat

    print("[CyxWiz] MATLAB-style functions loaded successfully")

except ImportError as e:
    # pycyxwiz not available, skip MATLAB-style functions
    print(f"[CyxWiz] pycyxwiz not found: {e}")
except AttributeError as e:
    # submodule not found (linalg, signal, etc.)
    print(f"[CyxWiz] MATLAB functions error: {e}")
except Exception as e:
    # Any other error
    print(f"[CyxWiz] Error loading MATLAB functions: {e}")

# ============================================================================
# DuckDB - Fast Analytics Database
# ============================================================================
try:
    import duckdb

    # Create default in-memory database connection
    db = duckdb.connect(':memory:')

    # Convenience function to run SQL and return result
    def sql(query):
        """Execute SQL query and return result as relation.

        Examples:
            sql("SELECT 1 + 1 AS result")
            sql("SELECT * FROM 'data.csv'")
            sql("SELECT * FROM 'data.parquet'")
        """
        return db.sql(query)

    # Quick data loading functions
    def read_csv(path, **kwargs):
        """Read CSV file into DuckDB relation."""
        return db.read_csv(path, **kwargs)

    def read_parquet(path, **kwargs):
        """Read Parquet file into DuckDB relation."""
        return db.read_parquet(path, **kwargs)

    def read_json(path, **kwargs):
        """Read JSON file into DuckDB relation."""
        return db.read_json(path, **kwargs)

    print("[CyxWiz] DuckDB loaded - use sql(), read_csv(), read_parquet(), read_json()")

except ImportError:
    print("[CyxWiz] DuckDB not installed - run: pip install duckdb")
except Exception as e:
    print(f"[CyxWiz] DuckDB error: {e}")

# ============================================================================
# Polars - Fast DataFrame Library
# ============================================================================
try:
    import polars as pl

    # Convenience aliases for common operations
    DataFrame = pl.DataFrame
    LazyFrame = pl.LazyFrame
    Series = pl.Series

    # Quick constructors
    def df(data=None, schema=None, **kwargs):
        """Create a Polars DataFrame.

        Examples:
            df({'a': [1, 2, 3], 'b': [4, 5, 6]})
            df([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}])
        """
        if data is None:
            return pl.DataFrame()
        return pl.DataFrame(data, schema=schema, **kwargs)

    def lf(data=None, schema=None, **kwargs):
        """Create a Polars LazyFrame for deferred execution."""
        if data is None:
            return pl.LazyFrame()
        return pl.DataFrame(data, schema=schema, **kwargs).lazy()

    # File reading shortcuts
    def pl_csv(path, **kwargs):
        """Read CSV file into Polars DataFrame."""
        return pl.read_csv(path, **kwargs)

    def pl_parquet(path, **kwargs):
        """Read Parquet file into Polars DataFrame."""
        return pl.read_parquet(path, **kwargs)

    def pl_json(path, **kwargs):
        """Read JSON file into Polars DataFrame."""
        return pl.read_json(path, **kwargs)

    def pl_excel(path, **kwargs):
        """Read Excel file into Polars DataFrame."""
        return pl.read_excel(path, **kwargs)

    # Scan (lazy) versions
    def scan_csv(path, **kwargs):
        """Lazily read CSV file."""
        return pl.scan_csv(path, **kwargs)

    def scan_parquet(path, **kwargs):
        """Lazily read Parquet file."""
        return pl.scan_parquet(path, **kwargs)

    # Column expression shortcut
    col = pl.col
    lit = pl.lit
    when = pl.when

    print("[CyxWiz] Polars loaded - use pl, df(), col(), scan_csv(), etc.")

except ImportError:
    print("[CyxWiz] Polars not installed - run: pip install polars")
except Exception as e:
    print(f"[CyxWiz] Polars error: {e}")
)PYTHON";

        py::exec(matlab_setup);
        matlab_aliases_initialized_ = true;
        spdlog::info("MATLAB-style aliases initialized");

    } catch (const py::error_already_set& e) {
        spdlog::error("Failed to initialize MATLAB aliases: {}", e.what());
    } catch (const std::exception& e) {
        spdlog::error("Failed to initialize MATLAB aliases: {}", e.what());
    }
}

} // namespace scripting
