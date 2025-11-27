#include "scripting_engine.h"
#include "../gui/panels/training_plot_panel.h"
#include <Python.h>  // For PyThreadState_SetAsyncExc, PyThread_get_thread_ident
#include <pybind11/embed.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <optional>

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
    // Stop any running script before destruction
    if (script_running_) {
        StopScript();
    }
    // Wait for thread to finish
    if (script_thread_ && script_thread_->joinable()) {
        script_thread_->join();
    }
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

    try {
        // Acquire GIL for this thread
        py::gil_scoped_acquire acquire;

        // Store the Python thread ID for async exception injection
        unsigned long tid = PyThread_get_thread_ident();
        python_thread_id_.store(tid);
        spdlog::info("Python thread ID: {}", tid);

        py::object sys = py::module_::import("sys");

        // Create output callback wrapper
        auto queue_func = [this](const std::string& text) {
            QueueOutput(text);
            if (output_callback_) {
                output_callback_(text);
            }
        };

        // Get the address of our atomic flag
        void* flag_addr = (void*)&shared_cancel_flag_;
        uintptr_t flag_addr_int = reinterpret_cast<uintptr_t>(flag_addr);

        // Create setup code that uses ctypes to read the flag directly from memory
        // This doesn't need the GIL for reading!
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
)";
        py::exec(setup_code);

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

    return result;
}

} // namespace scripting
