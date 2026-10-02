#include "scripting_engine.h"
#include "../gui/panels/training_plot_panel.h"
#include "../core/project_manager.h"
#include "../core/engine_config.h"
#include "../data/data_table.h"
#include <Python.h>  // For PyThreadState_SetAsyncExc, PyThread_get_thread_ident
#include <pybind11/embed.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <optional>
#include <filesystem>
#include <algorithm>
#include <iterator>
#include <sstream>

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
    // The language worker uses Python: it ends first.
    if (language_) {
        language_->Stop();
        language_.reset();
    }
    if (variables_) {
        variables_->Stop();
        variables_.reset();
    }

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
    return python_engine_ && python_engine_->IsInitialized();
}

bool ScriptingEngine::EnsurePythonInitialized(std::string* error_out) {
    if (!python_engine_) {
        if (error_out) {
            *error_out = "Python engine not available";
        }
        return false;
    }

    // Check if a project is active - Python should only initialize with a project
    auto& pm = cyxwiz::ProjectManager::Instance();
    if (!pm.HasActiveProject()) {
        if (error_out) {
            *error_out = "No project loaded. Python not initialized.\nCreate or open a project to use Python.";
        }
        spdlog::warn("Python initialization blocked: no active project");
        return false;
    }

    // Starting now would use the base interpreter and need a restart to move
    // to the project environment once it exists.
    if (!python_engine_->IsInitialized() &&
        pm.IsPythonEnvSetupPending(pm.GetProjectRoot())) {
        if (error_out) {
            *error_out = "The project's Python environment is still being set up. "
                         "Python will be available in a few seconds.";
        }
        spdlog::info("Python initialization deferred: project environment setup in progress");
        return false;
    }

    if (python_engine_->IsInitialized()) {
        std::string mismatch;
        if (python_engine_->HasInterpreterMismatch(&mismatch)) {
            std::string msg = "Python interpreter already initialized with a different runtime; restart required (" + mismatch + ")";
            spdlog::warn("{}", msg);
            if (error_out) {
                *error_out = msg;
            }
            return false;
        }
        return true;
    }

    std::string init_error;
    if (!python_engine_->Initialize(&init_error)) {
        if (error_out) {
            *error_out = init_error.empty() ? "Failed to initialize Python" : init_error;
        }
        std::lock_guard<std::mutex> lock(init_error_mutex_);
        last_init_error_ = init_error.empty() ? "Failed to initialize Python" : init_error;
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(init_error_mutex_);
        last_init_error_.clear();
    }
    return true;
}

std::string ScriptingEngine::GetLastInitError() const {
    std::lock_guard<std::mutex> lock(init_error_mutex_);
    return last_init_error_;
}

ScriptingEngine::InterpreterInfo ScriptingEngine::GetInterpreterInfo() {
    InterpreterInfo info;
    if (!python_engine_) {
        return info;
    }
    info.initialized = python_engine_->IsInitialized();
    if (info.initialized) {
        info.interpreter_path = python_engine_->ActiveInterpreterPath();
        info.source = python_engine_->ActiveSource();
        python_engine_->HasInterpreterMismatch(&info.mismatch);
        if (cached_python_version_.empty() && !python_busy_ &&
            !command_running_ && !script_running_) {
            try {
                py::gil_scoped_acquire gil;
                py::object vi = py::module_::import("sys").attr("version_info");
                cached_python_version_ = std::to_string(vi.attr("major").cast<int>()) + "." +
                    std::to_string(vi.attr("minor").cast<int>()) + "." +
                    std::to_string(vi.attr("micro").cast<int>());
            } catch (const std::exception& e) {
                spdlog::debug("GetInterpreterInfo: version unavailable: {}", e.what());
            }
        }
        info.version = cached_python_version_;
    } else {
        info.interpreter_path = python_engine_->PreviewInterpreterPath();
    }
    return info;
}

std::vector<std::string> ScriptingEngine::CompleteSync(const std::string& text,
                                                       size_t max_results) {
    std::vector<std::string> matches;
    if (text.empty() || !python_engine_ || !python_engine_->IsInitialized() ||
        python_busy_ || command_running_ || script_running_) {
        return matches;
    }
    try {
        py::gil_scoped_acquire gil;
        py::object namespace_dict = py::module_::import("__main__").attr("__dict__");
        py::object completer =
            py::module_::import("rlcompleter").attr("Completer")(namespace_dict);
        for (int state = 0; matches.size() < max_results; ++state) {
            py::object match = completer.attr("complete")(text, state);
            if (match.is_none()) break;
            std::string value = match.cast<std::string>();
            if (std::find(matches.begin(), matches.end(), value) == matches.end()) {
                matches.push_back(std::move(value));
            }
        }
    } catch (const std::exception& e) {
        spdlog::debug("CompleteSync: {}", e.what());
        matches.clear();
    }
    return matches;
}

bool ScriptingEngine::ResetSession(std::string* error_out) {
    if (!python_engine_ || !python_engine_->IsInitialized()) {
        return true;  // nothing to clear yet
    }
    if (!IsSafeForNewCommand()) {
        if (error_out) *error_out = "Python is busy; stop the running command first.";
        return false;
    }
    try {
        py::gil_scoped_acquire gil;
        // Clear from C++: Python code run in __main__ would delete its own
        // loop variables mid-loop.
        static const char* const kKeep[] = {"__name__", "__doc__", "__package__",
                                            "__loader__", "__spec__", "__builtins__",
                                            "__annotations__"};
        py::dict ns = py::module_::import("__main__").attr("__dict__");
        py::list keys = ns.attr("keys")();
        for (const auto& key : keys) {
            const std::string name = py::str(key);
            if (std::find(std::begin(kKeep), std::end(kKeep), name) == std::end(kKeep)) {
                PyDict_DelItem(ns.ptr(), key.ptr());
            }
        }
    } catch (const std::exception& e) {
        if (error_out) *error_out = e.what();
        return false;
    }
    matlab_aliases_initialized_ = false;
    training_dashboard_registered_ = false;
    return true;
}

std::string ScriptingEngine::GetPythonRuntimeDiagnostics() {
    if (!IsInitialized()) {
        return "Python engine not initialized. Open the Python console or run a script to initialize.";
    }

    if (!IsSafeForNewCommand()) {
        return "Python interpreter is busy.";
    }

    try {
        py::gil_scoped_acquire acquire;

        py::module_ sys = py::module_::import("sys");
        py::module_ os = py::module_::import("os");

        py::object site;
        bool has_site = true;
        try {
            site = py::module_::import("site");
        } catch (const py::error_already_set&) {
            has_site = false;
        }

        auto getenv_str = [&os](const char* key) -> std::string {
            py::object val = os.attr("getenv")(key);
            if (val.is_none()) {
                return "<unset>";
            }
            return py::str(val).cast<std::string>();
        };

        std::ostringstream out;
        out << "exe: " << py::str(sys.attr("executable")).cast<std::string>() << "\n";
        out << "version: " << py::str(sys.attr("version")).cast<std::string>() << "\n";
        out << "prefix: " << py::str(sys.attr("prefix")).cast<std::string>() << "\n";
        out << "base_prefix: " << py::str(sys.attr("base_prefix")).cast<std::string>() << "\n";
        out << "PYTHONHOME: " << getenv_str("PYTHONHOME") << "\n";
        out << "PYTHONPATH: " << getenv_str("PYTHONPATH") << "\n";

        if (has_site) {
            try {
                py::object getsite = site.attr("getsitepackages");
                py::object sites = getsite();
                out << "site-packages:\n";
                for (auto item : sites) {
                    out << "  " << py::str(item).cast<std::string>() << "\n";
                }
            } catch (const py::error_already_set&) {
                out << "site-packages: n/a\n";
            }

            try {
                py::object usersite = site.attr("getusersitepackages")();
                out << "user-site: " << py::str(usersite).cast<std::string>() << "\n";
            } catch (const py::error_already_set&) {
                out << "user-site: n/a\n";
            }
        } else {
            out << "site-packages: n/a\n";
            out << "user-site: n/a\n";
        }

        out << "sys.path:\n";
        py::list sys_path = sys.attr("path").cast<py::list>();
        for (auto item : sys_path) {
            out << "  " << py::str(item).cast<std::string>() << "\n";
        }

        return out.str();

    } catch (const py::error_already_set& e) {
        return std::string("Failed to collect Python runtime diagnostics: ") + e.what();
    } catch (const std::exception& e) {
        return std::string("Failed to collect Python runtime diagnostics: ") + e.what();
    }
}

bool ScriptingEngine::ReloadPythonForProject() {
    if (!python_engine_) {
        spdlog::warn("ReloadPythonForProject: scripting engine not initialized");
        return false;
    }

    if (!IsSafeForNewCommand()) {
        spdlog::warn("ReloadPythonForProject: Python interpreter is busy");
        return false;
    }

    std::string error;
    if (!python_engine_->ReloadForProject(&error)) {
        if (!error.empty()) {
            spdlog::warn("ReloadPythonForProject: {}", error);
        }
        return false;
    }

    return true;
}

ExecutionResult ScriptingEngine::ExecuteCommand(const std::string& command) {
    ExecutionResult result;
    result.success = false;

    std::string init_error;
    if (!EnsurePythonInitialized(&init_error)) {
        result.error_message = init_error.empty() ? "Python initialization failed" : init_error;
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
        return ExecuteCommandDirect(command, is_internal);
    }

    // Use Python's own threading for timeout (safer than C++ threading with pybind11)
    return ExecuteCommandWithPythonTimeout(command);
}

ExecutionResult ScriptingEngine::ExecuteCommandDirect(const std::string& command, bool suppress_output_callback) {
    ExecutionResult result;
    result.success = false;

    // Safety check: don't try to acquire GIL if Python is busy in another thread
    // This prevents deadlock/freeze when async command is running
    if (python_busy_ || command_running_ || script_running_) {
        result.error_message = "Python interpreter is busy";
        spdlog::debug("ExecuteCommandDirect: skipped - Python busy");
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

        (void)suppress_output_callback;  // synchronous commands return their output

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
        exec(''')" + escaped_command + R"(''', __import__('__main__').__dict__)
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

    std::string init_error;
    if (!EnsurePythonInitialized(&init_error)) {
        ExecutionResult result;
        result.success = false;
        result.error_message = init_error.empty() ? "Python initialization failed" : init_error;
        {
            std::lock_guard<std::mutex> lock(command_result_mutex_);
            async_command_result_ = result;
        }
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

    // The command's wait loop polls this flag and stops the command thread
    // at its next line. No PyErr_SetInterrupt: that KeyboardInterrupt is
    // delivered to whichever Python code the main thread runs next (for
    // example a later Restart), not to the command.
    shared_cancel_flag_.store(1);
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

    // Mark Python as busy before any GIL operations
    python_busy_ = true;

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

        // The source reaches Python as a value, never spliced into code.
        py::module_::import("__main__").attr("_cyxwiz_repl_source") = command;
        // Polled while the command runs, so Interrupt stops it at the next
        // Python line (a blocking C call finishes first).
        py::module_::import("__main__").attr("_cyxwiz_repl_cancelled") =
            py::cpp_function([]() { return shared_cancel_flag_.load() != 0; });

        double timeout_secs = console_timeout_seconds_;

        // Interactive semantics like the standard REPL: the last expression's
        // value is echoed, tracebacks show the user's own lines, and stdout,
        // stderr (warnings), the exception type and the traceback stay apart.
        std::string timeout_code = R"(
import sys, threading, io, ast, traceback, linecache

_cmd_result = {'output': '', 'stderr': '', 'error': '', 'traceback': '',
               'exc_type': '', 'success': False, 'timeout': False}
_cmd_cancel = [False]

def _cmd_trace(frame, event, arg):
    if _cmd_cancel[0]:
        raise KeyboardInterrupt("Command timeout")
    return _cmd_trace

def _run_command():
    global _cmd_result
    ns = __import__('__main__').__dict__
    src = ns.pop('_cyxwiz_repl_source', '')
    linecache.cache['<console>'] = (len(src), None, src.splitlines(True), '<console>')
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    sys.settrace(_cmd_trace)
    try:
        tree = ast.parse(src, '<console>', 'exec')
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = tree.body.pop()
        exec(compile(tree, '<console>', 'exec'), ns)
        if last is not None:
            value = eval(compile(ast.Expression(last.value), '<console>', 'eval'), ns)
            if value is not None:
                ns['_'] = value
                print(repr(value))
        _cmd_result['success'] = True
    except KeyboardInterrupt:
        _cmd_result['error'] = 'Command interrupted'
        _cmd_result['exc_type'] = 'KeyboardInterrupt'
    except BaseException as e:
        _cmd_result['exc_type'] = type(e).__name__
        _cmd_result['error'] = ''.join(traceback.format_exception_only(type(e), e)).strip()
        frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename != '<string>']
        if frames:
            _cmd_result['traceback'] = ('Traceback (most recent call last):\n' +
                ''.join(traceback.format_list(frames)) + _cmd_result['error'])
        else:
            _cmd_result['traceback'] = _cmd_result['error']
    finally:
        sys.settrace(None)
        _cmd_result['output'] = sys.stdout.getvalue()
        _cmd_result['stderr'] = sys.stderr.getvalue()
        sys.stdout = old_stdout
        sys.stderr = old_stderr

_cmd_thread = threading.Thread(target=_run_command)
_cmd_thread.start()
_cmd_deadline = __import__('time').monotonic() + )" + std::to_string(timeout_secs) + R"(
_cmd_cancelled = __import__('__main__').__dict__.pop('_cyxwiz_repl_cancelled', lambda: False)
while _cmd_thread.is_alive():
    _cmd_thread.join(0.05)
    if not _cmd_cancel[0] and _cmd_cancelled():
        _cmd_cancel[0] = True  # user interrupt: stop at the next line
    if __import__('time').monotonic() > _cmd_deadline:
        _cmd_cancel[0] = True
        _cmd_thread.join(timeout=2.0)
        if _cmd_thread.is_alive():
            _cmd_result['error'] = 'Command timed out and could not be stopped'
        else:
            _cmd_result['error'] = 'Command interrupted (timeout)'
        _cmd_result['timeout'] = True
        _cmd_result['success'] = False
        break
)";

        py::exec(timeout_code);

        // Get the result from Python
        py::dict cmd_result = py::eval("_cmd_result").cast<py::dict>();

        result.success = cmd_result["success"].cast<bool>();
        result.output = cmd_result["output"].cast<std::string>();
        result.stderr_output = cmd_result["stderr"].cast<std::string>();
        result.error_message = cmd_result["error"].cast<std::string>();
        result.traceback = cmd_result["traceback"].cast<std::string>();
        result.exception_type = cmd_result["exc_type"].cast<std::string>();
        result.timeout_exceeded = cmd_result["timeout"].cast<bool>();

        if (result.timeout_exceeded) {
            spdlog::warn("Async command timed out after {} seconds", timeout_secs);
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

    // Track command completion time and error status for cooldown
    last_command_end_time_ = std::chrono::steady_clock::now();
    last_command_had_error_ = !result.success;

    // Clear Python busy flag - GIL is now fully released
    python_busy_ = false;

    command_running_ = false;
    shared_cancel_flag_.store(0);  // Reset cancel flag

    spdlog::info("CommandAsyncWorker: completed, success={}", result.success);
}

bool ScriptingEngine::IsSafeForNewCommand() const {
    // Not safe if a command or script is running
    if (command_running_ || script_running_) {
        return false;
    }

    // Not safe if Python is currently busy (even if command_running_ just turned false)
    if (python_busy_) {
        return false;
    }

    // After an error, wait for a short cooldown before allowing new commands
    // This gives Python time to clean up exception state
    if (last_command_had_error_) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_command_end_time_).count();
        if (elapsed_ms < POST_ERROR_COOLDOWN_MS) {
            return false;
        }
    }

    return true;
}

ExecutionResult ScriptingEngine::ExecuteScript(const std::string& script) {
    std::string init_error;
    if (!EnsurePythonInitialized(&init_error)) {
        ExecutionResult result;
        result.success = false;
        result.error_message = init_error.empty() ? "Python initialization failed" : init_error;
        return result;
    }

    // Lazily register training dashboard on first script execution
    EnsureTrainingDashboardRegistered();

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

    std::string init_error;
    if (!EnsurePythonInitialized(&init_error)) {
        result.error_message = init_error.empty() ? "Python initialization failed" : init_error;
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
    // Defer Python module import - importing cyxwiz_plotting at startup can segfault
    // when stdin is not a terminal. The panel pointer is stored and registration
    // happens lazily on first script execution that needs it.
    training_plot_panel_ = panel;
    spdlog::info("Training Dashboard panel stored for deferred registration");
}

void ScriptingEngine::EnsureTrainingDashboardRegistered() {
    if (!training_plot_panel_ || training_dashboard_registered_) return;

    try {
        py::gil_scoped_acquire acquire;
        py::module_ plotting_module = py::module_::import("cyxwiz_plotting");
        py::object set_func = plotting_module.attr("set_training_plot_panel");
        set_func(py::cast(training_plot_panel_, py::return_value_policy::reference));
        training_dashboard_registered_ = true;
        spdlog::info("Training Dashboard registered with Python successfully");
    } catch (const py::error_already_set& e) {
        spdlog::warn("Failed to register Training Dashboard with Python: {}", e.what());
    } catch (const std::exception& e) {
        spdlog::warn("Exception while registering Training Dashboard: {}", e.what());
    } catch (...) {
        spdlog::warn("Unknown error registering Training Dashboard with Python");
    }
}

// ========== Async Execution Implementation ==========

bool ScriptingEngine::ExecuteScriptAsync(const std::string& script, RunCallbacks callbacks) {
    // Don't start if already running
    if (script_running_) {
        spdlog::warn("Script already running, ignoring new execution request");
        return false;
    }

    std::string init_error;
    if (!EnsurePythonInitialized(&init_error)) {
        ExecutionResult result;
        result.success = false;
        result.error_message = init_error.empty() ? "Python initialization failed" : init_error;
        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            async_result_ = result;
        }
        if (callbacks.on_complete) {
            callbacks.on_complete(result);
        }
        return true;
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
    script_thread_ = std::make_unique<std::thread>(&ScriptingEngine::ScriptWorker, this, script, std::move(callbacks));

    spdlog::info("Script execution started in background thread");
    return true;
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

void ScriptingEngine::ScriptWorker(const std::string& script, RunCallbacks callbacks) {
    spdlog::debug("Script worker thread started");

    ExecutionResult result = ExecuteWithStreaming(script, callbacks);

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

    // This run's completion callback, if any
    if (callbacks.on_complete) {
        try {
            callbacks.on_complete(result);
        } catch (const std::exception& e) {
            spdlog::error("Exception in completion callback: {}", e.what());
        }
    }

    spdlog::debug("Script worker thread finished");
}

LanguageService& ScriptingEngine::Language() {
    std::lock_guard<std::mutex> lock(language_mutex_);
    if (!language_) language_ = std::make_unique<LanguageService>(this);
    return *language_;
}

VariablesService& ScriptingEngine::Variables() {
    std::lock_guard<std::mutex> lock(language_mutex_);
    if (!variables_) variables_ = std::make_unique<VariablesService>(this);
    return *variables_;
}

bool ScriptingEngine::RunActive() const { return script_running_ || command_running_ || python_busy_; }

namespace {
void DropFromPath(py::list& path, const std::string& dir) {
    for (size_t i = 0; i < py::len(path); ++i) {
        if (py::str(path[i]).cast<std::string>() == dir) {
            path.attr("pop")(i);
            return;
        }
    }
}

// A module of <exe dir>/python_tools, imported once and kept by sys.modules;
// the folder is on sys.path only while importing. Needs the GIL.
py::object BundledTool(const char* name) {
    py::module_ sys_module = py::module_::import("sys");
    if (sys_module.attr("modules").attr("__contains__")(name).cast<bool>()) return sys_module.attr("modules")[name];
    const std::string dir = (cyxwiz::core::ExecutableDirectory() / "python_tools").string();
    py::list path = sys_module.attr("path");
    path.insert(0, dir);
    py::object module;
    try {
        module = py::module_::import(name);
    } catch (...) {
        DropFromPath(path, dir);
        throw;
    }
    DropFromPath(path, dir);
    return module;
}

// The namespace of a scope: the session's __main__, or a notebook's own
// (an empty dict when that notebook has not run yet).
py::dict ScopeNamespace(const std::string& scope) {
    auto main = py::module_::import("__main__").attr("__dict__").cast<py::dict>();
    if (scope.empty()) return main;
    if (main.contains("_cyxwiz_notebook_ns")) {
        auto all = main["_cyxwiz_notebook_ns"].cast<py::dict>();
        if (all.contains(scope)) return all[py::str(scope)].cast<py::dict>();
    }
    return py::dict();
}

cyxwiz::DataTable::CellValue CellFrom(const py::handle& v) {
    if (v.is_none()) return std::monostate{};
    if (py::isinstance<py::bool_>(v)) return std::string(v.cast<bool>() ? "True" : "False");
    if (py::isinstance<py::int_>(v)) {
        try {
            return static_cast<int64_t>(v.cast<long long>());
        } catch (const py::cast_error&) {
            return py::str(v).cast<std::string>();  // beyond 64 bits
        }
    }
    if (py::isinstance<py::float_>(v)) return v.cast<double>();
    return py::str(v).cast<std::string>();
}
}  // namespace

std::string ScriptingEngine::CallVariablesTool(const std::string& function, const std::string& args_json,
                                               const std::string& scope, bool* busy) {
    if (busy) *busy = false;
    if (!IsInitialized()) return {};
    if (RunActive()) {
        if (busy) *busy = true;
        return {};
    }
    try {
        py::gil_scoped_acquire acquire;
        py::object tool = BundledTool("cyxwiz_vars");
        py::module_ json = py::module_::import("json");
        py::dict kwargs = json.attr("loads")(args_json);
        py::object result = tool.attr(function.c_str())(ScopeNamespace(scope), **kwargs);
        return json.attr("dumps")(result).cast<std::string>();
    } catch (const py::error_already_set& e) {
        spdlog::warn("Variables: {} failed: {}", function, e.what());
        return {};
    } catch (const std::exception& e) {
        spdlog::warn("Variables: {} failed: {}", function, e.what());
        return {};
    }
}

bool ScriptingEngine::ReadVariableTable(const std::string& scope, const std::string& path_json, long long max_rows,
                                        const std::vector<int>& index, VariableTable* out, bool* busy) {
    if (busy) *busy = false;
    if (!out) return false;
    if (!IsInitialized()) {
        out->error = "Python is not running";
        return false;
    }
    if (RunActive()) {
        if (busy) *busy = true;
        return false;
    }
    try {
        py::gil_scoped_acquire acquire;
        py::object tool = BundledTool("cyxwiz_vars");
        py::module_ json = py::module_::import("json");
        py::list idx;
        for (int i : index) idx.append(i);
        py::object rows_arg = max_rows > 0 ? py::object(py::int_(max_rows)) : py::object(py::none());
        py::dict data = tool.attr("table")(ScopeNamespace(scope), json.attr("loads")(path_json), rows_arg, idx);
        if (data.contains("error")) {
            out->error = data["error"].cast<std::string>();
            return false;
        }
        out->kind = data["kind"].cast<std::string>();
        for (auto n : data["shape"].cast<py::list>()) out->shape.push_back(n.cast<long long>());
        out->rows = data["rows"].cast<long long>();
        out->shown = data["shown"].cast<long long>();
        for (auto n : data["slice"].cast<py::list>()) out->slice.push_back(n.cast<int>());
        py::list columns = data["columns"].cast<py::list>();
        const bool has_index = !data["index"].is_none();
        std::vector<std::string> headers;
        if (has_index) {
            headers.push_back(data["index_name"].cast<std::string>());
            out->dtypes.push_back("index");
        }
        std::vector<py::list> cols;
        for (auto c : columns) {
            headers.push_back(c["name"].cast<std::string>());
            out->dtypes.push_back(c["dtype"].cast<std::string>());
            cols.push_back(c["values"].cast<py::list>());
        }
        auto table = std::make_shared<cyxwiz::DataTable>();
        table->SetHeaders(headers);
        py::list index_values = has_index ? data["index"].cast<py::list>() : py::list();
        const size_t n = static_cast<size_t>(out->shown);
        for (size_t r = 0; r < n; ++r) {
            cyxwiz::DataTable::Row row;
            row.reserve(headers.size());
            if (has_index) row.push_back(CellFrom(index_values[r]));
            for (auto& col : cols) row.push_back(r < py::len(col) ? CellFrom(col[r]) : cyxwiz::DataTable::CellValue{});
            table->AddRow(std::move(row));
        }
        out->table = std::move(table);
        return true;
    } catch (const py::error_already_set& e) {
        out->error = e.what();
        return false;
    } catch (const std::exception& e) {
        out->error = e.what();
        return false;
    }
}

std::string ScriptingEngine::LanguageToolsError() const {
    std::lock_guard<std::mutex> lock(language_mutex_);
    return language_error_;
}

std::string ScriptingEngine::CallLanguageTool(const std::string& function, const std::string& args_json,
                                              const std::string& notebook_namespace) {
    if (!IsInitialized()) return {};
    try {
        py::gil_scoped_acquire acquire;
        // The module is imported once per interpreter and kept by sys.modules
        // (a C++ static would outlive Python and release it after shutdown).
        py::module_ sys_module = py::module_::import("sys");
        py::object intel;
        if (sys_module.attr("modules").attr("__contains__")("cyxwiz_intel").cast<bool>()) {
            intel = sys_module.attr("modules")["cyxwiz_intel"];
        } else {
            const auto tools = cyxwiz::core::ExecutableDirectory() / "python_tools";
            py::list path = sys_module.attr("path");
            const std::string dir = tools.string();
            path.insert(0, dir);
            try {
                intel = py::module_::import("cyxwiz_intel");
            } catch (const py::error_already_set& e) {
                std::lock_guard<std::mutex> lock(language_mutex_);
                language_error_ = std::string("The Script Editor's language tools are missing: ") + e.what();
            }
            // Only cyxwiz_intel came from there; it loads jedi/pyflakes itself.
            for (size_t i = 0; i < py::len(path); ++i) {
                if (py::str(path[i]).cast<std::string>() == dir) {
                    path.attr("pop")(i);
                    break;
                }
            }
            if (!intel) return {};
            const py::dict status = intel.attr("status")();
            std::lock_guard<std::mutex> lock(language_mutex_);
            language_error_ = status["ok"].cast<bool>() ? std::string() : status["error"].cast<std::string>();
            spdlog::info("Script Editor language tools: {}", language_error_.empty()
                                                                 ? "jedi " + status["jedi"].cast<std::string>()
                                                                 : language_error_);
        }
        py::module_ json = py::module_::import("json");
        py::dict kwargs = json.attr("loads")(args_json);
        if (!notebook_namespace.empty()) {
            auto main = py::module_::import("__main__").attr("__dict__").cast<py::dict>();
            if (main.contains("_cyxwiz_notebook_ns")) {
                auto all = main["_cyxwiz_notebook_ns"].cast<py::dict>();
                if (all.contains(notebook_namespace)) kwargs["namespace"] = all[py::str(notebook_namespace)];
            }
        }
        py::object result = intel.attr(function.c_str())(**kwargs);
        return json.attr("dumps")(result).cast<std::string>();
    } catch (const py::error_already_set& e) {
        spdlog::debug("Language tool {} failed: {}", function, e.what());
        return {};
    } catch (const std::exception& e) {
        spdlog::debug("Language tool {} failed: {}", function, e.what());
        return {};
    }
}

bool ScriptingEngine::DropNotebookNamespace(const std::string& key) {
    if (script_running_) return false;
    if (!IsInitialized()) return true;  // nothing ran yet
    try {
        py::gil_scoped_acquire acquire;
        auto main = py::module_::import("__main__").attr("__dict__").cast<py::dict>();
        if (main.contains("_cyxwiz_notebook_ns")) {
            auto all = main["_cyxwiz_notebook_ns"].cast<py::dict>();
            if (all.contains(key)) {
                auto ns = all[py::str(key)].cast<py::dict>();
                ns.clear();  // break cycles through the namespace before it goes
                all.attr("pop")(key);
            }
        }
        py::module_::import("gc").attr("collect")();
    } catch (const py::error_already_set& e) {
        spdlog::warn("Could not reset notebook namespace: {}", e.what());
    }
    return true;
}

bool ScriptingEngine::ExportNotebookValueToCsv(const std::string& key, int count, const std::string& path,
                                               std::string* error) {
    if (script_running_) {
        if (error) *error = "A script is running; try again when it finishes";
        return false;
    }
    if (!IsInitialized()) {
        if (error) *error = "Python is not running; run the cell again";
        return false;
    }
    try {
        py::gil_scoped_acquire acquire;
        auto main = py::module_::import("__main__").attr("__dict__").cast<py::dict>();
        py::object export_fn = main.contains("_cyxwiz_export_value") ? py::object(main["_cyxwiz_export_value"]) : py::none();
        if (export_fn.is_none()) {
            if (error) *error = "Run a cell first";
            return false;
        }
        const std::string reason = export_fn(key, count, path).cast<std::string>();
        if (!reason.empty()) {
            if (error) *error = reason;
            return false;
        }
        return true;
    } catch (const py::error_already_set& e) {
        if (error) *error = e.what();
        return false;
    }
}

bool ScriptingEngine::ExportNotebookVariableToCsv(const std::string& key, const std::string& name, const std::string& path,
                                                  std::string* error) {
    if (script_running_ || command_running_) {
        if (error) *error = "A script is running; try again when it finishes";
        return false;
    }
    if (!IsInitialized()) {
        if (error) *error = "Python is not running";
        return false;
    }
    try {
        py::gil_scoped_acquire acquire;
        auto main = py::module_::import("__main__").attr("__dict__").cast<py::dict>();
        if (!main.contains("_cyxwiz_export_variable")) {
            if (error) *error = "Run a cell first";
            return false;
        }
        const std::string reason = py::object(main["_cyxwiz_export_variable"])(key, name, path).cast<std::string>();
        if (!reason.empty() && error) *error = reason;
        return reason.empty();
    } catch (const py::error_already_set& e) {
        if (error) *error = e.what();
        return false;
    }
}

bool ScriptingEngine::NotebookVariablesJson(const std::string& key, std::string* json) {
    if (script_running_ || command_running_) return false;
    if (!IsInitialized()) {
        if (json) *json = "[]";
        return true;
    }
    try {
        py::gil_scoped_acquire acquire;
        auto main = py::module_::import("__main__").attr("__dict__").cast<py::dict>();
        if (!main.contains("_cyxwiz_notebook_variables")) {
            if (json) *json = "[]";
            return true;
        }
        const std::string text = py::object(main["_cyxwiz_notebook_variables"])(key).cast<std::string>();
        if (json) *json = text;
        return true;
    } catch (const py::error_already_set& e) {
        spdlog::warn("Notebook variables: {}", e.what());
        return false;
    }
}

ExecutionResult ScriptingEngine::ExecuteWithStreaming(const std::string& script, const RunCallbacks& callbacks) {
    const OutputCallback& on_output = callbacks.on_output;
    ExecutionResult result;
    result.success = false;

    if (!IsInitialized()) {
        result.error_message = "Scripting engine not initialized";
        return result;
    }

    // MATLAB-style names are installed once per interpreter, like the
    // console does; re-running them before every script overwrote the
    // user's own svd/norm/zeros... (TOFIX133 P0 item 11).
    if (!matlab_aliases_initialized_) {
        InitializeMatlabAliases();
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
        auto queue_func = [this, on_output](const std::string& text) {
            QueueOutput(text);
            if (on_output) {
                on_output(text);
            }
        };

        auto queue_stderr = [this, callbacks](const std::string& text) {
            QueueOutput(text);
            if (callbacks.on_stderr) callbacks.on_stderr(text);
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

        // Register cancellation through a C++ callback instead of ctypes.
        // Some project virtual environments lack _ctypes, and importing ctypes there
        // caused script runs to fail before user code executed.
        std::string setup_code = R"(
import sys

def _cyxwiz_is_cancelled():
    return False

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

# Notebook cells (TOFIX133 P4): one namespace per notebook, kept between
# runs until Restart; the last expression's value is echoed as in Jupyter.
try:
    _cyxwiz_notebook_ns
except NameError:
    _cyxwiz_notebook_ns = {}

def _cyxwiz_export_value(key, count, path):
    ns = _cyxwiz_notebook_ns.get(key)
    if ns is None:
        return 'The notebook was restarted; run the cell again'
    value = ns.get('Out', {}).get(count)
    if value is None:
        return 'This value is no longer in memory; run the cell again'
    if not hasattr(value, 'to_csv'):
        return 'Only tables (pandas DataFrame or Series) open in the Table Viewer'
    value.to_csv(path)
    return ''

def _cyxwiz_export_variable(key, name, path):
    ns = _cyxwiz_notebook_ns.get(key)
    if ns is None or name not in ns:
        return 'This variable is gone (the notebook was restarted)'
    value = ns[name]
    if not hasattr(value, 'to_csv'):
        return 'Only tables (pandas DataFrame or Series) open in the Table Viewer'
    value.to_csv(path)
    return ''

def _cyxwiz_notebook_variables(key):
    import json, reprlib, types
    ns = _cyxwiz_notebook_ns.get(key) or {}
    short = reprlib.Repr()
    short.maxstring = 160
    short.maxother = 160
    short.maxlist = short.maxtuple = short.maxset = short.maxfrozenset = short.maxdeque = 12
    short.maxdict = 12
    short.maxlevel = 2
    out = []
    for name, value in list(ns.items()):
        if name.startswith('_') or name in ('Out', 'In'):
            continue
        if isinstance(value, (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, type)):
            continue
        try:
            type_name = type(value).__name__
            size = ''
            shape = getattr(value, 'shape', None)
            if shape is not None and not callable(shape):
                try:
                    size = str(tuple(shape))
                except Exception:
                    pass
            elif isinstance(value, (list, tuple)):
                size = '(%d,)' % len(value)
            elif isinstance(value, (dict, set, frozenset)):
                size = '%d items' % len(value)
            elif isinstance(value, (str, bytes)):
                size = 'len=%d' % len(value)
            dtype = getattr(value, 'dtype', None)
            if dtype is not None and shape is not None and type_name not in ('DataFrame', 'Series'):
                type_name = '%s[%s]' % (type_name, dtype)
            try:
                if type_name == 'DataFrame':
                    cols = [str(c) for c in list(value.columns)[:12]]
                    text = 'columns: ' + ', '.join(cols) + (', ...' if len(value.columns) > 12 else '')
                elif type_name == 'Series':
                    text = '%s, dtype %s' % (value.name if value.name is not None else 'unnamed', value.dtype)
                else:
                    text = short.repr(value)
            except Exception:
                text = '<no preview>'
            out.append({'name': name, 'type': type_name, 'size': size, 'value': text.replace('\n', ' '),
                        'table': hasattr(value, 'to_csv')})
        except Exception:
            pass
    out.sort(key=lambda v: v['name'].lower())
    return json.dumps(out)

def _cyxwiz_frames(tb, skip):
    import traceback
    return [(f.filename, f.lineno or 0, f.name, f.line or '', '') for f in traceback.extract_tb(tb)[skip:]]

def _cyxwiz_run_cell(src, key, filename, count=0):
    import ast, builtins, linecache, traceback
    ns = _cyxwiz_notebook_ns.get(key)
    if ns is None:
        ns = {'__name__': '__main__', '__builtins__': builtins, 'Out': {}}
        _cyxwiz_notebook_ns[key] = ns
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
    out = {'ok': False, 'repr': None, 'html': None, 'ename': '', 'evalue': '', 'frames': [], 'traceback': ''}
    try:
        tree = ast.parse(src, filename, 'exec')
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = tree.body.pop()
        exec(compile(tree, filename, 'exec'), ns)
        if last is not None:
            value = eval(compile(ast.Expression(last.value), filename, 'eval'), ns)
            if value is not None:
                ns['_'] = value
                if count:
                    ns.setdefault('Out', {})[count] = value
                out['repr'] = repr(value)
                html = getattr(value, '_repr_html_', None)
                if callable(html):
                    try:
                        h = html()
                        if isinstance(h, str) and len(h) < 2_000_000:
                            out['html'] = h
                    except Exception:
                        pass
        out['ok'] = True
    except KeyboardInterrupt:
        raise
    except BaseException as e:
        out['ename'] = type(e).__name__
        out['evalue'] = str(e)
        frames = traceback.extract_tb(e.__traceback__)[1:]  # not this helper
        out['frames'] = _cyxwiz_frames(e.__traceback__, 1)
        if isinstance(e, SyntaxError) and e.filename == filename:
            out['frames'].append((filename, e.lineno or 0, '<module>', (e.text or '').strip(), ''))
            out['evalue'] = e.msg
        # "Caused by": the exception this one was raised from (or during).
        cause = e.__cause__ or (None if e.__suppress_context__ else e.__context__)
        depth = 0
        while cause is not None and depth < 5:
            chained = _cyxwiz_frames(cause.__traceback__, 0)
            label = type(cause).__name__ + (': ' + str(cause) if str(cause) else '')
            if chained:
                f = chained[0]
                chained[0] = (f[0], f[1], f[2], f[3], label)
            else:
                chained = [('', 0, '', '', label)]
            out['frames'].extend(chained)
            cause = cause.__cause__ or (None if cause.__suppress_context__ else cause.__context__)
            depth += 1
        out['traceback'] = ''.join(traceback.format_list(frames)) + ''.join(traceback.format_exception_only(type(e), e))
    return out

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


)";
        py::exec(setup_code);
        py::globals()["_cyxwiz_is_cancelled"] = py::cpp_function([this]() {
            return shared_cancel_flag_.load() != 0;
        });


        // Setup matplotlib capture with our callback
        py::object setup_matplotlib = py::eval("_cyxwiz_setup_matplotlib_capture");
        setup_matplotlib(py::cpp_function(plot_capture_func));

        // Create output object
        py::object output_class = py::eval("_CyxWizOutput");
        py::object output_obj = output_class(py::cpp_function(queue_func));

        // Save original stdout/stderr
        py::object original_stdout = sys.attr("stdout");
        py::object original_stderr = sys.attr("stderr");

        // stderr apart when the caller asks for it (notebook cells)
        py::object error_obj = callbacks.on_stderr ? output_class(py::cpp_function(queue_stderr)) : output_obj;

        // Redirect
        sys.attr("stdout") = output_obj;
        sys.attr("stderr") = error_obj;

        // Set the trace function
        py::exec("sys.settrace(_cyxwiz_trace)");

        try {
            // Execute the user script
            if (callbacks.notebook_namespace.empty()) {
                py::exec(script);
                result.success = true;
            } else {
                const std::string filename = callbacks.cell_filename.empty() ? "<cell>" : callbacks.cell_filename;
                auto out = py::eval("_cyxwiz_run_cell")(script, callbacks.notebook_namespace, filename,
                                                        callbacks.execution_count)
                               .cast<py::dict>();
                result.success = out["ok"].cast<bool>();
                if (!out["repr"].is_none()) result.result_repr = out["repr"].cast<std::string>();
                if (!out["html"].is_none()) result.result_html = out["html"].cast<std::string>();
                if (!result.success) {
                    result.exception_type = out["ename"].cast<std::string>();
                    result.exception_value = out["evalue"].cast<std::string>();
                    result.traceback = out["traceback"].cast<std::string>();
                    result.error_message = result.exception_value.empty()
                                               ? result.exception_type
                                               : result.exception_type + ": " + result.exception_value;
                    for (auto item : out["frames"].cast<py::list>()) {
                        auto t = item.cast<py::tuple>();
                        result.frames.push_back({t[0].cast<std::string>(), t[1].cast<int>(), t[2].cast<std::string>(),
                                                 t[3].cast<std::string>(), t[4].cast<std::string>()});
                    }
                }
            }
            output_obj.attr("flush")();
            if (callbacks.on_stderr) error_obj.attr("flush")();
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
            error_obj.attr("_callback") = py::none();
            error_obj.attr("_buffer") = "";
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

    } catch (const py::error_already_set& e) {
        result.success = false;
        result.error_message = e.what();
        spdlog::error("Execution error: {}", e.what());
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
        spdlog::error("Execution error: {}", e.what());
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
# MATLAB-style Console functions (flat namespace)
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
    # No flat "filter": it would hide Python's builtin. Use cyx.signal.filter.
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
    def _cyxwiz_report_pycyxwiz(error):
        import importlib.util
        import sys
        spec = importlib.util.find_spec("pycyxwiz")
        if spec is None:
            print("[CyxWiz] pycyxwiz not found on sys.path")
        else:
            print(f"[CyxWiz] pycyxwiz found at {getattr(spec, 'origin', None)} but failed to load: {error}")
            print("[CyxWiz] Likely ABI mismatch or missing DLL dependencies.")
            print(f"[CyxWiz] Python: {sys.version}")
    _cyxwiz_report_pycyxwiz(e)
    del _cyxwiz_report_pycyxwiz
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
