#include "scripting_engine.h"
#include "../gui/panels/training_plot_panel.h"
#include "../core/project_manager.h"
#include <spdlog/spdlog.h>
#include <fstream>
#include <optional>
#include <filesystem>
#include <algorithm>

#ifdef CYXWIZ_HAS_PYTHON

#include <Python.h>
#include <pybind11/embed.h>

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
    if (script_running_) {
        StopScript();
    }
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

    if (!matlab_aliases_initialized_) {
        InitializeMatlabAliases();
    }

    try {
        py::gil_scoped_acquire acquire;
        py::object sys = py::module_::import("sys");
        py::object io = py::module_::import("io");
        py::object stdout_capture = io.attr("StringIO")();
        py::object stderr_capture = io.attr("StringIO")();
        py::object original_stdout = sys.attr("stdout");
        py::object original_stderr = sys.attr("stderr");
        sys.attr("stdout") = stdout_capture;
        sys.attr("stderr") = stderr_capture;

        py::object py_result;
        bool has_result = false;

        try {
            py::exec(command);
        } catch (const py::error_already_set&) {
            try {
                py_result = py::eval(command);
                has_result = true;
            } catch (const py::error_already_set&) {
                sys.attr("stdout") = original_stdout;
                sys.attr("stderr") = original_stderr;
                throw;
            }
        }

        stdout_capture.attr("seek")(0);
        stderr_capture.attr("seek")(0);
        std::string stdout_str = py::str(stdout_capture.attr("read")());
        std::string stderr_str = py::str(stderr_capture.attr("read")());
        sys.attr("stdout") = original_stdout;
        sys.attr("stderr") = original_stderr;

        result.success = true;
        if (has_result && !py_result.is_none()) {
            result.output = py::str(py_result);
            result.output += "\n";
        }
        if (!stdout_str.empty()) {
            result.output += stdout_str;
        }
        if (!stderr_str.empty()) {
            result.error_message = stderr_str;
            result.success = false;
        }
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
    if (sandbox_enabled_ && sandbox_) {
        auto sandbox_result = sandbox_->Execute(script);
        return ConvertSandboxResult(sandbox_result);
    }
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
        std::ifstream file(filepath);
        if (!file.is_open()) {
            result.error_message = "Failed to open file: " + filepath;
            return result;
        }
        std::string script((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
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
        py::gil_scoped_acquire acquire;
        py::module_ plotting_module = py::module_::import("cyxwiz_plotting");
        py::object set_func = plotting_module.attr("set_training_plot_panel");
        set_func(py::cast(panel, py::return_value_policy::reference));
        spdlog::info("Training Dashboard registered with Python successfully");
    } catch (const py::error_already_set& e) {
        spdlog::error("Failed to register Training Dashboard with Python: {}", e.what());
    } catch (const std::exception& e) {
        spdlog::error("Exception while registering Training Dashboard: {}", e.what());
    }
}

void ScriptingEngine::SetCompletionCallback(CompletionCallback callback) {
    completion_callback_ = callback;
}

void ScriptingEngine::ExecuteScriptAsync(const std::string& script) {
    if (script_running_) {
        spdlog::warn("Script already running, ignoring new execution request");
        return;
    }

    if (script_thread_ && script_thread_->joinable()) {
        script_thread_->join();
    }

    {
        std::lock_guard<std::mutex> lock(result_mutex_);
        async_result_.reset();
    }
    {
        std::lock_guard<std::mutex> lock(output_mutex_);
        std::queue<std::string> empty;
        std::swap(output_queue_, empty);
    }

    cancel_requested_ = false;
    script_running_ = true;
    script_thread_ = std::make_unique<std::thread>(&ScriptingEngine::ScriptWorker, this, script);
    spdlog::info("Script execution started in background thread");
}

void ScriptingEngine::StopScript() {
    if (!script_running_) return;
    spdlog::info("Requesting script cancellation...");
    cancel_requested_ = true;
    shared_cancel_flag_.store(1);
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

    if (cancel_requested_) {
        result.was_cancelled = true;
        result.error_message = "Script execution cancelled by user";
    }

    {
        std::lock_guard<std::mutex> lock(result_mutex_);
        async_result_ = result;
    }

    script_running_ = false;

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

    shared_cancel_flag_.store(0);
    python_thread_id_.store(0);

    {
        std::lock_guard<std::mutex> lock(plot_mutex_);
        plot_queue_.clear();
    }

    try {
        py::gil_scoped_acquire acquire;
        unsigned long tid = PyThread_get_thread_ident();
        python_thread_id_.store(tid);

        py::object sys = py::module_::import("sys");
        py::object os = py::module_::import("os");

        std::string project_root;
        if (cyxwiz::ProjectManager::Instance().HasActiveProject()) {
            project_root = cyxwiz::ProjectManager::Instance().GetProjectRoot();
            std::replace(project_root.begin(), project_root.end(), '\\', '/');
            try {
                os.attr("chdir")(project_root);
                py::list sys_path = sys.attr("path").cast<py::list>();
                bool found = false;
                for (size_t i = 0; i < sys_path.size(); ++i) {
                    std::string path_entry = py::str(sys_path[i]);
                    std::replace(path_entry.begin(), path_entry.end(), '\\', '/');
                    if (path_entry == project_root) { found = true; break; }
                }
                if (!found) sys_path.insert(0, project_root);
            } catch (const py::error_already_set& e) {
                spdlog::warn("Failed to set Python working directory: {}", e.what());
            }
        }

        auto queue_func = [this](const std::string& text) {
            QueueOutput(text);
            if (output_callback_) output_callback_(text);
        };

        auto plot_capture_func = [this](py::bytes png_data, int width, int height, const std::string& label) {
            CapturedPlot plot;
            std::string data_str = png_data;
            plot.png_data = std::vector<unsigned char>(data_str.begin(), data_str.end());
            plot.width = width;
            plot.height = height;
            plot.label = label;
            QueuePlot(plot);
        };

        void* flag_addr = (void*)&shared_cancel_flag_;
        uintptr_t flag_addr_int = reinterpret_cast<uintptr_t>(flag_addr);

        std::string setup_code = R"(
import sys
import ctypes
_cyxwiz_cancel_flag_addr = )" + std::to_string(flag_addr_int) + R"(
_cyxwiz_cancel_ptr = ctypes.cast(_cyxwiz_cancel_flag_addr, ctypes.POINTER(ctypes.c_int))
def _cyxwiz_is_cancelled():
    return _cyxwiz_cancel_ptr[0] != 0
class _CyxWizOutput:
    def __init__(self, callback):
        self._callback = callback
        self._buffer = ""
    def write(self, text):
        if self._callback is None: return
        if _cyxwiz_is_cancelled(): raise KeyboardInterrupt("Script cancelled")
        self._buffer += text
        if '\n' in self._buffer:
            lines = self._buffer.split('\n')
            for line in lines[:-1]:
                if self._callback is not None: self._callback(line + '\n')
            self._buffer = lines[-1]
    def flush(self):
        if self._buffer and self._callback is not None:
            try: self._callback(self._buffer)
            except: pass
            self._buffer = ""
    def getvalue(self): return ""
def _cyxwiz_trace(frame, event, arg):
    if _cyxwiz_is_cancelled(): raise KeyboardInterrupt("Script cancelled by user")
    return _cyxwiz_trace
_cyxwiz_plot_capture_callback = None
def _cyxwiz_setup_matplotlib_capture(capture_callback):
    global _cyxwiz_plot_capture_callback
    _cyxwiz_plot_capture_callback = capture_callback
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        _original_show = plt.show
        def _cyxwiz_show(*args, **kwargs):
            global _cyxwiz_plot_capture_callback
            import io
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                dpi = fig.dpi
                width = int(fig.get_figwidth() * dpi)
                height = int(fig.get_figheight() * dpi)
                title = ""
                if fig._suptitle: title = fig._suptitle.get_text()
                elif len(fig.axes) > 0 and fig.axes[0].get_title(): title = fig.axes[0].get_title()
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
                buf.seek(0)
                png_data = buf.read()
                buf.close()
                if _cyxwiz_plot_capture_callback is not None:
                    _cyxwiz_plot_capture_callback(png_data, width, height, title)
            plt.close('all')
        plt.show = _cyxwiz_show
    except ImportError: pass
)";
        py::exec(setup_code);

        py::object setup_matplotlib = py::eval("_cyxwiz_setup_matplotlib_capture");
        setup_matplotlib(py::cpp_function(plot_capture_func));

        py::object output_class = py::eval("_CyxWizOutput");
        py::object output_obj = output_class(py::cpp_function(queue_func));

        py::object original_stdout = sys.attr("stdout");
        py::object original_stderr = sys.attr("stderr");
        sys.attr("stdout") = output_obj;
        sys.attr("stderr") = output_obj;

        py::exec("sys.settrace(_cyxwiz_trace)");

        try {
            py::exec(script);
            output_obj.attr("flush")();
            result.success = true;
        } catch (const py::error_already_set& e) {
            if (e.matches(PyExc_KeyboardInterrupt)) {
                result.success = false;
                result.was_cancelled = true;
                result.error_message = "Script cancelled";
            } else {
                result.success = false;
                result.error_message = e.what();
            }
        }

        try {
            output_obj.attr("_callback") = py::none();
            output_obj.attr("_buffer") = "";
        } catch (...) {}

        try {
            sys.attr("stdout") = original_stdout;
            sys.attr("stderr") = original_stderr;
            PyEval_SetTrace(nullptr, nullptr);
        } catch (...) {}

        PyErr_Clear();

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
        PyErr_Clear();
    }

    python_thread_id_.store(0);

    {
        std::lock_guard<std::mutex> lock(plot_mutex_);
        result.plots = std::move(plot_queue_);
        plot_queue_.clear();
    }

    return result;
}

void ScriptingEngine::InitializeMatlabAliases() {
    if (matlab_aliases_initialized_) return;
    spdlog::info("Initializing MATLAB-style aliases...");

    try {
        py::gil_scoped_acquire acquire;
        std::string matlab_setup = R"(
try:
    import pycyxwiz
    cyx = pycyxwiz
except ImportError as e:
    print(f"[CyxWiz] pycyxwiz not found: {e}")
except Exception as e:
    print(f"[CyxWiz] Error loading MATLAB functions: {e}")
)";
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

#else // !CYXWIZ_HAS_PYTHON

// Stub implementations when Python is disabled
namespace scripting {

std::atomic<int> ScriptingEngine::shared_cancel_flag_{0};

ScriptingEngine::ScriptingEngine() : sandbox_enabled_(false) {
    python_engine_ = std::make_unique<PythonEngine>();
    sandbox_ = std::make_unique<PythonSandbox>();
}

ScriptingEngine::~ScriptingEngine() {}

void ScriptingEngine::EnableSandbox(bool) {}
void ScriptingEngine::SetSandboxConfig(const PythonSandbox::Config&) {}
PythonSandbox::Config ScriptingEngine::GetSandboxConfig() const { return PythonSandbox::Config(); }
bool ScriptingEngine::IsInitialized() const { return false; }
void ScriptingEngine::SetOutputCallback(OutputCallback) {}

ExecutionResult ScriptingEngine::ExecuteCommand(const std::string&) {
    ExecutionResult result;
    result.success = false;
    result.error_message = "Python support disabled";
    return result;
}

ExecutionResult ScriptingEngine::ExecuteScript(const std::string&) {
    ExecutionResult result;
    result.success = false;
    result.error_message = "Python support disabled";
    return result;
}

ExecutionResult ScriptingEngine::ExecuteFile(const std::string&) {
    ExecutionResult result;
    result.success = false;
    result.error_message = "Python support disabled";
    return result;
}

ExecutionResult ScriptingEngine::ConvertSandboxResult(const PythonSandbox::ExecutionResult&) {
    ExecutionResult result;
    result.success = false;
    result.error_message = "Python support disabled";
    return result;
}

void ScriptingEngine::RegisterTrainingDashboard(cyxwiz::TrainingPlotPanel*) {}
void ScriptingEngine::SetCompletionCallback(CompletionCallback) {}
void ScriptingEngine::ExecuteScriptAsync(const std::string&) {}
void ScriptingEngine::StopScript() {}
bool ScriptingEngine::IsScriptRunning() const { return false; }
std::optional<ExecutionResult> ScriptingEngine::GetAsyncResult() { return std::nullopt; }
std::string ScriptingEngine::GetPendingOutput() { return ""; }
void ScriptingEngine::QueueOutput(const std::string&) {}
void ScriptingEngine::QueuePlot(const CapturedPlot&) {}
std::vector<CapturedPlot> ScriptingEngine::GetPendingPlots() { return {}; }
void ScriptingEngine::ScriptWorker(const std::string&) {}
ExecutionResult ScriptingEngine::ExecuteWithStreaming(const std::string&) {
    ExecutionResult result;
    result.success = false;
    result.error_message = "Python support disabled";
    return result;
}
void ScriptingEngine::InitializeMatlabAliases() {}

} // namespace scripting

#endif // CYXWIZ_HAS_PYTHON
