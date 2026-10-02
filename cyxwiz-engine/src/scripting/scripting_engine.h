#pragma once

#include "language_service.h"
#include "python_engine.h"
#include "python_sandbox.h"
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <chrono>

// Forward declarations
namespace cyxwiz {
    class TrainingPlotPanel;
}

namespace scripting {

/**
 * Captured plot/image from matplotlib or other plotting libraries
 */
struct CapturedPlot {
    std::vector<unsigned char> png_data;  // PNG image data
    int width = 0;
    int height = 0;
    std::string label;  // Optional label (e.g., figure title)
};

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

    // Async execution info
    bool was_cancelled{false};

    // Interactive (REPL) detail: stderr on success (warnings), the exception
    // type and the formatted traceback with the user's own frames.
    std::string stderr_output;
    std::string exception_type;
    std::string traceback;

    // Captured matplotlib/plotting figures
    std::vector<CapturedPlot> plots;

    // Notebook cells (TOFIX133 P4): repr of the last expression's value
    // (empty when it was None or the cell ended with a statement), and the
    // exception's frames, innermost last.
    std::string result_repr;
    std::string result_html;   // the value's _repr_html_ (pandas tables), when it has one
    std::string exception_value;
    struct Frame {
        std::string file;
        int line = 0;
        std::string function;
        std::string code;
        std::string cause;     // set on the first frame of a chained exception ("URLError: ...")
    };
    std::vector<Frame> frames;
};

/**
 * ScriptingEngine - High-level wrapper around PythonEngine
 * Adds output capture, error handling, sandbox security, and async execution
 */
class ScriptingEngine {
public:
    ScriptingEngine();
    ~ScriptingEngine();

    // ========== Synchronous Execution (blocks caller) ==========
    // Execute single command (REPL-style) - WARNING: blocks for timeout duration
    ExecutionResult ExecuteCommand(const std::string& command);

    // ========== Async Command Execution (for console commands) ==========
    // Start command execution in background (non-blocking)
    void ExecuteCommandAsync(const std::string& command);

    // Check if a console command is currently running
    bool IsCommandRunning() const { return command_running_.load(); }

    // Stop currently running console command
    void StopCommand();

    // Get result of async command (if finished)
    std::optional<ExecutionResult> GetCommandResult();

    // Execute multi-line script
    ExecutionResult ExecuteScript(const std::string& script);

    // Execute script file
    ExecutionResult ExecuteFile(const std::string& filepath);

    // ========== Asynchronous Execution (non-blocking) ==========
    using CompletionCallback = std::function<void(const ExecutionResult&)>;
    using OutputCallback = std::function<void(const std::string&)>;
    // Callbacks for one run only (TOFIX133 P0 item 9: they used to be global
    // and never cleared, so output went to whichever cell or panel set them
    // last). Both run on the worker thread; on_complete also runs at once if
    // Python cannot start.
    struct RunCallbacks {
        OutputCallback on_output;
        CompletionCallback on_complete;
        // Notebook cells (TOFIX133 P4, decision D4). Set notebook_namespace
        // to run in that notebook's own namespace with the last expression
        // echoed; empty runs in __main__ as before. on_stderr empty: stderr
        // goes to on_output.
        OutputCallback on_stderr;
        std::string notebook_namespace;
        std::string cell_filename;  // shown in tracebacks, e.g. "Cell In[3]"
        int execution_count = 0;    // the value is kept as Out[n] in the namespace
    };
    // Starts the script in a background thread and returns at once. Returns
    // false when another script is still running (nothing is started and no
    // callback is called).
    bool ExecuteScriptAsync(const std::string& script, RunCallbacks callbacks = {});

    // Restart for a notebook: forgets the namespace its cells ran in.
    // Returns false (nothing done) while a script is running.
    bool DropNotebookNamespace(const std::string& key);

    // Script Editor language intelligence (TOFIX133 P3). CallLanguageTool runs
    // one function of python_tools/cyxwiz_intel.py with JSON keyword arguments
    // (and a notebook's namespace when given) under the GIL, on the caller's
    // thread; it returns the result as JSON, or "" when Python is not running
    // or the tools are missing. Language() runs them on a worker thread.
    std::string CallLanguageTool(const std::string& function, const std::string& args_json,
                                 const std::string& notebook_namespace = {});
    LanguageService& Language();
    // Starts Python now (UI thread), as a first run would; false with the reason.
    bool StartPython(std::string* error = nullptr) { return EnsurePythonInitialized(error); }
    // Whether the bundled tools loaded ("" until tried; else the reason).
    std::string LanguageToolsError() const;

    // Writes Out[count] of a notebook (a pandas DataFrame or Series) to a CSV
    // file, for the Table Viewer. False with a reason while a script runs, after
    // Restart, or when the value is not a table.
    bool ExportNotebookValueToCsv(const std::string& key, int count, const std::string& path, std::string* error);
    // The same for a variable of the notebook (a table in its Variables panel).
    bool ExportNotebookVariableToCsv(const std::string& key, const std::string& name, const std::string& path,
                                     std::string* error);

    // A notebook's variables as JSON [{name, type, size, value, table}],
    // summarised with reprlib (cheap, under the GIL on the caller's thread).
    // False while a script or command runs.
    bool NotebookVariablesJson(const std::string& key, std::string* json);

    // Stop currently running script
    // Sends interrupt signal to Python interpreter
    void StopScript();

    // Check if a script is currently running
    bool IsScriptRunning() const;

    // Get the result of the last async execution (if finished)
    // Returns nullopt if still running or no async execution started
    std::optional<ExecutionResult> GetAsyncResult();

    // Get any pending output from the running script
    // Call this periodically from GUI to get real-time output
    std::string GetPendingOutput();

    // ========== Output & Configuration ==========
    // Sandbox configuration
    void EnableSandbox(bool enable);
    bool IsSandboxEnabled() const { return sandbox_enabled_; }
    void SetSandboxConfig(const PythonSandbox::Config& config);
    PythonSandbox::Config GetSandboxConfig() const;

    // Verbose logging (includes internal Variable Explorer commands)
    void SetVerboseLogging(bool enable) { verbose_logging_ = enable; }
    bool IsVerboseLogging() const { return verbose_logging_; }
    bool* GetVerboseLoggingPtr() { return &verbose_logging_; }

    // Console timeout configuration (for interactive commands)
    void SetConsoleTimeout(double seconds) { console_timeout_seconds_ = seconds; }
    double GetConsoleTimeout() const { return console_timeout_seconds_; }
    double* GetConsoleTimeoutPtr() { return &console_timeout_seconds_; }

    // Check if engine is initialized
    bool IsInitialized() const;

    // Runtime diagnostics for current Python interpreter
    std::string GetPythonRuntimeDiagnostics();

    // Reload interpreter config for active project (picks up python_env.json)
    bool ReloadPythonForProject();

    // ---- Python REPL support (tofix121) ----
    struct InterpreterInfo {
        bool initialized = false;
        std::string version;           // "3.12.14" once started
        std::string interpreter_path;  // running, else the one the next start uses
        std::string source;            // "project" / "system" when running
        std::string mismatch;          // non-empty: restart needed
    };
    // Cheap enough for UI refreshes; the preview path is resolved without logging.
    InterpreterInfo GetInterpreterInfo();
    // Names completing `text` (identifiers, dotted attributes) from the REPL
    // namespace, using rlcompleter directly (no code built from user text).
    // Empty while a command or script runs or before Python starts.
    std::vector<std::string> CompleteSync(const std::string& text, size_t max_results = 50);
    // Clears the REPL's variables and imports (the embedded interpreter keeps
    // running; it cannot be restarted inside the Engine process).
    bool ResetSession(std::string* error_out = nullptr);
    // Last message from a failed Python start, empty when none.
    std::string GetLastInitError() const;

    // Register Training Dashboard with Python module (deferred - stores panel pointer)
    void RegisterTrainingDashboard(cyxwiz::TrainingPlotPanel* panel);
    // Actually register with Python (called lazily when scripts run)
    void EnsureTrainingDashboardRegistered();

private:
    mutable std::mutex init_error_mutex_;
    std::string last_init_error_;
    std::string cached_python_version_;
    cyxwiz::TrainingPlotPanel* training_plot_panel_{nullptr};
    bool training_dashboard_registered_{false};
    std::unique_ptr<PythonEngine> python_engine_;
    std::unique_ptr<LanguageService> language_;  // stopped before Python ends
    mutable std::mutex language_mutex_;
    std::string language_error_;
    std::unique_ptr<PythonSandbox> sandbox_;
    bool sandbox_enabled_;
    bool verbose_logging_{false};  // Log all commands including internal ones
    double console_timeout_seconds_{30.0};  // Console command timeout (default 30s)

    // ========== Async execution state ==========
    std::unique_ptr<std::thread> script_thread_;
    std::atomic<bool> script_running_{false};
    std::atomic<bool> cancel_requested_{false};

    // Thread-safe output queue
    std::mutex output_mutex_;
    std::queue<std::string> output_queue_;

    // Thread-safe plot queue
    std::mutex plot_mutex_;
    std::vector<CapturedPlot> plot_queue_;

    // Result storage
    std::mutex result_mutex_;
    std::optional<ExecutionResult> async_result_;

    // Worker thread function
    void ScriptWorker(const std::string& script, RunCallbacks callbacks);

    // Internal execution with output streaming
    ExecutionResult ExecuteWithStreaming(const std::string& script, const RunCallbacks& callbacks);

    // Convert sandbox result to engine result
    ExecutionResult ConvertSandboxResult(const PythonSandbox::ExecutionResult& sandbox_result);

    // Queue output for async retrieval
    void QueueOutput(const std::string& output);

    // Queue plot for async retrieval
    void QueuePlot(const CapturedPlot& plot);

    // Get pending plots and clear queue
    std::vector<CapturedPlot> GetPendingPlots();

    // Shared cancellation flag - accessible from Python without GIL
    static std::atomic<int> shared_cancel_flag_;

    // Python thread ID for async exception injection
    std::atomic<unsigned long> python_thread_id_{0};

    // MATLAB-style aliases initialization
    bool matlab_aliases_initialized_{false};
    void InitializeMatlabAliases();

    // Ensure Python is initialized before use
    bool EnsurePythonInitialized(std::string* error_out = nullptr);

    // Console command execution with timeout
    std::mutex command_mutex_;
    std::condition_variable command_cv_;
    std::atomic<bool> command_finished_{false};
    ExecutionResult command_result_;
    ExecutionResult ExecuteCommandDirect(const std::string& command, bool suppress_output_callback);
    ExecutionResult ExecuteCommandWithPythonTimeout(const std::string& command);
    void ExecuteCommandWorker(const std::string& command);

    // Async command execution (for non-blocking console)
    std::unique_ptr<std::thread> command_thread_;
    std::atomic<bool> command_running_{false};
    std::atomic<bool> command_stop_requested_{false};
    std::mutex command_result_mutex_;
    std::optional<ExecutionResult> async_command_result_;
    void CommandAsyncWorker(const std::string& command);

    // Post-command cooldown tracking (to prevent racing with Python cleanup)
    std::chrono::steady_clock::time_point last_command_end_time_;
    std::atomic<bool> last_command_had_error_{false};
    std::atomic<bool> python_busy_{false};  // True while any Python operation is in progress
    static constexpr int POST_ERROR_COOLDOWN_MS = 500;  // Wait 500ms after error (increased)

public:
    // Static method for Python to check cancellation (no GIL needed)
    static int GetCancelFlag() { return shared_cancel_flag_.load(); }
    static void SetCancelFlag(int val) { shared_cancel_flag_.store(val); }

    // Check if it's safe to run commands (no async running, past cooldown period)
    bool IsSafeForNewCommand() const;
};

} // namespace scripting
