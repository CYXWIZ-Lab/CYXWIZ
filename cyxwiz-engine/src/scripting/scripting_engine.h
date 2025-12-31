#pragma once

#include "python_engine.h"
#include "python_sandbox.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>

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

    // Captured matplotlib/plotting figures
    std::vector<CapturedPlot> plots;
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
    // Execute single command (REPL-style)
    ExecutionResult ExecuteCommand(const std::string& command);

    // Execute multi-line script
    ExecutionResult ExecuteScript(const std::string& script);

    // Execute script file
    ExecutionResult ExecuteFile(const std::string& filepath);

    // ========== Asynchronous Execution (non-blocking) ==========
    // Start script execution in background thread
    // Returns immediately, script runs in background
    void ExecuteScriptAsync(const std::string& script);

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

    // Completion callback (called when async script finishes)
    using CompletionCallback = std::function<void(const ExecutionResult&)>;
    void SetCompletionCallback(CompletionCallback callback);

    // ========== Output & Configuration ==========
    // Output callback (for real-time output streaming)
    using OutputCallback = std::function<void(const std::string&)>;
    void SetOutputCallback(OutputCallback callback);

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

    // Register Training Dashboard with Python module
    void RegisterTrainingDashboard(cyxwiz::TrainingPlotPanel* panel);

private:
    std::unique_ptr<PythonEngine> python_engine_;
    std::unique_ptr<PythonSandbox> sandbox_;
    OutputCallback output_callback_;
    CompletionCallback completion_callback_;
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
    void ScriptWorker(const std::string& script);

    // Internal execution with output streaming
    ExecutionResult ExecuteWithStreaming(const std::string& script);

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

    // Console command execution with timeout
    std::mutex command_mutex_;
    std::condition_variable command_cv_;
    std::atomic<bool> command_finished_{false};
    ExecutionResult command_result_;
    ExecutionResult ExecuteCommandDirect(const std::string& command);
    void ExecuteCommandWorker(const std::string& command);

public:
    // Static method for Python to check cancellation (no GIL needed)
    static int GetCancelFlag() { return shared_cancel_flag_.load(); }
    static void SetCancelFlag(int val) { shared_cancel_flag_.store(val); }
};

} // namespace scripting
