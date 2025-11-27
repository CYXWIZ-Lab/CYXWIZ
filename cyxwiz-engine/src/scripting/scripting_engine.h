#pragma once

#include "python_engine.h"
#include "python_sandbox.h"
#include <string>
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

    // ========== Async execution state ==========
    std::unique_ptr<std::thread> script_thread_;
    std::atomic<bool> script_running_{false};
    std::atomic<bool> cancel_requested_{false};

    // Thread-safe output queue
    std::mutex output_mutex_;
    std::queue<std::string> output_queue_;

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

    // Shared cancellation flag - accessible from Python without GIL
    static std::atomic<int> shared_cancel_flag_;

    // Python thread ID for async exception injection
    std::atomic<unsigned long> python_thread_id_{0};
public:
    // Static method for Python to check cancellation (no GIL needed)
    static int GetCancelFlag() { return shared_cancel_flag_.load(); }
    static void SetCancelFlag(int val) { shared_cancel_flag_.store(val); }
};

} // namespace scripting
