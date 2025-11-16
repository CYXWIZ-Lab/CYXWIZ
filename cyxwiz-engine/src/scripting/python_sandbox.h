#pragma once

#include <string>
#include <vector>
#include <unordered_set>
#include <chrono>
#include <memory>

namespace scripting {

/**
 * PythonSandbox - Secure Python execution environment
 *
 * Provides:
 * - Restricted builtins (blocks dangerous functions)
 * - Module whitelist (only approved imports)
 * - Execution timeout
 * - Memory and CPU tracking
 * - File system access restrictions
 */
class PythonSandbox {
public:
    struct Config {
        // Execution limits
        std::chrono::seconds timeout{60};           // Max execution time
        size_t max_memory_mb{1024};                 // Max memory usage (1GB)

        // Allowed modules
        std::unordered_set<std::string> allowed_modules{
            "math",
            "random",
            "json",
            "datetime",
            "collections",
            "itertools",
            "functools",
            "re",
            // CyxWiz modules
            "pycyxwiz",
            "cyxwiz_plotting"
        };

        // Blocked builtins (dangerous functions)
        std::unordered_set<std::string> blocked_builtins{
            "exec",
            "eval",
            "compile",
            "__import__",
            "open",
            "input",
            "breakpoint",
            "exit",
            "quit"
        };

        // File access restrictions
        bool allow_file_read{false};
        bool allow_file_write{false};
        std::string allowed_directory{""};  // If empty, no file access
    };

    PythonSandbox();
    explicit PythonSandbox(const Config& config);
    ~PythonSandbox();

    // Execute code in sandbox
    struct ExecutionResult {
        bool success;
        std::string output;
        std::string error_message;

        // Resource usage
        std::chrono::milliseconds execution_time{0};
        size_t peak_memory_bytes{0};

        // Security violations
        bool timeout_exceeded{false};
        bool memory_exceeded{false};
        bool security_violation{false};
        std::string violation_reason;
    };

    ExecutionResult Execute(const std::string& code);
    ExecutionResult ExecuteFile(const std::string& filepath);

    // Configuration
    void SetConfig(const Config& config);
    const Config& GetConfig() const { return config_; }

    // Security checks
    bool IsModuleAllowed(const std::string& module_name) const;
    bool IsBuiltinAllowed(const std::string& builtin_name) const;
    bool IsPathAllowed(const std::string& path) const;

private:
    // Setup sandbox environment
    void SetupRestrictedBuiltins();
    void SetupImportHook();
    void SetupFileAccessHook();
    void CleanupHooks();

    // Execution monitoring
    void StartMonitoring();
    void StopMonitoring();
    bool CheckResourceLimits(ExecutionResult& result);

    // Security validation
    bool ValidateCode(const std::string& code, std::string& error);
    bool CheckASTForDangerousPatterns(const std::string& code, std::string& error);

    // Data
    Config config_;
    bool initialized_;

    // Monitoring state
    std::chrono::steady_clock::time_point start_time_;
    size_t initial_memory_;
    bool monitoring_active_;
};

} // namespace scripting
