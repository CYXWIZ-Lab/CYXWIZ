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

        // Allowed modules (Python standard library + CyxWiz modules)
        std::unordered_set<std::string> allowed_modules{
            // Core Python (required for sandbox cleanup)
            "builtins",
            "sys",
            "os",
            "io",
            "csv",
            "time",

            // Math and data
            "math",
            "random",
            "statistics",

            // Data structures
            "json",
            "datetime",
            "collections",
            "itertools",
            "functools",

            // Text processing
            "re",
            "string",
            "textwrap",  // Used by Python's traceback formatting
            "enum",      // Used by re module
            "copyreg",   // Used by pickle/copy
            "types",     // Used internally by many modules
            "_sre",      // C extension for re module
            "sre_compile",  // re internals
            "sre_parse",    // re internals
            "sre_constants", // re internals
            "_constants",    // Python 3.14+ re internals

            // File formats (safe reading)
            "pathlib",
            "tempfile",

            // Optional: Scientific computing (if installed)
            "numpy",
            "pandas",
            "matplotlib",
            "scipy",

            // Data analysis (DuckDB + Polars)
            "polars",
            "polars.functions",
            "polars.datatypes",
            "polars.io",
            "polars.lazy",
            "duckdb",
            "pyarrow",
            "pyarrow.parquet",
            "pyarrow.csv",
            "pyarrow.json",

            // CyxWiz modules
            "pycyxwiz",
            "cyxwiz_plotting"
        };

        // Blocked builtins (dangerous functions)
        // Note: 'open' is NOT blocked - we use allow_file_read/write to control it
        // Note: '__import__' is NOT blocked - we need it for our import hook to work
        //       The import hook itself controls which modules can be imported
        std::unordered_set<std::string> blocked_builtins{
            "exec",
            "eval",
            "compile",
            // "__import__",  // Removed - needed for import hook to function
            // "open",        // Removed - allow file reading with restrictions
            "input",
            "breakpoint",
            "exit",
            "quit"
        };

        // File access restrictions
        // Default: Allow file reading (safe for templates), deny writing
        bool allow_file_read{true};   // Allow reading files
        bool allow_file_write{false};  // Deny writing files (security)
        std::string allowed_directory{""};  // Empty = current directory
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
    void SetupTimeoutWatchdog();
    void RemoveTimeoutWatchdog();
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
