#include "python_sandbox.h"
#include "python_engine.h"
#include <pybind11/pybind11.h>
#include <pybind11/eval.h>
#include <spdlog/spdlog.h>
#include <sstream>
#include <fstream>
#include <thread>
#include <future>
#include <filesystem>

namespace py = pybind11;

namespace scripting {

PythonSandbox::PythonSandbox()
    : initialized_(false)
    , monitoring_active_(false)
{
    spdlog::info("PythonSandbox created with default configuration");
}

PythonSandbox::PythonSandbox(const Config& config)
    : config_(config)
    , initialized_(false)
    , monitoring_active_(false)
{
    spdlog::info("PythonSandbox created with custom configuration");
    spdlog::info("  Timeout: {}s", config_.timeout.count());
    spdlog::info("  Max memory: {} MB", config_.max_memory_mb);
    spdlog::info("  Allowed modules: {}", config_.allowed_modules.size());
}

PythonSandbox::~PythonSandbox() {
    CleanupHooks();
}

void PythonSandbox::SetConfig(const Config& config) {
    config_ = config;
    spdlog::info("PythonSandbox configuration updated");
}

bool PythonSandbox::IsModuleAllowed(const std::string& module_name) const {
    // Check if module is in whitelist
    return config_.allowed_modules.find(module_name) != config_.allowed_modules.end();
}

bool PythonSandbox::IsBuiltinAllowed(const std::string& builtin_name) const {
    // Check if builtin is NOT in blocklist
    return config_.blocked_builtins.find(builtin_name) == config_.blocked_builtins.end();
}

bool PythonSandbox::IsPathAllowed(const std::string& path) const {
    if (config_.allowed_directory.empty()) {
        return false;  // No file access allowed
    }

    try {
        std::filesystem::path file_path(path);
        std::filesystem::path allowed_path(config_.allowed_directory);

        // Normalize paths
        file_path = std::filesystem::absolute(file_path);
        allowed_path = std::filesystem::absolute(allowed_path);

        // Check if file_path is within allowed_directory
        auto [root_end, nothing] = std::mismatch(
            allowed_path.begin(), allowed_path.end(),
            file_path.begin(), file_path.end()
        );

        return root_end == allowed_path.end();
    } catch (const std::exception& e) {
        spdlog::error("Path validation error: {}", e.what());
        return false;
    }
}

void PythonSandbox::SetupRestrictedBuiltins() {
    try {
        py::module_ builtins = py::module_::import("builtins");

        // Get original builtins dict
        py::dict original_builtins = builtins.attr("__dict__");

        // Create restricted builtins dict
        py::dict restricted_builtins;

        // Copy allowed builtins
        for (auto item : original_builtins) {
            std::string name = py::str(item.first);

            // Skip blocked builtins
            if (!IsBuiltinAllowed(name)) {
                spdlog::debug("Blocking builtin: {}", name);
                continue;
            }

            restricted_builtins[item.first] = item.second;
        }

        // Store restricted builtins in __main__ module
        py::module_ main = py::module_::import("__main__");
        main.attr("__builtins__") = restricted_builtins;

        spdlog::info("Restricted builtins configured ({} allowed, {} blocked)",
            restricted_builtins.size(), config_.blocked_builtins.size());

    } catch (const py::error_already_set& e) {
        spdlog::error("Failed to setup restricted builtins: {}", e.what());
    }
}

void PythonSandbox::SetupImportHook() {
    try {
        // Create custom import hook
        std::string import_hook_code = R"(
import sys
import builtins

class SandboxImportHook:
    def __init__(self, allowed_modules):
        self.allowed_modules = set(allowed_modules)
        self.original_import = builtins.__import__

    def __call__(self, name, *args, **kwargs):
        # Check if module or its parent is allowed
        module_parts = name.split('.')
        for i in range(len(module_parts)):
            partial_name = '.'.join(module_parts[:i+1])
            if partial_name in self.allowed_modules:
                return self.original_import(name, *args, **kwargs)

        # Module not allowed
        raise ImportError(f"Module '{name}' is not allowed in sandbox environment")

# Install the hook
allowed_modules = __ALLOWED_MODULES__
builtins.__import__ = SandboxImportHook(allowed_modules)
)";

        // Replace placeholder with actual allowed modules
        std::ostringstream modules_list;
        modules_list << "[";
        bool first = true;
        for (const auto& module : config_.allowed_modules) {
            if (!first) modules_list << ", ";
            modules_list << "'" << module << "'";
            first = false;
        }
        modules_list << "]";

        size_t pos = import_hook_code.find("__ALLOWED_MODULES__");
        if (pos != std::string::npos) {
            import_hook_code.replace(pos, 19, modules_list.str());
        }

        // Execute the hook setup
        py::exec(import_hook_code);

        spdlog::info("Import hook installed with {} allowed modules", config_.allowed_modules.size());

    } catch (const py::error_already_set& e) {
        spdlog::error("Failed to setup import hook: {}", e.what());
    }
}

void PythonSandbox::SetupFileAccessHook() {
    // TODO: Implement file access restrictions
    // This would require monkey-patching the 'open' builtin and 'os' module
    spdlog::debug("File access hook not yet implemented");
}

void PythonSandbox::CleanupHooks() {
    try {
        // Restore original __import__
        std::string cleanup_code = R"(
import builtins
import sys
if hasattr(builtins.__import__, 'original_import'):
    builtins.__import__ = builtins.__import__.original_import
)";
        py::exec(cleanup_code);

        spdlog::debug("Sandbox hooks cleaned up");
    } catch (const py::error_already_set& e) {
        spdlog::error("Failed to cleanup hooks: {}", e.what());
    }
}

bool PythonSandbox::ValidateCode(const std::string& code, std::string& error) {
    // Basic validation - check for obviously dangerous patterns
    std::vector<std::string> dangerous_patterns = {
        "os.system",
        "subprocess.",
        "eval(",
        "exec(",
        "__import__",
        "compile(",
    };

    for (const auto& pattern : dangerous_patterns) {
        if (code.find(pattern) != std::string::npos) {
            error = "Code contains dangerous pattern: " + pattern;
            spdlog::warn("Security violation: {}", error);
            return false;
        }
    }

    return true;
}

bool PythonSandbox::CheckASTForDangerousPatterns(const std::string& code, std::string& error) {
    try {
        // Use Python's AST module to analyze code structure
        py::module_ ast = py::module_::import("ast");

        // Parse the code into an AST
        py::object tree;
        try {
            tree = ast.attr("parse")(code);
        } catch (const py::error_already_set& e) {
            // Syntax error in code - let it through, Python will catch it later
            // AST check is for security, not syntax validation
            return true;
        }

        // Walk the AST and check for dangerous constructs
        py::object ast_walk = ast.attr("walk");
        py::object ast_Call = ast.attr("Call");
        py::object ast_Name = ast.attr("Name");
        py::object ast_Attribute = ast.attr("Attribute");

        for (auto node : ast_walk(tree)) {
            // Check for dangerous function calls
            if (py::isinstance(node, ast_Call)) {
                py::object func = node.attr("func");
                if (py::isinstance(func, ast_Name)) {
                    std::string func_id = py::str(func.attr("id"));
                    if (func_id == "exec" || func_id == "eval" ||
                        func_id == "compile" || func_id == "__import__") {
                        error = "Dangerous function call: " + func_id;
                        return false;
                    }
                }
            }

            // Check for private attribute access
            if (py::isinstance(node, ast_Attribute)) {
                std::string attr = py::str(node.attr("attr"));
                if (attr.length() >= 2 && attr.substr(0, 2) == "__") {
                    error = "Access to private attribute: " + attr;
                    return false;
                }
            }
        }

        return true;

    } catch (const py::error_already_set& e) {
        // If AST analysis fails for any reason, allow the code through
        // The actual execution will catch real errors
        spdlog::debug("AST analysis error (non-fatal): {}", e.what());
        return true;
    }
}

void PythonSandbox::StartMonitoring() {
    start_time_ = std::chrono::steady_clock::now();
    initial_memory_ = 0;  // TODO: Get actual memory usage
    monitoring_active_ = true;
}

void PythonSandbox::StopMonitoring() {
    monitoring_active_ = false;
}

bool PythonSandbox::CheckResourceLimits(ExecutionResult& result) {
    if (!monitoring_active_) return true;

    // Check timeout
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_);
    result.execution_time = elapsed;

    if (elapsed > config_.timeout) {
        result.timeout_exceeded = true;
        result.error_message = "Execution timeout exceeded (" +
            std::to_string(config_.timeout.count()) + "s)";
        spdlog::warn("Execution timeout: {} ms", elapsed.count());
        return false;
    }

    // TODO: Check memory limit (requires platform-specific code)

    return true;
}

PythonSandbox::ExecutionResult PythonSandbox::Execute(const std::string& code) {
    ExecutionResult result;
    result.success = false;

    // Validate code before execution
    std::string validation_error;
    if (!ValidateCode(code, validation_error)) {
        result.security_violation = true;
        result.violation_reason = validation_error;
        result.error_message = "Security violation: " + validation_error;
        return result;
    }

    // AST-based security check
    if (!CheckASTForDangerousPatterns(code, validation_error)) {
        result.security_violation = true;
        result.violation_reason = validation_error;
        result.error_message = "Security violation: " + validation_error;
        return result;
    }

    // Setup sandbox environment
    SetupRestrictedBuiltins();
    SetupImportHook();
    SetupFileAccessHook();

    // Start monitoring
    StartMonitoring();

    try {
        // Execute with timeout using std::async
        auto future = std::async(std::launch::async, [&]() {
            try {
                // Redirect stdout/stderr
                py::object sys = py::module_::import("sys");
                py::object io = py::module_::import("io");

                py::object stdout_capture = io.attr("StringIO")();
                py::object stderr_capture = io.attr("StringIO")();

                py::object original_stdout = sys.attr("stdout");
                py::object original_stderr = sys.attr("stderr");

                sys.attr("stdout") = stdout_capture;
                sys.attr("stderr") = stderr_capture;

                // Execute code
                py::exec(code);

                // Restore stdout/stderr
                sys.attr("stdout") = original_stdout;
                sys.attr("stderr") = original_stderr;

                // Get captured output
                result.output = py::str(stdout_capture.attr("getvalue")());
                std::string stderr_output = py::str(stderr_capture.attr("getvalue")());

                if (!stderr_output.empty()) {
                    result.output += "\nStderr: " + stderr_output;
                }

                result.success = true;

            } catch (const py::error_already_set& e) {
                result.error_message = e.what();
                result.success = false;
            }
        });

        // Wait for execution with timeout
        auto status = future.wait_for(config_.timeout);

        if (status == std::future_status::timeout) {
            result.timeout_exceeded = true;
            result.error_message = "Execution timeout exceeded (" +
                std::to_string(config_.timeout.count()) + "s)";
            result.success = false;
        }

    } catch (const std::exception& e) {
        result.error_message = std::string("Execution error: ") + e.what();
        result.success = false;
    }

    // Stop monitoring and check limits
    StopMonitoring();
    CheckResourceLimits(result);

    // Cleanup
    CleanupHooks();

    return result;
}

PythonSandbox::ExecutionResult PythonSandbox::ExecuteFile(const std::string& filepath) {
    ExecutionResult result;

    // Check if file access is allowed
    if (!IsPathAllowed(filepath)) {
        result.security_violation = true;
        result.violation_reason = "File access not allowed: " + filepath;
        result.error_message = result.violation_reason;
        return result;
    }

    // Read file
    try {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            result.error_message = "Failed to open file: " + filepath;
            return result;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string code = buffer.str();

        // Execute the code
        return Execute(code);

    } catch (const std::exception& e) {
        result.error_message = std::string("File read error: ") + e.what();
        return result;
    }
}

} // namespace scripting
