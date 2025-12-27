#include "python_sandbox.h"
#include <spdlog/spdlog.h>

#ifdef CYXWIZ_HAS_PYTHON

#include "python_engine.h"
#include <pybind11/pybind11.h>
#include <pybind11/eval.h>
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
    return config_.allowed_modules.find(module_name) != config_.allowed_modules.end();
}

bool PythonSandbox::IsBuiltinAllowed(const std::string& builtin_name) const {
    return config_.blocked_builtins.find(builtin_name) == config_.blocked_builtins.end();
}

bool PythonSandbox::IsPathAllowed(const std::string& path) const {
    if (config_.allowed_directory.empty()) {
        return false;
    }

    try {
        std::filesystem::path file_path(path);
        std::filesystem::path allowed_path(config_.allowed_directory);
        file_path = std::filesystem::absolute(file_path);
        allowed_path = std::filesystem::absolute(allowed_path);
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
        py::dict original_builtins = builtins.attr("__dict__");
        py::dict restricted_builtins;

        for (auto item : original_builtins) {
            std::string name = py::str(item.first);
            if (!IsBuiltinAllowed(name)) {
                spdlog::debug("Blocking builtin: {}", name);
                continue;
            }
            restricted_builtins[item.first] = item.second;
        }

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
        std::string import_hook_code = R"(
import sys
import builtins

class SandboxImportHook:
    def __init__(self, allowed_modules):
        self.allowed_modules = set(allowed_modules)
        self.original_import = builtins.__import__

    def __call__(self, name, *args, **kwargs):
        if name in sys.modules:
            return self.original_import(name, *args, **kwargs)
        module_parts = name.split('.')
        for i in range(len(module_parts)):
            partial_name = '.'.join(module_parts[:i+1])
            if partial_name in self.allowed_modules:
                return self.original_import(name, *args, **kwargs)
        raise ImportError(f"Module '{name}' is not allowed in sandbox environment")

allowed_modules = __ALLOWED_MODULES__
builtins.__import__ = SandboxImportHook(allowed_modules)
)";

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

        py::exec(import_hook_code);
        spdlog::info("Import hook installed with {} allowed modules", config_.allowed_modules.size());
    } catch (const py::error_already_set& e) {
        spdlog::error("Failed to setup import hook: {}", e.what());
    }
}

void PythonSandbox::SetupFileAccessHook() {
    spdlog::debug("File access hook not yet implemented");
}

void PythonSandbox::CleanupHooks() {
    try {
        py::module_ builtins = py::module_::import("builtins");
        py::object current_import = builtins.attr("__import__");
        if (py::hasattr(current_import, "original_import")) {
            py::object original_import = current_import.attr("original_import");
            builtins.attr("__import__") = original_import;
            spdlog::debug("Sandbox import hook cleaned up");
        }
    } catch (const py::error_already_set& e) {
        spdlog::error("Failed to cleanup hooks: {}", e.what());
    }
}

bool PythonSandbox::ValidateCode(const std::string& code, std::string& error) {
    std::vector<std::string> dangerous_patterns = {
        "os.system", "subprocess.", "eval(", "exec(", "__import__", "compile("
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
        py::module_ ast = py::module_::import("ast");
        py::object tree;
        try {
            tree = ast.attr("parse")(code);
        } catch (const py::error_already_set&) {
            return true;
        }

        py::object ast_walk = ast.attr("walk");
        py::object ast_Call = ast.attr("Call");
        py::object ast_Name = ast.attr("Name");
        py::object ast_Attribute = ast.attr("Attribute");

        for (auto node : ast_walk(tree)) {
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
        spdlog::debug("AST analysis error (non-fatal): {}", e.what());
        return true;
    }
}

void PythonSandbox::StartMonitoring() {
    start_time_ = std::chrono::steady_clock::now();
    initial_memory_ = 0;
    monitoring_active_ = true;
}

void PythonSandbox::StopMonitoring() {
    monitoring_active_ = false;
}

bool PythonSandbox::CheckResourceLimits(ExecutionResult& result) {
    if (!monitoring_active_) return true;
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
    return true;
}

PythonSandbox::ExecutionResult PythonSandbox::Execute(const std::string& code) {
    ExecutionResult result;
    result.success = false;

    std::string validation_error;
    if (!ValidateCode(code, validation_error)) {
        result.security_violation = true;
        result.violation_reason = validation_error;
        result.error_message = "Security violation: " + validation_error;
        return result;
    }

    if (!CheckASTForDangerousPatterns(code, validation_error)) {
        result.security_violation = true;
        result.violation_reason = validation_error;
        result.error_message = "Security violation: " + validation_error;
        return result;
    }

    SetupRestrictedBuiltins();
    SetupImportHook();
    SetupFileAccessHook();
    StartMonitoring();

    try {
        py::object sys = py::module_::import("sys");
        py::object io = py::module_::import("io");
        py::object stdout_capture = io.attr("StringIO")();
        py::object stderr_capture = io.attr("StringIO")();
        py::object original_stdout = sys.attr("stdout");
        py::object original_stderr = sys.attr("stderr");
        sys.attr("stdout") = stdout_capture;
        sys.attr("stderr") = stderr_capture;

        py::exec(code);

        sys.attr("stdout") = original_stdout;
        sys.attr("stderr") = original_stderr;
        result.output = py::str(stdout_capture.attr("getvalue")());
        std::string stderr_output = py::str(stderr_capture.attr("getvalue")());
        if (!stderr_output.empty()) {
            result.output += "\nStderr: " + stderr_output;
        }
        result.success = true;
    } catch (const py::error_already_set& e) {
        result.error_message = e.what();
        result.success = false;
    } catch (const std::exception& e) {
        result.error_message = std::string("Execution error: ") + e.what();
        result.success = false;
    }

    StopMonitoring();
    CheckResourceLimits(result);
    CleanupHooks();
    return result;
}

PythonSandbox::ExecutionResult PythonSandbox::ExecuteFile(const std::string& filepath) {
    ExecutionResult result;

    if (!IsPathAllowed(filepath)) {
        result.security_violation = true;
        result.violation_reason = "File access not allowed: " + filepath;
        result.error_message = result.violation_reason;
        return result;
    }

    try {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            result.error_message = "Failed to open file: " + filepath;
            return result;
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        return Execute(buffer.str());
    } catch (const std::exception& e) {
        result.error_message = std::string("File read error: ") + e.what();
        return result;
    }
}

} // namespace scripting

#else // !CYXWIZ_HAS_PYTHON

// Stub implementations when Python is disabled
namespace scripting {

PythonSandbox::PythonSandbox() : initialized_(false), monitoring_active_(false) {}
PythonSandbox::PythonSandbox(const Config& config) : config_(config), initialized_(false), monitoring_active_(false) {}
PythonSandbox::~PythonSandbox() {}

void PythonSandbox::SetConfig(const Config& config) { config_ = config; }
bool PythonSandbox::IsModuleAllowed(const std::string&) const { return false; }
bool PythonSandbox::IsBuiltinAllowed(const std::string&) const { return false; }
bool PythonSandbox::IsPathAllowed(const std::string&) const { return false; }

void PythonSandbox::SetupRestrictedBuiltins() {}
void PythonSandbox::SetupImportHook() {}
void PythonSandbox::SetupFileAccessHook() {}
void PythonSandbox::CleanupHooks() {}

bool PythonSandbox::ValidateCode(const std::string&, std::string& error) {
    error = "Python support disabled";
    return false;
}

bool PythonSandbox::CheckASTForDangerousPatterns(const std::string&, std::string& error) {
    error = "Python support disabled";
    return false;
}

void PythonSandbox::StartMonitoring() {}
void PythonSandbox::StopMonitoring() {}
bool PythonSandbox::CheckResourceLimits(ExecutionResult&) { return false; }

PythonSandbox::ExecutionResult PythonSandbox::Execute(const std::string&) {
    ExecutionResult result;
    result.success = false;
    result.error_message = "Python support disabled";
    return result;
}

PythonSandbox::ExecutionResult PythonSandbox::ExecuteFile(const std::string&) {
    ExecutionResult result;
    result.success = false;
    result.error_message = "Python support disabled";
    return result;
}

} // namespace scripting

#endif // CYXWIZ_HAS_PYTHON
