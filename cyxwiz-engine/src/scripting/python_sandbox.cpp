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

// Platform-specific includes for memory monitoring
#ifdef _WIN32
    #include <windows.h>
    #include <psapi.h>
#else
    #include <sys/resource.h>
    #include <unistd.h>
    #if defined(__APPLE__)
        #include <mach/mach.h>
    #endif
#endif

namespace py = pybind11;

namespace {

/**
 * Get current process memory usage in bytes
 * Cross-platform implementation
 */
size_t GetCurrentMemoryUsage() {
#ifdef _WIN32
    // Windows: Use GetProcessMemoryInfo
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;  // Physical memory currently in use
    }
    return 0;
#elif defined(__APPLE__)
    // macOS: Use mach API
    struct mach_task_basic_info info;
    mach_msg_type_number_t size = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &size) == KERN_SUCCESS) {
        return info.resident_size;
    }
    return 0;
#else
    // Linux: Read from /proc/self/status
    std::ifstream status_file("/proc/self/status");
    std::string line;
    while (std::getline(status_file, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            // Parse VmRSS value (in kB)
            size_t kb = 0;
            std::istringstream iss(line.substr(6));
            iss >> kb;
            return kb * 1024;  // Convert to bytes
        }
    }
    return 0;
#endif
}

} // anonymous namespace

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
    // Skip CleanupHooks during shutdown - it calls Python code which can crash
    // when Python interpreter is in an unstable state during process exit.
    // Since we're not calling py::finalize_interpreter(), OS cleanup will
    // handle everything anyway.
    spdlog::info("~PythonSandbox: skipping CleanupHooks (OS will cleanup)");
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
        # Allow empty module names (used by Python internals for relative imports)
        if not name:
            return self.original_import(name, *args, **kwargs)

        # Debug: Log import attempt
        print(f"[SANDBOX] Import request: {name}")

        # Check if module is already imported (avoids re-checking)
        if name in sys.modules:
            print(f"[SANDBOX] {name} already in sys.modules, allowing")
            return self.original_import(name, *args, **kwargs)

        # Check if module or its parent is allowed
        module_parts = name.split('.')
        for i in range(len(module_parts)):
            partial_name = '.'.join(module_parts[:i+1])
            if partial_name in self.allowed_modules:
                print(f"[SANDBOX] {name} matched whitelist as {partial_name}, allowing")
                return self.original_import(name, *args, **kwargs)

        # Module not allowed
        print(f"[SANDBOX] {name} NOT in whitelist, blocking")
        print(f"[SANDBOX] Whitelist has {len(self.allowed_modules)} modules")
        raise ImportError(f"Module '{name}' is not allowed in sandbox environment")

# Install the hook
allowed_modules = __ALLOWED_MODULES__
print(f"[SANDBOX] Installing import hook with {len(allowed_modules)} allowed modules:")
print(f"[SANDBOX] Allowed: {sorted(allowed_modules)[:10]}...")  # Show first 10
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

        // Debug: Log the allowed modules
        std::ostringstream debug_modules;
        for (const auto& module : config_.allowed_modules) {
            debug_modules << module << " ";
        }
        spdlog::debug("Allowed modules: {}", debug_modules.str());

    } catch (const py::error_already_set& e) {
        spdlog::error("Failed to setup import hook: {}", e.what());
    }
}

void PythonSandbox::SetupFileAccessHook() {
    try {
        // Build configuration for the hook
        std::string allowed_dir_escaped = config_.allowed_directory;
        // Escape backslashes for Python string
        size_t pos = 0;
        while ((pos = allowed_dir_escaped.find("\\", pos)) != std::string::npos) {
            allowed_dir_escaped.replace(pos, 1, "\\\\");
            pos += 2;
        }

        std::string file_hook_code = R"(
import builtins
import os
import os.path

class SandboxFileHook:
    def __init__(self, original_open, allow_read, allow_write, allowed_dir):
        self.original_open = original_open
        self.allow_read = allow_read
        self.allow_write = allow_write
        self.allowed_dir = allowed_dir if allowed_dir else None

    def is_path_allowed(self, filepath):
        """Check if path is within allowed directory"""
        if self.allowed_dir is None:
            return True  # No restriction on directory
        try:
            abs_path = os.path.abspath(filepath)
            abs_allowed = os.path.abspath(self.allowed_dir)
            # Use commonpath to check containment
            return os.path.commonpath([abs_path, abs_allowed]) == abs_allowed
        except (ValueError, OSError):
            return False

    def is_write_mode(self, mode):
        """Check if mode involves writing"""
        write_chars = {'w', 'a', 'x', '+'}
        return any(c in mode for c in write_chars)

    def __call__(self, file, mode='r', *args, **kwargs):
        filepath = str(file)

        # Check path restriction
        if not self.is_path_allowed(filepath):
            raise PermissionError(f"[Sandbox] Access denied: '{filepath}' is outside allowed directory")

        # Check read permission
        if 'r' in mode and not self.allow_read:
            raise PermissionError(f"[Sandbox] File reading is disabled")

        # Check write permission
        if self.is_write_mode(mode) and not self.allow_write:
            raise PermissionError(f"[Sandbox] File writing is disabled")

        return self.original_open(file, mode, *args, **kwargs)

# Store original open
_sandbox_original_open = builtins.open

# Configuration from C++
_sandbox_allow_read = __ALLOW_READ__
_sandbox_allow_write = __ALLOW_WRITE__
_sandbox_allowed_dir = __ALLOWED_DIR__

# Install hook
builtins.open = SandboxFileHook(_sandbox_original_open, _sandbox_allow_read, _sandbox_allow_write, _sandbox_allowed_dir)

# Also restrict os module file operations
if hasattr(os, 'remove') and not _sandbox_allow_write:
    _sandbox_original_remove = os.remove
    def _sandbox_remove(path):
        raise PermissionError(f"[Sandbox] File deletion is disabled")
    os.remove = _sandbox_remove

if hasattr(os, 'unlink') and not _sandbox_allow_write:
    _sandbox_original_unlink = os.unlink
    def _sandbox_unlink(path):
        raise PermissionError(f"[Sandbox] File deletion is disabled")
    os.unlink = _sandbox_unlink

if hasattr(os, 'rename') and not _sandbox_allow_write:
    _sandbox_original_rename = os.rename
    def _sandbox_rename(src, dst):
        raise PermissionError(f"[Sandbox] File renaming is disabled")
    os.rename = _sandbox_rename

if hasattr(os, 'mkdir') and not _sandbox_allow_write:
    _sandbox_original_mkdir = os.mkdir
    def _sandbox_mkdir(path, *args, **kwargs):
        raise PermissionError(f"[Sandbox] Directory creation is disabled")
    os.mkdir = _sandbox_mkdir

if hasattr(os, 'makedirs') and not _sandbox_allow_write:
    _sandbox_original_makedirs = os.makedirs
    def _sandbox_makedirs(path, *args, **kwargs):
        raise PermissionError(f"[Sandbox] Directory creation is disabled")
    os.makedirs = _sandbox_makedirs

if hasattr(os, 'rmdir') and not _sandbox_allow_write:
    _sandbox_original_rmdir = os.rmdir
    def _sandbox_rmdir(path):
        raise PermissionError(f"[Sandbox] Directory deletion is disabled")
    os.rmdir = _sandbox_rmdir
)";

        // Replace placeholders with actual values
        std::string allow_read_str = config_.allow_file_read ? "True" : "False";
        std::string allow_write_str = config_.allow_file_write ? "True" : "False";
        std::string allowed_dir_str = allowed_dir_escaped.empty() ? "None" : "'" + allowed_dir_escaped + "'";

        pos = file_hook_code.find("__ALLOW_READ__");
        if (pos != std::string::npos) {
            file_hook_code.replace(pos, 14, allow_read_str);
        }

        pos = file_hook_code.find("__ALLOW_WRITE__");
        if (pos != std::string::npos) {
            file_hook_code.replace(pos, 15, allow_write_str);
        }

        pos = file_hook_code.find("__ALLOWED_DIR__");
        if (pos != std::string::npos) {
            file_hook_code.replace(pos, 15, allowed_dir_str);
        }

        // Execute the hook setup
        py::exec(file_hook_code);

        spdlog::info("File access hook installed (read={}, write={}, dir={})",
            config_.allow_file_read, config_.allow_file_write,
            config_.allowed_directory.empty() ? "<any>" : config_.allowed_directory);

    } catch (const py::error_already_set& e) {
        spdlog::error("Failed to setup file access hook: {}", e.what());
    }
}

void PythonSandbox::SetupTimeoutWatchdog() {
    try {
        // Cross-platform timeout implementation:
        // - Unix (Linux/macOS): Use signal.alarm (SIGALRM) for robust timeout
        //   that can interrupt C extensions like numpy
        // - Windows: Use sys.settrace (checks elapsed time on each line)
        // - Fallback: sys.settrace works everywhere but can't interrupt C code

        long timeout_ms = std::chrono::duration_cast<std::chrono::milliseconds>(config_.timeout).count();

        std::string timeout_code = R"PY(
import sys
import time

class _SandboxTimeoutWatchdog:
    """
    Cross-platform timeout watchdog.

    On Unix: Uses signal.alarm (SIGALRM) which can interrupt C extensions
    On Windows: Uses sys.settrace which only works for pure Python code
    """

    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
        self.original_trace = None
        self.original_alarm_handler = None
        self.timed_out = False
        self._check_count = 0
        self._check_interval = 100  # Check every N trace calls for performance
        self._use_signal = False

        # Check if signal.alarm is available (Unix only)
        try:
            import signal
            if hasattr(signal, 'SIGALRM') and hasattr(signal, 'alarm'):
                self._use_signal = True
                self._signal = signal
        except ImportError:
            pass

    def _alarm_handler(self, signum, frame):
        """Signal handler for SIGALRM (Unix only)"""
        self.timed_out = True
        raise TimeoutError(f"Execution timeout exceeded ({self.timeout_seconds}s)")

    def start(self):
        self.start_time = time.time()
        self.timed_out = False
        self._check_count = 0

        if self._use_signal:
            # Unix: Use signal.alarm for robust timeout
            self.original_alarm_handler = self._signal.signal(
                self._signal.SIGALRM, self._alarm_handler
            )
            # signal.alarm only accepts integers, so we use setitimer for sub-second
            if hasattr(self._signal, 'setitimer'):
                self._signal.setitimer(
                    self._signal.ITIMER_REAL,
                    self.timeout_seconds
                )
            else:
                # Fallback to alarm (integer seconds, rounded up)
                import math
                self._signal.alarm(math.ceil(self.timeout_seconds))

        # Always also use sys.settrace as a backup/supplement
        self.original_trace = sys.gettrace()
        sys.settrace(self._trace_callback)

    def stop(self):
        # Stop sys.settrace
        sys.settrace(self.original_trace)
        self.original_trace = None

        if self._use_signal:
            # Cancel any pending alarm
            if hasattr(self._signal, 'setitimer'):
                self._signal.setitimer(self._signal.ITIMER_REAL, 0)
            else:
                self._signal.alarm(0)
            # Restore original handler
            if self.original_alarm_handler is not None:
                self._signal.signal(self._signal.SIGALRM, self.original_alarm_handler)
                self.original_alarm_handler = None

    def _trace_callback(self, frame, event, arg):
        """Called on each Python line execution (backup timeout check)"""
        # Only check timeout periodically for performance
        self._check_count += 1
        if self._check_count >= self._check_interval:
            self._check_count = 0
            if time.time() - self.start_time > self.timeout_seconds:
                self.timed_out = True
                # Raise exception to interrupt execution
                raise TimeoutError(f"Execution timeout exceeded ({self.timeout_seconds}s)")
        return self._trace_callback

# Create global watchdog instance
_sandbox_timeout_watchdog = _SandboxTimeoutWatchdog(__TIMEOUT_SECONDS__)
)PY";

        // Replace placeholder with actual timeout value
        double timeout_seconds = static_cast<double>(timeout_ms) / 1000.0;
        size_t pos = timeout_code.find("__TIMEOUT_SECONDS__");
        if (pos != std::string::npos) {
            timeout_code.replace(pos, 19, std::to_string(timeout_seconds));
        }

        py::exec(timeout_code);
        spdlog::info("Timeout watchdog configured ({}s)", timeout_seconds);

    } catch (const py::error_already_set& e) {
        spdlog::error("Failed to setup timeout watchdog: {}", e.what());
    }
}

void PythonSandbox::RemoveTimeoutWatchdog() {
    try {
        // Stop the watchdog and restore original trace
        // Use Python try/except to avoid import errors during exception formatting
        py::exec(R"(
try:
    if "_sandbox_timeout_watchdog" in dir():
        _sandbox_timeout_watchdog.stop()
except:
    pass  # Silently ignore cleanup errors
)");
        spdlog::debug("Timeout watchdog removed");
    } catch (const py::error_already_set& e) {
        // Silently ignore - cleanup is best-effort
        spdlog::debug("Timeout watchdog cleanup: {}", e.what());
    }
}

void PythonSandbox::CleanupHooks() {
    try {
        // Use pybind11 C++ API directly to avoid import issues
        // py::module_::import() uses Python C API and bypasses __import__ hook
        py::module_ builtins = py::module_::import("builtins");

        // Cleanup import hook
        py::object current_import = builtins.attr("__import__");
        if (py::hasattr(current_import, "original_import")) {
            py::object original_import = current_import.attr("original_import");
            builtins.attr("__import__") = original_import;
            spdlog::debug("Sandbox import hook cleaned up");
        }

        // Cleanup file access hook (open)
        py::object current_open = builtins.attr("open");
        if (py::hasattr(current_open, "original_open")) {
            py::object original_open = current_open.attr("original_open");
            builtins.attr("open") = original_open;
            spdlog::debug("Sandbox file hook cleaned up");
        }

        // Cleanup os module hooks
        py::module_ main = py::module_::import("__main__");
        std::string cleanup_code = R"(
import os
# Restore os module functions if originals were saved
if '_sandbox_original_remove' in dir():
    os.remove = _sandbox_original_remove
if '_sandbox_original_unlink' in dir():
    os.unlink = _sandbox_original_unlink
if '_sandbox_original_rename' in dir():
    os.rename = _sandbox_original_rename
if '_sandbox_original_mkdir' in dir():
    os.mkdir = _sandbox_original_mkdir
if '_sandbox_original_makedirs' in dir():
    os.makedirs = _sandbox_original_makedirs
if '_sandbox_original_rmdir' in dir():
    os.rmdir = _sandbox_original_rmdir
)";
        py::exec(cleanup_code);
        spdlog::debug("Sandbox os module hooks cleaned up");

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
    initial_memory_ = GetCurrentMemoryUsage();
    monitoring_active_ = true;
    spdlog::debug("Memory monitoring started (baseline: {} MB)", initial_memory_ / (1024 * 1024));
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

    // Check memory limit
    size_t current_memory = GetCurrentMemoryUsage();
    size_t memory_delta = (current_memory > initial_memory_) ? (current_memory - initial_memory_) : 0;
    result.peak_memory_bytes = memory_delta;

    size_t max_memory_bytes = config_.max_memory_mb * 1024 * 1024;
    if (memory_delta > max_memory_bytes) {
        result.memory_exceeded = true;
        result.error_message = "Memory limit exceeded (" +
            std::to_string(memory_delta / (1024 * 1024)) + " MB > " +
            std::to_string(config_.max_memory_mb) + " MB limit)";
        spdlog::warn("Memory limit exceeded: {} MB (limit: {} MB)",
            memory_delta / (1024 * 1024), config_.max_memory_mb);
        return false;
    }

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
    SetupTimeoutWatchdog();

    // Start monitoring
    StartMonitoring();

    // Objects for stdout/stderr capture (declared outside try for cleanup)
    py::object sys;
    py::object io;
    py::object stdout_capture;
    py::object stderr_capture;
    py::object original_stdout;
    py::object original_stderr;
    bool streams_redirected = false;

    try {
        // Redirect stdout/stderr
        sys = py::module_::import("sys");
        io = py::module_::import("io");

        stdout_capture = io.attr("StringIO")();
        stderr_capture = io.attr("StringIO")();

        original_stdout = sys.attr("stdout");
        original_stderr = sys.attr("stderr");

        sys.attr("stdout") = stdout_capture;
        sys.attr("stderr") = stderr_capture;
        streams_redirected = true;

        // Start the timeout watchdog before execution
        py::exec("_sandbox_timeout_watchdog.start()");

        // Execute code with timeout protection via sys.settrace
        py::exec(code);

        // Stop the timeout watchdog
        py::exec("_sandbox_timeout_watchdog.stop()");

        // Restore stdout/stderr
        sys.attr("stdout") = original_stdout;
        sys.attr("stderr") = original_stderr;
        streams_redirected = false;

        // Get captured output
        result.output = py::str(stdout_capture.attr("getvalue")());
        std::string stderr_output = py::str(stderr_capture.attr("getvalue")());

        if (!stderr_output.empty()) {
            result.output += "\nStderr: " + stderr_output;
        }

        result.success = true;

    } catch (const py::error_already_set& e) {
        // Stop watchdog on error
        try {
            py::exec("if '_sandbox_timeout_watchdog' in dir(): _sandbox_timeout_watchdog.stop()");
        } catch (...) {}

        // Restore streams if redirected
        if (streams_redirected) {
            try {
                sys.attr("stdout") = original_stdout;
                sys.attr("stderr") = original_stderr;
            } catch (...) {}
        }

        // Check if this was a timeout
        std::string error_str = e.what();
        if (error_str.find("TimeoutError") != std::string::npos ||
            error_str.find("Execution timeout exceeded") != std::string::npos) {
            result.timeout_exceeded = true;
            result.error_message = "Execution timeout exceeded (" +
                std::to_string(config_.timeout.count()) + "s)";
            spdlog::warn("Script execution timed out after {}s", config_.timeout.count());
        } else {
            result.error_message = error_str;
        }
        result.success = false;

        // Try to get any captured output before the error
        try {
            if (stdout_capture) {
                result.output = py::str(stdout_capture.attr("getvalue")());
            }
        } catch (...) {}

    } catch (const std::exception& e) {
        // Stop watchdog on error
        try {
            py::exec("if '_sandbox_timeout_watchdog' in dir(): _sandbox_timeout_watchdog.stop()");
        } catch (...) {}

        // Restore streams if redirected
        if (streams_redirected) {
            try {
                sys.attr("stdout") = original_stdout;
                sys.attr("stderr") = original_stderr;
            } catch (...) {}
        }

        result.error_message = std::string("Execution error: ") + e.what();
        result.success = false;
    }

    // Stop monitoring and check limits
    StopMonitoring();
    CheckResourceLimits(result);

    // Cleanup (also removes timeout watchdog)
    RemoveTimeoutWatchdog();
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
