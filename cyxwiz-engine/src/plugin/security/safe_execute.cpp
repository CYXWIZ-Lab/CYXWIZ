#include "safe_execute.h"
#include <spdlog/spdlog.h>
#include <sstream>
#include <iomanip>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace cyxwiz::plugin::security {

// ============================================================================
// Platform-specific crash isolation
// ============================================================================

#ifdef _WIN32

// MSVC forbids __try/__except in functions with C++ objects that have destructors.
// We use a plain-C trampoline that invokes a function pointer under SEH protection.

struct SEHTrampoline {
    void (*fn)(void*);
    void* arg;
};

// This function MUST NOT have any C++ objects with destructors.
// It catches all hardware exceptions (access violation, stack overflow, etc.).
#pragma warning(push)
#pragma warning(disable: 4611) // interaction between '_setjmp' and C++ object destruction
static int InvokeSEH(SEHTrampoline* trampoline, DWORD* exception_code_out) {
    __try {
        trampoline->fn(trampoline->arg);
        return 1;  // success
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        *exception_code_out = GetExceptionCode();
        return 0;  // crash
    }
}
#pragma warning(pop)

const char* ExceptionCodeToString(uint32_t code) {
    switch (code) {
        case EXCEPTION_ACCESS_VIOLATION:      return "Access Violation";
        case EXCEPTION_STACK_OVERFLOW:        return "Stack Overflow";
        case EXCEPTION_INT_DIVIDE_BY_ZERO:    return "Integer Divide by Zero";
        case EXCEPTION_FLT_DIVIDE_BY_ZERO:    return "Float Divide by Zero";
        case EXCEPTION_ILLEGAL_INSTRUCTION:   return "Illegal Instruction";
        case EXCEPTION_ARRAY_BOUNDS_EXCEEDED: return "Array Bounds Exceeded";
        case EXCEPTION_DATATYPE_MISALIGNMENT: return "Data Misalignment";
        case EXCEPTION_IN_PAGE_ERROR:         return "In-Page Error";
        case EXCEPTION_PRIV_INSTRUCTION:      return "Privileged Instruction";
        default:                              return "Unknown Exception";
    }
}

SafeExecuteResult SafeExecute(
    const std::string& plugin_id,
    const std::string& operation,
    std::function<void()> callback)
{
    SafeExecuteResult result;

    // Trampoline to bridge std::function to plain C function pointer
    auto* cb_ptr = &callback;
    auto trampoline_fn = [](void* arg) {
        auto* fn = static_cast<std::function<void()>*>(arg);
        (*fn)();
    };

    SEHTrampoline trampoline{trampoline_fn, cb_ptr};
    DWORD exception_code = 0;

    int seh_ok = InvokeSEH(&trampoline, &exception_code);

    if (seh_ok) {
        result.success = true;
    } else {
        result.crashed = true;
        result.fault_code = exception_code;
        std::ostringstream hex_code;
        hex_code << "0x" << std::hex << std::uppercase << exception_code;
        result.error_message = std::string("Plugin '") + plugin_id +
            "' crashed during " + operation + ": " +
            ExceptionCodeToString(exception_code) +
            " (" + hex_code.str() + ")";
        spdlog::error("{}", result.error_message);
    }

    return result;
}

#else  // Unix

#include <csignal>
#include <csetjmp>

static thread_local sigjmp_buf safe_execute_jmp_buf;
static thread_local volatile sig_atomic_t safe_execute_signal = 0;
static thread_local struct sigaction old_sigsegv, old_sigbus, old_sigfpe, old_sigabrt;

static void SafeExecuteSignalHandler(int sig) {
    safe_execute_signal = sig;
    siglongjmp(safe_execute_jmp_buf, 1);
}

const char* ExceptionCodeToString(uint32_t code) {
    switch (code) {
        case SIGSEGV: return "Segmentation Fault";
        case SIGBUS:  return "Bus Error";
        case SIGFPE:  return "Floating Point Exception";
        case SIGABRT: return "Abort";
        default:      return "Unknown Signal";
    }
}

SafeExecuteResult SafeExecute(
    const std::string& plugin_id,
    const std::string& operation,
    std::function<void()> callback)
{
    SafeExecuteResult result;

    // Install signal handlers
    struct sigaction sa;
    sa.sa_handler = SafeExecuteSignalHandler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;

    sigaction(SIGSEGV, &sa, &old_sigsegv);
    sigaction(SIGBUS, &sa, &old_sigbus);
    sigaction(SIGFPE, &sa, &old_sigfpe);
    sigaction(SIGABRT, &sa, &old_sigabrt);

    safe_execute_signal = 0;

    if (sigsetjmp(safe_execute_jmp_buf, 1) == 0) {
        // Normal path
        callback();
        result.success = true;
    } else {
        // Signal caught
        result.crashed = true;
        result.fault_code = safe_execute_signal;
        result.error_message = std::string("Plugin '") + plugin_id +
            "' crashed during " + operation + ": " +
            ExceptionCodeToString(safe_execute_signal);
        spdlog::error("{}", result.error_message);
    }

    // Restore original handlers
    sigaction(SIGSEGV, &old_sigsegv, nullptr);
    sigaction(SIGBUS, &old_sigbus, nullptr);
    sigaction(SIGFPE, &old_sigfpe, nullptr);
    sigaction(SIGABRT, &old_sigabrt, nullptr);

    return result;
}

#endif  // Platform

// ============================================================================
// SafeExecuteBool — convenience for bool-returning lifecycle methods
// ============================================================================

SafeExecuteResult SafeExecuteBool(
    const std::string& plugin_id,
    const std::string& operation,
    std::function<bool()> callback,
    bool& result_out)
{
    result_out = false;
    bool temp_result = false;

    // First try C++ exception handling around the SEH wrapper
    try {
        auto seh_result = SafeExecute(plugin_id, operation, [&]() {
            temp_result = callback();
        });
        // Only propagate result if execution succeeded (no crash)
        if (seh_result.success) {
            result_out = temp_result;
        }
        return seh_result;
    } catch (const std::exception& e) {
        SafeExecuteResult result;
        result.error_message = std::string("Plugin '") + plugin_id +
            "' threw during " + operation + ": " + e.what();
        spdlog::error("{}", result.error_message);
        return result;
    } catch (...) {
        SafeExecuteResult result;
        result.error_message = std::string("Plugin '") + plugin_id +
            "' threw unknown exception during " + operation;
        spdlog::error("{}", result.error_message);
        return result;
    }
}

} // namespace cyxwiz::plugin::security
