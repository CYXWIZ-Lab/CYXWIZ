#pragma once

#include <functional>
#include <string>
#include <cstdint>

namespace cyxwiz::plugin::security {

struct SafeExecuteResult {
    bool success = false;
    bool crashed = false;          // True if SEH/signal caught a hardware fault
    uint32_t fault_code = 0;       // Windows: GetExceptionCode(), Unix: signal number
    std::string error_message;
};

// Execute a plugin callback with crash isolation.
// Catches both C++ exceptions and hardware faults (access violation, stack overflow).
// On Windows uses SEH (__try/__except), on Unix uses sigsetjmp/siglongjmp.
SafeExecuteResult SafeExecute(
    const std::string& plugin_id,
    const std::string& operation,   // e.g. "OnLoad", "Render", "OnEpochEnd"
    std::function<void()> callback
);

// Convenience for bool-returning lifecycle methods (OnLoad, OnInitialize).
SafeExecuteResult SafeExecuteBool(
    const std::string& plugin_id,
    const std::string& operation,
    std::function<bool()> callback,
    bool& result_out
);

// Convert a Windows exception code to a human-readable string.
const char* ExceptionCodeToString(uint32_t code);

} // namespace cyxwiz::plugin::security
