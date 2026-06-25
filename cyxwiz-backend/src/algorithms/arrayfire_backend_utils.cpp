#include "arrayfire_backend_utils.h"

#include <cstdlib>
#include <mutex>
#include <set>
#include <sstream>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {
namespace {

std::mutex g_backend_fallback_log_mutex;
std::set<std::string> g_backend_fallback_log_keys;

} // namespace

const char* BackendFallbackReasonName(BackendFallbackReason reason) {
    switch (reason) {
    case BackendFallbackReason::CudaJitParamOverflow:
        return "cuda_jit_param_overflow";
    case BackendFallbackReason::ArrayFireJitCompileFailure:
        return "arrayfire_jit_compile_failure";
    case BackendFallbackReason::GpuBackendException:
        return "gpu_backend_exception";
    }
    return "gpu_backend_exception";
}

bool IsCudaJitFormalParameterOverflow(const char* message) {
    if (message == nullptr) {
        return false;
    }
    const std::string text(message);
    return text.find("Formal parameter space overflowed") != std::string::npos ||
           text.find("formal parameter") != std::string::npos;
}

BackendFallbackReason ClassifyArrayFireBackendFallbackReason(
    const char* message) {
    if (IsCudaJitFormalParameterOverflow(message)) {
        return BackendFallbackReason::CudaJitParamOverflow;
    }
    if (message == nullptr) {
        return BackendFallbackReason::GpuBackendException;
    }
    const std::string text(message);
    if (text.find("NVRTC") != std::string::npos ||
        text.find("JIT") != std::string::npos ||
        text.find("compile") != std::string::npos ||
        text.find("Compile") != std::string::npos) {
        return BackendFallbackReason::ArrayFireJitCompileFailure;
    }
    return BackendFallbackReason::GpuBackendException;
}

std::string BuildTensorShapeContext(
    const char* tensor_name,
    const std::vector<size_t>& shape) {
    std::ostringstream out;
    out << (tensor_name ? tensor_name : "tensor") << "=[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out << "x";
        }
        out << shape[i];
    }
    out << "]";
    return out.str();
}

std::string CurrentArrayFireBackendName() {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        switch (af::getActiveBackend()) {
        case AF_BACKEND_CUDA:
            return "cuda";
        case AF_BACKEND_OPENCL:
            return "opencl";
        case AF_BACKEND_CPU:
            return "cpu";
        default:
            return "unknown";
        }
    } catch (...) {
        return "unknown";
    }
#else
    return "unavailable";
#endif
}

std::string BuildArrayFireBackendFallbackContext(
    const std::string& shape_or_node_context,
    const std::string& backend_name) {
    std::string context = "backend=";
    context += backend_name.empty() ? "unknown" : backend_name;
    if (!shape_or_node_context.empty()) {
        context += "; ";
        context += shape_or_node_context;
    }
    return context;
}

std::string BuildArrayFireBackendFallbackMessage(
    const char* operation_name,
    BackendFallbackReason reason,
    bool include_error_text,
    const char* error_message,
    const std::string& context) {
    std::string message = std::string("ArrayFire ") +
        (operation_name ? operation_name : "operation") +
        " failed (reason=" + BackendFallbackReasonName(reason) +
        "); falling back to CPU. Training continues, but this path may be slower.";
    if (!context.empty()) {
        message += " Context: ";
        message += context;
        message += ".";
    }
    if (include_error_text && error_message != nullptr &&
        error_message[0] != '\0') {
        message += " Error: ";
        message += error_message;
    }
    return message;
}

bool ShouldLogArrayFireBackendFallbackOnce(
    const char* operation_name,
    BackendFallbackReason reason,
    const std::string& context) {
    std::string key = operation_name ? operation_name : "unknown";
    key += ":";
    key += BackendFallbackReasonName(reason);
    if (!context.empty()) {
        key += ":";
        key += context;
    }
    std::lock_guard<std::mutex> lock(g_backend_fallback_log_mutex);
    return g_backend_fallback_log_keys.insert(key).second;
}

bool ShouldForceArrayFireBackendFallbackForTesting(
    const char* operation_name) {
#ifdef NDEBUG
    (void)operation_name;
    return false;
#else
    const char* value = std::getenv("CYXWIZ_TEST_FORCE_ARRAYFIRE_FALLBACK");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    const std::string requested(value);
    size_t start = 0;
    while (start <= requested.size()) {
        const size_t separator = requested.find_first_of(",;", start);
        const size_t end = separator == std::string::npos
            ? requested.size()
            : separator;
        const size_t first =
            end > start ? requested.find_first_not_of(" \t\r\n", start) :
                          std::string::npos;
        const size_t last =
            first != std::string::npos && first < end
                ? requested.find_last_not_of(" \t\r\n", end - 1)
                : std::string::npos;
        if (first != std::string::npos && first < end &&
            last != std::string::npos) {
            const std::string token =
                requested.substr(first, last - first + 1);
            if (token == "*" ||
                token == (operation_name ? operation_name : "")) {
                return true;
            }
        }
        if (separator == std::string::npos) {
            break;
        }
        start = separator + 1;
    }
    return false;
#endif
}

} // namespace cyxwiz
