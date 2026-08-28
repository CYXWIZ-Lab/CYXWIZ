#include "arrayfire_backend_utils.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {
namespace {

std::mutex g_backend_fallback_log_mutex;
std::set<std::string> g_backend_fallback_log_keys;
thread_local ArrayFireFallbackPolicy g_arrayfire_fallback_policy =
    ArrayFireFallbackPolicy::AllowNativeCpuFallback;
thread_local ArrayFireNativeCpuFallbackObserver
    g_arrayfire_native_cpu_fallback_observer = nullptr;
thread_local ArrayFireHostSyncObserver
    g_arrayfire_host_sync_observer = nullptr;
thread_local ArrayFireHostSyncAttribution
    g_arrayfire_host_sync_attribution;

std::string ToLowerAscii(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text;
}

bool ContainsAny(const std::string& text,
                 const std::initializer_list<const char*> needles) {
    for (const char* needle : needles) {
        if (needle != nullptr && text.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

} // namespace

ArrayFireFallbackPolicy GetArrayFireFallbackPolicy() {
    return g_arrayfire_fallback_policy;
}

void SetArrayFireFallbackPolicy(ArrayFireFallbackPolicy policy) {
    g_arrayfire_fallback_policy = policy;
}

ScopedArrayFireFallbackPolicy::ScopedArrayFireFallbackPolicy(
    ArrayFireFallbackPolicy policy)
    : previous_(GetArrayFireFallbackPolicy()) {
    SetArrayFireFallbackPolicy(policy);
}

ScopedArrayFireFallbackPolicy::~ScopedArrayFireFallbackPolicy() {
    SetArrayFireFallbackPolicy(previous_);
}

ScopedArrayFireNativeCpuFallbackObserver::
    ScopedArrayFireNativeCpuFallbackObserver(
        ArrayFireNativeCpuFallbackObserver observer)
    : previous_(GetArrayFireNativeCpuFallbackObserver()) {
    SetArrayFireNativeCpuFallbackObserver(observer);
}

ScopedArrayFireNativeCpuFallbackObserver::
    ~ScopedArrayFireNativeCpuFallbackObserver() {
    SetArrayFireNativeCpuFallbackObserver(previous_);
}

ScopedArrayFireHostSyncObserver::ScopedArrayFireHostSyncObserver(
    ArrayFireHostSyncObserver observer)
    : previous_(GetArrayFireHostSyncObserver()) {
    SetArrayFireHostSyncObserver(observer);
}

ScopedArrayFireHostSyncObserver::~ScopedArrayFireHostSyncObserver() {
    SetArrayFireHostSyncObserver(previous_);
}

ScopedArrayFireHostSyncAttribution::ScopedArrayFireHostSyncAttribution(
    ArrayFireHostSyncCategory category,
    std::string operation_name)
    : previous_(GetArrayFireHostSyncAttribution()) {
    SetArrayFireHostSyncAttribution({category, std::move(operation_name)});
}

ScopedArrayFireHostSyncAttribution::~ScopedArrayFireHostSyncAttribution() {
    SetArrayFireHostSyncAttribution(previous_);
}

bool IsArrayFireNativeCpuFallbackForbidden() {
    return GetArrayFireFallbackPolicy() ==
           ArrayFireFallbackPolicy::ForbidNativeCpuFallback;
}

ArrayFireNativeCpuFallbackObserver
GetArrayFireNativeCpuFallbackObserver() {
    return g_arrayfire_native_cpu_fallback_observer;
}

void SetArrayFireNativeCpuFallbackObserver(
    ArrayFireNativeCpuFallbackObserver observer) {
    g_arrayfire_native_cpu_fallback_observer = observer;
}

ArrayFireHostSyncObserver GetArrayFireHostSyncObserver() {
    return g_arrayfire_host_sync_observer;
}

void SetArrayFireHostSyncObserver(ArrayFireHostSyncObserver observer) {
    g_arrayfire_host_sync_observer = observer;
}

ArrayFireHostSyncAttribution GetArrayFireHostSyncAttribution() {
    return g_arrayfire_host_sync_attribution;
}

void SetArrayFireHostSyncAttribution(
    const ArrayFireHostSyncAttribution& attribution) {
    g_arrayfire_host_sync_attribution = attribution;
}

const char* ArrayFireHostSyncCategoryName(
    ArrayFireHostSyncCategory category) {
    switch (category) {
        case ArrayFireHostSyncCategory::LossScalarReadback:
            return "loss_scalar_readback";
        case ArrayFireHostSyncCategory::MetricScalarReadback:
            return "metric_scalar_readback";
        case ArrayFireHostSyncCategory::LayoutConversion:
            return "layout_conversion";
        case ArrayFireHostSyncCategory::DebugSampleDump:
            return "debug_sample_dump";
        case ArrayFireHostSyncCategory::LayerCpuPath:
            return "layer_cpu_path";
        case ArrayFireHostSyncCategory::OptimizerCpuPath:
            return "optimizer_cpu_path";
        case ArrayFireHostSyncCategory::LossCpuPath:
            return "loss_cpu_path";
        case ArrayFireHostSyncCategory::CheckpointOutput:
            return "checkpoint_output";
        case ArrayFireHostSyncCategory::AlgorithmCpuPath:
            return "algorithm_cpu_path";
        case ArrayFireHostSyncCategory::OutputMaterialization:
            return "output_materialization";
        case ArrayFireHostSyncCategory::Unknown:
        default:
            return "unknown";
    }
}

void NotifyArrayFireHostSync(ArrayFireHostSyncEvent event) {
    const auto observer = GetArrayFireHostSyncObserver();
    if (!observer) {
        return;
    }

    const auto attribution = GetArrayFireHostSyncAttribution();
    if (event.attribution_category.empty()) {
        event.attribution_category =
            ArrayFireHostSyncCategoryName(attribution.category);
    }
    if (event.attribution_operation.empty()) {
        event.attribution_operation = attribution.operation_name;
    }
    if (event.selected_backend.empty()) {
        event.selected_backend = CurrentArrayFireBackendName();
    }
    observer(event);
}

const char* BackendFallbackReasonName(BackendFallbackReason reason) {
    switch (reason) {
    case BackendFallbackReason::BackendUnavailable:
        return "backend_unavailable";
    case BackendFallbackReason::CudaJitParamOverflow:
        return "cuda_jit_param_overflow";
    case BackendFallbackReason::ArrayFireJitCompileFailure:
        return "arrayfire_jit_compile_failure";
    case BackendFallbackReason::GpuBackendException:
        return "gpu_backend_exception";
    case BackendFallbackReason::GpuOutOfMemory:
        return "gpu_out_of_memory";
    case BackendFallbackReason::UnsupportedDtype:
        return "unsupported_dtype";
    case BackendFallbackReason::UnsupportedShape:
        return "unsupported_shape";
    case BackendFallbackReason::UnsupportedOperation:
        return "unsupported_operation";
    case BackendFallbackReason::BackendCompileTimeout:
        return "backend_compile_timeout";
    case BackendFallbackReason::BackendInternalError:
        return "backend_internal_error";
    }
    return "backend_internal_error";
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
        return BackendFallbackReason::BackendInternalError;
    }
    const std::string text = ToLowerAscii(message);
    if (ContainsAny(text, {
            "backend unavailable",
            "backend is unavailable",
            "no arrayfire backend"})) {
        return BackendFallbackReason::BackendUnavailable;
    }
    if (ContainsAny(text, {
            "out of memory",
            "cuda_error_memory_allocation",
            "memory allocation",
            "allocation failed",
            "failed to allocate",
            "cl_mem_object_allocation_failure",
            "af_err_no_mem"})) {
        return BackendFallbackReason::GpuOutOfMemory;
    }
    if (ContainsAny(text, {
            "unsupported dtype",
            "unsupported type",
            "type not supported",
            "invalid type",
            "dtype not supported",
            "af_err_type"})) {
        return BackendFallbackReason::UnsupportedDtype;
    }
    if (ContainsAny(text, {
            "unsupported shape",
            "shape not supported",
            "invalid shape",
            "invalid dimension",
            "dimension mismatch",
            "dims mismatch",
            "af_err_size"})) {
        return BackendFallbackReason::UnsupportedShape;
    }
    if (ContainsAny(text, {
            "timeout",
            "timed out",
            "compile timeout",
            "execution timeout",
            "launch timeout"})) {
        return BackendFallbackReason::BackendCompileTimeout;
    }
    if (ContainsAny(text, {
            "device lost",
            "device reset",
            "gpu device",
            "cuda_error_unknown",
            "cuda_error_launch_failure",
            "af_err_device"})) {
        return BackendFallbackReason::GpuBackendException;
    }
    if (ContainsAny(text, {
            "nvrtc",
            "jit",
            "compile",
            "compilation",
            "program build",
            "build program"})) {
        return BackendFallbackReason::ArrayFireJitCompileFailure;
    }
    return BackendFallbackReason::BackendInternalError;
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
        case AF_BACKEND_ONEAPI:
            return "oneapi";
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

bool IsCurrentArrayFireBackendGpu() {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::Backend backend = af::getActiveBackend();
        return backend == AF_BACKEND_CUDA || backend == AF_BACKEND_OPENCL;
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

bool IsCurrentArrayFireBackendAvailable() {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::Backend backend = af::getActiveBackend();
        return backend == AF_BACKEND_CUDA ||
               backend == AF_BACKEND_OPENCL ||
               backend == AF_BACKEND_ONEAPI ||
               backend == AF_BACKEND_CPU;
    } catch (...) {
        return false;
    }
#else
    return false;
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
        "); falling back to native CPU. Training continues, but this path may be slower.";
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

void ThrowIfArrayFireNativeCpuFallbackForbidden(
    const char* operation_name,
    BackendFallbackReason reason,
    const char* error_message,
    const std::string& context) {
    if (const auto observer = GetArrayFireNativeCpuFallbackObserver()) {
        ArrayFireNativeCpuFallbackEvent event;
        event.operation_name = operation_name ? operation_name : "operation";
        event.reason_code = BackendFallbackReasonName(reason);
        event.selected_backend = CurrentArrayFireBackendName();
        event.context = context;
        event.error_message = error_message ? error_message : "";
        event.fallback_forbidden = IsArrayFireNativeCpuFallbackForbidden();
        observer(event);
    }

    if (!IsArrayFireNativeCpuFallbackForbidden()) {
        return;
    }

    std::string message = std::string("ArrayFire ") +
        (operation_name ? operation_name : "operation") +
        " failed (reason=" + BackendFallbackReasonName(reason) +
        "); native CPU fallback is forbidden by the current execution policy.";
    if (!context.empty()) {
        message += " Context: ";
        message += context;
        message += ".";
    }
    if (error_message != nullptr && error_message[0] != '\0') {
        message += " Error: ";
        message += error_message;
    }
    throw std::runtime_error(message);
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
