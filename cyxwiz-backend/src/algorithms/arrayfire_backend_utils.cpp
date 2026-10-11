#include "arrayfire_backend_utils.h"

#include <algorithm>
#include <chrono>
#include "cyxwiz/debug_hooks.h"
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <arrayfire.h>

namespace cyxwiz {

namespace {
std::atomic<uint64_t> g_attention_dropout_streams{0};
}  // namespace

namespace {
bool ProfileStageSyncEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("CYXWIZ_PROFILE_STAGE_SYNC");
        return value && *value && std::string(value) != "0";
    }();
    return enabled;
}

int64_t SteadyNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
}  // namespace

ScopedProfileSpan::ScopedProfileSpan(const char* name)
    : name_(name), active_(BackendDebugHooks::HasDebugEventCallback()), start_ns_(0) {
    if (!active_) return;
    if (ProfileStageSyncEnabled()) af::sync();
    start_ns_ = SteadyNowNs();
}

ScopedProfileSpan::~ScopedProfileSpan() {
    if (!active_) return;
    if (ProfileStageSyncEnabled()) {
        try {
            af::sync();
        } catch (...) {
        }
    }
    const double ms = static_cast<double>(SteadyNowNs() - start_ns_) / 1.0e6;
    BackendDebugHooks::EmitDebugEvent("ModelSpan", std::string("name=") + name_ + " duration_ms=" + std::to_string(ms));
}

uint64_t NextAttentionDropoutStream() {
    return g_attention_dropout_streams.fetch_add(1) + 1;
}

std::string SeedCurrentArrayFireRandomEngine(uint64_t seed) {
    g_attention_dropout_streams = 0;
    const auto type = af::getDefaultRandomEngine().getType();
    int major = 0, minor = 0, patch = 0;
    const af_err version_status = af_get_version(&major, &minor, &patch);
    if (version_status != AF_SUCCESS)
        throw std::runtime_error("Cannot identify ArrayFire RNG runtime version");
    af::setSeed(seed);
    return "arrayfire/" + std::to_string(major) + "." + std::to_string(minor) + "." +
        std::to_string(patch) + "/default_type=" + std::to_string(static_cast<int>(type));
}

namespace {

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
        case ArrayFireHostSyncCategory::LossInputValidation:
            return "loss_input_validation";
        case ArrayFireHostSyncCategory::MetricInputValidation:
            return "metric_input_validation";
        case ArrayFireHostSyncCategory::ProviderHostStaging:
            return "provider_host_staging";
        case ArrayFireHostSyncCategory::CheckpointOutput:
            return "checkpoint_output";
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
}

bool IsCurrentArrayFireBackendAvailable() {
    try {
        const af::Backend backend = af::getActiveBackend();
        return backend == AF_BACKEND_CUDA ||
               backend == AF_BACKEND_OPENCL ||
               backend == AF_BACKEND_ONEAPI ||
               backend == AF_BACKEND_CPU;
    } catch (...) {
        return false;
    }
}

} // namespace cyxwiz
