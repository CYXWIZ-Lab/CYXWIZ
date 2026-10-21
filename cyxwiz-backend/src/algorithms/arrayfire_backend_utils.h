#pragma once

#include "cyxwiz/api_export.h"
#include "cyxwiz/backend_fallback_reason.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cyxwiz {

// Reset the current thread/device default ArrayFire RNG and the attention
// dropout stream counter; return the ArrayFire generator identity. Does not
// restore stream continuation.
CYXWIZ_API std::string SeedCurrentArrayFireRandomEngine(uint64_t seed);

// Next attention-dropout mask stream: fused attention derives each call's mask
// seed from ArrayFire's seed and this counter. Reset to zero by
// SeedCurrentArrayFireRandomEngine, so a seeded run repeats in the same
// process (the counter used to run on across runs).
CYXWIZ_API uint64_t NextAttentionDropoutStream();

// Profiling span (TOFIX118 P8): when a BackendDebugHooks listener is set,
// times the enclosing scope and reports it as a "ModelSpan" event
// ("name=<name> duration_ms=<ms>"). With CYXWIZ_PROFILE_STAGE_SYNC=1 it waits
// for the device first, so the time is the span's device work. Free when
// nothing listens.
class CYXWIZ_API ScopedProfileSpan {
public:
    explicit ScopedProfileSpan(const char* name);
    ~ScopedProfileSpan();
    ScopedProfileSpan(const ScopedProfileSpan&) = delete;
    ScopedProfileSpan& operator=(const ScopedProfileSpan&) = delete;

private:
    const char* name_;
    bool active_;
    int64_t start_ns_;
};

enum class ArrayFireHostSyncCategory {
    Unknown,
    LossScalarReadback,
    MetricScalarReadback,
    LayoutConversion,
    DebugSampleDump,
    CheckpointOutput,
    OutputMaterialization,
    LossInputValidation,
    MetricInputValidation,
    // Provider tenants that cannot share ArrayFire's device memory (oneAPI)
    // stage inputs through host memory.
    ProviderHostStaging,
};

struct ArrayFireHostSyncAttribution {
    ArrayFireHostSyncCategory category = ArrayFireHostSyncCategory::Unknown;
    std::string operation_name;
};

struct ArrayFireHostSyncEvent {
    std::string operation_name;
    std::string selected_backend;
    std::string reason_code;
    std::string attribution_category;
    std::string attribution_operation;
    std::vector<size_t> tensor_shape;
    std::string tensor_dtype;
    std::string tensor_layout;
    std::string context;
    uint64_t bytes = 0;
};

using ArrayFireHostSyncObserver =
    void (*)(const ArrayFireHostSyncEvent& event);

class CYXWIZ_API ScopedArrayFireHostSyncObserver {
public:
    explicit ScopedArrayFireHostSyncObserver(
        ArrayFireHostSyncObserver observer);
    ~ScopedArrayFireHostSyncObserver();

    ScopedArrayFireHostSyncObserver(
        const ScopedArrayFireHostSyncObserver&) = delete;
    ScopedArrayFireHostSyncObserver& operator=(
        const ScopedArrayFireHostSyncObserver&) = delete;

private:
    ArrayFireHostSyncObserver previous_;
};

class CYXWIZ_API ScopedArrayFireHostSyncAttribution {
public:
    ScopedArrayFireHostSyncAttribution(
        ArrayFireHostSyncCategory category,
        std::string operation_name);
    ~ScopedArrayFireHostSyncAttribution();

    ScopedArrayFireHostSyncAttribution(
        const ScopedArrayFireHostSyncAttribution&) = delete;
    ScopedArrayFireHostSyncAttribution& operator=(
        const ScopedArrayFireHostSyncAttribution&) = delete;

private:
    ArrayFireHostSyncAttribution previous_;
};

CYXWIZ_API ArrayFireHostSyncObserver GetArrayFireHostSyncObserver();
CYXWIZ_API void SetArrayFireHostSyncObserver(
    ArrayFireHostSyncObserver observer);
CYXWIZ_API ArrayFireHostSyncAttribution GetArrayFireHostSyncAttribution();
CYXWIZ_API void SetArrayFireHostSyncAttribution(
    const ArrayFireHostSyncAttribution& attribution);
CYXWIZ_API const char* ArrayFireHostSyncCategoryName(
    ArrayFireHostSyncCategory category);
CYXWIZ_API void NotifyArrayFireHostSync(ArrayFireHostSyncEvent event);
CYXWIZ_API bool IsCudaJitFormalParameterOverflow(const char* message);
CYXWIZ_API BackendFallbackReason ClassifyArrayFireBackendFallbackReason(
    const char* message);

CYXWIZ_API std::string BuildTensorShapeContext(
    const char* tensor_name,
    const std::vector<size_t>& shape);
CYXWIZ_API std::string CurrentArrayFireBackendName();
CYXWIZ_API bool IsCurrentArrayFireBackendAvailable();

} // namespace cyxwiz
