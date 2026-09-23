#pragma once

#include "api_export.h"
#include "backend_fallback_reason.h"
#include "recurrent_cuda_placement.h"

#include <string>
#include <vector>

namespace cyxwiz {

// Observation-facing spellings of the failure taxonomy. Derived from the
// typed BackendFallbackReason authority so the strings cannot drift; add new
// reasons to backend_fallback_reason.h, not here.
namespace BackendPlacementObservationReason {
inline constexpr const char* CudaJitParamOverflow =
    BackendFallbackReasonName(BackendFallbackReason::CudaJitParamOverflow);
inline constexpr const char* ArrayFireJitCompileFailure =
    BackendFallbackReasonName(BackendFallbackReason::ArrayFireJitCompileFailure);
inline constexpr const char* GpuBackendException =
    BackendFallbackReasonName(BackendFallbackReason::GpuBackendException);
inline constexpr const char* GpuOutOfMemory =
    BackendFallbackReasonName(BackendFallbackReason::GpuOutOfMemory);
inline constexpr const char* UnsupportedDtype =
    BackendFallbackReasonName(BackendFallbackReason::UnsupportedDtype);
inline constexpr const char* UnsupportedShape =
    BackendFallbackReasonName(BackendFallbackReason::UnsupportedShape);
inline constexpr const char* BackendCompileTimeout =
    BackendFallbackReasonName(BackendFallbackReason::BackendCompileTimeout);
inline constexpr const char* BackendInternalError =
    BackendFallbackReasonName(BackendFallbackReason::BackendInternalError);
} // namespace BackendPlacementObservationReason

namespace BackendPlacementObservationSource {
inline constexpr const char* RuntimeFallback = "runtime_fallback";
inline constexpr const char* PreflightProbe = "preflight_probe";
inline constexpr const char* Test = "test";
} // namespace BackendPlacementObservationSource

namespace BackendPlacementProbeScope {
inline constexpr const char* NormalCompile = "normal_compile";
inline constexpr const char* DeepPreflight = "deep_preflight";
} // namespace BackendPlacementProbeScope

struct BackendPlacementObservation {
    std::string op_type;
    std::string backend;
    std::string device;
    std::string dtype;
    std::string shape_signature;
    std::string reason_code;
    std::string source = BackendPlacementObservationSource::RuntimeFallback;
    std::string detail;
    std::string timestamp;
    std::string probe_outcome;
    std::string probe_scope;
};

enum class BackendPlacementProbeOutcome {
    Safe,
    Unsafe,
    Timeout,
    Unsupported,
    Inconclusive,
};

struct BackendPlacementProbeResult {
    BackendPlacementProbeOutcome outcome = BackendPlacementProbeOutcome::Inconclusive;
    std::string reason_code;
    std::string detail;
    bool has_observation = false;
    BackendPlacementObservation observation;
};

CYXWIZ_API const char* BackendPlacementProbeOutcomeName(
    BackendPlacementProbeOutcome outcome);

CYXWIZ_API std::string BuildRecurrentCudaPlacementShapeSignature(
    const RecurrentCudaPlacementRequest& request);

CYXWIZ_API std::string BuildDensePlacementShapeSignature(
    const std::vector<size_t>& input_shape,
    size_t out_features);

CYXWIZ_API std::string BuildEmbeddingPlacementShapeSignature(
    size_t num_embeddings,
    size_t embedding_dim,
    const std::vector<size_t>& input_shape,
    const std::string& index_dtype);

CYXWIZ_API std::string BuildActivationPlacementShapeSignature(
    const std::vector<size_t>& input_shape,
    const std::string& dtype);

CYXWIZ_API std::string BuildLinearPlacementShapeSignature(
    const std::vector<size_t>& lhs_shape,
    const std::vector<size_t>& rhs_shape,
    const std::vector<size_t>& output_shape,
    const std::string& dtype,
    bool use_bias);

CYXWIZ_API std::string BuildLossPlacementShapeSignature(
    const std::vector<size_t>& prediction_shape,
    const std::vector<size_t>& target_shape,
    const std::string& reduction,
    const std::string& dtype);

CYXWIZ_API std::string BuildTensorOpPlacementShapeSignature(
    const std::vector<std::vector<size_t>>& input_shapes,
    const std::vector<size_t>& output_shape,
    const std::string& dtype,
    const std::string& attributes);

// Input-shape-only by design (tofix67 slice 7 key redesign): runtime
// Forward failures happen BEFORE an output exists, so an output-bearing key
// could never be formed by the writers that matter. Compiler lookups pass
// the batchless compile-time input shape; runtime writers pass
// StripBatchDimensionForPlacementSignature of the layer input.
CYXWIZ_API std::string BuildTensorLayerPlacementShapeSignature(
    const std::vector<size_t>& input_shape);

CYXWIZ_API std::string CurrentBackendPlacementDeviceSignature();

// Runtime tensors carry a leading batch dimension; compiler placement
// signatures exclude it (compiler shapes are batchless). Runtime observation
// writers MUST normalize a runtime tensor shape with this before building a
// shape signature — otherwise the recorded key can never match a compile-time
// lookup and the evidence is silently dead (tofix67 slice 6 audit finding).
CYXWIZ_API std::vector<size_t> StripBatchDimensionForPlacementSignature(
    const std::vector<size_t>& runtime_shape);

// Canonical observation-store key: op|backend|device|dtype|shape. The single
// authority for key formation — typed views (GpuExecutionKey) and the store
// itself both use this.
CYXWIZ_API std::string BuildBackendPlacementObservationKey(
    const std::string& op_type,
    const std::string& backend,
    const std::string& device,
    const std::string& dtype,
    const std::string& shape_signature);

CYXWIZ_API void RecordBackendPlacementObservation(
    const BackendPlacementObservation& observation);

CYXWIZ_API void RecordBackendPlacementObservationForActiveDevice(
    const std::string& op_type,
    const std::string& backend,
    const std::string& dtype,
    const std::string& shape_signature,
    const std::string& reason_code,
    const std::string& source,
    const std::string& detail);

CYXWIZ_API bool TryGetBackendPlacementObservation(
    const std::string& op_type,
    const std::string& backend,
    const std::string& device,
    const std::string& dtype,
    const std::string& shape_signature,
    BackendPlacementObservation& observation);

CYXWIZ_API bool TryGetBackendPlacementObservationForActiveDevice(
    const std::string& op_type,
    const std::string& backend,
    const std::string& dtype,
    const std::string& shape_signature,
    BackendPlacementObservation& observation);

CYXWIZ_API void RecordRecurrentCudaPlacementObservation(
    const RecurrentCudaPlacementRequest& request,
    const std::string& reason_code,
    const std::string& source,
    const std::string& detail);

CYXWIZ_API void RecordRecurrentCudaPreflightProbeFailure(
    const RecurrentCudaPlacementRequest& request,
    const std::string& reason_code,
    const std::string& detail);

CYXWIZ_API bool TryRunRecurrentCudaPreflightProbe(
    const RecurrentCudaPlacementRequest& request,
    BackendPlacementObservation& failure_observation);

CYXWIZ_API BackendPlacementProbeResult RunRecurrentCudaPreflightProbe(
    const RecurrentCudaPlacementRequest& request);

CYXWIZ_API bool TryGetRecurrentCudaPlacementObservation(
    const RecurrentCudaPlacementRequest& request,
    BackendPlacementObservation& observation);

CYXWIZ_API std::vector<BackendPlacementObservation>
SnapshotBackendPlacementObservations();

CYXWIZ_API bool SaveBackendPlacementObservationCache(
    const std::string& path,
    std::string* error_message = nullptr);

CYXWIZ_API bool LoadBackendPlacementObservationCache(
    const std::string& path,
    std::string* error_message = nullptr);

CYXWIZ_API void ClearBackendPlacementObservationCacheForTesting();

} // namespace cyxwiz
