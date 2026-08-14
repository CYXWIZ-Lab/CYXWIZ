#pragma once

#include <cyxwiz/device.h>

#include <filesystem>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cyxwiz {

inline constexpr const char* kCyxWizComputeContractId =
    "cyxwiz-arrayfire-training-v1";
inline constexpr const char* kRouteQualificationOperationManifestId =
    "arrayfire-route-ops-v1";
inline constexpr const char* kRouteQualificationMatrixId =
    "cyxwiz-route-qualification-v1";
inline constexpr const char* kRoutePerformanceBenchmarkId =
    "cyxwiz-dense-compute-v1";
inline constexpr int kRouteQualificationOperationCount =
    0
#define CYXWIZ_ROUTE_OPERATION(name) + 1
#include "arrayfire_route_operations.def"
#undef CYXWIZ_ROUTE_OPERATION
    ;

inline const char* RouteQualificationEvidenceLabel(
    std::string_view matrix_id) {
    if (matrix_id.empty()) return "Not available";
    if (matrix_id == kRouteQualificationMatrixId) {
        return "CyxWiz route qualification v1";
    }
    return "Legacy route qualification evidence";
}

enum class RouteFailureStage {
    None,
    Package,
    BackendLoad,
    Enumeration,
    Identity,
    Activation,
    Operation,
    Numerical,
    StrictTraining,
    Policy,
    Evidence
};

enum class RouteFailureCategory {
    None,
    PackageAbsent,
    PackageInvalid,
    AbiMismatch,
    DependencyMissing,
    ProviderMissing,
    BackendLoadFailed,
    DeviceNotEnumerated,
    IdentityMismatch,
    ActivationFailed,
    BackendSubstitution,
    UnsupportedOperation,
    OperationFailed,
    OutOfMemory,
    NumericalMismatch,
    NonFiniteResult,
    Timeout,
    ChildProcessCrash,
    NativeFallback,
    ResidencyViolation,
    PolicyBlocked,
    Cancelled,
    EvidenceStale,
    MalformedEvidence
};

struct RouteFailureDiagnostic {
    RouteFailureStage stage = RouteFailureStage::None;
    RouteFailureCategory category = RouteFailureCategory::None;
    std::string operation;
    std::string probe_stage;
    int error_code = 0;
    int timeout_ms = 0;
    std::string observed_fact;
    std::string bounded_interpretation;
    std::string recommended_action;
    std::string evidence_id;
};

struct RouteQualificationRecord {
    DeviceType type = DeviceType::CPU;
    int device_id = 0;
    std::string physical_fingerprint;
    std::string provider;
    std::string driver_version;
    std::string runtime_version;
    // Signed pack that supplied this route. CPU routes use base_pack_id.
    std::string pack_id;
    int operation_count = 0;
    int pass_count = 0;
    int unavailable_count = 0;
    int failure_count = 0;
    int timeout_count = 0;
    int crash_count = 0;
    bool certified = false;
    std::string display_name;
    DeviceKind device_kind = DeviceKind::Unknown;
    bool device_kind_known = false;
    std::string identity_source;
    RouteFailureDiagnostic failure;
    std::string benchmark_id;
    int benchmark_sample_count = 0;
    int benchmark_iterations_per_sample = 0;
    double benchmark_median_iteration_ms = 0.0;
    std::string benchmark_message;
};

struct BackendPackQualificationIdentity {
    DeviceType type = DeviceType::CPU;
    std::string pack_id;
};

struct RuntimeQualificationIdentity {
    std::string runtime_set_id;
    std::uint64_t generation = 0;
    std::string base_pack_id;
    std::vector<BackendPackQualificationIdentity> backend_packs;
};

struct RouteQualificationSnapshot {
    int schema = 1;
    std::string matrix_id;
    std::string pack_id;
    std::string runtime_set_id;
    std::uint64_t runtime_generation = 0;
    std::string base_pack_id;
    std::string compute_contract_id;
    std::string operation_manifest_id;
    std::string captured_at;
    std::string report_sha256;
    std::vector<RouteQualificationRecord> routes;
};

struct RouteQualificationLoadResult {
    bool loaded = false;
    std::string matrix_id;
    size_t route_count = 0;
    std::string message;
};

struct RouteQualificationDecision {
    bool qualified = false;
    bool evidence_available = false;
    std::string matrix_id;
    std::string message;
    std::string display_name;
    DeviceKind device_kind = DeviceKind::Unknown;
    bool display_name_available = false;
    bool device_kind_known = false;
    std::string identity_source;
    RouteFailureDiagnostic failure;
};

enum class RouteTrainingAuthorizationStatus {
    Ready,
    NoEvidence,
    MatrixRejected,
    DiagnosticOnly
};

struct RouteTrainingAuthorizationDecision {
    RouteTrainingAuthorizationStatus status =
        RouteTrainingAuthorizationStatus::NoEvidence;
    bool authorized = false;
    std::string message;
    RouteFailureDiagnostic failure;
};

RouteQualificationLoadResult LoadAndInstallRouteQualificationSnapshot(
    const std::filesystem::path& path);
bool SaveRouteQualificationSnapshotAtomic(
    const std::filesystem::path& path,
    const RouteQualificationSnapshot& snapshot,
    std::string& error);
void InstallRouteQualificationSnapshot(RouteQualificationSnapshot snapshot);
void ClearRouteQualificationSnapshot();
std::optional<RouteQualificationSnapshot> GetRouteQualificationSnapshot();
std::optional<RuntimeQualificationIdentity>
ReadActiveRuntimeQualificationIdentity(std::string& error);
std::string RuntimePackIdForRoute(
    const RuntimeQualificationIdentity& identity,
    DeviceType type);
std::string ValidateRuntimeQualificationIdentity(
    const RuntimeQualificationIdentity& identity);
RouteQualificationDecision EvaluateRouteQualification(
    const DeviceInfo& route);
RouteQualificationDecision EvaluateRouteQualification(
    const DeviceInfo& route,
    const std::optional<RouteQualificationSnapshot>& snapshot);
RouteTrainingAuthorizationDecision EvaluateRouteTrainingAuthorization(
    const DeviceInfo& route,
    const RouteQualificationDecision& qualification);
const char* RouteTrainingAuthorizationStatusName(
    RouteTrainingAuthorizationStatus status);
const char* RouteFailureStageName(RouteFailureStage stage);
const char* RouteFailureCategoryName(RouteFailureCategory category);

}  // namespace cyxwiz
