#pragma once

#include "route_qualification_snapshot.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace cyxwiz {

enum class RouteProbeStatus {
    Passed,
    Unavailable,
    Failed,
    TimedOut,
    Crashed,
    Cancelled,
    InfrastructureFailure
};

struct RouteProbeInvocation {
    std::filesystem::path executable;
    DeviceType type = DeviceType::CPU;
    int device_id = 0;
    std::string operation;
    std::chrono::milliseconds timeout{20000};
    size_t output_limit_bytes = 64 * 1024;
    std::filesystem::path runtime_root;
    std::filesystem::path working_directory;
    std::vector<std::filesystem::path> runtime_dll_directories;
    std::optional<RuntimeQualificationIdentity> runtime_identity;
    bool enumerate_backend = false;
};

struct RouteProbeResult {
    RouteProbeStatus status = RouteProbeStatus::InfrastructureFailure;
    int exit_code = 0;
    std::string output;
    std::string last_probe_stage;
    std::string infrastructure_error;
};

struct IsolatedRouteDiscoveryResult {
    RouteProbeStatus status = RouteProbeStatus::InfrastructureFailure;
    std::vector<DeviceInfo> routes;
    std::string message;
};

using RouteQualificationCancelCheck = std::function<bool()>;
using RouteProbeRunner = std::function<RouteProbeResult(
    const RouteProbeInvocation&,
    const RouteQualificationCancelCheck&)>;

enum class RouteQualificationRunStatus {
    Idle,
    Running,
    Completed,
    Cancelled,
    InvalidRequest,
    Busy,
    InfrastructureFailure,
    PublishFailure
};

struct RouteQualificationProgress {
    RouteQualificationRunStatus status = RouteQualificationRunStatus::Idle;
    size_t route_index = 0;
    size_t route_count = 0;
    size_t operation_index = 0;
    size_t operation_count = 0;
    std::string backend;
    int device_id = -1;
    std::string operation;
    std::string message;
};

struct RouteQualificationOptions {
    std::filesystem::path probe_executable;
    std::filesystem::path cache_path;
    std::string matrix_id;
    std::string pack_id;
    std::string runtime_version;
    std::optional<RuntimeQualificationIdentity> runtime_identity;
    std::chrono::milliseconds operation_timeout{20000};
    size_t output_limit_bytes = 64 * 1024;
    bool benchmark_verified_routes = false;
    std::chrono::milliseconds benchmark_timeout{60000};
    std::filesystem::path probe_runtime_root;
    std::filesystem::path probe_working_directory;
    std::vector<std::filesystem::path> probe_runtime_dll_directories;
};

enum class RuntimeQualificationFailurePolicy {
    KeepInstalledUnqualified,
    RequireRollback
};

enum class RuntimeQualificationDisposition {
    Qualified,
    InstalledUnqualified,
    RollbackRequired
};

struct RouteQualificationRunResult {
    RouteQualificationRunStatus status =
        RouteQualificationRunStatus::InvalidRequest;
    bool published = false;
    std::string message;
    std::optional<RouteQualificationSnapshot> snapshot;
};

struct RuntimeQualificationResult {
    RouteQualificationRunResult qualification;
    RuntimeQualificationDisposition disposition =
        RuntimeQualificationDisposition::InstalledUnqualified;
    RouteFailureDiagnostic diagnostic;
};

std::span<const std::string_view> RequiredRouteQualificationOperations();
RouteProbeResult RunIsolatedRouteProbe(
    const RouteProbeInvocation& invocation,
    const RouteQualificationCancelCheck& should_cancel);
IsolatedRouteDiscoveryResult DiscoverIsolatedBackendRoutes(
    RouteProbeInvocation invocation,
    const RouteQualificationCancelCheck& should_cancel = {});

class RouteQualificationService {
public:
    explicit RouteQualificationService(RouteProbeRunner runner = {});

    RouteQualificationRunResult VerifyRoute(
        const DeviceInfo& route,
        const RouteQualificationOptions& options,
        std::function<void(const RouteQualificationProgress&)> on_progress = {});
    RouteQualificationRunResult VerifyAll(
        const std::vector<DeviceInfo>& routes,
        const RouteQualificationOptions& options,
        std::function<void(const RouteQualificationProgress&)> on_progress = {});
    RuntimeQualificationResult VerifyStagedRuntimeRoutes(
        const std::vector<DeviceInfo>& affected_routes,
        const RuntimeQualificationIdentity& identity,
        RuntimeQualificationFailurePolicy failure_policy,
        RouteQualificationOptions options,
        std::function<void(const RouteQualificationProgress&)> on_progress = {});
    RouteQualificationRunResult ReconcileRuntimeEvidence(
        const RuntimeQualificationIdentity& identity,
        const RouteQualificationOptions& options);

    void Cancel();
    RouteQualificationProgress GetProgress() const;
    std::optional<RouteQualificationSnapshot> GetPublishedSnapshot() const;

private:
    RouteQualificationRunResult Verify(
        const std::vector<DeviceInfo>& routes,
        const RouteQualificationOptions& options,
        bool merge_current_snapshot,
        const std::function<void(const RouteQualificationProgress&)>&
            on_progress);
    void SetProgress(
        RouteQualificationProgress progress,
        const std::function<void(const RouteQualificationProgress&)>&
            on_progress);

    RouteProbeRunner runner_;
    std::atomic<bool> cancel_requested_{false};
    mutable std::mutex state_mutex_;
    std::mutex run_mutex_;
    RouteQualificationProgress progress_;
    std::optional<RouteQualificationSnapshot> published_snapshot_;
};

const char* RouteQualificationRunStatusName(
    RouteQualificationRunStatus status);
const char* RuntimeQualificationDispositionName(
    RuntimeQualificationDisposition disposition);

}  // namespace cyxwiz
