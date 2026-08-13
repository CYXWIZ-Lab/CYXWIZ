#include "route_qualification_service.h"

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <af/version.h>
#endif

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <thread>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace cyxwiz {
namespace {

constexpr std::string_view kOperations[] = {
#define CYXWIZ_ROUTE_OPERATION(name) #name,
#include "arrayfire_route_operations.def"
#undef CYXWIZ_ROUTE_OPERATION
};
static_assert(static_cast<int>(std::size(kOperations)) ==
              kRouteQualificationOperationCount);

const char* BackendName(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "cpu";
        case DeviceType::CUDA: return "cuda";
        case DeviceType::OPENCL: return "opencl";
        case DeviceType::ONEAPI: return "oneapi";
        default: return nullptr;
    }
}

std::string CurrentRuntimeVersion() {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    return AF_VERSION;
#else
    return {};
#endif
}

std::string UtcTimestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto value = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &value);
#else
    gmtime_r(&value, &utc);
#endif
    std::ostringstream stream;
    stream << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return stream.str();
}

std::string LastField(const std::string& text, std::string_view prefix) {
    size_t cursor = 0;
    std::string result;
    while ((cursor = text.find(prefix, cursor)) != std::string::npos) {
        cursor += prefix.size();
        const size_t end = text.find_first_of(" \r\n", cursor);
        result = text.substr(cursor, end == std::string::npos
                                        ? std::string::npos
                                        : end - cursor);
    }
    return result;
}

int ParseErrorCode(const std::string& text) {
    const std::string value = LastField(text, "code=");
    int result = 0;
    if (!value.empty()) {
        const auto parsed = std::from_chars(
            value.data(), value.data() + value.size(), result);
        if (parsed.ec != std::errc{}) return 0;
    }
    return result;
}

int ParsePositiveIntField(const std::string& text, std::string_view prefix) {
    const std::string value = LastField(text, prefix);
    int result = 0;
    if (value.empty()) return 0;
    const auto parsed = std::from_chars(
        value.data(), value.data() + value.size(), result);
    return parsed.ec == std::errc{} && result > 0 ? result : 0;
}

double ParsePositiveDoubleField(
    const std::string& text, std::string_view prefix) {
    const std::string value = LastField(text, prefix);
    if (value.empty()) return 0.0;
    std::istringstream stream(value);
    double result = 0.0;
    stream >> result;
    return stream && stream.eof() && result > 0.0 ? result : 0.0;
}

const char* ProbeStatusName(RouteProbeStatus status) {
    switch (status) {
        case RouteProbeStatus::Passed: return "passed";
        case RouteProbeStatus::Unavailable: return "unavailable";
        case RouteProbeStatus::Failed: return "failed";
        case RouteProbeStatus::TimedOut: return "timed out";
        case RouteProbeStatus::Crashed: return "crashed";
        case RouteProbeStatus::Cancelled: return "cancelled";
        default: return "could not run";
    }
}

RouteFailureDiagnostic FailureFor(
    const RouteProbeResult& probe,
    std::string_view operation,
    const std::string& evidence_id,
    int timeout_ms) {
    RouteFailureDiagnostic failure;
    failure.stage = probe.status == RouteProbeStatus::Unavailable
        ? RouteFailureStage::Enumeration
        : RouteFailureStage::Operation;
    failure.operation = std::string(operation);
    failure.probe_stage = !probe.last_probe_stage.empty()
        ? probe.last_probe_stage
        : LastField(probe.output, "stage=");
    failure.error_code = probe.exit_code != 0
        ? probe.exit_code
        : ParseErrorCode(probe.output);
    failure.evidence_id = evidence_id;
    switch (probe.status) {
        case RouteProbeStatus::Unavailable:
            failure.category = RouteFailureCategory::DeviceNotEnumerated;
            failure.observed_fact = "The exact route was unavailable while probing '" +
                std::string(operation) + "'";
            failure.bounded_interpretation =
                "The backend loaded, but this device ordinal did not enumerate";
            failure.recommended_action =
                "Refresh device inventory and verify an enumerated route";
            break;
        case RouteProbeStatus::TimedOut:
            failure.category = RouteFailureCategory::Timeout;
            failure.timeout_ms = timeout_ms;
            failure.observed_fact = "Operation '" + std::string(operation) +
                "' timed out after " + std::to_string(timeout_ms) + " ms";
            failure.bounded_interpretation =
                "The exact provider/device/runtime route did not complete the released operation contract";
            failure.recommended_action =
                "Update the provider or driver and verify again, or select another verified route";
            break;
        case RouteProbeStatus::Crashed:
            failure.category = RouteFailureCategory::ChildProcessCrash;
            failure.observed_fact = "Operation '" + std::string(operation) +
                "' terminated its isolated child process";
            failure.bounded_interpretation =
                "The exact provider/device/runtime route crashed during the released operation contract";
            failure.recommended_action =
                "Update the provider or driver and verify again, or select another verified route";
            break;
        default:
            failure.category = RouteFailureCategory::OperationFailed;
            failure.observed_fact = "Operation '" + std::string(operation) +
                "' returned an error";
            failure.bounded_interpretation =
                "The exact provider/device/runtime route did not pass the released operation contract";
            failure.recommended_action =
                "Review the error evidence, then verify again or select another verified route";
            break;
    }
    if (!failure.probe_stage.empty()) {
        failure.observed_fact += " at probe stage '" +
            failure.probe_stage + "'";
    }
    return failure;
}

RouteQualificationRecord RecordFor(const DeviceInfo& route) {
    RouteQualificationRecord record;
    record.type = route.type;
    record.device_id = route.device_id;
    record.physical_fingerprint = route.physical_fingerprint_known
        ? route.physical_fingerprint
        : std::string{};
    record.provider = route.provider_known ? route.provider : std::string{};
    record.driver_version = route.driver_version_known
        ? route.driver_version
        : std::string{};
    record.runtime_version = CurrentRuntimeVersion();
    record.display_name = route.name_known && !route.name_is_fallback
        ? route.name
        : std::string{};
    record.device_kind = route.kind;
    record.device_kind_known = route.kind != DeviceKind::Unknown;
    record.identity_source = route.physical_fingerprint_known
        ? "device_inventory_stable_identity"
        : (route.metadata_status == DeviceMetadataStatus::Available
               ? "arrayfire_device_metadata"
               : std::string{});
    record.operation_count = static_cast<int>(std::size(kOperations));
    return record;
}

#ifdef _WIN32
std::wstring QuoteWindowsArgument(const std::wstring& value) {
    std::wstring quoted = L"\"";
    size_t slashes = 0;
    for (wchar_t character : value) {
        if (character == L'\\') {
            ++slashes;
        } else if (character == L'\"') {
            quoted.append(slashes * 2 + 1, L'\\');
            quoted.push_back(character);
            slashes = 0;
        } else {
            quoted.append(slashes, L'\\');
            slashes = 0;
            quoted.push_back(character);
        }
    }
    quoted.append(slashes * 2, L'\\');
    quoted.push_back(L'\"');
    return quoted;
}

void DrainPipe(HANDLE pipe, std::string& output, size_t limit) {
    std::array<char, 4096> buffer{};
    for (;;) {
        DWORD available = 0;
        if (!PeekNamedPipe(pipe, nullptr, 0, nullptr, &available, nullptr) ||
            available == 0) {
            return;
        }
        const DWORD wanted = static_cast<DWORD>(std::min<size_t>(
            {available, buffer.size(), limit - output.size()}));
        if (wanted == 0) return;
        DWORD read = 0;
        if (!ReadFile(pipe, buffer.data(), wanted, &read, nullptr) || read == 0)
            return;
        output.append(buffer.data(), read);
    }
}
#endif

}  // namespace

std::span<const std::string_view> RequiredRouteQualificationOperations() {
    return kOperations;
}

RouteProbeResult RunIsolatedRouteProbe(
    const RouteProbeInvocation& invocation,
    const RouteQualificationCancelCheck& should_cancel) {
    RouteProbeResult result;
    const char* backend = BackendName(invocation.type);
    if (!backend || invocation.device_id < 0 || invocation.operation.empty() ||
        invocation.executable.empty() ||
        !std::filesystem::is_regular_file(invocation.executable) ||
        invocation.timeout.count() <= 0 || invocation.output_limit_bytes == 0) {
        result.infrastructure_error = "Invalid isolated route probe invocation";
        return result;
    }

#ifdef _WIN32
    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE read_pipe = nullptr;
    HANDLE write_pipe = nullptr;
    if (!CreatePipe(&read_pipe, &write_pipe, &security, 0) ||
        !SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0)) {
        if (read_pipe) CloseHandle(read_pipe);
        if (write_pipe) CloseHandle(write_pipe);
        result.infrastructure_error = "Could not create isolated probe output pipe";
        return result;
    }
    const auto close_pipes = [&] {
        if (read_pipe) CloseHandle(read_pipe);
        if (write_pipe) CloseHandle(write_pipe);
    };

    const std::wstring backend_w(backend, backend + std::char_traits<char>::length(backend));
    const std::wstring operation_w(
        invocation.operation.begin(), invocation.operation.end());
    std::wstring command = QuoteWindowsArgument(invocation.executable.wstring()) +
        L" --backend " + backend_w + L" --device " +
        std::to_wstring(invocation.device_id) + L" --operation " + operation_w;
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdOutput = write_pipe;
    startup.hStdError = write_pipe;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(
            invocation.executable.c_str(), command.data(), nullptr, nullptr,
            TRUE, CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP, nullptr,
            invocation.executable.parent_path().c_str(), &startup, &process)) {
        close_pipes();
        result.infrastructure_error = "Could not launch isolated route probe: " +
            std::system_category().message(static_cast<int>(GetLastError()));
        return result;
    }
    CloseHandle(write_pipe);
    write_pipe = nullptr;

    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (job) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                                     &limits, sizeof(limits)) ||
            !AssignProcessToJobObject(job, process.hProcess)) {
            CloseHandle(job);
            job = nullptr;
        }
    }
    const auto terminate = [&] {
        if (job) TerminateJobObject(job, 1);
        else TerminateProcess(process.hProcess, 1);
    };

    const auto deadline = std::chrono::steady_clock::now() + invocation.timeout;
    bool done = false;
    while (!done) {
        DrainPipe(read_pipe, result.output, invocation.output_limit_bytes);
        if (result.output.size() >= invocation.output_limit_bytes) {
            terminate();
            result.infrastructure_error = "Isolated route probe exceeded its output limit";
            break;
        }
        if (should_cancel && should_cancel()) {
            terminate();
            result.status = RouteProbeStatus::Cancelled;
            break;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            terminate();
            result.status = RouteProbeStatus::TimedOut;
            break;
        }
        done = WaitForSingleObject(process.hProcess, 10) == WAIT_OBJECT_0;
    }
    WaitForSingleObject(process.hProcess, 2000);
    DrainPipe(read_pipe, result.output, invocation.output_limit_bytes);
    DWORD exit_code = 0;
    GetExitCodeProcess(process.hProcess, &exit_code);
    result.exit_code = static_cast<int>(exit_code);
    if (done && result.infrastructure_error.empty()) {
        if (exit_code == 0) result.status = RouteProbeStatus::Passed;
        else if (exit_code == 77) result.status = RouteProbeStatus::Unavailable;
        else if ((exit_code & 0x80000000UL) != 0)
            result.status = RouteProbeStatus::Crashed;
        else result.status = RouteProbeStatus::Failed;
    }
    result.last_probe_stage = LastField(result.output, "stage=");
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    if (job) CloseHandle(job);
    close_pipes();
    return result;
#else
    int pipe_fds[2] = {-1, -1};
    if (pipe(pipe_fds) != 0) {
        result.infrastructure_error = "Could not create isolated probe output pipe";
        return result;
    }
    const pid_t child = fork();
    if (child == 0) {
        setpgid(0, 0);
        dup2(pipe_fds[1], STDOUT_FILENO);
        dup2(pipe_fds[1], STDERR_FILENO);
        close(pipe_fds[0]);
        close(pipe_fds[1]);
        const std::string device_id = std::to_string(invocation.device_id);
        execl(invocation.executable.c_str(), invocation.executable.c_str(),
              "--backend", backend, "--device", device_id.c_str(),
              "--operation", invocation.operation.c_str(), nullptr);
        _exit(127);
    }
    close(pipe_fds[1]);
    if (child < 0) {
        close(pipe_fds[0]);
        result.infrastructure_error = "Could not launch isolated route probe";
        return result;
    }
    fcntl(pipe_fds[0], F_SETFL, fcntl(pipe_fds[0], F_GETFL) | O_NONBLOCK);
    const auto deadline = std::chrono::steady_clock::now() + invocation.timeout;
    int wait_status = 0;
    bool done = false;
    std::array<char, 4096> buffer{};
    while (!done) {
        const ssize_t count = read(pipe_fds[0], buffer.data(),
                                   std::min(buffer.size(),
                                            invocation.output_limit_bytes - result.output.size()));
        if (count > 0) result.output.append(buffer.data(), static_cast<size_t>(count));
        if (result.output.size() >= invocation.output_limit_bytes) {
            kill(-child, SIGKILL);
            result.infrastructure_error = "Isolated route probe exceeded its output limit";
            break;
        }
        if (should_cancel && should_cancel()) {
            kill(-child, SIGKILL);
            result.status = RouteProbeStatus::Cancelled;
            break;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            kill(-child, SIGKILL);
            result.status = RouteProbeStatus::TimedOut;
            break;
        }
        const pid_t waited = waitpid(child, &wait_status, WNOHANG);
        done = waited == child;
        if (!done) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!done) waitpid(child, &wait_status, 0);
    for (;;) {
        const size_t remaining = invocation.output_limit_bytes - result.output.size();
        if (remaining == 0) break;
        const ssize_t count = read(pipe_fds[0], buffer.data(),
                                   std::min(buffer.size(), remaining));
        if (count <= 0) break;
        result.output.append(buffer.data(), static_cast<size_t>(count));
    }
    close(pipe_fds[0]);
    if (done && result.infrastructure_error.empty()) {
        if (WIFSIGNALED(wait_status)) {
            result.status = RouteProbeStatus::Crashed;
            result.exit_code = 128 + WTERMSIG(wait_status);
        } else {
            result.exit_code = WEXITSTATUS(wait_status);
            if (result.exit_code == 0) result.status = RouteProbeStatus::Passed;
            else if (result.exit_code == 77) result.status = RouteProbeStatus::Unavailable;
            else result.status = RouteProbeStatus::Failed;
        }
    }
    result.last_probe_stage = LastField(result.output, "stage=");
    return result;
#endif
}

RouteQualificationService::RouteQualificationService(RouteProbeRunner runner)
    : runner_(runner ? std::move(runner) : RunIsolatedRouteProbe) {}

RouteQualificationRunResult RouteQualificationService::VerifyRoute(
    const DeviceInfo& route,
    const RouteQualificationOptions& options,
    std::function<void(const RouteQualificationProgress&)> on_progress) {
    return Verify({route}, options, true, on_progress);
}

RouteQualificationRunResult RouteQualificationService::VerifyAll(
    const std::vector<DeviceInfo>& routes,
    const RouteQualificationOptions& options,
    std::function<void(const RouteQualificationProgress&)> on_progress) {
    return Verify(routes, options, false, on_progress);
}

RouteQualificationRunResult RouteQualificationService::Verify(
    const std::vector<DeviceInfo>& routes,
    const RouteQualificationOptions& options,
    bool merge_current_snapshot,
    const std::function<void(const RouteQualificationProgress&)>& on_progress) {
    std::unique_lock<std::mutex> run_lock(run_mutex_, std::try_to_lock);
    if (!run_lock.owns_lock()) {
        return {RouteQualificationRunStatus::Busy, false,
                "Route qualification is already running", std::nullopt};
    }
    if (routes.empty() || options.matrix_id.empty() ||
        options.cache_path.empty() || options.probe_executable.empty() ||
        options.operation_timeout.count() <= 0 ||
        options.output_limit_bytes == 0 ||
        (options.benchmark_verified_routes &&
         options.benchmark_timeout.count() <= 0)) {
        return {RouteQualificationRunStatus::InvalidRequest, false,
                "Route qualification request is incomplete", std::nullopt};
    }
    for (size_t route_index = 0; route_index < routes.size(); ++route_index) {
        const auto& route = routes[route_index];
        if (!BackendName(route.type) || route.device_id < 0) {
            return {RouteQualificationRunStatus::InvalidRequest, false,
                    "Route qualification contains an unsupported route",
                    std::nullopt};
        }
        const auto duplicate = std::find_if(
            routes.begin(), routes.begin() + route_index,
            [&](const DeviceInfo& candidate) {
                return candidate.type == route.type &&
                       candidate.device_id == route.device_id;
            });
        if (duplicate != routes.begin() + route_index) {
            return {RouteQualificationRunStatus::InvalidRequest, false,
                    "Route qualification contains a duplicate exact route",
                    std::nullopt};
        }
    }

    cancel_requested_.store(false);
    RouteQualificationSnapshot snapshot;
    if (merge_current_snapshot) {
        const auto current = GetRouteQualificationSnapshot();
        if (current.has_value() && current->pack_id == options.pack_id &&
            current->compute_contract_id == kCyxWizComputeContractId &&
            current->operation_manifest_id ==
                kRouteQualificationOperationManifestId) {
            snapshot = *current;
        }
    }
    snapshot.schema = 1;
    snapshot.matrix_id = options.matrix_id;
    snapshot.pack_id = options.pack_id;
    snapshot.compute_contract_id = kCyxWizComputeContractId;
    snapshot.operation_manifest_id =
        kRouteQualificationOperationManifestId;
    snapshot.captured_at = UtcTimestamp();
    snapshot.report_sha256.clear();

    const auto operations = RequiredRouteQualificationOperations();
    for (size_t route_index = 0; route_index < routes.size(); ++route_index) {
        const auto& route = routes[route_index];
        RouteQualificationRecord record = RecordFor(route);
        for (size_t operation_index = 0;
             operation_index < operations.size(); ++operation_index) {
            RouteQualificationProgress progress;
            progress.status = RouteQualificationRunStatus::Running;
            progress.route_index = route_index;
            progress.route_count = routes.size();
            progress.operation_index = operation_index;
            progress.operation_count = operations.size();
            progress.backend = BackendName(route.type);
            progress.device_id = route.device_id;
            progress.operation = std::string(operations[operation_index]);
            progress.message = "Verifying exact compute route";
            SetProgress(progress, on_progress);

            RouteProbeInvocation invocation;
            invocation.executable = options.probe_executable;
            invocation.type = route.type;
            invocation.device_id = route.device_id;
            invocation.operation = progress.operation;
            invocation.timeout = options.operation_timeout;
            invocation.output_limit_bytes = options.output_limit_bytes;
            const auto probe = runner_(invocation, [this] {
                return cancel_requested_.load();
            });
            if (probe.status == RouteProbeStatus::Cancelled ||
                cancel_requested_.load()) {
                progress.status = RouteQualificationRunStatus::Cancelled;
                progress.message = "Route qualification cancelled; accepted evidence was unchanged";
                SetProgress(progress, on_progress);
                return {progress.status, false, progress.message, std::nullopt};
            }
            if (probe.status == RouteProbeStatus::InfrastructureFailure) {
                progress.status = RouteQualificationRunStatus::InfrastructureFailure;
                progress.message = probe.infrastructure_error.empty()
                    ? "The isolated route probe could not run"
                    : probe.infrastructure_error;
                SetProgress(progress, on_progress);
                return {progress.status, false, progress.message, std::nullopt};
            }
            switch (probe.status) {
                case RouteProbeStatus::Passed: ++record.pass_count; break;
                case RouteProbeStatus::Unavailable: ++record.unavailable_count; break;
                case RouteProbeStatus::Failed: ++record.failure_count; break;
                case RouteProbeStatus::TimedOut: ++record.timeout_count; break;
                case RouteProbeStatus::Crashed: ++record.crash_count; break;
                default: break;
            }
            if (probe.status != RouteProbeStatus::Passed &&
                record.failure.category == RouteFailureCategory::None) {
                record.failure = FailureFor(
                    probe, operations[operation_index], options.matrix_id,
                    static_cast<int>(std::min<int64_t>(
                        options.operation_timeout.count(),
                        (std::numeric_limits<int>::max)())));
            }
        }
        record.certified = record.pass_count == record.operation_count;
        if (options.benchmark_verified_routes && record.certified) {
            RouteQualificationProgress progress;
            progress.status = RouteQualificationRunStatus::Running;
            progress.route_index = route_index;
            progress.route_count = routes.size();
            progress.operation_index = operations.size();
            progress.operation_count = operations.size();
            progress.backend = BackendName(route.type);
            progress.device_id = route.device_id;
            progress.operation = "dense_compute_benchmark";
            progress.message =
                "Benchmarking this verified route with a fixed dense workload";
            SetProgress(progress, on_progress);

            RouteProbeInvocation invocation;
            invocation.executable = options.probe_executable;
            invocation.type = route.type;
            invocation.device_id = route.device_id;
            invocation.operation = "dense_compute_benchmark";
            invocation.timeout = options.benchmark_timeout;
            invocation.output_limit_bytes = options.output_limit_bytes;
            const auto benchmark = runner_(invocation, [this] {
                return cancel_requested_.load();
            });
            if (benchmark.status == RouteProbeStatus::Cancelled ||
                cancel_requested_.load()) {
                progress.status = RouteQualificationRunStatus::Cancelled;
                progress.message =
                    "Route verification cancelled; accepted evidence was unchanged";
                SetProgress(progress, on_progress);
                return {progress.status, false, progress.message, std::nullopt};
            }

            const std::string benchmark_id =
                LastField(benchmark.output, "benchmark_id=");
            const int sample_count = ParsePositiveIntField(
                benchmark.output, "samples=");
            const int iterations_per_sample = ParsePositiveIntField(
                benchmark.output, "iterations_per_sample=");
            const double median_ms = ParsePositiveDoubleField(
                benchmark.output, "median_iteration_ms=");
            if (benchmark.status == RouteProbeStatus::Passed &&
                benchmark_id == kRoutePerformanceBenchmarkId &&
                sample_count > 0 && iterations_per_sample > 0 &&
                median_ms > 0.0) {
                record.benchmark_id = benchmark_id;
                record.benchmark_sample_count = sample_count;
                record.benchmark_iterations_per_sample =
                    iterations_per_sample;
                record.benchmark_median_iteration_ms = median_ms;
                record.benchmark_message =
                    "Fixed dense compute benchmark completed";
            } else {
                record.benchmark_message =
                    std::string("Performance benchmark ") +
                    ProbeStatusName(benchmark.status);
                if (!benchmark.infrastructure_error.empty()) {
                    record.benchmark_message += ": " +
                        benchmark.infrastructure_error;
                } else if (benchmark.status == RouteProbeStatus::Passed) {
                    record.benchmark_message +=
                        ": result fields were incomplete";
                }
            }
        } else if (options.benchmark_verified_routes) {
            record.benchmark_message =
                "Not benchmarked because this route failed verification";
        }
        const auto existing = std::find_if(
            snapshot.routes.begin(), snapshot.routes.end(),
            [&](const RouteQualificationRecord& candidate) {
                return candidate.type == record.type &&
                       candidate.device_id == record.device_id;
            });
        if (existing == snapshot.routes.end()) snapshot.routes.push_back(std::move(record));
        else *existing = std::move(record);
    }

    std::string publish_error;
    if (!SaveRouteQualificationSnapshotAtomic(
            options.cache_path, snapshot, publish_error)) {
        RouteQualificationProgress progress;
        progress.status = RouteQualificationRunStatus::PublishFailure;
        progress.route_count = routes.size();
        progress.operation_count = operations.size();
        progress.message = publish_error;
        SetProgress(progress, on_progress);
        return {progress.status, false, publish_error, std::nullopt};
    }
    InstallRouteQualificationSnapshot(snapshot);
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        published_snapshot_ = snapshot;
    }
    RouteQualificationProgress complete;
    complete.status = RouteQualificationRunStatus::Completed;
    complete.route_index = routes.size();
    complete.route_count = routes.size();
    complete.operation_index = 0;
    complete.operation_count = operations.size();
    complete.message = "Route qualification evidence published";
    SetProgress(complete, on_progress);
    return {complete.status, true, complete.message, std::move(snapshot)};
}

void RouteQualificationService::Cancel() {
    cancel_requested_.store(true);
}

RouteQualificationProgress RouteQualificationService::GetProgress() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return progress_;
}

std::optional<RouteQualificationSnapshot>
RouteQualificationService::GetPublishedSnapshot() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return published_snapshot_;
}

void RouteQualificationService::SetProgress(
    RouteQualificationProgress progress,
    const std::function<void(const RouteQualificationProgress&)>& on_progress) {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        progress_ = progress;
    }
    if (on_progress) on_progress(progress);
}

const char* RouteQualificationRunStatusName(
    RouteQualificationRunStatus status) {
    switch (status) {
        case RouteQualificationRunStatus::Idle: return "idle";
        case RouteQualificationRunStatus::Running: return "running";
        case RouteQualificationRunStatus::Completed: return "completed";
        case RouteQualificationRunStatus::Cancelled: return "cancelled";
        case RouteQualificationRunStatus::InvalidRequest: return "invalid_request";
        case RouteQualificationRunStatus::Busy: return "busy";
        case RouteQualificationRunStatus::InfrastructureFailure:
            return "infrastructure_failure";
        case RouteQualificationRunStatus::PublishFailure: return "publish_failure";
        default: return "unknown";
    }
}

}  // namespace cyxwiz
