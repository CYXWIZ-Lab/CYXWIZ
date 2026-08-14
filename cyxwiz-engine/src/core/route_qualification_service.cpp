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

#include <nlohmann/json.hpp>

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

bool IsCompleteRuntimeIdentity(
    const RuntimeQualificationIdentity& identity) {
    return ValidateRuntimeQualificationIdentity(identity).empty();
}

bool IsCompatibleRuntimeSnapshot(
    const RouteQualificationSnapshot& snapshot,
    const RuntimeQualificationIdentity& identity) {
    return snapshot.runtime_set_id == identity.runtime_set_id &&
           snapshot.base_pack_id == identity.base_pack_id &&
           snapshot.compute_contract_id == kCyxWizComputeContractId &&
           snapshot.operation_manifest_id ==
               kRouteQualificationOperationManifestId;
}

void RemoveEvidenceForChangedPacks(
    RouteQualificationSnapshot& snapshot,
    const RuntimeQualificationIdentity& identity) {
    snapshot.routes.erase(
        std::remove_if(
            snapshot.routes.begin(), snapshot.routes.end(),
            [&](const RouteQualificationRecord& record) {
                const std::string active_pack =
                    RuntimePackIdForRoute(identity, record.type);
                return active_pack.empty() || record.pack_id != active_pack;
            }),
        snapshot.routes.end());
}

bool HasRuntimeOverride(const RouteProbeInvocation& invocation) {
    return !invocation.runtime_root.empty() ||
           !invocation.working_directory.empty() ||
           !invocation.runtime_dll_directories.empty();
}

bool HasExactKeys(
    const nlohmann::json& value,
    std::initializer_list<std::string_view> keys) {
    if (!value.is_object() || value.size() != keys.size()) return false;
    return std::all_of(keys.begin(), keys.end(), [&](std::string_view key) {
        return value.contains(std::string(key));
    });
}

bool ReadBoundedOptionalText(
    const nlohmann::json& object,
    const char* key,
    size_t maximum_size,
    std::string& output,
    bool& known) {
    const auto& value = object.at(key);
    output.clear();
    known = false;
    if (value.is_null()) return true;
    if (!value.is_string()) return false;
    output = value.get<std::string>();
    if (output.empty() || output.size() > maximum_size ||
        std::any_of(output.begin(), output.end(), [](unsigned char character) {
            return character < 0x20;
        })) {
        output.clear();
        return false;
    }
    known = true;
    return true;
}

bool ParseDeviceKind(const nlohmann::json& value, DeviceKind& output) {
    if (!value.is_string()) return false;
    const auto name = value.get<std::string>();
    if (name == "unknown") output = DeviceKind::Unknown;
    else if (name == "cpu") output = DeviceKind::CPU;
    else if (name == "gpu") output = DeviceKind::GPU;
    else if (name == "accelerator") output = DeviceKind::Accelerator;
    else return false;
    return true;
}

bool ParseIdentityConfidence(
    const nlohmann::json& value,
    DeviceIdentityConfidence& output) {
    if (!value.is_string()) return false;
    const auto name = value.get<std::string>();
    if (name == "unknown") output = DeviceIdentityConfidence::Unknown;
    else if (name == "backend_local") {
        output = DeviceIdentityConfidence::BackendLocal;
    } else if (name == "provider_reported") {
        output = DeviceIdentityConfidence::ProviderReported;
    } else if (name == "stable_hardware") {
        output = DeviceIdentityConfidence::StableHardware;
    } else return false;
    return true;
}

bool ParseMetadataStatus(
    const nlohmann::json& value,
    DeviceMetadataStatus& output) {
    if (!value.is_string()) return false;
    const auto name = value.get<std::string>();
    if (name == "not_queried") output = DeviceMetadataStatus::NotQueried;
    else if (name == "available") output = DeviceMetadataStatus::Available;
    else if (name == "unsupported") output = DeviceMetadataStatus::Unsupported;
    else if (name == "failed") output = DeviceMetadataStatus::Failed;
    else return false;
    return true;
}

bool ParseRouteInventory(
    const std::string& output,
    DeviceType type,
    std::vector<DeviceInfo>& routes,
    std::string& error) {
    constexpr std::string_view prefix = "route_inventory_json=";
    const auto prefix_position = output.rfind(prefix);
    if (prefix_position == std::string::npos) {
        error = "Isolated route inventory output is missing";
        return false;
    }
    const auto start = prefix_position + prefix.size();
    const auto end = output.find_first_of("\r\n", start);
    nlohmann::json document;
    try {
        document = nlohmann::json::parse(
            output.substr(start, end == std::string::npos
                                     ? std::string::npos
                                     : end - start));
    } catch (const std::exception& exception) {
        error = std::string("Isolated route inventory is not valid JSON: ") +
                exception.what();
        return false;
    }
    const char* backend = BackendName(type);
    if (!backend ||
        !HasExactKeys(document, {"schema_version", "backend", "routes"}) ||
        !document["schema_version"].is_number_unsigned() ||
        document["schema_version"].get<std::uint64_t>() != 1 ||
        !document["backend"].is_string() ||
        document["backend"].get<std::string>() != backend ||
        !document["routes"].is_array() ||
        document["routes"].size() > 64) {
        error = "Isolated route inventory envelope is invalid";
        return false;
    }
    routes.clear();
    routes.reserve(document["routes"].size());
    int expected_device_id = 0;
    for (const auto& route : document["routes"]) {
        if (!HasExactKeys(
                route,
                {"device_id", "name", "kind", "identity_confidence",
                 "provider", "driver_version", "physical_fingerprint",
                 "metadata_status", "metadata_error_code",
                 "metadata_message"}) ||
            !route["device_id"].is_number_unsigned() ||
            route["device_id"].get<std::uint64_t>() !=
                static_cast<std::uint64_t>(expected_device_id)) {
            error = "Isolated route inventory entry is invalid";
            return false;
        }
        DeviceInfo parsed;
        parsed.type = type;
        parsed.device_id = expected_device_id++;
        parsed.backend_available = true;
        parsed.device_selectable = true;
        bool metadata_message_known = false;
        if (!ReadBoundedOptionalText(
                route, "name", 512, parsed.name, parsed.name_known) ||
            !ParseDeviceKind(route["kind"], parsed.kind) ||
            !ParseIdentityConfidence(
                route["identity_confidence"], parsed.identity_confidence) ||
            !ReadBoundedOptionalText(
                route, "provider", 512, parsed.provider,
                parsed.provider_known) ||
            !ReadBoundedOptionalText(
                route, "driver_version", 512, parsed.driver_version,
                parsed.driver_version_known) ||
            !ReadBoundedOptionalText(
                route, "physical_fingerprint", 512,
                parsed.physical_fingerprint,
                parsed.physical_fingerprint_known) ||
            !ParseMetadataStatus(
                route["metadata_status"], parsed.metadata_status) ||
            !ReadBoundedOptionalText(
                route, "metadata_message", 1024,
                parsed.metadata_message, metadata_message_known)) {
            error = "Isolated route inventory metadata is invalid";
            return false;
        }
        std::int64_t metadata_error = 0;
        const auto& metadata_error_value = route["metadata_error_code"];
        if (metadata_error_value.is_number_unsigned()) {
            const auto unsigned_error =
                metadata_error_value.get<std::uint64_t>();
            if (unsigned_error >
                static_cast<std::uint64_t>(
                    (std::numeric_limits<int>::max)())) {
                error = "Isolated route inventory error code is invalid";
                return false;
            }
            metadata_error = static_cast<std::int64_t>(unsigned_error);
        } else if (metadata_error_value.is_number_integer()) {
            metadata_error = metadata_error_value.get<std::int64_t>();
        } else {
            error = "Isolated route inventory error code is invalid";
            return false;
        }
        if (metadata_error < (std::numeric_limits<int>::min)() ||
            metadata_error > (std::numeric_limits<int>::max)()) {
            error = "Isolated route inventory error code is invalid";
            return false;
        }
        parsed.metadata_error_code = static_cast<int>(metadata_error);
        parsed.name_is_fallback = !parsed.name_known;
        routes.push_back(std::move(parsed));
    }
    return true;
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

std::wstring WidenIdentity(const std::string& value) {
    return std::wstring(value.begin(), value.end());
}

bool BuildRuntimeEnvironment(
    const RouteProbeInvocation& invocation,
    std::vector<wchar_t>& output,
    std::string& error) {
    if (!invocation.runtime_identity.has_value() ||
        !IsCompleteRuntimeIdentity(*invocation.runtime_identity) ||
        invocation.runtime_root.empty() ||
        !invocation.runtime_root.is_absolute() ||
        invocation.working_directory.empty() ||
        !invocation.working_directory.is_absolute() ||
        invocation.runtime_dll_directories.empty()) {
        error = "Candidate probe runtime identity and paths are incomplete";
        return false;
    }
    std::error_code filesystem_error;
    const auto canonical_root = std::filesystem::weakly_canonical(
        invocation.runtime_root, filesystem_error);
    if (filesystem_error ||
        !std::filesystem::is_directory(canonical_root, filesystem_error) ||
        filesystem_error) {
        error = "Candidate probe runtime root is unavailable";
        return false;
    }
    const auto contained = [&](const std::filesystem::path& path) {
        const auto canonical =
            std::filesystem::weakly_canonical(path, filesystem_error);
        if (filesystem_error) return false;
        const auto relative = canonical.lexically_relative(canonical_root);
        return !relative.empty() && !relative.is_absolute() &&
               std::none_of(
                   relative.begin(), relative.end(),
                   [](const std::filesystem::path& component) {
                       return component == "..";
                   });
    };
    if (!std::filesystem::is_directory(
            invocation.working_directory, filesystem_error) ||
        filesystem_error || !contained(invocation.working_directory) ||
        !std::filesystem::is_regular_file(
            invocation.executable, filesystem_error) || filesystem_error ||
        !contained(invocation.executable)) {
        error =
            "Candidate probe executable or working directory is outside the runtime root";
        return false;
    }
    std::wstring path_value;
    for (const auto& directory : invocation.runtime_dll_directories) {
        if (directory.empty() || !directory.is_absolute() ||
            !std::filesystem::is_directory(directory, filesystem_error) ||
            filesystem_error || !contained(directory)) {
            error = "Candidate probe DLL directory is unavailable";
            return false;
        }
        if (!path_value.empty()) path_value.push_back(L';');
        path_value += directory.native();
    }
    std::array<wchar_t, MAX_PATH + 1> system_directory{};
    const UINT system_length = ::GetSystemDirectoryW(
        system_directory.data(), static_cast<UINT>(system_directory.size()));
    if (system_length == 0 || system_length >= system_directory.size()) {
        error = "Cannot resolve the Windows system directory for the candidate probe";
        return false;
    }
    path_value.push_back(L';');
    path_value.append(system_directory.data(), system_length);

    constexpr std::wstring_view removed_names[] = {
        L"PATH", L"CYXWIZ_ACTIVE_RUNTIME_ROOT", L"CYXWIZ_RUNTIME_SET_ID",
        L"CYXWIZ_RUNTIME_GENERATION", L"CYXWIZ_BASE_PACK_ID",
        L"CYXWIZ_RUNTIME_PACK_CUDA", L"CYXWIZ_RUNTIME_PACK_OPENCL",
        L"CYXWIZ_RUNTIME_PACK_ONEAPI", L"AF_PATH", L"AF_PLUGIN_PATH",
        L"CYXWIZ_ARRAYFIRE_DIR", L"AF_BUILD_PATH",
        L"AF_BUILD_LIB_CUSTOM_PATH", L"PYTHONHOME", L"PYTHONPATH"};
    const auto removed = [&](std::wstring_view name) {
        return std::any_of(
            std::begin(removed_names), std::end(removed_names),
            [&](std::wstring_view candidate) {
                return _wcsicmp(
                           std::wstring(name).c_str(),
                           std::wstring(candidate).c_str()) == 0;
            });
    };

    std::vector<std::wstring> variables;
    wchar_t* environment = ::GetEnvironmentStringsW();
    if (!environment) {
        error = "Cannot read the process environment for the candidate probe";
        return false;
    }
    for (const wchar_t* cursor = environment; *cursor != L'\0';) {
        std::wstring variable(cursor);
        cursor += variable.size() + 1;
        const auto separator = variable.find(L'=');
        if (separator == 0 || separator == std::wstring::npos ||
            removed(std::wstring_view(variable).substr(0, separator))) {
            continue;
        }
        variables.push_back(std::move(variable));
    }
    ::FreeEnvironmentStringsW(environment);

    const auto& identity = *invocation.runtime_identity;
    variables.push_back(L"PATH=" + path_value);
    variables.push_back(
        L"CYXWIZ_ACTIVE_RUNTIME_ROOT=" + invocation.runtime_root.native());
    variables.push_back(
        L"CYXWIZ_RUNTIME_SET_ID=" + WidenIdentity(identity.runtime_set_id));
    variables.push_back(
        L"CYXWIZ_RUNTIME_GENERATION=" +
        std::to_wstring(identity.generation));
    variables.push_back(
        L"CYXWIZ_BASE_PACK_ID=" + WidenIdentity(identity.base_pack_id));
    for (const auto& pack : identity.backend_packs) {
        const wchar_t* name = pack.type == DeviceType::CUDA
            ? L"CYXWIZ_RUNTIME_PACK_CUDA="
            : pack.type == DeviceType::OPENCL
                ? L"CYXWIZ_RUNTIME_PACK_OPENCL="
                : pack.type == DeviceType::ONEAPI
                    ? L"CYXWIZ_RUNTIME_PACK_ONEAPI="
                    : nullptr;
        if (!name) {
            error = "Candidate probe runtime contains an unsupported pack identity";
            return false;
        }
        variables.push_back(std::wstring(name) + WidenIdentity(pack.pack_id));
    }
    std::sort(
        variables.begin(), variables.end(),
        [](const std::wstring& left, const std::wstring& right) {
            return _wcsicmp(left.c_str(), right.c_str()) < 0;
        });
    size_t characters = 1;
    for (const auto& variable : variables) {
        characters += variable.size() + 1;
    }
    output.clear();
    output.reserve(characters);
    for (const auto& variable : variables) {
        output.insert(output.end(), variable.begin(), variable.end());
        output.push_back(L'\0');
    }
    output.push_back(L'\0');
    return true;
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
    if (!backend || invocation.device_id < 0 ||
        (!invocation.enumerate_backend && invocation.operation.empty()) ||
        invocation.executable.empty() ||
        !std::filesystem::is_regular_file(invocation.executable) ||
        invocation.timeout.count() <= 0 || invocation.output_limit_bytes == 0) {
        result.infrastructure_error = "Invalid isolated route probe invocation";
        return result;
    }

#ifdef _WIN32
    std::vector<wchar_t> runtime_environment;
    std::string runtime_environment_error;
    const bool runtime_override = HasRuntimeOverride(invocation);
    if (runtime_override && !BuildRuntimeEnvironment(
            invocation, runtime_environment, runtime_environment_error)) {
        result.infrastructure_error = std::move(runtime_environment_error);
        return result;
    }
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
    std::wstring command = QuoteWindowsArgument(invocation.executable.wstring());
    if (invocation.enumerate_backend) {
        command += L" --enumerate-backend " + backend_w;
    } else {
        const std::wstring operation_w(
            invocation.operation.begin(), invocation.operation.end());
        command += L" --backend " + backend_w + L" --device " +
            std::to_wstring(invocation.device_id) + L" --operation " +
            operation_w;
    }
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdOutput = write_pipe;
    startup.hStdError = write_pipe;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION process{};
    const auto working_directory = runtime_override
        ? invocation.working_directory
        : invocation.executable.parent_path();
    if (!CreateProcessW(
            invocation.executable.c_str(), command.data(), nullptr, nullptr,
            TRUE, CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP |
                      (runtime_override ? CREATE_UNICODE_ENVIRONMENT : 0),
            runtime_override ? runtime_environment.data() : nullptr,
            working_directory.c_str(), &startup, &process)) {
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
    if (HasRuntimeOverride(invocation)) {
        result.infrastructure_error =
            "Candidate probe runtime overrides are currently supported on Windows";
        return result;
    }
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
        if (invocation.enumerate_backend) {
            execl(invocation.executable.c_str(), invocation.executable.c_str(),
                  "--enumerate-backend", backend, nullptr);
        } else {
            const std::string device_id =
                std::to_string(invocation.device_id);
            execl(invocation.executable.c_str(), invocation.executable.c_str(),
                  "--backend", backend, "--device", device_id.c_str(),
                  "--operation", invocation.operation.c_str(), nullptr);
        }
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

IsolatedRouteDiscoveryResult DiscoverIsolatedBackendRoutes(
    RouteProbeInvocation invocation,
    const RouteQualificationCancelCheck& should_cancel) {
    invocation.enumerate_backend = true;
    invocation.operation.clear();
    invocation.device_id = 0;
    const auto probe = RunIsolatedRouteProbe(invocation, should_cancel);
    IsolatedRouteDiscoveryResult result;
    result.status = probe.status;
    if (probe.status != RouteProbeStatus::Passed) {
        result.message = probe.infrastructure_error.empty()
            ? "Isolated backend route discovery failed"
            : probe.infrastructure_error;
        return result;
    }
    if (!ParseRouteInventory(
            probe.output, invocation.type, result.routes, result.message)) {
        result.status = RouteProbeStatus::InfrastructureFailure;
        result.routes.clear();
        return result;
    }
    result.message = result.routes.empty()
        ? "The candidate backend exposed no routes"
        : "Candidate backend routes discovered";
    return result;
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

RuntimeQualificationResult
RouteQualificationService::VerifyStagedRuntimeRoutes(
    const std::vector<DeviceInfo>& affected_routes,
    const RuntimeQualificationIdentity& identity,
    RuntimeQualificationFailurePolicy failure_policy,
    RouteQualificationOptions options,
    std::function<void(const RouteQualificationProgress&)> on_progress) {
    options.runtime_identity = identity;
    auto qualification = Verify(
        affected_routes, options, true, on_progress);

    RuntimeQualificationResult result;
    result.qualification = std::move(qualification);
    bool qualified = result.qualification.status ==
                         RouteQualificationRunStatus::Completed &&
                     result.qualification.snapshot.has_value();
    if (qualified) {
        for (const auto& route : affected_routes) {
            const auto record = std::find_if(
                result.qualification.snapshot->routes.begin(),
                result.qualification.snapshot->routes.end(),
                [&](const RouteQualificationRecord& candidate) {
                    return candidate.type == route.type &&
                           candidate.device_id == route.device_id;
                });
            if (record == result.qualification.snapshot->routes.end() ||
                !record->certified) {
                qualified = false;
                if (record != result.qualification.snapshot->routes.end()) {
                    result.diagnostic = record->failure;
                }
                break;
            }
        }
    }
    if (qualified) {
        result.disposition = RuntimeQualificationDisposition::Qualified;
        return result;
    }

    result.disposition = failure_policy ==
            RuntimeQualificationFailurePolicy::RequireRollback
        ? RuntimeQualificationDisposition::RollbackRequired
        : RuntimeQualificationDisposition::InstalledUnqualified;
    if (result.diagnostic.category == RouteFailureCategory::None) {
        result.diagnostic.stage = RouteFailureStage::Policy;
        result.diagnostic.category = RouteFailureCategory::PolicyBlocked;
        result.diagnostic.observed_fact = result.qualification.message.empty()
            ? "The staged runtime routes were not qualified"
            : result.qualification.message;
        result.diagnostic.bounded_interpretation =
            "Package presence does not establish route compatibility";
        result.diagnostic.recommended_action =
            result.disposition == RuntimeQualificationDisposition::RollbackRequired
                ? "Restore the previously active runtime set"
                : "Keep the pack disabled until its routes pass verification";
        result.diagnostic.evidence_id = options.matrix_id;
    }
    return result;
}

RouteQualificationRunResult
RouteQualificationService::ReconcileRuntimeEvidence(
    const RuntimeQualificationIdentity& identity,
    const RouteQualificationOptions& options) {
    std::unique_lock<std::mutex> run_lock(run_mutex_, std::try_to_lock);
    if (!run_lock.owns_lock()) {
        return {RouteQualificationRunStatus::Busy, false,
                "Route qualification is already running", std::nullopt};
    }
    if (!IsCompleteRuntimeIdentity(identity) || options.cache_path.empty() ||
        options.matrix_id.empty()) {
        return {RouteQualificationRunStatus::InvalidRequest, false,
                "Runtime evidence reconciliation request is incomplete",
                std::nullopt};
    }

    RouteQualificationSnapshot snapshot;
    if (const auto current = GetRouteQualificationSnapshot();
        current.has_value() &&
        IsCompatibleRuntimeSnapshot(*current, identity)) {
        snapshot = *current;
        RemoveEvidenceForChangedPacks(snapshot, identity);
    }
    snapshot.schema = 1;
    snapshot.matrix_id = options.matrix_id;
    snapshot.pack_id = identity.runtime_set_id;
    snapshot.runtime_set_id = identity.runtime_set_id;
    snapshot.runtime_generation = identity.generation;
    snapshot.base_pack_id = identity.base_pack_id;
    snapshot.compute_contract_id = kCyxWizComputeContractId;
    snapshot.operation_manifest_id = kRouteQualificationOperationManifestId;
    snapshot.captured_at = UtcTimestamp();
    snapshot.report_sha256.clear();

    std::string publish_error;
    if (!SaveRouteQualificationSnapshotAtomic(
            options.cache_path, snapshot, publish_error)) {
        return {RouteQualificationRunStatus::PublishFailure, false,
                publish_error, std::nullopt};
    }
    InstallRouteQualificationSnapshot(snapshot);
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        published_snapshot_ = snapshot;
    }
    return {RouteQualificationRunStatus::Completed, true,
            "Runtime pack evidence reconciled", std::move(snapshot)};
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
    const bool probe_runtime_override =
        !options.probe_runtime_root.empty() ||
        !options.probe_working_directory.empty() ||
        !options.probe_runtime_dll_directories.empty();
    if (routes.empty() || options.matrix_id.empty() ||
        options.cache_path.empty() || options.probe_executable.empty() ||
        options.operation_timeout.count() <= 0 ||
        options.output_limit_bytes == 0 ||
        (options.benchmark_verified_routes &&
         options.benchmark_timeout.count() <= 0)) {
        return {RouteQualificationRunStatus::InvalidRequest, false,
                "Route qualification request is incomplete", std::nullopt};
    }
    if (probe_runtime_override &&
        (!options.runtime_identity.has_value() ||
         options.probe_runtime_root.empty() ||
         !options.probe_runtime_root.is_absolute() ||
         options.probe_working_directory.empty() ||
         !options.probe_working_directory.is_absolute() ||
         options.probe_runtime_dll_directories.empty() ||
         std::any_of(
             options.probe_runtime_dll_directories.begin(),
             options.probe_runtime_dll_directories.end(),
             [](const std::filesystem::path& directory) {
                 return directory.empty() || !directory.is_absolute();
             }))) {
        return {RouteQualificationRunStatus::InvalidRequest, false,
                "Candidate route qualification runtime is incomplete",
                std::nullopt};
    }
    if (options.runtime_identity.has_value() &&
        !IsCompleteRuntimeIdentity(*options.runtime_identity)) {
        return {RouteQualificationRunStatus::InvalidRequest, false,
                "Route qualification runtime identity is incomplete",
                std::nullopt};
    }
    for (size_t route_index = 0; route_index < routes.size(); ++route_index) {
        const auto& route = routes[route_index];
        if (!BackendName(route.type) || route.device_id < 0) {
            return {RouteQualificationRunStatus::InvalidRequest, false,
                    "Route qualification contains an unsupported route",
                    std::nullopt};
        }
        if (options.runtime_identity.has_value() &&
            RuntimePackIdForRoute(*options.runtime_identity, route.type).empty()) {
            return {RouteQualificationRunStatus::InvalidRequest, false,
                    "Route qualification has no signed pack identity for an affected route",
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
        if (current.has_value() && options.runtime_identity.has_value() &&
            IsCompatibleRuntimeSnapshot(
                *current, *options.runtime_identity)) {
            snapshot = *current;
            RemoveEvidenceForChangedPacks(
                snapshot, *options.runtime_identity);
        } else if (current.has_value() &&
            !options.runtime_identity.has_value() &&
            current->pack_id == options.pack_id &&
            current->compute_contract_id == kCyxWizComputeContractId &&
            current->operation_manifest_id ==
                kRouteQualificationOperationManifestId) {
            snapshot = *current;
        }
    }
    snapshot.schema = 1;
    snapshot.matrix_id = options.matrix_id;
    if (options.runtime_identity.has_value()) {
        snapshot.pack_id = options.runtime_identity->runtime_set_id;
        snapshot.runtime_set_id = options.runtime_identity->runtime_set_id;
        snapshot.runtime_generation = options.runtime_identity->generation;
        snapshot.base_pack_id = options.runtime_identity->base_pack_id;
    } else {
        snapshot.pack_id = options.pack_id;
        snapshot.runtime_set_id.clear();
        snapshot.runtime_generation = 0;
        snapshot.base_pack_id.clear();
    }
    snapshot.compute_contract_id = kCyxWizComputeContractId;
    snapshot.operation_manifest_id =
        kRouteQualificationOperationManifestId;
    snapshot.captured_at = UtcTimestamp();
    snapshot.report_sha256.clear();

    const auto operations = RequiredRouteQualificationOperations();
    for (size_t route_index = 0; route_index < routes.size(); ++route_index) {
        const auto& route = routes[route_index];
        RouteQualificationRecord record = RecordFor(route);
        if (options.runtime_identity.has_value()) {
            record.pack_id = RuntimePackIdForRoute(
                *options.runtime_identity, route.type);
        }
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
            invocation.runtime_root = options.probe_runtime_root;
            invocation.working_directory =
                options.probe_working_directory;
            invocation.runtime_dll_directories =
                options.probe_runtime_dll_directories;
            invocation.runtime_identity = options.runtime_identity;
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
            invocation.runtime_root = options.probe_runtime_root;
            invocation.working_directory =
                options.probe_working_directory;
            invocation.runtime_dll_directories =
                options.probe_runtime_dll_directories;
            invocation.runtime_identity = options.runtime_identity;
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

const char* RuntimeQualificationDispositionName(
    RuntimeQualificationDisposition disposition) {
    switch (disposition) {
        case RuntimeQualificationDisposition::Qualified: return "qualified";
        case RuntimeQualificationDisposition::InstalledUnqualified:
            return "installed_unqualified";
        case RuntimeQualificationDisposition::RollbackRequired:
            return "rollback_required";
        default: return "unknown";
    }
}

}  // namespace cyxwiz
