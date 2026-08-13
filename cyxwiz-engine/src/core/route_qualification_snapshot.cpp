#include "route_qualification_snapshot.h"

#include <nlohmann/json.hpp>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <af/version.h>
#endif

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace cyxwiz {
namespace {

std::mutex& SnapshotMutex() {
    static std::mutex mutex;
    return mutex;
}

std::optional<RouteQualificationSnapshot>& SnapshotSlot() {
    static std::optional<RouteQualificationSnapshot> snapshot;
    return snapshot;
}

std::optional<DeviceType> ParseBackend(const std::string& backend) {
    if (backend == "cpu") return DeviceType::CPU;
    if (backend == "cuda") return DeviceType::CUDA;
    if (backend == "opencl") return DeviceType::OPENCL;
    if (backend == "oneapi") return DeviceType::ONEAPI;
    return std::nullopt;
}

const char* BackendJsonName(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "cpu";
        case DeviceType::CUDA: return "cuda";
        case DeviceType::OPENCL: return "opencl";
        case DeviceType::ONEAPI: return "oneapi";
        default: return nullptr;
    }
}

std::string Lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return value;
}

bool SameKnownValue(const std::string& expected,
                    const std::string& actual) {
    return expected.empty() ||
           (!actual.empty() && Lower(expected) == Lower(actual));
}

std::string CurrentArrayFireVersion() {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    return AF_VERSION;
#else
    return {};
#endif
}

int RequiredInt(const nlohmann::json& value, const char* key) {
    if (!value.contains(key) || !value.at(key).is_number_integer()) {
        throw std::runtime_error(std::string("Missing integer field '") +
                                 key + "'");
    }
    return value.at(key).get<int>();
}

std::string OptionalString(const nlohmann::json& value, const char* key) {
    if (!value.contains(key) || value.at(key).is_null()) return {};
    if (!value.at(key).is_string()) {
        throw std::runtime_error(std::string("Field '") + key +
                                 "' must be a string or null");
    }
    return value.at(key).get<std::string>();
}

DeviceKind ParseOptionalDeviceKind(const nlohmann::json& value,
                                   bool& known) {
    known = false;
    const std::string kind = OptionalString(value, "device_kind");
    if (kind.empty() || kind == "unknown") return DeviceKind::Unknown;
    known = true;
    if (kind == "cpu") return DeviceKind::CPU;
    if (kind == "gpu") return DeviceKind::GPU;
    if (kind == "accelerator") return DeviceKind::Accelerator;
    throw std::runtime_error(
        "Qualification route has an unknown device_kind");
}

int OptionalInt(const nlohmann::json& value, const char* key) {
    if (!value.contains(key) || value.at(key).is_null()) return 0;
    if (!value.at(key).is_number_integer()) {
        throw std::runtime_error(std::string("Field '") + key +
                                 "' must be an integer or null");
    }
    return value.at(key).get<int>();
}

double OptionalDouble(const nlohmann::json& value, const char* key) {
    if (!value.contains(key) || value.at(key).is_null()) return 0.0;
    if (!value.at(key).is_number()) {
        throw std::runtime_error(std::string("Field '") + key +
                                 "' must be a number or null");
    }
    return value.at(key).get<double>();
}

RouteFailureStage ParseFailureStage(const std::string& value) {
    if (value.empty() || value == "none") return RouteFailureStage::None;
    if (value == "package") return RouteFailureStage::Package;
    if (value == "backend_load") return RouteFailureStage::BackendLoad;
    if (value == "enumeration") return RouteFailureStage::Enumeration;
    if (value == "identity") return RouteFailureStage::Identity;
    if (value == "activation") return RouteFailureStage::Activation;
    if (value == "operation") return RouteFailureStage::Operation;
    if (value == "numerical") return RouteFailureStage::Numerical;
    if (value == "strict_training") return RouteFailureStage::StrictTraining;
    if (value == "policy") return RouteFailureStage::Policy;
    if (value == "evidence") return RouteFailureStage::Evidence;
    throw std::runtime_error("Qualification route has unknown failure_stage");
}

RouteFailureCategory ParseFailureCategory(const std::string& value) {
    if (value.empty() || value == "none") return RouteFailureCategory::None;
    if (value == "package_absent") return RouteFailureCategory::PackageAbsent;
    if (value == "package_invalid") return RouteFailureCategory::PackageInvalid;
    if (value == "abi_mismatch") return RouteFailureCategory::AbiMismatch;
    if (value == "dependency_missing") return RouteFailureCategory::DependencyMissing;
    if (value == "provider_missing") return RouteFailureCategory::ProviderMissing;
    if (value == "backend_load_failed") return RouteFailureCategory::BackendLoadFailed;
    if (value == "device_not_enumerated") return RouteFailureCategory::DeviceNotEnumerated;
    if (value == "identity_mismatch") return RouteFailureCategory::IdentityMismatch;
    if (value == "activation_failed") return RouteFailureCategory::ActivationFailed;
    if (value == "backend_substitution") return RouteFailureCategory::BackendSubstitution;
    if (value == "unsupported_operation") return RouteFailureCategory::UnsupportedOperation;
    if (value == "operation_failed") return RouteFailureCategory::OperationFailed;
    if (value == "out_of_memory") return RouteFailureCategory::OutOfMemory;
    if (value == "numerical_mismatch") return RouteFailureCategory::NumericalMismatch;
    if (value == "non_finite_result") return RouteFailureCategory::NonFiniteResult;
    if (value == "timeout") return RouteFailureCategory::Timeout;
    if (value == "child_process_crash") return RouteFailureCategory::ChildProcessCrash;
    if (value == "native_fallback") return RouteFailureCategory::NativeFallback;
    if (value == "residency_violation") return RouteFailureCategory::ResidencyViolation;
    if (value == "policy_blocked") return RouteFailureCategory::PolicyBlocked;
    if (value == "cancelled") return RouteFailureCategory::Cancelled;
    if (value == "evidence_stale") return RouteFailureCategory::EvidenceStale;
    if (value == "malformed_evidence") return RouteFailureCategory::MalformedEvidence;
    throw std::runtime_error("Qualification route has unknown failure_category");
}

RouteFailureDiagnostic DeriveFailureDiagnostic(
    const RouteQualificationRecord& route,
    const std::string& matrix_id) {
    if (route.failure.category != RouteFailureCategory::None) {
        return route.failure;
    }

    RouteFailureDiagnostic failure;
    failure.stage = RouteFailureStage::Operation;
    failure.evidence_id = matrix_id;
    std::ostringstream observed;
    if (route.crash_count > 0) {
        failure.category = RouteFailureCategory::ChildProcessCrash;
        observed << route.crash_count
                 << " required operation(s) terminated an isolated child process";
    } else if (route.timeout_count > 0) {
        failure.category = RouteFailureCategory::Timeout;
        observed << route.timeout_count
                 << " required operation(s) exceeded the probe timeout";
    } else if (route.failure_count > 0) {
        failure.category = RouteFailureCategory::OperationFailed;
        observed << route.failure_count
                 << " required operation(s) returned an error";
    } else if (route.unavailable_count > 0) {
        failure.category = RouteFailureCategory::UnsupportedOperation;
        observed << route.unavailable_count
                 << " required operation(s) were unavailable";
    } else if (!route.certified ||
               route.pass_count != route.operation_count) {
        failure.stage = RouteFailureStage::Evidence;
        failure.category = RouteFailureCategory::MalformedEvidence;
        observed << "The route did not satisfy the complete operation matrix";
    } else {
        return failure;
    }
    if (!route.failure.operation.empty()) {
        failure.operation = route.failure.operation;
        observed << "; first failing operation was '"
                 << route.failure.operation << "'";
    }
    failure.observed_fact = observed.str();
    failure.bounded_interpretation =
        "The runtime package was present, but this exact backend, device, "
        "provider, and driver combination did not complete the released "
        "operation contract";
    failure.recommended_action =
        "Update the provider or driver and verify again, or select another "
        "verified route";
    return failure;
}

const char* DeviceKindJsonName(DeviceKind kind, bool known) {
    if (!known) return nullptr;
    switch (kind) {
        case DeviceKind::CPU: return "cpu";
        case DeviceKind::GPU: return "gpu";
        case DeviceKind::Accelerator: return "accelerator";
        default: return nullptr;
    }
}

std::string ValidateRouteRecord(
    const RouteQualificationRecord& route) {
    if (!BackendJsonName(route.type) || route.device_id < 0) {
        return "Qualification snapshot contains an invalid route";
    }
    if (route.operation_count != kRouteQualificationOperationCount) {
        return "Qualification route does not cover the current operation manifest";
    }
    if (route.pass_count < 0 || route.unavailable_count < 0 ||
        route.failure_count < 0 || route.timeout_count < 0 ||
        route.crash_count < 0) {
        return "Qualification route contains a negative outcome count";
    }
    const int64_t classified = static_cast<int64_t>(route.pass_count) +
        route.unavailable_count + route.failure_count + route.timeout_count +
        route.crash_count;
    if (classified != route.operation_count) {
        return "Qualification route contains inconsistent outcome counts";
    }
    const bool outcomes_certify =
        route.pass_count == route.operation_count &&
        route.unavailable_count == 0 && route.failure_count == 0 &&
        route.timeout_count == 0 && route.crash_count == 0;
    if (route.certified != outcomes_certify) {
        return "Qualification route certification disagrees with outcomes";
    }
    if ((!route.display_name.empty() || route.device_kind_known) &&
        route.identity_source.empty()) {
        return "Qualification identity requires identity_source";
    }
    if (route.device_kind_known &&
        !DeviceKindJsonName(route.device_kind, true)) {
        return "Qualification route has an unknown device_kind";
    }
    const bool has_failure_category =
        route.failure.category != RouteFailureCategory::None;
    const bool has_failure_stage =
        route.failure.stage != RouteFailureStage::None;
    if (has_failure_category != has_failure_stage ||
        (has_failure_category && route.failure.observed_fact.empty())) {
        return "Qualification failure requires category, stage, and observed_fact";
    }
    if (route.certified && has_failure_category) {
        return "Passed qualification route cannot contain a failure";
    }
    const bool has_benchmark = !route.benchmark_id.empty();
    if (has_benchmark != (route.benchmark_median_iteration_ms > 0.0) ||
        (has_benchmark &&
         (route.benchmark_sample_count <= 0 ||
          route.benchmark_iterations_per_sample <= 0))) {
        return "Qualification route contains inconsistent benchmark evidence";
    }
    return {};
}

bool PublishAtomic(const std::filesystem::path& temporary,
                   const std::filesystem::path& target,
                   std::string& error) {
#ifdef _WIN32
    if (!MoveFileExW(temporary.c_str(), target.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        error = "could not publish qualification cache atomically: " +
                std::system_category().message(
                    static_cast<int>(GetLastError()));
        return false;
    }
    return true;
#else
    std::error_code ec;
    std::filesystem::rename(temporary, target, ec);
    if (ec) {
        error = "could not publish qualification cache atomically: " +
                ec.message();
        return false;
    }
    return true;
#endif
}

RouteQualificationSnapshot ParseSnapshot(const nlohmann::json& document) {
    if (!document.is_object()) {
        throw std::runtime_error("Qualification snapshot root must be an object");
    }

    RouteQualificationSnapshot snapshot;
    snapshot.schema = RequiredInt(document, "schema");
    if (snapshot.schema != 1) {
        throw std::runtime_error("Unsupported qualification snapshot schema");
    }
    snapshot.matrix_id = OptionalString(document, "matrix_id");
    snapshot.pack_id = OptionalString(document, "pack_id");
    snapshot.compute_contract_id =
        OptionalString(document, "compute_contract_id");
    snapshot.operation_manifest_id =
        OptionalString(document, "operation_manifest_id");
    snapshot.captured_at = OptionalString(document, "captured_at");
    snapshot.report_sha256 = OptionalString(document, "report_sha256");
    if (snapshot.matrix_id.empty()) {
        throw std::runtime_error("Qualification snapshot has no matrix_id");
    }
    if (!document.contains("routes") || !document.at("routes").is_array()) {
        throw std::runtime_error("Qualification snapshot routes must be an array");
    }

    for (const auto& route_json : document.at("routes")) {
        const auto type = ParseBackend(OptionalString(route_json, "backend"));
        if (!type.has_value()) {
            throw std::runtime_error("Qualification route has an unknown backend");
        }

        RouteQualificationRecord route;
        route.type = *type;
        route.device_id = RequiredInt(route_json, "device_id");
        route.physical_fingerprint =
            OptionalString(route_json, "physical_fingerprint");
        route.provider = OptionalString(route_json, "provider");
        route.driver_version = OptionalString(route_json, "driver_version");
        route.runtime_version = OptionalString(route_json, "runtime_version");
        route.operation_count = RequiredInt(route_json, "operation_count");
        route.pass_count = RequiredInt(route_json, "pass_count");
        route.unavailable_count = RequiredInt(route_json, "unavailable_count");
        route.failure_count = RequiredInt(route_json, "failure_count");
        route.timeout_count = RequiredInt(route_json, "timeout_count");
        route.crash_count = RequiredInt(route_json, "crash_count");
        route.display_name = OptionalString(route_json, "display_name");
        route.device_kind =
            ParseOptionalDeviceKind(route_json, route.device_kind_known);
        route.identity_source =
            OptionalString(route_json, "identity_source");
        route.failure.stage = ParseFailureStage(
            OptionalString(route_json, "failure_stage"));
        route.failure.category = ParseFailureCategory(
            OptionalString(route_json, "failure_category"));
        route.failure.operation =
            OptionalString(route_json, "failed_operation");
        route.failure.probe_stage =
            OptionalString(route_json, "probe_stage");
        route.failure.error_code = OptionalInt(route_json, "error_code");
        route.failure.timeout_ms = OptionalInt(route_json, "timeout_ms");
        route.failure.observed_fact =
            OptionalString(route_json, "observed_fact");
        route.failure.bounded_interpretation =
            OptionalString(route_json, "bounded_interpretation");
        route.failure.recommended_action =
            OptionalString(route_json, "recommended_action");
        route.failure.evidence_id =
            OptionalString(route_json, "evidence_id");
        route.benchmark_id =
            OptionalString(route_json, "benchmark_id");
        route.benchmark_sample_count =
            OptionalInt(route_json, "benchmark_sample_count");
        route.benchmark_iterations_per_sample =
            OptionalInt(route_json, "benchmark_iterations_per_sample");
        route.benchmark_median_iteration_ms =
            OptionalDouble(route_json, "benchmark_median_iteration_ms");
        route.benchmark_message =
            OptionalString(route_json, "benchmark_message");
        if (!route_json.contains("certified") ||
            !route_json.at("certified").is_boolean()) {
            throw std::runtime_error(
                "Qualification route certified must be a boolean");
        }
        route.certified = route_json.at("certified").get<bool>();

        if (const std::string validation = ValidateRouteRecord(route);
            !validation.empty()) {
            throw std::runtime_error(validation);
        }
        const auto duplicate = std::find_if(
            snapshot.routes.begin(), snapshot.routes.end(),
            [&](const RouteQualificationRecord& existing) {
                return existing.type == route.type &&
                       existing.device_id == route.device_id;
            });
        if (duplicate != snapshot.routes.end()) {
            throw std::runtime_error(
                "Qualification snapshot contains a duplicate route");
        }
        snapshot.routes.push_back(std::move(route));
    }

    if (snapshot.routes.empty()) {
        throw std::runtime_error("Qualification snapshot contains no routes");
    }
    return snapshot;
}

}  // namespace

RouteQualificationLoadResult LoadAndInstallRouteQualificationSnapshot(
    const std::filesystem::path& path) {
    RouteQualificationLoadResult result;
    try {
        std::ifstream stream(path);
        if (!stream) {
            result.message = "Qualification snapshot not found: " +
                             path.string();
            return result;
        }
        nlohmann::json document;
        stream >> document;
        auto snapshot = ParseSnapshot(document);
        result.matrix_id = snapshot.matrix_id;
        result.route_count = snapshot.routes.size();
        InstallRouteQualificationSnapshot(std::move(snapshot));
        result.loaded = true;
        result.message = "Qualification snapshot loaded";
    } catch (const std::exception& error) {
        result.message = error.what();
    }
    return result;
}

bool SaveRouteQualificationSnapshotAtomic(
    const std::filesystem::path& path,
    const RouteQualificationSnapshot& snapshot,
    std::string& error) {
    error.clear();
    if (snapshot.schema != 1 || snapshot.matrix_id.empty() ||
        snapshot.routes.empty()) {
        error = "qualification snapshot is incomplete";
        return false;
    }

    nlohmann::json document = {
        {"schema", snapshot.schema},
        {"matrix_id", snapshot.matrix_id},
        {"pack_id", snapshot.pack_id.empty()
             ? nlohmann::json(nullptr)
             : nlohmann::json(snapshot.pack_id)},
        {"compute_contract_id", snapshot.compute_contract_id.empty()
             ? nlohmann::json(nullptr)
             : nlohmann::json(snapshot.compute_contract_id)},
        {"operation_manifest_id", snapshot.operation_manifest_id.empty()
             ? nlohmann::json(nullptr)
             : nlohmann::json(snapshot.operation_manifest_id)},
        {"captured_at", snapshot.captured_at.empty()
             ? nlohmann::json(nullptr)
             : nlohmann::json(snapshot.captured_at)},
        {"report_sha256", snapshot.report_sha256.empty()
             ? nlohmann::json(nullptr)
             : nlohmann::json(snapshot.report_sha256)},
        {"routes", nlohmann::json::array()}};

    std::vector<std::pair<DeviceType, int>> seen_routes;
    seen_routes.reserve(snapshot.routes.size());
    for (const auto& route : snapshot.routes) {
        const char* backend = BackendJsonName(route.type);
        if (const std::string validation = ValidateRouteRecord(route);
            !validation.empty()) {
            error = validation;
            return false;
        }
        const auto key = std::make_pair(route.type, route.device_id);
        if (std::find(seen_routes.begin(), seen_routes.end(), key) !=
            seen_routes.end()) {
            error = "Qualification snapshot contains a duplicate route";
            return false;
        }
        seen_routes.push_back(key);
        nlohmann::json record = {
            {"backend", backend},
            {"device_id", route.device_id},
            {"physical_fingerprint", route.physical_fingerprint.empty()
                 ? nlohmann::json(nullptr)
                 : nlohmann::json(route.physical_fingerprint)},
            {"provider", route.provider.empty()
                 ? nlohmann::json(nullptr)
                 : nlohmann::json(route.provider)},
            {"driver_version", route.driver_version.empty()
                 ? nlohmann::json(nullptr)
                 : nlohmann::json(route.driver_version)},
            {"runtime_version", route.runtime_version.empty()
                 ? nlohmann::json(nullptr)
                 : nlohmann::json(route.runtime_version)},
            {"display_name", route.display_name.empty()
                 ? nlohmann::json(nullptr)
                 : nlohmann::json(route.display_name)},
            {"device_kind", DeviceKindJsonName(
                 route.device_kind, route.device_kind_known)
                 ? nlohmann::json(DeviceKindJsonName(
                       route.device_kind, route.device_kind_known))
                 : nlohmann::json(nullptr)},
            {"identity_source", route.identity_source.empty()
                 ? nlohmann::json(nullptr)
                 : nlohmann::json(route.identity_source)},
            {"operation_count", route.operation_count},
            {"pass_count", route.pass_count},
            {"unavailable_count", route.unavailable_count},
            {"failure_count", route.failure_count},
            {"timeout_count", route.timeout_count},
            {"crash_count", route.crash_count},
            {"certified", route.certified}};

        record["benchmark_id"] = route.benchmark_id.empty()
            ? nlohmann::json(nullptr)
            : nlohmann::json(route.benchmark_id);
        record["benchmark_sample_count"] = route.benchmark_sample_count == 0
            ? nlohmann::json(nullptr)
            : nlohmann::json(route.benchmark_sample_count);
        record["benchmark_iterations_per_sample"] =
            route.benchmark_iterations_per_sample == 0
                ? nlohmann::json(nullptr)
                : nlohmann::json(route.benchmark_iterations_per_sample);
        record["benchmark_median_iteration_ms"] =
            route.benchmark_median_iteration_ms <= 0.0
                ? nlohmann::json(nullptr)
                : nlohmann::json(route.benchmark_median_iteration_ms);
        record["benchmark_message"] = route.benchmark_message.empty()
            ? nlohmann::json(nullptr)
            : nlohmann::json(route.benchmark_message);

        const auto failure = DeriveFailureDiagnostic(route, snapshot.matrix_id);
        if (failure.category != RouteFailureCategory::None) {
            record["failure_stage"] = RouteFailureStageName(failure.stage);
            record["failure_category"] =
                RouteFailureCategoryName(failure.category);
            record["failed_operation"] = failure.operation.empty()
                ? nlohmann::json(nullptr)
                : nlohmann::json(failure.operation);
            record["probe_stage"] = failure.probe_stage.empty()
                ? nlohmann::json(nullptr)
                : nlohmann::json(failure.probe_stage);
            record["error_code"] = failure.error_code == 0
                ? nlohmann::json(nullptr)
                : nlohmann::json(failure.error_code);
            record["timeout_ms"] = failure.timeout_ms == 0
                ? nlohmann::json(nullptr)
                : nlohmann::json(failure.timeout_ms);
            record["observed_fact"] = failure.observed_fact;
            record["bounded_interpretation"] =
                failure.bounded_interpretation;
            record["recommended_action"] = failure.recommended_action;
            record["evidence_id"] = failure.evidence_id.empty()
                ? nlohmann::json(snapshot.matrix_id)
                : nlohmann::json(failure.evidence_id);
        }
        document["routes"].push_back(std::move(record));
    }

    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
        error = "could not create qualification cache directory: " +
                ec.message();
        return false;
    }
    const auto nonce =
        std::chrono::steady_clock::now().time_since_epoch().count();
    std::filesystem::path temporary = path;
    temporary += ".tmp." + std::to_string(nonce);
    try {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            error = "could not open temporary qualification cache";
            return false;
        }
        output << document.dump(2) << '\n';
        output.flush();
        if (!output.good()) {
            error = "could not write temporary qualification cache";
            output.close();
            std::filesystem::remove(temporary, ec);
            return false;
        }
        output.close();
        if (!PublishAtomic(temporary, path, error)) {
            std::filesystem::remove(temporary, ec);
            return false;
        }
        return true;
    } catch (const std::exception& exception) {
        error = "qualification cache save failed: " +
                std::string(exception.what());
        std::filesystem::remove(temporary, ec);
        return false;
    }
}

void InstallRouteQualificationSnapshot(RouteQualificationSnapshot snapshot) {
    std::lock_guard<std::mutex> lock(SnapshotMutex());
    SnapshotSlot() = std::move(snapshot);
}

void ClearRouteQualificationSnapshot() {
    std::lock_guard<std::mutex> lock(SnapshotMutex());
    SnapshotSlot().reset();
}

std::optional<RouteQualificationSnapshot> GetRouteQualificationSnapshot() {
    std::lock_guard<std::mutex> lock(SnapshotMutex());
    return SnapshotSlot();
}

RouteQualificationDecision EvaluateRouteQualification(
    const DeviceInfo& route) {
    return EvaluateRouteQualification(route, GetRouteQualificationSnapshot());
}

RouteQualificationDecision EvaluateRouteQualification(
    const DeviceInfo& route,
    const std::optional<RouteQualificationSnapshot>& snapshot) {
    RouteQualificationDecision decision;
    if (!snapshot.has_value()) {
        decision.message =
            "No isolated route qualification snapshot is installed";
        return decision;
    }
    decision.matrix_id = snapshot->matrix_id;

    const auto record = std::find_if(
        snapshot->routes.begin(), snapshot->routes.end(),
        [&](const RouteQualificationRecord& candidate) {
            return candidate.type == route.type &&
                   candidate.device_id == route.device_id;
        });
    if (record == snapshot->routes.end()) {
        decision.message =
            "This route has no retained qualification evidence";
        return decision;
    }
    decision.evidence_available = true;

    if ((!snapshot->compute_contract_id.empty() &&
         snapshot->compute_contract_id != kCyxWizComputeContractId) ||
        (!snapshot->operation_manifest_id.empty() &&
         snapshot->operation_manifest_id !=
             kRouteQualificationOperationManifestId)) {
        decision.message =
            "The retained qualification evidence uses a different compute contract or operation manifest";
        decision.failure.stage = RouteFailureStage::Evidence;
        decision.failure.category = RouteFailureCategory::EvidenceStale;
        decision.failure.observed_fact = decision.message;
        decision.failure.bounded_interpretation =
            "The required CyxWiz computation contract changed after this evidence was captured";
        decision.failure.recommended_action =
            "Verify the route again with the current operation manifest";
        decision.failure.evidence_id = snapshot->matrix_id;
        return decision;
    }

    if (const std::string validation = ValidateRouteRecord(*record);
        !validation.empty()) {
        decision.message = validation;
        decision.failure.stage = RouteFailureStage::Evidence;
        decision.failure.category = RouteFailureCategory::MalformedEvidence;
        decision.failure.observed_fact = decision.message;
        decision.failure.bounded_interpretation =
            "The retained evidence does not prove the complete current operation contract";
        decision.failure.recommended_action =
            "Verify the route again with the current operation manifest";
        decision.failure.evidence_id = snapshot->matrix_id;
        return decision;
    }

    if (!record->physical_fingerprint.empty() &&
        (!route.physical_fingerprint_known ||
         !SameKnownValue(record->physical_fingerprint,
                         route.physical_fingerprint))) {
        decision.message =
            "The discovered route identity differs from the retained qualification evidence";
        decision.failure.stage = RouteFailureStage::Evidence;
        decision.failure.category = RouteFailureCategory::EvidenceStale;
        decision.failure.observed_fact = decision.message;
        decision.failure.bounded_interpretation =
            "The cached evidence belongs to a different physical route";
        decision.failure.recommended_action =
            "Verify the currently discovered route";
        decision.failure.evidence_id = snapshot->matrix_id;
        return decision;
    }
    if (!SameKnownValue(record->provider,
                        route.provider_known ? route.provider : std::string{}) ||
        !SameKnownValue(record->driver_version,
                        route.driver_version_known
                            ? route.driver_version
                            : std::string{})) {
        decision.message =
            "The route provider or driver differs from the retained qualification evidence";
        decision.failure.stage = RouteFailureStage::Evidence;
        decision.failure.category = RouteFailureCategory::EvidenceStale;
        decision.failure.observed_fact = decision.message;
        decision.failure.bounded_interpretation =
            "The provider or driver changed after this evidence was captured";
        decision.failure.recommended_action =
            "Verify the route again with the current provider and driver";
        decision.failure.evidence_id = snapshot->matrix_id;
        return decision;
    }
    if (!record->runtime_version.empty() &&
        !SameKnownValue(record->runtime_version,
                        CurrentArrayFireVersion())) {
        decision.message =
            "The ArrayFire runtime version differs from the retained qualification evidence";
        decision.failure.stage = RouteFailureStage::Evidence;
        decision.failure.category = RouteFailureCategory::EvidenceStale;
        decision.failure.observed_fact = decision.message;
        decision.failure.bounded_interpretation =
            "The ArrayFire runtime changed after this evidence was captured";
        decision.failure.recommended_action =
            "Verify the route again with the active ArrayFire runtime";
        decision.failure.evidence_id = snapshot->matrix_id;
        return decision;
    }
    decision.display_name = record->display_name;
    decision.device_kind = record->device_kind;
    decision.display_name_available = !record->display_name.empty();
    decision.device_kind_known = record->device_kind_known;
    decision.identity_source = record->identity_source;
    decision.failure = DeriveFailureDiagnostic(*record, snapshot->matrix_id);
    if (record->crash_count > 0 || record->timeout_count > 0 ||
        record->failure_count > 0 || record->unavailable_count > 0) {
        std::ostringstream message;
        message << "Route failed isolated verification: crash="
                << record->crash_count
                << " timeout=" << record->timeout_count
                << " failed=" << record->failure_count
                << " unavailable=" << record->unavailable_count;
        decision.message = message.str();
        return decision;
    }
    if (!record->certified ||
        record->pass_count != record->operation_count) {
        decision.message =
            "Route did not pass the complete qualification operation set";
        return decision;
    }

    decision.qualified = true;
    decision.message = "Route passed isolated verification";
    return decision;
}

RouteTrainingAuthorizationDecision EvaluateRouteTrainingAuthorization(
    const DeviceInfo& route,
    const RouteQualificationDecision& qualification) {
    (void)route;
    RouteTrainingAuthorizationDecision decision;
    if (!qualification.evidence_available) {
        decision.status = RouteTrainingAuthorizationStatus::NoEvidence;
        decision.message = qualification.message;
        decision.failure = qualification.failure;
        return decision;
    }
    if (!qualification.qualified) {
        decision.status = RouteTrainingAuthorizationStatus::MatrixRejected;
        decision.message = qualification.message;
        decision.failure = qualification.failure;
        return decision;
    }
    decision.status = RouteTrainingAuthorizationStatus::Ready;
    decision.authorized = true;
    decision.message =
        "Exact route passed its isolated matrix and is authorized for training";
    return decision;
}

const char* RouteTrainingAuthorizationStatusName(
    RouteTrainingAuthorizationStatus status) {
    switch (status) {
        case RouteTrainingAuthorizationStatus::Ready:
            return "ready";
        case RouteTrainingAuthorizationStatus::NoEvidence:
            return "no_evidence";
        case RouteTrainingAuthorizationStatus::MatrixRejected:
            return "matrix_rejected";
        case RouteTrainingAuthorizationStatus::DiagnosticOnly:
            return "diagnostic_only";
        default:
            return "unknown";
    }
}

const char* RouteFailureStageName(RouteFailureStage stage) {
    switch (stage) {
        case RouteFailureStage::Package: return "package";
        case RouteFailureStage::BackendLoad: return "backend_load";
        case RouteFailureStage::Enumeration: return "enumeration";
        case RouteFailureStage::Identity: return "identity";
        case RouteFailureStage::Activation: return "activation";
        case RouteFailureStage::Operation: return "operation";
        case RouteFailureStage::Numerical: return "numerical";
        case RouteFailureStage::StrictTraining: return "strict_training";
        case RouteFailureStage::Policy: return "policy";
        case RouteFailureStage::Evidence: return "evidence";
        default: return "none";
    }
}

const char* RouteFailureCategoryName(RouteFailureCategory category) {
    switch (category) {
        case RouteFailureCategory::PackageAbsent: return "package_absent";
        case RouteFailureCategory::PackageInvalid: return "package_invalid";
        case RouteFailureCategory::AbiMismatch: return "abi_mismatch";
        case RouteFailureCategory::DependencyMissing: return "dependency_missing";
        case RouteFailureCategory::ProviderMissing: return "provider_missing";
        case RouteFailureCategory::BackendLoadFailed: return "backend_load_failed";
        case RouteFailureCategory::DeviceNotEnumerated: return "device_not_enumerated";
        case RouteFailureCategory::IdentityMismatch: return "identity_mismatch";
        case RouteFailureCategory::ActivationFailed: return "activation_failed";
        case RouteFailureCategory::BackendSubstitution: return "backend_substitution";
        case RouteFailureCategory::UnsupportedOperation: return "unsupported_operation";
        case RouteFailureCategory::OperationFailed: return "operation_failed";
        case RouteFailureCategory::OutOfMemory: return "out_of_memory";
        case RouteFailureCategory::NumericalMismatch: return "numerical_mismatch";
        case RouteFailureCategory::NonFiniteResult: return "non_finite_result";
        case RouteFailureCategory::Timeout: return "timeout";
        case RouteFailureCategory::ChildProcessCrash: return "child_process_crash";
        case RouteFailureCategory::NativeFallback: return "native_fallback";
        case RouteFailureCategory::ResidencyViolation: return "residency_violation";
        case RouteFailureCategory::PolicyBlocked: return "policy_blocked";
        case RouteFailureCategory::Cancelled: return "cancelled";
        case RouteFailureCategory::EvidenceStale: return "evidence_stale";
        case RouteFailureCategory::MalformedEvidence: return "malformed_evidence";
        default: return "none";
    }
}

}  // namespace cyxwiz
