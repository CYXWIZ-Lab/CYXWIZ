#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cyxwiz/device.h>

#include "../../cyxwiz-backend/src/core/device_identity.h"
#include "../../cyxwiz-backend/src/core/device_probe.h"
#include "../../cyxwiz-engine/src/core/compute_runtime_config.h"
#include "../../cyxwiz-engine/src/core/compute_runtime_paths.h"
#include "../../cyxwiz-engine/src/core/execution_device_preferences.h"
#include "../../cyxwiz-engine/src/core/route_qualification_service.h"
#include "../../cyxwiz-engine/src/core/route_recommendation.h"
#ifdef CYXWIZ_HAS_BACKEND_PACK_QUALIFICATION_ADAPTER
#include "../../cyxwiz-engine/src/core/backend_pack_qualification_adapter.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <tuple>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>

namespace {
class ArrayFireStateGuard {
public:
    ArrayFireStateGuard()
        : backend_(af::getActiveBackend()), device_(af::getDevice()) {}

    ~ArrayFireStateGuard() {
        try {
            af::setBackend(backend_);
            af::setDevice(device_);
        } catch (af::exception&) {
        }
    }

private:
    af::Backend backend_;
    int device_;
};
}  // namespace
#endif

namespace {

class FakeArrayFireProbeAdapter final
    : public cyxwiz::detail::ArrayFireProbeAdapter {
public:
    std::map<cyxwiz::DeviceType, cyxwiz::detail::DeviceProbeStatus>
        backend_status;
    std::map<cyxwiz::DeviceType, cyxwiz::detail::DeviceCountProbeResult>
        device_counts;
    std::map<std::pair<cyxwiz::DeviceType, int>,
             cyxwiz::detail::DeviceProbeStatus>
        device_status;
    std::map<std::pair<cyxwiz::DeviceType, int>, cyxwiz::DeviceInfo> metadata;

    cyxwiz::detail::DeviceProbeStatus SelectBackend(
        cyxwiz::DeviceType type) override {
        return backend_status.at(type);
    }

    cyxwiz::detail::DeviceCountProbeResult GetDeviceCount(
        cyxwiz::DeviceType type) override {
        return device_counts.at(type);
    }

    cyxwiz::detail::DeviceProbeStatus SelectDevice(
        cyxwiz::DeviceType type,
        int device_id) override {
        return device_status.at({type, device_id});
    }

    cyxwiz::DeviceInfo QuerySelectedDeviceMetadata(
        cyxwiz::DeviceType type,
        int device_id) override {
        return metadata.at({type, device_id});
    }
};

cyxwiz::detail::DeviceProbeStatus ProbeSuccess() {
    return {true, 0, {}};
}

cyxwiz::DeviceInfo AvailableMetadata(const std::string& name) {
    cyxwiz::DeviceInfo info{};
    info.name = name;
    info.name_known = true;
    info.metadata_status = cyxwiz::DeviceMetadataStatus::Available;
    return info;
}

class RouteQualificationStateGuard {
public:
    RouteQualificationStateGuard()
        : previous_(cyxwiz::GetRouteQualificationSnapshot()) {}

    ~RouteQualificationStateGuard() {
        if (previous_.has_value()) {
            cyxwiz::InstallRouteQualificationSnapshot(
                std::move(*previous_));
        } else {
            cyxwiz::ClearRouteQualificationSnapshot();
        }
    }

private:
    std::optional<cyxwiz::RouteQualificationSnapshot> previous_;
};

class EnvironmentGuard {
public:
    explicit EnvironmentGuard(std::initializer_list<const char*> names) {
        for (const char* name : names) {
            const char* value = std::getenv(name);
            saved_.push_back({name, value
                ? std::optional<std::string>(value)
                : std::nullopt});
        }
    }

    ~EnvironmentGuard() {
        for (const auto& [name, value] : saved_) {
            Set(name.c_str(), value ? value->c_str() : nullptr);
        }
    }

    static void Set(const char* name, const char* value) {
#ifdef _WIN32
        _putenv_s(name, value ? value : "");
#else
        if (value) setenv(name, value, 1);
        else unsetenv(name);
#endif
    }

private:
    std::vector<std::pair<std::string, std::optional<std::string>>> saved_;
};

std::vector<cyxwiz::DeviceInfo> InstallCertifiedInventorySnapshot(
    const std::string& matrix_id) {
    const auto inventory = cyxwiz::Device::GetAvailableDevices();
    cyxwiz::RouteQualificationSnapshot snapshot;
    snapshot.matrix_id = matrix_id;
    snapshot.routes.reserve(inventory.size());
    for (const auto& device : inventory) {
        cyxwiz::RouteQualificationRecord record;
        record.type = device.type;
        record.device_id = device.device_id;
        record.physical_fingerprint = device.physical_fingerprint_known
            ? device.physical_fingerprint
            : std::string{};
        record.provider = device.provider_known
            ? device.provider
            : std::string{};
        record.driver_version = device.driver_version_known
            ? device.driver_version
            : std::string{};
        record.operation_count = cyxwiz::kRouteQualificationOperationCount;
        record.pass_count = cyxwiz::kRouteQualificationOperationCount;
        record.certified = true;
        snapshot.routes.push_back(std::move(record));
    }
    cyxwiz::InstallRouteQualificationSnapshot(std::move(snapshot));
    return inventory;
}

TEST_CASE("Device kind and identity names preserve unknown values",
          "[device][identity]") {
    CHECK(std::string(cyxwiz::DeviceKindName(cyxwiz::DeviceKind::CPU)) ==
          "cpu");
    CHECK(std::string(cyxwiz::DeviceKindName(cyxwiz::DeviceKind::GPU)) ==
          "gpu");
    CHECK(std::string(cyxwiz::DeviceKindName(
              cyxwiz::DeviceKind::Accelerator)) == "accelerator");
    CHECK(std::string(cyxwiz::DeviceKindName(
              static_cast<cyxwiz::DeviceKind>(999))) == "unknown");
    CHECK(std::string(cyxwiz::DeviceIdentityConfidenceName(
              cyxwiz::DeviceIdentityConfidence::BackendLocal)) ==
          "backend_local");
    CHECK(std::string(cyxwiz::DeviceIdentityConfidenceName(
              static_cast<cyxwiz::DeviceIdentityConfidence>(999))) ==
          "unknown");
}

TEST_CASE("Physical identity uses stable provider fields and never names",
          "[device][identity]") {
    cyxwiz::DeviceInfo cuda{};
    cuda.type = cyxwiz::DeviceType::CUDA;
    cuda.name = "CUDA display name";
    cuda.identity_confidence =
        cyxwiz::DeviceIdentityConfidence::BackendLocal;
    cuda.hardware_uuid = "ABCDEF0123456789";
    cuda.hardware_uuid_known = true;
    cyxwiz::detail::FinalizeDeviceIdentity(cuda);

    cyxwiz::DeviceInfo opencl{};
    opencl.type = cyxwiz::DeviceType::OPENCL;
    opencl.name = "Different OpenCL display name";
    opencl.identity_confidence =
        cyxwiz::DeviceIdentityConfidence::BackendLocal;
    opencl.hardware_uuid = "abcdef0123456789";
    opencl.hardware_uuid_known = true;
    cyxwiz::detail::FinalizeDeviceIdentity(opencl);

    CHECK(cuda.physical_fingerprint == "uuid:abcdef0123456789");
    CHECK(opencl.physical_fingerprint == cuda.physical_fingerprint);
    CHECK(cuda.identity_confidence ==
          cyxwiz::DeviceIdentityConfidence::StableHardware);

    cyxwiz::DeviceInfo name_only{};
    name_only.name = cuda.name;
    name_only.identity_confidence =
        cyxwiz::DeviceIdentityConfidence::BackendLocal;
    cyxwiz::detail::FinalizeDeviceIdentity(name_only);
    CHECK_FALSE(name_only.physical_fingerprint_known);
    CHECK(name_only.physical_fingerprint.empty());
    CHECK(name_only.identity_confidence ==
          cyxwiz::DeviceIdentityConfidence::BackendLocal);
}

TEST_CASE("PCI identity is backend independent and requires vendor evidence",
          "[device][identity]") {
    cyxwiz::DeviceInfo first{};
    first.type = cyxwiz::DeviceType::CUDA;
    first.hardware_vendor_id = 0x10de;
    first.hardware_vendor_id_known = true;
    first.pci_domain = 0;
    first.pci_bus = 1;
    first.pci_device = 2;
    first.pci_function = 0;
    first.pci_location_known = true;
    cyxwiz::detail::FinalizeDeviceIdentity(first);

    cyxwiz::DeviceInfo second = first;
    second.type = cyxwiz::DeviceType::OPENCL;
    second.physical_fingerprint.clear();
    second.physical_fingerprint_known = false;
    cyxwiz::detail::FinalizeDeviceIdentity(second);

    CHECK(first.physical_fingerprint == "pci:10de:0000:01:02.0");
    CHECK(second.physical_fingerprint == first.physical_fingerprint);

    cyxwiz::DeviceInfo unproven{};
    unproven.pci_bus = 1;
    unproven.pci_device = 2;
    unproven.pci_location_known = true;
    cyxwiz::detail::FinalizeDeviceIdentity(unproven);
    CHECK_FALSE(unproven.physical_fingerprint_known);
    CHECK(unproven.identity_confidence ==
          cyxwiz::DeviceIdentityConfidence::ProviderReported);
}

TEST_CASE("Physical route re-resolution survives ordinal changes without backend substitution",
          "[device][identity]") {
    cyxwiz::DeviceInfo cuda_first{};
    cuda_first.type = cyxwiz::DeviceType::CUDA;
    cuda_first.device_id = 0;
    cuda_first.physical_fingerprint = "uuid:first";
    cuda_first.physical_fingerprint_known = true;

    cyxwiz::DeviceInfo cuda_selected = cuda_first;
    cuda_selected.device_id = 1;
    cuda_selected.name = "Duplicate display name";
    cuda_selected.physical_fingerprint = "uuid:selected";

    cyxwiz::DeviceInfo opencl_same_hardware = cuda_selected;
    opencl_same_hardware.type = cyxwiz::DeviceType::OPENCL;
    opencl_same_hardware.device_id = 0;

    const std::vector<cyxwiz::DeviceInfo> reordered = {
        cuda_first, opencl_same_hardware, cuda_selected};
    const auto resolved = cyxwiz::ResolvePhysicalDeviceRoute(
        reordered, cyxwiz::DeviceType::CUDA, "UUID:SELECTED");
    CHECK(resolved.status ==
          cyxwiz::DeviceRouteResolutionStatus::Resolved);
    CHECK(resolved.type == cyxwiz::DeviceType::CUDA);
    CHECK(resolved.device_id == 1);

    const auto wrong_backend = cyxwiz::ResolvePhysicalDeviceRoute(
        {opencl_same_hardware},
        cyxwiz::DeviceType::CUDA,
        "uuid:selected");
    CHECK(wrong_backend.status ==
          cyxwiz::DeviceRouteResolutionStatus::NotFound);

    const auto missing = cyxwiz::ResolvePhysicalDeviceRoute(
        reordered, cyxwiz::DeviceType::CUDA, {});
    CHECK(missing.status ==
          cyxwiz::DeviceRouteResolutionStatus::FingerprintMissing);

    auto duplicate = cuda_selected;
    duplicate.device_id = 2;
    const auto ambiguous = cyxwiz::ResolvePhysicalDeviceRoute(
        {cuda_selected, duplicate},
        cyxwiz::DeviceType::CUDA,
        "uuid:selected");
    CHECK(ambiguous.status ==
          cyxwiz::DeviceRouteResolutionStatus::Ambiguous);
}

TEST_CASE("Device selection transaction commits only after exact revalidation",
          "[device][selection][transaction]") {
    cyxwiz::PendingExecutionDeviceSelection candidate{
        cyxwiz::DeviceType::CUDA, 2, "uuid:selected"};
    cyxwiz::DeviceInfo route{};
    route.type = candidate.type;
    route.device_id = candidate.device_id;
    route.physical_fingerprint = candidate.physical_fingerprint;
    route.physical_fingerprint_known = true;

    int inventory_calls = 0;
    int commit_calls = 0;
    cyxwiz::PendingExecutionDeviceSelection committed;
    cyxwiz::DeviceSelectionTransactionHooks hooks;
    hooks.inventory = [&] {
        ++inventory_calls;
        return std::vector<cyxwiz::DeviceInfo>{route};
    };
    hooks.qualify = [](const auto&, std::string&) { return true; };
    hooks.activate = [](auto type, int device_id) {
        cyxwiz::DeviceActivationResult result;
        result.requested_type = type;
        result.requested_device_id = device_id;
        result.effective_type = type;
        result.effective_device_id = device_id;
        result.success = true;
        result.execution_validated = true;
        result.stage = cyxwiz::DeviceActivationStage::Complete;
        return result;
    };
    hooks.restore = [] {
        cyxwiz::DeviceActivationResult result;
        result.success = true;
        result.stage = cyxwiz::DeviceActivationStage::Complete;
        return result;
    };
    hooks.commit = [&](const auto& selection) {
        ++commit_calls;
        committed = selection;
    };

    const auto result =
        cyxwiz::RunDeviceSelectionTransaction(candidate, hooks);
    CHECK(result.committed);
    CHECK(result.status ==
          cyxwiz::DeviceSelectionTransactionStatus::Committed);
    CHECK(result.stage == cyxwiz::DeviceSelectionTransactionStage::Complete);
    CHECK(inventory_calls == 2);
    CHECK(commit_calls == 1);
    CHECK(committed.type == candidate.type);
    CHECK(committed.device_id == candidate.device_id);
    CHECK(committed.physical_fingerprint == candidate.physical_fingerprint);
}

TEST_CASE("Route qualification snapshot rejects unsafe and stale evidence",
          "[device][selection][qualification]") {
    struct SnapshotReset {
        ~SnapshotReset() { cyxwiz::ClearRouteQualificationSnapshot(); }
    } reset;
    cyxwiz::ClearRouteQualificationSnapshot();

    cyxwiz::DeviceInfo route{};
    route.type = cyxwiz::DeviceType::CUDA;
    route.device_id = 2;
    route.provider = "NVIDIA CUDA";
    route.provider_known = true;
    route.driver_version = "581.57";
    route.driver_version_known = true;
    route.physical_fingerprint = "uuid:selected";
    route.physical_fingerprint_known = true;

    auto record = cyxwiz::RouteQualificationRecord{};
    record.type = route.type;
    record.device_id = route.device_id;
    record.provider = route.provider;
    record.driver_version = route.driver_version;
    record.physical_fingerprint = route.physical_fingerprint;
#ifdef CYXWIZ_HAS_ARRAYFIRE
    record.runtime_version = AF_VERSION;
#endif
    record.operation_count = cyxwiz::kRouteQualificationOperationCount;
    record.pass_count = cyxwiz::kRouteQualificationOperationCount;
    record.certified = true;
    record.display_name = "NVIDIA Test GPU";
    record.device_kind = cyxwiz::DeviceKind::GPU;
    record.device_kind_known = true;
    record.identity_source = "test_inventory";

    cyxwiz::RouteQualificationSnapshot snapshot;
    snapshot.matrix_id = "test-matrix";
    snapshot.routes = {record};
    cyxwiz::InstallRouteQualificationSnapshot(snapshot);

    const auto qualified = cyxwiz::EvaluateRouteQualification(route);
    CHECK(qualified.qualified);
    CHECK(qualified.evidence_available);
    CHECK(qualified.matrix_id == "test-matrix");
    CHECK(qualified.display_name == "NVIDIA Test GPU");
    CHECK(qualified.display_name_available);
    CHECK(qualified.device_kind == cyxwiz::DeviceKind::GPU);
    CHECK(qualified.device_kind_known);
    CHECK(qualified.identity_source == "test_inventory");
    const auto ready =
        cyxwiz::EvaluateRouteTrainingAuthorization(route, qualified);
    CHECK(ready.authorized);
    CHECK(ready.status ==
          cyxwiz::RouteTrainingAuthorizationStatus::Ready);

    auto wrong_runtime = record;
    wrong_runtime.runtime_version = "0.0.0";
    snapshot.routes = {wrong_runtime};
    cyxwiz::InstallRouteQualificationSnapshot(snapshot);
    const auto runtime_stale = cyxwiz::EvaluateRouteQualification(route);
    CHECK_FALSE(runtime_stale.qualified);
    CHECK_FALSE(runtime_stale.display_name_available);
    CHECK(runtime_stale.message.find("runtime version differs") !=
          std::string::npos);

    snapshot.routes = {record};
    snapshot.compute_contract_id = "obsolete-compute-contract";
    snapshot.operation_manifest_id =
        cyxwiz::kRouteQualificationOperationManifestId;
    cyxwiz::InstallRouteQualificationSnapshot(snapshot);
    const auto contract_stale = cyxwiz::EvaluateRouteQualification(route);
    CHECK_FALSE(contract_stale.qualified);
    CHECK(contract_stale.failure.category ==
          cyxwiz::RouteFailureCategory::EvidenceStale);
    CHECK(contract_stale.message.find("different compute contract") !=
          std::string::npos);
    snapshot.compute_contract_id.clear();
    snapshot.operation_manifest_id.clear();

    auto unsafe = record;
    unsafe.pass_count = unsafe.operation_count - 1;
    unsafe.crash_count = 1;
    unsafe.certified = false;
    snapshot.routes = {unsafe};
    cyxwiz::InstallRouteQualificationSnapshot(snapshot);
    const auto crash = cyxwiz::EvaluateRouteQualification(route);
    CHECK_FALSE(crash.qualified);
    CHECK(crash.evidence_available);
    CHECK(crash.message.find("crash=1") != std::string::npos);
    const auto rejected =
        cyxwiz::EvaluateRouteTrainingAuthorization(route, crash);
    CHECK_FALSE(rejected.authorized);
    CHECK(rejected.status ==
          cyxwiz::RouteTrainingAuthorizationStatus::MatrixRejected);

    cyxwiz::RouteQualificationSnapshot identity_snapshot;
    identity_snapshot.matrix_id = "test-matrix";
    identity_snapshot.routes = {record};
    cyxwiz::InstallRouteQualificationSnapshot(
        std::move(identity_snapshot));
    route.physical_fingerprint = "uuid:replacement";
    const auto stale = cyxwiz::EvaluateRouteQualification(route);
    CHECK_FALSE(stale.qualified);
    CHECK_FALSE(stale.display_name_available);
    CHECK(stale.message.find("identity differs") != std::string::npos);

    cyxwiz::ClearRouteQualificationSnapshot();
    const auto missing = cyxwiz::EvaluateRouteQualification(route);
    CHECK_FALSE(missing.qualified);
    CHECK_FALSE(missing.evidence_available);
    const auto no_evidence =
        cyxwiz::EvaluateRouteTrainingAuthorization(route, missing);
    CHECK(no_evidence.status ==
          cyxwiz::RouteTrainingAuthorizationStatus::NoEvidence);

    route.type = cyxwiz::DeviceType::ONEAPI;
    const auto exact_route_ready =
        cyxwiz::EvaluateRouteTrainingAuthorization(route, qualified);
    CHECK(exact_route_ready.authorized);
    CHECK(exact_route_ready.status ==
          cyxwiz::RouteTrainingAuthorizationStatus::Ready);
}

TEST_CASE("Route qualification JSON is validated before installation",
          "[device][selection][qualification]") {
    struct SnapshotReset {
        ~SnapshotReset() { cyxwiz::ClearRouteQualificationSnapshot(); }
    } reset;
    cyxwiz::ClearRouteQualificationSnapshot();

    const auto path = std::filesystem::temp_directory_path() /
        "cyxwiz-route-qualification-test.json";
    {
        std::ofstream output(path, std::ios::trunc);
        REQUIRE(output.good());
        output << R"({
  "schema": 1,
  "matrix_id": "json-test",
  "pack_id": "test-pack",
  "captured_at": "2026-08-12T00:00:00Z",
  "report_sha256": "test",
  "routes": [{
    "backend": "opencl",
    "device_id": 1,
    "physical_fingerprint": "pci:8086:0000:00:02.0",
    "provider": null,
    "driver_version": null,
    "runtime_version": "3.10.0",
    "display_name": "Intel(R) Test Graphics",
    "device_kind": "gpu",
    "identity_source": "test_selector",
    "operation_count": )" << cyxwiz::kRouteQualificationOperationCount << R"(,
    "pass_count": )" << cyxwiz::kRouteQualificationOperationCount << R"(,
    "unavailable_count": 0,
    "failure_count": 0,
    "timeout_count": 0,
    "crash_count": 0,
    "certified": true
  }]
})";
    }

    const auto loaded =
        cyxwiz::LoadAndInstallRouteQualificationSnapshot(path);
    std::error_code remove_error;
    std::filesystem::remove(path, remove_error);
    INFO(loaded.message);
    REQUIRE(loaded.loaded);
    CHECK(loaded.matrix_id == "json-test");
    CHECK(loaded.route_count == 1);

    cyxwiz::DeviceInfo route{};
    route.type = cyxwiz::DeviceType::OPENCL;
    route.device_id = 1;
    route.physical_fingerprint = "pci:8086:0000:00:02.0";
    route.physical_fingerprint_known = true;
    const auto decision = cyxwiz::EvaluateRouteQualification(route);
    CHECK(decision.qualified);
    CHECK(decision.display_name == "Intel(R) Test Graphics");
    CHECK(decision.device_kind == cyxwiz::DeviceKind::GPU);
    CHECK(decision.identity_source == "test_selector");
}

TEST_CASE("Qualification evidence must cover the complete operation manifest",
          "[device][selection][qualification][evidence]") {
    RouteQualificationStateGuard state_guard;
    const auto root = std::filesystem::temp_directory_path() /
        "cyxwiz-route-qualification-completeness";
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);

    cyxwiz::RouteQualificationRecord incomplete;
    incomplete.type = cyxwiz::DeviceType::CPU;
    incomplete.device_id = 0;
    incomplete.operation_count =
        cyxwiz::kRouteQualificationOperationCount - 1;
    incomplete.pass_count = incomplete.operation_count;
    incomplete.certified = true;
    cyxwiz::RouteQualificationSnapshot snapshot;
    snapshot.matrix_id = "incomplete-evidence";
    snapshot.compute_contract_id = cyxwiz::kCyxWizComputeContractId;
    snapshot.operation_manifest_id =
        cyxwiz::kRouteQualificationOperationManifestId;
    snapshot.routes = {incomplete};
    cyxwiz::InstallRouteQualificationSnapshot(snapshot);

    cyxwiz::DeviceInfo route;
    route.type = cyxwiz::DeviceType::CPU;
    route.device_id = 0;
    const auto decision = cyxwiz::EvaluateRouteQualification(route);
    CHECK_FALSE(decision.qualified);
    CHECK(decision.evidence_available);
    CHECK(decision.failure.category ==
          cyxwiz::RouteFailureCategory::MalformedEvidence);

    std::string error;
    CHECK_FALSE(cyxwiz::SaveRouteQualificationSnapshotAtomic(
        root / "incomplete.json", snapshot, error));
    CHECK(error.find("current operation manifest") != std::string::npos);

    auto complete = incomplete;
    complete.operation_count = cyxwiz::kRouteQualificationOperationCount;
    complete.pass_count = cyxwiz::kRouteQualificationOperationCount;
    snapshot.routes = {complete, complete};
    error.clear();
    CHECK_FALSE(cyxwiz::SaveRouteQualificationSnapshotAtomic(
        root / "duplicate.json", snapshot, error));
    CHECK(error.find("duplicate route") != std::string::npos);
    std::filesystem::remove_all(root, cleanup_error);
}

TEST_CASE("Machine compute runtime configuration persists atomically",
          "[device][selection][runtime-config]") {
    const auto root = std::filesystem::temp_directory_path() /
        "cyxwiz-compute-runtime-config-test";
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
    const cyxwiz::ScopedComputeRuntimeRootOverrideForTesting override(root);
    const auto path = cyxwiz::GetComputeRuntimeConfigPath();

    cyxwiz::ComputeRuntimeConfig config;
    config.default_fallback_policy =
        cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback;
    config.preferred_route = cyxwiz::PreferredComputeRoute{
        cyxwiz::DeviceType::OPENCL, 2, "pci:8086:0000:00:02.0"};
    std::string error;
    REQUIRE(cyxwiz::SaveComputeRuntimeConfigAtomic(path, config, error));
    INFO(error);

    const auto loaded = cyxwiz::LoadComputeRuntimeConfig(path);
    INFO(loaded.message);
    REQUIRE(loaded.loaded);
    REQUIRE(loaded.config.preferred_route.has_value());
    CHECK(loaded.config.preferred_route->type ==
          cyxwiz::DeviceType::OPENCL);
    CHECK(loaded.config.preferred_route->last_device_id == 2);
    CHECK(loaded.config.preferred_route->physical_fingerprint ==
          "pci:8086:0000:00:02.0");
    CHECK(loaded.config.default_fallback_policy ==
          cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);

    REQUIRE(cyxwiz::UpdatePreferredComputeRouteAtomic(
        path,
        {cyxwiz::DeviceType::CUDA, 0, "uuid:nvidia"},
        error));
    REQUIRE(cyxwiz::UpdateDefaultFallbackPolicyAtomic(
        path,
        cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback,
        error));
    const auto updated = cyxwiz::LoadComputeRuntimeConfig(path);
    REQUIRE(updated.loaded);
    REQUIRE(updated.config.preferred_route.has_value());
    CHECK(updated.config.preferred_route->type == cyxwiz::DeviceType::CUDA);
    CHECK(updated.config.default_fallback_policy ==
          cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);

    {
        std::ofstream corrupt(path, std::ios::binary | std::ios::trunc);
        corrupt << "{not-json";
    }
    const auto invalid = cyxwiz::LoadComputeRuntimeConfig(path);
    CHECK(invalid.file_exists);
    CHECK_FALSE(invalid.loaded);
    CHECK_FALSE(cyxwiz::UpdatePreferredComputeRouteAtomic(
        path,
        {cyxwiz::DeviceType::CPU, 0, {}},
        error));
    CHECK(error.find("invalid") != std::string::npos);
    std::filesystem::remove_all(root, cleanup_error);
}

TEST_CASE("Qualification cache preserves typed failure diagnostics",
          "[device][selection][qualification][diagnostics]") {
    struct SnapshotReset {
        ~SnapshotReset() { cyxwiz::ClearRouteQualificationSnapshot(); }
    } reset;
    const auto root = std::filesystem::temp_directory_path() /
        "cyxwiz-route-diagnostic-cache-test";
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
    const auto path = root / "route-qualification.json";

    cyxwiz::RouteQualificationRecord record;
    record.type = cyxwiz::DeviceType::ONEAPI;
    record.device_id = 1;
    record.display_name = "Intel Test CPU";
    record.device_kind = cyxwiz::DeviceKind::CPU;
    record.device_kind_known = true;
    record.identity_source = "test_selector";
    record.operation_count = cyxwiz::kRouteQualificationOperationCount;
    record.pass_count = cyxwiz::kRouteQualificationOperationCount - 1;
    record.timeout_count = 1;
    record.failure.stage = cyxwiz::RouteFailureStage::Operation;
    record.failure.category = cyxwiz::RouteFailureCategory::Timeout;
    record.failure.operation = "sum";
    record.failure.probe_stage = "expression_begin";
    record.failure.timeout_ms = 20000;
    record.failure.observed_fact =
        "Operation 'sum' timed out after 20000 ms at expression_begin";
    record.failure.bounded_interpretation =
        "The exact route did not complete a required reduction";
    record.failure.recommended_action =
        "Update the provider and verify again";

    cyxwiz::RouteQualificationSnapshot snapshot;
    snapshot.matrix_id = "diagnostic-roundtrip";
    snapshot.routes = {record};
    std::string error;
    REQUIRE(cyxwiz::SaveRouteQualificationSnapshotAtomic(
        path, snapshot, error));
    INFO(error);
    const auto loaded =
        cyxwiz::LoadAndInstallRouteQualificationSnapshot(path);
    INFO(loaded.message);
    REQUIRE(loaded.loaded);

    cyxwiz::DeviceInfo route;
    route.type = cyxwiz::DeviceType::ONEAPI;
    route.device_id = 1;
    const auto decision = cyxwiz::EvaluateRouteQualification(route);
    CHECK_FALSE(decision.qualified);
    CHECK(decision.display_name == "Intel Test CPU");
    CHECK(decision.failure.category ==
          cyxwiz::RouteFailureCategory::Timeout);
    CHECK(decision.failure.operation == "sum");
    CHECK(decision.failure.probe_stage == "expression_begin");
    CHECK(decision.failure.timeout_ms == 20000);
    CHECK(decision.failure.observed_fact.find("sum") != std::string::npos);
    CHECK(std::string(cyxwiz::RouteFailureCategoryName(
              decision.failure.category)) == "timeout");

    const auto authorization =
        cyxwiz::EvaluateRouteTrainingAuthorization(route, decision);
    CHECK(authorization.status ==
          cyxwiz::RouteTrainingAuthorizationStatus::MatrixRejected);
    CHECK(authorization.failure.category ==
          cyxwiz::RouteFailureCategory::Timeout);
    std::filesystem::remove_all(root, cleanup_error);
}

TEST_CASE("Qualification service publishes a complete exact-route matrix",
          "[device][selection][qualification][service]") {
    RouteQualificationStateGuard state_guard;
    cyxwiz::ClearRouteQualificationSnapshot();
    const auto root = std::filesystem::temp_directory_path() /
        "cyxwiz-route-qualification-service-pass";
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);

    std::vector<std::string> operations;
    cyxwiz::RouteQualificationService service(
        [&](const cyxwiz::RouteProbeInvocation& invocation,
            const cyxwiz::RouteQualificationCancelCheck&) {
            operations.push_back(invocation.operation);
            cyxwiz::RouteProbeResult result;
            result.status = cyxwiz::RouteProbeStatus::Passed;
            result.output = "probe_event stage=read_complete";
            return result;
        });
    cyxwiz::DeviceInfo route;
    route.type = cyxwiz::DeviceType::OPENCL;
    route.device_id = 2;
    route.name = "Intel Test CPU";
    route.name_known = true;
    route.kind = cyxwiz::DeviceKind::CPU;
    route.provider = "Intel(R) Corporation";
    route.provider_known = true;
    route.driver_version = "test-driver";
    route.driver_version_known = true;
    route.physical_fingerprint = "pci:8086:0000:00:02.0";
    route.physical_fingerprint_known = true;
    route.metadata_status = cyxwiz::DeviceMetadataStatus::Available;

    cyxwiz::RouteQualificationOptions options;
    options.probe_executable = root / "fake-probe";
    options.cache_path = root / "route-qualification.json";
    options.matrix_id = "service-pass";
    options.pack_id = "test-pack";
    std::vector<cyxwiz::RouteQualificationProgress> progress;
    const auto result = service.VerifyRoute(
        route, options,
        [&](const auto& update) { progress.push_back(update); });

    INFO(result.message);
    REQUIRE(result.status ==
            cyxwiz::RouteQualificationRunStatus::Completed);
    REQUIRE(result.published);
    REQUIRE(result.snapshot.has_value());
    REQUIRE(result.snapshot->routes.size() == 1);
    const auto& record = result.snapshot->routes.front();
    CHECK(record.certified);
    CHECK(record.operation_count == static_cast<int>(
              cyxwiz::RequiredRouteQualificationOperations().size()));
    CHECK(record.pass_count == record.operation_count);
    CHECK(record.physical_fingerprint == route.physical_fingerprint);
    CHECK(record.provider == route.provider);
    CHECK(record.display_name == route.name);
    CHECK(record.device_kind == cyxwiz::DeviceKind::CPU);
    CHECK(operations.size() ==
          cyxwiz::RequiredRouteQualificationOperations().size());
    CHECK(operations.front() == "route_metadata");
    CHECK(operations.back() == "cyxwiz_dropout_forward_backward");
    REQUIRE_FALSE(progress.empty());
    CHECK(progress.back().status ==
          cyxwiz::RouteQualificationRunStatus::Completed);
    CHECK(std::filesystem::is_regular_file(options.cache_path));

    const auto installed = cyxwiz::EvaluateRouteQualification(route);
    CHECK(installed.qualified);
    CHECK(installed.matrix_id == options.matrix_id);
    std::filesystem::remove_all(root, cleanup_error);
}

TEST_CASE("Qualification service rejects duplicate exact routes",
          "[device][selection][qualification][service]") {
    RouteQualificationStateGuard state_guard;
    const auto root = std::filesystem::temp_directory_path() /
        "cyxwiz-route-qualification-service-duplicate";
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);

    bool probe_called = false;
    cyxwiz::RouteQualificationService service(
        [&](const cyxwiz::RouteProbeInvocation&,
            const cyxwiz::RouteQualificationCancelCheck&) {
            probe_called = true;
            return cyxwiz::RouteProbeResult{};
        });
    cyxwiz::DeviceInfo route;
    route.type = cyxwiz::DeviceType::CPU;
    route.device_id = 0;
    cyxwiz::RouteQualificationOptions options;
    options.probe_executable = root / "fake-probe";
    options.cache_path = root / "route-qualification.json";
    options.matrix_id = "duplicate-routes";
    options.pack_id = "test-pack";

    const auto result = service.VerifyAll({route, route}, options);
    CHECK(result.status ==
          cyxwiz::RouteQualificationRunStatus::InvalidRequest);
    CHECK_FALSE(result.published);
    CHECK_FALSE(probe_called);
    CHECK_FALSE(std::filesystem::exists(options.cache_path));
    std::filesystem::remove_all(root, cleanup_error);
}

TEST_CASE("Verify all benchmarks certified routes and recommends the fastest",
          "[device][selection][qualification][service][benchmark]") {
    RouteQualificationStateGuard state_guard;
    const auto root = std::filesystem::temp_directory_path() /
        "cyxwiz-route-qualification-service-benchmark";
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);

    std::vector<std::pair<cyxwiz::DeviceType, int>> benchmarked;
    cyxwiz::RouteQualificationService service(
        [&](const cyxwiz::RouteProbeInvocation& invocation,
            const cyxwiz::RouteQualificationCancelCheck&) {
            cyxwiz::RouteProbeResult result;
            if (invocation.operation == "dense_compute_benchmark") {
                benchmarked.emplace_back(
                    invocation.type, invocation.device_id);
                const double median =
                    invocation.type == cyxwiz::DeviceType::CUDA ? 2.5 : 8.0;
                result.status = cyxwiz::RouteProbeStatus::Passed;
                result.output =
                    "benchmark_result benchmark_id=cyxwiz-dense-compute-v1 "
                    "samples=5 iterations_per_sample=3 median_iteration_ms=" +
                    std::to_string(median);
                return result;
            }
            if (invocation.type == cyxwiz::DeviceType::ONEAPI &&
                invocation.operation == "sum") {
                result.status = cyxwiz::RouteProbeStatus::Crashed;
                result.last_probe_stage = "expression_begin";
                return result;
            }
            result.status = cyxwiz::RouteProbeStatus::Passed;
            return result;
        });

    cyxwiz::DeviceInfo cpu;
    cpu.type = cyxwiz::DeviceType::CPU;
    cpu.device_id = 0;
    cyxwiz::DeviceInfo cuda;
    cuda.type = cyxwiz::DeviceType::CUDA;
    cuda.device_id = 0;
    cyxwiz::DeviceInfo oneapi;
    oneapi.type = cyxwiz::DeviceType::ONEAPI;
    oneapi.device_id = 0;

    cyxwiz::RouteQualificationOptions options;
    options.probe_executable = root / "fake-probe";
    options.cache_path = root / "route-qualification.json";
    options.matrix_id = cyxwiz::kRouteQualificationMatrixId;
    options.pack_id = "test-pack";
    options.benchmark_verified_routes = true;
    const auto result = service.VerifyAll({cpu, cuda, oneapi}, options);

    INFO(result.message);
    REQUIRE(result.published);
    REQUIRE(result.snapshot.has_value());
    REQUIRE(benchmarked.size() == 2);
    CHECK(std::find(benchmarked.begin(), benchmarked.end(),
                    std::make_pair(cyxwiz::DeviceType::ONEAPI, 0)) ==
          benchmarked.end());
    const auto fastest =
        cyxwiz::RecommendFastestVerifiedRoute(result.snapshot);
    REQUIRE(fastest.has_value());
    CHECK(fastest->type == cyxwiz::DeviceType::CUDA);
    CHECK(fastest->device_id == 0);
    CHECK(fastest->benchmark_id ==
          cyxwiz::kRoutePerformanceBenchmarkId);
    CHECK(fastest->sample_count == 5);
    CHECK(fastest->iterations_per_sample == 3);
    CHECK(fastest->median_iteration_ms == Catch::Approx(2.5));

    cyxwiz::ClearRouteQualificationSnapshot();
    const auto loaded = cyxwiz::LoadAndInstallRouteQualificationSnapshot(
        options.cache_path);
    REQUIRE(loaded.loaded);
    const auto persisted_fastest = cyxwiz::RecommendFastestVerifiedRoute(
        cyxwiz::GetRouteQualificationSnapshot());
    REQUIRE(persisted_fastest.has_value());
    CHECK(persisted_fastest->type == cyxwiz::DeviceType::CUDA);
    CHECK(persisted_fastest->median_iteration_ms == Catch::Approx(2.5));
    std::filesystem::remove_all(root, cleanup_error);
}

TEST_CASE("Qualification service attributes the first exact operation failure",
          "[device][selection][qualification][service][diagnostics]") {
    RouteQualificationStateGuard state_guard;
    const auto root = std::filesystem::temp_directory_path() /
        "cyxwiz-route-qualification-service-failure";
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);

    cyxwiz::RouteQualificationService service(
        [](const cyxwiz::RouteProbeInvocation& invocation,
           const cyxwiz::RouteQualificationCancelCheck&) {
            cyxwiz::RouteProbeResult result;
            if (invocation.operation == "sum") {
                result.status = cyxwiz::RouteProbeStatus::Crashed;
                result.exit_code = static_cast<int>(0xC0000005u);
                result.output =
                    "probe_event operation=sum stage=expression_begin";
                result.last_probe_stage = "expression_begin";
            } else {
                result.status = cyxwiz::RouteProbeStatus::Passed;
            }
            return result;
        });
    cyxwiz::DeviceInfo route;
    route.type = cyxwiz::DeviceType::ONEAPI;
    route.device_id = 0;
    route.name = "Intel Test GPU";
    route.name_known = true;
    route.kind = cyxwiz::DeviceKind::GPU;
    route.metadata_status = cyxwiz::DeviceMetadataStatus::Available;

    cyxwiz::RouteQualificationOptions options;
    options.probe_executable = root / "fake-probe";
    options.cache_path = root / "route-qualification.json";
    options.matrix_id = "service-crash";
    options.pack_id = "test-pack";
    const auto result = service.VerifyAll({route}, options);

    INFO(result.message);
    REQUIRE(result.published);
    REQUIRE(result.snapshot.has_value());
    REQUIRE(result.snapshot->routes.size() == 1);
    const auto& record = result.snapshot->routes.front();
    CHECK_FALSE(record.certified);
    CHECK(record.crash_count == 1);
    CHECK(record.pass_count + record.crash_count == record.operation_count);
    CHECK(record.failure.stage == cyxwiz::RouteFailureStage::Operation);
    CHECK(record.failure.category ==
          cyxwiz::RouteFailureCategory::ChildProcessCrash);
    CHECK(record.failure.operation == "sum");
    CHECK(record.failure.probe_stage == "expression_begin");
    CHECK(record.failure.observed_fact.find("sum") != std::string::npos);

    const auto installed = cyxwiz::EvaluateRouteQualification(route);
    CHECK_FALSE(installed.qualified);
    CHECK(installed.failure.category ==
          cyxwiz::RouteFailureCategory::ChildProcessCrash);
    std::filesystem::remove_all(root, cleanup_error);
}

TEST_CASE("Qualification cancellation preserves previously accepted evidence",
          "[device][selection][qualification][service][cancel]") {
    RouteQualificationStateGuard state_guard;
    const auto root = std::filesystem::temp_directory_path() /
        "cyxwiz-route-qualification-service-cancel";
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
    const auto cache_path = root / "route-qualification.json";

    cyxwiz::RouteQualificationRecord accepted_record;
    accepted_record.type = cyxwiz::DeviceType::CPU;
    accepted_record.device_id = 0;
    accepted_record.operation_count =
        cyxwiz::kRouteQualificationOperationCount;
    accepted_record.pass_count =
        cyxwiz::kRouteQualificationOperationCount;
    accepted_record.certified = true;
    cyxwiz::RouteQualificationSnapshot accepted;
    accepted.matrix_id = "accepted-before-cancel";
    accepted.pack_id = "accepted-pack";
    accepted.routes = {accepted_record};
    std::string save_error;
    REQUIRE(cyxwiz::SaveRouteQualificationSnapshotAtomic(
        cache_path, accepted, save_error));
    cyxwiz::InstallRouteQualificationSnapshot(accepted);
    std::ifstream before_stream(cache_path, std::ios::binary);
    const std::string before{
        std::istreambuf_iterator<char>(before_stream),
        std::istreambuf_iterator<char>()};

    cyxwiz::RouteQualificationService* service_pointer = nullptr;
    cyxwiz::RouteQualificationService service(
        [&](const cyxwiz::RouteProbeInvocation&,
            const cyxwiz::RouteQualificationCancelCheck& should_cancel) {
            service_pointer->Cancel();
            CHECK(should_cancel());
            cyxwiz::RouteProbeResult result;
            result.status = cyxwiz::RouteProbeStatus::Cancelled;
            return result;
        });
    service_pointer = &service;
    cyxwiz::DeviceInfo route;
    route.type = cyxwiz::DeviceType::CUDA;
    route.device_id = 0;
    cyxwiz::RouteQualificationOptions options;
    options.probe_executable = root / "fake-probe";
    options.cache_path = cache_path;
    options.matrix_id = "replacement-cancelled";
    options.pack_id = "replacement-pack";

    const auto result = service.VerifyAll({route}, options);
    CHECK(result.status == cyxwiz::RouteQualificationRunStatus::Cancelled);
    CHECK_FALSE(result.published);
    CHECK_FALSE(result.snapshot.has_value());
    const auto installed = cyxwiz::GetRouteQualificationSnapshot();
    REQUIRE(installed.has_value());
    CHECK(installed->matrix_id == accepted.matrix_id);
    std::ifstream after_stream(cache_path, std::ios::binary);
    const std::string after{
        std::istreambuf_iterator<char>(after_stream),
        std::istreambuf_iterator<char>()};
    CHECK(after == before);
    std::filesystem::remove_all(root, cleanup_error);
}

TEST_CASE("Single-route verification does not relabel legacy evidence",
          "[device][selection][qualification][service][migration]") {
    RouteQualificationStateGuard state_guard;
    const auto root = std::filesystem::temp_directory_path() /
        "cyxwiz-route-qualification-service-legacy";
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);

    cyxwiz::RouteQualificationRecord legacy_record;
    legacy_record.type = cyxwiz::DeviceType::CPU;
    legacy_record.device_id = 0;
    legacy_record.operation_count =
        cyxwiz::kRouteQualificationOperationCount;
    legacy_record.pass_count =
        cyxwiz::kRouteQualificationOperationCount;
    legacy_record.certified = true;
    cyxwiz::RouteQualificationSnapshot legacy;
    legacy.matrix_id = "shared-matrix";
    legacy.pack_id = "shared-pack";
    legacy.routes = {legacy_record};
    cyxwiz::InstallRouteQualificationSnapshot(legacy);

    cyxwiz::RouteQualificationService service(
        [](const cyxwiz::RouteProbeInvocation&,
           const cyxwiz::RouteQualificationCancelCheck&) {
            cyxwiz::RouteProbeResult result;
            result.status = cyxwiz::RouteProbeStatus::Passed;
            return result;
        });
    cyxwiz::DeviceInfo measured;
    measured.type = cyxwiz::DeviceType::OPENCL;
    measured.device_id = 1;
    cyxwiz::RouteQualificationOptions options;
    options.probe_executable = root / "fake-probe";
    options.cache_path = root / "route-qualification.json";
    options.matrix_id = legacy.matrix_id;
    options.pack_id = legacy.pack_id;
    const auto result = service.VerifyRoute(measured, options);

    REQUIRE(result.published);
    REQUIRE(result.snapshot.has_value());
    CHECK(result.snapshot->compute_contract_id ==
          cyxwiz::kCyxWizComputeContractId);
    CHECK(result.snapshot->operation_manifest_id ==
          cyxwiz::kRouteQualificationOperationManifestId);
    REQUIRE(result.snapshot->routes.size() == 1);
    CHECK(result.snapshot->routes.front().type ==
          cyxwiz::DeviceType::OPENCL);
    std::filesystem::remove_all(root, cleanup_error);
}

TEST_CASE("Staged runtime verification keeps only unchanged pack evidence",
          "[device][selection][qualification][service][runtime-pack]") {
    RouteQualificationStateGuard state_guard;
    const auto root = std::filesystem::temp_directory_path() /
        "cyxwiz-runtime-pack-qualification-selective";
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);

    cyxwiz::RouteQualificationSnapshot accepted;
    accepted.matrix_id = cyxwiz::kRouteQualificationMatrixId;
    accepted.pack_id = "set-v1";
    accepted.runtime_set_id = "set-v1";
    accepted.runtime_generation = 4;
    accepted.base_pack_id = "base-v1";
    accepted.compute_contract_id = cyxwiz::kCyxWizComputeContractId;
    accepted.operation_manifest_id =
        cyxwiz::kRouteQualificationOperationManifestId;
    for (const auto& identity : {
             std::tuple{cyxwiz::DeviceType::CPU, 0, "base-v1"},
             std::tuple{cyxwiz::DeviceType::CUDA, 0, "cuda-v1"},
             std::tuple{cyxwiz::DeviceType::OPENCL, 0, "opencl-v1"}}) {
        cyxwiz::RouteQualificationRecord record;
        record.type = std::get<0>(identity);
        record.device_id = std::get<1>(identity);
        record.pack_id = std::get<2>(identity);
        record.operation_count = cyxwiz::kRouteQualificationOperationCount;
        record.pass_count = record.operation_count;
        record.certified = true;
        accepted.routes.push_back(std::move(record));
    }
    cyxwiz::InstallRouteQualificationSnapshot(accepted);

    cyxwiz::RouteQualificationService service(
        [](const cyxwiz::RouteProbeInvocation&,
           const cyxwiz::RouteQualificationCancelCheck&) {
            cyxwiz::RouteProbeResult result;
            result.status = cyxwiz::RouteProbeStatus::Passed;
            return result;
        });
    cyxwiz::RuntimeQualificationIdentity runtime;
    runtime.runtime_set_id = "set-v1";
    runtime.generation = 5;
    runtime.base_pack_id = "base-v1";
    runtime.backend_packs = {
        {cyxwiz::DeviceType::CUDA, "cuda-v1"},
        {cyxwiz::DeviceType::OPENCL, "opencl-v2"}};
    cyxwiz::DeviceInfo affected;
    affected.type = cyxwiz::DeviceType::OPENCL;
    affected.device_id = 0;
    cyxwiz::RouteQualificationOptions options;
    options.probe_executable = root / "fake-probe";
    options.cache_path = root / "route-qualification.json";
    options.matrix_id = cyxwiz::kRouteQualificationMatrixId;

    const auto result = service.VerifyStagedRuntimeRoutes(
        {affected}, runtime,
        cyxwiz::RuntimeQualificationFailurePolicy::KeepInstalledUnqualified,
        options);

    INFO(result.qualification.message);
    REQUIRE(result.disposition ==
            cyxwiz::RuntimeQualificationDisposition::Qualified);
    REQUIRE(result.qualification.snapshot.has_value());
    CHECK(result.qualification.snapshot->runtime_generation == 5);
    REQUIRE(result.qualification.snapshot->routes.size() == 3);
    const auto pack_for = [&](cyxwiz::DeviceType type) {
        const auto route = std::find_if(
            result.qualification.snapshot->routes.begin(),
            result.qualification.snapshot->routes.end(),
            [type](const auto& record) { return record.type == type; });
        REQUIRE(route != result.qualification.snapshot->routes.end());
        return route->pack_id;
    };
    CHECK(pack_for(cyxwiz::DeviceType::CPU) == "base-v1");
    CHECK(pack_for(cyxwiz::DeviceType::CUDA) == "cuda-v1");
    CHECK(pack_for(cyxwiz::DeviceType::OPENCL) == "opencl-v2");

    cyxwiz::ClearRouteQualificationSnapshot();
    const auto loaded = cyxwiz::LoadAndInstallRouteQualificationSnapshot(
        options.cache_path);
    REQUIRE(loaded.loaded);
    const auto roundtrip = cyxwiz::GetRouteQualificationSnapshot();
    REQUIRE(roundtrip.has_value());
    CHECK(roundtrip->runtime_set_id == runtime.runtime_set_id);
    CHECK(roundtrip->base_pack_id == runtime.base_pack_id);
    CHECK(roundtrip->runtime_generation == runtime.generation);
    std::filesystem::remove_all(root, cleanup_error);
}

TEST_CASE("Staged runtime failure returns typed pack policy disposition",
          "[device][selection][qualification][service][runtime-pack][diagnostics]") {
    RouteQualificationStateGuard state_guard;
    const auto root = std::filesystem::temp_directory_path() /
        "cyxwiz-runtime-pack-qualification-policy";
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);

    cyxwiz::RouteQualificationService service(
        [](const cyxwiz::RouteProbeInvocation& invocation,
           const cyxwiz::RouteQualificationCancelCheck&) {
            cyxwiz::RouteProbeResult result;
            result.status = invocation.operation == "sum"
                ? cyxwiz::RouteProbeStatus::TimedOut
                : cyxwiz::RouteProbeStatus::Passed;
            result.last_probe_stage = "expression_begin";
            return result;
        });
    cyxwiz::RuntimeQualificationIdentity runtime;
    runtime.runtime_set_id = "set-v1";
    runtime.generation = 7;
    runtime.base_pack_id = "base-v1";
    runtime.backend_packs = {
        {cyxwiz::DeviceType::ONEAPI, "oneapi-v1"}};
    cyxwiz::DeviceInfo affected;
    affected.type = cyxwiz::DeviceType::ONEAPI;
    affected.device_id = 0;
    cyxwiz::RouteQualificationOptions options;
    options.probe_executable = root / "fake-probe";
    options.cache_path = root / "route-qualification.json";
    options.matrix_id = cyxwiz::kRouteQualificationMatrixId;
    options.operation_timeout = std::chrono::milliseconds(25);

    const auto keep = service.VerifyStagedRuntimeRoutes(
        {affected}, runtime,
        cyxwiz::RuntimeQualificationFailurePolicy::KeepInstalledUnqualified,
        options);
    REQUIRE(keep.qualification.published);
    CHECK(keep.disposition ==
          cyxwiz::RuntimeQualificationDisposition::InstalledUnqualified);
    CHECK(keep.diagnostic.category ==
          cyxwiz::RouteFailureCategory::Timeout);
    CHECK(keep.diagnostic.timeout_ms == 25);

    ++runtime.generation;
    options.cache_path = root / "route-qualification-rollback.json";
    const auto rollback = service.VerifyStagedRuntimeRoutes(
        {affected}, runtime,
        cyxwiz::RuntimeQualificationFailurePolicy::RequireRollback,
        options);
    REQUIRE(rollback.qualification.published);
    CHECK(rollback.disposition ==
          cyxwiz::RuntimeQualificationDisposition::RollbackRequired);
    CHECK(rollback.diagnostic.category ==
          cyxwiz::RouteFailureCategory::Timeout);
    std::filesystem::remove_all(root, cleanup_error);
}

TEST_CASE("Staged runtime verification propagates an exact candidate probe runtime",
          "[device][selection][qualification][service][runtime-pack][process]") {
    RouteQualificationStateGuard state_guard;
    const auto root = std::filesystem::temp_directory_path() /
        "cyxwiz-runtime-pack-candidate-probe";
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
    const auto base = root / "base" / "base-v1";
    const auto pack = root / "packs" / "opencl" / "opencl-v2";
    std::filesystem::create_directories(base);
    std::filesystem::create_directories(pack);

    std::vector<cyxwiz::RouteProbeInvocation> invocations;
    cyxwiz::RouteQualificationService service(
        [&](const cyxwiz::RouteProbeInvocation& invocation,
            const cyxwiz::RouteQualificationCancelCheck&) {
            invocations.push_back(invocation);
            cyxwiz::RouteProbeResult result;
            result.status = cyxwiz::RouteProbeStatus::Passed;
            if (invocation.operation == "dense_compute_benchmark") {
                result.output =
                    "benchmark_id=cyxwiz-dense-compute-v1 samples=3 "
                    "iterations_per_sample=2 median_iteration_ms=1.25";
            }
            return result;
        });
    cyxwiz::RuntimeQualificationIdentity identity;
    identity.runtime_set_id = "set-v1";
    identity.generation = 8;
    identity.base_pack_id = "base-v1";
    identity.backend_packs = {
        {cyxwiz::DeviceType::OPENCL, "opencl-v2"}};
    cyxwiz::DeviceInfo affected;
    affected.type = cyxwiz::DeviceType::OPENCL;
    affected.device_id = 1;
    cyxwiz::RouteQualificationOptions options;
    options.probe_executable = base / "cyxwiz-route-probe.exe";
    options.cache_path = root / "route-qualification.json";
    options.matrix_id = cyxwiz::kRouteQualificationMatrixId;
    options.runtime_identity = identity;
    options.benchmark_verified_routes = true;
    options.probe_runtime_root = root;
    options.probe_working_directory = base;
    options.probe_runtime_dll_directories = {pack, base};

    const auto result = service.VerifyStagedRuntimeRoutes(
        {affected}, identity,
        cyxwiz::RuntimeQualificationFailurePolicy::KeepInstalledUnqualified,
        options);

    INFO(result.qualification.message);
    REQUIRE(result.disposition ==
            cyxwiz::RuntimeQualificationDisposition::Qualified);
    REQUIRE(invocations.size() ==
            cyxwiz::RequiredRouteQualificationOperations().size() + 1);
    for (const auto& invocation : invocations) {
        CHECK(invocation.runtime_root == root);
        CHECK(invocation.working_directory == base);
        CHECK(invocation.runtime_dll_directories ==
              std::vector<std::filesystem::path>{pack, base});
        REQUIRE(invocation.runtime_identity.has_value());
        CHECK(invocation.runtime_identity->runtime_set_id ==
              identity.runtime_set_id);
        CHECK(invocation.runtime_identity->generation == identity.generation);
        REQUIRE(invocation.runtime_identity->backend_packs.size() == 1);
        CHECK(invocation.runtime_identity->backend_packs.front().pack_id ==
              "opencl-v2");
    }

    bool incomplete_probe_called = false;
    cyxwiz::RouteQualificationService incomplete_service(
        [&](const cyxwiz::RouteProbeInvocation&,
            const cyxwiz::RouteQualificationCancelCheck&) {
            incomplete_probe_called = true;
            return cyxwiz::RouteProbeResult{};
        });
    options.probe_working_directory.clear();
    const auto incomplete = incomplete_service.VerifyStagedRuntimeRoutes(
        {affected}, identity,
        cyxwiz::RuntimeQualificationFailurePolicy::KeepInstalledUnqualified,
        options);
    CHECK(incomplete.qualification.status ==
          cyxwiz::RouteQualificationRunStatus::InvalidRequest);
    CHECK_FALSE(incomplete_probe_called);
    std::filesystem::remove_all(root, cleanup_error);
}

#ifdef CYXWIZ_HAS_BACKEND_PACK_QUALIFICATION_ADAPTER
TEST_CASE("Backend pack lifecycle qualification uses exact staged routes",
          "[device][qualification][runtime-pack][lifecycle]") {
    RouteQualificationStateGuard state_guard;
    const auto root = std::filesystem::temp_directory_path() /
        "cyxwiz-backend-pack-qualification-adapter";
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
    const auto base = root / "base" / "base-v1";
    const auto pack = root / "packs" / "opencl" / "opencl-v2";
    std::filesystem::create_directories(pack / "runtime");
    std::filesystem::create_directories(base);
    std::ofstream(base / "cyxwiz-engine.exe", std::ios::binary).put('\0');
    std::ofstream(base / "cyxwiz-route-probe.exe", std::ios::binary).put('\0');
    std::ofstream(pack / "runtime" / "afopencl.dll", std::ios::binary).put('\0');

    std::vector<cyxwiz::RouteProbeInvocation> qualification_invocations;
    auto service = std::make_shared<cyxwiz::RouteQualificationService>(
        [&](const cyxwiz::RouteProbeInvocation& invocation,
            const cyxwiz::RouteQualificationCancelCheck&) {
            qualification_invocations.push_back(invocation);
            cyxwiz::RouteProbeResult result;
            result.status = cyxwiz::RouteProbeStatus::Passed;
            return result;
        });
    cyxwiz::BackendPackQualificationAdapterOptions options;
    options.runtime_root = std::filesystem::absolute(root);
    options.probe_executable =
        std::filesystem::absolute(base / "cyxwiz-route-probe.exe");
    options.cache_path =
        std::filesystem::absolute(root / "route-qualification.json");

    bool discovery_called = false;
    const auto hook = cyxwiz::CreateBackendPackQualificationHook(
        service, options,
        [&](cyxwiz::RouteProbeInvocation invocation,
            const cyxwiz::RouteQualificationCancelCheck&) {
            discovery_called = true;
            CHECK(invocation.enumerate_backend);
            CHECK(invocation.type == cyxwiz::DeviceType::OPENCL);
            CHECK(invocation.runtime_root ==
                  std::filesystem::canonical(root));
            REQUIRE(invocation.runtime_identity.has_value());
            CHECK(invocation.runtime_identity->generation == 2);
            cyxwiz::DeviceInfo route;
            route.type = cyxwiz::DeviceType::OPENCL;
            route.device_id = 0;
            route.name = "Fixture GPU";
            route.name_known = true;
            route.kind = cyxwiz::DeviceKind::GPU;
            route.identity_confidence =
                cyxwiz::DeviceIdentityConfidence::ProviderReported;
            route.metadata_status = cyxwiz::DeviceMetadataStatus::Available;
            cyxwiz::IsolatedRouteDiscoveryResult result;
            result.status = cyxwiz::RouteProbeStatus::Passed;
            result.routes.push_back(std::move(route));
            return result;
        });

    cyxwiz::runtime::VerifiedBackendPackManifest manifest;
    manifest.pack_id = "opencl-v2";
    manifest.backend = "opencl";
    manifest.runtime_set_id = "runtime-v1";
    manifest.companion_base_id = "base-v1";
    manifest.arrayfire_version = "3.10.signed-pack";
    manifest.compatibility.operation_matrix_id =
        cyxwiz::kRouteQualificationMatrixId;
    cyxwiz::runtime::ActiveRuntimeState candidate;
    candidate.runtime_set_id = "runtime-v1";
    candidate.generation = 2;
    candidate.base_pack_id = "base-v1";
    candidate.packs.push_back({"opencl", "opencl-v2"});

    const auto qualified = hook(manifest, pack, candidate);
    INFO(qualified.message);
    REQUIRE(discovery_called);
    CHECK(qualified.disposition ==
          cyxwiz::runtime::BackendPackQualificationDisposition::Qualified);
    const auto published = cyxwiz::GetRouteQualificationSnapshot();
    REQUIRE(published.has_value());
    REQUIRE(published->routes.size() == 1);
    CHECK(published->routes.front().runtime_version ==
          manifest.arrayfire_version);
    REQUIRE_FALSE(qualification_invocations.empty());
    for (const auto& invocation : qualification_invocations) {
        REQUIRE(invocation.runtime_identity.has_value());
        CHECK(invocation.runtime_identity->runtime_set_id == "runtime-v1");
        CHECK(invocation.runtime_identity->backend_packs.front().pack_id ==
              "opencl-v2");
        CHECK(invocation.runtime_dll_directories.back() == pack / "runtime");
    }

    options.failure_policy =
        cyxwiz::RuntimeQualificationFailurePolicy::RequireRollback;
    const auto empty_hook = cyxwiz::CreateBackendPackQualificationHook(
        service, options,
        [](cyxwiz::RouteProbeInvocation,
           const cyxwiz::RouteQualificationCancelCheck&) {
            cyxwiz::IsolatedRouteDiscoveryResult result;
            result.status = cyxwiz::RouteProbeStatus::Passed;
            result.message = "Candidate backend exposed no routes";
            return result;
        });
    const auto empty = empty_hook(manifest, pack, candidate);
    CHECK(empty.disposition ==
          cyxwiz::runtime::BackendPackQualificationDisposition::RollbackRequired);

    const auto mismatched = hook(
        manifest, root / "packs" / "opencl" / "different", candidate);
    CHECK(mismatched.disposition ==
          cyxwiz::runtime::BackendPackQualificationDisposition::InstalledUnqualified);
    CHECK(mismatched.message.find("does not match") != std::string::npos);
    std::filesystem::remove_all(root, cleanup_error);
}
#endif

TEST_CASE("Runtime pack removal invalidates only its retained routes",
          "[device][selection][qualification][service][runtime-pack]") {
    RouteQualificationStateGuard state_guard;
    const auto root = std::filesystem::temp_directory_path() /
        "cyxwiz-runtime-pack-qualification-removal";
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);

    cyxwiz::RouteQualificationSnapshot accepted;
    accepted.matrix_id = cyxwiz::kRouteQualificationMatrixId;
    accepted.runtime_set_id = "set-v1";
    accepted.base_pack_id = "base-v1";
    accepted.compute_contract_id = cyxwiz::kCyxWizComputeContractId;
    accepted.operation_manifest_id =
        cyxwiz::kRouteQualificationOperationManifestId;
    for (const auto& identity : {
             std::pair{cyxwiz::DeviceType::CPU, "base-v1"},
             std::pair{cyxwiz::DeviceType::CUDA, "cuda-v1"}}) {
        cyxwiz::RouteQualificationRecord record;
        record.type = identity.first;
        record.pack_id = identity.second;
        record.operation_count = cyxwiz::kRouteQualificationOperationCount;
        record.pass_count = record.operation_count;
        record.certified = true;
        accepted.routes.push_back(std::move(record));
    }
    cyxwiz::InstallRouteQualificationSnapshot(accepted);

    cyxwiz::RuntimeQualificationIdentity cpu_only;
    cpu_only.runtime_set_id = "set-v1";
    cpu_only.generation = 9;
    cpu_only.base_pack_id = "base-v1";
    cyxwiz::RouteQualificationOptions options;
    options.cache_path = root / "route-qualification.json";
    options.matrix_id = cyxwiz::kRouteQualificationMatrixId;
    cyxwiz::RouteQualificationService service;
    const auto result = service.ReconcileRuntimeEvidence(cpu_only, options);

    INFO(result.message);
    REQUIRE(result.published);
    REQUIRE(result.snapshot.has_value());
    REQUIRE(result.snapshot->routes.size() == 1);
    CHECK(result.snapshot->routes.front().type == cyxwiz::DeviceType::CPU);
    CHECK(result.snapshot->routes.front().pack_id == "base-v1");
    std::filesystem::remove_all(root, cleanup_error);
}

TEST_CASE("Active packaged runtime identity rejects evidence from another pack",
          "[device][selection][qualification][runtime-pack][evidence]") {
    EnvironmentGuard environment({
        "CYXWIZ_ACTIVE_RUNTIME_ROOT", "CYXWIZ_RUNTIME_SET_ID",
        "CYXWIZ_RUNTIME_GENERATION", "CYXWIZ_BASE_PACK_ID",
        "CYXWIZ_RUNTIME_PACK_CUDA", "CYXWIZ_RUNTIME_PACK_OPENCL",
        "CYXWIZ_RUNTIME_PACK_ONEAPI"});
    EnvironmentGuard::Set("CYXWIZ_ACTIVE_RUNTIME_ROOT", "runtime-root");
    EnvironmentGuard::Set("CYXWIZ_RUNTIME_SET_ID", "set-v1");
    EnvironmentGuard::Set("CYXWIZ_RUNTIME_GENERATION", "2");
    EnvironmentGuard::Set("CYXWIZ_BASE_PACK_ID", "base-v1");
    EnvironmentGuard::Set("CYXWIZ_RUNTIME_PACK_CUDA", nullptr);
    EnvironmentGuard::Set("CYXWIZ_RUNTIME_PACK_OPENCL", "opencl-v2");
    EnvironmentGuard::Set("CYXWIZ_RUNTIME_PACK_ONEAPI", nullptr);

    cyxwiz::RouteQualificationRecord record;
    record.type = cyxwiz::DeviceType::OPENCL;
    record.pack_id = "opencl-v1";
    record.operation_count = cyxwiz::kRouteQualificationOperationCount;
    record.pass_count = record.operation_count;
    record.certified = true;
    cyxwiz::RouteQualificationSnapshot snapshot;
    snapshot.matrix_id = cyxwiz::kRouteQualificationMatrixId;
    snapshot.runtime_set_id = "set-v1";
    snapshot.runtime_generation = 1;
    snapshot.base_pack_id = "base-v1";
    snapshot.compute_contract_id = cyxwiz::kCyxWizComputeContractId;
    snapshot.operation_manifest_id =
        cyxwiz::kRouteQualificationOperationManifestId;
    snapshot.routes = {record};
    cyxwiz::DeviceInfo route;
    route.type = cyxwiz::DeviceType::OPENCL;

    const auto stale = cyxwiz::EvaluateRouteQualification(route, snapshot);
    CHECK_FALSE(stale.qualified);
    CHECK(stale.evidence_available);
    CHECK(stale.failure.category ==
          cyxwiz::RouteFailureCategory::EvidenceStale);

    EnvironmentGuard::Set("CYXWIZ_RUNTIME_PACK_OPENCL", "opencl-v1");
    const auto current = cyxwiz::EvaluateRouteQualification(route, snapshot);
    CHECK(current.qualified);
}

TEST_CASE("Route recommendations require stable identity and certification",
          "[device][selection][recommendation]") {
    auto cpu = cyxwiz::DeviceInfo{};
    cpu.type = cyxwiz::DeviceType::CPU;
    cpu.device_id = 0;
    cpu.kind = cyxwiz::DeviceKind::CPU;

    auto cuda = cyxwiz::DeviceInfo{};
    cuda.type = cyxwiz::DeviceType::CUDA;
    cuda.device_id = 0;
    cuda.kind = cyxwiz::DeviceKind::GPU;
    cuda.identity_confidence =
        cyxwiz::DeviceIdentityConfidence::StableHardware;
    cuda.physical_fingerprint = "uuid:nvidia";
    cuda.physical_fingerprint_known = true;
    cuda.hardware_vendor_id = 0x10de;
    cuda.hardware_vendor_id_known = true;

    auto opencl = cuda;
    opencl.type = cyxwiz::DeviceType::OPENCL;

    const auto qualify = [](const cyxwiz::DeviceInfo& route) {
        cyxwiz::RouteQualificationRecord record;
        record.type = route.type;
        record.device_id = route.device_id;
        record.physical_fingerprint = route.physical_fingerprint_known
            ? route.physical_fingerprint
            : std::string{};
        record.operation_count = cyxwiz::kRouteQualificationOperationCount;
        record.pass_count = cyxwiz::kRouteQualificationOperationCount;
        record.certified = true;
        return record;
    };
    cyxwiz::RouteQualificationSnapshot snapshot;
    snapshot.matrix_id = "recommendation-test";
    snapshot.routes = {qualify(cpu), qualify(cuda), qualify(opencl)};

    const auto result = cyxwiz::RecommendExecutionRoutes(
        cuda, {cpu, cuda, opencl}, snapshot);
    REQUIRE(result.recommendations.size() == 2);
    CHECK(result.recommendations[0].route.type ==
          cyxwiz::DeviceType::OPENCL);
    CHECK(result.recommendations[0].remediation ==
          cyxwiz::RouteRecommendationClass::SamePhysicalDevice);
    CHECK(result.recommendations[1].route.type ==
          cyxwiz::DeviceType::CPU);
    CHECK(result.recommendations[1].remediation ==
          cyxwiz::RouteRecommendationClass::ArrayFireCpuRecovery);

    auto uncertain_opencl = opencl;
    uncertain_opencl.identity_confidence =
        cyxwiz::DeviceIdentityConfidence::ProviderReported;
    const auto uncertain = cyxwiz::RecommendExecutionRoutes(
        cuda, {cpu, uncertain_opencl}, snapshot);
    REQUIRE(uncertain.recommendations.size() == 1);
    CHECK(uncertain.recommendations[0].route.type ==
          cyxwiz::DeviceType::CPU);
    REQUIRE(uncertain.rejections.size() == 1);
    CHECK(uncertain.rejections[0].reason.find("Stable physical identity") !=
          std::string::npos);

    auto stale_snapshot = snapshot;
    stale_snapshot.routes[2].physical_fingerprint = "uuid:replacement";
    const auto stale = cyxwiz::RecommendExecutionRoutes(
        cuda, {cpu, opencl}, stale_snapshot);
    REQUIRE(stale.recommendations.size() == 1);
    CHECK(stale.recommendations[0].route.type == cyxwiz::DeviceType::CPU);
    REQUIRE(stale.rejections.size() == 1);
    CHECK(stale.rejections[0].reason.find("identity differs") !=
          std::string::npos);
}

TEST_CASE("Route recommendation ordering covers Intel GPU and CPU paths",
          "[device][selection][recommendation]") {
    const auto certified = [](const cyxwiz::DeviceInfo& route) {
        cyxwiz::RouteQualificationRecord record;
        record.type = route.type;
        record.device_id = route.device_id;
        record.physical_fingerprint = route.physical_fingerprint_known
            ? route.physical_fingerprint
            : std::string{};
        record.operation_count = cyxwiz::kRouteQualificationOperationCount;
        record.pass_count = cyxwiz::kRouteQualificationOperationCount;
        record.certified = true;
        return record;
    };

    cyxwiz::DeviceInfo af_cpu{};
    af_cpu.type = cyxwiz::DeviceType::CPU;
    af_cpu.kind = cyxwiz::DeviceKind::CPU;

    cyxwiz::DeviceInfo oneapi_gpu{};
    oneapi_gpu.type = cyxwiz::DeviceType::ONEAPI;
    oneapi_gpu.kind = cyxwiz::DeviceKind::GPU;
    oneapi_gpu.identity_confidence =
        cyxwiz::DeviceIdentityConfidence::StableHardware;
    oneapi_gpu.physical_fingerprint = "pci:intel-gpu";
    oneapi_gpu.physical_fingerprint_known = true;
    auto opencl_gpu = oneapi_gpu;
    opencl_gpu.type = cyxwiz::DeviceType::OPENCL;

    cyxwiz::RouteQualificationSnapshot gpu_snapshot;
    gpu_snapshot.matrix_id = "intel-gpu";
    gpu_snapshot.routes = {certified(af_cpu), certified(opencl_gpu)};
    const auto gpu = cyxwiz::RecommendExecutionRoutes(
        oneapi_gpu, {af_cpu, opencl_gpu}, gpu_snapshot);
    REQUIRE(gpu.recommendations.size() == 2);
    CHECK(gpu.recommendations[0].route.type ==
          cyxwiz::DeviceType::OPENCL);

    cyxwiz::DeviceInfo opencl_cpu{};
    opencl_cpu.type = cyxwiz::DeviceType::OPENCL;
    opencl_cpu.device_id = 2;
    opencl_cpu.kind = cyxwiz::DeviceKind::CPU;
    opencl_cpu.identity_confidence =
        cyxwiz::DeviceIdentityConfidence::StableHardware;
    opencl_cpu.physical_fingerprint = "uuid:intel-cpu";
    opencl_cpu.physical_fingerprint_known = true;
    auto oneapi_cpu = opencl_cpu;
    oneapi_cpu.type = cyxwiz::DeviceType::ONEAPI;

    cyxwiz::RouteQualificationSnapshot cpu_snapshot;
    cpu_snapshot.matrix_id = "cpu-path";
    cpu_snapshot.routes = {certified(af_cpu), certified(oneapi_cpu)};
    const auto cpu = cyxwiz::RecommendExecutionRoutes(
        opencl_cpu, {oneapi_cpu, af_cpu}, cpu_snapshot);
    REQUIRE(cpu.recommendations.size() == 2);
    CHECK(cpu.recommendations[0].route.type == cyxwiz::DeviceType::ONEAPI);
    CHECK(cpu.recommendations[1].route.type == cyxwiz::DeviceType::CPU);

    oneapi_cpu.kind = cyxwiz::DeviceKind::Unknown;
    oneapi_cpu.identity_confidence =
        cyxwiz::DeviceIdentityConfidence::Unknown;
    oneapi_cpu.physical_fingerprint_known = false;
    const auto unknown = cyxwiz::RecommendExecutionRoutes(
        oneapi_cpu, {opencl_cpu, af_cpu}, cpu_snapshot);
    REQUIRE(unknown.recommendations.size() == 1);
    CHECK(unknown.recommendations[0].route.type ==
          cyxwiz::DeviceType::CPU);
}

TEST_CASE("Device selection transaction preserves state at every failure stage",
          "[device][selection][transaction]") {
    using Stage = cyxwiz::DeviceSelectionTransactionStage;
    using Status = cyxwiz::DeviceSelectionTransactionStatus;

    cyxwiz::PendingExecutionDeviceSelection candidate{
        cyxwiz::DeviceType::CUDA, 0, "uuid:selected"};
    cyxwiz::DeviceInfo route{};
    route.type = candidate.type;
    route.device_id = candidate.device_id;
    route.physical_fingerprint = candidate.physical_fingerprint;
    route.physical_fingerprint_known = true;

    const auto run_failure = [&](Stage injected_stage,
                                 Status expected_status) {
        int inventory_calls = 0;
        int commit_calls = 0;
        cyxwiz::DeviceSelectionTransactionHooks hooks;
        hooks.inventory = [&] {
            ++inventory_calls;
            if (injected_stage == Stage::Inventory) {
                return std::vector<cyxwiz::DeviceInfo>{};
            }
            if (injected_stage == Stage::Revalidation &&
                inventory_calls == 2) {
                return std::vector<cyxwiz::DeviceInfo>{};
            }
            return std::vector<cyxwiz::DeviceInfo>{route};
        };
        hooks.qualify = [&](const auto&, std::string& message) {
            if (injected_stage == Stage::Qualification) {
                message = "not qualified";
                return false;
            }
            return true;
        };
        hooks.activate = [&](auto type, int device_id) {
            cyxwiz::DeviceActivationResult result;
            result.requested_type = type;
            result.requested_device_id = device_id;
            result.effective_type = type;
            result.effective_device_id = device_id;
            result.success = injected_stage != Stage::Activation;
            result.execution_validated = result.success;
            if (injected_stage == Stage::Complete) {
                result.effective_device_id = device_id + 1;
            }
            result.message = result.success ? "" : "activation failed";
            return result;
        };
        hooks.restore = [&] {
            cyxwiz::DeviceActivationResult result;
            result.success = injected_stage != Stage::Restore;
            result.message = result.success ? "" : "restore failed";
            return result;
        };
        hooks.commit = [&](const auto&) {
            ++commit_calls;
            if (injected_stage == Stage::Commit) {
                throw std::runtime_error("commit failed");
            }
        };

        const auto result =
            cyxwiz::RunDeviceSelectionTransaction(candidate, hooks);
        CHECK_FALSE(result.committed);
        CHECK(result.status == expected_status);
        CHECK(commit_calls == (injected_stage == Stage::Commit ? 1 : 0));
    };

    run_failure(Stage::Inventory, Status::RouteNotFound);
    run_failure(Stage::Qualification, Status::NotQualified);
    run_failure(Stage::Activation, Status::ActivationFailed);
    run_failure(Stage::Complete, Status::EffectiveRouteMismatch);
    run_failure(Stage::Restore, Status::RestoreFailed);
    run_failure(Stage::Revalidation, Status::RevalidationFailed);
    run_failure(Stage::Commit, Status::CommitFailed);

    auto changed = route;
    changed.physical_fingerprint = "uuid:changed";
    cyxwiz::DeviceSelectionTransactionHooks identity_hooks;
    identity_hooks.inventory = [&] {
        return std::vector<cyxwiz::DeviceInfo>{changed};
    };
    identity_hooks.qualify = [](const auto&, std::string&) { return true; };
    identity_hooks.activate = [](auto, int) {
        return cyxwiz::DeviceActivationResult{};
    };
    identity_hooks.restore = [] {
        return cyxwiz::DeviceActivationResult{};
    };
    identity_hooks.commit = [](const auto&) {};
    const auto identity =
        cyxwiz::RunDeviceSelectionTransaction(candidate, identity_hooks);
    CHECK_FALSE(identity.committed);
    CHECK(identity.status == Status::IdentityMismatch);
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
TEST_CASE("Production selection transaction restores process route and commits atomically",
          "[device][selection][transaction][arrayfire]") {
    ArrayFireStateGuard restore;
    cyxwiz::ClearPendingExecutionDeviceSelection();
    cyxwiz::ClearSavedExecutionDeviceSelection();

    const auto original_backend = af::getActiveBackend();
    const int original_device = af::getDevice();
    const auto devices = cyxwiz::Device::GetAvailableDevices();
    const auto cpu = std::find_if(
        devices.begin(), devices.end(), [](const auto& device) {
            return device.type == cyxwiz::DeviceType::CPU;
        });
    REQUIRE(cpu != devices.end());

    const cyxwiz::PendingExecutionDeviceSelection candidate{
        cpu->type,
        cpu->device_id,
        cpu->physical_fingerprint_known
            ? cpu->physical_fingerprint
            : std::string{}};
    const auto missing_evidence =
        cyxwiz::CommitExecutionDeviceSelection(candidate);
    CHECK_FALSE(missing_evidence.committed);
    CHECK(missing_evidence.status ==
          cyxwiz::DeviceSelectionTransactionStatus::NotQualified);
    CHECK_FALSE(cyxwiz::GetPendingExecutionDeviceSelection().has_value());
    CHECK_FALSE(cyxwiz::GetSavedExecutionDeviceSelection().has_value());
    CHECK(af::getActiveBackend() == original_backend);
    CHECK(af::getDevice() == original_device);

    cyxwiz::RouteQualificationRecord qualification;
    qualification.type = cpu->type;
    qualification.device_id = cpu->device_id;
    qualification.physical_fingerprint = candidate.physical_fingerprint;
    qualification.provider = cpu->provider_known ? cpu->provider : std::string{};
    qualification.driver_version =
        cpu->driver_version_known ? cpu->driver_version : std::string{};
    qualification.operation_count =
        cyxwiz::kRouteQualificationOperationCount;
    qualification.pass_count =
        cyxwiz::kRouteQualificationOperationCount;
    qualification.certified = true;
    cyxwiz::RouteQualificationSnapshot snapshot;
    snapshot.matrix_id = "production-transaction-test";
    snapshot.routes = {qualification};
    cyxwiz::InstallRouteQualificationSnapshot(std::move(snapshot));
    const auto committed =
        cyxwiz::CommitExecutionDeviceSelection(candidate);
    INFO(committed.message);
    REQUIRE(committed.committed);
    CHECK(committed.status ==
          cyxwiz::DeviceSelectionTransactionStatus::Committed);
    REQUIRE(cyxwiz::GetPendingExecutionDeviceSelection().has_value());
    REQUIRE(cyxwiz::GetSavedExecutionDeviceSelection().has_value());
    CHECK(cyxwiz::GetPendingExecutionDeviceSelection()->type == candidate.type);
    CHECK(cyxwiz::GetSavedExecutionDeviceSelection()->device_id ==
          candidate.device_id);
    CHECK(af::getActiveBackend() == original_backend);
    CHECK(af::getDevice() == original_device);

    const auto pending_before =
        cyxwiz::GetPendingExecutionDeviceSelection();
    const auto saved_before = cyxwiz::GetSavedExecutionDeviceSelection();
    const auto rejected = cyxwiz::CommitExecutionDeviceSelection(
        {cyxwiz::DeviceType::CUDA, 999999, "uuid:missing"});
    CHECK_FALSE(rejected.committed);
    CHECK(rejected.status ==
          cyxwiz::DeviceSelectionTransactionStatus::RouteNotFound);
    CHECK(cyxwiz::GetPendingExecutionDeviceSelection()->type ==
          pending_before->type);
    CHECK(cyxwiz::GetSavedExecutionDeviceSelection()->type ==
          saved_before->type);

    cyxwiz::ClearPendingExecutionDeviceSelection();
    cyxwiz::ClearSavedExecutionDeviceSelection();
    cyxwiz::ClearRouteQualificationSnapshot();
}
#endif

} // namespace

TEST_CASE("ArrayFire probe keeps metadata-limited devices and siblings",
          "[device][probe]") {
    FakeArrayFireProbeAdapter adapter;
    adapter.backend_status[cyxwiz::DeviceType::ONEAPI] = ProbeSuccess();
    adapter.device_counts[cyxwiz::DeviceType::ONEAPI] =
        {ProbeSuccess(), 2};
    adapter.device_status[{cyxwiz::DeviceType::ONEAPI, 0}] = ProbeSuccess();
    adapter.device_status[{cyxwiz::DeviceType::ONEAPI, 1}] = ProbeSuccess();

    cyxwiz::DeviceInfo limited{};
    limited.name = "oneAPI device 0";
    limited.name_is_fallback = true;
    limited.metadata_status = cyxwiz::DeviceMetadataStatus::Unsupported;
    limited.metadata_error_code = 301;
    limited.metadata_message = "metadata unsupported";
    adapter.metadata[{cyxwiz::DeviceType::ONEAPI, 0}] = limited;
    adapter.metadata[{cyxwiz::DeviceType::ONEAPI, 1}] =
        AvailableMetadata("oneAPI provider device");

    const auto result = cyxwiz::detail::ProbeAvailableArrayFireDevices(
        adapter, {cyxwiz::DeviceType::ONEAPI});

    REQUIRE(result.devices.size() == 2);
    CHECK(result.devices[0].backend_available);
    CHECK(result.devices[0].device_selectable);
    CHECK(result.devices[0].metadata_status ==
          cyxwiz::DeviceMetadataStatus::Unsupported);
    CHECK(result.devices[0].kind == cyxwiz::DeviceKind::Unknown);
    CHECK(result.devices[0].identity_confidence ==
          cyxwiz::DeviceIdentityConfidence::Unknown);
    CHECK(result.devices[1].metadata_status ==
          cyxwiz::DeviceMetadataStatus::Available);
    REQUIRE(result.failures.size() == 1);
    CHECK(result.failures[0].stage ==
          cyxwiz::detail::DeviceProbeStage::Metadata);
    CHECK(result.failures[0].device_id == 0);
    CHECK(result.failures[0].error_code == 301);
}

TEST_CASE("ArrayFire probe isolates backend enumeration and device failures",
          "[device][probe]") {
    FakeArrayFireProbeAdapter adapter;
    adapter.backend_status[cyxwiz::DeviceType::CUDA] =
        {false, 101, "backend unavailable"};
    adapter.backend_status[cyxwiz::DeviceType::OPENCL] = ProbeSuccess();
    adapter.device_counts[cyxwiz::DeviceType::OPENCL] =
        {{false, 202, "enumeration failed"}, 0};
    adapter.backend_status[cyxwiz::DeviceType::CPU] = ProbeSuccess();
    adapter.device_counts[cyxwiz::DeviceType::CPU] = {ProbeSuccess(), 2};
    adapter.device_status[{cyxwiz::DeviceType::CPU, 0}] =
        {false, 303, "device selection failed"};
    adapter.device_status[{cyxwiz::DeviceType::CPU, 1}] = ProbeSuccess();
    adapter.metadata[{cyxwiz::DeviceType::CPU, 1}] =
        AvailableMetadata("CPU device 1");

    const auto result = cyxwiz::detail::ProbeAvailableArrayFireDevices(
        adapter,
        {cyxwiz::DeviceType::CUDA,
         cyxwiz::DeviceType::OPENCL,
         cyxwiz::DeviceType::CPU});

    REQUIRE(result.devices.size() == 1);
    CHECK(result.devices[0].type == cyxwiz::DeviceType::CPU);
    CHECK(result.devices[0].device_id == 1);
    REQUIRE(result.failures.size() == 3);
    CHECK(result.failures[0].stage ==
          cyxwiz::detail::DeviceProbeStage::BackendSelection);
    CHECK(result.failures[1].stage ==
          cyxwiz::detail::DeviceProbeStage::Enumeration);
    CHECK(result.failures[2].stage ==
          cyxwiz::detail::DeviceProbeStage::DeviceSelection);
}

TEST_CASE("ArrayFire probe distinguishes a loaded backend with no devices",
          "[device][probe]") {
    FakeArrayFireProbeAdapter adapter;
    adapter.backend_status[cyxwiz::DeviceType::OPENCL] = ProbeSuccess();
    adapter.device_counts[cyxwiz::DeviceType::OPENCL] = {ProbeSuccess(), 0};

    const auto result = cyxwiz::detail::ProbeAvailableArrayFireDevices(
        adapter, {cyxwiz::DeviceType::OPENCL});

    CHECK(result.devices.empty());
    REQUIRE(result.failures.size() == 1);
    CHECK(result.failures[0].type == cyxwiz::DeviceType::OPENCL);
    CHECK(result.failures[0].stage ==
          cyxwiz::detail::DeviceProbeStage::Enumeration);
    CHECK(result.failures[0].error_code == 0);
    CHECK(result.failures[0].message ==
          "Backend loaded but exposed no compatible devices");
}

TEST_CASE("Device enumeration", "[device]") {
    auto devices = cyxwiz::Device::GetAvailableDevices();
    REQUIRE(devices.size() >= 1);

    for (const auto& device : devices) {
        INFO("backend type=" << static_cast<int>(device.type)
             << " device=" << device.device_id);
        CHECK(device.backend_available);
        CHECK(device.device_selectable);
        CHECK(static_cast<int>(device.identity_confidence) >=
              static_cast<int>(
                  cyxwiz::DeviceIdentityConfidence::BackendLocal));
        if (device.type == cyxwiz::DeviceType::CPU) {
            CHECK(device.kind == cyxwiz::DeviceKind::CPU);
        } else if (device.type == cyxwiz::DeviceType::CUDA) {
            CHECK(device.kind == cyxwiz::DeviceKind::GPU);
        } else if (device.type == cyxwiz::DeviceType::OPENCL) {
            CHECK(device.kind != cyxwiz::DeviceKind::Unknown);
            CHECK(device.provider_known);
            CHECK(device.driver_version_known);
            CHECK(device.hardware_vendor_id_known);
        }
        CHECK_FALSE(device.name.empty());
        CHECK(device.name_known != device.name_is_fallback);
        if (!device.memory_total_known) {
            CHECK(device.memory_total == 0);
        }
        if (!device.memory_available_known) {
            CHECK(device.memory_available == 0);
        }
    }
}

TEST_CASE("Provider UUID correlates CUDA and OpenCL routes when reported",
          "[device][identity][arrayfire]") {
    const auto devices = cyxwiz::Device::GetAvailableDevices();
    const auto cuda = std::find_if(
        devices.begin(), devices.end(), [](const auto& device) {
            return device.type == cyxwiz::DeviceType::CUDA &&
                   device.hardware_uuid_known;
        });
    const auto opencl = std::find_if(
        devices.begin(), devices.end(), [](const auto& device) {
            return device.type == cyxwiz::DeviceType::OPENCL &&
                   device.hardware_vendor_id_known &&
                   device.hardware_vendor_id == 0x10de &&
                   device.hardware_uuid_known;
        });
    if (cuda == devices.end() || opencl == devices.end()) {
        SKIP("CUDA/OpenCL provider UUID pair is not exposed on this machine");
    }

    CHECK(cuda->physical_fingerprint_known);
    CHECK(opencl->physical_fingerprint_known);
    CHECK(cuda->identity_confidence ==
          cyxwiz::DeviceIdentityConfidence::StableHardware);
    CHECK(opencl->identity_confidence ==
          cyxwiz::DeviceIdentityConfidence::StableHardware);
    CHECK(cuda->physical_fingerprint == opencl->physical_fingerprint);
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
TEST_CASE("Device enumeration restores the active ArrayFire state",
          "[device][arrayfire]") {
    const auto original_backend = af::getActiveBackend();
    const int original_device = af::getDevice();

    REQUIRE_NOTHROW(cyxwiz::Device::GetAvailableDevices());

    CHECK(af::getActiveBackend() == original_backend);
    CHECK(af::getDevice() == original_device);
}

TEST_CASE("Direct device metadata query targets the requested backend and restores state",
          "[device][arrayfire]") {
    ArrayFireStateGuard restore;
    const auto cpu_activation =
        cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(false);
    REQUIRE(cpu_activation.success);

    const auto devices = cyxwiz::Device::GetAvailableDevices();
    const auto other = std::find_if(
        devices.begin(), devices.end(), [](const auto& device) {
            return device.type != cyxwiz::DeviceType::CPU;
        });
    if (other == devices.end()) {
        SKIP("No accelerator backend is available for cross-backend metadata query");
    }

    const auto info =
        cyxwiz::Device(other->type, other->device_id).GetInfo();
    CHECK(info.type == other->type);
    CHECK(info.device_id == other->device_id);
    CHECK(info.backend_available);
    CHECK(info.device_selectable);
    CHECK(af::getActiveBackend() == AF_BACKEND_CPU);
    CHECK(af::getDevice() == 0);
}

TEST_CASE("oneAPI discovery survives unsupported metadata",
          "[device][arrayfire][oneapi]") {
    const auto original_backend = af::getActiveBackend();
    const int original_device = af::getDevice();
    int oneapi_count = 0;
    try {
        af::setBackend(AF_BACKEND_ONEAPI);
        oneapi_count = af::getDeviceCount();
    } catch (const af::exception&) {
        oneapi_count = 0;
    }
    af::setBackend(original_backend);
    af::setDevice(original_device);

    if (oneapi_count == 0) {
        SKIP("ArrayFire oneAPI backend is not installed on this machine");
    }

    const auto devices = cyxwiz::Device::GetAvailableDevices();
    for (int device_id = 0; device_id < oneapi_count; ++device_id) {
        const auto match = std::find_if(
            devices.begin(), devices.end(), [device_id](const auto& device) {
                return device.type == cyxwiz::DeviceType::ONEAPI &&
                       device.device_id == device_id;
            });
        REQUIRE(match != devices.end());
        CHECK(match->backend_available);
        CHECK(match->device_selectable);

        if (match->metadata_status ==
            cyxwiz::DeviceMetadataStatus::Unsupported) {
            CHECK(match->metadata_error_code == AF_ERR_NOT_SUPPORTED);
            CHECK(match->name_is_fallback);
            CHECK_FALSE(match->name_known);
            CHECK_FALSE(match->memory_total_known);
        } else {
            CHECK(match->metadata_status ==
                  cyxwiz::DeviceMetadataStatus::Available);
            CHECK(match->name_known);
            CHECK_FALSE(match->name_is_fallback);
        }
    }
}

TEST_CASE("Exact activation validates CPU and rejects an invalid device",
          "[device][arrayfire][activation]") {
    ArrayFireStateGuard restore;

    const auto cpu =
        cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true);
    REQUIRE(cpu.success);
    CHECK(cpu.execution_validated);
    CHECK(cpu.stage == cyxwiz::DeviceActivationStage::Complete);
    CHECK(cpu.effective_type == cyxwiz::DeviceType::CPU);
    CHECK(cpu.effective_device_id == 0);

    const auto invalid =
        cyxwiz::Device(cyxwiz::DeviceType::CPU, 999999)
            .ActivateExact(true);
    CHECK_FALSE(invalid.success);
    CHECK(invalid.stage == cyxwiz::DeviceActivationStage::DeviceSelection);
    CHECK_FALSE(invalid.execution_validated);
    CHECK(af::getActiveBackend() == AF_BACKEND_CPU);
    CHECK(af::getDevice() == cpu.effective_device_id);
}

TEST_CASE("Exact activation switches CPU to oneAPI and back when installed",
          "[device][arrayfire][activation][oneapi]") {
    ArrayFireStateGuard restore;
    int oneapi_count = 0;
    try {
        af::setBackend(AF_BACKEND_ONEAPI);
        oneapi_count = af::getDeviceCount();
    } catch (af::exception&) {
        SKIP("ArrayFire oneAPI backend is not installed on this machine");
    }
    if (oneapi_count == 0) {
        SKIP("ArrayFire oneAPI backend has no device on this machine");
    }

    const auto cpu_before =
        cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true);
    REQUIRE(cpu_before.success);

    const auto oneapi =
        cyxwiz::Device(cyxwiz::DeviceType::ONEAPI, 0).ActivateExact(true);
    INFO("stage=" << cyxwiz::DeviceActivationStageName(oneapi.stage)
                  << " error=" << oneapi.error_code
                  << " message=" << oneapi.message);
    REQUIRE(oneapi.success);
    CHECK(oneapi.execution_validated);
    CHECK(oneapi.effective_type == cyxwiz::DeviceType::ONEAPI);
    CHECK(oneapi.effective_device_id == 0);

    const auto cpu_after =
        cyxwiz::Device(cyxwiz::DeviceType::CPU, 0).ActivateExact(true);
    REQUIRE(cpu_after.success);
    CHECK(cpu_after.execution_validated);
    CHECK(cpu_after.effective_type == cyxwiz::DeviceType::CPU);
}

TEST_CASE("Run preflight preserves requested and effective selection truth",
           "[device][arrayfire][activation][context]") {
    ArrayFireStateGuard restore;
    RouteQualificationStateGuard qualification_restore;
    InstallCertifiedInventorySnapshot("preflight-selection-truth-test");
    cyxwiz::ClearPendingExecutionDeviceSelection();

    cyxwiz::SetPendingExecutionDeviceSelection(
        cyxwiz::DeviceType::CPU, 999999);
    const auto context = cyxwiz::PrepareExecutionDeviceForRun(
        cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);
    CHECK(context.valid);
    CHECK(context.requested_backend == "arrayfire_cpu");
    CHECK(context.requested_device_id == 999999);
    CHECK(context.effective_backend == "arrayfire_cpu");
    CHECK(context.effective_device_id == 0);
    CHECK(context.activation_succeeded);
    CHECK(context.execution_validated);
    CHECK(context.selection_fallback_applied);
    CHECK(context.preflight_stage == "complete");
    CHECK_FALSE(cyxwiz::GetPendingExecutionDeviceSelection().has_value());
}

TEST_CASE("Run preflight never activates a stale saved hardware ordinal",
           "[device][arrayfire][activation][context][identity]") {
    ArrayFireStateGuard restore;
    RouteQualificationStateGuard qualification_restore;
    InstallCertifiedInventorySnapshot("preflight-stale-identity-test");
    cyxwiz::ClearPendingExecutionDeviceSelection();
    cyxwiz::ClearSavedExecutionDeviceSelection();
    cyxwiz::CommitExecutionDeviceSelectionState(
        {cyxwiz::DeviceType::CUDA, 0, "uuid:not-installed"});

    const auto compatible = cyxwiz::PrepareExecutionDeviceForRun(
        cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);
    CHECK(compatible.valid);
    CHECK(compatible.requested_backend == "arrayfire_cuda");
    CHECK(compatible.effective_backend == "arrayfire_cpu");
    CHECK(compatible.selection_fallback_applied);
    CHECK(compatible.error.find(
              "physical device identity could not be resolved") !=
          std::string::npos);
    CHECK_FALSE(cyxwiz::GetPendingExecutionDeviceSelection().has_value());
    REQUIRE(cyxwiz::GetSavedExecutionDeviceSelection().has_value());

    CHECK_THROWS_AS(
        cyxwiz::PrepareExecutionDeviceForRun(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback),
        std::runtime_error);
    CHECK(cyxwiz::GetSavedExecutionDeviceSelection().has_value());

    cyxwiz::ClearPendingExecutionDeviceSelection();
    cyxwiz::ClearSavedExecutionDeviceSelection();
}

TEST_CASE("Strict run preflight retains a failed pending selection",
          "[device][arrayfire][activation][context][strict]") {
    ArrayFireStateGuard restore;
    cyxwiz::ClearPendingExecutionDeviceSelection();
    cyxwiz::SetPendingExecutionDeviceSelection(
        cyxwiz::DeviceType::CPU, 999999);

    CHECK_THROWS_AS(
        cyxwiz::PrepareExecutionDeviceForRun(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback),
        std::runtime_error);
    CHECK(cyxwiz::GetPendingExecutionDeviceSelection().has_value());
    cyxwiz::ClearPendingExecutionDeviceSelection();
}

TEST_CASE("Failed exact oneAPI evidence follows compatibility and strict policy",
           "[device][arrayfire][activation][context][oneapi]") {
    ArrayFireStateGuard restore;
    RouteQualificationStateGuard qualification_restore;
    const auto inventory =
        InstallCertifiedInventorySnapshot("preflight-oneapi-policy-test");
    if (std::none_of(inventory.begin(), inventory.end(), [](const auto& route) {
            return route.type == cyxwiz::DeviceType::ONEAPI &&
                   route.device_id == 0;
        })) {
        SKIP("ArrayFire oneAPI route 0 is not installed on this machine");
    }
    auto snapshot = cyxwiz::GetRouteQualificationSnapshot();
    REQUIRE(snapshot.has_value());
    auto oneapi_record = std::find_if(
        snapshot->routes.begin(), snapshot->routes.end(), [](const auto& route) {
            return route.type == cyxwiz::DeviceType::ONEAPI &&
                   route.device_id == 0;
        });
    REQUIRE(oneapi_record != snapshot->routes.end());
    oneapi_record->pass_count = 0;
    oneapi_record->crash_count = oneapi_record->operation_count;
    oneapi_record->certified = false;
    oneapi_record->failure.stage = cyxwiz::RouteFailureStage::Operation;
    oneapi_record->failure.category =
        cyxwiz::RouteFailureCategory::ChildProcessCrash;
    oneapi_record->failure.operation = "sum";
    oneapi_record->failure.observed_fact =
        "Operation 'sum' terminated its isolated child process";
    cyxwiz::InstallRouteQualificationSnapshot(*snapshot);
    cyxwiz::ClearPendingExecutionDeviceSelection();
    cyxwiz::SetPendingExecutionDeviceSelection(
        cyxwiz::DeviceType::ONEAPI, 0);
    const auto compatible = cyxwiz::PrepareExecutionDeviceForRun(
        cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);
    CHECK(compatible.valid);
    CHECK(compatible.requested_backend == "arrayfire_oneapi");
    CHECK(compatible.effective_backend == "arrayfire_cpu");
    CHECK(compatible.execution_validated);
    CHECK(compatible.selection_fallback_applied);
    CHECK(compatible.error.find("Route failed isolated verification") !=
          std::string::npos);
    CHECK_FALSE(compatible.requested_qualification.qualified);
    CHECK(compatible.requested_qualification.matrix_id ==
          "preflight-oneapi-policy-test");
    CHECK(compatible.effective_qualification.qualified);
    CHECK_FALSE(cyxwiz::GetPendingExecutionDeviceSelection().has_value());

    cyxwiz::SetPendingExecutionDeviceSelection(
        cyxwiz::DeviceType::ONEAPI, 0);
    CHECK_THROWS_AS(
        cyxwiz::PrepareExecutionDeviceForRun(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback),
        std::runtime_error);
    CHECK(cyxwiz::GetPendingExecutionDeviceSelection().has_value());
    cyxwiz::ClearPendingExecutionDeviceSelection();
}
#endif
