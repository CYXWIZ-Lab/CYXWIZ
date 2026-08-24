#include "backend_pack_qualification_adapter.h"

#include <algorithm>
#include <system_error>
#include <utility>

namespace cyxwiz {
namespace {

std::optional<DeviceType> BackendType(const std::string& backend) {
    if (backend == "cpu") return DeviceType::CPU;
    if (backend == "cuda") return DeviceType::CUDA;
    if (backend == "opencl") return DeviceType::OPENCL;
    if (backend == "oneapi") return DeviceType::ONEAPI;
    return std::nullopt;
}

runtime::BackendPackQualificationDisposition FailureDisposition(
    RuntimeQualificationFailurePolicy policy) {
    return policy == RuntimeQualificationFailurePolicy::RequireRollback
        ? runtime::BackendPackQualificationDisposition::RollbackRequired
        : runtime::BackendPackQualificationDisposition::InstalledUnqualified;
}

runtime::BackendPackQualificationDisposition MapDisposition(
    RuntimeQualificationDisposition disposition) {
    switch (disposition) {
        case RuntimeQualificationDisposition::Qualified:
            return runtime::BackendPackQualificationDisposition::Qualified;
        case RuntimeQualificationDisposition::RollbackRequired:
            return runtime::BackendPackQualificationDisposition::RollbackRequired;
        case RuntimeQualificationDisposition::InstalledUnqualified:
            return runtime::BackendPackQualificationDisposition::InstalledUnqualified;
    }
    return runtime::BackendPackQualificationDisposition::InstalledUnqualified;
}

runtime::BackendPackQualificationDecision Failure(
    RuntimeQualificationFailurePolicy policy,
    std::string message) {
    return {FailureDisposition(policy), std::move(message)};
}

std::optional<RuntimeQualificationIdentity> BuildIdentity(
    const runtime::ActiveRuntimeState& candidate,
    std::string& error) {
    RuntimeQualificationIdentity identity;
    identity.runtime_set_id = candidate.runtime_set_id;
    identity.generation = candidate.generation;
    identity.base_pack_id = candidate.base_pack_id;
    for (const auto& pack : candidate.packs) {
        const auto type = BackendType(pack.backend);
        if (!type) {
            error = "Candidate runtime contains an unsupported backend";
            return std::nullopt;
        }
        identity.backend_packs.push_back({*type, pack.pack_id});
    }
    error = ValidateRuntimeQualificationIdentity(identity);
    if (!error.empty()) return std::nullopt;
    return identity;
}

runtime::BackendPackQualificationDecision QualifyCandidate(
    const std::shared_ptr<RouteQualificationService>& service,
    const BackendPackQualificationAdapterOptions& options,
    const BackendPackRouteDiscovery& discover,
    const runtime::VerifiedBackendPackManifest& manifest,
    const std::filesystem::path& installed_directory,
    const runtime::ActiveRuntimeState& candidate) {
    if (!service || options.runtime_root.empty() ||
        !options.runtime_root.is_absolute() ||
        options.probe_executable.empty() ||
        !options.probe_executable.is_absolute() ||
        options.cache_path.empty() || !options.cache_path.is_absolute() ||
        options.operation_timeout.count() <= 0 ||
        options.discovery_timeout.count() <= 0 ||
        options.output_limit_bytes == 0) {
        return Failure(
            options.failure_policy,
            "Staged route qualification is not configured");
    }
    const auto type = BackendType(manifest.backend);
    const bool base =
        manifest.kind == runtime::BackendPackManifestKind::Base;
    if (!type || manifest.pack_id.empty() ||
        manifest.compatibility.operation_matrix_id.empty() ||
        candidate.runtime_set_id != manifest.runtime_set_id ||
        candidate.base_pack_id !=
            (base ? manifest.pack_id : manifest.companion_base_id)) {
        return Failure(
            options.failure_policy,
            "Staged pack and prospective runtime identities do not match");
    }

    runtime::ActiveRuntime resolved;
    std::string error;
    if (!runtime::ResolveRuntimeState(
            options.runtime_root, candidate, resolved, error)) {
        return Failure(options.failure_policy, std::move(error));
    }
    std::error_code filesystem_error;
    const auto resolved_directory = [&]() -> std::filesystem::path {
        if (base) return resolved.base_directory;
        const auto resolved_pack = std::find_if(
            resolved.packs.begin(), resolved.packs.end(),
            [&](const runtime::ActivePack& pack) {
                return pack.backend == manifest.backend &&
                       pack.pack_id == manifest.pack_id;
            });
        return resolved_pack == resolved.packs.end()
            ? std::filesystem::path{} : resolved_pack->directory;
    }();
    if (resolved_directory.empty() ||
        !std::filesystem::equivalent(
            resolved_directory, installed_directory, filesystem_error) ||
        filesystem_error) {
        return Failure(
            options.failure_policy,
            "Installed pack directory does not match the prospective runtime");
    }
    const auto identity = BuildIdentity(candidate, error);
    if (!identity) return Failure(options.failure_policy, std::move(error));

    RouteProbeInvocation discovery;
    discovery.executable = options.probe_executable;
    discovery.type = *type;
    discovery.enumerate_backend = true;
    discovery.timeout = options.discovery_timeout;
    discovery.output_limit_bytes = options.output_limit_bytes;
    discovery.runtime_root = resolved.runtime_root;
    discovery.working_directory = resolved.base_directory;
    discovery.runtime_dll_directories = resolved.dll_directories;
    discovery.runtime_identity = *identity;
    const auto routes = discover(discovery, {});
    if (routes.status != RouteProbeStatus::Passed || routes.routes.empty()) {
        return Failure(
            options.failure_policy,
            routes.message.empty()
                ? "The staged backend exposed no qualifying routes"
                : routes.message);
    }

    RouteQualificationOptions qualification_options;
    qualification_options.probe_executable = options.probe_executable;
    qualification_options.cache_path = options.cache_path;
    qualification_options.matrix_id =
        manifest.compatibility.operation_matrix_id;
    qualification_options.pack_id = manifest.pack_id;
    qualification_options.runtime_version = manifest.arrayfire_version;
    qualification_options.operation_timeout = options.operation_timeout;
    qualification_options.output_limit_bytes = options.output_limit_bytes;
    qualification_options.probe_runtime_root = resolved.runtime_root;
    qualification_options.probe_working_directory = resolved.base_directory;
    qualification_options.probe_runtime_dll_directories =
        resolved.dll_directories;
    const auto result = service->VerifyStagedRuntimeRoutes(
        routes.routes, *identity, options.failure_policy,
        std::move(qualification_options));
    std::string message = result.qualification.message;
    if (result.disposition != RuntimeQualificationDisposition::Qualified &&
        !result.diagnostic.observed_fact.empty()) {
        message = result.diagnostic.observed_fact;
        if (!result.diagnostic.recommended_action.empty()) {
            message += ". " + result.diagnostic.recommended_action;
        }
    }
    if (message.empty()) {
        message = result.disposition == RuntimeQualificationDisposition::Qualified
            ? "Staged backend routes passed local qualification"
            : "Staged backend routes did not pass local qualification";
    }
    return {MapDisposition(result.disposition), std::move(message)};
}

}  // namespace

runtime::BackendPackQualificationHook CreateBackendPackQualificationHook(
    std::shared_ptr<RouteQualificationService> service,
    BackendPackQualificationAdapterOptions options,
    BackendPackRouteDiscovery discover) {
    if (!discover) discover = DiscoverIsolatedBackendRoutes;
    return [service = std::move(service), options = std::move(options),
            discover = std::move(discover)](
               const runtime::VerifiedBackendPackManifest& manifest,
               const std::filesystem::path& installed_directory,
               const runtime::ActiveRuntimeState& candidate) {
        return QualifyCandidate(
            service, options, discover, manifest, installed_directory,
            candidate);
    };
}

}  // namespace cyxwiz
