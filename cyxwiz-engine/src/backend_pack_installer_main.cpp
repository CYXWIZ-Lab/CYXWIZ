#include "core/backend_pack_qualification_adapter.h"
#include "core/compute_runtime_paths.h"
#include "installer/installer_progress_channel.h"

#include "backend_pack_lifecycle_service.h"
#include "backend_pack_metadata_cache.h"
#include "backend_pack_platform.h"
#include "backend_pack_state_service.h"
#include "product_release_version.h"
#include "product_registration.h"

#include <cyxwiz/version.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#else
#include <unistd.h>
#endif

namespace {

struct Options {
    std::filesystem::path runtime_root;
    std::filesystem::path metadata_root;
    std::string pack_id;
    std::string deactivate_backend;
    std::string progress_token;
    bool base = false;
    bool base_update = false;
    bool repair = false;
    bool offline = false;
    bool all_users = false;
};

template <typename Character>
bool IsIdentifier(const std::basic_string<Character>& value) {
    if (value.empty() || value.size() > 128 ||
        !((value.front() >= static_cast<Character>('a') &&
           value.front() <= static_cast<Character>('z')) ||
          (value.front() >= static_cast<Character>('0') &&
           value.front() <= static_cast<Character>('9')))) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](Character character) {
        return (character >= static_cast<Character>('a') &&
                character <= static_cast<Character>('z')) ||
               (character >= static_cast<Character>('0') &&
                character <= static_cast<Character>('9')) ||
               character == static_cast<Character>('.') ||
               character == static_cast<Character>('_') ||
               character == static_cast<Character>('-');
    });
}

std::filesystem::path ExecutableDirectory() {
#ifdef _WIN32
    std::vector<wchar_t> buffer(32768);
    const DWORD length = ::GetModuleFileNameW(
        nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0 || length >= buffer.size()) return {};
    return std::filesystem::path(
        std::wstring(buffer.data(), length)).parent_path();
#elif defined(__APPLE__)
    std::uint32_t size = 0;
    ::_NSGetExecutablePath(nullptr, &size);
    std::vector<char> buffer(size);
    if (size == 0 || ::_NSGetExecutablePath(buffer.data(), &size) != 0) {
        return {};
    }
    std::error_code error;
    const auto executable = std::filesystem::weakly_canonical(
        std::filesystem::path(buffer.data()), error);
    return error ? std::filesystem::path{} : executable.parent_path();
#else
    std::vector<char> buffer(4096);
    const auto length = ::readlink(
        "/proc/self/exe", buffer.data(), buffer.size());
    if (length <= 0 ||
        static_cast<std::size_t>(length) >= buffer.size()) {
        return {};
    }
    return std::filesystem::path(
        std::string(buffer.data(), static_cast<std::size_t>(length)))
        .parent_path();
#endif
}

std::string CurrentUtc() {
    const auto value = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
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

std::string ClientVersion() {
    return std::to_string(CYXWIZ_VERSION_MAJOR) + "." +
           std::to_string(CYXWIZ_VERSION_MINOR) + "." +
           std::to_string(CYXWIZ_VERSION_PATCH);
}

#ifdef _WIN32
bool ParseOptions(
    int argc,
    wchar_t** argv,
    Options& output,
    std::string& error) {
    bool saw_runtime_root = false;
    bool saw_metadata_root = false;
    bool saw_pack_id = false;
    bool saw_base_pack_id = false;
    bool saw_base_update_pack_id = false;
    bool saw_deactivate_backend = false;
    bool saw_all_users = false;
    bool saw_progress_token = false;
    for (int index = 1; index < argc; ++index) {
        const std::wstring_view argument(argv[index]);
        if (argument == L"--runtime-root" && !saw_runtime_root &&
            index + 1 < argc) {
            output.runtime_root = argv[++index];
            saw_runtime_root = true;
        } else if (argument == L"--metadata-root" &&
                   !saw_metadata_root && index + 1 < argc) {
            output.metadata_root = argv[++index];
            saw_metadata_root = true;
        } else if (argument == L"--pack-id" && !saw_pack_id &&
                   index + 1 < argc) {
            const std::wstring value(argv[++index]);
            if (!IsIdentifier(value)) {
                error = "--pack-id must be a safe ASCII identifier";
                return false;
            }
            output.pack_id.clear();
            output.pack_id.reserve(value.size());
            for (const wchar_t character : value) {
                output.pack_id.push_back(static_cast<char>(character));
            }
            saw_pack_id = true;
        } else if (argument == L"--base-pack-id" &&
                   !saw_base_pack_id && index + 1 < argc) {
            const std::wstring value(argv[++index]);
            if (!IsIdentifier(value)) {
                error = "--base-pack-id must be a safe ASCII identifier";
                return false;
            }
            output.pack_id.clear();
            output.pack_id.reserve(value.size());
            for (const wchar_t character : value) {
                output.pack_id.push_back(static_cast<char>(character));
            }
            output.base = true;
            saw_base_pack_id = true;
        } else if (argument == L"--update-base-pack-id" &&
                   !saw_base_update_pack_id && index + 1 < argc) {
            const std::wstring value(argv[++index]);
            if (!IsIdentifier(value)) {
                error = "--update-base-pack-id must be a safe ASCII identifier";
                return false;
            }
            output.pack_id.clear();
            output.pack_id.reserve(value.size());
            for (const wchar_t character : value) {
                output.pack_id.push_back(static_cast<char>(character));
            }
            output.base = true;
            output.base_update = true;
            saw_base_update_pack_id = true;
        } else if (argument == L"--deactivate-backend" &&
                   !saw_deactivate_backend && index + 1 < argc) {
            const std::wstring value(argv[++index]);
            if (!IsIdentifier(value)) {
                error =
                    "--deactivate-backend must be a safe ASCII identifier";
                return false;
            }
            output.deactivate_backend.clear();
            output.deactivate_backend.reserve(value.size());
            for (const wchar_t character : value) {
                output.deactivate_backend.push_back(
                    static_cast<char>(character));
            }
            saw_deactivate_backend = true;
        } else if (argument == L"--repair" && !output.repair) {
            output.repair = true;
        } else if (argument == L"--offline" && !output.offline) {
            output.offline = true;
        } else if (argument == L"--all-users" && !saw_all_users) {
            output.all_users = true;
            saw_all_users = true;
        } else if (argument == L"--progress-token" &&
                   !saw_progress_token && index + 1 < argc) {
            const std::wstring value(argv[++index]);
            const bool valid_token =
                value.size() == 32 &&
                std::all_of(
                    value.begin(), value.end(), [](wchar_t character) {
                        return (character >= L'0' && character <= L'9') ||
                               (character >= L'a' && character <= L'f');
                    });
            if (!valid_token) {
                error = "--progress-token must be a 32-character lowercase hexadecimal token";
                return false;
            }
            output.progress_token.clear();
            output.progress_token.reserve(value.size());
            for (const wchar_t character : value) {
                output.progress_token.push_back(
                    static_cast<char>(character));
            }
            saw_progress_token = true;
        } else {
            error = "Unsupported, duplicate, or incomplete installer argument";
            return false;
        }
    }
    if (!saw_runtime_root || !output.runtime_root.is_absolute() ||
        (saw_metadata_root && !output.metadata_root.is_absolute()) ||
        static_cast<int>(saw_pack_id) +
                static_cast<int>(saw_base_pack_id) +
                static_cast<int>(saw_base_update_pack_id) +
                static_cast<int>(saw_deactivate_backend) != 1 ||
        ((saw_base_pack_id || saw_base_update_pack_id) && output.repair) ||
        (saw_deactivate_backend && (output.repair || output.offline))) {
        error = "--runtime-root and exactly one pack operation are required";
        return false;
    }
    if (!saw_metadata_root) output.metadata_root = output.runtime_root;
    return true;
}
#else
bool ParseOptions(
    int argc,
    char** argv,
    Options& output,
    std::string& error) {
    bool saw_runtime_root = false;
    bool saw_metadata_root = false;
    bool saw_pack_id = false;
    bool saw_base_pack_id = false;
    bool saw_base_update_pack_id = false;
    bool saw_deactivate_backend = false;
    bool saw_all_users = false;
    bool saw_progress_token = false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--runtime-root" && !saw_runtime_root &&
            index + 1 < argc) {
            output.runtime_root = argv[++index];
            saw_runtime_root = true;
        } else if (argument == "--metadata-root" &&
                   !saw_metadata_root && index + 1 < argc) {
            output.metadata_root = argv[++index];
            saw_metadata_root = true;
        } else if (argument == "--pack-id" && !saw_pack_id &&
                   index + 1 < argc) {
            output.pack_id = argv[++index];
            if (!IsIdentifier(output.pack_id)) {
                error = "--pack-id must be a safe ASCII identifier";
                return false;
            }
            saw_pack_id = true;
        } else if (argument == "--base-pack-id" &&
                   !saw_base_pack_id && index + 1 < argc) {
            output.pack_id = argv[++index];
            if (!IsIdentifier(output.pack_id)) {
                error = "--base-pack-id must be a safe ASCII identifier";
                return false;
            }
            output.base = true;
            saw_base_pack_id = true;
        } else if (argument == "--update-base-pack-id" &&
                   !saw_base_update_pack_id && index + 1 < argc) {
            output.pack_id = argv[++index];
            if (!IsIdentifier(output.pack_id)) {
                error = "--update-base-pack-id must be a safe ASCII identifier";
                return false;
            }
            output.base = true;
            output.base_update = true;
            saw_base_update_pack_id = true;
        } else if (argument == "--deactivate-backend" &&
                   !saw_deactivate_backend && index + 1 < argc) {
            output.deactivate_backend = argv[++index];
            if (!IsIdentifier(output.deactivate_backend)) {
                error =
                    "--deactivate-backend must be a safe ASCII identifier";
                return false;
            }
            saw_deactivate_backend = true;
        } else if (argument == "--repair" && !output.repair) {
            output.repair = true;
        } else if (argument == "--offline" && !output.offline) {
            output.offline = true;
        } else if (argument == "--all-users" && !saw_all_users) {
            output.all_users = true;
            saw_all_users = true;
        } else if (argument == "--progress-token" &&
                   !saw_progress_token && index + 1 < argc) {
            output.progress_token = argv[++index];
            if (!cyxwiz::installer::IsInstallerProgressToken(
                    output.progress_token)) {
                error = "--progress-token must be a 32-character lowercase hexadecimal token";
                return false;
            }
            saw_progress_token = true;
        } else {
            error = "Unsupported, duplicate, or incomplete installer argument";
            return false;
        }
    }
    if (!saw_runtime_root || !output.runtime_root.is_absolute() ||
        (saw_metadata_root && !output.metadata_root.is_absolute()) ||
        static_cast<int>(saw_pack_id) +
                static_cast<int>(saw_base_pack_id) +
                static_cast<int>(saw_base_update_pack_id) +
                static_cast<int>(saw_deactivate_backend) != 1 ||
        ((saw_base_pack_id || saw_base_update_pack_id) && output.repair) ||
        (saw_deactivate_backend && (output.repair || output.offline))) {
        error = "--runtime-root and exactly one pack operation are required";
        return false;
    }
    if (!saw_metadata_root) output.metadata_root = output.runtime_root;
    return true;
}
#endif

}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
    Options options;
    std::string error;
    if (!ParseOptions(argc, argv, options, error)) {
        std::cerr << "CyxWiz backend-pack installer: " << error << '\n';
        return 78;
    }
    const auto progress_path = options.progress_token.empty()
        ? std::filesystem::path{}
        : cyxwiz::installer::InstallerProgressPath(
              options.runtime_root, options.progress_token);
    const auto publish_progress = [&](std::string stage,
                                      std::string message,
                                      std::uint64_t completed_bytes = 0,
                                      std::uint64_t total_bytes = 0,
                                      std::size_t component_index = 0,
                                      std::size_t component_count = 0) {
        if (progress_path.empty()) return;
        cyxwiz::installer::PublishInstallerProgress(
            progress_path,
            {std::move(stage), completed_bytes, total_bytes,
             component_index, component_count, std::move(message)});
    };
    if (!options.deactivate_backend.empty()) {
        publish_progress("removing", "Deactivating the selected backend");
        if (options.deactivate_backend != "cuda" &&
            options.deactivate_backend != "opencl" &&
            options.deactivate_backend != "oneapi") {
            std::cerr << "CyxWiz backend-pack installer: unsupported optional backend\n";
            return 78;
        }
        cyxwiz::runtime::ActiveRuntimeState active;
        if (!cyxwiz::runtime::LoadActiveRuntimeState(
                options.runtime_root / "active-runtime.json", active,
                error)) {
            std::cerr << "Cannot load the packaged runtime: " << error << '\n';
            return 1;
        }
        const bool already_inactive = std::none_of(
            active.packs.begin(), active.packs.end(),
            [&](const auto& pack) {
                return pack.backend == options.deactivate_backend;
            });
        if (already_inactive) {
            std::cout << options.deactivate_backend
                      << " is already inactive\n";
            return 0;
        }
        cyxwiz::runtime::BackendPackStateService state_service(
            options.runtime_root);
        const auto result = state_service.DeactivateOptionalPack(
            options.deactivate_backend);
        publish_progress(
            result.status == cyxwiz::runtime::BackendPackStateStatus::Completed
                ? "complete" : "failed",
            result.message);
        std::cout << result.message << '\n';
        return result.status ==
                cyxwiz::runtime::BackendPackStateStatus::Completed
            ? 0 : 1;
    }
    const auto executable_directory = ExecutableDirectory();
    if (executable_directory.empty()) {
        std::cerr << "Cannot resolve the installer executable directory\n";
        return 78;
    }
    auto trust = cyxwiz::runtime::BackendPackTrustStore::Load(
        options.metadata_root / "trust" / "trusted-keys.json", error);
    if (!trust) {
        std::cerr << "Cannot load the bundled trust store: " << error << '\n';
        return 78;
    }

    const auto metadata_root = options.metadata_root.lexically_normal();
    const auto runtime_root = options.runtime_root.lexically_normal();
    if (options.base && metadata_root != runtime_root) {
        cyxwiz::runtime::BackendPackLifecycleService metadata_service(
            metadata_root,
            cyxwiz::runtime::BackendPackMetadataVerifier(
                *trust, ClientVersion(),
                std::string(
                    cyxwiz::runtime::CurrentBackendPackPlatformId()),
                std::string(
                    cyxwiz::runtime::CurrentBackendPackArchitectureId())));
        cyxwiz::runtime::VerifiedBackendPackCatalogSnapshot snapshot;
        if (!metadata_service.ReadCatalogSnapshot(
                CurrentUtc(), snapshot, error) ||
            !cyxwiz::runtime::PublishVerifiedBackendPackMetadata(
                metadata_root / "trust" / "trusted-keys.json",
                snapshot, runtime_root, error)) {
            std::cerr << "Cannot seed verified installer metadata: "
                      << error << '\n';
            return 1;
        }
    }

    auto qualification_service =
        std::make_shared<cyxwiz::RouteQualificationService>();
    cyxwiz::BackendPackQualificationAdapterOptions qualification_options;
    qualification_options.runtime_root = options.runtime_root;
    qualification_options.probe_executable = options.base
        ? options.runtime_root / "base" / options.pack_id /
              cyxwiz::runtime::CurrentRouteProbeExecutableName()
        : executable_directory /
              cyxwiz::runtime::CurrentRouteProbeExecutableName();
    qualification_options.cache_path =
        cyxwiz::GetRouteQualificationCachePath();
    if (!options.base && !std::filesystem::is_regular_file(
            qualification_options.probe_executable)) {
        std::cerr << "Qualification helper is missing: "
                  << qualification_options.probe_executable.string() << '\n';
        return 78;
    }

    auto verifier = cyxwiz::runtime::BackendPackMetadataVerifier(
        std::move(*trust), ClientVersion(),
        std::string(cyxwiz::runtime::CurrentBackendPackPlatformId()),
        std::string(cyxwiz::runtime::CurrentBackendPackArchitectureId()));
    cyxwiz::runtime::BackendPackLifecycleService lifecycle(
        options.runtime_root, std::move(verifier),
        cyxwiz::runtime::BackendPackExecutionActiveCheck{},
        cyxwiz::CreateBackendPackQualificationHook(
            qualification_service, std::move(qualification_options)),
        [publish_progress](
            const cyxwiz::runtime::BackendPackLifecycleProgress& progress) {
            std::cout
                << cyxwiz::runtime::BackendPackLifecycleStageName(
                       progress.stage)
                << ": " << progress.message << '\n';
            publish_progress(
                cyxwiz::runtime::BackendPackLifecycleStageName(
                    progress.stage),
                progress.message, progress.completed_bytes,
                progress.total_bytes, progress.component_index,
                progress.component_count);
        });

    cyxwiz::runtime::BackendPackDeliveryRequest request;
    request.catalog_path =
        cyxwiz::runtime::BackendPackCurrentCatalogPath(
            options.metadata_root);
    request.manifest_path =
        cyxwiz::runtime::BackendPackCachedManifestPath(
            options.metadata_root, options.pack_id);
    request.current_utc = CurrentUtc();
    request.pack_id = options.pack_id;
    request.repair = options.repair;
    request.source = options.offline
        ? cyxwiz::runtime::BackendPackDeliverySource::OfflineSibling
        : cyxwiz::runtime::BackendPackDeliverySource::CatalogHttps;
    const auto result = options.base_update
        ? lifecycle.DeliverBaseUpdate(request)
        : (options.base ? lifecycle.DeliverBase(request)
                        : lifecycle.Deliver(request));
    std::cout << result.message << '\n';
    if (result.status == cyxwiz::runtime::
            BackendPackLifecycleStatus::InstalledAndActivated) {
        if (options.base) {
            publish_progress(
                "registering",
                "Registering CyxWiz application shortcuts and launcher");
            cyxwiz::runtime::ProductRegistrationRequest registration;
            registration.install_root = options.runtime_root.parent_path();
            registration.runtime_root = options.runtime_root;
            registration.scope = options.all_users
                ? cyxwiz::runtime::ProductInstallScope::AllUsers
                : cyxwiz::runtime::ProductInstallScope::CurrentUser;
            std::string installed_version;
            if (!cyxwiz::runtime::LoadProductReleaseVersion(
                    result.installed_directory, installed_version, error)) {
                const auto message =
                    "Cannot register the activated CyxWiz release: " + error;
                std::cerr << message << '\n';
                publish_progress("failed", message);
                return 3;
            }
            registration.product_version = std::move(installed_version);
            const auto registered =
                cyxwiz::runtime::RegisterInstalledProduct(registration);
            std::cout << registered.message << '\n';
            if (!registered.registered) {
                publish_progress("failed", registered.message);
                return 3;
            }
            publish_progress("complete", registered.message);
        }
        return 0;
    }
    if (result.status == cyxwiz::runtime::
            BackendPackLifecycleStatus::InstalledUnqualified) {
        return 2;
    }
    return 1;
}
