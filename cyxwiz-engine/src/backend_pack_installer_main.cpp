#include "core/backend_pack_qualification_adapter.h"
#include "core/compute_runtime_paths.h"

#include "backend_pack_lifecycle_service.h"
#include "backend_pack_platform.h"

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
    std::string pack_id;
    bool repair = false;
    bool offline = false;
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
    bool saw_pack_id = false;
    for (int index = 1; index < argc; ++index) {
        const std::wstring_view argument(argv[index]);
        if (argument == L"--runtime-root" && !saw_runtime_root &&
            index + 1 < argc) {
            output.runtime_root = argv[++index];
            saw_runtime_root = true;
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
        } else if (argument == L"--repair" && !output.repair) {
            output.repair = true;
        } else if (argument == L"--offline" && !output.offline) {
            output.offline = true;
        } else {
            error = "Unsupported, duplicate, or incomplete installer argument";
            return false;
        }
    }
    if (!saw_runtime_root || !output.runtime_root.is_absolute() ||
        !saw_pack_id || output.pack_id.empty()) {
        error = "--runtime-root and --pack-id are required";
        return false;
    }
    return true;
}
#else
bool ParseOptions(
    int argc,
    char** argv,
    Options& output,
    std::string& error) {
    bool saw_runtime_root = false;
    bool saw_pack_id = false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--runtime-root" && !saw_runtime_root &&
            index + 1 < argc) {
            output.runtime_root = argv[++index];
            saw_runtime_root = true;
        } else if (argument == "--pack-id" && !saw_pack_id &&
                   index + 1 < argc) {
            output.pack_id = argv[++index];
            if (!IsIdentifier(output.pack_id)) {
                error = "--pack-id must be a safe ASCII identifier";
                return false;
            }
            saw_pack_id = true;
        } else if (argument == "--repair" && !output.repair) {
            output.repair = true;
        } else if (argument == "--offline" && !output.offline) {
            output.offline = true;
        } else {
            error = "Unsupported, duplicate, or incomplete installer argument";
            return false;
        }
    }
    if (!saw_runtime_root || !output.runtime_root.is_absolute() ||
        !saw_pack_id || output.pack_id.empty()) {
        error = "--runtime-root and --pack-id are required";
        return false;
    }
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
    const auto executable_directory = ExecutableDirectory();
    if (executable_directory.empty()) {
        std::cerr << "Cannot resolve the installer executable directory\n";
        return 78;
    }
    auto trust = cyxwiz::runtime::BackendPackTrustStore::Load(
        options.runtime_root / "trust" / "trusted-keys.json", error);
    if (!trust) {
        std::cerr << "Cannot load the bundled trust store: " << error << '\n';
        return 78;
    }

    auto qualification_service =
        std::make_shared<cyxwiz::RouteQualificationService>();
    cyxwiz::BackendPackQualificationAdapterOptions qualification_options;
    qualification_options.runtime_root = options.runtime_root;
    qualification_options.probe_executable =
        executable_directory /
        cyxwiz::runtime::CurrentRouteProbeExecutableName();
    qualification_options.cache_path =
        cyxwiz::GetRouteQualificationCachePath();
    if (!std::filesystem::is_regular_file(
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
        [](const cyxwiz::runtime::BackendPackLifecycleProgress& progress) {
            std::cout
                << cyxwiz::runtime::BackendPackLifecycleStageName(
                       progress.stage)
                << ": " << progress.message << '\n';
        });

    cyxwiz::runtime::BackendPackDeliveryRequest request;
    request.catalog_path =
        cyxwiz::runtime::BackendPackCurrentCatalogPath(
            options.runtime_root);
    request.manifest_path =
        cyxwiz::runtime::BackendPackCachedManifestPath(
            options.runtime_root, options.pack_id);
    request.current_utc = CurrentUtc();
    request.pack_id = options.pack_id;
    request.repair = options.repair;
    request.source = options.offline
        ? cyxwiz::runtime::BackendPackDeliverySource::OfflineSibling
        : cyxwiz::runtime::BackendPackDeliverySource::CatalogHttps;
    const auto result = lifecycle.Deliver(request);
    std::cout << result.message << '\n';
    if (result.status == cyxwiz::runtime::
            BackendPackLifecycleStatus::InstalledAndActivated) {
        return 0;
    }
    if (result.status == cyxwiz::runtime::
            BackendPackLifecycleStatus::InstalledUnqualified) {
        return 2;
    }
    return 1;
}
