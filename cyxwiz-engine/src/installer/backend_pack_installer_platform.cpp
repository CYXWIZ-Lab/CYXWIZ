#include "backend_pack_installer_platform.h"

#include <cyxwiz/version.h>

#include "backend_pack_platform.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cwctype>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#else
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "backend_pack_lifecycle_service.h"
#include "backend_pack_state_service.h"
#include "core/backend_pack_catalog_adapter.h"

namespace cyxwiz::installer {
namespace {

bool IsIdentifier(const std::string& value) {
    if (value.empty() || value.size() > 128 ||
        !std::isalnum(static_cast<unsigned char>(value.front()))) {
        return false;
    }
    return std::all_of(
        value.begin(), value.end(), [](unsigned char character) {
            return std::islower(character) || std::isdigit(character) ||
                   character == '.' || character == '_' || character == '-';
        });
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

std::filesystem::path HelperPath(
    const std::filesystem::path& executable_directory) {
    return executable_directory /
        runtime::CurrentBackendPackInstallerExecutableName();
}

#ifdef _WIN32

std::wstring QuoteWindowsArgument(const std::wstring& value) {
    std::wstring quoted = L"\"";
    std::size_t backslashes = 0;
    for (const wchar_t character : value) {
        if (character == L'\\') {
            ++backslashes;
            continue;
        }
        if (character == L'\"') {
            quoted.append(backslashes * 2 + 1, L'\\');
            quoted.push_back(character);
            backslashes = 0;
            continue;
        }
        quoted.append(backslashes, L'\\');
        backslashes = 0;
        quoted.push_back(character);
    }
    quoted.append(backslashes * 2, L'\\');
    quoted.push_back(L'\"');
    return quoted;
}

int RunHelper(
    const std::filesystem::path& helper,
    const std::filesystem::path& runtime_root,
    const std::wstring& operation,
    const std::string& value,
    std::string& error) {
    std::wstring command = QuoteWindowsArgument(helper.native()) +
        L" --runtime-root " + QuoteWindowsArgument(runtime_root.native()) +
        L" " + operation + L" " + QuoteWindowsArgument(
            std::wstring(value.begin(), value.end()));
    std::vector<wchar_t> mutable_command(command.begin(), command.end());
    mutable_command.push_back(L'\0');
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!::CreateProcessW(
            helper.c_str(), mutable_command.data(), nullptr, nullptr, FALSE,
            CREATE_NO_WINDOW, nullptr, helper.parent_path().c_str(),
            &startup, &process)) {
        error = "Cannot launch the signed pack helper; Win32 error " +
                std::to_string(::GetLastError());
        return -1;
    }
    ::CloseHandle(process.hThread);
    const DWORD wait = ::WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exit_code = 1;
    if (wait == WAIT_OBJECT_0 &&
        ::GetExitCodeProcess(process.hProcess, &exit_code)) {
        ::CloseHandle(process.hProcess);
        return exit_code <= static_cast<DWORD>(
                   std::numeric_limits<int>::max())
            ? static_cast<int>(exit_code) : -1;
    }
    error = "Waiting for the signed pack helper failed";
    ::CloseHandle(process.hProcess);
    return -1;
}

std::vector<std::string> RecommendedBackends() {
    bool nvidia = false;
    bool intel = false;
    bool amd = false;
    for (DWORD index = 0;; ++index) {
        DISPLAY_DEVICEW device{};
        device.cb = sizeof(device);
        if (!::EnumDisplayDevicesW(nullptr, index, &device, 0)) break;
        if ((device.StateFlags & DISPLAY_DEVICE_MIRRORING_DRIVER) != 0) {
            continue;
        }
        std::wstring name(device.DeviceString);
        std::transform(
            name.begin(), name.end(), name.begin(),
            [](wchar_t value) { return std::towlower(value); });
        nvidia = nvidia || name.find(L"nvidia") != std::wstring::npos;
        intel = intel || name.find(L"intel") != std::wstring::npos;
        amd = amd || name.find(L"amd") != std::wstring::npos ||
              name.find(L"radeon") != std::wstring::npos;
    }
    std::vector<std::string> backends;
    if (nvidia) backends.push_back("cuda");
    if (intel || amd) backends.push_back("opencl");
    return backends;
}

#else

int RunHelper(
    const std::filesystem::path& helper,
    const std::filesystem::path& runtime_root,
    const char* operation,
    const std::string& value,
    std::string& error) {
    const auto helper_text = helper.string();
    const auto root_text = runtime_root.string();
    const pid_t child = ::fork();
    if (child < 0) {
        error = "Cannot fork the signed pack helper";
        return -1;
    }
    if (child == 0) {
        ::execl(
            helper_text.c_str(), helper_text.c_str(), "--runtime-root",
            root_text.c_str(), operation, value.c_str(),
            static_cast<char*>(nullptr));
        ::_exit(127);
    }
    int status = 0;
    if (::waitpid(child, &status, 0) != child || !WIFEXITED(status)) {
        error = "Waiting for the signed pack helper failed";
        return -1;
    }
    return WEXITSTATUS(status);
}

std::vector<std::string> RecommendedBackends() {
    std::vector<std::string> backends;
#if defined(__linux__)
    bool nvidia = std::filesystem::is_regular_file(
        "/proc/driver/nvidia/version");
    bool intel_or_amd = false;
    std::error_code error;
    const std::filesystem::path drm_root("/sys/class/drm");
    for (std::filesystem::directory_iterator iterator(drm_root, error), end;
         !error && iterator != end; iterator.increment(error)) {
        std::ifstream vendor(iterator->path() / "device" / "vendor");
        std::string value;
        if (!(vendor >> value)) continue;
        std::transform(
            value.begin(), value.end(), value.begin(),
            [](unsigned char character) {
                return static_cast<char>(std::tolower(character));
            });
        nvidia = nvidia || value == "0x10de";
        intel_or_amd = intel_or_amd || value == "0x8086" ||
            value == "0x1002";
    }
    if (nvidia) backends.push_back("cuda");
    if (intel_or_amd) backends.push_back("opencl");
#endif
    return backends;
}

#endif

class DesktopInstallerPlatform final : public BackendPackInstallerPlatform {
public:
    DesktopInstallerPlatform(
        std::filesystem::path runtime_root,
        std::filesystem::path executable_directory)
        : runtime_root_(std::move(runtime_root)),
          executable_directory_(std::move(executable_directory)) {}

    InstallerCatalogState Refresh() override {
        InstallerCatalogState state;
        if (!runtime_root_.is_absolute()) {
            state.message = "The CyxWiz runtime root must be absolute";
            return state;
        }
        runtime::ActiveRuntimeState active;
        std::string error;
        if (!runtime::LoadActiveRuntimeState(
                runtime_root_ / "active-runtime.json", active, error)) {
            state.message = "Cannot load the packaged runtime: " + error;
            return state;
        }
        runtime::VerifiedBackendPackCatalogSnapshot active_only;
        state.records = BuildBackendPackCatalogRecords(active_only, active);
        auto trust = runtime::BackendPackTrustStore::Load(
            runtime_root_ / "trust" / "trusted-keys.json", error);
        if (!trust) {
            state.message = "Cannot load the bundled trust store: " + error;
            return state;
        }
        runtime::BackendPackLifecycleService lifecycle(
            runtime_root_, runtime::BackendPackMetadataVerifier(
                std::move(*trust), ClientVersion(),
                std::string(runtime::CurrentBackendPackPlatformId()),
                std::string(runtime::CurrentBackendPackArchitectureId())));
        runtime::VerifiedBackendPackCatalogSnapshot snapshot;
        if (!lifecycle.ReadCatalogSnapshot(CurrentUtc(), snapshot, error)) {
            state.message = "Cannot verify the current pack catalog: " + error;
            return state;
        }
        state.records = BuildBackendPackCatalogRecords(snapshot, active);
        const auto recommended = RecommendedBackends();
        for (auto& record : state.records) {
            record.recommended =
                record.catalog_support == BackendPackCatalogSupport::Supported &&
                std::find(
                    recommended.begin(), recommended.end(), record.backend) !=
                    recommended.end();
        }
        state.available = true;
        state.catalog_id = snapshot.catalog.catalog_id;
        state.message = "Verified signed catalog " + state.catalog_id;
        return state;
    }

    InstallerOperationResult InstallOrUpdate(
        const std::string& pack_id) override {
        InstallerOperationResult result;
        if (!IsIdentifier(pack_id)) {
            result.message = "The selected pack identity is invalid";
            return result;
        }
        const auto helper = HelperPath(executable_directory_);
        if (!std::filesystem::is_regular_file(helper)) {
            result.message =
                "The signed backend-pack helper is missing from this installation";
            return result;
        }
        std::string error;
        const int exit_code = RunHelper(
            helper, runtime_root_,
#ifdef _WIN32
            L"--pack-id",
#else
            "--pack-id",
#endif
            pack_id, error);
        if (exit_code == 0) {
            result.succeeded = true;
            result.activated = true;
            result.message =
                pack_id + " was installed, qualified, and activated";
        } else if (exit_code == 2) {
            result.succeeded = true;
            result.message = pack_id +
                " was installed but local qualification did not authorize activation";
        } else {
            result.message = error.empty()
                ? "Pack installation failed with helper exit code " +
                      std::to_string(exit_code)
                : std::move(error);
        }
        return result;
    }

    InstallerOperationResult DeactivateBackend(
        const std::string& backend) override {
        InstallerOperationResult result;
        if (backend != "cuda" && backend != "opencl" &&
            backend != "oneapi") {
            result.message = "The optional backend identity is invalid";
            return result;
        }
        const auto helper = HelperPath(executable_directory_);
        if (!std::filesystem::is_regular_file(helper)) {
            result.message =
                "The signed backend-pack helper is missing from this installation";
            return result;
        }
        std::string error;
        const int exit_code = RunHelper(
            helper, runtime_root_,
#ifdef _WIN32
            L"--deactivate-backend",
#else
            "--deactivate-backend",
#endif
            backend, error);
        if (exit_code == 0) {
            result.succeeded = true;
            result.message = backend +
                " was deactivated; its package files remain installed";
        } else {
            result.message = error.empty()
                ? "Backend deactivation failed with helper exit code " +
                      std::to_string(exit_code)
                : std::move(error);
        }
        return result;
    }

    std::string PlatformName() const override {
#ifdef _WIN32
        return "Windows";
#elif defined(__APPLE__)
        return "macOS";
#else
        return "Linux";
#endif
    }

private:
    std::filesystem::path runtime_root_;
    std::filesystem::path executable_directory_;
};

}  // namespace

std::unique_ptr<BackendPackInstallerPlatform>
CreateBackendPackInstallerPlatform(
    std::filesystem::path runtime_root,
    std::filesystem::path executable_directory) {
    return std::make_unique<DesktopInstallerPlatform>(
        std::move(runtime_root), std::move(executable_directory));
}

}  // namespace cyxwiz::installer
