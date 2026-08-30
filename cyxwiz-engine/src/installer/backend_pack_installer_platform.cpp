#include "backend_pack_installer_platform.h"

#include <cyxwiz/version.h>

#include "backend_pack_platform.h"
#include "installer_cancellation_channel.h"
#include "installer_helper_session.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>
#include <utility>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>

#else
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "backend_pack_lifecycle_service.h"
#include "backend_pack_metadata_refresh.h"
#include "backend_pack_state_service.h"
#include "core/backend_pack_catalog_adapter.h"
#include "core/backend_pack_decision_reconciliation.h"
#include "core/compute_runtime_paths.h"
#include "core/route_qualification_snapshot.h"

namespace cyxwiz::installer {
namespace {

using InstallerHelperStarted =
    std::function<void(const std::string&)>;
using InstallerHelperFinished =
    std::function<void(const std::string&)>;

class InstallerHelperControlScope {
public:
    InstallerHelperControlScope(
        std::string token, const InstallerHelperStarted& started,
        const InstallerHelperFinished& finished)
        : token_(std::move(token)), finished_(finished) {
        ClearInstallerCancellation(token_);
        if (started) started(token_);
    }

    ~InstallerHelperControlScope() {
        if (finished_) finished_(token_);
        ClearInstallerCancellation(token_);
    }

private:
    std::string token_;
    InstallerHelperFinished finished_;
};

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

std::optional<DeviceType> BackendType(const std::string& backend) {
    if (backend == "cpu") return DeviceType::CPU;
    if (backend == "cuda") return DeviceType::CUDA;
    if (backend == "opencl") return DeviceType::OPENCL;
    if (backend == "oneapi") return DeviceType::ONEAPI;
    return std::nullopt;
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

void PollHelperProgress(
    const std::filesystem::path& progress_path,
    const InstallerOperationDetailObserver& observer,
    std::optional<InstallerHelperProgress>& previous) {
    if (!observer) return;
    const auto current = ReadInstallerProgress(progress_path);
    if (!current.has_value()) return;
    const bool changed = !previous.has_value() ||
        current->stage != previous->stage ||
        current->completed_bytes != previous->completed_bytes ||
        current->total_bytes != previous->total_bytes ||
        current->component_index != previous->component_index ||
        current->component_count != previous->component_count ||
        current->message != previous->message;
    if (changed) {
        observer(*current);
        previous = current;
    }
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

int WaitForHelper(
    HANDLE process, const std::filesystem::path& progress_path,
    const InstallerOperationDetailObserver& observer, std::string& error) {
    std::optional<InstallerHelperProgress> previous;
    DWORD wait = WAIT_TIMEOUT;
    while ((wait = ::WaitForSingleObject(process, 100)) == WAIT_TIMEOUT) {
        PollHelperProgress(progress_path, observer, previous);
    }
    PollHelperProgress(progress_path, observer, previous);
    std::error_code cleanup_error;
    std::filesystem::remove(progress_path, cleanup_error);
    DWORD exit_code = 1;
    if (wait == WAIT_OBJECT_0 &&
        ::GetExitCodeProcess(process, &exit_code)) {
        if (exit_code != 0 && previous.has_value() &&
            previous->stage == "failed" && !previous->message.empty()) {
            error = previous->message;
        }
        ::CloseHandle(process);
        return exit_code <= static_cast<DWORD>(
                   std::numeric_limits<int>::max())
            ? static_cast<int>(exit_code) : -1;
    }
    error = "Waiting for the signed pack helper failed";
    ::CloseHandle(process);
    return -1;
}

int RunHelper(
    const std::filesystem::path& helper,
    const std::filesystem::path& runtime_root,
    const std::filesystem::path& metadata_root,
    bool elevate,
    InstallerPackageSource package_source,
    const std::wstring& operation,
    const std::string& value,
    const InstallerOperationDetailObserver& observer,
    const InstallerHelperStarted& started,
    const InstallerHelperFinished& finished,
    std::string& error) {
    const auto progress_token = CreateInstallerProgressToken();
    const InstallerHelperControlScope control(
        progress_token, started, finished);
    const auto progress_path =
        InstallerProgressPath(runtime_root, progress_token);
    const std::wstring parameters =
        L"--runtime-root " + QuoteWindowsArgument(runtime_root.native()) +
        L" --metadata-root " + QuoteWindowsArgument(metadata_root.native()) +
        L" " + operation + L" " + QuoteWindowsArgument(
            std::wstring(value.begin(), value.end())) +
        L" --progress-token " + QuoteWindowsArgument(
            std::wstring(progress_token.begin(), progress_token.end())) +
        L" --parent-pid " +
            std::to_wstring(CurrentInstallerProcessId()) +
        (package_source == InstallerPackageSource::OfflineSibling
             ? L" --offline" : L"") +
        (elevate ? L" --all-users" : L"");
    if (elevate) {
        SHELLEXECUTEINFOW execute{};
        execute.cbSize = sizeof(execute);
        execute.fMask = SEE_MASK_NOCLOSEPROCESS;
        execute.lpVerb = L"runas";
        execute.lpFile = helper.c_str();
        execute.lpParameters = parameters.c_str();
        execute.lpDirectory = helper.parent_path().c_str();
        execute.nShow = SW_HIDE;
        if (!::ShellExecuteExW(&execute) || !execute.hProcess) {
            const auto code = ::GetLastError();
            error = code == ERROR_CANCELLED
                ? "System-wide installation authorization was cancelled"
                : "Cannot launch the authorized pack helper; Win32 error " +
                      std::to_string(code);
            return -1;
        }
        return WaitForHelper(
            execute.hProcess, progress_path, observer, error);
    }
    std::wstring command = QuoteWindowsArgument(helper.native()) + L" " +
        parameters;
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
    return WaitForHelper(process.hProcess, progress_path, observer, error);
}

#else

int RunHelper(
    const std::filesystem::path& helper,
    const std::filesystem::path& runtime_root,
    const std::filesystem::path& metadata_root,
    bool elevate,
    InstallerPackageSource package_source,
    const char* operation,
    const std::string& value,
    const InstallerOperationDetailObserver& observer,
    const InstallerHelperStarted& started,
    const InstallerHelperFinished& finished,
    std::string& error) {
    if (elevate && ::geteuid() != 0) {
        error =
            "System-wide installation requires launching CyxWiz Installer with administrative privileges";
        return -1;
    }
    const auto helper_text = helper.string();
    const auto root_text = runtime_root.string();
    const auto metadata_text = metadata_root.string();
    const std::string operation_text(operation);
    const auto progress_token = CreateInstallerProgressToken();
    const InstallerHelperControlScope control(
        progress_token, started, finished);
    const auto progress_path =
        InstallerProgressPath(runtime_root, progress_token);
    std::vector<std::string> arguments{
        helper_text, "--runtime-root", root_text,
        "--metadata-root", metadata_text, operation_text, value,
        "--progress-token", progress_token};
    arguments.emplace_back("--parent-pid");
    arguments.emplace_back(std::to_string(CurrentInstallerProcessId()));
    if (package_source == InstallerPackageSource::OfflineSibling) {
        arguments.emplace_back("--offline");
    }
    if (elevate) arguments.emplace_back("--all-users");
    std::vector<char*> native_arguments;
    native_arguments.reserve(arguments.size() + 1);
    for (auto& argument : arguments) {
        native_arguments.push_back(argument.data());
    }
    native_arguments.push_back(nullptr);
    const pid_t child = ::fork();
    if (child < 0) {
        error = "Cannot fork the signed pack helper";
        return -1;
    }
    if (child == 0) {
        ::execv(helper_text.c_str(), native_arguments.data());
        ::_exit(127);
    }
    std::optional<InstallerHelperProgress> previous;
    int status = 0;
    pid_t waited = 0;
    while ((waited = ::waitpid(child, &status, WNOHANG)) == 0) {
        PollHelperProgress(progress_path, observer, previous);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    PollHelperProgress(progress_path, observer, previous);
    std::error_code cleanup_error;
    std::filesystem::remove(progress_path, cleanup_error);
    if (waited != child || !WIFEXITED(status)) {
        error = "Waiting for the signed pack helper failed";
        return -1;
    }
    const int exit_code = WEXITSTATUS(status);
    if (exit_code != 0 && previous.has_value() &&
        previous->stage == "failed" && !previous->message.empty()) {
        error = previous->message;
    }
    return exit_code;
}

#endif

class DesktopInstallerPlatform final : public BackendPackInstallerPlatform {
public:
    DesktopInstallerPlatform(
        std::filesystem::path runtime_root,
        std::filesystem::path metadata_root,
        std::filesystem::path executable_directory,
        CyxWizInstallScope scope,
        std::string catalog_url,
        InstallerPackageSource package_source)
        : runtime_root_(std::move(runtime_root)),
          metadata_root_(std::move(metadata_root)),
          executable_directory_(std::move(executable_directory)),
          catalog_url_(std::move(catalog_url)),
          package_source_(package_source),
          elevate_(scope == CyxWizInstallScope::AllUsers) {}

    void BeginPlanExecution() override {
        const std::scoped_lock lock(helper_mutex_);
        plan_running_ = true;
        cancellation_requested_ = false;
        active_helper_token_.clear();
    }

    void EndPlanExecution() override {
        const std::scoped_lock lock(helper_mutex_);
        plan_running_ = false;
        cancellation_requested_ = false;
        active_helper_token_.clear();
    }

    InstallerCatalogState Refresh() override {
        InstallerCatalogState state;
        state.cuda_prerequisite = DetectInstallerCudaPrerequisite();
        if (!runtime_root_.is_absolute()) {
            state.message = "The CyxWiz runtime root must be absolute";
            return state;
        }
        runtime::ActiveRuntimeState active;
        std::string error;
        const auto active_path = runtime_root_ / "active-runtime.json";
        std::error_code filesystem_error;
        const bool active_exists =
            std::filesystem::exists(active_path, filesystem_error);
        if (filesystem_error) {
            state.message = "Cannot inspect the installation state: " +
                filesystem_error.message();
            return state;
        }
        if (active_exists) {
            if (!std::filesystem::is_regular_file(
                    active_path, filesystem_error) || filesystem_error ||
                !runtime::LoadActiveRuntimeState(
                    active_path, active, error)) {
                state.message =
                    "The existing installation is incomplete or invalid: " +
                    (error.empty() ? filesystem_error.message() : error);
                return state;
            }
            state.mode = CyxWizInstallerMode::Maintenance;
        } else {
            state.mode = CyxWizInstallerMode::FreshInstall;
        }
        RuntimeQualificationIdentity active_identity;
        active_identity.runtime_set_id = active.runtime_set_id;
        active_identity.generation = active.generation;
        active_identity.base_pack_id = active.base_pack_id;
        for (const auto& pack : active.packs) {
            const auto backend = BackendType(pack.backend);
            if (backend.has_value()) {
                active_identity.backend_packs.push_back(
                    {*backend, pack.pack_id});
            }
        }
        ClearRouteQualificationSnapshot();
        const auto qualification_load =
            LoadAndInstallRouteQualificationSnapshot(
                GetRouteQualificationCachePath());
        state.verification = BuildInstallerVerificationSummary(
            qualification_load.loaded
                ? GetRouteQualificationSnapshot()
                : std::optional<RouteQualificationSnapshot>{},
            active_identity);
        runtime::VerifiedBackendPackCatalogSnapshot active_only;
        state.records = BuildBackendPackCatalogRecords(active_only, active);
        const auto catalog_root = CurrentCatalogRoot();
        if (!catalog_root.is_absolute()) {
            state.message = "The signed metadata root must be absolute";
            return state;
        }
        auto trust = runtime::BackendPackTrustStore::Load(
            catalog_root / "trust" / "trusted-keys.json", error);
        if (!trust) {
            state.message = "Cannot load the bundled trust store: " + error;
            return state;
        }
        runtime::BackendPackLifecycleService lifecycle(
            catalog_root, runtime::BackendPackMetadataVerifier(
                std::move(*trust), ClientVersion(),
                std::string(runtime::CurrentBackendPackPlatformId()),
                std::string(runtime::CurrentBackendPackArchitectureId())));
        runtime::VerifiedBackendPackCatalogSnapshot snapshot;
        if (!lifecycle.ReadCatalogSnapshot(CurrentUtc(), snapshot, error)) {
            state.message = "Cannot verify the current pack catalog: " + error;
            return state;
        }
        state.records = BuildBackendPackCatalogRecords(snapshot, active);
        ReconcileBackendPackDecisionEvidence(
            state.records, state.verification);
        state.available = true;
        state.catalog_id = snapshot.catalog.catalog_id;
        state.message = "Verified signed catalog " + state.catalog_id;
        return state;
    }

    InstallerCatalogRefreshResult RefreshOnline() override {
        InstallerCatalogRefreshResult result;
        if (catalog_url_.empty()) {
            result.message =
                "Online catalog source is not configured; the packaged verified catalog remains active";
            return result;
        }
        const auto catalog_root = CurrentCatalogRoot();
        std::string error;
        auto trust = runtime::BackendPackTrustStore::Load(
            catalog_root / "trust" / "trusted-keys.json", error);
        if (!trust) {
            result.message = "Cannot load the catalog trust store: " + error;
            return result;
        }
        runtime::BackendPackMetadataVerifier verifier(
            std::move(*trust), ClientVersion(),
            std::string(runtime::CurrentBackendPackPlatformId()),
            std::string(runtime::CurrentBackendPackArchitectureId()));
        runtime::HttpsBackendPackMetadataSource source;
        runtime::BackendPackMetadataRefreshRequest request;
        request.catalog_url = catalog_url_;
        request.trusted_keys_path =
            catalog_root / "trust" / "trusted-keys.json";
        request.destination_root = runtime_root_;
        request.current_utc = CurrentUtc();
        const auto refreshed = runtime::RefreshBackendPackMetadata(
            request, verifier, source);
        result.succeeded = refreshed.status ==
            runtime::BackendPackMetadataRefreshStatus::Refreshed;
        if (result.succeeded) online_refresh_succeeded_ = true;
        result.message = refreshed.message;
        return result;
    }

    InstallerOperationResult InstallOrUpdate(
        const std::string& pack_id,
        const InstallerOperationDetailObserver& observer) override {
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
            helper, runtime_root_, CurrentCatalogRoot(), elevate_,
            package_source_,
#ifdef _WIN32
            L"--pack-id",
#else
            "--pack-id",
#endif
            pack_id, observer,
            [this](const std::string& token) { BeginHelper(token); },
            [this](const std::string& token) { FinishHelper(token); }, error);
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

    InstallerOperationResult InstallBase(
        const std::string& pack_id,
        const InstallerOperationDetailObserver& observer) override {
        InstallerOperationResult result;
        if (!IsIdentifier(pack_id)) {
            result.message = "The required base identity is invalid";
            return result;
        }
        const auto helper = HelperPath(executable_directory_);
        if (!std::filesystem::is_regular_file(helper)) {
            result.message =
                "The signed product-installation helper is missing";
            return result;
        }
        std::string error;
        const int exit_code = RunHelper(
            helper, runtime_root_, CurrentCatalogRoot(), elevate_,
            package_source_,
#ifdef _WIN32
            L"--base-pack-id",
#else
            "--base-pack-id",
#endif
            pack_id, observer,
            [this](const std::string& token) { BeginHelper(token); },
            [this](const std::string& token) { FinishHelper(token); }, error);
        if (exit_code == 0) {
            result.succeeded = true;
            result.activated = true;
            result.message =
                pack_id + " was installed, CPU-qualified, and activated";
        } else if (exit_code == 3) {
            result.succeeded = true;
            result.activated = true;
            result.message = pack_id +
                " was installed and activated, but operating-system launch integration could not be registered; the stable launcher remains available in the installation folder";
        } else if (exit_code == 2) {
            result.succeeded = true;
            result.message = pack_id +
                " was installed but CPU qualification did not authorize activation";
        } else {
            result.message = error.empty()
                ? "Base installation failed with helper exit code " +
                      std::to_string(exit_code)
                : std::move(error);
        }
        return result;
    }

    InstallerOperationResult UpdateBase(
        const std::string& pack_id,
        const InstallerOperationDetailObserver& observer) override {
        InstallerOperationResult result;
        if (!IsIdentifier(pack_id)) {
            result.message = "The CPU-base update identity is invalid";
            return result;
        }
        const auto helper = HelperPath(executable_directory_);
        if (!std::filesystem::is_regular_file(helper)) {
            result.message =
                "The signed product-update helper is missing";
            return result;
        }
        std::string error;
        const int exit_code = RunHelper(
            helper, runtime_root_, CurrentCatalogRoot(), elevate_,
            package_source_,
#ifdef _WIN32
            L"--update-base-pack-id",
#else
            "--update-base-pack-id",
#endif
            pack_id, observer,
            [this](const std::string& token) { BeginHelper(token); },
            [this](const std::string& token) { FinishHelper(token); }, error);
        if (exit_code == 0) {
            result.succeeded = true;
            result.activated = true;
            result.message = pack_id +
                " was installed, CPU-qualified, and activated as the current Engine";
        } else if (exit_code == 3) {
            result.succeeded = true;
            result.activated = true;
            result.message = pack_id +
                " was activated, but operating-system launch integration could not be refreshed";
        } else if (exit_code == 2) {
            result.succeeded = true;
            result.message = pack_id +
                " was installed but CPU qualification kept the previous Engine active";
        } else {
            result.message = error.empty()
                ? "Base update failed with helper exit code " +
                      std::to_string(exit_code)
                : std::move(error);
        }
        return result;
    }

    InstallerOperationResult DeactivateBackend(
        const std::string& backend,
        const InstallerOperationDetailObserver& observer) override {
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
            helper, runtime_root_, CurrentCatalogRoot(), elevate_,
            package_source_,
#ifdef _WIN32
            L"--deactivate-backend",
#else
            "--deactivate-backend",
#endif
            backend, observer,
            [this](const std::string& token) { BeginHelper(token); },
            [this](const std::string& token) { FinishHelper(token); }, error);
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

    InstallerOperationResult RequestCancellation() override {
        InstallerOperationResult result;
        const std::scoped_lock lock(helper_mutex_);
        if (!plan_running_) {
            result.message = "No installation operation is active";
            return result;
        }
        cancellation_requested_ = true;
        if (active_helper_token_.empty()) {
            result.succeeded = true;
            result.message =
                "Cancellation requested; CyxWiz will stop before the next component";
            return result;
        }
        result.succeeded =
            RequestInstallerCancellation(active_helper_token_);
        result.message = result.succeeded
            ? "Cancellation requested; CyxWiz is stopping safely"
            : "Cannot publish the cancellation request";
        return result;
    }

    InstallerOperationResult LaunchEngine() override {
        return LaunchStableBootstrapper(false);
    }

    InstallerOperationResult OpenInstalledManager() override {
        return LaunchStableBootstrapper(true);
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
    void BeginHelper(const std::string& token) {
        const std::scoped_lock lock(helper_mutex_);
        active_helper_token_ = token;
        if (cancellation_requested_) {
            RequestInstallerCancellation(active_helper_token_);
        }
    }

    void FinishHelper(const std::string& token) {
        const std::scoped_lock lock(helper_mutex_);
        if (active_helper_token_ == token) active_helper_token_.clear();
    }

    InstallerOperationResult LaunchStableBootstrapper(
        bool installer_mode) const {
        InstallerOperationResult result;
        const auto launcher = runtime_root_.parent_path() /
            runtime::CurrentRuntimeBootstrapperExecutableName();
        if (!std::filesystem::is_regular_file(launcher)) {
            result.message =
                "The installed CyxWiz launcher is missing";
            return result;
        }
#ifdef _WIN32
        std::wstring command = QuoteWindowsArgument(launcher.native());
        if (installer_mode) command += L" --installer";
        std::vector<wchar_t> mutable_command(command.begin(), command.end());
        mutable_command.push_back(L'\0');
        STARTUPINFOW startup{};
        startup.cb = sizeof(startup);
        PROCESS_INFORMATION process{};
        if (!::CreateProcessW(
                launcher.c_str(), mutable_command.data(), nullptr, nullptr,
                FALSE, 0, nullptr, launcher.parent_path().c_str(),
                &startup, &process)) {
            result.message = "Cannot launch CyxWiz; Win32 error " +
                std::to_string(::GetLastError());
            return result;
        }
        ::CloseHandle(process.hThread);
        ::CloseHandle(process.hProcess);
#else
        const pid_t intermediate = ::fork();
        if (intermediate < 0) {
            result.message = "Cannot fork the installed CyxWiz launcher";
            return result;
        }
        if (intermediate == 0) {
            const pid_t child = ::fork();
            if (child == 0) {
                ::setsid();
                ::chdir(launcher.parent_path().c_str());
                if (installer_mode) {
                    ::execl(launcher.c_str(), launcher.c_str(),
                            "--installer", nullptr);
                } else {
                    ::execl(launcher.c_str(), launcher.c_str(), nullptr);
                }
                ::_exit(127);
            }
            ::_exit(child < 0 ? 127 : 0);
        }
        int status = 0;
        if (::waitpid(intermediate, &status, 0) != intermediate ||
            !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            result.message = "Cannot launch the installed CyxWiz process";
            return result;
        }
#endif
        result.succeeded = true;
        result.activated = true;
        result.message = installer_mode
            ? "Installed CyxWiz manager opened"
            : "CyxWiz launched";
        return result;
    }
    std::filesystem::path CurrentCatalogRoot() const {
        if (package_source_ == InstallerPackageSource::OfflineSibling) {
            return metadata_root_;
        }
        std::error_code error;
        const bool has_active_runtime = std::filesystem::is_regular_file(
            runtime_root_ / "active-runtime.json", error);
        if (error || (!has_active_runtime && !online_refresh_succeeded_)) {
            return metadata_root_;
        }
        const bool has_cached_catalog = std::filesystem::is_regular_file(
            runtime::BackendPackCurrentCatalogPath(runtime_root_), error);
        if (!error && has_cached_catalog && std::filesystem::is_regular_file(
                runtime_root_ / "trust" / "trusted-keys.json", error) &&
            !error) {
            return runtime_root_;
        }
        return metadata_root_;
    }

    std::filesystem::path runtime_root_;
    std::filesystem::path metadata_root_;
    std::filesystem::path executable_directory_;
    std::string catalog_url_;
    InstallerPackageSource package_source_ =
        InstallerPackageSource::CatalogHttps;
    bool online_refresh_succeeded_ = false;
    bool elevate_ = false;
    std::mutex helper_mutex_;
    bool plan_running_ = false;
    bool cancellation_requested_ = false;
    std::string active_helper_token_;
};

}  // namespace

std::unique_ptr<BackendPackInstallerPlatform>
CreateBackendPackInstallerPlatform(
    std::filesystem::path runtime_root,
    std::filesystem::path metadata_root,
    std::filesystem::path executable_directory,
    CyxWizInstallScope scope,
    std::string catalog_url,
    InstallerPackageSource package_source) {
    return std::make_unique<DesktopInstallerPlatform>(
        std::move(runtime_root), std::move(metadata_root),
        std::move(executable_directory), scope, std::move(catalog_url),
        package_source);
}

std::filesystem::path DefaultCyxWizInstallRoot(
    CyxWizInstallScope scope) {
#ifdef _WIN32
    const wchar_t* variable = scope == CyxWizInstallScope::AllUsers
        ? _wgetenv(L"ProgramFiles") : _wgetenv(L"LOCALAPPDATA");
    if (variable && *variable) {
        return std::filesystem::path(variable) / "CyxWiz";
    }
    return scope == CyxWizInstallScope::AllUsers
        ? std::filesystem::path("C:\\Program Files\\CyxWiz")
        : std::filesystem::temp_directory_path() / "CyxWiz";
#elif defined(__APPLE__)
    if (scope == CyxWizInstallScope::AllUsers) {
        return "/Applications/CyxWiz";
    }
    const char* home = std::getenv("HOME");
    return home && *home
        ? std::filesystem::path(home) / "Applications" / "CyxWiz"
        : std::filesystem::temp_directory_path() / "CyxWiz";
#else
    if (scope == CyxWizInstallScope::AllUsers) return "/opt/cyxwiz";
    const char* data_home = std::getenv("XDG_DATA_HOME");
    if (data_home && *data_home) {
        return std::filesystem::path(data_home) / "cyxwiz";
    }
    const char* home = std::getenv("HOME");
    return home && *home
        ? std::filesystem::path(home) / ".local" / "share" / "cyxwiz"
        : std::filesystem::temp_directory_path() / "cyxwiz";
#endif
}

}  // namespace cyxwiz::installer
