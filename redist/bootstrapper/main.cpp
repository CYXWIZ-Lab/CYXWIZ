#include "runtime_layout.h"
#include "backend_pack_maintenance_request.h"
#include "backend_pack_platform.h"
#include "product_removal_handoff.h"
#include "product_removal_protocol.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace {

std::filesystem::path ExecutableDirectory() {
    std::vector<wchar_t> buffer(32768);
    const DWORD length = ::GetModuleFileNameW(
        nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0 || length >= buffer.size()) {
        return {};
    }
    return std::filesystem::path(std::wstring(buffer.data(), length)).parent_path();
}

std::wstring QuoteArgument(const std::wstring& argument) {
    if (!argument.empty() &&
        argument.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
        return argument;
    }
    std::wstring quoted = L"\"";
    std::size_t backslashes = 0;
    for (const wchar_t character : argument) {
        if (character == L'\\') {
            ++backslashes;
        } else if (character == L'\"') {
            quoted.append(backslashes * 2 + 1, L'\\');
            quoted.push_back(L'\"');
            backslashes = 0;
        } else {
            quoted.append(backslashes, L'\\');
            backslashes = 0;
            quoted.push_back(character);
        }
    }
    quoted.append(backslashes * 2, L'\\');
    quoted.push_back(L'\"');
    return quoted;
}

std::wstring RestrictedPath(
    const cyxwiz::runtime::ActiveRuntime& runtime) {
    std::wstring value;
    for (const auto& directory : runtime.dll_directories) {
        if (!value.empty()) {
            value.push_back(L';');
        }
        value += directory.native();
    }
    std::vector<wchar_t> system_directory(32768);
    const UINT length = ::GetSystemDirectoryW(
        system_directory.data(), static_cast<UINT>(system_directory.size()));
    if (length > 0 && length < system_directory.size()) {
        if (!value.empty()) {
            value.push_back(L';');
        }
        value.append(system_directory.data(), length);
    }
    return value;
}

bool IsValidHandle(HANDLE handle) {
    return handle != nullptr && handle != INVALID_HANDLE_VALUE;
}

class ChildStandardHandles {
public:
    ChildStandardHandles() = default;
    ChildStandardHandles(const ChildStandardHandles&) = delete;
    ChildStandardHandles& operator=(const ChildStandardHandles&) = delete;

    ~ChildStandardHandles() {
        for (const HANDLE handle : {input_, output_, error_}) {
            if (IsValidHandle(handle)) {
                ::CloseHandle(handle);
            }
        }
    }

    bool Configure(STARTUPINFOW& startup, std::string& error) {
        if (!DuplicateOrOpenNull(
                STD_INPUT_HANDLE, GENERIC_READ, input_, error) ||
            !DuplicateOrOpenNull(
                STD_OUTPUT_HANDLE, GENERIC_WRITE, output_, error) ||
            !DuplicateOrOpenNull(
                STD_ERROR_HANDLE, GENERIC_WRITE, error_, error)) {
            return false;
        }
        startup.dwFlags |= STARTF_USESTDHANDLES;
        startup.hStdInput = input_;
        startup.hStdOutput = output_;
        startup.hStdError = error_;
        return true;
    }

private:
    static bool DuplicateOrOpenNull(
        DWORD standard_handle,
        DWORD fallback_access,
        HANDLE& output,
        std::string& error) {
        const HANDLE source = ::GetStdHandle(standard_handle);
        if (IsValidHandle(source)) {
            if (::DuplicateHandle(
                    ::GetCurrentProcess(), source, ::GetCurrentProcess(),
                    &output, 0, TRUE, DUPLICATE_SAME_ACCESS)) {
                return true;
            }
        } else {
            SECURITY_ATTRIBUTES security{
                sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
            output = ::CreateFileW(
                L"NUL", fallback_access, FILE_SHARE_READ | FILE_SHARE_WRITE,
                &security, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (IsValidHandle(output)) {
                return true;
            }
        }
        error = "cannot prepare inherited standard handles; Win32 error " +
            std::to_string(::GetLastError());
        return false;
    }

    HANDLE input_ = nullptr;
    HANDLE output_ = nullptr;
    HANDLE error_ = nullptr;
};

int Fail(const std::filesystem::path& runtime_root, const std::string& message) {
    cyxwiz::runtime::AppendBootstrapDiagnostic(runtime_root, "launch failed: " + message);
    std::cerr << "CyxWiz launch failed: " << message << '\n';
    std::cerr << "Diagnostic log: " << (runtime_root / "bootstrapper.log").string() << '\n';
    return 78;
}

std::wstring WidenIdentifier(const std::string& value) {
    return std::wstring(value.begin(), value.end());
}

bool SetRuntimeIdentityEnvironment(
    const cyxwiz::runtime::ActiveRuntime& runtime) {
    const std::wstring runtime_set = WidenIdentifier(runtime.runtime_set_id);
    const std::wstring generation = std::to_wstring(runtime.generation);
    const std::wstring base_pack = WidenIdentifier(runtime.base_pack_id);
    if (!::SetEnvironmentVariableW(
            L"CYXWIZ_RUNTIME_SET_ID", runtime_set.c_str()) ||
        !::SetEnvironmentVariableW(
            L"CYXWIZ_RUNTIME_GENERATION", generation.c_str()) ||
        !::SetEnvironmentVariableW(
            L"CYXWIZ_BASE_PACK_ID", base_pack.c_str())) {
        return false;
    }
    for (const wchar_t* name : {
             L"CYXWIZ_RUNTIME_PACK_CUDA",
             L"CYXWIZ_RUNTIME_PACK_OPENCL",
             L"CYXWIZ_RUNTIME_PACK_ONEAPI"}) {
        ::SetEnvironmentVariableW(name, nullptr);
    }
    for (const auto& pack : runtime.packs) {
        const wchar_t* name = pack.backend == "cuda"
            ? L"CYXWIZ_RUNTIME_PACK_CUDA"
            : pack.backend == "opencl"
                ? L"CYXWIZ_RUNTIME_PACK_OPENCL"
                : L"CYXWIZ_RUNTIME_PACK_ONEAPI";
        const std::wstring pack_id = WidenIdentifier(pack.pack_id);
        if (!::SetEnvironmentVariableW(name, pack_id.c_str())) return false;
    }
    return true;
}

bool RunBackendPackRepair(
    const std::filesystem::path& active_base_directory,
    const std::filesystem::path& runtime_root,
    const cyxwiz::runtime::BackendPackMaintenanceRequest& request,
    std::string& message) {
    const auto helper =
        active_base_directory /
        cyxwiz::runtime::CurrentBackendPackInstallerExecutableName();
    if (!std::filesystem::is_regular_file(helper)) {
        message = "Backend-pack repair helper is missing";
        return false;
    }
    std::wstring command = QuoteArgument(helper.native()) +
        L" --runtime-root " + QuoteArgument(runtime_root.native()) +
        L" --pack-id " + QuoteArgument(WidenIdentifier(request.pack_id)) +
        L" --repair";
    std::vector<wchar_t> mutable_command(command.begin(), command.end());
    mutable_command.push_back(L'\0');
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!::CreateProcessW(
            helper.c_str(), mutable_command.data(), nullptr, nullptr,
            FALSE, 0, nullptr, active_base_directory.c_str(),
            &startup, &process)) {
        message = "Cannot launch backend-pack repair helper; Win32 error " +
            std::to_string(::GetLastError());
        return false;
    }
    ::CloseHandle(process.hThread);
    const DWORD wait = ::WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exit_code = 1;
    if (wait == WAIT_OBJECT_0) {
        ::GetExitCodeProcess(process.hProcess, &exit_code);
    }
    ::CloseHandle(process.hProcess);
    if (wait != WAIT_OBJECT_0) {
        message = "Waiting for backend-pack repair helper failed";
        return false;
    }
    message = exit_code == 0
        ? "Backend pack repaired, locally qualified, and reactivated"
        : "Backend-pack repair helper failed with exit code " +
              std::to_string(exit_code);
    return exit_code == 0;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    const auto executable_directory = ExecutableDirectory();
    if (executable_directory.empty()) {
        std::cerr << "CyxWiz launch failed: cannot resolve bootstrapper location\n";
        return 78;
    }

    std::filesystem::path runtime_root = executable_directory / "runtime";
    int first_forwarded_argument = 1;
    if (argc >= 3 && std::wstring_view(argv[1]) == L"--runtime-root") {
        runtime_root = argv[2];
        first_forwarded_argument = 3;
    }
    bool installer_mode = false;
    if (argc > first_forwarded_argument &&
        std::wstring_view(argv[first_forwarded_argument]) == L"--installer") {
        installer_mode = true;
        ++first_forwarded_argument;
    }

    cyxwiz::runtime::ActiveRuntime runtime;
    std::string error;
    if (!cyxwiz::runtime::ResolveActiveRuntime(runtime_root, runtime, error)) {
        return Fail(runtime_root, error);
    }
    cyxwiz::runtime::ActiveRuntimeState launched_runtime;
    launched_runtime.runtime_set_id = runtime.runtime_set_id;
    launched_runtime.generation = runtime.generation;
    launched_runtime.base_pack_id = runtime.base_pack_id;
    for (const auto& pack : runtime.packs) {
        launched_runtime.packs.push_back({pack.backend, pack.pack_id});
    }
    if (!::SetDefaultDllDirectories(
            LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32)) {
        return Fail(runtime.runtime_root,
                    "cannot restrict bootstrapper DLL search; Win32 error " +
                        std::to_string(::GetLastError()));
    }

    const auto restricted_path = RestrictedPath(runtime);
    if (!::SetEnvironmentVariableW(L"PATH", restricted_path.c_str()) ||
        !::SetEnvironmentVariableW(
            L"CYXWIZ_ACTIVE_RUNTIME_ROOT", runtime.runtime_root.c_str()) ||
        !SetRuntimeIdentityEnvironment(runtime)) {
        return Fail(runtime.runtime_root,
                    "cannot prepare child runtime environment; Win32 error " +
                        std::to_string(::GetLastError()));
    }
    for (const wchar_t* variable : {
             L"AF_PATH", L"AF_PLUGIN_PATH", L"CYXWIZ_ARRAYFIRE_DIR",
             L"AF_BUILD_PATH", L"AF_BUILD_LIB_CUSTOM_PATH",
             L"PYTHONHOME", L"PYTHONPATH"}) {
        if (!::SetEnvironmentVariableW(variable, nullptr) &&
            ::GetLastError() != ERROR_ENVVAR_NOT_FOUND) {
            return Fail(runtime.runtime_root,
                        "cannot remove inherited runtime override; Win32 error " +
                            std::to_string(::GetLastError()));
        }
    }

    const auto launched_executable = installer_mode
        ? runtime.base_directory /
              cyxwiz::runtime::CurrentInstallerManagerExecutableName()
        : runtime.engine_executable;
    if (!std::filesystem::is_regular_file(launched_executable)) {
        return Fail(
            runtime.runtime_root,
            installer_mode
                ? "active base does not contain the CyxWiz Installer"
                : "active base does not contain the CyxWiz Engine");
    }
    std::wstring command_line = QuoteArgument(launched_executable.native());
    if (installer_mode) {
        command_line += L" --runtime-root ";
        command_line += QuoteArgument(runtime.runtime_root.native());
        command_line += L" --product-removal-host";
    }
    for (int index = first_forwarded_argument; index < argc; ++index) {
        command_line.push_back(L' ');
        command_line += QuoteArgument(argv[index]);
    }
    std::vector<wchar_t> mutable_command(command_line.begin(), command_line.end());
    mutable_command.push_back(L'\0');

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    ChildStandardHandles standard_handles;
    if (!standard_handles.Configure(startup, error)) {
        return Fail(runtime.runtime_root, error);
    }
    PROCESS_INFORMATION process{};
    // Preserve the caller's standard streams so command-line verification and
    // diagnostics emitted by the Engine remain observable through this stable
    // launcher. The bootstrapper owns no other inheritable handles here.
    const BOOL created = ::CreateProcessW(
        launched_executable.c_str(), mutable_command.data(), nullptr, nullptr,
        TRUE, 0, nullptr, runtime.base_directory.c_str(), &startup, &process);
    if (!created) {
        return Fail(runtime.runtime_root,
                    "CreateProcessW failed for the active base; Win32 error " +
                        std::to_string(::GetLastError()));
    }
    ::CloseHandle(process.hThread);
    cyxwiz::runtime::AppendBootstrapDiagnostic(
        runtime.runtime_root,
        std::string(installer_mode ? "launched installer runtime_set="
                                   : "launched runtime_set=") +
            runtime.runtime_set_id +
            " generation=" + std::to_string(runtime.generation) +
            " base=" + runtime.base_pack_id);

    const DWORD wait_result = ::WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exit_code = 78;
    if (wait_result == WAIT_OBJECT_0) {
        ::GetExitCodeProcess(process.hProcess, &exit_code);
    } else {
        cyxwiz::runtime::AppendBootstrapDiagnostic(
            runtime.runtime_root,
            "wait for child failed with Win32 error " +
                std::to_string(::GetLastError()));
    }
    ::CloseHandle(process.hProcess);
    if (wait_result == WAIT_OBJECT_0 && installer_mode &&
        exit_code == static_cast<DWORD>(
            cyxwiz::runtime::kProductRemovalRequestedExitCode)) {
        auto handoff = cyxwiz::runtime::SchedulePendingProductRemoval(
            executable_directory, error);
        if (handoff.status != cyxwiz::runtime::
                ProductRemovalHandoffStatus::Scheduled) {
            return Fail(
                runtime.runtime_root,
                "cannot schedule queued product removal: " + error);
        }
        cyxwiz::runtime::AppendBootstrapDiagnostic(
            runtime.runtime_root,
            "product removal queued; detached finalizer is waiting for exit");
        handoff.parent_lifetime.PreserveUntilProcessExit();
        return 0;
    }
    if (wait_result == WAIT_OBJECT_0 && !installer_mode) {
        const auto maintenance =
            cyxwiz::runtime::ApplyPendingBackendPackMaintenance(
                runtime.runtime_root, launched_runtime,
                [&](const auto& request, std::string& message) {
                    return RunBackendPackRepair(
                        runtime.base_directory, runtime.runtime_root,
                        request, message);
                });
        if (maintenance.status != cyxwiz::runtime::
                BackendPackMaintenanceApplyStatus::NoRequest) {
            cyxwiz::runtime::AppendBootstrapDiagnostic(
                runtime.runtime_root,
                std::string("backend maintenance ") +
                    cyxwiz::runtime::BackendPackMaintenanceApplyStatusName(
                        maintenance.status) +
                    ": " + maintenance.message);
        }
    }
    return static_cast<int>(exit_code);
}
