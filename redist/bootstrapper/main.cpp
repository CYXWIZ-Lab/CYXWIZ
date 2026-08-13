#include "runtime_layout.h"

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

int Fail(const std::filesystem::path& runtime_root, const std::string& message) {
    cyxwiz::runtime::AppendBootstrapDiagnostic(runtime_root, "launch failed: " + message);
    std::cerr << "CyxWiz launch failed: " << message << '\n';
    std::cerr << "Diagnostic log: " << (runtime_root / "bootstrapper.log").string() << '\n';
    return 78;
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

    cyxwiz::runtime::ActiveRuntime runtime;
    std::string error;
    if (!cyxwiz::runtime::ResolveActiveRuntime(runtime_root, runtime, error)) {
        return Fail(runtime_root, error);
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
            L"CYXWIZ_ACTIVE_RUNTIME_ROOT", runtime.runtime_root.c_str())) {
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

    std::wstring command_line = QuoteArgument(runtime.engine_executable.native());
    for (int index = first_forwarded_argument; index < argc; ++index) {
        command_line.push_back(L' ');
        command_line += QuoteArgument(argv[index]);
    }
    std::vector<wchar_t> mutable_command(command_line.begin(), command_line.end());
    mutable_command.push_back(L'\0');

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    const BOOL created = ::CreateProcessW(
        runtime.engine_executable.c_str(), mutable_command.data(), nullptr, nullptr,
        FALSE, 0, nullptr, runtime.base_directory.c_str(), &startup, &process);
    if (!created) {
        return Fail(runtime.runtime_root,
                    "CreateProcessW failed for the active base; Win32 error " +
                        std::to_string(::GetLastError()));
    }
    ::CloseHandle(process.hThread);
    cyxwiz::runtime::AppendBootstrapDiagnostic(
        runtime.runtime_root,
        "launched runtime_set=" + runtime.runtime_set_id +
            " generation=" + std::to_string(runtime.generation) +
            " base=" + runtime.base_pack_id);

    const DWORD wait_result = ::WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exit_code = 78;
    if (wait_result == WAIT_OBJECT_0) {
        ::GetExitCodeProcess(process.hProcess, &exit_code);
    } else {
        cyxwiz::runtime::AppendBootstrapDiagnostic(
            runtime.runtime_root,
            "wait for Engine failed with Win32 error " + std::to_string(::GetLastError()));
    }
    ::CloseHandle(process.hProcess);
    return static_cast<int>(exit_code);
}
