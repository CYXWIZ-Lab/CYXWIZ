#include "arrayfire_backend_discovery_isolation.h"

#include <cstdlib>
#include <filesystem>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace cyxwiz {
namespace {

#ifdef _WIN32
bool ReadEnvironmentValue(
    const wchar_t* name,
    std::wstring& value,
    bool& present,
    std::string& error) {
    ::SetLastError(ERROR_SUCCESS);
    const DWORD required = ::GetEnvironmentVariableW(name, nullptr, 0);
    if (required == 0) {
        const DWORD code = ::GetLastError();
        if (code == ERROR_ENVVAR_NOT_FOUND || code == ERROR_SUCCESS) {
            value.clear();
            present = false;
            return true;
        }
        error = "Cannot inspect the process environment; Win32 error " +
            std::to_string(code);
        return false;
    }
    std::wstring buffer(required, L'\0');
    const DWORD length =
        ::GetEnvironmentVariableW(name, buffer.data(), required);
    if (length == 0 || length >= required) {
        error = "Cannot read the process environment; Win32 error " +
            std::to_string(::GetLastError());
        return false;
    }
    buffer.resize(length);
    value = std::move(buffer);
    present = true;
    return true;
}
#endif

}  // namespace

ScopedArrayFireBackendDiscoveryIsolation::
    ~ScopedArrayFireBackendDiscoveryIsolation() {
#ifdef _WIN32
    if (applied_) {
        ::SetEnvironmentVariableW(
            L"ProgramFiles",
            previous_program_files_present_
                ? previous_program_files_.c_str()
                : nullptr);
    }
#endif
}

bool ScopedArrayFireBackendDiscoveryIsolation::Apply(std::string& error) {
    error.clear();
#ifdef _WIN32
    if (applied_) return true;
    const wchar_t* configured_root =
        ::_wgetenv(L"CYXWIZ_ACTIVE_RUNTIME_ROOT");
    if (!configured_root || *configured_root == L'\0') return true;

    const std::filesystem::path runtime_root(configured_root);
    if (!runtime_root.is_absolute()) {
        error = "The active runtime root must be absolute before ArrayFire "
                "backend discovery";
        return false;
    }
    std::error_code filesystem_error;
    if (!std::filesystem::is_directory(runtime_root, filesystem_error) ||
        filesystem_error) {
        error = "The active runtime root is unavailable before ArrayFire "
                "backend discovery";
        return false;
    }
    if (!ReadEnvironmentValue(
            L"ProgramFiles", previous_program_files_,
            previous_program_files_present_, error)) {
        return false;
    }
    const auto isolated_fallback =
        runtime_root / ".cyxwiz-isolated-program-files";
    if (!::SetEnvironmentVariableW(
            L"ProgramFiles", isolated_fallback.c_str())) {
        error = "Cannot isolate ArrayFire backend discovery; Win32 error " +
            std::to_string(::GetLastError());
        return false;
    }
    applied_ = true;
#endif
    return true;
}

}  // namespace cyxwiz
