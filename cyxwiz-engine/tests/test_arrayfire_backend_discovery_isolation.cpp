#include "core/arrayfire_backend_discovery_isolation.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

#ifdef _WIN32
std::wstring EnvironmentValue(const wchar_t* name) {
    const DWORD required = ::GetEnvironmentVariableW(name, nullptr, 0);
    if (required == 0) return {};
    std::wstring value(required, L'\0');
    const DWORD length =
        ::GetEnvironmentVariableW(name, value.data(), required);
    Check(length > 0 && length < required, "cannot read test environment");
    value.resize(length);
    return value;
}
#endif

}  // namespace

int main() {
#ifdef _WIN32
    const std::filesystem::path runtime_root =
        std::filesystem::temp_directory_path() /
        ("cyxwiz-arrayfire-discovery-isolation-test-" +
         std::to_string(::GetCurrentProcessId()));
    std::error_code filesystem_error;
    std::filesystem::create_directories(runtime_root, filesystem_error);
    Check(!filesystem_error, "cannot create test runtime root");

    const std::wstring previous_runtime =
        EnvironmentValue(L"CYXWIZ_ACTIVE_RUNTIME_ROOT");
    const std::wstring previous_program_files =
        EnvironmentValue(L"ProgramFiles");
    const std::wstring marker = L"C:\\cyxwiz-program-files-marker";
    Check(::SetEnvironmentVariableW(
              L"CYXWIZ_ACTIVE_RUNTIME_ROOT", runtime_root.c_str()) != FALSE,
          "cannot set test runtime identity");
    Check(::SetEnvironmentVariableW(L"ProgramFiles", marker.c_str()) != FALSE,
          "cannot set test Program Files marker");
    {
        cyxwiz::ScopedArrayFireBackendDiscoveryIsolation isolation;
        std::string error;
        Check(isolation.Apply(error), error);
        Check(EnvironmentValue(L"ProgramFiles") ==
                  (runtime_root / ".cyxwiz-isolated-program-files").native(),
              "packaged discovery must redirect the global fallback");
    }
    Check(EnvironmentValue(L"ProgramFiles") == marker,
          "discovery isolation must restore Program Files");

    ::SetEnvironmentVariableW(
        L"CYXWIZ_ACTIVE_RUNTIME_ROOT",
        previous_runtime.empty() ? nullptr : previous_runtime.c_str());
    ::SetEnvironmentVariableW(
        L"ProgramFiles",
        previous_program_files.empty() ? nullptr : previous_program_files.c_str());
    std::filesystem::remove_all(runtime_root, filesystem_error);
    Check(!filesystem_error, "cannot remove test runtime root");
#else
    cyxwiz::ScopedArrayFireBackendDiscoveryIsolation isolation;
    std::string error;
    Check(isolation.Apply(error), error);
#endif
    std::cout << "ArrayFire backend discovery isolation test passed\n";
    return 0;
}
