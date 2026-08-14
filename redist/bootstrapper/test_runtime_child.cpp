#include "windows_dll_search.h"

#include <iostream>
#include <string>
#include <vector>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace {

std::wstring EnvironmentValue(const wchar_t* name) {
    const DWORD required = ::GetEnvironmentVariableW(name, nullptr, 0);
    if (required == 0) {
        return {};
    }
    std::vector<wchar_t> value(required);
    const DWORD length = ::GetEnvironmentVariableW(name, value.data(), required);
    return std::wstring(value.data(), length);
}

}  // namespace

int main() {
    if (EnvironmentValue(L"CYXWIZ_ACTIVE_RUNTIME_ROOT").empty()) {
        std::cerr << "child did not receive the active runtime root\n";
        return 1;
    }
    if (EnvironmentValue(L"CYXWIZ_RUNTIME_SET_ID") != L"set-v1" ||
        EnvironmentValue(L"CYXWIZ_RUNTIME_GENERATION") != L"1" ||
        EnvironmentValue(L"CYXWIZ_BASE_PACK_ID") != L"base-v1" ||
        EnvironmentValue(L"CYXWIZ_RUNTIME_PACK_OPENCL") != L"opencl-v1" ||
        !EnvironmentValue(L"CYXWIZ_RUNTIME_PACK_CUDA").empty() ||
        !EnvironmentValue(L"CYXWIZ_RUNTIME_PACK_ONEAPI").empty()) {
        std::cerr << "child did not receive the exact active runtime identity\n";
        return 5;
    }
    if (!EnvironmentValue(L"AF_PATH").empty() ||
        !EnvironmentValue(L"AF_PLUGIN_PATH").empty() ||
        !EnvironmentValue(L"AF_BUILD_PATH").empty() ||
        !EnvironmentValue(L"AF_BUILD_LIB_CUSTOM_PATH").empty() ||
        !EnvironmentValue(L"PYTHONPATH").empty()) {
        std::cerr << "child received an inherited developer runtime override\n";
        return 2;
    }
    if (EnvironmentValue(L"PATH").find(L"cyxwiz-untrusted-marker") != std::wstring::npos) {
        std::cerr << "child received the inherited developer PATH\n";
        return 3;
    }
    std::string error;
    if (!cyxwiz::runtime::ConfigureActiveRuntimeDllSearchFromEnvironment(error)) {
        std::cerr << "child could not configure restricted DLL search: " << error << '\n';
        return 4;
    }
    return 0;
}
