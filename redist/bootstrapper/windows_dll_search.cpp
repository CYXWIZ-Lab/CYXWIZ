#include "windows_dll_search.h"

#include "runtime_layout.h"

#include <cstdlib>
#include <mutex>
#include <vector>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace cyxwiz::runtime {
namespace {

std::mutex g_search_mutex;
std::vector<DLL_DIRECTORY_COOKIE> g_directory_cookies;

}  // namespace

bool ConfigureActiveRuntimeDllSearchFromEnvironment(std::string& error) {
    const wchar_t* configured_root = _wgetenv(L"CYXWIZ_ACTIVE_RUNTIME_ROOT");
    if (configured_root == nullptr || configured_root[0] == L'\0') {
        return true;
    }

    ActiveRuntime runtime;
    if (!ResolveActiveRuntime(configured_root, runtime, error)) {
        AppendBootstrapDiagnostic(configured_root, "engine runtime validation failed: " + error);
        return false;
    }

    std::lock_guard<std::mutex> lock(g_search_mutex);
    if (!g_directory_cookies.empty()) {
        error = "active runtime DLL search was already configured";
        AppendBootstrapDiagnostic(runtime.runtime_root, "engine DLL setup failed: " + error);
        return false;
    }
    if (!::SetDefaultDllDirectories(
            LOAD_LIBRARY_SEARCH_APPLICATION_DIR |
            LOAD_LIBRARY_SEARCH_SYSTEM32 |
            LOAD_LIBRARY_SEARCH_USER_DIRS)) {
        error = "SetDefaultDllDirectories failed with Win32 error " +
                std::to_string(::GetLastError());
        AppendBootstrapDiagnostic(runtime.runtime_root, "engine DLL setup failed: " + error);
        return false;
    }
    for (const auto& directory : runtime.dll_directories) {
        auto cookie = ::AddDllDirectory(directory.c_str());
        if (cookie == nullptr) {
            error = "AddDllDirectory failed for " + directory.string() +
                    " with Win32 error " + std::to_string(::GetLastError());
            for (auto existing : g_directory_cookies) {
                ::RemoveDllDirectory(existing);
            }
            g_directory_cookies.clear();
            AppendBootstrapDiagnostic(runtime.runtime_root, "engine DLL setup failed: " + error);
            return false;
        }
        g_directory_cookies.push_back(cookie);
    }
    AppendBootstrapDiagnostic(
        runtime.runtime_root,
        "engine DLL search configured for runtime_set=" + runtime.runtime_set_id +
            " generation=" + std::to_string(runtime.generation));
    return true;
}

}  // namespace cyxwiz::runtime
