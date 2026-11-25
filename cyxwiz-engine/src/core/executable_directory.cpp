// The folder of the running executable: what ships beside the Engine
// (resources/, python_tools/, plugins/) is read from here, never from the
// working folder (an installed Engine works in the user's state folder, and a
// Python script run moves the working folder to the project).
#include "core/user_folders.h"

#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#include <climits>
#else
#include <unistd.h>
#include <climits>
#endif

namespace cyxwiz::core {

std::filesystem::path ExecutableDirectory() {
#ifdef _WIN32
    std::vector<wchar_t> buffer(32768);
    const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0 || length >= buffer.size()) return {};
    return std::filesystem::path(std::wstring(buffer.data(), length)).parent_path();
#elif defined(__APPLE__)
    char path[PATH_MAX];
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) != 0) return {};
    std::error_code ec;
    const auto canonical = std::filesystem::weakly_canonical(path, ec);
    return (ec ? std::filesystem::path(path) : canonical).parent_path();
#else
    char path[PATH_MAX];
    const ssize_t count = readlink("/proc/self/exe", path, sizeof(path));
    if (count <= 0) return {};
    return std::filesystem::path(std::string(path, static_cast<std::size_t>(count))).parent_path();
#endif
}

}  // namespace cyxwiz::core
