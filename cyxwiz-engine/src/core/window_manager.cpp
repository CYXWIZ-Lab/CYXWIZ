#include "window_manager.h"
#include <spdlog/spdlog.h>
#include <filesystem>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#include <processthreadsapi.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
// Define PATH_MAX if not available
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif
#endif

namespace cyxwiz {
namespace core {

bool WindowManager::LaunchWindow(const std::string& project_path) {
    if (project_path.empty()) {
        spdlog::error("WindowManager::LaunchWindow - project_path is empty");
        return false;
    }

    // Validate that the project path exists
    if (!std::filesystem::exists(project_path)) {
        spdlog::error("WindowManager::LaunchWindow - project does not exist: {}", project_path);
        return false;
    }

    // Get the executable path
    std::string exe_path = GetExecutablePath();
    if (exe_path.empty()) {
        spdlog::error("WindowManager::LaunchWindow - failed to get executable path");
        return false;
    }

    // Build command-line arguments
    std::vector<std::string> args;
    args.push_back(exe_path);  // argv[0] is the executable name
    args.push_back("--project");
    args.push_back(project_path);

    // Launch the new process
    bool success = LaunchProcess(exe_path, args);
    if (success) {
        spdlog::info("WindowManager::LaunchWindow - launched new window with project: {}", project_path);
    } else {
        spdlog::error("WindowManager::LaunchWindow - failed to launch new window");
    }

    return success;
}

bool WindowManager::LaunchWindowWithDialog() {
    // Get the executable path
    std::string exe_path = GetExecutablePath();
    if (exe_path.empty()) {
        spdlog::error("WindowManager::LaunchWindowWithDialog - failed to get executable path");
        return false;
    }

    // Build command-line arguments (no --project, will show dialog)
    std::vector<std::string> args;
    args.push_back(exe_path);  // argv[0] is the executable name

    // Launch the new process
    bool success = LaunchProcess(exe_path, args);
    if (success) {
        spdlog::info("WindowManager::LaunchWindowWithDialog - launched new window with project selection dialog");
    } else {
        spdlog::error("WindowManager::LaunchWindowWithDialog - failed to launch new window");
    }

    return success;
}

bool WindowManager::RestartEngine(const std::string& project_path) {
    spdlog::info("WindowManager::RestartEngine - restarting engine with project: {}", project_path);

    // Restarting is just launching a new instance with the same project
    // The caller is responsible for closing the current instance after this succeeds
    return LaunchWindow(project_path);
}

void WindowManager::SwitchProject(const std::string& new_project_path) {
    // TODO: This will be implemented in the Application class
    // For now, just launch a new window and let the user close the current one
    spdlog::info("WindowManager::SwitchProject - launching new window with project: {}", new_project_path);
    spdlog::warn("WindowManager::SwitchProject - automatic close not implemented, please close this window manually");

    LaunchWindow(new_project_path);
}

std::vector<std::string> WindowManager::GetRunningProjects() {
    // TODO: Implement process enumeration to find other engine instances
    // This requires platform-specific code:
    // - Windows: EnumProcesses + GetModuleFileNameEx
    // - Linux: Parse /proc/*/cmdline
    // - macOS: Use ps command or sysctl

    spdlog::warn("WindowManager::GetRunningProjects - not yet implemented");
    return {};
}

std::string WindowManager::GetExecutablePath() {
#ifdef _WIN32
    // Windows: Use GetModuleFileNameA
    char path[MAX_PATH];
    DWORD length = GetModuleFileNameA(NULL, path, MAX_PATH);
    if (length == 0 || length >= MAX_PATH) {
        spdlog::error("WindowManager::GetExecutablePath - GetModuleFileNameA failed");
        return "";
    }
    return std::string(path);

#elif defined(__linux__)
    // Linux: Read /proc/self/exe symlink
    char path[PATH_MAX];
    ssize_t length = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (length == -1) {
        spdlog::error("WindowManager::GetExecutablePath - readlink failed");
        return "";
    }
    path[length] = '\0';
    return std::string(path);

#elif defined(__APPLE__)
    // macOS: Use _NSGetExecutablePath
    char path[PATH_MAX];
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) != 0) {
        spdlog::error("WindowManager::GetExecutablePath - _NSGetExecutablePath failed");
        return "";
    }
    // Resolve symlinks and relative paths
    char resolved_path[PATH_MAX];
    if (realpath(path, resolved_path) == nullptr) {
        spdlog::error("WindowManager::GetExecutablePath - realpath failed");
        return "";
    }
    return std::string(resolved_path);

#else
    spdlog::error("WindowManager::GetExecutablePath - unsupported platform");
    return "";
#endif
}

bool WindowManager::LaunchProcess(const std::string& executable_path,
                                  const std::vector<std::string>& args) {
#ifdef _WIN32
    // Windows: Use CreateProcessA

    // Build command line string (need to quote paths with spaces)
    std::string cmdline;
    for (const auto& arg : args) {
        if (!cmdline.empty()) {
            cmdline += " ";
        }
        // Quote argument if it contains spaces
        if (arg.find(' ') != std::string::npos) {
            cmdline += "\"" + arg + "\"";
        } else {
            cmdline += arg;
        }
    }

    spdlog::debug("WindowManager::LaunchProcess - command line: {}", cmdline);

    // CreateProcessA requires a modifiable string
    std::vector<char> cmdline_buf(cmdline.begin(), cmdline.end());
    cmdline_buf.push_back('\0');

    STARTUPINFOA si = {};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi = {};

    BOOL success = CreateProcessA(
        executable_path.c_str(),  // lpApplicationName
        cmdline_buf.data(),       // lpCommandLine
        NULL,                     // lpProcessAttributes
        NULL,                     // lpThreadAttributes
        FALSE,                    // bInheritHandles
        0,                        // dwCreationFlags
        NULL,                     // lpEnvironment
        NULL,                     // lpCurrentDirectory
        &si,                      // lpStartupInfo
        &pi                       // lpProcessInformation
    );

    if (!success) {
        DWORD error = GetLastError();
        spdlog::error("WindowManager::LaunchProcess - CreateProcessA failed with error: {}", error);
        return false;
    }

    // Close process and thread handles (we don't need to wait)
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    return true;

#else
    // Linux/macOS: Use fork + execv

    pid_t pid = fork();
    if (pid == -1) {
        spdlog::error("WindowManager::LaunchProcess - fork() failed");
        return false;
    }

    if (pid == 0) {
        // Child process

        // Convert std::vector<std::string> to char* array for execv
        std::vector<char*> argv;
        for (const auto& arg : args) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        argv.push_back(nullptr);  // execv requires NULL-terminated array

        // Execute the new process
        execv(executable_path.c_str(), argv.data());

        // If execv returns, it failed
        spdlog::error("WindowManager::LaunchProcess - execv() failed");
        _exit(1);
    }

    // Parent process
    spdlog::debug("WindowManager::LaunchProcess - spawned child process with PID: {}", pid);

    // We don't wait for the child process (it runs independently)
    return true;

#endif
}

} // namespace core
} // namespace cyxwiz
