#include "file_browser_launch.h"

#include <filesystem>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#elif defined(__APPLE__) || defined(__linux__)
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace cyxwiz {

namespace fs = std::filesystem;

bool LaunchAssetInFileBrowser(const std::string& absolute_path, bool is_directory) {
    if (absolute_path.empty()) {
        return false;
    }

#ifdef _WIN32
    (void)is_directory;
    const std::string params = "/select,\"" + absolute_path + "\"";
    const HINSTANCE result = ShellExecuteA(
        nullptr, "open", "explorer.exe", params.c_str(), nullptr, SW_SHOWNORMAL);
    return reinterpret_cast<INT_PTR>(result) > 32;
#elif defined(__APPLE__)
    const pid_t pid = fork();
    if (pid == 0) {
        if (is_directory) {
            execlp("open", "open", absolute_path.c_str(), static_cast<char*>(nullptr));
        } else {
            execlp("open", "open", "-R", absolute_path.c_str(), static_cast<char*>(nullptr));
        }
        _exit(127);
    }
    int status = 0;
    return pid > 0 && waitpid(pid, &status, 0) == pid &&
           WIFEXITED(status) && WEXITSTATUS(status) == 0;
#else
    const std::string directory = is_directory
        ? absolute_path
        : fs::path(absolute_path).parent_path().string();
    if (directory.empty()) {
        return false;
    }
    const pid_t pid = fork();
    if (pid == 0) {
        execlp("xdg-open", "xdg-open", directory.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }
    int status = 0;
    return pid > 0 && waitpid(pid, &status, 0) == pid &&
           WIFEXITED(status) && WEXITSTATUS(status) == 0;
#endif
}

}  // namespace cyxwiz
