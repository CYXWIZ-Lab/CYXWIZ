#include "ui_platform.h"

#include <filesystem>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#else
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace cyxwiz::ui {

namespace {

#ifndef _WIN32
bool Run(const char* program, const char* arg1, const char* arg2) {
    const pid_t pid = fork();
    if (pid == 0) {
        if (arg2) execlp(program, program, arg1, arg2, static_cast<char*>(nullptr));
        else execlp(program, program, arg1, static_cast<char*>(nullptr));
        _exit(127);
    }
    int status = 0;
    return pid > 0 && waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0;
}
#endif

}  // namespace

bool OpenUrl(const std::string& url) {
    if (url.empty()) return false;
#ifdef _WIN32
    const HINSTANCE result = ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    return reinterpret_cast<INT_PTR>(result) > 32;
#elif defined(__APPLE__)
    return Run("open", url.c_str(), nullptr);
#else
    return Run("xdg-open", url.c_str(), nullptr);
#endif
}

bool ShowInFileManager(const std::string& path) {
    if (path.empty()) return false;
    std::error_code ec;
    const std::filesystem::path p = std::filesystem::absolute(path, ec);
    const std::string absolute = ec ? path : p.string();
    const bool is_directory = std::filesystem::is_directory(p, ec);
#ifdef _WIN32
    const std::string params = is_directory ? "\"" + absolute + "\"" : "/select,\"" + absolute + "\"";
    const HINSTANCE result = ShellExecuteA(nullptr, "open", "explorer.exe", params.c_str(), nullptr, SW_SHOWNORMAL);
    return reinterpret_cast<INT_PTR>(result) > 32;
#elif defined(__APPLE__)
    return is_directory ? Run("open", absolute.c_str(), nullptr) : Run("open", "-R", absolute.c_str());
#else
    const std::string folder = is_directory ? absolute : p.parent_path().string();
    return Run("xdg-open", folder.c_str(), nullptr);
#endif
}

}  // namespace cyxwiz::ui
