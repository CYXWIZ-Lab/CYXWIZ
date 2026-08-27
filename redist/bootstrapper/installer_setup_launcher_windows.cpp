#include "installer_setup_launcher.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <vector>

namespace cyxwiz::runtime {

bool LaunchInstallerAndWait(
    const std::filesystem::path& installer_path,
    int& exit_code,
    std::string& error) {
    exit_code = -1;
    std::error_code filesystem_error;
    if (!installer_path.is_absolute() ||
        !std::filesystem::is_regular_file(installer_path, filesystem_error) ||
        filesystem_error) {
        error = "Verified installer entry point is missing";
        return false;
    }
    std::wstring command = L"\"" + installer_path.wstring() + L"\"";
    std::vector<wchar_t> writable(command.begin(), command.end());
    writable.push_back(L'\0');
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    const auto working_directory = installer_path.parent_path().wstring();
    if (!CreateProcessW(
            installer_path.c_str(), writable.data(), nullptr, nullptr, FALSE,
            0, nullptr, working_directory.c_str(), &startup, &process)) {
        error = "Cannot launch verified installer (Windows error " +
            std::to_string(GetLastError()) + ")";
        return false;
    }
    CloseHandle(process.hThread);
    const DWORD wait = WaitForSingleObject(process.hProcess, INFINITE);
    DWORD child_exit = 0;
    const bool complete = wait == WAIT_OBJECT_0 &&
        GetExitCodeProcess(process.hProcess, &child_exit);
    CloseHandle(process.hProcess);
    if (!complete) {
        error = "Cannot collect verified installer exit status";
        return false;
    }
    exit_code = static_cast<int>(child_exit);
    return true;
}

}  // namespace cyxwiz::runtime
