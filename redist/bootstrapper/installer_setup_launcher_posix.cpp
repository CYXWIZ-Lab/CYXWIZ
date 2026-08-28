#include "installer_setup_launcher.h"

#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

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
    const pid_t child = fork();
    if (child < 0) {
        error = "Cannot fork verified installer: " + std::string(std::strerror(errno));
        return false;
    }
    if (child == 0) {
        const auto executable = installer_path.string();
        char* const arguments[] = {const_cast<char*>(executable.c_str()), nullptr};
        execv(executable.c_str(), arguments);
        _exit(127);
    }
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        error = "Cannot wait for verified installer: " +
            std::string(std::strerror(errno));
        return false;
    }
    if (WIFEXITED(status)) {
        exit_code = WEXITSTATUS(status);
        return true;
    }
    error = "Verified installer terminated abnormally";
    return false;
}

}  // namespace cyxwiz::runtime
