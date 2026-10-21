#include "product_removal_handoff_platform.h"

#include "backend_pack_platform.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace cyxwiz::runtime::detail {
namespace {

constexpr off_t kMaximumFinalizerBytes = 16 * 1024 * 1024;

void RemoveStaging(
    const std::filesystem::path& finalizer,
    const std::filesystem::path& directory) {
    if (!finalizer.empty()) ::unlink(finalizer.c_str());
    if (!directory.empty()) ::rmdir(directory.c_str());
}

bool CopyExactExecutable(
    const std::filesystem::path& source,
    const std::filesystem::path& destination,
    std::string& error) {
    const int input = ::open(source.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (input < 0) {
        error = "Cannot open the exact product removal finalizer: " +
            std::string(std::strerror(errno));
        return false;
    }
    struct stat status{};
    if (::fstat(input, &status) != 0 || !S_ISREG(status.st_mode) ||
        status.st_size <= 0 || status.st_size > kMaximumFinalizerBytes) {
        const int code = errno == 0 ? EINVAL : errno;
        ::close(input);
        error = "The product removal finalizer is not an exact regular file: " +
            std::string(std::strerror(code));
        return false;
    }
    const int output = ::open(
        destination.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0700);
    if (output < 0) {
        const int code = errno;
        ::close(input);
        error = "Cannot create the staged product removal finalizer: " +
            std::string(std::strerror(code));
        return false;
    }
    bool succeeded = true;
    char buffer[64 * 1024];
    for (;;) {
        const auto bytes = ::read(input, buffer, sizeof(buffer));
        if (bytes == 0) break;
        if (bytes < 0) {
            if (errno == EINTR) continue;
            succeeded = false;
            break;
        }
        ssize_t offset = 0;
        while (offset < bytes) {
            const auto written = ::write(
                output, buffer + offset,
                static_cast<std::size_t>(bytes - offset));
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) {
                succeeded = false;
                break;
            }
            offset += written;
        }
        if (!succeeded) break;
    }
    if (succeeded && ::fsync(output) != 0) succeeded = false;
    const int operation_error = errno == 0 ? EIO : errno;
    ::close(output);
    ::close(input);
    if (!succeeded) {
        ::unlink(destination.c_str());
        error = "Cannot copy the complete product removal finalizer: " +
            std::string(std::strerror(operation_error));
    }
    return succeeded;
}

}  // namespace

bool LaunchDetachedProductRemovalFinalizer(
    const std::filesystem::path& source_finalizer,
    const std::filesystem::path& install_root,
    std::string_view install_id,
    ProductRemovalHandoff& handoff,
    std::string& error) {
    std::error_code filesystem_error;
    const auto temporary = std::filesystem::temp_directory_path(filesystem_error);
    if (filesystem_error) {
        error = "Cannot resolve the product-removal temporary directory";
        return false;
    }
    auto pattern = (temporary /
        ("cyxwiz-removal-" + std::string(install_id) + "-XXXXXX")).string();
    std::vector<char> mutable_pattern(pattern.begin(), pattern.end());
    mutable_pattern.push_back('\0');
    const char* created_directory = ::mkdtemp(mutable_pattern.data());
    if (created_directory == nullptr) {
        error = "Cannot create an exclusive product-removal temporary directory: " +
            std::string(std::strerror(errno));
        return false;
    }
    const std::filesystem::path directory(created_directory);
    const auto staged = directory /
        std::string(CurrentProductRemovalFinalizerExecutableName());
    const auto result = directory / "result.txt";
    if (!CopyExactExecutable(source_finalizer, staged, error)) {
        RemoveStaging(staged, directory);
        return false;
    }

    int lifetime[2] = {-1, -1};
    int launch_error[2] = {-1, -1};
    if (::pipe(lifetime) != 0 || ::pipe(launch_error) != 0 ||
        ::fcntl(lifetime[1], F_SETFD, FD_CLOEXEC) == -1 ||
        ::fcntl(launch_error[1], F_SETFD, FD_CLOEXEC) == -1) {
        const int code = errno;
        for (int descriptor : {lifetime[0], lifetime[1],
                               launch_error[0], launch_error[1]}) {
            if (descriptor >= 0) ::close(descriptor);
        }
        RemoveStaging(staged, directory);
        error = "Cannot create the product-removal process boundary: " +
            std::string(std::strerror(code));
        return false;
    }

    const pid_t intermediate = ::fork();
    if (intermediate < 0) {
        const int code = errno;
        for (int descriptor : {lifetime[0], lifetime[1],
                               launch_error[0], launch_error[1]}) {
            ::close(descriptor);
        }
        RemoveStaging(staged, directory);
        error = "Cannot fork the product removal finalizer: " +
            std::string(std::strerror(code));
        return false;
    }
    if (intermediate == 0) {
        ::close(lifetime[1]);
        ::close(launch_error[0]);
        int child_error = 0;
        if (::setsid() < 0) {
            child_error = errno;
        } else {
            const pid_t detached = ::fork();
            if (detached < 0) {
                child_error = errno;
            } else if (detached > 0) {
                ::_exit(0);
            } else {
                const std::string lifetime_text = std::to_string(lifetime[0]);
                std::vector<std::string> arguments{
                    staged.string(), "--install-root", install_root.string(),
                    "--parent-lifetime-fd", lifetime_text};
                std::vector<char*> values;
                values.reserve(arguments.size() + 1);
                for (auto& argument : arguments) values.push_back(argument.data());
                values.push_back(nullptr);
                ::execv(staged.c_str(), values.data());
                child_error = errno;
            }
        }
        const auto ignored = ::write(
            launch_error[1], &child_error, sizeof(child_error));
        (void)ignored;
        ::_exit(78);
    }

    ::close(lifetime[0]);
    ::close(launch_error[1]);
    int child_error = 0;
    const auto error_bytes = ::read(
        launch_error[0], &child_error, sizeof(child_error));
    ::close(launch_error[0]);
    int intermediate_status = 0;
    const bool reaped = ::waitpid(intermediate, &intermediate_status, 0) >= 0;
    if (error_bytes != 0 || !reaped ||
        !WIFEXITED(intermediate_status) || WEXITSTATUS(intermediate_status) != 0) {
        ::close(lifetime[1]);
        RemoveStaging(staged, directory);
        error = error_bytes > 0
            ? "Cannot launch the detached product removal finalizer: " +
                  std::string(std::strerror(child_error))
            : "Cannot confirm the detached product removal finalizer launch";
        return false;
    }
    handoff.staged_finalizer = staged;
    handoff.result_path = result;
    handoff.parent_lifetime = ProductRemovalParentLifetime(
        static_cast<std::uintptr_t>(lifetime[1]));
    return true;
}

void CloseProductRemovalLifetimeToken(std::uintptr_t token) noexcept {
    ::close(static_cast<int>(token));
}

}  // namespace cyxwiz::runtime::detail
