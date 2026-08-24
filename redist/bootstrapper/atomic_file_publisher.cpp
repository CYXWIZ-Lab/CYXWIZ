#include "atomic_file_publisher.h"

#include <chrono>
#include <fstream>
#include <system_error>
#include <utility>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace cyxwiz::runtime {
namespace {

class RemoveTemporaryFile {
public:
    explicit RemoveTemporaryFile(std::filesystem::path path)
        : path_(std::move(path)) {}
    ~RemoveTemporaryFile() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

private:
    std::filesystem::path path_;
};

bool FlushFileToStorage(
    const std::filesystem::path& path,
    std::string& error) {
#ifdef _WIN32
    const HANDLE file = ::CreateFileW(
        path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        error = "Cannot reopen temporary publication file; Win32 error " +
            std::to_string(::GetLastError());
        return false;
    }
    const bool flushed = ::FlushFileBuffers(file) != FALSE;
    const DWORD flush_error = flushed ? ERROR_SUCCESS : ::GetLastError();
    ::CloseHandle(file);
    if (!flushed) {
        error = "Cannot flush temporary publication file; Win32 error " +
            std::to_string(flush_error);
    }
    return flushed;
#else
    const int file = ::open(path.c_str(), O_RDONLY);
    if (file < 0) {
        error = "Cannot reopen temporary publication file";
        return false;
    }
    const bool flushed = ::fsync(file) == 0;
    ::close(file);
    if (!flushed) error = "Cannot flush temporary publication file";
    return flushed;
#endif
}

}  // namespace

bool PublishRegularFileAtomic(
    const std::filesystem::path& source,
    const std::filesystem::path& destination,
    std::uintmax_t maximum_bytes,
    std::string& error,
    AtomicFilePublishValidator validator) {
    if (!source.is_absolute() || !destination.is_absolute() ||
        maximum_bytes == 0) {
        error = "Absolute source/destination paths and a byte bound are required";
        return false;
    }
    std::error_code filesystem_error;
    const auto source_status = std::filesystem::symlink_status(
        source, filesystem_error);
    if (filesystem_error ||
        source_status.type() != std::filesystem::file_type::regular) {
        error = "Publication source is not a regular file: " + source.string();
        return false;
    }
    const auto size = std::filesystem::file_size(source, filesystem_error);
    if (filesystem_error || size > maximum_bytes) {
        error = filesystem_error
            ? "Cannot inspect publication source: " +
                  filesystem_error.message()
            : "Publication source exceeds its bounded file size";
        return false;
    }
    std::filesystem::create_directories(
        destination.parent_path(), filesystem_error);
    if (filesystem_error) {
        error = "Cannot create the publication directory: " +
            filesystem_error.message();
        return false;
    }
    auto temporary = destination;
    temporary += ".part-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    RemoveTemporaryFile cleanup(temporary);
    std::ifstream input(source, std::ios::binary);
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!input || !output) {
        error = "Cannot open the publication source or temporary file";
        return false;
    }
    output << input.rdbuf();
    output.flush();
    if (input.bad() || !output) {
        error = "Cannot copy the complete publication source";
        return false;
    }
    output.close();
    const auto copied_size =
        std::filesystem::file_size(temporary, filesystem_error);
    if (filesystem_error || copied_size != size) {
        error = "Temporary publication file is incomplete";
        return false;
    }
    std::filesystem::permissions(
        temporary, source_status.permissions(), filesystem_error);
    if (filesystem_error) {
        error = "Cannot preserve publication file permissions: " +
            filesystem_error.message();
        return false;
    }
    if (validator && !validator(temporary, error)) return false;
    if (!FlushFileToStorage(temporary, error)) return false;
#ifdef _WIN32
    if (!::MoveFileExW(
            temporary.c_str(), destination.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        error = "Cannot atomically publish file; Win32 error " +
            std::to_string(::GetLastError());
        return false;
    }
#else
    std::filesystem::rename(temporary, destination, filesystem_error);
    if (filesystem_error) {
        error = "Cannot atomically publish file: " +
            filesystem_error.message();
        return false;
    }
#endif
    error.clear();
    return true;
}

}  // namespace cyxwiz::runtime
