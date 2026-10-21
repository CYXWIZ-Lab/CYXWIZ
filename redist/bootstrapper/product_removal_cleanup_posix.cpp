#include "product_removal_cleanup_platform.h"

#include "product_installation_receipt.h"
#include "product_removal_request.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <string>

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace cyxwiz::runtime::detail {
namespace {

constexpr std::uint64_t kMaximumEntries = 1'000'000;
constexpr unsigned int kMaximumDepth = 256;

class OwnedDescriptor {
public:
    explicit OwnedDescriptor(int descriptor = -1) : descriptor_(descriptor) {}
    ~OwnedDescriptor() { Close(); }
    void Close() {
        if (descriptor_ >= 0) ::close(descriptor_);
        descriptor_ = -1;
    }
    OwnedDescriptor(const OwnedDescriptor&) = delete;
    OwnedDescriptor& operator=(const OwnedDescriptor&) = delete;
    int get() const { return descriptor_; }

private:
    int descriptor_ = -1;
};

bool IsEvidenceName(const char* name) {
    return std::strcmp(name, ".cyxwiz-installation.json") == 0 ||
        std::strcmp(name, ".cyxwiz-removal-request.json") == 0;
}

bool InspectTree(
    int directory,
    dev_t root_device,
    unsigned int depth,
    std::uint64_t& entries,
    std::string& error) {
    if (depth > kMaximumDepth) {
        error = "Product cleanup exceeded its directory-depth bound";
        return false;
    }
    const int duplicate = ::openat(
        directory, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (duplicate < 0) {
        error = "Cannot duplicate a quarantine directory handle: " +
            std::string(std::strerror(errno));
        return false;
    }
    DIR* stream = ::fdopendir(duplicate);
    if (stream == nullptr) {
        ::close(duplicate);
        error = "Cannot enumerate the product quarantine: " +
            std::string(std::strerror(errno));
        return false;
    }
    errno = 0;
    while (const dirent* entry = ::readdir(stream)) {
        if (std::strcmp(entry->d_name, ".") == 0 ||
            std::strcmp(entry->d_name, "..") == 0) {
            errno = 0;
            continue;
        }
        if (++entries > kMaximumEntries) {
            ::closedir(stream);
            error = "Product cleanup exceeded its entry-count bound";
            return false;
        }
        struct stat status{};
        if (::fstatat(
                directory, entry->d_name, &status,
                AT_SYMLINK_NOFOLLOW) != 0) {
            const int code = errno;
            ::closedir(stream);
            error = "Cannot inspect a product quarantine entry: " +
                std::string(std::strerror(code));
            return false;
        }
        if (!S_ISDIR(status.st_mode)) {
            errno = 0;
            continue;
        }
        if (status.st_dev != root_device) {
            ::closedir(stream);
            error = "Product cleanup refuses to cross a mounted filesystem";
            return false;
        }
        OwnedDescriptor child(::openat(
            directory, entry->d_name,
            O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
        struct stat opened{};
        if (child.get() < 0 || ::fstat(child.get(), &opened) != 0 ||
            opened.st_dev != status.st_dev || opened.st_ino != status.st_ino) {
            const int code = errno == 0 ? EINVAL : errno;
            ::closedir(stream);
            error = "A product quarantine directory changed during preflight: " +
                std::string(std::strerror(code));
            return false;
        }
        if (!InspectTree(
                child.get(), root_device, depth + 1, entries, error)) {
            ::closedir(stream);
            return false;
        }
        errno = 0;
    }
    const int read_error = errno;
    ::closedir(stream);
    if (read_error != 0) {
        error = "Cannot complete product quarantine enumeration: " +
            std::string(std::strerror(read_error));
        return false;
    }
    return true;
}

bool RemoveTree(
    int directory,
    dev_t root_device,
    unsigned int depth,
    bool preserve_evidence,
    ProductRemovalCleanupResult& result,
    std::string& error) {
    const int duplicate = ::openat(
        directory, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (duplicate < 0) {
        error = "Cannot duplicate a quarantine cleanup handle: " +
            std::string(std::strerror(errno));
        return false;
    }
    DIR* stream = ::fdopendir(duplicate);
    if (stream == nullptr) {
        ::close(duplicate);
        error = "Cannot enumerate quarantine cleanup entries: " +
            std::string(std::strerror(errno));
        return false;
    }
    errno = 0;
    while (const dirent* entry = ::readdir(stream)) {
        if (std::strcmp(entry->d_name, ".") == 0 ||
            std::strcmp(entry->d_name, "..") == 0 ||
            (preserve_evidence && IsEvidenceName(entry->d_name))) {
            errno = 0;
            continue;
        }
        struct stat status{};
        if (::fstatat(
                directory, entry->d_name, &status,
                AT_SYMLINK_NOFOLLOW) != 0) {
            const int code = errno;
            ::closedir(stream);
            error = "Cannot revalidate a quarantine entry before deletion: " +
                std::string(std::strerror(code));
            return false;
        }
        int flags = 0;
        if (S_ISDIR(status.st_mode)) {
            if (status.st_dev != root_device || depth >= kMaximumDepth) {
                ::closedir(stream);
                error = "Product cleanup refuses an unsafe directory boundary";
                return false;
            }
            OwnedDescriptor child(::openat(
                directory, entry->d_name,
                O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
            struct stat opened{};
            if (child.get() < 0 || ::fstat(child.get(), &opened) != 0 ||
                opened.st_dev != status.st_dev || opened.st_ino != status.st_ino ||
                !RemoveTree(
                    child.get(), root_device, depth + 1, false,
                    result, error)) {
                if (error.empty()) {
                    error = "A quarantine directory changed during cleanup";
                }
                ::closedir(stream);
                return false;
            }
            flags = AT_REMOVEDIR;
        }
        if (::unlinkat(directory, entry->d_name, flags) != 0) {
            const int code = errno;
            ::closedir(stream);
            error = "Cannot remove a product quarantine entry: " +
                std::string(std::strerror(code));
            return false;
        }
        ++result.removed_entries;
        errno = 0;
    }
    const int read_error = errno;
    ::closedir(stream);
    if (read_error != 0) {
        error = "Cannot complete quarantine cleanup enumeration: " +
            std::string(std::strerror(read_error));
        return false;
    }
    return true;
}

}  // namespace

bool CleanupQuarantineNoFollow(
    const QuarantinedProductInstallation& quarantined,
    ProductRemovalCleanupResult& result,
    std::string& error) {
    const auto parent_path = quarantined.quarantine_root.parent_path();
    const auto name = quarantined.quarantine_root.filename().string();
    OwnedDescriptor parent(::open(
        parent_path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
    OwnedDescriptor root(parent.get() < 0 ? -1 : ::openat(
        parent.get(), name.c_str(),
        O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
    struct stat root_status{};
    if (parent.get() < 0 || root.get() < 0 ||
        ::fstat(root.get(), &root_status) != 0) {
        error = "Cannot open the exact product quarantine for cleanup: " +
            std::string(std::strerror(errno));
        return false;
    }
    struct stat pinned_path{};
    if (!ValidateQuarantinedProductInstallation(quarantined, error) ||
        ::fstatat(
            parent.get(), name.c_str(), &pinned_path,
            AT_SYMLINK_NOFOLLOW) != 0 ||
        pinned_path.st_dev != root_status.st_dev ||
        pinned_path.st_ino != root_status.st_ino) {
        if (error.empty()) {
            error = "The product quarantine changed while cleanup pinned it";
        }
        return false;
    }
    std::uint64_t entries = 0;
    if (!InspectTree(root.get(), root_status.st_dev, 0, entries, error) ||
        !RemoveTree(root.get(), root_status.st_dev, 0, true, result, error)) {
        return false;
    }
    const char* request = ".cyxwiz-removal-request.json";
    struct stat request_status{};
    if (::fstatat(root.get(), request, &request_status, AT_SYMLINK_NOFOLLOW) == 0) {
        if (!S_ISREG(request_status.st_mode) ||
            ::unlinkat(root.get(), request, 0) != 0) {
            error = "Cannot remove the exact product removal request evidence";
            return false;
        }
        ++result.removed_entries;
    } else if (errno != ENOENT) {
        error = "Cannot inspect product removal request evidence";
        return false;
    }
    const char* receipt = ".cyxwiz-installation.json";
    struct stat receipt_status{};
    if (::fstatat(root.get(), receipt, &receipt_status, AT_SYMLINK_NOFOLLOW) != 0 ||
        !S_ISREG(receipt_status.st_mode) ||
        ::unlinkat(root.get(), receipt, 0) != 0) {
        error = "Cannot remove the final product ownership receipt";
        return false;
    }
    ++result.removed_entries;
    root.Close();
    struct stat remaining{};
    if (::fstatat(parent.get(), name.c_str(), &remaining, AT_SYMLINK_NOFOLLOW) != 0 ||
        remaining.st_dev != root_status.st_dev ||
        remaining.st_ino != root_status.st_ino ||
        ::unlinkat(parent.get(), name.c_str(), AT_REMOVEDIR) != 0) {
        error = "Cannot remove the empty product quarantine root";
        return false;
    }
    ++result.removed_entries;
    return true;
}

}  // namespace cyxwiz::runtime::detail
