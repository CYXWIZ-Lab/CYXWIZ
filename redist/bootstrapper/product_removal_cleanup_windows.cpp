#include "product_removal_cleanup_platform.h"

#include <cstdint>
#include <filesystem>
#include <string>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace cyxwiz::runtime::detail {
namespace {

constexpr std::uint64_t kMaximumEntries = 1'000'000;
constexpr unsigned int kMaximumDepth = 256;

class OwnedHandle {
public:
    explicit OwnedHandle(HANDLE handle = INVALID_HANDLE_VALUE)
        : handle_(handle) {}
    ~OwnedHandle() {
        if (handle_ != INVALID_HANDLE_VALUE) ::CloseHandle(handle_);
    }
    OwnedHandle(const OwnedHandle&) = delete;
    OwnedHandle& operator=(const OwnedHandle&) = delete;
    HANDLE get() const { return handle_; }
    bool valid() const { return handle_ != INVALID_HANDLE_VALUE; }

private:
    HANDLE handle_ = INVALID_HANDLE_VALUE;
};

bool IsEvidenceName(const wchar_t* name) {
    return ::CompareStringOrdinal(
               name, -1, L".cyxwiz-installation.json", -1, TRUE) ==
            CSTR_EQUAL ||
        ::CompareStringOrdinal(
               name, -1, L".cyxwiz-removal-request.json", -1, TRUE) ==
            CSTR_EQUAL;
}

OwnedHandle OpenEntry(
    const std::filesystem::path& path,
    bool directory,
    DWORD access = FILE_READ_ATTRIBUTES) {
    return OwnedHandle(::CreateFileW(
        path.c_str(), access,
        FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT |
            (directory ? FILE_FLAG_BACKUP_SEMANTICS : 0),
        nullptr));
}

bool ReadAttributes(
    HANDLE handle,
    FILE_ATTRIBUTE_TAG_INFO& attributes,
    std::string& error) {
    if (!::GetFileInformationByHandleEx(
            handle, FileAttributeTagInfo, &attributes, sizeof(attributes))) {
        error = "Cannot inspect an exact quarantine entry; Win32 error " +
            std::to_string(::GetLastError());
        return false;
    }
    return true;
}

bool Enumerate(
    const std::filesystem::path& directory,
    const auto& operation,
    std::string& error) {
    WIN32_FIND_DATAW entry{};
    const auto pattern = directory / "*";
    const HANDLE search = ::FindFirstFileW(pattern.c_str(), &entry);
    if (search == INVALID_HANDLE_VALUE) {
        const DWORD code = ::GetLastError();
        if (code == ERROR_FILE_NOT_FOUND) return true;
        error = "Cannot enumerate the product quarantine; Win32 error " +
            std::to_string(code);
        return false;
    }
    bool succeeded = true;
    do {
        if (::CompareStringOrdinal(entry.cFileName, -1, L".", -1, FALSE) ==
                CSTR_EQUAL ||
            ::CompareStringOrdinal(entry.cFileName, -1, L"..", -1, FALSE) ==
                CSTR_EQUAL) {
            continue;
        }
        if (!operation(entry, error)) {
            succeeded = false;
            break;
        }
    } while (::FindNextFileW(search, &entry));
    const DWORD enumeration_error = ::GetLastError();
    ::FindClose(search);
    if (succeeded && enumeration_error != ERROR_NO_MORE_FILES) {
        error = "Product quarantine enumeration changed unexpectedly; Win32 "
            "error " + std::to_string(enumeration_error);
        return false;
    }
    return succeeded;
}

bool InspectTree(
    const std::filesystem::path& directory,
    unsigned int depth,
    std::uint64_t& entries,
    std::string& error) {
    if (depth > kMaximumDepth) {
        error = "Product cleanup exceeded its directory-depth bound";
        return false;
    }
    auto opened = OpenEntry(directory, true);
    FILE_ATTRIBUTE_TAG_INFO attributes{};
    if (!opened.valid() || !ReadAttributes(opened.get(), attributes, error) ||
        (attributes.FileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0 ||
        (attributes.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
        if (error.empty()) {
            error = "Product cleanup refuses a redirected directory";
        }
        return false;
    }
    return Enumerate(
        directory,
        [&](const WIN32_FIND_DATAW& entry, std::string& nested_error) {
            if (++entries > kMaximumEntries) {
                nested_error = "Product cleanup exceeded its entry-count bound";
                return false;
            }
            if ((entry.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0 &&
                (entry.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) == 0) {
                return InspectTree(
                    directory / entry.cFileName, depth + 1,
                    entries, nested_error);
            }
            return true;
        },
        error);
}

bool DeleteExactEntry(
    const std::filesystem::path& path,
    bool directory,
    bool require_regular,
    std::string& error) {
    auto opened = OpenEntry(
        path, directory, DELETE | FILE_READ_ATTRIBUTES);
    FILE_ATTRIBUTE_TAG_INFO attributes{};
    if (!opened.valid() || !ReadAttributes(opened.get(), attributes, error)) {
        if (error.empty()) {
            error = "Cannot open an exact quarantine entry for deletion";
        }
        return false;
    }
    const bool actual_directory =
        (attributes.FileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
    const bool redirected =
        (attributes.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
    if (actual_directory != directory || (require_regular && redirected)) {
        error = "A quarantine entry changed before exact deletion";
        return false;
    }
    FILE_DISPOSITION_INFO disposition{};
    disposition.DeleteFile = TRUE;
    if (!::SetFileInformationByHandle(
            opened.get(), FileDispositionInfo,
            &disposition, sizeof(disposition))) {
        error = "Cannot remove an exact product quarantine entry; Win32 error " +
            std::to_string(::GetLastError());
        return false;
    }
    return true;
}

bool RemoveTree(
    const std::filesystem::path& directory,
    unsigned int depth,
    bool preserve_evidence,
    ProductRemovalCleanupResult& result,
    std::string& error) {
    auto opened = OpenEntry(directory, true);
    FILE_ATTRIBUTE_TAG_INFO directory_attributes{};
    if (!opened.valid() ||
        !ReadAttributes(opened.get(), directory_attributes, error) ||
        (directory_attributes.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
        if (error.empty()) error = "A quarantine directory became redirected";
        return false;
    }
    return Enumerate(
        directory,
        [&](const WIN32_FIND_DATAW& entry, std::string& nested_error) {
            if (preserve_evidence && IsEvidenceName(entry.cFileName)) return true;
            const auto path = directory / entry.cFileName;
            const bool is_directory =
                (entry.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
            const bool is_reparse =
                (entry.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
            if (is_directory && !is_reparse) {
                if (depth >= kMaximumDepth ||
                    !RemoveTree(
                        path, depth + 1, false, result, nested_error)) {
                    return false;
                }
            }
            if (!DeleteExactEntry(path, is_directory, false, nested_error)) {
                return false;
            }
            ++result.removed_entries;
            return true;
        },
        error);
}

bool DeleteEvidenceIfPresent(
    const std::filesystem::path& path,
    bool required,
    ProductRemovalCleanupResult& result,
    std::string& error) {
    const DWORD attributes = ::GetFileAttributesW(path.c_str());
    if (attributes == INVALID_FILE_ATTRIBUTES) {
        if (!required && ::GetLastError() == ERROR_FILE_NOT_FOUND) return true;
        error = "Required product removal evidence is missing";
        return false;
    }
    if (!DeleteExactEntry(path, false, true, error)) return false;
    ++result.removed_entries;
    return true;
}

}  // namespace

bool CleanupQuarantineNoFollow(
    const QuarantinedProductInstallation& quarantined,
    ProductRemovalCleanupResult& result,
    std::string& error) {
    auto root = OpenEntry(quarantined.quarantine_root, true, DELETE | FILE_READ_ATTRIBUTES);
    FILE_ATTRIBUTE_TAG_INFO root_attributes{};
    if (!root.valid() || !ReadAttributes(root.get(), root_attributes, error) ||
        (root_attributes.FileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0 ||
        (root_attributes.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
        if (error.empty()) error = "Cannot open the exact product quarantine";
        return false;
    }
    if (!ValidateQuarantinedProductInstallation(quarantined, error)) {
        error = "The pinned product quarantine failed live validation: " + error;
        return false;
    }
    std::uint64_t entries = 0;
    if (!InspectTree(quarantined.quarantine_root, 0, entries, error) ||
        !RemoveTree(
            quarantined.quarantine_root, 0, true, result, error) ||
        !DeleteEvidenceIfPresent(
            quarantined.quarantine_root / ".cyxwiz-removal-request.json",
            false, result, error) ||
        !DeleteEvidenceIfPresent(
            quarantined.quarantine_root / ".cyxwiz-installation.json",
            true, result, error)) {
        return false;
    }
    FILE_DISPOSITION_INFO disposition{};
    disposition.DeleteFile = TRUE;
    if (!::SetFileInformationByHandle(
            root.get(), FileDispositionInfo, &disposition, sizeof(disposition))) {
        error = "Cannot remove the empty product quarantine root; Win32 error " +
            std::to_string(::GetLastError());
        return false;
    }
    ++result.removed_entries;
    return true;
}

}  // namespace cyxwiz::runtime::detail
