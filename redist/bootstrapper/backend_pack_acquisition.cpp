#include "backend_pack_acquisition.h"
#include "backend_pack_hash.h"
#include "backend_pack_path.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace cyxwiz::runtime {
namespace {

bool IsArchiveFileName(std::string_view value) {
    return value.size() <= 255 &&
           IsCanonicalBackendPackRelativePath(value) &&
           value.find('/') == std::string_view::npos &&
           std::all_of(value.begin(), value.end(), [](unsigned char character) {
               return character >= 0x20 && character < 0x7f &&
                      std::string_view{"\"<>|?*"}.find(character) ==
                          std::string_view::npos;
           });
}

std::string PercentEncodeUrlPathSegment(std::string_view value) {
    std::ostringstream encoded;
    encoded << std::uppercase << std::hex;
    for (const unsigned char character : value) {
        if ((character >= 'a' && character <= 'z') ||
            (character >= 'A' && character <= 'Z') ||
            (character >= '0' && character <= '9') || character == '-' ||
            character == '_' || character == '.' || character == '~') {
            encoded << static_cast<char>(character);
        } else {
            encoded << '%' << std::setw(2) << std::setfill('0')
                    << static_cast<unsigned int>(character);
        }
    }
    return encoded.str();
}

std::filesystem::path PartialPath(const std::filesystem::path& destination) {
    auto path = destination;
    path += ".part";
    return path;
}

bool IsRegularNonLink(
    const std::filesystem::path& path,
    std::string& error) {
    std::error_code filesystem_error;
    const auto status = std::filesystem::symlink_status(path, filesystem_error);
    if (filesystem_error || !std::filesystem::is_regular_file(status) ||
        std::filesystem::is_symlink(status)) {
        error = "Artifact path is not a regular non-link file: " + path.string();
        return false;
    }
    return true;
}

bool IsSafeWritablePartial(
    const std::filesystem::path& path,
    std::string& error) {
    if (!IsRegularNonLink(path, error)) return false;
#ifdef _WIN32
    HANDLE file = ::CreateFileW(
        path.c_str(), FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
        OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        error = "Cannot inspect partial artifact link ownership";
        return false;
    }
    BY_HANDLE_FILE_INFORMATION information{};
    const BOOL inspected = ::GetFileInformationByHandle(file, &information);
    ::CloseHandle(file);
    if (!inspected || information.nNumberOfLinks != 1 ||
        (information.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
        error = "Partial artifact must be an unlinked regular file";
        return false;
    }
#else
    struct stat information {};
    if (::lstat(path.c_str(), &information) != 0 ||
        !S_ISREG(information.st_mode) || information.st_nlink != 1) {
        error = "Partial artifact must be an unlinked regular file";
        return false;
    }
#endif
    return true;
}

bool VerifyArtifact(
    const std::filesystem::path& path,
    std::uint64_t expected_size,
    const std::string& expected_sha256,
    std::string& error) {
    if (!IsRegularNonLink(path, error)) return false;
    std::error_code filesystem_error;
    const auto size = std::filesystem::file_size(path, filesystem_error);
    if (filesystem_error || size != expected_size) {
        error = "Artifact byte size differs from signed metadata";
        return false;
    }
    std::string digest;
    if (!Sha256File(path, digest, error)) return false;
    if (digest != expected_sha256) {
        error = "Artifact SHA-256 differs from signed metadata";
        return false;
    }
    return true;
}

bool FlushFileToDisk(
    const std::filesystem::path& path,
    std::string& error) {
#ifdef _WIN32
    HANDLE file = ::CreateFileW(
        path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        error = "Cannot open completed partial artifact for flushing";
        return false;
    }
    const BOOL flushed = ::FlushFileBuffers(file);
    const DWORD flush_error = flushed ? ERROR_SUCCESS : ::GetLastError();
    ::CloseHandle(file);
    if (!flushed) {
        error = "Cannot flush completed partial artifact; Win32 error " +
                std::to_string(flush_error);
        return false;
    }
#else
    const int descriptor = ::open(
        path.c_str(), O_WRONLY | O_CLOEXEC | O_NOFOLLOW);
    if (descriptor < 0) {
        error = "Cannot safely open completed partial artifact: " +
            std::string(std::strerror(errno));
        return false;
    }
    struct stat information {};
    const bool safe = ::fstat(descriptor, &information) == 0 &&
        S_ISREG(information.st_mode) && information.st_nlink == 1;
    const bool flushed = safe && ::fsync(descriptor) == 0;
    const int flush_error = errno;
    ::close(descriptor);
    if (!flushed) {
        error = safe
            ? "Cannot flush completed partial artifact: " +
                  std::string(std::strerror(flush_error))
            : "Completed partial artifact changed before flushing";
        return false;
    }
#endif
    return true;
}

bool PublishArtifact(
    const std::filesystem::path& partial,
    const std::filesystem::path& destination,
    std::string& error) {
#ifdef _WIN32
    if (!::MoveFileExW(
            partial.c_str(), destination.c_str(), MOVEFILE_WRITE_THROUGH)) {
        error = "Cannot atomically publish downloaded artifact; Win32 error " +
                std::to_string(::GetLastError());
        return false;
    }
#else
    std::error_code filesystem_error;
    std::filesystem::rename(partial, destination, filesystem_error);
    if (filesystem_error) {
        error = "Cannot atomically publish downloaded artifact: " +
                filesystem_error.message();
        return false;
    }
    const int directory = ::open(
        destination.parent_path().c_str(),
        O_RDONLY | O_CLOEXEC | O_DIRECTORY);
    if (directory < 0 || ::fsync(directory) != 0) {
        const int publish_error = errno;
        if (directory >= 0) ::close(directory);
        error = "Cannot durably publish downloaded artifact: " +
            std::string(std::strerror(publish_error));
        return false;
    }
    ::close(directory);
#endif
    return true;
}

}  // namespace

bool ResolveHttpsBackendPackArchiveUrl(
    std::string_view manifest_url,
    std::string_view archive_file_name,
    std::string& archive_url,
    std::string& error) {
    constexpr std::string_view scheme = "https://";
    archive_url.clear();
    error.clear();
    if (!manifest_url.starts_with(scheme) ||
        manifest_url.size() > 4096 ||
        std::any_of(
            manifest_url.begin(), manifest_url.end(),
            [](unsigned char character) {
                return character <= 0x20 || character >= 0x7f ||
                       character == '\\';
            }) ||
        manifest_url.find_first_of("?#") != std::string_view::npos ||
        !IsArchiveFileName(archive_file_name)) {
        error = "Signed manifest or archive source is invalid";
        return false;
    }
    const auto authority_end = manifest_url.find('/', scheme.size());
    if (authority_end == std::string_view::npos ||
        authority_end == scheme.size() ||
        manifest_url.substr(scheme.size(), authority_end - scheme.size())
                .find('@') != std::string_view::npos) {
        error = "Signed manifest HTTPS authority is invalid";
        return false;
    }
    const auto file_separator = manifest_url.rfind('/');
    if (file_separator < authority_end ||
        file_separator + 1 == manifest_url.size()) {
        error = "Signed manifest URL must identify a file";
        return false;
    }
    archive_url.assign(manifest_url.substr(0, file_separator + 1));
    archive_url += PercentEncodeUrlPathSegment(archive_file_name);
    if (archive_url.size() > 4096) {
        archive_url.clear();
        error = "Derived backend-pack archive URL is too long";
        return false;
    }
    return true;
}

bool ResolveOfflineBackendPackArchivePath(
    const std::filesystem::path& manifest_path,
    std::string_view archive_file_name,
    std::filesystem::path& archive_path,
    std::string& error) {
    archive_path.clear();
    error.clear();
    if (!manifest_path.is_absolute() || !manifest_path.has_filename() ||
        !IsArchiveFileName(archive_file_name)) {
        error = "Offline manifest or archive source is invalid";
        return false;
    }
    archive_path = manifest_path.parent_path() /
        BackendPackNativeRelativePath(archive_file_name);
    return true;
}

OfflineBackendPackArtifactSource::OfflineBackendPackArtifactSource(
    std::filesystem::path path)
    : path_(std::move(path)) {}

std::string OfflineBackendPackArtifactSource::Description() const {
    return "offline:" + path_.string();
}

bool OfflineBackendPackArtifactSource::TransferFrom(
    std::uint64_t offset,
    std::uint64_t expected_size,
    const BackendPackArtifactChunk& consume,
    const BackendPackArtifactCancelCheck& cancelled,
    std::string& error) {
    if (offset > expected_size || !path_.is_absolute() ||
        !IsRegularNonLink(path_, error)) {
        if (error.empty()) error = "Offline artifact path must be absolute";
        return false;
    }
    std::error_code filesystem_error;
    if (std::filesystem::file_size(path_, filesystem_error) != expected_size ||
        filesystem_error) {
        error = "Offline artifact byte size differs from signed metadata";
        return false;
    }
    std::ifstream stream(path_, std::ios::binary);
    if (!stream || offset >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::streamoff>::max())) {
        error = "Cannot open or seek the offline artifact";
        return false;
    }
    stream.seekg(static_cast<std::streamoff>(offset));
    if (!stream) {
        error = "Cannot resume the offline artifact at the requested offset";
        return false;
    }
    std::vector<char> buffer(1024 * 1024);
    std::uint64_t transferred = offset;
    while (transferred < expected_size) {
        if (cancelled()) {
            error = "Artifact acquisition cancelled";
            return false;
        }
        const auto remaining = expected_size - transferred;
        const auto request = static_cast<std::streamsize>(
            std::min<std::uint64_t>(remaining, buffer.size()));
        stream.read(buffer.data(), request);
        const auto count = stream.gcount();
        if (count <= 0 ||
            !consume(buffer.data(), static_cast<std::size_t>(count), error)) {
            if (error.empty()) error = "Cannot read the offline artifact";
            return false;
        }
        transferred += static_cast<std::uint64_t>(count);
    }
    if (stream.peek() != std::char_traits<char>::eof()) {
        error = "Offline artifact changed while it was being copied";
        return false;
    }
    return true;
}

BackendPackArtifactAcquirer::BackendPackArtifactAcquirer(
    BackendPackAcquisitionObserver observer)
    : observer_(std::move(observer)) {}

BackendPackAcquisitionResult BackendPackArtifactAcquirer::Acquire(
    BackendPackArtifactSource& source,
    const std::filesystem::path& destination,
    std::uint64_t expected_size,
    std::string expected_sha256,
    std::uint64_t disk_budget_bytes,
    BackendPackAcquisitionRetryPolicy retry) {
    std::unique_lock<std::mutex> acquisition_lock(
        acquisition_mutex_, std::try_to_lock);
    if (!acquisition_lock.owns_lock()) {
        return {BackendPackAcquisitionStatus::Busy,
                "An artifact acquisition is already running"};
    }
    cancel_requested_.store(false);
    BackendPackAcquisitionProgress progress;
    progress.stage = BackendPackAcquisitionStage::Preparing;
    progress.source = source.Description();
    progress.destination = destination;
    progress.total_bytes = expected_size;
    progress.message = "Preparing resumable artifact acquisition";
    SetProgress(progress);

    if (destination.empty() || !destination.is_absolute() ||
        destination.filename().empty() ||
        expected_size == 0 || !IsLowercaseSha256(expected_sha256) ||
        retry.maximum_attempts == 0 || retry.maximum_attempts > 5 ||
        retry.backoff < std::chrono::milliseconds(0) ||
        retry.backoff > std::chrono::seconds(30)) {
        return Finish(
            BackendPackAcquisitionStatus::InvalidRequest,
            "Artifact destination, byte size, and SHA-256 are required");
    }
    if (disk_budget_bytes > 0 && expected_size > disk_budget_bytes) {
        return Finish(
            BackendPackAcquisitionStatus::DiskBudgetExceeded,
            "Artifact exceeds the approved disk budget");
    }
    std::error_code filesystem_error;
    std::filesystem::create_directories(
        destination.parent_path(), filesystem_error);
    if (filesystem_error) {
        return Finish(
            BackendPackAcquisitionStatus::FilesystemFailure,
            "Cannot create artifact destination directory: " +
                filesystem_error.message());
    }
    if (std::filesystem::exists(destination, filesystem_error)) {
        std::string error;
        if (VerifyArtifact(
                destination, expected_size, expected_sha256, error)) {
            return Finish(
                BackendPackAcquisitionStatus::AlreadyPresent,
                "Verified artifact is already present", destination);
        }
        return Finish(BackendPackAcquisitionStatus::IntegrityFailure, error);
    }

    const auto partial = PartialPath(destination);
    std::uint64_t offset = 0;
    if (std::filesystem::exists(partial, filesystem_error)) {
        std::string error;
        if (!IsSafeWritablePartial(partial, error)) {
            return Finish(
                BackendPackAcquisitionStatus::FilesystemFailure, error);
        }
        offset = std::filesystem::file_size(partial, filesystem_error);
        if (filesystem_error || offset > expected_size) {
            return Finish(
                BackendPackAcquisitionStatus::IntegrityFailure,
                "Partial artifact is larger than signed metadata");
        }
        if (offset == expected_size) {
            if (VerifyArtifact(
                    partial, expected_size, expected_sha256, error)) {
                if (!FlushFileToDisk(partial, error) ||
                    !PublishArtifact(partial, destination, error)) {
                    return Finish(
                        BackendPackAcquisitionStatus::FilesystemFailure,
                        error);
                }
                return Finish(
                    BackendPackAcquisitionStatus::Downloaded,
                    "Verified completed partial artifact was published",
                    destination, offset);
            }
            std::filesystem::remove(partial, filesystem_error);
            if (filesystem_error) {
                return Finish(
                    BackendPackAcquisitionStatus::FilesystemFailure,
                    "Cannot discard a corrupt completed partial artifact");
            }
            offset = 0;
        }
    }
    const auto disk = std::filesystem::space(
        destination.parent_path(), filesystem_error);
    if (filesystem_error || disk.available < expected_size - offset) {
        return Finish(
            BackendPackAcquisitionStatus::DiskBudgetExceeded,
            "Insufficient free space for the remaining artifact bytes");
    }

    progress.stage = BackendPackAcquisitionStage::Transferring;
    progress.resumed_bytes = offset;
    progress.completed_bytes = offset;
    progress.message = offset == 0 ? "Transferring artifact" :
                                     "Resuming partial artifact";
    SetProgress(progress);
    std::ofstream output(
        partial, std::ios::binary |
            (offset == 0 ? std::ios::trunc : std::ios::app));
    if (!output) {
        return Finish(
            BackendPackAcquisitionStatus::FilesystemFailure,
            "Cannot open partial artifact for writing");
    }
    std::uint64_t completed = offset;
    std::string error;
    bool transferred = false;
    std::size_t attempts = 0;
    for (; attempts < retry.maximum_attempts; ++attempts) {
        error.clear();
        transferred = source.TransferFrom(
            completed, expected_size,
            [&](const char* bytes, std::size_t size,
                std::string& sink_error) {
                if (size > expected_size - completed) {
                    sink_error =
                        "Artifact source exceeded its signed byte size";
                    return false;
                }
                output.write(bytes, static_cast<std::streamsize>(size));
                if (!output) {
                    sink_error = "Cannot write the partial artifact";
                    return false;
                }
                completed += size;
                progress.completed_bytes = completed;
                SetProgress(progress);
                return !cancel_requested_.load();
            },
            [&] { return cancel_requested_.load(); }, error);
        if (transferred || cancel_requested_.load() || !output ||
            attempts + 1 >= retry.maximum_attempts) {
            break;
        }
        output.flush();
        if (!output) break;
        progress.message =
            "Connection interrupted; retrying from " +
            std::to_string(completed) + " downloaded bytes (attempt " +
            std::to_string(attempts + 2) + " of " +
            std::to_string(retry.maximum_attempts) + ")";
        SetProgress(progress);
        const auto delay = retry.backoff *
            static_cast<std::chrono::milliseconds::rep>(attempts + 1);
        auto waited = std::chrono::milliseconds(0);
        while (waited < delay && !cancel_requested_.load()) {
            const auto slice = std::min(
                std::chrono::milliseconds(100), delay - waited);
            std::this_thread::sleep_for(slice);
            waited += slice;
        }
    }
    output.flush();
    const bool output_ok = static_cast<bool>(output);
    output.close();
    if (!transferred) {
        return Finish(
            cancel_requested_.load()
                ? BackendPackAcquisitionStatus::Interrupted
                : BackendPackAcquisitionStatus::SourceFailure,
            cancel_requested_.load()
                ? (error.empty() ? "Artifact transfer cancelled" : error)
                : "Artifact transfer failed after " +
                      std::to_string(attempts + 1) +
                      " attempt(s); the partial download is preserved for "
                      "resume: " +
                      (error.empty() ? "source unavailable" : error),
            {}, offset);
    }
    if (!output_ok || completed != expected_size) {
        return Finish(
            BackendPackAcquisitionStatus::FilesystemFailure,
            "Artifact transfer ended before the signed byte size", {}, offset);
    }
    if (!FlushFileToDisk(partial, error)) {
        return Finish(
            BackendPackAcquisitionStatus::FilesystemFailure, error, {}, offset);
    }

    progress.stage = BackendPackAcquisitionStage::Verifying;
    progress.message = "Verifying completed artifact hash";
    SetProgress(progress);
    if (!VerifyArtifact(partial, expected_size, expected_sha256, error)) {
        std::filesystem::remove(partial, filesystem_error);
        return Finish(
            BackendPackAcquisitionStatus::IntegrityFailure, error, {}, offset);
    }
    progress.stage = BackendPackAcquisitionStage::Publishing;
    progress.message = "Publishing verified artifact atomically";
    SetProgress(progress);
    if (!PublishArtifact(partial, destination, error)) {
        if (std::filesystem::exists(destination, filesystem_error) &&
            VerifyArtifact(
                destination, expected_size, expected_sha256, error)) {
            std::filesystem::remove(partial, filesystem_error);
            return Finish(
                BackendPackAcquisitionStatus::AlreadyPresent,
                "Another transaction published the verified artifact",
                destination, offset);
        }
        return Finish(
            BackendPackAcquisitionStatus::FilesystemFailure, error, {}, offset);
    }
    return Finish(
        BackendPackAcquisitionStatus::Downloaded,
        "Artifact downloaded and verified", destination, offset);
}

void BackendPackArtifactAcquirer::Cancel() {
    cancel_requested_.store(true);
}

BackendPackAcquisitionProgress BackendPackArtifactAcquirer::GetProgress() const {
    std::lock_guard<std::mutex> lock(progress_mutex_);
    return progress_;
}

BackendPackAcquisitionResult BackendPackArtifactAcquirer::Finish(
    BackendPackAcquisitionStatus status,
    std::string message,
    std::filesystem::path artifact_path,
    std::uint64_t resumed_bytes) {
    auto progress = GetProgress();
    progress.stage = status == BackendPackAcquisitionStatus::Downloaded ||
                             status == BackendPackAcquisitionStatus::AlreadyPresent
                         ? BackendPackAcquisitionStage::Complete
                         : BackendPackAcquisitionStage::Failed;
    progress.message = message;
    SetProgress(progress);
    return {status, std::move(message), std::move(artifact_path), resumed_bytes};
}

void BackendPackArtifactAcquirer::SetProgress(
    BackendPackAcquisitionProgress progress) {
    {
        std::lock_guard<std::mutex> lock(progress_mutex_);
        progress_ = progress;
    }
    if (observer_) observer_(progress);
}

const char* BackendPackAcquisitionStatusName(
    BackendPackAcquisitionStatus status) {
    switch (status) {
        case BackendPackAcquisitionStatus::Downloaded: return "downloaded";
        case BackendPackAcquisitionStatus::AlreadyPresent:
            return "already_present";
        case BackendPackAcquisitionStatus::Busy: return "busy";
        case BackendPackAcquisitionStatus::InvalidRequest:
            return "invalid_request";
        case BackendPackAcquisitionStatus::DiskBudgetExceeded:
            return "disk_budget_exceeded";
        case BackendPackAcquisitionStatus::SourceFailure:
            return "source_failure";
        case BackendPackAcquisitionStatus::FilesystemFailure:
            return "filesystem_failure";
        case BackendPackAcquisitionStatus::IntegrityFailure:
            return "integrity_failure";
        case BackendPackAcquisitionStatus::Interrupted: return "interrupted";
        default: return "unknown";
    }
}

const char* BackendPackAcquisitionStageName(
    BackendPackAcquisitionStage stage) {
    switch (stage) {
        case BackendPackAcquisitionStage::Idle: return "idle";
        case BackendPackAcquisitionStage::Preparing: return "preparing";
        case BackendPackAcquisitionStage::Transferring: return "transferring";
        case BackendPackAcquisitionStage::Verifying: return "verifying";
        case BackendPackAcquisitionStage::Publishing: return "publishing";
        case BackendPackAcquisitionStage::Complete: return "complete";
        case BackendPackAcquisitionStage::Failed: return "failed";
        default: return "unknown";
    }
}

}  // namespace cyxwiz::runtime
