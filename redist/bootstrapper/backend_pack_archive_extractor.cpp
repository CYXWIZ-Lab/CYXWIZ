#include "backend_pack_archive_extractor.h"
#include "backend_pack_hash.h"
#include "backend_pack_path.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <archive.h>
#include <archive_entry.h>

#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <utility>

namespace cyxwiz::runtime {
namespace {

struct ArchiveCloser {
    void operator()(archive* value) const {
        if (value) archive_read_free(value);
    }
};
using ArchiveHandle = std::unique_ptr<archive, ArchiveCloser>;

std::string ArchiveError(archive* value, const char* action) {
    const char* detail = archive_error_string(value);
    return std::string(action) + (detail ? ": " + std::string(detail) : "");
}

class RemoveIncompleteExtraction {
public:
    explicit RemoveIncompleteExtraction(std::filesystem::path path)
        : path_(std::move(path)) {}
    ~RemoveIncompleteExtraction() {
        if (keep_) return;
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    void Keep() { keep_ = true; }

private:
    std::filesystem::path path_;
    bool keep_ = false;
};

}  // namespace

BackendPackArchiveExtractor::BackendPackArchiveExtractor(
    BackendPackExtractionObserver observer)
    : observer_(std::move(observer)) {}

BackendPackExtractionResult BackendPackArchiveExtractor::Extract(
    const std::filesystem::path& archive_path,
    const VerifiedBackendPackManifest& manifest,
    const std::filesystem::path& destination,
    std::uint64_t disk_budget_bytes) {
    std::unique_lock<std::mutex> extraction_lock(
        extraction_mutex_, std::try_to_lock);
    if (!extraction_lock.owns_lock()) {
        return {BackendPackExtractionStatus::Busy,
                "An archive extraction is already running"};
    }
    cancel_requested_.store(false);
    BackendPackExtractionProgress progress;
    progress.stage = BackendPackExtractionStage::Preparing;
    progress.pack_id = manifest.pack_id;
    progress.archive_path = archive_path;
    progress.destination = destination;
    progress.component_count = manifest.components.size();
    progress.message = "Validating signed archive identity";
    SetProgress(progress);

    if (archive_path.empty() || !archive_path.is_absolute() ||
        destination.empty() || !destination.is_absolute() ||
        destination.filename().empty() || manifest.pack_id.empty() ||
        manifest.components.empty() || manifest.archive.size == 0 ||
        archive_path.filename().string() != manifest.archive.file_name ||
        !IsLowercaseSha256(manifest.archive.sha256)) {
        return Finish(
            BackendPackExtractionStatus::InvalidRequest,
            "Archive, manifest identity, and extraction destination are required");
    }
    std::error_code filesystem_error;
    const auto archive_status =
        std::filesystem::symlink_status(archive_path, filesystem_error);
    if (filesystem_error ||
        !std::filesystem::is_regular_file(archive_status) ||
        std::filesystem::is_symlink(archive_status) ||
        std::filesystem::file_size(archive_path, filesystem_error) !=
            manifest.archive.size || filesystem_error) {
        return Finish(
            BackendPackExtractionStatus::IntegrityFailure,
            "Archive file is missing, linked, or differs from signed size");
    }
    std::string error;
    std::string archive_digest;
    if (!Sha256File(archive_path, archive_digest, error) ||
        archive_digest != manifest.archive.sha256) {
        return Finish(
            BackendPackExtractionStatus::IntegrityFailure,
            error.empty() ? "Archive SHA-256 differs from signed metadata" :
                            error);
    }

    std::uint64_t total_bytes = 0;
    std::map<std::string, const VerifiedPackComponent*> expected;
    for (const auto& component : manifest.components) {
        if (!IsCanonicalBackendPackRelativePath(component.relative_path) ||
            !IsLowercaseSha256(component.sha256) ||
            component.size > static_cast<std::uint64_t>(
                                 std::numeric_limits<la_int64_t>::max()) ||
            total_bytes > std::numeric_limits<std::uint64_t>::max() -
                              component.size ||
            !expected.emplace(
                FoldBackendPackPath(component.relative_path), &component).second) {
            return Finish(
                BackendPackExtractionStatus::InvalidRequest,
                "Signed component inventory is unsafe or overflows byte limits");
        }
        total_bytes += component.size;
    }
    progress.total_bytes = total_bytes;
    SetProgress(progress);
    if (disk_budget_bytes > 0 && total_bytes > disk_budget_bytes) {
        return Finish(
            BackendPackExtractionStatus::DiskBudgetExceeded,
            "Extracted components exceed the approved disk budget");
    }
    if (std::filesystem::exists(destination, filesystem_error)) {
        return Finish(
            BackendPackExtractionStatus::FilesystemFailure,
            "Extraction destination already exists");
    }
    std::filesystem::create_directories(
        destination.parent_path(), filesystem_error);
    if (filesystem_error) {
        return Finish(
            BackendPackExtractionStatus::FilesystemFailure,
            "Cannot create extraction parent directory: " +
                filesystem_error.message());
    }
    const auto disk = std::filesystem::space(
        destination.parent_path(), filesystem_error);
    if (filesystem_error || disk.available < total_bytes) {
        return Finish(
            BackendPackExtractionStatus::DiskBudgetExceeded,
            "Insufficient free space for extracted components");
    }
    if (!std::filesystem::create_directory(destination, filesystem_error) ||
        filesystem_error) {
        return Finish(
            BackendPackExtractionStatus::FilesystemFailure,
            "Cannot create a new extraction directory");
    }
    RemoveIncompleteExtraction cleanup(destination);

    ArchiveHandle archive_reader(archive_read_new());
    if (!archive_reader ||
        archive_read_support_filter_none(archive_reader.get()) != ARCHIVE_OK ||
        archive_read_support_format_zip(archive_reader.get()) != ARCHIVE_OK ||
#ifdef _WIN32
        archive_read_open_filename_w(
            archive_reader.get(), archive_path.c_str(), 1024 * 1024) !=
#else
        archive_read_open_filename(
            archive_reader.get(), archive_path.c_str(), 1024 * 1024) !=
#endif
            ARCHIVE_OK) {
        return Finish(
            BackendPackExtractionStatus::ArchiveFailure,
            archive_reader
                ? ArchiveError(archive_reader.get(), "Cannot open ZIP archive")
                : "Cannot allocate ZIP reader");
    }

    progress.stage = BackendPackExtractionStage::Extracting;
    progress.message = "Extracting exact signed component inventory";
    SetProgress(progress);
    std::set<std::string> observed;
    archive_entry* entry = nullptr;
    for (;;) {
        const int next = archive_read_next_header(
            archive_reader.get(), &entry);
        if (next == ARCHIVE_EOF) break;
        if (next != ARCHIVE_OK) {
            return Finish(
                BackendPackExtractionStatus::ArchiveFailure,
                ArchiveError(archive_reader.get(), "Cannot read ZIP entry"));
        }
        if (cancel_requested_.load()) {
            return Finish(
                BackendPackExtractionStatus::Interrupted,
                "Archive extraction cancelled");
        }
        const char* raw_path = archive_entry_pathname_utf8(entry);
        if (!raw_path) raw_path = archive_entry_pathname(entry);
        const std::string relative = raw_path ? raw_path : "";
        const auto folded = FoldBackendPackPath(relative);
        const auto expected_entry = expected.find(folded);
        const auto declared_size = archive_entry_size(entry);
        if (!IsCanonicalBackendPackRelativePath(relative) ||
            expected_entry == expected.end() ||
            relative != expected_entry->second->relative_path ||
            !observed.insert(folded).second ||
            archive_entry_filetype(entry) != AE_IFREG ||
            archive_entry_symlink(entry) != nullptr ||
            archive_entry_hardlink(entry) != nullptr || declared_size < 0 ||
            static_cast<std::uint64_t>(declared_size) !=
                expected_entry->second->size) {
            return Finish(
                BackendPackExtractionStatus::IntegrityFailure,
                "ZIP entries differ from the exact signed component inventory");
        }
        const auto target =
            destination / BackendPackNativeRelativePath(relative);
        std::filesystem::create_directories(
            target.parent_path(), filesystem_error);
        if (filesystem_error) {
            return Finish(
                BackendPackExtractionStatus::FilesystemFailure,
                "Cannot create extracted component directory");
        }
        std::ofstream output(target, std::ios::binary | std::ios::trunc);
        if (!output) {
            return Finish(
                BackendPackExtractionStatus::FilesystemFailure,
                "Cannot create extracted component file");
        }
        std::uint64_t written = 0;
        for (;;) {
            const void* block = nullptr;
            std::size_t block_size = 0;
            la_int64_t block_offset = 0;
            const int data = archive_read_data_block(
                archive_reader.get(), &block, &block_size, &block_offset);
            if (data == ARCHIVE_EOF) break;
            if (data != ARCHIVE_OK || block_offset < 0 ||
                static_cast<std::uint64_t>(block_offset) != written ||
                block_size > expected_entry->second->size - written) {
                return Finish(
                    BackendPackExtractionStatus::ArchiveFailure,
                    "ZIP component data is sparse, corrupt, or exceeds signed size");
            }
            output.write(
                static_cast<const char*>(block),
                static_cast<std::streamsize>(block_size));
            if (!output) {
                return Finish(
                    BackendPackExtractionStatus::FilesystemFailure,
                    "Cannot write extracted component data");
            }
            written += block_size;
            progress.completed_bytes += block_size;
            SetProgress(progress);
            if (cancel_requested_.load()) {
                return Finish(
                    BackendPackExtractionStatus::Interrupted,
                    "Archive extraction cancelled");
            }
        }
        output.flush();
        if (!output || written != expected_entry->second->size) {
            return Finish(
                BackendPackExtractionStatus::IntegrityFailure,
                "Extracted component differs from signed byte size");
        }
        ++progress.component_index;
        SetProgress(progress);
    }
    if (observed.size() != expected.size()) {
        return Finish(
            BackendPackExtractionStatus::IntegrityFailure,
            "ZIP archive is missing signed components");
    }

    progress.stage = BackendPackExtractionStage::Verifying;
    progress.message = "Verifying every extracted component hash";
    SetProgress(progress);
    for (const auto& [folded, component] : expected) {
        (void)folded;
        const auto path =
            destination /
            BackendPackNativeRelativePath(component->relative_path);
        std::string digest;
        if (!Sha256File(path, digest, error) || digest != component->sha256) {
            return Finish(
                BackendPackExtractionStatus::IntegrityFailure,
                error.empty()
                    ? "Extracted component SHA-256 differs from signed metadata"
                    : error);
        }
    }
    cleanup.Keep();
    return Finish(
        BackendPackExtractionStatus::Extracted,
        "Archive extracted and verified", destination);
}

void BackendPackArchiveExtractor::Cancel() {
    cancel_requested_.store(true);
}

BackendPackExtractionProgress BackendPackArchiveExtractor::GetProgress() const {
    std::lock_guard<std::mutex> lock(progress_mutex_);
    return progress_;
}

BackendPackExtractionResult BackendPackArchiveExtractor::Finish(
    BackendPackExtractionStatus status,
    std::string message,
    std::filesystem::path extracted_directory) {
    auto progress = GetProgress();
    progress.stage = status == BackendPackExtractionStatus::Extracted
        ? BackendPackExtractionStage::Complete
        : BackendPackExtractionStage::Failed;
    progress.message = message;
    SetProgress(progress);
    return {status, std::move(message), std::move(extracted_directory)};
}

void BackendPackArchiveExtractor::SetProgress(
    BackendPackExtractionProgress progress) {
    {
        std::lock_guard<std::mutex> lock(progress_mutex_);
        progress_ = progress;
    }
    if (observer_) observer_(progress);
}

const char* BackendPackExtractionStatusName(
    BackendPackExtractionStatus status) {
    switch (status) {
        case BackendPackExtractionStatus::Extracted: return "extracted";
        case BackendPackExtractionStatus::Busy: return "busy";
        case BackendPackExtractionStatus::InvalidRequest:
            return "invalid_request";
        case BackendPackExtractionStatus::DiskBudgetExceeded:
            return "disk_budget_exceeded";
        case BackendPackExtractionStatus::IntegrityFailure:
            return "integrity_failure";
        case BackendPackExtractionStatus::ArchiveFailure:
            return "archive_failure";
        case BackendPackExtractionStatus::FilesystemFailure:
            return "filesystem_failure";
        case BackendPackExtractionStatus::Interrupted: return "interrupted";
        default: return "unknown";
    }
}

const char* BackendPackExtractionStageName(
    BackendPackExtractionStage stage) {
    switch (stage) {
        case BackendPackExtractionStage::Idle: return "idle";
        case BackendPackExtractionStage::Preparing: return "preparing";
        case BackendPackExtractionStage::Extracting: return "extracting";
        case BackendPackExtractionStage::Verifying: return "verifying";
        case BackendPackExtractionStage::Complete: return "complete";
        case BackendPackExtractionStage::Failed: return "failed";
        default: return "unknown";
    }
}

}  // namespace cyxwiz::runtime
