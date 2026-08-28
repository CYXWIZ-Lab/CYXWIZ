#pragma once

#include "backend_pack_metadata_verifier.h"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>

namespace cyxwiz::runtime {

struct VerifiedInstallerBundle;

enum class BackendPackExtractionStage {
    Idle,
    Preparing,
    Extracting,
    Verifying,
    Complete,
    Failed
};

enum class BackendPackExtractionStatus {
    Extracted,
    Busy,
    InvalidRequest,
    DiskBudgetExceeded,
    IntegrityFailure,
    ArchiveFailure,
    FilesystemFailure,
    Interrupted
};

struct BackendPackExtractionProgress {
    BackendPackExtractionStage stage = BackendPackExtractionStage::Idle;
    std::string pack_id;
    std::filesystem::path archive_path;
    std::filesystem::path destination;
    std::size_t component_index = 0;
    std::size_t component_count = 0;
    std::uint64_t completed_bytes = 0;
    std::uint64_t total_bytes = 0;
    std::string message;
};

struct BackendPackExtractionResult {
    BackendPackExtractionStatus status =
        BackendPackExtractionStatus::InvalidRequest;
    std::string message;
    std::filesystem::path extracted_directory;
};

using BackendPackExtractionObserver =
    std::function<void(const BackendPackExtractionProgress&)>;

class BackendPackArchiveExtractor {
public:
    explicit BackendPackArchiveExtractor(
        BackendPackExtractionObserver observer = {});

    BackendPackExtractionResult Extract(
        const std::filesystem::path& archive_path,
        const VerifiedBackendPackManifest& manifest,
        const std::filesystem::path& destination,
        std::uint64_t disk_budget_bytes);
    BackendPackExtractionResult ExtractInstallerBundle(
        const std::filesystem::path& archive_path,
        const VerifiedInstallerBundle& bundle,
        const std::filesystem::path& destination,
        std::uint64_t disk_budget_bytes);
    void Cancel();
    BackendPackExtractionProgress GetProgress() const;

private:
    BackendPackExtractionResult Finish(
        BackendPackExtractionStatus status,
        std::string message,
        std::filesystem::path extracted_directory = {});
    void SetProgress(BackendPackExtractionProgress progress);

    BackendPackExtractionObserver observer_;
    std::atomic<bool> cancel_requested_{false};
    std::mutex extraction_mutex_;
    mutable std::mutex progress_mutex_;
    BackendPackExtractionProgress progress_;
};

const char* BackendPackExtractionStatusName(
    BackendPackExtractionStatus status);
const char* BackendPackExtractionStageName(
    BackendPackExtractionStage stage);

}  // namespace cyxwiz::runtime
