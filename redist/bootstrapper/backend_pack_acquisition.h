#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>

namespace cyxwiz::runtime {

bool ResolveHttpsBackendPackArchiveUrl(
    std::string_view manifest_url,
    std::string_view archive_file_name,
    std::string& archive_url,
    std::string& error);
bool ResolveOfflineBackendPackArchivePath(
    const std::filesystem::path& manifest_path,
    std::string_view archive_file_name,
    std::filesystem::path& archive_path,
    std::string& error);

using BackendPackArtifactChunk =
    std::function<bool(const char*, std::size_t, std::string&)>;
using BackendPackArtifactCancelCheck = std::function<bool()>;

class BackendPackArtifactSource {
public:
    virtual ~BackendPackArtifactSource() = default;
    virtual std::string Description() const = 0;
    virtual bool TransferFrom(
        std::uint64_t offset,
        std::uint64_t expected_size,
        const BackendPackArtifactChunk& consume,
        const BackendPackArtifactCancelCheck& cancelled,
        std::string& error) = 0;
};

class OfflineBackendPackArtifactSource final
    : public BackendPackArtifactSource {
public:
    explicit OfflineBackendPackArtifactSource(std::filesystem::path path);
    std::string Description() const override;
    bool TransferFrom(
        std::uint64_t offset,
        std::uint64_t expected_size,
        const BackendPackArtifactChunk& consume,
        const BackendPackArtifactCancelCheck& cancelled,
        std::string& error) override;

private:
    std::filesystem::path path_;
};

class HttpsBackendPackArtifactSource final
    : public BackendPackArtifactSource {
public:
    explicit HttpsBackendPackArtifactSource(
        std::string url,
        std::chrono::milliseconds timeout = std::chrono::seconds(60));
    std::string Description() const override;
    bool TransferFrom(
        std::uint64_t offset,
        std::uint64_t expected_size,
        const BackendPackArtifactChunk& consume,
        const BackendPackArtifactCancelCheck& cancelled,
        std::string& error) override;

private:
    std::string url_;
    std::chrono::milliseconds timeout_;
};

enum class BackendPackAcquisitionStage {
    Idle,
    Preparing,
    Transferring,
    Verifying,
    Publishing,
    Complete,
    Failed
};

enum class BackendPackAcquisitionStatus {
    Downloaded,
    AlreadyPresent,
    Busy,
    InvalidRequest,
    DiskBudgetExceeded,
    SourceFailure,
    FilesystemFailure,
    IntegrityFailure,
    Interrupted
};

struct BackendPackAcquisitionProgress {
    BackendPackAcquisitionStage stage = BackendPackAcquisitionStage::Idle;
    std::string source;
    std::filesystem::path destination;
    std::uint64_t resumed_bytes = 0;
    std::uint64_t completed_bytes = 0;
    std::uint64_t total_bytes = 0;
    std::string message;
};

struct BackendPackAcquisitionResult {
    BackendPackAcquisitionStatus status =
        BackendPackAcquisitionStatus::InvalidRequest;
    std::string message;
    std::filesystem::path artifact_path;
    std::uint64_t resumed_bytes = 0;
};

using BackendPackAcquisitionObserver =
    std::function<void(const BackendPackAcquisitionProgress&)>;

class BackendPackArtifactAcquirer {
public:
    explicit BackendPackArtifactAcquirer(
        BackendPackAcquisitionObserver observer = {});

    BackendPackAcquisitionResult Acquire(
        BackendPackArtifactSource& source,
        const std::filesystem::path& destination,
        std::uint64_t expected_size,
        std::string expected_sha256,
        std::uint64_t disk_budget_bytes);
    void Cancel();
    BackendPackAcquisitionProgress GetProgress() const;

private:
    BackendPackAcquisitionResult Finish(
        BackendPackAcquisitionStatus status,
        std::string message,
        std::filesystem::path artifact_path = {},
        std::uint64_t resumed_bytes = 0);
    void SetProgress(BackendPackAcquisitionProgress progress);

    BackendPackAcquisitionObserver observer_;
    std::atomic<bool> cancel_requested_{false};
    std::mutex acquisition_mutex_;
    mutable std::mutex progress_mutex_;
    BackendPackAcquisitionProgress progress_;
};

const char* BackendPackAcquisitionStatusName(
    BackendPackAcquisitionStatus status);
const char* BackendPackAcquisitionStageName(
    BackendPackAcquisitionStage stage);

}  // namespace cyxwiz::runtime
