#include "backend_pack_acquisition.h"
#include "backend_pack_archive_extractor.h"
#include "backend_pack_hash.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <archive.h>
#include <archive_entry.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace cyxwiz::runtime;

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        root_ = std::filesystem::temp_directory_path() /
            ("cyxwiz-pack-delivery-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root_);
    }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(root_, error);
    }
    const std::filesystem::path& Path() const { return root_; }

private:
    std::filesystem::path root_;
};

class MemorySource final : public BackendPackArtifactSource {
public:
    explicit MemorySource(
        std::string bytes,
        std::uint64_t fail_after = std::numeric_limits<std::uint64_t>::max())
        : bytes_(std::move(bytes)), fail_after_(fail_after) {}

    std::string Description() const override { return "memory:test"; }

    bool TransferFrom(
        std::uint64_t offset,
        std::uint64_t expected_size,
        const BackendPackArtifactChunk& consume,
        const BackendPackArtifactCancelCheck& cancelled,
        std::string& error) override {
        opened_at = offset;
        if (expected_size != bytes_.size() || offset > bytes_.size()) {
            error = "memory source size mismatch";
            return false;
        }
        std::uint64_t cursor = offset;
        while (cursor < bytes_.size()) {
            if (cancelled()) {
                error = "memory source cancelled";
                return false;
            }
            if (cursor >= fail_after_) {
                error = "simulated source interruption";
                return false;
            }
            const auto size = static_cast<std::size_t>(
                std::min<std::uint64_t>(65536, bytes_.size() - cursor));
            if (!consume(bytes_.data() + cursor, size, error)) return false;
            cursor += size;
        }
        return true;
    }

    std::uint64_t opened_at = 0;

private:
    std::string bytes_;
    std::uint64_t fail_after_;
};

struct ArchiveItem {
    std::string path;
    std::string bytes;
    bool symlink = false;
};

struct ArchiveWriteCloser {
    void operator()(archive* value) const {
        if (!value) return;
        archive_write_close(value);
        archive_write_free(value);
    }
};

bool WriteZip(
    const std::filesystem::path& path,
    const std::vector<ArchiveItem>& items) {
    std::filesystem::create_directories(path.parent_path());
    std::unique_ptr<archive, ArchiveWriteCloser> writer(archive_write_new());
    if (!writer || archive_write_set_format_zip(writer.get()) != ARCHIVE_OK ||
        archive_write_open_filename_w(writer.get(), path.c_str()) != ARCHIVE_OK) {
        return false;
    }
    for (const auto& item : items) {
        archive_entry* raw_entry = archive_entry_new();
        if (!raw_entry) return false;
        const auto free_entry = [](archive_entry* entry) {
            archive_entry_free(entry);
        };
        std::unique_ptr<archive_entry, decltype(free_entry)> entry(
            raw_entry, free_entry);
        archive_entry_set_pathname_utf8(entry.get(), item.path.c_str());
        archive_entry_set_perm(entry.get(), 0644);
        if (item.symlink) {
            archive_entry_set_filetype(entry.get(), AE_IFLNK);
            archive_entry_set_symlink(entry.get(), "outside");
            archive_entry_set_size(entry.get(), 0);
        } else {
            archive_entry_set_filetype(entry.get(), AE_IFREG);
            archive_entry_set_size(
                entry.get(), static_cast<la_int64_t>(item.bytes.size()));
        }
        if (archive_write_header(writer.get(), entry.get()) != ARCHIVE_OK)
            return false;
        if (!item.symlink && !item.bytes.empty() &&
            archive_write_data(
                writer.get(), item.bytes.data(), item.bytes.size()) !=
                static_cast<la_ssize_t>(item.bytes.size())) {
            return false;
        }
    }
    const int closed = archive_write_close(writer.get());
    archive_write_free(writer.release());
    return closed == ARCHIVE_OK;
}

std::string Hash(const std::string& bytes) {
    std::string digest;
    std::string error;
    return Sha256Bytes(bytes, digest, error) ? digest : "";
}

VerifiedBackendPackManifest Manifest(
    const std::filesystem::path& archive_path,
    std::vector<VerifiedPackComponent> components) {
    VerifiedBackendPackManifest manifest;
    manifest.pack_id = "opencl-v1";
    manifest.backend = "opencl";
    manifest.runtime_set_id = "set-v1";
    manifest.companion_base_id = "base-v1";
    manifest.components = std::move(components);
    manifest.archive.file_name = archive_path.filename().string();
    std::error_code filesystem_error;
    manifest.archive.size =
        std::filesystem::file_size(archive_path, filesystem_error);
    std::string error;
    Sha256File(archive_path, manifest.archive.sha256, error);
    return manifest;
}

bool Expect(bool condition, const std::string& message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}

}  // namespace

int main() {
    TemporaryDirectory temporary;
    std::string bytes(2 * 1024 * 1024, '\0');
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<char>(i % 251);
    }
    const auto digest = Hash(bytes);
    const auto download = temporary.Path() / "downloads" / "pack.zip";

    MemorySource interrupted(bytes, 128 * 1024);
    BackendPackArtifactAcquirer acquirer;
    auto acquired = acquirer.Acquire(
        interrupted, download, bytes.size(), digest, bytes.size());
    if (!Expect(
            acquired.status == BackendPackAcquisitionStatus::SourceFailure,
            "interrupted source did not preserve resumable state")) return 1;
    auto partial = download;
    partial += ".part";
    const auto partial_size = std::filesystem::file_size(partial);
    if (!Expect(partial_size > 0 && partial_size < bytes.size(),
                "partial artifact size is not resumable")) return 1;

    MemorySource resumed(bytes);
    acquired = acquirer.Acquire(
        resumed, download, bytes.size(), digest, bytes.size());
    if (!Expect(
            acquired.status == BackendPackAcquisitionStatus::Downloaded,
            acquired.message) ||
        !Expect(resumed.opened_at == partial_size,
                "source did not resume at the partial offset") ||
        !Expect(acquired.resumed_bytes == partial_size,
                "result omitted resumed byte count")) return 1;

    acquired = acquirer.Acquire(
        resumed, download, bytes.size(), digest, bytes.size());
    if (!Expect(
            acquired.status == BackendPackAcquisitionStatus::AlreadyPresent,
            "verified existing artifact was not reused")) return 1;

    MemorySource budget_source(bytes);
    acquired = acquirer.Acquire(
        budget_source, temporary.Path() / "budget.zip", bytes.size(), digest,
        bytes.size() - 1);
    if (!Expect(
            acquired.status == BackendPackAcquisitionStatus::DiskBudgetExceeded,
            "download disk budget was not enforced")) return 1;

    const auto hardlink_victim = temporary.Path() / "hardlink-victim.bin";
    {
        std::ofstream victim(hardlink_victim, std::ios::binary);
        victim << "must-not-change";
    }
    const auto hardlink_destination = temporary.Path() / "hardlink-pack.zip";
    auto hardlink_partial = hardlink_destination;
    hardlink_partial += ".part";
    std::error_code hardlink_error;
    std::filesystem::create_hard_link(
        hardlink_victim, hardlink_partial, hardlink_error);
    if (!Expect(!hardlink_error, "cannot create hard-link security fixture"))
        return 1;
    acquired = acquirer.Acquire(
        budget_source, hardlink_destination, bytes.size(), digest,
        bytes.size());
    std::ifstream victim(hardlink_victim, std::ios::binary);
    const std::string victim_bytes{
        std::istreambuf_iterator<char>{victim},
        std::istreambuf_iterator<char>{}};
    if (!Expect(
            acquired.status == BackendPackAcquisitionStatus::FilesystemFailure &&
                victim_bytes == "must-not-change",
            "writable partial hard link was accepted")) return 1;

    const auto offline_copy = temporary.Path() / "offline-copy.zip";
    OfflineBackendPackArtifactSource offline(download);
    acquired = acquirer.Acquire(
        offline, offline_copy, bytes.size(), digest, bytes.size());
    if (!Expect(
            acquired.status == BackendPackAcquisitionStatus::Downloaded,
            "offline source did not use the shared acquisition path")) return 1;

    const auto cancel_download = temporary.Path() / "cancel-download.zip";
    BackendPackArtifactAcquirer* cancelling_acquirer = nullptr;
    BackendPackArtifactAcquirer cancellation_acquirer(
        [&](const BackendPackAcquisitionProgress& progress) {
            if (progress.completed_bytes > 0 && cancelling_acquirer)
                cancelling_acquirer->Cancel();
        });
    cancelling_acquirer = &cancellation_acquirer;
    MemorySource cancellation_source(bytes);
    acquired = cancellation_acquirer.Acquire(
        cancellation_source, cancel_download, bytes.size(), digest,
        bytes.size());
    auto cancelled_partial = cancel_download;
    cancelled_partial += ".part";
    if (!Expect(
            acquired.status == BackendPackAcquisitionStatus::Interrupted &&
                std::filesystem::is_regular_file(cancelled_partial),
            "cancelled acquisition did not retain resumable partial state"))
        return 1;
    MemorySource cancellation_resume(bytes);
    acquired = acquirer.Acquire(
        cancellation_resume, cancel_download, bytes.size(), digest,
        bytes.size());
    if (!Expect(
            acquired.status == BackendPackAcquisitionStatus::Downloaded &&
                acquired.resumed_bytes > 0,
            "cancelled acquisition could not be resumed")) return 1;

    HttpsBackendPackArtifactSource insecure("http://example.test/pack.zip");
    acquired = acquirer.Acquire(
        insecure, temporary.Path() / "insecure.zip", bytes.size(), digest,
        bytes.size());
    if (!Expect(
            acquired.status == BackendPackAcquisitionStatus::SourceFailure,
            "insecure online source URL was accepted")) return 1;

    const std::vector<ArchiveItem> valid_items = {
        {"runtime/afopencl.dll", "plugin"},
        {"THIRD_PARTY_LICENSES/ArrayFire/LICENSE.txt", "license"}};
    const std::vector<VerifiedPackComponent> valid_components = {
        {"runtime/afopencl.dll", 6, Hash("plugin")},
        {"THIRD_PARTY_LICENSES/ArrayFire/LICENSE.txt", 7, Hash("license")}};
    const auto valid_archive =
        temporary.Path() / "valid" / "opencl-v1.zip";
    if (!Expect(WriteZip(valid_archive, valid_items),
                "cannot create valid ZIP fixture")) return 1;
    auto manifest = Manifest(valid_archive, valid_components);
    BackendPackArchiveExtractor extractor;
    auto extracted = extractor.Extract(
        valid_archive, manifest, temporary.Path() / "extract-valid", 1024);
    if (!Expect(
            extracted.status == BackendPackExtractionStatus::Extracted,
            extracted.message) ||
        !Expect(
            std::filesystem::is_regular_file(
                extracted.extracted_directory / "runtime" / "afopencl.dll"),
            "valid component was not extracted")) return 1;

    const auto extra_archive =
        temporary.Path() / "extra" / "opencl-v1.zip";
    auto extra_items = valid_items;
    extra_items.push_back({"unexpected.cmd", "bad"});
    if (!Expect(WriteZip(extra_archive, extra_items),
                "cannot create extra-entry ZIP fixture")) return 1;
    manifest = Manifest(extra_archive, valid_components);
    const auto extra_destination = temporary.Path() / "extract-extra";
    extracted = extractor.Extract(
        extra_archive, manifest, extra_destination, 1024);
    if (!Expect(
            extracted.status == BackendPackExtractionStatus::IntegrityFailure &&
                !std::filesystem::exists(extra_destination),
            "unexpected ZIP entry was accepted or staging was retained")) return 1;

    const auto traversal_archive =
        temporary.Path() / "traversal" / "opencl-v1.zip";
    if (!Expect(
            WriteZip(traversal_archive, {{"../escape.txt", "escape"}}),
            "cannot create traversal ZIP fixture")) return 1;
    manifest = Manifest(traversal_archive, valid_components);
    const auto traversal_destination = temporary.Path() / "extract-traversal";
    extracted = extractor.Extract(
        traversal_archive, manifest, traversal_destination, 1024);
    if (!Expect(
            extracted.status == BackendPackExtractionStatus::IntegrityFailure &&
                !std::filesystem::exists(temporary.Path() / "escape.txt") &&
                !std::filesystem::exists(traversal_destination),
            "ZIP traversal entry escaped or left staging behind")) return 1;

    const auto link_archive =
        temporary.Path() / "link" / "opencl-v1.zip";
    if (!Expect(
            WriteZip(link_archive, {{"runtime/afopencl.dll", "", true}}),
            "cannot create link ZIP fixture")) return 1;
    manifest = Manifest(
        link_archive,
        {{"runtime/afopencl.dll", 0, Hash("")}});
    const auto link_destination = temporary.Path() / "extract-link";
    extracted = extractor.Extract(
        link_archive, manifest, link_destination, 1024);
    if (!Expect(
            extracted.status == BackendPackExtractionStatus::IntegrityFailure &&
                !std::filesystem::exists(link_destination),
            "ZIP link entry was accepted")) return 1;

    const auto corrupt_destination = temporary.Path() / "extract-corrupt";
    manifest = Manifest(valid_archive, valid_components);
    manifest.components.front().sha256 = std::string(64, '0');
    extracted = extractor.Extract(
        valid_archive, manifest, corrupt_destination, 1024);
    if (!Expect(
            extracted.status == BackendPackExtractionStatus::IntegrityFailure &&
                !std::filesystem::exists(corrupt_destination),
            "component corruption did not remove incomplete extraction")) return 1;

    manifest = Manifest(valid_archive, valid_components);
    extracted = extractor.Extract(
        valid_archive, manifest, temporary.Path() / "extract-budget", 12);
    if (!Expect(
            extracted.status == BackendPackExtractionStatus::DiskBudgetExceeded,
            "extraction disk budget was not enforced")) return 1;

    const auto cancel_archive =
        temporary.Path() / "cancel" / "opencl-v1.zip";
    const std::string large(2 * 1024 * 1024, 'x');
    if (!Expect(
            WriteZip(cancel_archive, {{"runtime/afopencl.dll", large}}),
            "cannot create cancellation ZIP fixture")) return 1;
    manifest = Manifest(
        cancel_archive,
        {{"runtime/afopencl.dll", large.size(), Hash(large)}});
    BackendPackArchiveExtractor* cancelling_extractor = nullptr;
    BackendPackArchiveExtractor cancellation(
        [&](const BackendPackExtractionProgress& progress) {
            if (progress.completed_bytes > 0 && cancelling_extractor)
                cancelling_extractor->Cancel();
        });
    cancelling_extractor = &cancellation;
    const auto cancel_destination = temporary.Path() / "extract-cancel";
    extracted = cancellation.Extract(
        cancel_archive, manifest, cancel_destination, large.size());
    if (!Expect(
            extracted.status == BackendPackExtractionStatus::Interrupted &&
                !std::filesystem::exists(cancel_destination),
            "cancelled extraction left staging behind")) return 1;

    std::cout << "backend pack delivery contract tests passed\n";
    return 0;
}
