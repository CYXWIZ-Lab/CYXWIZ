#pragma once

#include <cstddef>
#include <filesystem>

namespace cyxwiz {

// File-ingress budgets, not a process RSS quota. Tests may use smaller limits.
struct XlsxArchiveLimits {
    size_t compressed_bytes = 64 * 1024 * 1024;
    size_t expanded_bytes = 64 * 1024 * 1024;
    size_t part_bytes = 16 * 1024 * 1024;
    size_t entries = 512;
    size_t xml_markup = 1000000;
    size_t xml_depth = 128;
};

// Owns a validated, stored (uncompressed) ZIP snapshot. No package entry is
// extracted as a filesystem path. The source is never reopened by OpenXLSX.
// V1 accepts UTF-8 XML/.rels parts only; binary attachments reject explicitly.
class PreparedXlsxArchive {
public:
    explicit PreparedXlsxArchive(const std::filesystem::path& source,
                                 const XlsxArchiveLimits& limits = {});
    ~PreparedXlsxArchive();
    PreparedXlsxArchive(const PreparedXlsxArchive&) = delete;
    PreparedXlsxArchive& operator=(const PreparedXlsxArchive&) = delete;
    const std::filesystem::path& Path() const { return staged_path_; }

private:
    void Cleanup() noexcept;
    std::filesystem::path directory_;
    std::filesystem::path staged_path_;
};

} // namespace cyxwiz
