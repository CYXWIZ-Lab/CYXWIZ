#pragma once
#include <cstddef>
#include <filesystem>
#include <functional>
#include <string>
#include <memory>
namespace arrow { class Table; }

namespace cyxwiz {
struct ArchiveTextLimits {
    size_t archive_bytes = 64 * 1024 * 1024;
    size_t member_bytes = 16 * 1024 * 1024;
    size_t entries = 10000;
    size_t total_text_bytes = 64 * 1024 * 1024;
};

// Exact, case-sensitive ZIP member selection. Returns original UTF-8 bytes.
// Rejects ambiguous names, links, encryption, NUL and invalid UTF-8. Throws
// std::runtime_error on failure; no partial result or filesystem extraction.
std::string ReadZipTextMember(const std::filesystem::path& path,
                             const std::string& member,
                             const ArchiveTextLimits& limits = {},
                             const std::function<bool()>& cancelled = {});
// One document row: source_path, archive_sha256, member_path, member_sha256,
// document_id and text. Hashes cover the same bytes used for parsing.
std::shared_ptr<arrow::Table> LoadZipTextTable(
    const std::filesystem::path& path, const std::string& member,
    const ArchiveTextLimits& limits = {},
    const std::function<bool()>& cancelled = {});
// LF/CRLF-separated exact paths, at most 4096 selections / 65536 bytes.
// Selection order is preserved; duplicate/empty/missing selections fail.
std::shared_ptr<arrow::Table> LoadZipTextSelection(
    const std::filesystem::path& path, const std::string& selection,
    const ArchiveTextLimits& limits = {},
    const std::function<bool()>& cancelled = {});
} // namespace cyxwiz
