#include "xlsx_archive_guard.h"

#include <archive.h>
#include <archive_entry.h>
#include <arrow/util/utf8.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace cyxwiz {
namespace {
void RequireArchive(int status, archive* handle, const char* operation) {
    if (status != ARCHIVE_OK) {
        const char* detail = archive_error_string(handle);
        throw std::runtime_error(std::string(operation) + ": " + (detail ? detail : "archive error"));
    }
}

void CheckXml(const std::string& xml, const XlsxArchiveLimits& limits, size_t& markup) {
    static const bool utf8_initialized = [] { arrow::util::InitializeUTF8(); return true; }();
    (void)utf8_initialized;
    if (xml.find('\0') != std::string::npos || !arrow::util::ValidateUTF8(xml))
        throw std::runtime_error("XLSX parts must be UTF-8 XML (NUL/UTF-16/UTF-32 unsupported)");
    size_t depth = 0;
    size_t offset = 0;
    while ((offset = xml.find('<', offset)) != std::string::npos) {
        if (++markup > limits.xml_markup) throw std::runtime_error("XLSX XML markup limit exceeded");
        const std::string_view tail(xml.data() + offset, xml.size() - offset);
        const auto skip_until = [&](std::string_view end, size_t start) {
            const auto found = xml.find(end, offset + start);
            if (found == std::string::npos) throw std::runtime_error("unterminated XLSX XML markup");
            return found + end.size();
        };
        if (tail.starts_with("<!--")) { offset = skip_until("-->", 4); continue; }
        if (tail.starts_with("<![CDATA[")) { offset = skip_until("]]>", 9); continue; }
        if (tail.starts_with("<?")) { offset = skip_until("?>", 2); continue; }
        if (tail.starts_with("<!")) throw std::runtime_error("XLSX XML declarations/DTDs are unsupported");
        const bool closing = tail.starts_with("</");
        size_t end = offset + 1;
        char quote = 0;
        for (; end < xml.size(); ++end) {
            const char c = xml[end];
            if (quote) { if (c == quote) quote = 0; }
            else if (c == '\'' || c == '"') quote = c;
            else if (c == '>') break;
            else if (c == '<') throw std::runtime_error("invalid XLSX XML tag");
        }
        if (end == xml.size()) throw std::runtime_error("unterminated XLSX XML tag");
        if (closing) {
            if (depth == 0) throw std::runtime_error("unbalanced XLSX XML tags");
            --depth;
        } else if (xml[end - 1] != '/') {
            if (++depth > limits.xml_depth) throw std::runtime_error("XLSX XML nesting limit exceeded");
        }
        offset = end + 1;
    }
    if (depth != 0) throw std::runtime_error("unbalanced XLSX XML tags");
    // This is a conservative resource scan, not a replacement XML parser.
}

void CheckPartName(const std::string& name) {
    if (name.empty() || name.size() > 256 || name.front() == '/' ||
        name.find_first_of("\\:") != std::string::npos)
        throw std::runtime_error("unsafe XLSX package part name");
    size_t start = 0;
    while (start < name.size()) {
        const auto slash = name.find('/', start);
        const auto segment = name.substr(start, slash == std::string::npos ? slash : slash - start);
        if (segment.empty() || segment == "." || segment == "..")
            throw std::runtime_error("unsafe XLSX package part name");
        if (slash == std::string::npos) break;
        start = slash + 1;
    }
}

void CheckZipMethod(archive* reader, const std::vector<char>& snapshot) {
    // No outer filters are enabled: the local ZIP header offset is also an
    // offset into our immutable compressed snapshot. Check before decoding.
    const auto position = archive_read_header_position(reader);
    if (position < 0 || static_cast<uint64_t>(position) > snapshot.size() ||
        snapshot.size() - static_cast<size_t>(position) < 30)
        throw std::runtime_error("invalid XLSX ZIP header offset");
    const char* header = snapshot.data() + static_cast<size_t>(position);
    if (std::string_view(header, 4) != std::string_view("PK\x03\x04", 4))
        throw std::runtime_error("invalid XLSX local ZIP header");
    const size_t name_size = static_cast<unsigned char>(header[26]) |
        (static_cast<unsigned>(static_cast<unsigned char>(header[27])) << 8);
    if (name_size > snapshot.size() - static_cast<size_t>(position) - 30)
        throw std::runtime_error("invalid XLSX ZIP part-name length");
    const std::string raw_name(header + 30, name_size);
    if (raw_name.find('\0') != std::string::npos)
        throw std::runtime_error("unsafe XLSX package part name");
    CheckPartName(raw_name); // Validate before LibArchive's path normalization too.
    const auto method = static_cast<unsigned char>(header[8]) |
        (static_cast<unsigned>(static_cast<unsigned char>(header[9])) << 8);
    if (method != 0 && method != 8)
        throw std::runtime_error("XLSX ZIP compression must be stored or Deflate");
}

std::vector<char> ReadSnapshot(const std::filesystem::path& source, size_t limit) {
    if (!std::filesystem::is_regular_file(source)) throw std::runtime_error("XLSX input must be a regular file");
    if (std::filesystem::file_size(source) > limit) throw std::runtime_error("XLSX compressed-file limit exceeded");
    std::ifstream input(source, std::ios::binary);
    if (!input) throw std::runtime_error("could not read XLSX input");
    std::vector<char> bytes;
    std::array<char, 32768> buffer;
    while (input.read(buffer.data(), buffer.size()) || input.gcount() != 0) {
        const auto count = static_cast<size_t>(input.gcount());
        if (count > limit - bytes.size()) throw std::runtime_error("XLSX compressed-file limit exceeded");
        bytes.insert(bytes.end(), buffer.data(), buffer.data() + count);
    }
    if (input.bad() || !input.eof()) throw std::runtime_error("XLSX source read failed");
    return bytes;
}

void Repack(const std::filesystem::path& source, const std::filesystem::path& staged,
            const XlsxArchiveLimits& limits) {
    const auto snapshot = ReadSnapshot(source, limits.compressed_bytes);
    std::unique_ptr<archive, decltype(&archive_read_free)> reader(archive_read_new(), archive_read_free);
    std::unique_ptr<archive, decltype(&archive_write_free)> writer(archive_write_new(), archive_write_free);
    if (!reader || !writer) throw std::bad_alloc();
    RequireArchive(archive_read_support_format_zip_streamable(reader.get()), reader.get(), "ZIP reader setup");
    RequireArchive(archive_read_open_memory(reader.get(), snapshot.data(), snapshot.size()), reader.get(), "ZIP open");
    RequireArchive(archive_write_set_format_zip(writer.get()), writer.get(), "ZIP writer setup");
    RequireArchive(archive_write_zip_set_compression_store(writer.get()), writer.get(), "ZIP store setup");
#ifdef _WIN32
    RequireArchive(archive_write_open_filename_w(writer.get(), staged.c_str()), writer.get(), "staging open");
#else
    RequireArchive(archive_write_open_filename(writer.get(), staged.c_str()), writer.get(), "staging open");
#endif
    size_t expanded = 0, entries = 0, markup = 0;
    std::set<std::string> names;
    archive_entry* entry = nullptr;
    int status;
    std::array<char, 32768> buffer;
    while ((status = archive_read_next_header(reader.get(), &entry)) != ARCHIVE_EOF) {
        RequireArchive(status, reader.get(), "ZIP entry");
        CheckZipMethod(reader.get(), snapshot);
        if (++entries > limits.entries) throw std::runtime_error("XLSX entry-count limit exceeded");
        const char* raw_name = archive_entry_pathname_utf8(entry);
        if (!raw_name) throw std::runtime_error("XLSX part name is not UTF-8");
        const std::string name(raw_name);
        CheckPartName(name);
        if (!names.insert(name).second) throw std::runtime_error("duplicate XLSX package part");
        if (archive_entry_is_encrypted(entry) || archive_entry_symlink(entry) || archive_entry_hardlink(entry))
            throw std::runtime_error("encrypted or linked XLSX package part is unsupported");
        const auto kind = archive_entry_filetype(entry);
        if (kind != AE_IFREG && kind != AE_IFDIR) throw std::runtime_error("unsupported XLSX package entry type");
        if (kind == AE_IFREG && !name.ends_with(".xml") && !name.ends_with(".rels"))
            throw std::runtime_error("binary/non-XML XLSX attachments are unsupported");
        if (archive_entry_size_is_set(entry) && (archive_entry_size(entry) < 0 ||
            static_cast<uint64_t>(archive_entry_size(entry)) > limits.part_bytes))
            throw std::runtime_error("XLSX part-byte limit exceeded");
        std::string part;
        for (;;) {
            const auto count = archive_read_data(reader.get(), buffer.data(), buffer.size());
            if (count < 0) throw std::runtime_error("XLSX ZIP decompression or checksum failed");
            if (count == 0) break;
            const auto size = static_cast<size_t>(count);
            if (kind == AE_IFDIR) throw std::runtime_error("XLSX directory entry contains data");
            if (size > limits.part_bytes - part.size()) throw std::runtime_error("XLSX part-byte limit exceeded");
            if (size > limits.expanded_bytes - expanded) throw std::runtime_error("XLSX expanded-byte limit exceeded");
            expanded += size;
            part.append(buffer.data(), size);
        }
        if (kind == AE_IFDIR) continue;
        CheckXml(part, limits, markup);
        std::unique_ptr<archive_entry, decltype(&archive_entry_free)> clean(archive_entry_new(), archive_entry_free);
        if (!clean) throw std::bad_alloc();
        archive_entry_set_pathname_utf8(clean.get(), name.c_str());
        archive_entry_set_filetype(clean.get(), AE_IFREG);
        archive_entry_set_perm(clean.get(), 0600);
        archive_entry_set_size(clean.get(), static_cast<la_int64_t>(part.size()));
        RequireArchive(archive_write_header(writer.get(), clean.get()), writer.get(), "staging header");
        const auto written = archive_write_data(writer.get(), part.data(), part.size());
        if (written < 0 || static_cast<size_t>(written) != part.size()) throw std::runtime_error("XLSX staging write failed");
        RequireArchive(archive_write_finish_entry(writer.get()), writer.get(), "staging entry close");
    }
    if (!names.contains("[Content_Types].xml") || !names.contains("xl/workbook.xml"))
        throw std::runtime_error("XLSX workbook package parts are missing");
    RequireArchive(archive_read_close(reader.get()), reader.get(), "ZIP close");
    RequireArchive(archive_write_close(writer.get()), writer.get(), "staging close");
}
} // namespace

PreparedXlsxArchive::PreparedXlsxArchive(const std::filesystem::path& source,
                                       const XlsxArchiveLimits& limits) {
    const XlsxArchiveLimits maximum;
    if (!limits.compressed_bytes || limits.compressed_bytes > maximum.compressed_bytes ||
        !limits.expanded_bytes || limits.expanded_bytes > maximum.expanded_bytes ||
        !limits.part_bytes || limits.part_bytes > maximum.part_bytes ||
        !limits.entries || limits.entries > maximum.entries ||
        !limits.xml_markup || limits.xml_markup > maximum.xml_markup ||
        !limits.xml_depth || limits.xml_depth > maximum.xml_depth)
        throw std::runtime_error("invalid XLSX resource limits (limits may only be tightened)");
    static std::atomic<uint64_t> sequence{0};
    const auto root = std::filesystem::temp_directory_path();
    for (int attempt = 0; attempt < 16; ++attempt) {
        const auto candidate = root / ("cyxwiz_xlsx_stage_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + "_" + std::to_string(sequence++));
        if (std::filesystem::create_directory(candidate)) { directory_ = candidate; break; }
    }
    if (directory_.empty()) throw std::runtime_error("could not create a unique XLSX staging directory");
    try {
#ifndef _WIN32
        std::filesystem::permissions(directory_, std::filesystem::perms::owner_all,
                                     std::filesystem::perm_options::replace);
#endif
        staged_path_ = directory_ / "workbook.xlsx";
        Repack(source, staged_path_, limits);
    } catch (...) {
        Cleanup();
        throw;
    }
}

void PreparedXlsxArchive::Cleanup() noexcept {
    std::error_code ec;
    if (!staged_path_.empty()) std::filesystem::remove(staged_path_, ec);
    if (ec) std::fprintf(stderr, "XLSX staged-file cleanup failed (error %d)\n", ec.value());
    if (!directory_.empty()) std::filesystem::remove(directory_, ec);
    if (ec) std::fprintf(stderr, "XLSX staging-directory cleanup failed (error %d)\n", ec.value());
}
PreparedXlsxArchive::~PreparedXlsxArchive() { Cleanup(); }
} // namespace cyxwiz
