#include "archive_text_source.h"
#include <archive.h>
#include <arrow/api.h>
#include <cyxwiz/utilities.h>
#include <archive_entry.h>
#include <arrow/util/utf8.h>
#include <array>
#include <fstream>
#include <memory>
#include <set>
#include <map>
#include <stdexcept>
#include <vector>

namespace cyxwiz {
namespace {
void CheckName(const std::string& name) {
    if (name.empty() || name.size() > 4096 || name.front() == '/' ||
        name.find_first_of("\\:") != std::string::npos)
        throw std::runtime_error("Archive text: invalid member name");
    size_t start = 0;
    while (start < name.size()) {
        const auto end = name.find('/', start);
        const auto part = name.substr(start, end == std::string::npos ? end : end - start);
        if (part.empty() || part == "." || part == "..")
            throw std::runtime_error("Archive text: invalid member path component");
        if (end == std::string::npos) break;
        start = end + 1;
    }
}
void Require(int code, archive* reader, const char* operation) {
    if (code != ARCHIVE_OK) {
        const auto* detail = archive_error_string(reader);
        throw std::runtime_error(std::string("Archive text: ") + operation + ": " +
                                 (detail ? detail : "invalid ZIP"));
    }
}
std::string Digest(const std::string& bytes) {
    const auto hash = Utilities::HashText(bytes, "sha256");
    if (!hash.success || hash.sha256_hash.size() != 64)
        throw std::runtime_error("Archive text: SHA-256 failed");
    return hash.sha256_hash;
}
struct Documents { std::vector<std::string> texts; std::string archive_hash; };

Documents ReadDocuments(const std::filesystem::path& path,
                             const std::vector<std::string>& members,
                             const ArchiveTextLimits& limits,
                             const std::function<bool()>& cancelled) {
    const auto check_cancel = [&] {
        if (cancelled && cancelled()) throw std::runtime_error("Archive text: cancelled");
    };
    check_cancel();
    if (members.empty() || members.size() > 4096)
        throw std::runtime_error("Archive text: select 1 to 4096 members");
    std::map<std::string, size_t> selected;
    for (size_t i = 0; i < members.size(); ++i) {
        CheckName(members[i]);
        if (members[i].back() == '/') throw std::runtime_error("Archive text: select a file member");
        if (!selected.emplace(members[i], i).second)
            throw std::runtime_error("Archive text: duplicate selection: " + members[i]);
    }
    if (!limits.archive_bytes || !limits.member_bytes || !limits.entries || !limits.total_text_bytes)
        throw std::runtime_error("Archive text: limits must be positive");
    // Snapshot in bounded chunks so later path replacement cannot change the
    // bytes being parsed. Actual bytes, not only declared file sizes, are bounded.
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Archive text: cannot open archive");
    std::array<char, 32768> buffer{};
    std::string snapshot;
    while (file) {
        check_cancel();
        file.read(buffer.data(), buffer.size());
        const auto count = static_cast<size_t>(file.gcount());
        if (count > limits.archive_bytes - snapshot.size())
            throw std::runtime_error("Archive text: archive byte limit exceeded");
        snapshot.append(buffer.data(), count);
    }
    if (!file.eof()) throw std::runtime_error("Archive text: archive read failed");
    std::unique_ptr<archive, decltype(&archive_read_free)> reader(archive_read_new(), archive_read_free);
    if (!reader) throw std::runtime_error("Archive text: cannot allocate ZIP reader");
    Require(archive_read_support_format_zip(reader.get()), reader.get(), "enable ZIP");
    Require(archive_read_open_memory(reader.get(), snapshot.data(), snapshot.size()), reader.get(), "open ZIP");
    std::set<std::string> names;
    std::vector<std::string> results(members.size());
    std::vector<bool> found(members.size(), false);
    size_t total_bytes = 0;
    size_t entries = 0;
    archive_entry* entry = nullptr;
    for (;;) {
        check_cancel();
        const auto status = archive_read_next_header(reader.get(), &entry);
        if (status == ARCHIVE_EOF) break;
        Require(status, reader.get(), "read header");
        if (++entries > limits.entries) throw std::runtime_error("Archive text: entry limit exceeded");
        const char* raw_name = archive_entry_pathname_utf8(entry);
        if (!raw_name) throw std::runtime_error("Archive text: member name is not UTF-8");
        const std::string name(raw_name);
        CheckName(name);
        if (!names.insert(name).second) throw std::runtime_error("Archive text: duplicate member name");
        if (archive_entry_is_encrypted(entry) || archive_entry_symlink(entry) || archive_entry_hardlink(entry))
            throw std::runtime_error("Archive text: encrypted or linked member unsupported");
        const auto type = archive_entry_filetype(entry);
        if (type != AE_IFREG && type != AE_IFDIR)
            throw std::runtime_error("Archive text: unsupported member type");
        const auto match = selected.find(name);
        if (match == selected.end()) {
            Require(archive_read_data_skip(reader.get()), reader.get(), "skip member");
            continue;
        }
        if (type != AE_IFREG) throw std::runtime_error("Archive text: selected member is not a file");
        if (archive_entry_size_is_set(entry) && (archive_entry_size(entry) < 0 ||
            static_cast<uint64_t>(archive_entry_size(entry)) > limits.member_bytes))
            throw std::runtime_error("Archive text: member byte limit exceeded");
        auto& result = results[match->second];
        for (;;) {
            check_cancel();
            const auto count = archive_read_data(reader.get(), buffer.data(), buffer.size());
            if (count < 0) Require(ARCHIVE_FATAL, reader.get(), "read member");
            if (count == 0) break;
            if (static_cast<size_t>(count) > limits.member_bytes - result.size())
                throw std::runtime_error("Archive text: member byte limit exceeded");
            if (static_cast<size_t>(count) > limits.total_text_bytes - total_bytes)
                throw std::runtime_error("Archive text: total selected text byte limit exceeded");
            total_bytes += static_cast<size_t>(count);
            result.append(buffer.data(), static_cast<size_t>(count));
        }
        found[match->second] = true;
    }
    for (size_t i = 0; i < members.size(); ++i)
        if (!found[i]) throw std::runtime_error("Archive text: selected member not found: " + members[i]);
    static const bool initialized = [] { arrow::util::InitializeUTF8(); return true; }();
    (void)initialized;
    for (size_t i = 0; i < results.size(); ++i) {
        check_cancel();
        if (results[i].find('\0') != std::string::npos || !arrow::util::ValidateUTF8(results[i]))
            throw std::runtime_error("Archive text: selected member must be UTF-8 text without NUL: " + members[i]);
    }
    check_cancel();
    return {std::move(results), Digest(snapshot)};
}
} // namespace

std::string ReadZipTextMember(const std::filesystem::path& path,
                             const std::string& member,
                             const ArchiveTextLimits& limits,
                             const std::function<bool()>& cancelled) {
    return ReadDocuments(path, {member}, limits, cancelled).texts.front();
}

namespace {
std::shared_ptr<arrow::Table> LoadMembers(const std::filesystem::path& path,
    const std::vector<std::string>& members, const ArchiveTextLimits& limits,
    const std::function<bool()>& cancelled) {
    const auto documents = ReadDocuments(path, members, limits, cancelled);
    std::array<arrow::StringBuilder, 6> builders;
    const auto source = std::filesystem::absolute(path).lexically_normal().generic_u8string();
    const std::string source_path(source.begin(), source.end());
    for (size_t i = 0; i < members.size(); ++i) {
        if (cancelled && cancelled()) throw std::runtime_error("Archive text: cancelled");
        const auto append = [&](size_t column, const std::string& value) {
            const auto status = builders[column].Append(value);
            if (!status.ok()) throw std::runtime_error(status.ToString());
        };
        append(0, source_path);
        append(1, documents.archive_hash);
        append(2, members[i]);
        append(3, Digest(documents.texts[i]));
        append(4, Digest(documents.archive_hash + "\n" + members[i]));
        append(5, documents.texts[i]);
    }
    const std::array<const char*, 6> names = {
        "source_path", "archive_sha256", "member_path", "member_sha256", "document_id", "text"};
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> columns;
    for (size_t i = 0; i < names.size(); ++i) {
        auto array = builders[i].Finish();
        if (!array.ok()) throw std::runtime_error(array.status().ToString());
        fields.push_back(arrow::field(names[i], arrow::utf8(), false));
        columns.push_back(*array);
    }
    if (cancelled && cancelled()) throw std::runtime_error("Archive text: cancelled");
    auto table = arrow::Table::Make(arrow::schema(fields), columns);
    const auto valid = table->ValidateFull();
    if (!valid.ok()) throw std::runtime_error(valid.ToString());
    return table;
}
} // namespace

std::shared_ptr<arrow::Table> LoadZipTextTable(const std::filesystem::path& path,
    const std::string& member, const ArchiveTextLimits& limits,
    const std::function<bool()>& cancelled) {
    return LoadMembers(path, {member}, limits, cancelled);
}

std::shared_ptr<arrow::Table> LoadZipTextSelection(const std::filesystem::path& path,
    const std::string& selection, const ArchiveTextLimits& limits,
    const std::function<bool()>& cancelled) {
    if (selection.empty() || selection.size() > 65536)
        throw std::runtime_error("Archive text: selection must contain 1 to 65536 bytes");
    std::vector<std::string> members;
    size_t start = 0;
    while (start < selection.size()) {
        const auto end = selection.find('\n', start);
        auto member = selection.substr(start, end == std::string::npos ? end : end - start);
        if (!member.empty() && member.back() == '\r') member.pop_back();
        if (member.empty() || member.find_first_of("\r\n") != std::string::npos ||
            member.find('\0') != std::string::npos)
            throw std::runtime_error("Archive text: empty or invalid member selection line");
        members.push_back(std::move(member));
        if (members.size() > 4096) throw std::runtime_error("Archive text: selection count exceeds 4096");
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return LoadMembers(path, members, limits, cancelled);
}
} // namespace cyxwiz
