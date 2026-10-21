#include "core/archive_text_source.h"
#include <archive.h>
#include <arrow/api.h>
#include <cyxwiz/utilities.h>
#include <archive_entry.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

void Check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void WriteZip(const std::filesystem::path& path,
              const std::vector<std::pair<std::string, std::string>>& files) {
    std::unique_ptr<archive, decltype(&archive_write_free)> writer(archive_write_new(), archive_write_free);
    Check(archive_write_set_format_zip(writer.get()) == ARCHIVE_OK, "ZIP format");
    Check(archive_write_open_filename(writer.get(), path.string().c_str()) == ARCHIVE_OK, "ZIP open");
    for (const auto& [name, text] : files) {
        std::unique_ptr<archive_entry, decltype(&archive_entry_free)> entry(archive_entry_new(), archive_entry_free);
        archive_entry_set_pathname(entry.get(), name.c_str());
        archive_entry_set_filetype(entry.get(), AE_IFREG);
        archive_entry_set_perm(entry.get(), 0644);
        archive_entry_set_size(entry.get(), text.size());
        Check(archive_write_header(writer.get(), entry.get()) == ARCHIVE_OK, "ZIP header");
        Check(archive_write_data(writer.get(), text.data(), text.size()) == static_cast<la_ssize_t>(text.size()), "ZIP write");
    }
    Check(archive_write_close(writer.get()) == ARCHIVE_OK, "ZIP close");
}
template<class F> void Reject(F f, const std::string& expected) {
    try { f(); } catch (const std::exception& e) {
        Check(std::string(e.what()).find(expected) != std::string::npos, e.what());
        return;
    }
    throw std::runtime_error("Expected rejection: " + expected);
}
int main(int argc, char** argv) {
    try {
        if (argc == 3) {
            const auto text = cyxwiz::ReadZipTextMember(argv[1], argv[2]);
            std::cout << "Selected member bytes=" << text.size() << "\n";
            return 0;
        }
        const auto path = std::filesystem::temp_directory_path() / "cyxwiz_archive_text_test.zip";
        const std::string original = "First\r\n\nCaf\xc3\xa9\tlast ";
        WriteZip(path, {{"notes.txt", "ignore"}, {"nested/text.txt", original}});
        Check(cyxwiz::ReadZipTextMember(path, "nested/text.txt") == original, "Exact UTF-8/whitespace preservation");
        const auto table = cyxwiz::LoadZipTextTable(path, "nested/text.txt");
        Check(table->ValidateFull().ok() && table->num_rows() == 1 && table->num_columns() == 6, "Document schema");
        const auto cell = [&](const char* name) {
            return std::static_pointer_cast<arrow::StringArray>(table->GetColumnByName(name)->chunk(0))->GetString(0);
        };
        Check(cell("text") == original, "Arrow text preservation");
        Check(cell("member_path") == "nested/text.txt", "Member identity");
        Check(cell("member_sha256") == cyxwiz::Utilities::HashText(original, "sha256").sha256_hash, "Member hash");
        Check(cell("archive_sha256") == cyxwiz::Utilities::HashFile(path.string(), "sha256").sha256_hash, "Snapshot hash");
        const auto multiple = cyxwiz::LoadZipTextSelection(path, "nested/text.txt\r\nnotes.txt\r\n");
        Check(multiple->num_rows() == 2 && multiple->num_columns() == 6, "Multi-member schema");
        Check(multiple->Slice(0, 1)->Equals(*table), "Single-member backward compatibility");
        Check(multiple->GetColumnByName("text")->GetScalar(1).ValueOrDie()->ToString() == "ignore", "Selection order, not archive order");
        Reject([&] {cyxwiz::LoadZipTextSelection(path, "notes.txt\nnotes.txt");}, "duplicate selection");
        {
            // Spaces/tabs around pasted member paths are trimmed.
            const auto trimmed = cyxwiz::LoadZipTextSelection(path, "  nested/text.txt \t\r\n\tnotes.txt  ");
            Check(trimmed && trimmed->num_rows() == 2, "trimmed member paths load both members");
        }
        Reject([&] {cyxwiz::LoadZipTextSelection(path, "notes.txt\n   \nnested/text.txt");}, "empty or invalid");
        Reject([&] {cyxwiz::LoadZipTextSelection(path, "notes.txt\n\nnested/text.txt");}, "empty or invalid");
        Reject([&] {cyxwiz::LoadZipTextSelection(path, "notes.txt\nmissing");}, "not found: missing");
        Reject([&] {cyxwiz::LoadZipTextSelection(path, std::string(65537, 'a'));}, "65536 bytes");
        auto combined_limit = cyxwiz::ArchiveTextLimits{};
        combined_limit.total_text_bytes = original.size();
        Reject([&] {cyxwiz::LoadZipTextSelection(path, "notes.txt\nnested/text.txt", combined_limit);}, "total selected text byte limit");
        Reject([&] {cyxwiz::LoadZipTextSelection(path, "notes.txt\nnested/text.txt", {}, [] {return true;});}, "cancelled");
        const auto prior_id = cell("document_id");
        Check(prior_id.size() == 64, "Document identity");
        Reject([&] {cyxwiz::ReadZipTextMember(path, "missing");}, "not found");
        Reject([&] {cyxwiz::ReadZipTextMember(path, "../text.txt");}, "path component");
        Reject([&] {cyxwiz::ReadZipTextMember(path, "nested/text.txt", {}, [] {return true;});}, "cancelled");
        int calls = 0;
        Reject([&] {cyxwiz::ReadZipTextMember(path, "nested/text.txt", {}, [&] {return ++calls >= 5;});}, "cancelled");
        auto limits = cyxwiz::ArchiveTextLimits{};
        limits.archive_bytes = 1;
        Reject([&] {cyxwiz::ReadZipTextMember(path, "nested/text.txt", limits);}, "archive byte limit");
        limits = {}; limits.member_bytes = 1;
        Reject([&] {cyxwiz::ReadZipTextMember(path, "nested/text.txt", limits);}, "member byte limit");
        limits = {}; limits.entries = 1;
        Reject([&] {cyxwiz::ReadZipTextMember(path, "nested/text.txt", limits);}, "entry limit");
        WriteZip(path, {{"text", "first"}, {"text", "second"}});
        Reject([&] {cyxwiz::ReadZipTextMember(path, "text");}, "duplicate");
        WriteZip(path, {{"text", std::string("x\0y",3)}});
        Reject([&] {cyxwiz::ReadZipTextMember(path, "text");}, "UTF-8");
        WriteZip(path, {{"text", std::string(1, static_cast<char>(0xff))}});
        Reject([&] {cyxwiz::ReadZipTextMember(path, "text");}, "UTF-8");
        WriteZip(path, {{"text", ""}});
        Check(cyxwiz::ReadZipTextMember(path, "text").empty(), "Empty text remains empty");
        {std::ofstream invalid(path, std::ios::binary); invalid << "not a zip";}
        Reject([&] {cyxwiz::ReadZipTextMember(path, "text");}, "Archive text:");
        std::filesystem::remove(path);
        std::cout << "Archive text reader regression passed\n";
        return 0;
    } catch (const std::exception& e) {std::cerr << e.what() << '\n'; return 1;}
}
