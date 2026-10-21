#ifdef CYXWIZ_HAS_XLSX
#include "core/xlsx_archive_guard.h"
#include "core/excel_table_adapter.h"
#include <OpenXLSX.hpp>
#include <archive.h>
#include <archive_entry.h>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
int checks = 0;
void Check(bool ok, const std::string& message) {
    ++checks;
    if (!ok) throw std::runtime_error("XLSX resource test: " + message);
}
using Parts = std::vector<std::pair<std::string, std::string>>;
void WriteZip(const std::filesystem::path& file, const Parts& parts) {
    std::unique_ptr<archive, decltype(&archive_write_free)> writer(archive_write_new(), archive_write_free);
    Check(writer != nullptr, "ZIP writer allocation");
    Check(archive_write_set_format_zip(writer.get()) == ARCHIVE_OK, "ZIP format");
    Check(archive_write_set_format_option(writer.get(), "zip", "compression", "deflate") == ARCHIVE_OK, "deflate fixture");
#ifdef _WIN32
    Check(archive_write_open_filename_w(writer.get(), file.c_str()) == ARCHIVE_OK, "fixture open");
#else
    Check(archive_write_open_filename(writer.get(), file.c_str()) == ARCHIVE_OK, "fixture open");
#endif
    for (const auto& [name, contents] : parts) {
        std::unique_ptr<archive_entry, decltype(&archive_entry_free)> entry(archive_entry_new(), archive_entry_free);
        Check(entry != nullptr, "ZIP entry allocation");
        archive_entry_set_pathname_utf8(entry.get(), name.c_str());
        archive_entry_set_filetype(entry.get(), AE_IFREG);
        archive_entry_set_perm(entry.get(), 0600);
        archive_entry_set_size(entry.get(), static_cast<la_int64_t>(contents.size()));
        Check(archive_write_header(writer.get(), entry.get()) == ARCHIVE_OK, "fixture header");
        Check(archive_write_data(writer.get(), contents.data(), contents.size()) == static_cast<la_ssize_t>(contents.size()), "fixture bytes");
        Check(archive_write_finish_entry(writer.get()) == ARCHIVE_OK, "fixture entry close");
    }
    Check(archive_write_close(writer.get()) == ARCHIVE_OK, "fixture close");
}
void Reject(const std::filesystem::path& file, const cyxwiz::XlsxArchiveLimits& limits,
            const std::string& reason) {
    bool rejected = false;
    try { cyxwiz::PreparedXlsxArchive staged(file, limits); }
    catch (const std::exception& error) {
        rejected = std::string(error.what()).find(reason) != std::string::npos;
        Check(rejected, "expected '" + reason + "', received: " + error.what());
    }
    Check(rejected, "must reject " + reason);
}
}

void RunXlsxArchiveGuardTests(const std::filesystem::path& work_dir, const std::string& workbook) {
    namespace fs = std::filesystem;
    using cyxwiz::PreparedXlsxArchive;
    using cyxwiz::XlsxArchiveLimits;
    const auto copy = work_dir / "snapshot_source.xlsx";
    fs::copy_file(workbook, copy);
    fs::path staged_path;
    {
        PreparedXlsxArchive snapshot(copy);
        staged_path = snapshot.Path();
        Check(fs::exists(staged_path), "validated snapshot exists");
        { std::ofstream out(copy, std::ios::binary); out << "source changed after validation"; }
        OpenXLSX::XLDocument doc;
        const auto utf8 = staged_path.u8string();
        doc.open(std::string(utf8.begin(), utf8.end()));
        Check(doc.workbook().worksheet("Sheet1").cell("A1").value().get<std::string>() == "id",
              "OpenXLSX reads the immutable validated snapshot, not changed source");
    }
    Check(!fs::exists(staged_path) && !fs::exists(staged_path.parent_path()), "snapshot lifetime cleans its own file and directory");
    const auto prepare = [&] {
        PreparedXlsxArchive snapshot(workbook);
        return snapshot.Path();
    };
    auto first_task = std::async(std::launch::async, prepare);
    auto second_task = std::async(std::launch::async, prepare);
    const auto first_path = first_task.get();
    const auto second_path = second_task.get();
    Check(first_path != second_path && !fs::exists(first_path) && !fs::exists(second_path),
          "concurrent reads own independent, cleaned snapshots");

    const auto raw = work_dir / "resource_limits.xlsx";
    const Parts base = {{"[Content_Types].xml", "<Types/>"}, {"xl/workbook.xml", "<workbook/>"}};
    WriteZip(raw, base);
    XlsxArchiveLimits limits;
    limits.compressed_bytes = 1;
    Reject(raw, limits, "compressed-file limit");
    limits = {};
    limits.entries = 1;
    Reject(raw, limits, "entry-count limit");
    limits = {};
    limits.part_bytes = 7;
    Reject(raw, limits, "part-byte limit");
    limits = {};
    limits.expanded_bytes = 12;
    Reject(raw, limits, "expanded-byte limit");
    limits = {};
    limits.xml_markup = 1;
    Reject(raw, limits, "markup limit");
    limits = {};
    limits.entries = 513;
    Reject(raw, limits, "only be tightened");
    {
        std::fstream out(raw, std::ios::binary | std::ios::in | std::ios::out);
        out.seekp(8);
        const char unsupported_method[2] = {12, 0}; // BZip2, rejected before decoder setup.
        out.write(unsupported_method, 2);
        Check(static_cast<bool>(out), "unsupported compression fixture");
    }
    Reject(raw, {}, "stored or Deflate");

    auto parts = base;
    parts.push_back({"xl/large.xml", "<data>" + std::string(100000, 'x') + "</data>"});
    WriteZip(raw, parts);
    Check(fs::file_size(raw) < 4096, "compressed expansion fixture is small on disk");
    limits = {};
    limits.part_bytes = 4096;
    Reject(raw, limits, "part-byte limit");
    for (const auto& [name, content, reason] : std::vector<std::tuple<std::string, std::string, std::string>>{
            {"xl/workbook.xml", "<second/>", "duplicate"},
            {"../escape.xml", "<r/>", "unsafe"},
            {"xl\\escape.xml", "<r/>", "unsafe"},
            {"xl/media/image.png", "binary", "binary/non-XML"},
            {"xl/deep.xml", "<a><b><c/></b></a>", "nesting limit"},
            {"xl/dtd.xml", "<!DOCTYPE r [<!ENTITY e 'x'>]><r>&e;</r>", "DTDs"},
            {"xl/utf16.xml", std::string("<\0r\0/\0>\0", 8), "UTF-8"},
            {"xl/bad_utf8.xml", std::string("<r>\xff</r>"), "UTF-8"}}) {
        parts = base;
        parts.push_back({name, content});
        WriteZip(raw, parts);
        if (name == "xl\\escape.xml") {
            // The Windows ZIP writer normalizes backslashes. Restore the hostile
            // name in both local and central headers so the reader sees it.
            std::ifstream input(raw, std::ios::binary);
            std::string bytes((std::istreambuf_iterator<char>(input)), {});
            input.close();
            size_t patched = 0;
            size_t offset = 0;
            while ((offset = bytes.find("xl/escape.xml", offset)) != std::string::npos) {
                bytes[offset + 2] = '\\';
                offset += name.size();
                ++patched;
            }
            Check(patched == 2, "hostile raw filename exists in both ZIP headers");
            std::ofstream output(raw, std::ios::binary | std::ios::trunc);
            output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
            Check(static_cast<bool>(output), "raw backslash fixture write");
        }
        limits = {};
        if (name == "xl/deep.xml") limits.xml_depth = 1;
        try { Reject(raw, limits, reason); }
        catch (const std::exception& error) { throw std::runtime_error(name + ": " + error.what()); }
    }
    // Fake closing tags in CDATA/comments and '>' inside quotes do not hide
    // subsequent real nesting from the resource scan.
    parts = base;
    parts.push_back({"xl/deep.xml", "<r q='>'><![CDATA[</r>]]><!-- </r> --><a><b></b></a></r>"});
    WriteZip(raw, parts);
    limits = {};
    limits.xml_depth = 2;
    Reject(raw, limits, "nesting limit");

    const auto repeated = work_dir / "shared_string_expansion.xlsx";
    {
        OpenXLSX::XLDocument doc;
        doc.create(repeated.string(), OpenXLSX::XLDoNotOverwrite);
        auto sheet = doc.workbook().worksheet("Sheet1");
        sheet.cell("A1").value() = "text";
        const std::string text(32760, 'x');
        for (uint32_t row = 2; row <= 2051; ++row) sheet.cell(OpenXLSX::XLCellReference(row, 1)).value() = text;
        doc.save();
    }
    Check(fs::file_size(repeated) < 1024 * 1024, "shared-string fixture is small on disk");
    std::string error;
    Check(!cyxwiz::ReadExcelTable(repeated.string(), {}, error) && error.find("Arrow string-byte limit") != std::string::npos,
          "shared strings cannot expand Arrow beyond its byte budget: " + error);
    const auto wide_headers = work_dir / "header_expansion.xlsx";
    {
        OpenXLSX::XLDocument doc;
        doc.create(wide_headers.string(), OpenXLSX::XLDoNotOverwrite);
        auto sheet = doc.workbook().worksheet("Sheet1");
        for (uint16_t col = 1; col <= 33; ++col)
            sheet.cell(OpenXLSX::XLCellReference(1, col)).value() =
                std::to_string(col) + std::string(32760, 'h');
        doc.save();
    }
    Check(!cyxwiz::ReadExcelTable(wide_headers.string(), {}, error) && error.find("header-byte limit") != std::string::npos,
          "schema headers cannot exceed their byte budget: " + error);
    cyxwiz::ExcelTableReadOptions options;
    options.max_cells = 1000001;
    Check(!cyxwiz::ReadExcelTable(workbook, options, error), "cell safety ceiling cannot be disabled");
    std::cout << "XLSX archive/resource guard: " << checks << " checks passed\n";
}
#endif
