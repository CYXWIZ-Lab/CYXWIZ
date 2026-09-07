#include "core/data_convert_service.h"
#include "core/excel_table_adapter.h"
#include <arrow/api.h>
#ifdef CYXWIZ_HAS_XLSX
#include <OpenXLSX.hpp>
#endif
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

void RunXlsxArchiveGuardTests(const std::filesystem::path&, const std::string&);

namespace {
int checks = 0;
void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
struct Workspace {
    std::filesystem::path path;
    Workspace() {
        const auto stem = "cyxwiz_xlsx_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
        path = std::filesystem::temp_directory_path() / stem;
        if (!std::filesystem::create_directory(path)) throw std::runtime_error("fixture directory collision");
    }
    ~Workspace() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};
}

int main() try {
    Workspace workspace;
    cyxwiz::ExcelTableReadOptions options;
    std::string error;
#ifdef CYXWIZ_HAS_XLSX
    using namespace OpenXLSX;
    const auto file = (workspace.path / "input.xlsx").string();
    {
        XLDocument doc;
        doc.create(file, XLDoNotOverwrite);
        auto sheet = doc.workbook().worksheet("Sheet1");
        sheet.cell("A1").value() = "id";
        sheet.cell("B1").value() = "value";
        sheet.cell("C1").value() = "name";
        sheet.cell("D1").value() = "flag";
        sheet.cell("A2").value() = int64_t{1};
        sheet.cell("B2").value() = 2;
        sheet.cell("C2").value() = "hello,\nworld";
        sheet.cell("D2").value() = true;
        sheet.cell("A3").value() = int64_t{2};
        sheet.cell("B3").value() = 2.5;
        // C3 deliberately missing: must stay null, not an empty string.
        sheet.cell("D3").value() = false;
        doc.workbook().addWorksheet("Second");
        auto second = doc.workbook().worksheet("Second");
        second.cell("A1").value() = "preamble";
        second.cell("A2").value() = "score";
        second.cell("A3").value() = 7.25;
        doc.save();
    }
    RunXlsxArchiveGuardTests(workspace.path, file);
    auto table = cyxwiz::ReadExcelTable(file, options, error);
    Check(table != nullptr, error);
    Check(table->num_rows() == 2 && table->num_columns() == 4, "worksheet shape");
    Check(table->field(0)->type()->id() == arrow::Type::INT64, "integer type");
    Check(table->field(1)->type()->id() == arrow::Type::DOUBLE, "mixed numeric promotion");
    Check(table->field(3)->type()->id() == arrow::Type::BOOL, "boolean type");
    Check(table->column(2)->null_count() == 1, "blank cell stays null");
    Check(table->column(2)->GetScalar(0).ValueOrDie()->ToString() == "hello,\nworld", "string preserved");
    Check(table->column(1)->GetScalar(1).ValueOrDie()->ToString() == "2.5", "fraction preserved");

    options.sheet_name = "Second";
    options.skip_rows = 1;
    auto second = cyxwiz::ReadExcelTable(file, options, error);
    Check(second && second->num_rows() == 1 && second->field(0)->name() == "score", error);
    options.skip_rows = 2;
    options.has_header = false;
    second = cyxwiz::ReadExcelTable(file, options, error);
    Check(second && second->field(0)->name() == "col_0" && second->num_rows() == 1, error);
    options.sheet_name = "Missing";
    Check(!cyxwiz::ReadExcelTable(file, options, error) && error.find("does not exist") != std::string::npos, "missing sheet rejects");
    options = {};
    options.max_cells = 2;
    Check(!cyxwiz::ReadExcelTable(file, options, error) && error.find("cell limit") != std::string::npos, "cell budget rejects");
    options = {};
    options.skip_rows = -1;
    Check(!cyxwiz::ReadExcelTable(file, options, error), "negative skip rejects");
    options.skip_rows = 99;
    Check(!cyxwiz::ReadExcelTable(file, options, error), "out-of-range skip rejects");
    options = {};

    const auto offset_file = (workspace.path / "offset.xlsx").string();
    {
        XLDocument doc;
        doc.create(offset_file, XLDoNotOverwrite);
        auto sheet = doc.workbook().worksheet("Sheet1");
        sheet.cell("B2").value() = "Report title";
        sheet.cell("B6").value() = "id";
        sheet.cell("C6").value() = "score";
        sheet.cell("B7").value() = 10;
        sheet.cell("C7").value() = 1.5;
        sheet.cell("B8").value() = 20; // C8 remains null.
        doc.workbook().addWorksheet("Wide");
        auto wide = doc.workbook().worksheet("Wide");
        wide.cell("AA3").value() = "value";
        wide.cell("AA4").value() = 42;
        doc.save();
    }
    Check(!cyxwiz::ReadExcelTable(offset_file, options, error) && error.find("!A1") != std::string::npos,
          "default must not silently locate an offset header");
    options.skip_rows = 5;
    Check(!cyxwiz::ReadExcelTable(offset_file, options, error) && error.find("!A6") != std::string::npos,
          "row skipping must not silently drop blank leading columns");
    options.start_column = "b";
    auto offset = cyxwiz::ReadExcelTable(offset_file, options, error);
    Check(offset && offset->num_rows() == 2 && offset->num_columns() == 2, error);
    Check(offset->field(0)->name() == "id" && offset->column(1)->null_count() == 1,
          "offset header and null preserved");
    Check(offset->column(0)->GetScalar(1).ValueOrDie()->ToString() == "20", "offset integer preserved");
    options.skip_rows = 6;
    options.has_header = false;
    auto no_header = cyxwiz::ReadExcelTable(offset_file, options, error);
    Check(no_header && no_header->field(0)->name() == "col_0" && no_header->num_rows() == 2,
          "no-header names are relative to selected columns");
    for (const auto* invalid : {"", "0", "A1", "A:B", "AAAA", "XFE", " A", "A "}) {
        options.start_column = invalid;
        Check(!cyxwiz::ReadExcelTable(offset_file, options, error) && error.find("start column") != std::string::npos,
              "invalid start column rejects");
    }
    options.start_column = "D";
    Check(!cyxwiz::ReadExcelTable(offset_file, options, error) && error.find("beyond") != std::string::npos,
          "start after final column rejects");
    options = {};
    options.sheet_name = "Wide";
    options.start_column = "AA";
    options.skip_rows = 2;
    auto wide = cyxwiz::ReadExcelTable(offset_file, options, error);
    Check(wide && wide->num_columns() == 1 && wide->column(0)->GetScalar(0).ValueOrDie()->ToString() == "42",
          "multi-letter start column and named sheet");
    options = {};

    cyxwiz::DataConvertOptions offset_conversion;
    offset_conversion.input_path = offset_file;
    offset_conversion.output_path = (workspace.path / "offset.parquet").string();
    offset_conversion.skip_rows = 5;
    offset_conversion.excel_start_column = "B";
    Check(cyxwiz::DataConvertService::Preview(offset_conversion).ok, "service offset preview");
    Check(cyxwiz::DataConvertService::Convert(offset_conversion).ok, "service offset conversion");
    Check(cyxwiz::DataConvertService::Convert(offset_conversion).skipped_fresh_output, "offset cache reused");
    offset_conversion.excel_start_column = "C";
    Check(!cyxwiz::DataConvertService::Convert(offset_conversion).ok, "changed start column cannot reuse old output");
    offset_conversion.overwrite = true;
    auto changed_column = cyxwiz::DataConvertService::Convert(offset_conversion);
    Check(changed_column.ok && !changed_column.skipped_fresh_output && changed_column.columns == 1,
          "changed start column regenerates selected schema");

    cyxwiz::DataConvertOptions conversion;
    conversion.input_path = file;
    conversion.output_path = (workspace.path / "output.parquet").string();
    auto preview = cyxwiz::DataConvertService::Preview(conversion);
    Check(preview.ok && preview.rows == 2, preview.error);
    auto result = cyxwiz::DataConvertService::Convert(conversion);
    Check(result.ok, result.error);
    cyxwiz::DataConvertOptions reload;
    reload.input_path = conversion.output_path;
    auto reloaded = cyxwiz::DataConvertService::LoadTable(reload, error);
    Check(reloaded && reloaded->Equals(*table, false), "XLSX -> Parquet -> disk values/schema");
    auto cached = cyxwiz::DataConvertService::Convert(conversion);
    Check(cached.ok && cached.skipped_fresh_output, "XLSX file cache reused");
    conversion.excel_sheet = "Second";
    conversion.skip_rows = 1;
    Check(!cyxwiz::DataConvertService::Convert(conversion).ok, "changed sheet cannot reuse cache when overwrite=false");
    conversion.overwrite = true;
    result = cyxwiz::DataConvertService::Convert(conversion);
    Check(result.ok && !result.skipped_fresh_output && result.rows_written == 1, result.error);
    conversion.output_path = (workspace.path / "rejected.xlsx").string();
    conversion.output_format = "xlsx";
    Check(!cyxwiz::DataConvertService::Convert(conversion).ok && !std::filesystem::exists(conversion.output_path), "Excel export disabled");

    const auto bad = (workspace.path / "bad.xlsx").string();
    { std::ofstream out(bad); out << "not a zip"; }
    Check(!cyxwiz::ReadExcelTable(bad, options, error), "malformed input rejects");
    Check(!cyxwiz::ReadExcelTable((workspace.path / "old.xls").string(), options, error), "legacy XLS rejects");

    const auto validate_bad_cell = [&](const char* kind) {
        XLDocument doc;
        doc.create(bad, XLForceOverwrite);
        auto sheet = doc.workbook().worksheet("Sheet1");
        sheet.cell("A1").value() = "value";
        sheet.cell("A2").value() = 1;
        if (std::string(kind) == "formula") sheet.cell("A3").formula() = "SUM(A2)";
        else if (std::string(kind) == "error") sheet.cell("A3").value() = std::numeric_limits<double>::quiet_NaN();
        else sheet.cell("A3").value() = "text";
        doc.save();
        doc.close();
        Check(!cyxwiz::ReadExcelTable(bad, options, error) && error.find("A3") != std::string::npos,
              std::string(kind) + " must reject with cell location: " + error);
    };
    validate_bad_cell("formula");
    validate_bad_cell("error");
    validate_bad_cell("mixed");
    {
        XLDocument doc;
        doc.create(bad, XLForceOverwrite);
        auto sheet = doc.workbook().worksheet("Sheet1");
        sheet.cell("A1").value() = "date_serial";
        sheet.cell("A2").value() = XLDateTime(43791.5);
        doc.save();
    }
    auto dates = cyxwiz::ReadExcelTable(bad, options, error);
    Check(dates && dates->field(0)->type()->id() == arrow::Type::DOUBLE &&
          dates->column(0)->GetScalar(0).ValueOrDie()->ToString() == "43791.5", "dates retain raw Excel serials");
    {
        XLDocument doc;
        doc.create(bad, XLForceOverwrite);
        auto sheet = doc.workbook().worksheet("Sheet1");
        sheet.cell("A1").value() = "value";
        sheet.cell("A2").value() = int64_t{9007199254740993LL};
        sheet.cell("A3").value() = 0.5;
        doc.save();
    }
    Check(!cyxwiz::ReadExcelTable(bad, options, error) && error.find("precision loss") != std::string::npos,
          "unsafe integer promotion rejects: " + error);
    {
        XLDocument doc;
        doc.create(bad, XLForceOverwrite);
        auto sheet = doc.workbook().worksheet("Sheet1");
        sheet.cell("A1").value() = "same";
        sheet.cell("B1").value() = "same";
        doc.save();
    }
    Check(!cyxwiz::ReadExcelTable(bad, options, error) && error.find("duplicate header") != std::string::npos,
          "duplicate headers reject");
    {
        XLDocument doc;
        doc.create(bad, XLForceOverwrite);
        auto sheet = doc.workbook().worksheet("Sheet1");
        sheet.cell("A1").value() = "value";
        sheet.mergeCells("A1:B1");
        doc.save();
    }
    Check(!cyxwiz::ReadExcelTable(bad, options, error) && error.find("merged cells") != std::string::npos,
          "merged cells reject");
    {
        XLDocument doc;
        doc.create(bad, XLForceOverwrite);
        doc.workbook().worksheet("Sheet1").cell("A1").value() = "value";
        doc.save();
    }
    auto header_only = cyxwiz::ReadExcelTable(bad, options, error);
    Check(header_only && header_only->num_rows() == 0 && header_only->field(0)->name() == "value", "header-only table");
    {
        XLDocument doc;
        doc.create(bad, XLForceOverwrite);
        doc.save();
    }
    Check(!cyxwiz::ReadExcelTable(bad, options, error), "empty worksheet rejects");
#else
    Check(!cyxwiz::ReadExcelTable("unavailable.xlsx", options, error), "disabled adapter rejects");
    Check(error.find("not compiled") != std::string::npos, "disabled error explains dependency");
    cyxwiz::DataConvertOptions conversion;
    conversion.input_path = (workspace.path / "input.xlsx").string();
    { std::ofstream out(conversion.input_path); out << "unavailable adapter fixture"; }
    conversion.output_path = (workspace.path / "output.parquet").string();
    Check(!cyxwiz::DataConvertService::Preview(conversion).ok, "disabled XLSX preview rejects");
    Check(!cyxwiz::DataConvertService::Convert(conversion).ok && !std::filesystem::exists(conversion.output_path),
          "disabled XLSX conversion cannot publish output");
#endif
    std::cout << "Excel adapter: " << checks << " checks passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
