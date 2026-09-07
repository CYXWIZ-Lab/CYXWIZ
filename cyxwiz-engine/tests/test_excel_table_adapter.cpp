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
