// Asset Browser light preview (TOFIX140): file kinds, the bounded CSV / TSV /
// Parquet / text peek, the large-file row estimate and the summary line.
#include "../src/core/asset_preview.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>
#include <arrow/ipc/writer.h>
#ifdef CYXWIZ_HAS_HDF5
#include <highfive/H5File.hpp>
#endif
#ifdef CYXWIZ_HAS_XLSX
#include <OpenXLSX.hpp>
#endif

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace cyxwiz::assets;
namespace fs = std::filesystem;

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        ++failures;
    }
}

void WriteText(const fs::path& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary);
    out << text;
}

}  // namespace

int main(int argc, char** argv) {
    // --keep <dir>: write the fixtures there and leave them (sample files for
    // a screen check of the Asset Browser preview).
    const bool keep = argc > 2 && std::string(argv[1]) == "--keep";
    const fs::path dir = keep ? fs::path(argv[2]) : fs::temp_directory_path() / "cyxwiz_asset_preview_test";
    if (!keep) fs::remove_all(dir);
    fs::create_directories(dir);

    // Kinds and labels.
    Check(KindOfPath("a/helix.csv", false) == FileKind::Table, "csv is a table");
    Check(KindOfPath("a/b.PARQUET", false) == FileKind::Columnar, "parquet is columnar (any case)");
    Check(KindOfPath("a/python_env.json", false) == FileKind::Config, "json is config");
    Check(KindOfPath("a/p5_api.py", false) == FileKind::Python, "py is python");
    Check(KindOfPath("a/eda.ipynb", false) == FileKind::Notebook, "ipynb is a notebook");
    Check(KindOfPath("a/g.cyxgraph", false) == FileKind::Graph, "cyxgraph is a graph");
    Check(KindOfPath("a/m.safetensors", false) == FileKind::Model, "safetensors is a model");
    Check(KindOfPath("a/humanoid.png", false) == FileKind::Image, "png is an image");
    Check(KindOfPath("a/datasets", true) == FileKind::Folder, "directory is a folder");
    Check(KindOfPath("a/LICENSE", false) == FileKind::Other, "no extension is other");
    Check(TypeLabel("a/helix.csv", false) == "CSV" && TypeLabel("a/x", false) == "File" &&
              TypeLabel("a/d", true) == "Folder", "type labels");

    // CSV: header, first rows, column kinds, exact row count, quoted fields.
    std::string csv = "t,x,label\n";
    for (int i = 0; i < 400; ++i) csv += std::to_string(i) + "," + std::to_string(i * 0.5) + ",\"inner, ring\"\n";
    WriteText(dir / "helix.csv", csv);
    FilePeek peek = PeekFile((dir / "helix.csv").string());
    Check(peek.ok && peek.table, "csv peeks as a table");
    Check(peek.columns == std::vector<std::string>({"t", "x", "label"}), "csv header");
    Check(peek.rows.size() == 12, "csv shows 12 rows");
    Check(peek.rows[0][2] == "inner, ring", "quoted field keeps its comma");
    Check(peek.numeric == std::vector<bool>({true, true, false}), "csv column kinds");
    Check(peek.total_rows == 400 && !peek.rows_estimated, "csv exact row count");
    Check(SummaryLine("CSV", "13.8 KB", peek) == "CSV \xC2\xB7 13.8 KB \xC2\xB7 400 rows \xC2\xB7 3 columns",
          "summary line: " + SummaryLine("CSV", "13.8 KB", peek));

    // No trailing newline still counts the last row; CRLF is stripped.
    WriteText(dir / "crlf.tsv", "a\tb\r\n1\t2\r\n3\t4");
    peek = PeekFile((dir / "crlf.tsv").string());
    Check(peek.ok && peek.total_rows == 2 && peek.rows[1][1] == "4", "tsv rows, CRLF, no final newline");

    // A large file: the row count is estimated from the head, not read in full.
    PeekLimits small;
    small.head_bytes = 256;
    small.exact_count_bytes = 512;
    peek = PeekFile((dir / "helix.csv").string(), small);
    Check(peek.ok && peek.rows_estimated && peek.total_rows > 300 && peek.total_rows < 500,
          "large csv rows estimated: " + std::to_string(peek.total_rows));
    Check(SummaryLine("CSV", "", peek).find("about ") != std::string::npos, "estimate says about");

    // Parquet: rows from the footer, first rows from one small batch.
    {
        arrow::Int64Builder ids;
        arrow::StringBuilder names;
        for (int i = 0; i < 50; ++i) {
            (void)ids.Append(i);
            (void)names.Append("row" + std::to_string(i));
        }
        std::shared_ptr<arrow::Array> a, b;
        (void)ids.Finish(&a);
        (void)names.Finish(&b);
        auto table = arrow::Table::Make(
            arrow::schema({arrow::field("id", arrow::int64()), arrow::field("name", arrow::utf8())}), {a, b});
        auto out = arrow::io::FileOutputStream::Open((dir / "t.parquet").string());
        Check(out.ok(), "parquet fixture opened");
        if (out.ok()) {
            Check(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *out, 20).ok(), "parquet written");
            (void)(*out)->Close();
        }
    }
    peek = PeekFile((dir / "t.parquet").string());
    Check(peek.ok && peek.table && peek.total_rows == 50, "parquet row count from the footer: " + peek.error);
    Check(peek.columns == std::vector<std::string>({"id", "name"}) &&
              peek.numeric == std::vector<bool>({true, false}), "parquet columns and kinds");
    Check(peek.rows.size() == 12 && peek.rows[3][1] == "row3", "parquet first rows");

    // Text: the first lines, and whether more follow.
    WriteText(dir / "python_env.json", "{\n  \"python\": {\n    \"interpreter_path\": \"python.exe\"\n  }\n}\n");
    peek = PeekFile((dir / "python_env.json").string());
    Check(peek.ok && !peek.table && peek.lines.size() == 5 && !peek.more_lines, "short text in full");
    Check(SummaryLine("JSON", "72 B", peek) == "JSON \xC2\xB7 72 B \xC2\xB7 5 lines", "text summary");
    std::string long_text;
    for (int i = 0; i < 40; ++i) long_text += "line " + std::to_string(i) + "\n";
    WriteText(dir / "notes.md", long_text);
    peek = PeekFile((dir / "notes.md").string());
    Check(peek.lines.size() == 12 && peek.more_lines, "long text cut at 12 lines");

    // Unreadable / empty.
    peek = PeekFile((dir / "missing.csv").string());
    Check(!peek.ok && !peek.error.empty(), "missing file reports an error");
    WriteText(dir / "empty.csv", "");
    peek = PeekFile((dir / "empty.csv").string());
    Check(!peek.ok && peek.error == "The file is empty.", "empty csv");

    // Arrow IPC (Feather v2): row count from the footer, rows from the first batch.
    {
        arrow::Int64Builder ids;
        for (int i = 0; i < 30; ++i) (void)ids.Append(i * 10);
        std::shared_ptr<arrow::Array> a;
        (void)ids.Finish(&a);
        auto table = arrow::Table::Make(arrow::schema({arrow::field("value", arrow::int64())}), {a});
        auto out = arrow::io::FileOutputStream::Open((dir / "t.feather").string());
        Check(out.ok(), "feather fixture opened");
        if (out.ok()) {
            auto writer = arrow::ipc::MakeFileWriter(*out, table->schema());
            Check(writer.ok() && (*writer)->WriteTable(*table, 8).ok() && (*writer)->Close().ok(), "feather written");
            (void)(*out)->Close();
        }
    }
    peek = PeekFile((dir / "t.feather").string());
    Check(peek.ok && peek.table && peek.total_rows == 30 && peek.rows.size() == 8 && peek.rows[2][0] == "20",
          "feather: 30 rows from the footer, the first batch shown: " + peek.error);

#ifdef CYXWIZ_HAS_HDF5
    // HDF5: the top-level objects, metadata only.
    {
        HighFive::File file((dir / "t.h5").string(), HighFive::File::Overwrite);
        file.createGroup("group");
        std::vector<std::vector<float>> data(4, std::vector<float>(3, 1.0f));
        file.createDataSet("features", data);
    }
    peek = PeekFile((dir / "t.h5").string());
    Check(peek.ok && peek.lines.size() == 2 && peek.line_noun == "object", "hdf5 lists two objects: " + peek.note);
    bool saw_dataset = false;
    for (const auto& line : peek.lines) saw_dataset |= line.find("features") != std::string::npos && line.find("4 x 3") != std::string::npos;
    Check(saw_dataset, "hdf5 dataset line shows its shape");
#endif

#ifdef CYXWIZ_HAS_XLSX
    // Excel: the first sheet of a small workbook.
    {
        OpenXLSX::XLDocument doc;
        doc.create((dir / "t.xlsx").string(), OpenXLSX::XLForceOverwrite);
        auto sheet = doc.workbook().worksheet("Sheet1");
        sheet.cell("A1").value() = "segment";
        sheet.cell("B1").value() = "spend";
        for (int r = 2; r <= 6; ++r) {
            sheet.cell("A" + std::to_string(r)).value() = r % 2 ? "east" : "delta";
            sheet.cell("B" + std::to_string(r)).value() = r * 1.5;
        }
        doc.save();
        doc.close();
    }
    peek = PeekFile((dir / "t.xlsx").string());
    Check(peek.ok && peek.table && peek.total_rows == 5 && peek.columns.size() == 2 && peek.rows[0][0] == "delta",
          "xlsx: first sheet rows: " + peek.note);
    PeekLimits tiny;
    tiny.max_workbook_bytes = 16;
    peek = PeekFile((dir / "t.xlsx").string(), tiny);
    Check(peek.ok && !peek.table && peek.note.find("Large workbook") != std::string::npos, "a large workbook is not read");
#endif

    if (!keep) fs::remove_all(dir);
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("asset preview: kinds, CSV / TSV / Parquet / text peeks, estimate and summary as expected. OK\n");
    return 0;
}
