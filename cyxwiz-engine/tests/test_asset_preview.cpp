// Asset Browser light preview (TOFIX140): file kinds, the bounded CSV / TSV /
// Parquet / text peek, the large-file row estimate and the summary line.
#include "../src/core/asset_preview.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

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

int main() {
    const fs::path dir = fs::temp_directory_path() / "cyxwiz_asset_preview_test";
    fs::remove_all(dir);
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

    fs::remove_all(dir);
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("asset preview: kinds, CSV / TSV / Parquet / text peeks, estimate and summary as expected. OK\n");
    return 0;
}
