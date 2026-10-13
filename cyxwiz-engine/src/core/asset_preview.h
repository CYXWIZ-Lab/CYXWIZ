#pragma once

// The Asset Browser's file kinds and its light preview (TOFIX140): a bounded
// peek at a file the project holds - the first rows of a table, the first
// lines of a text file - so a click shows what the file is. It never loads or
// registers the data: a Data Input is the one way data enters the Engine
// (on-disk, lazily loaded, any size). No ImGui here.

#include <cstdint>
#include <string>
#include <vector>

namespace cyxwiz::assets {

enum class FileKind {
    Folder,
    Table,        // .csv .tsv
    Spreadsheet,  // .xlsx .xls
    Columnar,     // .parquet .arrow .feather .h5 .hdf5 .hdf
    Config,       // .json .yaml .yml .toml .ini .cfg .conf
    Python,       // .py .pyw .cyx
    Notebook,     // .ipynb
    Graph,        // .cyxgraph
    Model,        // .onnx .pt .pth .safetensors .bin .ckpt .checkpoint .gguf .lora
    Image,        // .png .jpg .jpeg .bmp .gif .webp .tif .tiff
    Audio,        // .wav .mp3 .flac .ogg .m4a
    Video,        // .mp4 .avi .mov .mkv .webm
    Text,         // .txt .md .log .rst
    Plugin,       // .dll .so .dylib
    Other,
};

FileKind KindOfPath(const std::string& path, bool is_directory);

// Short type name for the preview header: the extension in capitals ("CSV",
// "PARQUET"), "Folder", or "File" without one.
std::string TypeLabel(const std::string& path, bool is_directory);

// Kinds whose first rows the preview reads as a table.
bool PeeksAsTable(const std::string& path);
// Kinds whose first lines the preview shows as text.
bool PeeksAsText(FileKind kind);

struct FilePeek {
    bool ok = false;
    std::string error;  // why the file could not be read

    // Table (CSV / TSV / Parquet): header, the first rows as text, and which
    // columns hold only numbers in those rows.
    bool table = false;
    std::vector<std::string> columns;
    std::vector<bool> numeric;
    std::vector<std::vector<std::string>> rows;
    int64_t total_rows = -1;       // -1: unknown
    bool rows_estimated = false;   // from the sampled head of a large file

    // Text: the first lines.
    std::vector<std::string> lines;
    bool more_lines = false;
};

struct PeekLimits {
    int max_rows = 12;
    int max_lines = 12;
    // Bytes read from the head of a text table; a larger file's row count is
    // estimated from them instead of reading the whole file.
    std::uintmax_t head_bytes = 1u << 20;
    std::uintmax_t exact_count_bytes = 16u << 20;
};

FilePeek PeekFile(const std::string& path, const PeekLimits& limits = {});

// "CSV · 13.8 KB · 400 rows · 5 columns" (rows prefixed "about" when
// estimated; text files give their shown line count only when complete).
std::string SummaryLine(const std::string& type_label, const std::string& size_text, const FilePeek& peek);

}  // namespace cyxwiz::assets
