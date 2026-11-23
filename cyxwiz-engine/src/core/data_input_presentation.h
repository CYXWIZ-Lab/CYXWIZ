#pragma once

// The Data Input dialog's data strip (TOFIX129 B-2): what the node loads and
// how far it got, above the tabs, the same on every tab. Pure data in, words
// out; the dialog draws it.

#include <cstdint>
#include <string>
#include <vector>

namespace cyxwiz::data_input_view {

// The status vocabulary's tones: loaded green, loading accent, not loaded
// orange, failed red.
enum class LoadTone { NotLoaded, Loading, Loaded, Failed };

// The dialog's backend tags (loaded_backend_).
enum class Backend { None = 0, ArrowMemory = 1, ParquetDisk = 2, ImageFolder = 3, AudioFolder = 4, Text = 5 };

struct StripInput {
    std::string source;        // the chosen file or folder ("" = none)
    std::string kind;          // "CSV table", "Image folder", ...
    uint64_t file_size = 0;    // bytes on disk, 0 = unknown or a folder

    bool loading = false;
    std::string loading_message;
    double loading_seconds = 0.0;

    bool loaded = false;
    std::string dataset_name;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t classes = -1;      // -1 = not known
    Backend backend = Backend::None;
    std::string time_column;   // a time series loaded in the order of this column
    uint64_t bytes = 0;
    bool bytes_estimate = false;
    double load_seconds = -1.0;
    int audit_errors = -1;     // -1 = no audit yet
    int audit_warnings = -1;

    bool failed = false;       // the last Apply failed (an earlier load may stay)
    std::string failure;
    // A Data Input whose file lives in CyxCloud (TOFIX164): `source` is the
    // file's name there, `cloud_dataset` the dataset's; file_size its size.
    bool cloud = false;
    std::string cloud_dataset;
};

// A number and its word ("52,681" "samples"), drawn as one fact.
struct Fact {
    std::string value;
    std::string label;
};

struct Strip {
    std::string title;         // file or folder name
    std::string state;         // "Loaded in 4.57 s", "Not loaded", ...
    LoadTone tone = LoadTone::NotLoaded;
    std::string detail;        // "as sentiment_mental_health (text)", the error, ...
    std::vector<Fact> facts;
    std::string note;          // what to do next, or how it is held
};

Strip BuildStrip(const StripInput& in);

std::string FormatCount(int64_t value);     // 52681 -> "52,681"
std::string FormatSize(uint64_t bytes);     // 31457280 -> "30.0 MB"
std::string FormatSeconds(double seconds);  // 4.567 -> "4.57 s", 0.42 -> "420 ms"

}  // namespace cyxwiz::data_input_view
