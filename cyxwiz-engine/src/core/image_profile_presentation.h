#pragma once

// Explore > Profile for an image dataset (TOFIX178 1.3e), also the Data
// Input dialog's Profile tab: counts, class balance, original sizes, how the
// graph's Resize treats them, and the prepared sets in cache/images. Pure
// data in and out; the view (gui/data_studio/image_profile_view) draws it.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::image_profile {

struct PreparedSet {
    int width = 0, height = 0, channels = 0;
    std::string mode;                // "exact", "fit", "fill", "center"
    std::vector<std::string> fixed;  // transforms done once
    uint64_t bytes = 0;
    bool in_use = false;             // the set this graph trains from
};

struct Input {
    std::string dataset;             // "cat_dog"
    std::string labels_source;       // "cat_dog.csv" or "" for class folders
    std::vector<std::string> class_names;  // by class index (display names)
    // Per image, from the index table: label (class index, -1 none), size and
    // channels (-1 when unreadable), file bytes.
    std::vector<int> labels, widths, heights, channels;
    std::vector<int64_t> bytes;
    // The graph's Resize (width 0: the graph has none).
    int resize_width = 0, resize_height = 0;
    std::string resize_mode;         // "exact", "fit", "fill", "center"
    std::vector<PreparedSet> sets;
    bool sets_listed = false;        // false while the cache is being read
    // Channel statistics of the set this graph trains from.
    std::vector<double> channel_mean, channel_std;  // empty until measured
    size_t stats_images = 0;
    int stats_width = 0, stats_height = 0;  // the measured set's size (after the fixed steps)
    std::string stats_device;        // "CUDA - GeForce GTX 1050 Ti"
    // While measuring: preparing (done of total images), then measuring.
    enum class StatsStep { Idle, Preparing, Measuring } stats_step = StatsStep::Idle;
    size_t stats_done = 0, stats_total = 0;
    std::string stats_error;
    bool normalize = false;          // the graph has a Normalize node
    float norm_mean = 0.0f, norm_std = 1.0f;
    // Quality: the Quality Analyzer's measurements at the Resize size, judged
    // with the default checks.
    enum class QualityStep { Looking, NotMeasured, Measuring, Measured } quality_step = QualityStep::Looking;
    size_t quality_done = 0, quality_total = 0;
    std::string quality_device;      // where it was measured
    std::string quality_error;
    size_t rejected = 0, blurry = 0, dark = 0, bright = 0, low_contrast = 0, duplicates = 0, multiple = 0;
    std::vector<size_t> class_total, class_rejected;  // by class index
};

struct Tile {
    std::string value, label;        // "25,000", "images"
};

struct ClassRow {
    int label = 0;
    std::string name;
    size_t count = 0;
    float share = 0.0f;              // of all images, 0..1
    std::string text;                // "12,500 · 50%"
};

struct Histogram {
    std::vector<int> edges;          // bin i is [edges[i], edges[i+1])
    std::vector<size_t> counts;
    size_t max_count = 0;
};

struct SetRow {
    std::string size, resize, fixed, bytes;
    bool in_use = false;
};

struct StatsRow {
    std::string what;                // "mean", "std"
    std::vector<std::string> values; // per channel
};

struct QualityRow {
    std::string reason;              // "blurry", "low contrast", ...
    size_t count = 0;
    std::string text;                // "1,691 \xC2\xB7 6.8%"
    float of_rejected = 0.0f;        // the bar: share of the images that fail
};

struct View {
    std::vector<Tile> tiles;
    std::string classes_title;       // "Classes · from cat_dog.csv"
    std::vector<ClassRow> classes;
    std::string balance;             // "Balanced." / "Largest class is 3.2x the smallest."
    std::vector<std::pair<std::string, std::string>> sizes;  // Width, Height, Square, Resize
    Histogram widths, heights;
    std::vector<SetRow> sets;
    std::string sets_note;
    // Channel statistics.
    std::string stats_title;         // "Channel statistics", then the set: "64 x 64, 25,000 images, on CUDA - ..."
    std::vector<std::string> channels;  // "R", "G", "B" (or "grey")
    std::vector<StatsRow> stats;     // empty until measured
    std::string stats_note;          // what Normalize makes of them, or what Measure does
    bool can_measure = false;        // the graph has a Resize and nothing runs
    // Quality.
    std::string quality_title;       // "Quality", then where and with what it was measured
    std::string quality_summary;     // "2,340 images fail a check (9.4%); 402 fail more than one."
    std::vector<QualityRow> quality;
    std::string quality_note;        // by class and the checks, or what Measure does
    bool can_measure_quality = false;
    std::string quality_button;      // "Measure quality" / "Measure again"
};

View Build(const Input& in);

// Bins of a round width (25, 50, 100, 200, 250, 500, 1000, ...) so that at
// most `max_bins` cover 0..max; values below 0 are left out.
Histogram MakeHistogram(const std::vector<int>& values, size_t max_bins = 12);

}  // namespace cyxwiz::image_profile
