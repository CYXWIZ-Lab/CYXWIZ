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
    std::string stats_device;        // "CUDA - GeForce GTX 1050 Ti"
    // While measuring: preparing (done of total images), then measuring.
    enum class StatsStep { Idle, Preparing, Measuring } stats_step = StatsStep::Idle;
    size_t stats_done = 0, stats_total = 0;
    std::string stats_error;
    bool normalize = false;          // the graph has a Normalize node
    float norm_mean = 0.0f, norm_std = 1.0f;
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
};

View Build(const Input& in);

// Bins of a round width (25, 50, 100, 200, 250, 500, 1000, ...) so that at
// most `max_bins` cover 0..max; values below 0 are left out.
Histogram MakeHistogram(const std::vector<int>& values, size_t max_bins = 12);

}  // namespace cyxwiz::image_profile
