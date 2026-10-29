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

struct View {
    std::vector<Tile> tiles;
    std::string classes_title;       // "Classes · from cat_dog.csv"
    std::vector<ClassRow> classes;
    std::string balance;             // "Balanced." / "Largest class is 3.2x the smallest."
    std::vector<std::pair<std::string, std::string>> sizes;  // Width, Height, Square, Resize
    Histogram widths, heights;
    std::vector<SetRow> sets;
    std::string sets_note;
};

View Build(const Input& in);

// Bins of a round width (25, 50, 100, 200, 250, 500, 1000, ...) so that at
// most `max_bins` cover 0..max; values below 0 are left out.
Histogram MakeHistogram(const std::vector<int>& values, size_t max_bins = 12);

}  // namespace cyxwiz::image_profile
