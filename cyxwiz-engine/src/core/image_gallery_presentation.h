#pragma once

// Explore > Images (TOFIX178 1.3e): an image dataset's thumbnails by page,
// filtered by class and sorted, with the picked image's details, as a grid,
// a list or one image in the viewer. Pure data in and out;
// gui/data_studio/image_gallery_view draws it.

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::image_gallery {

// One image, from the index table.
struct Row {
    std::string path, file;
    int label = -1;
    int width = -1, height = -1;  // -1 when unreadable
    int channels = -1;
    int64_t bytes = 0;
    bool readable = true;
    // From the index's quality columns, once quality is measured.
    bool measured = false;
    float blur = 0.0f, brightness = 0.0f, contrast = 0.0f;
    std::string verdict;          // "" passes, else "blurry, low contrast"
    std::string duplicate_of;     // the kept image's file name
};

// The first four are the grid's Sort buttons; the list's column headers
// also sort by name, class and size on disk.
enum class Sort { DatasetOrder, Smallest, Largest, Blurriest, Name, Class, LargestFile };

struct Input {
    std::vector<Row> rows;
    std::vector<std::string> class_names;  // by class index (display names)
    std::string filter = "all";            // "all", "class:<index>", "unreadable", "fails", "reason:<reason>"
    Sort sort = Sort::DatasetOrder;
    size_t page = 0;
    size_t per_page = 60;
    long long selected = -1;               // a row index, -1 none
    size_t strip_radius = 5;               // the viewer's filmstrip: images each side of the selected
};

struct Chip {
    std::string key, label;
    size_t count = 0;
    std::string count_text;  // "25,000"
    bool on = false;
};

struct View {
    std::vector<Chip> chips;
    std::vector<size_t> shown;             // row indices on this page, in order
    std::vector<std::string> tags;         // per shown image: its class name
    std::vector<std::string> flags;        // per shown image: its first reason ("blurry +1"), "" passes
    std::vector<std::string> sizes;        // per shown image: "1023 x 768" (a times sign), "unreadable"
    std::vector<std::string> channels;     // per shown image: "colour", "grey", ...
    std::vector<std::string> on_disk;      // per shown image: "93.3 KB"
    std::vector<std::string> verdicts;     // per shown image: "passes", its verdict, "not measured"
    bool quality = false;                  // quality is measured: its chips and the Blurriest sort apply
    size_t matching = 0;                   // rows the filter keeps
    size_t page = 0, pages = 0;
    bool has_previous = false, has_next = false;
    std::string showing;                   // "1 to 60 of 25,000"
    long long selected = -1;               // kept only when it matches the filter
    std::string detail_title;              // the selected file's name
    std::vector<std::pair<std::string, std::string>> detail;
    // The viewer: where the selected image is among the matching ones
    // ("1 of 25,000"), its neighbours, and the filmstrip around it.
    std::string position;
    long long previous = -1, next = -1;
    std::vector<size_t> strip;             // row indices, the selected among them
    // Each matching image in order, for moving the pick by keys; row indices.
    std::vector<size_t> order;
};

View Build(const Input& in);

}  // namespace cyxwiz::image_gallery
