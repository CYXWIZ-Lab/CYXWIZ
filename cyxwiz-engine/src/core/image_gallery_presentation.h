#pragma once

// Explore > Images (TOFIX178 1.3e): an image dataset's thumbnails by page,
// filtered by class and sorted, with the picked image's details. Pure data
// in and out; gui/data_studio/image_gallery_view draws it.

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
};

enum class Sort { DatasetOrder, Smallest, Largest };

struct Input {
    std::vector<Row> rows;
    std::vector<std::string> class_names;  // by class index (display names)
    std::string filter = "all";            // "all", "class:<index>", "unreadable"
    Sort sort = Sort::DatasetOrder;
    size_t page = 0;
    size_t per_page = 60;
    long long selected = -1;               // a row index, -1 none
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
    size_t matching = 0;                   // rows the filter keeps
    size_t page = 0, pages = 0;
    bool has_previous = false, has_next = false;
    std::string showing;                   // "1 to 60 of 25,000"
    long long selected = -1;               // kept only when it matches the filter
    std::string detail_title;              // the selected file's name
    std::vector<std::pair<std::string, std::string>> detail;
};

View Build(const Input& in);

}  // namespace cyxwiz::image_gallery
