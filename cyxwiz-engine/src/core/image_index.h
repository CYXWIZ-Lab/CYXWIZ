#pragma once

// The index table of an image dataset (TOFIX178 1.3e): one row per image,
// so every table tool (Query, the Explore tabs, the Table Viewer) works on
// images too. Built from the image files' headers (no decode for JPEG, PNG
// and 16/24-bit BMP), in dataset order, and registered as "<dataset>_index"
// (DataRegistry::ImageIndexName).
//
// Columns: path, file, label (class index), label_name, width, height,
// channels, bytes, readable.

#include "data_registry.h"
#include "prepared_image_set.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace arrow {
class Table;
}

namespace cyxwiz {

// What an image file's header says, as the Engine's decode (OpenCV
// IMREAD_UNCHANGED) sees the file: the stored size (EXIF orientation is not
// applied) and the channels it gives (grey 1, colour 3, with alpha 4).
struct ImageHeader {
    bool readable = false;
    int width = 0;
    int height = 0;
    int channels = 0;
};

// Reads the header of JPEG, PNG and 16/24-bit BMP files; other files are
// decoded once. readable is false when neither works.
ImageHeader ReadImageHeader(const std::string& path);

// A name per class from its files' names: the part before the first '.',
// '_', '-' or digit that every file of the class shares ("cat.1.jpg" ->
// "cat"), when each class has its own; "" for a class without one or whose
// name already says it.
std::vector<std::string> SuggestClassNames(const std::vector<std::string>& files, const std::vector<int>& labels,
                                           const std::vector<std::string>& class_names);

// The index rows, in the listing's order. label_names gives a display name
// per class index (empty or missing: the class name). progress(done, total)
// every few hundred files; null when cancel became true.
std::shared_ptr<arrow::Table> BuildImageIndex(const ImageSetListing& listing,
                                              const std::vector<std::string>& label_names,
                                              const std::function<void(size_t, size_t)>& progress = {},
                                              const std::atomic<bool>* cancel = nullptr, int threads = 0);

}  // namespace cyxwiz
