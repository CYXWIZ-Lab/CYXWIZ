#pragma once

// The images a training run reads, decoded once (TOFIX178 1.3a): every image
// of an image Data Input decoded to RGB and resized to the Resize node's
// size, as uint8 [N, H, W, 3], kept in one file under the project's
// cache/images and opened memory-mapped. Training gathers its batches from
// it instead of decoding every file again each epoch. Any image Data Input:
// class folders or a labels CSV, any size, 8- or 16-bit, grey, RGB or RGBA.
// No ImGui.

#include "data_registry.h"
#include "../preprocessing/preprocessing_config.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

// Where image caches live: <project>/cache/<kind> with a project open, the
// system temp folder's cyxwiz/<kind> without one ("quality", "images").
void SetImageCacheProjectRoot(const std::string& root);
std::filesystem::path ImageCacheDir(const std::string& kind);

// The files of an image Data Input in dataset order, with their labels and
// class names; no pixels are read.
struct ImageSetListing {
    std::vector<std::string> files;
    std::vector<int> labels;
    std::vector<std::string> class_names;
};
ImageSetListing ListImageSet(const DataRegistry::ImageDatasetEntry& entry);

// "path|bytes|time" of every file, sorted: changes when any file does.
std::vector<std::string> ImageFileIdentityRows(const ImageSetListing& listing);

// Cheap: the folder's and its subfolders' times (and the labels CSV's), which
// change when files are added, removed or renamed.
std::string ImageSetSignature(const DataRegistry::ImageDatasetEntry& entry);

struct PreparedImageSpec {
    int width = 0;
    int height = 0;
    ResizeMode mode = ResizeMode::Exact;
};

// The compiled Resize node's size and mode (no mode: exact).
inline PreparedImageSpec PreparedImageSpecFrom(const ImagePreprocessingConfig& config) {
    return {config.target_width, config.target_height,
            config.resize_mode == ResizeMode::None ? ResizeMode::Exact : config.resize_mode};
}

// One file decoded to uint8 RGB [height, width, 3] at `spec`, as training
// sees it: exact stretches to the size; fit scales to fit inside and pads
// with black; fill scales to cover and crops the centre; center crops (or
// pads) the centre without scaling. Shrinking uses INTER_AREA, enlarging
// LANCZOS4, in float, rounded once at the end. False when the file cannot
// be read.
bool DecodePreparedImage(const std::string& path, const PreparedImageSpec& spec, uint8_t* out);

class PreparedImageSet {
public:
    ~PreparedImageSet();
    PreparedImageSet(const PreparedImageSet&) = delete;
    PreparedImageSet& operator=(const PreparedImageSet&) = delete;

    size_t Count() const { return labels_.size(); }
    size_t Height() const { return height_; }
    size_t Width() const { return width_; }
    size_t Channels() const { return 3; }
    size_t RowBytes() const { return height_ * width_ * 3; }
    // Image i's pixels, [H, W, 3] row-major; valid while the set lives.
    const uint8_t* Row(size_t i) const { return pixels_ + i * RowBytes(); }
    const std::vector<int>& Labels() const { return labels_; }
    const std::vector<std::string>& Files() const { return files_; }
    const std::vector<std::string>& ClassNames() const { return class_names_; }
    // Files that could not be read: black, label kept.
    const std::vector<std::string>& Unreadable() const { return unreadable_; }
    const std::filesystem::path& Path() const { return path_; }
    const std::string& Key() const { return key_; }

    // The set in `path`; nullptr (with the reason) when it is not a complete set.
    static std::shared_ptr<const PreparedImageSet> Open(const std::filesystem::path& path,
                                                        std::string* error = nullptr);

private:
    PreparedImageSet() = default;

    std::filesystem::path path_;
    std::string key_;
    size_t height_ = 0;
    size_t width_ = 0;
    std::vector<int> labels_;
    std::vector<std::string> files_;
    std::vector<std::string> class_names_;
    std::vector<std::string> unreadable_;
    const uint8_t* pixels_ = nullptr;
    void* mapping_ = nullptr;  // the platform's mapping of the file
    size_t mapped_bytes_ = 0;
};

// The key of `entry` at `spec`: its files (each one's size and time), its
// labels CSV, the size and the mode.
std::string PreparedImageSetKey(const DataRegistry::ImageDatasetEntry& entry, const PreparedImageSpec& spec);
std::filesystem::path PreparedImageSetPath(const std::string& key);

// Images done of total; return false to stop.
using PreparedImageProgress = std::function<bool(size_t done, size_t total)>;

struct PreparedImageResult {
    std::shared_ptr<const PreparedImageSet> set;  // nullptr: see error
    bool from_cache = false;
    double seconds = 0.0;  // decoding time (0 when from the cache)
    std::string error;
};

// The set of `entry` at `spec`: opened from cache/images when it is there,
// otherwise decoded on `threads` workers (0: every core), written there and
// opened. The cache folder keeps sets for 30 days and at most 10 GB.
PreparedImageResult PrepareImageSet(const DataRegistry::ImageDatasetEntry& entry,
                                    const PreparedImageSpec& spec,
                                    int threads = 0,
                                    const PreparedImageProgress& progress = {});

}  // namespace cyxwiz
