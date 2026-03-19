#pragma once

#include "../dataset_base.h"
#include "../data_registry.h"
#include "../image_utils.h"
#include "image_csv_dataset.h"  // For LRUCache template
#include <spdlog/spdlog.h>
#include <filesystem>
#include <algorithm>
#include <map>

namespace fs = std::filesystem;

namespace cyxwiz {

/**
 * ImageFolder Dataset Implementation
 * Expects directory structure: root/class_name/image.jpg
 * Loads images using stb_image, resizes to consistent dimensions
 * Includes LRU cache for loaded images
 */
class ImageFolderDataset : public Dataset {
public:
    ImageFolderDataset(const std::string& path, int target_width = 224, int target_height = 224, size_t cache_size = 100)
        : path_(path), target_width_(target_width), target_height_(target_height), image_cache_(cache_size) {
        LoadData();
    }

    size_t Size() const override { return image_paths_.size(); }

    std::pair<std::vector<float>, int> GetItem(size_t index) const override {
        if (index >= image_paths_.size()) return {{}, -1};

        // Check cache first
        auto cached = image_cache_.Get(index);
        if (cached.has_value()) {
            return {cached.value(), labels_[index]};
        }

        // Lazy load the image
        std::vector<float> image = LoadImage(image_paths_[index]);

        // Store in cache
        image_cache_.Put(index, image);

        return {std::move(image), labels_[index]};
    }

    DatasetInfo GetInfo() const override {
        DatasetInfo info;
        info.name = fs::path(path_).filename().string();
        info.path = path_;
        info.type = DatasetType::ImageFolder;
        info.shape = {static_cast<size_t>(target_width_),
                      static_cast<size_t>(target_height_),
                      static_cast<size_t>(channels_)};
        info.num_samples = image_paths_.size();
        info.num_classes = class_names_.size();
        info.class_names = class_names_;
        info.train_count = train_indices_.size();
        info.val_count = val_indices_.size();
        info.test_count = test_indices_.size();
        // Estimate memory (lazy loading, but estimate full load)
        info.memory_usage = image_paths_.size() * target_width_ * target_height_ * channels_ * sizeof(float);
        info.is_loaded = !image_paths_.empty();
        return info;
    }

private:
    void LoadData() {
        if (!fs::exists(path_) || !fs::is_directory(path_)) {
            spdlog::error("ImageFolder path does not exist or is not a directory: {}", path_);
            return;
        }

        // Scan for class directories
        std::vector<std::string> class_dirs;
        for (const auto& entry : fs::directory_iterator(path_)) {
            if (entry.is_directory()) {
                class_dirs.push_back(entry.path().filename().string());
            }
        }

        if (class_dirs.empty()) {
            spdlog::error("No class directories found in: {}", path_);
            return;
        }

        // Sort class names for consistent label assignment
        std::sort(class_dirs.begin(), class_dirs.end());
        class_names_ = class_dirs;

        // Build class name to label mapping
        std::map<std::string, int> class_to_label;
        for (size_t i = 0; i < class_names_.size(); i++) {
            class_to_label[class_names_[i]] = static_cast<int>(i);
        }

        // Scan for images in each class directory
        std::vector<std::string> valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tga"};

        for (const auto& class_name : class_names_) {
            fs::path class_path = fs::path(path_) / class_name;
            int label = class_to_label[class_name];

            for (const auto& entry : fs::directory_iterator(class_path)) {
                if (!entry.is_regular_file()) continue;

                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                if (std::find(valid_extensions.begin(), valid_extensions.end(), ext) != valid_extensions.end()) {
                    image_paths_.push_back(entry.path().string());
                    labels_.push_back(label);
                }
            }
        }

        if (image_paths_.empty()) {
            spdlog::error("No valid images found in: {}", path_);
            return;
        }

        // Detect channels from first image using OpenCV
        std::vector<float> temp_data;
        int width, height, channels;
        if (!ImageUtils::LoadImage(image_paths_[0], temp_data, width, height, channels)) {
            spdlog::warn("Could not get info for first image, defaulting to 3 channels");
            channels_ = 3;
        } else {
            channels_ = channels;
        }

        spdlog::info("Loaded ImageFolder dataset: {} images, {} classes from {}",
            image_paths_.size(), class_names_.size(), path_);

        // Apply default split
        SetSplit(SplitConfig{});
    }

    std::vector<float> LoadImage(const std::string& path) const {
        // Load image using OpenCV (via ImageUtils)
        std::vector<float> result;
        int width, height, channels;

        if (!ImageUtils::LoadImage(path, result, width, height, channels)) {
            spdlog::warn("Failed to load image: {}", path);
            return std::vector<float>(target_width_ * target_height_ * channels_, 0.0f);
        }

        // Resize if needed using OpenCV high-quality resize
        if (width != target_width_ || height != target_height_) {
            // Use Area method for downscaling (best quality), Lanczos for upscaling
            ImageUtils::ResizeMethod method = (target_width_ < width || target_height_ < height)
                ? ImageUtils::ResizeMethod::Area
                : ImageUtils::ResizeMethod::Lanczos;

            if (!ImageUtils::ResizeImage(result, width, height, channels,
                                          target_width_, target_height_, method)) {
                spdlog::warn("Failed to resize image: {}", path);
                return std::vector<float>(target_width_ * target_height_ * channels_, 0.0f);
            }
        }

        return result;
    }

    std::string path_;
    int target_width_;
    int target_height_;
    int channels_ = 3;
    std::vector<std::string> image_paths_;
    std::vector<int> labels_;
    std::vector<std::string> class_names_;
    mutable LRUCache<size_t, std::vector<float>> image_cache_;
};

} // namespace cyxwiz
