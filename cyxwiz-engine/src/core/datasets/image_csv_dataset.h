#pragma once

#include "../dataset_base.h"
#include "../data_registry.h"
#include "../image_utils.h"
#include <vector>
#include <string>
#include <map>
#include <functional>

namespace cyxwiz {

/**
 * LRU Cache for lazy image loading
 */
template <typename K, typename V>
class LRUCache {
public:
    explicit LRUCache(size_t capacity) : capacity_(capacity) {}

    std::optional<V> Get(const K& key) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            misses_++;
            return std::nullopt;
        }

        // Move to front (most recently used)
        order_.erase(it->second.second);
        order_.push_front(key);
        it->second.second = order_.begin();

        hits_++;
        return it->second.first;
    }

    void Put(const K& key, const V& value) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = cache_.find(key);
        if (it != cache_.end()) {
            // Update existing
            it->second.first = value;
            order_.erase(it->second.second);
            order_.push_front(key);
            it->second.second = order_.begin();
            return;
        }

        // Evict if at capacity
        if (cache_.size() >= capacity_) {
            K lru_key = order_.back();
            cache_.erase(lru_key);
            order_.pop_back();
        }

        // Insert new
        order_.push_front(key);
        cache_[key] = {value, order_.begin()};
    }

    size_t Size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.size();
    }

    size_t GetHits() const { return hits_; }
    size_t GetMisses() const { return misses_; }

private:
    size_t capacity_;
    mutable std::mutex mutex_;
    mutable std::map<K, std::pair<V, typename std::list<K>::iterator>> cache_;
    mutable std::list<K> order_;  // front = MRU, back = LRU
    mutable size_t hits_ = 0;
    mutable size_t misses_ = 0;
};

/**
 * Image dataset loader with CSV labels
 *
 * Supports two modes:
 * 1. CSV mode: Images in folder + labels in CSV file
 * 2. Folder mode: Images in class subfolders (ImageNet structure)
 *
 * Features:
 * - Lazy image loading with LRU cache
 * - Auto header detection in CSV files
 * - Auto column detection (filename, label)
 * - Recursive folder scanning
 * - High-quality image resizing (Area/Lanczos)
 * - Support for multiple image formats (jpg, png, bmp, gif, tga, tiff, webp)
 */
class ImageCSVDataset : public Dataset {
public:
    ImageCSVDataset(const std::string& image_folder, const std::string& csv_path,
                    int target_width = 224, int target_height = 224, size_t cache_size = 100,
                    const std::string& filename_col = "", const std::string& label_col = "");

    // Dataset interface
    size_t Size() const override;
    std::pair<std::vector<float>, int> GetItem(size_t index) const override;
    DatasetInfo GetInfo() const override;

private:
    // Initialization
    void LoadData();
    void LoadFromCSV(const std::vector<std::string>& valid_extensions);
    void LoadFromFolder(std::function<bool(const std::string&)> is_image);

    // CSV parsing
    std::vector<std::string> ParseCSVLine(const std::string& line);

    // Image loading
    std::vector<float> LoadImage(const std::string& path) const;

    // Member variables
    std::string image_folder_;
    std::string csv_path_;
    std::string filename_col_;
    std::string label_col_;
    int target_width_;
    int target_height_;
    int channels_ = 3;
    std::vector<std::string> image_paths_;
    std::vector<int> labels_;
    std::vector<std::string> class_names_;
    mutable LRUCache<size_t, std::vector<float>> image_cache_;
};

} // namespace cyxwiz
