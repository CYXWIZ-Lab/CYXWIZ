#include "image_csv_dataset.h"
#include <spdlog/spdlog.h>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cctype>

namespace fs = std::filesystem;

namespace cyxwiz {

ImageCSVDataset::ImageCSVDataset(const std::string& image_folder, const std::string& csv_path,
                int target_width, int target_height, size_t cache_size,
                const std::string& filename_col, const std::string& label_col)
    : image_folder_(image_folder), csv_path_(csv_path),
      target_width_(target_width), target_height_(target_height),
      filename_col_(filename_col), label_col_(label_col),
      image_cache_(cache_size) {
    LoadData();
}

size_t ImageCSVDataset::Size() const {
    return image_paths_.size();
}

std::pair<std::vector<float>, int> ImageCSVDataset::GetItem(size_t index) const {
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

DatasetInfo ImageCSVDataset::GetInfo() const {
    DatasetInfo info;
    info.name = fs::path(image_folder_).filename().string() + (csv_path_.empty() ? " (Folder)" : " (CSV)");
    info.path = image_folder_;
    info.type = DatasetType::ImageCSV;
    info.shape = {static_cast<size_t>(target_width_),
                  static_cast<size_t>(target_height_),
                  static_cast<size_t>(channels_)};
    info.num_samples = image_paths_.size();
    info.num_classes = class_names_.size();
    info.class_names = class_names_;
    info.train_count = train_indices_.size();
    info.val_count = val_indices_.size();
    info.test_count = test_indices_.size();

    // Memory usage is the cache size, not total dataset size (lazy loading)
    size_t image_bytes = target_width_ * target_height_ * channels_ * sizeof(float);
    size_t cached_images = image_cache_.Size();
    info.memory_usage = cached_images * image_bytes;  // Actual memory in use
    info.cache_usage = cached_images * image_bytes;
    info.is_loaded = !image_paths_.empty();
    info.is_streaming = true;  // Mark as streaming/lazy loading

    // Track cache stats
    info.cache_hits = image_cache_.GetHits();
    info.cache_misses = image_cache_.GetMisses();

    return info;
}

void ImageCSVDataset::LoadData() {
    // Validate image folder
    if (!fs::exists(image_folder_) || !fs::is_directory(image_folder_)) {
        spdlog::error("ImageDataset: Image folder does not exist: {}", image_folder_);
        return;
    }

    // Valid image extensions
    std::vector<std::string> valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tga", ".tiff", ".webp"};
    auto is_image = [&](const std::string& ext) {
        std::string lower_ext = ext;
        std::transform(lower_ext.begin(), lower_ext.end(), lower_ext.begin(), ::tolower);
        return std::find(valid_extensions.begin(), valid_extensions.end(), lower_ext) != valid_extensions.end();
    };

    // Check if CSV is provided and exists
    bool use_csv = !csv_path_.empty() && fs::exists(csv_path_);

    if (use_csv) {
        // CSV mode: load images with labels from CSV file
        LoadFromCSV(valid_extensions);
    } else {
        // Folder mode: scan subfolders, use subfolder names as class labels
        LoadFromFolder(is_image);
    }

    if (image_paths_.empty()) {
        spdlog::error("ImageDataset: No valid images found");
        return;
    }

    // Detect channels from first image using OpenCV
    std::vector<float> temp_data;
    int width, height, channels;
    if (!ImageUtils::LoadImage(image_paths_[0], temp_data, width, height, channels)) {
        spdlog::warn("ImageDataset: Could not get info for first image, defaulting to 3 channels");
        channels_ = 3;
    } else {
        channels_ = channels;
    }

    if (use_csv) {
        spdlog::info("ImageDataset: Loaded {} images, {} classes from {} + {}",
            image_paths_.size(), class_names_.size(), image_folder_, csv_path_);
    } else {
        spdlog::info("ImageDataset: Loaded {} images, {} classes from folder {}",
            image_paths_.size(), class_names_.size(), image_folder_);
    }

    // Apply default split
    SetSplit(SplitConfig{});
}

void ImageCSVDataset::LoadFromCSV(const std::vector<std::string>& valid_extensions) {
    std::ifstream file(csv_path_);
    if (!file.is_open()) {
        spdlog::error("ImageDataset: Failed to open CSV file: {}", csv_path_);
        return;
    }

    std::string line;
    std::vector<std::string> headers;
    int filename_idx = 0;
    int label_idx = 1;

    // Read first line to detect headers
    if (std::getline(file, line)) {
        std::vector<std::string> first_row = ParseCSVLine(line);

        // Check if first line looks like a header
        bool likely_header = false;
        for (const auto& cell : first_row) {
            std::string lower = cell;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

            // Exact matches for common header names
            if (lower == "filename" || lower == "file" || lower == "image" ||
                lower == "label" || lower == "class" || lower == "target" ||
                lower == "id" || lower == "name" || lower == "path" ||
                lower == "dx" || lower == "category" || lower == "diagnosis") {
                likely_header = true;
                break;
            }

            // Suffix matches
            if (lower.length() > 3) {
                if (lower.substr(lower.length() - 3) == "_id" ||
                    lower.substr(lower.length() - 5 > 0 ? lower.length() - 5 : 0) == "_name" ||
                    lower.substr(lower.length() - 5 > 0 ? lower.length() - 5 : 0) == "_path" ||
                    lower.substr(lower.length() - 5 > 0 ? lower.length() - 5 : 0) == "_file" ||
                    lower.substr(lower.length() - 6 > 0 ? lower.length() - 6 : 0) == "_class" ||
                    lower.substr(lower.length() - 6 > 0 ? lower.length() - 6 : 0) == "_label") {
                    likely_header = true;
                    break;
                }
            }

            // Substring matches
            if (lower.find("image") != std::string::npos ||
                lower.find("file") != std::string::npos ||
                lower.find("label") != std::string::npos ||
                lower.find("class") != std::string::npos ||
                lower.find("name") != std::string::npos) {
                likely_header = true;
                break;
            }

            // Check for typical header patterns
            bool is_numeric = !lower.empty();
            for (char c : lower) {
                if (!std::isdigit(c) && c != '.' && c != '-' && c != '+' && c != 'e') {
                    is_numeric = false;
                    break;
                }
            }
            if (!is_numeric && lower.length() < 30 &&
                lower.find('/') == std::string::npos &&
                lower.find('\\') == std::string::npos &&
                lower.find('.') == std::string::npos) {
                if (lower == "age" || lower == "sex" || lower == "type" ||
                    lower == "date" || lower == "time" || lower == "index" ||
                    lower == "row" || lower == "col" || lower == "value") {
                    likely_header = true;
                    break;
                }
            }
        }

        if (likely_header) {
            headers = first_row;

            // Find column indices
            if (!filename_col_.empty()) {
                for (size_t i = 0; i < headers.size(); i++) {
                    if (headers[i] == filename_col_) {
                        filename_idx = static_cast<int>(i);
                        break;
                    }
                }
            } else {
                // Auto-detect filename column
                std::vector<std::string> filename_priorities = {
                    "image_id", "imageid", "image_name", "imagename",
                    "filename", "file_name", "file", "image",
                    "path", "image_path", "img", "photo", "picture"
                };
                for (const auto& prio : filename_priorities) {
                    bool found = false;
                    for (size_t i = 0; i < headers.size(); i++) {
                        std::string lower = headers[i];
                        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                        if (lower == prio) {
                            filename_idx = static_cast<int>(i);
                            found = true;
                            break;
                        }
                    }
                    if (found) break;
                }
            }

            if (!label_col_.empty()) {
                for (size_t i = 0; i < headers.size(); i++) {
                    if (headers[i] == label_col_) {
                        label_idx = static_cast<int>(i);
                        break;
                    }
                }
            } else {
                // Auto-detect label column
                std::vector<std::string> label_priorities = {
                    "label", "labels", "class", "class_name", "classname",
                    "target", "category", "dx", "diagnosis", "y"
                };
                for (const auto& prio : label_priorities) {
                    bool found = false;
                    for (size_t i = 0; i < headers.size(); i++) {
                        std::string lower = headers[i];
                        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                        if (lower == prio) {
                            label_idx = static_cast<int>(i);
                            found = true;
                            break;
                        }
                    }
                    if (found) break;
                }
            }

            spdlog::info("ImageDataset: Using column {} for filenames, column {} for labels",
                filename_idx < static_cast<int>(headers.size()) ? headers[filename_idx] : std::to_string(filename_idx),
                label_idx < static_cast<int>(headers.size()) ? headers[label_idx] : std::to_string(label_idx));
        } else {
            // No header, process first line as data
            file.seekg(0);
        }
    }

    // Build label mapping
    std::map<std::string, int> label_map;
    std::vector<std::pair<std::string, std::string>> raw_data;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::vector<std::string> row = ParseCSVLine(line);
        if (row.size() <= static_cast<size_t>(std::max(filename_idx, label_idx))) {
            continue;
        }

        std::string filename = row[filename_idx];
        std::string label_str = row[label_idx];

        // Trim whitespace
        filename.erase(0, filename.find_first_not_of(" \t\r\n"));
        filename.erase(filename.find_last_not_of(" \t\r\n") + 1);
        label_str.erase(0, label_str.find_first_not_of(" \t\r\n"));
        label_str.erase(label_str.find_last_not_of(" \t\r\n") + 1);

        raw_data.emplace_back(filename, label_str);

        if (label_map.find(label_str) == label_map.end()) {
            label_map[label_str] = static_cast<int>(label_map.size());
        }
    }

    file.close();

    // Build class names
    class_names_.resize(label_map.size());
    for (const auto& [name, idx] : label_map) {
        class_names_[idx] = name;
    }

    // Build filename -> full path map by scanning folder
    std::map<std::string, std::string> filename_to_path;
    auto scan_dir = [&](const fs::path& dir, auto& self) -> void {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.is_directory()) {
                self(entry.path(), self);
            } else if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::string lower_ext = ext;
                std::transform(lower_ext.begin(), lower_ext.end(), lower_ext.begin(), ::tolower);
                if (std::find(valid_extensions.begin(), valid_extensions.end(), lower_ext) != valid_extensions.end()) {
                    std::string full_path = entry.path().string();
                    std::string stem = entry.path().stem().string();
                    std::string name_with_ext = entry.path().filename().string();

                    filename_to_path[stem] = full_path;
                    filename_to_path[name_with_ext] = full_path;
                }
            }
        }
    };
    scan_dir(image_folder_, scan_dir);

    spdlog::info("ImageDataset: Scanned {} image files in folder tree", filename_to_path.size() / 2);

    // Process each row
    for (const auto& [filename, label_str] : raw_data) {
        std::string resolved_path;

        // Try direct lookup
        auto it = filename_to_path.find(filename);
        if (it != filename_to_path.end()) {
            resolved_path = it->second;
        } else {
            // Try direct path
            fs::path image_path = fs::path(image_folder_) / filename;
            if (fs::exists(image_path)) {
                resolved_path = image_path.string();
            } else {
                // Try adding extensions
                for (const auto& ext : valid_extensions) {
                    fs::path test_path = fs::path(image_folder_) / (filename + ext);
                    if (fs::exists(test_path)) {
                        resolved_path = test_path.string();
                        break;
                    }
                }
            }
        }

        if (resolved_path.empty()) {
            spdlog::warn("ImageDataset: Image not found: {}", filename);
            continue;
        }

        image_paths_.push_back(resolved_path);
        labels_.push_back(label_map[label_str]);
    }
}

void ImageCSVDataset::LoadFromFolder(std::function<bool(const std::string&)> is_image) {
    std::map<std::string, int> label_map;
    bool has_subfolders = false;

    // Check if we have class subfolders
    for (const auto& entry : fs::directory_iterator(image_folder_)) {
        if (entry.is_directory()) {
            has_subfolders = true;
            break;
        }
    }

    if (has_subfolders) {
        // Class subfolder mode
        for (const auto& class_dir : fs::directory_iterator(image_folder_)) {
            if (!class_dir.is_directory()) continue;

            std::string class_name = class_dir.path().filename().string();

            // Skip hidden folders
            if (class_name.empty() || class_name[0] == '.') continue;

            // Assign class index
            if (label_map.find(class_name) == label_map.end()) {
                label_map[class_name] = static_cast<int>(label_map.size());
            }
            int class_idx = label_map[class_name];

            // Scan images in class folder
            for (const auto& img_entry : fs::directory_iterator(class_dir.path())) {
                if (!img_entry.is_regular_file()) continue;

                std::string ext = img_entry.path().extension().string();
                if (!is_image(ext)) continue;

                image_paths_.push_back(img_entry.path().string());
                labels_.push_back(class_idx);
            }
        }

        // Build class names
        class_names_.resize(label_map.size());
        for (const auto& [name, idx] : label_map) {
            class_names_[idx] = name;
        }

        spdlog::info("ImageDataset: Found {} class subfolders", class_names_.size());
    } else {
        // Flat folder mode
        class_names_.push_back("default");

        for (const auto& entry : fs::directory_iterator(image_folder_)) {
            if (!entry.is_regular_file()) continue;

            std::string ext = entry.path().extension().string();
            if (!is_image(ext)) continue;

            image_paths_.push_back(entry.path().string());
            labels_.push_back(0);
        }

        spdlog::info("ImageDataset: Flat folder mode, {} images with single class", image_paths_.size());
    }
}

std::vector<std::string> ImageCSVDataset::ParseCSVLine(const std::string& line) {
    std::vector<std::string> result;
    std::string current;
    bool in_quotes = false;

    for (size_t i = 0; i < line.size(); i++) {
        char c = line[i];

        if (c == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                // Escaped quote
                current += '"';
                i++;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (c == ',' && !in_quotes) {
            result.push_back(current);
            current.clear();
        } else {
            current += c;
        }
    }
    result.push_back(current);

    return result;
}

std::vector<float> ImageCSVDataset::LoadImage(const std::string& path) const {
    std::vector<float> result;
    int width, height, channels;

    if (!ImageUtils::LoadImage(path, result, width, height, channels)) {
        spdlog::warn("ImageCSV: Failed to load image: {}", path);
        return std::vector<float>(target_width_ * target_height_ * channels_, 0.0f);
    }

    // Resize if needed
    if (width != target_width_ || height != target_height_) {
        ImageUtils::ResizeMethod method = (target_width_ < width || target_height_ < height)
            ? ImageUtils::ResizeMethod::Area
            : ImageUtils::ResizeMethod::Lanczos;

        if (!ImageUtils::ResizeImage(result, width, height, channels,
                                      target_width_, target_height_, method)) {
            spdlog::warn("ImageCSV: Failed to resize image: {}", path);
            return std::vector<float>(target_width_ * target_height_ * channels_, 0.0f);
        }
    }

    return result;
}

} // namespace cyxwiz
