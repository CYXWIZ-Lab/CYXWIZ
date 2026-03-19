#include "streaming_dataset.h"
#include <spdlog/spdlog.h>
#include <stb_image.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace fs = std::filesystem;

namespace cyxwiz {

StreamingDataset::StreamingDataset(const std::string& path, const StreamingConfig& config)
    : path_(path), config_(config) {
    Initialize();
}

size_t StreamingDataset::Size() const {
    return estimated_total_size_;
}

std::pair<std::vector<float>, int> StreamingDataset::GetItem(size_t index) const {
    // For streaming, we need to ensure the chunk containing this index is loaded
    size_t chunk_idx = index / config_.chunk_size;
    size_t local_idx = index % config_.chunk_size;

    // Check if chunk is in buffer
    auto it = chunk_buffer_.find(chunk_idx);
    if (it != chunk_buffer_.end()) {
        const auto& chunk = it->second;
        if (local_idx < chunk.samples.size()) {
            return {chunk.samples[local_idx], chunk.labels[local_idx]};
        }
    }

    // Load chunk
    LoadChunk(chunk_idx);

    it = chunk_buffer_.find(chunk_idx);
    if (it != chunk_buffer_.end() && local_idx < it->second.samples.size()) {
        return {it->second.samples[local_idx], it->second.labels[local_idx]};
    }

    return {{}, -1};
}

DatasetInfo StreamingDataset::GetInfo() const {
    DatasetInfo info;
    info.name = fs::path(path_).stem().string();
    info.path = path_;
    info.type = detected_type_;
    info.shape = shape_;
    info.num_samples = estimated_total_size_;
    info.num_classes = num_classes_;
    info.train_count = train_indices_.size();
    info.val_count = val_indices_.size();
    info.test_count = test_indices_.size();
    info.memory_usage = GetBufferMemoryUsage();
    info.is_loaded = true;
    info.is_streaming = true;
    return info;
}

bool StreamingDataset::IsStreaming() const {
    return true;
}

bool StreamingDataset::HasNext() const {
    return current_position_ < estimated_total_size_;
}

std::pair<std::vector<float>, int> StreamingDataset::GetNext() {
    if (current_position_ >= estimated_total_size_) {
        return {{}, -1};
    }
    auto result = GetItem(current_position_);
    current_position_++;
    return result;
}

void StreamingDataset::ResetStream() {
    current_position_ = 0;
    chunk_buffer_.clear();
}

void StreamingDataset::Initialize() {
    // Detect dataset type and estimate size
    detected_type_ = DataRegistry::DetectType(path_);

    if (!fs::exists(path_)) {
        spdlog::error("Streaming dataset path does not exist: {}", path_);
        return;
    }

    // Estimate total size based on file/directory
    EstimateSize();

    spdlog::info("Initialized streaming dataset: {} (est. {} samples, {} chunks)",
                path_, estimated_total_size_, (estimated_total_size_ + config_.chunk_size - 1) / config_.chunk_size);
}

void StreamingDataset::EstimateSize() {
    if (detected_type_ == DatasetType::CSV) {
        // Count lines in CSV
        std::ifstream file(path_);
        estimated_total_size_ = std::count(
            std::istreambuf_iterator<char>(file),
            std::istreambuf_iterator<char>(), '\n');
        if (estimated_total_size_ > 0) estimated_total_size_--;  // Subtract header
    }
    else if (detected_type_ == DatasetType::ImageFolder) {
        // Count images
        for (const auto& entry : fs::recursive_directory_iterator(path_)) {
            if (entry.is_regular_file()) {
                auto ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                    estimated_total_size_++;
                }
            }
        }
    }
    else {
        estimated_total_size_ = 10000;  // Default estimate
    }
}

void StreamingDataset::LoadChunk(size_t chunk_idx) const {
    // Evict old chunks if buffer is full
    while (chunk_buffer_.size() >= max_chunks_in_buffer_) {
        // Remove oldest chunk
        if (!chunk_access_order_.empty()) {
            size_t oldest = chunk_access_order_.front();
            chunk_access_order_.erase(chunk_access_order_.begin());
            chunk_buffer_.erase(oldest);
        }
    }

    // Load the chunk
    DataChunk chunk;
    size_t start_idx = chunk_idx * config_.chunk_size;
    size_t end_idx = std::min(start_idx + config_.chunk_size, estimated_total_size_);

    if (detected_type_ == DatasetType::CSV) {
        LoadCSVChunk(chunk, start_idx, end_idx);
    }
    else if (detected_type_ == DatasetType::ImageFolder) {
        LoadImageChunk(chunk, start_idx, end_idx);
    }

    chunk_buffer_[chunk_idx] = std::move(chunk);
    chunk_access_order_.push_back(chunk_idx);
}

void StreamingDataset::LoadCSVChunk(DataChunk& chunk, size_t start_idx, size_t end_idx) const {
    std::ifstream file(path_);
    if (!file) return;

    std::string line;
    size_t current_line = 0;

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line) && current_line < end_idx) {
        if (current_line >= start_idx) {
            std::vector<std::string> tokens;
            std::stringstream ss(line);
            std::string token;

            while (std::getline(ss, token, ',')) {
                token.erase(0, token.find_first_not_of(" \t\r\n"));
                token.erase(token.find_last_not_of(" \t\r\n") + 1);
                tokens.push_back(token);
            }

            if (!tokens.empty()) {
                std::vector<float> sample;
                for (size_t i = 0; i < tokens.size() - 1; i++) {
                    try {
                        sample.push_back(std::stof(tokens[i]));
                    } catch (...) {
                        sample.push_back(0.0f);
                    }
                }

                int label = 0;
                try {
                    label = std::stoi(tokens.back());
                } catch (...) {}

                chunk.samples.push_back(std::move(sample));
                chunk.labels.push_back(label);
            }
        }
        current_line++;
    }

    // Update shape if this is first chunk
    if (shape_.empty() && !chunk.samples.empty()) {
        shape_ = {chunk.samples[0].size()};
    }
}

void StreamingDataset::LoadImageChunk(DataChunk& chunk, size_t start_idx, size_t end_idx) const {
    // Collect image paths first (if not already done)
    if (image_paths_.empty()) {
        CollectImagePaths();
    }

    for (size_t i = start_idx; i < end_idx && i < image_paths_.size(); i++) {
        const auto& [path, label] = image_paths_[i];

        int width, height, channels;
        unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 0);

        if (data) {
            std::vector<float> sample(width * height * channels);
            for (int j = 0; j < width * height * channels; j++) {
                sample[j] = data[j] / 255.0f;
            }
            stbi_image_free(data);

            chunk.samples.push_back(std::move(sample));
            chunk.labels.push_back(label);

            // Update shape
            if (shape_.empty()) {
                shape_ = {static_cast<size_t>(width), static_cast<size_t>(height), static_cast<size_t>(channels)};
            }
        }
    }
}

void StreamingDataset::CollectImagePaths() const {
    if (!fs::is_directory(path_)) return;

    std::vector<std::string> class_dirs;
    for (const auto& entry : fs::directory_iterator(path_)) {
        if (entry.is_directory()) {
            class_dirs.push_back(entry.path().filename().string());
        }
    }
    std::sort(class_dirs.begin(), class_dirs.end());

    std::map<std::string, int> class_to_label;
    for (size_t i = 0; i < class_dirs.size(); i++) {
        class_to_label[class_dirs[i]] = static_cast<int>(i);
    }

    for (const auto& class_name : class_dirs) {
        fs::path class_path = fs::path(path_) / class_name;
        int label = class_to_label[class_name];

        for (const auto& entry : fs::directory_iterator(class_path)) {
            if (!entry.is_regular_file()) continue;

            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                image_paths_.push_back({entry.path().string(), label});
            }
        }
    }

    num_classes_ = class_dirs.size();
}

size_t StreamingDataset::GetBufferMemoryUsage() const {
    size_t total = 0;
    for (const auto& [_, chunk] : chunk_buffer_) {
        for (const auto& sample : chunk.samples) {
            total += sample.size() * sizeof(float);
        }
        total += chunk.labels.size() * sizeof(int);
    }
    return total;
}

} // namespace cyxwiz
