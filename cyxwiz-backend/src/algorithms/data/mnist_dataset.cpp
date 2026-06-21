#include "cyxwiz/dataloader.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

namespace cyxwiz {

// ============================================================================
// Helper Functions
// ============================================================================

namespace {

// Reverse bytes for big-endian to little-endian conversion
uint32_t ReverseInt(uint32_t i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
}

} // anonymous namespace

// ============================================================================
// MNISTDataset Implementation
// ============================================================================

MNISTDataset::MNISTDataset(const std::string& path, Split split,
                           bool normalize, bool flatten)
    : normalize_(normalize), flatten_(flatten)
{
    // Determine sample size
    sample_size_ = flatten ? 784 : 28 * 28;

    // Find and load image file
    std::string image_path = FindImageFile(path, split);
    if (image_path.empty()) {
        throw std::runtime_error("Could not find MNIST image file in: " + path);
    }

    // Find and load label file
    std::string label_path = FindLabelFile(path, split);
    if (label_path.empty()) {
        throw std::runtime_error("Could not find MNIST label file in: " + path);
    }

    spdlog::debug("Loading MNIST images from: {}", image_path);
    if (!LoadImages(image_path)) {
        throw std::runtime_error("Failed to load MNIST images from: " + image_path);
    }

    spdlog::debug("Loading MNIST labels from: {}", label_path);
    if (!LoadLabels(label_path)) {
        throw std::runtime_error("Failed to load MNIST labels from: " + label_path);
    }

    spdlog::info("Loaded MNIST {} set: {} samples",
                 (split == Split::Train ? "train" : "test"), Size());
}

std::pair<std::vector<float>, int> MNISTDataset::GetItem(size_t index) const {
    if (index >= Size()) {
        throw std::out_of_range("Index out of bounds: " + std::to_string(index));
    }

    size_t start = index * sample_size_;
    std::vector<float> data(images_.begin() + start,
                            images_.begin() + start + sample_size_);

    return {data, labels_[index]};
}

std::vector<size_t> MNISTDataset::GetShape() const {
    if (flatten_) {
        return {784};
    } else {
        return {28, 28};
    }
}

std::vector<std::string> MNISTDataset::GetClassNames() const {
    return {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
}

bool MNISTDataset::Download(const std::string& path, bool verbose) {
    // TODO: Implement download from http://yann.lecun.com/exdb/mnist/
    // For now, just check if files exist
    if (verbose) {
        spdlog::warn("MNIST download not implemented. Please download manually from:");
        spdlog::warn("  http://yann.lecun.com/exdb/mnist/");
        spdlog::warn("Extract files to: {}", path);
    }
    return false;
}

bool MNISTDataset::LoadImages(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        spdlog::error("Could not open file: {}", filepath);
        return false;
    }

    // Read header
    uint32_t magic_number = 0;
    uint32_t num_images = 0;
    uint32_t num_rows = 0;
    uint32_t num_cols = 0;

    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = ReverseInt(magic_number);

    if (magic_number != 2051) {
        spdlog::error("Invalid MNIST image file magic number: {}", magic_number);
        return false;
    }

    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
    file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));

    num_images = ReverseInt(num_images);
    num_rows = ReverseInt(num_rows);
    num_cols = ReverseInt(num_cols);

    spdlog::debug("MNIST: {} images, {}x{} pixels", num_images, num_rows, num_cols);

    // Read image data
    size_t image_size = num_rows * num_cols;
    images_.resize(num_images * image_size);

    std::vector<unsigned char> buffer(image_size);
    for (uint32_t i = 0; i < num_images; i++) {
        file.read(reinterpret_cast<char*>(buffer.data()), image_size);

        for (size_t j = 0; j < image_size; j++) {
            float pixel = static_cast<float>(buffer[j]);
            if (normalize_) {
                pixel /= 255.0f;
            }
            images_[i * image_size + j] = pixel;
        }
    }

    return true;
}

bool MNISTDataset::LoadLabels(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        spdlog::error("Could not open file: {}", filepath);
        return false;
    }

    // Read header
    uint32_t magic_number = 0;
    uint32_t num_labels = 0;

    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = ReverseInt(magic_number);

    if (magic_number != 2049) {
        spdlog::error("Invalid MNIST label file magic number: {}", magic_number);
        return false;
    }

    file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));
    num_labels = ReverseInt(num_labels);

    spdlog::debug("MNIST: {} labels", num_labels);

    // Read labels
    labels_.resize(num_labels);
    std::vector<unsigned char> buffer(num_labels);
    file.read(reinterpret_cast<char*>(buffer.data()), num_labels);

    for (uint32_t i = 0; i < num_labels; i++) {
        labels_[i] = static_cast<int>(buffer[i]);
    }

    return true;
}

std::string MNISTDataset::FindImageFile(const std::string& path, Split split) {
    std::vector<std::string> candidates;

    if (split == Split::Train) {
        candidates = {
            "train-images-idx3-ubyte",
            "train-images.idx3-ubyte",
            "train-images-idx3-ubyte.gz",
        };
    } else {
        candidates = {
            "t10k-images-idx3-ubyte",
            "t10k-images.idx3-ubyte",
            "t10k-images-idx3-ubyte.gz",
        };
    }

    for (const auto& name : candidates) {
        fs::path full_path = fs::path(path) / name;
        if (fs::exists(full_path)) {
            return full_path.string();
        }
    }

    // Also check for extracted files in subdirectories
    for (const auto& entry : fs::directory_iterator(path)) {
        if (entry.is_directory()) {
            std::string result = FindImageFile(entry.path().string(), split);
            if (!result.empty()) {
                return result;
            }
        }
    }

    return "";
}

std::string MNISTDataset::FindLabelFile(const std::string& path, Split split) {
    std::vector<std::string> candidates;

    if (split == Split::Train) {
        candidates = {
            "train-labels-idx1-ubyte",
            "train-labels.idx1-ubyte",
            "train-labels-idx1-ubyte.gz",
        };
    } else {
        candidates = {
            "t10k-labels-idx1-ubyte",
            "t10k-labels.idx1-ubyte",
            "t10k-labels-idx1-ubyte.gz",
        };
    }

    for (const auto& name : candidates) {
        fs::path full_path = fs::path(path) / name;
        if (fs::exists(full_path)) {
            return full_path.string();
        }
    }

    // Also check for extracted files in subdirectories
    for (const auto& entry : fs::directory_iterator(path)) {
        if (entry.is_directory()) {
            std::string result = FindLabelFile(entry.path().string(), split);
            if (!result.empty()) {
                return result;
            }
        }
    }

    return "";
}

} // namespace cyxwiz
