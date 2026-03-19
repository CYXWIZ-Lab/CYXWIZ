#pragma once

#include <string>
#include <vector>
#include <cstddef>

namespace cyxwiz {

/**
 * Dataset type enumeration
 */
enum class DatasetType {
    None,
    CSV,
    TSV,
    JSON,               // JSON data files
    TXT,                // Plain text files
    ImageFolder,
    ImageCSV,           // Images in folder + labels in CSV file
    MNIST,
    FashionMNIST,
    CIFAR10,
    CIFAR100,
    HuggingFace,
    Kaggle,
    Custom,
    HDF5,               // HDF5 data files (.h5, .hdf5)
    Streaming,
    Audio,
    TimeSeries,
    ARFF                // Weka ARFF files (.arff)
};

/**
 * Dataset split enumeration
 */
enum class DatasetSplit {
    Train,
    Validation,
    Test,
    All
};

/**
 * Split configuration
 */
struct SplitConfig {
    float train_ratio = 0.8f;
    float val_ratio = 0.1f;
    float test_ratio = 0.1f;
    bool stratified = true;
    bool shuffle = true;
    int seed = 42;
};

/**
 * Dataset information structure
 */
struct DatasetInfo {
    std::string name;                    // Unique identifier
    std::string path;                    // Source path
    DatasetType type = DatasetType::None;
    std::vector<size_t> shape;           // Sample shape (e.g., [28, 28, 1])
    size_t num_samples = 0;
    size_t num_classes = 0;
    std::vector<std::string> class_names;

    // Split information
    size_t train_count = 0;
    size_t val_count = 0;
    size_t test_count = 0;
    float train_ratio = 0.8f;
    float val_ratio = 0.1f;
    float test_ratio = 0.1f;

    // Memory information
    size_t memory_usage = 0;             // Bytes
    size_t cache_usage = 0;              // Bytes in cache
    bool is_loaded = false;
    bool is_streaming = false;

    // Cache stats per dataset
    size_t cache_hits = 0;
    size_t cache_misses = 0;

    // Get formatted shape string
    std::string GetShapeString() const {
        if (shape.empty()) return "[]";
        std::string result = "[";
        for (size_t i = 0; i < shape.size(); i++) {
            if (i > 0) result += ", ";
            result += std::to_string(shape[i]);
        }
        result += "]";
        return result;
    }

    // Get formatted memory usage string
    std::string GetMemoryString() const {
        const char* units[] = {"B", "KB", "MB", "GB"};
        int unit_index = 0;
        double size = static_cast<double>(memory_usage);
        while (size >= 1024.0 && unit_index < 3) {
            size /= 1024.0;
            unit_index++;
        }
        char buffer[32];
        snprintf(buffer, sizeof(buffer), "%.2f %s", size, units[unit_index]);
        return buffer;
    }
};

} // namespace cyxwiz
