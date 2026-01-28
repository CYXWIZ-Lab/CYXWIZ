#pragma once

#include <string>
#include <vector>
#include <memory>

namespace cyxwiz {

/**
 * Describes an HDF5 dataset or group
 */
struct HDF5NodeInfo {
    std::string name;           // Node name (e.g., "images")
    std::string path;           // Full path (e.g., "/train/images")
    bool is_group = false;      // true = group, false = dataset
    bool is_expanded = false;   // UI state for tree view

    // Dataset-specific info (only valid if !is_group)
    std::string dtype;          // e.g., "float32", "uint8", "int32"
    std::vector<size_t> shape;  // e.g., [10000, 224, 224, 3]
    bool is_chunked = false;
    std::string compression;    // e.g., "gzip", "none"
    size_t size_bytes = 0;      // Total size in bytes

    // Children (only valid if is_group)
    std::vector<HDF5NodeInfo> children;

    // Helper methods
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

    std::string GetSizeString() const {
        const char* units[] = {"B", "KB", "MB", "GB"};
        int unit_idx = 0;
        double size = static_cast<double>(size_bytes);
        while (size >= 1024.0 && unit_idx < 3) {
            size /= 1024.0;
            unit_idx++;
        }
        char buf[32];
        snprintf(buf, sizeof(buf), "%.2f %s", size, units[unit_idx]);
        return buf;
    }

    // Check if this dataset looks like image data
    bool LooksLikeImageData() const {
        if (is_group) return false;
        // 3D: [N, H, W] grayscale or 4D: [N, H, W, C] color
        if (shape.size() == 3 || shape.size() == 4) {
            // Reasonable image dimensions
            if (shape.size() >= 3 && shape[1] >= 8 && shape[1] <= 4096 &&
                shape[2] >= 8 && shape[2] <= 4096) {
                return true;
            }
        }
        return false;
    }

    // Check if this dataset looks like label data
    bool LooksLikeLabelData() const {
        if (is_group) return false;
        // 1D array of integers
        if (shape.size() == 1 &&
            (dtype == "int32" || dtype == "int64" || dtype == "uint8" || dtype == "float32")) {
            return true;
        }
        return false;
    }
};

/**
 * HDF5 file browser utility
 * Provides functions to browse HDF5 file structure without loading data
 */
class HDF5Browser {
public:
    /**
     * Browse an HDF5 file and return its structure
     * @param path Path to HDF5 file
     * @return Root node containing all groups and datasets
     */
    static HDF5NodeInfo Browse(const std::string& path);

    /**
     * Get info for a specific dataset
     * @param path Path to HDF5 file
     * @param dataset_path Path within the file (e.g., "/images")
     * @return Dataset info, or empty node if not found
     */
    static HDF5NodeInfo GetDatasetInfo(const std::string& path, const std::string& dataset_path);

    /**
     * Check if file is valid HDF5
     * @param path Path to file
     * @return true if valid HDF5 file
     */
    static bool IsValidHDF5(const std::string& path);

    /**
     * Auto-detect data and label paths in an HDF5 file
     * @param path Path to HDF5 file
     * @param[out] data_path Detected data path (empty if not found)
     * @param[out] label_path Detected label path (empty if not found)
     * @return true if at least data_path was found
     */
    static bool AutoDetectPaths(const std::string& path,
                                 std::string& data_path,
                                 std::string& label_path);

    /**
     * Get flat list of all datasets (non-groups) in file
     * @param path Path to HDF5 file
     * @return Vector of dataset infos
     */
    static std::vector<HDF5NodeInfo> ListDatasets(const std::string& path);

    /**
     * Read preview samples from a dataset
     * @param path Path to HDF5 file
     * @param dataset_path Path within the file
     * @param num_samples Number of samples to read
     * @param[out] data Output data (flattened)
     * @param[out] sample_shape Shape of each sample
     * @return true on success
     */
    static bool ReadPreviewSamples(const std::string& path,
                                    const std::string& dataset_path,
                                    size_t num_samples,
                                    std::vector<float>& data,
                                    std::vector<size_t>& sample_shape);

private:
    static void BrowseGroup(void* group_ptr, HDF5NodeInfo& parent);
    static std::string DTypeToString(void* dtype_ptr);
};

} // namespace cyxwiz
