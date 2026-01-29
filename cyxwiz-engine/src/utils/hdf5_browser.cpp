#include "hdf5_browser.h"
#include <spdlog/spdlog.h>
#include <filesystem>
#include <algorithm>

#ifdef CYXWIZ_HAS_HDF5
#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5Group.hpp>
#endif

namespace cyxwiz {

#ifdef CYXWIZ_HAS_HDF5

std::string HDF5Browser::DTypeToString(void* dtype_ptr) {
    auto& dtype = *static_cast<HighFive::DataType*>(dtype_ptr);

    if (dtype == HighFive::AtomicType<float>()) return "float32";
    if (dtype == HighFive::AtomicType<double>()) return "float64";
    if (dtype == HighFive::AtomicType<uint8_t>()) return "uint8";
    if (dtype == HighFive::AtomicType<int8_t>()) return "int8";
    if (dtype == HighFive::AtomicType<uint16_t>()) return "uint16";
    if (dtype == HighFive::AtomicType<int16_t>()) return "int16";
    if (dtype == HighFive::AtomicType<uint32_t>()) return "uint32";
    if (dtype == HighFive::AtomicType<int32_t>()) return "int32";
    if (dtype == HighFive::AtomicType<uint64_t>()) return "uint64";
    if (dtype == HighFive::AtomicType<int64_t>()) return "int64";

    return "unknown";
}

void HDF5Browser::BrowseGroup(void* group_ptr, HDF5NodeInfo& parent) {
    auto& group = *static_cast<HighFive::Group*>(group_ptr);

    // Get all object names in this group
    std::vector<std::string> names = group.listObjectNames();

    for (const auto& name : names) {
        HDF5NodeInfo child;
        child.name = name;
        child.path = parent.path + "/" + name;
        if (child.path.substr(0, 2) == "//") {
            child.path = child.path.substr(1);  // Remove double slash at root
        }

        try {
            auto obj_type = group.getObjectType(name);

            if (obj_type == HighFive::ObjectType::Group) {
                child.is_group = true;
                auto sub_group = group.getGroup(name);
                BrowseGroup(&sub_group, child);
            } else if (obj_type == HighFive::ObjectType::Dataset) {
                child.is_group = false;
                auto dataset = group.getDataSet(name);
                auto dims = dataset.getDimensions();
                auto dtype = dataset.getDataType();

                child.shape = dims;
                child.dtype = DTypeToString(&dtype);

                // Calculate size
                size_t elem_size = dtype.getSize();
                size_t total_elems = 1;
                for (auto d : dims) total_elems *= d;
                child.size_bytes = total_elems * elem_size;

                // Check for chunking and compression
                auto props = dataset.getCreatePropertyList();
                // Note: HighFive doesn't expose easy chunking/compression info
                // We'll leave these as defaults for now
                child.is_chunked = false;
                child.compression = "unknown";
            }

            parent.children.push_back(std::move(child));
        } catch (const std::exception& e) {
            spdlog::warn("HDF5Browser: Could not read object '{}': {}", name, e.what());
        }
    }
}

HDF5NodeInfo HDF5Browser::Browse(const std::string& path) {
    HDF5NodeInfo root;
    root.name = std::filesystem::path(path).filename().string();
    root.path = "/";
    root.is_group = true;

    try {
        HighFive::File file(path, HighFive::File::ReadOnly);
        auto root_group = file.getGroup("/");
        BrowseGroup(&root_group, root);
    } catch (const std::exception& e) {
        spdlog::error("HDF5Browser: Failed to browse '{}': {}", path, e.what());
    }

    return root;
}

HDF5NodeInfo HDF5Browser::GetDatasetInfo(const std::string& path, const std::string& dataset_path) {
    HDF5NodeInfo info;
    info.name = std::filesystem::path(dataset_path).filename().string();
    info.path = dataset_path;

    try {
        HighFive::File file(path, HighFive::File::ReadOnly);

        if (!file.exist(dataset_path)) {
            return info;
        }

        auto dataset = file.getDataSet(dataset_path);
        auto dims = dataset.getDimensions();
        auto dtype = dataset.getDataType();

        info.is_group = false;
        info.shape = dims;
        info.dtype = DTypeToString(&dtype);

        size_t elem_size = dtype.getSize();
        size_t total_elems = 1;
        for (auto d : dims) total_elems *= d;
        info.size_bytes = total_elems * elem_size;

    } catch (const std::exception& e) {
        spdlog::error("HDF5Browser: Failed to get dataset info: {}", e.what());
    }

    return info;
}

bool HDF5Browser::IsValidHDF5(const std::string& path) {
    try {
        HighFive::File file(path, HighFive::File::ReadOnly);
        return true;
    } catch (...) {
        return false;
    }
}

bool HDF5Browser::AutoDetectPaths(const std::string& path,
                                   std::string& data_path,
                                   std::string& label_path) {
    data_path.clear();
    label_path.clear();

    try {
        HighFive::File file(path, HighFive::File::ReadOnly);

        // Common data path names (priority order)
        const std::vector<std::string> data_names = {
            "images", "data", "features", "X", "x", "inputs",
            "x_train", "train/images", "train/data"
        };

        // Common label path names
        const std::vector<std::string> label_names = {
            "labels", "targets", "y", "Y", "outputs",
            "y_train", "train/labels", "train/targets"
        };

        // Find data dataset
        for (const auto& name : data_names) {
            if (file.exist(name)) {
                try {
                    auto ds = file.getDataSet(name);
                    auto dims = ds.getDimensions();
                    if (dims.size() >= 1) {
                        data_path = "/" + name;
                        break;
                    }
                } catch (...) {}
            }
        }

        // Find labels dataset
        for (const auto& name : label_names) {
            if (file.exist(name)) {
                try {
                    auto ds = file.getDataSet(name);
                    auto dims = ds.getDimensions();
                    if (dims.size() == 1) {
                        label_path = "/" + name;
                        break;
                    }
                } catch (...) {}
            }
        }

    } catch (const std::exception& e) {
        spdlog::error("HDF5Browser: AutoDetect failed: {}", e.what());
        return false;
    }

    return !data_path.empty();
}

std::vector<HDF5NodeInfo> HDF5Browser::ListDatasets(const std::string& path) {
    std::vector<HDF5NodeInfo> datasets;

    auto root = Browse(path);

    // Recursive lambda to collect all datasets
    std::function<void(const HDF5NodeInfo&)> collect = [&](const HDF5NodeInfo& node) {
        if (!node.is_group) {
            datasets.push_back(node);
        }
        for (const auto& child : node.children) {
            collect(child);
        }
    };

    for (const auto& child : root.children) {
        collect(child);
    }

    return datasets;
}

bool HDF5Browser::ReadPreviewSamples(const std::string& path,
                                      const std::string& dataset_path,
                                      size_t num_samples,
                                      std::vector<float>& data,
                                      std::vector<size_t>& sample_shape) {
    data.clear();
    sample_shape.clear();

    try {
        HighFive::File file(path, HighFive::File::ReadOnly);

        if (!file.exist(dataset_path)) {
            spdlog::error("HDF5Browser: Dataset '{}' not found", dataset_path);
            return false;
        }

        auto dataset = file.getDataSet(dataset_path);
        auto dims = dataset.getDimensions();
        auto dtype = dataset.getDataType();

        if (dims.empty()) return false;

        // Clamp num_samples to available
        num_samples = std::min(num_samples, dims[0]);

        // Calculate sample shape and size
        size_t sample_size = 1;
        for (size_t i = 1; i < dims.size(); i++) {
            sample_shape.push_back(dims[i]);
            sample_size *= dims[i];
        }
        if (dims.size() == 1) {
            sample_shape = {1};
            sample_size = 1;
        }

        // Read samples - need to handle different dimensionalities
        data.resize(num_samples * sample_size);

        // Build selection for reading sample by sample
        std::vector<size_t> offset(dims.size(), 0);
        std::vector<size_t> count(dims.size(), 1);
        for (size_t d = 1; d < dims.size(); d++) {
            count[d] = dims[d];
        }

        // Determine normalization
        bool normalize = (dtype == HighFive::AtomicType<uint8_t>());
        float scale = normalize ? 1.0f / 255.0f : 1.0f;

        // Read sample by sample to handle any dimensionality
        for (size_t s = 0; s < num_samples; s++) {
            offset[0] = s;
            auto selection = dataset.select(offset, count);
            size_t out_offset = s * sample_size;

            if (dims.size() == 1) {
                // 1D: [N] - scalar per sample
                if (dtype == HighFive::AtomicType<float>()) {
                    float val;
                    selection.read(val);
                    data[out_offset] = val;
                } else if (dtype == HighFive::AtomicType<uint8_t>()) {
                    uint8_t val;
                    selection.read(val);
                    data[out_offset] = static_cast<float>(val) * scale;
                } else if (dtype == HighFive::AtomicType<int32_t>()) {
                    int32_t val;
                    selection.read(val);
                    data[out_offset] = static_cast<float>(val);
                } else if (dtype == HighFive::AtomicType<double>()) {
                    double val;
                    selection.read(val);
                    data[out_offset] = static_cast<float>(val);
                }
            } else if (dims.size() == 2) {
                // 2D: [N, features]
                if (dtype == HighFive::AtomicType<float>()) {
                    std::vector<float> row(sample_size);
                    selection.read(row);
                    std::copy(row.begin(), row.end(), data.begin() + out_offset);
                } else if (dtype == HighFive::AtomicType<uint8_t>()) {
                    std::vector<uint8_t> row(sample_size);
                    selection.read(row);
                    for (size_t i = 0; i < sample_size; i++) {
                        data[out_offset + i] = static_cast<float>(row[i]) * scale;
                    }
                } else if (dtype == HighFive::AtomicType<int32_t>()) {
                    std::vector<int32_t> row(sample_size);
                    selection.read(row);
                    for (size_t i = 0; i < sample_size; i++) {
                        data[out_offset + i] = static_cast<float>(row[i]);
                    }
                } else if (dtype == HighFive::AtomicType<double>()) {
                    std::vector<double> row(sample_size);
                    selection.read(row);
                    for (size_t i = 0; i < sample_size; i++) {
                        data[out_offset + i] = static_cast<float>(row[i]);
                    }
                }
            } else if (dims.size() == 3) {
                // 3D: [N, H, W] - grayscale images
                size_t h = dims[1], w = dims[2];
                if (dtype == HighFive::AtomicType<float>()) {
                    std::vector<std::vector<float>> img(h, std::vector<float>(w));
                    selection.read(img);
                    size_t k = 0;
                    for (const auto& row : img) {
                        for (float val : row) {
                            data[out_offset + k++] = val;
                        }
                    }
                } else if (dtype == HighFive::AtomicType<uint8_t>()) {
                    std::vector<std::vector<uint8_t>> img(h, std::vector<uint8_t>(w));
                    selection.read(img);
                    size_t k = 0;
                    for (const auto& row : img) {
                        for (uint8_t val : row) {
                            data[out_offset + k++] = static_cast<float>(val) * scale;
                        }
                    }
                } else if (dtype == HighFive::AtomicType<double>()) {
                    std::vector<std::vector<double>> img(h, std::vector<double>(w));
                    selection.read(img);
                    size_t k = 0;
                    for (const auto& row : img) {
                        for (double val : row) {
                            data[out_offset + k++] = static_cast<float>(val);
                        }
                    }
                }
            } else if (dims.size() == 4) {
                // 4D: [N, H, W, C] - color images
                size_t h = dims[1], w = dims[2], c = dims[3];
                if (dtype == HighFive::AtomicType<float>()) {
                    std::vector<std::vector<std::vector<float>>> img(h,
                        std::vector<std::vector<float>>(w, std::vector<float>(c)));
                    selection.read(img);
                    size_t k = 0;
                    for (const auto& row : img) {
                        for (const auto& pixel : row) {
                            for (float val : pixel) {
                                data[out_offset + k++] = val;
                            }
                        }
                    }
                } else if (dtype == HighFive::AtomicType<uint8_t>()) {
                    std::vector<std::vector<std::vector<uint8_t>>> img(h,
                        std::vector<std::vector<uint8_t>>(w, std::vector<uint8_t>(c)));
                    selection.read(img);
                    size_t k = 0;
                    for (const auto& row : img) {
                        for (const auto& pixel : row) {
                            for (uint8_t val : pixel) {
                                data[out_offset + k++] = static_cast<float>(val) * scale;
                            }
                        }
                    }
                } else if (dtype == HighFive::AtomicType<double>()) {
                    std::vector<std::vector<std::vector<double>>> img(h,
                        std::vector<std::vector<double>>(w, std::vector<double>(c)));
                    selection.read(img);
                    size_t k = 0;
                    for (const auto& row : img) {
                        for (const auto& pixel : row) {
                            for (double val : pixel) {
                                data[out_offset + k++] = static_cast<float>(val);
                            }
                        }
                    }
                }
            } else {
                spdlog::warn("HDF5Browser: Unsupported dimensionality {} for preview", dims.size());
                return false;
            }
        }

        return true;

    } catch (const std::exception& e) {
        spdlog::error("HDF5Browser: Failed to read preview: {}", e.what());
        return false;
    }
}

#else // !CYXWIZ_HAS_HDF5

// Stub implementations when HDF5 is not available
HDF5NodeInfo HDF5Browser::Browse(const std::string& path) {
    HDF5NodeInfo root;
    root.name = "HDF5 support not available";
    root.is_group = true;
    return root;
}

HDF5NodeInfo HDF5Browser::GetDatasetInfo(const std::string&, const std::string&) {
    return HDF5NodeInfo{};
}

bool HDF5Browser::IsValidHDF5(const std::string&) {
    return false;
}

bool HDF5Browser::AutoDetectPaths(const std::string&, std::string&, std::string&) {
    return false;
}

std::vector<HDF5NodeInfo> HDF5Browser::ListDatasets(const std::string&) {
    return {};
}

bool HDF5Browser::ReadPreviewSamples(const std::string&, const std::string&,
                                      size_t, std::vector<float>&, std::vector<size_t>&) {
    return false;
}

void HDF5Browser::BrowseGroup(void*, HDF5NodeInfo&) {}
std::string HDF5Browser::DTypeToString(void*) { return ""; }

#endif // CYXWIZ_HAS_HDF5

} // namespace cyxwiz
