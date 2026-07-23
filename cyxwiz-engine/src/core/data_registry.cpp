#include "data_registry.h"
#include "dataset_base.h"
#include "arrow_dataset.h"
#include "parquet_backed_dataset.h"
#include <arrow/compute/api.h>  // for arrow::compute::Cast (integer column compaction)
#include <limits>              // for std::numeric_limits (Arrow block_size cap)

// Cross-platform headers for GetAvailableMemoryBytes(). Each branch is gated
// by the platform preprocessor so only one is ever compiled, and the fallback
// path uses a safe conservative default so builds without an implementation
// still work.
#ifdef _WIN32
// <windows.h> defines LoadImage as a preprocessor macro (LoadImageA or
// LoadImageW depending on UNICODE). That collides with
// cyxwiz::ImageUtils::LoadImage - under unity builds the ImageFolderDataset
// code compiled into this TU gets its LoadImage call rewritten to
// LoadImageA, which then fails at link time. NOGDI / WIN32_LEAN_AND_MEAN
// don't exclude it because LoadImage lives in <winuser.h>, not GDI.
// The surgical fix is to #undef it right after the include.
#define WIN32_LEAN_AND_MEAN
#define NOGDI
#include <windows.h>
#ifdef LoadImage
#undef LoadImage
#endif
#elif defined(__linux__)
#include <sys/sysinfo.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#endif
#include "annotation_manager.h"
#include "image_utils.h"
#include "../preprocessing/preprocessing_config.h"
#include "../transforms/transform.h"
#include "../plugin/registries/plugin_data_loader_registry.h"

// Dataset implementations
#include "datasets/kaggle_dataset.h"
#include "datasets/hdf5_dataset.h"
#include "datasets/image_csv_dataset.h"
#include "datasets/streaming_dataset.h"
#include "datasets/mnist_dataset.h"
#include "datasets/cifar10_dataset.h"
#include "datasets/csv_dataset.h"
#include "datasets/tsv_dataset.h"
#include "datasets/json_dataset.h"
#include "datasets/txt_dataset.h"
#include "datasets/image_folder_dataset.h"
#include "datasets/huggingface_dataset.h"
#include "datasets/custom_dataset.h"
#include <spdlog/spdlog.h>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <random>
#include <sstream>
#include <cstring>
#include <set>
#include <list>
#include <optional>
#include <unordered_map>
#include <numeric>
#include <chrono>
#include <ctime>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <nlohmann/json.hpp>

// stb_image for image loading (implementation in stb_image_impl.cpp)
#include <stb_image.h>

// HDF5 support (optional - only if HighFive is available)
#ifdef CYXWIZ_HAS_HDF5
#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#endif

namespace fs = std::filesystem;

namespace cyxwiz {

// =============================================================================
// DataRegistry Implementation
// =============================================================================

DatasetHandle DataRegistry::LoadDataset(const std::string& path, const std::string& name) {
    DatasetType type = DetectType(path);

    switch (type) {
        case DatasetType::MNIST:
            return LoadMNIST(path, name.empty() ? "mnist" : name);
        case DatasetType::CIFAR10:
            return LoadCIFAR10(path, name.empty() ? "cifar10" : name);
        case DatasetType::CSV:
            return LoadCSV(path, name);
        case DatasetType::TSV:
            return LoadTSV(path, name);
        case DatasetType::JSON:
            return LoadJSON(path, name);
        case DatasetType::TXT:
            return LoadTXT(path, name);
        case DatasetType::ImageFolder:
            return LoadImageFolder(path, name);
        case DatasetType::HDF5:
            return LoadHDF5(path, name);
        default: {
            // Check if a plugin data loader can handle this format
            try {
                auto ext = std::filesystem::path(path).extension().string();
                if (!ext.empty() && cyxwiz::plugin::PluginDataLoaderRegistry::Instance().HasLoaderForExtension(ext)) {
                    spdlog::info("DataRegistry: Plugin data loader available for '{}' (bridge not yet implemented)", path);
                    // TODO: Bridge PluginDataset to DatasetHandle with adapter class
                }
            } catch (const std::exception& e) {
                spdlog::warn("DataRegistry: Plugin loader check failed: {}", e.what());
            }
            spdlog::error("Unknown dataset type for path: {}", path);
            return DatasetHandle();
        }
    }
}

DatasetHandle DataRegistry::LoadMNIST(const std::string& path, const std::string& name) {
    std::string unique_name = GenerateUniqueName(name);

    try {
        auto dataset = std::make_shared<MNISTDataset>(path);
        if (dataset->Size() == 0) {
            return DatasetHandle();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (on_loaded_) {
            on_loaded_(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered MNIST dataset as '{}'", unique_name);
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load MNIST dataset: {}", e.what());
        return DatasetHandle();
    }
}

DatasetHandle DataRegistry::LoadCIFAR10(const std::string& path, const std::string& name) {
    std::string unique_name = GenerateUniqueName(name);

    try {
        auto dataset = std::make_shared<CIFAR10Dataset>(path);
        if (dataset->Size() == 0) {
            return DatasetHandle();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (on_loaded_) {
            on_loaded_(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered CIFAR-10 dataset as '{}'", unique_name);
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load CIFAR-10 dataset: {}", e.what());
        return DatasetHandle();
    }
}

DatasetHandle DataRegistry::LoadCSV(const std::string& path, const std::string& name) {
    std::string base_name = name.empty() ? fs::path(path).stem().string() : name;
    std::string unique_name = GenerateUniqueName(base_name);

    try {
        auto dataset = std::make_shared<CSVDataset>(path);
        if (dataset->Size() == 0) {
            return DatasetHandle();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (on_loaded_) {
            on_loaded_(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered CSV dataset as '{}'", unique_name);
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load CSV dataset: {}", e.what());
        return DatasetHandle();
    }
}

DatasetHandle DataRegistry::LoadTSV(const std::string& path, const std::string& name) {
    std::string base_name = name.empty() ? fs::path(path).stem().string() : name;
    std::string unique_name = GenerateUniqueName(base_name);

    try {
        auto dataset = std::make_shared<TSVDataset>(path);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (on_loaded_) {
            on_loaded_(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered TSV dataset as '{}'", unique_name);
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load TSV dataset: {}", e.what());
        return DatasetHandle();
    }
}

DatasetHandle DataRegistry::LoadJSON(const std::string& path, const std::string& name) {
    std::string base_name = name.empty() ? fs::path(path).stem().string() : name;
    std::string unique_name = GenerateUniqueName(base_name);

    try {
        auto dataset = std::make_shared<JSONDataset>(path);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (on_loaded_) {
            on_loaded_(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered JSON dataset as '{}'", unique_name);
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load JSON dataset: {}", e.what());
        return DatasetHandle();
    }
}

DatasetHandle DataRegistry::LoadTXT(const std::string& path, const std::string& name) {
    std::string base_name = name.empty() ? fs::path(path).stem().string() : name;
    std::string unique_name = GenerateUniqueName(base_name);

    try {
        auto dataset = std::make_shared<TXTDataset>(path);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (on_loaded_) {
            on_loaded_(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered TXT dataset as '{}'", unique_name);
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load TXT dataset: {}", e.what());
        return DatasetHandle();
    }
}

DatasetHandle DataRegistry::LoadImageFolder(const std::string& path, const std::string& name) {
    std::string base_name = name.empty() ? fs::path(path).filename().string() : name;
    std::string unique_name = GenerateUniqueName(base_name);

    try {
        auto dataset = std::make_shared<ImageFolderDataset>(path);
        if (dataset->Size() == 0) {
            return DatasetHandle();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (on_loaded_) {
            on_loaded_(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered ImageFolder dataset as '{}'", unique_name);
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load ImageFolder dataset: {}", e.what());
        return DatasetHandle();
    }
}

DatasetHandle DataRegistry::LoadImageCSV(const std::string& image_folder, const std::string& csv_path,
                                          const std::string& name, int target_width, int target_height,
                                          size_t cache_size) {
    std::string base_name = name.empty() ? fs::path(image_folder).filename().string() : name;
    std::string unique_name = GenerateUniqueName(base_name);

    try {
        auto dataset = std::make_shared<ImageCSVDataset>(image_folder, csv_path, target_width, target_height, cache_size);
        if (dataset->Size() == 0) {
            spdlog::warn("ImageCSV dataset loaded with 0 samples");
            return DatasetHandle();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (on_loaded_) {
            on_loaded_(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered ImageCSV dataset as '{}' with {} samples", unique_name, dataset->Size());
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load ImageCSV dataset: {}", e.what());
        return DatasetHandle();
    }
}

DatasetHandle DataRegistry::LoadHuggingFace(const HuggingFaceConfig& config, const std::string& name) {
    std::string base_name = name.empty() ? config.dataset_name : name;
    std::string unique_name = GenerateUniqueName(base_name);

    try {
        auto dataset = std::make_shared<HuggingFaceDataset>(config);
        if (dataset->Size() == 0) {
            spdlog::warn("HuggingFace dataset '{}' loaded with 0 samples", config.dataset_name);
            return DatasetHandle();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (on_loaded_) {
            on_loaded_(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered HuggingFace dataset '{}' as '{}'", config.dataset_name, unique_name);
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load HuggingFace dataset '{}': {}", config.dataset_name, e.what());
        return DatasetHandle();
    }
}

DatasetHandle DataRegistry::LoadStreamingDataset(const std::string& path, const StreamingConfig& config, const std::string& name) {
    std::string base_name = name.empty() ? fs::path(path).stem().string() : name;
    std::string unique_name = GenerateUniqueName(base_name);

    try {
        auto dataset = std::make_shared<StreamingDataset>(path, config);
        if (dataset->Size() == 0) {
            spdlog::warn("Streaming dataset '{}' has estimated 0 samples", path);
            return DatasetHandle();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (on_loaded_) {
            on_loaded_(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered streaming dataset as '{}' ({} estimated samples)",
                    unique_name, dataset->Size());
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load streaming dataset '{}': {}", path, e.what());
        return DatasetHandle();
    }
}

DatasetHandle DataRegistry::LoadKaggle(const KaggleConfig& config, const std::string& name) {
    std::string base_name = name;
    if (base_name.empty()) {
        // Extract name from dataset_slug or competition
        if (!config.dataset_slug.empty()) {
            size_t pos = config.dataset_slug.find_last_of('/');
            base_name = (pos != std::string::npos) ? config.dataset_slug.substr(pos + 1) : config.dataset_slug;
        } else {
            base_name = config.competition;
        }
    }
    std::string unique_name = GenerateUniqueName(base_name);

    try {
        auto dataset = std::make_shared<KaggleDataset>(config);
        if (dataset->Size() == 0) {
            spdlog::warn("Kaggle dataset loaded with 0 samples");
            return DatasetHandle();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (on_loaded_) {
            on_loaded_(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered Kaggle dataset as '{}' ({} samples)",
                    unique_name, dataset->Size());
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load Kaggle dataset: {}", e.what());
        return DatasetHandle();
    }
}

DatasetHandle DataRegistry::LoadCustom(const CustomConfig& config, const std::string& name) {
    try {
        std::string unique_name = name.empty() ? GenerateUniqueName("custom") : name;

        // Check if already loaded
        if (HasDataset(unique_name)) {
            spdlog::warn("Dataset '{}' already loaded, returning existing", unique_name);
            return GetDataset(unique_name);
        }

        spdlog::info("Loading custom dataset from: {}", config.data_path);

        auto dataset = std::make_shared<CustomDataset>(config);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (on_loaded_) {
            on_loaded_(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered custom dataset as '{}' ({} samples)",
                    unique_name, dataset->Size());
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load custom dataset: {}", e.what());
        return DatasetHandle();
    }
}

DatasetHandle DataRegistry::LoadHDF5(const std::string& path, const std::string& name,
                                      const HDF5Config& config) {
#ifdef CYXWIZ_HAS_HDF5
    std::string base_name = name.empty() ? fs::path(path).stem().string() : name;
    std::string unique_name = GenerateUniqueName(base_name);

    try {
        auto dataset = std::make_shared<HDF5Dataset>(path, config);
        if (dataset->Size() == 0) {
            spdlog::warn("HDF5 dataset is empty: {}", path);
            return DatasetHandle();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (on_loaded_) {
            on_loaded_(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered HDF5 dataset '{}': {} samples", unique_name, dataset->Size());
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load HDF5 dataset: {}", e.what());
        return DatasetHandle();
    }
#else
    spdlog::error("HDF5 support not compiled (HighFive library missing)");
    return DatasetHandle();
#endif
}

bool DataRegistry::ExportHDF5(const std::string& name, const std::string& filepath,
                               const HDF5ExportConfig& config) {
    auto handle = GetDataset(name);
    if (!handle.IsValid()) {
        spdlog::error("ExportHDF5: Dataset '{}' not found", name);
        return false;
    }
    return ExportHDF5(handle, filepath, config);
}

bool DataRegistry::ExportHDF5(DatasetHandle handle, const std::string& filepath,
                               const HDF5ExportConfig& config) {
#ifdef CYXWIZ_HAS_HDF5
    if (!handle.IsValid()) {
        spdlog::error("ExportHDF5: Invalid dataset handle");
        return false;
    }

    try {
        auto info = handle.GetInfo();
        spdlog::info("Exporting dataset to HDF5: {} ({} samples)", filepath, info.num_samples);

        // Create HDF5 file
        HighFive::File file(filepath, HighFive::File::Overwrite);

        // Get sample shape
        std::vector<size_t> sample_shape = info.shape;
        if (sample_shape.empty()) {
            sample_shape = {1};  // Scalar samples
        }

        // Calculate total data shape: [N, ...sample_shape]
        std::vector<size_t> data_shape;
        data_shape.push_back(info.num_samples);

        // Handle NCHW transpose if requested
        bool do_nchw = config.store_as_nchw && sample_shape.size() == 3;
        if (do_nchw) {
            // Input is NHWC [H, W, C], output should be NCHW [C, H, W]
            data_shape.push_back(sample_shape[2]);  // C
            data_shape.push_back(sample_shape[0]);  // H
            data_shape.push_back(sample_shape[1]);  // W
        } else {
            for (auto d : sample_shape) {
                data_shape.push_back(d);
            }
        }

        // Set up chunking
        std::vector<size_t> chunk_dims = data_shape;
        if (config.chunked && info.num_samples > config.chunk_samples) {
            chunk_dims[0] = std::min(config.chunk_samples, info.num_samples);
        }

        // Create property list for compression and chunking
        HighFive::DataSetCreateProps props;
        if (config.chunked) {
            props.add(HighFive::Chunking(chunk_dims));
        }
        if (config.compress && config.chunked) {
            props.add(HighFive::Deflate(config.compression_level));
        }

        // Calculate sample size
        size_t sample_size = 1;
        for (auto d : sample_shape) {
            sample_size *= d;
        }

        // Read all data from dataset
        spdlog::info("Reading {} samples from dataset...", info.num_samples);
        std::vector<float> all_data;
        all_data.reserve(info.num_samples * sample_size);

        std::vector<int> all_labels;
        all_labels.reserve(info.num_samples);

        for (size_t i = 0; i < info.num_samples; i++) {
            auto [data, label] = handle.GetSample(i);

            // Transpose NHWC to NCHW if requested
            if (do_nchw && sample_shape.size() == 3) {
                size_t H = sample_shape[0];
                size_t W = sample_shape[1];
                size_t C = sample_shape[2];
                std::vector<float> transposed(sample_size);

                // NHWC [h, w, c] -> NCHW [c, h, w]
                for (size_t h = 0; h < H; h++) {
                    for (size_t w = 0; w < W; w++) {
                        for (size_t c = 0; c < C; c++) {
                            size_t src_idx = h * W * C + w * C + c;  // NHWC
                            size_t dst_idx = c * H * W + h * W + w;  // NCHW
                            transposed[dst_idx] = data[src_idx];
                        }
                    }
                }
                all_data.insert(all_data.end(), transposed.begin(), transposed.end());
            } else {
                all_data.insert(all_data.end(), data.begin(), data.end());
            }

            all_labels.push_back(label);

            if ((i + 1) % 1000 == 0) {
                spdlog::info("Read {}/{} samples", i + 1, info.num_samples);
            }
        }

        // Create and write data dataset
        if (config.store_as_uint8) {
            // Convert float [0,1] to uint8 [0,255]
            std::vector<uint8_t> uint8_data(all_data.size());
            for (size_t i = 0; i < all_data.size(); i++) {
                float val = std::clamp(all_data[i], 0.0f, 1.0f) * 255.0f;
                uint8_data[i] = static_cast<uint8_t>(val);
            }

            auto data_ds = file.createDataSet<uint8_t>(config.data_path,
                HighFive::DataSpace(data_shape), props);
            data_ds.write_raw(uint8_data.data());
            spdlog::info("Wrote data as uint8 to {}", config.data_path);
        } else {
            auto data_ds = file.createDataSet<float>(config.data_path,
                HighFive::DataSpace(data_shape), props);
            data_ds.write_raw(all_data.data());
            spdlog::info("Wrote data as float32 to {}", config.data_path);
        }

        // Create and write labels dataset
        auto label_ds = file.createDataSet<int>(config.label_path,
            HighFive::DataSpace({info.num_samples}));
        label_ds.write(all_labels);
        spdlog::info("Wrote {} labels to {}", info.num_samples, config.label_path);

        // Write metadata
        if (config.include_metadata) {
            auto root = file.getGroup("/");

            // Number of samples
            root.createAttribute("num_samples", info.num_samples);

            // Number of classes
            root.createAttribute("num_classes", info.num_classes);

            // Sample shape
            if (!sample_shape.empty()) {
                root.createAttribute("sample_shape", sample_shape);
            }

            // Data format
            std::string format = do_nchw ? "NCHW" : "NHWC";
            root.createAttribute("data_format", format);

            // Class names if provided
            if (!config.class_names.empty()) {
                root.createAttribute("class_names", config.class_names);
            }

            spdlog::info("Wrote metadata attributes");
        }

        spdlog::info("Successfully exported dataset to: {}", filepath);
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Failed to export HDF5 dataset: {}", e.what());
        return false;
    }
#else
    spdlog::error("HDF5 support not compiled (HighFive library missing)");
    return false;
#endif
}

DatasetHandle DataRegistry::GetDataset(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = datasets_.find(name);
    if (it != datasets_.end()) {
        // Update LRU access time
        last_access_times_[name] = std::chrono::steady_clock::now();
        return DatasetHandle(it->second, name);
    }
    return DatasetHandle();
}

bool DataRegistry::HasDataset(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return datasets_.find(name) != datasets_.end();
}

std::vector<DatasetInfo> DataRegistry::ListDatasets() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<DatasetInfo> result;
    result.reserve(datasets_.size());

    for (const auto& [name, dataset] : datasets_) {
        auto info = dataset->GetInfo();
        info.name = name;
        result.push_back(info);
    }

    return result;
}

std::vector<std::string> DataRegistry::GetDatasetNames() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> names;
    names.reserve(datasets_.size());

    for (const auto& [name, _] : datasets_) {
        names.push_back(name);
    }

    return names;
}

DatasetType DataRegistry::DetectType(const std::string& path) {
    if (!fs::exists(path)) {
        return DatasetType::None;
    }

    // Check for directory-based datasets
    if (fs::is_directory(path)) {
        // MNIST
        if (fs::exists(path + "/train-images-idx3-ubyte") ||
            fs::exists(path + "/train-images.idx3-ubyte")) {
            return DatasetType::MNIST;
        }

        // CIFAR-10
        if (fs::exists(path + "/data_batch_1.bin")) {
            return DatasetType::CIFAR10;
        }

        // Check for image folder structure
        for (const auto& entry : fs::directory_iterator(path)) {
            if (entry.is_directory()) {
                for (const auto& sub : fs::directory_iterator(entry)) {
                    auto ext = sub.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                        return DatasetType::ImageFolder;
                    }
                }
            }
        }
    } else {
        // File-based datasets
        auto ext = fs::path(path).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".csv") return DatasetType::CSV;
        if (ext == ".tsv") return DatasetType::TSV;
        if (ext == ".json") return DatasetType::JSON;
        if (ext == ".txt") return DatasetType::TXT;
        if (ext == ".h5" || ext == ".hdf5" || ext == ".hdf") return DatasetType::HDF5;
    }

    return DatasetType::None;
}

std::string DataRegistry::TypeToString(DatasetType type) {
    switch (type) {
        case DatasetType::None: return "None";
        case DatasetType::CSV: return "CSV";
        case DatasetType::TSV: return "TSV";
        case DatasetType::JSON: return "JSON";
        case DatasetType::TXT: return "TXT";
        case DatasetType::ImageFolder: return "ImageFolder";
        case DatasetType::ImageCSV: return "ImageCSV";
        case DatasetType::MNIST: return "MNIST";
        case DatasetType::FashionMNIST: return "FashionMNIST";
        case DatasetType::CIFAR10: return "CIFAR-10";
        case DatasetType::CIFAR100: return "CIFAR-100";
        case DatasetType::HuggingFace: return "HuggingFace";
        case DatasetType::Kaggle: return "Kaggle";
        case DatasetType::Custom: return "Custom";
        case DatasetType::HDF5: return "HDF5";
        default: return "Unknown";
    }
}

size_t DataRegistry::GetTotalMemoryUsage() const {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t total = 0;
    for (const auto& [_, dataset] : datasets_) {
        total += dataset->GetInfo().memory_usage;
    }
    return total;
}

void DataRegistry::SetMemoryLimit(size_t bytes) {
    memory_limit_ = bytes;
}

MemoryStats DataRegistry::GetMemoryStats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    MemoryStats stats;
    stats.memory_limit = memory_limit_;
    stats.datasets_count = datasets_.size();
    stats.cache_hits = total_cache_hits_;
    stats.cache_misses = total_cache_misses_;
    stats.cache_evictions = total_cache_evictions_;

    // Sum up memory from all datasets
    for (const auto& [_, dataset] : datasets_) {
        auto info = dataset->GetInfo();
        stats.total_allocated += info.memory_usage;
        stats.total_cached += info.cache_usage;
    }

    // Update peak usage
    if (stats.total_allocated > peak_usage_) {
        peak_usage_ = stats.total_allocated;
    }
    stats.peak_usage = peak_usage_;

    // Get texture memory from TextureManager
    // (Note: TextureManager tracks its own memory)
    stats.texture_memory = 0;  // Will be set by caller if needed
    stats.texture_count = 0;

    return stats;
}

void DataRegistry::ResetCacheStats() {
    std::lock_guard<std::mutex> lock(mutex_);
    total_cache_hits_ = 0;
    total_cache_misses_ = 0;
    total_cache_evictions_ = 0;
}

// =============================================================================
// Memory Optimization
// =============================================================================

bool DataRegistry::IsMemoryPressure() const {
    size_t current = GetTotalMemoryUsage();
    return current >= static_cast<size_t>(memory_limit_ * memory_pressure_threshold_);
}

void DataRegistry::EvictOldest() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (datasets_.empty()) {
        return;
    }

    // Find the least recently accessed dataset
    std::string oldest_name;
    auto oldest_time = std::chrono::steady_clock::time_point::max();

    for (const auto& [name, _] : datasets_) {
        auto it = last_access_times_.find(name);
        auto access_time = (it != last_access_times_.end())
            ? it->second
            : std::chrono::steady_clock::time_point::min();  // Never accessed = oldest

        if (access_time < oldest_time) {
            oldest_time = access_time;
            oldest_name = name;
        }
    }

    if (!oldest_name.empty()) {
        auto info = datasets_[oldest_name]->GetInfo();
        spdlog::info("Memory eviction: unloading '{}' ({} bytes)", oldest_name, info.memory_usage);

        datasets_.erase(oldest_name);
        last_access_times_.erase(oldest_name);
        total_cache_evictions_++;

        if (on_unloaded_) {
            on_unloaded_(oldest_name);
        }
    }
}

void DataRegistry::TrimMemory(size_t target_bytes) {
    // If target is 0, use memory_limit_
    size_t target = (target_bytes > 0) ? target_bytes : memory_limit_;

    size_t current = GetTotalMemoryUsage();

    // Notify about memory pressure if callback is set
    if (current > memory_limit_ && on_memory_pressure_) {
        on_memory_pressure_(current, memory_limit_);
    }

    // Keep evicting until we're under target
    int eviction_count = 0;
    while (current > target && !datasets_.empty()) {
        EvictOldest();
        current = GetTotalMemoryUsage();
        eviction_count++;

        // Safety limit to prevent infinite loop
        if (eviction_count > 100) {
            spdlog::warn("TrimMemory: Safety limit reached after {} evictions", eviction_count);
            break;
        }
    }

    if (eviction_count > 0) {
        spdlog::info("TrimMemory: Evicted {} datasets, new usage: {} bytes", eviction_count, current);
    }
}

// =============================================================================
// Arrow Format-Specific Loaders (for DataInput node pipeline execution)
// =============================================================================

// -----------------------------------------------------------------------------
// CompactIntegerColumns
//
// Arrow's CSV reader promotes every integer column it sees to int64 regardless
// of the actual value range. For ML datasets like MNIST where pixel values fit
// in [0, 255] this means an 8x memory waste (8 bytes per cell instead of 1)
// AND 8x memory traffic per batch because the ArrowDatasetBatcher has to read
// int64 and cast to float.
//
// This helper walks the table once after load, computes min/max per integer
// column, and casts each column down to the smallest (unsigned preferred)
// integer type that still holds the observed range. It uses arrow::compute::Cast
// which is part of the Arrow core library we already link.
//
// Returns the downcast table (or the original if nothing could be compacted).
// Logs total memory savings at info level.
// -----------------------------------------------------------------------------
static std::shared_ptr<arrow::Table> CompactIntegerColumns(
    const std::shared_ptr<arrow::Table>& table) {
    if (!table) return table;

    size_t bytes_before = 0;
    size_t bytes_after = 0;
    int cols_compacted = 0;

    std::vector<std::shared_ptr<arrow::ChunkedArray>> new_columns;
    new_columns.reserve(table->num_columns());

    for (int i = 0; i < table->num_columns(); ++i) {
        auto col = table->column(i);
        auto type_id = col->type()->id();

        // Track original size for logging
        for (int c = 0; c < col->num_chunks(); ++c) {
            auto chunk = col->chunk(c);
            for (const auto& buf : chunk->data()->buffers) {
                if (buf) bytes_before += buf->size();
            }
        }

        // Only compact signed integer columns that Arrow CSV reader produces.
        // int8/int16/int32 are rare but handled; we skip already-unsigned types
        // (they're presumably intentional) and float/string/etc.
        bool is_signed_int = (type_id == arrow::Type::INT64 ||
                              type_id == arrow::Type::INT32 ||
                              type_id == arrow::Type::INT16 ||
                              type_id == arrow::Type::INT8);
        if (!is_signed_int) {
            new_columns.push_back(col);
            // Still count final size
            for (int c = 0; c < col->num_chunks(); ++c) {
                auto chunk = col->chunk(c);
                for (const auto& buf : chunk->data()->buffers) {
                    if (buf) bytes_after += buf->size();
                }
            }
            continue;
        }

        // Find min/max across all chunks by iterating raw values.
        // (arrow::compute::MinMax would also work but iterating is fine and
        // keeps us independent of the Arrow compute registry initialization.)
        int64_t min_val = std::numeric_limits<int64_t>::max();
        int64_t max_val = std::numeric_limits<int64_t>::min();
        bool all_nonnull = (col->null_count() == 0);
        bool have_data = false;

        auto observe = [&](int64_t v) {
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
            have_data = true;
        };

        for (int c = 0; c < col->num_chunks(); ++c) {
            auto chunk = col->chunk(c);
            int64_t n = chunk->length();
            if (n == 0) continue;

            switch (type_id) {
                case arrow::Type::INT64: {
                    auto arr = std::static_pointer_cast<arrow::Int64Array>(chunk);
                    const int64_t* data = arr->raw_values();
                    if (all_nonnull) {
                        for (int64_t r = 0; r < n; ++r) observe(data[r]);
                    } else {
                        for (int64_t r = 0; r < n; ++r)
                            if (!arr->IsNull(r)) observe(data[r]);
                    }
                    break;
                }
                case arrow::Type::INT32: {
                    auto arr = std::static_pointer_cast<arrow::Int32Array>(chunk);
                    const int32_t* data = arr->raw_values();
                    if (all_nonnull) {
                        for (int64_t r = 0; r < n; ++r) observe(data[r]);
                    } else {
                        for (int64_t r = 0; r < n; ++r)
                            if (!arr->IsNull(r)) observe(data[r]);
                    }
                    break;
                }
                case arrow::Type::INT16: {
                    auto arr = std::static_pointer_cast<arrow::Int16Array>(chunk);
                    const int16_t* data = arr->raw_values();
                    if (all_nonnull) {
                        for (int64_t r = 0; r < n; ++r) observe(data[r]);
                    } else {
                        for (int64_t r = 0; r < n; ++r)
                            if (!arr->IsNull(r)) observe(data[r]);
                    }
                    break;
                }
                case arrow::Type::INT8: {
                    auto arr = std::static_pointer_cast<arrow::Int8Array>(chunk);
                    const int8_t* data = arr->raw_values();
                    if (all_nonnull) {
                        for (int64_t r = 0; r < n; ++r) observe(data[r]);
                    } else {
                        for (int64_t r = 0; r < n; ++r)
                            if (!arr->IsNull(r)) observe(data[r]);
                    }
                    break;
                }
                default: break;
            }
        }

        // Pick smallest fitting target type
        std::shared_ptr<arrow::DataType> target_type;
        if (!have_data) {
            // Empty or all-null column — leave it alone
            target_type = col->type();
        } else if (min_val >= 0 && max_val <= 255) {
            target_type = arrow::uint8();
        } else if (min_val >= -128 && max_val <= 127) {
            target_type = arrow::int8();
        } else if (min_val >= 0 && max_val <= 65535) {
            target_type = arrow::uint16();
        } else if (min_val >= -32768 && max_val <= 32767) {
            target_type = arrow::int16();
        } else if (min_val >= 0 && max_val <= 4294967295LL) {
            target_type = arrow::uint32();
        } else if (min_val >= -2147483648LL && max_val <= 2147483647LL) {
            target_type = arrow::int32();
        } else {
            target_type = col->type();  // already int64, can't shrink
        }

        // Skip if target type is the same as source
        if (target_type->Equals(*col->type())) {
            new_columns.push_back(col);
            for (int c = 0; c < col->num_chunks(); ++c) {
                auto chunk = col->chunk(c);
                for (const auto& buf : chunk->data()->buffers) {
                    if (buf) bytes_after += buf->size();
                }
            }
            continue;
        }

        // Cast each chunk to the target type
        arrow::compute::CastOptions cast_opts;
        cast_opts.allow_int_overflow = false;
        std::vector<std::shared_ptr<arrow::Array>> new_chunks;
        new_chunks.reserve(col->num_chunks());
        bool cast_failed = false;
        for (int c = 0; c < col->num_chunks(); ++c) {
            auto chunk = col->chunk(c);
            auto cast_result = arrow::compute::Cast(*chunk, target_type, cast_opts);
            if (!cast_result.ok()) {
                spdlog::warn("CompactIntegerColumns: cast failed for column {} ({} -> {}): {}",
                             table->field(i)->name(),
                             col->type()->ToString(),
                             target_type->ToString(),
                             cast_result.status().ToString());
                cast_failed = true;
                break;
            }
            new_chunks.push_back(cast_result.ValueOrDie());
        }

        if (cast_failed) {
            new_columns.push_back(col);
            for (int c = 0; c < col->num_chunks(); ++c) {
                auto chunk = col->chunk(c);
                for (const auto& buf : chunk->data()->buffers) {
                    if (buf) bytes_after += buf->size();
                }
            }
        } else {
            auto new_col = std::make_shared<arrow::ChunkedArray>(new_chunks, target_type);
            new_columns.push_back(new_col);
            cols_compacted++;
            for (const auto& chunk : new_chunks) {
                for (const auto& buf : chunk->data()->buffers) {
                    if (buf) bytes_after += buf->size();
                }
            }
        }
    }

    if (cols_compacted == 0) {
        return table;
    }

    // Build new schema with updated field types
    arrow::FieldVector new_fields;
    new_fields.reserve(table->num_columns());
    for (int i = 0; i < table->num_columns(); ++i) {
        auto old_field = table->field(i);
        new_fields.push_back(
            std::make_shared<arrow::Field>(old_field->name(),
                                            new_columns[i]->type(),
                                            old_field->nullable()));
    }
    auto new_schema = std::make_shared<arrow::Schema>(new_fields, table->schema()->metadata());

    auto new_table = arrow::Table::Make(new_schema, new_columns, table->num_rows());

    double savings_mb = (static_cast<double>(bytes_before) - bytes_after) / (1024.0 * 1024.0);
    double pct = bytes_before > 0 ? (100.0 * (bytes_before - bytes_after)) / bytes_before : 0.0;
    spdlog::info("CompactIntegerColumns: compacted {} columns, {:.1f} MB -> {:.1f} MB "
                 "(saved {:.1f} MB, {:.1f}%)",
                 cols_compacted,
                 bytes_before / (1024.0 * 1024.0),
                 bytes_after / (1024.0 * 1024.0),
                 savings_mb,
                 pct);

    return new_table;
}

std::shared_ptr<ArrowDataset> DataRegistry::LoadCSVToArrow(
    const std::string& path, const std::string& name,
    bool has_header, char delimiter, int skip_rows, int64_t max_rows,
    const std::vector<std::string>& missing_value_tokens) {

    std::string unique_name = GenerateUniqueName(name.empty() ? fs::path(path).stem().string() : name);

    try {
        // Configure CSV reading options
        auto read_options = arrow::csv::ReadOptions::Defaults();
        read_options.skip_rows = skip_rows;

        // Auto-size block_size so the entire file loads as a single Arrow
        // chunk whenever possible. Single-chunk columns hit the fast
        // raw_values() direct-pointer path in ArrowDatasetBatcher; multi-
        // chunk columns fall back to a slower per-row scan. Arrow's default
        // is 1 MB, which splits anything non-trivial into many chunks.
        //
        // We pick block_size = max(file_size + 1 MB headroom, 64 MB), capped
        // at INT32_MAX since Arrow's block_size is a signed 32-bit int.
        // Files >2 GB simply fall back to multi-chunk (still correct).
        int64_t file_size_bytes = 0;
        try {
            file_size_bytes = static_cast<int64_t>(fs::file_size(path));
        } catch (...) {
            // If we can't stat the file, fall through to a sensible default
            file_size_bytes = 0;
        }
        constexpr int64_t kMinBlock = 64 * 1024 * 1024;         // 64 MB
        constexpr int64_t kMaxBlock = std::numeric_limits<int32_t>::max();
        int64_t target = std::max<int64_t>(file_size_bytes + (1 << 20), kMinBlock);
        target = std::min<int64_t>(target, kMaxBlock);
        read_options.block_size = static_cast<int32_t>(target);
        spdlog::info("LoadCSVToArrow: file_size={} MB, block_size={} MB (auto)",
                     file_size_bytes / (1024 * 1024),
                     read_options.block_size / (1024 * 1024));

        auto parse_options = arrow::csv::ParseOptions::Defaults();
        parse_options.delimiter = delimiter;

        auto convert_options = MakeTabularCsvConvertOptions(missing_value_tokens);

        // Handle header
        if (!has_header) {
            read_options.autogenerate_column_names = true;
        }

        auto dataset = ArrowDataset::FromCSV(path, unique_name, read_options, parse_options, convert_options);
        if (!dataset) {
            spdlog::error("LoadCSVToArrow: Failed to load {}", path);
            return nullptr;
        }

        // Apply row cap from Limit Rows tab. Slice is zero-copy (Arrow table
        // buffers are shared) so this is cheap — but note that the full file
        // was still parsed into RAM before the slice. For a true lazy row cap
        // we'd need arrow::csv::StreamingReader; deferred.
        if (max_rows > 0) {
            auto table = dataset->GetArrowTable();
            if (table && table->num_rows() > max_rows) {
                auto sliced = table->Slice(0, max_rows);
                spdlog::info("LoadCSVToArrow: max_rows={} applied, sliced from {} to {} rows",
                             max_rows, table->num_rows(), sliced->num_rows());
                dataset = std::make_shared<ArrowDataset>(sliced, unique_name);
            }
        }

        // Compact integer columns to the smallest fitting type. For a CSV like
        // mnist_784 where Arrow promotes uint8 pixels to int64, this reduces
        // memory by 8x and makes per-batch reads 8x less data-bound. Runs
        // after slicing so the min/max scan only looks at the kept rows.
        auto compacted = CompactIntegerColumns(dataset->GetArrowTable());
        if (compacted && compacted.get() != dataset->GetArrowTable().get()) {
            dataset = std::make_shared<ArrowDataset>(compacted, unique_name);
        }

        std::lock_guard<std::mutex> lock(mutex_);
        arrow_datasets_[unique_name] = dataset;
        RememberTabularSourcePathUnlocked(unique_name, path);

        spdlog::info("LoadCSVToArrow: Loaded '{}' as '{}' ({} rows, {} cols)",
                    path, unique_name, dataset->GetNumRows(), dataset->GetNumColumns());
        return dataset;

    } catch (const std::exception& e) {
        spdlog::error("LoadCSVToArrow exception: {}", e.what());
        return nullptr;
    }
}

// -----------------------------------------------------------------------------
// GetAvailableMemoryBytes - cross-platform available RAM detection
//
// Returns the amount of physical memory the OS says is currently available for
// allocation (not the total RAM, which doesn't account for what other processes
// are using). Used by LoadTabularCSV to decide whether a dataset will fit in
// memory or needs the disk-backed Parquet path.
//
// Conservative fallback: if detection fails, returns 2 GB. That's enough to
// comfortably load MNIST-scale datasets in memory, and anything larger will
// take the disk-backed path as a safe default.
// -----------------------------------------------------------------------------
static size_t GetAvailableMemoryBytes() {
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memInfo)) {
        return static_cast<size_t>(memInfo.ullAvailPhys);
    }
    return 2ULL * 1024 * 1024 * 1024;  // 2 GB fallback
#elif defined(__linux__)
    // Linux: sysinfo() gives us freeram in units of si.mem_unit bytes.
    // freeram is approximate (doesn't count page cache as "free"), but
    // that's fine for our 75%-threshold decision — we'd rather err toward
    // disk-backed mode than OOM during load.
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        return static_cast<size_t>(si.freeram) * static_cast<size_t>(si.mem_unit);
    }
    return 2ULL * 1024 * 1024 * 1024;
#elif defined(__APPLE__)
    // macOS: host_statistics64() gives us free + inactive pages, which is
    // the closest analog to Windows' AvailPhys. pagesize is queried via
    // sysctl.
    vm_size_t page_size = 0;
    host_page_size(mach_host_self(), &page_size);
    vm_statistics64_data_t vmstat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&vmstat), &count) == KERN_SUCCESS) {
        const uint64_t free_bytes = (static_cast<uint64_t>(vmstat.free_count) +
                                     static_cast<uint64_t>(vmstat.inactive_count)) *
                                    static_cast<uint64_t>(page_size);
        return static_cast<size_t>(free_bytes);
    }
    return 2ULL * 1024 * 1024 * 1024;
#else
    // Unknown platform — conservative fallback. Small datasets still take
    // the in-memory path; anything larger goes disk-backed.
    return 2ULL * 1024 * 1024 * 1024;
#endif
}

// -----------------------------------------------------------------------------
// LoadTabularCSV - automatic in-memory vs disk-backed dispatcher
//
// Decision rule:
//   - If force_disk_backed is true, always take the Parquet path.
//   - If file_size < 0.75 * available RAM, take the in-memory Arrow path
//     (LoadCSVToArrow). This is the fast common case.
//   - Otherwise, convert the CSV to a Snappy-compressed Parquet cache in
//     the system temp dir (with mtime-based cache freshness check), open
//     it via memory-mapped reads, and register as ParquetBackedDataset.
//
// On success the dataset is queryable via the matching accessor. The caller
// uses the returned TabularLoadBackend tag to know which map to query.
// -----------------------------------------------------------------------------
DataRegistry::TabularLoadBackend DataRegistry::LoadTabularCSV(
    const std::string& path, const std::string& name,
    bool has_header, char delimiter, int skip_rows,
    int64_t max_rows, bool force_disk_backed,
    const std::vector<std::string>& missing_value_tokens) {

    // File size check (required for dispatch decision and for logging)
    int64_t file_size_bytes = 0;
    try {
        file_size_bytes = static_cast<int64_t>(fs::file_size(path));
    } catch (const std::exception& e) {
        spdlog::error("LoadTabularCSV: cannot stat '{}': {}", path, e.what());
        return TabularLoadBackend::Failed;
    }

    const size_t available_ram = GetAvailableMemoryBytes();
    const size_t memory_threshold = static_cast<size_t>(available_ram * 0.75);
    const bool file_fits_in_ram = static_cast<size_t>(file_size_bytes) < memory_threshold;
    const bool use_disk_backed = force_disk_backed || !file_fits_in_ram;

    spdlog::info("LoadTabularCSV: '{}' file={:.1f} MB, available RAM={:.1f} MB, "
                 "threshold={:.1f} MB, force_disk={}, decision={}",
                 path,
                 file_size_bytes / (1024.0 * 1024.0),
                 available_ram / (1024.0 * 1024.0),
                 memory_threshold / (1024.0 * 1024.0),
                 force_disk_backed ? "true" : "false",
                 use_disk_backed ? "disk-backed" : "in-memory");

    // Clear any stale entries under the same name in the OTHER map before
    // (re-)registering. Without this, repeatedly re-Applying a dataset
    // (especially when toggling the Force disk-backed flag) leaves orphan
    // entries in both maps and MainWindow::StartTrainingFromGraph routes
    // to whichever one it checks first, which is usually the stale one.
    UnregisterTabularDataset(name);

    if (!use_disk_backed) {
        // Fast path: fits in RAM, use the existing in-memory loader.
        auto dataset = LoadCSVToArrow(path, name, has_header, delimiter,
                                      skip_rows, max_rows, missing_value_tokens);
        return dataset ? TabularLoadBackend::InMemory : TabularLoadBackend::Failed;
    }

    // Slow path: file is too big (or forced). Convert to a Parquet cache
    // next to the system temp dir, open it via memory-mapped reads.
    const std::string cache_signature =
        std::string(has_header ? "header=1" : "header=0") +
        "|delimiter=" + std::to_string(static_cast<unsigned char>(delimiter)) +
        "|skip=" + std::to_string(skip_rows) +
        "|nulls=" + MissingValueTokensSignature(missing_value_tokens);
    const std::string cache_path =
        ParquetBackedDataset::GetCacheFilePath(path, cache_signature);

    if (ParquetBackedDataset::IsCacheFresh(path, cache_path)) {
        spdlog::info("LoadTabularCSV: reusing existing Parquet cache at '{}'", cache_path);
    } else {
        spdlog::info("LoadTabularCSV: converting CSV to Parquet cache at '{}'", cache_path);
        if (!ParquetBackedDataset::ConvertCsvToParquet(path, cache_path,
                                                       has_header, delimiter, skip_rows,
                                                       missing_value_tokens)) {
            spdlog::error("LoadTabularCSV: CSV-to-Parquet conversion failed for '{}'", path);
            return TabularLoadBackend::Failed;
        }
    }

    std::string unique_name = GenerateUniqueName(name.empty() ? fs::path(path).stem().string() : name);
    auto pq_dataset = ParquetBackedDataset::Open(cache_path, unique_name);
    if (!pq_dataset) {
        spdlog::error("LoadTabularCSV: failed to open Parquet cache '{}'", cache_path);
        return TabularLoadBackend::Failed;
    }

    RegisterParquetBacked(unique_name, pq_dataset);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        RememberTabularSourcePathUnlocked(unique_name, path);
    }

    // Note: max_rows is intentionally ignored on the disk-backed path for
    // now. Applying it would require reading a row-limit subset into memory,
    // which defeats the purpose. When we want Limit Rows to work for large
    // datasets, it'll be implemented via the batcher's row-group scan.
    if (max_rows > 0) {
        spdlog::warn("LoadTabularCSV: max_rows={} requested but ignored on disk-backed path "
                     "(not yet supported; will be applied by the Parquet batcher in a future version)",
                     max_rows);
    }

    return TabularLoadBackend::DiskBacked;
}

std::shared_ptr<ArrowDataset> DataRegistry::LoadParquetToArrow(
    const std::string& path, const std::string& name) {

    std::string unique_name = GenerateUniqueName(name.empty() ? fs::path(path).stem().string() : name);

    try {
        auto dataset = ArrowDataset::FromParquet(path, unique_name);
        if (!dataset) {
            spdlog::error("LoadParquetToArrow: Failed to load {}", path);
            return nullptr;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        arrow_datasets_[unique_name] = dataset;
        RememberTabularSourcePathUnlocked(unique_name, path);

        spdlog::info("LoadParquetToArrow: Loaded '{}' as '{}' ({} rows, {} cols)",
                    path, unique_name, dataset->GetNumRows(), dataset->GetNumColumns());
        return dataset;

    } catch (const std::exception& e) {
        spdlog::error("LoadParquetToArrow exception: {}", e.what());
        return nullptr;
    }
}

std::shared_ptr<ArrowDataset> DataRegistry::LoadJSONToArrow(
    const std::string& path, const std::string& name, bool json_lines) {

    std::string unique_name = GenerateUniqueName(name.empty() ? fs::path(path).stem().string() : name);

    try {
        // JSON loading via DuckDB (Arrow doesn't have native JSON reader)
        // For now, fall back to FromFile which auto-detects
        auto dataset = ArrowDataset::FromFile(path, unique_name);
        if (!dataset) {
            spdlog::warn("LoadJSONToArrow: Direct load failed, JSON may need DuckDB conversion");
            // TODO: Use DuckDB to read JSON and convert to Arrow
            return nullptr;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        arrow_datasets_[unique_name] = dataset;
        RememberTabularSourcePathUnlocked(unique_name, path);

        spdlog::info("LoadJSONToArrow: Loaded '{}' as '{}' ({} rows, {} cols)",
                    path, unique_name, dataset->GetNumRows(), dataset->GetNumColumns());
        return dataset;

    } catch (const std::exception& e) {
        spdlog::error("LoadJSONToArrow exception: {}", e.what());
        return nullptr;
    }
}

std::shared_ptr<ArrowDataset> DataRegistry::LoadExcelToArrow(
    const std::string& path, const std::string& name, int sheet_idx) {

    std::string unique_name = GenerateUniqueName(name.empty() ? fs::path(path).stem().string() : name);

    try {
        // Excel requires external library (xlsxio or similar)
        // For now, log and return nullptr - pipeline should handle gracefully
        spdlog::warn("LoadExcelToArrow: Excel support requires additional library (sheet {})", sheet_idx);
        // TODO: Integrate xlsxio or use DuckDB's Excel extension
        return nullptr;

    } catch (const std::exception& e) {
        spdlog::error("LoadExcelToArrow exception: {}", e.what());
        return nullptr;
    }
}

std::shared_ptr<ArrowDataset> DataRegistry::LoadImageFolderToArrow(
    const std::string& path, const std::string& name) {

    std::string unique_name = GenerateUniqueName(name.empty() ? fs::path(path).stem().string() : name);

    try {
        // Build a table with file paths and labels from directory structure
        // Format: image_path | label_name | label_id
        std::vector<std::string> image_paths;
        std::vector<std::string> label_names;
        std::vector<int32_t> label_ids;
        std::map<std::string, int32_t> label_map;
        int32_t next_label = 0;

        for (const auto& entry : fs::recursive_directory_iterator(path)) {
            if (!entry.is_regular_file()) continue;

            auto ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".gif") {
                image_paths.push_back(entry.path().string());

                // Use parent directory name as label
                std::string label = entry.path().parent_path().filename().string();
                label_names.push_back(label);

                if (label_map.find(label) == label_map.end()) {
                    label_map[label] = next_label++;
                }
                label_ids.push_back(label_map[label]);
            }
        }

        if (image_paths.empty()) {
            spdlog::warn("LoadImageFolderToArrow: No images found in {}", path);
            return nullptr;
        }

        // Build Arrow table
        arrow::StringBuilder path_builder;
        arrow::StringBuilder label_builder;
        arrow::Int32Builder id_builder;

        for (size_t i = 0; i < image_paths.size(); i++) {
            (void)path_builder.Append(image_paths[i]);
            (void)label_builder.Append(label_names[i]);
            (void)id_builder.Append(label_ids[i]);
        }

        std::shared_ptr<arrow::Array> path_array, label_array, id_array;
        (void)path_builder.Finish(&path_array);
        (void)label_builder.Finish(&label_array);
        (void)id_builder.Finish(&id_array);

        auto schema = arrow::schema({
            arrow::field("image_path", arrow::utf8()),
            arrow::field("label_name", arrow::utf8()),
            arrow::field("label_id", arrow::int32())
        });

        auto table = arrow::Table::Make(schema, {path_array, label_array, id_array});
        auto dataset = std::make_shared<ArrowDataset>(table, unique_name);

        std::lock_guard<std::mutex> lock(mutex_);
        arrow_datasets_[unique_name] = dataset;

        spdlog::info("LoadImageFolderToArrow: Loaded {} images with {} classes from '{}'",
                    image_paths.size(), label_map.size(), path);
        return dataset;

    } catch (const std::exception& e) {
        spdlog::error("LoadImageFolderToArrow exception: {}", e.what());
        return nullptr;
    }
}

std::shared_ptr<ArrowDataset> DataRegistry::LoadMLDatasetToArrow(
    const std::string& ml_type, const std::string& name) {

    std::string unique_name = GenerateUniqueName(name.empty() ? ml_type : name);

    try {
        // ML datasets (MNIST, CIFAR, etc.) need conversion from binary format to Arrow
        // For now, placeholder that creates metadata table
        spdlog::warn("LoadMLDatasetToArrow: '{}' - ML datasets require loading via DatasetHandle first", ml_type);
        // TODO: Bridge existing LoadMNIST/LoadCIFAR10 to Arrow format
        return nullptr;

    } catch (const std::exception& e) {
        spdlog::error("LoadMLDatasetToArrow exception: {}", e.what());
        return nullptr;
    }
}

// =============================================================================
// Arrow Export Methods (for DataOutput node pipeline execution)
// =============================================================================

bool DataRegistry::ExportArrowToCSV(const std::string& dataset_name, const std::string& output_path) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = arrow_datasets_.find(dataset_name);
    if (it == arrow_datasets_.end()) {
        spdlog::error("ExportArrowToCSV: Dataset '{}' not found", dataset_name);
        return false;
    }

    bool success = it->second->ExportCSV(output_path);
    if (success) {
        spdlog::info("ExportArrowToCSV: Exported '{}' to '{}'", dataset_name, output_path);
    }
    return success;
}

bool DataRegistry::ExportArrowToParquet(const std::string& dataset_name, const std::string& output_path) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = arrow_datasets_.find(dataset_name);
    if (it == arrow_datasets_.end()) {
        spdlog::error("ExportArrowToParquet: Dataset '{}' not found", dataset_name);
        return false;
    }

    bool success = it->second->ExportParquet(output_path);
    if (success) {
        spdlog::info("ExportArrowToParquet: Exported '{}' to '{}'", dataset_name, output_path);
    }
    return success;
}

bool DataRegistry::ExportArrowToJSON(const std::string& dataset_name, const std::string& output_path) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = arrow_datasets_.find(dataset_name);
    if (it == arrow_datasets_.end()) {
        spdlog::error("ExportArrowToJSON: Dataset '{}' not found", dataset_name);
        return false;
    }

    // JSON export - Arrow doesn't have native JSON writer, use custom implementation
    // For now, write as CSV with .json extension (will need proper JSON serialization)
    spdlog::warn("ExportArrowToJSON: Native JSON export not implemented, consider CSV");
    // TODO: Implement proper JSON export using nlohmann/json
    return false;
}

} // namespace cyxwiz
