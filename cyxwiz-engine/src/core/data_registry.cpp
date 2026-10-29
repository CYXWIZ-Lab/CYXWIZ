#include "data_registry.h"
#include "json_lines_reader.h"
#include "dataset_base.h"
#include "arrow_dataset.h"
#include "parquet_backed_dataset.h"
#include "sparse_feature_dataset.h"
#include "data_budget.h"
#include "sha256_digest.h"
#include "hashing_output_stream.h"
#include "data_input_formats.h"
#include "split_partitioning.h"
#include <arrow/csv/writer.h>
#include <arrow/record_batch.h>
#include <arrow/io/buffered.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>
#include <arrow/compute/api.h>  // for arrow::compute::Cast (integer column compaction)
#include <arrow/array/concatenate.h>  // SortArrowTableByColumn
#include <limits>              // for std::numeric_limits (Arrow block_size cap)

#include "annotation_manager.h"
#include "image_utils.h"
#include "../preprocessing/preprocessing_config.h"
#include "../transforms/transform.h"

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
        default:
            // Plugin formats load through Data Input (PluginDataLoaderRegistry).
            spdlog::error("Unknown dataset type for path: {}", path);
            return DatasetHandle();
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
            InvalidateTabularPublicationUnlocked(unique_name);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (auto callback = GetDatasetLoadedCallback()) {
            callback(unique_name, handle.GetInfo());
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
            InvalidateTabularPublicationUnlocked(unique_name);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (auto callback = GetDatasetLoadedCallback()) {
            callback(unique_name, handle.GetInfo());
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
            InvalidateTabularPublicationUnlocked(unique_name);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (auto callback = GetDatasetLoadedCallback()) {
            callback(unique_name, handle.GetInfo());
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
            InvalidateTabularPublicationUnlocked(unique_name);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (auto callback = GetDatasetLoadedCallback()) {
            callback(unique_name, handle.GetInfo());
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
            InvalidateTabularPublicationUnlocked(unique_name);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (auto callback = GetDatasetLoadedCallback()) {
            callback(unique_name, handle.GetInfo());
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
            InvalidateTabularPublicationUnlocked(unique_name);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (auto callback = GetDatasetLoadedCallback()) {
            callback(unique_name, handle.GetInfo());
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
            InvalidateTabularPublicationUnlocked(unique_name);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (auto callback = GetDatasetLoadedCallback()) {
            callback(unique_name, handle.GetInfo());
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
            InvalidateTabularPublicationUnlocked(unique_name);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (auto callback = GetDatasetLoadedCallback()) {
            callback(unique_name, handle.GetInfo());
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
            InvalidateTabularPublicationUnlocked(unique_name);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (auto callback = GetDatasetLoadedCallback()) {
            callback(unique_name, handle.GetInfo());
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
            InvalidateTabularPublicationUnlocked(unique_name);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (auto callback = GetDatasetLoadedCallback()) {
            callback(unique_name, handle.GetInfo());
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
            InvalidateTabularPublicationUnlocked(unique_name);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (auto callback = GetDatasetLoadedCallback()) {
            callback(unique_name, handle.GetInfo());
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
            InvalidateTabularPublicationUnlocked(unique_name);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (auto callback = GetDatasetLoadedCallback()) {
            callback(unique_name, handle.GetInfo());
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
            InvalidateTabularPublicationUnlocked(unique_name);
            datasets_[unique_name] = dataset;
        }

        auto handle = DatasetHandle(dataset, unique_name);

        if (auto callback = GetDatasetLoadedCallback()) {
            callback(unique_name, handle.GetInfo());
        }

        spdlog::info("Registered HDF5 dataset '{}': {} samples", unique_name, dataset->Size());
        return handle;

    } catch (const std::exception& e) {
        spdlog::error("Failed to load HDF5 dataset: {}", e.what());
        return DatasetHandle();
    }
#else
    (void)path; (void)name; (void)config;
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
            std::vector<hsize_t> hdf5_chunk_dims(chunk_dims.begin(), chunk_dims.end());
            props.add(HighFive::Chunking(hdf5_chunk_dims));
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
    (void)handle; (void)filepath; (void)config;
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

LoadedDataReport DataRegistry::LoadedData() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return LoadedDataUnlocked();
}

LoadedDataReport DataRegistry::LoadedDataUnlocked() const {
    LoadedDataReportBuilder builder;
    const auto source = [this](const std::string& name) {
        const auto it = tabular_source_paths_by_name_.find(name);
        return it == tabular_source_paths_by_name_.end() ? std::string() : it->second;
    };
    for (const auto& [name, dataset] : arrow_datasets_) {
        if (!dataset) continue;
        LoadedDataEntry e;
        e.name = name;
        e.kind = LoadedDataKind::TableInMemory;
        e.path = source(name);
        builder.AddTable(std::move(e), dataset->GetArrowTable());
    }
    for (const auto& [name, dataset] : parquet_backed_datasets_) {
        if (!dataset) continue;
        LoadedDataEntry e;
        e.name = name;
        e.kind = LoadedDataKind::TableOnDisk;
        e.disk_bytes = dataset->GetFileSizeBytes();
        e.path = dataset->GetFilePath();
        e.rows = dataset->GetNumRows();
        builder.Add(std::move(e));
    }
    for (const auto& [name, dataset] : sparse_feature_datasets_) {
        if (!dataset) continue;
        LoadedDataEntry e;
        e.name = name;
        e.kind = LoadedDataKind::SparseFeatures;
        e.memory_bytes = dataset->GetEstimatedHostMemoryBytes();
        builder.Add(std::move(e));
    }
    // Image, audio and text datasets read their files as training goes:
    // they hold their file lists, not the data.
    const auto files = [&builder](const std::string& name, LoadedDataKind kind, const std::string& path, size_t n) {
        LoadedDataEntry e;
        e.name = name;
        e.kind = kind;
        e.path = path;
        e.rows = static_cast<int64_t>(n);
        builder.Add(std::move(e));
    };
    for (const auto& [name, entry] : image_dataset_entries_) files(name, LoadedDataKind::Images, entry.folder_path, entry.num_images);
    for (const auto& [name, entry] : audio_dataset_entries_) files(name, LoadedDataKind::Audio, entry.folder_path, entry.num_samples);
    for (const auto& [name, entry] : text_dataset_entries_) files(name, LoadedDataKind::Text, entry.source_path, entry.num_samples);
    for (const auto& [name, dataset] : datasets_) {
        if (!dataset) continue;
        LoadedDataEntry e;
        e.name = name;
        e.kind = LoadedDataKind::Other;
        e.memory_bytes = dataset->GetInfo().memory_usage;
        builder.Add(std::move(e));
    }
    return builder.Finish();
}

std::function<bool(const std::filesystem::path&)> DataRegistry::FileInUse() const {
    std::set<std::string> held;
    for (const auto& e : LoadedData().entries)
        if (!e.path.empty()) held.insert(std::filesystem::path(e.path).lexically_normal().string());
    return [held = std::move(held)](const std::filesystem::path& p) {
        return held.count(p.lexically_normal().string()) > 0;
    };
}

MemoryStats DataRegistry::GetMemoryStats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    MemoryStats stats;
    stats.memory_limit = memory_limit_;
    stats.cache_hits = total_cache_hits_;
    stats.cache_misses = total_cache_misses_;
    stats.cache_evictions = total_cache_evictions_;

    const LoadedDataReport loaded = LoadedDataUnlocked();
    stats.datasets_count = loaded.entries.size();
    stats.total_allocated = loaded.memory_bytes;
    for (const auto& [_, dataset] : datasets_) {
        if (dataset) stats.total_cached += dataset->GetInfo().cache_usage;
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

        InvalidateTabularPublicationUnlocked(oldest_name);
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
    MemoryPressureCallback callback;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        callback = on_memory_pressure_;
    }
    if (current > memory_limit_ && callback) {
        callback(current, memory_limit_);
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
    const std::vector<std::string>& missing_value_tokens,
    const std::vector<std::string>& selected_columns,
    char decimal_point,
    bool use_threads,
    const CsvProgressCallback& progress_callback,
    char quote_char) {

    std::string unique_name = GenerateUniqueName(name.empty() ? fs::path(path).stem().string() : name);

    try {
        // Configure CSV reading options
        auto read_options = arrow::csv::ReadOptions::Defaults();
        read_options.skip_rows = skip_rows;
        read_options.use_threads = use_threads;

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
            file_size_bytes = 0;
        }
        if (max_rows <= 0) {
            constexpr int64_t kResponsiveMaxBlock = 64 * 1024 * 1024;
            constexpr int64_t kMaxBlock = std::numeric_limits<int32_t>::max();
            int64_t target = 0;
            if (use_threads) {
                target = std::max<int64_t>(
                    file_size_bytes + (1 << 20), kResponsiveMaxBlock);
                target = std::min<int64_t>(target, kMaxBlock);
            } else {
                // Interactive loads already run on a background worker. Use
                // bounded streaming blocks rather than one source-sized block
                // so parsing cannot monopolize every Arrow worker or demand a
                // second source-sized allocation. Exact sizing also avoids an
                // Arrow 21 non-threaded empty-block failure on small files.
                target = std::clamp<int64_t>(
                    file_size_bytes, 1, kResponsiveMaxBlock);
            }
            read_options.block_size = static_cast<int32_t>(target);
            spdlog::info(
                "LoadCSVToArrow: file_size={} MB, block_size={} MB, threads={}",
                file_size_bytes / (1024 * 1024),
                read_options.block_size / (1024 * 1024),
                use_threads);
        } else {
            spdlog::info(
                "LoadCSVToArrow: streaming source with max_rows={} and "
                "block_size={} KB",
                max_rows,
                read_options.block_size / 1024);
        }

        auto parse_options = arrow::csv::ParseOptions::Defaults();
        // Quoted values may span lines (free text, reviews): read them as one value.
        parse_options.newlines_in_values = true;
        parse_options.delimiter = delimiter;
        ApplyCsvQuoteChar(parse_options, quote_char);

        auto convert_options = MakeTabularCsvConvertOptions(
            missing_value_tokens, decimal_point);
        convert_options.include_columns = selected_columns;

        // Handle header
        if (!has_header) {
            read_options.autogenerate_column_names = true;
        }

        const bool preflight_widen_integers =
            file_size_bytes > read_options.block_size &&
            CsvSourceSamplesContainDecimalValues(
                path, decimal_point, read_options.block_size);
        if (preflight_widen_integers) {
            spdlog::info(
                "LoadCSVToArrow: schema preflight found decimal values beyond "
                "the inference block; widening inferred integer columns before parsing");
        }

        bool cancelled = false;
        auto load_table_attempt = [&](bool promote_inferred_integers,
                                      bool is_retry,
                                      size_t& promoted_integer_columns,
                                      std::string& attempt_error)
            -> std::shared_ptr<arrow::Table> {
            std::shared_ptr<arrow::io::ReadableFile> input_source;
            auto reader_result = MakeStableCsvStreamingReader(
                path, read_options, parse_options, convert_options,
                &promoted_integer_columns, promote_inferred_integers,
                &input_source);
            if (!reader_result.ok()) {
                attempt_error = "reader creation failed: " +
                    reader_result.status().ToString();
                return nullptr;
            }

            auto reader = reader_result.ValueOrDie();
            std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
            int64_t rows_read = 0;
            while (max_rows <= 0 || rows_read < max_rows) {
                std::shared_ptr<arrow::RecordBatch> batch;
                auto status = reader->ReadNext(&batch);
                if (!status.ok()) {
                    attempt_error = "stream read failed: " + status.ToString();
                    return nullptr;
                }
                if (!batch) break;

                if (max_rows > 0) {
                    const int64_t remaining = max_rows - rows_read;
                    if (batch->num_rows() > remaining) {
                        batch = batch->Slice(0, remaining);
                    }
                }
                rows_read += batch->num_rows();
                batches.push_back(std::move(batch));

                if (progress_callback && file_size_bytes > 0 && input_source) {
                    auto position = input_source->Tell();
                    if (position.ok()) {
                        const float source_fraction = std::clamp(
                            static_cast<float>(position.ValueOrDie()) /
                                static_cast<float>(file_size_bytes),
                            0.0f, 1.0f);
                        const float base = is_retry ? 0.8f : 0.0f;
                        const float span = is_retry ? 0.19f : 0.99f;
                        const std::string message = is_retry
                            ? "Reading CSV with widened numeric schema"
                            : (promote_inferred_integers
                                ? "Reading CSV with preflight schema"
                                : "Reading CSV");
                        if (!progress_callback(base + span * source_fraction,
                                               message)) {
                            cancelled = true;
                            attempt_error = "cancelled";
                            return nullptr;
                        }
                    }
                }
            }

            auto result = arrow::Table::FromRecordBatches(
                reader->schema(), batches);
            if (!result.ok()) {
                attempt_error = "table assembly failed: " +
                    result.status().ToString();
                return nullptr;
            }
            return result.ValueOrDie();
        };

        size_t promoted_integer_columns = 0;
        std::string initial_error;
        auto table = load_table_attempt(
            preflight_widen_integers, false,
            promoted_integer_columns, initial_error);
        if (!table) {
            if (cancelled) return nullptr;
            if (preflight_widen_integers) {
                spdlog::error(
                    "LoadCSVToArrow: preflight schema read failed for '{}': {}",
                    path, initial_error);
                return nullptr;
            }
            if (progress_callback &&
                !progress_callback(0.8f, "Retrying with widened numeric schema")) {
                return nullptr;
            }
            std::string retry_error;
            table = load_table_attempt(
                true, true, promoted_integer_columns, retry_error);
            if (!table) {
                if (cancelled) return nullptr;
                spdlog::error(
                    "LoadCSVToArrow: read failed for '{}': {}; numeric fallback also failed: {}",
                    path, initial_error, retry_error);
                return nullptr;
            }
            spdlog::info(
                "LoadCSVToArrow: initial read failed for '{}' ({}); retry succeeded after promoting {} inferred integer columns",
                path, initial_error, promoted_integer_columns);
        }
        if (progress_callback &&
            !progress_callback(0.99f, "Optimizing loaded columns")) {
            return nullptr;
        }
        std::shared_ptr<ArrowDataset> dataset = std::make_shared<ArrowDataset>(
            table, unique_name);
        if (!dataset) {
            spdlog::error("LoadCSVToArrow: Failed to load {}", path);
            return nullptr;
        }

        // Defensively clamp the final table. The bounded path already stops
        // its streaming reader at max_rows, so this should normally be a no-op.
        if (max_rows > 0) {
            auto bounded_table = dataset->GetArrowTable();
            if (bounded_table && bounded_table->num_rows() > max_rows) {
                auto sliced = bounded_table->Slice(0, max_rows);
                spdlog::info("LoadCSVToArrow: max_rows={} applied, sliced from {} to {} rows",
                             max_rows, bounded_table->num_rows(), sliced->num_rows());
                dataset = std::make_shared<ArrowDataset>(sliced, unique_name);
            }
        }

        // Compact integer columns to the smallest fitting type. For a CSV like
        // mnist_784 where Arrow promotes uint8 pixels to int64, this reduces
        // memory by 8x and makes per-batch reads 8x less data-bound. Runs
        // after bounded ingestion so the min/max scan only sees kept rows.
        auto compacted = CompactIntegerColumns(dataset->GetArrowTable());
        if (compacted && compacted.get() != dataset->GetArrowTable().get()) {
            dataset = std::make_shared<ArrowDataset>(compacted, unique_name);
        }

        std::lock_guard<std::mutex> lock(mutex_);
        InvalidateTabularPublicationUnlocked(unique_name);
        arrow_datasets_[unique_name] = dataset;
        parquet_backed_datasets_.erase(unique_name);
        sparse_feature_datasets_.erase(unique_name);
        RememberTabularSourcePathUnlocked(unique_name, path);

        spdlog::info("LoadCSVToArrow: Loaded '{}' as '{}' ({} rows, {} cols)",
                    path, unique_name, dataset->GetNumRows(), dataset->GetNumColumns());
        if (progress_callback) {
            (void)progress_callback(1.0f, "CSV loaded");
        }
        return dataset;

    } catch (const std::exception& e) {
        spdlog::error("LoadCSVToArrow exception: {}", e.what());
        return nullptr;
    }
}


static bool PersistArrowCacheAtomically(
    const std::shared_ptr<ArrowDataset>& dataset,
    const std::string& cache_path) {
    if (!dataset || cache_path.empty()) return false;

    const std::string temporary_path = ParquetBackedDataset::UniqueTempPath(cache_path);

    try {
        const fs::path parent = fs::path(cache_path).parent_path();
        if (!parent.empty()) fs::create_directories(parent);
        if (!dataset->ExportParquet(temporary_path)) {
            std::error_code cleanup_error;
            fs::remove(temporary_path, cleanup_error);
            return false;
        }

        if (fs::exists(cache_path)) {
            fs::remove(cache_path);
        }
        fs::rename(temporary_path, cache_path);
        return true;
    } catch (const std::exception& e) {
        spdlog::warn("PersistArrowCacheAtomically: cache write failed for '{}': {}",
                     cache_path, e.what());
        std::error_code cleanup_error;
        fs::remove(temporary_path, cleanup_error);
        return false;
    }
}

// -----------------------------------------------------------------------------
// LoadTabularCSV - automatic in-memory vs disk-backed dispatcher
//
// Decision rule:
//   - If force_disk_backed is true, always take the Parquet path.
//   - If file_size < 0.75 * available RAM, take the in-memory Arrow path
//     (LoadCSVToArrow). This is the fast common case.
//   - Otherwise, convert the CSV to a Snappy-compressed persistent Parquet
//     cache (with parser-contract identity and mtime freshness), open it via
//     memory-mapped reads, and register as ParquetBackedDataset.
// In-memory loads use the same cache contract for fast restoration after an
// application restart. The UI supplies the active project's cache directory;
// headless callers fall back to the system temp directory.
//
// On success the dataset is queryable via the matching accessor. The caller
// uses the returned TabularLoadBackend tag to know which map to query.
// -----------------------------------------------------------------------------
DataRegistry::TabularLoadBackend DataRegistry::LoadTabularCSV(
    const std::string& path, const std::string& name,
    bool has_header, char delimiter, int skip_rows,
    int64_t max_rows, bool force_disk_backed,
    const std::vector<std::string>& missing_value_tokens,
    const std::vector<std::string>& selected_columns,
    char decimal_point,
    bool use_threads,
    const CsvProgressCallback& progress_callback,
    std::string* load_status,
    const std::string& cache_directory,
    char quote_char) {

    if (load_status) load_status->clear();

    // File size check (required for dispatch decision and for logging)
    int64_t file_size_bytes = 0;
    try {
        file_size_bytes = static_cast<int64_t>(fs::file_size(path));
    } catch (const std::exception& e) {
        spdlog::error("LoadTabularCSV: cannot stat '{}': {}", path, e.what());
        return TabularLoadBackend::Failed;
    }

    // The table is about the size of the CSV; above the data budget's result
    // threshold it stays on disk (TOFIX154 D1/D2).
    const uint64_t memory_threshold = DataResultThresholdBytes();
    const bool use_disk_backed =
        force_disk_backed || static_cast<uint64_t>(file_size_bytes) > memory_threshold;

    spdlog::info("LoadTabularCSV: '{}' file={}, data budget={}, threshold={}, force_disk={}, decision={}",
                 path, FormatDataBytes(static_cast<uint64_t>(file_size_bytes)),
                 FormatDataBytes(DataBudgetBytes()), FormatDataBytes(memory_threshold),
                 force_disk_backed ? "true" : "false",
                 use_disk_backed ? "disk-backed" : "in-memory");

    const std::string cache_signature =
        std::string(has_header ? "header=1" : "header=0") +
        "|delimiter=" + std::to_string(static_cast<unsigned char>(delimiter)) +
        "|decimal=" + std::to_string(static_cast<unsigned char>(decimal_point)) +
        "|quote=" + std::to_string(static_cast<unsigned char>(quote_char)) +
        "|skip=" + std::to_string(skip_rows) +
        "|max_rows=" + std::to_string(std::max<int64_t>(0, max_rows)) +
        "|source_size=" + std::to_string(file_size_bytes) +
        "|nulls=" + MissingValueTokensSignature(missing_value_tokens) +
        "|columns=" + [&selected_columns]() {
            std::string signature;
            for (const auto& column : selected_columns) {
                signature += std::to_string(column.size()) + ":" + column + ";";
            }
            return signature;
        }();
    const std::string cache_path = ParquetBackedDataset::GetCacheFilePath(
        path, cache_signature, cache_directory);
    const bool cache_is_fresh =
        ParquetBackedDataset::IsCacheFresh(path, cache_path);

    if (!use_disk_backed) {
        // The same file with the same options already in memory under
        // another name (a Data Input applied, then run): share its table.
        const std::string load_key = TabularLoadKey(path, "csv|" + cache_signature);
        const std::string final_name = name.empty() ? fs::path(path).stem().string() : name;
        {
            std::string shared_from;
            std::lock_guard<std::mutex> lock(mutex_);
            if (const auto shared = ShareLoadedTableUnlocked(load_key, final_name, path, &shared_from)) {
                if (load_status) *load_status = shared_from == final_name ? "Already loaded" : "Shared with " + shared_from + " (already loaded)";
                spdlog::info("LoadTabularCSV: '{}' is already in memory as '{}'; '{}' shares that table ({} rows)", path,
                             shared_from, final_name, shared->GetNumRows());
                if (progress_callback) (void)progress_callback(1.0f, "Already loaded");
                return TabularLoadBackend::InMemory;
            }
        }
        if (cache_is_fresh) {
            if (progress_callback &&
                !progress_callback(0.05f, "Restoring persistent ingestion cache")) {
                return TabularLoadBackend::Failed;
            }
            auto cached_dataset = ArrowDataset::FromParquet(
                cache_path, final_name);
            if (cached_dataset) {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    InvalidateTabularPublicationUnlocked(final_name);
                    arrow_datasets_[final_name] = cached_dataset;
                    parquet_backed_datasets_.erase(final_name);
                    sparse_feature_datasets_.erase(final_name);
                    RememberTabularSourcePathUnlocked(final_name, path);
                    RecordTabularLoadUnlocked(final_name, load_key, cached_dataset);
                }
                if (load_status) {
                    *load_status = "Restored from persistent ingestion cache";
                }
                spdlog::info(
                    "LoadTabularCSV: persistent cache hit '{}' -> '{}' ({} rows, {} cols)",
                    cache_path, final_name,
                    cached_dataset->GetNumRows(), cached_dataset->GetNumColumns());
                if (progress_callback) {
                    (void)progress_callback(1.0f, "Persistent cache restored");
                }
                return TabularLoadBackend::InMemory;
            }
            spdlog::warn(
                "LoadTabularCSV: persistent cache '{}' could not be opened; rebuilding from source",
                cache_path);
        }

        // First load: parse into Arrow, then persist that exact canonical
        // table for restart-safe restoration. Cache failure is non-fatal.
        auto dataset = LoadCSVToArrow(path, name, has_header, delimiter,
                                      skip_rows, max_rows, missing_value_tokens,
                                      selected_columns, decimal_point,
                                      use_threads, progress_callback, quote_char);
        if (!dataset) {
            // LoadCSVToArrow only publishes after parsing succeeds, so an
            // earlier valid registration remains available on failure.
            return TabularLoadBackend::Failed;
        }
        {
            // Backend switches are committed only after the replacement is
            // ready. This avoids a registry gap while a large CSV is parsed.
            std::lock_guard<std::mutex> lock(mutex_);
            if (parquet_backed_datasets_.count(dataset->GetName())) {
                InvalidateTabularPublicationUnlocked(dataset->GetName());
                parquet_backed_datasets_.erase(dataset->GetName());
            }
            RecordTabularLoadUnlocked(dataset->GetName(), load_key, dataset);
        }
        if (progress_callback) {
            (void)progress_callback(0.995f, "Writing persistent ingestion cache");
        }
        const bool cache_written =
            PersistArrowCacheAtomically(dataset, cache_path);
        if (cache_written) {
            spdlog::info(
                "LoadTabularCSV: persistent ingestion cache ready at '{}'",
                cache_path);
            if (load_status) {
                *load_status = "Loaded in memory; persistent cache ready";
            }
        } else {
            spdlog::warn(
                "LoadTabularCSV: dataset loaded, but persistent cache write failed for '{}'",
                cache_path);
            if (load_status) *load_status = "Loaded in memory";
        }
        return TabularLoadBackend::InMemory;
    }

    // Disk-backed runtime: use the same persistent cache but keep it mapped.
    if (cache_is_fresh) {
        spdlog::info("LoadTabularCSV: reusing existing Parquet cache at '{}'", cache_path);
        if (load_status) *load_status = "Opened persistent ingestion cache";
        if (progress_callback &&
            !progress_callback(0.95f, "Opening existing Parquet cache")) {
            return TabularLoadBackend::Failed;
        }
    } else {
        spdlog::info("LoadTabularCSV: converting CSV to Parquet cache at '{}'", cache_path);
        // The conversion reads every byte of the source: its SHA-256 comes
        // with it, for a lineage that would otherwise read the file again.
        uint64_t source_bytes = 0;
        const std::string stamp = FileStamp(path, &source_bytes);
        std::unique_ptr<Sha256Hasher> hasher;
        uint64_t hashed_bytes = 0;
        bool hash_ok = false;
        CsvSourceObserver observer;
        observer.restart = [&] {
            hasher = std::make_unique<Sha256Hasher>();
            hashed_bytes = 0;
            hash_ok = true;
        };
        observer.bytes = [&](const uint8_t* data, int64_t n) {
            std::string error;
            if (hash_ok) hash_ok = hasher->Update(std::string_view(reinterpret_cast<const char*>(data), static_cast<size_t>(n)), error);
            hashed_bytes += static_cast<uint64_t>(n);
        };
        if (!ParquetBackedDataset::ConvertCsvToParquet(path, cache_path,
                                                       has_header, delimiter, skip_rows,
                                                       max_rows,
                                                       missing_value_tokens,
                                                       selected_columns,
                                                       decimal_point,
                                                       use_threads,
                                                       progress_callback,
                                                       quote_char,
                                                       &observer)) {
            spdlog::error("LoadTabularCSV: CSV-to-Parquet conversion failed for '{}'", path);
            return TabularLoadBackend::Failed;
        }
        // Only a source read to its end, unchanged meanwhile (a row limit stops early).
        std::string digest, error;
        if (hash_ok && hasher && hashed_bytes == source_bytes && FileStamp(path) == stamp && hasher->Finish(digest, error)) {
            RememberFileDigest(stamp, fs::path(cache_path).parent_path(), digest);
            spdlog::info("LoadTabularCSV: source SHA-256 recorded while converting ({})", digest.substr(0, 12));
        }
        if (load_status) *load_status = "Created persistent ingestion cache";
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
        // As above, remove the superseded backend only after the new one is
        // open and registered. Consumers always see a valid dataset.
        InvalidateTabularPublicationUnlocked(unique_name);
        arrow_datasets_.erase(unique_name);
        RememberTabularSourcePathUnlocked(unique_name, path);
    }

    if (progress_callback) {
        (void)progress_callback(1.0f, "Disk-backed dataset ready");
    }

    return TabularLoadBackend::DiskBacked;
}

DataRegistry::TabularLoadBackend DataRegistry::LoadTabularParquet(
    const std::string& path, const std::string& name, bool force_disk_backed,
    const std::vector<std::string>& selected_columns, std::string* load_status,
    const std::string& cache_directory) {
    if (load_status) load_status->clear();
    // Opening reads only the metadata; the decision needs nothing more.
    auto source = ParquetBackedDataset::Open(path, name);
    if (!source) return TabularLoadBackend::Failed;
    const uint64_t threshold = DataResultThresholdBytes();
    uint64_t bytes = source->GetInMemoryBytesEstimate();
    if (!selected_columns.empty() && source->GetNumColumns() > 0) {
        bytes = bytes * selected_columns.size() / static_cast<uint64_t>(source->GetNumColumns());
    }
    const bool use_disk_backed = force_disk_backed || bytes > threshold;
    spdlog::info("LoadTabularParquet: '{}' table~{}, data budget={}, threshold={}, force_disk={}, decision={}",
                 path, FormatDataBytes(bytes), FormatDataBytes(DataBudgetBytes()), FormatDataBytes(threshold),
                 force_disk_backed ? "true" : "false", use_disk_backed ? "disk-backed" : "in-memory");

    if (!use_disk_backed) {
        source.reset();
        std::string columns;
        for (const auto& column : selected_columns) columns += std::to_string(column.size()) + ":" + column + ";";
        const std::string load_key = TabularLoadKey(path, "parquet|columns=" + columns);
        {
            std::string shared_from;
            std::lock_guard<std::mutex> lock(mutex_);
            if (const auto shared = ShareLoadedTableUnlocked(load_key, name, path, &shared_from)) {
                if (load_status) *load_status = shared_from == name ? "Already loaded" : "Shared with " + shared_from + " (already loaded)";
                spdlog::info("LoadTabularParquet: '{}' is already in memory as '{}'; '{}' shares that table", path, shared_from, name);
                return TabularLoadBackend::InMemory;
            }
        }
        auto dataset = LoadParquetToArrow(path, name);
        if (!dataset) return TabularLoadBackend::Failed;
        if (!selected_columns.empty()) {
            auto projected = dataset->SelectColumns(selected_columns);
            if (!projected || projected->GetNumColumns() != static_cast<int64_t>(selected_columns.size())) {
                UnregisterTabularDataset(dataset->GetName());
                spdlog::error("LoadTabularParquet: selected columns do not match '{}'", path);
                return TabularLoadBackend::Failed;
            }
            dataset = RegisterArrowTable(projected->GetArrowTable(), dataset->GetName());
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            RecordTabularLoadUnlocked(dataset->GetName(), load_key, dataset);
        }
        if (load_status) *load_status = "Loaded in memory";
        return TabularLoadBackend::InMemory;
    }

    std::string open_path = path;
    if (!selected_columns.empty()) {
        std::string signature = "parquet_columns=";
        for (const auto& column : selected_columns) signature += std::to_string(column.size()) + ":" + column + ";";
        open_path = ParquetBackedDataset::GetCacheFilePath(path, signature, cache_directory);
        if (!ParquetBackedDataset::IsCacheFresh(path, open_path)) {
            source.reset();
            std::string error;
            if (!ParquetBackedDataset::WriteColumnProjection(path, open_path, selected_columns, error)) {
                spdlog::error("LoadTabularParquet: column selection of '{}' failed: {}", path, error);
                return TabularLoadBackend::Failed;
            }
        }
        if (load_status) *load_status = "Selected columns kept on disk";
    } else if (load_status) {
        *load_status = "Opened in place";
    }
    const std::string unique_name = GenerateUniqueName(name.empty() ? fs::path(path).stem().string() : name);
    auto dataset = open_path == path && source ? source : ParquetBackedDataset::Open(open_path, unique_name);
    if (!dataset) return TabularLoadBackend::Failed;
    RegisterParquetBacked(unique_name, dataset);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        InvalidateTabularPublicationUnlocked(unique_name);
        arrow_datasets_.erase(unique_name);
        RememberTabularSourcePathUnlocked(unique_name, path);
    }
    return TabularLoadBackend::DiskBacked;
}

void DataRegistry::SetExternalReaderOpener(ExternalReaderOpener opener) {
    std::lock_guard<std::mutex> lock(mutex_);
    external_opener_ = std::move(opener);
}

DataRegistry::TabularLoadBackend DataRegistry::LoadTabularExternal(
    const std::string& format, const std::string& path, const std::string& name, bool force_disk_backed,
    const std::vector<std::string>& selected_columns, std::string* load_status,
    const std::string& cache_directory, const std::function<bool()>& should_cancel) {
    if (load_status) load_status->clear();
    const auto fail = [&](const std::string& why) {
        spdlog::error("LoadTabularExternal: '{}' ({}): {}", path, format, why);
        if (load_status) *load_status = why;
        return TabularLoadBackend::Failed;
    };
    data_input::ExternalFormat external;
    if (!data_input::FindExternalFormat(format, external))
        return fail(data_input::UnsupportedReason(format, false));
    ExternalReaderOpener opener;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        opener = external_opener_;
    }
    if (!opener) return fail("No reader is installed for plugin formats");

    // One Parquet copy per file and format version, rewritten when the file changes.
    const std::string cache_path =
        ParquetBackedDataset::GetCacheFilePath(path, "format=" + format + "@" + external.version, cache_directory);
    if (!ParquetBackedDataset::IsCacheFresh(path, cache_path)) {
        std::shared_ptr<arrow::RecordBatchReader> reader;
        std::string error;
        if (!opener(format, path, reader, error) || !reader)
            return fail(error.empty() ? "The format's reader could not open the file" : error);

        const std::string temp_path = ParquetBackedDataset::UniqueTempPath(cache_path);
        auto opened = arrow::io::FileOutputStream::Open(temp_path);
        if (!opened.ok()) return fail("Cannot write the cache file " + temp_path + ": " + opened.status().message());
        std::shared_ptr<arrow::io::FileOutputStream> out = *opened;
        parquet::WriterProperties::Builder properties;
        properties.compression(parquet::Compression::SNAPPY);
        auto made = parquet::arrow::FileWriter::Open(*reader->schema(), arrow::default_memory_pool(), out,
                                                     properties.build(),
                                                     parquet::ArrowWriterProperties::Builder().store_schema()->build());
        if (!made.ok()) {
            (void)out->Close();
            std::error_code ignored;
            fs::remove(temp_path, ignored);
            return fail("Cannot write the cache file: " + made.status().message());
        }
        std::unique_ptr<parquet::arrow::FileWriter> writer = std::move(*made);
        arrow::Status status;
        int64_t rows = 0;
        while (status.ok()) {
            if (should_cancel && should_cancel()) {
                status = arrow::Status::Cancelled("cancelled");
                break;
            }
            std::shared_ptr<arrow::RecordBatch> batch;
            status = reader->ReadNext(&batch);
            if (!status.ok() || !batch) break;
            status = writer->WriteRecordBatch(*batch);
            rows += batch->num_rows();
        }
        reader.reset();  // the plugin's stream is released before the file is used
        const arrow::Status closed = writer->Close();
        const arrow::Status out_closed = out->Close();
        if (status.ok() && !closed.ok()) status = closed;
        if (status.ok() && !out_closed.ok()) status = out_closed;
        std::error_code ec;
        if (status.ok()) fs::rename(temp_path, cache_path, ec);
        if (!status.ok() || ec) {
            fs::remove(temp_path, ec);
            if (status.IsCancelled()) return fail("Cancelled");
            return fail(status.ok() ? "Cannot place the cache file: " + ec.message() : status.message());
        }
        spdlog::info("LoadTabularExternal: '{}' ({}) streamed {} rows into {}", path, external.label, rows, cache_path);
    }
    const auto backend = LoadTabularParquet(cache_path, name, force_disk_backed, selected_columns, load_status,
                                            cache_directory);
    if (backend == TabularLoadBackend::Failed && load_status && load_status->empty())
        *load_status = "The converted file could not be read; check the selected columns";
    if (backend != TabularLoadBackend::Failed) {
        // The table came from `path`, not from its cache copy: Preview and
        // the dialogs find it by the file the user chose.
        std::lock_guard<std::mutex> lock(mutex_);
        if (const auto it = tabular_dataset_by_source_path_.find(NormalizeTabularSourcePath(cache_path));
            it != tabular_dataset_by_source_path_.end()) {
            const std::string loaded = it->second;
            RememberTabularSourcePathUnlocked(loaded, path);
        }
    }
    return backend;
}

std::optional<TableRef> DataRegistry::GetTableRef(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (const auto it = arrow_datasets_.find(name); it != arrow_datasets_.end() && it->second) {
        TableRef ref;
        ref.name = name;
        ref.kind = TableRef::Kind::Memory;
        ref.rows = it->second->GetNumRows();
        ref.columns = it->second->GetNumColumns();
        ref.bytes = it->second->GetMemoryUsage();
        return ref;
    }
    if (const auto it = parquet_backed_datasets_.find(name); it != parquet_backed_datasets_.end() && it->second) {
        TableRef ref;
        ref.name = name;
        ref.kind = TableRef::Kind::Parquet;
        ref.rows = it->second->GetNumRows();
        ref.columns = it->second->GetNumColumns();
        ref.bytes = it->second->GetInMemoryBytesEstimate();
        ref.path = it->second->GetFilePath();
        ref.row_groups = it->second->GetNumRowGroups();
        return ref;
    }
    return std::nullopt;
}

std::shared_ptr<arrow::Table> DataRegistry::ReadTablePreview(const std::string& name, int64_t max_rows) {
    if (const auto memory = GetArrowDataset(name)) {
        const auto table = memory->GetArrowTable();
        return table ? table->Slice(0, std::max<int64_t>(0, max_rows)) : nullptr;
    }
    const auto disk = GetParquetBackedDataset(name);
    if (!disk) return nullptr;
    std::vector<int> groups;
    int64_t rows = 0;
    for (int i = 0; i < disk->GetNumRowGroups() && rows < max_rows; ++i) {
        groups.push_back(i);
        rows += disk->GetRowGroupSize(i);
    }
    if (groups.empty()) return arrow::Table::MakeEmpty(disk->GetSchema()).ValueOr(nullptr);
    const auto table = disk->ReadRowGroups(groups);
    return table ? table->Slice(0, max_rows) : nullptr;
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
        InvalidateTabularPublicationUnlocked(unique_name);
        arrow_datasets_[unique_name] = dataset;
        parquet_backed_datasets_.erase(unique_name);
        sparse_feature_datasets_.erase(unique_name);
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
    if (!json_lines) {
        spdlog::error(
            "LoadJSONToArrow: '{}' is plain JSON; tabular input reads JSON Lines (one object per line, .jsonl)",
            path);
        return nullptr;
    }
    std::string error;
    auto table = ReadJsonLinesTable(path, false, error);
    if (!table) {
        spdlog::error("LoadJSONToArrow: {}", error);
        return nullptr;
    }
    auto dataset = RegisterArrowTable(table, name.empty() ? fs::path(path).stem().string() : name);
    if (dataset) {
        spdlog::info("LoadJSONToArrow: loaded JSON Lines '{}' ({} rows, {} cols)",
                     path, dataset->GetNumRows(), dataset->GetNumColumns());
    }
    return dataset;
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

bool DataRegistry::VisitTablePieces(const std::string& name,
                                    const std::function<bool(const std::shared_ptr<arrow::Table>&, std::string&)>& visit,
                                    std::shared_ptr<arrow::Schema>* schema, std::string& error) {
    if (const auto memory = GetArrowDataset(name)) {
        const auto table = memory->GetArrowTable();
        if (!table) {
            error = "'" + name + "' has no table";
            return false;
        }
        if (schema) *schema = table->schema();
        return visit(table, error);
    }
    const auto disk = GetParquetBackedDataset(name);
    if (!disk) {
        error = "'" + name + "' is not a table";
        return false;
    }
    if (schema) *schema = disk->GetSchema();
    for (int group = 0; group < disk->GetNumRowGroups(); ++group) {
        const auto piece = disk->ReadRowGroup(group);
        if (!piece) {
            error = "could not read row group " + std::to_string(group) + " of '" + disk->GetFilePath() + "'";
            return false;
        }
        if (!visit(piece, error)) return false;
    }
    return true;
}

bool DataRegistry::VisitExportedPieces(const std::string& name,
                                       const std::function<bool(const std::shared_ptr<arrow::Table>&, std::string&)>& visit,
                                       std::shared_ptr<arrow::Schema>* schema, std::string& error) {
    const bool ok = VisitTablePieces(name, [&visit](const std::shared_ptr<arrow::Table>& piece, std::string& why) {
        const int index = piece->schema()->GetFieldIndex(kSplitPartitionColumn);
        if (index < 0) return visit(piece, why);
        auto without = piece->RemoveColumn(index);
        if (!without.ok()) {
            why = without.status().ToString();
            return false;
        }
        return visit(*without, why);
    }, schema, error);
    if (schema && *schema) {
        const int index = (*schema)->GetFieldIndex(kSplitPartitionColumn);
        if (index >= 0) {
            if (auto without = (*schema)->RemoveField(index); without.ok()) *schema = *without;
        }
    }
    return ok;
}

namespace {

// An export's file, buffered and hashed as it is written.
struct ExportFile {
    std::shared_ptr<HashingOutputStream> hashed;
    std::shared_ptr<arrow::io::OutputStream> buffered;
};

arrow::Result<ExportFile> OpenExportFile(const std::string& path) {
    ARROW_ASSIGN_OR_RAISE(auto file, arrow::io::FileOutputStream::Open(path));
    ExportFile out;
    out.hashed = std::make_shared<HashingOutputStream>(file);
    ARROW_ASSIGN_OR_RAISE(out.buffered, arrow::io::BufferedOutputStream::Create(1 << 20, arrow::default_memory_pool(), out.hashed));
    return out;
}

}  // namespace

bool DataRegistry::ExportTableCSV(const std::string& name, const std::string& output_path, TabularExport* written) {
    std::string error;
    auto opened = OpenExportFile(output_path);
    if (!opened.ok()) {
        spdlog::error("ExportTableCSV: cannot open '{}': {}", output_path, opened.status().ToString());
        return false;
    }
    ExportFile file = *opened;
    std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
    TabularExport facts;
    const bool ok = VisitExportedPieces(name, [&](const std::shared_ptr<arrow::Table>& piece, std::string& why) {
        if (!writer) {
            auto options = arrow::csv::WriteOptions::Defaults();
            options.delimiter = ',';
            options.include_header = true;
            auto made = arrow::csv::MakeCSVWriter(file.buffered, piece->schema(), options);
            if (!made.ok()) {
                why = made.status().ToString();
                return false;
            }
            writer = *made;
        }
        if (auto status = writer->WriteTable(*piece); !status.ok()) {
            why = status.ToString();
            return false;
        }
        facts.rows += piece->num_rows();
        return true;
    }, &facts.schema, error);
    // A table with no row groups still gets its header.
    if (ok && !writer && facts.schema) {
        auto options = arrow::csv::WriteOptions::Defaults();
        auto made = arrow::csv::MakeCSVWriter(file.buffered, facts.schema, options);
        if (made.ok()) writer = *made;
    }
    arrow::Status closed = writer ? writer->Close() : arrow::Status::OK();
    if (closed.ok()) closed = file.buffered->Close();
    if (!ok || !closed.ok()) {
        spdlog::error("ExportTableCSV: '{}' to '{}' failed: {}", name, output_path, ok ? closed.ToString() : error);
        return false;
    }
    facts.bytes = file.hashed->bytes();
    facts.sha256 = file.hashed->Finish();
    spdlog::info("ExportTableCSV: '{}' to '{}' ({} rows)", name, output_path, facts.rows);
    if (written) *written = std::move(facts);
    return true;
}

bool DataRegistry::ExportTableParquet(const std::string& name, const std::string& output_path, TabularExport* written) {
    std::string error;
    auto opened = OpenExportFile(output_path);
    if (!opened.ok()) {
        spdlog::error("ExportTableParquet: cannot open '{}': {}", output_path, opened.status().ToString());
        return false;
    }
    ExportFile file = *opened;
    std::unique_ptr<parquet::arrow::FileWriter> writer;
    TabularExport facts;
    const auto open_writer = [&](const std::shared_ptr<arrow::Schema>& schema, std::string& why) {
        parquet::WriterProperties::Builder properties;
        properties.compression(parquet::Compression::SNAPPY);
        auto made = parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), file.buffered, properties.build(),
                                                     parquet::ArrowWriterProperties::Builder().store_schema()->build());
        if (!made.ok()) {
            why = made.status().ToString();
            return false;
        }
        writer = std::move(*made);
        return true;
    };
    bool ok = VisitExportedPieces(name, [&](const std::shared_ptr<arrow::Table>& piece, std::string& why) {
        if (!writer && !open_writer(piece->schema(), why)) return false;
        if (auto status = writer->WriteTable(*piece, 1024 * 1024); !status.ok()) {
            why = status.ToString();
            return false;
        }
        facts.rows += piece->num_rows();
        return true;
    }, &facts.schema, error);
    if (ok && !writer && facts.schema) ok = open_writer(facts.schema, error);  // no row groups: an empty file with the schema
    arrow::Status closed = writer ? writer->Close() : arrow::Status::OK();
    if (closed.ok()) closed = file.buffered->Close();
    if (!ok || !closed.ok()) {
        spdlog::error("ExportTableParquet: '{}' to '{}' failed: {}", name, output_path, ok ? closed.ToString() : error);
        return false;
    }
    facts.bytes = file.hashed->bytes();
    facts.sha256 = file.hashed->Finish();
    spdlog::info("ExportTableParquet: '{}' to '{}' ({} rows)", name, output_path, facts.rows);
    if (written) *written = std::move(facts);
    return true;
}

bool DataRegistry::SortArrowTableByColumn(const std::string& name, const std::string& column,
                                          std::string& error) {
    const auto dataset = GetArrowDataset(name);
    if (!dataset || !dataset->GetArrowTable()) {
        error = "'" + name + "' is not a table in memory";
        return false;
    }
    const auto table = dataset->GetArrowTable();
    const auto chunked = table->GetColumnByName(column);
    if (!chunked) {
        error = "The time column '" + column + "' is not in the table";
        return false;
    }
    const auto fail = [&](const arrow::Status& status) {
        error = "Could not sort by '" + column + "': " + status.ToString();
        return false;
    };
    // This Arrow build has no sort kernel (no arrow_compute): order the row
    // numbers here, then Take. Text compares as text; numbers, dates and
    // times through int64 / float64; missing values go last; ties keep
    // the file's order.
    auto joined = arrow::Concatenate(chunked->chunks());
    if (!joined.ok()) return fail(joined.status());
    const auto values = *joined;
    const auto type = values->type_id();
    std::vector<int64_t> order(static_cast<size_t>(values->length()));
    for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int64_t>(i);
    const auto nulls_last = [&](int64_t a, int64_t b, const auto& less) {
        const bool na = values->IsNull(a), nb = values->IsNull(b);
        if (na || nb) return !na && nb;
        return less(a, b);
    };
    if (type == arrow::Type::STRING || type == arrow::Type::LARGE_STRING) {
        const auto text = [&](int64_t row) {
            return type == arrow::Type::STRING
                ? std::static_pointer_cast<arrow::StringArray>(values)->GetView(row)
                : std::static_pointer_cast<arrow::LargeStringArray>(values)->GetView(row);
        };
        std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
            return nulls_last(a, b, [&](int64_t x, int64_t y) { return text(x) < text(y); });
        });
    } else if (arrow::is_floating(type)) {
        auto cast = arrow::compute::Cast(*values, arrow::float64());
        if (!cast.ok()) return fail(cast.status());
        const auto numbers = std::static_pointer_cast<arrow::DoubleArray>(*cast);
        std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
            return nulls_last(a, b, [&](int64_t x, int64_t y) { return numbers->Value(x) < numbers->Value(y); });
        });
    } else if (arrow::is_integer(type) || arrow::is_temporal(type) || type == arrow::Type::BOOL) {
        std::shared_ptr<arrow::Array> as_int = values;
        if (type != arrow::Type::INT64) {
            // Dates and times are stored as integers: read them as such.
            std::shared_ptr<arrow::Array> storage = values;
            if (arrow::is_temporal(type)) {
                auto view = values->View(arrow::bit_width(type) == 32 ? arrow::int32() : arrow::int64());
                if (!view.ok()) return fail(view.status());
                storage = *view;
            }
            auto cast = arrow::compute::Cast(*storage, arrow::int64());
            if (!cast.ok()) return fail(cast.status());
            as_int = *cast;
        }
        const auto numbers = std::static_pointer_cast<arrow::Int64Array>(as_int);
        std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
            return nulls_last(a, b, [&](int64_t x, int64_t y) { return numbers->Value(x) < numbers->Value(y); });
        });
    } else {
        error = "The time column '" + column + "' holds " + values->type()->ToString() +
                ", which cannot be put in order";
        return false;
    }
    arrow::Int64Builder builder;
    if (auto status = builder.AppendValues(order); !status.ok()) return fail(status);
    std::shared_ptr<arrow::Array> indices;
    if (auto status = builder.Finish(&indices); !status.ok()) return fail(status);
    auto sorted = arrow::compute::Take(arrow::Datum(table), arrow::Datum(indices));
    if (!sorted.ok()) return fail(sorted.status());
    auto combined = sorted->table()->CombineChunks();
    if (!combined.ok()) return fail(combined.status());
    // Same name, same source file: only the row order changes.
    std::lock_guard<std::mutex> lock(mutex_);
    InvalidateTabularPublicationUnlocked(name);
    arrow_datasets_[name] = std::make_shared<ArrowDataset>(*combined, name);
    last_access_times_[name] = std::chrono::steady_clock::now();
    return true;
}

} // namespace cyxwiz
