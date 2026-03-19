#include "data_registry.h"
#include "dataset_base.h"
#include "arrow_dataset.h"
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
// Dataset Base Class Implementation
// =============================================================================

std::pair<std::vector<std::vector<float>>, std::vector<int>>
Dataset::GetBatch(const std::vector<size_t>& indices) const {
    std::vector<std::vector<float>> samples;
    std::vector<int> labels;
    samples.reserve(indices.size());
    labels.reserve(indices.size());

    for (size_t idx : indices) {
        auto [sample, label] = GetItem(idx);
        samples.push_back(std::move(sample));
        labels.push_back(label);
    }

    return {std::move(samples), std::move(labels)};
}

void Dataset::SetSplit(const SplitConfig& config) {
    split_config_ = config;

    // Create all indices
    all_indices_.resize(Size());
    for (size_t i = 0; i < Size(); i++) {
        all_indices_[i] = i;
    }

    // Shuffle if requested
    if (config.shuffle) {
        std::mt19937 rng(config.seed);
        std::shuffle(all_indices_.begin(), all_indices_.end(), rng);
    }

    // Calculate split sizes
    size_t total = Size();
    size_t train_size = static_cast<size_t>(total * config.train_ratio);
    size_t val_size = static_cast<size_t>(total * config.val_ratio);
    size_t test_size = total - train_size - val_size;

    // Assign indices to splits
    train_indices_.clear();
    val_indices_.clear();
    test_indices_.clear();

    train_indices_.reserve(train_size);
    val_indices_.reserve(val_size);
    test_indices_.reserve(test_size);

    for (size_t i = 0; i < train_size; i++) {
        train_indices_.push_back(all_indices_[i]);
    }
    for (size_t i = train_size; i < train_size + val_size; i++) {
        val_indices_.push_back(all_indices_[i]);
    }
    for (size_t i = train_size + val_size; i < total; i++) {
        test_indices_.push_back(all_indices_[i]);
    }

    spdlog::info("Dataset split: train={}, val={}, test={}",
        train_indices_.size(), val_indices_.size(), test_indices_.size());
}

const std::vector<size_t>& Dataset::GetSplitIndices(DatasetSplit split) const {
    switch (split) {
        case DatasetSplit::Train: return train_indices_;
        case DatasetSplit::Validation: return val_indices_;
        case DatasetSplit::Test: return test_indices_;
        case DatasetSplit::All: return all_indices_;
        default: return all_indices_;
    }
}

// =============================================================================
// DatasetHandle Implementation
// =============================================================================

DatasetHandle::DatasetHandle(std::shared_ptr<Dataset> dataset, const std::string& name)
    : dataset_(std::move(dataset)), name_(name) {}

DatasetInfo DatasetHandle::GetInfo() const {
    if (!IsValid()) return DatasetInfo{};
    return dataset_->GetInfo();
}

size_t DatasetHandle::Size() const {
    if (!IsValid()) return 0;
    return dataset_->Size();
}

size_t DatasetHandle::Size(DatasetSplit split) const {
    if (!IsValid()) return 0;
    return dataset_->GetSplitIndices(split).size();
}

std::pair<std::vector<float>, int> DatasetHandle::GetSample(size_t index) const {
    if (!IsValid()) return {{}, -1};
    return dataset_->GetItem(index);
}

std::pair<std::vector<std::vector<float>>, std::vector<int>>
DatasetHandle::GetBatch(const std::vector<size_t>& indices) const {
    if (!IsValid()) return {{}, {}};
    return dataset_->GetBatch(indices);
}

const std::vector<size_t>& DatasetHandle::GetTrainIndices() const {
    static std::vector<size_t> empty;
    if (!IsValid()) return empty;
    return dataset_->GetTrainIndices();
}

const std::vector<size_t>& DatasetHandle::GetValIndices() const {
    static std::vector<size_t> empty;
    if (!IsValid()) return empty;
    return dataset_->GetValIndices();
}

const std::vector<size_t>& DatasetHandle::GetTestIndices() const {
    static std::vector<size_t> empty;
    if (!IsValid()) return empty;
    return dataset_->GetTestIndices();
}

const std::vector<size_t>& DatasetHandle::GetSplitIndices(DatasetSplit split) const {
    static std::vector<size_t> empty;
    if (!IsValid()) return empty;
    return dataset_->GetSplitIndices(split);
}

void DatasetHandle::ApplySplit(const SplitConfig& config) {
    if (IsValid()) {
        dataset_->SetSplit(config);
    }
}

// =============================================================================
// DataRegistry Implementation
// =============================================================================

DataRegistry& DataRegistry::Instance() {
    static DataRegistry instance;
    return instance;
}

std::string DataRegistry::GenerateUniqueName(const std::string& base_name) {
    std::string name = base_name;
    if (name.empty()) {
        name = "dataset";
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (datasets_.find(name) == datasets_.end()) {
        return name;
    }

    // Add suffix to make unique
    int suffix = 1;
    while (datasets_.find(name + "_" + std::to_string(suffix)) != datasets_.end()) {
        suffix++;
    }
    return name + "_" + std::to_string(suffix);
}

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

void DataRegistry::UnloadDataset(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = datasets_.find(name);
    if (it != datasets_.end()) {
        datasets_.erase(it);
        spdlog::info("Unloaded dataset: {}", name);

        if (on_unloaded_) {
            on_unloaded_(name);
        }
    }
}

void DataRegistry::UnloadAll() {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> names;
    for (const auto& [name, _] : datasets_) {
        names.push_back(name);
    }

    datasets_.clear();

    for (const auto& name : names) {
        if (on_unloaded_) {
            on_unloaded_(name);
        }
    }

    spdlog::info("Unloaded all datasets");
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

DatasetPreview DataRegistry::GetPreview(const std::string& path, int max_samples) {
    DatasetPreview preview;
    preview.type = DetectType(path);

    if (!fs::exists(path)) {
        return preview;
    }

    // Get file size
    if (fs::is_regular_file(path)) {
        preview.file_size = fs::file_size(path);
    } else if (fs::is_directory(path)) {
        for (const auto& entry : fs::recursive_directory_iterator(path)) {
            if (entry.is_regular_file()) {
                preview.file_size += entry.file_size();
            }
        }
    }

    // Generate preview based on type
    switch (preview.type) {
        case DatasetType::CSV: {
            std::ifstream file(path);
            if (!file) return preview;

            std::string line;
            int line_count = 0;

            while (std::getline(file, line) && line_count <= max_samples) {
                std::vector<std::string> tokens;
                std::stringstream ss(line);
                std::string token;

                while (std::getline(ss, token, ',')) {
                    token.erase(0, token.find_first_not_of(" \t\r\n"));
                    token.erase(token.find_last_not_of(" \t\r\n") + 1);
                    tokens.push_back(token);
                }

                if (line_count == 0) {
                    // Check if header
                    try {
                        (void)std::stof(tokens[0]);  // Just checking if it's numeric
                        preview.rows.push_back(tokens);
                    } catch (...) {
                        preview.columns = tokens;
                    }
                } else {
                    preview.rows.push_back(tokens);
                }
                line_count++;
            }

            // Count total lines
            file.clear();
            file.seekg(0);
            preview.num_samples = std::count(
                std::istreambuf_iterator<char>(file),
                std::istreambuf_iterator<char>(), '\n');
            if (!preview.columns.empty()) preview.num_samples--;

            break;
        }

        case DatasetType::MNIST: {
            preview.shape = {28, 28, 1};
            preview.num_classes = 10;

            // Quick count from header
            std::string images_file = path + "/train-images-idx3-ubyte";
            if (!fs::exists(images_file)) {
                images_file = path + "/train-images.idx3-ubyte";
            }

            if (fs::exists(images_file)) {
                std::ifstream file(images_file, std::ios::binary);
                file.seekg(4);  // Skip magic

                uint32_t num;
                file.read(reinterpret_cast<char*>(&num), 4);
                // Convert from big-endian
                preview.num_samples = ((num & 0xFF) << 24) | ((num & 0xFF00) << 8) |
                                     ((num & 0xFF0000) >> 8) | ((num & 0xFF000000) >> 24);
            }
            break;
        }

        case DatasetType::CIFAR10: {
            preview.shape = {32, 32, 3};
            preview.num_classes = 10;
            preview.num_samples = 50000;  // Standard CIFAR-10 training set
            break;
        }

        default:
            break;
    }

    return preview;
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
// Configuration Export/Import
// =============================================================================

std::string DataRegistry::SerializeConfig(const DatasetInfo& info, const SplitConfig& split) {
    nlohmann::json j;

    // Dataset info
    j["name"] = info.name;
    j["path"] = info.path;
    j["type"] = TypeToString(info.type);
    j["shape"] = info.shape;
    j["num_samples"] = info.num_samples;
    j["num_classes"] = info.num_classes;
    j["class_names"] = info.class_names;

    // Split config
    j["split"]["train_ratio"] = split.train_ratio;
    j["split"]["val_ratio"] = split.val_ratio;
    j["split"]["test_ratio"] = split.test_ratio;
    j["split"]["stratified"] = split.stratified;
    j["split"]["shuffle"] = split.shuffle;
    j["split"]["seed"] = split.seed;

    // Metadata
    j["version"] = "1.0";
    j["exported_at"] = std::time(nullptr);

    return j.dump(2);
}

bool DataRegistry::DeserializeConfig(const std::string& json_str, DatasetInfo& info, SplitConfig& split) {
    try {
        nlohmann::json j = nlohmann::json::parse(json_str);

        // Dataset info
        info.name = j.value("name", "");
        info.path = j.value("path", "");

        std::string type_str = j.value("type", "None");
        if (type_str == "CSV") info.type = DatasetType::CSV;
        else if (type_str == "TSV") info.type = DatasetType::TSV;
        else if (type_str == "ImageFolder") info.type = DatasetType::ImageFolder;
        else if (type_str == "ImageCSV") info.type = DatasetType::ImageCSV;
        else if (type_str == "MNIST") info.type = DatasetType::MNIST;
        else if (type_str == "FashionMNIST") info.type = DatasetType::FashionMNIST;
        else if (type_str == "CIFAR10") info.type = DatasetType::CIFAR10;
        else if (type_str == "CIFAR100") info.type = DatasetType::CIFAR100;
        else if (type_str == "HuggingFace") info.type = DatasetType::HuggingFace;
        else if (type_str == "Kaggle") info.type = DatasetType::Kaggle;
        else if (type_str == "Custom") info.type = DatasetType::Custom;
        else info.type = DatasetType::None;

        if (j.contains("shape")) {
            info.shape = j["shape"].get<std::vector<size_t>>();
        }
        info.num_samples = j.value("num_samples", size_t(0));
        info.num_classes = j.value("num_classes", size_t(0));
        if (j.contains("class_names")) {
            info.class_names = j["class_names"].get<std::vector<std::string>>();
        }

        // Split config
        if (j.contains("split")) {
            auto& s = j["split"];
            split.train_ratio = s.value("train_ratio", 0.8f);
            split.val_ratio = s.value("val_ratio", 0.1f);
            split.test_ratio = s.value("test_ratio", 0.1f);
            split.stratified = s.value("stratified", true);
            split.shuffle = s.value("shuffle", true);
            split.seed = s.value("seed", 42);
        }

        return true;
    } catch (const std::exception& e) {
        spdlog::error("Failed to deserialize dataset config: {}", e.what());
        return false;
    }
}

bool DataRegistry::ExportConfig(const std::string& name, const std::string& filepath) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = datasets_.find(name);
    if (it == datasets_.end()) {
        spdlog::error("Cannot export config: dataset '{}' not found", name);
        return false;
    }

    DatasetInfo info = it->second->GetInfo();
    SplitConfig split;
    split.train_ratio = info.train_ratio;
    split.val_ratio = info.val_ratio;
    split.test_ratio = info.test_ratio;

    std::string json_str = SerializeConfig(info, split);

    std::ofstream file(filepath);
    if (!file.is_open()) {
        spdlog::error("Cannot open file for writing: {}", filepath);
        return false;
    }

    file << json_str;
    file.close();

    spdlog::info("Exported dataset config '{}' to {}", name, filepath);
    return true;
}

bool DataRegistry::ExportConfig(const std::string& name, const std::string& filepath, const SplitConfig& split) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = datasets_.find(name);
    if (it == datasets_.end()) {
        spdlog::error("Cannot export config: dataset '{}' not found", name);
        return false;
    }

    DatasetInfo info = it->second->GetInfo();
    std::string json_str = SerializeConfig(info, split);

    std::ofstream file(filepath);
    if (!file.is_open()) {
        spdlog::error("Cannot open file for writing: {}", filepath);
        return false;
    }

    file << json_str;
    file.close();

    spdlog::info("Exported dataset config '{}' to {} (custom split: {:.0f}/{:.0f}/{:.0f})",
                 name, filepath, split.train_ratio * 100, split.val_ratio * 100, split.test_ratio * 100);
    return true;
}

bool DataRegistry::ImportConfig(const std::string& filepath, std::string& out_name) {
    SplitConfig ignored_split;
    return ImportConfig(filepath, out_name, ignored_split);
}

bool DataRegistry::ImportConfig(const std::string& filepath, std::string& out_name, SplitConfig& out_split) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        spdlog::error("Cannot open config file: {}", filepath);
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();

    DatasetInfo info;
    SplitConfig split;

    if (!DeserializeConfig(buffer.str(), info, split)) {
        return false;
    }

    // Load the dataset using the config
    if (info.path.empty()) {
        spdlog::error("Config file does not specify a dataset path");
        return false;
    }

    DatasetHandle handle;

    // Check if dataset with same name or path is already loaded
    bool already_loaded = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);

        // First check by name
        if (!info.name.empty()) {
            auto it = datasets_.find(info.name);
            if (it != datasets_.end()) {
                handle = DatasetHandle(it->second, info.name);
                already_loaded = true;
                spdlog::info("Dataset '{}' already loaded, applying config only", info.name);
            }
        }

        // If not found by name, check by path
        if (!already_loaded) {
            for (const auto& [name, dataset] : datasets_) {
                DatasetInfo existing_info = dataset->GetInfo();
                if (existing_info.path == info.path) {
                    handle = DatasetHandle(dataset, name);
                    already_loaded = true;
                    spdlog::info("Dataset with path '{}' already loaded as '{}', applying config only",
                                 info.path, name);
                    break;
                }
            }
        }
    }

    // Only load if not already present
    if (!already_loaded) {
        // Report progress
        if (on_progress_) {
            on_progress_(0.0f, "Loading dataset from config...");
        }

        handle = LoadDataset(info.path, info.name);
        if (!handle.IsValid()) {
            spdlog::error("Failed to load dataset from path: {}", info.path);
            return false;
        }
    }

    // Apply split configuration
    handle.ApplySplit(split);

    out_name = handle.GetName();
    out_split = split;  // Return the split config from the file

    if (on_progress_) {
        on_progress_(1.0f, already_loaded ? "Config applied" : "Dataset loaded successfully");
    }

    spdlog::info("Imported dataset config from {}, {} '{}' (split: {:.0f}/{:.0f}/{:.0f})", filepath,
                 already_loaded ? "applied to existing" : "loaded as", out_name,
                 split.train_ratio * 100, split.val_ratio * 100, split.test_ratio * 100);
    return true;
}

// =============================================================================
// Dataset Versioning
// =============================================================================

std::vector<DataRegistry::DatasetVersion> DataRegistry::GetVersionHistory(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = version_history_.find(name);
    if (it != version_history_.end()) {
        return it->second;
    }
    return {};
}

bool DataRegistry::SaveVersion(const std::string& name, const std::string& description) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = datasets_.find(name);
    if (it == datasets_.end()) {
        spdlog::error("Cannot save version: dataset '{}' not found", name);
        return false;
    }

    DatasetInfo info = it->second->GetInfo();

    // Create version entry
    DatasetVersion version;

    // Generate version ID (simple incrementing)
    auto& history = version_history_[name];
    version.version_id = "v" + std::to_string(history.size() + 1);

    // Timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    char time_buf[32];
    std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", std::localtime(&time_t_now));
    version.timestamp = time_buf;

    version.description = description.empty() ? "Auto-saved version" : description;
    version.num_samples = info.num_samples;

    // Simple checksum based on sample count and memory usage
    std::stringstream ss;
    ss << info.num_samples << "_" << info.memory_usage << "_" << info.num_classes;
    version.checksum = ss.str();

    history.push_back(version);

    spdlog::info("Saved version {} for dataset '{}'", version.version_id, name);
    return true;
}

// =============================================================================
// Preprocessing Configuration Management
// =============================================================================

void DataRegistry::SetPreprocessingConfig(const std::string& dataset_id, const PreprocessingConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    preprocessing_configs_[dataset_id] = config;
    spdlog::info("DataRegistry: Set preprocessing config for dataset '{}'", dataset_id);
}

PreprocessingConfig DataRegistry::GetPreprocessingConfig(const std::string& dataset_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = preprocessing_configs_.find(dataset_id);
    if (it != preprocessing_configs_.end()) {
        return it->second;
    }
    // Return empty config if not found
    PreprocessingConfig empty_config;
    empty_config.dataset_id = dataset_id;
    return empty_config;
}

bool DataRegistry::HasPreprocessingConfig(const std::string& dataset_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return preprocessing_configs_.find(dataset_id) != preprocessing_configs_.end();
}

void DataRegistry::ClearPreprocessingConfig(const std::string& dataset_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    preprocessing_configs_.erase(dataset_id);
    spdlog::info("DataRegistry: Cleared preprocessing config for dataset '{}'", dataset_id);
}

// =============================================================================
// Augmentation Pipeline Management
// =============================================================================

void DataRegistry::SetAugmentationPipeline(const std::string& dataset_id,
                                            std::shared_ptr<transforms::Compose> pipeline) {
    std::lock_guard<std::mutex> lock(mutex_);
    augmentation_pipelines_[dataset_id] = pipeline;
    spdlog::info("DataRegistry: Set augmentation pipeline for dataset '{}'", dataset_id);
}

std::shared_ptr<transforms::Compose> DataRegistry::GetAugmentationPipeline(
    const std::string& dataset_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = augmentation_pipelines_.find(dataset_id);
    if (it != augmentation_pipelines_.end()) {
        return it->second;
    }
    return nullptr;
}

bool DataRegistry::HasAugmentationPipeline(const std::string& dataset_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return augmentation_pipelines_.find(dataset_id) != augmentation_pipelines_.end();
}

void DataRegistry::ClearAugmentationPipeline(const std::string& dataset_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    augmentation_pipelines_.erase(dataset_id);
    spdlog::info("DataRegistry: Cleared augmentation pipeline for dataset '{}'", dataset_id);
}

// =============================================================================
// Annotation Manager
// =============================================================================

AnnotationManager& DataRegistry::GetAnnotationManager() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!annotation_manager_) {
        annotation_manager_ = std::make_unique<AnnotationManager>();
    }
    return *annotation_manager_;
}

const AnnotationManager& DataRegistry::GetAnnotationManager() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!annotation_manager_) {
        annotation_manager_ = std::make_unique<AnnotationManager>();
    }
    return *annotation_manager_;
}

// =============================================================================
// Apache Arrow Integration (Data Studio Foundation - Phase 0)
// =============================================================================

std::shared_ptr<ArrowDataset> DataRegistry::LoadArrowTable(
    const std::string& path, const std::string& name) {

    std::lock_guard<std::mutex> lock(mutex_);

    // Generate unique name if not provided
    std::string dataset_name = name.empty() ? GenerateUniqueName("arrow_data") : name;

    // Check if already loaded
    if (arrow_datasets_.find(dataset_name) != arrow_datasets_.end()) {
        spdlog::warn("Arrow dataset '{}' already loaded, returning existing instance", dataset_name);
        return arrow_datasets_[dataset_name];
    }

    // Load from file using ArrowDataset factory
    auto dataset = ArrowDataset::FromFile(path, dataset_name);
    if (!dataset) {
        spdlog::error("Failed to load Arrow dataset from: {}", path);
        return nullptr;
    }

    // Store in registry
    arrow_datasets_[dataset_name] = dataset;
    last_access_times_[dataset_name] = std::chrono::steady_clock::now();

    spdlog::info("Loaded Arrow dataset '{}': {} rows, {} columns, {} bytes",
                 dataset_name,
                 dataset->GetNumRows(),
                 dataset->GetNumColumns(),
                 dataset->GetMemoryUsage());

    // Trigger callback
    if (on_loaded_) {
        DatasetInfo info;
        info.name = dataset_name;
        info.path = path;
        info.num_samples = dataset->GetNumRows();
        info.memory_usage = dataset->GetMemoryUsage();
        info.is_loaded = true;
        on_loaded_(dataset_name, info);
    }

    return dataset;
}

std::shared_ptr<ArrowDataset> DataRegistry::RegisterArrowTable(
    std::shared_ptr<arrow::Table> table, const std::string& name) {

    std::lock_guard<std::mutex> lock(mutex_);

    // Generate unique name if not provided
    std::string dataset_name = name.empty() ? GenerateUniqueName("arrow_data") : name;

    // Check if already exists
    if (arrow_datasets_.find(dataset_name) != arrow_datasets_.end()) {
        spdlog::warn("Arrow dataset '{}' already exists, overwriting", dataset_name);
    }

    // Create ArrowDataset wrapper
    auto dataset = std::make_shared<ArrowDataset>(table, dataset_name);

    // Store in registry
    arrow_datasets_[dataset_name] = dataset;
    last_access_times_[dataset_name] = std::chrono::steady_clock::now();

    spdlog::info("Registered Arrow dataset '{}': {} rows, {} columns, {} bytes",
                 dataset_name,
                 dataset->GetNumRows(),
                 dataset->GetNumColumns(),
                 dataset->GetMemoryUsage());

    // Trigger callback
    if (on_loaded_) {
        DatasetInfo info;
        info.name = dataset_name;
        info.num_samples = dataset->GetNumRows();
        info.memory_usage = dataset->GetMemoryUsage();
        info.is_loaded = true;
        on_loaded_(dataset_name, info);
    }

    return dataset;
}

std::shared_ptr<ArrowDataset> DataRegistry::GetArrowDataset(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = arrow_datasets_.find(name);
    if (it == arrow_datasets_.end()) {
        spdlog::warn("Arrow dataset '{}' not found in registry", name);
        return nullptr;
    }

    // Update last access time for LRU
    last_access_times_[name] = std::chrono::steady_clock::now();

    return it->second;
}

bool DataRegistry::IsArrowDataset(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return arrow_datasets_.find(name) != arrow_datasets_.end();
}

} // namespace cyxwiz
