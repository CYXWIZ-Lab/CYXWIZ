#pragma once

#include "../dataset_base.h"
#include "../data_registry.h"
#include <vector>
#include <string>
#include <random>

namespace cyxwiz {

/**
 * Kaggle dataset loader
 *
 * Supports loading Kaggle datasets with three strategies:
 * 1. From cached binary file (fast)
 * 2. From downloaded files in cache directory (CSV or image folders)
 * 3. Predefined synthetic datasets (Titanic, Iris, Fashion-MNIST, Digits)
 *
 * Features:
 * - Binary caching for fast reload
 * - CSV parsing with auto label encoding
 * - Image folder loading (ImageNet structure)
 * - Built-in synthetic datasets for testing
 * - Auto train/val/test splitting
 */
class KaggleDataset : public Dataset {
public:
    explicit KaggleDataset(const KaggleConfig& config);

    // Dataset interface
    size_t Size() const override;
    std::pair<std::vector<float>, int> GetItem(size_t index) const override;
    DatasetInfo GetInfo() const override;

private:
    // Initialization
    void Initialize();
    std::string GetDatasetName() const;
    std::string GetCacheDir() const;

    // Loading strategies
    bool LoadFromCache(const std::string& cache_file);
    void SaveToCache(const std::string& cache_file);
    bool LoadFromDownloadedFiles(const std::string& cache_dir);
    bool LoadCSVFile(const std::string& path);
    bool LoadImageFolder(const std::string& path);

    // Predefined datasets
    bool LoadPredefinedDataset(const std::string& name);
    bool CreateTitanicDataset();
    bool CreateIrisDataset();
    bool CreateFashionMNISTDataset();
    bool CreateDigitsDataset();

    // Split management
    void SetupSplits();
    size_t CalculateMemoryUsage() const;

    // Member variables
    KaggleConfig config_;
    std::vector<std::vector<float>> samples_;
    std::vector<int> labels_;
    std::vector<size_t> shape_;
    size_t num_classes_ = 0;
    std::vector<std::string> class_names_;
};

} // namespace cyxwiz
