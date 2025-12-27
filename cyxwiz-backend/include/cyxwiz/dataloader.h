#pragma once

#include "api_export.h"
#include "tensor.h"
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <functional>
#include <fstream>

namespace cyxwiz {

/**
 * A batch of training data
 */
struct CYXWIZ_API DataBatch {
    Tensor data;          // [batch_size, ...] - input features
    Tensor labels;        // [batch_size] or [batch_size, num_classes] if one-hot
    size_t size = 0;      // Actual batch size (may be smaller for last batch)

    bool IsValid() const { return size > 0; }
};

/**
 * Dataset split enumeration
 */
enum class Split {
    Train,
    Test,
    Validation
};

/**
 * Abstract base class for datasets
 */
class CYXWIZ_API DatasetBase {
public:
    virtual ~DatasetBase() = default;

    // Get the number of samples
    virtual size_t Size() const = 0;

    // Get a single sample (data, label)
    virtual std::pair<std::vector<float>, int> GetItem(size_t index) const = 0;

    // Get sample shape (e.g., {28, 28} for MNIST)
    virtual std::vector<size_t> GetShape() const = 0;

    // Get number of classes
    virtual size_t NumClasses() const = 0;

    // Get class names (optional)
    virtual std::vector<std::string> GetClassNames() const { return {}; }
};

/**
 * MNIST Dataset - Loads MNIST handwritten digit dataset
 *
 * File format: IDX file format (standard MNIST files)
 * - train-images-idx3-ubyte / train-images.idx3-ubyte
 * - train-labels-idx1-ubyte / train-labels.idx1-ubyte
 * - t10k-images-idx3-ubyte / t10k-images.idx3-ubyte
 * - t10k-labels-idx1-ubyte / t10k-labels.idx1-ubyte
 */
class CYXWIZ_API MNISTDataset : public DatasetBase {
public:
    /**
     * Create MNIST dataset
     * @param path Directory containing MNIST files
     * @param split Train or Test split
     * @param normalize If true, normalize pixel values to [0, 1]
     * @param flatten If true, flatten images to 784 elements
     */
    MNISTDataset(const std::string& path, Split split = Split::Train,
                 bool normalize = true, bool flatten = true);

    size_t Size() const override { return images_.size() / sample_size_; }
    std::pair<std::vector<float>, int> GetItem(size_t index) const override;
    std::vector<size_t> GetShape() const override;
    size_t NumClasses() const override { return 10; }
    std::vector<std::string> GetClassNames() const override;

    // MNIST-specific accessors
    int GetWidth() const { return 28; }
    int GetHeight() const { return 28; }
    bool IsFlattened() const { return flatten_; }
    bool IsNormalized() const { return normalize_; }

    // Static method to download MNIST if not present
    static bool Download(const std::string& path, bool verbose = true);

private:
    std::vector<float> images_;      // All image data (flattened)
    std::vector<int> labels_;        // All labels
    size_t sample_size_;             // Size of each sample in floats
    bool normalize_;
    bool flatten_;

    // Load IDX file format
    bool LoadImages(const std::string& filepath);
    bool LoadLabels(const std::string& filepath);

    // Find MNIST files in directory
    std::string FindImageFile(const std::string& path, Split split);
    std::string FindLabelFile(const std::string& path, Split split);
};

/**
 * FashionMNIST Dataset - Same format as MNIST, different images
 */
class CYXWIZ_API FashionMNISTDataset : public MNISTDataset {
public:
    FashionMNISTDataset(const std::string& path, Split split = Split::Train,
                        bool normalize = true, bool flatten = true)
        : MNISTDataset(path, split, normalize, flatten) {}

    std::vector<std::string> GetClassNames() const override {
        return {"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"};
    }
};

/**
 * Synthetic Dataset - Generate random data for testing
 */
class CYXWIZ_API SyntheticDataset : public DatasetBase {
public:
    /**
     * Create synthetic dataset
     * @param num_samples Number of samples to generate
     * @param input_shape Shape of each sample (e.g., {784} or {28, 28})
     * @param num_classes Number of classes (labels will be random in [0, num_classes))
     * @param seed Random seed for reproducibility
     */
    SyntheticDataset(size_t num_samples, const std::vector<size_t>& input_shape,
                     size_t num_classes = 10, int seed = 42);

    size_t Size() const override { return num_samples_; }
    std::pair<std::vector<float>, int> GetItem(size_t index) const override;
    std::vector<size_t> GetShape() const override { return shape_; }
    size_t NumClasses() const override { return num_classes_; }

private:
    size_t num_samples_;
    std::vector<size_t> shape_;
    size_t num_classes_;
    mutable std::mt19937 rng_;
    size_t sample_size_;
};

/**
 * DataLoader - Iterates over a dataset in batches
 *
 * Usage:
 *   auto dataset = std::make_shared<MNISTDataset>("./data", Split::Train);
 *   DataLoader loader(dataset, 32, true);
 *
 *   for (int epoch = 0; epoch < num_epochs; epoch++) {
 *       loader.Reset();
 *       while (!loader.IsEpochComplete()) {
 *           DataBatch batch = loader.GetNextBatch();
 *           // Train on batch...
 *       }
 *   }
 */
class CYXWIZ_API DataLoader {
public:
    /**
     * Create a DataLoader
     * @param dataset Shared pointer to dataset
     * @param batch_size Number of samples per batch
     * @param shuffle Whether to shuffle data each epoch
     * @param drop_last Drop last batch if smaller than batch_size
     * @param seed Random seed for shuffling
     */
    DataLoader(std::shared_ptr<DatasetBase> dataset,
               size_t batch_size,
               bool shuffle = true,
               bool drop_last = false,
               int seed = 42);

    /**
     * Get the next batch
     * @return DataBatch with data and labels tensors
     */
    DataBatch GetNextBatch();

    /**
     * Reset to beginning of epoch (re-shuffles if shuffle=true)
     */
    void Reset();

    /**
     * Check if current epoch is complete
     */
    bool IsEpochComplete() const;

    /**
     * Get total number of batches per epoch
     */
    size_t NumBatches() const;

    /**
     * Get current batch index (0-based)
     */
    size_t GetCurrentBatch() const { return current_batch_; }

    /**
     * Get total number of samples
     */
    size_t NumSamples() const { return indices_.size(); }

    /**
     * Get batch size
     */
    size_t GetBatchSize() const { return batch_size_; }

    // Preprocessing options
    void SetOneHotEncoding(bool enabled, size_t num_classes = 0);
    void SetNormalization(float mean, float std);

    // Dataset accessors
    std::shared_ptr<DatasetBase> GetDataset() const { return dataset_; }
    std::vector<size_t> GetShape() const { return dataset_->GetShape(); }
    size_t NumClasses() const { return dataset_->NumClasses(); }

private:
    std::shared_ptr<DatasetBase> dataset_;
    size_t batch_size_;
    bool shuffle_;
    bool drop_last_;

    std::vector<size_t> indices_;     // Sample indices (shuffled)
    size_t current_index_ = 0;        // Current position in indices_
    size_t current_batch_ = 0;        // Current batch number

    std::mt19937 rng_;

    // Preprocessing
    bool one_hot_ = false;
    size_t one_hot_classes_ = 0;
    bool normalize_ = false;
    float norm_mean_ = 0.0f;
    float norm_std_ = 1.0f;

    // Shuffle indices
    void ShuffleIndices();

    // Convert vector to Tensor
    Tensor VectorToTensor(const std::vector<float>& data,
                          const std::vector<size_t>& shape);

    // Convert labels to Tensor
    Tensor LabelsToTensor(const std::vector<int>& labels);

    // Convert labels to one-hot Tensor
    Tensor LabelsToOneHot(const std::vector<int>& labels);
};

/**
 * Helper function to load MNIST quickly
 * @param path Directory containing MNIST files
 * @param batch_size Batch size for the loader
 * @param train If true, load training set, else test set
 * @return DataLoader ready for iteration
 */
CYXWIZ_API DataLoader LoadMNIST(const std::string& path,
                                 size_t batch_size = 32,
                                 bool train = true,
                                 bool shuffle = true);

/**
 * Create a simple dataset from numpy-style arrays
 * @param data Input data tensor [num_samples, ...]
 * @param labels Label tensor [num_samples]
 * @param batch_size Batch size for the loader
 * @param shuffle Whether to shuffle
 * @return DataLoader ready for iteration
 */
CYXWIZ_API DataLoader CreateDataLoader(const Tensor& data,
                                        const Tensor& labels,
                                        size_t batch_size = 32,
                                        bool shuffle = true);

} // namespace cyxwiz
