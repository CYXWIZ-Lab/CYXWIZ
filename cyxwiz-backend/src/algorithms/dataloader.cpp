#include "cyxwiz/dataloader.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <filesystem>
#include <cstring>

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

// ============================================================================
// SyntheticDataset Implementation
// ============================================================================

SyntheticDataset::SyntheticDataset(size_t num_samples, const std::vector<size_t>& input_shape,
                                   size_t num_classes, int seed)
    : num_samples_(num_samples), shape_(input_shape), num_classes_(num_classes), rng_(seed)
{
    sample_size_ = 1;
    for (size_t dim : shape_) {
        sample_size_ *= dim;
    }
}

std::pair<std::vector<float>, int> SyntheticDataset::GetItem(size_t index) const {
    // Generate deterministic random data based on index
    std::mt19937 local_rng(static_cast<unsigned>(index));
    std::uniform_real_distribution<float> data_dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> label_dist(0, static_cast<int>(num_classes_) - 1);

    std::vector<float> data(sample_size_);
    for (size_t i = 0; i < sample_size_; i++) {
        data[i] = data_dist(local_rng);
    }

    int label = label_dist(local_rng);
    return {data, label};
}

// ============================================================================
// DataLoader Implementation
// ============================================================================

DataLoader::DataLoader(std::shared_ptr<DatasetBase> dataset,
                       size_t batch_size,
                       bool shuffle,
                       bool drop_last,
                       int seed)
    : dataset_(std::move(dataset))
    , batch_size_(batch_size)
    , shuffle_(shuffle)
    , drop_last_(drop_last)
    , rng_(seed)
{
    // Initialize indices
    size_t num_samples = dataset_->Size();
    indices_.resize(num_samples);
    for (size_t i = 0; i < num_samples; i++) {
        indices_[i] = i;
    }

    if (shuffle_) {
        ShuffleIndices();
    }
}

DataBatch DataLoader::GetNextBatch() {
    DataBatch batch;

    if (IsEpochComplete()) {
        return batch;
    }

    // Determine actual batch size
    size_t remaining = indices_.size() - current_index_;
    size_t actual_batch_size = std::min(batch_size_, remaining);

    if (drop_last_ && actual_batch_size < batch_size_) {
        return batch;
    }

    // Collect samples
    std::vector<std::vector<float>> batch_data;
    std::vector<int> batch_labels;
    batch_data.reserve(actual_batch_size);
    batch_labels.reserve(actual_batch_size);

    for (size_t i = 0; i < actual_batch_size; i++) {
        size_t idx = indices_[current_index_ + i];
        auto [data, label] = dataset_->GetItem(idx);

        // Apply normalization if enabled
        if (normalize_) {
            for (float& val : data) {
                val = (val - norm_mean_) / norm_std_;
            }
        }

        batch_data.push_back(std::move(data));
        batch_labels.push_back(label);
    }

    current_index_ += actual_batch_size;
    current_batch_++;

    // Build data tensor
    std::vector<size_t> data_shape = {actual_batch_size};
    auto sample_shape = dataset_->GetShape();
    data_shape.insert(data_shape.end(), sample_shape.begin(), sample_shape.end());

    // Flatten all data into single vector
    size_t sample_size = 1;
    for (size_t dim : sample_shape) {
        sample_size *= dim;
    }

    std::vector<float> flat_data;
    flat_data.reserve(actual_batch_size * sample_size);
    for (const auto& sample : batch_data) {
        flat_data.insert(flat_data.end(), sample.begin(), sample.end());
    }

    batch.data = VectorToTensor(flat_data, data_shape);

    // Build labels tensor
    if (one_hot_) {
        batch.labels = LabelsToOneHot(batch_labels);
    } else {
        batch.labels = LabelsToTensor(batch_labels);
    }

    batch.size = actual_batch_size;
    return batch;
}

void DataLoader::Reset() {
    current_index_ = 0;
    current_batch_ = 0;

    if (shuffle_) {
        ShuffleIndices();
    }
}

bool DataLoader::IsEpochComplete() const {
    if (drop_last_) {
        return current_index_ + batch_size_ > indices_.size();
    }
    return current_index_ >= indices_.size();
}

size_t DataLoader::NumBatches() const {
    if (drop_last_) {
        return indices_.size() / batch_size_;
    }
    return (indices_.size() + batch_size_ - 1) / batch_size_;
}

void DataLoader::SetOneHotEncoding(bool enabled, size_t num_classes) {
    one_hot_ = enabled;
    one_hot_classes_ = (num_classes > 0) ? num_classes : dataset_->NumClasses();
}

void DataLoader::SetNormalization(float mean, float std) {
    normalize_ = true;
    norm_mean_ = mean;
    norm_std_ = std;
}

void DataLoader::ShuffleIndices() {
    std::shuffle(indices_.begin(), indices_.end(), rng_);
}

Tensor DataLoader::VectorToTensor(const std::vector<float>& data,
                                  const std::vector<size_t>& shape) {
    return Tensor(shape, data.data(), DataType::Float32);
}

Tensor DataLoader::LabelsToTensor(const std::vector<int>& labels) {
    std::vector<size_t> shape = {labels.size()};

    // Convert int to int32_t
    std::vector<int32_t> labels_i32(labels.begin(), labels.end());
    return Tensor(shape, labels_i32.data(), DataType::Int32);
}

Tensor DataLoader::LabelsToOneHot(const std::vector<int>& labels) {
    size_t num_samples = labels.size();
    std::vector<size_t> shape = {num_samples, one_hot_classes_};

    std::vector<float> one_hot(num_samples * one_hot_classes_, 0.0f);
    for (size_t i = 0; i < num_samples; i++) {
        int label = labels[i];
        if (label >= 0 && static_cast<size_t>(label) < one_hot_classes_) {
            one_hot[i * one_hot_classes_ + label] = 1.0f;
        }
    }

    return Tensor(shape, one_hot.data(), DataType::Float32);
}

// ============================================================================
// Helper Functions
// ============================================================================

DataLoader LoadMNIST(const std::string& path,
                     size_t batch_size,
                     bool train,
                     bool shuffle) {
    Split split = train ? Split::Train : Split::Test;
    auto dataset = std::make_shared<MNISTDataset>(path, split, true, true);
    return DataLoader(dataset, batch_size, shuffle);
}

DataLoader CreateDataLoader(const Tensor& data,
                            const Tensor& labels,
                            size_t batch_size,
                            bool shuffle) {
    // Create a wrapper dataset from tensors
    class TensorDataset : public DatasetBase {
    public:
        TensorDataset(const Tensor& data, const Tensor& labels)
            : data_(data.Clone()), labels_(labels.Clone()) {
            const auto& shape = data.Shape();
            num_samples_ = shape[0];
            sample_size_ = data.NumElements() / num_samples_;

            // Store shape without batch dimension
            for (size_t i = 1; i < shape.size(); i++) {
                sample_shape_.push_back(shape[i]);
            }
            if (sample_shape_.empty()) {
                sample_shape_.push_back(sample_size_);
            }
        }

        size_t Size() const override { return num_samples_; }

        std::pair<std::vector<float>, int> GetItem(size_t index) const override {
            const float* data_ptr = data_.Data<float>();
            const int32_t* label_ptr = labels_.Data<int32_t>();

            size_t offset = index * sample_size_;
            std::vector<float> sample(data_ptr + offset, data_ptr + offset + sample_size_);
            int label = static_cast<int>(label_ptr[index]);

            return {sample, label};
        }

        std::vector<size_t> GetShape() const override { return sample_shape_; }
        size_t NumClasses() const override { return 10; }  // Default

    private:
        Tensor data_;
        Tensor labels_;
        size_t num_samples_;
        size_t sample_size_;
        std::vector<size_t> sample_shape_;
    };

    auto dataset = std::make_shared<TensorDataset>(data, labels);
    return DataLoader(dataset, batch_size, shuffle);
}

} // namespace cyxwiz
