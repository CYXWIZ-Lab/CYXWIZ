#include "image_dataset_batcher.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>

namespace cyxwiz {

ImageDatasetBatcher::ImageDatasetBatcher(
    const DataRegistry::ImageDatasetEntry& entry,
    const ImagePreprocessingConfig& preprocess_config,
    int batch_size,
    float train_split,
    bool shuffle)
    : batch_size_(batch_size), shuffle_(shuffle), rng_(42)
{
    // Extract target dimensions from the Resize config. If no Resize node
    // was in the graph, fall back to 224x224 which is the most common
    // default for image classification.
    if (preprocess_config.resize_mode != ResizeMode::None &&
        preprocess_config.target_width > 0 && preprocess_config.target_height > 0) {
        target_width_ = preprocess_config.target_width;
        target_height_ = preprocess_config.target_height;
    }
    // No else — if no Resize node is in the graph, the compile gate
    // (Phase 1.4) should have caught it as an error. The member defaults
    // (224x224) only apply as a last-resort fallback.

    channels_ = preprocess_config.convert_to_grayscale ? 1 : 3;

    // Create the underlying dataset with the target size baked in. This
    // avoids double resize: the dataset decodes+resizes in one pass via
    // ImageUtils::LoadImage + ResizeImage. The ImageTransform below then
    // only handles augmentation / blur / enhancement — NOT resize.
    if (entry.layout == 1 && !entry.labels_csv.empty()) {
        auto csv_ds = std::make_shared<ImageCSVDataset>(
            entry.folder_path, entry.labels_csv,
            target_width_, target_height_, 200);
        dataset_ = csv_ds;
        spdlog::info("ImageDatasetBatcher: created ImageCSVDataset {}x{}, {} samples",
                     target_width_, target_height_, csv_ds->Size());
    } else {
        auto folder_ds = std::make_shared<ImageFolderDataset>(entry.folder_path);
        dataset_ = folder_ds;
        spdlog::info("ImageDatasetBatcher: created ImageFolderDataset {}x{}, {} samples",
                     target_width_, target_height_, folder_ds->Size());
    }

    if (!dataset_ || dataset_->Size() == 0) {
        spdlog::error("ImageDatasetBatcher: dataset is empty or null");
        return;
    }

    // Build the ImageTransform for augmentation / blur / enhancement.
    // Set resize_mode to None since the dataset already resized to target.
    ImagePreprocessingConfig aug_config = preprocess_config;
    aug_config.resize_mode = ResizeMode::None;
    aug_config.target_width = 0;
    aug_config.target_height = 0;
    transform_ = std::make_unique<ImageTransform>(aug_config);

    // Split indices: first train_split fraction for training.
    size_t total = dataset_->Size();
    size_t train_count = static_cast<size_t>(total * train_split);
    if (train_count == 0) train_count = total;

    train_indices_.resize(train_count);
    std::iota(train_indices_.begin(), train_indices_.end(), 0);

    num_classes_ = entry.num_classes;
    if (num_classes_ == 0) {
        auto info = dataset_->GetInfo();
        num_classes_ = info.num_classes;
    }

    Reset();
    spdlog::info("ImageDatasetBatcher: {} train samples, {} classes, batch_size={}",
                 train_indices_.size(), num_classes_, batch_size_);
}

Batch ImageDatasetBatcher::GetNextBatch() {
    Batch batch;
    if (!dataset_ || current_idx_ >= epoch_order_.size()) return batch;

    size_t actual_size = std::min(static_cast<size_t>(batch_size_),
                                   epoch_order_.size() - current_idx_);
    if (actual_size == 0) return batch;

    size_t sample_dim = static_cast<size_t>(target_width_) *
                        static_cast<size_t>(target_height_) *
                        static_cast<size_t>(channels_);

    std::vector<float> batch_data;
    batch_data.reserve(actual_size * sample_dim);
    std::vector<float> batch_labels;

    if (do_onehot_ && num_classes_ > 0) {
        batch_labels.resize(actual_size * num_classes_, 0.0f);
    } else {
        batch_labels.reserve(actual_size);
    }

    for (size_t i = 0; i < actual_size; ++i) {
        size_t idx = epoch_order_[current_idx_ + i];
        auto [pixels, label] = dataset_->GetItem(idx);

        if (pixels.empty()) {
            // Fill with zeros on bad samples
            pixels.resize(sample_dim, 0.0f);
            label = 0;
        }

        // Apply augmentation transforms (resize already done by dataset)
        if (transform_) {
            std::vector<size_t> shape = {
                static_cast<size_t>(target_height_),
                static_cast<size_t>(target_width_),
                static_cast<size_t>(channels_)};
            Tensor img_tensor(shape, pixels.data(), DataType::Float32);
            Tensor transformed = transform_->Apply(img_tensor);
            const float* tdata = transformed.Data<float>();
            size_t tcount = transformed.NumElements();
            pixels.assign(tdata, tdata + tcount);
        }

        // Normalize if requested
        if (do_normalize_) {
            for (float& v : pixels) {
                v = (v - norm_mean_) / norm_std_;
            }
        }

        // Flatten is default for image → Dense head
        batch_data.insert(batch_data.end(), pixels.begin(), pixels.end());

        if (do_onehot_ && num_classes_ > 0) {
            if (label >= 0 && static_cast<size_t>(label) < num_classes_) {
                batch_labels[i * num_classes_ + label] = 1.0f;
            }
        } else {
            batch_labels.push_back(static_cast<float>(label));
        }
    }

    size_t feature_dim = flatten_ ? sample_dim : sample_dim;
    batch.data = Tensor({actual_size, feature_dim}, batch_data.data(), DataType::Float32);

    if (do_onehot_ && num_classes_ > 0) {
        batch.labels = Tensor({actual_size, num_classes_}, batch_labels.data(), DataType::Float32);
    } else {
        batch.labels = Tensor({actual_size}, batch_labels.data(), DataType::Float32);
    }

    batch.size = actual_size;
    current_idx_ += actual_size;

    return batch;
}

void ImageDatasetBatcher::Reset() {
    epoch_order_ = train_indices_;
    if (shuffle_) {
        std::shuffle(epoch_order_.begin(), epoch_order_.end(), rng_);
    }
    current_idx_ = 0;
}

bool ImageDatasetBatcher::IsEpochComplete() const {
    return current_idx_ >= epoch_order_.size();
}

size_t ImageDatasetBatcher::GetNumBatches() const {
    if (batch_size_ <= 0) return 0;
    return (epoch_order_.size() + batch_size_ - 1) / batch_size_;
}

size_t ImageDatasetBatcher::GetNumSamples() const {
    return train_indices_.size();
}

void ImageDatasetBatcher::SetNormalization(float mean, float std_dev) {
    norm_mean_ = mean;
    norm_std_ = (std_dev > 0.0f) ? std_dev : 1.0f;
    do_normalize_ = true;
}

void ImageDatasetBatcher::SetOneHotEncoding(size_t num_classes) {
    num_classes_ = num_classes;
    do_onehot_ = true;
}

void ImageDatasetBatcher::SetFlatten(bool flatten) {
    flatten_ = flatten;
}

} // namespace cyxwiz
