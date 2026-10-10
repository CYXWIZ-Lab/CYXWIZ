#include "image_dataset_batcher.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>
#include <thread>
#include <utility>

namespace cyxwiz {

ImageDatasetBatcher::ImageDatasetBatcher(
    const DataRegistry::ImageDatasetEntry& entry,
    const ImagePreprocessingConfig& preprocess_config,
    int batch_size,
    float train_split,
    bool shuffle,
    int num_workers,
    uint32_t seed)
    : batch_size_(batch_size), shuffle_(shuffle),
      num_workers_(std::max(0, num_workers)),
      augmentation_rng_(seed ^ 0x9E3779B9u), rng_(seed)
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

    decoded_shape_ = {static_cast<size_t>(target_height_), static_cast<size_t>(target_width_), 3};
    sample_shape_ = decoded_shape_;

    // Create the underlying dataset with the target size baked in, so the
    // dataset decodes and resizes in one pass (ImageUtils::LoadImage +
    // ResizeImage). The image transforms run on the device afterwards.
    if (entry.layout == 1 && !entry.labels_csv.empty()) {
        auto csv_ds = std::make_shared<ImageCSVDataset>(
            entry.folder_path, entry.labels_csv,
            target_width_, target_height_, 200);
        dataset_ = csv_ds;
        spdlog::info("ImageDatasetBatcher: created ImageCSVDataset {}x{}, {} samples",
                     target_width_, target_height_, csv_ds->Size());
    } else {
        auto folder_ds = std::make_shared<ImageFolderDataset>(
            entry.folder_path, target_width_, target_height_);
        dataset_ = folder_ds;
        spdlog::info("ImageDatasetBatcher: created ImageFolderDataset {}x{}, {} samples",
                     target_width_, target_height_, folder_ds->Size());
    }

    if (!dataset_ || dataset_->Size() == 0) {
        spdlog::error("ImageDatasetBatcher: dataset is empty or null");
        return;
    }

    // Shuffled train/val split (see audio_dataset_batcher.cpp for the
    // full rationale). Sequential split was leaking class-imbalanced
    // val sets that made the reported val metrics meaningless.
    size_t total = dataset_->Size();
    size_t train_count = static_cast<size_t>(total * train_split);
    if (train_count == 0) train_count = total;
    if (train_count > total) train_count = total;

    std::vector<size_t> all_indices(total);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::shuffle(all_indices.begin(), all_indices.end(), rng_);

    train_indices_.assign(all_indices.begin(), all_indices.begin() + train_count);
    val_indices_.assign(all_indices.begin() + train_count, all_indices.end());

    num_classes_ = entry.num_classes;
    if (num_classes_ == 0) {
        auto info = dataset_->GetInfo();
        num_classes_ = info.num_classes;
    }

    Reset();
    spdlog::info("ImageDatasetBatcher: {} train / {} val samples, {} classes, batch_size={}, num_workers={}",
                 train_indices_.size(), val_indices_.size(), num_classes_, batch_size_, num_workers_);
}

Batch ImageDatasetBatcher::GetNextBatch() {
    Batch batch;
    if (!dataset_ || IsEpochComplete()) return batch;

    size_t actual_size = std::min(static_cast<size_t>(batch_size_),
                                   epoch_order_.size() - current_idx_);
    if (actual_size == 0) return batch;

    const size_t sample_dim = decoded_shape_.Size();

    std::vector<float> batch_data(actual_size * sample_dim, 0.0f);
    std::vector<float> batch_labels;

    if (!scalar_label_mode_ && do_onehot_ && num_classes_ > 0) {
        batch_labels.resize(actual_size * num_classes_, 0.0f);
    } else {
        batch_labels.reserve(actual_size);
    }

    // Decode on the CPU workers straight into the batch rows; a sample that
    // fails to load stays zeros with label 0.
    std::vector<int> labels(actual_size, 0);
    auto load_range = [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            const size_t idx = epoch_order_[current_idx_ + i];
            auto [pixels, label] = dataset_->GetItem(idx);
            if (pixels.size() == sample_dim) {
                std::copy(pixels.begin(), pixels.end(), batch_data.begin() + i * sample_dim);
                labels[i] = label;
            }
        }
    };

    if (num_workers_ > 1 && actual_size > 1) {
        size_t worker_count = std::min(static_cast<size_t>(num_workers_), actual_size);
        size_t chunk_size = (actual_size + worker_count - 1) / worker_count;
        std::vector<std::thread> workers;
        workers.reserve(worker_count);

        for (size_t worker = 0; worker < worker_count; ++worker) {
            size_t begin = worker * chunk_size;
            size_t end = std::min(actual_size, begin + chunk_size);
            if (begin >= end) break;
            workers.emplace_back(load_range, begin, end);
        }

        for (auto& worker : workers) {
            worker.join();
        }
    } else {
        load_range(0, actual_size);
    }

    for (size_t i = 0; i < actual_size; ++i) {
        const int label = labels[i];
        if (!scalar_label_mode_ && do_onehot_ && num_classes_ > 0) {
            if (label >= 0 && static_cast<size_t>(label) < num_classes_) {
                batch_labels[i * num_classes_ + label] = 1.0f;
            }
        } else {
            batch_labels.push_back(static_cast<float>(label));
        }
    }

    const bool onehot = !scalar_label_mode_ && do_onehot_ && num_classes_ > 0;
    if (onehot) {
        batch.labels = Tensor({actual_size, num_classes_}, batch_labels.data(), DataType::Float32);
    } else if (scalar_label_mode_) {
        batch.labels = Tensor({actual_size, 1}, batch_labels.data(), DataType::Float32);
    } else {
        batch.labels = Tensor({actual_size}, batch_labels.data(), DataType::Float32);
    }

    // One upload, then the transforms, the batch mix (one-hot labels mixed
    // alongside) and Normalize on the device; the batch stays there for the model.
    Tensor rows({actual_size, sample_dim}, batch_data.data(), DataType::Float32);
    if (!augmentation_.Empty()) {
        rows = augmentation_.Apply(rows, decoded_shape_, current_phase_ == BatcherPhase::Train,
                                   augmentation_rng_, onehot ? &batch.labels : nullptr);
    }
    batch.data = flatten_
        ? rows
        : rows.Reshape({actual_size, sample_shape_.height, sample_shape_.width, sample_shape_.channels});

    batch.size = actual_size;
    current_idx_ += actual_size;

    return batch;
}

void ImageDatasetBatcher::Reset() {
    // Active index set depends on the current phase. Val never shuffles.
    if (current_phase_ == BatcherPhase::Val) {
        epoch_order_ = val_indices_;
    } else {
        epoch_order_ = train_indices_;
        if (shuffle_) {
            std::shuffle(epoch_order_.begin(), epoch_order_.end(), rng_);
        }
    }
    current_idx_ = 0;
}

void ImageDatasetBatcher::SetPhase(BatcherPhase phase) {
    current_phase_ = phase;
    // Caller should Reset() afterwards; training_executor's
    // RunValidationArrow does that via batcher.Reset().
}

bool ImageDatasetBatcher::IsEpochComplete() const {
    if (current_idx_ >= epoch_order_.size()) return true;
    return drop_last_ && batch_size_ > 0 &&
           current_phase_ == BatcherPhase::Train &&
           epoch_order_.size() - current_idx_ < static_cast<size_t>(batch_size_);
}

size_t ImageDatasetBatcher::GetNumBatches() const {
    if (batch_size_ <= 0) return 0;
    if (drop_last_ && current_phase_ == BatcherPhase::Train) {
        return epoch_order_.size() / static_cast<size_t>(batch_size_);
    }
    return (epoch_order_.size() + batch_size_ - 1) / batch_size_;
}

size_t ImageDatasetBatcher::GetNumSamples() const {
    return train_indices_.size();
}

void ImageDatasetBatcher::SetNormalization(float mean, float std_dev) {
    augmentation_.normalize = true;
    augmentation_.mean = mean;
    augmentation_.std_dev = (std_dev > 0.0f) ? std_dev : 1.0f;
}

void ImageDatasetBatcher::SetImageTransforms(const image::ImageAugmentation& compiled) {
    augmentation_.ops = compiled.ops;
    augmentation_.mix = compiled.mix;
    augmentation_.mix_alpha = compiled.mix_alpha;
    augmentation_.mix_probability = compiled.mix_probability;
    sample_shape_ = augmentation_.ShapeAfter(decoded_shape_);
}

void ImageDatasetBatcher::SetOneHotEncoding(size_t num_classes) {
    num_classes_ = num_classes;
    do_onehot_ = true;
}

void ImageDatasetBatcher::SetFlatten(bool flatten) {
    flatten_ = flatten;
}

} // namespace cyxwiz
