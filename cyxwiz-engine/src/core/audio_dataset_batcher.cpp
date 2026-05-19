#include "audio_dataset_batcher.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <thread>
#include <utility>

namespace cyxwiz {

AudioDatasetBatcher::AudioDatasetBatcher(
    const DataRegistry::AudioDatasetEntry& entry,
    const AudioPreprocessingConfig& preprocess_config,
    int batch_size,
    float train_split,
    bool shuffle,
    int num_workers)
    : batch_size_(batch_size), shuffle_(shuffle),
      num_workers_(std::max(0, num_workers)), rng_(42)
{
    // Build the AudioDatasetConfig. Start from the entry's dialog-baked
    // defaults, then let graph preprocessing nodes override them. The
    // registry entry stores feature_type as int (0/1/2) so the dialog
    // and JSON serialization can round-trip without depending on the
    // enum class — convert here.
    AudioDatasetConfig cfg;
    switch (entry.feature_type) {
        case 0: cfg.feature_type = AudioDatasetConfig::FeatureType::Spectrogram; break;
        case 2: cfg.feature_type = AudioDatasetConfig::FeatureType::MFCC; break;
        case 1:
        default: cfg.feature_type = AudioDatasetConfig::FeatureType::MelSpectrogram; break;
    }
    cfg.target_sr      = entry.target_sr;
    cfg.n_fft          = entry.n_fft;
    cfg.hop_length     = entry.hop_length;
    cfg.n_mels         = entry.n_mels;
    cfg.fmin           = entry.fmin;
    cfg.fmax           = entry.fmax;
    cfg.n_mfcc         = entry.n_mfcc;
    cfg.max_duration   = entry.max_duration;
    cfg.labeled_subdirs = entry.labeled_subdirs;
    cfg.csv_path       = entry.csv_path;       // FlatWithCSV mode if non-empty
    cfg.filename_col   = entry.filename_col;
    cfg.label_col      = entry.label_col;

    // Apply graph-node overrides (Phase 2.1). If the user dropped a
    // Spectrogram / MelSpectrogram / MFCC node on the canvas, the
    // graph compiler's extractors have already written to
    // preprocess_config.has_feature_node and the params. Override
    // the dialog defaults here. If no feature node was present,
    // has_feature_node is false and we keep the entry's defaults.
    if (preprocess_config.has_feature_node) {
        switch (preprocess_config.feature_type) {
            case AudioFeatureType::Spectrogram:
                cfg.feature_type = AudioDatasetConfig::FeatureType::Spectrogram; break;
            case AudioFeatureType::MelSpectrogram:
                cfg.feature_type = AudioDatasetConfig::FeatureType::MelSpectrogram; break;
            case AudioFeatureType::MFCC:
                cfg.feature_type = AudioDatasetConfig::FeatureType::MFCC; break;
            case AudioFeatureType::None:
            default:
                break;
        }
        cfg.n_fft      = preprocess_config.n_fft;
        cfg.hop_length = preprocess_config.hop_length;
        cfg.n_mels     = preprocess_config.n_mels;
        cfg.fmin       = preprocess_config.fmin;
        cfg.fmax       = preprocess_config.fmax;
        cfg.n_mfcc     = preprocess_config.n_mfcc;
        spdlog::info("AudioDatasetBatcher: graph feature config overrides dialog "
                     "defaults (feature_type={}, n_fft={}, n_mels={}, n_mfcc={})",
                     static_cast<int>(preprocess_config.feature_type),
                     cfg.n_fft, cfg.n_mels, cfg.n_mfcc);
    }

    // Augmentation overrides (applied regardless of feature node mode
    // — augmentation is orthogonal to feature type).
    if (preprocess_config.has_augmentation) {
        cfg.noise_level  = preprocess_config.noise_level;
        cfg.time_stretch = preprocess_config.time_stretch;
        cfg.pitch_shift  = preprocess_config.pitch_shift;
        spdlog::info("AudioDatasetBatcher: graph augmentation enabled "
                     "(noise_level={}, time_stretch={}, pitch_shift={})",
                     cfg.noise_level, cfg.time_stretch, cfg.pitch_shift);
    }

    try {
        dataset_ = std::make_shared<AudioDataset>(entry.folder_path, cfg);
    } catch (const std::exception& e) {
        spdlog::error("AudioDatasetBatcher: failed to construct AudioDataset: {}", e.what());
        return;
    }

    if (!dataset_ || dataset_->Size() == 0) {
        spdlog::error("AudioDatasetBatcher: dataset is empty or null");
        return;
    }

    // Probe one sample to discover the feature shape. Cheaper than
    // recomputing from config and avoids a config/extraction mismatch.
    auto info = dataset_->GetInfo();
    if (info.shape.size() >= 2) {
        feature_rows_ = static_cast<int>(info.shape[0]);
        feature_cols_ = static_cast<int>(info.shape[1]);
    } else {
        // Fallback: recompute from config
        int max_samples = static_cast<int>(cfg.target_sr * cfg.max_duration);
        feature_cols_ = (max_samples + cfg.n_fft) / cfg.hop_length;
        switch (cfg.feature_type) {
            case AudioDatasetConfig::FeatureType::Spectrogram:
                feature_rows_ = cfg.n_fft / 2 + 1; break;
            case AudioDatasetConfig::FeatureType::MelSpectrogram:
                feature_rows_ = cfg.n_mels; break;
            case AudioDatasetConfig::FeatureType::MFCC:
                feature_rows_ = cfg.n_mfcc; break;
        }
    }

    // Shuffled train/val split.
    //
    // Previously: first 80% of file_paths_ was train, last 20% was val.
    // Since AudioDataset::ScanDirectory iterates class subdirs alphabetically
    // and appends all files per class, the sequential split made the val set
    // heavily imbalanced (for a 2-class binary: train = class_a + some class_b,
    // val = only class_b). A model that learned to always predict class_b
    // scored 100% val accuracy with zero real skill — that's where the
    // suspicious "100% val on epoch 1" came from.
    //
    // Fix: shuffle the full index list first, THEN split. Random sampling
    // approximately balances classes in expectation. A true stratified
    // split (shuffle within each class then split) would be even better
    // but this is sufficient to eliminate the pathological bias.
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
        num_classes_ = info.num_classes;
    }

    Reset();
    spdlog::info("AudioDatasetBatcher: {} train / {} val samples, "
                 "{} classes, feature shape [{} x {}], batch_size={}, num_workers={}",
                 train_indices_.size(), val_indices_.size(), num_classes_,
                 feature_rows_, feature_cols_, batch_size_, num_workers_);
}

Batch AudioDatasetBatcher::GetNextBatch() {
    Batch batch;
    if (!dataset_ || current_idx_ >= epoch_order_.size()) return batch;

    size_t actual_size = std::min(static_cast<size_t>(batch_size_),
                                   epoch_order_.size() - current_idx_);
    if (actual_size == 0) return batch;

    size_t sample_dim = static_cast<size_t>(feature_rows_) *
                        static_cast<size_t>(feature_cols_);

    std::vector<float> batch_data;
    batch_data.reserve(actual_size * sample_dim);
    std::vector<float> batch_labels;

    if (do_onehot_ && num_classes_ > 0) {
        batch_labels.resize(actual_size * num_classes_, 0.0f);
    } else {
        batch_labels.reserve(actual_size);
    }

    std::vector<std::pair<std::vector<float>, int>> samples(actual_size);

    auto load_range = [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            size_t idx = epoch_order_[current_idx_ + i];
            samples[i] = dataset_->GetItem(idx);
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
        auto& [features, label] = samples[i];

        if (features.size() != sample_dim) {
            // Bad sample (decode/extract failed): zero-fill so the batch
            // still has uniform shape. AudioDataset already logs the warning.
            features.assign(sample_dim, 0.0f);
            label = 0;
        }

        // Always apply per-sample z-score normalization for audio.
        // Mel spectrogram values live in the dB range (-80 to ~0) and
        // would saturate any Dense head if fed raw — softmax pins to one
        // class on the first batch. Z-score makes each sample mean=0 std=1,
        // which Xavier-init linear layers expect. This is universal best
        // practice in audio ML and removes a major footgun: the user's
        // Normalize node with default mean=0/std=1 is otherwise a no-op
        // and they wonder why training collapses.
        {
            double sum = 0.0;
            double sumsq = 0.0;
            for (float v : features) { sum += v; sumsq += v * v; }
            double mean = sum / static_cast<double>(features.size());
            double var = sumsq / static_cast<double>(features.size()) - mean * mean;
            float stdev = (var > 1e-8) ? static_cast<float>(std::sqrt(var)) : 1.0f;
            float fmean = static_cast<float>(mean);
            for (float& v : features) v = (v - fmean) / stdev;
        }

        // Optional explicit user normalization (rarely useful for audio
        // since z-score above already centered/scaled, but kept for
        // backward compatibility with the IBatcher interface).
        if (do_normalize_) {
            for (float& v : features) {
                v = (v - norm_mean_) / norm_std_;
            }
        }

        batch_data.insert(batch_data.end(), features.begin(), features.end());

        if (do_onehot_ && num_classes_ > 0) {
            if (label >= 0 && static_cast<size_t>(label) < num_classes_) {
                batch_labels[i * num_classes_ + label] = 1.0f;
            }
        } else {
            batch_labels.push_back(static_cast<float>(label));
        }
    }

    if (flatten_) {
        batch.data = Tensor({actual_size, sample_dim},
                            batch_data.data(), DataType::Float32);
    } else {
        batch.data = Tensor({actual_size,
                             static_cast<size_t>(feature_rows_),
                             static_cast<size_t>(feature_cols_)},
                            batch_data.data(), DataType::Float32);
    }

    if (do_onehot_ && num_classes_ > 0) {
        batch.labels = Tensor({actual_size, num_classes_},
                              batch_labels.data(), DataType::Float32);
    } else {
        batch.labels = Tensor({actual_size},
                              batch_labels.data(), DataType::Float32);
    }

    batch.size = actual_size;
    current_idx_ += actual_size;
    return batch;
}

void AudioDatasetBatcher::Reset() {
    // Pick the active index set based on current phase. For validation
    // we never shuffle — deterministic iteration order makes val metrics
    // reproducible across epochs.
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

void AudioDatasetBatcher::SetPhase(BatcherPhase phase) {
    current_phase_ = phase;
    // Caller is expected to call Reset() afterwards so the new epoch
    // order picks up. The training executor's RunValidationArrow does
    // exactly that.
}

bool AudioDatasetBatcher::IsEpochComplete() const {
    return current_idx_ >= epoch_order_.size();
}

size_t AudioDatasetBatcher::GetNumBatches() const {
    if (batch_size_ <= 0) return 0;
    return (epoch_order_.size() + batch_size_ - 1) / batch_size_;
}

size_t AudioDatasetBatcher::GetNumSamples() const {
    return train_indices_.size();
}

void AudioDatasetBatcher::SetNormalization(float mean, float std_dev) {
    norm_mean_ = mean;
    norm_std_ = (std_dev > 0.0f) ? std_dev : 1.0f;
    do_normalize_ = true;
}

void AudioDatasetBatcher::SetOneHotEncoding(size_t num_classes) {
    num_classes_ = num_classes;
    do_onehot_ = true;
}

void AudioDatasetBatcher::SetFlatten(bool flatten) {
    flatten_ = flatten;
}

} // namespace cyxwiz
