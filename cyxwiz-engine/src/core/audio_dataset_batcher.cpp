#include "audio_dataset_batcher.h"
#include "batch_uploads.h"
#include "classification_decision.h"
#include "class_label_range.h"
#include "cyxwiz/portable_random.h"
#include <cyxwiz/error_codes.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <exception>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <filesystem>
#include <string>
#include <thread>
#include <utility>

namespace cyxwiz {

AudioDatasetBatcher::AudioDatasetBatcher(
    const DataRegistry::AudioDatasetEntry& entry,
    const AudioPreprocessingConfig& preprocess_config,
    int batch_size,
    const SampleSplitSettings& split,
    bool shuffle,
    int num_workers,
    uint32_t shuffle_seed)
    : batch_size_(batch_size), shuffle_(shuffle),
      num_workers_(std::max(0, num_workers)), rng_(shuffle_seed)
{
    // Loading settings come from the Data Input entry; the features come only
    // from the graph's Spectrogram / Mel Spectrogram / MFCC node (TOFIX129 B-9).
    if (!preprocess_config.has_feature_node ||
        preprocess_config.feature_type == AudioFeatureType::None) {
        spdlog::error("AudioDatasetBatcher: [{}] {}", errors::Compiler::AudioFeaturesMissing,
                      kAudioFeaturesMissingMessage);
        return;
    }

    AudioDatasetConfig cfg;
    switch (preprocess_config.feature_type) {
        case AudioFeatureType::Spectrogram:
            cfg.feature_type = AudioDatasetConfig::FeatureType::Spectrogram; break;
        case AudioFeatureType::MFCC:
            cfg.feature_type = AudioDatasetConfig::FeatureType::MFCC; break;
        case AudioFeatureType::MelSpectrogram:
        default:
            cfg.feature_type = AudioDatasetConfig::FeatureType::MelSpectrogram; break;
    }
    cfg.n_fft          = preprocess_config.n_fft;
    cfg.hop_length     = preprocess_config.hop_length;
    cfg.n_mels         = preprocess_config.n_mels;
    cfg.fmin           = preprocess_config.fmin;
    cfg.fmax           = preprocess_config.fmax;
    cfg.n_mfcc         = preprocess_config.n_mfcc;
    cfg.target_sr      = entry.target_sr;
    cfg.max_duration   = entry.max_duration;
    cfg.labeled_subdirs = entry.labeled_subdirs;
    cfg.csv_path       = entry.csv_path;       // FlatWithCSV mode if non-empty
    cfg.filename_col   = entry.filename_col;
    cfg.label_col      = entry.label_col;
    spdlog::info("AudioDatasetBatcher: features from the graph node (feature_type={}, n_fft={}, "
                 "hop_length={}, n_mels={}, n_mfcc={})",
                 static_cast<int>(preprocess_config.feature_type),
                 cfg.n_fft, cfg.hop_length, cfg.n_mels, cfg.n_mfcc);

    // Augmentation is orthogonal to the feature type.
    if (preprocess_config.has_augmentation) {
        cfg.noise_level  = preprocess_config.noise_level;
        cfg.time_stretch = preprocess_config.time_stretch;
        cfg.pitch_shift  = preprocess_config.pitch_shift;
        cfg.augmentation_seed = shuffle_seed ^ 0x9E3779B9u;  // as image augmentation
        spdlog::info("AudioDatasetBatcher: graph augmentation enabled "
                     "(noise_level={}, time_stretch={}, pitch_shift={})",
                     cfg.noise_level, cfg.time_stretch, cfg.pitch_shift);
    }

    try {
        dataset_ = std::make_shared<AudioDataset>(entry.folder_path, cfg);
    } catch (const std::exception& e) {
        spdlog::error("AudioDatasetBatcher: failed to construct AudioDataset: {}", e.what());
        load_error_ = e.what();
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

    // Train / validation / test by the shared rule (by class when
    // stratified); the split has its own seed, apart from the batch order.
    std::vector<size_t> all_indices(dataset_->Size());
    std::iota(all_indices.begin(), all_indices.end(), 0);
    auto parts = SplitSamples(all_indices, dataset_->Labels(), split);
    train_indices_ = std::move(parts.train);
    val_indices_ = std::move(parts.val);
    test_indices_ = std::move(parts.test);

    num_classes_ = entry.num_classes;
    if (num_classes_ == 0) {
        num_classes_ = info.num_classes;
    }

    Reset();
    spdlog::info("AudioDatasetBatcher: {} train / {} validation / {} test samples, "
                 "{} classes, feature shape [{} x {}], batch_size={}, num_workers={}",
                 train_indices_.size(), val_indices_.size(), test_indices_.size(), num_classes_,
                 feature_rows_, feature_cols_, batch_size_, num_workers_);
}

Batch AudioDatasetBatcher::GetNextBatch() {
    if (!dataset_ || IsEpochComplete()) return {};
    const size_t actual_size = std::min(static_cast<size_t>(batch_size_), epoch_order_.size() - current_idx_);
    if (actual_size == 0) return {};
    // The AudioAugmentation node's random noise, stretch and pitch shift are
    // for training clips only: validation and test read each clip as it is.
    Batch batch = AssembleBatch(epoch_order_.data() + current_idx_, actual_size,
                                current_phase_ == BatcherPhase::Train);
    current_idx_ += actual_size;
    batch.last = IsEpochComplete();
    return batch;
}

size_t AudioDatasetBatcher::SourceSampleCount() const {
    return dataset_ ? dataset_->Size() : 0;
}

std::optional<Batch> AudioDatasetBatcher::GetSourceSamples(const std::vector<size_t>& indices) {
    if (!dataset_ || indices.empty()) return std::nullopt;
    for (size_t index : indices)
        if (index >= dataset_->Size())
            throw std::out_of_range("clip " + std::to_string(index) + " is outside the " +
                                    std::to_string(dataset_->Size()) + " clips");
    // Items by number come back all, in order, or not at all: a batch that
    // left out an unreadable clip would pair the next clips' rows with the
    // names and numbers the caller asked for.
    Batch batch = AssembleBatch(indices.data(), indices.size(), false);
    if (batch.size != indices.size()) {
        for (const size_t index : indices)
            if (std::find(batch.row_ids.begin(), batch.row_ids.end(), static_cast<int64_t>(index)) ==
                batch.row_ids.end())
                throw std::runtime_error("clip " + std::to_string(index) + " ('" + SourceSampleName(index) +
                                         "') could not be read (see the log)");
    }
    return batch;
}

std::optional<BatcherPhase> AudioDatasetBatcher::SourceSamplePart(size_t index) const {
    for (const BatcherPhase phase : {BatcherPhase::Train, BatcherPhase::Val, BatcherPhase::Test}) {
        const auto& part = Part(phase);
        if (std::find(part.begin(), part.end(), index) != part.end()) return phase;
    }
    return std::nullopt;
}

std::string AudioDatasetBatcher::SourceSampleName(size_t index) const {
    return index < dataset_->Size() ? std::filesystem::path(dataset_->GetFilePath(index)).filename().string()
                                     : std::string();
}

std::optional<std::vector<size_t>> AudioDatasetBatcher::TrainClassCounts() const {
    if (!dataset_) return std::nullopt;
    const auto& labels = dataset_->Labels();
    std::vector<size_t> counts;
    for (const size_t index : Part(BatcherPhase::Train))
        if (index < labels.size()) CountClass(counts, labels[index]);
    return counts;
}

std::vector<std::string> AudioDatasetBatcher::ClassNames() const { return dataset_->GetClassNames(); }

Batch AudioDatasetBatcher::AssembleBatch(const size_t* indices, size_t actual_size, bool augment) {
    Batch batch;

    size_t sample_dim = static_cast<size_t>(feature_rows_) *
                        static_cast<size_t>(feature_cols_);

    std::vector<float> batch_data;
    batch_data.reserve(actual_size * sample_dim);
    std::vector<float> batch_labels;

    if (!scalar_label_mode_ && do_onehot_ && num_classes_ > 0) {
        batch_labels.resize(actual_size * num_classes_, 0.0f);
    } else {
        batch_labels.reserve(actual_size);
    }

    std::vector<std::pair<std::vector<float>, int>> samples(actual_size);

    // A worker's exception is carried to this thread and rethrown after the
    // join: thrown inside a std::thread it would end the process
    // (TOFIX223 0.12).
    std::mutex error_mutex;
    std::exception_ptr worker_error;
    auto load_range = [&](size_t begin, size_t end) {
        try {
            for (size_t i = begin; i < end; ++i) {
                size_t idx = indices[i];
                samples[i] = dataset_->GetItem(idx, augment);
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!worker_error) worker_error = std::current_exception();
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
    if (worker_error) std::rethrow_exception(worker_error);

    // A clip that gave no features of the expected size (decode failure,
    // silence, a different length) is left out of the batch: a zero-filled
    // sample with its label, or class 0, would be trained on (TOFIX223
    // 0.12). AudioDataset logs each one.
    std::vector<size_t> kept;
    kept.reserve(actual_size);
    for (size_t i = 0; i < actual_size; ++i) {
        if (samples[i].first.size() == sample_dim) {
            kept.push_back(i);
        } else if (!samples[i].first.empty()) {
            spdlog::warn("AudioDatasetBatcher: leaving out '{}': {} feature values, the set's shape needs {}",
                         dataset_->GetFilePath(indices[i]), samples[i].first.size(), sample_dim);
        }
    }
    if (kept.empty()) {
        throw std::runtime_error("AudioDatasetBatcher: none of the " + std::to_string(actual_size) +
                                 " clips of this batch could be read (see the log for each file)");
    }
    if (kept.size() < actual_size && !scalar_label_mode_ && do_onehot_ && num_classes_ > 0) {
        batch_labels.assign(kept.size() * num_classes_, 0.0f);
    }

    for (size_t k = 0; k < kept.size(); ++k) {
        const size_t i = kept[k];
        auto& [features, label] = samples[i];

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

        if (!scalar_label_mode_ && do_onehot_ && num_classes_ > 0) {
            CheckClassLabelInWidth("AudioDatasetBatcher", label, num_classes_);
            batch_labels[k * num_classes_ + static_cast<size_t>(label)] = 1.0f;
        } else {
            batch_labels.push_back(static_cast<float>(label));
        }
    }

    const size_t used = kept.size();
    if (flatten_) {
        batch.data = HostBatchTensor("AudioDatasetBatcher", "features", {used, sample_dim}, batch_data.data());
    } else {
        batch.data = HostBatchTensor("AudioDatasetBatcher", "features",
                                     {used, static_cast<size_t>(feature_rows_), static_cast<size_t>(feature_cols_)},
                                     batch_data.data());
    }

    if (!scalar_label_mode_ && do_onehot_ && num_classes_ > 0) {
        batch.labels = HostBatchTensor("AudioDatasetBatcher", "labels", {used, num_classes_}, batch_labels.data());
    } else if (scalar_label_mode_) {
        batch.labels = HostBatchTensor("AudioDatasetBatcher", "labels", {used, 1}, batch_labels.data());
    } else {
        batch.labels = HostBatchTensor("AudioDatasetBatcher", "labels", {used}, batch_labels.data());
    }

    // Clip numbers in the dataset: test results name the clips with them.
    batch.row_ids.clear();
    for (const size_t i : kept) batch.row_ids.push_back(static_cast<int64_t>(indices[i]));
    batch.size = used;
    return batch;
}

void AudioDatasetBatcher::Reset() {
    // The current phase's part; only train batches are shuffled, so
    // validation and test metrics are reproducible across epochs.
    epoch_order_ = Part(current_phase_);
    if (current_phase_ == BatcherPhase::Train && pending_epoch_seed_) {
        rng_.seed(static_cast<std::mt19937::result_type>(*pending_epoch_seed_));
        // The clips' augmentation draws from the epoch's seed too, so a
        // resumed run replays them (a running round count could not)
        augmentation_round_ = static_cast<uint32_t>(*pending_epoch_seed_ ^ (*pending_epoch_seed_ >> 32));
        pending_epoch_seed_.reset();
    } else if (current_phase_ == BatcherPhase::Train) {
        ++augmentation_round_;
    }
    if (current_phase_ == BatcherPhase::Train && dataset_) dataset_->SetAugmentationRound(augmentation_round_);
    if (current_phase_ == BatcherPhase::Train && shuffle_) cyxwiz::portable::Shuffle(epoch_order_.begin(), epoch_order_.end(), rng_);
    current_idx_ = 0;
}

void AudioDatasetBatcher::SetPhase(BatcherPhase phase) {
    current_phase_ = phase;
    // Caller is expected to call Reset() afterwards so the new epoch
    // order picks up. The training executor's Evaluate does exactly that.
}

bool AudioDatasetBatcher::IsEpochComplete() const {
    if (current_idx_ >= epoch_order_.size()) return true;
    return drop_last_ && batch_size_ > 0 &&
           current_phase_ == BatcherPhase::Train &&
           epoch_order_.size() - current_idx_ < static_cast<size_t>(batch_size_);
}

size_t AudioDatasetBatcher::GetNumBatches() const {
    if (batch_size_ <= 0) return 0;
    if (drop_last_ && current_phase_ == BatcherPhase::Train) {
        return epoch_order_.size() / static_cast<size_t>(batch_size_);
    }
    return (epoch_order_.size() + batch_size_ - 1) / batch_size_;
}

size_t AudioDatasetBatcher::GetNumSamples() const {
    return Part(current_phase_).size();
}

const std::vector<size_t>& AudioDatasetBatcher::Part(BatcherPhase phase) const {
    return phase == BatcherPhase::Val ? val_indices_ : phase == BatcherPhase::Test ? test_indices_ : train_indices_;
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

SampleSplitSettings AudioTrainingSplit(const TrainingConfiguration& config) {
    return {config.train_ratio, config.val_ratio, config.test_ratio, config.stratified,
            static_cast<uint32_t>(config.split_seed)};
}

std::shared_ptr<AudioDatasetBatcher> MakeAudioTrainingBatcher(const DataRegistry::AudioDatasetEntry& entry,
                                                              const TrainingConfiguration& config, int batch_size,
                                                              const SampleSplitSettings& split) {
    auto batcher = std::make_shared<AudioDatasetBatcher>(entry, config.audio_preprocessing, batch_size, split,
                                                         config.shuffle, config.num_workers,
                                                         static_cast<uint32_t>(config.dataloader_seed));
    batcher->SetDropLast(config.drop_last);
    if (config.preprocessing.has_normalization)
        batcher->SetNormalization(config.preprocessing.norm_mean, config.preprocessing.norm_std);
    if (UsesScalarBinaryTargets(config.loss_type)) {
        batcher->SetScalarLabelMode(true);
    } else if (config.preprocessing.has_onehot && config.preprocessing.num_classes > 0) {
        batcher->SetOneHotEncoding(config.preprocessing.num_classes);
    }
    return batcher;
}

void ApplyAudioInputShape(const AudioDatasetBatcher& batcher, TrainingConfiguration& config) {
    const auto rows = static_cast<size_t>(batcher.GetFeatureRows());
    const auto cols = static_cast<size_t>(batcher.GetFeatureCols());
    config.input_size = rows * cols;
    config.input_shape = {rows, cols};
    // [rows, cols] = [bins, frames] per sample, torch [N, C, L]: a Conv1D
    // convolves over the frames with one channel per bin.
    config.sequence_input_shape = {cols, rows};
}

} // namespace cyxwiz
