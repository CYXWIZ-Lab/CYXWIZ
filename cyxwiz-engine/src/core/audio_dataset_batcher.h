#pragma once

#include "dataset_batcher.h"
#include "data_registry.h"
#include "graph_compiler.h"
#include "formats/audio_dataset.h"
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace cyxwiz {

/**
 * AudioDatasetBatcher — IBatcher implementation for audio datasets.
 *
 * Wraps an AudioDataset (which lazily decodes wav/flac/etc. and runs
 * the feature extraction per GetItem call) and stacks samples into
 * fixed-size batches. The output tensor shape is:
 *
 *   default:  [batch_size, feature_rows, feature_cols]    (3D)
 *   flatten:  [batch_size, feature_rows * feature_cols]   (2D, for Dense head)
 *
 * Where feature_rows / feature_cols come from the feature type:
 *   Spectrogram:    n_fft/2 + 1   ×   n_frames
 *   MelSpectrogram: n_mels        ×   n_frames
 *   MFCC:           n_mfcc        ×   n_frames
 *
 * Uses the same train/val split-by-fraction and shuffle pattern as
 * ImageDatasetBatcher. No augmentation pipeline is plumbed yet — that
 * comes when AudioAugmentation node extractors land in graph_compiler.
 */
class AudioDatasetBatcher : public IBatcher {
public:
    // preprocess_config is populated by the audio-domain extractors in
    // GraphCompiler (Spectrogram/MelSpectrogram/MFCC/AudioAugmentation
    // nodes). When preprocess_config.has_feature_node is false, the
    // batcher falls back to the dialog-baked feature config stored on
    // the AudioDatasetEntry — the "no node required" UX.
    AudioDatasetBatcher(const DataRegistry::AudioDatasetEntry& entry,
                        const AudioPreprocessingConfig& preprocess_config,
                        int batch_size,
                        float train_split = 0.8f,
                        bool shuffle = true,
                        int num_workers = 0,
                        uint32_t seed = 42);

    // IBatcher interface
    Batch GetNextBatch() override;
    void Reset() override;
    bool IsEpochComplete() const override;
    size_t GetNumBatches() const override;
    size_t GetNumSamples() const override;

    void SetNormalization(float mean, float std_dev) override;
    void SetOneHotEncoding(size_t num_classes) override;
    void SetScalarLabelMode(bool enable) override { scalar_label_mode_ = enable; }
    void SetFlatten(bool flatten) override;
    void SetDropLast(bool drop_last) override { drop_last_ = drop_last; }
    void SetPhase(BatcherPhase phase) override;

    // Audio-specific accessors
    int GetFeatureRows() const { return feature_rows_; }
    int GetFeatureCols() const { return feature_cols_; }
    size_t GetNumValSamples() const { return val_indices_.size(); }

private:
    std::shared_ptr<AudioDataset> dataset_;

    int batch_size_;
    bool shuffle_;
    int num_workers_ = 0;
    bool drop_last_ = false;
    bool flatten_ = true;   // default flatten — most audio classifiers use Dense head

    int feature_rows_ = 0;  // n_fft/2+1, n_mels, or n_mfcc
    int feature_cols_ = 0;  // time frames

    float norm_mean_ = 0.0f;
    float norm_std_ = 1.0f;
    bool do_normalize_ = false;

    size_t num_classes_ = 0;
    bool do_onehot_ = false;
    bool scalar_label_mode_ = false;

    // Separate train/val index sets (shuffled split at construction).
    // SetPhase toggles current_phase_; Reset() copies the appropriate
    // set into epoch_order_ for iteration.
    std::vector<size_t> train_indices_;
    std::vector<size_t> val_indices_;
    BatcherPhase current_phase_ = BatcherPhase::Train;

    std::vector<size_t> epoch_order_;
    size_t current_idx_ = 0;

    std::mt19937 rng_;
};

} // namespace cyxwiz
