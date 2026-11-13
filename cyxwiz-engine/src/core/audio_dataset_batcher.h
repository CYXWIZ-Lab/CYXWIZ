#pragma once

#include "dataset_batcher.h"
#include "sample_split.h"
#include "data_registry.h"
#include "graph_compiler.h"
#include "formats/audio_dataset.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

namespace cyxwiz {

inline constexpr const char* kAudioFeaturesMissingMessage =
    "audio needs a Mel Spectrogram, Spectrogram or MFCC node after the Data Loader";

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
 * The clips split once into train, validation and test (SplitSamples, the
 * rule every training path uses; TOFIX178 1.5); the phase picks the part a
 * pass reads, and only train batches are shuffled.
 */
class AudioDatasetBatcher : public IBatcher {
public:
    // The entry gives the loading settings; preprocess_config (filled by the
    // Spectrogram/MelSpectrogram/MFCC/AudioAugmentation extractors in
    // GraphCompiler) gives the features. Without a feature node the batcher
    // refuses: it holds no dataset and GetNumSamples() is 0.
    AudioDatasetBatcher(const DataRegistry::AudioDatasetEntry& entry,
                        const AudioPreprocessingConfig& preprocess_config,
                        int batch_size,
                        const SampleSplitSettings& split = {},
                        bool shuffle = true,
                        int num_workers = 0,
                        uint32_t shuffle_seed = 42);

    // IBatcher interface
    Batch GetNextBatch() override;
    void Reset() override;
    bool IsEpochComplete() const override;
    size_t GetNumBatches() const override;
    size_t GetNumSamples() const override;
    // Clips by number, as validation sees them: no augmentation (TOFIX178
    // 6.0b, Local Debug's picked sample).
    std::optional<Batch> GetSourceSamples(const std::vector<size_t>& indices) override;
    size_t SourceSampleCount() const override;
    // The part (of the three this batcher holds) the clip falls in.
    std::optional<BatcherPhase> SourceSamplePart(size_t index) const override;
    std::string SourceSampleName(size_t index) const override;
    std::vector<std::string> ClassNames() const override;

    void SetNormalization(float mean, float std_dev) override;
    void SetOneHotEncoding(size_t num_classes) override;
    void SetScalarLabelMode(bool enable) override { scalar_label_mode_ = enable; }
    void SetFlatten(bool flatten) override;
    void SetDropLast(bool drop_last) override { drop_last_ = drop_last; }
    void SetPhase(BatcherPhase phase) override;

    // Audio-specific accessors
    int GetFeatureRows() const { return feature_rows_; }
    int GetFeatureCols() const { return feature_cols_; }
    // The clips in a part (Train, Val or Test), whatever the current phase.
    size_t SamplesIn(BatcherPhase phase) const { return Part(phase).size(); }
    const std::vector<size_t>& Part(BatcherPhase phase) const;

private:
    // These clips as one batch: the features, per-clip z-score, Normalize and
    // the label form; augmented when `augment`.
    Batch AssembleBatch(const size_t* indices, size_t n, bool augment);

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

    // The three parts; SetPhase picks one, Reset() copies it into
    // epoch_order_ for iteration.
    std::vector<size_t> train_indices_;
    std::vector<size_t> val_indices_;
    std::vector<size_t> test_indices_;
    BatcherPhase current_phase_ = BatcherPhase::Train;

    std::vector<size_t> epoch_order_;
    size_t current_idx_ = 0;

    std::mt19937 rng_;
    uint32_t augmentation_round_ = 0;  // training epochs begun (augmentation draws)
};

// The Data Split's settings for the training dataset's audio batcher.
SampleSplitSettings AudioTrainingSplit(const TrainingConfiguration& config);
// The audio batcher training reads for `entry` (TrainingManager, Local
// Debug): the graph's feature node, `split`'s parts, drop-last, Normalize and
// the label form.
std::shared_ptr<AudioDatasetBatcher> MakeAudioTrainingBatcher(const DataRegistry::AudioDatasetEntry& entry,
                                                              const TrainingConfiguration& config, int batch_size,
                                                              const SampleSplitSettings& split);
// The model input the batcher's features give: rows x cols flattened, the
// [bins, frames] shape and, for a Conv1D, [frames, bins].
void ApplyAudioInputShape(const AudioDatasetBatcher& batcher, TrainingConfiguration& config);

} // namespace cyxwiz
