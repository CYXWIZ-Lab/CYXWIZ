#pragma once

#include "../data_registry.h"
#include <cyxwiz/audio_processing.h>
#include <string>
#include <vector>

namespace cyxwiz {

/**
 * Configuration for loading audio data as an ML dataset
 */
struct AudioDatasetConfig {
    // Feature extraction mode
    enum class FeatureType { Spectrogram, MelSpectrogram, MFCC };
    FeatureType feature_type = FeatureType::MelSpectrogram;

    // Audio loading
    int target_sr = 16000;        // Target sample rate

    // Spectrogram params
    int n_fft = 512;
    int hop_length = 256;

    // Mel params
    int n_mels = 128;
    float fmin = 0.0f;
    float fmax = 0.0f;

    // MFCC params
    int n_mfcc = 13;

    // Dataset structure
    bool labeled_subdirs = true;  // Subdirectories = class labels
    float max_duration = 5.0f;    // Max audio duration in seconds (pad/truncate)
};

/**
 * AudioDataset - Loads a directory of audio files and extracts
 * spectrogram/mel/MFCC features for ML training.
 *
 * Directory structure (labeled_subdirs=true):
 *   audio_dir/
 *     class_0/  file1.wav, file2.wav, ...
 *     class_1/  file1.wav, file2.wav, ...
 *
 * Or flat directory (labeled_subdirs=false):
 *   audio_dir/  file1.wav, file2.wav, ...
 */
class AudioDataset : public Dataset {
public:
    /**
     * Load audio dataset from directory
     * @param directory Path to audio directory
     * @param config Configuration for feature extraction
     */
    AudioDataset(const std::string& directory, const AudioDatasetConfig& config);

    // Dataset interface
    size_t Size() const override;
    std::pair<std::vector<float>, int> GetItem(size_t index) const override;
    DatasetInfo GetInfo() const override;

    // Class label info
    const std::vector<std::string>& GetClassNames() const { return class_names_; }
    int GetNumClasses() const { return static_cast<int>(class_names_.size()); }

private:
    AudioDatasetConfig config_;
    std::vector<std::string> file_paths_;
    std::vector<int> labels_;
    std::vector<std::string> class_names_;
    int feature_rows_ = 0;       // Rows in feature map
    int feature_cols_ = 0;       // Cols in feature map (time frames)

    void ScanDirectory(const std::string& directory);
    AudioFeatures ExtractFeatures(const std::string& filepath) const;
};

} // namespace cyxwiz
