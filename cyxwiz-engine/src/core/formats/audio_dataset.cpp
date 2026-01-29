#include "audio_dataset.h"
#include <filesystem>
#include <algorithm>
#include <stdexcept>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

namespace cyxwiz {

AudioDataset::AudioDataset(const std::string& directory, const AudioDatasetConfig& config)
    : config_(config) {
    ScanDirectory(directory);

    // Compute expected feature dimensions from config
    int max_samples = static_cast<int>(config_.target_sr * config_.max_duration);
    int n_frames = (max_samples + config_.n_fft) / config_.hop_length;  // Approximate

    switch (config_.feature_type) {
        case AudioDatasetConfig::FeatureType::Spectrogram:
            feature_rows_ = config_.n_fft / 2 + 1;
            break;
        case AudioDatasetConfig::FeatureType::MelSpectrogram:
            feature_rows_ = config_.n_mels;
            break;
        case AudioDatasetConfig::FeatureType::MFCC:
            feature_rows_ = config_.n_mfcc;
            break;
    }
    feature_cols_ = n_frames;
}

void AudioDataset::ScanDirectory(const std::string& directory) {
    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        throw std::runtime_error("AudioDataset: directory not found: " + directory);
    }

    // Supported extensions
    auto is_audio = [](const std::string& ext) {
        return ext == ".wav" || ext == ".flac" || ext == ".ogg" || ext == ".aiff" || ext == ".aif";
    };

    if (config_.labeled_subdirs) {
        // Each subdirectory is a class
        std::vector<std::string> subdirs;
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_directory()) {
                subdirs.push_back(entry.path().filename().string());
            }
        }
        std::sort(subdirs.begin(), subdirs.end());
        class_names_ = subdirs;

        for (int cls = 0; cls < static_cast<int>(subdirs.size()); cls++) {
            fs::path cls_dir = fs::path(directory) / subdirs[cls];
            for (const auto& entry : fs::recursive_directory_iterator(cls_dir)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (is_audio(ext)) {
                        file_paths_.push_back(entry.path().string());
                        labels_.push_back(cls);
                    }
                }
            }
        }
    } else {
        // Flat directory, no labels
        class_names_ = {"audio"};
        for (const auto& entry : fs::recursive_directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (is_audio(ext)) {
                    file_paths_.push_back(entry.path().string());
                    labels_.push_back(0);
                }
            }
        }
    }

    spdlog::info("AudioDataset: found {} audio files in {} classes from '{}'",
                 file_paths_.size(), class_names_.size(), directory);

    if (file_paths_.empty()) {
        throw std::runtime_error("AudioDataset: no audio files found in: " + directory);
    }

    // Initialize split indices
    all_indices_.resize(file_paths_.size());
    for (size_t i = 0; i < file_paths_.size(); i++) all_indices_[i] = i;
}

AudioFeatures AudioDataset::ExtractFeatures(const std::string& filepath) const {
    // Load audio
    AudioData audio = AudioProcessing::LoadAudio(filepath, config_.target_sr);
    if (!audio.valid) {
        AudioFeatures empty;
        empty.error_message = audio.error_message;
        return empty;
    }

    // Pad or truncate to max_duration
    size_t max_samples = static_cast<size_t>(config_.target_sr * config_.max_duration);
    if (audio.num_samples > max_samples) {
        audio.samples.resize(max_samples);
        audio.num_samples = max_samples;
    } else if (audio.num_samples < max_samples) {
        audio.samples.resize(max_samples, 0.0f);  // Zero-pad
        audio.num_samples = max_samples;
    }
    audio.duration_seconds = config_.max_duration;

    // Extract features based on config
    switch (config_.feature_type) {
        case AudioDatasetConfig::FeatureType::Spectrogram: {
            SpectrogramConfig sc;
            sc.n_fft = config_.n_fft;
            sc.hop_length = config_.hop_length;
            return AudioProcessing::ComputeSpectrogram(audio, sc);
        }
        case AudioDatasetConfig::FeatureType::MelSpectrogram: {
            MelConfig mc;
            mc.n_fft = config_.n_fft;
            mc.hop_length = config_.hop_length;
            mc.n_mels = config_.n_mels;
            mc.fmin = config_.fmin;
            mc.fmax = config_.fmax;
            return AudioProcessing::ComputeMelSpectrogram(audio, mc);
        }
        case AudioDatasetConfig::FeatureType::MFCC: {
            MFCCConfig cc;
            cc.n_fft = config_.n_fft;
            cc.hop_length = config_.hop_length;
            cc.n_mels = config_.n_mels;
            cc.fmin = config_.fmin;
            cc.fmax = config_.fmax;
            cc.n_mfcc = config_.n_mfcc;
            return AudioProcessing::ComputeMFCC(audio, cc);
        }
    }

    AudioFeatures empty;
    empty.error_message = "Unknown feature type";
    return empty;
}

size_t AudioDataset::Size() const {
    return file_paths_.size();
}

std::pair<std::vector<float>, int> AudioDataset::GetItem(size_t index) const {
    if (index >= file_paths_.size()) {
        return {{}, -1};
    }

    AudioFeatures features = ExtractFeatures(file_paths_[index]);
    if (!features.valid) {
        spdlog::warn("AudioDataset: failed to extract features for '{}': {}",
                     file_paths_[index], features.error_message);
        // Return zeros as fallback
        return {std::vector<float>(feature_rows_ * feature_cols_, 0.0f), labels_[index]};
    }

    return {features.data, labels_[index]};
}

DatasetInfo AudioDataset::GetInfo() const {
    DatasetInfo info;
    info.name = "AudioDataset";
    info.num_samples = file_paths_.size();
    info.shape = {static_cast<size_t>(feature_rows_), static_cast<size_t>(feature_cols_)};
    info.num_classes = class_names_.size();
    info.class_names = class_names_;
    return info;
}

} // namespace cyxwiz
