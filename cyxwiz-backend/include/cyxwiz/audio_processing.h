#pragma once

#include "api_export.h"
#include <vector>
#include <string>
#include <cstdint>

namespace cyxwiz {

// ============================================================================
// Audio Data Structures
// ============================================================================

/**
 * Raw audio data in memory
 */
struct CYXWIZ_API AudioData {
    std::vector<float> samples;       // Mono audio samples (normalized to [-1, 1])
    int sample_rate = 16000;          // Sample rate in Hz
    int original_sample_rate = 0;     // Original sample rate before resampling
    int channels = 1;                 // Number of channels (stored as mono)
    size_t num_samples = 0;           // Number of samples
    float duration_seconds = 0.0f;    // Duration in seconds
    std::string filepath;             // Source file path
    bool valid = false;
    std::string error_message;
};

/**
 * Spectrogram configuration
 */
struct CYXWIZ_API SpectrogramConfig {
    int n_fft = 512;                  // FFT window size
    int hop_length = 256;             // Hop between windows (default: n_fft / 2)
    int win_length = 0;              // Window length (default: n_fft)
    bool center = true;               // Pad signal to center frames
    // Window type: 0 = Hann, 1 = Hamming, 2 = Rectangular
    int window_type = 0;
};

/**
 * Mel spectrogram configuration (extends SpectrogramConfig)
 */
struct CYXWIZ_API MelConfig : public SpectrogramConfig {
    int n_mels = 128;                 // Number of mel filterbanks
    float fmin = 0.0f;               // Minimum frequency for mel
    float fmax = 0.0f;               // Maximum frequency (0 = sample_rate/2)
};

/**
 * MFCC configuration
 */
struct CYXWIZ_API MFCCConfig : public MelConfig {
    int n_mfcc = 13;                  // Number of MFCCs to return
    bool use_energy = true;           // Replace first coefficient with log energy
};

/**
 * 2D feature map result (e.g., spectrogram, mel spectrogram, MFCC)
 */
struct CYXWIZ_API AudioFeatures {
    std::vector<float> data;          // Flattened [rows, cols] row-major
    int rows = 0;                     // Frequency bins (or n_mels / n_mfcc)
    int cols = 0;                     // Time frames
    bool valid = false;
    std::string error_message;
};

// ============================================================================
// Audio Processing Class
// Uses libsndfile for I/O and FFTW3 for FFT
// ============================================================================

class CYXWIZ_API AudioProcessing {
public:
    // ==================== Loading ====================

    /**
     * Load audio from file (WAV, FLAC, OGG, AIFF via libsndfile)
     * @param filepath Path to audio file
     * @param target_sr Target sample rate (resamples if different, 0 = keep original)
     * @return AudioData with samples
     */
    static AudioData LoadAudio(const std::string& filepath, int target_sr = 16000);

    // ==================== Feature Extraction ====================

    /**
     * Compute magnitude spectrogram (STFT) using FFTW3
     * @param audio Audio data
     * @param config Spectrogram configuration
     * @return AudioFeatures [n_fft/2 + 1, time_frames]
     */
    static AudioFeatures ComputeSpectrogram(const AudioData& audio,
                                            const SpectrogramConfig& config = {});

    /**
     * Compute mel spectrogram
     * @param audio Audio data
     * @param config Mel spectrogram configuration
     * @return AudioFeatures [n_mels, time_frames]
     */
    static AudioFeatures ComputeMelSpectrogram(const AudioData& audio,
                                               const MelConfig& config = {});

    /**
     * Compute MFCCs (Mel-Frequency Cepstral Coefficients)
     * @param audio Audio data
     * @param config MFCC configuration
     * @return AudioFeatures [n_mfcc, time_frames]
     */
    static AudioFeatures ComputeMFCC(const AudioData& audio,
                                     const MFCCConfig& config = {});

    // ==================== Augmentation ====================

    /**
     * Add Gaussian noise
     */
    static AudioData AddNoise(const AudioData& audio, float snr_db = 20.0f);

    /**
     * Time stretch (change speed without changing pitch)
     */
    static AudioData TimeStretch(const AudioData& audio, float rate = 1.0f);

    /**
     * Pitch shift (simplified via resampling)
     */
    static AudioData PitchShift(const AudioData& audio, float semitones = 0.0f);

    // ==================== Utility ====================

    /**
     * Resample audio to target sample rate (linear interpolation)
     */
    static AudioData Resample(const AudioData& audio, int target_sr);

    /**
     * Normalize audio to peak amplitude
     */
    static AudioData Normalize(const AudioData& audio);

    /**
     * Trim silence from beginning and end
     */
    static AudioData TrimSilence(const AudioData& audio, float threshold_db = -40.0f);

private:
    // Window functions
    static std::vector<float> HannWindow(int size);
    static std::vector<float> HammingWindow(int size);

    // Mel filterbank
    static float HzToMel(float hz);
    static float MelToHz(float mel);
    static std::vector<std::vector<float>> CreateMelFilterbank(
        int n_mels, int n_fft, int sample_rate, float fmin, float fmax);

    // DCT for MFCC
    static std::vector<std::vector<float>> DCTMatrix(int n_mfcc, int n_mels);
};

} // namespace cyxwiz
