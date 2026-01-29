#include "cyxwiz/audio_processing.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstring>
#include <spdlog/spdlog.h>

#ifdef CYXWIZ_HAS_SNDFILE
#include <sndfile.h>
#endif

#ifdef CYXWIZ_HAS_FFTW
#include <fftw3.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cyxwiz {

// ============================================================================
// Audio Loading via libsndfile
// ============================================================================

AudioData AudioProcessing::LoadAudio(const std::string& filepath, int target_sr) {
    AudioData result;
    result.filepath = filepath;

#ifdef CYXWIZ_HAS_SNDFILE
    SF_INFO sfinfo;
    sfinfo.format = 0;
    SNDFILE* sndfile = sf_open(filepath.c_str(), SFM_READ, &sfinfo);

    if (!sndfile) {
        result.error_message = "Cannot open audio file: " + filepath +
                               " (" + sf_strerror(nullptr) + ")";
        return result;
    }

    int num_channels = sfinfo.channels;
    sf_count_t total_frames = sfinfo.frames;

    // Read all samples as float (interleaved if multi-channel)
    std::vector<float> raw(total_frames * num_channels);
    sf_count_t read = sf_readf_float(sndfile, raw.data(), total_frames);
    sf_close(sndfile);

    if (read <= 0) {
        result.error_message = "Failed to read audio data from: " + filepath;
        return result;
    }

    // Mix to mono by averaging channels
    result.samples.resize(read);
    if (num_channels == 1) {
        result.samples.assign(raw.begin(), raw.begin() + read);
    } else {
        for (sf_count_t i = 0; i < read; i++) {
            float sum = 0.0f;
            for (int ch = 0; ch < num_channels; ch++) {
                sum += raw[i * num_channels + ch];
            }
            result.samples[i] = sum / num_channels;
        }
    }

    result.sample_rate = sfinfo.samplerate;
    result.original_sample_rate = sfinfo.samplerate;
    result.channels = 1;
    result.num_samples = result.samples.size();
    result.duration_seconds = static_cast<float>(result.num_samples) / result.sample_rate;
    result.valid = true;

    // Resample if needed
    if (target_sr > 0 && target_sr != result.sample_rate) {
        result = Resample(result, target_sr);
    }

    spdlog::info("AudioProcessing: loaded {} ({:.2f}s, {}Hz, {} samples)",
                 filepath, result.duration_seconds, result.sample_rate, result.num_samples);

    return result;
#else
    result.error_message = "Audio loading requires libsndfile (not available in this build)";
    return result;
#endif // CYXWIZ_HAS_SNDFILE
}

// ============================================================================
// Window Functions
// ============================================================================

std::vector<float> AudioProcessing::HannWindow(int size) {
    std::vector<float> w(size);
    for (int i = 0; i < size; i++) {
        w[i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / (size - 1)));
    }
    return w;
}

std::vector<float> AudioProcessing::HammingWindow(int size) {
    std::vector<float> w(size);
    for (int i = 0; i < size; i++) {
        w[i] = 0.54f - 0.46f * std::cos(2.0f * static_cast<float>(M_PI) * i / (size - 1));
    }
    return w;
}

// ============================================================================
// Spectrogram via FFTW3
// ============================================================================

AudioFeatures AudioProcessing::ComputeSpectrogram(const AudioData& audio,
                                                   const SpectrogramConfig& config) {
    AudioFeatures result;

    if (!audio.valid || audio.samples.empty()) {
        result.error_message = "Invalid audio data";
        return result;
    }

    int n_fft = config.n_fft;
    int hop = config.hop_length > 0 ? config.hop_length : n_fft / 2;
    int win_len = config.win_length > 0 ? config.win_length : n_fft;

    // Create window
    std::vector<float> window;
    if (config.window_type == 1) {
        window = HammingWindow(win_len);
    } else {
        window = HannWindow(win_len);
    }

    // Pad signal if center=true
    std::vector<float> padded;
    if (config.center) {
        int pad = n_fft / 2;
        padded.resize(audio.num_samples + 2 * pad, 0.0f);
        std::copy(audio.samples.begin(), audio.samples.end(), padded.begin() + pad);
    } else {
        padded = audio.samples;
    }

    int n_frames = static_cast<int>((padded.size() - win_len) / hop) + 1;
    if (n_frames <= 0) {
        result.error_message = "Audio too short for given FFT parameters";
        return result;
    }

    int freq_bins = n_fft / 2 + 1;
    result.rows = freq_bins;
    result.cols = n_frames;
    result.data.resize(freq_bins * n_frames);

#ifdef CYXWIZ_HAS_FFTW
    // Allocate FFTW buffers
    double* fft_in = fftw_alloc_real(n_fft);
    fftw_complex* fft_out = fftw_alloc_complex(freq_bins);
    fftw_plan plan = fftw_plan_dft_r2c_1d(n_fft, fft_in, fft_out, FFTW_ESTIMATE);

    for (int frame = 0; frame < n_frames; frame++) {
        // Zero-fill and apply window
        std::memset(fft_in, 0, n_fft * sizeof(double));
        int start = frame * hop;
        for (int i = 0; i < win_len && (start + i) < static_cast<int>(padded.size()); i++) {
            fft_in[i] = static_cast<double>(padded[start + i] * window[i]);
        }

        // Execute FFT
        fftw_execute(plan);

        // Compute magnitude spectrum
        for (int f = 0; f < freq_bins; f++) {
            double re = fft_out[f][0];
            double im = fft_out[f][1];
            result.data[f * n_frames + frame] = static_cast<float>(std::sqrt(re * re + im * im));
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(fft_in);
    fftw_free(fft_out);

    result.valid = true;
#else
    result.error_message = "Spectrogram computation requires FFTW3 (not available in this build)";
#endif // CYXWIZ_HAS_FFTW

    return result;
}

// ============================================================================
// Mel Spectrogram
// ============================================================================

float AudioProcessing::HzToMel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float AudioProcessing::MelToHz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

std::vector<std::vector<float>> AudioProcessing::CreateMelFilterbank(
    int n_mels, int n_fft, int sample_rate, float fmin, float fmax) {

    if (fmax <= 0.0f) fmax = static_cast<float>(sample_rate) / 2.0f;

    int freq_bins = n_fft / 2 + 1;
    float mel_min = HzToMel(fmin);
    float mel_max = HzToMel(fmax);

    // Create n_mels + 2 equally spaced mel points
    std::vector<float> mel_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
    }

    // Convert to Hz and then to FFT bin indices
    std::vector<int> bin_indices(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++) {
        float hz = MelToHz(mel_points[i]);
        bin_indices[i] = static_cast<int>(std::floor((n_fft + 1) * hz / sample_rate));
    }

    // Create triangular filters
    std::vector<std::vector<float>> filterbank(n_mels, std::vector<float>(freq_bins, 0.0f));

    for (int m = 0; m < n_mels; m++) {
        int f_left = bin_indices[m];
        int f_center = bin_indices[m + 1];
        int f_right = bin_indices[m + 2];

        for (int f = f_left; f < f_center && f < freq_bins; f++) {
            if (f_center != f_left) {
                filterbank[m][f] = static_cast<float>(f - f_left) / (f_center - f_left);
            }
        }
        for (int f = f_center; f < f_right && f < freq_bins; f++) {
            if (f_right != f_center) {
                filterbank[m][f] = static_cast<float>(f_right - f) / (f_right - f_center);
            }
        }
    }

    return filterbank;
}

AudioFeatures AudioProcessing::ComputeMelSpectrogram(const AudioData& audio,
                                                      const MelConfig& config) {
    AudioFeatures spec = ComputeSpectrogram(audio, config);
    if (!spec.valid) return spec;

    auto filterbank = CreateMelFilterbank(config.n_mels, config.n_fft,
                                          audio.sample_rate, config.fmin, config.fmax);

    int freq_bins = spec.rows;
    int n_frames = spec.cols;

    AudioFeatures result;
    result.rows = config.n_mels;
    result.cols = n_frames;
    result.data.resize(config.n_mels * n_frames);

    for (int m = 0; m < config.n_mels; m++) {
        for (int t = 0; t < n_frames; t++) {
            float sum = 0.0f;
            for (int f = 0; f < freq_bins; f++) {
                sum += filterbank[m][f] * spec.data[f * n_frames + t];
            }
            result.data[m * n_frames + t] = std::log(std::max(sum, 1e-10f));
        }
    }

    result.valid = true;
    return result;
}

// ============================================================================
// MFCC
// ============================================================================

std::vector<std::vector<float>> AudioProcessing::DCTMatrix(int n_mfcc, int n_mels) {
    std::vector<std::vector<float>> dct(n_mfcc, std::vector<float>(n_mels));

    for (int k = 0; k < n_mfcc; k++) {
        for (int n = 0; n < n_mels; n++) {
            dct[k][n] = std::cos(static_cast<float>(M_PI) * k * (2 * n + 1) / (2 * n_mels));
        }
    }

    float norm0 = std::sqrt(1.0f / n_mels);
    float normk = std::sqrt(2.0f / n_mels);
    for (int n = 0; n < n_mels; n++) dct[0][n] *= norm0;
    for (int k = 1; k < n_mfcc; k++) {
        for (int n = 0; n < n_mels; n++) dct[k][n] *= normk;
    }

    return dct;
}

AudioFeatures AudioProcessing::ComputeMFCC(const AudioData& audio, const MFCCConfig& config) {
    AudioFeatures mel = ComputeMelSpectrogram(audio, config);
    if (!mel.valid) return mel;

    int n_mels = mel.rows;
    int n_frames = mel.cols;

    auto dct = DCTMatrix(config.n_mfcc, n_mels);

    AudioFeatures result;
    result.rows = config.n_mfcc;
    result.cols = n_frames;
    result.data.resize(config.n_mfcc * n_frames);

    for (int k = 0; k < config.n_mfcc; k++) {
        for (int t = 0; t < n_frames; t++) {
            float sum = 0.0f;
            for (int m = 0; m < n_mels; m++) {
                sum += dct[k][m] * mel.data[m * n_frames + t];
            }
            result.data[k * n_frames + t] = sum;
        }
    }

    result.valid = true;
    return result;
}

// ============================================================================
// Augmentation
// ============================================================================

AudioData AudioProcessing::AddNoise(const AudioData& audio, float snr_db) {
    AudioData result = audio;
    if (!audio.valid) return result;

    float signal_power = 0.0f;
    for (float s : audio.samples) signal_power += s * s;
    signal_power /= audio.num_samples;

    float noise_power = signal_power / std::pow(10.0f, snr_db / 10.0f);
    float noise_std = std::sqrt(noise_power);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, noise_std);

    for (size_t i = 0; i < result.num_samples; i++) {
        result.samples[i] += dist(rng);
    }

    return result;
}

AudioData AudioProcessing::TimeStretch(const AudioData& audio, float rate) {
    if (!audio.valid || rate <= 0.0f) return audio;

    AudioData result;
    size_t new_length = static_cast<size_t>(audio.num_samples / rate);
    result.samples.resize(new_length);

    for (size_t i = 0; i < new_length; i++) {
        float src_idx = i * rate;
        int idx0 = static_cast<int>(src_idx);
        float frac = src_idx - idx0;
        int idx1 = std::min(idx0 + 1, static_cast<int>(audio.num_samples) - 1);
        idx0 = std::min(idx0, static_cast<int>(audio.num_samples) - 1);
        result.samples[i] = audio.samples[idx0] * (1.0f - frac) + audio.samples[idx1] * frac;
    }

    result.sample_rate = audio.sample_rate;
    result.original_sample_rate = audio.original_sample_rate;
    result.channels = 1;
    result.num_samples = new_length;
    result.duration_seconds = static_cast<float>(new_length) / result.sample_rate;
    result.filepath = audio.filepath;
    result.valid = true;
    return result;
}

AudioData AudioProcessing::PitchShift(const AudioData& audio, float semitones) {
    if (!audio.valid || semitones == 0.0f) return audio;

    float rate = std::pow(2.0f, semitones / 12.0f);
    AudioData stretched = TimeStretch(audio, rate);
    int new_sr = static_cast<int>(audio.sample_rate * rate);
    stretched.sample_rate = new_sr;
    return Resample(stretched, audio.sample_rate);
}

// ============================================================================
// Utility
// ============================================================================

AudioData AudioProcessing::Resample(const AudioData& audio, int target_sr) {
    if (!audio.valid || target_sr == audio.sample_rate) return audio;

    AudioData result;
    float ratio = static_cast<float>(target_sr) / audio.sample_rate;
    size_t new_length = static_cast<size_t>(audio.num_samples * ratio);

    result.samples.resize(new_length);

    for (size_t i = 0; i < new_length; i++) {
        float src_idx = static_cast<float>(i) / ratio;
        int idx0 = static_cast<int>(src_idx);
        float frac = src_idx - idx0;
        int idx1 = std::min(idx0 + 1, static_cast<int>(audio.num_samples) - 1);
        idx0 = std::min(idx0, static_cast<int>(audio.num_samples) - 1);
        result.samples[i] = audio.samples[idx0] * (1.0f - frac) + audio.samples[idx1] * frac;
    }

    result.sample_rate = target_sr;
    result.original_sample_rate = audio.original_sample_rate;
    result.channels = 1;
    result.num_samples = new_length;
    result.duration_seconds = static_cast<float>(new_length) / target_sr;
    result.filepath = audio.filepath;
    result.valid = true;
    return result;
}

AudioData AudioProcessing::Normalize(const AudioData& audio) {
    if (!audio.valid) return audio;
    AudioData result = audio;

    float max_val = 0.0f;
    for (float s : result.samples) max_val = std::max(max_val, std::abs(s));

    if (max_val > 0.0f) {
        float scale = 1.0f / max_val;
        for (float& s : result.samples) s *= scale;
    }

    return result;
}

AudioData AudioProcessing::TrimSilence(const AudioData& audio, float threshold_db) {
    if (!audio.valid) return audio;

    float peak = 0.0f;
    for (float s : audio.samples) peak = std::max(peak, std::abs(s));
    if (peak == 0.0f) return audio;

    float threshold = peak * std::pow(10.0f, threshold_db / 20.0f);

    size_t start = 0, end = audio.num_samples;
    for (size_t i = 0; i < audio.num_samples; i++) {
        if (std::abs(audio.samples[i]) > threshold) { start = i; break; }
    }
    for (size_t i = audio.num_samples; i > 0; i--) {
        if (std::abs(audio.samples[i - 1]) > threshold) { end = i; break; }
    }

    if (start >= end) return audio;

    AudioData result;
    result.samples.assign(audio.samples.begin() + start, audio.samples.begin() + end);
    result.sample_rate = audio.sample_rate;
    result.original_sample_rate = audio.original_sample_rate;
    result.channels = 1;
    result.num_samples = result.samples.size();
    result.duration_seconds = static_cast<float>(result.num_samples) / result.sample_rate;
    result.filepath = audio.filepath;
    result.valid = true;
    return result;
}

} // namespace cyxwiz
