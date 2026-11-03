// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

#include "api_export.h"
#include <vector>
#include <string>
#include <complex>

namespace cyxwiz {

// ============================================================================
// Result Structures
// ============================================================================

// The device does the arithmetic (ArrayFire, float64 where the device has it,
// float32 otherwise; TOFIX129 D7); every figure here is checked against
// numpy.fft, scipy.signal and PyWavelets in test_computation_truth_signal.

struct CYXWIZ_API FFTResult {
    std::vector<double> magnitude;                    // |FFT|, two-sided (numpy.fft.fft)
    std::vector<double> phase;                        // angle(FFT) in radians
    std::vector<double> frequencies;                  // numpy.fft.fftfreq(n, 1 / sample_rate)
    std::vector<std::complex<double>> complex_output; // Raw complex FFT output
    int n = 0;                                        // Number of samples
    double sample_rate = 1.0;                         // Sample rate (Hz)
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API FFT2DResult {
    std::vector<std::vector<double>> magnitude;       // 2D magnitude
    std::vector<std::vector<double>> phase;           // 2D phase
    std::vector<std::vector<std::complex<double>>> complex_output;
    int rows = 0;
    int cols = 0;
    bool success = false;
    std::string error_message;
};

// A one-sided spectrum (Spectrum, Welch): `frequencies` 0 .. Nyquist.
struct CYXWIZ_API SpectrumResult {
    std::vector<double> frequencies;  // numpy.fft.rfftfreq(n, 1 / sample_rate) (Spectrum), scipy.signal.welch's (Welch)
    std::vector<double> amplitude;    // Spectrum only: 2 |X| / sum(window), DC and Nyquist not doubled (the amplitude of a tone)
    std::vector<double> psd;          // power density, scipy.signal.periodogram / welch (window hann, detrend constant, density)
    int n = 0;                        // samples (Spectrum) or samples per segment (Welch)
    int segments = 1;                 // Welch: segments averaged
    double sample_rate = 1.0;
    std::string window;
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API ConvolutionResult {
    std::vector<double> output;
    int output_size = 0;
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API Convolution2DResult {
    std::vector<std::vector<double>> output;
    int rows = 0;
    int cols = 0;
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API FilterCoefficients {
    std::vector<double> b;                  // Numerator coefficients (FIR, scipy.signal.firwin)
    std::vector<double> a;                  // Denominator coefficients (IIR, [1] for FIR)
    std::vector<double> freq_response_mag;  // |H| on freq_axis (scipy.signal.freqz)
    std::vector<double> freq_response_phase;// angle(H), wrapped
    std::vector<double> freq_axis;          // 0 .. Nyquist inclusive, num_points of them
    int order = 0;                          // taps (the length of b)
    double cutoff_low = 0.0;                // Lower cutoff frequency
    double cutoff_high = 0.0;               // Upper cutoff frequency (for bandpass/stop)
    double sample_rate = 1.0;
    std::string filter_type;                // "lowpass", "highpass", "bandpass", "bandstop"
    std::string window = "hamming";
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API SpectrogramResult {
    std::vector<std::vector<double>> spectrogram;  // [time_frame][freq_bin] power density (scipy.signal.spectrogram, psd, density)
    std::vector<double> times;                      // frame centres (seconds)
    std::vector<double> frequencies;                // 0 .. Nyquist (Hz)
    int num_frames = 0;
    int num_bins = 0;
    double duration = 0.0;                          // Total duration (seconds)
    bool success = false;
    std::string error_message;
};

// pywt.wavedec(x, wavelet, mode="symmetric", level): details[0] is the finest
// level (cD_1), details[levels - 1] the coarsest (cD_levels).
struct CYXWIZ_API WaveletResult {
    std::vector<double> approximation;              // cA_levels
    std::vector<std::vector<double>> details;       // cD_1 .. cD_levels
    int levels = 0;
    int max_levels = 0;                             // pywt.dwt_max_level(n, filter length)
    int original_size = 0;
    std::string wavelet_name;
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API DenoiseResult {
    std::vector<double> denoised;                   // the signal rebuilt from the thresholded details
    double sigma = 0.0;                             // noise level: median |cD_1| / 0.6745
    double threshold = 0.0;                         // the threshold used (universal: sigma sqrt(2 ln n), or the one given)
    bool success = false;
    std::string error_message;
};

// ============================================================================
// Signal Processing Class
// ============================================================================

class CYXWIZ_API SignalProcessing {
public:
    // ==================== FFT Operations ====================

    // 1D Fast Fourier Transform, two-sided (numpy.fft.fft)
    static FFTResult FFT(
        const std::vector<double>& signal,
        double sample_rate = 1.0
    );

    // 2D Fast Fourier Transform (for images)
    static FFT2DResult FFT2D(
        const std::vector<std::vector<double>>& image
    );

    // Inverse FFT (1D): the real part of numpy.fft.ifft
    static std::vector<double> IFFT(
        const std::vector<std::complex<double>>& spectrum
    );

    // Inverse FFT (2D): the real part of numpy.fft.ifft2
    static std::vector<std::vector<double>> IFFT2D(
        const std::vector<std::vector<std::complex<double>>>& spectrum
    );

    // ==================== Spectrum ====================

    // One-sided amplitude spectrum and power density of the whole signal:
    // the mean removed (unless remove_mean is false), a periodic window
    // ("hann", "hamming", "blackman", "boxcar"), one device FFT.
    static SpectrumResult Spectrum(
        const std::vector<double>& signal,
        double sample_rate = 1.0,
        const std::string& window = "hann",
        bool remove_mean = true
    );

    // Welch's power density: segments of `nperseg` samples overlapping by
    // half, each with its mean removed and windowed, their power averaged
    // (scipy.signal.welch, scaling "density"). One batched device FFT.
    static SpectrumResult Welch(
        const std::vector<double>& signal,
        double sample_rate = 1.0,
        int nperseg = 256,
        const std::string& window = "hann"
    );

    // ==================== Convolution ====================

    // 1D Convolution, numpy.convolve: "full" (n + k - 1), "same" (max(n, k),
    // centred), "valid" (max - min + 1)
    static ConvolutionResult Convolve1D(
        const std::vector<double>& signal,
        const std::vector<double>& kernel,
        const std::string& mode = "same"
    );

    // 2D Convolution
    static Convolution2DResult Convolve2D(
        const std::vector<std::vector<double>>& image,
        const std::vector<std::vector<double>>& kernel,
        const std::string& mode = "same"
    );

    // ==================== Filter Design ====================
    // Windowed-sinc FIR filters as scipy.signal.firwin(taps, cutoff, window,
    // pass_zero, scale=True, fs): `taps` is the number of coefficients (odd
    // for highpass and bandstop; 3 or more). The response is scipy.signal.freqz
    // on 512 points from 0 to Nyquist inclusive.

    static FilterCoefficients DesignFIR(
        const std::string& filter_type,    // "lowpass", "highpass", "bandpass", "bandstop"
        double cutoff_low,
        double cutoff_high,                // bandpass / bandstop only
        double sample_rate,
        int taps,
        const std::string& window = "hamming"
    );

    static FilterCoefficients DesignLowpass(
        double cutoff_freq,
        double sample_rate,
        int taps = 31
    );

    static FilterCoefficients DesignHighpass(
        double cutoff_freq,
        double sample_rate,
        int taps = 31
    );

    static FilterCoefficients DesignBandpass(
        double low_freq,
        double high_freq,
        double sample_rate,
        int taps = 31
    );

    static FilterCoefficients DesignBandstop(
        double low_freq,
        double high_freq,
        double sample_rate,
        int taps = 31
    );

    // Zero-phase FIR apply, the kernel centred on each sample; beyond the ends
    // the signal repeats its edge value (numpy.convolve(pad(x, "edge"), b,
    // "valid")); `a` is ignored (FIR only)
    static std::vector<double> ApplyFilter(
        const std::vector<double>& signal,
        const FilterCoefficients& filter
    );

    // Frequency response of b on num_points from 0 to Nyquist inclusive
    static void ComputeFrequencyResponse(
        FilterCoefficients& filter,
        int num_points = 512
    );

    // ==================== Spectrogram ====================

    // STFT power density (scipy.signal.spectrogram, window periodic, no
    // detrend, scaling "density", mode "psd"): one batched device FFT over
    // every frame; times are the frame centres
    static SpectrogramResult ComputeSpectrogram(
        const std::vector<double>& signal,
        int window_size = 256,
        int hop_size = 128,
        double sample_rate = 1.0,
        const std::string& window_type = "hann"
    );

    // ==================== Wavelet Transform ====================

    // The wavelets the backend knows (PyWavelets' filters): haar, db1..db8,
    // sym2..sym8, coif1..coif5
    static std::vector<std::string> WaveletNames();
    // pywt.dwt_max_level(n, filter length); 0 for an unknown wavelet
    static int WaveletMaxLevel(int n, const std::string& wavelet);

    // Discrete Wavelet Transform, pywt.wavedec(x, wavelet, mode="symmetric", level)
    static WaveletResult DWT(
        const std::vector<double>& signal,
        const std::string& wavelet = "haar",
        int levels = 3
    );

    // Inverse Discrete Wavelet Transform, pywt.waverec, trimmed to the original size
    static std::vector<double> IDWT(
        const WaveletResult& coeffs
    );

    // Wavelet denoising: the details thresholded (soft or hard) at
    // `threshold`, or at the universal threshold sigma sqrt(2 ln n) when
    // `threshold` <= 0, then rebuilt
    static DenoiseResult WaveletDenoise(
        const std::vector<double>& signal,
        const std::string& wavelet = "db4",
        int levels = 3,
        double threshold = 0.0,
        bool soft = true
    );

    // ==================== Window Functions ====================

    // scipy.signal.get_window(name, size, fftbins=periodic): "hann",
    // "hamming", "blackman", "boxcar" / "rectangular"; empty for an unknown name
    static std::vector<double> Window(const std::string& name, int size, bool periodic);

    static std::vector<double> HammingWindow(int size);      // symmetric
    static std::vector<double> HannWindow(int size);         // symmetric
    static std::vector<double> BlackmanWindow(int size);     // symmetric
    static std::vector<double> RectangularWindow(int size);

    // ==================== Signal Generation ====================

    // Generate sine wave
    static std::vector<double> GenerateSineWave(
        double frequency,
        double sample_rate,
        int num_samples,
        double amplitude = 1.0,
        double phase = 0.0
    );

    // Generate square wave
    static std::vector<double> GenerateSquareWave(
        double frequency,
        double sample_rate,
        int num_samples,
        double amplitude = 1.0
    );

    // Generate sawtooth wave
    static std::vector<double> GenerateSawtoothWave(
        double frequency,
        double sample_rate,
        int num_samples,
        double amplitude = 1.0
    );

    // Generate white noise
    static std::vector<double> GenerateWhiteNoise(
        int num_samples,
        double amplitude = 1.0
    );

    // Add noise to signal
    static std::vector<double> AddNoise(
        const std::vector<double>& signal,
        double snr_db
    );

    // ==================== Signal Analysis ====================

    // Find peaks in signal (greedy left to right: a local maximum of at least
    // min_height, min_distance from the peaks found before it)
    struct Peak {
        int index;
        double value;
        double frequency;  // If applicable
    };

    static std::vector<Peak> FindPeaks(
        const std::vector<double>& signal,
        double min_height = 0.0,
        int min_distance = 1
    );

    // scipy.signal.find_peaks(x, prominence, distance): local maxima (a
    // plateau counts once, at its middle), the highest kept first when two
    // are closer than `distance`, then those whose prominence (the drop to
    // the higher of the two bases) reaches `prominence`. A sequential host
    // walk: it depends on each peak's neighbours, not an array formula.
    struct ProminentPeak {
        int index = 0;
        double value = 0.0;
        double prominence = 0.0;
        int left_base = 0;
        int right_base = 0;
    };

    static std::vector<ProminentPeak> FindPeaksByProminence(
        const std::vector<double>& signal,
        double prominence,
        int distance = 1
    );

    // Compute power spectral density
    static FFTResult PowerSpectralDensity(
        const std::vector<double>& signal,
        double sample_rate = 1.0
    );

    // ==================== Utility ====================

    // Zero-pad signal to next power of 2
    static std::vector<double> ZeroPadToPowerOf2(
        const std::vector<double>& signal
    );

    // Resample signal
    static std::vector<double> Resample(
        const std::vector<double>& signal,
        int target_size
    );

    // Normalize signal to [-1, 1]
    static std::vector<double> Normalize(
        const std::vector<double>& signal
    );

    // Remove DC offset
    static std::vector<double> RemoveDC(
        const std::vector<double>& signal
    );

private:
    // Helper: Check if size is power of 2
    static bool IsPowerOf2(int n);

    // Helper: Next power of 2
    static int NextPowerOf2(int n);

    // Helper: PyWavelets' analysis and synthesis filters; false for an unknown name
    static bool GetWaveletFilters(
        const std::string& wavelet,
        std::vector<double>& dec_lo,
        std::vector<double>& dec_hi,
        std::vector<double>& rec_lo,
        std::vector<double>& rec_hi
    );
};

} // namespace cyxwiz
