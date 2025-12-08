#pragma once

#include "api_export.h"
#include <vector>
#include <string>
#include <complex>

namespace cyxwiz {

// ============================================================================
// Result Structures
// ============================================================================

struct CYXWIZ_API FFTResult {
    std::vector<double> magnitude;                    // |FFT|
    std::vector<double> phase;                        // angle(FFT) in radians
    std::vector<double> frequencies;                  // Frequency bins (Hz)
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
    std::vector<double> b;                  // Numerator coefficients (FIR)
    std::vector<double> a;                  // Denominator coefficients (IIR, [1] for FIR)
    std::vector<double> freq_response_mag;  // Magnitude response
    std::vector<double> freq_response_phase;// Phase response
    std::vector<double> freq_axis;          // Frequency axis for response
    int order = 0;
    double cutoff_low = 0.0;                // Lower cutoff frequency
    double cutoff_high = 0.0;               // Upper cutoff frequency (for bandpass/stop)
    double sample_rate = 1.0;
    std::string filter_type;                // "lowpass", "highpass", "bandpass", "bandstop"
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API SpectrogramResult {
    std::vector<std::vector<double>> spectrogram;  // [time_frame][freq_bin] power
    std::vector<double> times;                      // Time axis (seconds)
    std::vector<double> frequencies;                // Frequency axis (Hz)
    int num_frames = 0;
    int num_bins = 0;
    double duration = 0.0;                          // Total duration (seconds)
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API WaveletResult {
    std::vector<double> approximation;              // Final approximation coefficients
    std::vector<std::vector<double>> details;       // Detail coefficients per level
    int levels = 0;
    int original_size = 0;
    std::string wavelet_name;
    bool success = false;
    std::string error_message;
};

// ============================================================================
// Signal Processing Class
// ============================================================================

class CYXWIZ_API SignalProcessing {
public:
    // ==================== FFT Operations ====================

    // 1D Fast Fourier Transform
    static FFTResult FFT(
        const std::vector<double>& signal,
        double sample_rate = 1.0
    );

    // 2D Fast Fourier Transform (for images)
    static FFT2DResult FFT2D(
        const std::vector<std::vector<double>>& image
    );

    // Inverse FFT (1D)
    static std::vector<double> IFFT(
        const std::vector<std::complex<double>>& spectrum
    );

    // Inverse FFT (2D)
    static std::vector<std::vector<double>> IFFT2D(
        const std::vector<std::vector<std::complex<double>>>& spectrum
    );

    // ==================== Convolution ====================

    // 1D Convolution
    // mode: "full" = full convolution, "same" = same size as input, "valid" = only valid overlap
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

    // Design lowpass filter
    static FilterCoefficients DesignLowpass(
        double cutoff_freq,
        double sample_rate,
        int order = 4
    );

    // Design highpass filter
    static FilterCoefficients DesignHighpass(
        double cutoff_freq,
        double sample_rate,
        int order = 4
    );

    // Design bandpass filter
    static FilterCoefficients DesignBandpass(
        double low_freq,
        double high_freq,
        double sample_rate,
        int order = 4
    );

    // Design bandstop (notch) filter
    static FilterCoefficients DesignBandstop(
        double low_freq,
        double high_freq,
        double sample_rate,
        int order = 4
    );

    // Apply filter to signal
    static std::vector<double> ApplyFilter(
        const std::vector<double>& signal,
        const FilterCoefficients& filter
    );

    // Compute frequency response of filter
    static void ComputeFrequencyResponse(
        FilterCoefficients& filter,
        int num_points = 512
    );

    // ==================== Spectrogram ====================

    // Compute spectrogram (STFT)
    static SpectrogramResult ComputeSpectrogram(
        const std::vector<double>& signal,
        int window_size = 256,
        int hop_size = 128,
        double sample_rate = 1.0,
        const std::string& window_type = "hann"
    );

    // ==================== Wavelet Transform ====================

    // Discrete Wavelet Transform
    // wavelet: "haar", "db1", "db2", "db3", "db4"
    static WaveletResult DWT(
        const std::vector<double>& signal,
        const std::string& wavelet = "haar",
        int levels = 3
    );

    // Inverse Discrete Wavelet Transform
    static std::vector<double> IDWT(
        const WaveletResult& coeffs
    );

    // ==================== Window Functions ====================

    static std::vector<double> HammingWindow(int size);
    static std::vector<double> HannWindow(int size);
    static std::vector<double> BlackmanWindow(int size);
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

    // Find peaks in signal
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

    // Helper: Get wavelet filter coefficients
    static void GetWaveletFilters(
        const std::string& wavelet,
        std::vector<double>& low_pass,
        std::vector<double>& high_pass
    );

    // Helper: Single level DWT decomposition
    static void DWTDecompose(
        const std::vector<double>& signal,
        const std::vector<double>& low_pass,
        const std::vector<double>& high_pass,
        std::vector<double>& approx,
        std::vector<double>& detail
    );

    // Helper: Single level IDWT reconstruction
    static std::vector<double> DWTReconstruct(
        const std::vector<double>& approx,
        const std::vector<double>& detail,
        const std::vector<double>& low_pass,
        const std::vector<double>& high_pass,
        int original_size
    );
};

} // namespace cyxwiz
