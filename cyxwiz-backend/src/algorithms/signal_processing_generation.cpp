#include <cyxwiz/signal_processing.h>

#include <cmath>
#include <random>

namespace cyxwiz {
namespace {
constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 2.0 * PI;
} // namespace

// ============================================================================
// Signal Generation
// ============================================================================

std::vector<double> SignalProcessing::GenerateSineWave(
    double frequency, double sample_rate, int num_samples, double amplitude, double phase) {

    std::vector<double> signal(num_samples);
    double dt = 1.0 / sample_rate;

    for (int i = 0; i < num_samples; i++) {
        double t = i * dt;
        signal[i] = amplitude * std::sin(TWO_PI * frequency * t + phase);
    }

    return signal;
}

std::vector<double> SignalProcessing::GenerateSquareWave(
    double frequency, double sample_rate, int num_samples, double amplitude) {

    std::vector<double> signal(num_samples);
    double dt = 1.0 / sample_rate;
    double period = 1.0 / frequency;

    for (int i = 0; i < num_samples; i++) {
        double t = i * dt;
        double phase = std::fmod(t, period) / period;
        signal[i] = (phase < 0.5) ? amplitude : -amplitude;
    }

    return signal;
}

std::vector<double> SignalProcessing::GenerateSawtoothWave(
    double frequency, double sample_rate, int num_samples, double amplitude) {

    std::vector<double> signal(num_samples);
    double dt = 1.0 / sample_rate;
    double period = 1.0 / frequency;

    for (int i = 0; i < num_samples; i++) {
        double t = i * dt;
        double phase = std::fmod(t, period) / period;
        signal[i] = amplitude * (2.0 * phase - 1.0);
    }

    return signal;
}

std::vector<double> SignalProcessing::GenerateWhiteNoise(int num_samples, double amplitude) {
    std::vector<double> signal(num_samples);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-amplitude, amplitude);

    for (int i = 0; i < num_samples; i++) {
        signal[i] = dis(gen);
    }

    return signal;
}

std::vector<double> SignalProcessing::AddNoise(const std::vector<double>& signal, double snr_db) {
    if (signal.empty()) return signal;

    // Calculate signal power
    double signal_power = 0.0;
    for (double s : signal) {
        signal_power += s * s;
    }
    signal_power /= signal.size();

    // Calculate noise power from SNR
    double noise_power = signal_power / std::pow(10.0, snr_db / 10.0);
    double noise_amplitude = std::sqrt(noise_power);

    // Generate noise and add to signal
    auto noise = GenerateWhiteNoise(static_cast<int>(signal.size()), noise_amplitude);
    std::vector<double> noisy_signal(signal.size());

    for (size_t i = 0; i < signal.size(); i++) {
        noisy_signal[i] = signal[i] + noise[i];
    }

    return noisy_signal;
}

} // namespace cyxwiz