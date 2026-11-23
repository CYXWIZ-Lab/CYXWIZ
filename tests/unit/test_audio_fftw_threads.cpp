// Spectrograms from several threads at once, as the audio batcher's workers
// compute features (TOFIX223 0.13, audit AUD-03): FFTW's planner is not
// thread-safe, so plans are made and destroyed under one lock. Every thread's
// result must equal the one computed alone.

#include <catch2/catch_test_macros.hpp>
#include <cyxwiz/audio_processing.h>

#include <atomic>
#include <cmath>
#include <string>
#include <thread>
#include <vector>

namespace {

cyxwiz::AudioData Tone(size_t samples, float frequency) {
    cyxwiz::AudioData audio;
    audio.sample_rate = 16000;
    audio.samples.resize(samples);
    for (size_t i = 0; i < samples; ++i)
        audio.samples[i] = 0.5f * std::sin(6.2831853f * frequency * static_cast<float>(i) / 16000.0f) +
                           0.1f * std::sin(0.37f * static_cast<float>(i));
    audio.num_samples = samples;
    audio.duration_seconds = static_cast<float>(samples) / 16000.0f;
    audio.valid = true;
    return audio;
}

cyxwiz::SpectrogramConfig Config(int n_fft) {
    cyxwiz::SpectrogramConfig config;
    config.n_fft = n_fft;
    config.hop_length = n_fft / 2;
    return config;
}

}  // namespace

TEST_CASE("Spectrograms computed on many threads match the single-threaded result", "[audio][fftw]") {
    const std::vector<int> sizes = {64, 128, 256, 512, 1024, 96, 160, 400};
    const auto audio = Tone(8000, 440.0f);

    std::vector<cyxwiz::AudioFeatures> reference;
    for (const int n_fft : sizes) reference.push_back(cyxwiz::AudioProcessing::ComputeSpectrogram(audio, Config(n_fft)));
    if (!reference.front().valid &&
        reference.front().error_message.find("requires FFTW3") != std::string::npos) {
        SKIP("this build has no FFTW");
    }
    for (const auto& r : reference) REQUIRE(r.valid);

    constexpr int kThreads = 8;
    constexpr int kRounds = 30;
    std::atomic<int> mismatches{0};
    std::atomic<int> errors{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (int round = 0; round < kRounds; ++round) {
                const size_t k = static_cast<size_t>(t + round) % sizes.size();
                const auto got = cyxwiz::AudioProcessing::ComputeSpectrogram(audio, Config(sizes[k]));
                if (!got.valid) {
                    ++errors;
                    continue;
                }
                if (got.data != reference[k].data) ++mismatches;
            }
        });
    }
    for (auto& thread : threads) thread.join();
    REQUIRE(errors.load() == 0);
    REQUIRE(mismatches.load() == 0);
}

// A window longer than the FFT was copied past the end of the n_fft buffer
// (a heap overflow). It is refused, as torch.stft refuses it.
TEST_CASE("A spectrogram window longer than n_fft is refused", "[audio]") {
    const auto audio = Tone(4000, 440.0f);
    auto config = Config(256);
    config.win_length = 512;
    const auto refused = cyxwiz::AudioProcessing::ComputeSpectrogram(audio, config);
    REQUIRE_FALSE(refused.valid);
    REQUIRE(refused.error_message.find("win_length") != std::string::npos);

    config.win_length = 256;
    const auto equal = cyxwiz::AudioProcessing::ComputeSpectrogram(audio, config);
    if (equal.error_message.find("requires FFTW3") == std::string::npos) REQUIRE(equal.valid);
}
