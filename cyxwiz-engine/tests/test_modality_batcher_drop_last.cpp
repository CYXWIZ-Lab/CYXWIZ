#include "../src/core/audio_dataset_batcher.h"
#include "../src/core/image_dataset_batcher.h"
#include "../src/core/image_utils.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

void WriteLe16(std::ofstream& out, uint16_t value) {
    out.put(static_cast<char>(value & 0xff));
    out.put(static_cast<char>((value >> 8) & 0xff));
}

void WriteLe32(std::ofstream& out, uint32_t value) {
    out.put(static_cast<char>(value & 0xff));
    out.put(static_cast<char>((value >> 8) & 0xff));
    out.put(static_cast<char>((value >> 16) & 0xff));
    out.put(static_cast<char>((value >> 24) & 0xff));
}

// two_tone: the second row is the inverse colour, so the image is not uniform.
void WriteBmp24(const fs::path& path, uint8_t red, uint8_t green, uint8_t blue,
                bool two_tone = false) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    Check(out.good(), "failed to create BMP fixture " + path.string());

    constexpr uint32_t width = 2;
    constexpr uint32_t height = 2;
    constexpr uint32_t row_bytes = 8;
    constexpr uint32_t pixel_bytes = row_bytes * height;
    constexpr uint32_t data_offset = 54;

    out.write("BM", 2);
    WriteLe32(out, data_offset + pixel_bytes);
    WriteLe16(out, 0);
    WriteLe16(out, 0);
    WriteLe32(out, data_offset);
    WriteLe32(out, 40);
    WriteLe32(out, width);
    WriteLe32(out, height);
    WriteLe16(out, 1);
    WriteLe16(out, 24);
    WriteLe32(out, 0);
    WriteLe32(out, pixel_bytes);
    WriteLe32(out, 2835);
    WriteLe32(out, 2835);
    WriteLe32(out, 0);
    WriteLe32(out, 0);

    for (uint32_t row = 0; row < height; ++row) {
        const bool inverse = two_tone && row == 1;
        for (uint32_t column = 0; column < width; ++column) {
            out.put(static_cast<char>(inverse ? 255 - blue : blue));
            out.put(static_cast<char>(inverse ? 255 - green : green));
            out.put(static_cast<char>(inverse ? 255 - red : red));
        }
        out.put(0);
        out.put(0);
    }
}

void WriteMonoWav16(const fs::path& path,
                    const std::vector<int16_t>& samples,
                    uint32_t sample_rate) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    Check(out.good(), "failed to create WAV fixture " + path.string());

    constexpr uint16_t channels = 1;
    constexpr uint16_t bits_per_sample = 16;
    constexpr uint16_t block_align = channels * bits_per_sample / 8;
    const uint32_t byte_rate = sample_rate * block_align;
    const uint32_t data_bytes =
        static_cast<uint32_t>(samples.size() * sizeof(int16_t));

    out.write("RIFF", 4);
    WriteLe32(out, 36 + data_bytes);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    WriteLe32(out, 16);
    WriteLe16(out, 1);
    WriteLe16(out, channels);
    WriteLe32(out, sample_rate);
    WriteLe32(out, byte_rate);
    WriteLe16(out, block_align);
    WriteLe16(out, bits_per_sample);
    out.write("data", 4);
    WriteLe32(out, data_bytes);
    for (int16_t sample : samples) {
        WriteLe16(out, static_cast<uint16_t>(sample));
    }
}

std::vector<int16_t> MakeWave(int offset) {
    std::vector<int16_t> samples(128);
    for (size_t i = 0; i < samples.size(); ++i) {
        const int phase = static_cast<int>((i + offset) % 8);
        samples[i] = static_cast<int16_t>((phase - 4) * 2500);
    }
    return samples;
}

std::vector<size_t> CollectBatchSizes(cyxwiz::IBatcher& batcher) {
    std::vector<size_t> sizes;
    batcher.Reset();
    while (!batcher.IsEpochComplete()) {
        auto batch = batcher.GetNextBatch();
        Check(batch.IsValid(), "incomplete modality batch should be valid");
        sizes.push_back(batch.size);
    }
    return sizes;
}

struct EpochPayload {
    std::vector<size_t> batch_sizes;
    std::vector<float> data;
    std::vector<float> labels;
};

EpochPayload CollectEpochPayload(cyxwiz::IBatcher& batcher) {
    EpochPayload payload;
    batcher.Reset();
    while (!batcher.IsEpochComplete()) {
        auto batch = batcher.GetNextBatch();
        Check(batch.IsValid(), "seeded modality batch should be valid");
        payload.batch_sizes.push_back(batch.size);

        const float* data = batch.data.ReadData<float>();
        payload.data.insert(payload.data.end(), data,
                            data + batch.data.NumElements());
        const float* labels = batch.labels.ReadData<float>();
        payload.labels.insert(payload.labels.end(), labels,
                              labels + batch.labels.NumElements());
    }
    return payload;
}

void CheckMatchingPayloads(const EpochPayload& first,
                           const EpochPayload& second,
                           const std::string& modality) {
    Check(first.batch_sizes == second.batch_sizes,
          modality + " matching seeds should reproduce batch boundaries");
    Check(first.data == second.data,
          modality + " matching seeds should reproduce sample order");
    Check(first.labels == second.labels,
          modality + " matching seeds should reproduce label order");
}

template <typename Batcher>
void CheckTrainOnlyDropLast(Batcher& batcher, const std::string& modality) {
    Check(batcher.GetNumSamples() == 3,
          modality + " should expose three Train samples");
    Check(batcher.GetNumValSamples() == 1,
          modality + " should expose one validation sample");
    Check(batcher.GetNumBatches() == 2,
          modality + " should keep its partial Train batch by default");
    Check(CollectBatchSizes(batcher) == std::vector<size_t>({2, 1}),
          modality + " should emit full then partial Train batches");

    batcher.SetDropLast(true);
    batcher.SetPhase(cyxwiz::BatcherPhase::Train);
    batcher.Reset();
    Check(batcher.GetNumBatches() == 1,
          modality + " drop_last should floor the Train batch count");
    Check(CollectBatchSizes(batcher) == std::vector<size_t>({2}),
          modality + " drop_last should suppress the partial Train batch");

    batcher.SetPhase(cyxwiz::BatcherPhase::Val);
    batcher.Reset();
    Check(batcher.GetNumBatches() == 1,
          modality + " validation should retain its partial batch");
    Check(CollectBatchSizes(batcher) == std::vector<size_t>({1}),
          modality + " validation should emit its one-sample partial batch");
}

void TestImageBatcher(const fs::path& root) {
    const auto image_root = root / "images";
    WriteBmp24(image_root / "class_a" / "a0.bmp", 255, 0, 0);
    WriteBmp24(image_root / "class_a" / "a1.bmp", 192, 32, 0);
    WriteBmp24(image_root / "class_b" / "b0.bmp", 0, 255, 0);
    WriteBmp24(image_root / "class_b" / "b1.bmp", 0, 192, 32);

    cyxwiz::DataRegistry::ImageDatasetEntry entry;
    entry.folder_path = image_root.string();
    entry.layout = 0;
    entry.num_images = 4;
    entry.num_classes = 2;

    cyxwiz::ImagePreprocessingConfig preprocessing;
    preprocessing.resize_mode = cyxwiz::ResizeMode::Exact;
    preprocessing.target_width = 2;
    preprocessing.target_height = 2;

    const auto images = cyxwiz::PrepareImageSet(entry, cyxwiz::PreparedImageSpecFrom(preprocessing)).set;
    Check(images != nullptr, "the images are prepared");
    cyxwiz::ImageDatasetBatcher batcher(images, 2, 0.75f, false, 17);
    CheckTrainOnlyDropLast(batcher, "ImageDatasetBatcher");

    cyxwiz::ImageDatasetBatcher first_seeded(images, 2, 0.75f, true, 1701);
    cyxwiz::ImageDatasetBatcher second_seeded(images, 2, 0.75f, true, 1701);
    CheckMatchingPayloads(CollectEpochPayload(first_seeded),
                          CollectEpochPayload(second_seeded),
                          "ImageDatasetBatcher");

    // Each sample is the whole image decoded at the Resize size: a class
    // subfolder dataset used to decode at its 224x224 default, and the batcher
    // then kept only the top-left 2x2 corner, one colour of a two-tone image.
    const auto two_tone_root = root / "two_tone";
    WriteBmp24(two_tone_root / "class_a" / "t0.bmp", 255, 0, 0, true);
    WriteBmp24(two_tone_root / "class_b" / "t1.bmp", 0, 255, 0, true);
    cyxwiz::DataRegistry::ImageDatasetEntry two_tone;
    two_tone.folder_path = two_tone_root.string();
    two_tone.layout = 0;
    two_tone.num_images = 2;
    two_tone.num_classes = 2;
    cyxwiz::ImageDatasetBatcher whole_images(
        cyxwiz::PrepareImageSet(two_tone, cyxwiz::PreparedImageSpecFrom(preprocessing)).set, 2, 1.0f, false, 17);
    const auto batch = whole_images.GetNextBatch();
    Check(batch.size == 2, "ImageDatasetBatcher should emit both two-tone images");
    const float* pixels = batch.data.ReadData<float>();
    for (size_t i = 0; i < batch.size; ++i) {
        const float* sample = pixels + i * 2 * 2 * 3;
        const float* bottom_right = sample + 3 * 3;
        float difference = 0.0f;
        for (int c = 0; c < 3; ++c) difference += std::abs(sample[c] - bottom_right[c]);
        Check(difference > 1.5f,
              "ImageDatasetBatcher should keep both rows of a 2x2 image at Resize 2x2");
    }
}

// A 2x2 8-bit BMP with a grayscale palette (OpenCV reads it as one channel).
void WriteGrayBmp8(const fs::path& path, uint8_t value) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    const uint32_t palette = 256 * 4, row = 4, pixels = row * 2;
    out.put('B');
    out.put('M');
    WriteLe32(out, 14 + 40 + palette + pixels);
    WriteLe32(out, 0);
    WriteLe32(out, 14 + 40 + palette);
    WriteLe32(out, 40);
    WriteLe32(out, 2);
    WriteLe32(out, 2);
    WriteLe16(out, 1);
    WriteLe16(out, 8);
    for (int i = 0; i < 6; ++i) WriteLe32(out, i == 4 ? 256 : 0);
    for (int i = 0; i < 256; ++i) {
        for (int c = 0; c < 3; ++c) out.put(static_cast<char>(i));
        out.put(0);
    }
    for (int y = 0; y < 2; ++y) {
        out.put(static_cast<char>(value));
        out.put(static_cast<char>(value));
        out.put(0);
        out.put(0);
    }
}

void TestGrayscaleImages(const fs::path& root) {
    // Grayscale files are read as RGB: they used to stay one channel and the
    // batcher gave them all-zero samples.
    const auto gray_root = root / "gray";
    WriteGrayBmp8(gray_root / "class_a" / "g0.bmp", 128);
    WriteGrayBmp8(gray_root / "class_b" / "g1.bmp", 128);
    cyxwiz::DataRegistry::ImageDatasetEntry entry;
    entry.folder_path = gray_root.string();
    entry.layout = 0;
    entry.num_images = 2;
    entry.num_classes = 2;
    cyxwiz::ImagePreprocessingConfig preprocessing;
    preprocessing.resize_mode = cyxwiz::ResizeMode::Exact;
    preprocessing.target_width = 2;
    preprocessing.target_height = 2;
    cyxwiz::ImageDatasetBatcher batcher(
        cyxwiz::PrepareImageSet(entry, cyxwiz::PreparedImageSpecFrom(preprocessing)).set, 2, 1.0f, false, 17);
    const auto batch = batcher.GetNextBatch();
    Check(batch.size == 2, "both grayscale images form a batch");
    const float* pixels = batch.data.ReadData<float>();
    for (size_t i = 0; i < batch.size * 2 * 2 * 3; ++i) {
        Check(std::abs(pixels[i] - 128.0f / 255.0f) < 0.01f,
              "a grayscale pixel 128 reads as 0.5 in each of R, G, B");
    }

    // A 16-bit file scales by 65535, not 255.
    const auto pgm = root / "gray16.pgm";
    {
        std::ofstream out(pgm, std::ios::binary);
        out << "P5\n1 1\n65535\n";
        out.put(static_cast<char>(0x80));
        out.put(0);
    }
    std::vector<float> data;
    int width = 0, height = 0, channels = 0;
    Check(cyxwiz::ImageUtils::LoadImage(pgm.string(), data, width, height, channels) && channels == 3 &&
              data.size() == 3 && std::abs(data[0] - 32768.0f / 65535.0f) < 0.001f,
          "a 16-bit grayscale pixel 32768 reads as 0.5 in R, G, B");
}

void TestAudioBatcher(const fs::path& root) {
    const auto audio_root = root / "audio";
    WriteMonoWav16(audio_root / "class_a" / "a0.wav", MakeWave(0), 8000);
    WriteMonoWav16(audio_root / "class_a" / "a1.wav", MakeWave(1), 8000);
    WriteMonoWav16(audio_root / "class_b" / "b0.wav", MakeWave(2), 8000);
    WriteMonoWav16(audio_root / "class_b" / "b1.wav", MakeWave(3), 8000);

    cyxwiz::DataRegistry::AudioDatasetEntry entry;
    entry.folder_path = audio_root.string();
    entry.labeled_subdirs = true;
    entry.target_sr = 8000;
    entry.max_duration = 0.016f;
    entry.num_samples = 4;
    entry.num_classes = 2;

    // Without a feature node the batcher refuses rather than invent features.
    cyxwiz::AudioPreprocessingConfig no_features;
    cyxwiz::AudioDatasetBatcher refused(entry, no_features, 2, 0.75f, false, 0, 19);
    Check(refused.GetNumSamples() == 0 && refused.GetFeatureRows() == 0,
          "AudioDatasetBatcher refuses audio without a feature node");

    cyxwiz::AudioPreprocessingConfig preprocessing;
    preprocessing.has_feature_node = true;
    preprocessing.feature_type = cyxwiz::AudioFeatureType::Spectrogram;
    preprocessing.n_fft = 16;
    preprocessing.hop_length = 8;
    cyxwiz::AudioDatasetBatcher batcher(
        entry, preprocessing, 2, 0.75f, false, 0, 19);
    CheckTrainOnlyDropLast(batcher, "AudioDatasetBatcher");

    cyxwiz::AudioDatasetBatcher first_seeded(
        entry, preprocessing, 2, 0.75f, true, 0, 1901);
    cyxwiz::AudioDatasetBatcher second_seeded(
        entry, preprocessing, 2, 0.75f, true, 0, 1901);
    CheckMatchingPayloads(CollectEpochPayload(first_seeded),
                          CollectEpochPayload(second_seeded),
                          "AudioDatasetBatcher");
}

} // namespace

int main() {
    const auto root = fs::temp_directory_path() /
        "cyxwiz_modality_batcher_drop_last";
    fs::remove_all(root);
    fs::create_directories(root);

    TestImageBatcher(root);
    TestGrayscaleImages(root);
    TestAudioBatcher(root);

    fs::remove_all(root);
    std::cout << "Image/audio batch-boundary and seed contracts passed\n";
    return 0;
}
