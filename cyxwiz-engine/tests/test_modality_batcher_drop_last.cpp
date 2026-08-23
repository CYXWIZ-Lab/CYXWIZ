#include "../src/core/audio_dataset_batcher.h"
#include "../src/core/image_dataset_batcher.h"

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

void WriteBmp24(const fs::path& path, uint8_t red, uint8_t green, uint8_t blue) {
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
        for (uint32_t column = 0; column < width; ++column) {
            out.put(static_cast<char>(blue));
            out.put(static_cast<char>(green));
            out.put(static_cast<char>(red));
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

    cyxwiz::ImageDatasetBatcher batcher(
        entry, preprocessing, 2, 0.75f, false, 0, 17);
    CheckTrainOnlyDropLast(batcher, "ImageDatasetBatcher");

    cyxwiz::ImageDatasetBatcher first_seeded(
        entry, preprocessing, 2, 0.75f, true, 0, 1701);
    cyxwiz::ImageDatasetBatcher second_seeded(
        entry, preprocessing, 2, 0.75f, true, 0, 1701);
    CheckMatchingPayloads(CollectEpochPayload(first_seeded),
                          CollectEpochPayload(second_seeded),
                          "ImageDatasetBatcher");
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
    entry.feature_type = 0;
    entry.target_sr = 8000;
    entry.n_fft = 16;
    entry.hop_length = 8;
    entry.max_duration = 0.016f;
    entry.num_samples = 4;
    entry.num_classes = 2;

    cyxwiz::AudioPreprocessingConfig preprocessing;
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
    TestAudioBatcher(root);

    fs::remove_all(root);
    std::cout << "Image/audio batch-boundary and seed contracts passed\n";
    return 0;
}
