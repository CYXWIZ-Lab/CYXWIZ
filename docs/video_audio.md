# Video and Audio Data Processing in CyxWiz

This document outlines how ML systems handle video and audio data, and the planned approach for implementing this functionality in CyxWiz.

## Table of Contents
1. [Video Data Processing](#video-data-processing)
2. [Audio Data Processing](#audio-data-processing)
3. [Implementation Architecture](#implementation-architecture)
4. [Libraries and Dependencies](#libraries-and-dependencies)
5. [Implementation Phases](#implementation-phases)
6. [API Design](#api-design)

---

## Video Data Processing

### The Standard Pipeline

```
Video File → Decode → Frame Extraction → Preprocessing → Tensor
   (MP4)     (FFmpeg)   (Sample frames)   (Resize, norm)   (N,C,H,W)
```

### Frame Sampling Approaches

| Approach | Description | Use Case |
|----------|-------------|----------|
| **Uniform Sampling** | Extract every Nth frame | Action recognition, video classification |
| **Keyframe Extraction** | Extract I-frames only | Efficient scene understanding |
| **Dense Sampling** | Extract all frames | Optical flow, fine-grained tasks |
| **Clip-based** | Extract short clips (e.g., 16 frames) | Video transformers, 3D CNNs |

### How Major ML Frameworks Handle Video

#### PyTorch
- Uses `torchvision.io.read_video()` which wraps FFmpeg
- Alternative: `decord` library for GPU-accelerated decoding
- Returns tensors of shape `[T, H, W, C]` (time, height, width, channels)

```python
# PyTorch example
import torchvision.io as io

video, audio, info = io.read_video("video.mp4", pts_unit='sec')
# video shape: [T, H, W, C]
```

#### TensorFlow
- Uses `tf.io.decode_video()` or preprocessing layers
- Can use `tf.data` pipeline for efficient loading

```python
# TensorFlow example
import tensorflow as tf

video = tf.io.read_file("video.mp4")
video = tf.io.decode_video(video)
```

### Video Tensor Formats

| Format | Shape | Description |
|--------|-------|-------------|
| THWC | `[T, H, W, C]` | Time-first (common in PyTorch) |
| CTHW | `[C, T, H, W]` | Channel-first (for 3D CNNs) |
| BTHWC | `[B, T, H, W, C]` | Batched format |
| BCTHW | `[B, C, T, H, W]` | Batched channel-first |

---

## Audio Data Processing

### The Standard Pipeline

```
Audio File → Decode → Resampling → Feature Extraction → Tensor
   (WAV)     (FFmpeg)  (16kHz)      (Mel-spectrogram)   (Freq,Time)
```

### Audio Feature Types

| Feature | Shape | Description | Use Case |
|---------|-------|-------------|----------|
| **Raw Waveform** | `[1, samples]` | Unprocessed audio samples | WaveNet, raw audio models |
| **Spectrogram** | `[freq_bins, time]` | STFT magnitude | General audio analysis |
| **Mel-Spectrogram** | `[n_mels, time]` | Mel-scaled spectrogram | Speech recognition, music |
| **MFCC** | `[n_mfcc, time]` | Mel-frequency cepstral coefficients | Traditional speech processing |
| **Log-Mel** | `[n_mels, time]` | Log-scaled mel-spectrogram | Deep learning audio models |

### Common Audio Parameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| Sample Rate | 16000, 22050, 44100 Hz | Samples per second |
| n_fft | 1024, 2048 | FFT window size |
| hop_length | 512 | Samples between frames |
| n_mels | 80, 128 | Number of mel bands |
| n_mfcc | 13, 40 | Number of MFCC coefficients |

### How Major ML Frameworks Handle Audio

#### PyTorch (torchaudio)
```python
import torchaudio
import torchaudio.transforms as T

waveform, sample_rate = torchaudio.load("audio.wav")

# Convert to mel-spectrogram
mel_transform = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=1024,
    hop_length=512,
    n_mels=80
)
mel_spec = mel_transform(waveform)
```

#### TensorFlow
```python
import tensorflow as tf

audio = tf.io.read_file("audio.wav")
audio, sample_rate = tf.audio.decode_wav(audio)

# STFT for spectrogram
stft = tf.signal.stft(audio, frame_length=1024, frame_step=512)
spectrogram = tf.abs(stft)
```

---

## Implementation Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Dataset Panel (UI)                        │
├─────────────────────────────────────────────────────────────┤
│  [Load Video Dataset]  [Load Audio Dataset]  [Load Images]   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Options:                                                 │ │
│  │ - Sample Rate: [every N frames]                         │ │
│  │ - Max Frames: [limit per video]                         │ │
│  │ - Resize: [target resolution]                           │ │
│  │ - Normalize: [yes/no]                                   │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              VideoDataset / AudioDataset                     │
├─────────────────────────────────────────────────────────────┤
│  Members:                                                    │
│  - file_paths: vector<string>    // List of media files     │
│  - labels: vector<int>           // Class labels            │
│  - sample_rate: int              // Frames per video        │
│  - transforms: TransformPipeline // Preprocessing chain     │
│  - cache: LRUCache               // Decoded frame cache     │
│                                                              │
│  Methods:                                                    │
│  - LoadFile(path) -> Tensor                                  │
│  - GetItem(index) -> (Tensor, Label)                        │
│  - GetBatch(indices) -> (BatchTensor, Labels)               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    MediaLoader (Backend)                     │
├─────────────────────────────────────────────────────────────┤
│  VideoLoader:                                                │
│  - DecodeVideo(path) -> vector<Frame>                       │
│  - ExtractFrames(path, sample_rate) -> vector<Tensor>       │
│  - GetMetadata(path) -> VideoInfo                           │
│                                                              │
│  AudioLoader:                                                │
│  - DecodeAudio(path) -> Waveform                            │
│  - Resample(waveform, target_rate) -> Waveform              │
│  - GetMetadata(path) -> AudioInfo                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    DataLoader (Batching)                     │
├─────────────────────────────────────────────────────────────┤
│  - batch_size: int                                           │
│  - shuffle: bool                                             │
│  - num_workers: int        // Parallel decoding threads     │
│  - prefetch_factor: int    // Batches to prefetch           │
│  - drop_last: bool         // Drop incomplete last batch    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                     Output Tensor Shapes:
                     Video: [B, T, C, H, W] or [B, C, T, H, W]
                     Audio: [B, C, T] or [B, n_mels, T]
```

### Transform Pipeline

#### Video Transforms

```cpp
namespace transforms {

// Spatial transforms (applied per frame)
class VideoResize {
    int target_height, target_width;
    InterpolationMode mode;  // BILINEAR, NEAREST, BICUBIC
};

class VideoCenterCrop {
    int crop_height, crop_width;
};

class VideoRandomCrop {
    int crop_height, crop_width;
};

class VideoNormalize {
    std::vector<float> mean;  // e.g., {0.485, 0.456, 0.406}
    std::vector<float> std;   // e.g., {0.229, 0.224, 0.225}
};

class VideoRandomHorizontalFlip {
    float probability;  // e.g., 0.5
};

// Temporal transforms
class TemporalUniformSample {
    int num_frames;  // Number of frames to sample
};

class TemporalRandomSample {
    int num_frames;
};

class TemporalCrop {
    int start_frame, num_frames;
};

class TemporalJitter {
    int max_jitter;  // Max frames to shift
};

// Color transforms
class ColorJitter {
    float brightness, contrast, saturation, hue;
};

class Grayscale { };

}  // namespace transforms
```

#### Audio Transforms

```cpp
namespace transforms {

// Waveform transforms
class AudioResample {
    int original_rate, target_rate;
};

class AudioNormalize {
    float target_db;  // e.g., -3.0 dB
};

class AudioPadOrTruncate {
    int target_length;  // In samples
};

// Feature extraction
class MelSpectrogram {
    int sample_rate;
    int n_fft;
    int hop_length;
    int n_mels;
    float f_min, f_max;
};

class MFCC {
    int sample_rate;
    int n_mfcc;
    int n_fft;
    int hop_length;
};

class Spectrogram {
    int n_fft;
    int hop_length;
    bool power;  // true for power spectrogram
};

// Augmentation (SpecAugment)
class TimeMask {
    int max_mask_length;
    int num_masks;
};

class FrequencyMask {
    int max_mask_length;
    int num_masks;
};

class TimeStretch {
    float min_rate, max_rate;
};

class PitchShift {
    int min_semitones, max_semitones;
};

}  // namespace transforms
```

---

## Libraries and Dependencies

### Option A: FFmpeg (Recommended)

**Pros:**
- Industry standard, used by all major applications
- Supports virtually all video/audio formats
- Hardware acceleration support (NVDEC, VAAPI, etc.)
- Well-documented, stable

**Cons:**
- Large dependency
- Complex API
- Licensing considerations (LGPL/GPL depending on features)

**CMake Integration:**
```cmake
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED
    libavcodec
    libavformat
    libavutil
    libswscale
    libswresample
)

target_link_libraries(cyxwiz-backend
    ${FFMPEG_LIBRARIES}
)
```

### Option B: OpenCV VideoCapture

**Pros:**
- Simpler API
- Already common in CV projects
- Cross-platform

**Cons:**
- Wraps FFmpeg anyway
- Less control over decoding
- May not support all formats

**CMake Integration:**
```cmake
find_package(OpenCV REQUIRED COMPONENTS videoio)
target_link_libraries(cyxwiz-backend ${OpenCV_LIBS})
```

### Option C: Decord (For Python bindings)

**Pros:**
- GPU-accelerated decoding
- Python-native
- Efficient random access

**Cons:**
- Primarily Python-focused
- Smaller community

### Option D: Pre-extracted Frames

**Pros:**
- No video decoding complexity
- Simpler implementation
- Works with existing image loading code

**Cons:**
- Requires preprocessing step
- More storage space needed
- Less flexible

---

## Implementation Phases

### Phase 1: Foundation (Video Loading)

**Goals:**
- Add FFmpeg or OpenCV dependency to CMake
- Create `VideoLoader` class for basic frame extraction
- Support MP4, AVI, MKV formats

**Files to Create:**
```
cyxwiz-backend/
├── include/cyxwiz/
│   └── video_loader.h
└── src/io/
    └── video_loader.cpp
```

**Basic API:**
```cpp
class VideoLoader {
public:
    struct VideoInfo {
        int width, height;
        int num_frames;
        double fps;
        double duration;
        std::string codec;
    };

    VideoInfo GetInfo(const std::string& path);
    std::vector<Tensor> LoadFrames(
        const std::string& path,
        int sample_rate = 1,    // Every Nth frame
        int max_frames = -1,    // -1 = all
        int start_frame = 0
    );
    Tensor LoadSingleFrame(const std::string& path, int frame_index);
};
```

### Phase 2: Dataset Integration

**Goals:**
- Create `VideoDataset` class extending base Dataset
- Add video loading options to Dataset Panel UI
- Implement basic transforms (resize, normalize)

**Files to Create:**
```
cyxwiz-backend/
├── include/cyxwiz/
│   └── video_dataset.h
└── src/data/
    └── video_dataset.cpp

cyxwiz-engine/
└── src/gui/panels/
    └── dataset_panel.cpp  (modify)
```

**Dataset API:**
```cpp
class VideoDataset : public Dataset {
public:
    struct Config {
        std::string root_path;
        int frames_per_video = 16;
        int resize_height = 224;
        int resize_width = 224;
        bool normalize = true;
        std::vector<float> mean = {0.485f, 0.456f, 0.406f};
        std::vector<float> std = {0.229f, 0.224f, 0.225f};
    };

    VideoDataset(const Config& config);

    size_t Size() const override;
    std::pair<Tensor, int> GetItem(size_t index) override;
    Tensor GetBatch(const std::vector<size_t>& indices) override;
};
```

### Phase 3: Audio Support

**Goals:**
- Add audio decoding (FFmpeg or libsndfile)
- Create `AudioLoader` and `AudioDataset` classes
- Implement spectrogram/MFCC transforms

**Files to Create:**
```
cyxwiz-backend/
├── include/cyxwiz/
│   ├── audio_loader.h
│   ├── audio_dataset.h
│   └── audio_transforms.h
└── src/
    ├── io/
    │   └── audio_loader.cpp
    ├── data/
    │   └── audio_dataset.cpp
    └── transforms/
        └── audio_transforms.cpp
```

**Audio API:**
```cpp
class AudioLoader {
public:
    struct AudioInfo {
        int sample_rate;
        int num_channels;
        int num_samples;
        double duration;
        std::string format;
    };

    AudioInfo GetInfo(const std::string& path);
    Tensor LoadWaveform(
        const std::string& path,
        int target_sample_rate = -1,  // -1 = original
        int max_duration_ms = -1      // -1 = full
    );
};

class AudioDataset : public Dataset {
public:
    struct Config {
        std::string root_path;
        int sample_rate = 16000;
        int max_duration_ms = 10000;
        FeatureType feature_type = FeatureType::MelSpectrogram;
        int n_mels = 80;
        int n_fft = 1024;
        int hop_length = 512;
    };

    AudioDataset(const Config& config);
    // ... similar to VideoDataset
};
```

### Phase 4: Advanced Features

**Goals:**
- GPU-accelerated decoding (NVIDIA Video Codec SDK)
- Efficient caching and prefetching
- Video/audio preview in UI
- Multi-worker parallel loading

**Caching System:**
```cpp
class MediaCache {
public:
    struct CacheConfig {
        size_t max_memory_bytes = 1024 * 1024 * 1024;  // 1GB
        bool enable_disk_cache = false;
        std::string disk_cache_path;
    };

    void Put(const std::string& key, const Tensor& data);
    std::optional<Tensor> Get(const std::string& key);
    void Clear();
    size_t GetMemoryUsage() const;
};
```

**Multi-worker Loading:**
```cpp
class ParallelDataLoader {
public:
    struct Config {
        int num_workers = 4;
        int prefetch_factor = 2;
        int batch_size = 32;
        bool pin_memory = true;  // For GPU transfer
    };

    ParallelDataLoader(Dataset* dataset, const Config& config);

    BatchIterator begin();
    BatchIterator end();
};
```

---

## API Design

### Complete Video Pipeline Example

```cpp
// 1. Create video dataset
VideoDataset::Config config;
config.root_path = "/path/to/videos";
config.frames_per_video = 16;
config.resize_height = 224;
config.resize_width = 224;

auto dataset = std::make_shared<VideoDataset>(config);

// 2. Add transforms
auto transforms = TransformPipeline()
    .Add<VideoRandomCrop>(224, 224)
    .Add<VideoRandomHorizontalFlip>(0.5f)
    .Add<VideoNormalize>(
        {0.485f, 0.456f, 0.406f},  // ImageNet mean
        {0.229f, 0.224f, 0.225f}   // ImageNet std
    );

dataset->SetTransforms(transforms);

// 3. Create data loader
DataLoader::Config loader_config;
loader_config.batch_size = 8;
loader_config.shuffle = true;
loader_config.num_workers = 4;

auto loader = std::make_shared<DataLoader>(dataset, loader_config);

// 4. Training loop
for (auto& [batch, labels] : *loader) {
    // batch shape: [8, 16, 3, 224, 224] = [B, T, C, H, W]
    // labels shape: [8]

    auto output = model->Forward(batch);
    auto loss = criterion->Compute(output, labels);
    // ...
}
```

### Complete Audio Pipeline Example

```cpp
// 1. Create audio dataset
AudioDataset::Config config;
config.root_path = "/path/to/audio";
config.sample_rate = 16000;
config.max_duration_ms = 10000;
config.feature_type = FeatureType::MelSpectrogram;
config.n_mels = 80;

auto dataset = std::make_shared<AudioDataset>(config);

// 2. Add transforms
auto transforms = TransformPipeline()
    .Add<AudioNormalize>(-3.0f)  // Normalize to -3dB
    .Add<MelSpectrogram>(16000, 1024, 512, 80)
    .Add<TimeMask>(30, 2)   // SpecAugment
    .Add<FrequencyMask>(15, 2);

dataset->SetTransforms(transforms);

// 3. Create data loader and train
auto loader = std::make_shared<DataLoader>(dataset, loader_config);

for (auto& [batch, labels] : *loader) {
    // batch shape: [B, n_mels, T] = [32, 80, 313]
    // ...
}
```

---

## File Format Support

### Video Formats

| Format | Container | Codecs | Priority |
|--------|-----------|--------|----------|
| MP4 | MPEG-4 Part 14 | H.264, H.265, VP9 | High |
| AVI | Audio Video Interleave | Various | High |
| MKV | Matroska | H.264, H.265, VP8/9 | High |
| MOV | QuickTime | H.264, ProRes | Medium |
| WebM | WebM | VP8, VP9, AV1 | Medium |
| FLV | Flash Video | H.264, VP6 | Low |

### Audio Formats

| Format | Description | Priority |
|--------|-------------|----------|
| WAV | Uncompressed PCM | High |
| MP3 | MPEG Audio Layer III | High |
| FLAC | Free Lossless Audio Codec | High |
| OGG | Ogg Vorbis | Medium |
| AAC | Advanced Audio Coding | Medium |
| M4A | MPEG-4 Audio | Medium |

---

## Performance Considerations

### Video Loading Optimization

1. **Seek Efficiency**: Use keyframe-based seeking for random access
2. **Batch Decoding**: Decode multiple frames in sequence when possible
3. **Memory Mapping**: Use memory-mapped I/O for large files
4. **Hardware Decoding**: Leverage GPU decoders (NVDEC, VAAPI)

### Audio Loading Optimization

1. **Streaming**: Stream large audio files instead of loading entirely
2. **Resampling**: Use efficient resampling algorithms (libsamplerate)
3. **FFT Optimization**: Use optimized FFT libraries (FFTW, MKL)

### Caching Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    Caching Hierarchy                         │
├─────────────────────────────────────────────────────────────┤
│  L1: GPU Memory Cache (fastest, limited)                    │
│      - Recently used batches                                │
│      - Size: ~2-4 GB                                        │
├─────────────────────────────────────────────────────────────┤
│  L2: CPU Memory Cache (fast, larger)                        │
│      - Decoded frames/spectrograms                          │
│      - Size: ~8-32 GB                                       │
├─────────────────────────────────────────────────────────────┤
│  L3: Disk Cache (slower, largest)                           │
│      - Pre-extracted frames as images                       │
│      - Pre-computed spectrograms as .npy                    │
│      - Size: Unlimited                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## References

- [PyTorch Video Tutorial](https://pytorch.org/tutorials/beginner/video_tutorial.html)
- [TorchAudio Documentation](https://pytorch.org/audio/stable/)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [Decord GitHub](https://github.com/dmlc/decord)
- [SpecAugment Paper](https://arxiv.org/abs/1904.08779)
- [Video Transformer Survey](https://arxiv.org/abs/2106.02624)

---

*Document created: 2024*
*Last updated: 2024*
