# OpenCV Integration Architecture

## Overview

OpenCV (Open Source Computer Vision Library) is integrated into the **CyxWiz Engine** as an optional dependency for advanced image processing, data augmentation, and computer vision operations. This document describes the architecture, integration points, and usage patterns.

## Architecture Decision

### Placement: Engine vs Backend

**Decision: OpenCV is placed in `cyxwiz-engine`, NOT `cyxwiz-backend`**

| Factor | Engine (Chosen) | Backend |
|--------|----------------|---------|
| **Primary Use** | Data loading, preprocessing, augmentation | ML computation |
| **Existing Image I/O** | STB library (texture_manager.cpp) | ArrayFire tensors |
| **Data Flow** | Loads images → preprocesses → sends to backend | Receives tensors for training |
| **GPU Library** | OpenGL (rendering) | ArrayFire (compute) |
| **Conversion Overhead** | None (stays in CPU/OpenCV space) | Would need OpenCV↔ArrayFire conversion |

### Rationale

1. **Data Pipeline Ownership**: The Engine owns the data loading pipeline (`data_registry.cpp`, `dataset_panel.cpp`)
2. **Augmentation Location**: Phase 4 augmentation system will be in Engine, feeding processed data to Backend
3. **No Tensor Conversion**: Avoids costly `cv::Mat` ↔ `af::array` conversions
4. **Optional Dependency**: Engine can fall back to STB when OpenCV is unavailable

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CyxWiz Engine                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────┐ │
│  │   Dataset Panel     │───▶│   Image Loader      │───▶│  Augmentation   │ │
│  │  (dataset_panel.cpp)│    │                     │    │    Pipeline     │ │
│  └─────────────────────┘    │  ┌───────────────┐  │    │                 │ │
│                             │  │ CYXWIZ_HAS_   │  │    │  ┌───────────┐  │ │
│                             │  │   OPENCV      │  │    │  │  Resize   │  │ │
│                             │  │    ▼          │  │    │  │  Crop     │  │ │
│                             │  │ ┌─────────┐   │  │    │  │  Flip     │  │ │
│                             │  │ │ OpenCV  │   │  │    │  │  Rotate   │  │ │
│                             │  │ │ imread  │   │  │    │  │  Color    │  │ │
│                             │  │ └─────────┘   │  │    │  │  Noise    │  │ │
│                             │  │    │         │  │    │  └───────────┘  │ │
│                             │  │    ▼ else    │  │    │        │        │ │
│                             │  │ ┌─────────┐   │  │    │        ▼        │ │
│                             │  │ │   STB   │   │  │    │  ┌───────────┐  │ │
│                             │  │ │  image  │   │  │    │  │ Normalize │  │ │
│                             │  │ └─────────┘   │  │    │  │ To Tensor │  │ │
│                             │  └───────────────┘  │    │  └───────────┘  │ │
│                             └─────────────────────┘    └────────┬────────┘ │
│                                                                  │          │
│  ┌─────────────────────────────────────────────────────────────▼────────┐ │
│  │                         Data Registry                                 │ │
│  │                      (data_registry.cpp)                              │ │
│  │                                                                       │ │
│  │   Manages loaded datasets, caching, and data access                   │ │
│  └───────────────────────────────────────┬───────────────────────────────┘ │
│                                          │                                  │
└──────────────────────────────────────────┼──────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CyxWiz Backend                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────┐ │
│  │      Tensor         │───▶│       Model         │───▶│    Training     │ │
│  │   (ArrayFire)       │    │    (Layers, etc.)   │    │     Loop        │ │
│  └─────────────────────┘    └─────────────────────┘    └─────────────────┘ │
│                                                                              │
│  Note: Backend receives normalized float arrays, no OpenCV dependency        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Build System Integration

### CMakeLists.txt Configuration

```cmake
# cyxwiz-engine/CMakeLists.txt

# Optional: OpenCV for image processing and augmentation
find_package(OpenCV CONFIG)
if(OpenCV_FOUND)
    message(STATUS "OpenCV ${OpenCV_VERSION} found - Image processing enabled")
    set(CYXWIZ_HAS_OPENCV ON)
else()
    message(WARNING "OpenCV not found - Image augmentation disabled (using STB fallback)")
    set(CYXWIZ_HAS_OPENCV OFF)
endif()

# Later in the file...
if(OpenCV_FOUND)
    target_include_directories(cyxwiz-engine PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(cyxwiz-engine PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(cyxwiz-engine PRIVATE CYXWIZ_HAS_OPENCV)
endif()
```

### vcpkg.json Dependency

```json
{
  "name": "opencv4",
  "default-features": true,
  "platform": "!android"
}
```

### OpenCV Modules Available (via vcpkg default features)

| Module | Purpose in CyxWiz |
|--------|-------------------|
| `core` | Basic data structures (cv::Mat) |
| `imgproc` | Image transformations, filtering, color conversion |
| `imgcodecs` | Image file I/O (PNG, JPEG, TIFF, WebP) |
| `highgui` | Window display (for debugging) |
| `dnn` | Deep neural network inference (optional) |
| `calib3d` | Camera calibration (future: 3D dataset support) |

## Code Integration Patterns

### 1. Conditional Compilation Pattern

```cpp
// image_loader.h
#pragma once

#ifdef CYXWIZ_HAS_OPENCV
#include <opencv2/opencv.hpp>
#endif

#include <vector>
#include <string>

namespace cyxwiz {

class ImageLoader {
public:
    // Returns image data as float vector [0.0, 1.0]
    // Shape: [height, width, channels]
    static bool LoadImage(const std::string& path,
                          std::vector<float>& data,
                          int& width, int& height, int& channels);

    // Resize image to target dimensions
    static bool ResizeImage(std::vector<float>& data,
                            int src_width, int src_height, int channels,
                            int dst_width, int dst_height);

private:
#ifdef CYXWIZ_HAS_OPENCV
    static bool LoadWithOpenCV(const std::string& path,
                               std::vector<float>& data,
                               int& width, int& height, int& channels);
#endif
    static bool LoadWithSTB(const std::string& path,
                            std::vector<float>& data,
                            int& width, int& height, int& channels);
};

} // namespace cyxwiz
```

### 2. Image Loading Implementation

```cpp
// image_loader.cpp
#include "image_loader.h"

#ifdef CYXWIZ_HAS_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#else
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#endif

namespace cyxwiz {

bool ImageLoader::LoadImage(const std::string& path,
                            std::vector<float>& data,
                            int& width, int& height, int& channels) {
#ifdef CYXWIZ_HAS_OPENCV
    return LoadWithOpenCV(path, data, width, height, channels);
#else
    return LoadWithSTB(path, data, width, height, channels);
#endif
}

#ifdef CYXWIZ_HAS_OPENCV
bool ImageLoader::LoadWithOpenCV(const std::string& path,
                                 std::vector<float>& data,
                                 int& width, int& height, int& channels) {
    cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        return false;
    }

    // Convert to float [0, 1]
    cv::Mat float_img;
    img.convertTo(float_img, CV_32F, 1.0 / 255.0);

    // Convert BGR to RGB if needed
    if (float_img.channels() == 3) {
        cv::cvtColor(float_img, float_img, cv::COLOR_BGR2RGB);
    } else if (float_img.channels() == 4) {
        cv::cvtColor(float_img, float_img, cv::COLOR_BGRA2RGBA);
    }

    width = float_img.cols;
    height = float_img.rows;
    channels = float_img.channels();

    // Flatten to vector (row-major, interleaved channels)
    data.resize(width * height * channels);
    std::memcpy(data.data(), float_img.data,
                data.size() * sizeof(float));

    return true;
}
#endif

bool ImageLoader::LoadWithSTB(const std::string& path,
                              std::vector<float>& data,
                              int& width, int& height, int& channels) {
    unsigned char* img_data = stbi_load(path.c_str(),
                                        &width, &height, &channels, 0);
    if (!img_data) {
        return false;
    }

    // Convert to float [0, 1]
    size_t size = width * height * channels;
    data.resize(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = img_data[i] / 255.0f;
    }

    stbi_image_free(img_data);
    return true;
}

bool ImageLoader::ResizeImage(std::vector<float>& data,
                              int src_width, int src_height, int channels,
                              int dst_width, int dst_height) {
#ifdef CYXWIZ_HAS_OPENCV
    // Create cv::Mat from data (no copy)
    cv::Mat src(src_height, src_width,
                channels == 1 ? CV_32FC1 :
                channels == 3 ? CV_32FC3 : CV_32FC4,
                data.data());

    cv::Mat dst;
    cv::resize(src, dst, cv::Size(dst_width, dst_height),
               0, 0, cv::INTER_LINEAR);

    data.resize(dst_width * dst_height * channels);
    std::memcpy(data.data(), dst.data, data.size() * sizeof(float));
    return true;
#else
    // Simple bilinear interpolation fallback
    // ... (basic implementation without OpenCV)
    return false; // Not implemented without OpenCV
#endif
}

} // namespace cyxwiz
```

### 3. Data Augmentation Pipeline

```cpp
// augmentation.h
#pragma once

#ifdef CYXWIZ_HAS_OPENCV
#include <opencv2/opencv.hpp>
#endif

#include <vector>
#include <random>

namespace cyxwiz {

struct AugmentationConfig {
    bool enabled = true;

    // Geometric transforms
    bool random_flip_horizontal = true;
    bool random_flip_vertical = false;
    float random_rotation_degrees = 15.0f;  // Max rotation angle
    float random_scale_min = 0.8f;
    float random_scale_max = 1.2f;

    // Color transforms
    float brightness_delta = 0.2f;   // ± adjustment
    float contrast_delta = 0.2f;     // ± adjustment
    float saturation_delta = 0.2f;   // ± adjustment
    float hue_delta = 0.1f;          // ± adjustment

    // Noise
    float gaussian_noise_std = 0.05f;

    // Crop
    bool random_crop = false;
    int crop_width = 224;
    int crop_height = 224;
};

class Augmentation {
public:
    explicit Augmentation(const AugmentationConfig& config);

    // Apply augmentation pipeline to image data
    // Input/Output: float data in [0, 1] range
    bool Apply(std::vector<float>& data,
               int width, int height, int channels,
               int& out_width, int& out_height);

    // Check if augmentation is available
    static bool IsAvailable();

private:
    AugmentationConfig config_;
    std::mt19937 rng_;

#ifdef CYXWIZ_HAS_OPENCV
    cv::Mat ApplyGeometricTransforms(const cv::Mat& img);
    cv::Mat ApplyColorTransforms(const cv::Mat& img);
    cv::Mat ApplyNoise(const cv::Mat& img);
    cv::Mat ApplyCrop(const cv::Mat& img);
#endif
};

} // namespace cyxwiz
```

```cpp
// augmentation.cpp
#include "augmentation.h"
#include <spdlog/spdlog.h>

namespace cyxwiz {

Augmentation::Augmentation(const AugmentationConfig& config)
    : config_(config)
    , rng_(std::random_device{}()) {
}

bool Augmentation::IsAvailable() {
#ifdef CYXWIZ_HAS_OPENCV
    return true;
#else
    return false;
#endif
}

bool Augmentation::Apply(std::vector<float>& data,
                         int width, int height, int channels,
                         int& out_width, int& out_height) {
#ifdef CYXWIZ_HAS_OPENCV
    if (!config_.enabled) {
        out_width = width;
        out_height = height;
        return true;
    }

    // Create cv::Mat from data
    int cv_type = channels == 1 ? CV_32FC1 :
                  channels == 3 ? CV_32FC3 : CV_32FC4;
    cv::Mat img(height, width, cv_type, data.data());

    // Apply transforms in order
    cv::Mat result = img.clone();

    result = ApplyGeometricTransforms(result);
    result = ApplyColorTransforms(result);
    result = ApplyNoise(result);

    if (config_.random_crop) {
        result = ApplyCrop(result);
    }

    // Copy back to data vector
    out_width = result.cols;
    out_height = result.rows;
    data.resize(out_width * out_height * channels);
    std::memcpy(data.data(), result.data, data.size() * sizeof(float));

    return true;
#else
    spdlog::warn("Augmentation requires OpenCV. Build with CYXWIZ_HAS_OPENCV.");
    out_width = width;
    out_height = height;
    return false;
#endif
}

#ifdef CYXWIZ_HAS_OPENCV

cv::Mat Augmentation::ApplyGeometricTransforms(const cv::Mat& img) {
    cv::Mat result = img.clone();

    // Random horizontal flip
    if (config_.random_flip_horizontal) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        if (dist(rng_) > 0.5f) {
            cv::flip(result, result, 1);  // Horizontal flip
        }
    }

    // Random vertical flip
    if (config_.random_flip_vertical) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        if (dist(rng_) > 0.5f) {
            cv::flip(result, result, 0);  // Vertical flip
        }
    }

    // Random rotation
    if (config_.random_rotation_degrees > 0) {
        std::uniform_real_distribution<float> dist(
            -config_.random_rotation_degrees,
            config_.random_rotation_degrees);
        float angle = dist(rng_);

        cv::Point2f center(result.cols / 2.0f, result.rows / 2.0f);
        cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::warpAffine(result, result, rot_mat, result.size(),
                       cv::INTER_LINEAR, cv::BORDER_REFLECT);
    }

    // Random scale
    if (config_.random_scale_min < config_.random_scale_max) {
        std::uniform_real_distribution<float> dist(
            config_.random_scale_min,
            config_.random_scale_max);
        float scale = dist(rng_);

        int new_width = static_cast<int>(result.cols * scale);
        int new_height = static_cast<int>(result.rows * scale);
        cv::resize(result, result, cv::Size(new_width, new_height));
    }

    return result;
}

cv::Mat Augmentation::ApplyColorTransforms(const cv::Mat& img) {
    if (img.channels() < 3) {
        return img;  // Skip for grayscale
    }

    cv::Mat result = img.clone();

    // Random brightness
    if (config_.brightness_delta > 0) {
        std::uniform_real_distribution<float> dist(
            -config_.brightness_delta,
            config_.brightness_delta);
        float delta = dist(rng_);
        result += cv::Scalar(delta, delta, delta);
    }

    // Random contrast
    if (config_.contrast_delta > 0) {
        std::uniform_real_distribution<float> dist(
            1.0f - config_.contrast_delta,
            1.0f + config_.contrast_delta);
        float factor = dist(rng_);
        result = (result - 0.5f) * factor + 0.5f;
    }

    // Random saturation and hue (convert to HSV)
    if (config_.saturation_delta > 0 || config_.hue_delta > 0) {
        cv::Mat hsv;
        // Scale to 0-255 for cvtColor
        cv::Mat temp;
        result.convertTo(temp, CV_8UC3, 255.0);
        cv::cvtColor(temp, hsv, cv::COLOR_RGB2HSV);
        hsv.convertTo(hsv, CV_32FC3, 1.0 / 255.0);

        std::vector<cv::Mat> hsv_channels;
        cv::split(hsv, hsv_channels);

        // Adjust hue
        if (config_.hue_delta > 0) {
            std::uniform_real_distribution<float> dist(
                -config_.hue_delta, config_.hue_delta);
            hsv_channels[0] += dist(rng_);
        }

        // Adjust saturation
        if (config_.saturation_delta > 0) {
            std::uniform_real_distribution<float> dist(
                1.0f - config_.saturation_delta,
                1.0f + config_.saturation_delta);
            hsv_channels[1] *= dist(rng_);
        }

        cv::merge(hsv_channels, hsv);
        hsv.convertTo(temp, CV_8UC3, 255.0);
        cv::cvtColor(temp, temp, cv::COLOR_HSV2RGB);
        temp.convertTo(result, CV_32FC3, 1.0 / 255.0);
    }

    // Clamp to [0, 1]
    cv::threshold(result, result, 0.0, 0.0, cv::THRESH_TOZERO);
    cv::threshold(result, result, 1.0, 1.0, cv::THRESH_TRUNC);

    return result;
}

cv::Mat Augmentation::ApplyNoise(const cv::Mat& img) {
    if (config_.gaussian_noise_std <= 0) {
        return img;
    }

    cv::Mat result = img.clone();
    cv::Mat noise(img.size(), img.type());

    std::normal_distribution<float> dist(0.0f, config_.gaussian_noise_std);
    for (int i = 0; i < noise.rows; ++i) {
        float* ptr = noise.ptr<float>(i);
        for (int j = 0; j < noise.cols * noise.channels(); ++j) {
            ptr[j] = dist(rng_);
        }
    }

    result += noise;

    // Clamp to [0, 1]
    cv::threshold(result, result, 0.0, 0.0, cv::THRESH_TOZERO);
    cv::threshold(result, result, 1.0, 1.0, cv::THRESH_TRUNC);

    return result;
}

cv::Mat Augmentation::ApplyCrop(const cv::Mat& img) {
    if (img.cols <= config_.crop_width || img.rows <= config_.crop_height) {
        // Image smaller than crop size, resize instead
        cv::Mat result;
        cv::resize(img, result,
                   cv::Size(config_.crop_width, config_.crop_height));
        return result;
    }

    // Random crop position
    std::uniform_int_distribution<int> dist_x(
        0, img.cols - config_.crop_width);
    std::uniform_int_distribution<int> dist_y(
        0, img.rows - config_.crop_height);

    int x = dist_x(rng_);
    int y = dist_y(rng_);

    return img(cv::Rect(x, y, config_.crop_width, config_.crop_height)).clone();
}

#endif // CYXWIZ_HAS_OPENCV

} // namespace cyxwiz
```

## Integration with Dataset Panel

```cpp
// In dataset_panel.cpp

void DatasetPanel::LoadImageDataset(const std::string& path) {
    // Check if OpenCV is available for advanced features
    if (Augmentation::IsAvailable()) {
        spdlog::info("OpenCV available - Advanced augmentation enabled");
        show_augmentation_options_ = true;
    } else {
        spdlog::info("OpenCV not available - Using basic image loading");
        show_augmentation_options_ = false;
    }

    // Load images...
}

void DatasetPanel::RenderAugmentationOptions() {
#ifdef CYXWIZ_HAS_OPENCV
    if (ImGui::CollapsingHeader("Data Augmentation")) {
        ImGui::Checkbox("Enable Augmentation", &aug_config_.enabled);

        if (aug_config_.enabled) {
            ImGui::Separator();
            ImGui::Text("Geometric Transforms");
            ImGui::Checkbox("Random Horizontal Flip", &aug_config_.random_flip_horizontal);
            ImGui::Checkbox("Random Vertical Flip", &aug_config_.random_flip_vertical);
            ImGui::SliderFloat("Max Rotation (degrees)", &aug_config_.random_rotation_degrees, 0, 45);
            ImGui::SliderFloat("Scale Min", &aug_config_.random_scale_min, 0.5f, 1.0f);
            ImGui::SliderFloat("Scale Max", &aug_config_.random_scale_max, 1.0f, 2.0f);

            ImGui::Separator();
            ImGui::Text("Color Transforms");
            ImGui::SliderFloat("Brightness Delta", &aug_config_.brightness_delta, 0, 0.5f);
            ImGui::SliderFloat("Contrast Delta", &aug_config_.contrast_delta, 0, 0.5f);
            ImGui::SliderFloat("Saturation Delta", &aug_config_.saturation_delta, 0, 0.5f);
            ImGui::SliderFloat("Hue Delta", &aug_config_.hue_delta, 0, 0.5f);

            ImGui::Separator();
            ImGui::Text("Noise");
            ImGui::SliderFloat("Gaussian Noise Std", &aug_config_.gaussian_noise_std, 0, 0.2f);

            ImGui::Separator();
            ImGui::Checkbox("Random Crop", &aug_config_.random_crop);
            if (aug_config_.random_crop) {
                ImGui::InputInt("Crop Width", &aug_config_.crop_width);
                ImGui::InputInt("Crop Height", &aug_config_.crop_height);
            }
        }
    }
#else
    ImGui::TextDisabled("Augmentation requires OpenCV (not available)");
#endif
}
```

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Image Dataset Loading Pipeline                        │
└──────────────────────────────────────────────────────────────────────────────┘

  ┌─────────┐     ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
  │ File    │────▶│ Image       │────▶│ Augmentation │────▶│ Normalization   │
  │ System  │     │ Decoder     │     │ Pipeline     │     │ & Tensor Conv   │
  └─────────┘     └─────────────┘     └──────────────┘     └─────────────────┘
       │               │                    │                      │
       │               │                    │                      │
       ▼               ▼                    ▼                      ▼
  ┌─────────┐     ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
  │ .png    │     │ #ifdef      │     │ Flip, Rotate │     │ Scale to [0,1]  │
  │ .jpg    │     │ OPENCV:     │     │ Scale, Crop  │     │ CHW/HWC format  │
  │ .bmp    │     │   cv::imread│     │ Color adjust │     │ Batch creation  │
  │ .tiff   │     │ #else:      │     │ Noise        │     │                 │
  │ .webp   │     │   stbi_load │     │              │     │                 │
  └─────────┘     └─────────────┘     └──────────────┘     └─────────────────┘
                                             │
                                             │ Only with OpenCV
                                             ▼
                                      ┌──────────────┐
                                      │ Without      │
                                      │ OpenCV:      │
                                      │ Pass-through │
                                      └──────────────┘
```

## Performance Considerations

### Memory Management

```cpp
// Efficient batch processing with OpenCV
class BatchProcessor {
public:
    // Process images in batches to manage memory
    void ProcessBatch(const std::vector<std::string>& paths,
                      int batch_size,
                      std::function<void(std::vector<float>&)> callback) {
#ifdef CYXWIZ_HAS_OPENCV
        std::vector<cv::Mat> batch;
        batch.reserve(batch_size);

        for (size_t i = 0; i < paths.size(); ++i) {
            cv::Mat img = cv::imread(paths[i]);
            if (!img.empty()) {
                // Process and add to batch
                batch.push_back(ProcessSingle(img));
            }

            if (batch.size() >= batch_size || i == paths.size() - 1) {
                // Convert batch to tensor and call callback
                auto tensor = BatchToTensor(batch);
                callback(tensor);
                batch.clear();
            }
        }
#endif
    }
};
```

### Thread Safety

```cpp
// Thread-local augmentation for parallel data loading
class ParallelDataLoader {
public:
    void LoadParallel(const std::vector<std::string>& paths, int num_threads) {
#ifdef CYXWIZ_HAS_OPENCV
        #pragma omp parallel num_threads(num_threads)
        {
            // Each thread has its own augmentation instance
            Augmentation aug(aug_config_);

            #pragma omp for
            for (size_t i = 0; i < paths.size(); ++i) {
                ProcessImage(paths[i], aug);
            }
        }
#endif
    }
};
```

## Future Extensions

### 1. Video Support
```cpp
#ifdef CYXWIZ_HAS_OPENCV
// Future: Video dataset support
class VideoDataset {
    cv::VideoCapture capture_;

    bool Open(const std::string& path) {
        return capture_.open(path);
    }

    bool ReadFrame(std::vector<float>& data) {
        cv::Mat frame;
        if (!capture_.read(frame)) return false;
        // Convert to tensor...
        return true;
    }
};
#endif
```

### 2. Camera Input
```cpp
#ifdef CYXWIZ_HAS_OPENCV
// Future: Live camera input for inference
class CameraInput {
    cv::VideoCapture camera_;

    bool Open(int device_id = 0) {
        return camera_.open(device_id);
    }
};
#endif
```

### 3. Advanced Augmentation (Albumentations-style)
```cpp
// Future: Composable augmentation pipeline
class AugmentationPipeline {
    std::vector<std::unique_ptr<AugmentationOp>> ops_;

public:
    AugmentationPipeline& Add(std::unique_ptr<AugmentationOp> op) {
        ops_.push_back(std::move(op));
        return *this;
    }

    cv::Mat Apply(const cv::Mat& input) {
        cv::Mat result = input;
        for (auto& op : ops_) {
            result = op->Apply(result);
        }
        return result;
    }
};
```

## Troubleshooting

### Build Issues

**"OpenCV not found"**
```bash
# Ensure vcpkg installed OpenCV
cmake --preset windows-release  # Will install via vcpkg
# Or manually:
vcpkg install opencv4:x64-windows
```

**"CYXWIZ_HAS_OPENCV not defined"**
- Check CMake output for "OpenCV found" message
- Verify `find_package(OpenCV CONFIG)` succeeds
- Check vcpkg toolchain is configured

### Runtime Issues

**"Augmentation not available"**
- Built without OpenCV support
- Use `Augmentation::IsAvailable()` to check

**"cv::imread returns empty"**
- File path incorrect
- Unsupported image format
- File permissions

## References

- [OpenCV Documentation](https://docs.opencv.org/)
- [vcpkg OpenCV Port](https://github.com/microsoft/vcpkg/tree/master/ports/opencv4)
- [cv::Mat Memory Layout](https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html)
