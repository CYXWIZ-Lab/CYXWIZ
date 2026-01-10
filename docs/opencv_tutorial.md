# OpenCV Integration Tutorial

Practical examples and usage patterns for CyxWiz OpenCV features.

## Quick Start: Run the Tests

```bash
# Build the test executable
cmake --build build --config Release --target test_opencv_integration

# Run with output directory to see generated images
./build/bin/Release/test_opencv_integration.exe ./test_output

# Check results
ls test_output/
# original.png, morph_erode.png, blur_gaussian.png, etc.
```

---

## Phase 1: Preprocessing Transforms

Use these during training data preparation to augment or clean images.

### Morphology: Remove Noise, Find Edges

```cpp
#include "preprocessing/morphology_transform.h"

// Your image: float vector [0,1], HWC format
std::vector<float> image = LoadYourImage();
int w = 128, h = 128, c = 1;  // grayscale

// Remove small bright spots (salt noise)
cyxwiz::MorphologyTransform::Open(image, w, h, c, 3);

// Fill small holes (pepper noise)
cyxwiz::MorphologyTransform::Close(image, w, h, c, 3);

// Find edges (useful for edge detection datasets)
cyxwiz::MorphologyTransform::Gradient(image, w, h, c, 3);
```

**When to use:**
- `Erode/Dilate`: Shrink/grow binary masks
- `Open`: Remove small bright spots, separate touching objects
- `Close`: Fill small holes, connect nearby objects
- `Gradient`: Edge detection, boundary finding

### Blur: Reduce Noise

```cpp
#include "preprocessing/blur_transform.h"

std::vector<float> image = LoadYourImage();
int w = 256, h = 256, c = 3;  // RGB

// Standard denoising (fast, good quality)
cyxwiz::BlurTransform::GaussianBlur(image, w, h, c, 5, 1.5f);

// Remove salt-and-pepper noise
cyxwiz::BlurTransform::MedianBlur(image, w, h, c, 5);

// Edge-preserving smoothing (slow but best quality)
cyxwiz::BlurTransform::BilateralFilter(image, w, h, c, 9, 75.0f, 75.0f);
```

**When to use:**
- `Gaussian`: General denoising, pre-processing before edge detection
- `Median`: Salt-and-pepper noise, preserves edges better than Gaussian
- `Bilateral`: High-quality denoising for final outputs, slow

### Perspective: Geometric Augmentation

```cpp
#include "preprocessing/perspective_transform.h"

std::vector<float> image = LoadYourImage();
int w = 224, h = 224, c = 3;

// Rotate for augmentation
cyxwiz::PerspectiveTransform::Rotate(image, w, h, c, 15.0f, 1.0f, true);

// Log-polar for rotation-invariant features
cyxwiz::PerspectiveTransform::LogPolar(image, w, h, c, 0.5f, 0.5f, 0.4f);
```

**When to use:**
- `Rotate`: Data augmentation, correcting tilted images
- `LogPolar`: Rotation-invariant feature extraction, radial analysis

### Pyramid: Multi-Scale Processing

```cpp
#include "preprocessing/pyramid_transform.h"

std::vector<float> image = LoadYourImage();
int w = 512, h = 512, c = 3;

// Downsample for faster processing
cyxwiz::PyramidTransform::PyrDown(image, w, h, c);
// Now w=256, h=256

// Downsample multiple times
cyxwiz::PyramidTransform::PyrDownMultiple(image, w, h, c, 3);
// Now w=64, h=64
```

**When to use:**
- `PyrDown`: Create image thumbnails, speed up processing
- `PyrUp`: Upscale for display (note: doesn't recover lost detail)

---

## Phase 2: Dataset Utilities

Tools for analyzing and preparing datasets.

### Connected Components: Count Objects

```cpp
#include "utils/segmentation_utils.h"

// Binary mask (float, 0=background, 1=foreground)
std::vector<float> binary_mask = GetBinaryMask();
int w = 128, h = 128;

std::vector<int32_t> labels;
auto components = cyxwiz::SegmentationUtils::FindConnectedComponents(
    binary_mask, w, h, 8, &labels);

std::cout << "Found " << components.size() << " objects\n";

for (const auto& comp : components) {
    std::cout << "Object " << comp.label
              << ": " << comp.area << " pixels"
              << " at (" << comp.centroid_x << "," << comp.centroid_y << ")\n";
}
```

### Flood Fill: Magic Wand Selection

```cpp
#include "utils/segmentation_utils.h"

std::vector<float> image = LoadYourImage();
int w = 256, h = 256, c = 3;

// Click at (100, 100), tolerance 0.1 for low/high diff
auto mask = cyxwiz::SegmentationUtils::FloodFillMask(
    image, w, h, c,
    100, 100,      // seed point
    0.1f, 0.1f     // lo_diff, hi_diff
);

// mask is uint8_t: 255 = selected, 0 = not selected
```

### Shape Analysis: Measure Objects

```cpp
#include "utils/shape_utils.h"

// Contour points from edge detection or segmentation
std::vector<cyxwiz::Point2D> contour = GetContour();

// Get bounding rectangle
auto rect = cyxwiz::ShapeUtils::MinAreaRect(contour);
std::cout << "Size: " << rect.width << "x" << rect.height
          << " angle: " << rect.angle << "\n";

// Compute shape descriptors
auto desc = cyxwiz::ShapeUtils::ComputeDescriptors(contour);
std::cout << "Circularity: " << desc.circularity << "\n";
std::cout << "Aspect ratio: " << desc.aspect_ratio << "\n";
std::cout << "Solidity: " << desc.solidity << "\n";

// Compare two shapes
double similarity = cyxwiz::ShapeUtils::MatchShapes(contour1, contour2, 1);
```

### Contour Extraction: Find Object Boundaries

```cpp
#include "utils/labeling_utils.h"

// Binary mask (float, 0-1)
std::vector<float> mask = GetBinaryMask();
int w = 128, h = 128;

auto contours = cyxwiz::LabelingUtils::ExtractContours(
    mask, w, h,
    cyxwiz::ContourMode::External,  // Only outer boundaries
    cyxwiz::ContourApprox::Simple   // Compress straight lines
);

for (const auto& contour : contours) {
    std::cout << "Contour with " << contour.points.size() << " points\n";

    // Convert to bounding box
    auto bbox = cyxwiz::LabelingUtils::ContourToBBox(contour.points, false);
    std::cout << "BBox: " << bbox.x << "," << bbox.y
              << " " << bbox.width << "x" << bbox.height << "\n";
}
```

### IoU Calculation: Evaluate Detection Quality

```cpp
#include "utils/labeling_utils.h"

cyxwiz::BoundingBox prediction;
prediction.x = 50; prediction.y = 50;
prediction.width = 100; prediction.height = 100;

cyxwiz::BoundingBox ground_truth;
ground_truth.x = 60; ground_truth.y = 55;
ground_truth.width = 90; ground_truth.height = 95;

float iou = cyxwiz::LabelingUtils::CalculateIoU(prediction, ground_truth);
std::cout << "IoU: " << iou << "\n";  // ~0.65

// Non-maximum suppression for multiple detections
std::vector<cyxwiz::BoundingBox> detections = GetDetections();
auto kept = cyxwiz::LabelingUtils::NonMaxSuppression(detections, 0.5f);
```

---

## Phase 3: DNN Inference

Pre-trained model inference using OpenCV DNN.

### Face Detection

```cpp
#include "inference/dnn_models.h"

cyxwiz::DNNModel model;
model.Load("models/res10_300x300_ssd.caffemodel",
           "models/deploy.prototxt",
           cyxwiz::DNNModelType::FaceDetector);

// Optional: Use GPU
model.SetBackendAndTarget(cyxwiz::DNNBackend::CUDA, cyxwiz::DNNTarget::CUDA);

std::vector<float> image = LoadYourImage();
auto faces = model.DetectFaces(image, w, h, c, 0.5f);

for (const auto& face : faces) {
    std::cout << "Face at (" << face.x << "," << face.y << ") "
              << face.width << "x" << face.height
              << " conf: " << face.confidence << "\n";
}
```

### Object Detection (YOLO)

```cpp
#include "inference/dnn_models.h"

cyxwiz::DNNModelInfo info;
info.type = cyxwiz::DNNModelType::YOLOv4;
info.model_path = "models/yolov4.weights";
info.config_path = "models/yolov4.cfg";
info.labels_path = "models/coco.names";
info.input_width = 416;
info.input_height = 416;

cyxwiz::DNNModel model;
model.Load(info);

auto detections = model.Detect(image, w, h, c, 0.5f, 0.4f);

for (const auto& det : detections) {
    std::cout << det.class_name << " (" << det.confidence << "): "
              << det.x << "," << det.y << " "
              << det.width << "x" << det.height << "\n";
}
```

---

## Phase 4: Interactive Tools (GUI)

These are used in the Dataset Panel's Interactive tab.

### Using in Code (Advanced)

```cpp
#include "gui/annotation/grabcut_editor.h"

// Create editor
cyxwiz::GrabCutEditor editor;
editor.SetImage(image_data, w, h, 3);

// Simulate rectangle selection
editor.OnMouseDown(50, 50, cyxwiz::MouseButton::Left);
editor.OnMouseMove(200, 200, true);
editor.OnMouseUp(200, 200, cyxwiz::MouseButton::Left);

// Run GrabCut
editor.RunGrabCut(5);

// Get result mask
auto mask = editor.GetFloatMask();
```

### GUI Usage

1. **Dataset Panel** > **Interactive** tab
2. Load image: Click "Load from Dataset (Preview)" or "Load Test Image"
3. Select tool: GrabCut, Watershed, Brush, or Polygon
4. Annotate the image
5. Export: Set filename, click "Export Mask"

---

## Common Patterns

### Preprocessing Pipeline

```cpp
void PreprocessForTraining(std::vector<float>& img, int& w, int& h, int c) {
    // 1. Denoise
    cyxwiz::BlurTransform::GaussianBlur(img, w, h, c, 3, 0.5f);

    // 2. Resize to model input size
    cyxwiz::ImageUtils::ResizeImage(img, w, h, c, 224, 224);
    w = 224; h = 224;

    // 3. Optional: Morphology cleanup for segmentation
    if (c == 1) {
        cyxwiz::MorphologyTransform::Open(img, w, h, c, 3);
    }
}
```

### Dataset Quality Check

```cpp
void AnalyzeDataset(const std::vector<std::string>& image_paths) {
    for (const auto& path : image_paths) {
        std::vector<float> img;
        int w, h, c;
        cyxwiz::ImageUtils::LoadImage(path, img, w, h, c);

        // Convert to grayscale for analysis
        if (c == 3) {
            cyxwiz::ImageUtils::ConvertColorSpace(img, w, h,
                cyxwiz::ImageUtils::ColorSpace::RGB,
                cyxwiz::ImageUtils::ColorSpace::Grayscale);
            c = 1;
        }

        // Find objects
        std::vector<int32_t> labels;
        auto components = cyxwiz::SegmentationUtils::FindConnectedComponents(
            img, w, h, 8, &labels);

        std::cout << path << ": " << components.size() << " objects\n";
    }
}
```

---

## Troubleshooting

### "OpenCV not found"
```bash
# Ensure OpenCV is installed via vcpkg
./vcpkg/vcpkg install opencv4
```

### "Model file not found" (DNN)
- Download model files separately (not included in repo)
- Place in `models/` directory
- Check paths in DNNModelInfo

### Slow bilateral filter
- Bilateral is O(n*k^2) - use smaller `d` parameter
- For real-time, use Gaussian instead

### Memory issues with large images
- Use PyramidTransform::PyrDown to reduce resolution first
- Process in tiles for very large images

---

## Test Output Reference

After running `test_opencv_integration.exe ./test_output`:

| File | What it shows |
|------|---------------|
| `original.png` | Test image with circle and gradient |
| `morph_erode.png` | Circle shrunk by erosion |
| `morph_dilate.png` | Circle expanded by dilation |
| `morph_gradient.png` | Circle edges only |
| `blur_gaussian.png` | Smooth blur |
| `blur_median.png` | Blocky blur (good for noise) |
| `blur_bilateral.png` | Smooth blur with sharp edges |
| `perspective_rotate45.png` | Image rotated 45 degrees |
| `perspective_logpolar.png` | Polar coordinate transform |
| `pyramid_down.png` | 64x64 downsampled |
| `pyramid_up.png` | 128x128 upsampled back |
| `segment_flood.png` | Flood fill from center |
| `label_mask.png` | Mask from extracted contour |
