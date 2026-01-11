# OpenCV Integration Guide for CyxWiz Engine

This document covers all OpenCV-based features integrated into CyxWiz Engine across 4 phases.

---

## Table of Contents

1. [Phase 1: Preprocessing Transforms](#phase-1-preprocessing-transforms)
   - [Morphology Transform](#morphology-transform)
   - [Blur Transform](#blur-transform)
   - [Perspective Transform](#perspective-transform)
   - [Pyramid Transform](#pyramid-transform)
2. [Phase 2: Dataset Utilities](#phase-2-dataset-utilities)
   - [Segmentation Utils](#segmentation-utils)
   - [Shape Utils](#shape-utils)
   - [Labeling Utils](#labeling-utils)
3. [Phase 3: DNN Inference](#phase-3-dnn-inference)
   - [DNN Models](#dnn-models)
   - [DNN Preprocessor](#dnn-preprocessor)
4. [Phase 4: Interactive Annotation Tools](#phase-4-interactive-annotation-tools)
   - [Annotation Tools](#annotation-tools)
   - [GrabCut Editor](#grabcut-editor)
   - [Watershed Editor](#watershed-editor)

---

## Phase 1: Preprocessing Transforms

Location: `cyxwiz-engine/src/preprocessing/`

These transforms are applied during the data preprocessing pipeline for training.

### Morphology Transform

**File**: `morphology_transform.h/cpp`

Morphological operations process images based on shapes, useful for noise removal, element extraction, and edge detection.

#### Operations

| Operation | Description | Use Case |
|-----------|-------------|----------|
| `Erode` | Shrinks bright regions | Remove small bright spots, separate connected objects |
| `Dilate` | Expands bright regions | Fill small holes, connect nearby objects |
| `Open` | Erosion → Dilation | Remove small bright spots while preserving shape |
| `Close` | Dilation → Erosion | Close small dark holes while preserving shape |
| `Gradient` | Dilation - Erosion | Edge detection, finding object boundaries |
| `TopHat` | Source - Opening | Extract small bright details on dark background |
| `BlackHat` | Closing - Source | Extract small dark details on bright background |

#### Kernel Shapes

- `Rect` - Rectangular kernel (default)
- `Cross` - Cross-shaped kernel
- `Ellipse` - Elliptical kernel

#### API Usage

```cpp
#include "preprocessing/morphology_transform.h"

// Using config
cyxwiz::MorphologyConfig config;
config.enabled = true;
config.operation = cyxwiz::MorphOperation::Open;
config.shape = cyxwiz::MorphShape::Ellipse;
config.kernel_size = 5;  // Must be odd
config.iterations = 1;

std::vector<float> image_data = /* your image */;
cyxwiz::MorphologyTransform::Apply(image_data, width, height, channels, config);

// Direct method
cyxwiz::MorphologyTransform::Erode(image_data, width, height, channels, 3, 1);
cyxwiz::MorphologyTransform::Dilate(image_data, width, height, channels, 5, 2);
cyxwiz::MorphologyTransform::Open(image_data, width, height, channels, 7);
```

#### GUI Usage

1. Go to **Dataset Panel** → **Data Pipeline** tab
2. Enable **Morphology Operations** section
3. Select operation, kernel shape, size (3-21, odd), and iterations
4. Preview changes in real-time

---

### Blur Transform

**File**: `blur_transform.h/cpp`

Blur operations for noise reduction and smoothing.

#### Blur Types

| Type | Description | Best For |
|------|-------------|----------|
| `Box` | Simple averaging | Fast uniform blur |
| `Gaussian` | Weighted by Gaussian | Natural-looking blur (default) |
| `Median` | Median value in neighborhood | Salt-and-pepper noise removal |
| `Bilateral` | Edge-preserving smoothing | High-quality denoising (slow) |

#### API Usage

```cpp
#include "preprocessing/blur_transform.h"

// Using config
cyxwiz::BlurConfig config;
config.enabled = true;
config.type = cyxwiz::BlurType::Gaussian;
config.kernel_size = 5;  // Must be odd
config.sigma = 0.0f;     // 0 = auto-calculate

cyxwiz::BlurTransform::Apply(image_data, width, height, channels, config);

// Direct methods
cyxwiz::BlurTransform::GaussianBlur(image_data, w, h, c, 5, 1.5f);
cyxwiz::BlurTransform::MedianBlur(image_data, w, h, c, 5);
cyxwiz::BlurTransform::BilateralFilter(image_data, w, h, c, 9, 75.0f, 75.0f);
```

#### GUI Usage

1. Go to **Dataset Panel** → **Data Pipeline** tab
2. Enable **Blur / Smoothing** section
3. Select blur type and kernel size
4. For bilateral: adjust sigma_color and sigma_space

---

### Perspective Transform

**File**: `perspective_transform.h/cpp`

Geometric transformations including perspective warp and polar coordinates.

#### Transform Types

| Type | Description | Use Case |
|------|-------------|----------|
| `Perspective` | 4-point homography warp | Document scanning, augmentation |
| `Affine` | 3-point affine warp | Preserves parallel lines |
| `LogPolar` | Log-polar coordinates | Rotation-invariant features |
| `LinearPolar` | Linear-polar coordinates | Radial analysis |

#### API Usage

```cpp
#include "preprocessing/perspective_transform.h"

// Perspective warp
std::array<float, 8> src_pts = {0,0, 1,0, 1,1, 0,1};  // Normalized corners
std::array<float, 8> dst_pts = {0.1f,0.1f, 0.9f,0, 1,1, 0,0.9f};
cyxwiz::PerspectiveTransform::WarpPerspective(image_data, w, h, c, src_pts, dst_pts);

// Rotation
cyxwiz::PerspectiveTransform::Rotate(image_data, w, h, c, 45.0f, 1.0f, true);

// Log-polar transform
cyxwiz::PerspectiveTransform::LogPolar(image_data, w, h, c, 0.5f, 0.5f, 0.5f);

// Auto-detect document corners
auto corners = cyxwiz::PerspectiveTransform::DetectDocumentCorners(image_data, w, h, c);
```

#### GUI Usage

1. Go to **Dataset Panel** → **Data Pipeline** tab
2. Enable **Perspective Transform** section
3. Select transform type
4. For perspective: use corner point controls or "Auto-Detect Corners"
5. For polar: set center point and max radius

---

### Pyramid Transform

**File**: `pyramid_transform.h/cpp`

Multi-scale image representations for efficient processing.

#### Operations

| Operation | Description | Output Size |
|-----------|-------------|-------------|
| `PyrDown` | Gaussian blur + subsample | width/2, height/2 |
| `PyrUp` | Upsample + Gaussian blur | width*2, height*2 |
| `BuildGaussian` | Get specific Gaussian level | Depends on level |
| `BuildLaplacian` | Get high-frequency details | Same as Gaussian level |

#### API Usage

```cpp
#include "preprocessing/pyramid_transform.h"

// Downsample once
cyxwiz::PyramidTransform::PyrDown(image_data, w, h, c);
// Now w = w/2, h = h/2

// Upsample once
cyxwiz::PyramidTransform::PyrUp(image_data, w, h, c);

// Downsample multiple times
cyxwiz::PyramidTransform::PyrDownMultiple(image_data, w, h, c, 3);

// Get Laplacian level (high-frequency details)
cyxwiz::PyramidTransform::BuildLaplacianLevel(image_data, w, h, c, 2);
```

#### GUI Usage

1. Go to **Dataset Panel** → **Data Pipeline** tab
2. Enable **Image Pyramids** section
3. Select operation and number of levels

---

## Phase 2: Dataset Utilities

Location: `cyxwiz-engine/src/utils/`

Utility classes for dataset analysis, segmentation, and labeling assistance.

### Segmentation Utils

**File**: `segmentation_utils.h/cpp`

Tools for automatic and semi-automatic segmentation.

#### Features

- **Watershed** - Marker-based region growing
- **GrabCut** - Interactive foreground extraction
- **Connected Components** - Find and analyze connected regions
- **Flood Fill** - Magic wand selection

#### API Usage

```cpp
#include "utils/segmentation_utils.h"

// Watershed segmentation
std::vector<int32_t> markers = /* initial markers, 0=unknown, >0=region ID */;
std::vector<int32_t> output_labels;
auto regions = cyxwiz::SegmentationUtils::Watershed(
    image_data, w, h, 3, markers, output_labels
);

// GrabCut with rectangle
auto mask = cyxwiz::SegmentationUtils::GrabCut(
    image_data, w, h, 3,
    rect_x, rect_y, rect_w, rect_h,
    5  // iterations
);

// Find connected components
std::vector<int32_t> labels;
auto components = cyxwiz::SegmentationUtils::FindConnectedComponents(
    binary_mask, w, h, 8, &labels
);

// Filter by area
int remaining = cyxwiz::SegmentationUtils::FilterComponentsByArea(
    binary_mask, w, h, 100, 10000  // min_area, max_area
);

// Flood fill mask
auto filled_mask = cyxwiz::SegmentationUtils::FloodFillMask(
    image_data, w, h, c, seed_x, seed_y, 0.1f, 0.1f
);
```

---

### Shape Utils

**File**: `shape_utils.h/cpp`

Shape analysis and geometric detection.

#### Features

- **Convex Hull** - Compute convex hull of contour
- **Ellipse/Rectangle Fitting** - Fit shapes to contours
- **Hough Lines/Circles** - Detect lines and circles
- **Shape Descriptors** - Circularity, aspect ratio, solidity
- **Moments** - Hu moments for shape matching

#### API Usage

```cpp
#include "utils/shape_utils.h"

std::vector<cyxwiz::Point2D> contour = /* your contour */;

// Convex hull
auto hull = cyxwiz::ShapeUtils::ConvexHull(contour);

// Fit ellipse (needs >= 5 points)
auto ellipse = cyxwiz::ShapeUtils::FitEllipse(contour);

// Minimum area rectangle
auto rect = cyxwiz::ShapeUtils::MinAreaRect(contour);

// Detect lines in edge image
auto lines = cyxwiz::ShapeUtils::HoughLines(
    edge_image, w, h,
    1.0,    // rho resolution
    0.0174, // theta resolution (~1 degree)
    50,     // threshold
    50.0,   // min line length
    10.0,   // max gap
    true    // probabilistic
);

// Detect circles
auto circles = cyxwiz::ShapeUtils::HoughCircles(
    gray_image, w, h,
    1.0, 20.0, 100.0, 30.0, 10, 100
);

// Shape descriptors
auto desc = cyxwiz::ShapeUtils::ComputeDescriptors(contour);
// desc.circularity, desc.aspect_ratio, desc.solidity, etc.

// Hu moments
auto moments = cyxwiz::ShapeUtils::ComputeMoments(contour);

// Match shapes
double similarity = cyxwiz::ShapeUtils::MatchShapes(contour1, contour2, 1);
```

---

### Labeling Utils

**File**: `labeling_utils.h/cpp`

Tools for dataset annotation and labeling.

#### Features

- **Contour Extraction** - Extract contours with hierarchy
- **Region Proposals** - Selective Search, Edge Boxes
- **Mask Operations** - Create, dilate, erode, fill holes
- **Bounding Boxes** - Convert between formats, NMS, IoU
- **RLE Encoding** - Run-length encoding for masks

#### API Usage

```cpp
#include "utils/labeling_utils.h"

// Extract contours
auto contours = cyxwiz::LabelingUtils::ExtractContours(
    binary_mask, w, h,
    cyxwiz::ContourMode::External,
    cyxwiz::ContourApprox::Simple
);

// Generate region proposals
auto proposals = cyxwiz::LabelingUtils::SelectiveSearch(
    image_data, w, h, 3, "fast"
);

// Create mask from contour
auto mask = cyxwiz::LabelingUtils::CreateMaskFromContour(
    contour.points, w, h, true  // filled
);

// Convert contour to bounding box
auto bbox = cyxwiz::LabelingUtils::ContourToBBox(contour.points, false);

// Non-maximum suppression
auto kept = cyxwiz::LabelingUtils::NonMaxSuppression(boxes, 0.5f);

// Calculate IoU
float iou = cyxwiz::LabelingUtils::CalculateIoU(box1, box2);

// Mask operations
cyxwiz::LabelingUtils::DilateMask(mask, w, h, 3, 1);
cyxwiz::LabelingUtils::FillHoles(mask, w, h);
cyxwiz::LabelingUtils::SmoothMaskBoundary(mask, w, h, 5, 0.5f);

// RLE encoding
auto rle = cyxwiz::LabelingUtils::PolygonToRLE(contour, w, h);
auto decoded_mask = cyxwiz::LabelingUtils::RLEToMask(rle, w, h);
```

---

## Phase 3: DNN Inference

Location: `cyxwiz-engine/src/inference/`

OpenCV DNN module integration for pre-trained model inference.

### DNN Models

**File**: `dnn_models.h/cpp`

Unified interface for various pre-trained models.

#### Supported Model Types

| Type | Description | Models |
|------|-------------|--------|
| `FaceDetector` | Face detection | res10_300x300 SSD |
| `YOLOv4` | Object detection | YOLOv4, YOLOv4-tiny |
| `MobileNetSSD` | Lightweight detection | MobileNet SSD |
| `OpenPose` | Body pose estimation | OpenPose COCO/BODY_25 |
| `AgeGender` | Age/gender classification | - |
| `TextDetection` | Text detection | EAST detector |
| `Custom` | User-provided model | ONNX, Caffe, TensorFlow |

#### Backends and Targets

**Backends**: Default, OpenCV, CUDA, OpenCL, Vulkan, InferenceEngine (OpenVINO)

**Targets**: CPU, OpenCL, OpenCL_FP16, CUDA, CUDA_FP16, Vulkan, FPGA

#### API Usage

```cpp
#include "inference/dnn_models.h"

cyxwiz::DNNModel model;

// Load pre-configured model
model.Load("path/to/model.caffemodel", "path/to/config.prototxt",
           cyxwiz::DNNModelType::FaceDetector);

// Or load with full info
cyxwiz::DNNModelInfo info;
info.type = cyxwiz::DNNModelType::YOLOv4;
info.model_path = "yolov4.weights";
info.config_path = "yolov4.cfg";
info.labels_path = "coco.names";
info.input_width = 416;
info.input_height = 416;
model.Load(info);

// Set backend/target
model.SetBackendAndTarget(cyxwiz::DNNBackend::CUDA, cyxwiz::DNNTarget::CUDA);

// Object detection
auto detections = model.Detect(image_data, w, h, c, 0.5f, 0.4f);
for (const auto& det : detections) {
    printf("Class: %s, Conf: %.2f, Box: (%.2f, %.2f, %.2f, %.2f)\n",
           det.class_name.c_str(), det.confidence,
           det.x, det.y, det.width, det.height);
}

// Face detection
auto faces = model.DetectFaces(image_data, w, h, c, 0.5f);

// Pose estimation
auto poses = model.EstimatePose(image_data, w, h, c, 0.1f);

// Classification
auto results = model.Classify(image_data, w, h, c, 5);

// Get inference time
double ms = model.GetInferenceTimeMs();
```

#### Model Registry

```cpp
// Register built-in models
cyxwiz::DNNModelRegistry::Instance().RegisterBuiltins();
cyxwiz::DNNModelRegistry::Instance().SetModelsDirectory("models/");

// Check available models
for (const auto& name : cyxwiz::DNNModelRegistry::Instance().GetModelNames()) {
    bool available = cyxwiz::DNNModelRegistry::Instance().IsAvailable(name);
    printf("%s: %s\n", name.c_str(), available ? "Available" : "Missing");
}

// Get model info
const auto* info = cyxwiz::DNNModelRegistry::Instance().Get("face_detector");
```

---

### DNN Preprocessor

**File**: `dnn_preprocessor.h/cpp`

Image preprocessing for neural network inference.

#### Features

- Resizing (bilinear, letterbox, center crop)
- Normalization (mean subtraction, std division, scaling)
- Color space conversion (RGB ↔ BGR)
- Data layout conversion (HWC → CHW)
- Coordinate mapping for detections

#### API Usage

```cpp
#include "inference/dnn_preprocessor.h"

// Get preset config
auto config = cyxwiz::DNNPreprocessor::YOLOConfig(416);
// Or: ImageNetConfig(224), SSDConfig(300), FaceDetectorConfig(), OpenPoseConfig(368)

// Custom config
cyxwiz::DNNPreprocessConfig config;
config.input_width = 224;
config.input_height = 224;
config.scale_factor = 1.0f / 255.0f;
config.mean = {0.485f, 0.456f, 0.406f};  // ImageNet mean
config.std_dev = {0.229f, 0.224f, 0.225f};  // ImageNet std
config.swap_rb = true;
config.keep_aspect_ratio = true;  // Letterbox
config.channels_first = true;  // NCHW format

// Preprocess
auto blob = cyxwiz::DNNPreprocessor::Preprocess(image_data, w, h, c, config);

// blob.data is ready for DNN inference
// blob.original_width, blob.original_height for coordinate mapping

// Map detection back to original coordinates
auto [orig_x, orig_y] = cyxwiz::DNNPreprocessor::MapToOriginal(det_x, det_y, blob);
auto box = cyxwiz::DNNPreprocessor::MapBoxToOriginal(x, y, w, h, blob);
```

---

## Phase 4: Interactive Annotation Tools

Location: `cyxwiz-engine/src/gui/annotation/`

Interactive tools for dataset annotation in the GUI.

### Annotation Tools

**File**: `annotation_tools.h/cpp`

Base framework for interactive annotation.

#### Tool Types

| Tool | Description | Use Case |
|------|-------------|----------|
| `RectangleSelect` | Draw rectangle selection | GrabCut initialization, cropping |
| `BrushPaint` | Paint with circular brush | Manual mask painting |
| `BrushErase` | Erase with brush | Refine masks |
| `PolygonDraw` | Draw polygon vertices | Precise object boundaries |
| `PointMarker` | Place point markers | Watershed initialization |
| `FloodFill` | Magic wand selection | Similar region selection |
| `Lasso` | Freehand selection | Quick selections |

#### Annotation Labels

```cpp
enum class AnnotationLabel {
    Unknown = 0,           // Not labeled
    Background = 1,        // Definite background
    Foreground = 2,        // Definite foreground
    ProbableBackground = 3, // For GrabCut refinement
    ProbableForeground = 4  // For GrabCut refinement
};
```

#### API Usage

```cpp
#include "gui/annotation/annotation_tools.h"

// Create brush tool
cyxwiz::BrushTool brush;
brush.SetImage(image_data, w, h, c);
brush.SetBrushRadius(15.0f);
brush.SetCurrentLabel(cyxwiz::AnnotationLabel::Foreground);

// Handle mouse events
brush.OnMouseDown(x, y, cyxwiz::MouseButton::Left);
brush.OnMouseMove(x, y, true);  // dragging
brush.OnMouseUp(x, y, cyxwiz::MouseButton::Left);

// Get mask
const auto& mask = brush.GetMask();
auto float_mask = mask.ToFloatMask();

// Undo/redo
if (brush.CanUndo()) brush.Undo();
if (brush.CanRedo()) brush.Redo();

// Rectangle selection
cyxwiz::RectangleSelectTool rect;
// ... mouse events ...
if (rect.HasSelection()) {
    auto sel = rect.GetSelection();
    // sel.x, sel.y, sel.width, sel.height
}

// Point markers
cyxwiz::PointMarkerTool markers;
// ... mouse events ...
for (const auto& marker : markers.GetMarkers()) {
    // marker.position, marker.marker_id, marker.label
}
```

---

### GrabCut Editor

**File**: `grabcut_editor.h/cpp`

Interactive foreground extraction using OpenCV GrabCut.

#### Workflow

1. **Rectangle Mode**: Draw rectangle around object
2. **Run GrabCut**: Initial segmentation (5 iterations)
3. **Refinement Mode**: Paint foreground/background strokes
4. **Refine**: Iterative refinement

#### API Usage

```cpp
#include "gui/annotation/grabcut_editor.h"

cyxwiz::GrabCutEditor editor;
editor.SetImage(image_data, w, h, 3);

// Set callback for segmentation updates
editor.SetSegmentationCallback([](const cyxwiz::AnnotationMask& mask) {
    // Handle updated mask
});

// In Rectangle mode, draw rectangle
editor.OnMouseDown(x1, y1, cyxwiz::MouseButton::Left);
editor.OnMouseMove(x2, y2, true);
editor.OnMouseUp(x2, y2, cyxwiz::MouseButton::Left);

// Run initial GrabCut
if (editor.HasRectangle()) {
    editor.RunGrabCut(5);  // 5 iterations
}

// Switch to refinement mode
editor.SetMode(cyxwiz::GrabCutMode::Refinement);
editor.SetCurrentLabel(cyxwiz::AnnotationLabel::Foreground);
editor.SetBrushRadius(10.0f);

// Paint refinement strokes
editor.OnMouseDown(x, y, cyxwiz::MouseButton::Left);
// ... painting ...
editor.OnMouseUp(x, y, cyxwiz::MouseButton::Left);

// Refine segmentation
editor.Refine(1);

// Get result
auto float_mask = editor.GetFloatMask();
```

#### GUI Usage

1. Go to **Dataset Panel** → **Interactive** tab
2. Select **GrabCut** tool
3. **Step 1**: Draw rectangle around the object
4. Click **"Run GrabCut"**
5. **Step 2**: Switch to Refinement mode
6. Paint green (foreground) or red (background) strokes
7. Click **"Refine"** to update
8. Repeat until satisfied

---

### Watershed Editor

**File**: `watershed_editor.h/cpp`

Marker-based segmentation using OpenCV Watershed.

#### Workflow

1. **Add Regions**: Define region types (background, foreground classes)
2. **Place Markers**: Click to place markers for each region
3. **Run Watershed**: Algorithm segments image based on markers
4. **Refine**: Add more markers and re-run

#### API Usage

```cpp
#include "gui/annotation/watershed_editor.h"

cyxwiz::WatershedEditor editor;
editor.SetImage(image_data, w, h, 3);

// Add region types
int bg_id = editor.AddRegionType("Background", false);
int fg_id = editor.AddRegionType("Object 1", true);
int fg2_id = editor.AddRegionType("Object 2", true);

// Set current region and place markers
editor.SetCurrentRegion(bg_id);
editor.OnMouseDown(bg_x, bg_y, cyxwiz::MouseButton::Left);

editor.SetCurrentRegion(fg_id);
editor.OnMouseDown(obj1_x, obj1_y, cyxwiz::MouseButton::Left);

editor.SetCurrentRegion(fg2_id);
editor.OnMouseDown(obj2_x, obj2_y, cyxwiz::MouseButton::Left);

// Run watershed
if (editor.GetMarkerCount() >= 2) {
    editor.RunWatershed();
}

// Get results
const auto& regions = editor.GetRegions();
for (const auto& [id, region] : regions) {
    printf("Region %d (%s): %d pixels\n",
           region.id, region.label.c_str(), region.pixel_count);
}

// Get foreground mask (all foreground regions combined)
auto fg_mask = editor.GetForegroundMask();
auto float_mask = editor.GetFloatMask();

// Get full region mask (each pixel = region ID)
const auto& region_mask = editor.GetRegionMask();
```

#### GUI Usage

1. Go to **Dataset Panel** → **Interactive** tab
2. Select **Watershed** tool
3. **Add Regions**: Click "Add Region" to create region types
4. **Select Region**: Click region in list to select
5. **Place Markers**: Click on image to place markers
6. Click **"Run Watershed"**
7. Add more markers and re-run to refine

---

## File Summary

### Phase 1: Preprocessing Transforms
| File | Size | Purpose |
|------|------|---------|
| `preprocessing/morphology_transform.h` | 6KB | Morphological operations |
| `preprocessing/morphology_transform.cpp` | 11KB | Implementation |
| `preprocessing/blur_transform.h` | 4KB | Blur/smoothing operations |
| `preprocessing/blur_transform.cpp` | 12KB | Implementation |
| `preprocessing/perspective_transform.h` | 6KB | Perspective/coordinate transforms |
| `preprocessing/perspective_transform.cpp` | 22KB | Implementation |
| `preprocessing/pyramid_transform.h` | 4KB | Image pyramids |
| `preprocessing/pyramid_transform.cpp` | 13KB | Implementation |

### Phase 2: Dataset Utilities
| File | Size | Purpose |
|------|------|---------|
| `utils/segmentation_utils.h` | 7KB | Segmentation algorithms |
| `utils/segmentation_utils.cpp` | 18KB | Implementation |
| `utils/shape_utils.h` | 8KB | Shape analysis |
| `utils/shape_utils.cpp` | 17KB | Implementation |
| `utils/labeling_utils.h` | 9KB | Labeling assistance |
| `utils/labeling_utils.cpp` | 23KB | Implementation |

### Phase 3: DNN Inference
| File | Size | Purpose |
|------|------|---------|
| `inference/dnn_models.h` | 10KB | DNN model wrapper |
| `inference/dnn_models.cpp` | 27KB | Implementation |
| `inference/dnn_preprocessor.h` | 6KB | DNN preprocessing |
| `inference/dnn_preprocessor.cpp` | 13KB | Implementation |

### Phase 4: Interactive Tools
| File | Size | Purpose |
|------|------|---------|
| `gui/annotation/annotation_tools.h` | 10KB | Base annotation framework |
| `gui/annotation/annotation_tools.cpp` | 23KB | Implementation |
| `gui/annotation/grabcut_editor.h` | 4KB | GrabCut editor |
| `gui/annotation/grabcut_editor.cpp` | 13KB | Implementation |
| `gui/annotation/watershed_editor.h` | 5KB | Watershed editor |
| `gui/annotation/watershed_editor.cpp` | 15KB | Implementation |
| `gui/panels/dataset_panel_interactive.cpp` | 15KB | Interactive tab UI |

---

## Dependencies

All features require OpenCV 4.x with the following modules:
- `core` - Basic structures
- `imgproc` - Image processing
- `dnn` - Deep learning inference (Phase 3)
- `ximgproc` - Extended image processing (optional, for selective search)

OpenCV is linked via CMake:
```cmake
find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn)
target_link_libraries(cyxwiz-engine PRIVATE ${OpenCV_LIBS})
```
