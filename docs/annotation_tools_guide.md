# CyxWiz Annotation Tools Guide

A comprehensive guide for ML engineers to create production-quality training datasets using CyxWiz's annotation system.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Annotation Tools](#annotation-tools)
   - [Brush Tool](#brush-tool)
   - [Polygon Tool](#polygon-tool)
   - [GrabCut Tool](#grabcut-tool)
   - [Watershed Tool](#watershed-tool)
4. [Batch Annotation Workflow](#batch-annotation-workflow)
5. [Class Management](#class-management)
6. [Export Formats](#export-formats)
   - [COCO JSON](#coco-json)
   - [YOLO](#yolo-format)
   - [Pascal VOC](#pascal-voc-xml)
7. [Training Integration](#training-integration)
8. [Use Cases](#use-cases)
   - [Semantic Segmentation](#semantic-segmentation)
   - [Instance Segmentation](#instance-segmentation)
   - [Object Detection](#object-detection)
   - [Medical Imaging](#medical-imaging)
   - [Satellite/Aerial Imagery](#satelliteaerial-imagery)
9. [Best Practices](#best-practices)
10. [Keyboard Shortcuts](#keyboard-shortcuts)
11. [API Reference](#api-reference)

---

## Overview

CyxWiz provides a complete annotation pipeline for creating training datasets for computer vision tasks:

| Feature | Description |
|---------|-------------|
| **4 Annotation Tools** | Brush, Polygon, GrabCut, Watershed |
| **Batch Workflow** | Navigate entire datasets with Prev/Next |
| **Class Management** | Create and assign semantic labels |
| **3 Export Formats** | COCO JSON, YOLO txt, Pascal VOC XML |
| **Training Integration** | Direct use via `GetAnnotatedBatch()` |
| **Persistence** | Annotations saved with project files |

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CyxWiz Engine                            │
├─────────────────────────────────────────────────────────────┤
│  Dataset Panel (Interactive Tab)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Brush Tool  │  │ Polygon Tool│  │ GrabCut     │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          ▼                                  │
│              ┌───────────────────┐                          │
│              │  Current Mask     │                          │
│              └─────────┬─────────┘                          │
│                        │ "Add Annotation"                   │
│                        ▼                                    │
│              ┌───────────────────┐                          │
│              │AnnotationManager  │                          │
│              │ - Per-image store │                          │
│              │ - Class labels    │                          │
│              └─────────┬─────────┘                          │
│                        │                                    │
│         ┌──────────────┼──────────────┐                     │
│         ▼              ▼              ▼                     │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│   │COCO JSON │  │ YOLO txt │  │ VOC XML  │                 │
│   └──────────┘  └──────────┘  └──────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Load a Dataset

```
File → Open Dataset → Select folder/file
```

Or via code:
```cpp
auto& registry = DataRegistry::Instance();
auto dataset = registry.LoadImageFolder("path/to/images", "my_dataset");
```

### 2. Open Interactive Tab

Navigate to **Dataset Panel** → **Interactive** tab

### 3. Add Classes

1. Type class name in "New class" field
2. Click **[Add Class]**
3. Repeat for all classes (e.g., "cat", "dog", "background")

### 4. Annotate Images

1. Select a tool (Brush, Polygon, GrabCut, Watershed)
2. Create mask on the image
3. Select class from dropdown
4. Click **[Add Annotation from Mask]**
5. Click **[Next >]** to move to next image

### 5. Export

Click **[COCO JSON]**, **[YOLO]**, or **[Pascal VOC]** to export annotations.

---

## Annotation Tools

### Brush Tool

**Best for:** Quick masks, touching up other tools, small objects

**How to use:**
1. Select **Brush** tool
2. Left-click and drag to paint foreground (white)
3. Right-click and drag to erase (black)
4. Adjust brush size with slider

**Parameters:**
| Parameter | Range | Description |
|-----------|-------|-------------|
| Brush Size | 1-100 | Radius in pixels |
| Opacity | 0-1 | Mask transparency for visibility |

**Tips:**
- Use large brush for filling, small brush for edges
- Zoom in (scroll wheel) for precise edges
- Right-click to fix mistakes immediately

---

### Polygon Tool

**Best for:** Objects with clear edges, architectural features, vehicles

**How to use:**
1. Select **Polygon** tool
2. Click to place vertices
3. Double-click or press Enter to close polygon
4. Polygon automatically fills as mask

**Controls:**
| Action | Result |
|--------|--------|
| Left-click | Add vertex |
| Double-click | Close polygon |
| Right-click | Remove last vertex |
| Escape | Cancel current polygon |

**Tips:**
- Place vertices at object corners/edges
- Fewer vertices = faster annotation
- Use for cars, buildings, signs, furniture

---

### GrabCut Tool

**Best for:** Complex objects with clear foreground/background separation

**How to use:**
1. Select **GrabCut** tool
2. Draw rectangle around object (rough bounding box)
3. Algorithm segments foreground automatically
4. Refine with foreground/background strokes if needed

**Refinement modes:**
| Mode | Color | Purpose |
|------|-------|---------|
| Foreground | Green | Mark definite foreground |
| Background | Red | Mark definite background |
| Probable FG | Light Green | Hint: likely foreground |
| Probable BG | Light Red | Hint: likely background |

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| Iterations | 5 | GrabCut iterations (more = better but slower) |

**Tips:**
- Initial rectangle should fully contain object with some margin
- Add foreground strokes on missed parts
- Add background strokes on incorrectly included areas
- Works best with good foreground/background contrast

---

### Watershed Tool

**Best for:** Touching/overlapping objects, cell segmentation, grains

**How to use:**
1. Select **Watershed** tool
2. Place markers on each object (different colors)
3. Click **[Run Watershed]**
4. Algorithm finds boundaries between marked regions

**Marker colors:**
- Each object gets a unique color
- Background gets its own color
- More markers = more objects detected

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| Marker Size | 5 | Size of marker brush |
| Use Distance Transform | true | Improves separation of touching objects |

**Tips:**
- Place one marker per object instance
- Marker should be in center of object
- Works well for cells, grains, bubbles, crowds

---

## Batch Annotation Workflow

### Navigation Controls

```
┌──────────────────────────────────────────────────────────────┐
│ Dataset: my_dataset (1000 images)                            │
│ Image: [< Prev] [  42 / 1000  ] [Next >]  Go to: [___] [Go] │
└──────────────────────────────────────────────────────────────┘
```

| Control | Action |
|---------|--------|
| **[< Prev]** | Go to previous image |
| **[Next >]** | Go to next image |
| **Go to: [n] [Go]** | Jump to specific image index |
| **Left Arrow** | Previous image (keyboard) |
| **Right Arrow** | Next image (keyboard) |

### Annotation List

```
┌──────────────────────────────────────────────────────────────┐
│ Annotations (3):                                             │
│   #1: cat (Polygon)                              [Delete]    │
│   #2: cat (Polygon)                              [Delete]    │
│   #3: dog (Polygon)                              [Delete]    │
└──────────────────────────────────────────────────────────────┘
```

- View all annotations for current image
- Click **[Delete]** to remove an annotation
- Annotations auto-save when navigating

### Efficient Workflow

1. **Plan classes first** - Add all classes before annotating
2. **Annotate in batches** - Do 50-100 images, then review
3. **Use appropriate tool** - GrabCut for complex objects, Polygon for simple
4. **Review periodically** - Check annotation quality mid-way

---

## Class Management

### Adding Classes

```cpp
// Via UI
// Type name → Click [Add Class]

// Via Code
auto& ann_mgr = DataRegistry::Instance().GetAnnotationManager();
ann_mgr.CreateAnnotationSet("my_dataset");
int cat_id = ann_mgr.AddClass("my_dataset", "cat");      // Returns 0
int dog_id = ann_mgr.AddClass("my_dataset", "dog");      // Returns 1
int bg_id = ann_mgr.AddClass("my_dataset", "background"); // Returns 2
```

### Class ID Assignment

| Class Name | Class ID | Mask Value |
|------------|----------|------------|
| (background) | 0 | 0 (implicit) |
| cat | 1 | 1 |
| dog | 2 | 2 |
| bird | 3 | 3 |

**Note:** Background (class 0) is implicit - pixels without annotations are background.

### Recommended Class Naming

| Domain | Example Classes |
|--------|-----------------|
| Autonomous Driving | car, truck, pedestrian, cyclist, road, sidewalk, building |
| Medical Imaging | tumor, healthy_tissue, organ, background |
| Satellite | building, road, vegetation, water, bare_soil |
| Retail | product, shelf, price_tag, person |

---

## Export Formats

### COCO JSON

**Best for:** Detectron2, MMDetection, TensorFlow Object Detection API

**File structure:**
```
annotations/
└── coco.json
```

**Format:**
```json
{
  "info": {
    "description": "CyxWiz Annotations",
    "version": "1.0",
    "date_created": "2026-01-11"
  },
  "images": [
    {
      "id": 0,
      "file_name": "image_0.png",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 0,
      "category_id": 1,
      "segmentation": {
        "counts": [10, 5, 20, 3, ...],
        "size": [480, 640]
      },
      "bbox": [100, 150, 200, 180],
      "area": 36000,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "cat"},
    {"id": 2, "name": "dog"}
  ]
}
```

**Export:**
```cpp
ann_mgr.ExportCOCO("my_dataset", "annotations/coco.json", "images/");
```

**Use with Detectron2:**
```python
from detectron2.data import register_coco_instances

register_coco_instances(
    "my_dataset_train",
    {},
    "annotations/coco.json",
    "images/"
)
```

---

### YOLO Format

**Best for:** YOLOv5, YOLOv8, Ultralytics

**File structure:**
```
yolo_annotations/
├── classes.txt          # Class names
├── image_0.txt          # Annotations for image_0
├── image_1.txt
└── ...
```

**Format (per image .txt):**
```
# class_id center_x center_y width height (normalized 0-1)
0 0.45 0.52 0.25 0.30
1 0.72 0.38 0.15 0.22
```

**classes.txt:**
```
cat
dog
bird
```

**Export:**
```cpp
ann_mgr.ExportYOLO("my_dataset", "yolo_annotations/", "classes.txt");
```

**Use with YOLOv8:**
```yaml
# dataset.yaml
path: /path/to/dataset
train: images/train
val: images/val

names:
  0: cat
  1: dog
  2: bird
```

```python
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')
model.train(data='dataset.yaml', epochs=100)
```

---

### Pascal VOC XML

**Best for:** Legacy tools, TensorFlow 1.x, some academic benchmarks

**File structure:**
```
voc_annotations/
├── image_0.xml
├── image_1.xml
└── ...
```

**Format:**
```xml
<annotation>
  <folder>images</folder>
  <filename>image_0.png</filename>
  <size>
    <width>640</width>
    <height>480</height>
    <depth>3</depth>
  </size>
  <object>
    <name>cat</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>150</ymin>
      <xmax>300</xmax>
      <ymax>330</ymax>
    </bndbox>
  </object>
</annotation>
```

**Export:**
```cpp
ann_mgr.ExportVOC("my_dataset", "voc_annotations/", "images/");
```

---

## Training Integration

### Using GetAnnotatedBatch()

Train semantic segmentation models directly in CyxWiz:

```cpp
#include "core/dataset_batcher.h"
#include "core/data_registry.h"

// Get dataset
auto dataset = DataRegistry::Instance().GetDataset("my_dataset");

// Create batcher
DatasetBatcher batcher(dataset, 32, DatasetSplit::Train, true);
batcher.Reset();

// Training loop
for (int epoch = 0; epoch < 100; epoch++) {
    batcher.Reset();

    while (batcher.HasNext()) {
        AnnotatedBatch batch = batcher.GetNextAnnotatedBatch("my_dataset");

        if (batch.IsValid() && batch.HasMasks()) {
            // batch.images: Tensor [B, H, W, C] - input images
            // batch.masks:  Tensor [B, H, W]    - class ID per pixel
            // batch.labels: Tensor [B]          - classification labels

            // Forward pass
            Tensor output = model.Forward(batch.images);

            // Compute segmentation loss
            Tensor loss = CrossEntropyLoss(output, batch.masks);

            // Backward pass
            model.Backward(loss);
            optimizer.Step();
        }
    }

    printf("Epoch %d complete\n", epoch);
}
```

### Mask Format

The segmentation mask is a 2D tensor where each pixel contains the class ID:

```
┌───────────────────────────────────────┐
│ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 │  0 = background
│ 0 0 0 1 1 1 1 1 0 0 0 2 2 2 2 0 0 0 0 │  1 = cat
│ 0 0 1 1 1 1 1 1 1 0 2 2 2 2 2 2 0 0 0 │  2 = dog
│ 0 0 1 1 1 1 1 1 1 0 2 2 2 2 2 2 0 0 0 │
│ 0 0 0 1 1 1 1 1 0 0 0 2 2 2 2 0 0 0 0 │
│ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 │
└───────────────────────────────────────┘
```

### Custom Mask Size

```cpp
// Resize masks to specific dimensions (e.g., for U-Net)
batcher.SetMaskSize(256, 256);  // Output masks will be 256x256
```

---

## Use Cases

### Semantic Segmentation

**Goal:** Classify every pixel into a category

**Tools:** Brush, Polygon, GrabCut

**Workflow:**
1. Add classes: road, car, pedestrian, building, sky, vegetation
2. Annotate each class per image
3. Export to COCO or use `GetAnnotatedBatch()`
4. Train U-Net, DeepLab, SegFormer

**Example datasets:** Cityscapes, ADE20K, PASCAL VOC

```cpp
// Training semantic segmentation
AnnotatedBatch batch = batcher.GetNextAnnotatedBatch("cityscapes");
// batch.masks contains class IDs 0-18 for Cityscapes
```

---

### Instance Segmentation

**Goal:** Detect and segment individual object instances

**Tools:** GrabCut, Polygon, Watershed (for touching objects)

**Workflow:**
1. Add classes: person, car, dog (no background class needed)
2. Annotate each instance separately
3. Export to COCO JSON (supports instance segmentation)
4. Train Mask R-CNN, YOLACT, SOLOv2

**Example datasets:** COCO, LVIS, Cityscapes Panoptic

```python
# Use with Detectron2 Mask R-CNN
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # person, car, dog
trainer = DefaultTrainer(cfg)
trainer.train()
```

---

### Object Detection

**Goal:** Detect objects with bounding boxes

**Tools:** Polygon (bbox computed automatically), GrabCut

**Workflow:**
1. Create segmentation masks (bboxes auto-generated)
2. Export to YOLO or COCO
3. Train YOLOv8, Faster R-CNN, SSD

**Note:** Even if you only need bboxes, segmentation masks provide higher quality bbox coordinates.

```bash
# Train YOLOv8 with exported annotations
yolo detect train data=dataset.yaml model=yolov8n.pt epochs=100
```

---

### Medical Imaging

**Goal:** Segment tumors, organs, lesions

**Tools:**
- **GrabCut** - Tumors with clear boundaries
- **Watershed** - Cell counting, touching nuclei
- **Brush** - Fine-tuning, small lesions

**Workflow:**
1. Classes: tumor, healthy_tissue, organ_boundary
2. Use GrabCut for initial segmentation
3. Refine with Brush tool
4. Export to COCO for training

**Considerations:**
- Use high zoom for precision
- Annotate in consistent viewing conditions
- Consider multiple annotators for consensus

```cpp
// Medical imaging often uses binary masks
std::vector<float> tumor_mask = ann_mgr.GetClassMask(
    "ct_scans", image_idx,
    1,  // tumor class ID
    512, 512
);
```

---

### Satellite/Aerial Imagery

**Goal:** Land use classification, building detection

**Tools:**
- **Polygon** - Buildings, roads, fields
- **Brush** - Vegetation, water bodies
- **Watershed** - Dense building areas

**Workflow:**
1. Classes: building, road, vegetation, water, bare_soil
2. Use Polygon for buildings (clear edges)
3. Use Brush for natural features
4. Export to any format

**Considerations:**
- Large images may need tiling
- Consistent class definitions across images
- Consider seasonal variations

---

## Best Practices

### Annotation Quality

| Practice | Reason |
|----------|--------|
| **Consistent boundaries** | Stay 1-2 pixels inside object edges |
| **Complete coverage** | Don't leave holes in masks |
| **Verify classes** | Double-check class assignment |
| **Handle occlusion** | Annotate visible parts only |

### Efficiency Tips

1. **Batch similar images** - Annotate all "cat" images together
2. **Use keyboard shortcuts** - Much faster than mouse
3. **Start with easy images** - Build confidence before hard cases
4. **Review in batches** - Check 50 images at a time

### Quality Control

```cpp
// Check annotation statistics
auto* ann_set = ann_mgr.GetAnnotationSet("my_dataset");
size_t total = ann_set->GetTotalAnnotationCount();
size_t annotated = ann_set->GetAnnotatedImageCount();
printf("Annotated %zu/%zu images with %zu annotations\n",
       annotated, dataset.Size(), total);
```

### Multi-Annotator Workflow

1. Split dataset among annotators
2. Annotate independently
3. Export each annotator's work
4. Compare and resolve disagreements
5. Merge into final dataset

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| **←** | Previous image |
| **→** | Next image |
| **B** | Select Brush tool |
| **P** | Select Polygon tool |
| **G** | Select GrabCut tool |
| **W** | Select Watershed tool |
| **Ctrl+Z** | Undo |
| **Ctrl+Y** | Redo |
| **Ctrl+S** | Save annotations |
| **Delete** | Clear current mask |
| **Enter** | Close polygon / Confirm |
| **Escape** | Cancel current operation |
| **[** | Decrease brush size |
| **]** | Increase brush size |
| **Scroll** | Zoom in/out |
| **Middle-drag** | Pan image |

---

## API Reference

### AnnotationManager

```cpp
class AnnotationManager {
public:
    // Dataset operations
    void CreateAnnotationSet(const std::string& dataset_id);
    AnnotationSet* GetAnnotationSet(const std::string& dataset_id);
    bool HasAnnotationSet(const std::string& dataset_id) const;

    // Class management
    int AddClass(const std::string& dataset_id, const std::string& class_name);
    const std::vector<std::string>& GetClasses(const std::string& dataset_id) const;

    // Annotation operations
    int AddAnnotation(const std::string& dataset_id, size_t image_idx,
                      int class_id, AnnotationType type,
                      const std::vector<Point2D>& polygon,
                      const BoundingBox& bbox = {},
                      int image_width = 0, int image_height = 0);
    bool RemoveAnnotation(const std::string& dataset_id, size_t image_idx, int ann_id);
    std::vector<Annotation>* GetAnnotations(const std::string& dataset_id, size_t image_idx);

    // Export
    bool ExportCOCO(const std::string& dataset_id, const std::string& output_path,
                    const std::string& image_dir = "") const;
    bool ExportYOLO(const std::string& dataset_id, const std::string& output_dir,
                    const std::string& classes_file = "") const;
    bool ExportVOC(const std::string& dataset_id, const std::string& output_dir,
                   const std::string& image_dir = "") const;

    // Training integration
    std::vector<float> GetSegmentationMask(const std::string& dataset_id,
                                            size_t image_idx,
                                            int width, int height) const;
    std::vector<float> GetClassMask(const std::string& dataset_id,
                                     size_t image_idx, int class_id,
                                     int width, int height) const;

    // Persistence
    nlohmann::json ToJson() const;
    void FromJson(const nlohmann::json& j);
};
```

### DatasetBatcher (Annotated)

```cpp
struct AnnotatedBatch {
    Tensor images;        // [B, H, W, C] - input images
    Tensor labels;        // [B] - classification labels
    Tensor masks;         // [B, H, W] - segmentation masks
    size_t size;          // Batch size

    bool HasMasks() const;
};

class DatasetBatcher {
public:
    // Get batch with segmentation masks
    AnnotatedBatch GetAnnotatedBatch(const std::string& dataset_id,
                                      const std::vector<size_t>& indices);
    AnnotatedBatch GetNextAnnotatedBatch(const std::string& dataset_id);

    // Check for annotations
    bool HasAnnotations(const std::string& dataset_id) const;

    // Configure mask output size
    void SetMaskSize(int width, int height);
};
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| GrabCut gives poor results | Ensure bounding box has margin around object |
| Polygon won't close | Double-click on first vertex |
| Export creates empty files | Check that annotations exist for images |
| Mask has wrong dimensions | Use `SetMaskSize()` or check dataset shape |
| Classes not showing | Call `CreateAnnotationSet()` first |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-11 | Initial release with full annotation system |

---

*For questions or issues, refer to CLAUDE.md or open an issue on GitHub.*
