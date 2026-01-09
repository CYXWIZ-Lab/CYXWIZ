# Data Pipeline Tab - Unified Preprocessing & Augmentation

## Overview

The **Data Pipeline** tab combines preprocessing and augmentation into a single, intuitive interface that clearly separates deterministic operations from random training-time transforms.

**Previous Structure (DEPRECATED):**
- ❌ Separate "Augmentation" tab
- ❌ Separate "Preprocessing" tab
- ❌ Confusing overlap between tabs
- ❌ Unclear execution order

**New Structure:**
- ✅ Single "Data Pipeline" tab
- ✅ Clear visual separation (blue vs orange headers)
- ✅ Unified preview with mode selector
- ✅ Obvious execution order (top to bottom)

---

## Tab Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Data Pipeline Tab                                │
├──────────────────────────────────────┬──────────────────────────────────┤
│  Configuration (60%)                 │  Preview (40%)                   │
│                                      │                                   │
│  🎛️ Preprocessing (Deterministic)    │  📷 Preview Mode:                │
│  Blue Header                         │  ┌─────────────────────────┐    │
│  • Dataset Statistics                │  │ ○ Preprocessing         │    │
│  • Image Enhancement                 │  │ ○ Augmentation          │    │
│  • Normalization                     │  └─────────────────────────┘    │
│  • Scaling                           │                                   │
│  [Apply Preprocessing]               │  Sample: 0 [◄] [►] [Random]     │
│                                      │  [Update Preview]                │
│  ───────────────────────────────     │  ─────────────────────────────   │
│                                      │                                   │
│  🔀 Training Augmentation (Random)   │  BEFORE          AFTER           │
│  Orange Header                       │  [Image]         [Image]         │
│  • 13 Presets                        │                                   │
│  • Transform Pipeline                │  Dimensions: 28x28x1 → 28x28x1  │
│  [Apply Augmentation]                │                                   │
└──────────────────────────────────────┴──────────────────────────────────┘
```

---

## Section 1: Preprocessing (Deterministic)

**Icon:** 🎛️ (ICON_FA_SLIDERS)
**Color:** Blue (#4D80B3)
**Purpose:** Applied once before training, same result every time

### Features

#### 1. Dataset Statistics
- Compute per-channel statistics (mean, std, min, max, median, percentiles)
- GPU-accelerated PCA computation (ArrayFire SVD)
- Cached results for fast reuse
- Progress indicator during computation

#### 2. Image Enhancement
Deterministic transforms for data cleanup:
- **Resize:** Target dimensions (width × height)
- **Color Conversion:** RGB ↔ Grayscale
- **CLAHE:** Contrast Limited Adaptive Histogram Equalization
- **Denoise:** Non-local means, bilateral, median, Gaussian
- **Sharpen:** Unsharp mask, Laplacian
- **Edge Detection:** Canny, Sobel, Laplacian, Scharr (4 methods)

#### 3. Normalization
Pre-defined strategies:
- **None:** No normalization
- **Auto-Detect:** Automatically detect dataset type
- **MNIST:** [0, 1] normalization for grayscale images
- **CIFAR-10:** Per-channel mean/std normalization
- **ImageNet:** Industry-standard mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- **Custom:** User-defined mean/std values

#### 4. Scaling
Advanced scaling strategies:
- **None:** No scaling
- **MinMax:** Scale to [min, max] range (default [0, 1])
- **Standard:** Z-score normalization (mean=0, std=1)
- **Robust:** Median-based, resistant to outliers
- **MaxAbs:** Scale by max absolute value to [-1, 1]
- **Quantile:** Transform to uniform [0, 1] or normal N(0, 1)
- **PCA Whitening:** Decorrelate features using eigenvalue decomposition (GPU-accelerated)

### Execution Pipeline

```
Original Data
    ↓
Image Enhancement (Resize, Color, CLAHE, Denoise, Sharpen, Edge Detection)
    ↓
Normalization (Dataset-specific mean/std)
    ↓
Scaling (MinMax, Standard, Robust, MaxAbs, Quantile, PCA)
    ↓
Preprocessed Data (Cached)
```

---

## Section 2: Training Augmentation (Random)

**Icon:** 🔀 (ICON_FA_SHUFFLE)
**Color:** Orange (#B37D4D)
**Purpose:** Applied during training with randomness, different every epoch

### Features

#### Augmentation Presets (13)

1. **None:** No augmentation (baseline)
2. **ImageNet Train:** Standard ImageNet preprocessing
   - RandomResizedCrop(224)
   - RandomHorizontalFlip
   - ColorJitter
   - Normalize(ImageNet mean/std)

3. **ImageNet Val:** Validation-time preprocessing
   - Resize(256)
   - CenterCrop(224)
   - Normalize(ImageNet mean/std)

4. **CIFAR-10 Train:** Common CIFAR augmentation
   - RandomCrop(32, padding=4)
   - RandomHorizontalFlip
   - Normalize(CIFAR mean/std)

5. **CIFAR-10 Val:** Validation-time
   - Normalize(CIFAR mean/std)

6. **Medical Imaging:** Specialized for medical data
   - RandomRotation(10°)
   - RandomAffine(translate=0.1, scale=0.9-1.1)
   - RandomHorizontalFlip
   - Normalize

7. **Self-Supervised:** For contrastive learning
   - RandomResizedCrop
   - RandomHorizontalFlip
   - ColorJitter (strong)
   - RandomGrayscale(p=0.2)
   - GaussianBlur

8. **Object Detection:** Spatial augmentation
   - RandomHorizontalFlip
   - RandomResize
   - ColorJitter

9. **Segmentation:** Preserves spatial structure
   - RandomCrop
   - RandomHorizontalFlip
   - RandomVerticalFlip
   - ColorJitter

10. **Facial Recognition:** Face-specific
    - RandomHorizontalFlip
    - RandomRotation(5°)
    - ColorJitter
    - RandomErasing

11. **Text Detection:** OCR-specific
    - RandomRotation(2°)
    - RandomAffine
    - GaussianBlur
    - RandomBrightness

12. **Video Frames:** Temporal consistency
    - RandomCrop
    - RandomHorizontalFlip
    - ColorJitter (mild)

13. **Custom:** User-defined pipeline

### Transform Categories

#### Geometric Transforms
- Resize, Crop, CenterCrop
- Rotate, Flip (Horizontal/Vertical)
- Affine, Perspective

#### Color Transforms
- ColorJitter (brightness, contrast, saturation, hue)
- Grayscale
- AutoContrast, Equalize

#### Noise/Blur
- GaussianNoise
- GaussianBlur
- MedianBlur

#### Advanced
- RandomErasing
- Cutout
- MixUp (future)

---

## Unified Preview System

### Preview Modes

#### 1. Preprocessing Mode
Shows: **Original → Preprocessed**
- Displays deterministic transform results
- Same output on every preview
- Useful for validating enhancement/normalization

#### 2. Augmentation Mode
Shows: **Original → Augmented**
- Displays random transform results
- **Auto-refresh:** Continuously applies new random transforms
- Different output on each preview
- Useful for validating augmentation variety

### Preview Controls

```
Preview Mode: [Preprocessing] [Augmentation]
Sample: 0 [◄] [►] [Random]
☐ Auto-refresh (Augmentation mode only)
[Update Preview]
```

---

## When to Use Each Section

### Use Preprocessing When:
- ✅ Normalizing pixel values to standard ranges
- ✅ Resizing images to consistent dimensions
- ✅ Applying CLAHE for better contrast
- ✅ Denoising low-quality images
- ✅ Sharpening blurry images
- ✅ Extracting edge features
- ✅ Whitening data (PCA decorrelation)
- ✅ Operations that should be **identical** for all samples

### Use Augmentation When:
- ✅ Increasing effective dataset size
- ✅ Preventing overfitting
- ✅ Making model robust to variations (rotation, flip, color)
- ✅ Simulating real-world conditions
- ✅ Operations that should be **random** per training batch

### Example Workflow

```
1. Load MNIST dataset
   ↓
2. Preprocessing:
   - Normalize to [0, 1]
   - Optional: Apply CLAHE for better contrast
   [Apply Preprocessing]
   ↓
3. Augmentation:
   - Select "Custom" preset
   - Add RandomRotation(±10°)
   - Add RandomAffine(translate=0.1)
   [Apply Augmentation]
   ↓
4. Training:
   - Preprocessing applied ONCE at load time
   - Augmentation applied EVERY batch during training
```

---

## Key Implementation Files

| File | Purpose |
|------|---------|
| `dataset_panel_datapipeline.cpp` | Main tab implementation |
| `dataset_panel_preprocessing.cpp` | Preprocessing logic (reused) |
| `dataset_panel.cpp` | Augmentation logic (reused) |
| `statistics_calculator.cpp` | Dataset statistics + PCA (GPU) |
| `preprocessing_config.h` | Config structs |
| `scaling_transform.cpp` | Scaling implementations |
| `transforms/transforms.h` | Augmentation transforms |

---

## Migration from Old Tabs

### Old Code (DEPRECATED)
```cpp
// Opening Augmentation tab
dataset_panel_->SetActiveTab(gui::DatasetTab::Augmentation);  // ❌ No longer exists

// Opening Preprocessing tab
dataset_panel_->SetActiveTab(gui::DatasetTab::Preprocessing);  // ❌ No longer exists
```

### New Code
```cpp
// Opening unified Data Pipeline tab
dataset_panel_->SetActiveTab(gui::DatasetTab::DataPipeline);  // ✅ Correct
```

---

## Benefits of Unified Tab

### 1. **Reduced Confusion**
- Single tab instead of two separate tabs
- Clear visual distinction (blue vs orange)
- Obvious execution order (top to bottom)

### 2. **Better Mental Model**
- **Preprocessing = Do once, deterministic**
- **Augmentation = Do every epoch, random**
- Pipeline flow matches actual execution

### 3. **Improved Workflow**
- Configure entire data pipeline in one place
- Unified preview shows both pipelines
- No need to switch between tabs

### 4. **Space Efficiency**
- 60/40 split (config/preview) maximizes space
- Collapsible sections for focused work
- Side-by-side before/after comparison

---

## Technical Details

### Preprocessing Pipeline Architecture

```cpp
// In DataRegistry::ApplyPreprocessing()
void ApplyPreprocessing(const PreprocessingConfig& config) {
    // 1. Compute statistics (once, cached)
    auto stats = StatisticsCalculator::Compute(dataset_id, registry);

    // 2. Create transforms
    ImageTransform img_transform(config);
    NormalizationTransform norm_transform(config);
    ScalingTransform scale_transform(config);

    // Initialize with stats
    norm_transform.Initialize(stats);
    scale_transform.Initialize(stats);

    // 3. Apply to all samples
    for (auto& sample : samples) {
        sample = img_transform.Apply(sample);      // Image enhancement
        sample = norm_transform.Apply(sample);     // Normalization
        sample = scale_transform.Apply(sample);    // Scaling
    }
}
```

### Augmentation Pipeline Architecture

```cpp
// In TrainingExecutor::GetBatch()
std::vector<Tensor> GetBatch(size_t batch_size) {
    std::vector<Tensor> batch;

    for (size_t i = 0; i < batch_size; ++i) {
        // 1. Load preprocessed sample (already cached)
        auto sample = dataset_.GetSample(indices[i]);

        // 2. Apply random augmentation
        if (augmentation_pipeline_) {
            sample = augmentation_pipeline_->apply(sample);  // Random transforms
        }

        batch.push_back(sample);
    }

    return batch;
}
```

### GPU-Accelerated PCA

```cpp
// In StatisticsCalculator::Compute() - ArrayFire SVD
af::array data_af(n_features, n_samples, data_flat.data());
data_af = af::transpose(data_af);  // [n_samples x n_features]

// Center data
af::array mean_af = af::mean(data_af, 0);
af::array centered = data_af - af::tile(mean_af, n_samples, 1);

// SVD: X = U Σ V^T
af::array U, S, Vt;
af::svd(U, S, Vt, centered);  // GPU-accelerated!

// Eigenvectors = V (columns), Eigenvalues = σ²/n
```

---

## Performance Considerations

### Preprocessing (One-Time Cost)
- **Statistics computation:** ~1-5 seconds (GPU) for 50K samples
- **PCA whitening:** ~2-10 seconds (GPU) depending on feature dimensions
- **Image transforms:** ~0.1ms per sample (CPU) or ~0.01ms (GPU with ArrayFire)
- **Total:** Minutes for large datasets, but only done ONCE

### Augmentation (Per-Batch Cost)
- **Random transforms:** ~0.1-0.5ms per sample (CPU-bound)
- **Negligible compared to training:** <1% of epoch time
- **Parallelized:** Transforms applied in parallel via DataLoader threads

---

## FAQ

### Q: Should I use both preprocessing and augmentation?
**A:** Yes! They serve different purposes:
- Preprocessing normalizes data to standard ranges
- Augmentation increases variety to prevent overfitting

### Q: Why is preprocessing slow?
**A:** It's computing statistics over the entire dataset (mean, std, PCA components). But it's only done once and cached.

### Q: Can I skip preprocessing?
**A:** Yes, but not recommended. Most models expect normalized inputs ([0, 1] or mean=0/std=1).

### Q: Does augmentation slow down training?
**A:** Slightly (<1% per epoch), but the benefits (reduced overfitting) far outweigh the cost.

### Q: Can I add custom transforms?
**A:** Yes! Select "Custom" preset in augmentation and add transforms manually.

### Q: What's the execution order?
```
Load Sample
    ↓
Preprocessing (cached, deterministic) ← Applied ONCE
    ↓
Augmentation (random, per-batch) ← Applied EVERY BATCH
    ↓
Training
```

---

## Future Enhancements

### Planned Features
- [ ] **Full Pipeline Preview:** Show Original → Preprocessed → Augmented in 3 columns
- [ ] **MixUp/CutMix:** Advanced augmentation strategies
- [ ] **AutoAugment:** Policy-based augmentation search
- [ ] **GPU Augmentation:** ArrayFire-accelerated transforms
- [ ] **Export Pipeline:** Save preprocessing + augmentation as ONNX graph

### Under Consideration
- [ ] **Dataset versioning:** Track preprocessing configs
- [ ] **A/B testing:** Compare augmentation strategies
- [ ] **Smart defaults:** Auto-select presets based on dataset type

---

## Related Documentation

- **Preprocessing Config:** `cyxwiz-engine/src/preprocessing/preprocessing_config.h`
- **Transform Library:** `cyxwiz-engine/src/transforms/`
- **Statistics Calculator:** `cyxwiz-engine/src/preprocessing/statistics_calculator.h`
- **Training Pipeline:** `cyxwiz-engine/src/core/training_executor.h`

---

**Version:** 1.0
**Last Updated:** 2026-01-09
**Author:** CyxWiz Development Team
