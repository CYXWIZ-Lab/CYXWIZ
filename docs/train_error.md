# Training Debugging Journey: MNIST MLP Stuck at 89% Accuracy

This document chronicles the debugging process for a critical training issue in CyxWiz Engine where MNIST classification was stuck at ~89% accuracy with abnormally high loss values.

## Initial Problem

**Symptoms:**
- Training accuracy stuck at ~89% (should be 95%+)
- Loss values around 8.5 (should be 0.3-0.5 for cross-entropy)
- Network appeared to learn something but couldn't improve

**Expected vs Actual:**
| Metric | Expected | Actual |
|--------|----------|--------|
| Loss (epoch 1) | ~2.3 (random) → ~0.5 | ~8.0 → ~8.6 |
| Accuracy | 10% → 95%+ | 10% → 89% (stuck) |

## Investigation Process

### Step 1: Check Graph Topology

First, we verified the node graph was correct:
- DatasetInput → Normalize → Dense(512) → ReLU → Dropout → Dense(256) → ReLU → Dropout → Dense(10) → CrossEntropyLoss → Adam

**Finding:** Graph looked correct, but labels from DatasetInput weren't connected to CrossEntropyLoss. Fixed by adding the connection.

### Step 2: Double Softmax Issue

**Problem:** Network had both a Softmax node AND CrossEntropyLoss.

CrossEntropyLoss internally applies softmax to logits before computing the loss. Having an explicit Softmax node before it caused "double softmax", which:
- Squashes gradients to near-zero
- Prevents learning

**Fix:** Modified `graph_compiler.cpp` to skip Softmax nodes when CrossEntropyLoss is detected:
```cpp
if (node->type == gui::NodeType::Softmax && using_cross_entropy) {
    spdlog::debug("GraphCompiler: Skipping Softmax (CrossEntropyLoss applies it internally)");
    continue;
}
```

**Result:** Accuracy improved from 10% to ~88%, but still stuck.

### Step 3: Normalization Investigation

**Hypothesis:** Data not normalized properly.

Added debug logging to verify normalization was being applied:
```cpp
spdlog::info("TrainingExecutor: Applying normalization (mean={}, std={})",
             config_.preprocessing.norm_mean, config_.preprocessing.norm_std);
```

**Finding:** Logs showed normalization WAS being applied with mean=0, std=255.

Checked input data range:
```
DEBUG: Input data range: [0.0000, 1.0000]
```

Data was correctly normalized to [0,1]. This wasn't the issue.

### Step 4: Wrong Normalization Values (Failed Attempt)

**Hypothesis:** MNIST data loader already normalizes to [0,1], so applying mean=0, std=255 would double-normalize.

We changed the pattern to use standard MNIST normalization:
```json
{"mean": "0.1307", "std": "0.3081"}
```

**Result:** Made things WORSE!
```
DEBUG: Input data range: [-0.4242, 827.2292]
```

The CSV-loaded MNIST data was actually raw 0-255 values, NOT pre-normalized. The original mean=0, std=255 was correct.

**Reverted:** Back to mean=0, std=255.

### Step 5: Loss Function Deep Dive

Added detailed debug logging to CrossEntropyLoss:
```cpp
spdlog::info("DEBUG CrossEntropy: pred dims=({},{},{},{})", ...);
spdlog::info("DEBUG CrossEntropy: First sample softmax probs:");
spdlog::info("DEBUG CrossEntropy: First sample per-class loss:");
```

**Critical Finding:**
```
DEBUG: First sample predictions:
  [0.3715, 0.4388, 0.0640, ...]  // Highest at index 1

DEBUG CrossEntropy: First sample softmax probs:
  [0.1277, 0.0940, 0.1081, 0.0619, 0.1296, 0.1316, ...]  // Highest at index 5!

DEBUG CrossEntropy: First sample per-class loss:
  [-0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, ...]  // All zeros!
```

**The predictions and softmax didn't correspond!** Index 1 had the highest logit, but index 5 had the highest softmax probability. And the per-class loss was all zeros when it should have a value at the target class.

This indicated **data scrambling** during tensor conversion.

## Root Cause: Row-Major vs Column-Major Layout

**The Bug:**
- CyxWiz Tensor uses **row-major** (C-style) memory layout
- ArrayFire uses **column-major** (Fortran-style) memory layout
- The `TensorToAf` helper directly copied data without transposing

**Example of the problem:**

For a [batch=32, classes=10] tensor:

**Row-major (CyxWiz):**
```
Memory: [s0_c0, s0_c1, ..., s0_c9, s1_c0, s1_c1, ..., s1_c9, ...]
        ^----- sample 0 ------^  ^----- sample 1 ------^
```

**Column-major (ArrayFire):**
```
Memory: [s0_c0, s1_c0, ..., s31_c0, s0_c1, s1_c1, ..., s31_c1, ...]
        ^---- class 0 for all ----^  ^---- class 1 for all ----^
```

When row-major data was written to ArrayFire's column-major array, the data was effectively transposed/scrambled. Predictions for sample 0 were scattered across all samples, and predictions for class 0 came from different samples.

## The Fix

Modified `TensorToAf` and `AfToTensor` in both `loss.cpp` and `layer.cpp`:

**TensorToAf (row-major → column-major):**
```cpp
static af::array TensorToAf(const Tensor& t) {
    const auto& shape = t.Shape();
    af::dim4 dims(1, 1, 1, 1);
    for (size_t i = 0; i < shape.size() && i < 4; i++) {
        dims[static_cast<unsigned int>(i)] = static_cast<dim_t>(shape[i]);
    }

    // For 2D arrays, swap dimensions to account for row-major input
    // We load as [cols, rows] then transpose to get [rows, cols] in column-major
    if (shape.size() == 2) {
        af::dim4 swapped_dims(dims[1], dims[0], 1, 1);
        af::array arr(swapped_dims, ToAfType(t.GetDataType()));
        arr.write(t.Data(), arr.bytes(), afHost);
        return af::transpose(arr);  // Now [rows, cols] in column-major
    }

    af::array arr(dims, ToAfType(t.GetDataType()));
    arr.write(t.Data(), arr.bytes(), afHost);
    return arr;
}
```

**AfToTensor (column-major → row-major):**
```cpp
static Tensor AfToTensor(const af::array& arr) {
    // ... dtype detection ...

    // For 2D arrays, transpose to row-major before copying to Tensor
    if (ndims == 2) {
        af::array transposed = af::transpose(arr);
        std::vector<size_t> shape = {
            static_cast<size_t>(arr.dims(0)),
            static_cast<size_t>(arr.dims(1))
        };
        Tensor result(shape, dtype);
        transposed.host(result.Data());
        return result;
    }
    // ... handle other dims ...
}
```

## Results After Fix

```
DEBUG CrossEntropy: First sample per-class loss:
  [-0.0000, 2.4074, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000]
```

Now the loss is correctly computed only at the target class index!

**Training Results:**
```
Epoch 1: loss=0.2243, acc=93.17%, val_loss=0.1193, val_acc=96.33%
Epoch 2: loss=0.1069, acc=96.71%, val_loss=0.1038, val_acc=96.86%
```

| Metric | Before Fix | After Fix |
|--------|------------|-----------|
| Epoch 1 Loss | 8.06 | 0.22 |
| Epoch 1 Accuracy | 87% | 93% |
| Epoch 2 Accuracy | 89% | 97% |
| Val Accuracy | 89% (stuck) | 97%+ |

## Saving the Trained Model

After successfully training the model, we can save it for later use:

### Using the UI

1. Go to **Tools > Model Export > Save Trained Model...** (or press `Ctrl+Shift+S`)
2. Choose a location and filename (e.g., `mnist_classifier.cyxmodel`)
3. A single `.cyxmodel` file is created containing both metadata and weights

### Using the API

```cpp
#include "core/training_manager.h"

auto& tm = TrainingManager::Instance();
if (tm.HasTrainedModel()) {
    tm.SaveModel("models/mnist_mlp.cyxmodel", "MNIST MLP", "97% accuracy on test set");
}
```

### Model File Format (`.cyxmodel`)

The `.cyxmodel` format is a single binary file with embedded JSON metadata:

```
┌─────────────────────────────────────────┐
│ Magic: "CYXW" (4 bytes)                 │
│ Version: 2 (4 bytes)                    │
│ JSON length (8 bytes)                   │
│ JSON metadata (variable)                │
│ Binary weights (rest of file)           │
└─────────────────────────────────────────┘
```

**Embedded JSON Metadata:**
```json
{
  "metadata": {
    "name": "MNIST MLP",
    "description": "97% accuracy on test set",
    "created_at": "2025-12-13 19:30:00",
    "framework": "CyxWiz",
    "format_version": "2.0"
  },
  "modules": [
    {"index": 0, "name": "Linear(784 -> 512)", "has_parameters": true, ...},
    {"index": 1, "name": "ReLU", "has_parameters": false},
    ...
  ]
}
```

### Loading for Inference

```cpp
// Create model with same architecture
cyxwiz::SequentialModel model;
model.Add(std::make_unique<cyxwiz::LinearModule>(784, 512));
model.Add(std::make_unique<cyxwiz::ReLUModule>());
model.Add(std::make_unique<cyxwiz::DropoutModule>(0.2));
model.Add(std::make_unique<cyxwiz::LinearModule>(512, 256));
model.Add(std::make_unique<cyxwiz::ReLUModule>());
model.Add(std::make_unique<cyxwiz::DropoutModule>(0.2));
model.Add(std::make_unique<cyxwiz::LinearModule>(256, 10));

// Load weights from .cyxmodel file
model.Load("models/mnist_mlp.cyxmodel");
model.SetTraining(false);  // Set to evaluation mode

// Inference
auto output = model.Forward(input);
```

See `docs/model_serialization.md` for complete API documentation.

## Key Lessons Learned

1. **Memory layout matters:** When interfacing between libraries with different memory layouts (row-major vs column-major), explicit conversion is required.

2. **Debug logging is essential:** Adding step-by-step debug output revealed the data scrambling that wasn't visible otherwise.

3. **Loss values are diagnostic:** An abnormally high loss (8.5 vs expected 0.5) indicated something fundamentally wrong, not just a tuning issue.

4. **Check intermediate values:** Comparing predictions to softmax outputs revealed they didn't correspond, pointing to the conversion bug.

5. **Don't assume normalization issues:** We initially suspected normalization, but the real issue was deeper in the data pipeline.

## Files Modified

### Training Bug Fix
- `cyxwiz-backend/src/algorithms/loss.cpp` - Fixed TensorToAf/AfToTensor
- `cyxwiz-backend/src/algorithms/layer.cpp` - Fixed TensorToAf/AfToTensor
- `cyxwiz-engine/src/core/graph_compiler.cpp` - Skip Softmax with CrossEntropyLoss
- `cyxwiz-engine/src/core/training_executor.cpp` - Added debug logging
- `docs/mnist_mlp.cyxgraph` - Pattern file adjustments during debugging

### Model Serialization
- `cyxwiz-backend/include/cyxwiz/sequential.h` - Added Save/Load methods
- `cyxwiz-backend/src/algorithms/sequential.cpp` - Implemented serialization
- `cyxwiz-engine/src/core/training_manager.h` - Added SaveModel method
- `cyxwiz-engine/src/core/training_manager.cpp` - Implemented SaveModel
- `cyxwiz-engine/src/gui/panels/toolbar.h` - Added save model callback
- `cyxwiz-engine/src/gui/panels/toolbar_tools_menu.cpp` - Added Model Export menu
- `cyxwiz-engine/src/gui/main_window.cpp` - Wired up save dialog
- `docs/model_serialization.md` - Full API documentation

## Future Considerations

### Training Infrastructure
1. **Centralize conversion helpers:** The TensorToAf/AfToTensor functions are duplicated in loss.cpp and layer.cpp. Consider moving to a shared header.

2. **Add unit tests:** Create tests that verify data integrity through the Tensor↔ArrayFire conversion.

3. **Document memory layout:** Add clear documentation about CyxWiz's row-major convention and the ArrayFire interface requirements.

### Model Export
4. **ONNX export:** Add export to ONNX format for interoperability with PyTorch, TensorFlow, etc.

5. **Architecture serialization:** Currently architecture must be recreated before loading weights. Consider serializing layer types and parameters for full model reconstruction.

6. **Checkpoint saving:** Add automatic checkpoint saving during training for crash recovery.

7. **Model compression:** Implement weight quantization (FP16, INT8) for smaller model files.
