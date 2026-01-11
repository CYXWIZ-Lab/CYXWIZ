# CyxWiz Model Serialization

This document describes the model serialization format and API for saving and loading trained models in CyxWiz.

## Overview

CyxWiz uses a single `.cyxmodel` file format that combines JSON metadata with binary weights. This design is similar to ONNX (single file, structured format) and provides:
- Single file for easy sharing and deployment
- Embedded JSON metadata for model inspection
- Efficient binary storage for weights
- Version compatibility checking
- Platform-independent storage

## File Format

Saving a model creates a single `.cyxmodel` file (e.g., `my_model.cyxmodel`).

### Binary Structure

```
┌─────────────────────────────────────────────┐
│ Magic Number: "CYXW" (4 bytes)              │  0x43595857
├─────────────────────────────────────────────┤
│ Format Version (4 bytes)                    │  Currently: 2
├─────────────────────────────────────────────┤
│ JSON Length (8 bytes)                       │  Size of metadata section
├─────────────────────────────────────────────┤
│ JSON Metadata (variable)                    │  Human-readable model info
├─────────────────────────────────────────────┤
│ Number of Modules (8 bytes)                 │
├─────────────────────────────────────────────┤
│ Module Parameters (variable)                │  Binary weights for each module
└─────────────────────────────────────────────┘
```

### Embedded JSON Metadata

The JSON section contains human-readable information:

```json
{
  "metadata": {
    "name": "MNIST Classifier",
    "description": "MLP for digit recognition - 97% accuracy",
    "created_at": "2025-12-13 19:30:00",
    "framework": "CyxWiz",
    "format_version": "2.0"
  },
  "modules": [
    {
      "index": 0,
      "name": "Linear(784 -> 512)",
      "has_parameters": true,
      "trainable": true,
      "parameters": [
        {"name": "weight", "shape": [784, 512], "dtype": "float32"},
        {"name": "bias", "shape": [512], "dtype": "float32"}
      ]
    },
    {
      "index": 1,
      "name": "ReLU",
      "has_parameters": false,
      "trainable": true
    }
  ]
}
```

### Module Parameter Binary Format

After the JSON section, module parameters are stored sequentially:

```
For each module:
  [num_params: 8 bytes]
  For each parameter:
    [name_length: 8 bytes]
    [name: name_length bytes]
    [ndims: 8 bytes]
    [shape: ndims * 8 bytes]
    [dtype: 4 bytes]
    [num_bytes: 8 bytes]
    [data: num_bytes bytes]
```

## API Usage

### Saving a Model

#### From TrainingManager (Recommended)
```cpp
#include "core/training_manager.h"

// After training completes
auto& tm = TrainingManager::Instance();
if (tm.HasTrainedModel()) {
    bool success = tm.SaveModel(
        "models/mnist_mlp.cyxmodel",  // Path to .cyxmodel file
        "MNIST Classifier",            // Model name (optional)
        "MLP for digit recognition"    // Description (optional)
    );

    if (success) {
        spdlog::info("Model saved successfully");
    }
}
```

#### Direct SequentialModel API
```cpp
#include <cyxwiz/sequential.h>

// Create and train model
cyxwiz::SequentialModel model;
model.Add(std::make_unique<cyxwiz::LinearModule>(784, 512));
model.Add(std::make_unique<cyxwiz::ReLUModule>());
model.Add(std::make_unique<cyxwiz::LinearModule>(512, 10));

// ... train the model ...

// Save with metadata
model.SetName("My Model");
model.SetDescription("Trained on custom dataset");
model.Save("path/to/model.cyxmodel");  // .cyxmodel extension added automatically if missing
```

### Loading a Model

```cpp
#include <cyxwiz/sequential.h>

// Create model with same architecture
cyxwiz::SequentialModel model;
model.Add(std::make_unique<cyxwiz::LinearModule>(784, 512));
model.Add(std::make_unique<cyxwiz::ReLUModule>());
model.Add(std::make_unique<cyxwiz::LinearModule>(512, 10));

// Load weights from .cyxmodel file
if (model.Load("path/to/model.cyxmodel")) {
    model.SetTraining(false);  // Set to evaluation mode

    // Use for inference
    auto output = model.Forward(input);
}
```

**Important:** The model architecture must be set up before calling `Load()`. The loader validates that the number of modules matches.

### Using the UI

1. Go to **Tools > Model Export > Save Trained Model...** (or press `Ctrl+Shift+S`)
2. Choose a location and filename (e.g., `mnist_classifier.cyxmodel`)
3. The model is saved as a single `.cyxmodel` file

## Python API

```python
import pycyxwiz as cx

# After training
model = cx.SequentialModel()
model.add(cx.LinearLayer(784, 512))
model.add(cx.ReLU())
model.add(cx.LinearLayer(512, 10))

# Train...

# Save
model.set_name("MNIST Classifier")
model.save("models/mnist_mlp.cyxmodel")

# Load
model2 = cx.SequentialModel()
model2.add(cx.LinearLayer(784, 512))
model2.add(cx.ReLU())
model2.add(cx.LinearLayer(512, 10))
model2.load("models/mnist_mlp.cyxmodel")
```

## Error Handling

The `Save()` and `Load()` methods return `bool` indicating success/failure. Errors are logged via spdlog:

| Error | Cause | Solution |
|-------|-------|----------|
| "Failed to create file" | Permission denied or invalid path | Check write permissions |
| "Failed to open file" | File doesn't exist or path issue | Verify file path |
| "Invalid magic number" | Not a CyxWiz model file | Ensure file has `.cyxmodel` extension and is valid |
| "Unsupported format version" | File from different CyxWiz version | Check version compatibility below |
| "Module count mismatch" | Architecture doesn't match saved model | Ensure same number of layers |

## Version Compatibility

| Format Version | CyxWiz Version | Notes |
|----------------|----------------|-------|
| 1 | 0.1.0 - 0.2.x | Legacy two-file format (.json + .bin) |
| 2 | 0.3.0+ | Single `.cyxmodel` file (current) |

The current version (2) does not load version 1 files. If you have legacy models, use CyxWiz 0.2.x to convert them.

## Best Practices

1. **Use `.cyxmodel` extension** for consistent file naming
2. **Always set model name/description** for easier identification later
3. **Save to project's `models/` directory** for organization
4. **Include training info in description** (dataset, epochs, accuracy)
5. **Verify architecture matches** before loading weights
6. **Set evaluation mode** after loading for inference (`SetTraining(false)`)

## Supported Data Types

| DataType | Description |
|----------|-------------|
| float32 | 32-bit floating point (default) |
| float64 | 64-bit floating point |
| int32 | 32-bit integer |
| int64 | 64-bit integer |
| uint8 | 8-bit unsigned integer |

## Future Enhancements

- [ ] ONNX export for interoperability
- [ ] Architecture serialization (load without pre-defining layers)
- [ ] Checkpoint saving during training
- [ ] Model compression options
- [ ] Encryption for model protection
