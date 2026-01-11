# Flatten Node Design

## Overview

The Flatten node is a critical layer for transitioning from convolutional (4D) to fully-connected (2D) layers in neural networks.

## Problem Statement

**Image Data Shape:**
- Input: `[batch, H, W, C]` (4D tensor)
- Conv2D/MaxPool2D layers: Accept 4D input
- Dense (fully-connected) layers: Accept 2D input `[batch, features]`

**Transition Requirement:**
When going from CNN layers → Dense layers, we need to **flatten** from `[batch, H, W, C]` → `[batch, H*W*C]`

## Design Decision: Node Editor Approach

**Selected Solution**: Add explicit Flatten node in Node Editor (matches PyTorch/TensorFlow conventions)

### Why Node Editor?

✅ **Standard ML Practice** - All major frameworks use explicit Flatten layers
✅ **Educational** - Users learn about CNN→FC transition
✅ **Flexible** - Can flatten at any point in the network
✅ **Explicit Control** - User sees the operation in the graph
✅ **Code Generation** - Can export correct Flatten() code

### Comparison with Other Frameworks

**PyTorch:**
```python
model = nn.Sequential(
    nn.Conv2d(3, 32, 3),
    nn.Flatten(),        # ← Explicit
    nn.Linear(32*26*26, 128)
)
```

**TensorFlow/Keras:**
```python
model = keras.Sequential([
    layers.Conv2D(32, 3),
    layers.Flatten(),    # ← Explicit
    layers.Dense(128)
])
```

**CyxWiz (Proposed):**
```
Input[28,28,1] → Conv2D[32] → ReLU → Flatten → Dense[128] → Softmax[10]
```

## Implementation Plan

### 1. Add NodeType::Flatten

**File**: `cyxwiz-engine/src/gui/node_editor.h`

```cpp
enum class NodeType {
    // ... existing types ...
    Flatten,  // NEW: Reshape [B,H,W,C] → [B,H*W*C]
};
```

### 2. Add Flatten Node Creation

**File**: `cyxwiz-engine/src/gui/node_editor_nodes.cpp`

```cpp
case NodeType::Flatten: {
    node.display_name = "Flatten";
    node.category = NodeCategory::Layer;
    node.color = IM_COL32(200, 150, 100, 255);

    node.inputs = {{"input", SocketType::Tensor}};
    node.outputs = {{"output", SocketType::Tensor}};

    node.description = "Flattens input from [B,H,W,C] to [B,H*W*C] for Dense layers";
}
```

### 3. Shape Inference

**File**: `cyxwiz-engine/src/gui/node_editor_validation.cpp`

```cpp
std::vector<size_t> InferFlattenOutputShape(const std::vector<size_t>& input_shape) {
    if (input_shape.size() == 4) {
        // [batch, H, W, C] → [batch, H*W*C]
        return {input_shape[0],
                input_shape[1] * input_shape[2] * input_shape[3]};
    } else if (input_shape.size() == 2) {
        // Already flattened, pass through
        return input_shape;
    }
    return {};  // Invalid
}
```

### 4. Smart Validation Warnings

**File**: `cyxwiz-engine/src/gui/node_editor_validation.cpp`

Detect Conv2D → Dense connection without Flatten:

```cpp
if (source_node.type == NodeType::Conv2D &&
    target_node.type == NodeType::Dense &&
    !HasFlattenInBetween(source_node, target_node)) {

    warnings.push_back({
        .node_id = target_node.id,
        .severity = ValidationSeverity::Warning,
        .category = ValidationCategory::ShapeMismatch,
        .message = "Dense layer expects 2D input but receiving 4D from Conv2D. "
                  "Insert a Flatten node between them.",
        .suggested_fix = "Auto-insert Flatten node"
    });
}
```

### 5. Auto-Insert Feature

Show dialog when invalid connection attempted:

```
┌──────────────────────────────────────────┐
│  Shape Mismatch Detected                 │
├──────────────────────────────────────────┤
│  Conv2D output: [batch, 28, 28, 32]      │
│  Dense input:   [batch, features]        │
│                                           │
│  Dense layers require flattened input.   │
│  Would you like to auto-insert Flatten?  │
│                                           │
│  [Auto-Insert Flatten] [Manual] [Cancel] │
└──────────────────────────────────────────┘
```

**Implementation**:
```cpp
void NodeEditor::ShowAutoInsertFlattenDialog(NodeID source_id, NodeID target_id) {
    ImGui::OpenPopup("Auto-Insert Flatten");

    if (ImGui::BeginPopupModal("Auto-Insert Flatten")) {
        ImGui::Text("Shape Mismatch Detected");
        ImGui::Separator();

        auto& source = GetNode(source_id);
        auto& target = GetNode(target_id);

        ImGui::Text("Source: %s [%s]", source.display_name.c_str(),
                    FormatShape(source.output_shape).c_str());
        ImGui::Text("Target: %s [%s]", target.display_name.c_str(),
                    FormatShape(target.input_shape).c_str());

        ImGui::Spacing();
        ImGui::TextWrapped("Dense layers require 2D input. Insert Flatten node?");

        if (ImGui::Button("Auto-Insert Flatten")) {
            // Create Flatten node between source and target
            auto flatten_id = CreateNode(NodeType::Flatten);
            auto& flatten = GetNode(flatten_id);

            // Position between source and target
            flatten.pos = {
                (source.pos.x + target.pos.x) / 2,
                (source.pos.y + target.pos.y) / 2
            };

            // Reconnect: source → flatten → target
            DisconnectNodes(source_id, target_id);
            ConnectNodes(source_id, flatten_id);
            ConnectNodes(flatten_id, target_id);

            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Manual")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
            DisconnectNodes(source_id, target_id);
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}
```

### 6. Code Generation

**File**: `cyxwiz-engine/src/gui/node_editor_codegen.cpp`

**PyTorch:**
```cpp
case NodeType::Flatten:
    pytorch_code += indent + "nn.Flatten(),\n";
    break;
```

**TensorFlow/Keras:**
```cpp
case NodeType::Flatten:
    tensorflow_code += indent + "layers.Flatten(),\n";
    break;
```

**PyCyxWiz:**
```cpp
case NodeType::Flatten:
    pycyxwiz_code += indent + "cyxwiz.Flatten(),\n";
    break;
```

### 7. Backend Implementation

**File**: `cyxwiz-backend/include/cyxwiz/layer.h`

```cpp
class FlattenLayer : public Layer {
public:
    FlattenLayer() = default;

    Tensor Forward(const Tensor& input) override {
        // Cache original shape for backward pass
        original_shape_ = input.Shape();

        // Flatten: [B, H, W, C] → [B, H*W*C]
        size_t batch_size = original_shape_[0];
        size_t flattened_size = 1;
        for (size_t i = 1; i < original_shape_.size(); ++i) {
            flattened_size *= original_shape_[i];
        }

        return input.Reshape({batch_size, flattened_size});
    }

    Tensor Backward(const Tensor& grad_output) override {
        // Reshape gradient back to original shape
        return grad_output.Reshape(original_shape_);
    }

private:
    std::vector<size_t> original_shape_;
};
```

## Use Cases

### Example 1: LeNet-5 (Classic CNN)

```
Input[28,28,1]
  ↓
Conv2D[6, kernel=5]
  ↓
ReLU
  ↓
MaxPool2D[2]
  ↓
Conv2D[16, kernel=5]
  ↓
ReLU
  ↓
MaxPool2D[2]
  ↓
Flatten              ← Critical transition point
  ↓
Dense[120]
  ↓
ReLU
  ↓
Dense[84]
  ↓
ReLU
  ↓
Dense[10]
```

### Example 2: Multi-Branch Architecture

```
Input[224,224,3]
      ↓
    Conv2D[64]
      ↓
    Split───────┐
      ↓         ↓
  Branch A   Branch B
  Conv2D     Conv2D
    ↓          ↓
  Flatten   Flatten  ← Multiple flatten points
    ↓          ↓
  Dense      Dense
      ↓         ↓
    Concat──────┘
      ↓
   Dense[10]
```

## Node Documentation

**Node Name**: Flatten

**Category**: Layer

**Description**: Reshapes input tensor from 4D (image) to 2D (vector) format for fully-connected layers.

**Input Shape**: `[batch, height, width, channels]`

**Output Shape**: `[batch, height * width * channels]`

**Parameters**: None

**Use Case**: Required when transitioning from convolutional layers to dense layers in CNNs.

**Common Patterns**:
- After last Conv2D/MaxPool2D before Dense layers
- In encoder-decoder architectures at bottleneck
- In multi-task learning when flattening different branches

**Example Networks Using Flatten**:
- LeNet-5
- AlexNet
- VGG
- ResNet (via GlobalAveragePooling or Flatten)

## Alternative: Why NOT Auto-Flatten in Data Pipeline?

**Problems with automatic flattening:**

❌ **Breaks CNN architectures**:
```
Input → Conv2D → (auto-flatten?) → Conv2D  # WRONG! Conv2D needs 4D
```

❌ **Hidden behavior**:
- User doesn't understand the transformation
- Makes debugging shape mismatches harder
- Exported code unclear

❌ **Loss of flexibility**:
- Can't control where flattening happens
- Doesn't support multi-branch architectures
- Violates explicit-is-better-than-implicit principle

## Testing Checklist

Once implemented, test the following scenarios:

- [ ] Create CNN with Flatten node (LeNet-5 architecture)
- [ ] Validate shape inference: [batch,28,28,32] → [batch,25088]
- [ ] Test validation warning: Conv2D → Dense (no Flatten)
- [ ] Test auto-insert feature
- [ ] Export to PyTorch code (verify nn.Flatten() generated)
- [ ] Export to TensorFlow code (verify layers.Flatten() generated)
- [ ] Train model with Flatten node (verify forward/backward pass)
- [ ] Test multi-branch architecture with multiple Flatten nodes
- [ ] Verify backward pass shape restoration
- [ ] Test edge case: already flattened input (should pass through)

## Estimated Effort

**Implementation Time**: 2-3 hours

**Breakdown**:
- Add NodeType enum and node creation: 30 min
- Shape inference logic: 30 min
- Validation warnings: 30 min
- Auto-insert dialog: 45 min
- Code generation (3 backends): 30 min
- Backend FlattenLayer implementation: 30 min
- Testing: 30 min

## Priority

**High Priority** - Required for any CNN architecture that uses Dense layers (which is most classification CNNs).

## Implementation Status

✅ **COMPLETE** - All features implemented and integrated

**Implementation Date**: 2026-01-09

**Completed Features**:
- ✅ Basic Flatten node (creation, pins, parameters)
- ✅ Shape inference system (automatic propagation throughout graph)
- ✅ Smart validation warnings (Conv2D→Dense detection)
- ✅ Auto-insert dialog (3-button modal with one-click fix)
- ✅ Visual warning indicators (yellow triangle icons on nodes)
- ✅ Code generation (PyTorch, TensorFlow, Keras, PyCyxWiz)
- ✅ Backend layer (FlattenLayer forward/backward passes with ArrayFire)
- ✅ Python bindings (pycyxwiz.Flatten())

**Usage**:
1. Create a CNN architecture in the Node Editor
2. Try to connect Conv2D/MaxPool2D → Dense directly
3. Dialog appears offering to auto-insert Flatten
4. Click "Auto-Insert Flatten" for one-click fix
5. OR manually add Flatten node to avoid warnings

**Known Limitations**:
- Only detects Conv2D/MaxPool2D→Dense pattern (can be extended to other 4D→2D mismatches)
- Assumes batch-first tensor layout (standard PyTorch/TensorFlow convention)
- Shape inference doesn't handle dynamic shapes (e.g., -1 dimensions)

---

**Document Version**: 1.0
**Created**: 2026-01-09
**Author**: CyxWiz Development Team
