# Activation Functions

pycyxwiz provides both class-based and functional activation functions for neural networks.

## Overview

```python
import pycyxwiz as cx

# Class-based
relu = cx.ReLU()
output = relu.forward(input_tensor)

# Functional
output = cx.relu(input_tensor)
```

## Class-Based Activations

All class-based activations inherit from `cx.Activation` and provide:
- `forward(input)`: Apply activation
- `backward(grad_output, input)`: Compute gradients

### ReLU

Rectified Linear Unit: `f(x) = max(0, x)`

```python
relu = cx.ReLU()

# Forward
output = relu.forward(input)  # max(0, x)

# Backward
grad_input = relu.backward(grad_output, input)  # 1 if x > 0 else 0
```

**Characteristics:**
- Simple and fast
- Sparsity (zeros out negative values)
- "Dying ReLU" problem for large negative inputs

---

### Sigmoid

Logistic function: `f(x) = 1 / (1 + exp(-x))`

```python
sigmoid = cx.Sigmoid()
output = sigmoid.forward(input)  # Values in (0, 1)
```

**Characteristics:**
- Output range: (0, 1)
- Good for binary classification output
- Can suffer from vanishing gradients

**Mathematical form:**
```
f(x) = σ(x) = 1 / (1 + e^(-x))
f'(x) = σ(x) * (1 - σ(x))
```

---

### Tanh

Hyperbolic tangent: `f(x) = tanh(x)`

```python
tanh = cx.Tanh()
output = tanh.forward(input)  # Values in (-1, 1)
```

**Characteristics:**
- Output range: (-1, 1)
- Zero-centered (unlike sigmoid)
- Still has vanishing gradient issues

**Mathematical form:**
```
f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
f'(x) = 1 - tanh²(x)
```

---

### GELU

Gaussian Error Linear Unit: `f(x) = x * Φ(x)`

```python
gelu = cx.GELU()
output = gelu.forward(input)
```

**Characteristics:**
- Used in BERT, GPT, and modern transformers
- Smooth approximation of ReLU
- Non-monotonic (allows small negative values)

**Approximation:**
```
f(x) ≈ 0.5x * (1 + tanh(√(2/π) * (x + 0.044715x³)))
```

---

### LeakyReLU

ReLU with non-zero slope for negative values.

```python
leaky_relu = cx.LeakyReLU(negative_slope=0.01)
output = leaky_relu.forward(input)

# Access slope value
print(leaky_relu.alpha)  # 0.01
```

**Parameters:**
- `negative_slope` (float): Slope for x < 0. Default: 0.01

**Mathematical form:**
```
f(x) = x if x > 0 else α * x
f'(x) = 1 if x > 0 else α
```

---

### ELU

Exponential Linear Unit.

```python
elu = cx.ELU(alpha=1.0)
output = elu.forward(input)

print(elu.alpha)  # 1.0
```

**Parameters:**
- `alpha` (float): Scale for negative values. Default: 1.0

**Mathematical form:**
```
f(x) = x if x > 0 else α * (exp(x) - 1)
```

**Characteristics:**
- Smooth for negative values
- Can produce negative outputs
- Self-normalizing for α ≈ 1.67 (SELU variant)

---

### Swish / SiLU

Self-gated activation: `f(x) = x * sigmoid(x)`

```python
swish = cx.Swish()
output = swish.forward(input)

# SiLU is an alias
silu = cx.SiLU()  # Same as Swish
```

**Characteristics:**
- Used in EfficientNet, modern CNNs
- Smooth and non-monotonic
- Better gradient flow than ReLU

**Mathematical form:**
```
f(x) = x * σ(x) = x / (1 + e^(-x))
```

---

### Mish

Smooth, self-regularizing activation.

```python
mish = cx.Mish()
output = mish.forward(input)
```

**Mathematical form:**
```
f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
```

**Characteristics:**
- Similar to Swish but smoother
- Used in YOLOv4
- Computationally more expensive

---

### Hardswish

Efficient approximation of Swish.

```python
hardswish = cx.Hardswish()
output = hardswish.forward(input)
```

**Mathematical form:**
```
f(x) = 0 if x ≤ -3
     = x if x ≥ 3
     = x * (x + 3) / 6 otherwise
```

**Characteristics:**
- Faster than Swish (no sigmoid)
- Used in MobileNetV3
- Good for mobile/edge deployment

---

### Softmax

Normalizes to probability distribution.

```python
softmax = cx.Softmax(dim=-1)
output = softmax.forward(logits)  # Sum to 1 along dim
```

**Parameters:**
- `dim` (int): Dimension to apply softmax. Default: -1 (last dimension)

**Mathematical form:**
```
f(x_i) = exp(x_i) / Σ exp(x_j)
```

**Usage:**
```python
# For classification (batch of logits)
logits = cx.Tensor.random([32, 10])  # Batch 32, 10 classes
softmax = cx.Softmax(dim=-1)
probabilities = softmax.forward(logits)  # Each row sums to 1
```

## Functional API

All activations are also available as functions in the `cx` module:

```python
import pycyxwiz as cx

# Direct application
output = cx.relu(input)
output = cx.sigmoid(input)
output = cx.tanh(input)
output = cx.softmax(input, dim=-1)
output = cx.gelu(input)
output = cx.leaky_relu(input, negative_slope=0.01)
output = cx.elu(input, alpha=1.0)
output = cx.swish(input)
output = cx.silu(input)  # Alias for swish
output = cx.mish(input)
```

### Function Signatures

| Function | Signature |
|----------|-----------|
| `relu(input)` | ReLU activation |
| `sigmoid(input)` | Sigmoid activation |
| `tanh(input)` | Tanh activation |
| `softmax(input, dim=-1)` | Softmax normalization |
| `gelu(input)` | GELU activation |
| `leaky_relu(input, negative_slope=0.01)` | Leaky ReLU |
| `elu(input, alpha=1.0)` | ELU activation |
| `swish(input)` | Swish/SiLU activation |
| `silu(input)` | Alias for swish |
| `mish(input)` | Mish activation |

## Utility Functions

### `flatten(input)`

Flatten spatial dimensions.

```python
# [batch, channels, height, width] → [batch, channels * height * width]
flattened = cx.flatten(input)
```

---

### `dropout(input, p=0.5, training=True)`

Apply dropout regularization.

```python
# During training
output = cx.dropout(input, p=0.5, training=True)

# During evaluation (no dropout)
output = cx.dropout(input, p=0.5, training=False)
```

**Parameters:**
- `input`: Input tensor
- `p` (float): Dropout probability. Default: 0.5
- `training` (bool): Whether in training mode. Default: True

## Comparison Table

| Activation | Range | Gradient | Use Case |
|------------|-------|----------|----------|
| ReLU | [0, ∞) | Sparse | Default for hidden layers |
| Sigmoid | (0, 1) | Vanishing | Binary classification output |
| Tanh | (-1, 1) | Vanishing | Hidden layers, RNNs |
| GELU | (-0.17, ∞) | Smooth | Transformers |
| LeakyReLU | (-∞, ∞) | Non-zero | Prevent dying ReLU |
| ELU | (-α, ∞) | Smooth | Deep networks |
| Swish | (-0.28, ∞) | Smooth | Modern CNNs |
| Mish | (-0.31, ∞) | Smooth | Object detection |
| Softmax | (0, 1) | - | Multi-class output |

## Visualization

```python
import pycyxwiz as cx
import numpy as np
import matplotlib.pyplot as plt

# Input range
x = np.linspace(-5, 5, 100)
x_list = x.tolist()

# Create activations
activations = {
    'ReLU': lambda x: cx.relu(cx.Tensor.from_numpy(np.array([x]).astype(np.float32))).to_numpy()[0],
    'Sigmoid': lambda x: cx.sigmoid(cx.Tensor.from_numpy(np.array([x]).astype(np.float32))).to_numpy()[0],
    'Tanh': lambda x: cx.tanh(cx.Tensor.from_numpy(np.array([x]).astype(np.float32))).to_numpy()[0],
    'GELU': lambda x: cx.gelu(cx.Tensor.from_numpy(np.array([x]).astype(np.float32))).to_numpy()[0],
    'Swish': lambda x: cx.swish(cx.Tensor.from_numpy(np.array([x]).astype(np.float32))).to_numpy()[0],
}

# Simplified visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for ax, (name, _) in zip(axes, activations.items()):
    # Compute activation values
    if name == 'ReLU':
        y = np.maximum(0, x)
    elif name == 'Sigmoid':
        y = 1 / (1 + np.exp(-x))
    elif name == 'Tanh':
        y = np.tanh(x)
    elif name == 'GELU':
        y = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    elif name == 'Swish':
        y = x / (1 + np.exp(-x))

    ax.plot(x, y, 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_title(name)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-2, 5)

axes[-1].axis('off')  # Hide extra subplot
plt.tight_layout()
plt.show()
```

## Best Practices

### Hidden Layers
```python
# Default choice
relu = cx.ReLU()

# For deep networks (prevent dying ReLU)
leaky = cx.LeakyReLU(negative_slope=0.1)

# For transformers
gelu = cx.GELU()
```

### Output Layers
```python
# Binary classification
sigmoid = cx.Sigmoid()

# Multi-class classification
softmax = cx.Softmax(dim=-1)

# Regression (no activation)
# output = layer.forward(input)
```

### Choosing Activations
1. **Start with ReLU** for most cases
2. **Use GELU** for transformer architectures
3. **Use Swish/SiLU** for efficient modern CNNs
4. **Use LeakyReLU/ELU** if you see dying neurons
5. **Use Sigmoid** only for output, not hidden layers

---

**Back to**: [pycyxwiz Index](index.md)
