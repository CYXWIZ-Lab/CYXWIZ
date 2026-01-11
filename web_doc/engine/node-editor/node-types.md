# Node Types Reference

Complete reference for all 80+ node types available in the CyxWiz Node Editor.

## Data Pipeline Nodes

### Dataset Input
**Category:** Data Pipeline
**Purpose:** Load data from the Dataset Registry

**Inputs:** None
**Outputs:**
- `data` (Tensor) - Feature data
- `labels` (Labels) - Target labels

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| dataset_name | string | "" | Name of loaded dataset |

**Usage:** First node in any training pipeline. Displays the currently loaded dataset name.

---

### Data Loader
**Category:** Data Pipeline
**Purpose:** Batch iterator with shuffle and drop_last options

**Inputs:**
- `data` (Tensor)
- `labels` (Labels)

**Outputs:**
- `batches` (Tensor) - Batched data

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| batch_size | int | 32 | Samples per batch |
| shuffle | bool | true | Shuffle each epoch |
| drop_last | bool | false | Drop incomplete batch |

---

### Augmentation
**Category:** Data Pipeline
**Purpose:** Apply data augmentation transforms

**Inputs:**
- `data` (Tensor)

**Outputs:**
- `augmented` (Tensor)

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| preset | enum | "ImageNet" | Transform preset |
| random_crop | bool | true | Enable random cropping |
| horizontal_flip | bool | true | Random horizontal flip |
| rotation | float | 15.0 | Max rotation degrees |

**Presets:** ImageNet, CIFAR-10, Medical, Self-Supervised, etc.

---

### Data Split
**Category:** Data Pipeline
**Purpose:** Split data into train/validation/test sets

**Inputs:**
- `data` (Tensor)
- `labels` (Labels)

**Outputs:**
- `train_data` (Tensor)
- `val_data` (Tensor)
- `test_data` (Tensor)

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| train_ratio | float | 0.7 | Training set ratio |
| val_ratio | float | 0.15 | Validation set ratio |
| shuffle | bool | true | Shuffle before split |
| stratify | bool | true | Maintain class ratios |

---

### Normalize
**Category:** Data Pipeline
**Purpose:** Normalize input values

**Inputs:**
- `data` (Tensor)

**Outputs:**
- `normalized` (Tensor)

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| mean | float[] | [0.485, 0.456, 0.406] | Channel means |
| std | float[] | [0.229, 0.224, 0.225] | Channel stds |

---

### One-Hot Encode
**Category:** Data Pipeline
**Purpose:** Convert labels to one-hot vectors

**Inputs:**
- `labels` (Labels)

**Outputs:**
- `one_hot` (Tensor)

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| num_classes | int | -1 | Number of classes (-1 = auto) |

---

## Core Layers

### Dense (Fully Connected)
**Category:** Core Layers
**Purpose:** Fully connected layer

**Inputs:**
- `input` (Tensor)

**Outputs:**
- `output` (Tensor)

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| units | int | 128 | Output features |
| use_bias | bool | true | Include bias term |
| activation | enum | "None" | Built-in activation |

**Shape:** `[batch, in_features]` -> `[batch, units]`

---

### Conv1D
**Category:** Core Layers
**Purpose:** 1D convolution for sequences

**Inputs:**
- `input` (Tensor) - Shape: [batch, channels, length]

**Outputs:**
- `output` (Tensor)

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| filters | int | 64 | Number of filters |
| kernel_size | int | 3 | Convolution kernel size |
| stride | int | 1 | Stride |
| padding | int | 0 | Zero padding |
| dilation | int | 1 | Dilation factor |

---

### Conv2D
**Category:** Core Layers
**Purpose:** 2D convolution for images

**Inputs:**
- `input` (Tensor) - Shape: [batch, channels, height, width]

**Outputs:**
- `output` (Tensor)

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| filters | int | 64 | Number of filters |
| kernel_size | int | 3 | Kernel size (square) |
| stride | int | 1 | Stride |
| padding | int | 0 | Zero padding |
| dilation | int | 1 | Dilation factor |
| groups | int | 1 | Group convolution |

---

### Conv3D
**Category:** Core Layers
**Purpose:** 3D convolution for volumes/video

**Parameters:** Same as Conv2D, but 3D versions

---

### DepthwiseConv2D
**Category:** Core Layers
**Purpose:** Depthwise separable convolution

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| kernel_size | int | 3 | Kernel size |
| stride | int | 1 | Stride |
| padding | int | 0 | Padding |
| depth_multiplier | int | 1 | Channel multiplier |

---

### MaxPool2D
**Category:** Core Layers
**Purpose:** Max pooling

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| kernel_size | int | 2 | Pool window size |
| stride | int | 2 | Stride |
| padding | int | 0 | Padding |

---

### AvgPool2D
**Category:** Core Layers
**Purpose:** Average pooling

**Parameters:** Same as MaxPool2D

---

### GlobalMaxPool
**Category:** Core Layers
**Purpose:** Global max pooling (reduce spatial dims)

**Shape:** `[batch, channels, H, W]` -> `[batch, channels]`

---

### GlobalAvgPool
**Category:** Core Layers
**Purpose:** Global average pooling

**Shape:** `[batch, channels, H, W]` -> `[batch, channels]`

---

### AdaptiveAvgPool
**Category:** Core Layers
**Purpose:** Adaptive average pooling to target size

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| output_size | int[] | [1, 1] | Target output size |

---

## Normalization Layers

### BatchNorm
**Category:** Normalization
**Purpose:** Batch normalization

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| num_features | int | -1 | Features (-1 = auto) |
| momentum | float | 0.1 | Running stats momentum |
| eps | float | 1e-5 | Numerical stability |
| affine | bool | true | Learnable params |

---

### LayerNorm
**Category:** Normalization
**Purpose:** Layer normalization

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| normalized_shape | int[] | [-1] | Shape to normalize |
| eps | float | 1e-5 | Numerical stability |

---

### GroupNorm
**Category:** Normalization
**Purpose:** Group normalization

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| num_groups | int | 32 | Number of groups |
| num_channels | int | -1 | Channels (-1 = auto) |

---

### InstanceNorm
**Category:** Normalization
**Purpose:** Instance normalization

**Parameters:** Similar to BatchNorm

---

## Regularization

### Dropout
**Category:** Regularization
**Purpose:** Randomly zero elements during training

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| p | float | 0.5 | Dropout probability |
| inplace | bool | false | In-place operation |

---

### Flatten
**Category:** Regularization
**Purpose:** Flatten tensor to 2D

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| start_dim | int | 1 | First dim to flatten |
| end_dim | int | -1 | Last dim to flatten |

**Shape:** `[batch, C, H, W]` -> `[batch, C*H*W]`

---

## Recurrent Layers

### RNN
**Category:** Recurrent
**Purpose:** Simple RNN layer

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| hidden_size | int | 128 | Hidden state size |
| num_layers | int | 1 | Stacked layers |
| bidirectional | bool | false | Bidirectional |
| dropout | float | 0.0 | Dropout between layers |

---

### LSTM
**Category:** Recurrent
**Purpose:** Long Short-Term Memory

**Parameters:** Same as RNN, plus:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| proj_size | int | 0 | Projection size |

---

### GRU
**Category:** Recurrent
**Purpose:** Gated Recurrent Unit

**Parameters:** Same as RNN

---

### Bidirectional
**Category:** Recurrent
**Purpose:** Wrapper for bidirectional RNN

---

### TimeDistributed
**Category:** Recurrent
**Purpose:** Apply layer to each timestep

---

### Embedding
**Category:** Recurrent
**Purpose:** Token embedding lookup

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| num_embeddings | int | 10000 | Vocabulary size |
| embedding_dim | int | 256 | Embedding dimension |
| padding_idx | int | -1 | Padding token index |

---

## Attention & Transformer

### MultiHeadAttention
**Category:** Attention
**Purpose:** Multi-head attention mechanism

**Inputs:**
- `query` (Tensor)
- `key` (Tensor)
- `value` (Tensor)

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| embed_dim | int | 512 | Embedding dimension |
| num_heads | int | 8 | Number of heads |
| dropout | float | 0.0 | Attention dropout |

---

### SelfAttention
**Category:** Attention
**Purpose:** Self-attention (Q=K=V)

**Inputs:**
- `input` (Tensor) - Used for Q, K, and V

---

### CrossAttention
**Category:** Attention
**Purpose:** Cross-attention (Q from one source, K/V from another)

**Inputs:**
- `query` (Tensor)
- `context` (Tensor) - Used for K and V

---

### LinearAttention
**Category:** Attention
**Purpose:** O(n) linear attention (Performer-style)

---

### TransformerEncoder
**Category:** Attention
**Purpose:** Full transformer encoder block

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| d_model | int | 512 | Model dimension |
| nhead | int | 8 | Attention heads |
| dim_feedforward | int | 2048 | FFN dimension |
| dropout | float | 0.1 | Dropout |
| num_layers | int | 6 | Encoder layers |

---

### TransformerDecoder
**Category:** Attention
**Purpose:** Full transformer decoder block

**Parameters:** Similar to encoder

---

### PositionalEncoding
**Category:** Attention
**Purpose:** Add positional information

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| d_model | int | 512 | Model dimension |
| max_len | int | 5000 | Max sequence length |
| dropout | float | 0.1 | Dropout |

---

## Activation Functions

| Node | Formula | Use Case |
|------|---------|----------|
| **ReLU** | max(0, x) | Most common, fast |
| **LeakyReLU** | max(0.01x, x) | Prevent dying neurons |
| **PReLU** | max(ax, x), a learned | Adaptive leaky |
| **ELU** | x if x>0, a(e^x-1) | Smooth negative |
| **SELU** | scale * ELU | Self-normalizing |
| **GELU** | x * Phi(x) | Transformers |
| **Swish** | x * sigmoid(x) | EfficientNet |
| **Mish** | x * tanh(softplus(x)) | YOLOv4 |
| **Sigmoid** | 1/(1+e^-x) | Binary output |
| **Tanh** | (e^x-e^-x)/(e^x+e^-x) | Range [-1,1] |
| **Softmax** | e^xi / sum(e^xj) | Classification |

---

## Shape Operations

### Reshape
**Purpose:** Change tensor shape

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| shape | int[] | [-1] | Target shape (-1 = infer) |

### Permute
**Purpose:** Reorder dimensions

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| dims | int[] | [0,2,1] | New dimension order |

### Squeeze
**Purpose:** Remove size-1 dimensions

### Unsqueeze
**Purpose:** Add dimension

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| dim | int | 0 | Dimension to add |

### View
**Purpose:** Reshape (must be contiguous)

### Split
**Purpose:** Split tensor into chunks

---

## Merge Operations

### Concatenate
**Purpose:** Concatenate tensors along dimension
**Variadic:** Accepts multiple inputs

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| dim | int | 1 | Concatenation dimension |

### Add
**Purpose:** Element-wise addition
**Variadic:** Accepts multiple inputs

### Multiply
**Purpose:** Element-wise multiplication

### Average
**Purpose:** Element-wise average

---

## Loss Functions

| Node | Use Case | Formula |
|------|----------|---------|
| **MSELoss** | Regression | mean((y-y')^2) |
| **CrossEntropyLoss** | Multi-class | -sum(y*log(y')) |
| **BCELoss** | Binary | -[y*log(y') + (1-y)*log(1-y')] |
| **BCEWithLogits** | Binary (logits) | Sigmoid + BCE |
| **L1Loss** | Robust regression | mean(abs(y-y')) |
| **SmoothL1Loss** | Object detection | Huber loss |
| **HuberLoss** | Robust | MSE if small, L1 if large |
| **NLLLoss** | After LogSoftmax | -log(py) |

---

## Optimizers

| Node | Algorithm | Good For |
|------|-----------|----------|
| **SGD** | Stochastic Gradient Descent | Simple, well-tuned |
| **Adam** | Adaptive moments | Most tasks |
| **AdamW** | Adam with weight decay | Transformers |
| **RMSprop** | Root mean square prop | RNNs |
| **Adagrad** | Adaptive gradient | Sparse data |
| **NAdam** | Nesterov Adam | Fast convergence |

**Common Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| lr | float | 0.001 | Learning rate |
| weight_decay | float | 0.0 | L2 regularization |
| momentum | float | 0.9 | Momentum (SGD) |
| betas | float[] | [0.9, 0.999] | Adam betas |

---

## Schedulers

| Node | Behavior |
|------|----------|
| **StepLR** | Multiply by gamma every step_size epochs |
| **CosineAnnealing** | Cosine annealing to min_lr |
| **ReduceOnPlateau** | Reduce when metric plateaus |
| **ExponentialLR** | Multiply by gamma each epoch |
| **WarmupScheduler** | Linear warmup then constant/decay |

---

## Output

### Output
**Purpose:** Mark final output of the model
**Required:** Every training graph must end with Output

**Inputs:**
- `prediction` (Tensor) - Model output
- `loss` (Loss) - Loss value for training

---

**Next**: [Connections](connections.md) | [Code Generation](code-generation.md)
