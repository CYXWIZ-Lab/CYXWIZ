# CyxWiz Backend TODO

This file tracks pending features and improvements for the cyxwiz-backend DLL.

## High Priority

### Inference Runtime
- [ ] Add inference capability to cyxwiz-backend (currently handled in cyxwiz-server-node)
- [ ] Consider lightweight inference API without heavy dependencies
- [ ] Support for running trained Sequential models directly
- [ ] Batch inference optimization

### DataLoader Enhancements
- [ ] Implement MNIST auto-download from http://yann.lecun.com/exdb/mnist/
- [ ] Add CIFAR-10/CIFAR-100 dataset support
- [ ] Add ImageNet dataset loader
- [ ] Add CSV/Tabular data loader
- [ ] Multi-threaded data loading (prefetch batches)
- [ ] Data augmentation pipeline integration

### Layers
- [x] Attention / Transformer layers (MultiHeadAttention, TransformerEncoder, TransformerDecoder)
- [x] Embedding layer (with pretrained weight loading, padding_idx, max_norm)

### Loss Functions
- [x] Huber Loss (implemented as SmoothL1Loss alias)
- [x] Focal Loss (for class imbalance, with alpha and gamma parameters)
- [x] Triplet Loss (metric learning with Euclidean/Cosine distance)
- [x] Contrastive Loss (similarity learning with margin)

### Optimizers
- [x] Adadelta optimizer
- [x] LAMB optimizer (Layer-wise Adaptive Moments for large batch training)
- [x] Learning rate warmup support (Linear and Cosine warmup)

## Medium Priority

### Model Serialization
- [ ] Save/Load Sequential model to .cyxmodel format
- [ ] Export to ONNX format
- [ ] Import from ONNX format
- [ ] PyTorch state_dict compatibility

### Memory Management
- [ ] Memory pool for tensor allocations
- [ ] Automatic memory defragmentation
- [ ] Memory usage statistics and limits

### Performance
- [ ] Tensor operation fusion
- [ ] Lazy evaluation for chained operations
- [ ] Multi-GPU support (data parallelism)
- [ ] Mixed precision training (FP16/BF16)

### Python Bindings
- [ ] NumPy array interoperability
- [ ] PyTorch tensor conversion
- [ ] Jupyter notebook integration

## Low Priority

### Utilities
- [ ] Model summary/print function
- [ ] Parameter count utility
- [ ] Gradient clipping utility
- [ ] Early stopping callback

### Testing
- [ ] Unit tests for all layers
- [ ] Integration tests for training loops
- [ ] Benchmark suite for performance regression
- [ ] Memory leak detection tests

### Documentation
- [ ] API documentation generation (Doxygen)
- [ ] Tutorial notebooks
- [ ] Performance tuning guide

---

## Completed

### Core
- [x] Tensor class with ArrayFire backend
- [x] Device abstraction (CPU/CUDA/OpenCL)
- [x] Sequential model container

### Layers
- [x] Linear layer (Dense layer)
- [x] Conv2D layer (2D convolution with im2col)
- [x] Conv1D layer (1D convolution)
- [x] MaxPool2D layer
- [x] AvgPool2D layer
- [x] GlobalAvgPool2D layer
- [x] BatchNorm2D layer
- [x] LayerNorm layer
- [x] InstanceNorm2D layer
- [x] GroupNorm layer
- [x] Dropout layer
- [x] Flatten layer
- [x] LSTM layer (multi-layer, bidirectional, dropout)
- [x] GRU layer (multi-layer, bidirectional, dropout)
- [x] Embedding layer (with pretrained weights, padding_idx, max_norm)
- [x] MultiHeadAttention layer
- [x] TransformerEncoderLayer (Pre-LN and Post-LN variants)
- [x] TransformerDecoderLayer (with cross-attention and causal masking)

### Activations
- [x] ReLU, Sigmoid, Tanh activations
- [x] LeakyReLU activation
- [x] ELU activation
- [x] SELU activation
- [x] PReLU activation (learnable)
- [x] GELU activation
- [x] Swish/SiLU activation
- [x] Softmax activation
- [x] Softplus, Softsign activations
- [x] Hardtanh, Hardsigmoid, Hardswish activations
- [x] Mish activation

### Loss Functions
- [x] MSE, CrossEntropy, BCE loss functions
- [x] Huber/SmoothL1 Loss
- [x] Focal Loss (class imbalance handling)
- [x] Triplet Loss (metric learning)
- [x] Contrastive Loss (similarity learning)

### Optimizers
- [x] SGD optimizer (with momentum, Nesterov)
- [x] Adam optimizer
- [x] AdamW optimizer
- [x] RMSprop optimizer (with ArrayFire GPU support)
- [x] AdaGrad optimizer (with ArrayFire GPU support)
- [x] NAdam optimizer (with ArrayFire GPU support)
- [x] Adadelta optimizer (with ArrayFire GPU support)
- [x] LAMB optimizer (Layer-wise Adaptive Moments for large batch training)
- [x] Learning rate schedulers (Step, Exponential, Cosine)
- [x] Learning rate warmup (Linear and Cosine warmup)

### Data
- [x] MNIST DataLoader
- [x] Data normalization and standardization

### Algorithms
- [x] Clustering algorithms (K-Means, etc.)
- [x] Dimensionality reduction (PCA, t-SNE, UMAP)
- [x] Signal processing (FFT, Convolution, Filters)
- [x] Time series analysis
- [x] Text processing utilities

### Python Bindings
- [x] Complete pybind11 bindings for all classes
- [x] All optimizers (SGD, Adam, AdamW, RMSprop, AdaGrad, NAdam, Adadelta, LAMB)
- [x] Learning rate warmup (LRWarmup class with Linear/Cosine warmup)
- [x] All layers (Conv1D, LayerNorm, InstanceNorm2D, GroupNorm, etc.)
- [x] All activations (SELU, PReLU, etc.)
- [x] Embedding layer bindings
- [x] LSTM and GRU layer bindings
- [x] MultiHeadAttention layer bindings
- [x] TransformerEncoderLayer and TransformerDecoderLayer bindings

### Examples
- [x] Python examples folder (examples/python/)
- [x] Embedding layer example (embedding_example.py)
- [x] LSTM layer example (lstm_example.py)
- [x] GRU layer example (gru_example.py)
- [x] Transformer layers example (transformer_example.py)

---

*Last updated: 2025-12-29*
