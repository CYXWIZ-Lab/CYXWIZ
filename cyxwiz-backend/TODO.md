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
- [ ] Attention / Transformer layers
- [ ] Embedding layer

### Activations
- [ ] Softmax activation
- [ ] LeakyReLU activation
- [ ] GELU activation
- [ ] Swish/SiLU activation

### Loss Functions
- [ ] Huber Loss
- [ ] Focal Loss
- [ ] Triplet Loss
- [ ] Contrastive Loss

### Optimizers
- [ ] AdaGrad optimizer
- [ ] Adadelta optimizer
- [ ] LAMB optimizer
- [ ] Learning rate warmup support

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
- [ ] Complete pybind11 bindings for all classes
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

- [x] Tensor class with ArrayFire backend
- [x] Device abstraction (CPU/CUDA/OpenCL)
- [x] Linear layer (Dense layer)
- [x] Conv2D layer (2D convolution with im2col)
- [x] MaxPool2D layer
- [x] AvgPool2D layer
- [x] GlobalAvgPool2D layer
- [x] BatchNorm2D layer
- [x] Dropout layer
- [x] Flatten layer
- [x] LSTM layer (multi-layer, bidirectional, dropout)
- [x] GRU layer (multi-layer, bidirectional, dropout)
- [x] ReLU, Sigmoid, Tanh activations
- [x] MSE, CrossEntropy, BCE loss functions
- [x] SGD, Adam, AdamW, RMSprop optimizers
- [x] Sequential model container
- [x] Learning rate schedulers (Step, Exponential, Cosine)
- [x] MNIST DataLoader
- [x] Data normalization and standardization
- [x] Clustering algorithms (K-Means, etc.)
- [x] Dimensionality reduction (PCA, t-SNE, UMAP)
- [x] Signal processing (FFT, Convolution, Filters)
- [x] Time series analysis
- [x] Text processing utilities

---

*Last updated: 2024-12-26*
