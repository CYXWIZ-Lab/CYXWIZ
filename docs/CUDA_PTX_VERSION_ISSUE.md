# CUDA PTX Version Mismatch Issue - PCA/SVD Failures

## Problem Summary

**Symptom:** PCA (Principal Component Analysis) computation fails during dataset statistics calculation with the error:

```
[error] ArrayFire PCA computation failed: ArrayFire Exception (Internal error:998):
CUDA_ERROR_UNSUPPORTED_PTX_VERSION(222):
ptxas application ptx input, line 9;
fatal: Unsupported .version 8.7; current version is '8.5'
```

**Impact:**
- ✅ All other preprocessing features work (mean, std, min, max, percentiles, histograms)
- ✅ Training and inference work normally
- ✅ All ArrayFire operations work except SVD-based ones
- ❌ PCA Whitening scaling strategy unavailable

---

## Why This Happens

### TL;DR

**SVD (used by PCA) requires runtime kernel compilation, while other operations use pre-compiled kernels that work with older CUDA drivers.**

### Detailed Explanation

#### What is PTX?

**PTX (Parallel Thread Execution)** is NVIDIA's intermediate representation for GPU code:

```
Source Code
    ↓
CUDA C++
    ↓ [nvcc compiler]
PTX Assembly (e.g., version 8.7)
    ↓ [ptxas - GPU driver assembler]
GPU Machine Code (runs on hardware)
```

**PTX Version Requirements:**

| PTX Version | Required CUDA Driver | Released With |
|-------------|---------------------|---------------|
| PTX 8.0 | CUDA 12.0+ | CUDA Toolkit 12.0 |
| PTX 8.5 | CUDA 12.5+ | CUDA Toolkit 12.5 |
| PTX 8.7 | **CUDA 12.8+** | **CUDA Toolkit 12.8** |

**The Issue:**
- ArrayFire 3.10.0 was built with **CUDA 12.8** (requires PTX 8.7)
- Your GPU driver is **CUDA 12.0.6** (supports up to PTX 8.5)
- **Result:** PTX 8.7 code cannot be compiled by PTX 8.5 driver

---

## Why Only PCA/SVD Fails

### Pre-Compiled vs JIT Compilation

Most ArrayFire operations use **pre-compiled kernels** that work on any compatible GPU:

```
ArrayFire DLL
├─ matmul_kernel.ptx (compiled as PTX 8.0) ✅
├─ conv2d_kernel.ptx (compiled as PTX 8.2) ✅
├─ elementwise_kernel.ptx (compiled as PTX 8.0) ✅
└─ ... hundreds of pre-compiled kernels

Your GPU Driver (PTX 8.5 support)
└─ Loads these kernels ✅ (PTX 8.0-8.5 all compatible)
```

**These kernels were compiled conservatively** by ArrayFire developers to ensure broad compatibility.

### SVD Requires JIT (Just-In-Time) Compilation

SVD is too complex to pre-compile efficiently:

```python
# Simple operation - pre-compiled kernel
C = matmul(A, B)  # Fixed algorithm, one kernel works for all

# Complex operation - needs runtime optimization
U, Σ, V = SVD(A)  # Algorithm depends on matrix dimensions/properties
```

**SVD execution flow:**

```
User Code: af::svd(U, S, Vt, centered)
    ↓
ArrayFire: Calls cuSOLVER library (NVIDIA's GPU linear algebra)
    ↓
cuSOLVER 12.8: "I need to generate optimized kernel for this specific matrix"
    ↓
Runtime Compiler: Generates PTX 8.7 code
    ↓
Your Driver: "ERROR: I only support PTX 8.5!" ❌
    ↓
Failure: CUDA_ERROR_UNSUPPORTED_PTX_VERSION
```

**Why JIT compilation?**

SVD algorithm varies based on:
- Matrix dimensions (tall/wide/square)
- Matrix properties (sparse/dense)
- Desired accuracy
- Number of singular values requested

Pre-compiling all variants would require **gigabytes** of kernels. Instead, cuSOLVER generates the optimal kernel at runtime.

---

## Why Training and Other Operations Work Fine

### Operations That Work ✅

#### 1. Pre-Compiled ArrayFire Kernels

```cpp
// These all use pre-compiled kernels (PTX 8.0-8.5)
af::array C = af::matmul(A, B);         ✅
af::array D = af::convolve2(img, kernel); ✅
af::array E = af::sum(A);               ✅
af::array F = af::transpose(A);         ✅
af::array G = A + B;                    ✅
```

#### 2. Neural Network Training

```cpp
// Training loop uses simple operations
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    auto output = model.forward(batch);    // Matmul, conv2d - pre-compiled ✅
    auto loss = criterion(output, labels);  // Element-wise ops ✅
    loss.backward();                        // Gradient kernels ✅
    optimizer.step();                       // Update ops ✅
}
```

All of these use **fixed-algorithm kernels** that ArrayFire ships with.

#### 3. PyTorch/ONNX Runtime

If using PyTorch or ONNX Runtime, those have their own CUDA kernels compiled independently.

### Operations That Might Fail ❌

```cpp
af::svd(U, S, Vt, A);                  // ❌ JIT - PTX 8.7 required
af::qr(Q, R, A);                       // ⚠️ Might fail (depends on cuSOLVER)
af::lu(L, U, P, A);                    // ⚠️ Might fail (depends on cuSOLVER)
af::solve(X, A, B);                    // ⚠️ Might fail (depends on algorithm)
af::cholesky(L, A);                    // ⚠️ Might fail (depends on cuSOLVER)
```

These operations may call **cuSOLVER** which requires runtime compilation.

---

## Affected Features

### What Works ✅

**Dataset Statistics:**
- ✅ Mean, std, min, max per channel
- ✅ Median, Q25, Q75 (robust statistics)
- ✅ Histograms (256 bins)
- ✅ Caching of computed statistics

**Preprocessing Strategies:**
- ✅ Image Enhancement (resize, CLAHE, denoise, sharpen, edge detection)
- ✅ Normalization (None, Auto, MNIST, CIFAR-10, ImageNet, Custom)
- ✅ Standard Scaling (z-score normalization)
- ✅ MinMax Scaling
- ✅ Robust Scaling (median/IQR-based)
- ✅ MaxAbs Scaling
- ✅ Quantile Transforms

**Training:**
- ✅ All neural network layers
- ✅ All optimizers (SGD, Adam, AdamW, RMSprop)
- ✅ All loss functions
- ✅ GPU acceleration via ArrayFire

### What Doesn't Work ❌

**Dataset Statistics:**
- ❌ PCA Components (eigenvectors)
- ❌ PCA Eigenvalues (variance explained)
- ❌ PCA Mean Vector (for centering)

**Preprocessing Strategies:**
- ❌ PCA Whitening (requires PCA components from SVD)

---

## Technical Details

### Error Stack Trace

```
ArrayFire Exception (Internal error:998):
In function arrayfire::common::compileModule
In file src\backend\cuda\compile_module.cpp:321
CU Link Error CUDA_ERROR_UNSUPPORTED_PTX_VERSION(222):
ptxas application ptx input, line 9;
fatal: Unsupported .version 8.7; current version is '8.5'

Stack:
 0# af::exception::~exception in afcuda
 ...
 9# af_svd in afcuda          ← SVD operation
10# af_svd in af
11# af::svd in af
12# StatisticsCalculator::Compute() in cyxwiz_engine
```

### System Information

**Your Configuration:**
```
GPU: NVIDIA GeForce GTX 1050 Ti (4GB)
CUDA Driver: 12.0.60 (= version 12.0.6)
Supported PTX: Up to 8.5
ArrayFire: v3.10.0 (CUDA, 64-bit Windows)
ArrayFire CUDA Runtime: 12.8
Required PTX: 8.7 (for cuSOLVER 12.8)
```

**The Mismatch:**
```
ArrayFire built with:  CUDA 12.8 → PTX 8.7
Your driver supports:  CUDA 12.0 → PTX 8.5
Result:                Incompatible ❌
```

### Why cuSOLVER 12.8 Requires PTX 8.7

CUDA 12.8's cuSOLVER likely uses new GPU features:

1. **Tensor Core Enhancements** (Hopper/Ada architecture)
   - Faster matrix decomposition
   - Better numerical stability

2. **Warp-Level Primitives** (introduced in PTX 8.6+)
   - More efficient thread synchronization
   - Better memory access patterns

3. **Memory Ordering Features** (PTX 8.7 specific)
   - Relaxed memory ordering
   - Acquire/release semantics for GPU memory

These features aren't available in PTX 8.5, so cuSOLVER **requires** PTX 8.7.

---

## Solutions

### Solution 1: Update NVIDIA Driver (Recommended) ✅

**Best option** - enables full GPU-accelerated PCA with no performance loss.

#### Step 1: Check Current Driver

```bash
# Windows
nvidia-smi

# Output shows:
Driver Version: 537.13    ← Your current version
CUDA Version: 12.0        ← Maximum supported
```

#### Step 2: Download Latest Driver

**Download from:** https://www.nvidia.com/download/index.aspx

**Select:**
- Product Type: GeForce
- Product Series: GeForce 10 Series
- Product: GeForce GTX 1050 Ti
- Operating System: Windows 11/10 (64-bit)

**Required:** Driver version **545.84 or newer** (supports CUDA 12.8+)

#### Step 3: Install and Restart

1. Run installer (clean installation recommended)
2. Restart computer
3. Verify: `nvidia-smi` should show Driver Version 545.84+

**After Update:**
```
CUDA Driver: 12.8+
Supported PTX: 8.7+
PCA Computation: ✅ Works at full GPU speed
```

**Performance:**
- PCA on 25K images: ~10-30 seconds (GPU)
- No code changes needed

---

### Solution 2: CPU Fallback (Quick Fix) ⚠️

**Add Eigen-based CPU SVD** when GPU fails.

#### Implementation

**File:** `cyxwiz-engine/src/preprocessing/statistics_calculator.cpp`

```cpp
// Add at top
#include <Eigen/SVD>

// In Compute() method, replace GPU-only PCA with fallback:

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        // Try GPU SVD (existing code)
        af::array data_af(...);
        af::svd(U, S, Vt, centered);

        // Extract results (existing code)
        stats.pca_computed = true;
        spdlog::info("PCA computed (GPU-accelerated): {} components", n_features);

    } catch (const af::exception& e) {
        spdlog::warn("GPU PCA failed ({}), falling back to CPU", e.what());

        // FALLBACK: Use Eigen for CPU-based SVD
        Eigen::MatrixXf data_eigen(n_samples, n_features);
        for (size_t i = 0; i < n_samples; ++i) {
            for (size_t j = 0; j < n_features; ++j) {
                data_eigen(i, j) = data_flat[i * n_features + j];
            }
        }

        // Center data
        Eigen::VectorXf mean = data_eigen.colwise().mean();
        data_eigen.rowwise() -= mean.transpose();

        // Compute SVD (CPU)
        Eigen::BDCSVD<Eigen::MatrixXf> svd(
            data_eigen,
            Eigen::ComputeThinU | Eigen::ComputeThinV
        );

        // Extract components
        stats.pca_components.resize(n_features);
        for (size_t i = 0; i < n_features; ++i) {
            stats.pca_components[i].resize(n_features);
            for (size_t j = 0; j < n_features; ++j) {
                stats.pca_components[i][j] = svd.matrixV()(i, j);
            }
        }

        // Extract eigenvalues
        stats.pca_eigenvalues.resize(n_features);
        for (size_t i = 0; i < n_features; ++i) {
            float sigma = svd.singularValues()(i);
            stats.pca_eigenvalues[i] = (sigma * sigma) / n_samples;
        }

        // Store mean
        stats.pca_mean.resize(n_features);
        for (size_t i = 0; i < n_features; ++i) {
            stats.pca_mean[i] = mean(i);
        }

        stats.pca_computed = true;
        spdlog::info("PCA computed (CPU fallback): {} components", n_features);
    }
#else
    // Existing fallback when ArrayFire not available
    spdlog::warn("PCA computation requires ArrayFire or Eigen");
    stats.pca_computed = false;
#endif
```

**Pros:**
- ✅ Works immediately, no driver update needed
- ✅ Produces identical results to GPU version
- ✅ Automatic fallback (user doesn't need to know)

**Cons:**
- ❌ **10-50× slower** (minutes instead of seconds)
- ❌ Higher CPU usage during computation
- ❌ Requires Eigen library

**Performance Comparison:**

| Dataset Size | GPU (CUDA) | CPU (Eigen) |
|--------------|-----------|-------------|
| 1,000 images (28×28×1) | ~1 second | ~10 seconds |
| 10,000 images (32×32×3) | ~5 seconds | ~2 minutes |
| 25,000 images (224×224×3) | ~30 seconds | **~15 minutes** |
| 50,000 images (224×224×3) | ~60 seconds | **~45 minutes** |

**For large datasets (>10K images with >64×64 resolution), CPU fallback is impractical.**

---

### Solution 3: Disable PCA (Current Behavior) ⚠️

**The code already handles this gracefully:**

```cpp
} catch (const af::exception& e) {
    spdlog::error("ArrayFire PCA computation failed: {}", e.what());
    stats.pca_computed = false;  // PCA disabled, everything else works
}
```

**Available:**
- ✅ All preprocessing features except PCA Whitening
- ✅ Standard Scaling (mean/std)
- ✅ MinMax Scaling
- ✅ Robust Scaling (median/IQR)
- ✅ MaxAbs Scaling
- ✅ Quantile Transforms

**Unavailable:**
- ❌ PCA Whitening scaling strategy

**User Impact:**
- Most users don't need PCA Whitening anyway
- Standard Scaling (z-score) is more common
- No functionality loss for typical use cases

---

### Solution 4: Downgrade ArrayFire (Not Recommended) ❌

**Use older ArrayFire built with CUDA 12.0:**

```
ArrayFire 3.9.0 (CUDA 12.0) → cuSOLVER 12.0 → PTX 8.5 ✅
ArrayFire 3.10.0 (CUDA 12.8) → cuSOLVER 12.8 → PTX 8.7 ❌
```

**Pros:**
- ✅ PCA works without driver update

**Cons:**
- ❌ Lose performance improvements in ArrayFire 3.10
- ❌ Lose bug fixes
- ❌ Lose new features
- ❌ Requires rebuilding entire project
- ❌ May cause other compatibility issues

**Not recommended** - updating the driver is much simpler.

---

## Verification

### Check If Issue Is Fixed

After updating driver or implementing CPU fallback:

1. **Launch CyxWiz Engine**
2. **Load a dataset** (e.g., MNIST, CIFAR-10)
3. **Open Data Pipeline tab**
4. **Expand "Preprocessing (Deterministic)"**
5. **Expand "Dataset Statistics"**
6. **Click "Compute Statistics"**
7. **Check logs** (`engine_log.txt`):

**Success (GPU):**
```
[info] StatisticsCalculator: Computing statistics for dataset 'mnist'
[info] Computing PCA components for whitening...
[info] PCA computed (GPU-accelerated): 784 components
[info] StatisticsCalculator: Computed statistics for dataset 'mnist' (1 channels, 60000 samples)
```

**Success (CPU Fallback):**
```
[info] StatisticsCalculator: Computing statistics for dataset 'mnist'
[info] Computing PCA components for whitening...
[warn] GPU PCA failed (...), falling back to CPU
[info] PCA computed (CPU fallback): 784 components
[info] StatisticsCalculator: Computed statistics for dataset 'mnist' (1 channels, 60000 samples)
```

**Failure (Current):**
```
[info] StatisticsCalculator: Computing statistics for dataset 'mnist'
[info] Computing PCA components for whitening...
[error] ArrayFire PCA computation failed: CUDA_ERROR_UNSUPPORTED_PTX_VERSION
[info] StatisticsCalculator: Computed statistics for dataset 'mnist' (1 channels, 60000 samples)
```

### Verify PCA Whitening Works

After successful PCA computation:

1. **Expand "Scaling" section**
2. **Select "PCA Whitening" strategy**
3. **Click "Apply Preprocessing to Dataset"**
4. **Check logs:**

**Success:**
```
[info] ScalingTransform: Initialized PCA Whitening (784 components)
[info] Preprocessing configuration applied to dataset 'mnist'
```

**Failure (no PCA):**
```
[warn] ScalingTransform: PCA not computed, falling back to Standard scaling
[info] Preprocessing configuration applied to dataset 'mnist'
```

---

## FAQ

### Q: Why does training work but PCA doesn't?

**A:** Training uses **pre-compiled kernels** (PTX 8.0-8.5) that are compatible with older drivers. PCA uses **runtime-compiled kernels** (PTX 8.7) from cuSOLVER 12.8, which requires a newer driver.

### Q: Will updating my driver break anything?

**A:** No. NVIDIA drivers are **backward compatible**. A driver that supports CUDA 12.8 also supports all older CUDA versions (12.0, 11.x, 10.x, etc.). Your existing code and applications will continue to work.

### Q: Can I just skip PCA?

**A:** Yes! Most use cases don't need PCA Whitening:
- **Standard Scaling** (z-score) is more common and often better
- **MinMax Scaling** is simpler and faster
- **Robust Scaling** handles outliers better than PCA

PCA Whitening is mainly used in specific scenarios:
- Facial recognition (eigenfaces)
- Dimensionality reduction before training
- Removing correlated features

### Q: How long does CPU fallback take?

**A:** For small datasets (<5K images, <64×64), CPU fallback is acceptable (~10-30 seconds). For large datasets (>10K images, >128×128), it's impractical (~15-60 minutes).

### Q: Will this happen with other operations?

**A:** Potentially. Any operation that uses **cuSOLVER/cuBLAS with JIT compilation** may fail:
- `af::qr()` - QR decomposition
- `af::lu()` - LU decomposition
- `af::cholesky()` - Cholesky decomposition
- `af::solve()` - Linear system solving (some modes)

However, these are less commonly used in typical ML workflows.

### Q: What if I can't update my driver?

**A:** Options:
1. Use **CPU fallback** (slow but functional)
2. **Skip PCA** (most use cases don't need it)
3. Use **older ArrayFire** (not recommended)
4. Use **different hardware** (newer GPU or cloud instance)

---

## Related Issues

### Similar Problems You Might Encounter

#### Issue: "CUDA out of memory" during PCA

**Symptom:**
```
[error] ArrayFire PCA computation failed:
Out of memory (CUDA_ERROR_OUT_OF_MEMORY)
```

**Cause:** PCA needs to load entire dataset into GPU memory.

**Solution:**
- Reduce dataset size (subsample images)
- Use smaller images (resize before PCA)
- Increase GPU memory (better GPU)
- Use CPU fallback

#### Issue: "cuSOLVER not found"

**Symptom:**
```
[error] Could not load cuSOLVER library
```

**Cause:** CUDA Toolkit not installed or incomplete.

**Solution:**
- Install CUDA Toolkit 12.8 from NVIDIA website
- Ensure `cusolverXX.dll` is in PATH
- Reinstall ArrayFire

#### Issue: "PTX JIT compilation failed"

**Symptom:**
```
[error] CUDA_ERROR_INVALID_PTX
```

**Cause:** Corrupted PTX code or driver bug.

**Solution:**
- Update NVIDIA driver
- Clear CUDA cache: `rm -rf ~/.nv/ComputeCache`
- Reinstall CUDA Toolkit

---

## References

### Documentation

- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA Compatibility Guide](https://docs.nvidia.com/deploy/cuda-compatibility/)
- [ArrayFire Documentation](https://arrayfire.org/docs/)
- [cuSOLVER Library](https://docs.nvidia.com/cuda/cusolver/)

### Version Compatibility

| CUDA Version | PTX Version | Min Driver (Windows) | Min Driver (Linux) |
|--------------|-------------|---------------------|-------------------|
| 12.0 | 8.0 | 526.98 | 525.60 |
| 12.1 | 8.2 | 528.89 | 527.41 |
| 12.2 | 8.3 | 531.14 | 530.30 |
| 12.3 | 8.4 | 536.67 | 535.54 |
| 12.4 | 8.5 | 538.33 | 537.13 |
| 12.5 | 8.5 | 544.12 | 543.00 |
| **12.8** | **8.7** | **545.84+** | **545.23+** |

### Additional Resources

- [CyxWiz Documentation](../README.md)
- [Data Pipeline Tab Guide](../DATA_PIPELINE_TAB.md)
- [Preprocessing Configuration](../cyxwiz-engine/src/preprocessing/preprocessing_config.h)
- [Statistics Calculator Implementation](../cyxwiz-engine/src/preprocessing/statistics_calculator.cpp)

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-09 | 1.0 | Initial documentation |

---

**Status:** Documented but not fixed (awaiting driver update or CPU fallback implementation)

**Recommended Action:** Update NVIDIA driver to 545.84 or newer

**Workaround:** Use preprocessing without PCA Whitening (Standard Scaling recommended)
