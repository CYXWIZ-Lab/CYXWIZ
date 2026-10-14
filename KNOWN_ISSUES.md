# Known issues

Problems we know about in CyxWiz Studio, mostly in the GPU runtimes it builds on: what you see,
which devices are affected, and what to do. Each entry says whether CyxWiz already works around it.
This file ships with every release package, next to `LICENSE`.

Last updated: 2026-10-14.

## GPU runtimes

### ArrayFire oneAPI: indexed min / max along rows return wrong results

- **Affects:** the oneAPI compute route (Intel GPUs), ArrayFire 3.10.0. Seen on Intel Iris Xe.
  OpenCL on the same Intel GPU, NVIDIA CUDA, and the CPU route are not affected.
- **What happens:** ArrayFire's `af::max(value, index, in, 1)` and `af::min(value, index, in, 1)`
  look at only part of each row when the array has few rows. They return a wrong maximum or minimum
  and the wrong position.
- **Effect in CyxWiz:** none. CyxWiz does these reductions along the other dimension, which is
  correct on every route.
  - Covered: KNN, logistic regression, naive Bayes and SVM predictions; GMM and k-means cluster
    labels; PCA component signs.
  - A source check in the test suite stops new code from calling the affected form.
- **If you write your own ArrayFire code** (Python scripts, plugins): avoid
  `af::max` / `af::min` with indices along dim 1 on oneAPI. Transpose so you reduce along dim 0.
- **Status:** reported upstream as
  [arrayfire/arrayfire#3732](https://github.com/arrayfire/arrayfire/issues/3732).
  The reproducer is in the source repository: `scripts/arrayfire_probes/oneapi_indexed_ops`.

### oneAPI does not run on Intel UHD 630

- **Affects:** the oneAPI route on older Intel integrated graphics (UHD 630, 8th/9th generation Core).
- **What happens:** the oneAPI runtime fails while it activates the device, before any work runs.
  The machine's other routes are unaffected.
- **What to do:** use OpenCL, which the Engine recommends on Intel GPUs and which is verified on the
  UHD 630.

## Precision

### GPUs without float64 compute analysis in float32

- **Affects:** GPUs without double-precision support, for example Intel Iris Xe (OpenCL and oneAPI).
- **What happens:** classic ML, clustering, data transforms, dimensionality reduction and linear
  algebra run in float32 there instead of float64. Results agree with the scikit-learn / PyTorch
  references to float32 precision (5e-5 relative), not to 1e-8. Iterative and chaotic methods agree
  less closely (for example t-SNE).
- **What to do:** for float64 results, choose a device that has it (NVIDIA CUDA or OpenCL, or the
  CPU route). The usage docs in the source repository (`docs/usage/`) give the tolerances for each
  feature.

## Reporting a new issue

Open an issue with:
- the device and compute route (Preferences > Devices);
- the CyxWiz version;
- what you expected and what you saw.

If you build from source, run the ArrayFire reproducers in `scripts/arrayfire_probes/` and include
their output.
