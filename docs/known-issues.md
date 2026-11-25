---
title: Known issues
description: Problems we know about in the Engine and the GPU runtimes it builds on, the devices affected, and what to do.
type: reference
area: general
audience: user
status: stable
since: 1.0.7
owner: general
last_reviewed: 2026-11-25
related:
  - troubleshooting/index.md
  - troubleshooting/gpu-and-drivers.md
  - concepts/numerical-accuracy.md
  - install/gpu-packs.md
redirect_from: []
---

Problems we know about in CyxWiz, most of them in the GPU runtimes it
builds on: what you see, which devices are affected, and what to do. Each
entry says whether CyxWiz already works around it. A short list of these
entries ships in every release package as `KNOWN_ISSUES.md`, next to
`LICENSE`; **Help > Known Issues** opens it.

An entry is removed when its fix ships. Problems that are not bugs but
limits of a feature are described on the feature's page.

## GPU runtimes

### ArrayFire oneAPI: indexed min and max along rows return wrong results

- **Affects:** the oneAPI compute route (Intel GPUs), ArrayFire 3.10.0. Seen
  on Intel Iris Xe. OpenCL on the same Intel GPU, NVIDIA CUDA and the CPU
  route are not affected.
- **What happens:** ArrayFire's `af::max(value, index, in, 1)` and
  `af::min(value, index, in, 1)` look at only part of each row when the
  array has few rows. They return a wrong maximum or minimum and the wrong
  position.
- **Effect in CyxWiz:** none. CyxWiz does these reductions along the other
  dimension, which is correct on every route. Covered: KNN, logistic
  regression, naive Bayes and SVM predictions; GMM and k-means cluster
  labels; PCA component signs. A source check in the test suite stops new
  code from calling the affected form.
- **If you write your own ArrayFire code** (Python scripts, plugins): avoid
  `af::max` and `af::min` with indices along dim 1 on oneAPI. Transpose so
  you reduce along dim 0.
- **Status:** reported upstream as
  [arrayfire/arrayfire#3732](https://github.com/arrayfire/arrayfire/issues/3732).
  The reproducer is in the source repository:
  `scripts/arrayfire_probes/oneapi_indexed_ops`.

### ArrayFire oneAPI: indexing a slice with an index array ignores the slice's offset

- **Affects:** the oneAPI compute route, ArrayFire 3.10.0. Seen on Intel
  Iris Xe. OpenCL on the same GPU, CUDA and the CPU route are not affected.
- **What happens:** for a 1-D slice that starts at a non-zero offset, such
  as `data(af::seq(10, 29))`, indexing it with an index array
  (`slice(idx)`) returns elements counted from the start of the whole array.
  `af::lookup(slice, idx)` is right.
- **Effect in CyxWiz:** none. Word2vec training (Word Embeddings) uses
  `af::lookup` there, so it trains the same on oneAPI as elsewhere. The
  parity suites (classic ML, clustering, data transforms, dimensionality
  reduction, linear algebra, ROC and PR, word vectors) pass on oneAPI.
- **If you write your own ArrayFire code** for oneAPI: index slices with
  `af::lookup`, or `copy()` the slice first.
- **Status:** reported upstream as
  [arrayfire/arrayfire#3733](https://github.com/arrayfire/arrayfire/issues/3733).
  The reproducer is in the source repository:
  `scripts/arrayfire_probes/oneapi_slice_index`.

### oneAPI does not run on Intel UHD 630

- **Affects:** the oneAPI route on older Intel integrated graphics (UHD 630,
  8th and 9th generation Core).
- **What happens:** the oneAPI runtime fails while it activates the device,
  before any work runs. The machine's other routes are unaffected.
- **What to do:** use OpenCL, which the Engine recommends on Intel GPUs and
  which is verified on the UHD 630.

## Precision

### GPUs without float64 compute analysis in float32

- **Affects:** GPUs without double-precision support, for example Intel Iris
  Xe (OpenCL and oneAPI).
- **What happens:** classic ML, clustering, data transforms, dimensionality
  reduction and linear algebra run in float32 there instead of float64.
  Results agree with the scikit-learn and PyTorch references to float32
  precision (5e-5 relative), not to 1e-8. Iterative and chaotic methods
  agree less closely (for example t-SNE).
- **What to do:** for float64 results, choose a device that has it (NVIDIA
  CUDA or OpenCL, or the CPU route). See
  [Numerical accuracy](concepts/numerical-accuracy.md).

## Text

### Right-to-left text is joined only in the dashboard and plots

- **Affects:** Arabic (and Hebrew) text in data.
- **What happens:** the Dashboard and plots (Plot window, plot cards) draw
  Arabic with its letters joined and in right-to-left order. Other windows
  (Table Viewer, Data Studio, Properties, the Script Editor) show the Arabic
  letters one by one, unjoined and left to right. Hebrew letters have no
  font yet and show as "?".
- **What to do:** read Arabic values in a dashboard or plot; the data itself
  is unchanged.

## Python (pycyxwiz)

### An optimizer's step() from Python changes no weights

- **Affects:** pycyxwiz 0.2.0, every device.
- **What happens:** `optimizer.step(parameters, gradients)`, called with the
  dicts of `get_parameters()` and `get_gradients()` (of a layer class such
  as `Dense`, or of a `Sequential`), returns without error but updates
  nothing: the dicts hold copies. `LRWarmup.step` behaves the same.
- **What to do:** train a `Sequential` of modules and call
  `model.update_parameters(optimizer)` after `backward`, as in
  [Optimizers and training loops](python/training-loops.md). Change the
  learning rate with `optimizer.set_learning_rate`.
- **Status:** open.

### The PyCyxWiz code export's training loop does not run

- **Affects:** the **Code** button of the Studio bar with **PyCyxWiz**
  chosen.
- **What happens:** the exported `train` method and the `__main__` block
  call names pycyxwiz does not have: calling a loss object, `loss.backward()`,
  `optimizer.step()` without arguments, `loss.item()` and `device.name()`.
  Running the exported file stops at the first of them (`'pycyxwiz.Device'
  object has no attribute 'name'`).
- **What to do:** keep the exported model definition and write the loop as
  in [Optimizers and training loops](python/training-loops.md).
- **Status:** open.

## Code export

### Some exports write a model other than the graph's, without a warning

- **Affects:** the **Code** button of the Studio bar.
- **What happens:** the TensorFlow and Keras exports write a branched graph
  as one straight chain and leave its merge nodes out; they also leave out
  layers they have no code for. The PyTorch export does not write a
  **Split**, so both branches after it read the whole tensor. The file is
  written with no message, and it describes another network.
- **What to do:** for a branched graph, export PyTorch (without Split) or
  PyCyxWiz, and compare the exported model's layers with the graph before
  you train it. See
  [What each framework writes](concepts/model-formats.md#what-each-framework-writes).
- **Status:** open.

## Report a new issue

Open an issue with:

- the device and compute route (**Preferences > Devices**);
- the CyxWiz version (**Help > About CyxWiz...**, **Copy details**);
- what you expected and what you saw.

If you build from source, run the ArrayFire reproducers in
`scripts/arrayfire_probes/` and include their output. The full list of what
to attach is in
[Crash dumps and logs](troubleshooting/crash-dumps-and-logs.md#what-to-attach-to-a-bug-report).
