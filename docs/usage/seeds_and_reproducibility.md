# Seeds and reproducibility

A seed makes a run repeat. With the same data, graph and seeds, the Engine draws
the same random numbers on Windows, Linux and macOS, and on the CPU, CUDA and
OpenCL devices.

## What a seed decides

| Seed | Where you set it | What it draws |
| --- | --- | --- |
| Data Split seed | Data Split node | Which rows go to Train, Validation and Test |
| DataLoader seed | DataLoader node (`seed`, `balance_seed`) | Batch order each epoch; under- and oversampling; the weighted sampler; image and audio augmentation |
| Model seed | DataLoader node (`model_seed`) | Weight initialisation, dropout masks and noise layers |
| Cross-validation seed | Results > Cross-validation | Fold membership (k-fold and stratified) and learning-curve subsets |
| Tool seeds | Explore tabs (Groups, Text), Tools > Utilities > Random, word embeddings, k-means, GMM, Random Forest, hyperparameter search, text generation | Each tool's own sample or start |

Each epoch reseeds the model draws from the model seed and the batch order from
the DataLoader seed, each with the epoch number, so a run resumed from a
checkpoint draws what the full run would have drawn. Image and audio datasets
are the exception for now: their batch order and augmentation after a resume
differ from the full run's, and the run says so (its data order is marked not
exact).

## How close is "the same"

- **Exactly the same:** batch order, splits and folds, sampled rows, crop
  positions, flips and choices between options. These are whole numbers and
  uniform values, and they come out bit for bit the same on every platform and
  device.
- **The same to the last digits:** normal, gamma, Beta, binomial and Poisson
  values (noise, weight initialisation, MixUp, AugMix, the weighted sampler).
  These go through `log`, `exp` and `cos`, which each operating system and
  device may round differently in the last digit (about 1e-7 for float32
  values on a GPU).
- **Training as a whole** repeats exactly on the same device and build. Across
  devices it starts from the same draws, but float sums on a GPU are added in
  a different order than on a CPU, so losses after many steps agree only
  closely, not exactly.

## Changed in this release

Before this release, seeds repeated only on one operating system. The C++
standard library's shuffle and distributions differ between Visual Studio, GCC
and Clang, and ArrayFire's normal draws (`af::randn`) differ between the CPU and
the GPU.
A seed now picks the same rows and numbers everywhere.

As a result, the same seed gives different splits, batch orders, augmentation
and initial weights than in earlier releases, once. Metrics of a rerun old
project can move a little for that reason.

## For plugin and Python authors

- C++: draw through `cyxwiz/portable_random.h` (`cyxwiz::portable::Shuffle`,
  `Below`, `UniformInt`, `Uniform`, `Normal`, `Gamma`, `Beta`, `Binomial`,
  `Poisson`, `Discrete`) on a `std::mt19937` or `std::mt19937_64`.
  `std::shuffle` and the `std::*_distribution` classes give different numbers
  with each compiler.
- On ArrayFire, use a Philox engine (`AF_RANDOM_ENGINE_PHILOX_4X32_10`) and make
  normal values from its uniform draws. `af::randn` and the Threefry and
  Mersenne engines differ between CPU and GPU. The probe
  `scripts/arrayfire_probes/random_engine_portability` prints a device's draws
  to compare.
- The Engine's own source is checked by `scripts/checks/check_random_sources.py`
  (ctest `random_sources_check`).
