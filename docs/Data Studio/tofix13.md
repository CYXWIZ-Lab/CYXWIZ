# To Fix 13 - CyxWiz Backend Engine Gap Analysis

Status: Reopened.

This item is not complete while major frontend or backend files remain
monolithic enough that engineers must read thousands of lines to locate one
concept. Some tensor parity work from this file was completed, but the broader
modularity goal remains active.

Current interpretation of this tofix:

- split large backend files into focused translation units and headers,
- split large frontend files into focused panels, dialogs, models, and services,
- keep one clear owner per concept, such as `optimizer/lsqr_optimizer`,
  `optimizer/msr_optimizer`, `loss/cross_entropy`, or `layer/dense`,
- avoid duplicate wrapper layers that hide the real implementation,
- preserve existing engine behavior while moving code behind smaller module
  boundaries,
- make it obvious where to look before adding, testing, or debugging a feature.

This note compares the backend engine currently vendored in this repo:

- `D:\Dev\CyxWiz_Claude\cyxwiz-backend`

against the more organized standalone backend:

- `D:\Dev\cyxwiz-backend`

The goal is not to blindly replace one with the other. The standalone
backend has the better production structure and broader ML framework
surface. The in-repo backend has some local Studio-facing work and a few
current-only algorithm areas that should be preserved.

## Short Version

The in-repo backend has a large gap in production readiness:

- tensor API is much smaller
- tests and benchmarks are mostly missing
- documentation is much thinner
- model lifecycle features are missing
- serialization, ONNX, quantization, autograd, metrics, callbacks, and
  model-zoo infrastructure are missing
- many layers are implemented in a monolithic `layer.h` / `layer.cpp`
  instead of organized per-layer files
- build configuration lacks the standalone backend's ONNX, JSON,
  OpenSSL, protobuf, CUDA memory, and test support

The standalone backend should become the main reference implementation,
but we must preserve current-only work such as:

- audio processing
- tokenizer
- RL interface
- Studio-specific Python binding/training helpers
- any graph/runtime behavior already wired into the current app

## Source Layout Comparison

### Current in-repo backend

Top-level structure:

- `include/cyxwiz`
- `src/core`
- `src/algorithms`
- `python`
- `examples/python`

Important characteristics:

- no `tests` directory inside `cyxwiz-backend`
- no `benchmarks` directory inside `cyxwiz-backend`
- no `cpp_docs` or `py_docs`
- no `third_party` ONNX or HTTP helper dependencies
- layer implementation is concentrated in `include/cyxwiz/layer.h` and
  `src/algorithms/layer.cpp`
- some features are present only in this version, such as audio,
  tokenizer, and RL interface files

### Standalone reference backend

Top-level structure:

- `include/cyxwiz`
- `src/core`
- `src/algorithms`
- `src/utils`
- `python/cyxwiz`
- `tests`
- `benchmarks`
- `cpp_docs`
- `py_docs`
- `examples`
- `third_party`

Important characteristics:

- organized by feature family
- separate headers and source files for layer groups, losses, metrics,
  data loading, model zoo, autograd, serialization, and utilities
- includes Python, C++, C, Rust, Go, and Lua examples
- includes a test suite and benchmark suite
- includes production notes, changelog, audit report, and roadmap docs

## Concrete File Gaps

The standalone backend has many source/header files that are not present
in the in-repo backend. Major missing groups:

### Tensor and core runtime

Missing or incomplete:

- `include/cyxwiz/exceptions.h`
- `include/cyxwiz/gpu_memory.h`
- `src/core/gpu_memory.cpp`
- safer tensor size calculation and overflow checks
- GPU array caching via dirty-state tracking
- stronger bounds checks in tensor indexing

The current in-repo `tensor.h` is much smaller than the reference:

- current `include/cyxwiz/tensor.h`: about 96 lines
- reference `include/cyxwiz/tensor.h`: about 447 lines

The current in-repo `tensor.cpp` is also much smaller:

- current `src/core/tensor.cpp`: about 767 lines
- reference `src/core/tensor.cpp`: about 4219 lines

This is one of the largest backend gaps.

### Tensor operations missing from current backend

The reference backend includes a much richer tensor API:

- shape operations: `View`, `Squeeze`, `Unsqueeze`, `Flatten`, `Permute`
- indexing and slicing: `At`, `Set`, `Slice`, `IndexSelect`
- concatenation and splitting: `Cat`, `Stack`, `Split`, `Chunk`
- reductions: `Sum`, `Mean`, `Max`, `Min`, `Prod`, `Std`, `Var`
- scalar and elementwise math: `Pow`, `Sqrt`, `Exp`, `Log`, `Abs`,
  `Sign`, `Clip`
- comparison operators
- logical operators
- broadcasting: `IsBroadcastable`, `BroadcastShape`, `BroadcastTo`,
  `Expand`
- batched matrix operations: `BatchMatMul`, `Dot`

These operations are foundational. Without them, higher-level layers,
masking, metrics, normalization, attention, and data transforms become
harder to implement correctly.

Status update 2026-06-02: the core backend tensor operation gap above is
closed in the in-repo C++ API through tensor parity slices 1-13. The
remaining work for these operations is exposure through Python/Studio
bindings and frontend surfaces, covered by the Python binding and Studio
integration sections below.

## Algorithm and ML Feature Gaps

## PyTorch Reference Gap

PyTorch is the best reference framework for how CyxWiz should feel as a
machine-learning backend. We do not need to copy PyTorch internals or add
PyTorch as a dependency, but CyxWiz should eventually cover the same
major algorithm families:

- tensor operations
- autograd
- neural network modules
- losses
- optimizers
- schedulers
- datasets and dataloaders
- metrics
- serialization and export
- distributed training
- quantization and inference utilities

Compared with PyTorch-style coverage, CyxWiz still lacks many algorithm
families or has only partial implementations.

### PyTorch-style tensor algorithms not yet complete

Missing or incomplete:

- advanced indexing
- boolean masking
- gather/scatter
- top-k and sort
- argmax and argmin
- cumulative operations such as `cumsum` and `cumprod`
- broadcasting across all elementwise operators
- batched matrix multiplication
- einsum-like tensor contraction
- full dtype conversion support
- device transfer helpers
- no-grad / inference-mode tensor behavior
- random distributions beyond simple random tensor creation
- numerical stability helpers such as `logsumexp`

These are foundational. Without them, higher-level algorithms become
special-case C++ code instead of reusable tensor expressions.

### PyTorch-style neural network algorithms not yet complete

Missing or incomplete:

- module container hierarchy similar to `nn.Module`
- parameter registration and recursive parameter discovery
- module train/eval mode propagation
- reusable `Sequential`, `ModuleList`, and `ModuleDict`-style containers
- activation modules beyond the current core set
- full convolution family:
  - `Conv1D`
  - `Conv2D`
  - `Conv3D`
  - transposed convolutions
  - depthwise/separable convolutions
- full pooling family:
  - max pooling
  - average pooling
  - adaptive pooling
  - global pooling
- normalization family:
  - batch norm
  - layer norm
  - group norm
  - instance norm
  - RMS norm
- recurrent family:
  - RNN
  - LSTM
  - GRU
  - packed/padded sequence support
- transformer family:
  - multi-head attention
  - encoder layer
  - decoder layer
  - full encoder/decoder stacks
  - causal masks and padding masks
- embedding family:
  - embedding
  - embedding bag
  - pretrained embedding loading
- regularization modules:
  - dropout
  - dropout2d/dropout3d
  - stochastic depth

CyxWiz has some of these pieces, but the implementation is not yet
organized or tested like a framework-level backend.

### PyTorch-style training algorithms not yet complete

Missing or incomplete:

- full automatic differentiation engine
- gradient accumulation utilities
- gradient clipping
- mixed precision training
- loss scaling
- parameter groups
- optimizer state serialization
- complete optimizer family:
  - SGD
  - Adam
  - AdamW
  - RMSprop
  - Adagrad
  - Adadelta
  - NAdam
  - LAMB
  - LBFGS
- complete scheduler family:
  - step
  - multi-step
  - exponential
  - cosine annealing
  - cosine warm restarts
  - reduce-on-plateau
  - one-cycle
  - warmup schedules
- callbacks or trainer hooks for:
  - early stopping
  - model checkpointing
  - progress reporting
  - CSV logging
  - learning-rate logging

### PyTorch-style data algorithms not yet complete

Missing or incomplete:

- dataset base class
- map-style dataset
- iterable dataset
- sampler
- random sampler
- weighted sampler
- batch sampler
- distributed sampler
- async / multi-worker dataloader
- prefetching
- collate functions
- transform composition
- image augmentation transforms
- text/token transform pipeline
- train/validation/test split helpers

The current backend has data-loader work, but it should be unified under
a clear `data/` namespace like the standalone backend.

### PyTorch-style model lifecycle algorithms not yet complete

Missing or incomplete:

- save/load model state
- save/load full model config
- checkpoint format
- optimizer checkpointing
- training resume
- ONNX export
- model registry
- model hub
- pretrained weight download/cache
- versioned model metadata
- inference mode
- batch inference helpers
- deployment-oriented quantization

### PyTorch-style evaluation algorithms not yet complete

Missing or incomplete:

- accuracy
- top-k accuracy
- precision
- recall
- F1 score
- confusion matrix
- ROC/AUC helpers
- regression metrics such as MAE, RMSE, R2
- per-class metrics
- validation loop helpers
- test-set evaluation helpers

These should be backend features, not only UI-side calculations.

### PyTorch-style distributed algorithms not yet complete

Missing or incomplete:

- reliable CPU collective backend tests
- NCCL backend build/runtime validation
- distributed sampler tests
- gradient bucket tests
- distributed checkpoint behavior
- metric aggregation across workers
- fault handling and timeout behavior

CyxWiz has distributed files already, but the feature needs test and
documentation parity before it can be considered production-ready.

### Layers

The current backend has many layer classes, but they are concentrated in
one large header/source pair:

- `include/cyxwiz/layer.h`
- `src/algorithms/layer.cpp`

Concrete example:

- current LSTM declaration: `cyxwiz-backend/include/cyxwiz/layer.h`
- current LSTM implementation: `cyxwiz-backend/src/algorithms/layer.cpp`
- current GRU declaration: `cyxwiz-backend/include/cyxwiz/layer.h`
- current GRU implementation: `cyxwiz-backend/src/algorithms/layer.cpp`

In this layout, recurrent layers, convolution layers, normalization
layers, pooling layers, attention, transformers, dropout, embedding,
upsampling, and pixel shuffle all compete inside the same layer module.
That makes ownership, testing, review, and future migration harder.

The reference backend organizes layers by file:

- `layers/conv1d`
- `layers/conv2d`
- `layers/conv_transpose2d`
- `layers/depthwise_separable_conv2d`
- `layers/dropout`
- `layers/batchnorm`
- `layers/layernorm`
- `layers/groupnorm`
- `layers/pooling`
- `layers/lstm`
- `layers/gru`
- `layers/embedding`
- `layers/attention`
- `layers/transformer`
- `layers/transformer_decoder`
- `layers/positional_encoding`

Concrete reference example:

- reference LSTM declaration:
  `D:\Dev\cyxwiz-backend\include\cyxwiz\layers\lstm.h`
- reference LSTM implementation:
  `D:\Dev\cyxwiz-backend\src\algorithms\layers\lstm.cpp`
- reference GRU declaration:
  `D:\Dev\cyxwiz-backend\include\cyxwiz\layers\gru.h`
- reference GRU implementation:
  `D:\Dev\cyxwiz-backend\src\algorithms\layers\gru.cpp`

This is the target structure we should move toward: one public header and
one implementation file per major layer family, plus a small base
`layer.h` that only defines shared layer behavior.

Missing or weaker in current:

- separate layer headers for clean public API
- `DepthwiseSeparableConv2D`
- `BatchNorm1D`
- `Dropout2D`
- `AdaptiveAvgPool2D`
- positional encoding as its own module
- clear per-layer tests
- clear per-layer benchmark coverage

Current-only layer work to preserve:

- monolithic implementations for several layers already exist
- `Upsample2DLayer`
- `PixelShuffleLayer`
- any Studio graph bindings expecting the current class names

### Losses

The current backend has a broad `loss.h` and `loss.cpp`, including:

- MSE
- L1
- SmoothL1 / Huber
- CrossEntropy
- BCE
- BCEWithLogits
- NLL
- KLDiv
- CosineEmbedding
- Focal
- Triplet
- Contrastive

The reference backend has a cleaner split under:

- `include/cyxwiz/losses`
- `src/algorithms/losses`

Migration should not reduce current loss coverage. The better target is
to keep the current broader loss set while moving toward the reference
layout and adding tests.

### Metrics

Missing from current:

- `include/cyxwiz/metric.h`
- `include/cyxwiz/metrics/accuracy.h`
- `include/cyxwiz/metrics/f1_score.h`
- `include/cyxwiz/metrics/precision.h`
- `include/cyxwiz/metrics/recall.h`
- `include/cyxwiz/metrics/confusion_matrix.h`
- matching `src/algorithms/metrics/*.cpp`

This is important for Studio training because the dashboard should not
depend on ad hoc Python-side metric calculations only.

### Training Infrastructure

Missing from current:

- callbacks system
- early stopping callback
- model checkpoint callback
- CSV logger
- learning-rate monitor
- progress callback
- gradient utilities
- initialization helpers
- regularization helpers
- mixed precision helpers
- LR finder
- profiler and progress utility modules

Reference files:

- `callbacks.h/.cpp`
- `gradient_utils.h/.cpp`
- `init.h/.cpp`
- `regularization.h/.cpp`
- `mixed_precision.h/.cpp`
- `lr_finder.h/.cpp`
- `progress.h/.cpp`
- `profiler.h/.cpp`

### Autograd

Missing from current:

- `autograd.h/.cpp`
- `autograd_engine.h/.cpp`
- `grad_fn.h/.cpp`
- `grad_functions.h/.cpp`
- `gradient_utils.h/.cpp`

This limits the backend's ability to become a general training engine.
Current training depends heavily on hand-written backward methods and
special-case layer logic.

### Data Loading and Transforms

Current backend has:

- `data_loader.h`
- `dataloader.h`
- `src/core/data_loader.cpp`
- `src/algorithms/dataloader.cpp`

Reference backend has a cleaner namespace:

- `data/dataset.h`
- `data/dataloader.h`
- `data/sampler.h`
- `data/async_dataloader.h`
- `data/transform.h`
- corresponding `src/algorithms/data/*.cpp`

Missing from current:

- abstract dataset interface
- sampler abstraction
- async/prefetch data loader
- composable transform pipeline
- image transform pipeline

Current-only work to preserve:

- any existing Studio data loader integration
- current `data_loader` behavior used by graph nodes

### Serialization, ONNX, and Model Lifecycle

Missing from current:

- `serialization.h/.cpp`
- `onnx_export.h/.cpp`
- `checkpointing.h/.cpp`
- `model_config.h/.cpp`
- `model_registry.h/.cpp`
- `model_hub.h/.cpp`
- pretrained architecture support:
  - ResNet
  - MobileNet
  - ViT

This is a major gap for deployment, reproducible training, pretrained
model loading, and export.

### Quantization and Inference

Missing from current:

- `quantization.h/.cpp`
- inference-mode helpers
- model quantization path
- model cache / model hub path
- GPU memory reporting helpers

This matters for production deployment, especially once Studio models
are exported and reused outside an interactive training session.

### Distributed Training

Both versions have distributed training files, but the standalone
backend has stronger integration in CMake and documentation.

Current has:

- `distributed/process_group`
- `distributed/cpu_backend`
- `distributed/nccl_backend`
- `distributed/ddp`
- `distributed/distributed_sampler`
- `distributed/distributed_trainer`

Need to verify:

- whether NCCL file is consistently built only when available
- whether CPU fallback behavior matches reference
- whether DDP works with the current monolithic layer/model design
- whether tests exist for sampler and metric aggregation

## Build System Gaps

Current CMake is simpler and has local dependencies:

- ArrayFire
- fmt
- spdlog
- optional pybind11 controlled by parent variable
- optional SndFile
- optional FFTW3
- optional DuckDB
- links `cyxwiz-protocol`

Reference CMake adds production features:

- explicit C++17 standard
- `BUILD_PYTHON_BINDINGS` option
- `CYXWIZ_ENABLE_ONNX` option
- `nlohmann_json`
- OpenSSL
- protobuf and ONNX proto generation
- `third_party/cpp-httplib`
- `src/utils`
- test executable
- CUDA backend detection
- GPU memory runtime support

Needed action:

- merge the reference build capabilities into the in-repo backend
- keep local optional dependencies used by current-only features
- avoid breaking the parent repo build and `cyxwiz-protocol` linkage

## Python Binding Gaps

The current `python/bindings.cpp` is large and includes Studio-facing
bindings for layers and training. It should not be overwritten blindly.

Reference binding adds or organizes:

- tensor operation bindings
- serialization bindings
- ONNX export bindings
- quantization bindings
- model hub/model registry bindings
- metrics bindings
- data loader abstractions
- TensorBoard/profiling helpers
- pretrained model architecture bindings

Needed action:

- create a binding parity checklist before code migration
- port missing reference bindings incrementally
- preserve current graph/training APIs already used by Studio
- add Python tests for every newly exposed binding

## Testing and Verification Gap

Current in-repo backend lacks a proper local test suite.

Reference has:

- `tests/test_phase1_ops.py`
- `tests/test_phase2_layers.py`
- `tests/test_phase4_5.py`
- `tests/test_tensor_operations.py`
- `tests/test_v020_features.py`
- root-level tests for activations, autograd, batchnorm, linear,
  GPU memory, inference mode, ONNX export, priority features,
  TensorBoard, and validation
- C++ tests such as `test_new_ops.cpp` and `test_simple.cpp`

Minimum required tests before migration is considered done:

- tensor shape and math operations
- layer forward/backward smoke tests
- recurrent layer forward/backward checks
- transformer attention mask checks
- loss and metric correctness checks
- serialization and checkpoint round trip
- Python binding smoke tests
- Studio graph training smoke test
- GPU and CPU fallback tests where possible

## Documentation Gap

Reference has:

- `cpp_docs`
- `py_docs`
- API docs
- examples docs
- performance docs
- troubleshooting docs
- `CHANGELOG.md`
- `AUDIT_REPORT.md`
- `PRODUCTION_ROADMAP.md`
- `TENSOR_API_ENHANCEMENTS.md`
- `DISTRIBUTED.md`
- `BENCHMARKS.md`

Current in-repo backend has only a smaller README and TODO.

Needed action:

- import or adapt reference docs into the main repo docs structure
- keep Studio-specific docs separate from backend API docs
- document current-only features that the reference backend lacks

## Current-Only Features To Preserve

These exist in the in-repo backend but are absent from the standalone
reference comparison:

- `audio_processing.h/.cpp`
- `tokenizer.h/.cpp`
- `rl_interface.h/.cpp`
- `stats_utils.h`
- `debug_hooks.h`
- `data_loader.h` and current `dataloader.h`
- optional SndFile, FFTW3, and DuckDB CMake support
- parent-project linkage to `cyxwiz-protocol`
- Studio-specific Python binding and training helpers

Do not delete these during migration unless a direct replacement exists.

## Recommended Migration Order

### Phase 1 - Baseline and protection

1. Freeze current behavior with smoke tests.
2. Add tests around the current Studio graph training path.
3. Add tests for current-only audio/tokenizer/RL/data-loader features.
4. Record current Python binding API names used by Studio.
5. Make sure CMake can build the current backend in isolation and inside
   the parent repo.

### Phase 2 - Core tensor parity

1. Port the reference tensor header/API carefully.
2. Port safe multiply and bounds checking.
3. Port GPU array cache dirty-state handling.
4. Port shape, slicing, broadcasting, reduction, comparison, and math
   operations.
5. Add Python bindings for tensor operations.
6. Add tensor operation tests before moving higher-level layers.

### Phase 3 - File structure cleanup

1. Split monolithic layer declarations into per-layer headers.
2. Split monolithic `layer.cpp` into per-layer source files.
3. Keep compatibility includes in `cyxwiz.h`.
4. Preserve class names expected by current bindings and Studio.
5. Add CMake source groups for each feature family.

### Phase 4 - Training and metrics

1. Port metric classes.
2. Port callback system.
3. Port gradient utilities, initialization, regularization, and mixed
   precision helpers.
4. Port async data loader and sampler abstractions.
5. Wire metrics and callbacks into the Studio training dashboard.

### Phase 5 - Model lifecycle

1. Port serialization.
2. Port checkpointing.
3. Port ONNX export and protobuf generation.
4. Port model config, registry, and hub.
5. Port ResNet, MobileNet, and ViT only after tensor/layer parity is
   stable.

### Phase 6 - Production hardening

1. Port quantization.
2. Port GPU memory tracking.
3. Port profiler and progress utilities.
4. Add benchmark suite.
5. Add docs for C++, Python, and Studio integration.
6. Add CI checks for build, tests, formatting, and at least CPU runtime.

## Risks

### Risk 1 - Blind overwrite loses Studio behavior

The current backend is integrated with this repo's Studio and graph
runtime. Replacing files wholesale may break Python bindings, graph node
execution, training dashboard behavior, or protocol integration.

Mitigation:

- migrate by feature group
- keep compatibility wrappers
- run Studio graph smoke tests after every group

### Risk 2 - Monolithic layer split causes API drift

Current layer names and constructors may already be used by Studio.

Mitigation:

- preserve public class names
- use forwarding headers during split
- avoid renaming Python classes until UI/runtime callers are updated

### Risk 3 - Build dependency expansion breaks parent repo

Reference backend depends on JSON, OpenSSL, protobuf, and optional ONNX.
The parent repo may not have these configured consistently.

Mitigation:

- add options for optional features
- keep ONNX disabled unless protobuf is found
- keep current optional SndFile/FFTW3/DuckDB behavior
- avoid requiring CUDA Toolkit when ArrayFire CPU/OpenCL is enough

### Risk 4 - Reference backend also has known TODOs

The reference backend is better, but not perfect. Its audit report notes
remaining work such as full SHA256 implementation, Conv2D im2col
optimization, async data-loader lock contention, CPU cache locality,
dual Layer/Module hierarchy, GPU CI, and numerical gradient tests.

Mitigation:

- treat reference as a roadmap and source of implementations, not as a
  finished product
- keep known reference TODOs visible in the migration tracker

## Definition of Done

The backend gap is closed when:

- in-repo backend builds inside the parent repo and standalone
- tensor API reaches reference parity
- layer files are organized by feature family
- current-only audio/tokenizer/RL functionality still works
- metrics, callbacks, serialization, checkpointing, ONNX, quantization,
  model hub, and model registry are available or explicitly feature
  gated
- Python bindings expose the migrated capabilities
- Studio training still works with existing graphs
- tests cover tensor ops, layers, losses, metrics, serialization,
  bindings, and Studio graph training
- docs explain both the backend API and the Studio integration path

## Immediate Next Step

Continue the layer modularity split. The implementation files already
started moving into `src/algorithms/layers`; the next low-risk work is to
move public layer declarations from the monolithic `include/cyxwiz/layer.h`
into focused `include/cyxwiz/layers/*.h` headers while keeping
`cyxwiz/layer.h` as a compatibility include for existing callers.

## Progress Log

### 2026-06-02 - Tensor parity slice 1

Started with the smallest safe tensor parity work from the standalone
reference backend:

- added overflow-checked tensor element and byte-size calculations
- implemented the already-declared `Tensor::RangeN`
- implemented the already-declared `Tensor::Reshape`
- implemented the already-declared 2D `Tensor::Transpose`
- added unit coverage for range creation, reshape, transpose, lazy
  host sync after ArrayFire operations, and overflow rejection

This does not attempt to port the full reference tensor API yet. It
closes public API holes that were already declared in the in-repo
header and creates a tested base for future shape/indexing/reduction
work.

### 2026-06-02 - Tensor parity slice 2 and first tensor split

Added the next shape-operation slice from the reference tensor API:

- `Tensor::View`
- `Tensor::Squeeze`
- `Tensor::Unsqueeze`
- `Tensor::Flatten`
- `Tensor::Flatten(start_dim, end_dim)`
- `Tensor::Transpose(dim0, dim1)`

This work also starts the translation-unit cleanup called out in this
gap analysis: the new shape operations live in
`src/core/tensor_shape.cpp`, and the backend CMake source list now
builds that file explicitly. This keeps `tensor.cpp` from absorbing
every parity addition and establishes the pattern for future splits
such as `tensor_indexing.cpp`, `tensor_reductions.cpp`, and
`tensor_broadcast.cpp`.

Added unit tests for each new operation, including arbitrary-dimension
transpose over a 3D tensor. The implementation keeps semantics simple
and conservative: these shape operations return materialized tensors
instead of introducing shared-storage views, because the current
`Tensor` ownership model does not yet have reference-counted storage or
stride metadata.

### 2026-06-02 - Tensor parity slice 3 and reductions split

Added scalar all-element tensor reductions:

- `Tensor::Sum`
- `Tensor::Mean`
- `Tensor::Max`
- `Tensor::Min`
- `Tensor::Prod`

This continues the modularization work by implementing reductions in
`src/core/tensor_reductions.cpp` and wiring that translation unit into
the backend CMake source list. The current slice intentionally covers
only whole-tensor scalar reductions; dimension-wise reductions remain
future tensor parity work.

Added float and integer reduction tests. Tensor-specific tests and the
broader non-service regression suite pass after this slice.

### 2026-06-02 - Tensor parity slice 4 and elementwise split

Added the next elementwise tensor API slice from the reference backend:

- scalar arithmetic operators for `float` scalars
- `Tensor::Pow(float)`
- `Tensor::Pow(const Tensor&)`
- `Tensor::Sqrt`
- `Tensor::Exp`
- `Tensor::Log`
- `Tensor::Abs`
- `Tensor::Sign`
- `Tensor::Clip`
- unary negation

This work lives in `src/core/tensor_elementwise.cpp`, continuing the
translation-unit split instead of adding more responsibilities to
`tensor.cpp`. The implementation is CPU-backed and typed. Scalar
arithmetic preserves the source tensor dtype; real-valued unary math
returns `Float64` for `Float64` inputs and `Float32` otherwise.

Added focused tests for scalar arithmetic, integer dtype behavior,
real-valued unary math, clipping, sign, absolute value, negation, and
tensor exponentiation with dtype promotion. Tensor-specific tests and
the broader non-service regression suite pass after this slice.

### 2026-06-02 - Tensor parity slice 5 and broadcasting split

Added the reference-style broadcasting API:

- `Tensor::IsBroadcastable`
- `Tensor::BroadcastShape`
- `Tensor::BroadcastTo`
- `Tensor::Expand`

The implementation lives in `src/core/tensor_broadcast.cpp`, keeping
broadcasting separate from the core tensor ownership and device-cache
code. Unlike the standalone reference implementation, this slice
materializes broadcasts for all existing in-repo dtypes by copying raw
elements with row-major stride mapping instead of limiting `Expand` to
`Float32`.

Added tests for NumPy-style broadcast compatibility, broadcast shape
calculation, invalid shapes, leading-dimension expansion, singleton
dimension repeats, and integer tensor broadcasting. Tensor-specific
tests and the broader non-service regression suite pass after this
slice.

### 2026-06-02 - Tensor parity slice 6 and indexing split

Added the reference-style indexing and slicing API:

- `Tensor::At`
- `Tensor::Set`
- `Tensor::Slice`
- `Tensor::IndexSelect`

The implementation lives in `src/core/tensor_indexing.cpp`, keeping
element access, slicing, and gather-style selection separate from
`tensor.cpp`. The public API follows the reference backend's `float`
element access contract for `At` and `Set`, while the implementation
handles all existing in-repo dtypes by casting at that boundary. Slice
and index-select copy raw row-major elements, so they preserve the
source dtype.

Added tests for bounds-checked element access, integer cast behavior,
stepped slicing, negative dimensions and indices, invalid slice
arguments, and index selection. Tensor-specific tests and the broader
non-service regression suite pass after this slice.

### 2026-06-02 - Tensor parity slice 7 and concat split

Added the reference-style concatenation and splitting API:

- `Tensor::Cat`
- `Tensor::Stack`
- `Tensor::Split(split_size, dim)`
- `Tensor::Split(sizes, dim)`
- `Tensor::Chunk`

The implementation lives in `src/core/tensor_concat.cpp`, keeping
concatenation and split behavior separate from the core tensor storage
and device-cache code. Concatenation validates rank, shape, and dtype,
then copies raw row-major elements so all existing in-repo dtypes are
preserved.

Added tests for concatenation across rows and columns, invalid concat
inputs, stack placement, fixed-size splits, explicit-size splits, and
chunking. Tensor-specific tests and the broader non-service regression
suite pass after this slice.

Audit note: after reviewing the new concat module, fixed `Chunk()` on an
empty split dimension so it returns an empty result consistently with
`Split()` instead of computing a zero split size. Added a regression
test for empty-dimension split and chunk behavior.

### 2026-06-02 - Tensor indexing audit amendment: typed accessors

Amended the indexing slice after audit by adding checked native-dtype
C++ element accessors:

- `Tensor::AtAs<T>`
- `Tensor::SetAs<T>`

These preserve the reference-compatible `At` / `Set` float boundary for
existing callers while giving C++ code a typed path for `Float32`,
`Float64`, `Int32`, `Int64`, and `UInt8` tensors. The typed accessors
reject dtype mismatches, rank mismatches, and out-of-range indices
instead of silently reinterpreting memory or truncating through `float`.

Added tests for native integer access, `Float64` access, dtype mismatch
errors, rank mismatch errors, and bounds checks. The targeted tensor
suite passes with 34/34 tests, and the broader non-service regression
suite passes with 70/70 selected tests.

### 2026-06-02 - Tensor modularity audit amendment: private utilities

Amended the modularity slice after audit by consolidating duplicated
tensor helper logic into a private core utility module:

- `src/core/tensor_utils.h`
- `src/core/tensor_utils.cpp`

The utility now owns shared tensor primitives for dtype byte size,
dimension normalization, checked dimension products, checked row-major
stride construction, safe multiplication, and raw row-major element
copying. The split tensor modules now call this private utility instead
of carrying separate local copies in shape, indexing, broadcasting,
concat, and core tensor code.

This keeps the public `Tensor` API unchanged while reducing drift risk
between translation units. The targeted tensor suite passes with 34/34
tests, and the broader non-service regression suite passes with 70/70
selected tests.

Audit note: reviewed the utility boundary after consolidation. Shared
helpers are now centralized in `tensor_utils`; split modules no longer
carry duplicate `ElementSize`, `NormalizeDim`, row-major stride, or raw
copy helpers. The remaining `SafeAdd` helper in concat is intentionally
local because it is concat-specific and not duplicated in the tensor
split modules.

### 2026-06-02 - Tensor parity slice 8: permute

Added the remaining shape operation called out in the initial tensor
gap list:

- `Tensor::Permute`

The implementation lives in `src/core/tensor_shape.cpp` with the other
shape operations. It validates rank, invalid dimensions, and duplicate
dimensions, supports negative dimension aliases, and materializes
row-major output using `tensor_utils` so all existing in-repo dtypes are
preserved. This intentionally improves on the standalone reference
implementation, which notes `Permute` as `Float32`-only.

Added tests for arbitrary 3D dimension reordering, dtype preservation
for `Float64`, negative dimensions, duplicate-dimension rejection,
rank-mismatch rejection, and out-of-range rejection. The targeted tensor
suite passes with 36/36 tests, and the broader non-service regression
suite passes with 72/72 selected tests.

### 2026-06-02 - Tensor parity slice 9: scalar variance and standard deviation

Extended the all-element reduction slice with the remaining scalar
statistical reductions from the reference tensor API:

- `Tensor::Var`
- `Tensor::Std`

These are population variance and population standard deviation over
all elements. The implementation lives in `src/core/tensor_reductions.cpp`
with the other scalar reductions. It returns `Float64` for `Float64`
tensors and `Float32` for all other existing in-repo dtypes, matching
the existing `Mean()` dtype boundary. Dimension-wise `Var(dim)` and
`Std(dim)` remain a later tensor parity slice.

Added tests for float tensors, integer tensors, `Float64` dtype
preservation, and empty-tensor rejection. The targeted tensor suite
passes with 38/38 tests, and the broader non-service regression suite
passes with 74/74 selected tests.

### 2026-06-02 - Tensor parity slice 10: comparison operators

Added reference-style tensor comparison operators:

- `Tensor::operator>`
- `Tensor::operator>=`
- `Tensor::operator<`
- `Tensor::operator<=`
- `Tensor::operator==`
- `Tensor::operator!=`

Each operator supports tensor-tensor comparisons and `float` scalar
comparisons. Tensor-tensor comparisons use existing broadcast shape
rules and materialize broadcasted operands through `Expand`. Results are
`UInt8` mask tensors with `1` for true and `0` for false. The
implementation lives in `src/core/tensor_comparison.cpp`, keeping
comparison logic separate from scalar arithmetic and unary elementwise
math.

Added tests for tensor masks, scalar comparisons, mixed-dtype
comparisons, broadcasting, invalid broadcast shapes, and result dtype.
The targeted tensor suite passes with 41/41 tests, and the broader
non-service regression suite passes with 77/77 selected tests.

### 2026-06-02 - Tensor parity slice 11: logical operators

Added reference-style logical tensor operators:

- `Tensor::operator&&`
- `Tensor::operator||`
- `Tensor::operator!`

Logical operators treat zero as false and nonzero as true. Tensor-tensor
`&&` and `||` use the existing broadcast shape rules and materialize
broadcasted operands through `Expand`. Results are `UInt8` mask tensors
with `1` for true and `0` for false. The implementation lives in
`src/core/tensor_logical.cpp`, keeping mask logic separate from
comparison and arithmetic operations.

Added tests for `UInt8` mask inputs, mixed-dtype truthiness,
broadcasting, unary logical not, invalid broadcast shapes, and result
dtype. The targeted tensor suite passes with 43/43 tests, and the
broader non-service regression suite passes with 79/79 selected tests.

### 2026-06-02 - Tensor parity slice 12: dimension-wise reductions

Extended the reduction module with dimension-wise reductions and
`keepdim` support:

- `Tensor::Sum(dim, keepdim)`
- `Tensor::Mean(dim, keepdim)`
- `Tensor::Max(dim, keepdim)`
- `Tensor::Min(dim, keepdim)`
- `Tensor::Prod(dim, keepdim)`
- `Tensor::Var(dim, keepdim)`
- `Tensor::Std(dim, keepdim)`

The implementation lives in `src/core/tensor_reductions.cpp` with the
existing scalar reductions. It normalizes positive and negative
dimensions, computes row-major outer/inner reduction spans, preserves
the input dtype for `Sum`, `Max`, `Min`, and `Prod`, and returns
`Float64` for `Float64` inputs or `Float32` otherwise for `Mean`, `Var`,
and `Std`. `Max` and `Min` seed from the first value on the reduced
axis so all-negative and all-positive slices reduce correctly. `Sum`
and `Prod` return identity values for empty reduced dimensions; `Max`,
`Min`, `Mean`, `Var`, and `Std` reject empty reduced dimensions.

Added tests for integer dtype preservation, `keepdim` output shapes,
negative-value max/min reductions, `Float64` statistical reductions,
invalid dimension validation, and empty-axis behavior. The targeted
tensor suite passes with 47/47 tests, and the broader non-service
regression suite passes with 83/83 selected tests.

### 2026-06-02 - Tensor parity slice 13: tensor linear algebra

Closed the remaining batched matrix operation gap from the initial
tensor operation list:

- `Tensor::Dot`
- `Tensor::BatchMatMul`

The implementation lives in the new `src/core/tensor_linalg.cpp`
translation unit so linear algebra does not expand the reduction or
elementwise modules. `Dot` supports matching-dtype 1D vectors and
returns a `{1}` tensor preserving the input dtype. `BatchMatMul`
supports matching-dtype 3D tensors with shape `{batch, rows, shared}`
and `{batch, shared, cols}`, returns `{batch, rows, cols}`, preserves
the input dtype, and handles empty batch dimensions without special
cases.

Added tests for row-major dot products, batched matrix multiplication,
dtype preservation, empty batches, and validation of rank, shape, and
dtype errors. The targeted tensor suite passes with 52/52 tests, and
the broader non-service regression suite passes with 88/88 selected
tests.

### 2026-06-21 - Layer header split slice 1: base and dense headers

Started the public layer declaration split without changing layer
behavior:

- added `include/cyxwiz/layers/layer_base.h` as the owner of the base
  `Layer` contract,
- added `include/cyxwiz/layers/dense.h` as the owner of `DenseLayer`,
- kept `include/cyxwiz/layer.h` as the compatibility aggregate include,
- updated `DenseLayer` implementation to include its focused header,
- decoupled `LinearLayer` from the monolithic layer aggregate by including
  the base layer header directly,
- registered the new headers in the backend CMake header list.

This keeps the public API stable while reducing the amount of unrelated
layer declaration code that must live in `layer.h`. The next layer-header
slice should move another coherent family, such as pooling or dropout, into
focused headers before larger recurrent or transformer declarations are
touched.

### 2026-06-21 - Layer header split slice 2: dropout and pooling headers

Continued the public layer declaration split without changing layer behavior:

- added `include/cyxwiz/layers/dropout.h` as the owner of `DropoutLayer`,
- added `include/cyxwiz/layers/pooling.h` as the owner of `MaxPool2DLayer`,
  `AvgPool2DLayer`, and `GlobalAvgPool2DLayer`,
- kept `include/cyxwiz/layer.h` as the compatibility aggregate include,
- updated the dropout and pooling implementations to include their focused
  headers,
- registered the new headers in the backend CMake header list.

This keeps the split small and coherent: stateless or simple regularization
and pooling layers moved out first, while larger convolution, recurrent,
normalization, and transformer declarations remain untouched for later slices.

### 2026-06-21 - Layer header split slice 3: flatten header

Moved the simple shape-only `FlattenLayer` declaration out of the layer
aggregate:

- added `include/cyxwiz/layers/flatten.h`,
- kept `include/cyxwiz/layer.h` as the compatibility aggregate include,
- updated the flatten implementation to include its focused header,
- registered the new header in the backend CMake header list.

This keeps behavior unchanged and continues reducing `layer.h` before touching
stateful families such as normalization, convolution, recurrent, or transformer
layers.

### 2026-06-21 - Layer header split slice 4: upsampling header

Moved the upsampling declaration family out of the layer aggregate:

- added `include/cyxwiz/layers/upsampling.h`,
- moved `UpsampleMode`, `Upsample2DLayer`, and `PixelShuffleLayer` into the
  focused header,
- kept `include/cyxwiz/layer.h` as the compatibility aggregate include,
- updated the upsampling implementation to include its focused header,
- registered the new header in the backend CMake header list.

This preserves the existing public names while giving image upsampling and
sub-pixel shuffle behavior a clear owner separate from the remaining monolithic
layer declarations.

### 2026-06-21 - Layer header split completion batch

Completed the remaining public layer declaration split:

- added `include/cyxwiz/layers/convolution.h` for `Conv1DLayer`,
  `Conv2DLayer`, and `ConvTranspose2DLayer`,
- added `include/cyxwiz/layers/normalization.h` for `BatchNorm2DLayer`,
  `LayerNormLayer`, `InstanceNorm2DLayer`, and `GroupNormLayer`,
- added `include/cyxwiz/layers/embedding.h` for `EmbeddingLayer`,
- added `include/cyxwiz/layers/recurrent.h` for `LSTMLayer` and `GRULayer`,
- added `include/cyxwiz/layers/attention.h` for `MultiHeadAttentionLayer`,
- added `include/cyxwiz/layers/transformer.h` for transformer encoder and
  decoder layers,
- reduced `include/cyxwiz/layer.h` to a compatibility aggregate include,
- updated layer implementation files to include their focused owning headers,
- registered the new headers in the backend CMake header list.

At this point the monolithic public layer declaration surface has been split by
family while preserving existing class names and aggregate include compatibility.

### 2026-06-21 - Optimizer header split slice 1: public optimizer declarations

Started the optimizer modularity cleanup by splitting the monolithic
`include/cyxwiz/optimizer.h` declarations into focused public headers:

- added `include/cyxwiz/optimizers/optimizer_base.h` for `Optimizer`,
  `OptimizerType`, and `CreateOptimizer`,
- added `include/cyxwiz/optimizers/sgd.h` for `SGDOptimizer`,
- added `include/cyxwiz/optimizers/adam.h` for `AdamOptimizer`,
  `AdamWOptimizer`, and `NAdamOptimizer`,
- added `include/cyxwiz/optimizers/adaptive.h` for `RMSpropOptimizer`,
  `AdaGradOptimizer`, and `AdadeltaOptimizer`,
- added `include/cyxwiz/optimizers/lamb.h` for `LAMBOptimizer`,
- added `include/cyxwiz/optimizers/lr_warmup.h` for `LRWarmup` and
  `WarmupType`,
- kept `include/cyxwiz/optimizer.h` as the compatibility aggregate include,
- updated the SGD and Adam-family implementations to include their focused
  owning headers,
- registered the new headers in the backend CMake header list.

This keeps optimizer behavior unchanged while giving optimizer families clear
public ownership, matching the completed layer declaration split pattern.

### 2026-06-21 - Loss header split slice 1: public loss declarations

Continued the backend declaration split by moving the monolithic
`include/cyxwiz/loss.h` public declarations into focused loss-family headers:

- added `include/cyxwiz/losses/loss_base.h` for `Loss`, `LossType`,
  `Reduction`, and `CreateLoss`,
- added `include/cyxwiz/losses/regression.h` for `MSELoss`, `L1Loss`,
  `SmoothL1Loss`, and the `HuberLoss` alias,
- added `include/cyxwiz/losses/classification.h` for `CrossEntropyLoss`,
  `NLLLoss`, and `FocalLoss`,
- added `include/cyxwiz/losses/probability.h` for `BCELoss`,
  `BCEWithLogitsLoss`, and `KLDivLoss`,
- added `include/cyxwiz/losses/metric_learning.h` for `CosineEmbeddingLoss`,
  `TripletLoss`, and `ContrastiveLoss`,
- kept `include/cyxwiz/loss.h` as the compatibility aggregate include,
- updated split loss implementation files to include their focused owning
  headers,
- registered the new headers in the backend CMake header list.

This keeps loss behavior unchanged while making regression, classification,
probability, and metric-learning loss ownership explicit.

### 2026-06-21 - Sequential implementation split slice 1: activation modules

Started the backend sequential implementation split by moving the activation
module wrappers out of the monolithic `src/algorithms/sequential.cpp` file:

- added `src/algorithms/sequential/activation_modules.cpp` for `ReLUModule`,
  `SigmoidModule`, `TanhModule`, `LeakyReLUModule`, `ELUModule`, `GELUModule`,
  `SwishModule`, and `MishModule`,
- kept the public `include/cyxwiz/sequential.h` API unchanged,
- kept `CreateModule` in `sequential.cpp` so factory behavior remains in the
  existing model container implementation,
- registered the new translation unit in `cyxwiz-backend/CMakeLists.txt`.

This targets active backend code and uses the existing `src/algorithms/sequential`
folder instead of adding another abstraction layer.

### 2026-06-21 - Sequential implementation split slice 2: regularization and shape modules

Continued the backend sequential implementation split by moving another
self-contained module family out of `src/algorithms/sequential.cpp`:

- added `src/algorithms/sequential/regularization_shape_modules.cpp` for
  `SoftmaxModule`, `DropoutModule`, and `FlattenModule`,
- kept the public `include/cyxwiz/sequential.h` API unchanged,
- kept `CreateModule` in `sequential.cpp` so factory behavior remains in the
  existing model container implementation,
- registered the new translation unit in `cyxwiz-backend/CMakeLists.txt`.

This continues shrinking the active monolithic sequential implementation while
preserving behavior and keeping each slice easy to validate.
### 2026-06-21 - Sequential implementation split slice 3: tensor modules

Continued the backend sequential implementation split by moving the tensor and
shape-oriented module wrappers out of `src/algorithms/sequential.cpp`:

- added `src/algorithms/sequential/tensor_modules.cpp` for `ReshapeModule`,
  `PermuteModule`, `TensorUnaryModule`, `TensorReductionModule`,
  `TensorShapeModule`, and `TensorMaskModule`,
- kept the public `include/cyxwiz/sequential.h` API unchanged,
- kept `SequentialModel` and `CreateModule` in `sequential.cpp`,
- registered the new translation unit in `cyxwiz-backend/CMakeLists.txt`.

This keeps graph-facing tensor module ownership separate from the sequential
model container implementation without changing runtime behavior.
### 2026-06-21 - Sequential implementation split slice 4: feedforward modules

Continued the backend sequential implementation split by moving the simple
feedforward and embedding module wrappers out of `src/algorithms/sequential.cpp`:

- added `src/algorithms/sequential/feedforward_modules.cpp` for `LinearModule`,
  `TimeDistributedDenseModule`, `EmbeddingModule`, and
  `PositionalEncodingModule`,
- kept the public `include/cyxwiz/sequential.h` API unchanged,
- kept recurrent, transformer, normalization, `SequentialModel`, and
  `CreateModule` in `sequential.cpp` for later smaller slices,
- registered the new translation unit in `cyxwiz-backend/CMakeLists.txt`.

This removes another self-contained block from the active monolithic sequential
implementation without changing runtime behavior.
### 2026-06-21 - Sequential implementation split slice 5: recurrent modules

Continued the backend sequential implementation split by moving recurrent module
wrappers out of `src/algorithms/sequential.cpp`:

- added `src/algorithms/sequential/recurrent_modules.cpp` for `LSTMModule` and
  `GRUModule`,
- kept the public `include/cyxwiz/sequential.h` API unchanged,
- kept transformer, normalization, `SequentialModel`, and `CreateModule` in
  `sequential.cpp` for later focused slices,
- registered the new translation unit in `cyxwiz-backend/CMakeLists.txt`.

This gives recurrent sequence modules a clear backend implementation owner while
preserving the existing model container and factory behavior.
### 2026-06-21 - Sequential implementation split slice 6: transformer modules

Continued the backend sequential implementation split by moving transformer
module wrappers out of `src/algorithms/sequential.cpp`:

- added `src/algorithms/sequential/transformer_modules.cpp` for
  `TransformerEncoderModule` and `TransformerDecoderModule`,
- kept the public `include/cyxwiz/sequential.h` API unchanged,
- kept normalization, `SequentialModel`, and `CreateModule` in
  `sequential.cpp` for the remaining focused slices,
- registered the new translation unit in `cyxwiz-backend/CMakeLists.txt`.

This gives transformer sequence modules a clear backend implementation owner
while preserving existing factory and container behavior.
### 2026-06-21 - Sequential implementation split slice 7: normalization modules

Completed the sequential wrapper extraction by moving normalization module logic
out of `src/algorithms/sequential.cpp`:

- added `src/algorithms/sequential/normalization_modules.cpp` for
  `BatchNormModule`,
- kept the public `include/cyxwiz/sequential.h` API unchanged,
- left `SequentialModel`, debug trace helpers, and `CreateModule` in
  `sequential.cpp` as the core model-container owner,
- registered the new translation unit in `cyxwiz-backend/CMakeLists.txt`.

At this point `sequential.cpp` no longer owns the individual module wrapper
implementations; those live under `src/algorithms/sequential/` by family.
### 2026-06-21 - Time series implementation split slice 1: statistics utilities

Started the backend time-series implementation split by moving the small
statistics and rolling-window utility block out of `src/algorithms/time_series.cpp`:

- added `src/algorithms/time_series/statistics.cpp` for `Mean`, `Variance`,
  `StdDev`, `RollingMean`, `RollingStd`, and `CenteredMovingAverage`,
- kept the public `include/cyxwiz/time_series.h` API unchanged,
- left decomposition, forecasting, differencing, spectral analysis, generators,
  and windowing in `time_series.cpp` for later focused slices,
- registered the new translation unit in `cyxwiz-backend/CMakeLists.txt`.

This starts reducing the active monolithic time-series implementation with a
self-contained helper family and no runtime behavior changes.
### 2026-06-21 - Time series implementation split slice 2: decomposition

Continued the backend time-series implementation split by moving decomposition
logic out of `src/algorithms/time_series.cpp`:

- added `src/algorithms/time_series/decomposition.cpp` for the decomposition
  implementation family,
- kept the public `include/cyxwiz/time_series.h` API unchanged,
- left stationarity/autocorrelation, differencing, spectral analysis,
  forecasting, generators, and windowing in `time_series.cpp` for later slices,
- registered the new translation unit in `cyxwiz-backend/CMakeLists.txt`.

This keeps seasonal/trend decomposition ownership separate from the remaining
forecasting and data-generation code without changing runtime behavior.
### 2026-06-21 - Time series implementation split slice 3: diagnostics

Continued the backend time-series implementation split by moving the next
coherent diagnostics section out of `src/algorithms/time_series.cpp`:

- added `src/algorithms/time_series/diagnostics.cpp` for stationarity,
  autocorrelation, and related diagnostic-test logic,
- kept the public `include/cyxwiz/time_series.h` API unchanged,
- left differencing, spectral analysis, forecasting, generators, and windowing
  in `time_series.cpp` for later focused slices,
- registered the new translation unit in `cyxwiz-backend/CMakeLists.txt`.

This separates diagnostic analysis ownership from decomposition and forecasting
without changing runtime behavior.
### 2026-06-21 - Time series implementation split slice 4: stationarity

Continued the backend time-series implementation split by moving stationarity
logic out of `src/algorithms/time_series.cpp`:

- added `src/algorithms/time_series/stationarity.cpp` for stationarity tests,
  ADF/KPSS checks, and the regular/seasonal differencing helpers used by those
  checks and forecasting code,
- kept the public `include/cyxwiz/time_series.h` API unchanged,
- left seasonality detection, spectral analysis, forecasting, generators, and
  windowing in `time_series.cpp` for later focused slices,
- registered the new translation unit in `cyxwiz-backend/CMakeLists.txt`.

This separates stationarity and differencing ownership from autocorrelation
 diagnostics and forecasting without changing runtime behavior.
## Reopened Status

Status: active.

The tensor parity slice described in this file is complete. The C++ backend
tensor API now covers the required parity operations, the API has been split
into focused tensor translation units, Python bindings expose the tensor
surface, Studio exposes the tensor nodes as concrete frontend nodes, and
PyCyxWiz code export can generate graph-aware tensor operation code.

That does not close the wider modularity task. The remaining work is to keep
breaking oversized engine and Studio files into clear translation units and
folders without changing behavior just for style.

Remaining modularity targets:

- backend layer files that still mix many unrelated layer implementations,
- optimizer files that should be grouped by optimizer family and algorithm,
- loss files that should isolate contracts such as cross entropy, NLL, MSE, and
  sequence losses,
- metrics/evaluation files that should separate classification, regression,
  sequence, benchmark, and experiment metrics,
- frontend files that mix panel layout, state, graph operations, dialogs, and
  execution behavior,
- any file large enough that feature ownership is unclear before reading the
  whole file.

Guardrails:

- do not rewrite working logic during a split unless a test-proven bug is being
  fixed,
- move one coherent module family at a time,
- keep public APIs stable unless a better typed boundary is part of the batch,
- add or preserve focused tests before claiming a split is safe,
- prefer smaller translation units over new abstraction layers.

Remaining local Studio C++ runtime execution for arbitrary tensor graph nodes
is not part of this modularity item. That work has been carried forward into
`tofix14.md` as a separate runtime adapter concern.











### 2026-06-21 - Time series implementation split slice 5: seasonality

Moved `DetectSeasonality` and `Periodogram` from `cyxwiz-backend/src/algorithms/time_series.cpp`
into `cyxwiz-backend/src/algorithms/time_series/seasonality.cpp` and registered the new translation unit in
`cyxwiz-backend/CMakeLists.txt`.

This separates seasonality and spectral detection ownership from the remaining forecasting, synthetic data, and
windowing helpers without changing the public `TimeSeries` API.

### 2026-06-21 - Time series implementation split slice 6: forecasting

Moved forecasting implementations from `cyxwiz-backend/src/algorithms/time_series.cpp` into
`cyxwiz-backend/src/algorithms/time_series/forecasting.cpp` and registered the new translation unit in
`cyxwiz-backend/CMakeLists.txt`.

This groups exponential smoothing, Holt/Holt-Winters, moving average forecasting, and ARIMA ownership separately
from synthetic data generation and ML windowing while preserving the public `TimeSeries` API.

### 2026-06-21 - Time series implementation split slice 7: generation

Moved synthetic time-series generators from `cyxwiz-backend/src/algorithms/time_series.cpp` into
`cyxwiz-backend/src/algorithms/time_series/generation.cpp` and registered the new translation unit in
`cyxwiz-backend/CMakeLists.txt`.

This isolates white-noise, random-walk, trend-seasonal, AR, MA, and ARIMA generation helpers from the remaining
ML windowing code while preserving the public `TimeSeries` API.

### 2026-06-21 - Time series implementation split slice 8: windowing

Moved ML windowing and feature-construction helpers from `cyxwiz-backend/src/algorithms/time_series.cpp` into
`cyxwiz-backend/src/algorithms/time_series/windowing.cpp` and registered the new translation unit in
`cyxwiz-backend/CMakeLists.txt`.

This completes the current `time_series.cpp` implementation split: statistics, decomposition, diagnostics,
stationarity, seasonality, forecasting, generation, and windowing now each have focused translation-unit ownership
while preserving the public `TimeSeries` API.

### 2026-06-21 - Clustering implementation split slice 1: evaluation

Moved cluster-evaluation metrics from `cyxwiz-backend/src/algorithms/clustering.cpp` into
`cyxwiz-backend/src/algorithms/clustering_evaluation.cpp` and registered the new translation unit in
`cyxwiz-backend/CMakeLists.txt`.

This isolates silhouette, Davies-Bouldin, Calinski-Harabasz, and matching non-ArrayFire fallback ownership from the
main clustering algorithms without changing the public `Clustering` API.

### 2026-06-21 - Clustering implementation split slice 2: KMeans

Moved KMeans centroid initialization, assignment/update helpers, inertia, main KMeans execution, elbow analysis,
and matching non-ArrayFire fallback stubs from `cyxwiz-backend/src/algorithms/clustering.cpp` into
`cyxwiz-backend/src/algorithms/clustering_kmeans.cpp` and registered the new translation unit in
`cyxwiz-backend/CMakeLists.txt`.

This separates KMeans ownership from DBSCAN, hierarchical clustering, and GMM while preserving the public
`Clustering` API.

### 2026-06-21 - Clustering implementation split slice 3: DBSCAN

Moved DBSCAN implementation, k-distance helper, and matching non-ArrayFire fallback stubs from
`cyxwiz-backend/src/algorithms/clustering.cpp` into `cyxwiz-backend/src/algorithms/clustering_dbscan.cpp` and
registered the new translation unit in `cyxwiz-backend/CMakeLists.txt`.

This separates DBSCAN ownership from hierarchical clustering and GMM while preserving the public `Clustering` API.

### 2026-06-21 - Clustering implementation split slice 4: hierarchical

Moved hierarchical clustering, dendrogram cutting, and matching non-ArrayFire fallback stubs from
`cyxwiz-backend/src/algorithms/clustering.cpp` into `cyxwiz-backend/src/algorithms/clustering_hierarchical.cpp` and
registered the new translation unit in `cyxwiz-backend/CMakeLists.txt`.

This separates hierarchical clustering ownership from GMM while preserving the public `Clustering` API.

### 2026-06-21 - Clustering implementation split slice 5: GMM

Moved Gaussian mixture model execution, EM helper functions, Gaussian PDF/log-likelihood helpers, and matching
non-ArrayFire fallback stub from `cyxwiz-backend/src/algorithms/clustering.cpp` into
`cyxwiz-backend/src/algorithms/clustering_gmm.cpp` and registered the new translation unit in
`cyxwiz-backend/CMakeLists.txt`.

This leaves `clustering.cpp` focused on shared ArrayFire conversion and distance helpers while preserving the public
`Clustering` API.

### 2026-06-21 - Linear algebra implementation split slice 1: tensor operations

Moved tensor-first linear algebra operations from `cyxwiz-backend/src/algorithms/linear_algebra.cpp` into
`cyxwiz-backend/src/algorithms/linear_algebra_tensor.cpp` and registered the new translation unit in
`cyxwiz-backend/CMakeLists.txt`.

This separates Tensor-backed multiply, transpose, inverse, norm, solve, and least-squares ownership from the
vector/matrix linear algebra implementation while preserving the public `LinearAlgebra` API.

### 2026-06-21 - Linear algebra implementation split slice 2: utility constructors

Moved identity, zeros, ones, diagonal extraction/construction, and low-rank approximation helpers from
`cyxwiz-backend/src/algorithms/linear_algebra.cpp` into
`cyxwiz-backend/src/algorithms/linear_algebra_utilities.cpp` and registered the new translation unit in
`cyxwiz-backend/CMakeLists.txt`.

This separates matrix construction and utility ownership from core vector/matrix operations while preserving the
public `LinearAlgebra` API.

### 2026-06-21 - Data Studio frontend split slice 1: query editor rendering

Moved QueryEditor UI rendering methods from `cyxwiz-engine/src/gui/data_studio/query_editor.cpp` into
`cyxwiz-engine/src/gui/data_studio/query_editor_render.cpp` and registered the new translation unit in
`cyxwiz-engine/CMakeLists.txt`.

This separates SQL editor/result-table rendering from DuckDB execution, dataset registration, and query-result
materialization while preserving the public `QueryEditor` API.

### 2026-06-21 - Data Studio frontend split slice 2: visualizer rendering

Moved Visualizer UI and plot rendering methods from `cyxwiz-engine/src/gui/data_studio/visualizer.cpp` into
`cyxwiz-engine/src/gui/data_studio/visualizer_render.cpp` and registered the new translation unit in
`cyxwiz-engine/CMakeLists.txt`.

This separates ImGui/ImPlot rendering ownership from dataset registration, plot creation, data loading, and plot
lifecycle operations while preserving the public `Visualizer` API.

### 2026-06-21 - Data Studio frontend split slice 3: analyzer rendering

Moved Analyzer UI rendering methods from `cyxwiz-engine/src/gui/data_studio/analyzer.cpp` into
`cyxwiz-engine/src/gui/data_studio/analyzer_render.cpp` and registered the new translation unit in
`cyxwiz-engine/CMakeLists.txt`.

This separates ImGui analysis/report rendering from dataset registration, analysis execution, and compute helpers
while preserving the public `Analyzer` API.

### 2026-06-21 - Data Studio frontend split slice 4: panel rendering

Moved DataStudioPanel window, toolbar, dataset selector, tab bar, and status bar rendering from
`cyxwiz-engine/src/gui/data_studio/data_studio_panel.cpp` into
`cyxwiz-engine/src/gui/data_studio/data_studio_panel_render.cpp` and registered the new translation unit in
`cyxwiz-engine/CMakeLists.txt`.

This separates the panel container UI from component construction and active-dataset propagation while preserving the
public `DataStudioPanel` API.
