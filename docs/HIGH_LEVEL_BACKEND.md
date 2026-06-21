# CyxWiz Backend High-Level Architecture

This document describes the CyxWiz backend from high-level compute down
to low-level compute execution.

The backend is intended to be a C++ machine-learning engine used by
CyxWiz Studio, graph execution, Python bindings, and deployment paths.

## Compute Stack

```text
+------------------------------------------------------------------+
|                    CyxWiz Studio / Graph UI                      |
|                                                                  |
|  Nodes, workflows, training dashboard, debugger, model export    |
+------------------------------------------------------------------+
                                |
                                v
+------------------------------------------------------------------+
|                    Runtime / Node Executor                       |
|                                                                  |
|  Converts graph nodes into backend calls                         |
|  Manages training runs, inference requests, data flow, errors     |
+------------------------------------------------------------------+
                                |
                                v
+------------------------------------------------------------------+
|                    Backend Public API Layer                      |
|                                                                  |
|  cyxwiz.h                                                        |
|  cyxwiz_c.h                                                      |
|  python/bindings.cpp                                             |
|                                                                  |
|  Exposes C++, C ABI, and Python access to backend features        |
+------------------------------------------------------------------+
                                |
                                v
+------------------------------------------------------------------+
|                    Model / Training Layer                        |
|                                                                  |
|  model.h / model.cpp                                             |
|  sequential.h / sequential.cpp                                   |
|  layer.h / layer.cpp                                             |
|  loss.h / loss.cpp                                               |
|  optimizer.h / optimizer.cpp                                     |
|  scheduler.h / scheduler.cpp                                     |
|                                                                  |
|  Builds trainable models, runs forward/backward, updates weights  |
+------------------------------------------------------------------+
                                |
                                v
+------------------------------------------------------------------+
|                    Algorithm Layer                               |
|                                                                  |
|  Neural layers: Linear, Conv2D, LSTM, GRU, Transformer, etc.      |
|  Activations: ReLU, Sigmoid, Tanh                                |
|  ML algorithms: clustering, dimensionality reduction, evaluation  |
|  Domain algorithms: text, tokenizer, audio, signal, time series   |
|  Interpretability: feature importance, model interpretability     |
|  Optimization / calculus utilities                               |
+------------------------------------------------------------------+
                                |
                                v
+------------------------------------------------------------------+
|                    Tensor / Data Layer                           |
|                                                                  |
|  tensor.h / tensor.cpp                                           |
|  data_loader.h / data_loader.cpp                                 |
|  dataloader.h / dataloader.cpp                                   |
|  data_transform.h / data_transform.cpp                           |
|                                                                  |
|  Owns tensor shape, dtype, memory layout, data batches, transforms|
+------------------------------------------------------------------+
                                |
                                v
+------------------------------------------------------------------+
|                    Device / Memory Runtime                       |
|                                                                  |
|  engine.h / engine.cpp                                           |
|  device.h / device.cpp                                           |
|  memory_manager.h / memory_manager.cpp                           |
|                                                                  |
|  Initializes backend, selects CPU/GPU, manages allocations        |
+------------------------------------------------------------------+
                                |
                                v
+------------------------------------------------------------------+
|                    Low-Level Compute Backends                    |
|                                                                  |
|  ArrayFire                                                       |
|  CUDA / OpenCL / CPU backend                                     |
|  FFTW3 for FFT features                                          |
|  libsndfile for audio I/O                                        |
|  DuckDB for optional SQL data loading                            |
|  WinSock / NCCL-style distributed communication paths             |
+------------------------------------------------------------------+
```

In short:

```text
Studio graph
   -> backend API
      -> model/training API
         -> layers/losses/optimizers/metrics
            -> tensor operations
               -> device + memory runtime
                  -> ArrayFire / CUDA / OpenCL / CPU
```

## Current Backend Layout

The current backend lives at:

```text
cyxwiz-backend/
```

Important current source areas:

```text
cyxwiz-backend/include/cyxwiz/
cyxwiz-backend/src/core/
cyxwiz-backend/src/algorithms/
cyxwiz-backend/python/
cyxwiz-backend/examples/python/
```

Current core runtime:

```text
include/cyxwiz/tensor.h
src/core/tensor.cpp

include/cyxwiz/device.h
src/core/device.cpp

include/cyxwiz/engine.h
src/core/engine.cpp

include/cyxwiz/memory_manager.h
src/core/memory_manager.cpp
```

Current public API:

```text
include/cyxwiz/cyxwiz.h
include/cyxwiz/cyxwiz_c.h
python/bindings.cpp
```

Current model/training files:

```text
include/cyxwiz/model.h
src/algorithms/model.cpp

include/cyxwiz/sequential.h
src/algorithms/sequential.cpp

include/cyxwiz/loss.h
src/algorithms/loss.cpp

include/cyxwiz/optimizer.h
src/algorithms/optimizer.cpp

include/cyxwiz/scheduler.h
src/algorithms/scheduler.cpp
```

## Current Structural Problem

The largest architecture problem is that many unrelated neural network
layers are concentrated in one large header and one large implementation
file:

```text
include/cyxwiz/layer.h
src/algorithms/layer.cpp
```

That module currently contains or owns many layer families:

```text
include/cyxwiz/layer.h
src/algorithms/layer.cpp
        |
        +-- Dense / Linear-style layers
        +-- Conv2D
        +-- MaxPool2D
        +-- AvgPool2D
        +-- GlobalAvgPool2D
        +-- BatchNorm2D
        +-- Flatten
        +-- Dropout
        +-- LSTM
        +-- GRU
        +-- Embedding
        +-- LayerNorm
        +-- InstanceNorm2D
        +-- GroupNorm
        +-- Conv1D
        +-- MultiHeadAttention
        +-- TransformerEncoder
        +-- TransformerDecoder
        +-- ConvTranspose2D
        +-- Upsample2D
        +-- PixelShuffle
```

Concrete example:

```text
Current LSTM declaration:
cyxwiz-backend/include/cyxwiz/layer.h

Current LSTM implementation:
cyxwiz-backend/src/algorithms/layer.cpp

Current GRU declaration:
cyxwiz-backend/include/cyxwiz/layer.h

Current GRU implementation:
cyxwiz-backend/src/algorithms/layer.cpp
```

This makes the backend harder to maintain because a change to one layer
family risks touching a very large shared file. It also makes review,
testing, documentation, and ownership less clear.

## Target Layer Layout

The target architecture should keep a small base `Layer` abstraction and
move each major layer family into its own header and implementation
file.

Target public headers:

```text
include/cyxwiz/layer.h              # base Layer only

include/cyxwiz/layers/
    linear.h
    conv1d.h
    conv2d.h
    conv_transpose2d.h
    depthwise_separable_conv2d.h
    lstm.h
    gru.h
    embedding.h
    attention.h
    transformer.h
    transformer_decoder.h
    positional_encoding.h
    batchnorm.h
    layernorm.h
    groupnorm.h
    pooling.h
    dropout.h
```

Target implementations:

```text
src/algorithms/layers/
    linear.cpp
    conv1d.cpp
    conv2d.cpp
    conv_transpose2d.cpp
    depthwise_separable_conv2d.cpp
    lstm.cpp
    gru.cpp
    embedding.cpp
    attention.cpp
    transformer.cpp
    transformer_decoder.cpp
    positional_encoding.cpp
    batchnorm.cpp
    layernorm.cpp
    groupnorm.cpp
    pooling.cpp
    dropout.cpp
```

Target recurrent layer example:

```text
LSTM:
include/cyxwiz/layers/lstm.h
src/algorithms/layers/lstm.cpp

GRU:
include/cyxwiz/layers/gru.h
src/algorithms/layers/gru.cpp
```

## Target Backend Modules

The backend should evolve toward this module layout:

```text
cyxwiz-backend/
    include/cyxwiz/
        cyxwiz.h
        cyxwiz_c.h
        api_export.h

        core-facing public headers:
            tensor.h
            device.h
            engine.h
            memory_manager.h

        layers/
        losses/
        metrics/
        data/
        models/
        distributed/

        optimizer.h
        scheduler.h
        serialization.h
        checkpointing.h
        quantization.h
        onnx_export.h
        model_config.h
        model_registry.h
        model_hub.h

    src/
        core/
        algorithms/
            layers/
            losses/
            metrics/
            data/
            models/
            distributed/
        utils/

    python/
    tests/
    benchmarks/
    cpp_docs/
    py_docs/
    examples/
```

## Compute Responsibilities By Layer

### Studio / Graph UI

Responsible for:

- visual graph creation
- user workflow configuration
- training dashboard
- debugger display
- model export controls
- mapping user intent to graph nodes

It should not own low-level tensor math.

### Runtime / Node Executor

Responsible for:

- executing graph nodes in order
- passing tensors and data batches between nodes
- managing run state
- reporting backend errors
- coordinating pause, continue, cancel, and debugger events

It should call the backend through stable APIs.

### Backend Public API

Responsible for:

- C++ public includes
- C ABI for external integrations
- Python bindings
- stable exported symbols
- initialization and shutdown entry points

This is the boundary between application/runtime code and the backend
engine.

### Model / Training Layer

Responsible for:

- model containers
- sequential graphs
- layer composition
- loss computation
- optimizer steps
- learning-rate scheduling
- forward and backward orchestration

This layer should feel similar to a PyTorch-style training API, but in
C++ and integrated with CyxWiz.

### Algorithm Layer

Responsible for:

- neural network layers
- activations
- clustering
- dimensionality reduction
- model evaluation
- feature importance
- interpretability
- linear algebra
- signal processing
- optimization utilities
- time series
- text/token processing
- audio processing
- RL interface

This layer should be organized by feature family, not by one large mixed
file.

### Tensor / Data Layer

Responsible for:

- tensor shape
- dtype
- tensor storage
- CPU/GPU array interop
- tensor operations
- data loading
- batching
- transforms

This is the foundation for all higher-level algorithms.

### Device / Memory Runtime

Responsible for:

- backend initialization
- CPU/GPU device selection
- memory allocation
- memory tracking
- backend feature detection
- runtime dispatch to ArrayFire/CPU/GPU paths

### Low-Level Compute Backends

Responsible for:

- actual numerical kernels
- GPU execution
- CPU fallback
- FFT operations
- audio file I/O
- SQL-backed data loading
- distributed communication primitives

Current external/optional compute dependencies include:

- ArrayFire
- CUDA through ArrayFire runtime selection
- OpenCL through ArrayFire runtime selection
- FFTW3
- libsndfile
- DuckDB
- WinSock on Windows for distributed CPU backend
- optional NCCL-style distributed GPU backend

## Architecture Fix Priorities

1. Expand tensor API to support framework-level algorithms.
2. Split monolithic layer files into per-layer modules.
3. Add organized `losses`, `metrics`, `data`, `models`, and `utils`
   folders.
4. Preserve current Studio-facing APIs while moving internals.
5. Add tests around tensor ops, layers, losses, metrics, Python bindings,
   and Studio graph training.
6. Add model lifecycle modules: serialization, checkpointing, ONNX,
   quantization, model registry, and model hub.
7. Keep low-level compute dependencies behind clean backend interfaces.

The design goal is a PyTorch-like backend architecture in C++:

```text
Tensor
  -> Layer / Module
      -> Model
          -> Loss + Optimizer + Scheduler
              -> Trainer / Evaluation / Export
```

with CyxWiz Studio using that backend through graph nodes and runtime
execution, not by duplicating backend compute in the UI.
