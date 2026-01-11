# CyxWiz Professional Debug System - Complete Design Document

## Vision

A **comprehensive, professional-grade debugging system** that provides complete visibility into every aspect of CyxWiz:
- Training computations
- Data pipelines
- Memory management
- GPU operations
- Network/gRPC calls
- Script execution
- Node editor state
- System resources
- Threading/concurrency
- File I/O
- Event system
- UI state

The debug system should be accessible from **Console Panel** with toggleable subsystems.

---

## Problem Statement

Training the same model on different PCs produces different results. We need:
1. Complete traceability of every operation
2. Cross-machine reproducibility
3. Real-time system monitoring
4. Professional debugging tools

---

## Root Causes of Training Differences

| Category | Causes |
|----------|--------|
| **Non-Determinism** | GPU atomics, cuDNN auto-selection, parallel reductions |
| **Random State** | Seeds, RNG state, dropout, initialization, shuffling |
| **Hardware** | GPU architecture, Tensor Cores, CPU instruction sets |
| **Software** | ArrayFire/CUDA/cuDNN versions, compiler flags |
| **Floating-Point** | Rounding modes, operation ordering, precision |

---

## Debug System Architecture

### Master Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          CyxWiz Debug System                                     │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                         Debug Controller (Singleton)                        │ │
│  │  - Master on/off switch                                                     │ │
│  │  - Subsystem toggles                                                        │ │
│  │  - Log routing                                                              │ │
│  │  - Performance monitoring                                                   │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                        │                                         │
│         ┌──────────────────────────────┼──────────────────────────────┐          │
│         │                              │                              │          │
│         ▼                              ▼                              ▼          │
│  ┌─────────────┐               ┌─────────────┐               ┌─────────────┐    │
│  │  Training   │               │   System    │               │  Network    │    │
│  │   Debug     │               │   Debug     │               │   Debug     │    │
│  └─────────────┘               └─────────────┘               └─────────────┘    │
│         │                              │                              │          │
│         ▼                              ▼                              ▼          │
│  ┌─────────────┐               ┌─────────────┐               ┌─────────────┐    │
│  │    Data     │               │    GPU      │               │   Script    │    │
│  │   Debug     │               │   Debug     │               │   Debug     │    │
│  └─────────────┘               └─────────────┘               └─────────────┘    │
│         │                              │                              │          │
│         ▼                              ▼                              ▼          │
│  ┌─────────────┐               ┌─────────────┐               ┌─────────────┐    │
│  │   Memory    │               │  Threading  │               │  Node Graph │    │
│  │   Debug     │               │   Debug     │               │   Debug     │    │
│  └─────────────┘               └─────────────┘               └─────────────┘    │
│         │                              │                              │          │
│         ▼                              ▼                              ▼          │
│  ┌─────────────┐               ┌─────────────┐               ┌─────────────┐    │
│  │   File I/O  │               │   Event     │               │     UI      │    │
│  │   Debug     │               │   Debug     │               │   Debug     │    │
│  └─────────────┘               └─────────────┘               └─────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                              Output Sinks                                   │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │ │
│  │  │ Console  │  │   File   │  │   JSON   │  │  Binary  │  │ Remote/Cloud │  │ │
│  │  │  Panel   │  │   Log    │  │  Trace   │  │  Dump    │  │   Upload     │  │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Debug Subsystems (All 12)

### 1. Training Debug 🏋️

Everything related to ML training computations.

```cpp
struct TrainingDebugConfig {
    bool enabled = false;

    // Forward Pass
    bool trace_forward = true;
    bool log_activations = true;
    bool log_activation_stats = true;     // min/max/mean/std
    bool log_activation_histograms = true; // For dead neuron detection
    bool log_layer_outputs = true;

    // Backward Pass
    bool trace_backward = true;
    bool log_gradients = true;
    bool log_gradient_stats = true;
    bool log_gradient_flow = true;        // Track gradient through layers
    bool detect_vanishing_gradients = true;
    bool detect_exploding_gradients = true;
    float gradient_explosion_threshold = 1000.0f;
    float gradient_vanishing_threshold = 1e-7f;

    // Optimizer
    bool trace_optimizer = true;
    bool log_weight_updates = true;
    bool log_learning_rate = true;
    bool log_momentum_state = true;
    bool log_adaptive_lr = true;          // For Adam, RMSprop

    // Loss
    bool trace_loss = true;
    bool log_loss_components = true;      // For composite losses
    bool log_loss_gradients = true;

    // Computation Details
    bool log_operation_sequence = true;
    bool log_tensor_shapes = true;
    bool log_tensor_dtypes = true;
    bool log_tensor_devices = true;
    bool log_tensor_values = false;       // HUGE output
    bool log_tensor_hashes = true;        // For reproducibility check

    // Numerical Stability
    bool check_nan = true;
    bool check_inf = true;
    bool check_underflow = true;
    bool check_overflow = true;
    bool halt_on_nan = false;

    // Reproducibility
    bool enable_deterministic = false;    // Slower but reproducible
    bool log_random_seeds = true;
    bool log_rng_state = true;
    bool save_rng_state_per_batch = false;

    // Checkpointing
    bool save_checkpoints = true;
    int checkpoint_interval = 100;        // Every N batches
    bool save_full_state = true;          // All weights, optimizer, RNG
};
```

**What it logs:**
```
[TRAIN] ═══════════════════════════════════════════════════════════
[TRAIN] Epoch 1/10, Batch 42/1000
[TRAIN] ───────────────────────────────────────────────────────────
[TRAIN] FORWARD PASS:
[TRAIN]   conv1: [32,1,28,28] → [32,32,26,26]
[TRAIN]     Activation: min=-2.34 max=4.56 mean=0.12 std=0.89 zeros=12.3%
[TRAIN]     Histogram: [0-0.5]=45% [0.5-1]=32% [1-2]=18% [>2]=5%
[TRAIN]   relu1: zeros=34.2% (potential dying ReLU)
[TRAIN]   conv2: [32,32,13,13] → [32,64,11,11]
[TRAIN]     Activation: min=-3.12 max=5.67 mean=0.23 std=1.12
[TRAIN]   fc1: [32,7744] → [32,128]
[TRAIN]   fc2: [32,128] → [32,10]
[TRAIN] ───────────────────────────────────────────────────────────
[TRAIN] LOSS: CrossEntropy = 2.3456
[TRAIN]   Predictions: softmax range [0.02, 0.45]
[TRAIN]   Target distribution: one-hot
[TRAIN] ───────────────────────────────────────────────────────────
[TRAIN] BACKWARD PASS:
[TRAIN]   fc2.grad:   norm=0.234 max=0.056 min=-0.048
[TRAIN]   fc1.grad:   norm=0.189 max=0.034 min=-0.031
[TRAIN]   conv2.grad: norm=0.145 max=0.028 min=-0.025
[TRAIN]   conv1.grad: norm=0.067 max=0.012 min=-0.011
[TRAIN]   ⚠ Gradient diminishing: conv1 = 28% of fc2
[TRAIN] ───────────────────────────────────────────────────────────
[TRAIN] OPTIMIZER (Adam, lr=0.001):
[TRAIN]   Weight updates: fc2=0.0045 fc1=0.0034 conv2=0.0023 conv1=0.0012
[TRAIN]   Momentum: β1=0.9 β2=0.999 step=42
[TRAIN]   Effective LR: fc2=0.00098 fc1=0.00097 (adaptive)
[TRAIN] ───────────────────────────────────────────────────────────
[TRAIN] BATCH SUMMARY:
[TRAIN]   Time: 23.4ms (fwd=8.2ms, bwd=12.1ms, opt=3.1ms)
[TRAIN]   Throughput: 1367 samples/sec
[TRAIN]   Loss trend: ↓ 2.3456 (prev: 2.4012, Δ=-2.3%)
[TRAIN] ═══════════════════════════════════════════════════════════
```

---

### 2. Data Pipeline Debug 📊

Everything about data loading, preprocessing, augmentation.

```cpp
struct DataDebugConfig {
    bool enabled = false;

    // Data Loading
    bool trace_data_loading = true;
    bool log_batch_composition = true;
    bool log_sample_indices = true;       // Which samples in each batch
    bool log_shuffle_order = true;
    bool log_worker_activity = true;      // Multi-threaded loading

    // Preprocessing
    bool trace_preprocessing = true;
    bool log_transform_chain = true;
    bool log_transform_params = true;
    bool log_normalization = true;        // Mean/std used

    // Augmentation
    bool trace_augmentation = true;
    bool log_augmentation_params = true;  // Random crop coords, flip, etc.
    bool log_augmentation_seeds = true;
    bool save_augmented_samples = false;  // Debug visualization

    // Data Statistics
    bool compute_batch_stats = true;
    bool compute_dataset_stats = true;
    bool detect_data_anomalies = true;    // NaN, outliers

    // Cache
    bool log_cache_hits = true;
    bool log_cache_misses = true;
    bool log_memory_usage = true;
};
```

**What it logs:**
```
[DATA] ═══════════════════════════════════════════════════════════
[DATA] Loading batch 42 (samples 1344-1375)
[DATA] ───────────────────────────────────────────────────────────
[DATA] Shuffle order for epoch 1: [23, 156, 78, 1024, 892, ...]
[DATA] Worker 0: loaded samples [1344-1351] in 2.3ms
[DATA] Worker 1: loaded samples [1352-1359] in 2.1ms
[DATA] Worker 2: loaded samples [1360-1367] in 2.4ms
[DATA] Worker 3: loaded samples [1368-1375] in 2.2ms
[DATA] ───────────────────────────────────────────────────────────
[DATA] Transform chain:
[DATA]   1. Resize(224, 224)
[DATA]   2. RandomHorizontalFlip(p=0.5) → applied to 18/32 samples
[DATA]   3. RandomRotation(15°) → angles: [-12°, 8°, 3°, -7°, ...]
[DATA]   4. ColorJitter(0.2) → brightness: [0.92, 1.08, 1.15, ...]
[DATA]   5. ToTensor()
[DATA]   6. Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
[DATA] ───────────────────────────────────────────────────────────
[DATA] Batch statistics:
[DATA]   Shape: [32, 3, 224, 224]
[DATA]   Range: min=-2.12, max=2.64
[DATA]   Mean per channel: [0.02, -0.01, 0.03]
[DATA]   Std per channel: [1.01, 0.99, 1.02]
[DATA]   Labels: [3, 7, 2, 9, 0, 1, 4, 6, ...] (distribution: balanced)
[DATA] ───────────────────────────────────────────────────────────
[DATA] Cache: 45% hit rate, 234MB used / 512MB allocated
[DATA] ═══════════════════════════════════════════════════════════
```

---

### 3. GPU Debug 🎮

GPU operations, memory, kernel execution.

```cpp
struct GPUDebugConfig {
    bool enabled = false;

    // Device Info
    bool log_device_info = true;
    bool log_driver_version = true;
    bool log_compute_capability = true;

    // Memory
    bool trace_allocations = true;
    bool trace_deallocations = true;
    bool log_memory_usage = true;
    bool log_memory_fragmentation = true;
    bool detect_memory_leaks = true;
    bool log_peak_memory = true;

    // Kernels
    bool trace_kernel_launches = true;
    bool log_kernel_params = true;        // Grid, block sizes
    bool log_kernel_timing = true;
    bool log_occupancy = true;

    // Transfers
    bool trace_host_device_transfers = true;
    bool log_transfer_sizes = true;
    bool log_transfer_timing = true;
    bool warn_unnecessary_transfers = true;

    // cuDNN/cuBLAS
    bool log_algorithm_selection = true;  // Which conv algo chosen
    bool log_workspace_usage = true;

    // Synchronization
    bool trace_synchronization = true;
    bool log_stream_activity = true;
    bool detect_race_conditions = true;

    // Errors
    bool check_cuda_errors = true;
    bool halt_on_cuda_error = true;
};
```

**What it logs:**
```
[GPU] ═══════════════════════════════════════════════════════════
[GPU] Device: NVIDIA RTX 4090 (Compute 8.9)
[GPU] Driver: 545.23.08 | CUDA: 12.3 | cuDNN: 8.9.5
[GPU] ───────────────────────────────────────────────────────────
[GPU] Memory Status:
[GPU]   Allocated: 4.2 GB / 24 GB (17.5%)
[GPU]   Peak: 6.8 GB
[GPU]   Fragmentation: 3.2%
[GPU]   Active allocations: 156
[GPU] ───────────────────────────────────────────────────────────
[GPU] Kernel: cudnn_conv2d_forward
[GPU]   Grid: (128, 1, 1), Block: (256, 1, 1)
[GPU]   Shared memory: 48 KB
[GPU]   Registers/thread: 64
[GPU]   Occupancy: 78%
[GPU]   Time: 0.234ms
[GPU]   Algorithm: CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
[GPU]   Workspace: 2.4 MB
[GPU] ───────────────────────────────────────────────────────────
[GPU] ⚠ Host→Device transfer: 32 MB (input batch)
[GPU]   Consider keeping data on GPU between batches
[GPU] ───────────────────────────────────────────────────────────
[GPU] CUDA Error Check: All operations successful
[GPU] ═══════════════════════════════════════════════════════════
```

---

### 4. Memory Debug 💾

System-wide memory tracking.

```cpp
struct MemoryDebugConfig {
    bool enabled = false;

    // Allocations
    bool trace_all_allocations = true;
    bool log_allocation_size = true;
    bool log_allocation_source = true;    // Stack trace
    bool log_allocation_type = true;      // Tensor, buffer, string, etc.

    // Tracking
    bool track_peak_usage = true;
    bool track_per_component_usage = true;
    bool track_memory_timeline = true;

    // Detection
    bool detect_leaks = true;
    bool detect_double_free = true;
    bool detect_use_after_free = true;
    bool detect_buffer_overflow = true;   // Guard pages

    // Warnings
    bool warn_large_allocations = true;
    size_t large_allocation_threshold = 100 * 1024 * 1024; // 100MB
    bool warn_fragmentation = true;
    float fragmentation_threshold = 0.3f;

    // Reporting
    bool periodic_memory_report = true;
    int report_interval_seconds = 60;
    bool log_top_allocations = true;
    int top_n_allocations = 10;
};
```

**What it logs:**
```
[MEM] ═══════════════════════════════════════════════════════════
[MEM] MEMORY REPORT (periodic, every 60s)
[MEM] ───────────────────────────────────────────────────────────
[MEM] System Memory:
[MEM]   Total: 64 GB
[MEM]   Used: 12.4 GB (19.4%)
[MEM]   Available: 51.6 GB
[MEM] ───────────────────────────────────────────────────────────
[MEM] CyxWiz Memory:
[MEM]   Total allocated: 2.8 GB
[MEM]   Peak: 4.1 GB
[MEM]   Active allocations: 1,245
[MEM] ───────────────────────────────────────────────────────────
[MEM] By Component:
[MEM]   Training tensors:    1.2 GB (428 allocations)
[MEM]   Data pipeline:       0.8 GB (312 allocations)
[MEM]   Model weights:       0.5 GB (156 allocations)
[MEM]   UI/Rendering:        0.2 GB (245 allocations)
[MEM]   Misc:                0.1 GB (104 allocations)
[MEM] ───────────────────────────────────────────────────────────
[MEM] Top 5 Allocations:
[MEM]   1. 512 MB - DataLoader batch cache
[MEM]   2. 256 MB - Conv2d weight tensor (layer conv1)
[MEM]   3. 128 MB - Activation buffer (forward pass)
[MEM]   4. 64 MB  - Gradient accumulator
[MEM]   5. 32 MB  - Optimizer momentum state
[MEM] ───────────────────────────────────────────────────────────
[MEM] ⚠ Warning: 3 potential memory leaks detected
[MEM]   - 256 KB at scripting/python_engine.cpp:342 (unreleased PyObject)
[MEM]   - 128 KB at gui/node_editor.cpp:891 (orphaned node)
[MEM]   - 64 KB at network/grpc_client.cpp:156 (unclosed stream)
[MEM] ═══════════════════════════════════════════════════════════
```

---

### 5. Network Debug 🌐

gRPC, HTTP, P2P communications.

```cpp
struct NetworkDebugConfig {
    bool enabled = false;

    // gRPC
    bool trace_grpc_calls = true;
    bool log_request_payload = true;
    bool log_response_payload = true;
    bool log_grpc_metadata = true;
    bool log_grpc_timing = true;
    bool log_grpc_errors = true;
    bool log_retry_attempts = true;

    // Connection
    bool trace_connections = true;
    bool log_connection_state = true;     // Connecting, connected, etc.
    bool log_reconnection_attempts = true;
    bool log_ssl_handshake = true;

    // Data Transfer
    bool log_bytes_sent = true;
    bool log_bytes_received = true;
    bool log_bandwidth = true;
    bool log_latency = true;

    // Protocol
    bool log_message_serialization = true;
    bool log_protobuf_details = true;

    // Errors
    bool log_network_errors = true;
    bool log_timeout = true;
    bool halt_on_network_error = false;
};
```

**What it logs:**
```
[NET] ═══════════════════════════════════════════════════════════
[NET] gRPC Call: JobService.SubmitJob
[NET] ───────────────────────────────────────────────────────────
[NET] Connection: central-server.cyxwiz.io:50051
[NET]   State: CONNECTED (since 10:23:45)
[NET]   TLS: Enabled (TLS 1.3)
[NET]   Latency: 45ms avg (last 100 calls)
[NET] ───────────────────────────────────────────────────────────
[NET] Request:
[NET]   Method: /cyxwiz.JobService/SubmitJob
[NET]   Metadata: {authorization: Bearer <token>, request-id: abc123}
[NET]   Payload size: 2.4 KB
[NET]   Payload: {
[NET]     "job_type": "TRAINING",
[NET]     "model_hash": "a3f2b7c9...",
[NET]     "dataset_id": "mnist-train",
[NET]     "config": { "epochs": 10, "batch_size": 32, ... }
[NET]   }
[NET] ───────────────────────────────────────────────────────────
[NET] Response:
[NET]   Status: OK (0)
[NET]   Time: 156ms
[NET]   Payload size: 512 bytes
[NET]   Payload: {
[NET]     "job_id": "job-12345",
[NET]     "status": "QUEUED",
[NET]     "estimated_start": "2024-01-15T10:25:00Z"
[NET]   }
[NET] ───────────────────────────────────────────────────────────
[NET] Bandwidth: ↑ 2.4 KB/s | ↓ 512 B/s | Total: 45.6 MB today
[NET] ═══════════════════════════════════════════════════════════
```

---

### 6. Script Debug 📜

Python script execution and bindings.

```cpp
struct ScriptDebugConfig {
    bool enabled = false;

    // Execution
    bool trace_script_execution = true;
    bool log_function_calls = true;
    bool log_function_args = true;
    bool log_return_values = true;
    bool log_exceptions = true;

    // Python State
    bool log_python_objects = true;
    bool log_reference_counts = true;
    bool detect_reference_leaks = true;
    bool log_garbage_collection = true;

    // Bindings
    bool trace_pybind_calls = true;
    bool log_type_conversions = true;
    bool log_buffer_protocol = true;

    // Performance
    bool profile_script_execution = true;
    bool log_slow_calls = true;
    float slow_call_threshold_ms = 10.0f;

    // REPL
    bool log_repl_input = true;
    bool log_repl_output = true;
};
```

**What it logs:**
```
[SCRIPT] ═══════════════════════════════════════════════════════
[SCRIPT] Executing: train_model.py
[SCRIPT] ─────────────────────────────────────────────────────────
[SCRIPT] Python: 3.11.5 | pybind11: 2.11.1 | pycyxwiz: 0.2.0
[SCRIPT] ─────────────────────────────────────────────────────────
[SCRIPT] Line 15: model = pycyxwiz.Sequential()
[SCRIPT]   → C++ call: Sequential::Sequential()
[SCRIPT]   ← Returned: <Sequential object at 0x7f8a...>
[SCRIPT]   Ref count: 1
[SCRIPT]
[SCRIPT] Line 16: model.add(pycyxwiz.Conv2d(1, 32, 3))
[SCRIPT]   → C++ call: Conv2d::Conv2d(in=1, out=32, kernel=3)
[SCRIPT]     Tensor allocation: weights [32,1,3,3] = 1.15 KB
[SCRIPT]     Tensor allocation: bias [32] = 128 bytes
[SCRIPT]   → C++ call: Sequential::add(Layer*)
[SCRIPT]   ← Success
[SCRIPT]
[SCRIPT] Line 20: loss = model.train_step(batch_x, batch_y)
[SCRIPT]   → Type conversion: numpy.ndarray → Tensor
[SCRIPT]     Shape: [32, 1, 28, 28], dtype: float32
[SCRIPT]     Buffer protocol: direct memory access (zero-copy)
[SCRIPT]   → C++ call: Sequential::train_step(Tensor, Tensor)
[SCRIPT]     [Training debug output here...]
[SCRIPT]   ← Returned: 2.3456 (float)
[SCRIPT]   Time: 23.4ms
[SCRIPT] ─────────────────────────────────────────────────────────
[SCRIPT] ⚠ Slow call detected: model.train_step took 23.4ms
[SCRIPT] GC: collected 45 objects, freed 2.3 KB
[SCRIPT] ═══════════════════════════════════════════════════════
```

---

### 7. Node Graph Debug 🔗

Node editor state and execution.

```cpp
struct NodeGraphDebugConfig {
    bool enabled = false;

    // Graph State
    bool log_node_creation = true;
    bool log_node_deletion = true;
    bool log_connection_creation = true;
    bool log_connection_deletion = true;
    bool log_graph_validation = true;

    // Execution
    bool trace_graph_execution = true;
    bool log_execution_order = true;      // Topological sort
    bool log_node_inputs = true;
    bool log_node_outputs = true;
    bool log_node_timing = true;

    // Code Generation
    bool log_codegen_steps = true;
    bool log_generated_code = true;
    bool log_code_optimization = true;

    // Serialization
    bool log_save_operations = true;
    bool log_load_operations = true;
    bool log_serialization_format = true;

    // Clipboard
    bool log_copy_paste = true;
    bool log_clipboard_contents = true;
};
```

**What it logs:**
```
[GRAPH] ═══════════════════════════════════════════════════════
[GRAPH] Graph: "mnist_classifier" (12 nodes, 11 connections)
[GRAPH] ─────────────────────────────────────────────────────────
[GRAPH] Graph Validation:
[GRAPH]   ✓ No cycles detected
[GRAPH]   ✓ All inputs connected
[GRAPH]   ✓ Shape inference passed
[GRAPH]   ✓ Data type compatibility verified
[GRAPH] ─────────────────────────────────────────────────────────
[GRAPH] Execution Order (topological):
[GRAPH]   1. DataInput (id=0) → output: [batch, 1, 28, 28]
[GRAPH]   2. Conv2d (id=1) → output: [batch, 32, 26, 26]
[GRAPH]   3. ReLU (id=2) → output: [batch, 32, 26, 26]
[GRAPH]   4. MaxPool2d (id=3) → output: [batch, 32, 13, 13]
[GRAPH]   5. Conv2d (id=4) → output: [batch, 64, 11, 11]
[GRAPH]   6. ReLU (id=5) → output: [batch, 64, 11, 11]
[GRAPH]   7. MaxPool2d (id=6) → output: [batch, 64, 5, 5]
[GRAPH]   8. Flatten (id=7) → output: [batch, 1600]
[GRAPH]   9. Linear (id=8) → output: [batch, 128]
[GRAPH]   10. ReLU (id=9) → output: [batch, 128]
[GRAPH]   11. Linear (id=10) → output: [batch, 10]
[GRAPH]   12. Output (id=11) → final output
[GRAPH] ─────────────────────────────────────────────────────────
[GRAPH] Node Execution: Conv2d (id=1)
[GRAPH]   Input: Tensor[32,1,28,28] from DataInput(id=0)
[GRAPH]   Params: kernel=3, stride=1, padding=0
[GRAPH]   Weights: [32,1,3,3] hash=a3f2b7c9...
[GRAPH]   Output: Tensor[32,32,26,26]
[GRAPH]   Time: 0.234ms
[GRAPH] ═══════════════════════════════════════════════════════
```

---

### 8. Threading Debug 🧵

Concurrency, async tasks, race conditions.

```cpp
struct ThreadingDebugConfig {
    bool enabled = false;

    // Thread Management
    bool log_thread_creation = true;
    bool log_thread_destruction = true;
    bool log_thread_names = true;
    bool log_thread_pool_activity = true;

    // Tasks
    bool trace_async_tasks = true;
    bool log_task_submission = true;
    bool log_task_completion = true;
    bool log_task_timing = true;
    bool log_task_dependencies = true;

    // Synchronization
    bool trace_locks = true;
    bool log_lock_acquisition = true;
    bool log_lock_release = true;
    bool log_lock_contention = true;
    bool detect_deadlocks = true;
    bool log_condition_variables = true;

    // Atomics
    bool trace_atomic_operations = true;
    bool log_memory_ordering = true;

    // Performance
    bool log_thread_cpu_time = true;
    bool log_context_switches = true;
};
```

**What it logs:**
```
[THREAD] ═══════════════════════════════════════════════════════
[THREAD] Thread Pool Status: 8 workers, 3 busy, 5 idle
[THREAD] ─────────────────────────────────────────────────────────
[THREAD] Active Threads:
[THREAD]   Main (id=1)        - UI/Event loop
[THREAD]   Worker-0 (id=2)    - Training batch 42
[THREAD]   Worker-1 (id=3)    - Data loading
[THREAD]   Worker-2 (id=4)    - Async file save
[THREAD]   Network (id=5)     - gRPC handler
[THREAD]   GPU-Stream (id=6)  - CUDA async
[THREAD] ─────────────────────────────────────────────────────────
[THREAD] Task: AsyncDataLoad
[THREAD]   Submitted by: Main (id=1)
[THREAD]   Picked up by: Worker-1 (id=3)
[THREAD]   Status: Running
[THREAD]   Progress: 45%
[THREAD]   Time elapsed: 1.2s
[THREAD] ─────────────────────────────────────────────────────────
[THREAD] Lock: training_mutex
[THREAD]   Held by: Worker-0 (id=2) for 12ms
[THREAD]   Waiting: Worker-3 (id=5) for 8ms
[THREAD]   ⚠ Contention detected (>5ms wait)
[THREAD] ─────────────────────────────────────────────────────────
[THREAD] Context switches: 45/sec (normal)
[THREAD] ═══════════════════════════════════════════════════════
```

---

### 9. File I/O Debug 📁

All file operations.

```cpp
struct FileIODebugConfig {
    bool enabled = false;

    // Operations
    bool trace_file_open = true;
    bool trace_file_close = true;
    bool trace_file_read = true;
    bool trace_file_write = true;
    bool trace_file_seek = true;

    // Details
    bool log_file_paths = true;
    bool log_operation_size = true;
    bool log_operation_timing = true;
    bool log_file_handles = true;

    // Errors
    bool log_file_errors = true;
    bool log_permission_issues = true;
    bool log_disk_full = true;

    // Resources
    bool log_open_file_count = true;
    bool warn_too_many_open_files = true;
    int max_open_files_warning = 100;

    // Serialization
    bool log_json_parsing = true;
    bool log_binary_format = true;
    bool log_model_save_load = true;
};
```

---

### 10. Event System Debug 📢

Application events, callbacks, signals.

```cpp
struct EventDebugConfig {
    bool enabled = false;

    // Event Flow
    bool trace_event_dispatch = true;
    bool log_event_type = true;
    bool log_event_source = true;
    bool log_event_handlers = true;
    bool log_event_timing = true;

    // UI Events
    bool trace_ui_events = true;
    bool log_mouse_events = false;        // Very noisy
    bool log_keyboard_events = true;
    bool log_window_events = true;

    // Custom Events
    bool trace_custom_events = true;
    bool log_training_events = true;
    bool log_network_events = true;
    bool log_project_events = true;

    // Callbacks
    bool trace_callbacks = true;
    bool log_callback_registration = true;
    bool log_callback_invocation = true;
    bool log_callback_timing = true;
};
```

---

### 11. UI Debug 🖥️

ImGui state and rendering.

```cpp
struct UIDebugConfig {
    bool enabled = false;

    // Rendering
    bool log_frame_timing = true;
    bool log_draw_calls = true;
    bool log_vertex_count = true;
    bool log_texture_uploads = true;

    // Layout
    bool log_window_layout = true;
    bool log_docking_state = true;
    bool log_panel_visibility = true;

    // Widgets
    bool trace_widget_interactions = true;
    bool log_button_clicks = true;
    bool log_input_changes = true;
    bool log_selection_changes = true;

    // State
    bool log_imgui_state = true;
    bool log_focus_changes = true;
    bool log_modal_dialogs = true;

    // Performance
    bool log_slow_frames = true;
    float slow_frame_threshold_ms = 16.67f; // 60 FPS
};
```

---

### 12. System Debug 🖧

OS, hardware, environment.

```cpp
struct SystemDebugConfig {
    bool enabled = false;

    // Environment
    bool capture_environment = true;
    bool log_env_variables = true;
    bool log_working_directory = true;
    bool log_command_line = true;

    // Hardware
    bool log_cpu_info = true;
    bool log_gpu_info = true;
    bool log_memory_info = true;
    bool log_disk_info = true;

    // OS
    bool log_os_version = true;
    bool log_system_locale = true;
    bool log_timezone = true;

    // Performance
    bool monitor_cpu_usage = true;
    bool monitor_gpu_usage = true;
    bool monitor_disk_io = true;
    bool monitor_network_io = true;

    // Versions
    bool log_library_versions = true;
    bool log_build_info = true;
    bool log_git_commit = true;
};
```

---

## Console Integration

The debug system should be controlled from the **Console Panel** with a dedicated debug tab.

### Console Debug Tab UI

```
┌─────────────────────────────────────────────────────────────────┐
│ [Console] [Output] [Debug] [Python REPL]                        │
├─────────────────────────────────────────────────────────────────┤
│ ┌─ Debug Control ─────────────────────────────────────────────┐ │
│ │ [●] Master Debug    Level: [▼ DETAILED]    [Save Config]   │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│ ┌─ Subsystems ────────────────────────────────────────────────┐ │
│ │ [✓] Training      [✓] Data Pipeline   [○] GPU              │ │
│ │ [✓] Memory        [○] Network         [○] Script           │ │
│ │ [○] Node Graph    [○] Threading       [○] File I/O         │ │
│ │ [○] Events        [○] UI              [✓] System           │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│ ┌─ Quick Actions ─────────────────────────────────────────────┐ │
│ │ [Capture State] [Compare Runs] [Export Trace] [Clear Logs] │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│ ┌─ Live Log ──────────────────────────────────────────────────┐ │
│ │ Filter: [All ▼] Search: [_____________] [Pause] [Auto-scroll]│ │
│ │ ─────────────────────────────────────────────────────────────│ │
│ │ [10:23:45] [TRAIN] Epoch 1, Batch 42                         │ │
│ │ [10:23:45] [TRAIN] Forward pass: conv1 → relu1 → pool1       │ │
│ │ [10:23:45] [TRAIN] Loss: 2.3456 (↓ 2.3%)                     │ │
│ │ [10:23:45] [MEM]   GPU: 4.2 GB / 24 GB                       │ │
│ │ [10:23:46] [DATA]  Loaded batch 43 (2.1ms)                   │ │
│ │ [10:23:46] [TRAIN] Epoch 1, Batch 43                         │ │
│ │ ⚠ [10:23:46] [TRAIN] Warning: Gradient norm > 100            │ │
│ │ ...                                                          │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Debug Commands (Python REPL)

```python
# Enable/disable debug subsystems
debug.enable("training")
debug.disable("network")
debug.set_level("VERBOSE")

# Quick toggles
debug.training(True)
debug.memory(True)
debug.gpu(True)

# Capture state for comparison
state = debug.capture_state()
debug.save_state("checkpoint_batch_42.dbg")

# Compare two runs
diff = debug.compare("run_pc1.trace", "run_pc2.trace")
diff.first_divergence()  # Find where results differ

# Export trace
debug.export_trace("training_trace.json")

# View specific subsystem config
debug.config.training.log_gradients = True
debug.config.memory.detect_leaks = True
```

---

## Trace File Format

### JSON Trace Format (for cross-machine comparison)

```json
{
  "version": "1.0",
  "timestamp": "2024-01-15T10:23:45.123Z",
  "environment": {
    "platform": "Windows 11",
    "cpu": "AMD Ryzen 9 7950X",
    "gpu": "NVIDIA RTX 4090",
    "cuda": "12.3",
    "arrayfire": "3.8.3",
    "cyxwiz": "0.2.0"
  },
  "training": {
    "seed": 42,
    "deterministic": true
  },
  "trace": [
    {
      "step": 0,
      "type": "forward",
      "layer": "conv1",
      "input_hash": "a3f2b7c9...",
      "output_hash": "b4e3c8d2...",
      "output_stats": {
        "min": -2.34,
        "max": 4.56,
        "mean": 0.12,
        "std": 0.89
      },
      "time_ms": 0.234
    },
    {
      "step": 0,
      "type": "backward",
      "layer": "conv1",
      "gradient_hash": "c5f4d9e3...",
      "gradient_stats": {
        "norm": 0.067,
        "max": 0.012
      }
    }
  ]
}
```

---

## Implementation Priority

### Phase 1: Core (Essential)
1. DebugController singleton
2. Training debug (forward, backward, loss)
3. Console debug tab UI
4. Basic logging

### Phase 2: Reproducibility (Critical for your issue)
5. Environment capture
6. Seed management
7. Deterministic mode
8. State save/restore
9. Trace file export

### Phase 3: Comparison Tools
10. Trace diff tool
11. Cross-machine comparison
12. Divergence detection

### Phase 4: Extended Debugging
13. Memory debug
14. GPU debug
15. Network debug
16. All other subsystems

---

## Questions for Discussion

1. **Highest Priority Subsystems?**
   - Training + Reproducibility seem most critical for your cross-PC issue

2. **Console Tab Layout?**
   - Proposed layout above acceptable?
   - Any additional quick actions needed?

3. **Trace Storage?**
   - JSON for portability, binary for performance, or both?

4. **Deterministic Mode Default?**
   - Enable by default in debug? (slower but reproducible)

5. **Remote Comparison?**
   - Upload traces to cloud for comparison?
   - Or local-only diff tool?

6. **Integration with Existing Panels?**
   - Properties panel show debug info for selected nodes?
   - PlotWindow show debug metrics?

7. **Performance Overhead Budget?**
   - Acceptable slowdown at each debug level?
   - Level 2: 10%? Level 3: 50%? Level 4: 5x?

---

## File Structure

```
cyxwiz-backend/
├── include/cyxwiz/debug/
│   ├── debug.h                  # Main debug header
│   ├── debug_controller.h       # Singleton controller
│   ├── debug_config.h           # All config structs
│   ├── subsystems/
│   │   ├── training_debug.h
│   │   ├── data_debug.h
│   │   ├── gpu_debug.h
│   │   ├── memory_debug.h
│   │   ├── network_debug.h
│   │   ├── script_debug.h
│   │   ├── graph_debug.h
│   │   ├── threading_debug.h
│   │   ├── file_debug.h
│   │   ├── event_debug.h
│   │   ├── ui_debug.h
│   │   └── system_debug.h
│   ├── reproducibility.h        # Seed/state management
│   ├── numerical_checker.h      # NaN/Inf detection
│   ├── environment_capture.h    # System info
│   └── diff_tool.h              # Run comparison
│
├── src/debug/
│   └── [implementations]

cyxwiz-engine/
├── src/gui/panels/
│   └── debug_panel.h/cpp        # Console debug tab
```

---

## Next Steps

1. Confirm architecture and priorities
2. Start with Phase 1: Core infrastructure
3. Build debug tab in Console panel
4. Test with your cross-PC scenario

**What are your thoughts? What should we adjust or add?**
 Questions for you:

  1. Any additional subsystems or features needed?
  2. Should deterministic mode be default when debug is on?
  3. Prefer cloud-based trace comparison or local-only diff tool?
  4. What's acceptable performance overhead per debug level?
  5. Ready to start implementation, or want to discuss more?