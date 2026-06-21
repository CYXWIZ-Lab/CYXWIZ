# Book to CyxWiz Map

This map is based on the PDF table of contents and initial repo inspection. It should be refined chapter by chapter as we read deeper.

## Highest-Value Chapters for CyxWiz

### Chapters 1-2: AI System Overview and Hardware

Relevant to:

- `cyxwiz-server-node/src/core/device_pool.*`
- `cyxwiz-server-node/src/core/metrics_collector.*`
- `cyxwiz-backend/include/cyxwiz/device.h`
- `cyxwiz-backend/src/core/device.cpp`

Production use:

- Define a real node capability model instead of a loose hardware report.
- Track GPU type, memory, compute backend, driver/runtime compatibility, thermal state, and sustained throughput.
- Introduce "goodput" as a core metric: accepted jobs completed within expected latency/quality constraints, not just GPU utilization.

Immediate tasks:

- Add a node capability schema used by registration, scheduling, and job admission.
- Add benchmark IDs and environment fingerprints to every performance result.
- Separate raw utilization metrics from useful work metrics.

### Chapter 3: OS, Docker, and Kubernetes Tuning for GPU Environments

Relevant to:

- `cyxwiz-server-node/resources/config/server_config.yaml`
- node setup documentation and deployment packaging
- future Linux production node images

Production use:

- Create host readiness checks for CUDA runtime, GPU driver, CPU affinity, memory limits, filesystem behavior, and container runtime.
- Treat Kubernetes and container tuning as deployment-layer concerns, not core backend logic.

Immediate tasks:

- Add a `cyxwiz-server-node --doctor` or equivalent diagnostics command.
- Document minimum GPU node runtime requirements.
- Record CUDA, driver, ArrayFire, ONNX Runtime, llama.cpp, and LibTorch versions in metrics.

### Chapter 4: Distributed Networking Communication

Relevant to:

- `cyxwiz-server-node/src/node_client.*`
- `cyxwiz-server-node/src/deployment_handler.*`
- `cyxwiz-server-node/src/job_execution_service.*`
- central server scheduling protocol

Production use:

- CyxWiz should measure network latency, bandwidth, heartbeat health, and job transfer time before adding advanced distributed GPU communication.
- NCCL/NIXL-style optimizations are future work for multi-GPU and disaggregated inference, not day-one core requirements.

Immediate tasks:

- Add network and transfer timing metrics per deployment.
- Add backpressure and admission control when the node cannot safely accept more work.
- Make heartbeat state precise: healthy, degraded, draining, offline, and recovering.

### Chapter 5: GPU-Based Storage I/O

Relevant to:

- model loading in `cyxwiz-server-node/src/model_loader.*`
- dataset loading in `cyxwiz-backend`
- checkpointing and artifact transfer

Production use:

- Model and dataset I/O must be measured separately from compute.
- Loading, caching, and checkpointing should have explicit budgets and failure handling.

Immediate tasks:

- Add model load time, model size, cache hit/miss, and artifact write timing metrics.
- Define a local artifact cache policy.
- Add checksum or content-addressed validation for downloaded model/job artifacts.

### Chapters 13-14: PyTorch Distributed, Compiler, Triton, XLA

Relevant to:

- future PyTorch loader support in server node
- backend performance benchmarking
- optional optimized kernels

Production use:

- These chapters are useful for future compatibility and performance work, but CyxWiz backend currently uses ArrayFire and C++ abstractions.
- Avoid adding compiler or Triton concepts into the backend core until benchmarks prove a bottleneck that ArrayFire cannot solve.

Immediate tasks:

- Add benchmark fixtures for tensor operations, layers, training loops, and inference paths.
- Keep custom kernels behind narrow backend-specific interfaces.

### Chapters 15-19: Inference Serving, Batching, Scheduling, KV Cache, Dynamic Optimization

Relevant to:

- `cyxwiz-server-node/src/deployment_manager.*`
- `cyxwiz-server-node/src/model_loader.*`
- `cyxwiz-server-node/src/core/metrics_storage.*`
- backend TODO item: inference runtime

Production use:

- This is the most important section for production server-node behavior.
- CyxWiz needs an inference service model with admission control, batching policy, timeouts, streaming response support, memory limits, and per-model SLO tracking.
- Advanced prefill/decode disaggregation and KV cache tuning should be later-stage work for LLM-specific deployments.

Immediate tasks:

- Define deployment state machine and SLO fields.
- Add per-model concurrency limits.
- Add request queue metrics and timeout behavior.
- Add optional batching policy for compatible models.
- Add memory budgeting before model load and request execution.

### Chapter 16: Profiling, Debugging, and Tuning Inference at Scale

Relevant to almost all production work.

Production use:

- Establish observability before large optimization changes.
- Track latency percentiles, throughput, failure causes, queue time, model load time, GPU memory, CPU memory, and thermal/power data where available.

Immediate tasks:

- Add structured metrics exports.
- Add benchmark baseline files checked into docs or CI artifacts.
- Create regression thresholds for key operations.

### Chapter 20 and Appendix: AI-Assisted Optimization and Checklist

Relevant to:

- future automation
- production readiness reviews
- CI performance gates

Production use:

- Use the checklist as a review input, not as a feature backlog.
- AI-assisted optimization should operate on measured bottlenecks only.

Immediate tasks:

- Convert applicable checklist items into CyxWiz production review questions.
- Build a recurring review doc after each implemented production milestone.

## Non-Goals for the First Production Pass

- Full Kubernetes operator.
- NCCL or NIXL integration.
- Custom Triton kernels.
- Disaggregated prefill/decode architecture.
- Runtime RL agents for optimization.
- Broad plugin hooks into backend internals.

These may become valuable, but only after CyxWiz has stable benchmarks, observability, and production job lifecycle guarantees.

