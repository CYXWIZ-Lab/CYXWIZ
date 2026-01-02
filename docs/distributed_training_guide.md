# CyxWiz Distributed Training Guide

This guide explains how to use CyxWiz's distributed training capabilities for data-parallel training across multiple GPUs or machines.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Python API Reference](#python-api-reference)
5. [Configuration](#configuration)
6. [Launching Distributed Training](#launching-distributed-training)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Overview

CyxWiz supports **Data Parallel** distributed training, where:
- The same model is replicated across multiple GPUs/machines
- Each replica processes a different subset of the data
- Gradients are synchronized (averaged) across all replicas after each backward pass
- All replicas have identical model parameters after each update

### Supported Backends

| Backend | Description | Platform | Performance |
|---------|-------------|----------|-------------|
| **CPU** | TCP socket-based ring all-reduce | All platforms | Moderate |
| **NCCL** | NVIDIA Collective Communications Library | Linux + NVIDIA GPUs | High |

### Key Benefits

- **Linear scaling**: Training speed scales nearly linearly with the number of GPUs
- **Larger effective batch size**: batch_size × world_size
- **Fault tolerance**: Training can continue if a node fails (with checkpointing)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CyxWiz Distributed Training                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│   │   Rank 0     │    │   Rank 1     │    │   Rank 2     │    ...           │
│   │   (Master)   │    │   (Worker)   │    │   (Worker)   │                  │
│   ├──────────────┤    ├──────────────┤    ├──────────────┤                  │
│   │ Model Copy   │    │ Model Copy   │    │ Model Copy   │                  │
│   │ Data Shard 0 │    │ Data Shard 1 │    │ Data Shard 2 │                  │
│   │ GPU 0        │    │ GPU 1        │    │ GPU 2        │                  │
│   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                  │
│          │                   │                   │                           │
│          └───────────────────┼───────────────────┘                           │
│                              │                                               │
│                     ┌────────▼────────┐                                      │
│                     │  AllReduce      │  ← Gradient Synchronization          │
│                     │  (NCCL or TCP)  │                                      │
│                     └─────────────────┘                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Training Flow

1. **Forward Pass**: Each rank processes its local batch
2. **Loss Computation**: Each rank computes loss on its batch
3. **Backward Pass**: Each rank computes gradients locally
4. **AllReduce**: Gradients are averaged across all ranks
5. **Parameter Update**: All ranks update parameters identically

---

## Quick Start

### 1. Set Environment Variables

```bash
export RANK=0              # This process's rank (0, 1, 2, ...)
export WORLD_SIZE=4        # Total number of processes
export LOCAL_RANK=0        # GPU index on this machine
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
```

### 2. Write Training Script

```python
import pycyxwiz as cx

# Initialize distributed
cx.distributed.init()

# Create model
model = cx.SequentialModel()
model.add(cx.LinearModule(784, 256))
model.add(cx.ReLUModule())
model.add(cx.LinearModule(256, 10))

# Create trainer
trainer = cx.distributed.DistributedTrainer(
    model,
    cx.CrossEntropyLoss(),
    cx.AdamOptimizer(0.001)
)

# Configure and train
config = cx.distributed.DistributedTrainingConfig()
config.epochs = 10
config.batch_size = 64  # Per-GPU batch size

history = trainer.fit(X_train, y_train, config)

# Save model (master only)
if trainer.is_master():
    model.save("model.cyxmodel")

cx.distributed.finalize()
```

### 3. Launch Training

```bash
# Using the launcher script
./launch_distributed.sh 4 python train.py

# Or manually (run each in a separate terminal)
RANK=0 WORLD_SIZE=4 python train.py &
RANK=1 WORLD_SIZE=4 python train.py &
RANK=2 WORLD_SIZE=4 python train.py &
RANK=3 WORLD_SIZE=4 python train.py &
```

---

## Python API Reference

### Initialization Functions

```python
import pycyxwiz as cx

# Initialize from environment variables
cx.distributed.init()

# Initialize with explicit config
config = cx.distributed.DistributedConfig()
config.rank = 0
config.world_size = 4
cx.distributed.init(config)

# Query state
cx.distributed.is_initialized()  # -> bool
cx.distributed.get_rank()        # -> int
cx.distributed.get_world_size()  # -> int
cx.distributed.get_local_rank()  # -> int
cx.distributed.is_master()       # -> bool (True if rank == 0)

# Cleanup
cx.distributed.finalize()
```

### DistributedConfig

```python
config = cx.distributed.DistributedConfig()
config.rank = 0                    # Global rank
config.world_size = 4              # Total processes
config.local_rank = 0              # GPU index on this machine
config.master_addr = "127.0.0.1"   # IP of rank 0
config.master_port = 29500         # Communication port
config.backend = cx.distributed.BackendType.CPU  # or NCCL
config.timeout_ms = 30000          # Connection timeout

# Create from environment
config = cx.distributed.DistributedConfig.from_environment()
```

### DistributedDataParallel (DDP)

```python
# Configuration
ddp_config = cx.distributed.DDPConfig()
ddp_config.broadcast_parameters = True   # Sync params from rank 0
ddp_config.bucket_size_mb = 25           # Gradient bucket size
ddp_config.find_unused_parameters = False

# Wrap model
ddp = cx.distributed.DistributedDataParallel(model, ddp_config)

# Training methods
output = ddp.forward(input_tensor)
grad_output = ddp.backward(grad_tensor)
synced_grads = ddp.sync_gradients()      # AllReduce gradients
ddp.update_parameters(optimizer)          # Sync + update

# Info methods
ddp.is_master()       # -> bool
ddp.get_rank()        # -> int
ddp.get_world_size()  # -> int
ddp.get_model()       # -> SequentialModel
```

### DistributedSampler

```python
# Create sampler
sampler = cx.distributed.DistributedSampler(
    dataset_size=10000,
    shuffle=True,
    seed=42,
    drop_last=False
)

# Per-epoch methods
sampler.set_epoch(epoch)     # IMPORTANT: call at start of each epoch
indices = sampler.get_indices()  # Get this rank's indices

# Info methods
sampler.local_size()    # Samples for this rank
sampler.total_size()    # Total dataset size
sampler.padded_size()   # Padded size (divisible by world_size)
```

### DistributedBatchIterator

```python
# Create iterator
batch_iter = cx.distributed.DistributedBatchIterator(sampler, batch_size=64)

# Iterate
while batch_iter.has_next():
    indices = batch_iter.next()  # -> list of indices
    # Use indices to get batch from dataset

# Reset for new epoch
batch_iter.reset(epoch=1)

# Info
batch_iter.num_batches()  # Total batches per epoch
```

### DistributedTrainer (High-Level API)

```python
# Create trainer
trainer = cx.distributed.DistributedTrainer(model, loss_fn, optimizer)

# Configure
config = cx.distributed.DistributedTrainingConfig()
config.epochs = 10
config.batch_size = 64              # Per-GPU
config.shuffle = True
config.seed = 42
config.verbose = True
config.log_every_n_batches = 10
config.checkpoint_every_n_epochs = 5
config.checkpoint_dir = "./checkpoints"
config.save_on_master_only = True
config.validation_split = 0.1

# Train
history = trainer.fit(X_train, y_train, config)
# Or with validation data:
history = trainer.fit(X_train, y_train, X_val, y_val, config)

# Evaluate
loss, accuracy = trainer.evaluate(X_test, y_test)

# Checkpointing
trainer.save_checkpoint("model.ckpt")
trainer.load_checkpoint("model.ckpt")

# Info
trainer.is_master()
trainer.get_rank()
trainer.get_world_size()
```

### DistributedTrainingHistory

```python
history = trainer.fit(...)

history.train_losses        # List[float] - per epoch
history.train_accuracies    # List[float] - per epoch
history.val_losses          # List[float] - per epoch
history.val_accuracies      # List[float] - per epoch
history.total_time_seconds  # float
history.samples_per_second  # float - throughput
history.effective_batch_size  # int - batch_size * world_size
history.world_size          # int
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RANK` | Global rank of this process | Required |
| `WORLD_SIZE` | Total number of processes | Required |
| `LOCAL_RANK` | GPU index on this machine | Same as RANK |
| `MASTER_ADDR` | IP address of rank 0 | 127.0.0.1 |
| `MASTER_PORT` | Communication port | 29500 |
| `DISTRIBUTED_BACKEND` | "nccl" or "cpu" | cpu |
| `DISTRIBUTED_TIMEOUT_MS` | Connection timeout | 30000 |

### Backend Selection

```python
# CPU Backend (works everywhere)
config.backend = cx.distributed.BackendType.CPU

# NCCL Backend (Linux + NVIDIA GPUs, fastest)
config.backend = cx.distributed.BackendType.NCCL
```

---

## Launching Distributed Training

### Single Machine, Multiple GPUs

**Using Launcher Script (Linux/Mac):**
```bash
./launch_distributed.sh 4 python train.py
```

**Using Launcher Script (Windows):**
```batch
launch_distributed.bat 4 train.py
```

**Manual Launch:**
```bash
# Terminal 1
RANK=0 LOCAL_RANK=0 WORLD_SIZE=4 MASTER_ADDR=127.0.0.1 python train.py

# Terminal 2
RANK=1 LOCAL_RANK=1 WORLD_SIZE=4 MASTER_ADDR=127.0.0.1 python train.py

# Terminal 3
RANK=2 LOCAL_RANK=2 WORLD_SIZE=4 MASTER_ADDR=127.0.0.1 python train.py

# Terminal 4
RANK=3 LOCAL_RANK=3 WORLD_SIZE=4 MASTER_ADDR=127.0.0.1 python train.py
```

### Multi-Machine Cluster

```bash
# Machine 1 (192.168.1.100) - 2 GPUs
RANK=0 LOCAL_RANK=0 WORLD_SIZE=4 MASTER_ADDR=192.168.1.100 python train.py &
RANK=1 LOCAL_RANK=1 WORLD_SIZE=4 MASTER_ADDR=192.168.1.100 python train.py &

# Machine 2 (192.168.1.101) - 2 GPUs
RANK=2 LOCAL_RANK=0 WORLD_SIZE=4 MASTER_ADDR=192.168.1.100 python train.py &
RANK=3 LOCAL_RANK=1 WORLD_SIZE=4 MASTER_ADDR=192.168.1.100 python train.py &
```

---

## Best Practices

### 1. Always Set Epoch in Sampler

```python
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # Ensures different shuffling each epoch
    # ... training loop
```

### 2. Only Save/Log on Master

```python
if trainer.is_master():
    print(f"Epoch {epoch}: Loss = {loss}")
    model.save("checkpoint.ckpt")
```

### 3. Scale Learning Rate with Batch Size

```python
# Linear scaling rule
base_lr = 0.001
base_batch_size = 64
effective_batch_size = batch_size * world_size
scaled_lr = base_lr * (effective_batch_size / base_batch_size)
```

### 4. Use Gradient Accumulation for Large Models

```python
accumulation_steps = 4
for i, batch in enumerate(batches):
    output = ddp.forward(batch)
    loss = loss_fn.forward(output, target) / accumulation_steps
    ddp.backward(loss_fn.backward(...))

    if (i + 1) % accumulation_steps == 0:
        ddp.sync_gradients()
        ddp.update_parameters(optimizer)
```

### 5. Handle Uneven Data Sizes

```python
# Use drop_last=True for consistent batch sizes
sampler = cx.distributed.DistributedSampler(
    dataset_size=len(dataset),
    drop_last=True  # Ensures all ranks have same number of batches
)
```

---

## Troubleshooting

### Connection Refused

```
Error: Connection refused to 127.0.0.1:29500
```

**Solutions:**
- Ensure rank 0 starts first
- Check firewall settings
- Verify MASTER_ADDR is correct
- Try a different port

### Timeout During Initialization

```
Error: Timeout waiting for peers
```

**Solutions:**
- Increase `timeout_ms` in config
- Check network connectivity between machines
- Ensure all processes are started

### NCCL Errors

```
Error: NCCL error: unhandled cuda error
```

**Solutions:**
- Ensure CUDA drivers are installed
- Check GPU availability with `nvidia-smi`
- Verify LOCAL_RANK matches available GPU indices
- Try CPU backend as fallback

### Gradient Mismatch

```
Warning: Gradients differ across ranks
```

**Solutions:**
- Ensure same model architecture on all ranks
- Verify `broadcast_parameters=True` in DDPConfig
- Check that random seeds are consistent

### Out of Memory

**Solutions:**
- Reduce batch_size per GPU
- Use gradient accumulation
- Enable gradient checkpointing
- Use mixed precision training

---

## Example Files

- `distributed_training_example.py` - Comprehensive Python example
- `launch_distributed.sh` - Linux/Mac launcher script
- `launch_distributed.bat` - Windows launcher script

---

## See Also

- [CyxWiz Backend API Documentation](../cyxwiz-backend/README.md)
- [Model Serialization Guide](./model_serialization.md)
- [MNIST Training Demo](./mnist_demo.md)
