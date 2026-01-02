#!/usr/bin/env python3
"""
CyxWiz Distributed Training Example
====================================

This script demonstrates how to use CyxWiz's distributed training capabilities
for data-parallel training across multiple GPUs or machines.

Requirements:
    - pycyxwiz library built and installed
    - For multi-GPU: NVIDIA GPUs with CUDA support
    - For NCCL backend: NCCL library installed (Linux)

Environment Variables (set before running):
    RANK        - Global rank of this process (0, 1, 2, ...)
    WORLD_SIZE  - Total number of processes
    LOCAL_RANK  - GPU index on this machine (usually same as RANK for single machine)
    MASTER_ADDR - IP address of rank 0 (default: 127.0.0.1)
    MASTER_PORT - Port for communication (default: 29500)
    DISTRIBUTED_BACKEND - "nccl" (GPU) or "cpu" (TCP sockets)

Usage:
    # Single machine, 4 GPUs (run in 4 terminals or use launcher script)
    RANK=0 LOCAL_RANK=0 WORLD_SIZE=4 python distributed_training_example.py
    RANK=1 LOCAL_RANK=1 WORLD_SIZE=4 python distributed_training_example.py
    RANK=2 LOCAL_RANK=2 WORLD_SIZE=4 python distributed_training_example.py
    RANK=3 LOCAL_RANK=3 WORLD_SIZE=4 python distributed_training_example.py

    # Or use the launcher script (see launch_distributed.sh below)
    ./launch_distributed.sh 4 python distributed_training_example.py

Author: CyxWiz Team
"""

import os
import sys
import time
import numpy as np

# Add pycyxwiz to path if running from build directory
script_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(script_dir, '..', 'build', 'bin', 'Release')
if os.path.exists(build_dir):
    # Add DLL directory for Windows
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(build_dir)
        # Add ArrayFire if available
        af_dir = r"C:\Program Files\ArrayFire\v3\lib"
        if os.path.exists(af_dir):
            os.add_dll_directory(af_dir)
    sys.path.insert(0, build_dir)

try:
    import pycyxwiz as cx
except ImportError as e:
    print(f"Error importing pycyxwiz: {e}")
    print("Make sure pycyxwiz is built and in your Python path.")
    print(f"Tried: {build_dir}")
    sys.exit(1)


def generate_synthetic_data(num_samples: int, input_dim: int, num_classes: int, seed: int = 42):
    """Generate synthetic classification data for demonstration."""
    np.random.seed(seed)

    # Generate random features
    X = np.random.randn(num_samples, input_dim).astype(np.float32)

    # Generate random labels
    y = np.random.randint(0, num_classes, size=num_samples).astype(np.int64)

    # Convert to one-hot for training
    y_onehot = np.zeros((num_samples, num_classes), dtype=np.float32)
    y_onehot[np.arange(num_samples), y] = 1.0

    return X, y_onehot, y


def print_rank(msg: str):
    """Print message with rank prefix (only useful info, not spam)."""
    rank = cx.distributed.get_rank() if cx.distributed.is_initialized() else -1
    print(f"[Rank {rank}] {msg}")


# =============================================================================
# Example 1: High-Level API (Recommended for most users)
# =============================================================================

def example_high_level_api():
    """
    High-level distributed training using DistributedTrainer.

    This is the recommended approach for most use cases.
    The trainer handles all the complexity of distributed training:
    - Data sharding across ranks
    - Gradient synchronization
    - Metric aggregation
    - Checkpointing
    """
    print("\n" + "="*60)
    print("Example 1: High-Level API (DistributedTrainer)")
    print("="*60 + "\n")

    # Initialize distributed training
    # This reads RANK, WORLD_SIZE, etc. from environment variables
    if not cx.distributed.init():
        print("Failed to initialize distributed training")
        print("Make sure RANK and WORLD_SIZE environment variables are set")
        return

    rank = cx.distributed.get_rank()
    world_size = cx.distributed.get_world_size()
    is_master = cx.distributed.is_master()

    print_rank(f"Initialized: rank {rank}/{world_size}")

    # Generate synthetic data (same on all ranks for this example)
    # In real use, you'd load your actual dataset
    num_samples = 10000
    input_dim = 784  # e.g., flattened 28x28 MNIST images
    num_classes = 10

    X_train, y_train, _ = generate_synthetic_data(num_samples, input_dim, num_classes)
    X_val, y_val, _ = generate_synthetic_data(2000, input_dim, num_classes, seed=123)

    if is_master:
        print(f"Dataset: {num_samples} training samples, {X_val.shape[0]} validation samples")
        print(f"Input dim: {input_dim}, Classes: {num_classes}")

    # Create model
    model = cx.SequentialModel()
    model.add(cx.LinearModule(input_dim, 512))
    model.add(cx.ReLUModule())
    model.add(cx.LinearModule(512, 256))
    model.add(cx.ReLUModule())
    model.add(cx.LinearModule(256, num_classes))

    if is_master:
        print(f"Model: {input_dim} -> 512 -> 256 -> {num_classes}")

    # Create loss and optimizer
    loss_fn = cx.CrossEntropyLoss()
    optimizer = cx.AdamOptimizer(learning_rate=0.001)

    # Create distributed trainer
    trainer = cx.distributed.DistributedTrainer(model, loss_fn, optimizer)

    # Configure training
    config = cx.distributed.DistributedTrainingConfig()
    config.epochs = 5
    config.batch_size = 64  # Per-GPU batch size
    config.shuffle = True
    config.seed = 42
    config.verbose = True
    config.log_every_n_batches = 50
    config.validation_split = 0.0  # We provide separate validation data
    config.save_on_master_only = True

    effective_batch_size = config.batch_size * world_size
    if is_master:
        print(f"Training config:")
        print(f"  - Epochs: {config.epochs}")
        print(f"  - Batch size per GPU: {config.batch_size}")
        print(f"  - Effective batch size: {effective_batch_size}")
        print(f"  - World size: {world_size}")

    # Convert numpy arrays to CyxWiz tensors
    X_train_tensor = cx.Tensor(X_train.flatten().tolist(), list(X_train.shape))
    y_train_tensor = cx.Tensor(y_train.flatten().tolist(), list(y_train.shape))
    X_val_tensor = cx.Tensor(X_val.flatten().tolist(), list(X_val.shape))
    y_val_tensor = cx.Tensor(y_val.flatten().tolist(), list(y_val.shape))

    # Train!
    print_rank("Starting training...")
    start_time = time.time()

    history = trainer.fit(X_train_tensor, y_train_tensor,
                         X_val_tensor, y_val_tensor, config)

    elapsed = time.time() - start_time

    # Print results (master only to avoid duplicate output)
    if is_master:
        print("\n" + "-"*40)
        print("Training Complete!")
        print("-"*40)
        print(f"Total time: {elapsed:.2f}s")
        print(f"Throughput: {history.samples_per_second:.0f} samples/sec")
        print(f"Effective batch size: {history.effective_batch_size}")

        if history.train_losses:
            print(f"Final train loss: {history.train_losses[-1]:.4f}")
        if history.val_losses:
            print(f"Final val loss: {history.val_losses[-1]:.4f}")
        if history.train_accuracies:
            print(f"Final train accuracy: {history.train_accuracies[-1]:.2%}")
        if history.val_accuracies:
            print(f"Final val accuracy: {history.val_accuracies[-1]:.2%}")

    # Cleanup
    cx.distributed.finalize()
    print_rank("Finalized")


# =============================================================================
# Example 2: Low-Level API (For advanced users who need fine control)
# =============================================================================

def example_low_level_api():
    """
    Low-level distributed training with manual control.

    Use this when you need:
    - Custom training loops
    - Fine-grained control over gradient synchronization
    - Custom data loading logic
    - Integration with existing training code
    """
    print("\n" + "="*60)
    print("Example 2: Low-Level API (Manual Control)")
    print("="*60 + "\n")

    # Manual configuration (instead of reading from environment)
    config = cx.distributed.DistributedConfig()
    config.rank = int(os.environ.get('RANK', 0))
    config.world_size = int(os.environ.get('WORLD_SIZE', 1))
    config.local_rank = int(os.environ.get('LOCAL_RANK', config.rank))
    config.master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    config.master_port = int(os.environ.get('MASTER_PORT', 29500))
    config.backend = cx.distributed.BackendType.CPU  # Use CPU backend for compatibility

    print(f"Config: rank={config.rank}, world_size={config.world_size}, "
          f"master={config.master_addr}:{config.master_port}")

    # Initialize
    if not cx.distributed.init(config):
        print("Failed to initialize")
        return

    rank = cx.distributed.get_rank()
    world_size = cx.distributed.get_world_size()
    is_master = rank == 0

    # Generate data
    num_samples = 5000
    input_dim = 784
    num_classes = 10
    X_train, y_train, _ = generate_synthetic_data(num_samples, input_dim, num_classes)

    # Create model
    model = cx.SequentialModel()
    model.add(cx.LinearModule(input_dim, 256))
    model.add(cx.ReLUModule())
    model.add(cx.LinearModule(256, num_classes))

    # Wrap model in DDP
    ddp_config = cx.distributed.DDPConfig()
    ddp_config.broadcast_parameters = True  # Sync parameters from rank 0
    ddp_config.bucket_size_mb = 25

    ddp = cx.distributed.DistributedDataParallel(model, ddp_config)
    print_rank(f"DDP initialized, is_master={ddp.is_master()}")

    # Create sampler for data sharding
    sampler = cx.distributed.DistributedSampler(
        dataset_size=num_samples,
        shuffle=True,
        seed=42,
        drop_last=False
    )

    print_rank(f"Sampler: total={num_samples}, local={sampler.local_size()}")

    # Setup training
    loss_fn = cx.CrossEntropyLoss()
    optimizer = cx.AdamOptimizer(learning_rate=0.001)

    batch_size = 32
    num_epochs = 3

    # Training loop
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Important for proper shuffling

        batch_iter = cx.distributed.DistributedBatchIterator(sampler, batch_size)

        epoch_loss = 0.0
        num_batches = 0

        while batch_iter.has_next():
            indices = batch_iter.next()

            # Get batch data
            X_batch = X_train[indices]
            y_batch = y_train[indices]

            # Convert to tensors
            X_tensor = cx.Tensor(X_batch.flatten().tolist(), list(X_batch.shape))
            y_tensor = cx.Tensor(y_batch.flatten().tolist(), list(y_batch.shape))

            # Forward pass
            output = ddp.forward(X_tensor)
            loss_val = loss_fn.forward(output, y_tensor)

            # Backward pass
            grad = loss_fn.backward(output, y_tensor)
            ddp.backward(grad)

            # Synchronize gradients (AllReduce across all ranks)
            ddp.sync_gradients()

            # Update parameters
            ddp.update_parameters(optimizer)

            # Accumulate loss (this is local loss, not synchronized)
            # In real code, you might want to all-reduce the loss too
            epoch_loss += 0.5  # Placeholder since we can't easily get loss value
            num_batches += 1

        # Print progress (master only)
        if is_master:
            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"Epoch {epoch+1}/{num_epochs}, Batches: {num_batches}, "
                  f"Avg Loss: {avg_loss:.4f}")

    # Save model (master only)
    if is_master:
        print("Training complete! (Model save would go here)")

    # Cleanup
    cx.distributed.finalize()
    print_rank("Finalized")


# =============================================================================
# Example 3: Using DistributedSampler standalone
# =============================================================================

def example_sampler_only():
    """
    Demonstrate DistributedSampler for data sharding.

    Useful when you want to integrate distributed data loading
    with your own training loop or framework.
    """
    print("\n" + "="*60)
    print("Example 3: DistributedSampler Standalone")
    print("="*60 + "\n")

    # Initialize distributed
    if not cx.distributed.init():
        print("Failed to initialize (make sure RANK and WORLD_SIZE are set)")
        return

    rank = cx.distributed.get_rank()
    world_size = cx.distributed.get_world_size()

    # Create sampler
    dataset_size = 100  # Small for demonstration
    sampler = cx.distributed.DistributedSampler(
        dataset_size=dataset_size,
        shuffle=True,
        seed=42,
        drop_last=False
    )

    print_rank(f"Dataset size: {dataset_size}")
    print_rank(f"Local size (samples for this rank): {sampler.local_size()}")
    print_rank(f"Padded size: {sampler.padded_size()}")

    # Get indices for this rank
    for epoch in range(2):
        sampler.set_epoch(epoch)
        indices = sampler.get_indices()

        print_rank(f"Epoch {epoch} indices (first 10): {indices[:10]}")
        print_rank(f"  Total indices for this rank: {len(indices)}")

    # Using batch iterator
    print_rank("\nUsing DistributedBatchIterator:")
    sampler.set_epoch(0)
    batch_iter = cx.distributed.DistributedBatchIterator(sampler, batch_size=16)

    batch_num = 0
    while batch_iter.has_next():
        batch_indices = batch_iter.next()
        if batch_num < 3:  # Only print first 3 batches
            print_rank(f"  Batch {batch_num}: {len(batch_indices)} samples, "
                      f"indices: {batch_indices[:5]}...")
        batch_num += 1

    print_rank(f"  Total batches: {batch_num}")

    cx.distributed.finalize()


# =============================================================================
# Example 4: Configuration Options
# =============================================================================

def example_configuration():
    """
    Demonstrate various configuration options.
    """
    print("\n" + "="*60)
    print("Example 4: Configuration Options")
    print("="*60 + "\n")

    # Method 1: From environment (recommended for production)
    print("Method 1: DistributedConfig.from_environment()")
    config1 = cx.distributed.DistributedConfig.from_environment()
    print(f"  rank={config1.rank}, world_size={config1.world_size}")
    print(f"  master={config1.master_addr}:{config1.master_port}")
    print(f"  backend={config1.backend}")

    # Method 2: Manual configuration
    print("\nMethod 2: Manual configuration")
    config2 = cx.distributed.DistributedConfig()
    config2.rank = 0
    config2.world_size = 4
    config2.local_rank = 0
    config2.master_addr = "192.168.1.100"
    config2.master_port = 29500
    config2.backend = cx.distributed.BackendType.CPU
    config2.timeout_ms = 60000  # 60 second timeout
    print(f"  rank={config2.rank}, world_size={config2.world_size}")
    print(f"  master={config2.master_addr}:{config2.master_port}")

    # DDP Configuration
    print("\nDDPConfig options:")
    ddp_config = cx.distributed.DDPConfig()
    print(f"  broadcast_parameters={ddp_config.broadcast_parameters}")
    print(f"  bucket_size_mb={ddp_config.bucket_size_mb}")
    print(f"  find_unused_parameters={ddp_config.find_unused_parameters}")

    # Training Configuration
    print("\nDistributedTrainingConfig options:")
    train_config = cx.distributed.DistributedTrainingConfig()
    print(f"  epochs={train_config.epochs}")
    print(f"  batch_size={train_config.batch_size}")
    print(f"  shuffle={train_config.shuffle}")
    print(f"  seed={train_config.seed}")
    print(f"  save_on_master_only={train_config.save_on_master_only}")
    print(f"  checkpoint_every_n_epochs={train_config.checkpoint_every_n_epochs}")
    print(f"  checkpoint_dir={train_config.checkpoint_dir}")
    print(f"  verbose={train_config.verbose}")
    print(f"  log_every_n_batches={train_config.log_every_n_batches}")
    print(f"  validation_split={train_config.validation_split}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the selected example."""
    print("="*60)
    print("CyxWiz Distributed Training Examples")
    print("="*60)

    # Check if distributed environment is set up
    rank = os.environ.get('RANK')
    world_size = os.environ.get('WORLD_SIZE')

    if rank is None or world_size is None:
        print("\nNo distributed environment detected.")
        print("Running configuration example only.\n")
        print("To run distributed examples, set environment variables:")
        print("  export RANK=0")
        print("  export WORLD_SIZE=2")
        print("  export LOCAL_RANK=0")
        print("  export MASTER_ADDR=127.0.0.1")
        print("  export MASTER_PORT=29500")
        print("\nOr use the launcher script:")
        print("  ./launch_distributed.sh 4 python distributed_training_example.py")
        print()
        example_configuration()
        return

    print(f"\nDistributed environment detected:")
    print(f"  RANK={rank}")
    print(f"  WORLD_SIZE={world_size}")
    print(f"  LOCAL_RANK={os.environ.get('LOCAL_RANK', 'not set')}")
    print(f"  MASTER_ADDR={os.environ.get('MASTER_ADDR', '127.0.0.1')}")
    print(f"  MASTER_PORT={os.environ.get('MASTER_PORT', '29500')}")

    # Run examples
    try:
        # Example 1: High-level API (most common use case)
        example_high_level_api()

        # Uncomment to run other examples:
        # example_low_level_api()
        # example_sampler_only()
        # example_configuration()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

        # Make sure to finalize even on error
        try:
            if cx.distributed.is_initialized():
                cx.distributed.finalize()
        except:
            pass


if __name__ == "__main__":
    main()
