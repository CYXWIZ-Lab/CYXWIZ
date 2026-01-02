#!/bin/bash
#
# CyxWiz Distributed Training Launcher
# =====================================
#
# This script launches multiple processes for distributed training
# on a single machine with multiple GPUs.
#
# Usage:
#   ./launch_distributed.sh <num_gpus> <command> [args...]
#
# Examples:
#   # Launch with 4 GPUs
#   ./launch_distributed.sh 4 python distributed_training_example.py
#
#   # Launch with 2 GPUs and custom arguments
#   ./launch_distributed.sh 2 python train.py --epochs 100 --batch-size 64
#
#   # Launch with specific backend
#   DISTRIBUTED_BACKEND=nccl ./launch_distributed.sh 4 python train.py
#

set -e

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <num_gpus> <command> [args...]"
    echo ""
    echo "Examples:"
    echo "  $0 4 python distributed_training_example.py"
    echo "  $0 2 python train.py --epochs 100"
    echo ""
    echo "Environment variables:"
    echo "  MASTER_ADDR    - IP address of rank 0 (default: 127.0.0.1)"
    echo "  MASTER_PORT    - Port for communication (default: 29500)"
    echo "  DISTRIBUTED_BACKEND - 'nccl' or 'cpu' (default: cpu)"
    exit 1
fi

NUM_GPUS=$1
shift
COMMAND="$@"

# Default values
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
DISTRIBUTED_BACKEND=${DISTRIBUTED_BACKEND:-cpu}

echo "=============================================="
echo "CyxWiz Distributed Training Launcher"
echo "=============================================="
echo "World size:  $NUM_GPUS"
echo "Master:      $MASTER_ADDR:$MASTER_PORT"
echo "Backend:     $DISTRIBUTED_BACKEND"
echo "Command:     $COMMAND"
echo "=============================================="
echo ""

# Array to store PIDs
PIDS=()

# Cleanup function
cleanup() {
    echo ""
    echo "Stopping all processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            kill $pid 2>/dev/null || true
        fi
    done
    wait
    echo "All processes stopped."
}

# Set up trap for cleanup
trap cleanup EXIT INT TERM

# Launch processes
for ((RANK=0; RANK<NUM_GPUS; RANK++)); do
    echo "Launching rank $RANK..."

    RANK=$RANK \
    LOCAL_RANK=$RANK \
    WORLD_SIZE=$NUM_GPUS \
    MASTER_ADDR=$MASTER_ADDR \
    MASTER_PORT=$MASTER_PORT \
    DISTRIBUTED_BACKEND=$DISTRIBUTED_BACKEND \
    $COMMAND &

    PIDS+=($!)

    # Small delay between launches to avoid race conditions
    sleep 0.5
done

echo ""
echo "All $NUM_GPUS processes launched."
echo "PIDs: ${PIDS[*]}"
echo ""
echo "Waiting for processes to complete..."
echo "(Press Ctrl+C to stop all processes)"
echo ""

# Wait for all processes
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait $pid; then
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "WARNING: $FAILED process(es) exited with errors"
    exit 1
else
    echo ""
    echo "All processes completed successfully!"
fi
