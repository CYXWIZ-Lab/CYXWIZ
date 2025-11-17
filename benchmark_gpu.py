#!/usr/bin/env python3
"""
GPU Performance Benchmark - CyxWiz vs NumPy
Tests ArrayFire GPU acceleration performance
"""
import sys
import os
import time
import numpy as np

# Setup PATH for DLLs (Windows)
dll_dirs = [
    r"D:\Dev\CyxWiz_Claude\build\windows-release\bin\Release",
    r"D:\Dev\CyxWiz_Claude\build\windows-release\lib\Release",
]
for dll_dir in dll_dirs:
    if dll_dir not in os.environ.get('PATH', ''):
        os.environ['PATH'] = dll_dir + os.pathsep + os.environ.get('PATH', '')

# Add pycyxwiz to path
build_dir = r"D:\Dev\CyxWiz_Claude\build\windows-release\lib\Release"
sys.path.insert(0, build_dir)

import pycyxwiz as cx

print("=" * 70)
print(" GPU Performance Benchmark: CyxWiz vs NumPy")
print("=" * 70)
print()

# Initialize CyxWiz
cx.initialize()
print(f"CyxWiz Version: {cx.get_version()}")

# Show available devices
devices = cx.Device.get_available_devices()
print(f"\nAvailable Devices: {len(devices)}")
for dev in devices:
    print(f"  - {dev.name} ({dev.type})")
    if dev.memory_total > 0:
        print(f"    Memory: {dev.memory_total / (1024**3):.2f} GB")

print()

# Test different tensor sizes
sizes = [100, 1000, 10000, 100000, 1000000]
iterations = 100

print(f"Running {iterations} iterations of element-wise addition")
print()
print(f"{'Size':<12} {'NumPy (ms)':<15} {'CyxWiz (ms)':<15} {'Speedup':<15} {'Status':<10}")
print("-" * 80)

results = []

for size in sizes:
    # Create test data
    a_np = np.ones(size, dtype=np.float32)
    b_np = np.ones(size, dtype=np.float32)

    # NumPy baseline
    start = time.perf_counter()
    for _ in range(iterations):
        c_np = a_np + b_np
    numpy_time = (time.perf_counter() - start) * 1000  # milliseconds

    # CyxWiz (should use GPU if available)
    a_cx = cx.Tensor.from_numpy(a_np)
    b_cx = cx.Tensor.from_numpy(b_np)

    # Warm-up
    _ = a_cx + b_cx

    # Timed run
    start = time.perf_counter()
    for _ in range(iterations):
        c_cx = a_cx + b_cx
    cyxwiz_time = (time.perf_counter() - start) * 1000  # milliseconds

    # Calculate speedup
    speedup = numpy_time / cyxwiz_time if cyxwiz_time > 0 else 0

    # Format speedup
    if speedup >= 1:
        speedup_str = f"{speedup:.2f}x faster"
        status = "GPU!" if speedup > 2 else "OK"
    else:
        speedup_str = f"{1/speedup:.2f}x slower"
        status = "CPU fallback" if speedup < 0.5 else "overhead"

    print(f"{size:<12} {numpy_time:<15.2f} {cyxwiz_time:<15.2f} {speedup_str:<15} {status:<10}")

    # Verify correctness
    c_result = c_cx.to_numpy()
    if not np.allclose(c_np, c_result):
        print(f"  ⚠️  WARNING: Results don't match for size {size}!")
        print(f"     Expected: {c_np[:5]}")
        print(f"     Got:      {c_result[:5]}")

    results.append({
        'size': size,
        'numpy_ms': numpy_time,
        'cyxwiz_ms': cyxwiz_time,
        'speedup': speedup
    })

print()
print("=" * 80)
print(" Summary")
print("=" * 80)

# Analyze results
gpu_active = any(r['speedup'] > 5 for r in results if r['size'] >= 10000)

if gpu_active:
    print("✅ GPU Acceleration: ACTIVE")
    print(f"   Maximum speedup: {max(r['speedup'] for r in results):.2f}x")
    print(f"   Large tensors (1M): {results[-1]['speedup']:.2f}x faster")
    print()
    print("   GPU is working correctly! 🚀")
else:
    print("⚠️  GPU Acceleration: NOT DETECTED")
    print("   All operations appear to be using CPU")
    print()
    print("Possible reasons:")
    print("  1. ArrayFire fell back to CPU backend")
    print("  2. GPU overhead dominates for these sizes")
    print("  3. Build doesn't include GPU support")
    print()
    print("Check logs for ArrayFire warnings.")

print()
print("=" * 80)

cx.shutdown()
