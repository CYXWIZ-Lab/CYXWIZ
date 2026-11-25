#!/usr/bin/env python3
"""
pycyxwiz Basic Usage Example

This example demonstrates the core functionality of the pycyxwiz Python bindings:
the backend, the devices, tensors, arithmetic, NumPy and the data types.

Run it with the Python pycyxwiz is built for (3.12), naming the folder that
holds pycyxwiz (and, if different, the folder of its DLLs); it needs numpy:

    python examples/python/pycyxwiz_basic.py build/windows-release/lib/Release build/windows-release/bin/Release

In the Engine's Script Editor pycyxwiz is already importable: run it with F5.
"""
import sys
import os
import numpy as np

folder = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
dll_folder = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else folder
for d in (folder, dll_folder, r"C:\Program Files\ArrayFire\v3\lib"):
    if os.path.isdir(d) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(d)
# ArrayFire loads its backend libraries through PATH.
os.environ["PATH"] = os.pathsep.join((folder, dll_folder)) + os.pathsep + os.environ.get("PATH", "")
sys.path.insert(0, folder)

import pycyxwiz as cx

print("=" * 70)
print(" pycyxwiz - Python Bindings for CyxWiz Backend")
print("=" * 70)
print()

# Initialize the backend
print("1. Initializing CyxWiz Backend...")
cx.initialize()
print(f"   Version: {cx.get_version()}")
print()

# Device enumeration
print("2. Available Devices:")
devices = cx.Device.get_available_devices()
for dev_info in devices:
    print(f"   - {dev_info.name} ({dev_info.type})")
    print(f"     Memory: {dev_info.memory_total / (1024**3):.2f} GB")
    print(f"     Compute Units: {dev_info.compute_units}")
print()

# Creating tensors
print("3. Creating Tensors:")
print("   Creating tensor with shape [3, 4]...")
t1 = cx.Tensor([3, 4], cx.DataType.Float32)
print(f"   {t1}")
print(f"   Shape: {t1.shape()}")
print(f"   Elements: {t1.num_elements()}")
print(f"   Dimensions: {t1.num_dimensions()}")
print()

# Factory methods
print("4. Factory Methods:")
zeros = cx.Tensor.zeros([2, 3])
print(f"   Zeros: {zeros}")

ones = cx.Tensor.ones([2, 3])
print(f"   Ones: {ones}")

random = cx.Tensor.random([2, 3])
print(f"   Random: {random}")
print()

# Arithmetic operations
print("5. Arithmetic Operations:")
a = cx.Tensor.ones([2, 2])
b = cx.Tensor.ones([2, 2])
print(f"   a = {a}")
print(f"   b = {b}")

c = a + b
print(f"   a + b = {c}")

d = a * b
print(f"   a * b = {d}")
print()

# NumPy interoperability
print("6. NumPy Interoperability:")
np_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
print(f"   NumPy array:\n{np_array}")

tensor = cx.Tensor.from_numpy(np_array)
print(f"   Converted to Tensor: {tensor}")

back_to_numpy = tensor.to_numpy()
print(f"   Converted back to NumPy:\n{back_to_numpy}")
print(f"   Data preserved: {np.allclose(np_array, back_to_numpy)}")
print()

# Different data types
print("7. Different Data Types:")
float64_tensor = cx.Tensor([3], cx.DataType.Float64)
print(f"   Float64: {float64_tensor}")

int32_tensor = cx.Tensor([3], cx.DataType.Int32)
print(f"   Int32: {int32_tensor}")

int64_tensor = cx.Tensor([3], cx.DataType.Int64)
print(f"   Int64: {int64_tensor}")

uint8_tensor = cx.Tensor([3], cx.DataType.UInt8)
print(f"   UInt8: {uint8_tensor}")
print()

# Shutdown
print("8. Shutting down...")
cx.shutdown()
print()

print("=" * 70)
print(" Example complete!")
print("=" * 70)
