#!/usr/bin/env python3
"""
pycyxwiz Basic Usage Example

This example demonstrates the core functionality of the pycyxwiz Python bindings.
"""
import sys
import os
import numpy as np

# Setup PATH for DLLs (Windows only)
if sys.platform == 'win32':
    dll_dirs = [
        r"D:\Dev\CyxWiz_Claude\build\windows-release\bin\Release",
        r"D:\Dev\CyxWiz_Claude\build\windows-release\lib\Release",
    ]
    for dll_dir in dll_dirs:
        if dll_dir not in os.environ.get('PATH', ''):
            os.environ['PATH'] = dll_dir + os.pathsep + os.environ.get('PATH', '')

    # Add module to path
    build_dir = r"D:\Dev\CyxWiz_Claude\build\windows-release\lib\Release"
    sys.path.insert(0, build_dir)

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
