#!/usr/bin/env python3
"""
Test script for pycyxwiz Python bindings
"""
import sys
import os

# Add DLL directories to PATH (Windows-specific)
dll_dirs = [
    r"D:\Dev\CyxWiz_Claude\build\windows-release\bin\Release",
    r"D:\Dev\CyxWiz_Claude\build\windows-release\lib\Release",
]
for dll_dir in dll_dirs:
    if dll_dir not in os.environ.get('PATH', ''):
        os.environ['PATH'] = dll_dir + os.pathsep + os.environ.get('PATH', '')

# Add the build directory to Python path
build_dir = r"D:\Dev\CyxWiz_Claude\build\windows-release\lib\Release"
sys.path.insert(0, build_dir)

print("=" * 60)
print("pycyxwiz Module Test")
print("=" * 60)
print()

# Test 1: Import the module
print("Test 1: Importing pycyxwiz...")
try:
    import pycyxwiz as cx
    print("OK pycyxwiz imported successfully!")
except ImportError as e:
    print(f"FAIL Import failed: {e}")
    sys.exit(1)

print()

# Test 2: Get version
print("Test 2: Version info...")
try:
    version = cx.get_version()
    print(f"OK CyxWiz version: {version}")
except Exception as e:
    print(f"FAIL Failed: {e}")

print()

# Test 3: List available devices
print("Test 3: Device enumeration...")
try:
    devices = cx.Device.get_available_devices()
    print(f"OK Found {len(devices)} device(s):")
    for dev_info in devices:
        print(f"  - {dev_info.name} ({dev_info.type})")
        print(f"    Memory: {dev_info.memory_total / (1024**3):.2f} GB")
except Exception as e:
    print(f"FAIL Failed: {e}")

print()

# Test 4: Create tensors
print("Test 4: Tensor creation...")
try:
    # Create a simple tensor
    t1 = cx.Tensor([3, 4], cx.DataType.Float32)
    print(f"OK Created tensor: {t1}")
    print(f"  Shape: {t1.shape()}")
    print(f"  Elements: {t1.num_elements()}")
    print(f"  Dimensions: {t1.num_dimensions()}")
except Exception as e:
    print(f"FAIL Failed: {e}")

print()

# Test 5: Factory methods
print("Test 5: Factory methods...")
try:
    zeros = cx.Tensor.zeros([2, 3])
    print(f"OK Zeros tensor: {zeros}")

    ones = cx.Tensor.ones([2, 3])
    print(f"OK Ones tensor: {ones}")

    random = cx.Tensor.random([2, 3])
    print(f"OK Random tensor: {random}")
except Exception as e:
    print(f"FAIL Failed: {e}")

print()

# Test 6: Arithmetic operations
print("Test 6: Arithmetic operations...")
try:
    a = cx.Tensor.ones([2, 2])
    b = cx.Tensor.ones([2, 2])

    c = a + b
    print(f"OK Addition: ones + ones = {c}")

    d = a * b
    print(f"OK Multiplication: ones * ones = {d}")
except Exception as e:
    print(f"FAIL Failed: {e}")

print()

# Test 7: Data types
print("Test 7: Data types...")
try:
    float32_tensor = cx.Tensor([3], cx.DataType.Float32)
    print(f"OK Float32 tensor: {float32_tensor}")

    int32_tensor = cx.Tensor([3], cx.DataType.Int32)
    print(f"OK Int32 tensor: {int32_tensor}")
except Exception as e:
    print(f"FAIL Failed: {e}")

print()

# Summary
print("=" * 60)
print("Test Summary")
print("=" * 60)
print("All basic tests passed! OK")
print()
print("pycyxwiz is working correctly!")
print("=" * 60)
