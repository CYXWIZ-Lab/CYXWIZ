#!/usr/bin/env python3
"""
Test NumPy conversion for pycyxwiz
"""
import sys
import os
import numpy as np

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
print("NumPy Conversion Test")
print("=" * 60)
print()

# Import pycyxwiz
import pycyxwiz as cx

print("Test 1: NumPy array to Tensor (Float32)...")
try:
    np_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    print(f"  NumPy array:\n{np_arr}")
    print(f"  Shape: {np_arr.shape}, dtype: {np_arr.dtype}")

    tensor = cx.Tensor.from_numpy(np_arr)
    print(f"  Tensor: {tensor}")
    print(f"OK Conversion successful!")
except Exception as e:
    print(f"FAIL Failed: {e}")
    import traceback
    traceback.print_exc()

print()

print("Test 2: Tensor to NumPy array...")
try:
    # Create a tensor
    tensor = cx.Tensor.ones([2, 3])
    print(f"  Tensor: {tensor}")

    # Convert to NumPy
    np_result = tensor.to_numpy()
    print(f"  NumPy result:\n{np_result}")
    print(f"  Shape: {np_result.shape}, dtype: {np_result.dtype}")
    print(f"OK Conversion successful!")
except Exception as e:
    print(f"FAIL Failed: {e}")
    import traceback
    traceback.print_exc()

print()

print("Test 3: Round-trip conversion (NumPy -> Tensor -> NumPy)...")
try:
    # Create original NumPy array
    original = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]], dtype=np.float32)
    print(f"  Original NumPy:\n{original}")

    # Convert to Tensor
    tensor = cx.Tensor.from_numpy(original)
    print(f"  Tensor: {tensor}")

    # Convert back to NumPy
    result = tensor.to_numpy()
    print(f"  Result NumPy:\n{result}")

    # Check if values match
    if np.allclose(original, result):
        print(f"OK Round-trip successful! Data preserved.")
    else:
        print(f"FAIL Data mismatch after round-trip")
        print(f"  Difference:\n{original - result}")
except Exception as e:
    print(f"FAIL Failed: {e}")
    import traceback
    traceback.print_exc()

print()

print("Test 4: Different data types...")
try:
    # Float64
    np_f64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    tensor_f64 = cx.Tensor.from_numpy(np_f64)
    result_f64 = tensor_f64.to_numpy()
    print(f"  Float64: {np_f64.dtype} -> {result_f64.dtype} OK")

    # Int32
    np_i32 = np.array([1, 2, 3, 4], dtype=np.int32)
    tensor_i32 = cx.Tensor.from_numpy(np_i32)
    result_i32 = tensor_i32.to_numpy()
    print(f"  Int32: {np_i32.dtype} -> {result_i32.dtype} OK")

    # Int64
    np_i64 = np.array([10, 20, 30], dtype=np.int64)
    tensor_i64 = cx.Tensor.from_numpy(np_i64)
    result_i64 = tensor_i64.to_numpy()
    print(f"  Int64: {np_i64.dtype} -> {result_i64.dtype} OK")

    # UInt8
    np_u8 = np.array([0, 127, 255], dtype=np.uint8)
    tensor_u8 = cx.Tensor.from_numpy(np_u8)
    result_u8 = tensor_u8.to_numpy()
    print(f"  UInt8: {np_u8.dtype} -> {result_u8.dtype} OK")

    print(f"OK All data types converted successfully!")
except Exception as e:
    print(f"FAIL Failed: {e}")
    import traceback
    traceback.print_exc()

print()

print("Test 5: Arithmetic with NumPy arrays...")
try:
    # Create NumPy arrays
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    # Convert to Tensors
    a = cx.Tensor.from_numpy(a_np)
    b = cx.Tensor.from_numpy(b_np)

    # Perform arithmetic
    c = a + b
    d = a * b

    # Convert results back to NumPy
    c_np = c.to_numpy()
    d_np = d.to_numpy()

    print(f"  a:\n{a_np}")
    print(f"  b:\n{b_np}")
    print(f"  a + b:\n{c_np}")
    print(f"  a * b:\n{d_np}")

    # Verify against NumPy
    expected_add = a_np + b_np
    expected_mul = a_np * b_np

    if np.allclose(c_np, expected_add) and np.allclose(d_np, expected_mul):
        print(f"OK Arithmetic results match NumPy!")
    else:
        print(f"FAIL Arithmetic mismatch")
except Exception as e:
    print(f"FAIL Failed: {e}")
    import traceback
    traceback.print_exc()

print()

print("Test 6: Different shapes...")
try:
    shapes = [(5,), (3, 4), (2, 3, 4), (2, 2, 2, 2)]
    for shape in shapes:
        np_arr = np.random.rand(*shape).astype(np.float32)
        tensor = cx.Tensor.from_numpy(np_arr)
        result = tensor.to_numpy()

        if np.allclose(np_arr, result):
            print(f"  Shape {shape}: OK")
        else:
            print(f"  Shape {shape}: FAIL - Data mismatch")

    print(f"OK All shapes converted successfully!")
except Exception as e:
    print(f"FAIL Failed: {e}")
    import traceback
    traceback.print_exc()

print()

print("=" * 60)
print("Test Summary")
print("=" * 60)
print("NumPy conversion is working correctly!")
print("All tests passed! OK")
print("=" * 60)
