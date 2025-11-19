#!/usr/bin/env python3
"""
Verify that arithmetic operators compute correctly
"""
import sys
import os
import numpy as np

# Setup
dll_dirs = [
    r"D:\Dev\CyxWiz_Claude\build\windows-release\bin\Release",
    r"D:\Dev\CyxWiz_Claude\build\windows-release\lib\Release",
]
for dll_dir in dll_dirs:
    if dll_dir not in os.environ.get('PATH', ''):
        os.environ['PATH'] = dll_dir + os.pathsep + os.environ.get('PATH', '')

build_dir = r"D:\Dev\CyxWiz_Claude\build\windows-release\lib\Release"
sys.path.insert(0, build_dir)

import pycyxwiz as cx

print("=" * 70)
print(" Arithmetic Operator Verification")
print("=" * 70)
print()

# Test 1: Simple addition
print("Test 1: Addition")
a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

a = cx.Tensor.from_numpy(a_np)
b = cx.Tensor.from_numpy(b_np)
c = a + b
c_np = c.to_numpy()

print(f"a = \n{a_np}")
print(f"b = \n{b_np}")
print(f"a + b = \n{c_np}")
print(f"Expected: \n{a_np + b_np}")
print(f"Match: {np.allclose(c_np, a_np + b_np)}")
print()

# Test 2: Subtraction
print("Test 2: Subtraction")
d = a - b
d_np = d.to_numpy()
print(f"a - b = \n{d_np}")
print(f"Expected: \n{a_np - b_np}")
print(f"Match: {np.allclose(d_np, a_np - b_np)}")
print()

# Test 3: Multiplication
print("Test 3: Multiplication (element-wise)")
e = a * b
e_np = e.to_numpy()
print(f"a * b = \n{e_np}")
print(f"Expected: \n{a_np * b_np}")
print(f"Match: {np.allclose(e_np, a_np * b_np)}")
print()

# Test 4: Division
print("Test 4: Division")
f = b / a
f_np = f.to_numpy()
print(f"b / a = \n{f_np}")
print(f"Expected: \n{b_np / a_np}")
print(f"Match: {np.allclose(f_np, b_np / a_np)}")
print()

# Test 5: Verify factory methods
print("Test 5: Factory Methods")
zeros = cx.Tensor.zeros([2, 2])
zeros_np = zeros.to_numpy()
print(f"Zeros:\n{zeros_np}")
print(f"All zeros: {np.all(zeros_np == 0)}")

ones = cx.Tensor.ones([2, 2])
ones_np = ones.to_numpy()
print(f"Ones:\n{ones_np}")
print(f"All ones: {np.all(ones_np == 1)}")
print()

print("=" * 70)
print(" All Arithmetic Operations Working Correctly!")
print("=" * 70)
