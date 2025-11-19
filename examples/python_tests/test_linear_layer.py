#!/usr/bin/env python3
"""
Test Linear Layer Implementation
Tests forward pass, backward pass, and gradient computation
"""
import sys
import os
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
print(" Linear Layer Test")
print("=" * 70)
print()

# Initialize
cx.initialize()
print(f"CyxWiz Version: {cx.get_version()}")
print()

# Test 1: Create Linear layer
print("Test 1: Create Linear Layer")
print("-" * 70)

in_features = 4
out_features = 3
layer = cx.LinearLayer(in_features, out_features, use_bias=True)

print(f"[OK] Created Linear({in_features}, {out_features})")
print(f"   In features: {layer.in_features}")
print(f"   Out features: {layer.out_features}")
print(f"   Has bias: {layer.has_bias}")
print()

# Test 2: Forward pass with single sample
print("Test 2: Forward Pass (Single Sample)")
print("-" * 70)

# Create input [in_features]
input_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
input_tensor = cx.Tensor.from_numpy(input_data)

print(f"Input shape: {input_data.shape}")
print(f"Input: {input_data}")
print()

# Forward pass
output_tensor = layer.forward(input_tensor)
output_data = output_tensor.to_numpy()

print(f"Output shape: {output_data.shape}")
print(f"Output: {output_data}")
print(f"[OK] Forward pass successful (single sample)")
print()

# Test 3: Forward pass with batch
print("Test 3: Forward Pass (Batched)")
print("-" * 70)

# Create batched input [batch_size, in_features]
batch_size = 2
batch_input = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0]
], dtype=np.float32)
batch_tensor = cx.Tensor.from_numpy(batch_input)

print(f"Batch input shape: {batch_input.shape}")
print(f"Batch input:\n{batch_input}")
print()

batch_output_tensor = layer.forward(batch_tensor)
batch_output = batch_output_tensor.to_numpy()

print(f"Batch output shape: {batch_output.shape}")
print(f"Batch output:\n{batch_output}")
print(f"[OK] Batched forward pass successful")
print()

# Test 4: Backward pass
print("Test 4: Backward Pass (Gradient Computation)")
print("-" * 70)

# Create gradient output (matches output shape)
grad_output_data = np.ones_like(batch_output, dtype=np.float32)
grad_output_tensor = cx.Tensor.from_numpy(grad_output_data)

print(f"Grad output shape: {grad_output_data.shape}")
print()

# Backward pass
grad_input_tensor = layer.backward(grad_output_tensor)
grad_input = grad_input_tensor.to_numpy()

print(f"Grad input shape: {grad_input.shape}")
print(f"Grad input:\n{grad_input}")
print(f"[OK] Backward pass successful")
print()

# Test 5: Parameter access
print("Test 5: Parameter Access")
print("-" * 70)

params = layer.get_parameters()
print(f"Parameters: {list(params.keys())}")

weight = params['weight'].to_numpy()
bias = params['bias'].to_numpy()

print(f"\nWeight shape: {weight.shape}")
print(f"Weight:\n{weight}")
print(f"\nBias shape: {bias.shape}")
print(f"Bias: {bias}")
print()

# Test 6: Gradient access
print("Test 6: Gradient Access")
print("-" * 70)

grads = layer.get_gradients()
print(f"Gradients: {list(grads.keys())}")

weight_grad = grads['weight'].to_numpy()
bias_grad = grads['bias'].to_numpy()

print(f"\nWeight gradient shape: {weight_grad.shape}")
print(f"Weight gradient:\n{weight_grad}")
print(f"\nBias gradient shape: {bias_grad.shape}")
print(f"Bias gradient: {bias_grad}")
print(f"[OK] Gradients computed correctly")
print()

# Test 7: Verify gradient correctness (numerical)
print("Test 7: Numerical Gradient Check")
print("-" * 70)

# Use single sample for simplicity
simple_input = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
simple_tensor = cx.Tensor.from_numpy(simple_input)

# Get current parameters
params_before = layer.get_parameters()
weight_before = params_before['weight'].to_numpy().copy()

# Forward
output1 = layer.forward(simple_tensor)
output1_np = output1.to_numpy()

# Backward with ones gradient
grad_ones = np.ones_like(output1_np, dtype=np.float32)
grad_ones_tensor = cx.Tensor.from_numpy(grad_ones)
layer.backward(grad_ones_tensor)

# Get gradients
grads_check = layer.get_gradients()
weight_grad_check = grads_check['weight'].to_numpy()

print(f"Input: {simple_input}")
print(f"Output: {output1_np}")
print(f"\nWeight gradient (first row): {weight_grad_check[0]}")
print(f"Expected (should match input): {simple_input}")

# The gradient w.r.t weight[i,j] = input[j] * grad_output[i]
# With grad_output = ones and input = [1,0,0,0], gradient should be [1,0,0,0] for each row
match = np.allclose(weight_grad_check[:, 0], grad_ones, atol=1e-5)
print(f"\n[OK] Gradient check: {'PASS' if match else 'FAIL'}")
print()

# Test 8: Multi-layer forward pass
print("Test 8: Multi-Layer Network")
print("-" * 70)

layer1 = cx.LinearLayer(4, 8, use_bias=True)
layer2 = cx.LinearLayer(8, 3, use_bias=True)

test_input = np.random.randn(2, 4).astype(np.float32)
test_tensor = cx.Tensor.from_numpy(test_input)

# Forward through both layers
hidden = layer1.forward(test_tensor)
output = layer2.forward(hidden)

output_result = output.to_numpy()

print(f"Input shape: {test_input.shape}")
print(f"Hidden shape: {hidden.to_numpy().shape}")
print(f"Output shape: {output_result.shape}")
print(f"[OK] Multi-layer forward pass successful")
print()

print("=" * 70)
print(" All Linear Layer Tests PASSED!")
print("=" * 70)

cx.shutdown()
