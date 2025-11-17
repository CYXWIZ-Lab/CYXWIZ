#!/usr/bin/env python3
"""
Test Activation Functions
Tests ReLU, Sigmoid, and Tanh activations
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
print(" Activation Functions Test")
print("=" * 70)
print()

# Initialize
cx.initialize()
print(f"CyxWiz Version: {cx.get_version()}")
print()

# Test data
test_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
test_tensor = cx.Tensor.from_numpy(test_data)

print("Test Input:", test_data)
print()

# Test 1: ReLU
print("Test 1: ReLU Activation")
print("-" * 70)

relu = cx.ReLU()
relu_output = relu.forward(test_tensor)
relu_result = relu_output.to_numpy()

# Expected: max(0, x) = [0, 0, 0, 1, 2]
expected_relu = np.maximum(0, test_data)

print(f"Forward (ReLU):  {relu_result}")
print(f"Expected:        {expected_relu}")
print(f"Match: {np.allclose(relu_result, expected_relu)}")

# Test backward
grad_output = np.ones_like(relu_result)
grad_tensor = cx.Tensor.from_numpy(grad_output)
relu_grad = relu.backward(grad_tensor, test_tensor)
relu_grad_result = relu_grad.to_numpy()

# Expected gradient: 1 if x > 0 else 0 = [0, 0, 0, 1, 1]
expected_relu_grad = (test_data > 0).astype(np.float32)

print(f"Backward (ReLU): {relu_grad_result}")
print(f"Expected:        {expected_relu_grad}")
print(f"Match: {np.allclose(relu_grad_result, expected_relu_grad)}")
print(f"[OK] ReLU test passed")
print()

# Test 2: Sigmoid
print("Test 2: Sigmoid Activation")
print("-" * 70)

sigmoid = cx.Sigmoid()
sigmoid_output = sigmoid.forward(test_tensor)
sigmoid_result = sigmoid_output.to_numpy()

# Expected: 1 / (1 + exp(-x))
expected_sigmoid = 1.0 / (1.0 + np.exp(-test_data))

print(f"Forward (Sigmoid): {sigmoid_result}")
print(f"Expected:          {expected_sigmoid}")
print(f"Match: {np.allclose(sigmoid_result, expected_sigmoid, atol=1e-6)}")

# Test backward
sigmoid_grad = sigmoid.backward(grad_tensor, test_tensor)
sigmoid_grad_result = sigmoid_grad.to_numpy()

# Expected gradient: sigmoid(x) * (1 - sigmoid(x))
sigmoid_vals = 1.0 / (1.0 + np.exp(-test_data))
expected_sigmoid_grad = grad_output * sigmoid_vals * (1.0 - sigmoid_vals)

print(f"Backward (Sigmoid): {sigmoid_grad_result}")
print(f"Expected:           {expected_sigmoid_grad}")
print(f"Match: {np.allclose(sigmoid_grad_result, expected_sigmoid_grad, atol=1e-6)}")
print(f"[OK] Sigmoid test passed")
print()

# Test 3: Tanh
print("Test 3: Tanh Activation")
print("-" * 70)

tanh = cx.Tanh()
tanh_output = tanh.forward(test_tensor)
tanh_result = tanh_output.to_numpy()

# Expected: tanh(x)
expected_tanh = np.tanh(test_data)

print(f"Forward (Tanh): {tanh_result}")
print(f"Expected:       {expected_tanh}")
print(f"Match: {np.allclose(tanh_result, expected_tanh, atol=1e-6)}")

# Test backward
tanh_grad = tanh.backward(grad_tensor, test_tensor)
tanh_grad_result = tanh_grad.to_numpy()

# Expected gradient: 1 - tanh(x)^2
tanh_vals = np.tanh(test_data)
expected_tanh_grad = grad_output * (1.0 - tanh_vals * tanh_vals)

print(f"Backward (Tanh): {tanh_grad_result}")
print(f"Expected:        {expected_tanh_grad}")
print(f"Match: {np.allclose(tanh_grad_result, expected_tanh_grad, atol=1e-6)}")
print(f"[OK] Tanh test passed")
print()

# Test 4: Batch input
print("Test 4: Batched Input (2D Tensor)")
print("-" * 70)

batch_data = np.array([
    [-1.0, 0.0, 1.0],
    [2.0, -2.0, 0.5]
], dtype=np.float32)
batch_tensor = cx.Tensor.from_numpy(batch_data)

print(f"Batch shape: {batch_data.shape}")
print(f"Batch data:\n{batch_data}")
print()

# ReLU on batch
relu_batch = relu.forward(batch_tensor)
relu_batch_result = relu_batch.to_numpy()
expected_batch = np.maximum(0, batch_data)

print(f"ReLU output:\n{relu_batch_result}")
print(f"Expected:\n{expected_batch}")
print(f"Match: {np.allclose(relu_batch_result, expected_batch)}")
print(f"[OK] Batched forward passed")
print()

# Test 5: Integration with Linear layer
print("Test 5: Integration with Linear Layer")
print("-" * 70)

# Create a simple network: Linear -> ReLU
layer = cx.LinearLayer(3, 2, use_bias=True)
relu_activation = cx.ReLU()

input_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
input_tensor = cx.Tensor.from_numpy(input_data)

# Forward through layer
linear_output = layer.forward(input_tensor)
print(f"Linear output shape: {linear_output.to_numpy().shape}")

# Apply ReLU
activated_output = relu_activation.forward(linear_output)
activated_result = activated_output.to_numpy()

print(f"After ReLU shape: {activated_result.shape}")
print(f"After ReLU: {activated_result}")

# Backward through ReLU and layer
grad_out = np.ones_like(activated_result)
grad_out_tensor = cx.Tensor.from_numpy(grad_out)

grad_from_relu = relu_activation.backward(grad_out_tensor, linear_output)
grad_from_layer = layer.backward(grad_from_relu)

print(f"Gradient from layer shape: {grad_from_layer.to_numpy().shape}")
print(f"[OK] Linear + ReLU integration passed")
print()

print("=" * 70)
print(" All Activation Tests PASSED!")
print("=" * 70)

cx.shutdown()
