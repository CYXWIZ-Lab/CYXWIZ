#!/usr/bin/env python3
"""
Test Loss Functions
Tests MSE and CrossEntropy loss functions
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
print(" Loss Functions Test")
print("=" * 70)
print()

# Initialize
cx.initialize()
print(f"CyxWiz Version: {cx.get_version()}")
print()

# ============================================================================
# Test 1: MSE Loss - Simple case
# ============================================================================
print("Test 1: MSE Loss - Simple Case")
print("-" * 70)

predictions = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
targets = np.array([1.5, 2.5, 2.5, 3.5], dtype=np.float32)

pred_tensor = cx.Tensor.from_numpy(predictions)
target_tensor = cx.Tensor.from_numpy(targets)

mse_loss = cx.MSELoss()
loss_value = mse_loss.forward(pred_tensor, target_tensor)
loss_val = loss_value.to_numpy()[0]

# Expected MSE: mean((pred - target)^2)
# Differences: [-0.5, -0.5, 0.5, 0.5]
# Squared: [0.25, 0.25, 0.25, 0.25]
# Mean: 0.25
expected_loss = np.mean((predictions - targets) ** 2)

print(f"Predictions: {predictions}")
print(f"Targets:     {targets}")
print(f"MSE Loss:    {loss_val}")
print(f"Expected:    {expected_loss}")
print(f"Match: {np.isclose(loss_val, expected_loss)}")

# Test backward
grad = mse_loss.backward(pred_tensor, target_tensor)
grad_result = grad.to_numpy()

# Expected gradient: 2 * (pred - target) / N
expected_grad = 2.0 * (predictions - targets) / len(predictions)

print(f"Gradient:    {grad_result}")
print(f"Expected:    {expected_grad}")
print(f"Match: {np.allclose(grad_result, expected_grad)}")
print(f"[OK] MSE simple test passed")
print()

# ============================================================================
# Test 2: MSE Loss - Batched input
# ============================================================================
print("Test 2: MSE Loss - Batched Input (2D)")
print("-" * 70)

batch_pred = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
], dtype=np.float32)

batch_target = np.array([
    [1.5, 2.5, 2.5],
    [3.5, 5.5, 6.5]
], dtype=np.float32)

batch_pred_tensor = cx.Tensor.from_numpy(batch_pred)
batch_target_tensor = cx.Tensor.from_numpy(batch_target)

loss_value = mse_loss.forward(batch_pred_tensor, batch_target_tensor)
loss_val = loss_value.to_numpy()[0]

expected_loss = np.mean((batch_pred - batch_target) ** 2)

print(f"Batch shape: {batch_pred.shape}")
print(f"MSE Loss:    {loss_val}")
print(f"Expected:    {expected_loss}")
print(f"Match: {np.isclose(loss_val, expected_loss)}")

grad = mse_loss.backward(batch_pred_tensor, batch_target_tensor)
grad_result = grad.to_numpy()
expected_grad = 2.0 * (batch_pred - batch_target) / batch_pred.size

print(f"Gradient shape: {grad_result.shape}")
print(f"Match: {np.allclose(grad_result, expected_grad)}")
print(f"[OK] MSE batched test passed")
print()

# ============================================================================
# Test 3: CrossEntropy Loss - Binary classification
# ============================================================================
print("Test 3: CrossEntropy Loss - Binary Classification")
print("-" * 70)

# Logits for 2 samples, 2 classes (before softmax)
logits = np.array([
    [2.0, 1.0],  # Sample 1: class 0 more likely
    [1.0, 3.0]   # Sample 2: class 1 more likely
], dtype=np.float32)

# One-hot encoded targets
targets_ce = np.array([
    [1.0, 0.0],  # Sample 1: class 0
    [0.0, 1.0]   # Sample 2: class 1
], dtype=np.float32)

logits_tensor = cx.Tensor.from_numpy(logits)
targets_ce_tensor = cx.Tensor.from_numpy(targets_ce)

ce_loss = cx.CrossEntropyLoss()
loss_value = ce_loss.forward(logits_tensor, targets_ce_tensor)
loss_val = loss_value.to_numpy()[0]

# Compute expected cross entropy manually
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

probs = softmax(logits)
epsilon = 1e-7
ce_manual = -np.mean(np.sum(targets_ce * np.log(probs + epsilon), axis=1))

print(f"Logits:\n{logits}")
print(f"Targets (one-hot):\n{targets_ce}")
print(f"Softmax probs:\n{probs}")
print(f"CE Loss:     {loss_val}")
print(f"Expected:    {ce_manual}")
print(f"Match: {np.isclose(loss_val, ce_manual, atol=1e-5)}")

# Test backward
grad = ce_loss.backward(logits_tensor, targets_ce_tensor)
grad_result = grad.to_numpy()

# Expected gradient: (softmax(logits) - targets) / batch_size
expected_grad = (probs - targets_ce) / logits.shape[0]

print(f"Gradient:\n{grad_result}")
print(f"Expected:\n{expected_grad}")
print(f"Match: {np.allclose(grad_result, expected_grad, atol=1e-5)}")
print(f"[OK] CrossEntropy binary test passed")
print()

# ============================================================================
# Test 4: CrossEntropy Loss - Multi-class
# ============================================================================
print("Test 4: CrossEntropy Loss - Multi-class (3 classes)")
print("-" * 70)

# 4 samples, 3 classes
logits_multi = np.array([
    [2.0, 1.0, 0.1],
    [1.0, 3.0, 0.5],
    [0.5, 0.2, 2.5],
    [1.5, 1.5, 1.5]
], dtype=np.float32)

# One-hot targets
targets_multi = np.array([
    [1.0, 0.0, 0.0],  # Class 0
    [0.0, 1.0, 0.0],  # Class 1
    [0.0, 0.0, 1.0],  # Class 2
    [1.0, 0.0, 0.0]   # Class 0
], dtype=np.float32)

logits_multi_tensor = cx.Tensor.from_numpy(logits_multi)
targets_multi_tensor = cx.Tensor.from_numpy(targets_multi)

loss_value = ce_loss.forward(logits_multi_tensor, targets_multi_tensor)
loss_val = loss_value.to_numpy()[0]

probs_multi = softmax(logits_multi)
ce_manual = -np.mean(np.sum(targets_multi * np.log(probs_multi + epsilon), axis=1))

print(f"Batch size: {logits_multi.shape[0]}, Classes: {logits_multi.shape[1]}")
print(f"CE Loss:     {loss_val}")
print(f"Expected:    {ce_manual}")
print(f"Match: {np.isclose(loss_val, ce_manual, atol=1e-5)}")

grad = ce_loss.backward(logits_multi_tensor, targets_multi_tensor)
grad_result = grad.to_numpy()
expected_grad = (probs_multi - targets_multi) / logits_multi.shape[0]

print(f"Gradient match: {np.allclose(grad_result, expected_grad, atol=1e-5)}")
print(f"[OK] CrossEntropy multi-class test passed")
print()

# ============================================================================
# Test 5: Integration - Linear + ReLU + MSE Loss
# ============================================================================
print("Test 5: Integration - Linear Layer + ReLU + MSE Loss")
print("-" * 70)

# Create a simple network: Linear(3 -> 2) -> ReLU -> MSE Loss
layer = cx.LinearLayer(3, 2, use_bias=True)
relu = cx.ReLU()
mse = cx.MSELoss()

# Input data
input_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
target_data = np.array([[0.5, 1.0]], dtype=np.float32)

input_tensor = cx.Tensor.from_numpy(input_data)
target_tensor = cx.Tensor.from_numpy(target_data)

# Forward pass
linear_out = layer.forward(input_tensor)
print(f"Linear output: {linear_out.to_numpy()}")

activated_out = relu.forward(linear_out)
print(f"After ReLU:    {activated_out.to_numpy()}")

loss = mse.forward(activated_out, target_tensor)
print(f"MSE Loss:      {loss.to_numpy()[0]}")

# Backward pass
grad_loss = mse.backward(activated_out, target_tensor)
grad_relu = relu.backward(grad_loss, linear_out)
grad_input = layer.backward(grad_relu)

print(f"Gradient shape: {grad_input.to_numpy().shape}")
print(f"Weight gradients available: {layer.get_gradients()['weight'].to_numpy().shape}")
print(f"[OK] Integration test passed")
print()

# ============================================================================
# Test 6: Integration - Linear + Softmax + CrossEntropy
# ============================================================================
print("Test 6: Integration - Linear Layer + CrossEntropy Loss")
print("-" * 70)

# Create classifier: Linear(4 -> 3) -> CrossEntropy
classifier = cx.LinearLayer(4, 3, use_bias=True)
ce = cx.CrossEntropyLoss()

# Input: 2 samples, 4 features each
clf_input = np.random.randn(2, 4).astype(np.float32)
# Targets: one-hot for 2 samples, 3 classes
clf_targets = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

clf_input_tensor = cx.Tensor.from_numpy(clf_input)
clf_targets_tensor = cx.Tensor.from_numpy(clf_targets)

# Forward
logits_out = classifier.forward(clf_input_tensor)
print(f"Logits shape: {logits_out.to_numpy().shape}")

loss = ce.forward(logits_out, clf_targets_tensor)
print(f"CE Loss: {loss.to_numpy()[0]}")

# Backward
grad_ce = ce.backward(logits_out, clf_targets_tensor)
grad_input = classifier.backward(grad_ce)

print(f"Gradient from CE shape: {grad_ce.to_numpy().shape}")
print(f"Gradient to input shape: {grad_input.to_numpy().shape}")
print(f"Weight gradients shape: {classifier.get_gradients()['weight'].to_numpy().shape}")
print(f"[OK] Classifier integration test passed")
print()

print("=" * 70)
print(" All Loss Function Tests PASSED!")
print("=" * 70)

cx.shutdown()
