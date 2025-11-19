"""
End-to-End Training Example: XOR Problem
This demonstrates a complete training pipeline with:
- Network construction (Linear layers + activations)
- Forward pass
- Loss computation (MSE)
- Backward pass (gradient computation)
- Parameter updates (SGD optimizer)
- Training loop over multiple epochs
"""

import sys
import os

# We're running from test_clean directory with all DLLs present
# Rename .pyd file if needed
pyd_name = "pycyxwiz.cp314-win_amd64.pyd"
if os.path.exists(pyd_name) and not os.path.exists("pycyxwiz.pyd"):
    import shutil
    shutil.copy(pyd_name, "pycyxwiz.pyd")

import pycyxwiz as cx
import numpy as np

def main():
    print("=" * 60)
    print("CyxWiz End-to-End Training Example: XOR Problem")
    print("=" * 60)

    # Initialize backend
    cx.initialize()
    print(f"CyxWiz Version: {cx.get_version()}")
    print()

    # XOR dataset
    # Inputs: [[0,0], [0,1], [1,0], [1,1]]
    # Outputs: [[0], [1], [1], [0]]
    X_data = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ], dtype=np.float32)

    y_data = np.array([
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ], dtype=np.float32)

    print("XOR Dataset:")
    print(f"Inputs shape: {X_data.shape}")
    print(f"Outputs shape: {y_data.shape}")
    print()

    # Create tensors from numpy arrays
    X = cx.Tensor.from_numpy(X_data)
    y = cx.Tensor.from_numpy(y_data)

    # Build network: 2 -> 4 -> 1 (with ReLU activation)
    # Layer 1: Linear(2, 4)
    # Activation: ReLU
    # Layer 2: Linear(4, 1)
    print("Network Architecture:")
    print("  Input: 2 features")
    print("  Hidden Layer: Linear(2 -> 4) + ReLU")
    print("  Output Layer: Linear(4 -> 1)")
    print()

    layer1 = cx.LinearLayer(2, 4)
    activation = cx.ReLU()
    layer2 = cx.LinearLayer(4, 1)

    # Initialize weights with small random values
    # For simplicity, we'll use the default initialization

    # Create loss and optimizer
    criterion = cx.MSELoss()
    optimizer_learning_rate = 0.1

    print(f"Loss Function: MSE")
    print(f"Optimizer: SGD (learning_rate={optimizer_learning_rate})")
    print()

    # Training loop
    num_epochs = 1000
    print_every = 100

    print(f"Training for {num_epochs} epochs...")
    print("-" * 60)

    for epoch in range(num_epochs):
        # Forward pass
        # h1 = layer1(X)
        h1 = layer1.forward(X)

        # a1 = relu(h1)
        a1 = activation.forward(h1)

        # output = layer2(a1)
        output = layer2.forward(a1)

        # Compute loss
        loss_tensor = criterion.forward(output, y)
        loss_value = loss_tensor.to_numpy()[0]

        # Backward pass
        # Gradient from loss
        grad_output = criterion.backward(output, y)

        # Backward through layer2 (layer caches input from forward)
        grad_a1 = layer2.backward(grad_output)

        # Backward through activation (needs both grad and input)
        grad_h1 = activation.backward(grad_a1, h1)

        # Backward through layer1 (layer caches input from forward)
        grad_X = layer1.backward(grad_h1)

        # Get gradients from layers
        layer1_grads = layer1.get_gradients()
        layer2_grads = layer2.get_gradients()

        # Manual parameter update (SGD: param -= lr * grad)
        # Update layer1 parameters
        layer1_params = layer1.get_parameters()

        # weights -= lr * grad_weights
        w1_data = layer1_params["weight"].to_numpy()
        w1_grad = layer1_grads["weight"].to_numpy()
        w1_data -= optimizer_learning_rate * w1_grad

        # bias -= lr * grad_bias
        b1_data = layer1_params["bias"].to_numpy()
        b1_grad = layer1_grads["bias"].to_numpy()
        b1_data -= optimizer_learning_rate * b1_grad

        # Set updated parameters
        layer1.set_parameters({
            "weight": cx.Tensor.from_numpy(w1_data),
            "bias": cx.Tensor.from_numpy(b1_data)
        })

        # Update layer2 parameters
        layer2_params = layer2.get_parameters()

        # weights -= lr * grad_weights
        w2_data = layer2_params["weight"].to_numpy()
        w2_grad = layer2_grads["weight"].to_numpy()
        w2_data -= optimizer_learning_rate * w2_grad

        # bias -= lr * grad_bias
        b2_data = layer2_params["bias"].to_numpy()
        b2_grad = layer2_grads["bias"].to_numpy()
        b2_data -= optimizer_learning_rate * b2_grad

        # Set updated parameters
        layer2.set_parameters({
            "weight": cx.Tensor.from_numpy(w2_data),
            "bias": cx.Tensor.from_numpy(b2_data)
        })

        # Print progress
        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{num_epochs} | Loss: {loss_value:.6f}")

    print("-" * 60)
    print()

    # Final evaluation
    print("Final Evaluation:")
    print("-" * 60)

    # Forward pass one more time
    h1 = layer1.forward(X)
    a1 = activation.forward(h1)
    output = layer2.forward(a1)

    predictions = output.to_numpy()
    targets = y.to_numpy()

    print("Input | Target | Prediction | Error")
    print("-" * 60)
    for i in range(4):
        input_vals = X_data[i]
        target = targets[i, 0]  # Extract scalar from array
        pred = predictions[i, 0]  # Extract scalar from array
        error = abs(target - pred)
        print(f"{input_vals} | {target:.4f} | {pred:.4f}    | {error:.4f}")

    print()

    # Check if network learned XOR
    final_loss = criterion.forward(output, y).to_numpy()[0]
    print(f"Final Loss: {final_loss:.6f}")

    if final_loss < 0.1:
        print("[SUCCESS] Network successfully learned XOR function!")
    else:
        print("[WARNING] Network did not fully converge. Try more epochs or adjust learning rate.")

    print()
    print("=" * 60)

    # Shutdown
    cx.shutdown()
    print("Training example completed.")

if __name__ == "__main__":
    main()
