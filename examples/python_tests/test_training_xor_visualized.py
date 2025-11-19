"""
Enhanced XOR Training with Real-time Visualization
This demonstrates end-to-end training with live plotting in the CyxWiz Engine GUI
"""

import sys
import os

# Add DLL directories to PATH
dll_dir1 = "D:/Dev/CyxWiz_Claude/build/windows-release/bin/Release"
dll_dir2 = "D:/Dev/CyxWiz_Claude/build/windows-release/lib/Release"
os.environ['PATH'] = dll_dir1 + os.pathsep + dll_dir2 + os.pathsep + os.environ.get('PATH', '')
sys.path.insert(0, dll_dir2)

import pycyxwiz as cx  # ML backend
import numpy as np
import time

# Try to import visualization module
try:
    import cyxwiz_plotting as plot
    HAS_PLOTTING = True
except ImportError:
    print("[WARNING] cyxwiz_plotting module not found. Training will proceed without visualization.")
    HAS_PLOTTING = False

def main():
    print("=" * 70)
    print("Enhanced XOR Training with Real-time Visualization")
    print("=" * 70)

    # Initialize backend
    cx.initialize()
    print(f"CyxWiz Version: {cx.get_version()}")
    print()

    # Get training plot panel (if available)
    training_panel = None
    if HAS_PLOTTING:
        try:
            training_panel = plot.get_training_plot_panel()
            if training_panel:
                print("[INFO] Connected to TrainingPlotPanel - real-time visualization enabled!")
                training_panel.clear()  # Clear any previous data
                training_panel.show_loss_plot(True)
                training_panel.set_auto_scale(True)
            else:
                print("[WARNING] TrainingPlotPanel not available. Running without visualization.")
        except Exception as e:
            print(f"[WARNING] Could not connect to TrainingPlotPanel: {e}")
            training_panel = None
    print()

    # XOR dataset
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

    # Create tensors
    X = cx.Tensor.from_numpy(X_data)
    y = cx.Tensor.from_numpy(y_data)

    # Build network: 2 -> 4 -> 1
    print("Network Architecture:")
    print("  Input: 2 features")
    print("  Hidden Layer: Linear(2 -> 4) + ReLU")
    print("  Output Layer: Linear(4 -> 1)")
    print()

    layer1 = cx.LinearLayer(2, 4)
    activation = cx.ReLU()
    layer2 = cx.LinearLayer(4, 1)

    # Loss and optimizer
    criterion = cx.MSELoss()
    optimizer_learning_rate = 0.1

    print(f"Loss Function: MSE")
    print(f"Optimizer: SGD (learning_rate={optimizer_learning_rate})")
    print(f"Real-time Visualization: {'ENABLED' if training_panel else 'DISABLED'}")
    print()

    # Training loop
    num_epochs = 1000
    print_every = 10  # Print more frequently for visual feedback

    print(f"Training for {num_epochs} epochs...")
    print("-" * 70)

    for epoch in range(num_epochs):
        # Forward pass
        h1 = layer1.forward(X)
        a1 = activation.forward(h1)
        output = layer2.forward(a1)

        # Compute loss
        loss_tensor = criterion.forward(output, y)
        loss_value = loss_tensor.to_numpy()[0]

        # Update plot panel (if available)
        if training_panel:
            training_panel.add_loss_point(epoch, loss_value)

            # Add learning rate as custom metric
            if epoch % 100 == 0:  # Less frequent to avoid clutter
                training_panel.add_custom_metric("Learning Rate", epoch, optimizer_learning_rate)

        # Backward pass
        grad_output = criterion.backward(output, y)
        grad_a1 = layer2.backward(grad_output)
        grad_h1 = activation.backward(grad_a1, h1)
        grad_X = layer1.backward(grad_h1)

        # Get gradients
        layer1_grads = layer1.get_gradients()
        layer2_grads = layer2.get_gradients()

        # Manual SGD parameter update
        # Layer 1
        layer1_params = layer1.get_parameters()
        w1_data = layer1_params["weight"].to_numpy()
        w1_grad = layer1_grads["weight"].to_numpy()
        w1_data -= optimizer_learning_rate * w1_grad

        b1_data = layer1_params["bias"].to_numpy()
        b1_grad = layer1_grads["bias"].to_numpy()
        b1_data -= optimizer_learning_rate * b1_grad

        layer1.set_parameters({
            "weight": cx.Tensor.from_numpy(w1_data),
            "bias": cx.Tensor.from_numpy(b1_data)
        })

        # Layer 2
        layer2_params = layer2.get_parameters()
        w2_data = layer2_params["weight"].to_numpy()
        w2_grad = layer2_grads["weight"].to_numpy()
        w2_data -= optimizer_learning_rate * w2_grad

        b2_data = layer2_params["bias"].to_numpy()
        b2_grad = layer2_grads["bias"].to_numpy()
        b2_data -= optimizer_learning_rate * b2_grad

        layer2.set_parameters({
            "weight": cx.Tensor.from_numpy(w2_data),
            "bias": cx.Tensor.from_numpy(b2_data)
        })

        # Print progress
        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{num_epochs} | Loss: {loss_value:.6f}")

        # Small delay to allow GUI to update (only if visualizing)
        if training_panel and epoch % 5 == 0:
            time.sleep(0.001)  # 1ms delay every 5 epochs

    print("-" * 70)
    print()

    # Final evaluation
    print("Final Evaluation:")
    print("-" * 70)

    h1 = layer1.forward(X)
    a1 = activation.forward(h1)
    output = layer2.forward(a1)

    predictions = output.to_numpy()
    targets = y.to_numpy()

    print("Input     | Target | Prediction | Error")
    print("-" * 70)
    for i in range(4):
        input_vals = X_data[i]
        target = targets[i, 0]
        pred = predictions[i, 0]
        error = abs(target - pred)
        print(f"{input_vals} | {target:.4f} | {pred:.6f}   | {error:.6f}")

    print()

    final_loss = criterion.forward(output, y).to_numpy()[0]
    print(f"Final Loss: {final_loss:.6f}")

    if final_loss < 0.1:
        print("[SUCCESS] Network successfully learned XOR function!")
    else:
        print("[INFO] Network partially converged. Loss: {:.6f}".format(final_loss))

    # Export metrics if visualization was enabled
    if training_panel:
        try:
            export_path = "training_metrics_xor.csv"
            training_panel.export_to_csv(export_path)
            print(f"\n[INFO] Training metrics exported to: {export_path}")
        except Exception as e:
            print(f"\n[WARNING] Could not export metrics: {e}")

    print()
    print("=" * 70)
    print("Training complete! Check the Training Visualization panel in the GUI.")
    print("=" * 70)

    # Shutdown
    cx.shutdown()

if __name__ == "__main__":
    main()
