#!/usr/bin/env python3
"""
CyxWiz Training Dashboard Example

Demonstrates how to log training metrics from Python scripts
to the native CyxWiz Training Dashboard panel.

Usage:
    Run this script from the CyxWiz Engine's Python console.
    The Training Dashboard panel will update in real-time.
"""

import cyxwiz_plotting as plt
import time
import math


def simulate_training():
    """
    Simulate a training loop and log metrics to the dashboard.

    This shows how to integrate with custom Python training code
    or wrap existing frameworks (PyTorch, TensorFlow, etc.)
    """
    print("Starting simulated training...")

    # Get the training dashboard panel
    dashboard = plt.get_training_plot_panel()

    if dashboard is None:
        print("ERROR: Training Dashboard not available!")
        print("Make sure you're running this from the CyxWiz Engine console.")
        return

    # Clear any previous data
    dashboard.clear()

    # Training parameters
    num_epochs = 50
    initial_lr = 0.01

    print(f"Training for {num_epochs} epochs...")
    print("-" * 40)

    for epoch in range(1, num_epochs + 1):
        # Simulate training metrics (exponential decay with noise)
        progress = epoch / num_epochs

        # Training loss: starts high, decreases with noise
        train_loss = 2.0 * math.exp(-3 * progress) + 0.1 * math.sin(epoch * 0.5) + 0.05

        # Validation loss: slightly higher than training (typical pattern)
        val_loss = train_loss * 1.1 + 0.02 * math.sin(epoch * 0.3)

        # Training accuracy: starts low, increases
        train_acc = 1.0 - math.exp(-4 * progress) + 0.02 * math.sin(epoch * 0.4)
        train_acc = min(0.99, max(0.0, train_acc))

        # Validation accuracy: slightly lower
        val_acc = train_acc * 0.95 + 0.01 * math.sin(epoch * 0.2)
        val_acc = min(0.99, max(0.0, val_acc))

        # Learning rate with decay
        lr = initial_lr * math.exp(-0.05 * epoch)

        # Log metrics to dashboard
        dashboard.add_loss_point(epoch, train_loss, val_loss)
        dashboard.add_accuracy_point(epoch, train_acc, val_acc)
        dashboard.add_custom_metric("learning_rate", epoch, lr)

        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: loss={train_loss:.4f}, acc={train_acc:.4f}, lr={lr:.6f}")

        # Small delay to visualize real-time updates
        time.sleep(0.05)

    print("-" * 40)
    print("Training complete!")
    print(f"Final train loss: {train_loss:.4f}")
    print(f"Final train accuracy: {train_acc:.4f}")


def simulate_gan_training():
    """
    Simulate GAN training with generator and discriminator losses.
    Shows how to use custom metrics for complex training scenarios.
    """
    print("Starting simulated GAN training...")

    dashboard = plt.get_training_plot_panel()

    if dashboard is None:
        print("ERROR: Training Dashboard not available!")
        return

    dashboard.clear()

    num_epochs = 100

    print(f"Training GAN for {num_epochs} epochs...")
    print("-" * 40)

    for epoch in range(1, num_epochs + 1):
        progress = epoch / num_epochs

        # Generator loss: oscillates as G and D compete
        g_loss = 0.7 + 0.3 * math.sin(epoch * 0.2) + 0.1 * math.exp(-2 * progress)

        # Discriminator loss: inverse pattern to generator
        d_loss = 0.7 - 0.2 * math.sin(epoch * 0.2) + 0.1 * math.exp(-2 * progress)

        # FID score (quality metric): improves over time
        fid_score = 150 * math.exp(-3 * progress) + 10 + 5 * math.sin(epoch * 0.1)

        # Log as custom metrics
        dashboard.add_custom_metric("G_loss", epoch, g_loss)
        dashboard.add_custom_metric("D_loss", epoch, d_loss)
        dashboard.add_custom_metric("FID", epoch, fid_score)

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: G={g_loss:.4f}, D={d_loss:.4f}, FID={fid_score:.2f}")

        time.sleep(0.02)

    print("-" * 40)
    print("GAN training complete!")


def pytorch_style_example():
    """
    Example showing how you would integrate with PyTorch.
    (Pseudocode - requires actual PyTorch installation)
    """
    print("""
# PyTorch Integration Example (pseudocode)
# ----------------------------------------

import torch
import cyxwiz_plotting as plt

# Get dashboard
dashboard = plt.get_training_plot_panel()
dashboard.clear()

# Your PyTorch model and data
model = YourModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    # Training loop
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    train_loss /= len(train_loader)
    train_acc = correct / total

    # Validation loop
    model.eval()
    val_loss, val_acc = evaluate(model, val_loader)

    # Log to CyxWiz Dashboard
    dashboard.add_loss_point(epoch, train_loss, val_loss)
    dashboard.add_accuracy_point(epoch, train_acc, val_acc)
    dashboard.add_custom_metric("lr", epoch, optimizer.param_groups[0]['lr'])
""")


def export_example():
    """
    Show how to export training metrics to CSV.
    """
    print("Running training and exporting to CSV...")

    dashboard = plt.get_training_plot_panel()

    if dashboard is None:
        print("ERROR: Training Dashboard not available!")
        return

    dashboard.clear()

    # Quick training simulation
    for epoch in range(1, 21):
        train_loss = 1.0 / epoch
        val_loss = 1.1 / epoch
        dashboard.add_loss_point(epoch, train_loss, val_loss)
        dashboard.add_accuracy_point(epoch, 1 - train_loss, 1 - val_loss)

    # Export to CSV
    filepath = "training_metrics.csv"
    success = dashboard.export_to_csv(filepath)

    if success:
        print(f"Metrics exported to: {filepath}")
    else:
        print("Export failed")


def main():
    """Run examples"""
    print("=" * 60)
    print("CyxWiz Training Dashboard Examples")
    print("=" * 60)
    print()

    # Check if dashboard is available
    dashboard = plt.get_training_plot_panel()
    if dashboard is None:
        print("WARNING: Training Dashboard not available.")
        print("Run this script from the CyxWiz Engine console.")
        print()
        pytorch_style_example()
        return

    print("Select an example to run:")
    print("  1. Basic classifier training simulation")
    print("  2. GAN training simulation")
    print("  3. Export to CSV example")
    print("  4. PyTorch integration info")
    print()

    # For demo, run the basic simulation
    simulate_training()


if __name__ == "__main__":
    main()
