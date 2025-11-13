#!/usr/bin/env python3
"""
CyxWiz Plotting System - Advanced Examples

This script demonstrates advanced plotting features including:
- Custom plot configurations
- Data manipulation
- Working with numpy arrays
- Plot management and lifecycle
- Integration with ML training loops

Usage:
    Run from CyxWiz Engine's Python console
"""

import cyxwiz_plotting as plt
import numpy as np
import time


def example_custom_config():
    """Create a plot with custom configuration"""
    print("Creating plot with custom configuration...")

    manager = plt.PlotManager.get_instance()

    # Create custom configuration
    config = plt.PlotConfig()
    config.title = "Custom Configured Plot"
    config.x_label = "Custom X"
    config.y_label = "Custom Y"
    config.type = plt.PlotType.Scatter
    config.backend = plt.BackendType.ImPlot
    config.auto_fit = True
    config.show_legend = True
    config.show_grid = True
    config.width = 1024
    config.height = 768

    plot_id = manager.create_plot(config)

    # Add data
    x = np.random.randn(200)
    y = np.random.randn(200)

    dataset = plt.PlotDataset()
    dataset.add_series("random_data")
    series = dataset.get_series("random_data")

    for xi, yi in zip(x, y):
        series.add_point(float(xi), float(yi))

    manager.add_dataset(plot_id, "random_data", dataset)

    print(f"Created custom plot: {plot_id}")
    plt.show_plot(plot_id)

    return plot_id


def example_dynamic_update():
    """Demonstrate dynamic plot updates"""
    print("Creating dynamically updating plot...")

    manager = plt.PlotManager.get_instance()

    config = plt.PlotConfig()
    config.title = "Dynamic Plot Update"
    config.x_label = "Iteration"
    config.y_label = "Value"
    config.type = plt.PlotType.Line
    config.backend = plt.BackendType.ImPlot

    plot_id = manager.create_plot(config)

    # Initial dataset
    dataset = plt.PlotDataset()
    dataset.add_series("evolving_data")
    manager.add_dataset(plot_id, "evolving_data", dataset)

    # Update data over time
    for i in range(50):
        x = i
        y = np.sin(i * 0.2) + np.random.randn() * 0.1

        # Get dataset and update
        ds = manager.get_dataset(plot_id, "evolving_data")
        if ds:
            series = ds.get_series("evolving_data")
            if series:
                series.add_point(float(x), float(y))

    print(f"Updated plot: {plot_id}")
    plt.show_plot(plot_id)

    return plot_id


def example_training_metrics():
    """Simulate ML training metrics visualization"""
    print("Simulating training metrics...")

    manager = plt.PlotManager.get_instance()

    # Create loss plot
    loss_config = plt.PlotConfig()
    loss_config.title = "Training Loss"
    loss_config.x_label = "Epoch"
    loss_config.y_label = "Loss"
    loss_config.type = plt.PlotType.Line
    loss_config.backend = plt.BackendType.ImPlot

    loss_plot_id = manager.create_plot(loss_config)

    # Create accuracy plot
    acc_config = plt.PlotConfig()
    acc_config.title = "Training Accuracy"
    acc_config.x_label = "Epoch"
    acc_config.y_label = "Accuracy (%)"
    acc_config.type = plt.PlotType.Line
    acc_config.backend = plt.BackendType.ImPlot

    acc_plot_id = manager.create_plot(acc_config)

    # Simulate training loop
    num_epochs = 100
    initial_loss = 2.5
    initial_acc = 30.0

    for epoch in range(num_epochs):
        # Simulate decreasing loss
        loss = initial_loss * np.exp(-epoch / 20.0) + np.random.randn() * 0.05

        # Simulate increasing accuracy
        acc = initial_acc + (95.0 - initial_acc) * (1 - np.exp(-epoch / 15.0))
        acc += np.random.randn() * 0.5

        # Update loss plot
        manager.update_realtime_plot(loss_plot_id, float(epoch), float(loss), "train_loss")

        # Update accuracy plot
        manager.update_realtime_plot(acc_plot_id, float(epoch), float(acc), "train_acc")

        # Also add validation metrics
        val_loss = loss * 1.1 + np.random.randn() * 0.03
        val_acc = acc * 0.95 + np.random.randn() * 0.5

        manager.update_realtime_plot(loss_plot_id, float(epoch), float(val_loss), "val_loss")
        manager.update_realtime_plot(acc_plot_id, float(epoch), float(val_acc), "val_acc")

    print(f"Created training plots: {loss_plot_id}, {acc_plot_id}")
    plt.show_plot(loss_plot_id)
    plt.show_plot(acc_plot_id)

    return loss_plot_id, acc_plot_id


def example_comparison_plots():
    """Create comparison plots for model evaluation"""
    print("Creating comparison plots...")

    manager = plt.PlotManager.get_instance()

    # Create configuration
    config = plt.PlotConfig()
    config.title = "Model Comparison"
    config.x_label = "Model"
    config.y_label = "Metric Value"
    config.type = plt.PlotType.Bar
    config.backend = plt.BackendType.ImPlot

    plot_id = manager.create_plot(config)

    # Models and their metrics
    models = ["ResNet", "VGG", "EfficientNet", "MobileNet"]
    accuracy = [92.3, 89.7, 94.1, 88.5]
    f1_scores = [91.8, 88.9, 93.5, 87.9]

    # Add accuracy dataset
    acc_dataset = plt.PlotDataset()
    acc_dataset.add_series("accuracy")
    acc_series = acc_dataset.get_series("accuracy")

    for i, val in enumerate(accuracy):
        acc_series.add_point(float(i), float(val))

    manager.add_dataset(plot_id, "accuracy", acc_dataset)

    # Add F1 score dataset
    f1_dataset = plt.PlotDataset()
    f1_dataset.add_series("f1_score")
    f1_series = f1_dataset.get_series("f1_score")

    for i, val in enumerate(f1_scores):
        f1_series.add_point(float(i), float(val))

    manager.add_dataset(plot_id, "f1_score", f1_dataset)

    print(f"Created comparison plot: {plot_id}")
    plt.show_plot(plot_id)

    return plot_id


def example_data_serialization():
    """Demonstrate saving and loading plot data"""
    print("Testing data serialization...")

    # Create a dataset
    dataset = plt.PlotDataset()
    dataset.add_series("test_data")
    series = dataset.get_series("test_data")

    x = np.linspace(0, 10, 50)
    y = np.sin(x)

    for xi, yi in zip(x, y):
        series.add_point(float(xi), float(yi))

    # Save to JSON
    filepath = "test_dataset.json"
    success = dataset.save_to_json(filepath)

    if success:
        print(f"Dataset saved to: {filepath}")

        # Load it back
        loaded_dataset = plt.PlotDataset()
        if loaded_dataset.load_from_json(filepath):
            print(f"Dataset loaded successfully")
            print(f"Series count: {loaded_dataset.get_series_count()}")
            print(f"Series names: {loaded_dataset.get_series_names()}")

            # Verify data
            loaded_series = loaded_dataset.get_series("test_data")
            if loaded_series:
                print(f"Data points: {loaded_series.size()}")
        else:
            print("Failed to load dataset")
    else:
        print("Failed to save dataset")


def example_plot_management():
    """Demonstrate plot lifecycle management"""
    print("Testing plot management...")

    manager = plt.PlotManager.get_instance()

    # Get current state
    initial_count = manager.get_plot_count()
    print(f"Initial plot count: {initial_count}")

    # Create several plots
    plot_ids = []
    for i in range(5):
        config = plt.PlotConfig()
        config.title = f"Test Plot {i+1}"
        config.x_label = "X"
        config.y_label = "Y"
        config.type = plt.PlotType.Line
        config.backend = plt.BackendType.ImPlot

        plot_id = manager.create_plot(config)
        plot_ids.append(plot_id)

    print(f"Created {len(plot_ids)} plots")
    print(f"Current plot count: {manager.get_plot_count()}")
    print(f"All plot IDs: {manager.get_all_plot_ids()}")

    # Check individual plots
    for plot_id in plot_ids:
        exists = manager.has_plot(plot_id)
        print(f"Plot {plot_id} exists: {exists}")

    # Delete specific plot
    if plot_ids:
        delete_id = plot_ids[0]
        success = manager.delete_plot(delete_id)
        print(f"Deleted plot {delete_id}: {success}")
        print(f"Plot count after deletion: {manager.get_plot_count()}")

    # Update plot configuration
    if len(plot_ids) > 1:
        update_id = plot_ids[1]
        config = manager.get_plot_config(update_id)
        config.title = "Updated Title"
        config.show_legend = False
        manager.update_plot_config(update_id, config)
        print(f"Updated plot {update_id} configuration")

        # Verify update
        new_config = manager.get_plot_config(update_id)
        print(f"New title: {new_config.title}")
        print(f"Show legend: {new_config.show_legend}")


def example_numpy_integration():
    """Demonstrate seamless numpy integration"""
    print("Testing NumPy integration...")

    # Create various numpy arrays
    x_list = [1, 2, 3, 4, 5]  # Python list
    y_array = np.array([1, 4, 9, 16, 25])  # NumPy array
    z_linspace = np.linspace(0, 10, 100)  # NumPy linspace
    w_random = np.random.randn(100)  # Random array

    # All should work seamlessly
    plot1 = plt.plot_line(x_list, y_array, title="List + Array")
    print(f"Created plot from list and array: {plot1}")

    plot2 = plt.plot_scatter(z_linspace, w_random, title="LinSpace + Random")
    print(f"Created plot from numpy functions: {plot2}")

    # 2D arrays (flattened automatically)
    matrix = np.random.randn(10, 10)
    flat_data = matrix.flatten()
    plot3 = plt.plot_histogram(flat_data, bins=20, title="2D Matrix Histogram")
    print(f"Created histogram from matrix: {plot3}")

    plt.show_plot(plot1)
    plt.show_plot(plot2)
    plt.show_plot(plot3)


def main():
    """Run all advanced examples"""
    print("=" * 60)
    print("CyxWiz Plotting System - Advanced Examples")
    print("=" * 60)

    manager = plt.PlotManager.get_instance()
    manager.initialize_python_backend()

    try:
        example_custom_config()
        print("\n" + "-" * 60 + "\n")

        example_dynamic_update()
        print("\n" + "-" * 60 + "\n")

        example_training_metrics()
        print("\n" + "-" * 60 + "\n")

        example_comparison_plots()
        print("\n" + "-" * 60 + "\n")

        example_data_serialization()
        print("\n" + "-" * 60 + "\n")

        example_plot_management()
        print("\n" + "-" * 60 + "\n")

        example_numpy_integration()
        print("\n" + "-" * 60 + "\n")

        print("=" * 60)
        print("All advanced examples completed successfully!")
        print(f"Total plots: {manager.get_plot_count()}")
        print("=" * 60)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
