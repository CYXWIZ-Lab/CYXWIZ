#!/usr/bin/env python3
"""
CyxWiz Plotting System - Basic Examples

This script demonstrates basic plotting capabilities using the CyxWiz
plotting bindings. These plots will appear in the CyxWiz Engine GUI.

Usage:
    Run this script from the CyxWiz Engine's Python console or
    execute it as a standalone script if the module is in PYTHONPATH.
"""

import cyxwiz_plotting as plt
import numpy as np


def example_line_plot():
    """Create a simple line plot"""
    print("Creating line plot...")

    # Generate data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create plot
    plot_id = plt.plot_line(
        x, y,
        title="Sine Wave",
        x_label="Time (s)",
        y_label="Amplitude"
    )

    print(f"Created plot: {plot_id}")

    # Show in GUI
    plt.show_plot(plot_id)

    return plot_id


def example_scatter_plot():
    """Create a scatter plot with random data"""
    print("Creating scatter plot...")

    # Generate random data
    np.random.seed(42)
    x = np.random.randn(100)
    y = 2 * x + np.random.randn(100) * 0.5

    # Create plot
    plot_id = plt.plot_scatter(
        x, y,
        title="Random Scatter",
        x_label="X Variable",
        y_label="Y Variable"
    )

    print(f"Created plot: {plot_id}")
    plt.show_plot(plot_id)

    return plot_id


def example_multiple_series():
    """Create a plot with multiple series"""
    print("Creating multi-series plot...")

    # Get PlotManager instance
    manager = plt.PlotManager.get_instance()

    # Create plot
    config = plt.PlotConfig()
    config.title = "Multiple Functions"
    config.x_label = "X"
    config.y_label = "Y"
    config.type = plt.PlotType.Line
    config.backend = plt.BackendType.ImPlot

    plot_id = manager.create_plot(config)

    # Add multiple series
    x = np.linspace(-np.pi, np.pi, 200)

    # Sine series
    dataset_sin = plt.PlotDataset()
    dataset_sin.add_series("sin")
    series_sin = dataset_sin.get_series("sin")
    for xi, yi in zip(x, np.sin(x)):
        series_sin.add_point(float(xi), float(yi))

    manager.add_dataset(plot_id, "sin", dataset_sin)

    # Cosine series
    dataset_cos = plt.PlotDataset()
    dataset_cos.add_series("cos")
    series_cos = dataset_cos.get_series("cos")
    for xi, yi in zip(x, np.cos(x)):
        series_cos.add_point(float(xi), float(yi))

    manager.add_dataset(plot_id, "cos", dataset_cos)

    print(f"Created multi-series plot: {plot_id}")
    plt.show_plot(plot_id)

    return plot_id


def example_bar_chart():
    """Create a bar chart"""
    print("Creating bar chart...")

    # Data
    categories = [1, 2, 3, 4, 5]
    values = [23, 45, 56, 78, 32]

    # Create plot
    plot_id = plt.plot_bar(
        categories, values,
        title="Sales by Region",
        x_label="Region",
        y_label="Sales ($1000s)"
    )

    print(f"Created bar chart: {plot_id}")
    plt.show_plot(plot_id)

    return plot_id


def example_histogram():
    """Create a histogram"""
    print("Creating histogram...")

    # Generate normal distribution data
    np.random.seed(42)
    data = np.random.randn(1000)

    # Create plot
    plot_id = plt.plot_histogram(
        data,
        bins=30,
        title="Normal Distribution",
        x_label="Value",
        y_label="Frequency"
    )

    print(f"Created histogram: {plot_id}")
    plt.show_plot(plot_id)

    return plot_id


def example_realtime_update():
    """Demonstrate real-time plot updates"""
    print("Creating real-time plot...")

    # Create empty plot
    manager = plt.PlotManager.get_instance()

    config = plt.PlotConfig()
    config.title = "Real-Time Data Stream"
    config.x_label = "Time"
    config.y_label = "Value"
    config.type = plt.PlotType.Line
    config.backend = plt.BackendType.ImPlot

    plot_id = manager.create_plot(config)

    # Simulate streaming data
    # NOTE: In a real application, you would call update_realtime_plot
    # from a separate thread or timer callback
    for i in range(100):
        t = i * 0.1
        value = np.sin(t) + np.random.randn() * 0.1

        manager.update_realtime_plot(plot_id, t, value, "sensor_data")

    print(f"Created real-time plot: {plot_id}")
    plt.show_plot(plot_id)

    return plot_id


def example_statistics():
    """Calculate and display statistics"""
    print("Calculating statistics...")

    # Create plot with data
    x = np.linspace(0, 10, 100)
    y = np.random.randn(100) * 2 + 5

    plot_id = plt.plot_scatter(x, y, title="Data with Statistics")

    # Calculate statistics
    manager = plt.PlotManager.get_instance()
    stats = manager.calculate_statistics(plot_id, "data")

    print(f"\nStatistics for '{plot_id}':")
    print(f"  Min:     {stats.min:.3f}")
    print(f"  Max:     {stats.max:.3f}")
    print(f"  Mean:    {stats.mean:.3f}")
    print(f"  Median:  {stats.median:.3f}")
    print(f"  Std Dev: {stats.std_dev:.3f}")
    print(f"  Q1:      {stats.q1:.3f}")
    print(f"  Q3:      {stats.q3:.3f}")

    plt.show_plot(plot_id)

    return plot_id


def example_save_plot():
    """Save a plot to file"""
    print("Creating and saving plot...")

    # Create plot
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x) * np.exp(-x/10)

    plot_id = plt.plot_line(x, y, title="Damped Sine Wave")

    # Save to file (requires Matplotlib backend)
    manager = plt.PlotManager.get_instance()

    if manager.is_backend_available(plt.BackendType.Matplotlib):
        filepath = "damped_sine.png"
        success = manager.save_plot(plot_id, filepath)

        if success:
            print(f"Plot saved to: {filepath}")
        else:
            print("Failed to save plot")
    else:
        print("Matplotlib backend not available")

    plt.show_plot(plot_id)

    return plot_id


def main():
    """Run all examples"""
    print("=" * 60)
    print("CyxWiz Plotting System - Basic Examples")
    print("=" * 60)

    # Initialize Python backend if needed
    manager = plt.PlotManager.get_instance()
    manager.initialize_python_backend()

    try:
        # Run examples
        example_line_plot()
        print()

        example_scatter_plot()
        print()

        example_multiple_series()
        print()

        example_bar_chart()
        print()

        example_histogram()
        print()

        example_realtime_update()
        print()

        example_statistics()
        print()

        example_save_plot()
        print()

        # Summary
        print("=" * 60)
        print(f"Total plots created: {manager.get_plot_count()}")
        print(f"Plot IDs: {manager.get_all_plot_ids()}")
        print("=" * 60)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup (optional)
        # manager.clear_all_plots()
        # plt.clear_plot_windows()
        pass


if __name__ == "__main__":
    main()
