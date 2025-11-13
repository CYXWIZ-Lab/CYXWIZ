# CyxWiz Plotting System - Python Examples

This directory contains example scripts demonstrating the CyxWiz plotting Python bindings.

## Overview

The CyxWiz plotting system provides a powerful Python API for creating visualizations directly in the Engine GUI. All plots are rendered using ImPlot (real-time) or Matplotlib (offline export).

## Files

### `plotting_basic.py`
Basic examples covering:
- Simple line plots
- Scatter plots
- Bar charts
- Histograms
- Multiple series in one plot
- Real-time data streaming
- Statistics calculation
- Saving plots to files

**Run from Engine console:**
```python
exec(open('python/examples/plotting_basic.py').read())
```

### `plotting_advanced.py`
Advanced features:
- Custom plot configurations
- Dynamic plot updates
- ML training metrics visualization
- Model comparison charts
- Data serialization (JSON)
- Plot lifecycle management
- NumPy array integration

**Run from Engine console:**
```python
exec(open('python/examples/plotting_advanced.py').read())
```

## Quick Start

### Import the module

```python
import cyxwiz_plotting as plt
import numpy as np
```

### Create a simple plot

```python
# Generate data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create and show plot
plot_id = plt.plot_line(x, y, title="Sine Wave")
plt.show_plot(plot_id)
```

### Use the PlotManager directly

```python
# Get manager instance
manager = plt.PlotManager.get_instance()

# Create custom plot
config = plt.PlotConfig()
config.title = "My Plot"
config.x_label = "X Axis"
config.y_label = "Y Axis"
config.type = plt.PlotType.Scatter
config.backend = plt.BackendType.ImPlot

plot_id = manager.create_plot(config)

# Add data
dataset = plt.PlotDataset()
dataset.add_series("my_data")
series = dataset.get_series("my_data")
series.add_point(1.0, 2.0)
series.add_point(2.0, 4.0)
series.add_point(3.0, 6.0)

manager.add_dataset(plot_id, "my_data", dataset)

# Display
plt.show_plot(plot_id)
```

## API Reference

### Convenience Functions

- `plot_line(x_data, y_data, ...)` - Create line plot
- `plot_scatter(x_data, y_data, ...)` - Create scatter plot
- `plot_bar(x_data, y_data, ...)` - Create bar chart
- `plot_histogram(data, bins=10, ...)` - Create histogram
- `show_plot(plot_id)` - Display plot in GUI

### PlotManager Methods

- `get_instance()` - Get singleton instance
- `create_plot(config)` - Create new plot
- `delete_plot(plot_id)` - Delete plot
- `has_plot(plot_id)` - Check if plot exists
- `add_dataset(plot_id, name, dataset)` - Add data
- `update_realtime_plot(plot_id, x, y, series_name)` - Stream data
- `save_plot(plot_id, filepath)` - Save to file
- `calculate_statistics(plot_id, dataset_name)` - Get stats
- `get_all_plot_ids()` - List all plots

### PlotTypes

- `Line` - Line plot
- `Scatter` - Scatter plot
- `Histogram` - Histogram
- `Bar` - Bar chart
- `BoxPlot` - Box and whisker plot
- `Heatmap` - 2D heatmap
- `KDE` - Kernel density estimation
- And more...

### BackendTypes

- `ImPlot` - Real-time rendering in GUI (recommended)
- `Matplotlib` - Offline rendering for export

## Integration with Training

Example of plotting training metrics:

```python
import cyxwiz_plotting as plt

manager = plt.PlotManager.get_instance()

# Create loss plot
config = plt.PlotConfig()
config.title = "Training Loss"
config.type = plt.PlotType.Line
loss_plot = manager.create_plot(config)

# Training loop
for epoch in range(100):
    # ... train model ...
    loss = compute_loss()

    # Update plot
    manager.update_realtime_plot(loss_plot, epoch, loss, "train_loss")

plt.show_plot(loss_plot)
```

## Data Formats

The plotting system accepts:
- Python lists: `[1, 2, 3, 4]`
- NumPy arrays: `np.array([1, 2, 3, 4])`
- NumPy generated: `np.linspace(0, 10, 100)`

All are automatically converted to the internal format.

## Thread Safety

**Important**: Plot rendering (`show_plot()`, `render_implot()`) must be called from the main GUI thread. Data updates (`update_realtime_plot()`, `add_dataset()`) are thread-safe and can be called from background threads (e.g., training loops).

## Persistence

Plots created via Python are managed by the PlotManager and persist until:
1. Explicitly deleted: `manager.delete_plot(plot_id)`
2. All plots cleared: `manager.clear_all_plots()`
3. Engine shutdown

Plot data can be saved/loaded:
```python
dataset = plt.PlotDataset()
# ... add data ...
dataset.save_to_json("my_data.json")

# Later...
loaded = plt.PlotDataset()
loaded.load_from_json("my_data.json")
```

## Troubleshooting

### "Plot ID not found"
- Ensure the plot was created successfully
- Check that you're using the correct plot_id string

### "Expected numpy array or list"
- Verify your data is a list or NumPy array
- Convert if needed: `list(my_data)` or `np.array(my_data)`

### "X and Y data must have the same length"
- Check array sizes: `len(x) == len(y)`
- Use numpy: `assert x.shape == y.shape`

### Plots not appearing
- Make sure you called `plt.show_plot(plot_id)`
- Verify the Engine GUI is running
- Check console for error messages

## Further Examples

See the full example scripts for more detailed demonstrations:
- Basic examples: `plotting_basic.py`
- Advanced examples: `plotting_advanced.py`

## Support

For questions or issues, refer to:
- Engine documentation: `docs/engine.md`
- Plotting system docs: `docs/plotting_system.md`
- API reference: Generated from C++ headers
