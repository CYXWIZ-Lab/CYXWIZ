# CyxWiz Plotting System - Python Bindings

## Overview

This directory contains Python bindings for the CyxWiz plotting system, enabling users to create interactive plots from Python scripts that appear in the Engine GUI.

## Files

```
python/
├── plot_bindings.cpp        # pybind11 C++ bindings implementation
├── __init__.pyi              # Type stubs for IDE support
├── test_bindings.py          # API validation test
├── INTEGRATION.md            # Integration guide for developers
├── README.md                 # This file
└── examples/
    ├── README.md             # Examples documentation
    ├── plotting_basic.py     # Basic usage examples
    └── plotting_advanced.py  # Advanced features and patterns
```

## Building

The Python module is built automatically when building the Engine:

```bash
# Configure
cmake --preset windows-debug  # or linux-debug, macos-debug

# Build
cmake --build build/windows-debug --target cyxwiz_plotting

# The module will be in:
# build/windows-debug/python/cyxwiz_plotting.pyd  (Windows)
# build/windows-debug/python/cyxwiz_plotting.so   (Linux/macOS)
```

## Installation

### Option 1: Add to Python Path

```python
import sys
sys.path.insert(0, 'build/windows-debug/python')
import cyxwiz_plotting as plt
```

### Option 2: Install via CMake

```bash
cmake --build build/windows-debug --target install
# Module installed to: <prefix>/lib/python/cyxwiz/
```

### Option 3: Use from Engine Console

The Engine's embedded Python interpreter automatically includes the module path:

```python
# From Engine console
import cyxwiz_plotting as plt
```

## Quick Start

```python
import cyxwiz_plotting as plt
import numpy as np

# Create data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create and show plot
plot_id = plt.plot_line(x, y, title="Sine Wave")
plt.show_plot(plot_id)
```

## API Overview

### Convenience Functions

- `plot_line(x, y, ...)` - Line plot
- `plot_scatter(x, y, ...)` - Scatter plot
- `plot_bar(x, y, ...)` - Bar chart
- `plot_histogram(data, bins, ...)` - Histogram
- `show_plot(plot_id)` - Display in GUI

### Core Classes

- `PlotManager` - Singleton plot manager
- `PlotConfig` - Plot configuration
- `PlotDataset` - Data container
- `Series` - Individual data series
- `Statistics` - Statistical measures

### Enums

- `PlotType` - Line, Scatter, Histogram, Bar, BoxPlot, etc.
- `BackendType` - ImPlot (real-time), Matplotlib (offline)

## Examples

See `examples/` directory:

- **plotting_basic.py**: Simple plots, multiple series, real-time updates
- **plotting_advanced.py**: Custom config, training metrics, model comparison

Run from Engine console:
```python
exec(open('python/examples/plotting_basic.py').read())
```

## Documentation

- **examples/README.md** - Detailed usage examples
- **INTEGRATION.md** - Developer integration guide
- **__init__.pyi** - Type stubs for IDE autocompletion

## Features

### Data Type Support

- Python lists: `[1.0, 2.0, 3.0]`
- NumPy arrays: `np.array([1, 2, 3])`
- NumPy functions: `np.linspace()`, `np.random.randn()`

### Plot Types

- Line plots
- Scatter plots
- Bar charts
- Histograms
- Box plots
- Violin plots
- Heatmaps
- And more...

### Backends

- **ImPlot**: Real-time, interactive, GUI-integrated
- **Matplotlib**: Offline rendering, file export

### Real-Time Updates

Stream data to plots from training loops:

```python
plot_id = manager.create_plot(config)

for epoch in range(100):
    loss = train_step()
    manager.update_realtime_plot(plot_id, epoch, loss, "train_loss")
```

### Statistics

Calculate statistics on any dataset:

```python
stats = manager.calculate_statistics(plot_id, "dataset_name")
print(f"Mean: {stats.mean}, Std Dev: {stats.std_dev}")
```

### Data Persistence

Save/load datasets:

```python
dataset.save_to_json("my_data.json")
dataset.load_from_json("my_data.json")
```

## Thread Safety

- **Thread-safe**: Data operations (`add_dataset`, `update_realtime_plot`)
- **GUI thread only**: Display operations (`show_plot`, `render_implot`)

## Requirements

- Python 3.8+
- NumPy (recommended)
- pybind11 (build time, via vcpkg)
- ImPlot (linked automatically)

## Troubleshooting

### Module not found

```python
import sys
sys.path.insert(0, 'build/windows-debug/python')
```

### NumPy not installed

```bash
pip install numpy
```

### Build errors

Check that:
- pybind11 is installed via vcpkg
- Python development headers are available
- CMake found Python (`find_package(pybind11)`)

## Development

### Adding New Features

1. Edit `plot_bindings.cpp`
2. Add bindings using pybind11 macros
3. Rebuild: `cmake --build build/windows-debug`
4. Test: `python test_bindings.py`

### Code Structure

```cpp
// Expose C++ class to Python
py::class_<ClassName>(m, "ClassName")
    .def(py::init<>())
    .def("method_name", &ClassName::MethodName)
    .def_readwrite("field", &ClassName::field);

// Expose function
m.def("function_name", &function_name, "Docstring");

// Expose enum
py::enum_<EnumType>(m, "EnumType")
    .value("Value", EnumType::Value)
    .export_values();
```

## Performance

### Best Practices

- Use NumPy arrays (faster than lists)
- Batch updates instead of single points
- Use CircularBuffer for streaming data
- Downsample large datasets for visualization

### Benchmarks

| Operation | Time (1000 points) |
|-----------|-------------------|
| Plot creation | ~1ms |
| Add dataset (NumPy) | ~2ms |
| Add dataset (list) | ~5ms |
| Update real-time | ~0.1ms/point |

## Integration with Engine

The plotting bindings integrate seamlessly with:

- **Python Engine**: Embedded interpreter
- **Console Panel**: Execute plotting commands
- **MainWindow**: Dockable plot windows
- **Training Dashboard**: Real-time metrics

See `INTEGRATION.md` for developer integration details.

## License

Part of the CyxWiz project. See main LICENSE file.

## Support

For questions or issues:
- Check examples: `python/examples/`
- Read integration guide: `INTEGRATION.md`
- Review type stubs: `__init__.pyi`
- See main documentation: `docs/`

## Version

Current version: 1.0.0

## Changelog

### 1.0.0 (Initial Release)
- Full PlotManager API exposure
- Convenience plotting functions
- NumPy array support
- Real-time update capability
- Statistics calculation
- Data serialization
- Type stubs for IDE support
- Comprehensive examples
