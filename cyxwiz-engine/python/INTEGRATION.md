# Python Plotting Bindings - Integration Guide

## Overview

This document describes how the Python plotting bindings integrate with the CyxWiz Engine and how to use them from Python scripts.

## Architecture

### Components

1. **plot_bindings.cpp**: The pybind11 wrapper that exposes C++ plotting API to Python
2. **PlotManager**: Singleton C++ class managing all plots
3. **PlotDataset**: Data container for plot series
4. **PlotWindow**: GUI window for displaying plots (optional integration)

### Build System

The Python module is built as a separate shared library (`cyxwiz_plotting.pyd` on Windows, `.so` on Linux/macOS) and can be imported independently of the main Engine executable.

#### CMake Configuration

```cmake
pybind11_add_module(cyxwiz_plotting python/plot_bindings.cpp)
target_link_libraries(cyxwiz_plotting PRIVATE imgui implot spdlog)
```

#### Build Output

After building, the module is placed in:
- Windows: `build/windows-debug/python/cyxwiz_plotting.pyd`
- Linux: `build/linux-debug/python/cyxwiz_plotting.so`
- macOS: `build/macos-debug/python/cyxwiz_plotting.so`

## Integration with Engine

### Method 1: Import from Built Module

Add the build output directory to Python's module search path:

```python
import sys
sys.path.insert(0, 'build/windows-debug/python')  # Adjust for your platform

import cyxwiz_plotting as plt
```

### Method 2: Install Module

After building, install the module:

```bash
cmake --build build/windows-debug --target install
```

Then import normally (if installation directory is in PYTHONPATH).

### Method 3: Embedded in Engine

The Engine's embedded Python interpreter should automatically add the module path. From the Engine console:

```python
import cyxwiz_plotting as plt
```

## Thread Safety Considerations

### Thread-Safe Operations

These can be called from any thread (e.g., training loops, background workers):

- `PlotManager.create_plot()`
- `PlotManager.add_dataset()`
- `PlotManager.update_realtime_plot()`
- `PlotDataset` manipulation
- Data conversion functions

### GUI Thread Only

These **must** be called from the main GUI thread:

- `show_plot()` - Creates PlotWindow in GUI
- `PlotManager.render_implot()` - Renders using ImGui/ImPlot

### Example: Background Training with Real-Time Plotting

```python
import threading
import cyxwiz_plotting as plt
import numpy as np

# Create plot on main thread
manager = plt.PlotManager.get_instance()
config = plt.PlotConfig()
config.title = "Training Progress"
config.type = plt.PlotType.Line
plot_id = manager.create_plot(config)

# Training function (runs in background)
def train_model():
    for epoch in range(100):
        loss = compute_loss()  # Your training code

        # This is THREAD-SAFE
        manager.update_realtime_plot(plot_id, epoch, loss, "train_loss")

        time.sleep(0.1)

# Start training in background
thread = threading.Thread(target=train_model)
thread.start()

# Show plot on MAIN THREAD
plt.show_plot(plot_id)  # Must be on GUI thread
```

## Data Conversion

The bindings automatically handle conversion between Python and C++ types:

### Supported Input Types

- **Python lists**: `[1.0, 2.0, 3.0]`
- **NumPy arrays**: `np.array([1, 2, 3])`
- **NumPy functions**: `np.linspace()`, `np.random.randn()`, etc.

### Conversion Function

The `to_double_vector()` helper handles both:

```cpp
std::vector<double> to_double_vector(const py::object& obj) {
    if (py::isinstance<py::array>(obj)) {
        // NumPy array - direct memory access
    } else if (py::isinstance<py::list>(obj)) {
        // Python list - iterate and convert
    }
}
```

### Performance Notes

- NumPy arrays are preferred (faster, no copy if already float64)
- Large datasets: use NumPy for best performance
- Small datasets: Python lists are fine

## PlotWindow Integration

### Current Implementation

The bindings create a global registry of PlotWindow instances:

```cpp
static std::vector<std::shared_ptr<cyxwiz::PlotWindow>> g_python_plot_windows;
```

When `show_plot()` is called, a new PlotWindow is created and added to this registry.

### Integration with MainWindow

To fully integrate with the Engine's docking system, MainWindow needs to:

1. Access the Python plot windows registry
2. Render each window in the docking layout
3. Handle window lifetime (close button, etc.)

#### Recommended MainWindow Integration

```cpp
// In main_window.h
#include "../python/plot_bindings.h"  // For get_python_plot_windows()

// In main_window.cpp Render()
void MainWindow::Render() {
    RenderDockSpace();

    // Existing panels
    if (node_editor_) node_editor_->Render();
    if (console_) console_->Render();
    // ...

    // Python-created plot windows
    auto& python_plots = get_python_plot_windows();
    for (auto& plot_window : python_plots) {
        if (plot_window) {
            plot_window->Render();
        }
    }
}
```

### Alternative: Event-Based Integration

For cleaner separation, use callbacks:

```cpp
// In plot_bindings.cpp
using PlotWindowCallback = std::function<void(std::shared_ptr<PlotWindow>)>;
static PlotWindowCallback g_window_created_callback;

void set_plot_window_callback(PlotWindowCallback callback) {
    g_window_created_callback = callback;
}

void show_plot(const std::string& plot_id) {
    auto window = std::make_shared<PlotWindow>(...);

    if (g_window_created_callback) {
        g_window_created_callback(window);
    }
}
```

Then in MainWindow:

```cpp
// At initialization
set_plot_window_callback([this](auto window) {
    python_plot_windows_.push_back(window);
});
```

## Error Handling

### C++ Exceptions to Python

pybind11 automatically converts C++ exceptions to Python exceptions:

```cpp
// C++ code
if (!mgr.HasPlot(plot_id)) {
    throw std::runtime_error("Plot ID not found: " + plot_id);
}
```

```python
# Python code
try:
    plt.show_plot("invalid_id")
except RuntimeError as e:
    print(f"Error: {e}")  # "Error: Plot ID not found: invalid_id"
```

### Common Exceptions

- `std::invalid_argument`: Invalid data format (wrong array types, mismatched sizes)
- `std::runtime_error`: Plot not found, backend unavailable, operation failed
- `pybind11::cast_error`: Type conversion failed (rare, usually caught by validation)

## Memory Management

### Object Lifetime

- **PlotManager**: Singleton, lives for entire program
- **Plots**: Managed by PlotManager, deleted explicitly or on shutdown
- **PlotDataset**: Copied when added to plot (safe to delete Python reference)
- **PlotWindow**: Shared pointers in global registry, cleaned on `clear_plot_windows()`

### Cleanup

```python
# Delete specific plot
manager.delete_plot(plot_id)

# Delete all plots (data only, not windows)
manager.clear_all_plots()

# Clear plot windows (GUI elements)
plt.clear_plot_windows()

# Full cleanup
manager.shutdown_python_backend()
```

## Backend Selection

### ImPlot Backend (Default)

- Real-time rendering
- Interactive (zoom, pan)
- Integrated into Engine GUI
- No file output directly

### Matplotlib Backend

- Offline rendering
- Save to file (PNG, PDF, SVG)
- Show in external window
- Requires Python matplotlib package

### Switching Backends

```python
# Set default for new plots
manager.set_default_backend(plt.BackendType.Matplotlib)

# Or per-plot
config = plt.PlotConfig()
config.backend = plt.BackendType.ImPlot
```

## Extending the Bindings

### Adding New Plot Types

1. **Add enum value** in PlotManager::PlotType (C++)
2. **Export in bindings**:
   ```cpp
   .value("NewType", PlotManager::PlotType::NewType, "Description")
   ```
3. **Add convenience function** (optional):
   ```cpp
   std::string plot_new_type(const py::object& data, ...) {
       // Implementation
   }
   m.def("plot_new_type", &plot_new_type, ...);
   ```

### Adding 3D Plotting Support

The 3D structures (Plot3D::SurfaceData, etc.) are already defined in C++. To expose them:

```cpp
// In plot_bindings.cpp
py::class_<Plot3D::Point3D>(m, "Point3D")
    .def(py::init<double, double, double>())
    .def_readwrite("x", &Plot3D::Point3D::x)
    .def_readwrite("y", &Plot3D::Point3D::y)
    .def_readwrite("z", &Plot3D::Point3D::z);

py::class_<Plot3D::SurfaceData>(m, "SurfaceData")
    .def(py::init<>())
    .def_readwrite("x_grid", &Plot3D::SurfaceData::x_grid)
    // ... etc
```

Then add convenience functions for 3D plots.

## Testing

### Unit Tests (C++)

Test the bindings from C++ side:

```cpp
// In tests/python_bindings_test.cpp
#include <pybind11/embed.h>

TEST_CASE("Python bindings basic") {
    py::scoped_interpreter guard{};

    auto plt = py::module::import("cyxwiz_plotting");
    auto manager = plt.attr("PlotManager").attr("get_instance")();

    // Test creation
    auto plot_id = manager.attr("create_plot")(/* ... */);
    // ... assertions
}
```

### Integration Tests (Python)

```python
# tests/test_plotting_python.py
import cyxwiz_plotting as plt
import numpy as np

def test_basic_plot():
    x = [1, 2, 3]
    y = [1, 4, 9]
    plot_id = plt.plot_line(x, y)
    assert plot_id != ""

def test_numpy_integration():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plot_id = plt..plot_line(x, y)
    assert plot_id != ""

# Run with pytest
```

## Troubleshooting

### "Module not found: cyxwiz_plotting"

**Solution**: Add build output to PYTHONPATH:
```python
import sys
sys.path.insert(0, 'build/windows-debug/python')
```

### "Undefined symbol: PlotManager::GetInstance()"

**Solution**: Ensure plotting source files are linked:
```cmake
target_link_libraries(cyxwiz_plotting PRIVATE
    # Add source files or link against main engine lib
)
```

### Circular dependency (Engine includes bindings, bindings include Engine)

**Solution**: Use forward declarations and separate headers:
```cpp
// plot_bindings_api.h
std::vector<std::shared_ptr<PlotWindow>>& get_python_plot_windows();
```

### Crashes when calling from Python

**Common causes**:
1. GUI functions called from background thread → Use main thread only
2. Invalid plot_id → Check with `has_plot()` first
3. Data size mismatch → Validate array sizes match

## Performance Optimization

### Large Datasets

For datasets > 10,000 points:

1. **Use NumPy arrays** (avoid Python lists)
2. **Batch updates** instead of point-by-point
3. **Consider downsampling** for visualization
4. **Use CircularBuffer** for streaming data

### Memory Efficiency

```python
# Good: Single allocation
x = np.linspace(0, 10, 10000)
y = np.sin(x)
plot_id = plt.plot_line(x, y)

# Bad: Multiple allocations and conversions
for i in range(10000):
    manager.update_realtime_plot(plot_id, i*0.001, np.sin(i*0.001))
```

## Future Enhancements

1. **Async plot updates** from background threads with queue
2. **Live plot sharing** between Engine instances
3. **Plot templates** (predefined styles)
4. **Animation support** (frame-by-frame rendering)
5. **Interactive callbacks** (click, hover events)
6. **3D plotting** full integration
7. **Subplots** (multiple plots in one window)
8. **Annotations** (text, arrows, shapes)

## References

- pybind11 documentation: https://pybind11.readthedocs.io/
- ImPlot documentation: https://github.com/epezent/implot
- NumPy C API: https://numpy.org/doc/stable/reference/c-api/
- CyxWiz plotting system: `docs/plotting_system.md`
