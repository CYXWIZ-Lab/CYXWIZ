# Phase 7, Session 1: ImPlot Integration - Real-time Training Visualization

## Status: IN PROGRESS

**Date**: November 17, 2025
**Session**: Phase 7, Session 1
**Focus**: Real-time training visualization with ImPlot

---

## Objectives

✓ Integrate ImPlot library into the build system
✓ Create TrainingPlotPanel GUI component for real-time metrics visualization
⏳ Create Python bindings for TrainingPlotPanel
⏳ Integrate with ML backend training pipeline
⏳ Test with XOR training example

---

## Discoveries

### Existing Plotting Infrastructure

We discovered that CyxWiz Engine **already has a comprehensive plotting system**:

1. **ImPlot Integration**
   - Already in `vcpkg.json` as a dependency
   - Headers installed: `implot.h`, `implot_internal.h`
   - Already linked to Engine in CMakeLists.txt

2. **PlotManager Class** (`src/plotting/plot_manager.h`)
   - Singleton manager for all plotting operations
   - Supports multiple backends: ImPlot (real-time) and Matplotlib (offline)
   - Multiple plot types: Line, Scatter, Histogram, Bar, Heatmap, BoxPlot, etc.
   - Statistics calculation capabilities
   - Dataset management
   - Real-time plot updates: `UpdateRealtimePlot()`

3. **PlotWindow Class** (`src/gui/panels/plot_window.h`)
   - Reusable dockable window for displaying plots
   - Supports 2D and 3D plot types
   - Auto-generation of sample data
   - Plot export functionality
   - Matplotlib texture rendering for offline plots

4. **Backend System**
   - `implot_backend.cpp/h` - Real-time plotting with ImPlot
   - `matplotlib_backend.cpp/h` - Python matplotlib integration
   - `plot_backend.h` - Abstract backend interface

This means the plotting infrastructure is **already complete** and battle-tested!

---

## What We Built

### TrainingPlotPanel - Purpose-Built for ML Training

Since the general plotting system exists, we created a **specialized panel** for ML training visualization:

**File**: `cyxwiz-engine/src/gui/panels/training_plot_panel.h/cpp`

**Features**:
1. **Real-time Metric Tracking**
   - Training loss and validation loss (dual-line plots)
   - Training accuracy and validation accuracy
   - Custom metrics support (unlimited additional metrics)

2. **Thread-Safe Data Updates**
   - Mutex-protected data structures
   - Safe to update from training thread
   - Methods:
     - `AddLossPoint(epoch, train_loss, val_loss)`
     - `AddAccuracyPoint(epoch, train_acc, val_acc)`
     - `AddCustomMetric(metric_name, epoch, value)`

3. **UI Controls**
   - Clear All button
   - Export to CSV
   - Auto-scaling toggle
   - Show/hide individual plots
   - Statistics display (last 10 epochs: mean, min, max)

4. **Data Management**
   - Configurable max points (default: 1000) to prevent memory issues
   - Automatic data trimming when limit reached
   - Separate storage for each metric series

5. **Visual Styling**
   - Color-coded lines:
     - Training Loss: Red
     - Validation Loss: Blue
     - Training Accuracy: Green
     - Validation Accuracy: Yellow
   - Golden ratio-based color generation for custom metrics
   - 2px line thickness for visibility

---

## Implementation Details

### TrainingPlotPanel Architecture

```cpp
class TrainingPlotPanel : public Panel {
    struct MetricSeries {
        std::vector<int> epochs;
        std::vector<double> values;
        std::string name;
        ImVec4 color;
    };

    // Thread-safe data storage
    MetricSeries train_loss_;
    MetricSeries val_loss_;
    MetricSeries train_accuracy_;
    MetricSeries val_accuracy_;
    std::vector<MetricSeries> custom_metrics_;

    std::mutex data_mutex_;  // Thread safety
};
```

### Integration into MainWindow

1. Added forward declaration in `main_window.h`
2. Added member: `std::unique_ptr<TrainingPlotPanel> training_plot_panel_`
3. Initialized in constructor
4. Rendered in `Render()` method

### CMakeLists.txt Updates

Added to ENGINE_SOURCES:
```cmake
src/gui/panels/training_plot_panel.cpp
```

Added to ENGINE_HEADERS:
```cmake
src/gui/panels/training_plot_panel.h
```

---

## Next Steps

### 1. Verify Build Success

- ✓ Compilation started
- ⏳ Waiting for build completion
- ⏳ Fix any compilation errors

### 2. Create Python Bindings

Need to expose TrainingPlotPanel to Python so scripts can update plots:

```python
# Example usage
import cyxwiz_plotting as plot

# Create or get training plot panel
panel = plot.get_training_plot_panel()

# Update during training loop
for epoch in range(num_epochs):
    loss = train_one_epoch()
    panel.add_loss_point(epoch, loss)
```

**Files to create**:
- Extend `cyxwiz-engine/python/plot_bindings.cpp` with TrainingPlotPanel bindings

### 3. Integrate with ML Backend

Connect TrainingPlotPanel to the ML training loop:

```python
import pycyxwiz as cx  # ML backend
import cyxwiz_plotting as plot  # Plotting

# Setup
network = cx.Sequential([...])
optimizer = cx.SGDOptimizer(learning_rate=0.1)
criterion = cx.MSELoss()
plot_panel = plot.get_training_plot_panel()

# Training loop
for epoch in range(num_epochs):
    # Forward
    output = network.forward(X)
    loss_val = criterion.forward(output, y)

    # Update plot
    plot_panel.add_loss_point(epoch, loss_val.to_numpy()[0])

    # Backward
    grad = criterion.backward(output, y)
    network.backward(grad)
    optimizer.step(network.get_parameters(), network.get_gradients())
```

### 4. Enhanced XOR Training Example

Create `test_training_xor_visualized.py`:
- Same training loop as before
- Add real-time plot updates
- Display in Engine GUI
- User can watch training progress live

### 5. Testing Plan

**Unit Tests**:
- TrainingPlotPanel creation and initialization
- Thread-safe data updates
- Data trimming when max_points exceeded
- Statistics calculation accuracy

**Integration Tests**:
- Python bindings work correctly
- Plot updates from Python
- Multi-threaded training with plot updates
- Export to CSV functionality

**Manual Tests**:
- Open Engine GUI
- Run visualized training script
- Verify plots update in real-time
- Test UI controls (clear, export, toggles)
- Verify no lag or frame drops

---

## Technical Notes

### Thread Safety Considerations

TrainingPlotPanel uses `std::mutex` to protect data:
- Training can run on a background thread
- Plot updates are queued and applied safely
- ImGui rendering happens on main thread only

### Performance Optimization

1. **Limited Data Points**: Default 1000 points prevents unbounded memory growth
2. **Batch Updates**: Could add `AddBatch()` method for efficiency
3. **Render Frequency**: ImPlot is very efficient, 60 FPS is achievable

### Design Decisions

**Why not use existing PlotManager?**
- TrainingPlotPanel is simpler and purpose-built
- Direct ImPlot integration (no abstraction layer)
- Optimized for the specific use case of training metrics
- PlotManager is more general-purpose and feature-rich

**Could we integrate them?**
- Yes! TrainingPlotPanel could use PlotManager internally
- Future enhancement: refactor to use PlotManager backend
- For now, keeping it simple for rapid prototyping

---

## File Changes Summary

### New Files
- `cyxwiz-engine/src/gui/panels/training_plot_panel.h` (68 lines)
- `cyxwiz-engine/src/gui/panels/training_plot_panel.cpp` (322 lines)
- `PHASE7_PLAN.md` (comprehensive phase plan)
- `FUTURE_WORK_ROADMAP.md` (long-term project roadmap)

### Modified Files
- `cyxwiz-engine/CMakeLists.txt` - Added training_plot_panel to sources/headers
- `cyxwiz-engine/src/gui/main_window.h` - Added TrainingPlotPanel forward declaration and member
- `cyxwiz-engine/src/gui/main_window.cpp` - Added include, initialization, and rendering

### Total Lines Added: ~450 lines

---

## Build Status

**Current**: Building cyxwiz-engine with TrainingPlotPanel...

**Expected Issues**:
- Possible missing ImPlot includes
- Potential namespace issues
- Link errors if ImPlot not properly linked

**Resolution Strategy**:
1. Check compilation errors
2. Add missing includes
3. Verify ImPlot linkage in CMakeLists.txt
4. Rebuild and test

---

## Session Time

- **Started**: 10:00 AM
- **Current**: 10:25 AM
- **Elapsed**: 25 minutes
- **Estimated Completion**: 11:30 AM (1.5 hours total)

---

## Key Takeaways

1. **Don't reinvent the wheel**: CyxWiz already had extensive plotting capabilities
2. **Purpose-built is better**: TrainingPlotPanel is simpler than general PlotManager for this use case
3. **Thread safety is critical**: ML training runs on background threads
4. **Start simple, iterate**: Basic functionality first, then enhance

---

## Next Session Preview

**Phase 7, Session 2: ImNodes Integration - Visual Node Editor**

Build a visual drag-and-drop interface for constructing neural network architectures:
- Node types: Input, Linear, Conv2D, Activation, Loss, Output
- Visual connections representing data flow
- Property editors for layer parameters
- Graph-to-model conversion
- Model validation and error checking

This will complement TrainingPlotPanel by providing a visual way to build models that can then be trained with real-time visualization!

---

**Last Updated**: November 17, 2025, 10:25 AM
