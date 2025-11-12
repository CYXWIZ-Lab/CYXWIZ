# CyxWiz Plotting System

## Overview

The CyxWiz Plotting System provides a flexible, dual-mode plotting infrastructure for the Engine desktop client. It supports both **real-time plotting** (integrated into ImGui) and **offline/statistical plotting** (using matplotlib for publication-quality outputs).

## Architecture

### Design Philosophy

The plotting system is designed with the following principles:

1. **Dual-Mode Architecture**: Seamlessly switch between real-time (ImPlot) and offline (matplotlib) rendering
2. **Backend Abstraction**: Common API regardless of rendering backend
3. **Data-Centric**: Efficient data structures optimized for both streaming and batch operations
4. **Extensible**: Easy to add new plot types and backends
5. **Python Integration**: Leverage matplotlib's statistical capabilities via embedded Python

### Component Overview

```
cyxwiz-engine/src/plotting/
├── plot_manager.h/cpp          # Singleton manager (orchestrates everything)
├── plot_dataset.h/cpp          # Data structures (PlotDataset, CircularBuffer)
├── backends/
│   ├── plot_backend.h          # Abstract backend interface
│   ├── implot_backend.h/cpp    # Real-time plotting (Dear ImPlot)
│   └── matplotlib_backend.h/cpp # Offline plotting (Python/matplotlib)
├── plot_types/                 # Specialized plot implementations
│   ├── histogram.h/cpp         # Histogram plots
│   ├── kde.h/cpp              # Kernel Density Estimation
│   ├── boxplot.h/cpp          # Box-and-whisker plots
│   ├── qqplot.h/cpp           # Q-Q plots for normality testing
│   └── mosaic.h/cpp           # Mosaic plots for categorical data
└── test_data_generator.h/cpp  # Fake data for testing
```

## Core Components

### 1. PlotManager (Singleton)

**Purpose**: Central orchestrator for all plotting operations

**Key Responsibilities**:
- Plot lifecycle management (create, delete, update)
- Backend selection and initialization
- Data routing to appropriate backend
- Statistics calculation

**Usage Example**:
```cpp
#include "plotting/plot_manager.h"

using namespace cyxwiz::plotting;

// Get singleton instance
auto& mgr = PlotManager::GetInstance();

// Create a real-time plot
PlotManager::PlotConfig config;
config.title = "Training Loss";
config.x_label = "Epoch";
config.y_label = "Loss";
config.type = PlotManager::PlotType::Line;
config.backend = PlotManager::BackendType::ImPlot;

std::string plot_id = mgr.CreatePlot(config);

// Update with real-time data
mgr.UpdateRealtimePlot(plot_id, epoch, loss_value, "train_loss");

// Render in ImGui loop
mgr.RenderImPlot(plot_id);
```

### 2. PlotDataset

**Purpose**: Container for plot data with multiple series support

**Features**:
- Multiple named series per dataset
- Automatic x-axis generation (if not provided)
- JSON serialization/deserialization
- Efficient memory management

**Usage Example**:
```cpp
#include "plotting/plot_dataset.h"

PlotDataset dataset;

// Add series
dataset.AddSeries("accuracy");
dataset.AddSeries("val_accuracy");

// Add data points
auto* acc_series = dataset.GetSeries("accuracy");
acc_series->AddPoint(1.0, 0.85);  // epoch 1, acc 0.85
acc_series->AddPoint(2.0, 0.89);  // epoch 2, acc 0.89

// Save to file
dataset.SaveToJSON("training_metrics.json");
```

### 3. CircularBuffer

**Purpose**: Memory-bounded buffer for real-time streaming data

**Features**:
- Fixed capacity (prevents memory growth)
- Automatic oldest-data eviction
- Fast statistics (min, max, mean)
- Optimized for high-frequency updates

**Usage Example**:
```cpp
#include "plotting/plot_dataset.h"

CircularBuffer buffer(1000);  // Keep last 1000 points

// Stream data
for (int i = 0; i < 10000; ++i) {
    buffer.AddPoint(i, compute_loss(i));
}

// Get data for rendering
const double* x = buffer.GetXData();
const double* y = buffer.GetYData();
size_t count = buffer.GetSize();  // Will be 1000 max

// Statistics
double mean = buffer.GetMeanY();
double min = buffer.GetMinY();
```

### 4. PlotBackend (Abstract Interface)

**Purpose**: Common API for all rendering backends

**Key Methods**:
```cpp
class PlotBackend {
public:
    // Lifecycle
    virtual bool Initialize(int width, int height) = 0;
    virtual void Shutdown() = 0;

    // Plot lifecycle
    virtual void BeginPlot(const char* title) = 0;
    virtual void EndPlot() = 0;

    // Plotting primitives
    virtual void PlotLine(const char* label, const double* x, const double* y, int count) = 0;
    virtual void PlotScatter(...) = 0;
    virtual void PlotHistogram(...) = 0;
    virtual void PlotHeatmap(...) = 0;

    // Configuration
    virtual void SetAxisLabel(int axis, const char* label) = 0;
    virtual void SetAxisLimits(int axis, double min, double max) = 0;

    // Export
    virtual bool SaveToFile(const char* filepath) = 0;
};
```

### 5. ImPlotBackend

**Purpose**: Real-time plotting integrated into ImGui render loop

**When to Use**:
- Live training metrics (loss, accuracy curves)
- Interactive data exploration
- Dashboard visualizations
- Any plot that needs frequent updates (>1 Hz)

**Characteristics**:
- **Performance**: Optimized for 60+ FPS rendering
- **Integration**: Seamless ImGui docking and viewports
- **Limitations**: Limited statistical plot types, no export to file

**Example Integration**:
```cpp
// In your ImGui render loop
void TrainingDashboardPanel::Render() {
    ImGui::Begin("Training Dashboard");

    auto& mgr = PlotManager::GetInstance();
    mgr.RenderImPlot(loss_plot_id_);
    mgr.RenderImPlot(accuracy_plot_id_);

    ImGui::End();
}
```

### 6. MatplotlibBackend

**Purpose**: Offline, publication-quality plotting via Python

**When to Use**:
- Statistical analysis plots (KDE, Q-Q plots, violin plots)
- Publication-quality figures
- Complex multi-panel layouts
- Exporting to PNG/PDF/SVG

**Characteristics**:
- **Quality**: Publication-ready output with LaTeX support
- **Flexibility**: Full matplotlib API access
- **Trade-off**: Slower, not suitable for real-time updates
- **Requirement**: Python interpreter with matplotlib installed

**Example Usage**:
```cpp
// Create offline plot
PlotManager::PlotConfig config;
config.backend = PlotManager::BackendType::Matplotlib;
config.type = PlotManager::PlotType::Histogram;

auto plot_id = mgr.CreatePlot(config);

// Add data
PlotDataset dataset;
dataset.AddSeries("distribution");
// ... populate data ...
mgr.AddDataset(plot_id, "dist", dataset);

// Export to file
mgr.SavePlotToFile(plot_id, "distribution.png");
```

## Supported Plot Types

### Real-Time Plots (ImPlot)

| Plot Type | Use Case | Status |
|-----------|----------|--------|
| **Line** | Training curves, time series | ✅ Implemented |
| **Scatter** | Data points, correlations | ✅ Implemented |
| **Bars** | Comparisons, categorical data | ✅ Implemented |
| **Histogram** | Distributions | ✅ Implemented |
| **Heatmap** | Confusion matrix, weight matrices | ✅ Implemented |

### Statistical Plots (Matplotlib)

| Plot Type | Use Case | Status |
|-----------|----------|--------|
| **Histogram** | Data distributions | ✅ Implemented |
| **Box Plot** | Quartile analysis | ✅ Implemented |
| **Violin Plot** | Distribution density | ✅ Implemented |
| **KDE** | Kernel density estimation | ✅ Implemented |
| **Q-Q Plot** | Normality testing | ✅ Implemented |
| **Mosaic Plot** | Categorical relationships | 🚧 Stub only |
| **Stem-and-Leaf** | Small datasets | 📋 Planned |
| **Dot Chart** | Categorical comparisons | 📋 Planned |

## Data Flow

### Real-Time Mode (Training Loop)

```
Training Loop
    ↓
PlotManager::UpdateRealtimePlot(epoch, loss)
    ↓
PlotDataset (CircularBuffer)
    ↓
ImPlotBackend::PlotLine()
    ↓
ImGui Render Loop
    ↓
GPU Rendering (60 FPS)
```

### Offline Mode (Analysis)

```
Completed Training Data
    ↓
PlotManager::CreatePlot(matplotlib backend)
    ↓
PlotDataset::LoadFromJSON()
    ↓
MatplotlibBackend::PlotHistogram/KDE/etc.
    ↓
Python matplotlib execution
    ↓
SaveToFile("analysis.png")
```

## Implementation Status

### ✅ Completed

- Core architecture (PlotManager, PlotDataset, CircularBuffer)
- Backend abstraction (PlotBackend interface)
- ImPlot backend (real-time plotting)
- Matplotlib backend (skeleton with command queuing)
- Basic plot types (line, scatter, bar, histogram)
- Statistics calculation (mean, median, std dev, quartiles)

### 🚧 In Progress

- Python/pybind11 integration for matplotlib
- CMakeLists.txt integration
- Advanced plot types (mosaic, stem-and-leaf)
- Test data generator

### 📋 Planned

- GUI test panel for plot development
- Integration with TrainingDashboardPanel
- Plot templates and presets
- Animation support for ImPlot
- Multi-plot layouts
- Plot interaction (zoom, pan, selection)

## Adding a New Plot Type

### Step 1: Define in PlotManager

```cpp
// In plot_manager.h
enum class PlotType {
    Line,
    Scatter,
    // ... existing types ...
    YourNewPlotType  // Add here
};
```

### Step 2: Create Specialized Class (Optional)

```cpp
// In plot_types/your_plot.h
namespace cyxwiz::plotting {

class YourPlot {
public:
    static void Render(PlotBackend* backend,
                      const PlotDataset& data,
                      const PlotConfig& config);

private:
    static void PreprocessData(const PlotDataset& data);
};

} // namespace
```

### Step 3: Implement Backend Support

```cpp
// In implot_backend.cpp (if real-time)
void ImPlotBackend::PlotYourType(const char* label, ...) {
    // ImPlot rendering code
    ImPlot::PlotCustom(...);
}

// In matplotlib_backend.cpp (if statistical)
void MatplotlibBackend::PlotYourType(const char* label, ...) {
    std::ostringstream cmd;
    cmd << "ax.your_plot_type(data, ...)\n";
    python_commands_ += cmd.str();
}
```

### Step 4: Add to PlotBackend Interface

```cpp
// In plot_backend.h
virtual void PlotYourType(const char* label,
                         const double* data,
                         int count) = 0;
```

### Step 5: Wire Up in PlotManager

```cpp
// In plot_manager.cpp
void PlotManager::RenderPlot(const std::string& plot_id) {
    auto* plot = GetPlot(plot_id);

    switch (plot->config.type) {
        case PlotType::YourNewPlotType:
            plot->backend->PlotYourType(...);
            break;
        // ... other cases ...
    }
}
```

## Testing

### Using Test Data Generator

```cpp
#include "plotting/test_data_generator.h"

// Generate normal distribution
auto data = TestDataGenerator::GenerateNormal(1000, 0.0, 1.0);

// Generate sine wave
auto sine = TestDataGenerator::GenerateSineWave(500, 2.0, 0.1);

// Generate training curves
auto loss_curve = TestDataGenerator::GenerateTrainingCurve(100, 2.5, 0.01);
```

### Unit Testing

```cpp
// In tests/unit/test_plotting.cpp
TEST_CASE("PlotDataset basic operations", "[plotting]") {
    PlotDataset dataset;
    dataset.AddSeries("test");

    auto* series = dataset.GetSeries("test");
    series->AddPoint(1.0, 2.0);

    REQUIRE(series->Size() == 1);
    REQUIRE(series->x_data[0] == 1.0);
}
```

### Integration Testing

Create a test panel in the Engine:

```cpp
// In gui/panels/plot_test_panel.cpp
void PlotTestPanel::Render() {
    ImGui::Begin("Plot Testing");

    if (ImGui::Button("Test Line Plot")) {
        auto data = TestDataGenerator::GenerateSineWave(100);
        // ... create and render plot ...
    }

    ImGui::End();
}
```

## Python Integration

### Requirements

The matplotlib backend requires:

1. **Python 3.8+** installed and in PATH
2. **matplotlib** package: `pip install matplotlib scipy`
3. **pybind11** (managed by vcpkg)

### Initialization

```cpp
// In application startup
auto& mgr = PlotManager::GetInstance();
if (!mgr.InitializePythonBackend()) {
    spdlog::warn("Python backend unavailable - matplotlib plots disabled");
}
```

### Python Command Execution

The matplotlib backend queues Python commands as strings:

```cpp
// Commands are accumulated during BeginPlot/Plot*/EndPlot
python_commands_ += "import matplotlib.pyplot as plt\n";
python_commands_ += "ax.plot(x, y)\n";
python_commands_ += "plt.savefig('output.png')\n";

// TODO: Execute via pybind11 when integration is complete
// py::exec(python_commands_);
```

## Performance Considerations

### Real-Time Plotting (ImPlot)

- **Target**: 60 FPS with 1000+ data points per plot
- **Optimization**: Use CircularBuffer to limit data points
- **Tip**: Reduce update frequency for expensive plots (e.g., update every 10 epochs instead of every epoch)

```cpp
// Good: Bounded memory
CircularBuffer buffer(1000);  // Max 1000 points
for (int i = 0; i < 1000000; ++i) {
    buffer.AddPoint(i, data[i]);  // Automatically evicts oldest
}

// Bad: Unbounded growth
std::vector<double> x, y;
for (int i = 0; i < 1000000; ++i) {
    x.push_back(i);  // Memory grows indefinitely
    y.push_back(data[i]);
}
```

### Matplotlib Plotting

- **Overhead**: Python interpreter startup (~100ms)
- **Best Practice**: Batch operations, avoid frequent small plots
- **Export**: PNG is faster than PDF/SVG for large datasets

## Common Patterns

### Pattern 1: Training Dashboard

```cpp
class TrainingDashboard {
    std::string loss_plot_id_;
    std::string acc_plot_id_;

    void Initialize() {
        auto& mgr = PlotManager::GetInstance();

        PlotManager::PlotConfig config;
        config.backend = PlotManager::BackendType::ImPlot;
        config.type = PlotManager::PlotType::Line;

        config.title = "Training Loss";
        loss_plot_id_ = mgr.CreatePlot(config);

        config.title = "Accuracy";
        acc_plot_id_ = mgr.CreatePlot(config);
    }

    void OnEpochComplete(int epoch, double loss, double acc) {
        auto& mgr = PlotManager::GetInstance();
        mgr.UpdateRealtimePlot(loss_plot_id_, epoch, loss);
        mgr.UpdateRealtimePlot(acc_plot_id_, epoch, acc);
    }

    void Render() {
        auto& mgr = PlotManager::GetInstance();
        ImGui::Begin("Training");
        mgr.RenderImPlot(loss_plot_id_);
        mgr.RenderImPlot(acc_plot_id_);
        ImGui::End();
    }
};
```

### Pattern 2: Post-Training Analysis

```cpp
void AnalyzeTrainingResults(const std::string& metrics_file) {
    auto& mgr = PlotManager::GetInstance();

    // Load data
    PlotDataset dataset;
    dataset.LoadFromJSON(metrics_file);

    // Create histogram
    PlotManager::PlotConfig config;
    config.backend = PlotManager::BackendType::Matplotlib;
    config.type = PlotManager::PlotType::Histogram;
    config.title = "Loss Distribution";

    auto plot_id = mgr.CreatePlot(config);
    mgr.AddDataset(plot_id, "loss", dataset);
    mgr.SavePlotToFile(plot_id, "loss_histogram.png");

    // Create KDE plot
    config.type = PlotManager::PlotType::KDE;
    auto kde_id = mgr.CreatePlot(config);
    mgr.AddDataset(kde_id, "loss", dataset);
    mgr.SavePlotToFile(kde_id, "loss_kde.png");
}
```

### Pattern 3: Multi-Series Comparison

```cpp
void CompareDifferentRuns() {
    PlotDataset dataset;

    // Add multiple runs
    dataset.AddSeries("run_1_loss");
    dataset.AddSeries("run_2_loss");
    dataset.AddSeries("run_3_loss");

    // Populate from files
    LoadRunData("run1.json", dataset.GetSeries("run_1_loss"));
    LoadRunData("run2.json", dataset.GetSeries("run_2_loss"));
    LoadRunData("run3.json", dataset.GetSeries("run_3_loss"));

    // Create multi-line plot
    auto& mgr = PlotManager::GetInstance();
    PlotManager::PlotConfig config;
    config.type = PlotManager::PlotType::Line;
    config.show_legend = true;

    auto plot_id = mgr.CreatePlot(config);
    mgr.AddDataset(plot_id, "comparison", dataset);
    mgr.RenderImPlot(plot_id);
}
```

## Debugging

### Enable Logging

```cpp
// In debug builds, plotting uses spdlog
#include <spdlog/spdlog.h>

spdlog::set_level(spdlog::level::debug);  // Show all debug messages
```

### Common Issues

**"ImPlot context not initialized"**
- Ensure `ImPlot::CreateContext()` is called before any plots
- PlotManager handles this automatically in `Initialize()`

**"Python backend unavailable"**
- Check Python installation: `python --version`
- Verify matplotlib: `python -c "import matplotlib"`
- Call `PlotManager::InitializePythonBackend()` at startup

**"Plot not rendering"**
- Verify `RenderImPlot()` is called inside ImGui render loop
- Check that plot ID is valid: `mgr.HasPlot(plot_id)`
- Ensure plot has data: `mgr.GetDataset(plot_id, "series_name") != nullptr`

## Contributing

### Guidelines

1. **Follow existing patterns**: Use PlotBackend interface, don't bypass PlotManager
2. **Add tests**: Every new plot type should have unit tests
3. **Document**: Update this README and add code comments
4. **Cross-platform**: Test on Windows, macOS, Linux
5. **Performance**: Profile with large datasets (10K+ points)

### Code Style

```cpp
// Good: Descriptive names, clear ownership
auto& mgr = PlotManager::GetInstance();
std::string plot_id = mgr.CreatePlot(config);

// Bad: Unclear ownership, abbreviations
auto m = PM::get();
auto id = m->create(c);
```

### Submitting Changes

1. Create feature branch: `git checkout -b feature/your-plot-type`
2. Implement and test
3. Update this README if adding new features
4. Submit PR with description and test results

## Future Enhancements

### Planned Features

- [ ] **3D Plotting**: Surface plots, 3D scatter using ImPlot3D
- [ ] **Animation**: Animated training visualizations
- [ ] **Subplots**: Multi-panel layouts (2x2 grid, etc.)
- [ ] **Interactivity**: Click to inspect data points
- [ ] **Plot Templates**: Predefined configurations for common use cases
- [ ] **Export Improvements**: Copy to clipboard, export to LaTeX/TikZ
- [ ] **Theming**: Custom color schemes, dark/light mode
- [ ] **Statistics Panel**: Automatic statistics display alongside plot

### Research Areas

- **GPU Acceleration**: Use ArrayFire for large dataset preprocessing
- **Streaming Protocols**: WebSocket support for remote plotting
- **Notebook Integration**: Jupyter-like interface within Engine

## References

### External Documentation

- **ImPlot**: https://github.com/epezent/implot
- **matplotlib**: https://matplotlib.org/stable/
- **pybind11**: https://pybind11.readthedocs.io/

### Internal Documentation

- `CLAUDE.md`: Project overview and build instructions
- `cyxwiz-engine/README.md`: Engine architecture
- `docs/architecture.md`: Overall system design

## License

Part of the CyxWiz project. See root LICENSE file.

---

**Last Updated**: 2025-11-12
**Status**: Phase 1 Implementation (Core Infrastructure Complete)
**Maintainer**: CyxWiz Core Team
