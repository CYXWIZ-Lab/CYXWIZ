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
6. **GUI Integration**: Matplotlib plots rendered as OpenGL textures in ImGui windows for seamless viewing

### Component Overview

```
cyxwiz-engine/src/plotting/
├── plot_manager.h/cpp          # Singleton manager (orchestrates everything)
├── plot_dataset.h/cpp          # Data structures (PlotDataset, CircularBuffer)
├── backends/
│   ├── plot_backend.h          # Abstract backend interface
│   ├── implot_backend.h/cpp    # Real-time plotting (Dear ImPlot)
│   └── matplotlib_backend.h/cpp # Offline plotting (Python/matplotlib)
├── gui/panels/
│   ├── plot_window.h/cpp       # Reusable plot display window (supports both backends)
│   └── plot_test_control.h/cpp # Test control panel for plot generation
├── plot_types/                 # Specialized plot implementations (PLANNED)
│   ├── histogram.h/cpp         # Histogram plots
│   ├── kde.h/cpp              # Kernel Density Estimation
│   ├── boxplot.h/cpp          # Box-and-whisker plots
│   ├── qqplot.h/cpp           # Q-Q plots for normality testing
│   └── mosaic.h/cpp           # Mosaic plots for categorical data
└── test_data_generator.h/cpp  # Fake data for testing (sine, cosine, normal, etc.)
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
- **Display**: Plots are rendered to PNG and loaded as OpenGL textures in ImGui windows

**Implementation Details**:
The matplotlib backend works in three stages:

1. **Plot Generation**: Python commands are accumulated (BeginPlot, PlotLine/Scatter/etc., EndPlot)
2. **Image Export**: matplotlib saves the figure to a temporary PNG file via Agg backend
3. **GUI Rendering**: PNG is loaded using stb_image and uploaded to OpenGL texture for display in PlotWindow

**Example Usage**:
```cpp
// Create offline plot
PlotManager::PlotConfig config;
config.backend = PlotManager::BackendType::Matplotlib;
config.type = PlotManager::PlotType::Line;

auto plot_id = mgr.CreatePlot(config);

// Add data
PlotDataset dataset;
dataset.AddSeries("distribution");
// ... populate data ...
mgr.AddDataset(plot_id, "dist", dataset);

// Save to file (triggers: BeginPlot -> PlotLine -> EndPlot -> savefig)
mgr.SavePlotToFile(plot_id, "distribution.png");

// Display in GUI
PlotWindow window("My Plot", PlotWindow::PlotWindowType::Line2D);
window.SetPlotId(plot_id);
window.Render();  // Shows matplotlib plot as texture
```

### 7. PlotWindow (GUI Component)

**Purpose**: Reusable dockable window for displaying plots

**Features**:
- Supports both ImPlot (direct rendering) and matplotlib (texture rendering)
- Menu bar with File/Edit/View/Tools/Window/Help
- Interactive controls (Regenerate Data, Save to File, adjust parameters)
- Export functionality (PNG, SVG)
- Automatic data generation for testing

**Usage Example**:
```cpp
// Create plot window
auto window = std::make_shared<PlotWindow>(
    "Line Plot - Sine Wave",
    PlotWindow::PlotWindowType::Line2D,
    true  // auto_generate test data
);

// Connect to existing plot
window->SetPlotId(plot_id);

// Render in main loop
window->Render();
```

### 8. PlotTestControlPanel (GUI Component)

**Purpose**: Interactive panel for testing plotting system

**Features**:
- Plot type selection (Line, Scatter, Bar, Histogram, etc.)
- Backend selection (ImPlot vs Matplotlib)
- Test data selection (Sine, Cosine, Normal, Exponential, etc.)
- Generate and Clear All buttons
- Automatic window management

**Usage**:
Access from main menu: `Plots > Test Control`

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

### ✅ Completed (Phase 1-5)

**Core Infrastructure**:
- ✅ PlotManager singleton with plot lifecycle management
- ✅ PlotDataset with multi-series support
- ✅ CircularBuffer for bounded real-time data
- ✅ PlotBackend abstract interface
- ✅ ImPlotBackend for real-time rendering
- ✅ MatplotlibBackend with Python integration (pybind11)
- ✅ Statistics calculation (mean, median, std dev, quartiles)

**GUI Components**:
- ✅ PlotWindow - Reusable dockable plot display window
- ✅ PlotTestControlPanel - Interactive testing interface
- ✅ Matplotlib texture rendering (stb_image + OpenGL)
- ✅ Menu integration (Plots > Test Control)

**Plot Types**:
- ✅ Line plots (ImPlot + Matplotlib)
- ✅ Scatter plots (ImPlot + Matplotlib)
- ✅ Bar charts (ImPlot + Matplotlib)
- ✅ Histograms (ImPlot + Matplotlib)
- ✅ Heatmaps (ImPlot)

**Data Generation**:
- ✅ TestDataGenerator with multiple patterns
  - Sine, Cosine, Normal Distribution
  - Exponential Decay, Random Scatter
  - Linear, Polynomial, Damped Oscillation

**File I/O**:
- ✅ SavePlotToFile with plot data rendering
- ✅ PNG export via matplotlib
- ✅ Temporary file management for matplotlib plots

### 🚧 Current Limitations

**Matplotlib Backend**:
- ⚠️ Some advanced plot types (3D, Polar, Pie) map to basic types
- ⚠️ Limited customization exposed to C++ API
- ⚠️ SVG export implemented but needs testing

**ImPlot Backend**:
- ⚠️ No file export capability (render-only)
- ⚠️ Limited statistical plot types

**GUI**:
- ⚠️ Menu bar in PlotWindow is stub (File/Edit/View/etc not implemented)
- ⚠️ Regenerate functionality uses fixed parameters

### 📋 Planned Enhancements (Contributors Welcome!)

**Priority 1 - Core Functionality**:
- [ ] Implement PlotWindow menu bar actions (Open, Save As, Copy, Export options)
- [ ] Add SVG and PDF export support with testing
- [ ] Implement proper 3D plotting support (Surface, 3D Scatter, 3D Line)
- [ ] Add Polar plot support in matplotlib backend
- [ ] Add Pie chart support in matplotlib backend
- [ ] Implement Stairs plot type
- [ ] Add Box plot implementation
- [ ] Add Violin plot implementation
- [ ] Add KDE (Kernel Density Estimation) plot

**Priority 2 - Advanced Plot Types**:
- [ ] Q-Q plots for normality testing
- [ ] Mosaic plots for categorical data
- [ ] Stem-and-leaf plots
- [ ] Contour plots
- [ ] Error bar plots
- [ ] Candlestick charts (for metrics visualization)

**Priority 3 - GUI Improvements**:
- [ ] Implement "Save to File" dialog with format selection
- [ ] Add customizable plot parameters UI
- [ ] Implement plot themes/color schemes
- [ ] Add zoom/pan controls for ImPlot
- [ ] Add data inspection tooltip (hover to see values)
- [ ] Multi-plot layouts (subplots, grid layouts)

**Priority 4 - Integration**:
- [ ] Connect to TrainingDashboardPanel for live training metrics
- [ ] Add plot templates for common ML use cases
- [ ] Implement data import from CSV/JSON
- [ ] Add plot comparison tools (overlay multiple runs)
- [ ] Animation support for training progress

**Priority 5 - Performance & Quality**:
- [ ] Optimize CircularBuffer for >10K points
- [ ] Add plot caching to avoid redundant matplotlib calls
- [ ] Implement progressive rendering for large datasets
- [ ] Add unit tests for all plot types
- [ ] Integration tests with GUI components
- [ ] Memory leak detection and profiling

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

We welcome contributions to the CyxWiz plotting system! This section provides comprehensive guidance for contributors.

### Quick Start for Contributors

1. **Setup Development Environment**:
   ```bash
   # Clone and build
   git clone https://github.com/cyxwiz/cyxwiz.git
   cd cyxwiz
   cmake --preset windows-release  # or linux-release, macos-release
   cmake --build build --config Release
   ```

2. **Test the Plotting System**:
   ```bash
   # Run the engine
   ./build/bin/Release/cyxwiz-engine.exe
   # Navigate to: Plots > Test Control
   # Try generating plots with different backends and data
   ```

3. **Choose a Task**: See "Planned Enhancements" section above for available tasks

### Guidelines

1. **Follow existing patterns**: Use PlotBackend interface, don't bypass PlotManager
2. **Add tests**: Every new plot type should have unit tests
3. **Document**: Update this README and add code comments
4. **Cross-platform**: Test on Windows, macOS, Linux if possible
5. **Performance**: Profile with large datasets (10K+ points)
6. **Incremental PRs**: Submit small, focused changes rather than large rewrites

### Code Style

**Naming Conventions**:
```cpp
// Good: Descriptive names, clear ownership
auto& mgr = PlotManager::GetInstance();
std::string plot_id = mgr.CreatePlot(config);
PlotDataset dataset;
CircularBuffer buffer(1000);

// Bad: Unclear ownership, abbreviations
auto m = PM::get();
auto id = m->create(c);
PlotDS ds;
CircBuf buf(1000);
```

**Comments**:
```cpp
// Good: Explain why, not what
// Use circular buffer to prevent memory growth during long training runs
CircularBuffer buffer(1000);

// Bad: Stating the obvious
// Create a circular buffer with capacity 1000
CircularBuffer buffer(1000);
```

**Error Handling**:
```cpp
// Good: Check return values and log errors
if (!plot_mgr.SavePlotToFile(plot_id, filepath)) {
    spdlog::error("Failed to save plot {} to {}", plot_id, filepath);
    return false;
}

// Bad: Ignore errors
plot_mgr.SavePlotToFile(plot_id, filepath);
```

### Architecture Decisions

When implementing new features, follow these design principles:

**1. Backend Abstraction**:
- Always implement new plot types in BOTH backends (ImPlot and Matplotlib) if applicable
- If a plot type only makes sense for one backend, document why in comments

**2. PlotManager Orchestration**:
- Never call backend methods directly from GUI code
- Route all operations through PlotManager
- Let PlotManager handle backend selection and data routing

**3. Data Ownership**:
- PlotDataset owns the data
- PlotManager owns PlotBackend instances
- PlotWindow owns GUI state (texture IDs, window flags)

**4. GUI Integration**:
- Keep plotting logic separate from GUI logic
- PlotWindow should be reusable for any plot type
- Use composition over inheritance for specialized plot windows

### Adding a New Plot Type (Detailed Guide)

Let's walk through adding a "Box Plot" as an example:

**Step 1: Define in plot_manager.h**:
```cpp
enum class PlotType {
    Line,
    Scatter,
    Bar,
    Histogram,
    BoxPlot,  // <- Add this
    // ... other types
};
```

**Step 2: Add to matplotlib_backend.h**:
```cpp
class MatplotlibBackend : public PlotBackend {
public:
    // ... existing methods ...
    void PlotBoxPlot(const char* label,
                     const std::vector<double>& data,
                     const std::vector<std::string>& labels);
};
```

**Step 3: Implement in matplotlib_backend.cpp**:
```cpp
void MatplotlibBackend::PlotBoxPlot(const char* label,
                                     const std::vector<double>& data,
                                     const std::vector<std::string>& labels) {
    if (!in_plot_) {
        spdlog::warn("PlotBoxPlot called outside BeginPlot/EndPlot");
        return;
    }

    std::ostringstream cmd;

    // Convert data to Python list
    cmd << "data = [";
    for (size_t i = 0; i < data.size(); ++i) {
        if (i > 0) cmd << ", ";
        cmd << data[i];
    }
    cmd << "]\n";

    // Create box plot
    cmd << "ax.boxplot(data, labels=['" << label << "'])\n";

    python_commands_ += cmd.str();
}
```

**Step 4: Update plot_manager.cpp SavePlotToFile**:
```cpp
switch (plot->config.type) {
    // ... existing cases ...
    case PlotType::BoxPlot:
        plot->backend->PlotBoxPlot(series.name.c_str(),
                                   series.y_data,
                                   {series.name});
        break;
}
```

**Step 5: Add to plot_test_control.cpp**:
```cpp
// In plot type selection array
const char* plot_types[] = {
    "Line Plot",
    "Scatter Plot",
    // ...
    "Box Plot",  // <- Add here
};

// In GeneratePlot() switch
case 4: plot_type = PlotManager::PlotType::BoxPlot; break;
```

**Step 6: Add test in plot_window.cpp** (if needed):
```cpp
case PlotWindowType::BoxPlot:
    GenerateBoxPlotData();
    break;
```

**Step 7: Write unit test**:
```cpp
// In tests/unit/test_plotting.cpp
TEST_CASE("Box plot generation", "[plotting][boxplot]") {
    auto& mgr = PlotManager::GetInstance();

    PlotManager::PlotConfig config;
    config.type = PlotManager::PlotType::BoxPlot;
    config.backend = PlotManager::BackendType::Matplotlib;

    auto plot_id = mgr.CreatePlot(config);

    PlotDataset dataset;
    dataset.AddSeries("quartiles");
    // Add test data...

    REQUIRE(mgr.SavePlotToFile(plot_id, "test_boxplot.png"));
}
```

**Step 8: Update this README**:
- Add BoxPlot to "Supported Plot Types" table
- Add example usage if unique
- Update "Implementation Status" to mark as completed

### Testing Your Changes

**Manual Testing**:
1. Run engine: `./build/bin/Release/cyxwiz-engine.exe`
2. Open "Plot Test Control" panel
3. Select your new plot type
4. Test with different data patterns
5. Test both ImPlot and Matplotlib backends
6. Verify export functionality

**Unit Testing**:
```bash
cd build/windows-release  # or your build directory
ctest --output-on-failure -R plotting
```

**Integration Testing**:
- Create multiple plots and verify they coexist
- Test window docking and undocking
- Test with very large datasets (>10K points)
- Test export to different formats (PNG, SVG, PDF)

### Submitting Changes

1. **Create Feature Branch**:
   ```bash
   git checkout master
   git pull origin master
   git checkout -b feature/box-plot-implementation
   ```

2. **Implement and Test**:
   ```bash
   # Make your changes
   # Build and test
   cmake --build build --config Release
   ctest
   ```

3. **Update Documentation**:
   - Add inline code comments
   - Update this README
   - Add usage examples if needed

4. **Commit with Clear Messages**:
   ```bash
   git add cyxwiz-engine/src/plotting/
   git commit -m "Add box plot support to matplotlib backend

   - Implement PlotBoxPlot in MatplotlibBackend
   - Add BoxPlot enum to PlotType
   - Wire up in PlotManager::SavePlotToFile
   - Add to PlotTestControlPanel options
   - Add unit test for box plot generation
   - Update README with box plot documentation"
   ```

5. **Push and Create PR**:
   ```bash
   git push origin feature/box-plot-implementation
   # Create PR on GitHub with description and screenshots
   ```

### PR Checklist

Before submitting, ensure:
- [ ] Code compiles on your platform (Windows/Linux/macOS)
- [ ] Unit tests pass (`ctest`)
- [ ] Manual testing completed (screenshots helpful)
- [ ] Code follows existing style (run clang-format if available)
- [ ] Added/updated comments and documentation
- [ ] README.md updated if adding features
- [ ] No compiler warnings introduced
- [ ] Performance tested with >1K data points
- [ ] Memory leaks checked (valgrind/Dr. Memory if available)

### Common Tasks for New Contributors

**Easy (Good First Issues)**:
- [ ] Add new test data patterns to TestDataGenerator (e.g., log curve, step function)
- [ ] Improve error messages with more context
- [ ] Add parameter validation to plot methods
- [ ] Improve PlotWindow button styling and layout
- [ ] Add keyboard shortcuts to PlotTestControlPanel

**Medium**:
- [ ] Implement SVG export and test thoroughly
- [ ] Add plot themes (dark mode, light mode, custom colors)
- [ ] Implement Stairs plot type
- [ ] Add tooltips to show data values on hover (ImPlot)
- [ ] Implement copy-to-clipboard for plots

**Advanced**:
- [ ] Implement true 3D plotting with matplotlib
- [ ] Add subplot/multi-panel layout support
- [ ] Implement plot animation for training progress
- [ ] Add Jupyter-style interactive controls
- [ ] Optimize CircularBuffer for millions of points

### Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue with reproduction steps
- **Design Decisions**: Discuss in PR or create RFC issue
- **Architecture**: Refer to `CLAUDE.md` and this README

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

**Last Updated**: 2025-11-14
**Status**: Phase 5 Complete - Dual-backend plotting with GUI integration fully operational
**Key Features**:
- ✅ ImPlot (real-time) and Matplotlib (offline) backends
- ✅ PlotWindow with texture rendering for matplotlib plots
- ✅ Interactive PlotTestControlPanel for testing
- ✅ TestDataGenerator with 8 data patterns
- ✅ SavePlotToFile with automatic plot rendering

**Next Phase**: Community contributions for advanced plot types and features
**Maintainer**: CyxWiz Core Team
