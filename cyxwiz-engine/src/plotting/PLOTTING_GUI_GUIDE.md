# CyxWiz Plotting GUI User Guide

## Overview

The CyxWiz Engine includes a comprehensive plotting system with an interactive GUI for testing, demonstration, and real-world usage. This guide covers how to use the plotting features through the graphical interface.

## Quick Start

### Opening the Plot Test Panel

1. **Launch CyxWiz Engine**
2. **Locate the "Plot Test Panel" window** (it opens automatically on startup)
3. If not visible, check the View menu or Window menu to enable it

The Plot Test Panel provides three main tabs:
- **Real-time Plots**: Interactive, live plotting demonstrations
- **Offline Plots**: Statistical and publication-quality plots
- **Test Controls**: Advanced plot generation and testing

---

## Real-time Plots Tab

### Training Simulation

The most powerful feature for ML training visualization.

**Controls:**
- **Start/Stop Training**: Begin or pause the training simulation
- **Reset**: Clear all training data and restart from epoch 0
- **Max Epochs**: Set the number of training iterations (10-1000)
- **Noise Level**: Adjust randomness in training curves (0.0-0.5)

**What It Shows:**
- **Training Loss**: Exponential decay curve showing loss reduction
- **Training Accuracy**: Logarithmic growth curve showing accuracy improvement
- Both plots update in real-time as simulation progresses

**How to Use:**
1. Set your desired max epochs (e.g., 100)
2. Adjust noise level (0.05 recommended for realistic curves)
3. Click "Start Training"
4. Watch live plots update every frame
5. Click "Stop Training" to pause
6. Click "Reset" to start over

**Use Cases:**
- Testing plot rendering performance
- Demonstrating real-time plotting capabilities
- Understanding training curve behaviors
- Visualizing overfitting scenarios

### Histogram

Visualize data distributions with interactive controls.

**Controls:**
- **Generate Random Data**: Create new random normal distribution
- **Samples**: Number of data points (100-10,000)
- **Bins**: Number of histogram bins (10-100)

**How to Use:**
1. Set number of samples (e.g., 1000)
2. Set number of bins (e.g., 50)
3. Click "Generate Random Data"
4. View histogram showing distribution

**What It Shows:**
- Frequency distribution of generated data
- Visual representation of normal distribution shape
- Bin counts across value ranges

### Scatter Plot

Visualize 2D clustered data patterns.

**Controls:**
- **Generate Clusters**: Create new clustered scatter data
- **Points**: Total number of data points (100-5000)

**How to Use:**
1. Set number of points
2. Click "Generate Clusters"
3. View scatter plot showing 3 clusters in 2D space

**What It Shows:**
- K-means style clustering
- 2D point distributions
- Cluster separation patterns

### Heatmap

Display confusion matrices and 2D density data.

**Controls:**
- **Generate Confusion Matrix**: Create 5x5 confusion matrix with 85% accuracy

**How to Use:**
1. Click "Generate Confusion Matrix"
2. View heatmap showing classification accuracy

**What It Shows:**
- 5x5 confusion matrix
- Diagonal showing correct classifications
- Off-diagonal showing misclassifications
- Color intensity representing frequency

---

## Offline Plots Tab

For statistical analysis and publication-quality output.

### Overview

Offline plots use matplotlib (Python) for advanced statistical visualizations. These are rendered to files rather than displayed in real-time.

**Current Status:**
- ⚠️ Python integration pending
- Commands are queued but not executed yet
- Will be fully functional after pybind11 integration

### Available Plots

#### KDE Plot (Kernel Density Estimation)
- **Purpose**: Smooth probability density estimation
- **Use Case**: Understanding data distribution shapes
- **Export**: Saves to `output/kde_plot.png`

**How to Use:**
1. Click "Generate KDE Plot"
2. Check `output/` folder for generated plot
3. View publication-quality density curve

#### Q-Q Plot (Quantile-Quantile)
- **Purpose**: Test for normality of data
- **Use Case**: Validating statistical assumptions
- **Export**: Saves to `output/qq_plot.png`

**How to Use:**
1. Click "Generate Q-Q Plot"
2. Points on straight line indicate normal distribution
3. Deviations show non-normality

#### Box Plot
- **Purpose**: Show quartile distributions
- **Use Case**: Identifying outliers and spread
- **Export**: Saves to `output/box_plot.png`

**How to Use:**
1. Click "Generate Box Plot"
2. View quartiles, median, whiskers, and outliers

### Export Options

**Filepath Input:**
- Enter custom output path (default: `output/plot.png`)
- Supports `.png`, `.pdf`, `.svg` formats

**Export Button:**
- Saves current plot with specified settings
- Creates directory if doesn't exist

---

## Test Controls Tab

Advanced plot generation and customization.

### Plot Type Selection

**Available Types:**
- **Line**: Continuous curves, training metrics
- **Scatter**: Point clouds, correlations
- **Bar**: Categorical comparisons
- **Histogram**: Distributions
- **Heatmap**: 2D matrices
- **Box**: Quartile analysis
- **Violin**: Distribution density
- **KDE**: Kernel density estimation

### Backend Selection

Choose rendering engine:

**ImPlot (Real-time):**
- ✅ Integrated into GUI
- ✅ 60+ FPS performance
- ✅ Interactive (zoom, pan)
- ❌ No file export
- **Best for**: Training dashboards, live metrics

**Matplotlib (Offline):**
- ✅ Publication quality
- ✅ File export (PNG/PDF/SVG)
- ✅ Advanced statistics
- ⚠️ Python integration pending
- **Best for**: Reports, papers, analysis

### Test Data Generation

**Data Types:**
1. **Normal Distribution**: Gaussian bell curve
2. **Sine Wave**: Periodic signal with noise
3. **Training Curve**: Realistic ML loss decay
4. **Clustered Data**: K-means style groups
5. **Spiral Pattern**: Non-linear separable data
6. **Random Walk**: Stochastic process
7. **Overfitting Example**: Train vs validation divergence

**Parameters:**
- **Num Samples**: 10 to 10,000 points
- **Num Bins**: 5 to 100 bins (for histograms)
- **Noise**: 0.0 to 1.0 (randomness level)

### Generate Test Plot

**Workflow:**
1. Select plot type (e.g., "Line")
2. Choose backend (e.g., "ImPlot (Real-time)")
3. Pick data type (e.g., "Training Curve")
4. Adjust parameters (e.g., 100 samples, 0.05 noise)
5. Click "Generate Test Plot"
6. New plot window appears with generated data

### Clear All Plots

- Removes all test plots
- Resets training simulation
- Frees memory
- Useful for starting fresh

### Statistics Display

Real-time statistics for training loss plot:
- **Mean**: Average loss value
- **Std Dev**: Variability in loss
- **Min/Max**: Loss range
- **Total Plots**: Number of active plots

---

## Integration with Training Dashboard

### Using Plots in Training Workflows

While PlotTestPanel is for demonstration, real training workflows use the same plotting infrastructure.

**TrainingDashboardPanel Integration:**

```cpp
#include "plotting/plot_manager.h"

void TrainingDashboardPanel::OnEpochComplete(int epoch, TrainingMetrics metrics) {
    auto& mgr = plotting::PlotManager::GetInstance();

    // Update loss plot
    mgr.UpdateRealtimePlot(loss_plot_id_, epoch, metrics.loss, "train_loss");
    mgr.UpdateRealtimePlot(loss_plot_id_, epoch, metrics.val_loss, "val_loss");

    // Update accuracy plot
    mgr.UpdateRealtimePlot(acc_plot_id_, epoch, metrics.accuracy, "train_acc");
    mgr.UpdateRealtimePlot(acc_plot_id_, epoch, metrics.val_accuracy, "val_acc");
}

void TrainingDashboardPanel::Render() {
    ImGui::Begin("Training Dashboard");

    mgr.RenderImPlot(loss_plot_id_);
    mgr.RenderImPlot(acc_plot_id_);

    ImGui::End();
}
```

---

## Tips and Best Practices

### Performance

**Real-time Plotting:**
- Keep data points < 10,000 for 60 FPS
- Use CircularBuffer to limit memory (automatic in PlotManager)
- Update plots every N frames if high frequency

**Memory Management:**
- Plots are automatically cleaned up on close
- Clear unused plots to free memory
- CircularBuffer prevents unbounded growth

### Visualization Tips

**Training Curves:**
- Use logarithmic scale for loss (if very large initial values)
- Plot both train and validation on same graph
- Add horizontal line for target accuracy

**Distributions:**
- Use 30-50 bins for histograms (sqrt(N) rule of thumb)
- Overlay KDE curve for smooth visualization
- Check for normality with Q-Q plot before statistical tests

**Comparisons:**
- Use scatter for correlations
- Use box plots for multiple groups
- Use heatmaps for confusion matrices

### Exporting Plots

**For Papers/Reports:**
1. Use Matplotlib backend
2. Set high DPI (300+)
3. Export to PDF for vector graphics
4. Use descriptive titles and labels

**For Presentations:**
1. Use ImPlot for screenshots
2. Adjust window size for aspect ratio
3. Enable legends
4. Use clear color schemes

---

## Troubleshooting

### Plot Not Showing

**Problem**: Plot window is empty or not rendering

**Solutions:**
- Check that plot ID is valid
- Verify data has been added to plot
- Ensure `RenderImPlot()` is called in render loop
- Check ImGui window is visible

### Training Simulation Not Updating

**Problem**: Training curves are flat or not moving

**Solutions:**
- Click "Start Training" button
- Check epoch counter is incrementing
- Verify noise level is reasonable (not 0 or > 1)
- Try "Reset" and restart

### Export Fails

**Problem**: Cannot export plot to file

**Solutions:**
- Ensure output directory exists
- Check file path is valid
- Verify matplotlib backend is selected
- Wait for Python integration (if pending)

### Performance Issues

**Problem**: GUI is slow or dropping frames

**Solutions:**
- Reduce number of data points
- Lower update frequency
- Close unused plots
- Reduce number of active plots

---

## Keyboard Shortcuts

(To be implemented in future versions)

**Planned Shortcuts:**
- `Ctrl+T`: Start/Stop training
- `Ctrl+R`: Reset training
- `Ctrl+G`: Generate test plot
- `Ctrl+E`: Export current plot
- `Ctrl+C`: Clear all plots

---

## Advanced Features

### Multi-Series Plots

Display multiple datasets on same plot:

```cpp
auto& mgr = plotting::PlotManager::GetInstance();

// Add multiple series
mgr.UpdateRealtimePlot(plot_id, epoch, train_loss, "train_loss");
mgr.UpdateRealtimePlot(plot_id, epoch, val_loss, "val_loss");
mgr.UpdateRealtimePlot(plot_id, epoch, test_loss, "test_loss");
```

Legend automatically shows all series.

### Custom Plot Configurations

Modify plot appearance:

```cpp
plotting::PlotManager::PlotConfig config;
config.title = "Custom Plot";
config.x_label = "Time (s)";
config.y_label = "Value";
config.width = 1200;
config.height = 600;
config.show_legend = true;
config.show_grid = true;
config.auto_fit = false;  // Manual axis limits

auto plot_id = mgr.CreatePlot(config);
```

### Data Persistence

Save plot data to JSON:

```cpp
plotting::PlotDataset dataset;
// ... populate data ...
dataset.SaveToJSON("training_metrics.json");

// Later, reload
dataset.LoadFromJSON("training_metrics.json");
```

---

## Future Enhancements

### Planned Features

- [ ] **Interactive zoom/pan** (ImPlot built-in)
- [ ] **Data inspection** (click to view values)
- [ ] **Plot templates** (pre-configured styles)
- [ ] **Animation export** (GIF/video)
- [ ] **Subplot layouts** (2x2 grids)
- [ ] **3D plotting** (surface, 3D scatter)
- [ ] **Theming** (dark/light modes)
- [ ] **Copy to clipboard** (for quick sharing)

### Matplotlib Integration

Once Python integration is complete:
- Full statistical plot library
- LaTeX rendering for equations
- Seaborn-style themes
- Advanced customization
- Direct NumPy/Pandas integration

---

## Support

### Getting Help

**Documentation:**
- See `cyxwiz-engine/src/plotting/README.md` for developer guide
- See `PLOTTING_SYSTEM.md` for architecture overview

**Debugging:**
- Enable debug logging: `spdlog::set_level(spdlog::level::debug)`
- Check console output for plot creation/update messages
- Verify plot IDs are valid

**Reporting Issues:**
- Include screenshot of problem
- Provide log output
- Describe expected vs actual behavior
- Mention OS and GPU

---

## Example Workflows

### 1. Test New ML Model Training

```
1. Open Plot Test Panel → Real-time Plots tab
2. Set Max Epochs to 100
3. Set Noise Level to 0.05
4. Click "Start Training"
5. Observe loss decay and accuracy growth
6. Stop when satisfied
7. Screenshot for documentation
```

### 2. Analyze Data Distribution

```
1. Open Plot Test Panel → Real-time Plots tab
2. Go to Histogram section
3. Set Samples to 5000
4. Set Bins to 50
5. Click "Generate Random Data"
6. Observe normal distribution shape
7. Repeat with different sample sizes
```

### 3. Generate Publication Figure

```
1. Open Plot Test Panel → Offline Plots tab
2. Select desired statistical plot (KDE/Q-Q/Box)
3. Click "Generate [Plot Type]"
4. Check output/ folder for PNG file
5. Use in paper/report
```

### 4. Test Custom Plot Types

```
1. Open Plot Test Panel → Test Controls tab
2. Select "Plot Type": Line
3. Select "Backend": ImPlot
4. Select "Data Type": Training Curve
5. Set Num Samples: 100
6. Set Noise: 0.1
7. Click "Generate Test Plot"
8. Observe new plot window
```

---

**Last Updated**: 2025-11-12
**Version**: 1.0
**Status**: Real-time plotting functional, Matplotlib integration pending
