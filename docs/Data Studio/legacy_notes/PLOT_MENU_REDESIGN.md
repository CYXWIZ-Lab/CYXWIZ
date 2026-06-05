# Plot Menu Redesign - Implementation Summary

## Overview

Redesigned the Plots menu to provide a view-only reference of available plot types (based on Matplotlib cheatsheet) and added an interactive "Test Control" window for testing the plotting system.

## Changes Made

### 1. **Plots Menu Structure (View-Only)**

The Plots menu now displays categorized plot types as **inactive/disabled menu items** for reference only. Users can see what plots are available without accidentally triggering actions.

**Categories (from cheatsheets.pdf):**

- **Basic 2D Plots**
  - plot() - Line plot
  - scatter() - Scatter plot
  - bar() / barh() - Bar chart
  - imshow() - Image display
  - contour() / contourf() - Contour plot
  - pcolormesh() - Pseudocolor plot
  - quiver() - Vector field
  - pie() - Pie chart
  - fill_between() - Filled area

- **Advanced 2D Plots**
  - step() - Step plot
  - boxplot() - Box plot
  - errorbar() - Error bar plot
  - hist() - Histogram
  - violinplot() - Violin plot
  - barbs() - Barbs plot
  - eventplot() - Event plot
  - hexbin() - Hexagonal binning

- **3D Plots**
  - plot3D() - 3D line plot
  - scatter3D() - 3D scatter
  - plot_surface() - Surface plot
  - plot_wireframe() - Wireframe
  - contour3D() - 3D contour

- **Polar Plots**
  - polar() - Polar plot

- **Statistical Plots**
  - hist() - Histogram
  - boxplot() - Box plot
  - violinplot() - Violin plot
  - kde plot - Density estimation

- **Specialized Plots**
  - heatmap - Heat map
  - streamplot() - Stream plot
  - specgram() - Spectrogram
  - spy() - Sparse matrix viz

### 2. **Plot Test Control Window**

Added a new interactive window accessible from `Plots > Test Control` that allows users to:

**Features:**
- **Plot Type Selection** - Choose from 13 plot types:
  - Line Plot
  - Scatter Plot
  - Bar Chart
  - Histogram
  - Box Plot
  - Stem Plot
  - Stairs Plot
  - Pie Chart
  - Heatmap
  - Polar Plot
  - 3D Surface
  - 3D Scatter
  - 3D Line

- **Backend Selection** - Choose between:
  - ImPlot (Real-time) - Fast, interactive plotting
  - Matplotlib (Offline) - Publication-quality exports

- **Test Data Selection** - Choose from 8 test patterns:
  - Sine Wave
  - Cosine Wave
  - Normal Distribution
  - Exponential Decay
  - Random Scatter
  - Linear
  - Polynomial
  - Damped Oscillation

- **Generate Plot Button** - Creates a new plot window with the selected configuration

- **Helpful Tips** - Color-coded tips about backend selection

## Implementation Details

### Files Modified

1. **toolbar.h** (cyxwiz-engine/src/gui/panels/)
   - Added `show_plot_test_control_` flag
   - Added `selected_plot_type_`, `selected_backend_`, `selected_test_data_` state variables
   - Added `RenderPlotTestControl()` method declaration

2. **toolbar.cpp** (cyxwiz-engine/src/gui/panels/)
   - Initialized new state variables in constructor
   - Completely rewrote `RenderPlotsMenu()` to show view-only categories
   - Added `RenderPlotTestControl()` implementation
   - Updated `Render()` to call `RenderPlotTestControl()` when window is open

### Key Design Decisions

1. **Inactive Submenus** - Used `ImGui::BeginMenu("Name", false)` to create disabled menu headers
2. **TextDisabled** - Used `ImGui::TextDisabled()` for plot type listings to indicate they're informational
3. **Auto-Resize Window** - Test Control uses `ImGuiWindowFlags_AlwaysAutoResize` for clean UI
4. **Descriptive Titles** - Generated plot titles include plot type, backend, and test data for clarity

## Usage

### Opening Test Control

1. Run CyxWiz Engine
2. Click `Plots` menu in the toolbar
3. Click `Test Control` (first menu item)
4. Test Control window appears

### Viewing Available Plots

1. Click `Plots` menu
2. Browse through the categorized lists
3. Note: These are view-only and cannot be clicked

### Generating Test Plots

1. Open Test Control window
2. Select desired plot type from dropdown
3. Select backend (ImPlot or Matplotlib)
4. Select test data pattern
5. Click "Generate Plot" button
6. New plot window appears with auto-generated test data

## Example Workflow

```
User Action: Plots > Test Control
Result: Test Control window opens

User Action: Select "Line Plot" + "ImPlot" + "Sine Wave"
User Action: Click "Generate Plot"
Result: New window titled "Line Plot (ImPlot - Real-time) - Sine Wave" appears

User Action: Select "Histogram" + "Matplotlib" + "Normal Distribution"
User Action: Click "Generate Plot"
Result: New window titled "Histogram (Matplotlib - Offline) - Normal Distribution" appears
```

## Benefits

1. **Educational** - Users can see all available plot types at a glance
2. **No Accidents** - View-only menus prevent accidental plot creation
3. **Centralized Testing** - Single control panel for all testing needs
4. **Backend Comparison** - Easy to compare ImPlot vs Matplotlib
5. **Test Data Variety** - Multiple patterns to test different plot characteristics
6. **Clean UI** - Organized, categorized menu structure

## Technical Notes

### Plot Type Mapping

The test control maps combo box selections to `PlotWindow::PlotWindowType` enum:

```cpp
case 0: Line2D
case 1: Scatter2D
case 2: Bar
case 3: Histogram
case 4: BoxPlot
case 5: Stem
case 6: Stair
case 7: PieChart
case 8: Heatmap
case 9: Polar
case 10: Surface3D
case 11: Scatter3D
case 12: Line3D
```

### Backend Selection

Currently, the backend selection is stored but not yet fully integrated with the plot generation logic. Future enhancements should:
- Pass backend preference to PlotWindow constructor
- Configure PlotManager to use selected backend
- Validate Matplotlib availability before selecting it

### Test Data Generation

Test data selection is stored but actual data generation happens in `PlotWindow` with auto-generated flag set to `true`. Future enhancements should:
- Implement different data generators for each pattern
- Pass test data type to PlotWindow
- Generate appropriate data based on plot type and pattern

## Future Enhancements

1. **Backend Integration** - Actually use selected backend in plot creation
2. **Custom Data Generation** - Generate different patterns based on test data selection
3. **Plot Gallery** - Add preview thumbnails for each plot type
4. **Export Settings** - Add DPI, format options for Matplotlib plots
5. **Real-time Preview** - Show small preview before generating full plot
6. **Favorite Combinations** - Save frequently used configurations
7. **Batch Generation** - Create multiple plots at once
8. **Template System** - Save and load plot configurations

## Build Status

✅ **Successfully compiles** on Windows with MSVC
- No errors, only warnings (unused parameters in other files)
- All new code compiles cleanly

## Testing Checklist

- [x] Menu displays correctly
- [x] Submenus are inactive (cannot click)
- [x] Test Control opens from menu
- [x] All dropdowns work
- [x] Generate Plot button creates windows
- [x] Window titles reflect selections
- [ ] Backend selection actually changes backend (TODO)
- [ ] Test data selection generates different patterns (TODO)

## References

- [Matplotlib Cheatsheet](../../../cheatsheets.pdf) - Source for plot categorization
- [PlotWindow Implementation](./plot_window.h) - Plot window types
- [ImGui Documentation](https://github.com/ocornut/imgui) - UI framework
- [ImPlot Documentation](https://github.com/epezent/implot) - Plotting library
