/**
 * plot_bindings.cpp - Python bindings for CyxWiz Plotting System
 *
 * Exposes the plotting API to Python for scripting and automation
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "../src/plotting/plot_manager.h"
#include "../src/plotting/plot_dataset.h"
#include "../src/plotting/plot_3d.h"
#include "../src/gui/panels/plot_window.h"
#include "../src/gui/panels/training_plot_panel.h"

#include <memory>
#include <vector>
#include <stdexcept>

namespace py = pybind11;
using namespace cyxwiz::plotting;

// Global registry for PlotWindows created from Python
// This ensures they persist and can be accessed by the GUI thread
static std::vector<std::shared_ptr<cyxwiz::PlotWindow>> g_python_plot_windows;

// Global TrainingPlotPanel instance (shared with MainWindow)
// These functions are defined in training_plot_panel_global.cpp
extern void set_training_plot_panel(cyxwiz::TrainingPlotPanel* panel);
extern cyxwiz::TrainingPlotPanel* get_training_plot_panel();

/**
 * Helper: Convert numpy array or list to std::vector<double>
 */
std::vector<double> to_double_vector(const py::object& obj) {
    std::vector<double> result;

    if (py::isinstance<py::array>(obj)) {
        // NumPy array
        py::array_t<double> arr = obj.cast<py::array_t<double>>();
        auto r = arr.unchecked<1>();
        result.reserve(r.shape(0));
        for (py::ssize_t i = 0; i < r.shape(0); ++i) {
            result.push_back(r(i));
        }
    } else if (py::isinstance<py::list>(obj)) {
        // Python list
        py::list list = obj.cast<py::list>();
        result.reserve(list.size());
        for (auto item : list) {
            result.push_back(item.cast<double>());
        }
    } else {
        throw std::invalid_argument("Expected numpy array or list");
    }

    return result;
}

/**
 * Helper: Generate X values [0, 1, 2, ...] for Y-only data
 */
std::vector<double> generate_x_indices(size_t size) {
    std::vector<double> x;
    x.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        x.push_back(static_cast<double>(i));
    }
    return x;
}

/**
 * Convenience function: Create and populate a line plot
 */
std::string plot_line(const py::object& x_data, const py::object& y_data,
                     const std::string& title = "Line Plot",
                     const std::string& x_label = "X",
                     const std::string& y_label = "Y",
                     const std::string& series_name = "data") {
    auto& mgr = PlotManager::GetInstance();

    // Convert data
    std::vector<double> x = to_double_vector(x_data);
    std::vector<double> y = to_double_vector(y_data);

    if (x.size() != y.size()) {
        throw std::invalid_argument("X and Y data must have the same length");
    }

    // Create plot
    PlotManager::PlotConfig config;
    config.title = title;
    config.x_label = x_label;
    config.y_label = y_label;
    config.type = PlotManager::PlotType::Line;
    config.backend = PlotManager::BackendType::ImPlot;

    std::string plot_id = mgr.CreatePlot(config);

    // Add data
    PlotDataset dataset;
    dataset.AddSeries(series_name);
    auto* series = dataset.GetSeries(series_name);

    for (size_t i = 0; i < x.size(); ++i) {
        series->AddPoint(x[i], y[i]);
    }

    mgr.AddDataset(plot_id, series_name, dataset);

    return plot_id;
}

/**
 * Convenience function: Create and populate a scatter plot
 */
std::string plot_scatter(const py::object& x_data, const py::object& y_data,
                        const std::string& title = "Scatter Plot",
                        const std::string& x_label = "X",
                        const std::string& y_label = "Y",
                        const std::string& series_name = "data") {
    auto& mgr = PlotManager::GetInstance();

    std::vector<double> x = to_double_vector(x_data);
    std::vector<double> y = to_double_vector(y_data);

    if (x.size() != y.size()) {
        throw std::invalid_argument("X and Y data must have the same length");
    }

    PlotManager::PlotConfig config;
    config.title = title;
    config.x_label = x_label;
    config.y_label = y_label;
    config.type = PlotManager::PlotType::Scatter;
    config.backend = PlotManager::BackendType::ImPlot;

    std::string plot_id = mgr.CreatePlot(config);

    PlotDataset dataset;
    dataset.AddSeries(series_name);
    auto* series = dataset.GetSeries(series_name);

    for (size_t i = 0; i < x.size(); ++i) {
        series->AddPoint(x[i], y[i]);
    }

    mgr.AddDataset(plot_id, series_name, dataset);

    return plot_id;
}

/**
 * Convenience function: Create histogram
 */
std::string plot_histogram(const py::object& data,
                          int bins = 10,
                          const std::string& title = "Histogram",
                          const std::string& x_label = "Value",
                          const std::string& y_label = "Frequency") {
    auto& mgr = PlotManager::GetInstance();

    std::vector<double> values = to_double_vector(data);

    PlotManager::PlotConfig config;
    config.title = title;
    config.x_label = x_label;
    config.y_label = y_label;
    config.type = PlotManager::PlotType::Histogram;
    config.backend = PlotManager::BackendType::ImPlot;

    std::string plot_id = mgr.CreatePlot(config);

    // TODO: Histogram binning logic should be in PlotManager or backend
    // For now, we'll just add the raw data
    PlotDataset dataset;
    dataset.AddSeries("histogram");
    auto* series = dataset.GetSeries("histogram");

    for (double val : values) {
        series->AddPoint(val, 1.0);  // Placeholder
    }

    mgr.AddDataset(plot_id, "histogram", dataset);

    return plot_id;
}

/**
 * Convenience function: Create bar chart
 */
std::string plot_bar(const py::object& x_data, const py::object& y_data,
                    const std::string& title = "Bar Chart",
                    const std::string& x_label = "Category",
                    const std::string& y_label = "Value") {
    auto& mgr = PlotManager::GetInstance();

    std::vector<double> x = to_double_vector(x_data);
    std::vector<double> y = to_double_vector(y_data);

    if (x.size() != y.size()) {
        throw std::invalid_argument("X and Y data must have the same length");
    }

    PlotManager::PlotConfig config;
    config.title = title;
    config.x_label = x_label;
    config.y_label = y_label;
    config.type = PlotManager::PlotType::Bar;
    config.backend = PlotManager::BackendType::ImPlot;

    std::string plot_id = mgr.CreatePlot(config);

    PlotDataset dataset;
    dataset.AddSeries("bar");
    auto* series = dataset.GetSeries("bar");

    for (size_t i = 0; i < x.size(); ++i) {
        series->AddPoint(x[i], y[i]);
    }

    mgr.AddDataset(plot_id, "bar", dataset);

    return plot_id;
}

/**
 * Show a plot in a dockable window
 * NOTE: This must be called from the main GUI thread
 */
void show_plot(const std::string& plot_id) {
    auto& mgr = PlotManager::GetInstance();

    if (!mgr.HasPlot(plot_id)) {
        throw std::runtime_error("Plot ID not found: " + plot_id);
    }

    // Get plot config to determine title
    auto config = mgr.GetPlotConfig(plot_id);

    // Create a PlotWindow (this will be managed by MainWindow's docking system)
    // For now, we store it in a global registry
    // TODO: Better integration with MainWindow's panel management
    auto window = std::make_shared<cyxwiz::PlotWindow>(
        config.title,
        cyxwiz::PlotWindow::PlotWindowType::Line2D,  // Will be determined by plot type
        false  // Don't auto-generate data
    );

    g_python_plot_windows.push_back(window);

    // The window will be rendered by MainWindow in the next frame
}

/**
 * Clear all Python-created plot windows
 */
void clear_plot_windows() {
    g_python_plot_windows.clear();
}

/**
 * Get all Python-created plot windows (for GUI integration)
 */
const std::vector<std::shared_ptr<cyxwiz::PlotWindow>>& get_python_plot_windows() {
    return g_python_plot_windows;
}

/**
 * Python module definition
 */
PYBIND11_MODULE(cyxwiz_plotting, m) {
    m.doc() = "CyxWiz Plotting System - Python Bindings";

    // ===== Enums =====

    py::enum_<PlotManager::BackendType>(m, "BackendType", py::arithmetic())
        .value("ImPlot", PlotManager::BackendType::ImPlot,
               "Real-time plotting integrated into ImGui")
        .value("Matplotlib", PlotManager::BackendType::Matplotlib,
               "Offline plotting using Matplotlib")
        .export_values();

    py::enum_<PlotManager::PlotType>(m, "PlotType", py::arithmetic())
        .value("Line", PlotManager::PlotType::Line,
               "Line plot")
        .value("Scatter", PlotManager::PlotType::Scatter,
               "Scatter plot")
        .value("Histogram", PlotManager::PlotType::Histogram,
               "Histogram")
        .value("BoxPlot", PlotManager::PlotType::BoxPlot,
               "Box plot")
        .value("Violin", PlotManager::PlotType::Violin,
               "Violin plot")
        .value("KDE", PlotManager::PlotType::KDE,
               "Kernel density estimation")
        .value("QQPlot", PlotManager::PlotType::QQPlot,
               "Q-Q plot")
        .value("MosaicPlot", PlotManager::PlotType::MosaicPlot,
               "Mosaic plot")
        .value("StemLeaf", PlotManager::PlotType::StemLeaf,
               "Stem-and-leaf plot")
        .value("DotChart", PlotManager::PlotType::DotChart,
               "Dot chart")
        .value("Heatmap", PlotManager::PlotType::Heatmap,
               "Heatmap")
        .value("Bar", PlotManager::PlotType::Bar,
               "Bar chart")
        .export_values();

    // ===== PlotConfig =====

    py::class_<PlotManager::PlotConfig>(m, "PlotConfig")
        .def(py::init<>())
        .def_readwrite("title", &PlotManager::PlotConfig::title,
                      "Plot title")
        .def_readwrite("x_label", &PlotManager::PlotConfig::x_label,
                      "X-axis label")
        .def_readwrite("y_label", &PlotManager::PlotConfig::y_label,
                      "Y-axis label")
        .def_readwrite("type", &PlotManager::PlotConfig::type,
                      "Plot type")
        .def_readwrite("backend", &PlotManager::PlotConfig::backend,
                      "Backend to use for rendering")
        .def_readwrite("auto_fit", &PlotManager::PlotConfig::auto_fit,
                      "Auto-fit axes to data")
        .def_readwrite("show_legend", &PlotManager::PlotConfig::show_legend,
                      "Show legend")
        .def_readwrite("show_grid", &PlotManager::PlotConfig::show_grid,
                      "Show grid")
        .def_readwrite("width", &PlotManager::PlotConfig::width,
                      "Plot width in pixels")
        .def_readwrite("height", &PlotManager::PlotConfig::height,
                      "Plot height in pixels");

    // ===== Statistics =====

    py::class_<PlotManager::Statistics>(m, "Statistics")
        .def(py::init<>())
        .def_readonly("min", &PlotManager::Statistics::min,
                     "Minimum value")
        .def_readonly("max", &PlotManager::Statistics::max,
                     "Maximum value")
        .def_readonly("mean", &PlotManager::Statistics::mean,
                     "Mean value")
        .def_readonly("median", &PlotManager::Statistics::median,
                     "Median value")
        .def_readonly("std_dev", &PlotManager::Statistics::std_dev,
                     "Standard deviation")
        .def_readonly("q1", &PlotManager::Statistics::q1,
                     "First quartile (25th percentile)")
        .def_readonly("q3", &PlotManager::Statistics::q3,
                     "Third quartile (75th percentile)");

    // ===== PlotDataset =====

    py::class_<PlotDataset::Series>(m, "Series")
        .def(py::init<>())
        .def_readwrite("name", &PlotDataset::Series::name,
                      "Series name")
        .def_readwrite("x_data", &PlotDataset::Series::x_data,
                      "X data points")
        .def_readwrite("y_data", &PlotDataset::Series::y_data,
                      "Y data points")
        .def("add_point", &PlotDataset::Series::AddPoint,
             "Add a single point to the series",
             py::arg("x"), py::arg("y"))
        .def("clear", &PlotDataset::Series::Clear,
             "Clear all data from the series")
        .def("size", &PlotDataset::Series::Size,
             "Get number of points in the series");

    py::class_<PlotDataset>(m, "PlotDataset")
        .def(py::init<>())
        .def("add_series", &PlotDataset::AddSeries,
             "Add a new series to the dataset",
             py::arg("name"))
        .def("has_series", &PlotDataset::HasSeries,
             "Check if a series exists",
             py::arg("name"))
        .def("get_series",
             static_cast<PlotDataset::Series* (PlotDataset::*)(const std::string&)>(
                 &PlotDataset::GetSeries),
             "Get a series by name",
             py::arg("name"),
             py::return_value_policy::reference_internal)
        .def("get_series_names", &PlotDataset::GetSeriesNames,
             "Get all series names")
        .def("get_series_count", &PlotDataset::GetSeriesCount,
             "Get number of series")
        .def("add_point",
             static_cast<void (PlotDataset::*)(double, double)>(&PlotDataset::AddPoint),
             "Add a point to the default series",
             py::arg("x"), py::arg("y"))
        .def("clear", &PlotDataset::Clear,
             "Clear all data")
        .def("is_empty", &PlotDataset::IsEmpty,
             "Check if dataset is empty")
        .def("save_to_json", &PlotDataset::SaveToJSON,
             "Save dataset to JSON file",
             py::arg("filepath"))
        .def("load_from_json", &PlotDataset::LoadFromJSON,
             "Load dataset from JSON file",
             py::arg("filepath"));

    // ===== PlotManager =====

    py::class_<PlotManager>(m, "PlotManager")
        .def_static("get_instance", &PlotManager::GetInstance,
                   "Get the singleton PlotManager instance",
                   py::return_value_policy::reference)
        .def("set_default_backend", &PlotManager::SetDefaultBackend,
             "Set the default backend for new plots",
             py::arg("backend"))
        .def("get_default_backend", &PlotManager::GetDefaultBackend,
             "Get the default backend")
        .def("is_backend_available", &PlotManager::IsBackendAvailable,
             "Check if a backend is available",
             py::arg("backend"))
        .def("create_plot",
             [](PlotManager& self, const std::string& title,
                const std::string& x_label, const std::string& y_label,
                PlotManager::PlotType type, PlotManager::BackendType backend) {
                 PlotManager::PlotConfig config;
                 config.title = title;
                 config.x_label = x_label;
                 config.y_label = y_label;
                 config.type = type;
                 config.backend = backend;
                 return self.CreatePlot(config);
             },
             "Create a new plot",
             py::arg("title"), py::arg("x_label"), py::arg("y_label"),
             py::arg("plot_type"), py::arg("backend_type"))
        .def("create_plot",
             static_cast<std::string (PlotManager::*)(const PlotManager::PlotConfig&)>(
                 &PlotManager::CreatePlot),
             "Create a new plot with full configuration",
             py::arg("config"))
        .def("delete_plot", &PlotManager::DeletePlot,
             "Delete a plot",
             py::arg("plot_id"))
        .def("has_plot", &PlotManager::HasPlot,
             "Check if a plot exists",
             py::arg("plot_id"))
        .def("clear_all_plots", &PlotManager::ClearAllPlots,
             "Delete all plots")
        .def("add_dataset", &PlotManager::AddDataset,
             "Add a dataset to a plot",
             py::arg("plot_id"), py::arg("dataset_name"), py::arg("dataset"))
        .def("remove_dataset", &PlotManager::RemoveDataset,
             "Remove a dataset from a plot",
             py::arg("plot_id"), py::arg("dataset_name"))
        .def("get_dataset", &PlotManager::GetDataset,
             "Get a dataset from a plot",
             py::arg("plot_id"), py::arg("dataset_name"),
             py::return_value_policy::reference_internal)
        .def("render_implot", &PlotManager::RenderImPlot,
             "Render a plot using ImPlot (GUI thread only)",
             py::arg("plot_id"))
        .def("update_realtime_plot", &PlotManager::UpdateRealtimePlot,
             "Update a real-time plot with a new data point",
             py::arg("plot_id"), py::arg("x"), py::arg("y"),
             py::arg("series_name") = "default")
        .def("save_plot", &PlotManager::SavePlotToFile,
             "Save a plot to file",
             py::arg("plot_id"), py::arg("filepath"))
        .def("show_plot", &PlotManager::ShowPlot,
             "Show a plot (matplotlib backend)",
             py::arg("plot_id"))
        .def("calculate_statistics", &PlotManager::CalculateStatistics,
             "Calculate statistics for a dataset",
             py::arg("plot_id"), py::arg("dataset_name"))
        .def("update_plot_config", &PlotManager::UpdatePlotConfig,
             "Update plot configuration",
             py::arg("plot_id"), py::arg("config"))
        .def("get_plot_config", &PlotManager::GetPlotConfig,
             "Get plot configuration",
             py::arg("plot_id"))
        .def("get_all_plot_ids", &PlotManager::GetAllPlotIds,
             "Get all plot IDs")
        .def("get_plot_count", &PlotManager::GetPlotCount,
             "Get total number of plots")
        .def("initialize_python_backend", &PlotManager::InitializePythonBackend,
             "Initialize the Python/Matplotlib backend")
        .def("shutdown_python_backend", &PlotManager::ShutdownPythonBackend,
             "Shutdown the Python/Matplotlib backend");

    // ===== TrainingPlotPanel =====

    py::class_<cyxwiz::TrainingPlotPanel>(m, "TrainingPlotPanel",
        "Real-time training visualization panel for ML training metrics")
        .def("add_loss_point", &cyxwiz::TrainingPlotPanel::AddLossPoint,
             "Add a loss data point for the current epoch",
             py::arg("epoch"), py::arg("train_loss"), py::arg("val_loss") = -1.0)
        .def("add_accuracy_point", &cyxwiz::TrainingPlotPanel::AddAccuracyPoint,
             "Add an accuracy data point for the current epoch",
             py::arg("epoch"), py::arg("train_acc"), py::arg("val_acc") = -1.0)
        .def("add_custom_metric", &cyxwiz::TrainingPlotPanel::AddCustomMetric,
             "Add a custom metric data point",
             py::arg("metric_name"), py::arg("epoch"), py::arg("value"))
        .def("clear", &cyxwiz::TrainingPlotPanel::Clear,
             "Clear all data from the panel")
        .def("reset_plots", &cyxwiz::TrainingPlotPanel::ResetPlots,
             "Reset all plots")
        .def("set_max_points", &cyxwiz::TrainingPlotPanel::SetMaxPoints,
             "Set maximum number of data points to keep",
             py::arg("max_points"))
        .def("export_to_csv", &cyxwiz::TrainingPlotPanel::ExportToCSV,
             "Export training metrics to CSV file",
             py::arg("filepath"))
        .def("export_plot_image", &cyxwiz::TrainingPlotPanel::ExportPlotImage,
             "Export plot as image (TODO: not yet implemented)",
             py::arg("filepath"))
        .def("show_loss_plot", &cyxwiz::TrainingPlotPanel::ShowLossPlot,
             "Show or hide the loss plot",
             py::arg("show"))
        .def("show_accuracy_plot", &cyxwiz::TrainingPlotPanel::ShowAccuracyPlot,
             "Show or hide the accuracy plot",
             py::arg("show"))
        .def("set_auto_scale", &cyxwiz::TrainingPlotPanel::SetAutoScale,
             "Enable or disable auto-scaling",
             py::arg("auto_scale"));

    // Global TrainingPlotPanel accessor functions
    m.def("set_training_plot_panel", &set_training_plot_panel,
          "Set the global TrainingPlotPanel instance (called by MainWindow)",
          py::arg("panel"));

    m.def("get_training_plot_panel", &get_training_plot_panel,
          "Get the global TrainingPlotPanel instance",
          py::return_value_policy::reference);

    // ===== Convenience Functions =====

    m.def("plot_line", &plot_line,
          "Create a line plot",
          py::arg("x_data"), py::arg("y_data"),
          py::arg("title") = "Line Plot",
          py::arg("x_label") = "X",
          py::arg("y_label") = "Y",
          py::arg("series_name") = "data");

    m.def("plot_scatter", &plot_scatter,
          "Create a scatter plot",
          py::arg("x_data"), py::arg("y_data"),
          py::arg("title") = "Scatter Plot",
          py::arg("x_label") = "X",
          py::arg("y_label") = "Y",
          py::arg("series_name") = "data");

    m.def("plot_histogram", &plot_histogram,
          "Create a histogram",
          py::arg("data"),
          py::arg("bins") = 10,
          py::arg("title") = "Histogram",
          py::arg("x_label") = "Value",
          py::arg("y_label") = "Frequency");

    m.def("plot_bar", &plot_bar,
          "Create a bar chart",
          py::arg("x_data"), py::arg("y_data"),
          py::arg("title") = "Bar Chart",
          py::arg("x_label") = "Category",
          py::arg("y_label") = "Value");

    m.def("show_plot", &show_plot,
          "Show a plot in a dockable window (GUI thread only)",
          py::arg("plot_id"));

    m.def("clear_plot_windows", &clear_plot_windows,
          "Clear all Python-created plot windows");

    // ===== Version Info =====

    m.attr("__version__") = "1.0.0";
}
