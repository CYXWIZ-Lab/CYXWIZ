"""
Type stubs for cyxwiz_plotting module

This file provides type hints for IDE autocompletion and static analysis.
The actual implementation is in C++ (plot_bindings.cpp).
"""

from typing import List, Tuple, Union, Optional
from enum import IntEnum
import numpy as np


class BackendType(IntEnum):
    """Plot backend selection"""
    ImPlot: int      # Real-time plotting integrated into ImGui
    Matplotlib: int  # Offline plotting using Matplotlib


class PlotType(IntEnum):
    """Plot type selection"""
    Line: int         # Line plot
    Scatter: int      # Scatter plot
    Histogram: int    # Histogram
    BoxPlot: int      # Box plot
    Violin: int       # Violin plot
    KDE: int          # Kernel density estimation
    QQPlot: int       # Q-Q plot
    MosaicPlot: int   # Mosaic plot
    StemLeaf: int     # Stem-and-leaf plot
    DotChart: int     # Dot chart
    Heatmap: int      # Heatmap
    Bar: int          # Bar chart


class PlotConfig:
    """Plot configuration"""
    title: str
    x_label: str
    y_label: str
    type: PlotType
    backend: BackendType
    auto_fit: bool
    show_legend: bool
    show_grid: bool
    width: int
    height: int

    def __init__(self) -> None: ...


class Statistics:
    """Statistical measures for a dataset"""
    min: float
    max: float
    mean: float
    median: float
    std_dev: float
    q1: float  # First quartile (25th percentile)
    q3: float  # Third quartile (75th percentile)


class Series:
    """Data series within a dataset"""
    name: str
    x_data: List[float]
    y_data: List[float]

    def __init__(self) -> None: ...

    def add_point(self, x: float, y: float) -> None:
        """Add a single point to the series"""
        ...

    def clear(self) -> None:
        """Clear all data from the series"""
        ...

    def size(self) -> int:
        """Get number of points in the series"""
        ...


class PlotDataset:
    """Container for plot data with multiple series"""

    def __init__(self) -> None: ...

    def add_series(self, name: str) -> None:
        """Add a new series to the dataset"""
        ...

    def has_series(self, name: str) -> bool:
        """Check if a series exists"""
        ...

    def get_series(self, name: str) -> Optional[Series]:
        """Get a series by name"""
        ...

    def get_series_names(self) -> List[str]:
        """Get all series names"""
        ...

    def get_series_count(self) -> int:
        """Get number of series"""
        ...

    def add_point(self, x: float, y: float) -> None:
        """Add a point to the default series"""
        ...

    def clear(self) -> None:
        """Clear all data"""
        ...

    def is_empty(self) -> bool:
        """Check if dataset is empty"""
        ...

    def save_to_json(self, filepath: str) -> bool:
        """Save dataset to JSON file"""
        ...

    def load_from_json(self, filepath: str) -> bool:
        """Load dataset from JSON file"""
        ...


class PlotManager:
    """Singleton manager for all plotting operations"""

    @staticmethod
    def get_instance() -> 'PlotManager':
        """Get the singleton PlotManager instance"""
        ...

    def set_default_backend(self, backend: BackendType) -> None:
        """Set the default backend for new plots"""
        ...

    def get_default_backend(self) -> BackendType:
        """Get the default backend"""
        ...

    def is_backend_available(self, backend: BackendType) -> bool:
        """Check if a backend is available"""
        ...

    def create_plot(
        self,
        title: str,
        x_label: str,
        y_label: str,
        plot_type: PlotType,
        backend_type: BackendType
    ) -> str:
        """Create a new plot (overload 1)"""
        ...

    def create_plot(self, config: PlotConfig) -> str:
        """Create a new plot with full configuration (overload 2)"""
        ...

    def delete_plot(self, plot_id: str) -> bool:
        """Delete a plot"""
        ...

    def has_plot(self, plot_id: str) -> bool:
        """Check if a plot exists"""
        ...

    def clear_all_plots(self) -> None:
        """Delete all plots"""
        ...

    def add_dataset(
        self,
        plot_id: str,
        dataset_name: str,
        dataset: PlotDataset
    ) -> bool:
        """Add a dataset to a plot"""
        ...

    def remove_dataset(self, plot_id: str, dataset_name: str) -> bool:
        """Remove a dataset from a plot"""
        ...

    def get_dataset(
        self,
        plot_id: str,
        dataset_name: str
    ) -> Optional[PlotDataset]:
        """Get a dataset from a plot"""
        ...

    def render_implot(self, plot_id: str) -> None:
        """Render a plot using ImPlot (GUI thread only)"""
        ...

    def update_realtime_plot(
        self,
        plot_id: str,
        x: float,
        y: float,
        series_name: str = "default"
    ) -> bool:
        """Update a real-time plot with a new data point"""
        ...

    def save_plot(self, plot_id: str, filepath: str) -> bool:
        """Save a plot to file"""
        ...

    def show_plot(self, plot_id: str) -> bool:
        """Show a plot (matplotlib backend)"""
        ...

    def calculate_statistics(
        self,
        plot_id: str,
        dataset_name: str
    ) -> Statistics:
        """Calculate statistics for a dataset"""
        ...

    def update_plot_config(self, plot_id: str, config: PlotConfig) -> bool:
        """Update plot configuration"""
        ...

    def get_plot_config(self, plot_id: str) -> PlotConfig:
        """Get plot configuration"""
        ...

    def get_all_plot_ids(self) -> List[str]:
        """Get all plot IDs"""
        ...

    def get_plot_count(self) -> int:
        """Get total number of plots"""
        ...

    def initialize_python_backend(self) -> bool:
        """Initialize the Python/Matplotlib backend"""
        ...

    def shutdown_python_backend(self) -> None:
        """Shutdown the Python/Matplotlib backend"""
        ...


# Type alias for data that can be plotted
PlottableData = Union[List[float], np.ndarray]


def plot_line(
    x_data: PlottableData,
    y_data: PlottableData,
    title: str = "Line Plot",
    x_label: str = "X",
    y_label: str = "Y",
    series_name: str = "data"
) -> str:
    """
    Create a line plot

    Args:
        x_data: X coordinates (list or numpy array)
        y_data: Y coordinates (list or numpy array)
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        series_name: Name for the data series

    Returns:
        Plot ID string for further manipulation
    """
    ...


def plot_scatter(
    x_data: PlottableData,
    y_data: PlottableData,
    title: str = "Scatter Plot",
    x_label: str = "X",
    y_label: str = "Y",
    series_name: str = "data"
) -> str:
    """
    Create a scatter plot

    Args:
        x_data: X coordinates
        y_data: Y coordinates
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        series_name: Name for the data series

    Returns:
        Plot ID string
    """
    ...


def plot_histogram(
    data: PlottableData,
    bins: int = 10,
    title: str = "Histogram",
    x_label: str = "Value",
    y_label: str = "Frequency"
) -> str:
    """
    Create a histogram

    Args:
        data: Data values
        bins: Number of bins
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label

    Returns:
        Plot ID string
    """
    ...


def plot_bar(
    x_data: PlottableData,
    y_data: PlottableData,
    title: str = "Bar Chart",
    x_label: str = "Category",
    y_label: str = "Value"
) -> str:
    """
    Create a bar chart

    Args:
        x_data: X coordinates (categories)
        y_data: Y values
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label

    Returns:
        Plot ID string
    """
    ...


def show_plot(plot_id: str) -> None:
    """
    Show a plot in a dockable window (GUI thread only)

    Args:
        plot_id: ID of the plot to display
    """
    ...


def clear_plot_windows() -> None:
    """Clear all Python-created plot windows"""
    ...


__version__: str
