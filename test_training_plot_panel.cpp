// Simple compilation test for TrainingPlotPanel
#include "cyxwiz-engine/src/gui/panels/training_plot_panel.h"
#include <iostream>

int main() {
    std::cout << "Testing TrainingPlotPanel compilation..." << std::endl;

    // Create panel (this tests that the class compiles correctly)
    cyxwiz::TrainingPlotPanel panel;

    // Test adding data points
    panel.AddLossPoint(0, 1.5, 1.2);
    panel.AddLossPoint(1, 1.2, 1.0);
    panel.AddLossPoint(2, 0.9, 0.8);

    panel.AddAccuracyPoint(0, 45.0, 50.0);
    panel.AddAccuracyPoint(1, 60.0, 62.0);
    panel.AddAccuracyPoint(2, 75.0, 73.0);

    // Test custom metrics
    panel.AddCustomMetric("Learning Rate", 0, 0.1);
    panel.AddCustomMetric("Learning Rate", 1, 0.05);
    panel.AddCustomMetric("Learning Rate", 2, 0.025);

    // Test export
    panel.ExportToCSV("test_metrics.csv");

    std::cout << "TrainingPlotPanel compilation successful!" << std::endl;
    std::cout << "All methods are callable and link correctly." << std::endl;

    return 0;
}
