#include "training_plot_panel.h"

// Global accessor for TrainingPlotPanel (shared between GUI and Python)
namespace {
    cyxwiz::TrainingPlotPanel* g_training_plot_panel = nullptr;
}

void set_training_plot_panel(cyxwiz::TrainingPlotPanel* panel) {
    g_training_plot_panel = panel;
}

cyxwiz::TrainingPlotPanel* get_training_plot_panel() {
    return g_training_plot_panel;
}
