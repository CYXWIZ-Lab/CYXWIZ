#include "visualizer.h"
#include "../../core/duckdb_connector.h"
#include <spdlog/spdlog.h>
#include <algorithm>

namespace cyxwiz {

Visualizer::Visualizer()
    : next_plot_id_(1)
    , selected_plot_id_(-1)
    , show_create_plot_dialog_(false)
    , selected_plot_type_(0)
{
    // Create separate ImPlot context for Data Studio
    context_ = ImPlot::CreateContext();
    ImPlot::SetCurrentContext(context_);

    spdlog::info("[Data Studio] Visualizer initialized");
}

Visualizer::~Visualizer() {
    if (context_) {
        ImPlot::SetCurrentContext(context_);
        ImPlot::DestroyContext(context_);
        context_ = nullptr;
    }
}

void Visualizer::Render() {
    // Set Data Studio ImPlot context
    ImPlot::SetCurrentContext(context_);

    RenderPlotToolbar();
    ImGui::Separator();

    // Split view: plot gallery on left, canvas on right
    if (ImGui::BeginChild("PlotGallery", ImVec2(200, 0), true)) {
        RenderPlotGallery();
    }
    ImGui::EndChild();

    ImGui::SameLine();

    if (ImGui::BeginChild("PlotCanvas", ImVec2(0, 0), true)) {
        RenderPlotCanvas();
    }
    ImGui::EndChild();

    // Create plot dialog
    if (show_create_plot_dialog_) {
        RenderCreatePlotDialog();
    }
}

void Visualizer::RenderPlotToolbar() {
    if (ImGui::Button("Create Plot")) {
        show_create_plot_dialog_ = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Delete Selected")) {
        if (selected_plot_id_ >= 0) {
            DeletePlot(selected_plot_id_);
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear All")) {
        ClearAllPlots();
    }
    ImGui::SameLine();
    ImGui::Text("| Plots: %zu", plots_.size());
}

void Visualizer::RenderPlotGallery() {
    ImGui::Text("Plots");
    ImGui::Separator();

    if (plots_.empty()) {
        ImGui::TextDisabled("No plots created yet");
        return;
    }

    // List of created plots
    for (const auto& plot : plots_) {
        bool is_selected = (plot.id == selected_plot_id_);
        if (ImGui::Selectable(plot.name.c_str(), is_selected)) {
            selected_plot_id_ = plot.id;
        }
    }
}

void Visualizer::RenderPlotCanvas() {
    if (selected_plot_id_ < 0) {
        ImGui::TextDisabled("No plot selected. Create a new plot or select from the gallery.");
        return;
    }

    // Find selected plot
    auto it = std::find_if(plots_.begin(), plots_.end(),
                          [this](const PlotConfig& p) { return p.id == selected_plot_id_; });

    if (it == plots_.end()) {
        ImGui::TextDisabled("Plot not found");
        return;
    }

    const auto& plot = *it;

    // Render plot based on type
    switch (plot.type) {
        case PlotType::Line:
            RenderLinePlot(plot);
            break;
        case PlotType::Scatter:
            RenderScatterPlot(plot);
            break;
        case PlotType::Bar:
            RenderBarChart(plot);
            break;
        case PlotType::Histogram:
            RenderHistogram(plot);
            break;
        case PlotType::Heatmap:
            RenderHeatmap(plot);
            break;
        case PlotType::Box:
            RenderBoxPlot(plot);
            break;
    }
}

void Visualizer::RenderCreatePlotDialog() {
    ImGui::OpenPopup("Create Plot");

    ImGui::SetNextWindowSize(ImVec2(400, 500), ImGuiCond_FirstUseEver);
    if (ImGui::BeginPopupModal("Create Plot", &show_create_plot_dialog_)) {
        ImGui::Text("Create New Plot");
        ImGui::Separator();

        // Plot type selection
        const char* plot_types[] = {"Line", "Scatter", "Bar", "Histogram", "Heatmap", "Box"};
        ImGui::Combo("Plot Type", &selected_plot_type_, plot_types, IM_ARRAYSIZE(plot_types));

        ImGui::Spacing();

        // Column selection
        ImGui::Text("Select Columns:");
        ImGui::Separator();

        // TODO: Phase 1 Week 2 - Load actual column names from dataset
        if (available_columns_.empty()) {
            ImGui::TextDisabled("No dataset loaded");
        } else {
            for (size_t i = 0; i < available_columns_.size(); i++) {
                ImGui::Checkbox(available_columns_[i].c_str(), &selected_columns_[i]);
            }
        }

        ImGui::Spacing();

        // Buttons
        if (ImGui::Button("Create")) {
            // Collect selected columns
            std::vector<std::string> selected;
            for (size_t i = 0; i < available_columns_.size(); i++) {
                if (selected_columns_[i]) {
                    selected.push_back(available_columns_[i]);
                }
            }

            if (!selected.empty()) {
                CreatePlot(plot_types[selected_plot_type_], selected);
                show_create_plot_dialog_ = false;
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
            show_create_plot_dialog_ = false;
        }

        ImGui::EndPopup();
    }
}

void Visualizer::RenderLinePlot(const PlotConfig& plot) {
    if (ImPlot::BeginPlot(plot.name.c_str(), ImVec2(-1, -1))) {
        // TODO: Phase 1 Week 2 - Plot actual data
        // Placeholder: plot dummy data
        ImPlot::PlotLine("Series 1", nullptr, nullptr, 0);

        ImPlot::EndPlot();
    }
}

void Visualizer::RenderScatterPlot(const PlotConfig& plot) {
    if (ImPlot::BeginPlot(plot.name.c_str(), ImVec2(-1, -1))) {
        // TODO: Phase 1 Week 2 - Plot actual data
        ImPlot::PlotScatter("Series 1", nullptr, nullptr, 0);
        ImPlot::EndPlot();
    }
}

void Visualizer::RenderBarChart(const PlotConfig& plot) {
    if (ImPlot::BeginPlot(plot.name.c_str(), ImVec2(-1, -1))) {
        // TODO: Phase 1 Week 2 - Plot actual data
        ImPlot::EndPlot();
    }
}

void Visualizer::RenderHistogram(const PlotConfig& plot) {
    if (ImPlot::BeginPlot(plot.name.c_str(), ImVec2(-1, -1))) {
        // TODO: Phase 1 Week 2 - Plot actual data
        ImPlot::EndPlot();
    }
}

void Visualizer::RenderHeatmap(const PlotConfig& plot) {
    if (ImPlot::BeginPlot(plot.name.c_str(), ImVec2(-1, -1))) {
        // TODO: Phase 1 Week 2 - Plot actual data
        ImPlot::EndPlot();
    }
}

void Visualizer::RenderBoxPlot(const PlotConfig& plot) {
    if (ImPlot::BeginPlot(plot.name.c_str(), ImVec2(-1, -1))) {
        // TODO: Phase 1 Week 2 - Plot actual data
        ImPlot::EndPlot();
    }
}

void Visualizer::SetActiveDataset(const std::string& dataset_name) {
    current_dataset_ = dataset_name;
    spdlog::info("[Data Studio] Visualizer set active dataset: {}", dataset_name);

    // TODO: Phase 1 Week 2 - Load column names from dataset
    available_columns_ = {"Column1", "Column2", "Column3"};
    selected_columns_.resize(available_columns_.size(), false);
}

void Visualizer::CreatePlot(const std::string& plot_type,
                           const std::vector<std::string>& columns) {
    PlotConfig plot;
    plot.id = next_plot_id_++;
    plot.name = plot_type + " Plot #" + std::to_string(plot.id);
    plot.columns = columns;
    plot.x_axis_label = "X";
    plot.y_axis_label = "Y";
    plot.show_legend = true;

    // Parse plot type
    if (plot_type == "Line") plot.type = PlotType::Line;
    else if (plot_type == "Scatter") plot.type = PlotType::Scatter;
    else if (plot_type == "Bar") plot.type = PlotType::Bar;
    else if (plot_type == "Histogram") plot.type = PlotType::Histogram;
    else if (plot_type == "Heatmap") plot.type = PlotType::Heatmap;
    else if (plot_type == "Box") plot.type = PlotType::Box;

    // Load data
    LoadPlotData(plot);

    plots_.push_back(plot);
    selected_plot_id_ = plot.id;

    spdlog::info("[Data Studio] Created plot: {} with {} columns",
                 plot.name, columns.size());
}

bool Visualizer::LoadPlotData(PlotConfig& plot) {
    // TODO: Phase 1 Week 2 - Load actual data from DuckDB
    // For now, generate placeholder data
    for (const auto& col : plot.columns) {
        std::vector<double> values;
        for (int i = 0; i < 100; i++) {
            values.push_back(static_cast<double>(i));
        }
        plot.data[col] = values;
    }
    return true;
}

void Visualizer::DeletePlot(int plot_id) {
    plots_.erase(
        std::remove_if(plots_.begin(), plots_.end(),
                      [plot_id](const PlotConfig& p) { return p.id == plot_id; }),
        plots_.end()
    );

    if (selected_plot_id_ == plot_id) {
        selected_plot_id_ = plots_.empty() ? -1 : plots_[0].id;
    }

    spdlog::info("[Data Studio] Deleted plot: {}", plot_id);
}

void Visualizer::ClearAllPlots() {
    plots_.clear();
    selected_plot_id_ = -1;
    spdlog::info("[Data Studio] Cleared all plots");
}

} // namespace cyxwiz
