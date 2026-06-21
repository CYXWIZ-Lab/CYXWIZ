#include "visualizer.h"

namespace cyxwiz {

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
                // vector<bool> doesn't support &selected_columns_[i], use temp bool
                bool selected = selected_columns_[i];
                if (ImGui::Checkbox(available_columns_[i].c_str(), &selected)) {
                    selected_columns_[i] = selected;
                }
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
        // Plot each column as a separate line series
        for (const auto& [col_name, values] : plot.data) {
            if (!values.empty()) {
                // Generate X values (0, 1, 2, ...)
                std::vector<double> x_values(values.size());
                for (size_t i = 0; i < values.size(); i++) {
                    x_values[i] = static_cast<double>(i);
                }

                ImPlot::PlotLine(col_name.c_str(), x_values.data(), values.data(),
                                static_cast<int>(values.size()));
            }
        }
        ImPlot::EndPlot();
    }
}

void Visualizer::RenderScatterPlot(const PlotConfig& plot) {
    if (ImPlot::BeginPlot(plot.name.c_str(), ImVec2(-1, -1))) {
        // For scatter plot, if 2 columns: use first as X, second as Y
        // If more columns: plot each against index
        if (plot.data.size() == 2) {
            auto it = plot.data.begin();
            const auto& x_data = it->second;
            const std::string& x_name = it->first;
            ++it;
            const auto& y_data = it->second;
            const std::string& y_name = it->first;

            size_t count = std::min(x_data.size(), y_data.size());
            if (count > 0) {
                ImPlot::PlotScatter((x_name + " vs " + y_name).c_str(),
                                   x_data.data(), y_data.data(), static_cast<int>(count));
            }
        } else {
            // Plot each column against index
            for (const auto& [col_name, values] : plot.data) {
                if (!values.empty()) {
                    std::vector<double> x_values(values.size());
                    for (size_t i = 0; i < values.size(); i++) {
                        x_values[i] = static_cast<double>(i);
                    }
                    ImPlot::PlotScatter(col_name.c_str(), x_values.data(), values.data(),
                                       static_cast<int>(values.size()));
                }
            }
        }
        ImPlot::EndPlot();
    }
}

void Visualizer::RenderBarChart(const PlotConfig& plot) {
    if (ImPlot::BeginPlot(plot.name.c_str(), ImVec2(-1, -1))) {
        // Plot each column as a separate bar series
        for (const auto& [col_name, values] : plot.data) {
            if (!values.empty()) {
                ImPlot::PlotBars(col_name.c_str(), values.data(), static_cast<int>(values.size()));
            }
        }
        ImPlot::EndPlot();
    }
}

void Visualizer::RenderHistogram(const PlotConfig& plot) {
    if (ImPlot::BeginPlot(plot.name.c_str(), ImVec2(-1, -1))) {
        // Plot histogram for each column
        for (const auto& [col_name, values] : plot.data) {
            if (!values.empty()) {
                // Use 20 bins by default
                int bins = 20;
                ImPlot::PlotHistogram(col_name.c_str(), values.data(),
                                     static_cast<int>(values.size()), bins);
            }
        }
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


} // namespace cyxwiz
