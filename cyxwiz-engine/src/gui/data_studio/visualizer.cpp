#include "visualizer.h"
#include "../../core/duckdb_connector.h"
#include "../../core/data_registry.h"
#include "../../core/arrow_dataset.h"
#include <spdlog/spdlog.h>
#include <algorithm>

namespace cyxwiz {

Visualizer::Visualizer()
    : next_plot_id_(1)
    , selected_plot_id_(-1)
    , show_create_plot_dialog_(false)
    , selected_plot_type_(0)
    , duckdb_(std::make_unique<DuckDBConnector>())
{
    // Create separate ImPlot context for Data Studio
    context_ = ImPlot::CreateContext();
    ImPlot::SetCurrentContext(context_);

    spdlog::info("[Data Studio] Visualizer initialized");
}

Visualizer::~Visualizer() {
    // Skip context cleanup - ImPlot contexts are cleaned up automatically
    // when ImGui shuts down. Explicit cleanup can cause crashes during
    // application shutdown if ImGui is already destroyed.
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

void Visualizer::SetActiveDataset(const std::string& dataset_name) {
    current_dataset_ = dataset_name;
    spdlog::info("[Data Studio] Visualizer set active dataset: {}", dataset_name);

    // Clear previous data
    available_columns_.clear();
    selected_columns_.clear();

    if (dataset_name.empty()) {
        return;
    }

    try {
        // Get Arrow dataset from DataRegistry
        auto& registry = DataRegistry::Instance();
        auto arrow_dataset = registry.GetArrowDataset(dataset_name);

        if (!arrow_dataset) {
            spdlog::warn("[Data Studio] Visualizer: dataset not found in registry");
            return;
        }

        auto arrow_table = arrow_dataset->GetArrowTable();
        if (!arrow_table) {
            spdlog::warn("[Data Studio] Visualizer: dataset has no Arrow table");
            return;
        }

        // Register table with DuckDB
        std::string table_name = "viz_table";
        if (!duckdb_->RegisterTable(table_name, arrow_table)) {
            spdlog::error("[Data Studio] Visualizer: failed to register table: {}",
                         duckdb_->GetLastError());
            return;
        }

        // Get column names from schema
        auto schema = duckdb_->GetTableSchema(table_name);
        for (const auto& col : schema) {
            available_columns_.push_back(col.name);
        }

        selected_columns_.resize(available_columns_.size(), false);

        // Unregister table (will re-register when loading plot data)
        duckdb_->UnregisterTable(table_name);

        spdlog::info("[Data Studio] Visualizer: loaded {} columns from {}",
                     available_columns_.size(), dataset_name);

    } catch (const std::exception& e) {
        spdlog::error("[Data Studio] Visualizer: failed to load dataset: {}", e.what());
    }
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
    if (current_dataset_.empty()) {
        spdlog::warn("[Data Studio] Visualizer: no dataset selected");
        return false;
    }

    if (plot.columns.empty()) {
        spdlog::warn("[Data Studio] Visualizer: no columns selected for plot");
        return false;
    }

    try {
        // Get Arrow dataset from DataRegistry
        auto& registry = DataRegistry::Instance();
        auto arrow_dataset = registry.GetArrowDataset(current_dataset_);

        if (!arrow_dataset) {
            spdlog::error("[Data Studio] Visualizer: dataset not found");
            return false;
        }

        auto arrow_table = arrow_dataset->GetArrowTable();
        if (!arrow_table) {
            spdlog::error("[Data Studio] Visualizer: dataset has no Arrow table");
            return false;
        }

        // Register table with DuckDB
        std::string table_name = "viz_data";
        if (!duckdb_->RegisterTable(table_name, arrow_table)) {
            spdlog::error("[Data Studio] Visualizer: failed to register table: {}",
                         duckdb_->GetLastError());
            return false;
        }

        // Query selected columns
        std::string cols = plot.columns[0];
        for (size_t i = 1; i < plot.columns.size(); i++) {
            cols += ", " + plot.columns[i];
        }

        std::string sql = "SELECT " + cols + " FROM " + table_name + " LIMIT 10000";
        auto result_table = duckdb_->Query(sql);

        if (!result_table) {
            spdlog::error("[Data Studio] Visualizer: query failed: {}", duckdb_->GetLastError());
            duckdb_->UnregisterTable(table_name);
            return false;
        }

        // Convert Arrow columns to double vectors
        plot.data.clear();
        for (size_t col_idx = 0; col_idx < plot.columns.size(); col_idx++) {
            const std::string& col_name = plot.columns[col_idx];
            auto column = result_table->column(static_cast<int>(col_idx));

            std::vector<double> values;
            values.reserve(result_table->num_rows());

            // Process each chunk
            for (int chunk_idx = 0; chunk_idx < column->num_chunks(); chunk_idx++) {
                auto chunk = column->chunk(chunk_idx);

                // Convert based on Arrow type
                auto type_id = chunk->type_id();

                if (type_id == arrow::Type::DOUBLE) {
                    auto typed_array = std::static_pointer_cast<arrow::DoubleArray>(chunk);
                    for (int64_t i = 0; i < typed_array->length(); i++) {
                        if (!typed_array->IsNull(i)) {
                            values.push_back(typed_array->Value(i));
                        } else {
                            values.push_back(0.0);  // Replace null with 0
                        }
                    }
                } else if (type_id == arrow::Type::FLOAT) {
                    auto typed_array = std::static_pointer_cast<arrow::FloatArray>(chunk);
                    for (int64_t i = 0; i < typed_array->length(); i++) {
                        if (!typed_array->IsNull(i)) {
                            values.push_back(static_cast<double>(typed_array->Value(i)));
                        } else {
                            values.push_back(0.0);
                        }
                    }
                } else if (type_id == arrow::Type::INT64) {
                    auto typed_array = std::static_pointer_cast<arrow::Int64Array>(chunk);
                    for (int64_t i = 0; i < typed_array->length(); i++) {
                        if (!typed_array->IsNull(i)) {
                            values.push_back(static_cast<double>(typed_array->Value(i)));
                        } else {
                            values.push_back(0.0);
                        }
                    }
                } else if (type_id == arrow::Type::INT32) {
                    auto typed_array = std::static_pointer_cast<arrow::Int32Array>(chunk);
                    for (int64_t i = 0; i < typed_array->length(); i++) {
                        if (!typed_array->IsNull(i)) {
                            values.push_back(static_cast<double>(typed_array->Value(i)));
                        } else {
                            values.push_back(0.0);
                        }
                    }
                } else {
                    // Unsupported type - skip
                    spdlog::warn("[Data Studio] Visualizer: unsupported column type for plotting: {}",
                                chunk->type()->ToString());
                }
            }

            plot.data[col_name] = values;
        }

        // Unregister table
        duckdb_->UnregisterTable(table_name);

        spdlog::info("[Data Studio] Visualizer: loaded plot data with {} columns, {} rows",
                     plot.columns.size(), plot.data.empty() ? 0 : plot.data.begin()->second.size());

        return true;

    } catch (const std::exception& e) {
        spdlog::error("[Data Studio] Visualizer: failed to load plot data: {}", e.what());
        return false;
    }
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
