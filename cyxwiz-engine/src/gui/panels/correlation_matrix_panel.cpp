#include "correlation_matrix_panel.h"
#include "../../data/data_table.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <algorithm>
#include <cmath>

namespace cyxwiz {

CorrelationMatrixPanel::CorrelationMatrixPanel() {
    std::memset(export_path_, 0, sizeof(export_path_));
}

CorrelationMatrixPanel::~CorrelationMatrixPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        is_computing_ = false;
        compute_thread_->join();
    }
}

void CorrelationMatrixPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(600, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_TABLE " Correlation Matrix###CorrelationMatrix", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_computing_.load()) {
            RenderLoadingIndicator();
        } else if (has_correlation_) {
            RenderHeatmap();
            RenderSelectedInfo();
        } else {
            ImGui::Spacing();
            ImGui::TextDisabled("Select a data table to compute correlations");
            ImGui::Spacing();
            ImGui::TextWrapped("Only numeric columns will be included in the correlation matrix.");
        }
    }
    ImGui::End();
}

void CorrelationMatrixPanel::RenderToolbar() {
    RenderDataSelector();

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_ARROWS_ROTATE " Compute")) {
        if (current_table_) {
            ComputeAsync(current_table_);
        }
    }

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_FILE_EXPORT " Export")) {
        ImGui::OpenPopup("ExportCorr");
    }

    if (ImGui::BeginPopup("ExportCorr")) {
        ImGui::InputText("Path", export_path_, sizeof(export_path_));
        if (ImGui::Button("Save CSV")) {
            ExportToCSV(export_path_);
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_GEAR)) {
        ImGui::OpenPopup("CorrSettings");
    }

    if (ImGui::BeginPopup("CorrSettings")) {
        ImGui::Text("Display Settings");
        ImGui::Separator();
        ImGui::SliderFloat("Cell Size", &cell_size_, 20.0f, 60.0f);
        ImGui::Checkbox("Show Values", &show_values_);
        ImGui::Checkbox("Show Legend", &show_legend_);
        ImGui::EndPopup();
    }
}

void CorrelationMatrixPanel::RenderDataSelector() {
    auto& registry = DataTableRegistry::Instance();
    auto table_names = registry.GetTableNames();

    ImGui::SetNextItemWidth(200);
    if (ImGui::BeginCombo("##TableSelect", selected_table_.empty() ?
                          "Select table..." : selected_table_.c_str())) {
        for (const auto& name : table_names) {
            bool is_selected = (name == selected_table_);
            if (ImGui::Selectable(name.c_str(), is_selected)) {
                selected_table_ = name;
                auto table = registry.GetTable(name);
                if (table) {
                    AnalyzeTable(table);
                }
            }
            if (is_selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
}

void CorrelationMatrixPanel::RenderLoadingIndicator() {
    ImGui::Spacing();
    ImGui::Text("Computing correlations...");
    ImGui::Spacing();

    // Simple spinner
    static float rotation = 0.0f;
    rotation += ImGui::GetIO().DeltaTime * 5.0f;
    ImGui::TextColored(ImVec4(0.5f, 0.7f, 1.0f, 1.0f),
                       "%s Computing...", ICON_FA_SPINNER);
}

ImVec4 CorrelationMatrixPanel::CorrelationToColor(double corr) const {
    // RdBu colormap: Red (-1) -> White (0) -> Blue (+1)
    // Clamp correlation
    corr = std::max(-1.0, std::min(1.0, corr));

    float r, g, b;

    if (corr < 0) {
        // Negative: Red to White
        float t = static_cast<float>(-corr);
        r = 1.0f;
        g = 1.0f - t * 0.6f;
        b = 1.0f - t * 0.6f;
    } else {
        // Positive: White to Blue
        float t = static_cast<float>(corr);
        r = 1.0f - t * 0.6f;
        g = 1.0f - t * 0.3f;
        b = 1.0f;
    }

    return ImVec4(r, g, b, 1.0f);
}

void CorrelationMatrixPanel::RenderHeatmap() {
    std::lock_guard<std::mutex> lock(corr_mutex_);

    size_t n = correlation_.column_names.size();
    if (n == 0) {
        ImGui::TextDisabled("No numeric columns found");
        return;
    }

    // Header with column names summary
    ImGui::Text("%zu numeric columns", n);
    if (show_legend_) {
        ImGui::SameLine();
        RenderLegend();
    }

    ImGui::Spacing();

    // Calculate content size
    float matrix_width = cell_size_ * n + 100;  // +100 for row labels
    float matrix_height = cell_size_ * n + 50;   // +50 for header

    ImGui::BeginChild("HeatmapScroll", ImVec2(0, 0), true,
                      ImGuiWindowFlags_HorizontalScrollbar);

    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // Draw column headers (rotated or abbreviated)
    float header_x = canvas_pos.x + 100;
    for (size_t j = 0; j < n; j++) {
        std::string label = correlation_.column_names[j];
        if (label.length() > 6) {
            label = label.substr(0, 5) + "..";
        }

        ImVec2 text_pos(header_x + j * cell_size_ + 2, canvas_pos.y + 5);
        draw_list->AddText(text_pos, IM_COL32(200, 200, 200, 255), label.c_str());
    }

    // Reset hover state
    hovered_i_ = -1;
    hovered_j_ = -1;

    // Draw matrix cells
    float start_y = canvas_pos.y + 40;

    for (size_t i = 0; i < n; i++) {
        // Row label
        std::string row_label = correlation_.column_names[i];
        if (row_label.length() > 12) {
            row_label = row_label.substr(0, 11) + "..";
        }

        ImVec2 label_pos(canvas_pos.x + 5, start_y + i * cell_size_ + cell_size_ / 2 - 6);
        draw_list->AddText(label_pos, IM_COL32(200, 200, 200, 255), row_label.c_str());

        for (size_t j = 0; j < n; j++) {
            double corr = correlation_.Get(i, j);
            ImVec4 color = CorrelationToColor(corr);

            ImVec2 cell_min(canvas_pos.x + 100 + j * cell_size_, start_y + i * cell_size_);
            ImVec2 cell_max(cell_min.x + cell_size_ - 2, cell_min.y + cell_size_ - 2);

            // Fill cell
            draw_list->AddRectFilled(cell_min, cell_max,
                                     ImGui::ColorConvertFloat4ToU32(color));

            // Border
            draw_list->AddRect(cell_min, cell_max, IM_COL32(80, 80, 80, 255));

            // Value text
            if (show_values_ && cell_size_ >= 25) {
                char buf[16];
                snprintf(buf, sizeof(buf), "%.2f", corr);

                // Use dark text on light backgrounds
                ImU32 text_color = (std::abs(corr) < 0.5) ?
                    IM_COL32(40, 40, 40, 255) : IM_COL32(255, 255, 255, 255);

                ImVec2 text_size = ImGui::CalcTextSize(buf);
                ImVec2 text_pos(cell_min.x + (cell_size_ - 2 - text_size.x) / 2,
                                cell_min.y + (cell_size_ - 2 - text_size.y) / 2);

                draw_list->AddText(text_pos, text_color, buf);
            }

            // Check hover
            ImVec2 mouse = ImGui::GetMousePos();
            if (mouse.x >= cell_min.x && mouse.x < cell_max.x &&
                mouse.y >= cell_min.y && mouse.y < cell_max.y) {
                hovered_i_ = static_cast<int>(i);
                hovered_j_ = static_cast<int>(j);

                // Highlight border
                draw_list->AddRect(cell_min, cell_max, IM_COL32(255, 255, 0, 255), 0.0f, 0, 2.0f);
            }
        }
    }

    // Adjust for scrollable content
    ImGui::Dummy(ImVec2(matrix_width, matrix_height));

    ImGui::EndChild();
}

void CorrelationMatrixPanel::RenderLegend() {
    // Simple horizontal gradient legend
    ImGui::SameLine(ImGui::GetWindowWidth() - 200);

    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    float legend_width = 120.0f;
    float legend_height = 15.0f;

    // Draw gradient
    for (int i = 0; i < static_cast<int>(legend_width); i++) {
        float t = static_cast<float>(i) / legend_width;  // 0 to 1
        double corr = t * 2.0 - 1.0;  // -1 to +1
        ImVec4 color = CorrelationToColor(corr);

        ImVec2 p1(pos.x + i, pos.y);
        ImVec2 p2(pos.x + i + 1, pos.y + legend_height);
        draw_list->AddRectFilled(p1, p2, ImGui::ColorConvertFloat4ToU32(color));
    }

    // Labels
    draw_list->AddText(ImVec2(pos.x - 15, pos.y), IM_COL32(200, 200, 200, 255), "-1");
    draw_list->AddText(ImVec2(pos.x + legend_width / 2 - 5, pos.y), IM_COL32(200, 200, 200, 255), "0");
    draw_list->AddText(ImVec2(pos.x + legend_width + 5, pos.y), IM_COL32(200, 200, 200, 255), "+1");

    ImGui::Dummy(ImVec2(legend_width + 30, legend_height));
}

void CorrelationMatrixPanel::RenderSelectedInfo() {
    if (hovered_i_ >= 0 && hovered_j_ >= 0) {
        std::lock_guard<std::mutex> lock(corr_mutex_);

        size_t i = static_cast<size_t>(hovered_i_);
        size_t j = static_cast<size_t>(hovered_j_);

        if (i < correlation_.column_names.size() && j < correlation_.column_names.size()) {
            double corr = correlation_.Get(i, j);

            ImGui::Separator();
            ImGui::Text("Correlation: %s vs %s",
                        correlation_.column_names[i].c_str(),
                        correlation_.column_names[j].c_str());
            ImGui::Text("Pearson r = %.4f", corr);

            // Interpretation
            const char* strength = "None";
            if (std::abs(corr) >= 0.8) strength = "Very Strong";
            else if (std::abs(corr) >= 0.6) strength = "Strong";
            else if (std::abs(corr) >= 0.4) strength = "Moderate";
            else if (std::abs(corr) >= 0.2) strength = "Weak";

            ImGui::Text("Strength: %s %s", strength, corr >= 0 ? "(positive)" : "(negative)");
        }
    }
}

void CorrelationMatrixPanel::AnalyzeTable(const std::string& table_name) {
    auto& registry = DataTableRegistry::Instance();
    auto table = registry.GetTable(table_name);
    if (table) {
        selected_table_ = table_name;
        AnalyzeTable(table);
    }
}

void CorrelationMatrixPanel::AnalyzeTable(std::shared_ptr<DataTable> table) {
    if (!table) return;
    current_table_ = table;
    ComputeAsync(table);
}

void CorrelationMatrixPanel::ComputeAsync(std::shared_ptr<DataTable> table) {
    if (is_computing_.load()) {
        spdlog::warn("Correlation computation already in progress");
        return;
    }

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;

    compute_thread_ = std::make_unique<std::thread>([this, table]() {
        try {
            DataAnalyzer analyzer;
            auto corr = analyzer.ComputeCorrelationMatrix(*table);

            {
                std::lock_guard<std::mutex> lock(corr_mutex_);
                correlation_ = std::move(corr);
                has_correlation_ = true;
            }

            spdlog::info("Correlation matrix computed: {}x{}",
                         correlation_.column_names.size(),
                         correlation_.column_names.size());

        } catch (const std::exception& e) {
            spdlog::error("Correlation computation error: {}", e.what());
        }

        is_computing_ = false;
    });
}

void CorrelationMatrixPanel::ExportToCSV(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(corr_mutex_);

    std::ofstream file(filepath);
    if (!file) {
        spdlog::error("Failed to open file: {}", filepath);
        return;
    }

    // Header row
    file << "";
    for (const auto& name : correlation_.column_names) {
        file << "," << name;
    }
    file << "\n";

    // Data rows
    for (size_t i = 0; i < correlation_.column_names.size(); i++) {
        file << correlation_.column_names[i];
        for (size_t j = 0; j < correlation_.column_names.size(); j++) {
            file << "," << correlation_.Get(i, j);
        }
        file << "\n";
    }

    spdlog::info("Correlation matrix exported to: {}", filepath);
}

} // namespace cyxwiz
