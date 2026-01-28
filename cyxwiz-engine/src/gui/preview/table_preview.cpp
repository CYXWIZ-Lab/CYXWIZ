#include "table_preview.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace gui {

void TablePreviewRenderer::LoadFromFile(const std::string& filepath, char delimiter, int max_rows) {
    // Avoid reloading same file
    static std::string last_path;
    if (filepath == last_path && !rows_.empty()) return;
    last_path = filepath;

    headers_.clear();
    rows_.clear();
    current_page_ = 0;

    std::ifstream file(filepath);
    if (!file) {
        spdlog::warn("TablePreview: Cannot open file: {}", filepath);
        return;
    }

    std::string line;
    bool first_line = true;

    while (std::getline(file, line) && static_cast<int>(rows_.size()) < max_rows) {
        if (line.empty()) continue;

        std::vector<std::string> tokens;
        std::stringstream ss(line);
        std::string token;

        // Handle quoted fields for CSV
        if (delimiter == ',') {
            bool in_quotes = false;
            std::string field;
            for (size_t i = 0; i < line.size(); i++) {
                char c = line[i];
                if (c == '"') {
                    in_quotes = !in_quotes;
                } else if (c == delimiter && !in_quotes) {
                    tokens.push_back(field);
                    field.clear();
                } else {
                    field += c;
                }
            }
            tokens.push_back(field);
        } else {
            while (std::getline(ss, token, delimiter)) {
                // Trim whitespace
                auto start = token.find_first_not_of(" \t\r");
                auto end = token.find_last_not_of(" \t\r");
                if (start != std::string::npos) {
                    tokens.push_back(token.substr(start, end - start + 1));
                } else {
                    tokens.push_back("");
                }
            }
        }

        if (first_line && has_header_) {
            // Detect if first row is header (non-numeric)
            bool is_header = false;
            for (const auto& t : tokens) {
                if (!t.empty()) {
                    try {
                        std::stod(t);
                    } catch (...) {
                        is_header = true;
                        break;
                    }
                }
            }

            if (is_header) {
                headers_ = tokens;
                first_line = false;
                continue;
            }
        }

        rows_.push_back(tokens);
        first_line = false;
    }

    // Generate default headers if none found
    if (headers_.empty() && !rows_.empty()) {
        for (size_t i = 0; i < rows_[0].size(); i++) {
            headers_.push_back("Col " + std::to_string(i));
        }
    }

    spdlog::info("TablePreview: Loaded {} rows, {} columns from {}", rows_.size(), headers_.size(), filepath);
}

void TablePreviewRenderer::LoadFromDataset(const cyxwiz::DatasetHandle& handle,
                                            const std::vector<std::string>& column_names) {
    // Avoid reloading if data exists
    if (!rows_.empty()) return;

    headers_ = column_names;
    rows_.clear();
    current_page_ = 0;

    if (!handle.IsValid()) return;

    size_t num_samples = std::min(handle.Size(), static_cast<size_t>(1000));

    for (size_t i = 0; i < num_samples; i++) {
        auto [data, label] = handle.GetSample(i);
        std::vector<std::string> row;

        for (float val : data) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.4f", val);
            row.push_back(buf);
        }
        row.push_back(std::to_string(label));
        rows_.push_back(std::move(row));
    }

    // Add "Label" column if not in headers
    if (!headers_.empty() && headers_.size() < rows_[0].size()) {
        headers_.push_back("Label");
    } else if (headers_.empty() && !rows_.empty()) {
        for (size_t i = 0; i < rows_[0].size() - 1; i++) {
            headers_.push_back("F" + std::to_string(i));
        }
        headers_.push_back("Label");
    }
}

void TablePreviewRenderer::SetData(const std::vector<std::string>& headers,
                                    const std::vector<std::vector<std::string>>& rows) {
    headers_ = headers;
    rows_ = rows;
    current_page_ = 0;
}

void TablePreviewRenderer::Render() {
    if (rows_.empty()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No data to display.");
        return;
    }

    // Info and pagination
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
        "%zu rows, %zu columns", rows_.size(), headers_.size());

    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    ImGui::InputInt("Rows/page", &rows_per_page_, 0, 0);
    rows_per_page_ = std::clamp(rows_per_page_, 10, 500);

    RenderPagination();
    ImGui::Separator();

    // Calculate page bounds
    int start_row = current_page_ * rows_per_page_;
    int end_row = std::min(start_row + rows_per_page_, static_cast<int>(rows_.size()));

    // Determine number of columns (cap at 50 for performance)
    int num_cols = std::min(static_cast<int>(headers_.size()), 50);

    ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollX |
                            ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit |
                            ImGuiTableFlags_Reorderable;

    // Add row number column
    if (ImGui::BeginTable("##TablePreview", num_cols + 1, flags, ImGui::GetContentRegionAvail())) {
        // Row number column
        ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 50);
        ImGui::TableSetupScrollFreeze(1, 1);

        // Data columns
        for (int c = 0; c < num_cols; c++) {
            float col_width = std::min(150.0f, std::max(60.0f, ImGui::CalcTextSize(headers_[c].c_str()).x + 20));
            ImGui::TableSetupColumn(headers_[c].c_str(), ImGuiTableColumnFlags_WidthFixed, col_width);
        }

        ImGui::TableHeadersRow();

        // Data rows
        for (int r = start_row; r < end_row; r++) {
            ImGui::TableNextRow();

            // Row number
            ImGui::TableNextColumn();
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
            ImGui::Text("%d", r + 1);
            ImGui::PopStyleColor();

            // Data cells
            const auto& row = rows_[r];
            for (int c = 0; c < num_cols; c++) {
                ImGui::TableNextColumn();
                if (c < static_cast<int>(row.size())) {
                    ImGui::TextUnformatted(row[c].c_str());
                }
            }
        }

        ImGui::EndTable();
    }

    // Show note if columns were truncated
    if (static_cast<int>(headers_.size()) > 50) {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
            "Showing 50 of %zu columns", headers_.size());
    }
}

void TablePreviewRenderer::RenderPagination() {
    int total_pages = std::max(1, (static_cast<int>(rows_.size()) + rows_per_page_ - 1) / rows_per_page_);

    ImGui::SameLine();
    if (ImGui::Button("<##prev") && current_page_ > 0) current_page_--;
    ImGui::SameLine();
    ImGui::Text("Page %d / %d", current_page_ + 1, total_pages);
    ImGui::SameLine();
    if (ImGui::Button(">##next") && current_page_ < total_pages - 1) current_page_++;
    ImGui::SameLine();
    if (ImGui::Button("<<##first")) current_page_ = 0;
    ImGui::SameLine();
    if (ImGui::Button(">>##last")) current_page_ = total_pages - 1;
}

} // namespace gui
