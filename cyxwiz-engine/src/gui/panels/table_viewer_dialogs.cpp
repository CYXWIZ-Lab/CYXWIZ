#include "table_viewer.h"
#include "../icons.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <cstring>
#include <sstream>
#include <string>

namespace cyxwiz {
// ============================================================================
// Export Dialog
// ============================================================================

void TableViewerPanel::RenderExportDialog() {
    if (!show_export_dialog_) return;

    ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(ICON_FA_FILE_EXPORT " Export Table", &show_export_dialog_)) {
        TableTab* tab = GetActiveTab();
        if (!tab || !tab->table) {
            ImGui::TextDisabled("No table to export");
            ImGui::End();
            return;
        }

        static int export_format = 0;  // 0=CSV, 1=TSV, 2=JSON
        static bool include_headers = true;
        static bool filtered_only = false;
        static char export_path[512] = "";

        ImGui::Text("Format:");
        ImGui::RadioButton("CSV (Comma-separated)", &export_format, 0);
        ImGui::RadioButton("TSV (Tab-separated)", &export_format, 1);
        ImGui::RadioButton("JSON", &export_format, 2);

        ImGui::Separator();
        ImGui::Text("Options:");
        ImGui::Checkbox("Include headers", &include_headers);
        ImGui::Checkbox("Export filtered rows only", &filtered_only);

        ImGui::Separator();
        ImGui::Text("Output Path:");
        ImGui::InputText("##ExportPath", export_path, sizeof(export_path));
        ImGui::SameLine();
        if (ImGui::Button("Browse...")) {
            // TODO: Open file dialog
        }

        ImGui::Separator();

        if (ImGui::Button(ICON_FA_FILE_EXPORT " Export", ImVec2(120, 0))) {
            // Build export content
            std::ostringstream ss;
            char delimiter = (export_format == 0) ? ',' : '\t';

            if (export_format < 2) {  // CSV/TSV
                // Headers
                if (include_headers) {
                    const auto& headers = tab->table->GetHeaders();
                    for (size_t i = 0; i < headers.size(); i++) {
                        if (i > 0) ss << delimiter;
                        ss << headers[i];
                    }
                    ss << "\n";
                }

                // Data
                const auto& indices = (filtered_only && tab->filter_mode_hide && !tab->filtered_indices.empty()) ?
                    tab->filtered_indices : tab->sorted_indices;

                for (size_t idx : indices) {
                    for (size_t c = 0; c < tab->table->GetColumnCount(); c++) {
                        if (c > 0) ss << delimiter;
                        ss << tab->table->GetCellAsString(idx, c);
                    }
                    ss << "\n";
                }
            } else {  // JSON
                ss << "[\n";
                const auto& headers = tab->table->GetHeaders();
                const auto& indices = (filtered_only && tab->filter_mode_hide && !tab->filtered_indices.empty()) ?
                    tab->filtered_indices : tab->sorted_indices;

                for (size_t i = 0; i < indices.size(); i++) {
                    size_t idx = indices[i];
                    ss << "  {";
                    for (size_t c = 0; c < tab->table->GetColumnCount(); c++) {
                        if (c > 0) ss << ", ";
                        ss << "\"" << headers[c] << "\": \"" << tab->table->GetCellAsString(idx, c) << "\"";
                    }
                    ss << "}";
                    if (i < indices.size() - 1) ss << ",";
                    ss << "\n";
                }
                ss << "]\n";
            }

            // Copy to clipboard for now (file saving needs platform dialog)
            ImGui::SetClipboardText(ss.str().c_str());
            spdlog::info("Exported table to clipboard ({} bytes)", ss.str().length());
            show_export_dialog_ = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            show_export_dialog_ = false;
        }
    }
    ImGui::End();
}

// ============================================================================
// Find/Replace Dialog
// ============================================================================

void TableViewerPanel::RenderFindDialog() {
    if (!show_find_dialog_) return;

    ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(ICON_FA_MAGNIFYING_GLASS " Find in Table", &show_find_dialog_)) {
        TableTab* tab = GetActiveTab();
        if (!tab || !tab->table) {
            ImGui::TextDisabled("No table loaded");
            ImGui::End();
            return;
        }

        static bool case_sensitive = false;
        static bool whole_cell = false;
        static int found_count = -1;

        ImGui::Text("Find:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        bool search_changed = ImGui::InputText("##FindInput", find_buffer_, sizeof(find_buffer_));

        ImGui::Checkbox("Case sensitive", &case_sensitive);
        ImGui::SameLine();
        ImGui::Checkbox("Whole cell match", &whole_cell);

        ImGui::Separator();

        if (ImGui::Button(ICON_FA_MAGNIFYING_GLASS " Find All", ImVec2(120, 0)) || search_changed) {
            // Use filter functionality
            tab->filter_text = find_buffer_;
            std::strncpy(tab->filter_buffer, find_buffer_, sizeof(tab->filter_buffer) - 1);

            // Count matches
            found_count = 0;
            std::string search_term = find_buffer_;
            for (size_t r = 0; r < tab->table->GetRowCount(); r++) {
                for (size_t c = 0; c < tab->table->GetColumnCount(); c++) {
                    std::string cell = tab->table->GetCellAsString(r, c);
                    if (cell.find(search_term) != std::string::npos) {
                        found_count++;
                    }
                }
            }
        }
        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_XMARK " Clear", ImVec2(120, 0))) {
            std::memset(find_buffer_, 0, sizeof(find_buffer_));
            tab->filter_text.clear();
            std::memset(tab->filter_buffer, 0, sizeof(tab->filter_buffer));
            found_count = -1;
        }

        if (found_count >= 0) {
            ImGui::Text("Found: %d matches", found_count);
        }
    }
    ImGui::End();
}

}  // namespace cyxwiz

