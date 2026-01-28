#include "hex_preview.h"
#include <imgui.h>
#include <fstream>
#include <spdlog/spdlog.h>
#include <filesystem>

namespace gui {

void HexPreviewRenderer::LoadFile(const std::string& filepath, size_t max_bytes) {
    // Avoid reloading
    static std::string last_path;
    if (filepath == last_path && !data_.empty()) return;
    last_path = filepath;

    data_.clear();
    file_size_ = 0;

    if (!std::filesystem::exists(filepath)) return;

    file_size_ = std::filesystem::file_size(filepath);
    size_t read_size = std::min(static_cast<size_t>(file_size_), max_bytes);

    std::ifstream f(filepath, std::ios::binary);
    if (f) {
        data_.resize(read_size);
        f.read(reinterpret_cast<char*>(data_.data()), read_size);
        data_.resize(f.gcount());
    }
}

void HexPreviewRenderer::SetData(const std::vector<uint8_t>& data) {
    data_ = data;
    file_size_ = data.size();
}

void HexPreviewRenderer::Render() {
    if (data_.empty()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No data.");
        return;
    }

    // Info bar
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
        "Showing %zu / %zu bytes", data_.size(), file_size_);
    ImGui::Separator();

    ImGui::BeginChild("##HexContent", ImVec2(0, 0), false);

    int total_rows = (static_cast<int>(data_.size()) + bytes_per_row_ - 1) / bytes_per_row_;

    // Use monospace font for hex
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));

    ImGuiListClipper clipper;
    clipper.Begin(total_rows);

    while (clipper.Step()) {
        for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
            size_t offset = static_cast<size_t>(row) * bytes_per_row_;

            // Offset column
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.7f, 1.0f, 1.0f));
            ImGui::Text("%08zX  ", offset);
            ImGui::PopStyleColor();
            ImGui::SameLine();

            // Hex bytes
            std::string hex_part;
            std::string ascii_part;

            for (int b = 0; b < bytes_per_row_; b++) {
                size_t idx = offset + b;

                if (b == 8) {
                    hex_part += " ";  // Extra space at midpoint
                }

                if (idx < data_.size()) {
                    char hex[4];
                    snprintf(hex, sizeof(hex), "%02X ", data_[idx]);
                    hex_part += hex;

                    // ASCII representation
                    char c = static_cast<char>(data_[idx]);
                    ascii_part += (c >= 32 && c < 127) ? c : '.';
                } else {
                    hex_part += "   ";
                    ascii_part += ' ';
                }
            }

            // Hex column
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.9f, 0.9f, 1.0f));
            ImGui::Text("%s", hex_part.c_str());
            ImGui::PopStyleColor();
            ImGui::SameLine();

            // Separator
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 0.3f, 0.3f, 1.0f));
            ImGui::Text(" | ");
            ImGui::PopStyleColor();
            ImGui::SameLine();

            // ASCII column
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.9f, 0.7f, 1.0f));
            ImGui::Text("%s", ascii_part.c_str());
            ImGui::PopStyleColor();
        }
    }

    ImGui::PopStyleVar();
    ImGui::EndChild();
}

} // namespace gui
