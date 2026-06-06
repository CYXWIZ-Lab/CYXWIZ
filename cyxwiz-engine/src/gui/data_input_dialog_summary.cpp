// DataInputDialog dataset summary rendering.

#include "node_config_dialog.h"

#include <string>

namespace gui {

void DataInputDialog::RenderDatasetSummaryPanel() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "DATASET SUMMARY");
    ImGui::Separator();
    ImGui::Spacing();

    const bool loaded =
        data_load_state_ == DataLoadState::InMemory &&
        !loaded_dataset_name_.empty();

    if (loaded) {
        ImGui::Text("Dataset:");
        ImGui::SameLine(120);
        ImGui::TextUnformatted(loaded_dataset_name_.c_str());

        ImGui::Text("Rows / samples:");
        ImGui::SameLine(120);
        ImGui::Text("%lld", static_cast<long long>(loaded_rows_));

        if (loaded_cols_ > 0) {
            ImGui::Text("Columns:");
            ImGui::SameLine(120);
            ImGui::Text("%lld", static_cast<long long>(loaded_cols_));
        }

        ImGui::Text("Backend:");
        ImGui::SameLine(120);
        ImGui::TextUnformatted(BackendSummary());

        ImGui::Text("Footprint:");
        ImGui::SameLine(120);
        if (loaded_backend_ == 2) {
            ImGui::Text("%s on disk", FormatBytes(loaded_memory_bytes_).c_str());
        } else if (loaded_memory_is_estimate_) {
            ImGui::Text("~%s if fully cached", FormatBytes(loaded_memory_bytes_).c_str());
        } else {
            ImGui::Text("%s RAM", FormatBytes(loaded_memory_bytes_).c_str());
        }

        if (last_load_elapsed_ms_ >= 0.0f) {
            ImGui::Text("Last Apply:");
            ImGui::SameLine(120);
            if (last_load_elapsed_ms_ < 1000.0f) {
                ImGui::Text("%.0f ms", last_load_elapsed_ms_);
            } else {
                ImGui::Text("%.2f s", last_load_elapsed_ms_ / 1000.0f);
            }
        }

        if (node_) {
            auto errors = node_->parameters.find("audit_errors");
            auto warnings = node_->parameters.find("audit_warnings");
            if (errors != node_->parameters.end() || warnings != node_->parameters.end()) {
                ImGui::Text("Audit:");
                ImGui::SameLine(120);
                ImGui::Text("%s errors, %s warnings",
                            errors != node_->parameters.end() ? errors->second.c_str() : "0",
                            warnings != node_->parameters.end() ? warnings->second.c_str() : "0");
            }
        }
        return;
    }

    if (is_loading_async_) {
        ImGui::Text("State:");
        ImGui::SameLine(120);
        ImGui::TextUnformatted("Loading");
        if (async_load_state_ && !async_load_state_->dataset_name.empty()) {
            ImGui::Text("Dataset:");
            ImGui::SameLine(120);
            ImGui::TextUnformatted(async_load_state_->dataset_name.c_str());
        }
        const std::string source = CurrentSourcePath();
        if (!source.empty()) {
            ImGui::Text("Source:");
            ImGui::SameLine(120);
            ImGui::TextUnformatted(source.c_str());
        }
        ImGui::TextWrapped("Apply path: %s", CurrentApplySummary().c_str());
        return;
    }

    const std::string source = CurrentSourcePath();
    if (source.empty()) {
        ImGui::TextDisabled("No source selected.");
    } else {
        ImGui::Text("Source:");
        ImGui::SameLine(120);
        ImGui::TextUnformatted(source.c_str());
    }

    ImGui::Text("Type:");
    ImGui::SameLine(120);
    ImGui::TextUnformatted(CurrentSourceLabel().c_str());

    if (preview_loaded_ && preview_error_.empty()) {
        ImGui::Text("Preview:");
        ImGui::SameLine(120);
        ImGui::Text("%zu rows, %zu columns",
                    preview_data_.size(), preview_columns_.size());
    }

    if (file_size_ > 0) {
        ImGui::Text("Disk size:");
        ImGui::SameLine(120);
        ImGui::TextUnformatted(FormatBytes(file_size_).c_str());
    }

    if (IsApplySupported()) {
        ImGui::TextWrapped("Apply path: %s", CurrentApplySummary().c_str());
    } else {
        ImGui::TextColored(ImVec4(0.9f, 0.4f, 0.3f, 1.0f),
                           "Apply path: %s", CurrentApplySummary().c_str());
    }
}

} // namespace gui
