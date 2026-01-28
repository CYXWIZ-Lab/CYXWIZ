#include "hdf5_preview.h"
#include "../icons.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <filesystem>

namespace fs = std::filesystem;

namespace gui {

void HDF5PreviewRenderer::LoadFile(const std::string& filepath) {
    if (filepath == current_path_ && file_loaded_) return;

    Clear();
    current_path_ = filepath;

    if (!fs::exists(filepath)) {
        spdlog::warn("HDF5 file not found: {}", filepath);
        return;
    }

    if (!cyxwiz::HDF5Browser::IsValidHDF5(filepath)) {
        spdlog::warn("Not a valid HDF5 file: {}", filepath);
        return;
    }

    root_node_ = cyxwiz::HDF5Browser::Browse(filepath);
    root_node_.is_expanded = true;  // Expand root by default
    file_loaded_ = true;

    spdlog::debug("Loaded HDF5 structure from: {}", filepath);
}

void HDF5PreviewRenderer::Clear() {
    current_path_.clear();
    root_node_ = cyxwiz::HDF5NodeInfo();
    file_loaded_ = false;
    selected_node_ = nullptr;
    selected_dataset_path_.clear();
    preview_data_.clear();
    preview_shape_.clear();
    preview_loaded_ = false;
    preview_sample_idx_ = 0;
}

void HDF5PreviewRenderer::Render() {
    if (!file_loaded_) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No HDF5 file loaded.");
        return;
    }

    // File info header
    ImGui::TextColored(ImVec4(0.5f, 0.7f, 1.0f, 1.0f), ICON_FA_DATABASE " %s",
                       fs::path(current_path_).filename().string().c_str());

    ImGui::Separator();

    // Two-column layout: tree on left, details on right
    float tree_width = ImGui::GetContentRegionAvail().x * 0.4f;

    // Left panel: Tree view
    ImGui::BeginChild("##HDF5Tree", ImVec2(tree_width, 0), true);
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Structure");
    ImGui::Separator();

    if (root_node_.children.empty() && !root_node_.is_group) {
        // Single dataset at root level
        RenderTreeNode(root_node_);
    } else {
        // Normal hierarchical structure
        for (auto& child : root_node_.children) {
            RenderTreeNode(child);
        }
    }
    ImGui::EndChild();

    ImGui::SameLine();

    // Right panel: Details and preview
    ImGui::BeginChild("##HDF5Details", ImVec2(0, 0), true);

    if (selected_node_ && !selected_node_->is_group) {
        RenderDatasetDetails();
        ImGui::Separator();
        RenderSamplePreview();
    } else if (selected_node_ && selected_node_->is_group) {
        ImGui::TextColored(ImVec4(0.5f, 0.7f, 1.0f, 1.0f), ICON_FA_FOLDER " Group: %s",
                           selected_node_->name.c_str());
        ImGui::Spacing();
        ImGui::Text("Path: %s", selected_node_->path.c_str());
        ImGui::Text("Children: %zu", selected_node_->children.size());
    } else {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                           "Select a dataset to view details");
    }

    ImGui::EndChild();
}

void HDF5PreviewRenderer::RenderTreeNode(cyxwiz::HDF5NodeInfo& node) {
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow |
                                ImGuiTreeNodeFlags_OpenOnDoubleClick |
                                ImGuiTreeNodeFlags_SpanAvailWidth;

    if (!node.is_group) {
        flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
    }

    if (selected_node_ == &node) {
        flags |= ImGuiTreeNodeFlags_Selected;
    }

    // Set open state
    if (node.is_expanded && node.is_group) {
        ImGui::SetNextItemOpen(true, ImGuiCond_Once);
    }

    // Icon based on type
    const char* icon = node.is_group ? ICON_FA_FOLDER : ICON_FA_TABLE_CELLS;
    if (!node.is_group) {
        if (node.LooksLikeImageData()) {
            icon = ICON_FA_IMAGE;
        } else if (node.LooksLikeLabelData()) {
            icon = ICON_FA_TAGS;
        }
    }

    // Color based on type
    ImVec4 color = node.is_group ? ImVec4(0.9f, 0.8f, 0.3f, 1.0f)  // Yellow for groups
                                  : ImVec4(0.4f, 0.8f, 0.4f, 1.0f); // Green for datasets

    ImGui::PushStyleColor(ImGuiCol_Text, color);
    bool opened = ImGui::TreeNodeEx(node.name.c_str(), flags, "%s %s", icon, node.name.c_str());
    ImGui::PopStyleColor();

    // Handle selection
    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
        selected_node_ = &node;
        selected_dataset_path_ = node.path;

        // Load preview if dataset selected
        if (!node.is_group && node.path != selected_dataset_path_) {
            preview_loaded_ = false;
        }
    }

    // Track expanded state
    if (node.is_group) {
        node.is_expanded = opened;
        if (opened) {
            for (auto& child : node.children) {
                RenderTreeNode(child);
            }
            ImGui::TreePop();
        }
    }
}

void HDF5PreviewRenderer::RenderDatasetDetails() {
    if (!selected_node_) return;

    ImGui::TextColored(ImVec4(0.5f, 0.7f, 1.0f, 1.0f), ICON_FA_TABLE_CELLS " Dataset: %s",
                       selected_node_->name.c_str());
    ImGui::Spacing();

    // Details table
    if (ImGui::BeginTable("##DatasetInfo", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 100);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

        // Path
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Path");
        ImGui::TableNextColumn();
        ImGui::Text("%s", selected_node_->path.c_str());

        // Shape
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Shape");
        ImGui::TableNextColumn();
        ImGui::Text("%s", selected_node_->GetShapeString().c_str());

        // Data type
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Type");
        ImGui::TableNextColumn();
        ImGui::Text("%s", selected_node_->dtype.c_str());

        // Size
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Size");
        ImGui::TableNextColumn();
        ImGui::Text("%s", selected_node_->GetSizeString().c_str());

        // Compression
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Compression");
        ImGui::TableNextColumn();
        ImGui::Text("%s", selected_node_->compression.empty() ? "none" : selected_node_->compression.c_str());

        // Chunked
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Chunked");
        ImGui::TableNextColumn();
        ImGui::Text("%s", selected_node_->is_chunked ? "Yes" : "No");

        // Inferred type
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Looks Like");
        ImGui::TableNextColumn();
        if (selected_node_->LooksLikeImageData()) {
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), ICON_FA_IMAGE " Image Data");
        } else if (selected_node_->LooksLikeLabelData()) {
            ImGui::TextColored(ImVec4(0.8f, 0.6f, 1.0f, 1.0f), ICON_FA_TAGS " Labels");
        } else {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Generic Data");
        }

        ImGui::EndTable();
    }
}

void HDF5PreviewRenderer::RenderSamplePreview() {
    if (!selected_node_ || selected_node_->is_group) return;

    ImGui::TextColored(ImVec4(0.5f, 0.7f, 1.0f, 1.0f), ICON_FA_EYE " Sample Preview");
    ImGui::Spacing();

    // Load preview data if needed
    if (!preview_loaded_) {
        preview_data_.clear();
        preview_shape_.clear();

        if (cyxwiz::HDF5Browser::ReadPreviewSamples(
                current_path_, selected_node_->path,
                MAX_PREVIEW_SAMPLES, preview_data_, preview_shape_)) {
            preview_loaded_ = true;
            preview_sample_idx_ = 0;
        }
    }

    if (!preview_loaded_ || preview_data_.empty()) {
        ImGui::TextColored(ImVec4(0.8f, 0.4f, 0.4f, 1.0f), "Could not load preview data");
        return;
    }

    // Calculate sample size
    size_t sample_size = 1;
    for (size_t dim : preview_shape_) {
        sample_size *= dim;
    }

    size_t num_samples = preview_data_.size() / sample_size;
    if (num_samples == 0) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No samples available");
        return;
    }

    // Sample navigation
    ImGui::Text("Sample:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    ImGui::SliderInt("##SampleIdx", &preview_sample_idx_, 0, static_cast<int>(num_samples) - 1);
    ImGui::SameLine();
    ImGui::Text("/ %zu", num_samples);

    ImGui::Spacing();

    // Get current sample data
    size_t offset = static_cast<size_t>(preview_sample_idx_) * sample_size;
    const float* sample_ptr = preview_data_.data() + offset;

    // Display based on shape
    if (selected_node_->LooksLikeImageData() && preview_shape_.size() >= 2) {
        // Image preview (simplified - show dimensions and some pixel values)
        int h = static_cast<int>(preview_shape_[0]);
        int w = static_cast<int>(preview_shape_[1]);
        int c = preview_shape_.size() > 2 ? static_cast<int>(preview_shape_[2]) : 1;

        ImGui::Text("Image: %dx%dx%d", h, w, c);

        // Show corner pixel values
        ImGui::BeginChild("##PixelPreview", ImVec2(0, 100), true);
        ImGui::Text("Sample pixel values:");
        for (int i = 0; i < std::min(5, h); i++) {
            std::string row;
            for (int j = 0; j < std::min(5, w); j++) {
                size_t idx = (i * w + j) * c;
                if (idx < sample_size) {
                    char buf[16];
                    snprintf(buf, sizeof(buf), "%.1f ", sample_ptr[idx]);
                    row += buf;
                }
            }
            if (w > 5) row += "...";
            ImGui::Text("%s", row.c_str());
        }
        if (h > 5) ImGui::Text("...");
        ImGui::EndChild();
    } else {
        // Tabular/generic preview
        ImGui::BeginChild("##DataPreview", ImVec2(0, 150), true);

        size_t display_count = std::min(sample_size, size_t(50));

        if (ImGui::BeginTable("##SampleData", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            for (size_t i = 0; i < display_count; i++) {
                if (i % 5 == 0) ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("[%zu] %.4f", i, sample_ptr[i]);
            }

            if (sample_size > display_count) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "... (%zu more)", sample_size - display_count);
            }

            ImGui::EndTable();
        }

        ImGui::EndChild();
    }
}

} // namespace gui
