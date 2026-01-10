#include "watershed_editor.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <algorithm>

#ifdef CYXWIZ_HAS_OPENCV
#include <opencv2/imgproc.hpp>
#endif

namespace cyxwiz {

// Predefined colors for different regions
const std::vector<ImU32> WatershedEditor::REGION_COLORS = {
    IM_COL32(255, 0, 0, 200),      // Red
    IM_COL32(0, 255, 0, 200),      // Green
    IM_COL32(0, 0, 255, 200),      // Blue
    IM_COL32(255, 255, 0, 200),    // Yellow
    IM_COL32(255, 0, 255, 200),    // Magenta
    IM_COL32(0, 255, 255, 200),    // Cyan
    IM_COL32(255, 128, 0, 200),    // Orange
    IM_COL32(128, 0, 255, 200),    // Purple
    IM_COL32(0, 255, 128, 200),    // Spring Green
    IM_COL32(255, 128, 128, 200),  // Light Red
    IM_COL32(128, 255, 128, 200),  // Light Green
    IM_COL32(128, 128, 255, 200),  // Light Blue
};

WatershedEditor::WatershedEditor() {
    marker_tool_ = std::make_unique<PointMarkerTool>();

    // Add default regions
    AddRegionType("Foreground", true);
    AddRegionType("Background", false);
    current_region_ = 1;  // Start with foreground
}

WatershedEditor::~WatershedEditor() = default;

void WatershedEditor::SetImage(const std::vector<float>& image_data, int width, int height, int channels) {
    image_data_ = image_data;
    image_width_ = width;
    image_height_ = height;
    image_channels_ = channels;

    marker_tool_->SetImage(image_data, width, height, channels);

    marker_mask_.resize(width * height, 0);
    region_mask_.resize(width * height, 0);

    Reset();
}

void WatershedEditor::Reset() {
    has_segmentation_ = false;
    marker_tool_->Clear();
    std::fill(marker_mask_.begin(), marker_mask_.end(), 0);
    std::fill(region_mask_.begin(), region_mask_.end(), 0);

    // Reset regions but keep default ones
    regions_.clear();
    region_foreground_.clear();
    next_region_id_ = 1;
    AddRegionType("Foreground", true);
    AddRegionType("Background", false);
    current_region_ = 1;
}

AnnotationMask WatershedEditor::GetForegroundMask() const {
    AnnotationMask mask(image_width_, image_height_);

    for (int y = 0; y < image_height_; ++y) {
        for (int x = 0; x < image_width_; ++x) {
            int region_id = region_mask_[y * image_width_ + x];
            if (region_id > 0 && IsRegionForeground(region_id)) {
                mask.Set(x, y, AnnotationLabel::Foreground);
            } else {
                mask.Set(x, y, AnnotationLabel::Background);
            }
        }
    }

    return mask;
}

std::vector<float> WatershedEditor::GetFloatMask() const {
    std::vector<float> mask(image_width_ * image_height_, 0.0f);

    for (int y = 0; y < image_height_; ++y) {
        for (int x = 0; x < image_width_; ++x) {
            int idx = y * image_width_ + x;
            int region_id = region_mask_[idx];
            if (region_id > 0 && IsRegionForeground(region_id)) {
                mask[idx] = 1.0f;
            }
        }
    }

    return mask;
}

size_t WatershedEditor::GetMarkerCount() const {
    return marker_tool_->GetMarkers().size();
}

int WatershedEditor::AddRegionType(const std::string& label, bool is_foreground) {
    int id = next_region_id_++;

    WatershedRegion region;
    region.id = id;
    region.label = label;
    region.color = GetRegionColor(id);
    region.pixel_count = 0;

    regions_[id] = region;
    region_foreground_[id] = is_foreground;

    return id;
}

void WatershedEditor::RemoveRegionType(int region_id) {
    regions_.erase(region_id);
    region_foreground_.erase(region_id);

    if (current_region_ == region_id) {
        current_region_ = regions_.empty() ? 0 : regions_.begin()->first;
    }
}

void WatershedEditor::SetCurrentRegion(int region_id) {
    if (regions_.find(region_id) != regions_.end()) {
        current_region_ = region_id;
    }
}

void WatershedEditor::SetRegionForeground(int region_id, bool is_foreground) {
    region_foreground_[region_id] = is_foreground;
}

bool WatershedEditor::IsRegionForeground(int region_id) const {
    auto it = region_foreground_.find(region_id);
    return it != region_foreground_.end() ? it->second : false;
}

ImU32 WatershedEditor::GetRegionColor(int region_id) const {
    if (region_id <= 0) return IM_COL32(128, 128, 128, 100);
    return REGION_COLORS[(region_id - 1) % REGION_COLORS.size()];
}

void WatershedEditor::UpdateMarkerMask() {
    std::fill(marker_mask_.begin(), marker_mask_.end(), 0);

    // Paint markers onto mask
    for (const auto& marker : marker_tool_->GetMarkers()) {
        int cx = static_cast<int>(marker.position.x);
        int cy = static_cast<int>(marker.position.y);
        int radius = static_cast<int>(marker_radius_);
        int region_id = current_region_;  // Use marker's associated region

        // Simple circle painting
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                if (dx * dx + dy * dy <= radius * radius) {
                    int x = cx + dx;
                    int y = cy + dy;
                    if (x >= 0 && x < image_width_ && y >= 0 && y < image_height_) {
                        marker_mask_[y * image_width_ + x] = region_id;
                    }
                }
            }
        }
    }
}

bool WatershedEditor::RunWatershed() {
#ifdef CYXWIZ_HAS_OPENCV
    if (GetMarkerCount() < 2) {
        spdlog::warn("WatershedEditor: Need at least 2 markers");
        return false;
    }

    if (image_channels_ != 3) {
        spdlog::error("WatershedEditor: Image must have 3 channels");
        return false;
    }

    if (on_progress_) {
        on_progress_(0.0f, "Running Watershed...");
    }

    try {
        // Convert float image to BGR Mat
        cv::Mat img(image_height_, image_width_, CV_8UC3);
        for (int y = 0; y < image_height_; ++y) {
            for (int x = 0; x < image_width_; ++x) {
                int idx = (y * image_width_ + x) * 3;
                img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    static_cast<uint8_t>(image_data_[idx + 2] * 255),
                    static_cast<uint8_t>(image_data_[idx + 1] * 255),
                    static_cast<uint8_t>(image_data_[idx + 0] * 255)
                );
            }
        }

        // Create marker image
        cv::Mat markers(image_height_, image_width_, CV_32S, cv::Scalar(0));

        // Paint markers based on point markers
        for (const auto& marker : marker_tool_->GetMarkers()) {
            int cx = static_cast<int>(marker.position.x);
            int cy = static_cast<int>(marker.position.y);
            int radius = static_cast<int>(marker_radius_);

            // Determine region ID based on marker label
            int region_id = (marker.label == AnnotationLabel::Foreground) ? 1 : 2;

            cv::circle(markers, cv::Point(cx, cy), radius, cv::Scalar(region_id), -1);
        }

        if (on_progress_) {
            on_progress_(0.3f, "Computing watershed...");
        }

        // Run watershed
        cv::watershed(img, markers);

        if (on_progress_) {
            on_progress_(0.9f, "Processing results...");
        }

        // Copy result to region_mask_
        for (int y = 0; y < image_height_; ++y) {
            for (int x = 0; x < image_width_; ++x) {
                int32_t val = markers.at<int32_t>(y, x);
                // Watershed marks boundaries as -1
                if (val == -1) {
                    region_mask_[y * image_width_ + x] = 0;  // Boundary
                } else {
                    region_mask_[y * image_width_ + x] = val;
                }
            }
        }

        // Update region pixel counts
        for (auto& pair : regions_) {
            pair.second.pixel_count = 0;
        }
        for (int i = 0; i < image_width_ * image_height_; ++i) {
            int region_id = region_mask_[i];
            if (regions_.find(region_id) != regions_.end()) {
                regions_[region_id].pixel_count++;
            }
        }

        has_segmentation_ = true;

        if (on_progress_) {
            on_progress_(1.0f, "Watershed complete");
        }

        if (on_segmentation_) {
            on_segmentation_(region_mask_);
        }

        return true;

    } catch (const cv::Exception& e) {
        spdlog::error("WatershedEditor: OpenCV error - {}", e.what());
        return false;
    }
#else
    spdlog::error("WatershedEditor: OpenCV not available");
    return false;
#endif
}

void WatershedEditor::OnMouseDown(float x, float y, MouseButton button) {
    if (button == MouseButton::Left) {
        // Add marker for current region
        is_painting_ = true;

        // Set marker label based on whether current region is foreground
        AnnotationLabel label = IsRegionForeground(current_region_) ?
            AnnotationLabel::Foreground : AnnotationLabel::Background;
        marker_tool_->SetCurrentLabel(label);
        marker_tool_->OnMouseDown(x, y, button);
    } else if (button == MouseButton::Right) {
        // Remove nearest marker
        marker_tool_->OnMouseDown(x, y, button);
    }
}

void WatershedEditor::OnMouseUp(float x, float y, MouseButton button) {
    is_painting_ = false;
    marker_tool_->OnMouseUp(x, y, button);
}

void WatershedEditor::OnMouseMove(float x, float y, bool dragging) {
    cursor_pos_ = {x, y};
    marker_tool_->OnMouseMove(x, y, dragging);

    // Continuous painting while dragging
    if (is_painting_ && dragging) {
        AnnotationLabel label = IsRegionForeground(current_region_) ?
            AnnotationLabel::Foreground : AnnotationLabel::Background;
        marker_tool_->SetCurrentLabel(label);
        marker_tool_->OnMouseDown(x, y, MouseButton::Left);
    }
}

void WatershedEditor::OnMouseScroll(float delta) {
    marker_radius_ = std::max(1.0f, std::min(30.0f, marker_radius_ + delta));
}

void WatershedEditor::OnKeyDown(int key) {
    // Enter to run watershed
    if (key == ImGuiKey_Enter || key == ImGuiKey_KeypadEnter) {
        RunWatershed();
    }
    // Escape to reset
    else if (key == ImGuiKey_Escape) {
        Reset();
    }
    // Number keys to select region
    else if (key >= ImGuiKey_1 && key <= ImGuiKey_9) {
        int region_idx = key - ImGuiKey_1 + 1;
        if (regions_.find(region_idx) != regions_.end()) {
            SetCurrentRegion(region_idx);
        }
    }
}

void WatershedEditor::RenderOverlay(void* draw_list_ptr, float offset_x, float offset_y, float scale) {
    auto* draw_list = static_cast<ImDrawList*>(draw_list_ptr);

    // Draw segmentation result
    if (has_segmentation_ && show_region_colors_) {
        // Draw colored overlay for each region (sampled for performance)
        for (int y = 0; y < image_height_; y += 4) {
            for (int x = 0; x < image_width_; x += 4) {
                int region_id = region_mask_[y * image_width_ + x];
                if (region_id > 0) {
                    ImU32 color = GetRegionColor(region_id);
                    // Make semi-transparent
                    color = (color & 0x00FFFFFF) | 0x40000000;

                    ImVec2 p1(offset_x + x * scale, offset_y + y * scale);
                    ImVec2 p2(offset_x + (x + 4) * scale, offset_y + (y + 4) * scale);
                    draw_list->AddRectFilled(p1, p2, color);
                }
            }
        }

        // Draw watershed boundaries (where region_mask_ == 0 after segmentation)
        for (int y = 1; y < image_height_ - 1; y += 2) {
            for (int x = 1; x < image_width_ - 1; x += 2) {
                if (region_mask_[y * image_width_ + x] == 0) {
                    // Check if this is a boundary (neighbors have different regions)
                    ImVec2 p(offset_x + x * scale, offset_y + y * scale);
                    draw_list->AddCircleFilled(p, 1.0f, IM_COL32(255, 255, 255, 200));
                }
            }
        }
    }

    // Draw markers
    for (const auto& marker : marker_tool_->GetMarkers()) {
        ImVec2 center(offset_x + marker.position.x * scale, offset_y + marker.position.y * scale);

        // Color based on label
        ImU32 color = (marker.label == AnnotationLabel::Foreground) ?
            IM_COL32(0, 255, 0, 255) : IM_COL32(255, 0, 0, 255);

        float radius = marker_radius_ * scale;
        draw_list->AddCircleFilled(center, radius, (color & 0x00FFFFFF) | 0x80000000);
        draw_list->AddCircle(center, radius, color, 16, 2.0f);
    }

    // Draw cursor
    ImVec2 cursor(offset_x + cursor_pos_.x * scale, offset_y + cursor_pos_.y * scale);
    ImU32 cursor_color = IsRegionForeground(current_region_) ?
        IM_COL32(0, 255, 0, 150) : IM_COL32(255, 0, 0, 150);
    draw_list->AddCircle(cursor, marker_radius_ * scale, cursor_color, 16, 2.0f);
}

void WatershedEditor::RenderControls() {
    ImGui::Text("Watershed Segmentation");
    ImGui::Separator();

    ImGui::Text("Markers: %zu", GetMarkerCount());

    ImGui::SliderFloat("Marker Size", &marker_radius_, 1.0f, 30.0f, "%.0f px");

    ImGui::Separator();
    ImGui::Text("Regions:");

    // Region selection
    for (const auto& pair : regions_) {
        const auto& region = pair.second;
        bool is_current = (current_region_ == region.id);
        bool is_fg = IsRegionForeground(region.id);

        ImGui::PushID(region.id);

        // Color indicator
        ImVec4 col = ImGui::ColorConvertU32ToFloat4(region.color);
        ImGui::ColorButton("##color", col, ImGuiColorEditFlags_NoTooltip, ImVec2(20, 20));
        ImGui::SameLine();

        // Radio button for selection
        if (ImGui::RadioButton(region.label.c_str(), is_current)) {
            SetCurrentRegion(region.id);
        }

        ImGui::SameLine();
        ImGui::TextDisabled("(%s)", is_fg ? "FG" : "BG");

        if (has_segmentation_) {
            ImGui::SameLine();
            ImGui::TextDisabled("[%d px]", region.pixel_count);
        }

        ImGui::PopID();
    }

    ImGui::Separator();

    // Add new region
    static char new_label[64] = "";
    static bool new_is_fg = true;

    ImGui::InputText("##newlabel", new_label, sizeof(new_label));
    ImGui::SameLine();
    ImGui::Checkbox("FG##new", &new_is_fg);
    ImGui::SameLine();
    if (ImGui::Button("Add Region")) {
        if (strlen(new_label) > 0) {
            AddRegionType(new_label, new_is_fg);
            new_label[0] = '\0';
        }
    }

    ImGui::Separator();

    ImGui::Checkbox("Show Region Colors", &show_region_colors_);

    ImGui::Separator();

    if (GetMarkerCount() >= 2) {
        if (ImGui::Button("Run Watershed", ImVec2(-1, 0))) {
            RunWatershed();
        }
    } else {
        ImGui::BeginDisabled();
        ImGui::Button("Run Watershed (need 2+ markers)", ImVec2(-1, 0));
        ImGui::EndDisabled();
    }

    if (ImGui::Button("Reset", ImVec2(-1, 0))) {
        Reset();
    }

    ImGui::Separator();
    ImGui::TextDisabled("Shortcuts:");
    ImGui::TextDisabled("  Left-click: Add marker");
    ImGui::TextDisabled("  Right-click: Remove marker");
    ImGui::TextDisabled("  Scroll: Adjust marker size");
    ImGui::TextDisabled("  Enter: Run watershed");
    ImGui::TextDisabled("  1-9: Select region");
}

} // namespace cyxwiz
