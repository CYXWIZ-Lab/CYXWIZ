#include "annotation_tools.h"
#include <imgui.h>
#include <algorithm>
#include <cmath>
#include <queue>

namespace cyxwiz {

// ============================================================================
// AnnotationMask Implementation
// ============================================================================

AnnotationMask::AnnotationMask(int width, int height)
    : width_(width), height_(height), data_(width * height, static_cast<uint8_t>(AnnotationLabel::Unknown))
{
}

void AnnotationMask::Resize(int width, int height) {
    width_ = width;
    height_ = height;
    data_.resize(width * height);
    Clear();
}

void AnnotationMask::Clear() {
    std::fill(data_.begin(), data_.end(), static_cast<uint8_t>(AnnotationLabel::Unknown));
}

void AnnotationMask::Fill(AnnotationLabel label) {
    std::fill(data_.begin(), data_.end(), static_cast<uint8_t>(label));
}

AnnotationLabel AnnotationMask::Get(int x, int y) const {
    if (x < 0 || x >= width_ || y < 0 || y >= height_) {
        return AnnotationLabel::Unknown;
    }
    return static_cast<AnnotationLabel>(data_[y * width_ + x]);
}

void AnnotationMask::Set(int x, int y, AnnotationLabel label) {
    if (x >= 0 && x < width_ && y >= 0 && y < height_) {
        data_[y * width_ + x] = static_cast<uint8_t>(label);
    }
}

void AnnotationMask::DrawCircle(int cx, int cy, int radius, AnnotationLabel label) {
    int r2 = radius * radius;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy <= r2) {
                Set(cx + dx, cy + dy, label);
            }
        }
    }
}

void AnnotationMask::DrawLine(int x1, int y1, int x2, int y2, int thickness, AnnotationLabel label) {
    // Bresenham's line algorithm with thickness
    int dx = std::abs(x2 - x1);
    int dy = std::abs(y2 - y1);
    int sx = x1 < x2 ? 1 : -1;
    int sy = y1 < y2 ? 1 : -1;
    int err = dx - dy;

    int x = x1, y = y1;
    int half_thick = thickness / 2;

    while (true) {
        // Draw circle at each point for thickness
        DrawCircle(x, y, half_thick, label);

        if (x == x2 && y == y2) break;

        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x += sx;
        }
        if (e2 < dx) {
            err += dx;
            y += sy;
        }
    }
}

void AnnotationMask::DrawRect(int x, int y, int w, int h, AnnotationLabel label, bool filled) {
    if (filled) {
        for (int dy = 0; dy < h; ++dy) {
            for (int dx = 0; dx < w; ++dx) {
                Set(x + dx, y + dy, label);
            }
        }
    } else {
        // Top and bottom edges
        for (int dx = 0; dx < w; ++dx) {
            Set(x + dx, y, label);
            Set(x + dx, y + h - 1, label);
        }
        // Left and right edges
        for (int dy = 0; dy < h; ++dy) {
            Set(x, y + dy, label);
            Set(x + w - 1, y + dy, label);
        }
    }
}

void AnnotationMask::FloodFill(int x, int y, AnnotationLabel label) {
    if (x < 0 || x >= width_ || y < 0 || y >= height_) return;

    AnnotationLabel target = Get(x, y);
    if (target == label) return;

    std::queue<std::pair<int, int>> queue;
    queue.push({x, y});

    while (!queue.empty()) {
        auto [cx, cy] = queue.front();
        queue.pop();

        if (cx < 0 || cx >= width_ || cy < 0 || cy >= height_) continue;
        if (Get(cx, cy) != target) continue;

        Set(cx, cy, label);

        queue.push({cx + 1, cy});
        queue.push({cx - 1, cy});
        queue.push({cx, cy + 1});
        queue.push({cx, cy - 1});
    }
}

std::vector<float> AnnotationMask::ToFloatMask() const {
    std::vector<float> result(width_ * height_);
    for (size_t i = 0; i < data_.size(); ++i) {
        AnnotationLabel label = static_cast<AnnotationLabel>(data_[i]);
        switch (label) {
            case AnnotationLabel::Foreground:
            case AnnotationLabel::ProbableForeground:
                result[i] = 1.0f;
                break;
            case AnnotationLabel::Background:
            case AnnotationLabel::ProbableBackground:
                result[i] = 0.0f;
                break;
            default:
                result[i] = 0.5f;  // Unknown
                break;
        }
    }
    return result;
}

AnnotationMask AnnotationMask::FromFloatMask(const std::vector<float>& mask, int width, int height, float threshold) {
    AnnotationMask result(width, height);
    for (int i = 0; i < width * height && i < static_cast<int>(mask.size()); ++i) {
        if (mask[i] >= threshold) {
            result.data_[i] = static_cast<uint8_t>(AnnotationLabel::Foreground);
        } else {
            result.data_[i] = static_cast<uint8_t>(AnnotationLabel::Background);
        }
    }
    return result;
}

// ============================================================================
// AnnotationTool Base Implementation
// ============================================================================

void AnnotationTool::SetImage(const std::vector<float>& image_data, int width, int height, int channels) {
    image_data_ = image_data;
    image_width_ = width;
    image_height_ = height;
    image_channels_ = channels;
    mask_.Resize(width, height);
    undo_stack_.clear();
    redo_stack_.clear();
}

void AnnotationTool::Clear() {
    SaveUndoState();
    mask_.Clear();
    NotifyMaskChanged();
}

void AnnotationTool::Undo() {
    if (undo_stack_.empty()) return;

    redo_stack_.push_back(mask_);
    mask_ = undo_stack_.back();
    undo_stack_.pop_back();
    NotifyMaskChanged();
}

void AnnotationTool::Redo() {
    if (redo_stack_.empty()) return;

    undo_stack_.push_back(mask_);
    mask_ = redo_stack_.back();
    redo_stack_.pop_back();
    NotifyMaskChanged();
}

void AnnotationTool::SaveUndoState() {
    undo_stack_.push_back(mask_);
    if (undo_stack_.size() > MAX_UNDO_LEVELS) {
        undo_stack_.erase(undo_stack_.begin());
    }
    redo_stack_.clear();
}

void AnnotationTool::NotifyMaskChanged() {
    if (on_mask_changed_) {
        on_mask_changed_(mask_);
    }
}

// ============================================================================
// RectangleSelectTool Implementation
// ============================================================================

void RectangleSelectTool::OnMouseDown(float x, float y, MouseButton button) {
    if (button == MouseButton::Left) {
        is_drawing_ = true;
        start_point_ = {x, y};
        selection_ = AnnotationRect();
    }
}

void RectangleSelectTool::OnMouseUp(float x, float y, MouseButton button) {
    if (button == MouseButton::Left && is_drawing_) {
        is_drawing_ = false;

        // Finalize selection
        float x1 = std::min(start_point_.x, x);
        float y1 = std::min(start_point_.y, y);
        float x2 = std::max(start_point_.x, x);
        float y2 = std::max(start_point_.y, y);

        selection_.x = x1;
        selection_.y = y1;
        selection_.width = x2 - x1;
        selection_.height = y2 - y1;
    }
}

void RectangleSelectTool::OnMouseMove(float x, float y, bool) {
    if (is_drawing_) {
        float x1 = std::min(start_point_.x, x);
        float y1 = std::min(start_point_.y, y);
        float x2 = std::max(start_point_.x, x);
        float y2 = std::max(start_point_.y, y);

        selection_.x = x1;
        selection_.y = y1;
        selection_.width = x2 - x1;
        selection_.height = y2 - y1;
    }
}

void RectangleSelectTool::RenderOverlay(void* draw_list_ptr, float offset_x, float offset_y, float scale) {
    auto* draw_list = static_cast<ImDrawList*>(draw_list_ptr);

    if (selection_.IsValid()) {
        ImVec2 p1(offset_x + selection_.x * scale, offset_y + selection_.y * scale);
        ImVec2 p2(offset_x + (selection_.x + selection_.width) * scale,
                  offset_y + (selection_.y + selection_.height) * scale);

        // Draw selection rectangle
        draw_list->AddRect(p1, p2, IM_COL32(0, 255, 0, 255), 0.0f, 0, 2.0f);
        draw_list->AddRectFilled(p1, p2, IM_COL32(0, 255, 0, 30));
    }
}

void RectangleSelectTool::RenderControls() {
    ImGui::Text("Rectangle Select Tool");
    ImGui::Separator();

    if (selection_.IsValid()) {
        ImGui::Text("Selection: (%.0f, %.0f) %.0fx%.0f",
                   selection_.x, selection_.y, selection_.width, selection_.height);

        if (ImGui::Button("Clear Selection")) {
            ClearSelection();
        }
    } else {
        ImGui::TextDisabled("Click and drag to select a region");
    }
}

// ============================================================================
// BrushTool Implementation
// ============================================================================

void BrushTool::OnMouseDown(float x, float y, MouseButton button) {
    if (button == MouseButton::Left || button == MouseButton::Right) {
        SaveUndoState();
        is_painting_ = true;
        last_point_ = {x, y};

        // Set label based on button
        AnnotationLabel paint_label = (button == MouseButton::Left) ?
            current_label_ : AnnotationLabel::Background;

        mask_.DrawCircle(static_cast<int>(x), static_cast<int>(y),
                        static_cast<int>(brush_radius_), paint_label);
        NotifyMaskChanged();
    }
}

void BrushTool::OnMouseUp(float, float, MouseButton) {
    is_painting_ = false;
}

void BrushTool::OnMouseMove(float x, float y, bool) {
    cursor_pos_ = {x, y};

    if (is_painting_) {
        // Determine label based on which button is held
        // (simplified - assumes left button for foreground)
        AnnotationLabel paint_label = current_label_;

        mask_.DrawLine(static_cast<int>(last_point_.x), static_cast<int>(last_point_.y),
                      static_cast<int>(x), static_cast<int>(y),
                      static_cast<int>(brush_radius_ * 2), paint_label);
        last_point_ = {x, y};
        NotifyMaskChanged();
    }
}

void BrushTool::OnMouseScroll(float delta) {
    brush_radius_ = std::max(1.0f, std::min(100.0f, brush_radius_ + delta * 2.0f));
}

void BrushTool::RenderOverlay(void* draw_list_ptr, float offset_x, float offset_y, float scale) {
    auto* draw_list = static_cast<ImDrawList*>(draw_list_ptr);

    // Draw brush cursor
    ImVec2 center(offset_x + cursor_pos_.x * scale, offset_y + cursor_pos_.y * scale);
    float radius = brush_radius_ * scale;

    ImU32 color = (current_label_ == AnnotationLabel::Foreground) ?
        IM_COL32(0, 255, 0, 200) : IM_COL32(255, 0, 0, 200);

    draw_list->AddCircle(center, radius, color, 32, 2.0f);
}

void BrushTool::RenderControls() {
    ImGui::Text("Brush Tool");
    ImGui::Separator();

    ImGui::SliderFloat("Brush Size", &brush_radius_, 1.0f, 100.0f, "%.0f px");

    ImGui::Text("Label:");
    if (ImGui::RadioButton("Foreground", current_label_ == AnnotationLabel::Foreground)) {
        current_label_ = AnnotationLabel::Foreground;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Background", current_label_ == AnnotationLabel::Background)) {
        current_label_ = AnnotationLabel::Background;
    }

    ImGui::Separator();
    ImGui::TextDisabled("Left-click: Paint foreground");
    ImGui::TextDisabled("Right-click: Paint background");
    ImGui::TextDisabled("Scroll: Adjust brush size");
}

// ============================================================================
// PointMarkerTool Implementation
// ============================================================================

void PointMarkerTool::OnMouseDown(float x, float y, MouseButton button) {
    if (button == MouseButton::Left) {
        SaveUndoState();

        PointMarker marker;
        marker.position = {x, y};
        marker.marker_id = next_marker_id_++;
        marker.label = current_label_;
        markers_.push_back(marker);

        // Also draw on mask
        mask_.DrawCircle(static_cast<int>(x), static_cast<int>(y), 3, current_label_);
        NotifyMaskChanged();
    } else if (button == MouseButton::Right) {
        // Remove nearest marker
        float min_dist = 20.0f;  // Minimum distance to select
        int remove_idx = -1;

        for (size_t i = 0; i < markers_.size(); ++i) {
            float dx = markers_[i].position.x - x;
            float dy = markers_[i].position.y - y;
            float dist = std::sqrt(dx * dx + dy * dy);
            if (dist < min_dist) {
                min_dist = dist;
                remove_idx = static_cast<int>(i);
            }
        }

        if (remove_idx >= 0) {
            SaveUndoState();
            markers_.erase(markers_.begin() + remove_idx);
        }
    }
}

void PointMarkerTool::OnMouseUp(float, float, MouseButton) {
    // Nothing special on mouse up
}

void PointMarkerTool::OnMouseMove(float x, float y, bool) {
    cursor_pos_ = {x, y};

    // Highlight nearest marker
    float min_dist = 20.0f;
    selected_marker_ = -1;

    for (size_t i = 0; i < markers_.size(); ++i) {
        float dx = markers_[i].position.x - x;
        float dy = markers_[i].position.y - y;
        float dist = std::sqrt(dx * dx + dy * dy);
        if (dist < min_dist) {
            min_dist = dist;
            selected_marker_ = static_cast<int>(i);
        }
    }
}

void PointMarkerTool::RenderOverlay(void* draw_list_ptr, float offset_x, float offset_y, float scale) {
    auto* draw_list = static_cast<ImDrawList*>(draw_list_ptr);

    // Draw all markers
    for (size_t i = 0; i < markers_.size(); ++i) {
        const auto& marker = markers_[i];
        ImVec2 center(offset_x + marker.position.x * scale, offset_y + marker.position.y * scale);

        ImU32 color = (marker.label == AnnotationLabel::Foreground) ?
            IM_COL32(0, 255, 0, 255) : IM_COL32(255, 0, 0, 255);

        bool is_selected = (static_cast<int>(i) == selected_marker_);
        float radius = is_selected ? 8.0f : 6.0f;

        draw_list->AddCircleFilled(center, radius, color);
        draw_list->AddCircle(center, radius, IM_COL32(255, 255, 255, 255), 12, 2.0f);

        // Draw marker ID
        char id_str[16];
        snprintf(id_str, sizeof(id_str), "%d", marker.marker_id);
        draw_list->AddText(ImVec2(center.x + 10, center.y - 6), IM_COL32(255, 255, 255, 255), id_str);
    }

    // Draw cursor
    ImVec2 cursor(offset_x + cursor_pos_.x * scale, offset_y + cursor_pos_.y * scale);
    ImU32 cursor_color = (current_label_ == AnnotationLabel::Foreground) ?
        IM_COL32(0, 255, 0, 128) : IM_COL32(255, 0, 0, 128);
    draw_list->AddCircle(cursor, 6.0f, cursor_color, 12, 2.0f);
}

void PointMarkerTool::RenderControls() {
    ImGui::Text("Point Marker Tool");
    ImGui::Separator();

    ImGui::Text("Markers: %zu", markers_.size());

    ImGui::Text("Label:");
    if (ImGui::RadioButton("Foreground##marker", current_label_ == AnnotationLabel::Foreground)) {
        current_label_ = AnnotationLabel::Foreground;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Background##marker", current_label_ == AnnotationLabel::Background)) {
        current_label_ = AnnotationLabel::Background;
    }

    if (ImGui::Button("Clear All Markers")) {
        Clear();
    }

    ImGui::Separator();
    ImGui::TextDisabled("Left-click: Add marker");
    ImGui::TextDisabled("Right-click: Remove nearest marker");
}

void PointMarkerTool::Clear() {
    AnnotationTool::Clear();
    markers_.clear();
    next_marker_id_ = 1;
    selected_marker_ = -1;
}

// ============================================================================
// PolygonTool Implementation
// ============================================================================

void PolygonTool::OnMouseDown(float x, float y, MouseButton button) {
    if (button == MouseButton::Left) {
        if (!is_drawing_) {
            // Start new polygon
            is_drawing_ = true;
            current_polygon_ = PolygonAnnotation();
            current_polygon_.label = current_label_;
        }

        // Add vertex
        current_polygon_.vertices.push_back({x, y});

        // Check if close to first vertex (to close polygon)
        if (current_polygon_.vertices.size() > 2) {
            float dx = current_polygon_.vertices[0].x - x;
            float dy = current_polygon_.vertices[0].y - y;
            float dist = std::sqrt(dx * dx + dy * dy);
            if (dist < 10.0f) {
                ClosePolygon();
            }
        }
    } else if (button == MouseButton::Right) {
        // Cancel current polygon
        is_drawing_ = false;
        current_polygon_ = PolygonAnnotation();
    }
}

void PolygonTool::OnMouseUp(float, float, MouseButton) {
    // Nothing special
}

void PolygonTool::OnMouseMove(float x, float y, bool) {
    cursor_pos_ = {x, y};
}

void PolygonTool::OnKeyDown(int key) {
    // Enter key closes polygon
    if (key == ImGuiKey_Enter || key == ImGuiKey_KeypadEnter) {
        if (is_drawing_ && current_polygon_.vertices.size() >= 3) {
            ClosePolygon();
        }
    }
    // Escape cancels
    else if (key == ImGuiKey_Escape) {
        is_drawing_ = false;
        current_polygon_ = PolygonAnnotation();
    }
}

void PolygonTool::RenderOverlay(void* draw_list_ptr, float offset_x, float offset_y, float scale) {
    auto* draw_list = static_cast<ImDrawList*>(draw_list_ptr);

    // Draw completed polygons
    for (const auto& poly : polygons_) {
        if (poly.vertices.size() < 3) continue;

        ImU32 color = (poly.label == AnnotationLabel::Foreground) ?
            IM_COL32(0, 255, 0, 100) : IM_COL32(255, 0, 0, 100);
        ImU32 outline = (poly.label == AnnotationLabel::Foreground) ?
            IM_COL32(0, 255, 0, 255) : IM_COL32(255, 0, 0, 255);

        std::vector<ImVec2> points;
        for (const auto& v : poly.vertices) {
            points.push_back(ImVec2(offset_x + v.x * scale, offset_y + v.y * scale));
        }

        draw_list->AddConvexPolyFilled(points.data(), static_cast<int>(points.size()), color);
        draw_list->AddPolyline(points.data(), static_cast<int>(points.size()), outline, ImDrawFlags_Closed, 2.0f);
    }

    // Draw current polygon being drawn
    if (is_drawing_ && !current_polygon_.vertices.empty()) {
        ImU32 color = (current_polygon_.label == AnnotationLabel::Foreground) ?
            IM_COL32(0, 255, 0, 200) : IM_COL32(255, 0, 0, 200);

        // Draw vertices
        for (const auto& v : current_polygon_.vertices) {
            ImVec2 p(offset_x + v.x * scale, offset_y + v.y * scale);
            draw_list->AddCircleFilled(p, 5.0f, color);
        }

        // Draw edges
        for (size_t i = 0; i < current_polygon_.vertices.size(); ++i) {
            const auto& v1 = current_polygon_.vertices[i];
            const auto& v2 = (i + 1 < current_polygon_.vertices.size()) ?
                current_polygon_.vertices[i + 1] : current_polygon_.vertices[0];

            ImVec2 p1(offset_x + v1.x * scale, offset_y + v1.y * scale);
            ImVec2 p2(offset_x + v2.x * scale, offset_y + v2.y * scale);

            if (i + 1 < current_polygon_.vertices.size()) {
                draw_list->AddLine(p1, p2, color, 2.0f);
            }
        }

        // Draw line to cursor
        if (!current_polygon_.vertices.empty()) {
            const auto& last = current_polygon_.vertices.back();
            ImVec2 p1(offset_x + last.x * scale, offset_y + last.y * scale);
            ImVec2 p2(offset_x + cursor_pos_.x * scale, offset_y + cursor_pos_.y * scale);
            draw_list->AddLine(p1, p2, IM_COL32(255, 255, 0, 200), 1.0f);
        }
    }
}

void PolygonTool::RenderControls() {
    ImGui::Text("Polygon Tool");
    ImGui::Separator();

    ImGui::Text("Polygons: %zu", polygons_.size());

    if (is_drawing_) {
        ImGui::Text("Drawing: %zu vertices", current_polygon_.vertices.size());

        if (current_polygon_.vertices.size() >= 3) {
            if (ImGui::Button("Close Polygon")) {
                ClosePolygon();
            }
        }

        if (ImGui::Button("Cancel")) {
            is_drawing_ = false;
            current_polygon_ = PolygonAnnotation();
        }
    }

    ImGui::Text("Label:");
    if (ImGui::RadioButton("Foreground##poly", current_label_ == AnnotationLabel::Foreground)) {
        current_label_ = AnnotationLabel::Foreground;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Background##poly", current_label_ == AnnotationLabel::Background)) {
        current_label_ = AnnotationLabel::Background;
    }

    if (ImGui::Button("Clear All Polygons")) {
        Clear();
    }

    ImGui::Separator();
    ImGui::TextDisabled("Left-click: Add vertex");
    ImGui::TextDisabled("Right-click: Cancel polygon");
    ImGui::TextDisabled("Enter: Close polygon");
}

void PolygonTool::Clear() {
    AnnotationTool::Clear();
    polygons_.clear();
    current_polygon_ = PolygonAnnotation();
    is_drawing_ = false;
}

void PolygonTool::ClosePolygon() {
    if (current_polygon_.vertices.size() < 3) return;

    SaveUndoState();

    current_polygon_.closed = true;
    polygons_.push_back(current_polygon_);

    // Rasterize polygon to mask (simple scanline fill)
    // Find bounding box
    float min_x = current_polygon_.vertices[0].x;
    float max_x = min_x;
    float min_y = current_polygon_.vertices[0].y;
    float max_y = min_y;

    for (const auto& v : current_polygon_.vertices) {
        min_x = std::min(min_x, v.x);
        max_x = std::max(max_x, v.x);
        min_y = std::min(min_y, v.y);
        max_y = std::max(max_y, v.y);
    }

    // Simple point-in-polygon test for each pixel in bounding box
    for (int y = static_cast<int>(min_y); y <= static_cast<int>(max_y); ++y) {
        for (int x = static_cast<int>(min_x); x <= static_cast<int>(max_x); ++x) {
            // Ray casting algorithm
            bool inside = false;
            size_t n = current_polygon_.vertices.size();

            for (size_t i = 0, j = n - 1; i < n; j = i++) {
                float xi = current_polygon_.vertices[i].x;
                float yi = current_polygon_.vertices[i].y;
                float xj = current_polygon_.vertices[j].x;
                float yj = current_polygon_.vertices[j].y;

                if (((yi > y) != (yj > y)) &&
                    (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
                    inside = !inside;
                }
            }

            if (inside) {
                mask_.Set(x, y, current_polygon_.label);
            }
        }
    }

    current_polygon_ = PolygonAnnotation();
    is_drawing_ = false;
    NotifyMaskChanged();
}

} // namespace cyxwiz
