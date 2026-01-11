#include "grabcut_editor.h"
#include <imgui.h>
#include <spdlog/spdlog.h>

#ifdef CYXWIZ_HAS_OPENCV
#include <opencv2/imgproc.hpp>
#endif

namespace cyxwiz {

GrabCutEditor::GrabCutEditor() {
    rect_tool_ = std::make_unique<RectangleSelectTool>();
    brush_tool_ = std::make_unique<BrushTool>();
}

GrabCutEditor::~GrabCutEditor() = default;

void GrabCutEditor::SetImage(const std::vector<float>& image_data, int width, int height, int channels) {
    image_data_ = image_data;
    image_width_ = width;
    image_height_ = height;
    image_channels_ = channels;

    rect_tool_->SetImage(image_data, width, height, channels);
    brush_tool_->SetImage(image_data, width, height, channels);

    gc_mask_.resize(width * height);
    result_mask_.Resize(width, height);

    Reset();
}

void GrabCutEditor::SetMode(GrabCutMode mode) {
    mode_ = mode;
}

const AnnotationMask& GrabCutEditor::GetMask() const {
    return result_mask_;
}

std::vector<float> GrabCutEditor::GetFloatMask() const {
    return result_mask_.ToFloatMask();
}

void GrabCutEditor::Reset() {
    mode_ = GrabCutMode::Rectangle;
    has_segmentation_ = false;
    rect_tool_->ClearSelection();
    brush_tool_->Clear();
    result_mask_.Clear();

    // Initialize gc_mask to probable background
    std::fill(gc_mask_.begin(), gc_mask_.end(), 2);  // cv::GC_PR_BGD
}

bool GrabCutEditor::HasRectangle() const {
    return rect_tool_->HasSelection();
}

bool GrabCutEditor::RunGrabCut(int iterations) {
#ifdef CYXWIZ_HAS_OPENCV
    if (!HasRectangle()) {
        spdlog::warn("GrabCutEditor: No rectangle selected");
        return false;
    }

    if (image_channels_ != 3) {
        spdlog::error("GrabCutEditor: Image must have 3 channels");
        return false;
    }

    if (on_progress_) {
        on_progress_(0.0f, "Running GrabCut...");
    }

    try {
        // Convert float image to BGR Mat
        cv::Mat img(image_height_, image_width_, CV_8UC3);
        for (int y = 0; y < image_height_; ++y) {
            for (int x = 0; x < image_width_; ++x) {
                int idx = (y * image_width_ + x) * 3;
                // RGB to BGR, scale to 0-255
                img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    static_cast<uint8_t>(image_data_[idx + 2] * 255),
                    static_cast<uint8_t>(image_data_[idx + 1] * 255),
                    static_cast<uint8_t>(image_data_[idx + 0] * 255)
                );
            }
        }

        // Get rectangle
        const auto& sel = rect_tool_->GetSelection();
        cv::Rect rect(
            static_cast<int>(sel.x),
            static_cast<int>(sel.y),
            static_cast<int>(sel.width),
            static_cast<int>(sel.height)
        );

        // Clamp rectangle to image bounds
        rect.x = std::max(0, rect.x);
        rect.y = std::max(0, rect.y);
        rect.width = std::min(rect.width, image_width_ - rect.x);
        rect.height = std::min(rect.height, image_height_ - rect.y);

        if (rect.width <= 0 || rect.height <= 0) {
            spdlog::error("GrabCutEditor: Invalid rectangle");
            return false;
        }

        // Create mask
        cv::Mat mask(image_height_, image_width_, CV_8UC1, cv::Scalar(cv::GC_BGD));

        // Initialize rectangle area as probable foreground
        mask(rect).setTo(cv::Scalar(cv::GC_PR_FGD));

        // GrabCut models
        cv::Mat bgdModel, fgdModel;

        // Run GrabCut
        cv::grabCut(img, mask, rect, bgdModel, fgdModel, iterations, cv::GC_INIT_WITH_RECT);

        // Copy mask to gc_mask_
        for (int y = 0; y < image_height_; ++y) {
            for (int x = 0; x < image_width_; ++x) {
                gc_mask_[y * image_width_ + x] = mask.at<uint8_t>(y, x);
            }
        }

        has_segmentation_ = true;
        mode_ = GrabCutMode::Refinement;

        UpdateResultMask();

        if (on_progress_) {
            on_progress_(1.0f, "GrabCut complete");
        }

        if (on_segmentation_) {
            on_segmentation_(result_mask_);
        }

        return true;

    } catch (const cv::Exception& e) {
        spdlog::error("GrabCutEditor: OpenCV error - {}", e.what());
        return false;
    }
#else
    spdlog::error("GrabCutEditor: OpenCV not available");
    return false;
#endif
}

bool GrabCutEditor::Refine(int iterations) {
#ifdef CYXWIZ_HAS_OPENCV
    if (!has_segmentation_) {
        spdlog::warn("GrabCutEditor: No initial segmentation");
        return false;
    }

    if (on_progress_) {
        on_progress_(0.0f, "Refining segmentation...");
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

        // Create mask from gc_mask_ with brush strokes applied
        cv::Mat mask(image_height_, image_width_, CV_8UC1);
        for (int y = 0; y < image_height_; ++y) {
            for (int x = 0; x < image_width_; ++x) {
                mask.at<uint8_t>(y, x) = gc_mask_[y * image_width_ + x];
            }
        }

        // Apply brush mask modifications
        const auto& brush_mask = brush_tool_->GetMask();
        for (int y = 0; y < image_height_; ++y) {
            for (int x = 0; x < image_width_; ++x) {
                AnnotationLabel label = brush_mask.Get(x, y);
                if (label == AnnotationLabel::Foreground) {
                    mask.at<uint8_t>(y, x) = cv::GC_FGD;
                } else if (label == AnnotationLabel::Background) {
                    mask.at<uint8_t>(y, x) = cv::GC_BGD;
                }
            }
        }

        // GrabCut models (will be re-initialized)
        cv::Mat bgdModel, fgdModel;

        // Get original rectangle for reference
        const auto& sel = rect_tool_->GetSelection();
        cv::Rect rect(
            static_cast<int>(sel.x),
            static_cast<int>(sel.y),
            static_cast<int>(sel.width),
            static_cast<int>(sel.height)
        );

        // Run GrabCut with mask
        cv::grabCut(img, mask, rect, bgdModel, fgdModel, iterations, cv::GC_INIT_WITH_MASK);

        // Update gc_mask_
        for (int y = 0; y < image_height_; ++y) {
            for (int x = 0; x < image_width_; ++x) {
                gc_mask_[y * image_width_ + x] = mask.at<uint8_t>(y, x);
            }
        }

        UpdateResultMask();

        // Clear brush strokes after applying
        brush_tool_->Clear();

        if (on_progress_) {
            on_progress_(1.0f, "Refinement complete");
        }

        if (on_segmentation_) {
            on_segmentation_(result_mask_);
        }

        return true;

    } catch (const cv::Exception& e) {
        spdlog::error("GrabCutEditor: OpenCV error - {}", e.what());
        return false;
    }
#else
    spdlog::error("GrabCutEditor: OpenCV not available");
    return false;
#endif
}

void GrabCutEditor::UpdateResultMask() {
    for (int y = 0; y < image_height_; ++y) {
        for (int x = 0; x < image_width_; ++x) {
            uint8_t gc_val = gc_mask_[y * image_width_ + x];

#ifdef CYXWIZ_HAS_OPENCV
            // Convert GrabCut mask values to annotation labels
            if (gc_val == cv::GC_FGD || gc_val == cv::GC_PR_FGD) {
                result_mask_.Set(x, y, AnnotationLabel::Foreground);
            } else {
                result_mask_.Set(x, y, AnnotationLabel::Background);
            }
#else
            // Fallback without OpenCV
            if (gc_val == 1 || gc_val == 3) {  // GC_FGD=1, GC_PR_FGD=3
                result_mask_.Set(x, y, AnnotationLabel::Foreground);
            } else {
                result_mask_.Set(x, y, AnnotationLabel::Background);
            }
#endif
        }
    }
}

void GrabCutEditor::OnMouseDown(float x, float y, MouseButton button) {
    if (mode_ == GrabCutMode::Rectangle) {
        rect_tool_->OnMouseDown(x, y, button);
    } else {
        brush_tool_->OnMouseDown(x, y, button);
    }
}

void GrabCutEditor::OnMouseUp(float x, float y, MouseButton button) {
    if (mode_ == GrabCutMode::Rectangle) {
        rect_tool_->OnMouseUp(x, y, button);
    } else {
        brush_tool_->OnMouseUp(x, y, button);
    }
}

void GrabCutEditor::OnMouseMove(float x, float y, bool dragging) {
    if (mode_ == GrabCutMode::Rectangle) {
        rect_tool_->OnMouseMove(x, y, dragging);
    } else {
        brush_tool_->OnMouseMove(x, y, dragging);
    }
}

void GrabCutEditor::OnMouseScroll(float delta) {
    if (mode_ == GrabCutMode::Refinement) {
        brush_tool_->OnMouseScroll(delta);
    }
}

void GrabCutEditor::OnKeyDown(int key) {
    // Enter to run/refine
    if (key == ImGuiKey_Enter || key == ImGuiKey_KeypadEnter) {
        if (mode_ == GrabCutMode::Rectangle && HasRectangle()) {
            RunGrabCut(iterations_);
        } else if (mode_ == GrabCutMode::Refinement) {
            Refine(1);
        }
    }
    // Escape to reset
    else if (key == ImGuiKey_Escape) {
        Reset();
    }
    // R to go back to rectangle mode
    else if (key == ImGuiKey_R) {
        if (mode_ == GrabCutMode::Refinement) {
            mode_ = GrabCutMode::Rectangle;
        }
    }
}

void GrabCutEditor::SetBrushRadius(float radius) {
    brush_tool_->SetBrushRadius(radius);
}

float GrabCutEditor::GetBrushRadius() const {
    return brush_tool_->GetBrushRadius();
}

void GrabCutEditor::SetCurrentLabel(AnnotationLabel label) {
    brush_tool_->SetCurrentLabel(label);
}

AnnotationLabel GrabCutEditor::GetCurrentLabel() const {
    return brush_tool_->GetCurrentLabel();
}

void GrabCutEditor::RenderOverlay(void* draw_list_ptr, float offset_x, float offset_y, float scale) {
    auto* draw_list = static_cast<ImDrawList*>(draw_list_ptr);

    // Draw segmentation result
    if (has_segmentation_) {
        // Draw semi-transparent overlay for foreground
        for (int y = 0; y < image_height_; y += 4) {  // Sample every 4th pixel for performance
            for (int x = 0; x < image_width_; x += 4) {
                if (result_mask_.Get(x, y) == AnnotationLabel::Foreground) {
                    ImVec2 p1(offset_x + x * scale, offset_y + y * scale);
                    ImVec2 p2(offset_x + (x + 4) * scale, offset_y + (y + 4) * scale);
                    draw_list->AddRectFilled(p1, p2, IM_COL32(0, 255, 0, 40));
                }
            }
        }

        // Draw contour (simplified - just outline the mask boundary)
        // This is a visual approximation; proper contour would need more computation
    }

    // Draw tool overlay
    if (mode_ == GrabCutMode::Rectangle) {
        rect_tool_->RenderOverlay(draw_list_ptr, offset_x, offset_y, scale);
    } else {
        brush_tool_->RenderOverlay(draw_list_ptr, offset_x, offset_y, scale);
    }
}

void GrabCutEditor::RenderControls() {
    ImGui::Text("GrabCut Segmentation");
    ImGui::Separator();

    // Mode indicator
    const char* mode_str = (mode_ == GrabCutMode::Rectangle) ? "Rectangle Selection" : "Refinement";
    ImGui::Text("Mode: %s", mode_str);

    ImGui::Separator();

    if (mode_ == GrabCutMode::Rectangle) {
        ImGui::TextWrapped("Draw a rectangle around the object you want to extract.");

        if (HasRectangle()) {
            const auto& sel = rect_tool_->GetSelection();
            ImGui::Text("Selection: %.0fx%.0f at (%.0f, %.0f)",
                       sel.width, sel.height, sel.x, sel.y);

            ImGui::SliderInt("Iterations", &iterations_, 1, 10);

            if (ImGui::Button("Run GrabCut", ImVec2(-1, 0))) {
                RunGrabCut(iterations_);
            }
        } else {
            ImGui::TextDisabled("Click and drag to select region");
        }
    } else {
        ImGui::TextWrapped("Paint foreground (left-click) or background (right-click) to refine.");

        // Brush controls
        float radius = GetBrushRadius();
        if (ImGui::SliderFloat("Brush Size", &radius, 1.0f, 50.0f, "%.0f px")) {
            SetBrushRadius(radius);
        }

        AnnotationLabel label = GetCurrentLabel();
        ImGui::Text("Paint Mode:");
        if (ImGui::RadioButton("Foreground", label == AnnotationLabel::Foreground)) {
            SetCurrentLabel(AnnotationLabel::Foreground);
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Background", label == AnnotationLabel::Background)) {
            SetCurrentLabel(AnnotationLabel::Background);
        }

        ImGui::Separator();

        if (ImGui::Button("Refine", ImVec2(-1, 0))) {
            Refine(1);
        }

        if (ImGui::Button("Back to Rectangle", ImVec2(-1, 0))) {
            mode_ = GrabCutMode::Rectangle;
        }
    }

    ImGui::Separator();

    if (ImGui::Button("Reset", ImVec2(-1, 0))) {
        Reset();
    }

    ImGui::Separator();
    ImGui::TextDisabled("Shortcuts:");
    ImGui::TextDisabled("  Enter: Run/Refine");
    ImGui::TextDisabled("  Escape: Reset");
    ImGui::TextDisabled("  R: Rectangle mode");
}

} // namespace cyxwiz
