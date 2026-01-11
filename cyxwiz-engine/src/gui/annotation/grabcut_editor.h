#pragma once

#include "annotation_tools.h"
#include <memory>
#include <functional>

namespace cyxwiz {

/**
 * GrabCut segmentation mode
 */
enum class GrabCutMode {
    Rectangle,     // Initial rectangle selection
    Refinement     // Brush refinement mode
};

/**
 * GrabCut Editor - Interactive foreground extraction using OpenCV GrabCut
 *
 * Workflow:
 * 1. User draws rectangle around object of interest
 * 2. GrabCut runs initial segmentation
 * 3. User refines with foreground/background brush strokes
 * 4. Each refinement triggers GrabCut iteration
 */
class GrabCutEditor {
public:
    GrabCutEditor();
    ~GrabCutEditor();

    /**
     * Set the image to segment
     * @param image_data Float image (0-1 range, HWC format, RGB)
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels (must be 3)
     */
    void SetImage(const std::vector<float>& image_data, int width, int height, int channels);

    /**
     * Get current mode
     */
    GrabCutMode GetMode() const { return mode_; }

    /**
     * Set mode
     */
    void SetMode(GrabCutMode mode);

    /**
     * Get the result mask
     */
    const AnnotationMask& GetMask() const;

    /**
     * Get the result as float mask (0-1)
     */
    std::vector<float> GetFloatMask() const;

    /**
     * Reset to initial state
     */
    void Reset();

    /**
     * Run GrabCut with current rectangle
     * @param iterations Number of GrabCut iterations
     * @return true if successful
     */
    bool RunGrabCut(int iterations = 5);

    /**
     * Refine segmentation with current brush strokes
     * @param iterations Number of GrabCut iterations
     * @return true if successful
     */
    bool Refine(int iterations = 1);

    /**
     * Check if initial rectangle has been drawn
     */
    bool HasRectangle() const;

    /**
     * Check if segmentation has been performed
     */
    bool HasSegmentation() const { return has_segmentation_; }

    // Mouse event handlers
    void OnMouseDown(float x, float y, MouseButton button);
    void OnMouseUp(float x, float y, MouseButton button);
    void OnMouseMove(float x, float y, bool dragging);
    void OnMouseScroll(float delta);

    // Keyboard handlers
    void OnKeyDown(int key);

    /**
     * Render overlay on image
     */
    void RenderOverlay(void* draw_list, float offset_x, float offset_y, float scale);

    /**
     * Render UI controls
     */
    void RenderControls();

    // Settings
    void SetIterations(int iters) { iterations_ = iters; }
    int GetIterations() const { return iterations_; }

    void SetBrushRadius(float radius);
    float GetBrushRadius() const;

    void SetCurrentLabel(AnnotationLabel label);
    AnnotationLabel GetCurrentLabel() const;

    // Callbacks
    using SegmentationCallback = std::function<void(const AnnotationMask&)>;
    void SetSegmentationCallback(SegmentationCallback callback) { on_segmentation_ = callback; }

    using ProgressCallback = std::function<void(float progress, const std::string& status)>;
    void SetProgressCallback(ProgressCallback callback) { on_progress_ = callback; }

private:
    // Image data
    std::vector<float> image_data_;
    int image_width_ = 0;
    int image_height_ = 0;
    int image_channels_ = 0;

    // Tools
    std::unique_ptr<RectangleSelectTool> rect_tool_;
    std::unique_ptr<BrushTool> brush_tool_;

    // State
    GrabCutMode mode_ = GrabCutMode::Rectangle;
    bool has_segmentation_ = false;
    int iterations_ = 5;

    // GrabCut internal mask (cv::GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD)
    std::vector<uint8_t> gc_mask_;

    // Result mask
    AnnotationMask result_mask_;

    // Callbacks
    SegmentationCallback on_segmentation_;
    ProgressCallback on_progress_;

    // Helper to convert mask
    void UpdateResultMask();
};

} // namespace cyxwiz
