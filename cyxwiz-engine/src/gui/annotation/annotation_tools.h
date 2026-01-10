#pragma once

#include <vector>
#include <string>
#include <functional>
#include <memory>

namespace cyxwiz {

/**
 * Annotation tool types
 */
enum class AnnotationToolType {
    None,
    RectangleSelect,   // Draw rectangle selection
    BrushPaint,        // Paint with brush
    BrushErase,        // Erase with brush
    PolygonDraw,       // Draw polygon vertices
    PointMarker,       // Place point markers
    FloodFill,         // Magic wand / flood fill
    Lasso              // Freehand selection
};

/**
 * Mouse button state
 */
enum class MouseButton {
    None,
    Left,
    Right,
    Middle
};

/**
 * Annotation label types for segmentation
 */
enum class AnnotationLabel {
    Unknown = 0,       // Not labeled
    Background = 1,    // Definite background
    Foreground = 2,    // Definite foreground
    ProbableBackground = 3,  // Probable background (for GrabCut)
    ProbableForeground = 4   // Probable foreground (for GrabCut)
};

/**
 * 2D point for annotation
 */
struct AnnotationPoint {
    float x = 0.0f;
    float y = 0.0f;

    AnnotationPoint() = default;
    AnnotationPoint(float x_, float y_) : x(x_), y(y_) {}
};

/**
 * Rectangle for selection
 */
struct AnnotationRect {
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;

    bool IsValid() const { return width > 0 && height > 0; }
    bool Contains(float px, float py) const {
        return px >= x && px < x + width && py >= y && py < y + height;
    }
};

/**
 * Brush stroke for painting
 */
struct BrushStroke {
    std::vector<AnnotationPoint> points;
    float radius = 5.0f;
    AnnotationLabel label = AnnotationLabel::Foreground;
};

/**
 * Polygon annotation
 */
struct PolygonAnnotation {
    std::vector<AnnotationPoint> vertices;
    bool closed = false;
    AnnotationLabel label = AnnotationLabel::Foreground;
};

/**
 * Point marker annotation
 */
struct PointMarker {
    AnnotationPoint position;
    int marker_id = 0;  // For watershed markers
    AnnotationLabel label = AnnotationLabel::Foreground;
};

/**
 * Annotation mask - stores per-pixel labels
 */
class AnnotationMask {
public:
    AnnotationMask() = default;
    AnnotationMask(int width, int height);

    void Resize(int width, int height);
    void Clear();
    void Fill(AnnotationLabel label);

    int GetWidth() const { return width_; }
    int GetHeight() const { return height_; }

    AnnotationLabel Get(int x, int y) const;
    void Set(int x, int y, AnnotationLabel label);

    // Draw operations
    void DrawCircle(int cx, int cy, int radius, AnnotationLabel label);
    void DrawLine(int x1, int y1, int x2, int y2, int thickness, AnnotationLabel label);
    void DrawRect(int x, int y, int w, int h, AnnotationLabel label, bool filled = true);
    void FloodFill(int x, int y, AnnotationLabel label, float tolerance = 0.0f);

    // Get raw data
    const std::vector<uint8_t>& GetData() const { return data_; }
    std::vector<uint8_t>& GetData() { return data_; }

    // Convert to float mask (0-1 for foreground probability)
    std::vector<float> ToFloatMask() const;

    // Create from float mask
    static AnnotationMask FromFloatMask(const std::vector<float>& mask, int width, int height, float threshold = 0.5f);

private:
    int width_ = 0;
    int height_ = 0;
    std::vector<uint8_t> data_;
};

/**
 * Base class for interactive annotation tools
 */
class AnnotationTool {
public:
    virtual ~AnnotationTool() = default;

    /**
     * Get tool type
     */
    virtual AnnotationToolType GetType() const = 0;

    /**
     * Get tool name
     */
    virtual std::string GetName() const = 0;

    /**
     * Set the image to annotate
     * @param image_data Float image (0-1 range, HWC format)
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels
     */
    virtual void SetImage(const std::vector<float>& image_data, int width, int height, int channels);

    /**
     * Get current annotation mask
     */
    virtual const AnnotationMask& GetMask() const { return mask_; }
    virtual AnnotationMask& GetMask() { return mask_; }

    /**
     * Clear all annotations
     */
    virtual void Clear();

    /**
     * Undo last operation
     */
    virtual void Undo();

    /**
     * Redo last undone operation
     */
    virtual void Redo();

    /**
     * Check if can undo
     */
    virtual bool CanUndo() const { return !undo_stack_.empty(); }

    /**
     * Check if can redo
     */
    virtual bool CanRedo() const { return !redo_stack_.empty(); }

    // Mouse event handlers (coordinates in image space 0-width, 0-height)
    virtual void OnMouseDown(float x, float y, MouseButton button) = 0;
    virtual void OnMouseUp(float x, float y, MouseButton button) = 0;
    virtual void OnMouseMove(float x, float y, bool dragging) = 0;
    virtual void OnMouseScroll(float delta) {}

    // Keyboard event handlers
    virtual void OnKeyDown(int key) {}
    virtual void OnKeyUp(int key) {}

    /**
     * Render tool overlay (called during ImGui rendering)
     * @param draw_list ImGui draw list
     * @param offset Screen offset for image
     * @param scale Display scale
     */
    virtual void RenderOverlay(void* draw_list, float offset_x, float offset_y, float scale) = 0;

    /**
     * Render tool-specific UI controls
     */
    virtual void RenderControls() = 0;

    // Brush settings
    void SetBrushRadius(float radius) { brush_radius_ = radius; }
    float GetBrushRadius() const { return brush_radius_; }

    void SetCurrentLabel(AnnotationLabel label) { current_label_ = label; }
    AnnotationLabel GetCurrentLabel() const { return current_label_; }

    // Callbacks
    using MaskChangedCallback = std::function<void(const AnnotationMask&)>;
    void SetMaskChangedCallback(MaskChangedCallback callback) { on_mask_changed_ = callback; }

protected:
    // Image data
    std::vector<float> image_data_;
    int image_width_ = 0;
    int image_height_ = 0;
    int image_channels_ = 0;

    // Annotation mask
    AnnotationMask mask_;

    // Tool settings
    float brush_radius_ = 10.0f;
    AnnotationLabel current_label_ = AnnotationLabel::Foreground;

    // Undo/redo stacks
    std::vector<AnnotationMask> undo_stack_;
    std::vector<AnnotationMask> redo_stack_;
    static constexpr size_t MAX_UNDO_LEVELS = 20;

    // Callback
    MaskChangedCallback on_mask_changed_;

    // Helper to save state for undo
    void SaveUndoState();

    // Notify mask changed
    void NotifyMaskChanged();
};

/**
 * Rectangle selection tool
 */
class RectangleSelectTool : public AnnotationTool {
public:
    AnnotationToolType GetType() const override { return AnnotationToolType::RectangleSelect; }
    std::string GetName() const override { return "Rectangle Select"; }

    void OnMouseDown(float x, float y, MouseButton button) override;
    void OnMouseUp(float x, float y, MouseButton button) override;
    void OnMouseMove(float x, float y, bool dragging) override;

    void RenderOverlay(void* draw_list, float offset_x, float offset_y, float scale) override;
    void RenderControls() override;

    const AnnotationRect& GetSelection() const { return selection_; }
    bool HasSelection() const { return selection_.IsValid(); }
    void ClearSelection() { selection_ = AnnotationRect(); }

private:
    AnnotationRect selection_;
    bool is_drawing_ = false;
    AnnotationPoint start_point_;
};

/**
 * Brush paint tool
 */
class BrushTool : public AnnotationTool {
public:
    AnnotationToolType GetType() const override { return AnnotationToolType::BrushPaint; }
    std::string GetName() const override { return "Brush"; }

    void OnMouseDown(float x, float y, MouseButton button) override;
    void OnMouseUp(float x, float y, MouseButton button) override;
    void OnMouseMove(float x, float y, bool dragging) override;
    void OnMouseScroll(float delta) override;

    void RenderOverlay(void* draw_list, float offset_x, float offset_y, float scale) override;
    void RenderControls() override;

private:
    bool is_painting_ = false;
    AnnotationPoint last_point_;
    AnnotationPoint cursor_pos_;
};

/**
 * Point marker tool (for watershed)
 */
class PointMarkerTool : public AnnotationTool {
public:
    AnnotationToolType GetType() const override { return AnnotationToolType::PointMarker; }
    std::string GetName() const override { return "Point Marker"; }

    void OnMouseDown(float x, float y, MouseButton button) override;
    void OnMouseUp(float x, float y, MouseButton button) override;
    void OnMouseMove(float x, float y, bool dragging) override;

    void RenderOverlay(void* draw_list, float offset_x, float offset_y, float scale) override;
    void RenderControls() override;

    void Clear() override;

    const std::vector<PointMarker>& GetMarkers() const { return markers_; }
    int GetNextMarkerId() const { return next_marker_id_; }

private:
    std::vector<PointMarker> markers_;
    int next_marker_id_ = 1;
    AnnotationPoint cursor_pos_;
    int selected_marker_ = -1;
};

/**
 * Polygon drawing tool
 */
class PolygonTool : public AnnotationTool {
public:
    AnnotationToolType GetType() const override { return AnnotationToolType::PolygonDraw; }
    std::string GetName() const override { return "Polygon"; }

    void OnMouseDown(float x, float y, MouseButton button) override;
    void OnMouseUp(float x, float y, MouseButton button) override;
    void OnMouseMove(float x, float y, bool dragging) override;
    void OnKeyDown(int key) override;

    void RenderOverlay(void* draw_list, float offset_x, float offset_y, float scale) override;
    void RenderControls() override;

    void Clear() override;
    void ClosePolygon();

    const std::vector<PolygonAnnotation>& GetPolygons() const { return polygons_; }

private:
    std::vector<PolygonAnnotation> polygons_;
    PolygonAnnotation current_polygon_;
    AnnotationPoint cursor_pos_;
    bool is_drawing_ = false;
};

} // namespace cyxwiz
