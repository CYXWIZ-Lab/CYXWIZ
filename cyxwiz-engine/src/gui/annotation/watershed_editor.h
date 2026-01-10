#pragma once

#include "annotation_tools.h"
#include <imgui.h>
#include <memory>
#include <functional>
#include <map>

namespace cyxwiz {

/**
 * Watershed region info
 */
struct WatershedRegion {
    int id = 0;
    std::string label;
    ImU32 color = 0;
    int pixel_count = 0;
};

/**
 * Watershed Editor - Interactive marker-based segmentation using OpenCV Watershed
 *
 * Workflow:
 * 1. User places point markers for different regions
 * 2. Each marker gets a unique ID and color
 * 3. Watershed algorithm segments image based on markers
 * 4. User can add more markers to refine
 */
class WatershedEditor {
public:
    WatershedEditor();
    ~WatershedEditor();

    /**
     * Set the image to segment
     * @param image_data Float image (0-1 range, HWC format, RGB)
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels (must be 3)
     */
    void SetImage(const std::vector<float>& image_data, int width, int height, int channels);

    /**
     * Get the result mask (each pixel has region ID)
     */
    const std::vector<int32_t>& GetRegionMask() const { return region_mask_; }

    /**
     * Get regions info
     */
    const std::map<int, WatershedRegion>& GetRegions() const { return regions_; }

    /**
     * Get binary mask for foreground regions
     */
    AnnotationMask GetForegroundMask() const;

    /**
     * Get float mask for foreground (0-1)
     */
    std::vector<float> GetFloatMask() const;

    /**
     * Reset to initial state
     */
    void Reset();

    /**
     * Run watershed segmentation
     * @return true if successful
     */
    bool RunWatershed();

    /**
     * Check if segmentation has been performed
     */
    bool HasSegmentation() const { return has_segmentation_; }

    /**
     * Get number of markers
     */
    size_t GetMarkerCount() const;

    /**
     * Add a new region type
     * @param label Region label
     * @param is_foreground Whether this region is foreground
     * @return Region ID
     */
    int AddRegionType(const std::string& label, bool is_foreground = true);

    /**
     * Remove region type
     */
    void RemoveRegionType(int region_id);

    /**
     * Set current region for marking
     */
    void SetCurrentRegion(int region_id);
    int GetCurrentRegion() const { return current_region_; }

    /**
     * Set region as foreground/background
     */
    void SetRegionForeground(int region_id, bool is_foreground);
    bool IsRegionForeground(int region_id) const;

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
    void SetMarkerRadius(float radius) { marker_radius_ = radius; }
    float GetMarkerRadius() const { return marker_radius_; }

    void SetShowRegionColors(bool show) { show_region_colors_ = show; }
    bool GetShowRegionColors() const { return show_region_colors_; }

    // Callbacks
    using SegmentationCallback = std::function<void(const std::vector<int32_t>&)>;
    void SetSegmentationCallback(SegmentationCallback callback) { on_segmentation_ = callback; }

    using ProgressCallback = std::function<void(float progress, const std::string& status)>;
    void SetProgressCallback(ProgressCallback callback) { on_progress_ = callback; }

private:
    // Image data
    std::vector<float> image_data_;
    int image_width_ = 0;
    int image_height_ = 0;
    int image_channels_ = 0;

    // Marker tool
    std::unique_ptr<PointMarkerTool> marker_tool_;

    // Region management
    std::map<int, WatershedRegion> regions_;
    std::map<int, bool> region_foreground_;  // Is region foreground?
    int next_region_id_ = 1;
    int current_region_ = 0;

    // Marker mask (same size as image, each pixel = marker region ID or 0)
    std::vector<int32_t> marker_mask_;

    // Result mask
    std::vector<int32_t> region_mask_;
    bool has_segmentation_ = false;

    // Settings
    float marker_radius_ = 5.0f;
    bool show_region_colors_ = true;

    // Predefined colors for regions
    static const std::vector<ImU32> REGION_COLORS;

    // Callbacks
    SegmentationCallback on_segmentation_;
    ProgressCallback on_progress_;

    // Mouse state
    AnnotationPoint cursor_pos_;
    bool is_painting_ = false;

    // Helper to get color for region
    ImU32 GetRegionColor(int region_id) const;

    // Update marker mask from markers
    void UpdateMarkerMask();
};

} // namespace cyxwiz
