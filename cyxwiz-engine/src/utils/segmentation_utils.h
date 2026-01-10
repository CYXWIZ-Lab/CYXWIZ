#pragma once

#include <vector>
#include <string>
#include <functional>
#include <cstdint>

namespace cyxwiz {

/**
 * Connected component information
 */
struct ConnectedComponent {
    int label = 0;                    // Component label ID
    int area = 0;                     // Area in pixels
    float centroid_x = 0.0f;          // Centroid X coordinate
    float centroid_y = 0.0f;          // Centroid Y coordinate
    int bbox_x = 0;                   // Bounding box X
    int bbox_y = 0;                   // Bounding box Y
    int bbox_width = 0;               // Bounding box width
    int bbox_height = 0;              // Bounding box height
};

/**
 * Watershed region result
 */
struct WatershedRegion {
    int label = 0;                    // Region label (0 = background, -1 = boundary)
    int area = 0;                     // Area in pixels
    std::vector<std::pair<int, int>> boundary_pixels;  // Boundary pixel coordinates
};

/**
 * GrabCut result masks
 */
enum class GrabCutMask : uint8_t {
    Background = 0,       // Definite background
    Foreground = 1,       // Definite foreground
    ProbBackground = 2,   // Probable background
    ProbForeground = 3    // Probable foreground
};

/**
 * GrabCut initialization mode
 */
enum class GrabCutMode {
    WithRect,    // Initialize with rectangle
    WithMask     // Initialize with mask
};

/**
 * Segmentation utilities using OpenCV
 *
 * Provides tools for:
 * - Watershed segmentation (marker-based region growing)
 * - GrabCut (interactive foreground extraction)
 * - Connected components analysis
 * - Flood fill operations
 */
class SegmentationUtils {
public:
    /**
     * Apply watershed segmentation
     * @param image_data Float image data (0-1 range) [HWC format]
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels (must be 3 for RGB)
     * @param markers Initial marker image (same size, int32 labels)
     *                0 = unknown, positive values = region labels
     * @param output_labels Output label image (modified in-place)
     * @return Vector of watershed regions
     */
    static std::vector<WatershedRegion> Watershed(
        const std::vector<float>& image_data,
        int width, int height, int channels,
        const std::vector<int32_t>& markers,
        std::vector<int32_t>& output_labels
    );

    /**
     * Generate automatic markers using distance transform
     * Useful for automatic watershed segmentation
     * @param binary_mask Binary mask (0 or 1 values)
     * @param width Image width
     * @param height Image height
     * @param dist_threshold Distance threshold for foreground markers (0-1)
     * @return Marker image with labeled regions
     */
    static std::vector<int32_t> GenerateWatershedMarkers(
        const std::vector<float>& binary_mask,
        int width, int height,
        float dist_threshold = 0.5f
    );

    /**
     * Apply GrabCut segmentation
     * @param image_data Float image data (0-1 range) [HWC format]
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels (must be 3 for RGB)
     * @param rect_x ROI rectangle X (for WithRect mode)
     * @param rect_y ROI rectangle Y
     * @param rect_width ROI rectangle width
     * @param rect_height ROI rectangle height
     * @param iterations Number of GrabCut iterations
     * @param mode Initialization mode
     * @return Binary foreground mask (0-1 values)
     */
    static std::vector<float> GrabCut(
        const std::vector<float>& image_data,
        int width, int height, int channels,
        int rect_x, int rect_y, int rect_width, int rect_height,
        int iterations = 5,
        GrabCutMode mode = GrabCutMode::WithRect
    );

    /**
     * Apply GrabCut with initial mask
     * @param image_data Float image data
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels
     * @param initial_mask Initial GrabCut mask (use GrabCutMask values)
     * @param iterations Number of iterations
     * @return Binary foreground mask (0-1 values)
     */
    static std::vector<float> GrabCutWithMask(
        const std::vector<float>& image_data,
        int width, int height, int channels,
        const std::vector<uint8_t>& initial_mask,
        int iterations = 5
    );

    /**
     * Find connected components in a binary image
     * @param binary_mask Binary mask (0 or 1 values)
     * @param width Image width
     * @param height Image height
     * @param connectivity 4 or 8 connectivity
     * @param output_labels Output label image
     * @return Vector of connected component info
     */
    static std::vector<ConnectedComponent> FindConnectedComponents(
        const std::vector<float>& binary_mask,
        int width, int height,
        int connectivity = 8,
        std::vector<int32_t>* output_labels = nullptr
    );

    /**
     * Filter connected components by area
     * @param binary_mask Binary mask (modified in-place)
     * @param width Image width
     * @param height Image height
     * @param min_area Minimum area to keep
     * @param max_area Maximum area to keep (0 = no max)
     * @return Number of components remaining
     */
    static int FilterComponentsByArea(
        std::vector<float>& binary_mask,
        int width, int height,
        int min_area = 0,
        int max_area = 0
    );

    /**
     * Apply flood fill from a seed point
     * @param image_data Float image data (modified in-place)
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels
     * @param seed_x Seed point X
     * @param seed_y Seed point Y
     * @param new_value New value to fill with
     * @param lo_diff Lower difference threshold
     * @param up_diff Upper difference threshold
     * @return Area of filled region in pixels
     */
    static int FloodFill(
        std::vector<float>& image_data,
        int width, int height, int channels,
        int seed_x, int seed_y,
        float new_value = 1.0f,
        float lo_diff = 0.1f,
        float up_diff = 0.1f
    );

    /**
     * Create a binary mask from flood fill
     * @param image_data Float image data (read-only)
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels
     * @param seed_x Seed point X
     * @param seed_y Seed point Y
     * @param lo_diff Lower difference threshold
     * @param up_diff Upper difference threshold
     * @return Binary mask of filled region
     */
    static std::vector<float> FloodFillMask(
        const std::vector<float>& image_data,
        int width, int height, int channels,
        int seed_x, int seed_y,
        float lo_diff = 0.1f,
        float up_diff = 0.1f
    );

private:
    static bool ValidateInput(const std::vector<float>& image_data,
                              int width, int height, int channels);
};

// Enum string conversions
std::string GrabCutMaskToString(GrabCutMask mask);
std::string GrabCutModeToString(GrabCutMode mode);

} // namespace cyxwiz
