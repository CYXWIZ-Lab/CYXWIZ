#pragma once

#include <vector>
#include <string>
#include <functional>
#include <cstdint>
#include "shape_utils.h"

namespace cyxwiz {

/**
 * Contour retrieval mode
 */
enum class ContourMode {
    External,    // Only outermost contours
    List,        // All contours without hierarchy
    CComp,       // Two-level hierarchy (external and holes)
    Tree         // Full hierarchy
};

/**
 * Contour approximation method
 */
enum class ContourApprox {
    None,           // Store all contour points
    Simple,         // Compress horizontal/vertical/diagonal segments
    TC89_L1,        // Teh-Chin L1 approximation
    TC89_KCOS       // Teh-Chin KCOS approximation
};

/**
 * Contour with hierarchy information
 */
struct Contour {
    std::vector<Point2D> points;
    int parent = -1;        // Index of parent contour (-1 = none)
    int child = -1;         // Index of first child contour
    int next = -1;          // Index of next sibling contour
    int prev = -1;          // Index of previous sibling contour
    int level = 0;          // Hierarchy level (0 = outermost)
};

/**
 * Region proposal for object detection
 */
struct RegionProposal {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
    float score = 0.0f;     // Objectness score (if available)
    int label = -1;         // Optional class label
};

/**
 * Bounding box with optional rotation
 */
struct BoundingBox {
    float x = 0.0f;         // Top-left X
    float y = 0.0f;         // Top-left Y
    float width = 0.0f;
    float height = 0.0f;
    float angle = 0.0f;     // Rotation angle (0 = axis-aligned)
    float confidence = 1.0f;
    int class_id = -1;
    std::string class_name;
};

/**
 * Labeling utilities for dataset annotation
 *
 * Provides tools for:
 * - Contour extraction and hierarchy
 * - Region proposals generation
 * - Mask creation and manipulation
 * - Bounding box operations
 */
class LabelingUtils {
public:
    /**
     * Extract contours from a binary mask
     * @param binary_mask Binary mask (0-1 values)
     * @param width Image width
     * @param height Image height
     * @param mode Contour retrieval mode
     * @param approx Approximation method
     * @return Vector of contours with hierarchy
     */
    static std::vector<Contour> ExtractContours(
        const std::vector<float>& binary_mask,
        int width, int height,
        ContourMode mode = ContourMode::List,
        ContourApprox approx = ContourApprox::Simple
    );

    /**
     * Generate selective search region proposals
     * @param image_data Float image data (0-1 range) [HWC format]
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels (3 for RGB)
     * @param mode "fast" or "quality"
     * @param base_k Base k for initial segmentation
     * @param inc_k Increment for k
     * @param sigma Gaussian blur sigma
     * @return Vector of region proposals
     */
    static std::vector<RegionProposal> SelectiveSearch(
        const std::vector<float>& image_data,
        int width, int height, int channels,
        const std::string& mode = "fast",
        int base_k = 150,
        int inc_k = 150,
        float sigma = 0.8f
    );

    /**
     * Generate edge boxes region proposals
     * @param image_data Float image data (0-1 range) [HWC format]
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels
     * @param max_boxes Maximum number of proposals
     * @param alpha Step size of sliding window
     * @param beta NMS threshold
     * @return Vector of region proposals
     */
    static std::vector<RegionProposal> EdgeBoxes(
        const std::vector<float>& image_data,
        int width, int height, int channels,
        int max_boxes = 100,
        float alpha = 0.65f,
        float beta = 0.75f
    );

    /**
     * Create binary mask from contour
     * @param contour Contour points
     * @param width Output mask width
     * @param height Output mask height
     * @param fill If true, fill interior; if false, draw outline only
     * @param thickness Line thickness (if fill = false)
     * @return Binary mask (0-1 values)
     */
    static std::vector<float> CreateMaskFromContour(
        const std::vector<Point2D>& contour,
        int width, int height,
        bool fill = true,
        int thickness = 1
    );

    /**
     * Create binary mask from multiple contours
     * @param contours Vector of contours
     * @param width Output mask width
     * @param height Output mask height
     * @param fill If true, fill interior
     * @return Binary mask (0-1 values)
     */
    static std::vector<float> CreateMaskFromContours(
        const std::vector<std::vector<Point2D>>& contours,
        int width, int height,
        bool fill = true
    );

    /**
     * Create binary mask from bounding box
     * @param bbox Bounding box
     * @param width Output mask width
     * @param height Output mask height
     * @param fill If true, fill interior
     * @return Binary mask (0-1 values)
     */
    static std::vector<float> CreateMaskFromBBox(
        const BoundingBox& bbox,
        int width, int height,
        bool fill = true
    );

    /**
     * Convert contour to bounding box
     * @param contour Contour points
     * @param rotated If true, return minimum area rotated rect
     * @return Bounding box
     */
    static BoundingBox ContourToBBox(
        const std::vector<Point2D>& contour,
        bool rotated = false
    );

    /**
     * Convert mask to bounding boxes (one per connected component)
     * @param binary_mask Binary mask (0-1 values)
     * @param width Mask width
     * @param height Mask height
     * @param min_area Minimum component area
     * @return Vector of bounding boxes
     */
    static std::vector<BoundingBox> MaskToBBoxes(
        const std::vector<float>& binary_mask,
        int width, int height,
        int min_area = 0
    );

    /**
     * Apply non-maximum suppression to bounding boxes
     * @param boxes Input bounding boxes
     * @param iou_threshold IoU threshold for suppression
     * @return Indices of kept boxes
     */
    static std::vector<int> NonMaxSuppression(
        const std::vector<BoundingBox>& boxes,
        float iou_threshold = 0.5f
    );

    /**
     * Calculate IoU (Intersection over Union) between two boxes
     * @param box1 First bounding box
     * @param box2 Second bounding box
     * @return IoU value (0-1)
     */
    static float CalculateIoU(const BoundingBox& box1, const BoundingBox& box2);

    /**
     * Dilate a binary mask
     * @param mask Binary mask (modified in-place)
     * @param width Mask width
     * @param height Mask height
     * @param kernel_size Dilation kernel size
     * @param iterations Number of iterations
     */
    static void DilateMask(
        std::vector<float>& mask,
        int width, int height,
        int kernel_size = 3,
        int iterations = 1
    );

    /**
     * Erode a binary mask
     * @param mask Binary mask (modified in-place)
     * @param width Mask width
     * @param height Mask height
     * @param kernel_size Erosion kernel size
     * @param iterations Number of iterations
     */
    static void ErodeMask(
        std::vector<float>& mask,
        int width, int height,
        int kernel_size = 3,
        int iterations = 1
    );

    /**
     * Smooth mask boundaries
     * @param mask Binary mask (modified in-place)
     * @param width Mask width
     * @param height Mask height
     * @param kernel_size Gaussian kernel size
     * @param threshold Threshold for binarization
     */
    static void SmoothMaskBoundary(
        std::vector<float>& mask,
        int width, int height,
        int kernel_size = 5,
        float threshold = 0.5f
    );

    /**
     * Fill holes in a binary mask
     * @param mask Binary mask (modified in-place)
     * @param width Mask width
     * @param height Mask height
     */
    static void FillHoles(
        std::vector<float>& mask,
        int width, int height
    );

    /**
     * Convert polygon annotation to RLE (Run-Length Encoding)
     * @param contour Contour points
     * @param width Image width
     * @param height Image height
     * @return RLE encoded mask (counts array)
     */
    static std::vector<int> PolygonToRLE(
        const std::vector<Point2D>& contour,
        int width, int height
    );

    /**
     * Convert RLE to binary mask
     * @param rle RLE encoded mask (counts array)
     * @param width Image width
     * @param height Image height
     * @return Binary mask (0-1 values)
     */
    static std::vector<float> RLEToMask(
        const std::vector<int>& rle,
        int width, int height
    );

private:
    static bool ValidateInput(const std::vector<float>& data,
                              int width, int height, int channels = 1);
};

// Enum string conversions
std::string ContourModeToString(ContourMode mode);
ContourMode StringToContourMode(const std::string& str);
std::string ContourApproxToString(ContourApprox approx);
ContourApprox StringToContourApprox(const std::string& str);

} // namespace cyxwiz
