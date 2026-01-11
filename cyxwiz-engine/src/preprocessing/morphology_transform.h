#pragma once

#include <vector>
#include <nlohmann/json.hpp>

namespace cyxwiz {

/**
 * Morphological operation types
 * Based on OpenCV morphological operations
 */
enum class MorphOperation {
    Erode,      // cv::MORPH_ERODE - Erodes image (shrinks bright regions)
    Dilate,     // cv::MORPH_DILATE - Dilates image (expands bright regions)
    Open,       // cv::MORPH_OPEN - Erosion followed by dilation (removes small bright spots)
    Close,      // cv::MORPH_CLOSE - Dilation followed by erosion (removes small dark spots)
    Gradient,   // cv::MORPH_GRADIENT - Dilation minus erosion (edge detection)
    TopHat,     // cv::MORPH_TOPHAT - Source minus opening (extracts bright details)
    BlackHat    // cv::MORPH_BLACKHAT - Closing minus source (extracts dark details)
};

/**
 * Structuring element (kernel) shapes
 */
enum class MorphShape {
    Rect,       // cv::MORPH_RECT - Rectangular kernel
    Cross,      // cv::MORPH_CROSS - Cross-shaped kernel
    Ellipse     // cv::MORPH_ELLIPSE - Elliptical kernel
};

/**
 * Configuration for morphological operations
 */
struct MorphologyConfig {
    bool enabled = false;
    MorphOperation operation = MorphOperation::Erode;
    MorphShape shape = MorphShape::Rect;
    int kernel_size = 3;    // Must be odd: 3, 5, 7, etc.
    int iterations = 1;     // Number of times to apply operation

    nlohmann::json ToJson() const;
    static MorphologyConfig FromJson(const nlohmann::json& j);

    bool operator==(const MorphologyConfig& other) const;
    bool operator!=(const MorphologyConfig& other) const { return !(*this == other); }
};

/**
 * Morphological transform operations using OpenCV
 *
 * Morphological operations are fundamental image processing techniques
 * that process images based on shapes. They are particularly useful for:
 * - Noise removal
 * - Extraction of individual elements
 * - Joining disparate elements
 * - Finding intensity bumps or holes
 */
class MorphologyTransform {
public:
    /**
     * Apply morphological operation to image using config
     * @param image_data Float image data (0-1 range) [modified in-place]
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels (1=grayscale, 3=RGB)
     * @param config Morphology configuration
     * @return true if successful, false otherwise
     */
    static bool Apply(
        std::vector<float>& image_data,
        int width, int height, int channels,
        const MorphologyConfig& config
    );

    /**
     * Apply morphological operation with explicit parameters
     * @param image_data Float image data (0-1 range) [modified in-place]
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels (1=grayscale, 3=RGB)
     * @param operation Type of morphological operation
     * @param shape Structuring element shape
     * @param kernel_size Kernel size (must be odd)
     * @param iterations Number of times to apply
     * @return true if successful, false otherwise
     */
    static bool ApplyOperation(
        std::vector<float>& image_data,
        int width, int height, int channels,
        MorphOperation operation,
        MorphShape shape = MorphShape::Rect,
        int kernel_size = 3,
        int iterations = 1
    );

    // Convenience methods for individual operations

    /**
     * Erode image - shrinks bright regions, enlarges dark regions
     * Useful for: removing small bright spots, separating connected objects
     */
    static bool Erode(
        std::vector<float>& image_data,
        int width, int height, int channels,
        int kernel_size = 3,
        int iterations = 1,
        MorphShape shape = MorphShape::Rect
    );

    /**
     * Dilate image - expands bright regions, shrinks dark regions
     * Useful for: filling small holes, connecting nearby objects
     */
    static bool Dilate(
        std::vector<float>& image_data,
        int width, int height, int channels,
        int kernel_size = 3,
        int iterations = 1,
        MorphShape shape = MorphShape::Rect
    );

    /**
     * Opening - erosion followed by dilation
     * Useful for: removing small bright spots while preserving shape
     */
    static bool Open(
        std::vector<float>& image_data,
        int width, int height, int channels,
        int kernel_size = 3,
        MorphShape shape = MorphShape::Rect
    );

    /**
     * Closing - dilation followed by erosion
     * Useful for: closing small dark holes while preserving shape
     */
    static bool Close(
        std::vector<float>& image_data,
        int width, int height, int channels,
        int kernel_size = 3,
        MorphShape shape = MorphShape::Rect
    );

    /**
     * Morphological gradient - difference between dilation and erosion
     * Useful for: edge detection, finding object boundaries
     */
    static bool Gradient(
        std::vector<float>& image_data,
        int width, int height, int channels,
        int kernel_size = 3,
        MorphShape shape = MorphShape::Rect
    );

    /**
     * Top Hat - difference between source and opening
     * Useful for: extracting small bright details on dark background
     */
    static bool TopHat(
        std::vector<float>& image_data,
        int width, int height, int channels,
        int kernel_size = 3,
        MorphShape shape = MorphShape::Rect
    );

    /**
     * Black Hat - difference between closing and source
     * Useful for: extracting small dark details on bright background
     */
    static bool BlackHat(
        std::vector<float>& image_data,
        int width, int height, int channels,
        int kernel_size = 3,
        MorphShape shape = MorphShape::Rect
    );

private:
    static bool ValidateInput(const std::vector<float>& image_data,
                              int width, int height, int channels);
    static int ValidateKernelSize(int kernel_size);
};

// Enum string conversions for serialization and UI
std::string MorphOperationToString(MorphOperation op);
MorphOperation StringToMorphOperation(const std::string& str);
std::string MorphShapeToString(MorphShape shape);
MorphShape StringToMorphShape(const std::string& str);

} // namespace cyxwiz
