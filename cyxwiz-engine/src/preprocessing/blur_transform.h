#pragma once

#include <vector>
#include <nlohmann/json.hpp>

namespace cyxwiz {

/**
 * Blur/smoothing algorithm types
 */
enum class BlurType {
    Box,        // cv::blur - Simple averaging (fast, uniform blur)
    Gaussian,   // cv::GaussianBlur - Weighted by Gaussian kernel (natural blur)
    Median,     // cv::medianBlur - Median value in neighborhood (removes salt/pepper noise)
    Bilateral   // cv::bilateralFilter - Edge-preserving smoothing (slow but high quality)
};

/**
 * Configuration for blur/smoothing operations
 */
struct BlurConfig {
    bool enabled = false;
    BlurType type = BlurType::Gaussian;
    int kernel_size = 5;        // Must be odd: 3, 5, 7, etc. (for median, must be 3, 5, or 7)
    float sigma = 0.0f;         // For Gaussian: 0 = auto-calculate from kernel size
    float sigma_color = 75.0f;  // For Bilateral: filter sigma in the color space
    float sigma_space = 75.0f;  // For Bilateral: filter sigma in the coordinate space

    nlohmann::json ToJson() const;
    static BlurConfig FromJson(const nlohmann::json& j);

    bool operator==(const BlurConfig& other) const;
    bool operator!=(const BlurConfig& other) const { return !(*this == other); }
};

/**
 * Blur/smoothing transform operations using OpenCV
 *
 * Blur operations are used for:
 * - Noise reduction
 * - Image preprocessing before edge detection
 * - Creating smooth gradients
 * - Simulating depth of field
 */
class BlurTransform {
public:
    /**
     * Apply blur operation to image using config
     * @param image_data Float image data (0-1 range) [modified in-place]
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels (1=grayscale, 3=RGB)
     * @param config Blur configuration
     * @return true if successful, false otherwise
     */
    static bool Apply(
        std::vector<float>& image_data,
        int width, int height, int channels,
        const BlurConfig& config
    );

    /**
     * Box blur (simple averaging)
     * Fast but produces blocky results at large kernel sizes
     * @param kernel_size Size of the blur kernel (must be odd)
     */
    static bool BoxBlur(
        std::vector<float>& image_data,
        int width, int height, int channels,
        int kernel_size = 5
    );

    /**
     * Gaussian blur (weighted averaging)
     * Most commonly used blur, produces natural-looking results
     * @param kernel_size Size of the Gaussian kernel (must be odd)
     * @param sigma Gaussian standard deviation (0 = auto from kernel size)
     */
    static bool GaussianBlur(
        std::vector<float>& image_data,
        int width, int height, int channels,
        int kernel_size = 5,
        float sigma = 0.0f
    );

    /**
     * Median blur (median filtering)
     * Excellent for removing salt-and-pepper noise while preserving edges
     * @param kernel_size Size of the median kernel (must be odd, max 255)
     */
    static bool MedianBlur(
        std::vector<float>& image_data,
        int width, int height, int channels,
        int kernel_size = 5
    );

    /**
     * Bilateral filter (edge-preserving smoothing)
     * Smooths while preserving edges - best quality but slowest
     * @param diameter Diameter of pixel neighborhood (use -1 for auto from sigma_space)
     * @param sigma_color Filter sigma in the color space (larger = more colors mixed)
     * @param sigma_space Filter sigma in coordinate space (larger = more distant pixels influence)
     */
    static bool BilateralFilter(
        std::vector<float>& image_data,
        int width, int height, int channels,
        int diameter = 9,
        float sigma_color = 75.0f,
        float sigma_space = 75.0f
    );

private:
    static bool ValidateInput(const std::vector<float>& image_data,
                              int width, int height, int channels);
    static int ValidateKernelSize(int kernel_size, BlurType type);
};

// Enum string conversions for serialization and UI
std::string BlurTypeToString(BlurType type);
BlurType StringToBlurType(const std::string& str);

} // namespace cyxwiz
