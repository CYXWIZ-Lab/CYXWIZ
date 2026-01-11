#pragma once

#include <vector>
#include <nlohmann/json.hpp>

namespace cyxwiz {

/**
 * Image pyramid operation types
 */
enum class PyramidOperation {
    PyrDown,        // Gaussian pyramid downscale (blur + subsample by 2)
    PyrUp,          // Gaussian pyramid upscale (upsample by 2 + blur)
    BuildGaussian,  // Build multi-level Gaussian pyramid and return specific level
    BuildLaplacian  // Build Laplacian pyramid (high-frequency detail at level)
};

/**
 * Configuration for pyramid operations
 */
struct PyramidConfig {
    bool enabled = false;
    PyramidOperation operation = PyramidOperation::PyrDown;
    int levels = 1;  // Number of pyramid levels (for multi-level operations)

    nlohmann::json ToJson() const;
    static PyramidConfig FromJson(const nlohmann::json& j);

    bool operator==(const PyramidConfig& other) const;
    bool operator!=(const PyramidConfig& other) const { return !(*this == other); }
};

/**
 * Image pyramid transform operations using OpenCV
 *
 * Image pyramids are multi-scale representations useful for:
 * - Multi-resolution processing
 * - Feature detection at different scales
 * - Image blending (Laplacian pyramid)
 * - Efficient template matching
 * - Anti-aliased downsampling
 */
class PyramidTransform {
public:
    /**
     * Apply pyramid operation using config
     * @param image_data Float image data (0-1 range) [modified in-place]
     * @param width Input/output image width (changes for Down/Up operations)
     * @param height Input/output image height (changes for Down/Up operations)
     * @param channels Number of channels (1=grayscale, 3=RGB)
     * @param config Pyramid configuration
     * @return true if successful, false otherwise
     */
    static bool Apply(
        std::vector<float>& image_data,
        int& width, int& height, int channels,
        const PyramidConfig& config
    );

    /**
     * Downsample image using Gaussian pyramid
     * Applies Gaussian blur then subsamples by factor of 2
     * Output size: (width/2, height/2) - rounded down
     * @return true if successful
     */
    static bool PyrDown(
        std::vector<float>& image_data,
        int& width, int& height, int channels
    );

    /**
     * Upsample image using Gaussian pyramid
     * Upsamples by factor of 2 then applies Gaussian blur
     * Output size: (width*2, height*2)
     * @return true if successful
     */
    static bool PyrUp(
        std::vector<float>& image_data,
        int& width, int& height, int channels
    );

    /**
     * Build Gaussian pyramid and return specified level
     * Level 0 is original image, level 1 is half size, etc.
     * @param level Target level (0 = original)
     * @return true if successful
     */
    static bool BuildGaussianLevel(
        std::vector<float>& image_data,
        int& width, int& height, int channels,
        int level
    );

    /**
     * Build Laplacian pyramid and return specified level
     * Laplacian level = Gaussian[n] - upsample(Gaussian[n+1])
     * Contains high-frequency details at each level
     * @param level Target level (higher = coarser detail)
     * @return true if successful
     */
    static bool BuildLaplacianLevel(
        std::vector<float>& image_data,
        int& width, int& height, int channels,
        int level
    );

    /**
     * Apply multiple pyrDown operations
     * More efficient than calling PyrDown in a loop
     * @param levels Number of times to downsample
     * @return true if successful
     */
    static bool PyrDownMultiple(
        std::vector<float>& image_data,
        int& width, int& height, int channels,
        int levels
    );

    /**
     * Apply multiple pyrUp operations
     * @param levels Number of times to upsample
     * @return true if successful
     */
    static bool PyrUpMultiple(
        std::vector<float>& image_data,
        int& width, int& height, int channels,
        int levels
    );

private:
    static bool ValidateInput(const std::vector<float>& image_data,
                              int width, int height, int channels);
};

// Enum string conversions
std::string PyramidOperationToString(PyramidOperation op);
PyramidOperation StringToPyramidOperation(const std::string& str);

} // namespace cyxwiz
