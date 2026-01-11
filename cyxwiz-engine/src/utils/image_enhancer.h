#pragma once

#include <vector>
#include <cstdint>

namespace cyxwiz {

/**
 * Image enhancement utilities using OpenCV
 * Provides CLAHE, denoising, and sharpening operations
 */
class ImageEnhancer {
public:
    /**
     * Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
     * @param image_data Float image data (0-1 range) [modified in-place]
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels (1=grayscale, 3=RGB)
     * @param clip_limit Threshold for contrast limiting (1.0-4.0, higher = more contrast)
     * @param tile_size Grid size for histogram equalization (8 or 16)
     * @return true if successful, false otherwise
     */
    static bool ApplyCLAHE(
        std::vector<float>& image_data,
        int width, int height, int channels,
        float clip_limit = 2.0f,
        int tile_size = 8
    );

    /**
     * Apply Gaussian denoising using Non-local Means
     * @param image_data Float image data (0-1 range) [modified in-place]
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels
     * @param strength Filter strength (10.0 = recommended)
     * @param color_strength Filter strength for color components (10.0 = recommended)
     * @param template_size Size in pixels of template patch (7 = recommended)
     * @param search_size Size in pixels of search area (21 = recommended)
     * @return true if successful, false otherwise
     */
    static bool ApplyDenoise(
        std::vector<float>& image_data,
        int width, int height, int channels,
        float strength = 10.0f,
        float color_strength = 10.0f,
        int template_size = 7,
        int search_size = 21
    );

    /**
     * Apply unsharp masking for sharpening
     * @param image_data Float image data (0-1 range) [modified in-place]
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels
     * @param amount Sharpening strength (1.0 = recommended)
     * @param radius Gaussian blur radius for unsharp mask (1.0 = recommended)
     * @param threshold Threshold for sharpening (0.0 = no threshold)
     * @return true if successful, false otherwise
     */
    static bool ApplySharpen(
        std::vector<float>& image_data,
        int width, int height, int channels,
        float amount = 1.0f,
        float radius = 1.0f,
        float threshold = 0.0f
    );

    /**
     * Edge detection types
     */
    enum class EdgeDetectorType {
        Canny,      // Canny edge detector (multi-stage, best general-purpose)
        Sobel,      // Sobel operator (gradient-based)
        Laplacian,  // Laplacian operator (second derivative)
        Scharr      // Scharr operator (high-precision gradient)
    };

    /**
     * Apply edge detection
     * @param image_data Float image data (0-1 range) [modified in-place]
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels
     * @param detector_type Type of edge detector to use
     * @param threshold1 For Canny: low threshold (50.0 recommended)
     * @param threshold2 For Canny: high threshold (150.0 recommended)
     * @param kernel_size For Sobel/Laplacian/Scharr: kernel size (3, 5, or 7)
     * @return true if successful, false otherwise
     */
    static bool ApplyEdgeDetection(
        std::vector<float>& image_data,
        int width, int height, int channels,
        EdgeDetectorType detector_type = EdgeDetectorType::Canny,
        float threshold1 = 50.0f,
        float threshold2 = 150.0f,
        int kernel_size = 3
    );

    /**
     * Enhancement presets for common use cases
     */
    enum class EnhancementPreset {
        None,       // No enhancement
        Light,      // CLAHE(1.5) + light sharpen
        Standard,   // CLAHE(2.0) + denoise + sharpen
        Strong,     // CLAHE(3.0) + strong denoise + sharpen
        Medical,    // CLAHE(2.5) only (preserve detail)
        Custom      // User-defined parameters
    };

    /**
     * Apply enhancement preset
     * @param image_data Float image data (0-1 range) [modified in-place]
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels
     * @param preset Enhancement preset to apply
     * @return true if successful, false otherwise
     */
    static bool ApplyPreset(
        std::vector<float>& image_data,
        int width, int height, int channels,
        EnhancementPreset preset
    );

private:
    // Helper methods
    static bool ValidateInput(const std::vector<float>& image_data,
                              int width, int height, int channels);
};

} // namespace cyxwiz
