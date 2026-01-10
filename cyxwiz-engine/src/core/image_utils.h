#pragma once

#include <string>
#include <vector>

namespace cyxwiz {

/**
 * ImageUtils - Unified image processing utilities using OpenCV
 *
 * Provides high-quality image operations with OpenCV to eliminate code duplication
 * across dataset loaders, preprocessing, and texture management.
 *
 * All methods use OpenCV for superior quality compared to manual implementations.
 */
class ImageUtils {
public:
    /**
     * Load image from file using OpenCV
     *
     * @param path Path to image file
     * @param data Output float data [0, 1] range
     * @param width Output image width
     * @param height Output image height
     * @param channels Output number of channels (1=Grayscale, 3=RGB)
     * @return true if successful, false otherwise
     *
     * Supported formats: JPG, PNG, BMP, TIFF, WebP, and more via OpenCV
     * Output is always RGB (OpenCV loads as BGR, we convert to RGB)
     */
    static bool LoadImage(const std::string& path,
                          std::vector<float>& data,
                          int& width, int& height, int& channels);

    /**
     * Resize methods available via OpenCV
     */
    enum class ResizeMethod {
        Nearest,      // cv::INTER_NEAREST - Fastest, lowest quality
        Bilinear,     // cv::INTER_LINEAR - Fast, good quality
        Bicubic,      // cv::INTER_CUBIC - High quality, smooth
        Lanczos,      // cv::INTER_LANCZOS4 - Highest quality, best for upscaling
        Area          // cv::INTER_AREA - Best for downscaling (anti-aliasing)
    };

    /**
     * Resize image using OpenCV interpolation
     *
     * @param data Input/output float data [0, 1] range
     * @param src_width Source image width
     * @param src_height Source image height
     * @param channels Number of channels (1 or 3)
     * @param dst_width Target width
     * @param dst_height Target height
     * @param method Resize interpolation method
     * @return true if successful, false otherwise
     *
     * Recommendations:
     * - Use Area for downscaling (reduces aliasing)
     * - Use Lanczos for upscaling (sharpest results)
     * - Use Bicubic for general purpose (good balance)
     */
    static bool ResizeImage(std::vector<float>& data,
                            int src_width, int src_height, int channels,
                            int dst_width, int dst_height,
                            ResizeMethod method = ResizeMethod::Bicubic);

    /**
     * Color space enumeration
     */
    enum class ColorSpace {
        RGB,        // Red, Green, Blue (standard)
        BGR,        // Blue, Green, Red (OpenCV default)
        Grayscale,  // Single channel luminance
        HSV,        // Hue, Saturation, Value (cylindrical color space)
        Lab,        // Lightness, a*, b* (perceptual color space)
        YCbCr       // Luma, Chroma blue, Chroma red (video color space)
    };

    /**
     * Convert between color spaces using OpenCV
     *
     * @param data Input/output float data [0, 1] range
     * @param width Image width
     * @param height Image height
     * @param from Source color space
     * @param to Target color space
     * @return true if successful, false otherwise
     *
     * Common conversions:
     * - RGB ↔ BGR: OpenCV uses BGR, most libraries use RGB
     * - RGB → Grayscale: Weighted conversion (0.299R + 0.587G + 0.114B)
     * - RGB → HSV: Useful for color-based filtering
     * - RGB → Lab: Perceptually uniform color differences
     */
    static bool ConvertColorSpace(std::vector<float>& data,
                                   int width, int height,
                                   ColorSpace from, ColorSpace to);

    /**
     * Convert uint8 pixels to float [0, 1] range
     *
     * @param src Source uint8 data [0, 255]
     * @param dst Destination float data [0, 1]
     * @param count Number of values to convert
     * @param scale Scale factor (default: 1/255)
     */
    static void NormalizePixels(const unsigned char* src, float* dst,
                                size_t count, float scale = 1.0f / 255.0f);

    /**
     * Convert float [0, 1] to uint8 pixels
     *
     * @param src Source float data [0, 1]
     * @param dst Destination uint8 data [0, 255]
     * @param count Number of values to convert
     * @param scale Scale factor (default: 255)
     */
    static void FloatToUint8(const float* src, unsigned char* dst,
                             size_t count, float scale = 255.0f);

    /**
     * Save image to file using OpenCV
     *
     * @param path Output file path (extension determines format)
     * @param data Float image data [0, 1] range
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels (1=Grayscale, 3=RGB)
     * @return true if successful, false otherwise
     *
     * Supported formats: JPG, PNG, BMP, TIFF, WebP, and more via OpenCV
     * RGB data is converted to BGR for OpenCV's imwrite
     */
    static bool SaveImage(const std::string& path,
                          const std::vector<float>& data,
                          int width, int height, int channels);

    /**
     * Save mask/segmentation to file
     *
     * @param path Output file path (PNG recommended for masks)
     * @param mask Float mask data [0, 1] range
     * @param width Mask width
     * @param height Mask height
     * @param binary If true, threshold at 0.5 to create binary mask (0/255)
     *               If false, scale to full range (0-255)
     * @return true if successful, false otherwise
     */
    static bool SaveMask(const std::string& path,
                         const std::vector<float>& mask,
                         int width, int height, bool binary = false);

    /**
     * Get list of supported image formats
     *
     * @return Vector of file extensions (e.g., "jpg", "png", "bmp")
     */
    static std::vector<std::string> GetSupportedFormats();

    /**
     * Check if OpenCV is available
     *
     * @return true (always, since we use OpenCV only)
     */
    static bool HasOpenCV() { return true; }

private:
    // Helper to convert OpenCV Mat to float vector
    static void MatToFloatVector(const void* mat, std::vector<float>& data);

    // Helper to convert float vector to OpenCV Mat
    static void FloatVectorToMat(const std::vector<float>& data,
                                  int width, int height, int channels,
                                  void* mat);
};

} // namespace cyxwiz
