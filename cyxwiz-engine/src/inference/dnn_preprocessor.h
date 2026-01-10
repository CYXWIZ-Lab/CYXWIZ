#pragma once

#include <string>
#include <vector>
#include <array>
#include <functional>

namespace cyxwiz {

/**
 * Color space for preprocessing
 */
enum class ColorSpace {
    RGB,
    BGR,
    Grayscale
};

/**
 * Preprocessing configuration for DNN models
 */
struct DNNPreprocessConfig {
    // Input size
    int input_width = 224;
    int input_height = 224;

    // Normalization
    float scale_factor = 1.0f / 255.0f;  // Scale pixel values
    std::array<float, 3> mean = {0.0f, 0.0f, 0.0f};  // Mean subtraction (BGR order)
    std::array<float, 3> std_dev = {1.0f, 1.0f, 1.0f};  // Std normalization

    // Color handling
    ColorSpace input_color = ColorSpace::RGB;  // Expected input format
    ColorSpace model_color = ColorSpace::BGR;  // Model expects BGR (most OpenCV models)
    bool swap_rb = true;  // Swap R and B channels

    // Spatial handling
    bool crop_center = false;  // Center crop to square before resize
    bool keep_aspect_ratio = false;  // Letterbox padding
    float letterbox_color = 0.5f;  // Fill color for letterbox (0-1)

    // Data format
    bool channels_first = true;  // NCHW vs NHWC (OpenCV DNN uses NCHW)
};

/**
 * Preprocessed image blob for DNN inference
 */
struct DNNBlob {
    std::vector<float> data;
    int batch_size = 1;
    int channels = 3;
    int height = 0;
    int width = 0;

    // Original image dimensions (for mapping output back)
    int original_width = 0;
    int original_height = 0;

    // Letterbox padding info (for coordinate mapping)
    float scale = 1.0f;
    int pad_left = 0;
    int pad_top = 0;
};

/**
 * DNN Preprocessor - Image preprocessing for neural network inference
 *
 * Handles common preprocessing operations:
 * - Resizing (with aspect ratio preservation or letterbox)
 * - Normalization (mean subtraction, std division, scaling)
 * - Color space conversion
 * - Data layout conversion (HWC to CHW)
 */
class DNNPreprocessor {
public:
    /**
     * Preprocess image for DNN inference
     * @param image_data Input image (float 0-1 range, HWC format)
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels (1 or 3)
     * @param config Preprocessing configuration
     * @return Preprocessed blob ready for DNN inference
     */
    static DNNBlob Preprocess(
        const std::vector<float>& image_data,
        int width, int height, int channels,
        const DNNPreprocessConfig& config
    );

    /**
     * Preprocess batch of images
     * @param images Vector of images (each float 0-1 range, HWC format)
     * @param widths Image widths
     * @param heights Image heights
     * @param channels Number of channels
     * @param config Preprocessing configuration
     * @return Preprocessed batch blob
     */
    static DNNBlob PreprocessBatch(
        const std::vector<std::vector<float>>& images,
        const std::vector<int>& widths,
        const std::vector<int>& heights,
        int channels,
        const DNNPreprocessConfig& config
    );

    /**
     * Map detection coordinates from model output space to original image space
     * @param x X coordinate (0-1 in model space)
     * @param y Y coordinate (0-1 in model space)
     * @param blob The blob used for preprocessing (contains mapping info)
     * @return Coordinates in original image space (0-1)
     */
    static std::pair<float, float> MapToOriginal(
        float x, float y,
        const DNNBlob& blob
    );

    /**
     * Map bounding box from model output space to original image space
     * @param x Box center x (0-1)
     * @param y Box center y (0-1)
     * @param w Box width (0-1)
     * @param h Box height (0-1)
     * @param blob The blob used for preprocessing
     * @return Box in original image space (x, y, w, h all 0-1)
     */
    static std::array<float, 4> MapBoxToOriginal(
        float x, float y, float w, float h,
        const DNNBlob& blob
    );

    // Preset configurations for common models

    /**
     * Get ImageNet preprocessing config
     * Standard ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
     */
    static DNNPreprocessConfig ImageNetConfig(int size = 224);

    /**
     * Get YOLO preprocessing config
     * Scale 1/255, no mean subtraction, letterbox to square
     */
    static DNNPreprocessConfig YOLOConfig(int size = 416);

    /**
     * Get SSD preprocessing config
     * Various configs depending on backbone
     */
    static DNNPreprocessConfig SSDConfig(int size = 300, bool mobilenet = true);

    /**
     * Get Face Detection (SSD) preprocessing config
     * OpenCV face detector: res10_300x300
     */
    static DNNPreprocessConfig FaceDetectorConfig();

    /**
     * Get OpenPose preprocessing config
     */
    static DNNPreprocessConfig OpenPoseConfig(int height = 368);

    /**
     * Get generic classification preprocessing config
     */
    static DNNPreprocessConfig ClassificationConfig(int size = 224);

private:
    // Resize with letterbox padding
    static void ResizeLetterbox(
        const std::vector<float>& src, int src_w, int src_h, int channels,
        std::vector<float>& dst, int dst_w, int dst_h,
        float& scale, int& pad_left, int& pad_top,
        float fill_color
    );

    // Resize with center crop
    static void ResizeCenterCrop(
        const std::vector<float>& src, int src_w, int src_h, int channels,
        std::vector<float>& dst, int dst_w, int dst_h
    );

    // Simple bilinear resize
    static void ResizeBilinear(
        const std::vector<float>& src, int src_w, int src_h, int channels,
        std::vector<float>& dst, int dst_w, int dst_h
    );

    // Convert HWC to CHW
    static void HWCToCHW(
        const std::vector<float>& hwc, int w, int h, int c,
        std::vector<float>& chw
    );

    // Apply normalization
    static void Normalize(
        std::vector<float>& data, int w, int h, int c,
        float scale,
        const std::array<float, 3>& mean,
        const std::array<float, 3>& std_dev
    );

    // Swap R and B channels
    static void SwapRB(std::vector<float>& data, int w, int h);
};

} // namespace cyxwiz
